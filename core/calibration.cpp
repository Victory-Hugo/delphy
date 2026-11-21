#include "calibration.h"

#include <algorithm>
#include <cmath>
#include <istream>
#include <numbers>
#include <ostream>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"

#include "human_dates.h"

namespace delphy {

namespace {

constexpr auto k_max_listed = 10;
constexpr auto k_max_abs_year = int64_t{10'000'000};
constexpr auto k_inf = std::numeric_limits<double>::infinity();

auto list_names(const std::vector<std::string>& names) -> std::string {
  auto shown = std::vector<std::string>(
      names.begin(), names.begin() + std::min<std::ptrdiff_t>(std::ssize(names), k_max_listed));
  auto result = absl::StrJoin(shown, ", ");
  if (std::ssize(names) > k_max_listed) {
    absl::StrAppend(&result, absl::StrFormat(", ... (%d in total)", std::ssize(names)));
  }
  return result;
}

auto throw_problems(const std::vector<std::string>& problems) -> void {
  auto msg = absl::StrFormat("Calibration configuration rejected: %d problem%s found",
                             std::ssize(problems), std::ssize(problems) == 1 ? "" : "s");
  auto i = 0;
  for (const auto& p : problems) {
    ++i;
    if (i > 25) {
      absl::StrAppend(&msg, absl::StrFormat("\n  ... and %d more", std::ssize(problems) - 25));
      break;
    }
    absl::StrAppend(&msg, absl::StrFormat("\n  %d. %s", i, p));
  }
  throw Calibration_error(msg);
}

}  // namespace

auto to_string(Age_units units) -> std::string_view {
  switch (units) {
    case Age_units::ka: return "ka";
    case Age_units::years_bp: return "years_bp";
    case Age_units::cal_bp: return "cal_bp";
    case Age_units::ce: return "ce";
    case Age_units::bce: return "bce";
  }
  return "?";
}

auto parse_age_units(std::string_view name) -> Age_units {
  if (name == "ka") { return Age_units::ka; }
  if (name == "years_bp") { return Age_units::years_bp; }
  if (name == "cal_bp") { return Age_units::cal_bp; }
  if (name == "ce") { return Age_units::ce; }
  if (name == "bce") { return Age_units::bce; }
  throw Calibration_error(absl::StrFormat(
      "Unknown age_units '%s'; expected one of ka, years_bp, cal_bp, ce, bce", name));
}

auto to_string(Calibration_distribution distribution) -> std::string_view {
  switch (distribution) {
    case Calibration_distribution::normal: return "normal";
    case Calibration_distribution::lognormal: return "lognormal";
    case Calibration_distribution::uniform: return "uniform";
  }
  return "?";
}

auto parse_calibration_distribution(std::string_view name) -> Calibration_distribution {
  if (name == "normal") { return Calibration_distribution::normal; }
  if (name == "lognormal") { return Calibration_distribution::lognormal; }
  if (name == "uniform") { return Calibration_distribution::uniform; }
  throw Calibration_error(absl::StrFormat(
      "Unknown distribution '%s'; expected one of normal, lognormal, uniform", name));
}

auto Age_prior::support_youngest_age() const -> double {
  auto result = youngest_age;
  if (distribution == Calibration_distribution::lognormal) {
    result = std::max(result, 0.0);
  }
  return result;
}

auto Age_prior::support_oldest_age() const -> double {
  return oldest_age;
}

auto Age_prior::log_density(double age_years) const -> double {
  if (not std::isfinite(age_years)) { return -k_inf; }
  if (age_years < youngest_age || age_years > oldest_age) { return -k_inf; }

  static const auto log_sqrt_2pi = 0.5 * std::log(2.0 * std::numbers::pi);
  switch (distribution) {
    case Calibration_distribution::normal: {
      auto z = (age_years - mean) / stddev;
      return -std::log(stddev) - log_sqrt_2pi - 0.5 * z * z;
    }
    case Calibration_distribution::lognormal: {
      if (age_years <= 0.0) { return -k_inf; }
      auto z = (std::log(age_years) - std::log(median)) / sigma;
      return -std::log(age_years) - std::log(sigma) - log_sqrt_2pi - 0.5 * z * z;
    }
    case Calibration_distribution::uniform: {
      return -std::log(oldest_age - youngest_age);
    }
  }
  return -k_inf;
}

// ---------------------------------------------------------------------------------------------
// Configuration file
// ---------------------------------------------------------------------------------------------

namespace {

struct Raw_value {
  std::string text;
  int line = 0;
};

auto reference_t_of(Age_units units, const std::optional<int64_t>& reference_year) -> double {
  switch (units) {
    case Age_units::ka:
    case Age_units::years_bp:
      return astronomical_year_to_day(reference_year.value());
    case Age_units::cal_bp:
      return astronomical_year_to_day(k_cal_bp_reference_year);
    case Age_units::ce:
    case Age_units::bce:
      return 0.0;  // 2020-01-01
  }
  return 0.0;
}

// Converts one location parameter (a date or an age, as written in the file) to years of age before
// the calibration's reference instant.  Throws Calibration_error for values that the convention
// cannot express.
auto location_to_age_years(Age_units units, double value, std::string_view key) -> double {
  if (not std::isfinite(value)) {
    throw Calibration_error(absl::StrFormat("%s must be a finite number", key));
  }
  switch (units) {
    case Age_units::ka:
      return value * 1000.0;
    case Age_units::years_bp:
    case Age_units::cal_bp:
      return value;
    case Age_units::ce:
    case Age_units::bce: {
      if (value != std::floor(value)) {
        throw Calibration_error(absl::StrFormat(
            "%s = %g is not a whole year, but the '%s' convention has a resolution of one calendar year.  "
            "Use age_units years_bp or cal_bp for fractional values", key, value, to_string(units)));
      }
      if (value < 1.0) {
        throw Calibration_error(absl::StrFormat(
            "%s = %g is not a valid %s year; %s years start at 1 (there is no year 0)",
            key, value, to_string(units), to_string(units)));
      }
      if (std::abs(value) > static_cast<double>(k_max_abs_year)) {
        throw Calibration_error(absl::StrFormat("%s = %g is out of range (|year| <= %d)",
                                                key, value, k_max_abs_year));
      }
      auto year = static_cast<int64_t>(value);
      auto astronomical_year = (units == Age_units::ce) ? year : (1 - year);
      return -astronomical_year_to_day(astronomical_year) / 365.0;
    }
  }
  return value;
}

auto scale_to_age_years(Age_units units, double value) -> double {
  return (units == Age_units::ka) ? value * 1000.0 : value;
}

struct Stanza {
  int line = 0;
  std::string name;
  std::vector<std::string> taxa;
  bool all_tips = false;
  bool saw_taxa = false;
  absl::flat_hash_map<std::string, Raw_value> keys;
};

auto parse_double(const Raw_value& v, std::string_view key) -> double {
  auto result = 0.0;
  if (not absl::SimpleAtod(v.text, &result)) {
    throw Calibration_error(absl::StrFormat("line %d: %s = '%s' is not a number", v.line, key, v.text));
  }
  if (not std::isfinite(result)) {
    throw Calibration_error(absl::StrFormat("line %d: %s = '%s' is not finite", v.line, key, v.text));
  }
  return result;
}

auto parse_int(const Raw_value& v, std::string_view key) -> int64_t {
  auto result = int64_t{0};
  if (not absl::SimpleAtoi(v.text, &result)) {
    throw Calibration_error(absl::StrFormat("line %d: %s = '%s' is not an integer", v.line, key, v.text));
  }
  return result;
}

// Builds one Calibration from a parsed stanza.  Throws Calibration_error on the first problem of this
// stanza; the caller collects the message.
auto stanza_to_calibration(const Stanza& stanza) -> Calibration {
  auto required = [&](std::string_view key) -> const Raw_value& {
    auto it = stanza.keys.find(key);
    if (it == stanza.keys.end()) {
      throw Calibration_error(absl::StrFormat(
          "calibration '%s' (line %d): missing required key '%s'", stanza.name, stanza.line, key));
    }
    return it->second;
  };
  auto optional_key = [&](std::string_view key) -> const Raw_value* {
    auto it = stanza.keys.find(key);
    return (it == stanza.keys.end()) ? nullptr : &it->second;
  };
  auto forbid = [&](std::string_view key, std::string_view why) -> void {
    if (stanza.keys.contains(key)) {
      throw Calibration_error(absl::StrFormat(
          "calibration '%s' (line %d): key '%s' is not allowed here: %s", stanza.name, stanza.line, key, why));
    }
  };

  auto result = Calibration{};
  result.name = stanza.name;
  result.config_line = stanza.line;
  result.all_tips = stanza.all_tips;
  result.taxa = stanza.taxa;

  if (not stanza.saw_taxa) {
    throw Calibration_error(absl::StrFormat(
        "calibration '%s' (line %d): missing required key 'taxa' (either 'taxa all' or a list of sample IDs)",
        stanza.name, stanza.line));
  }
  if (not result.all_tips && std::ssize(result.taxa) < 2) {
    throw Calibration_error(absl::StrFormat(
        "calibration '%s' (line %d): a calibration needs at least 2 sample IDs, because it constrains the age of "
        "their common ancestor.  The sampling age of a single sample belongs in the sample metadata, not here",
        stanza.name, stanza.line));
  }

  result.units = parse_age_units(required("age_units").text);
  if (result.units == Age_units::ka || result.units == Age_units::years_bp) {
    auto year = parse_int(required("age_reference_year"), "age_reference_year");
    if (std::abs(year) > k_max_abs_year || year == 0) {
      throw Calibration_error(absl::StrFormat(
          "calibration '%s' (line %d): age_reference_year = %d is not a valid year CE (there is no year 0; "
          "|year| <= %d)", stanza.name, stanza.line, year, k_max_abs_year));
    }
    result.reference_year = year;
  } else {
    forbid("age_reference_year", absl::StrFormat(
        "the '%s' convention fixes the reference itself (%s)", to_string(result.units),
        result.units == Age_units::cal_bp ? "1950 CE" : "the calendar year is absolute"));
  }
  result.reference_t = reference_t_of(result.units, result.reference_year);

  auto& prior = result.prior;
  prior.distribution = parse_calibration_distribution(required("distribution").text);

  auto convert_location = [&](const Raw_value& v, std::string_view key) {
    return location_to_age_years(result.units, parse_double(v, key), key);
  };

  switch (prior.distribution) {
    case Calibration_distribution::normal: {
      prior.mean = convert_location(required("mean"), "mean");
      auto stddev = scale_to_age_years(result.units, parse_double(required("stddev"), "stddev"));
      if (not (stddev > 0.0)) {
        throw Calibration_error(absl::StrFormat(
            "calibration '%s' (line %d): stddev must be > 0 (got %g).  A strictly fixed node age is not supported: "
            "it is a point mass that the time proposals can never hit, and it needs its own operator rather than a "
            "zero-width density", stanza.name, stanza.line, stddev));
      }
      prior.stddev = stddev;
      forbid("median", "it belongs to distribution lognormal");
      forbid("sigma", "it belongs to distribution lognormal");
      break;
    }
    case Calibration_distribution::lognormal: {
      if (result.units == Age_units::ce || result.units == Age_units::bce) {
        throw Calibration_error(absl::StrFormat(
            "calibration '%s' (line %d): distribution lognormal needs an age convention (ka, years_bp or cal_bp); "
            "a lognormal density on a calendar year has no meaning", stanza.name, stanza.line));
      }
      prior.median = convert_location(required("median"), "median");
      if (not (prior.median > 0.0)) {
        throw Calibration_error(absl::StrFormat(
            "calibration '%s' (line %d): median must be > 0 (got %g)", stanza.name, stanza.line, prior.median));
      }
      auto sigma = parse_double(required("sigma"), "sigma");
      if (not (sigma > 0.0)) {
        throw Calibration_error(absl::StrFormat(
            "calibration '%s' (line %d): sigma must be > 0 (got %g).  A strictly fixed node age is not supported: "
            "it is a point mass that the time proposals can never hit, and it needs its own operator rather than a "
            "zero-width density", stanza.name, stanza.line, sigma));
      }
      prior.sigma = sigma;
      forbid("mean", "it belongs to distribution normal");
      forbid("stddev", "it belongs to distribution normal");
      break;
    }
    case Calibration_distribution::uniform: {
      prior.oldest_age = convert_location(required("oldest"), "oldest");
      prior.youngest_age = convert_location(required("youngest"), "youngest");
      forbid("mean", "it belongs to distribution normal");
      forbid("stddev", "it belongs to distribution normal");
      forbid("median", "it belongs to distribution lognormal");
      forbid("sigma", "it belongs to distribution lognormal");
      break;
    }
  }

  if (prior.distribution != Calibration_distribution::uniform) {
    if (auto* v = optional_key("oldest"); v != nullptr) {
      prior.oldest_age = convert_location(*v, "oldest");
    }
    if (auto* v = optional_key("youngest"); v != nullptr) {
      prior.youngest_age = convert_location(*v, "youngest");
    }
  }

  if (not (prior.oldest_age > prior.youngest_age)) {
    throw Calibration_error(absl::StrFormat(
        "calibration '%s' (line %d): 'oldest' must be strictly older than 'youngest', but after conversion to years "
        "of age they are %g and %g.  In the '%s' convention, remember that larger values are %s",
        stanza.name, stanza.line, prior.oldest_age, prior.youngest_age, to_string(result.units),
        (result.units == Age_units::ce) ? "younger" : "older"));
  }

  if (auto* v = optional_key("monophyly"); v != nullptr) {
    if (v->text == "required") {
      result.require_monophyly = true;
    } else if (v->text == "not_required") {
      if (result.all_tips) {
        throw Calibration_error(absl::StrFormat(
            "calibration '%s' (line %d): 'monophyly not_required' has no meaning with 'taxa all', whose target is "
            "always the root", stanza.name, stanza.line));
      }
      result.require_monophyly = false;
    } else {
      throw Calibration_error(absl::StrFormat(
          "line %d: monophyly = '%s' is not recognized; use 'required' or 'not_required'", v->line, v->text));
    }
  }

  auto known_keys = absl::flat_hash_set<std::string>{
    "monophyly", "age_units", "age_reference_year", "distribution",
    "mean", "stddev", "median", "sigma", "oldest", "youngest"};
  auto unknown = std::vector<std::string>{};
  for (const auto& [key, v] : stanza.keys) {
    if (not known_keys.contains(key)) { unknown.push_back(absl::StrFormat("'%s' (line %d)", key, v.line)); }
  }
  if (not unknown.empty()) {
    std::ranges::sort(unknown);
    throw Calibration_error(absl::StrFormat(
        "calibration '%s' (line %d): unknown key(s) %s", stanza.name, stanza.line, absl::StrJoin(unknown, ", ")));
  }

  return result;
}

}  // namespace

auto read_calibration_config(std::istream& is) -> Calibration_set {
  auto problems = std::vector<std::string>{};
  auto stanzas = std::vector<Stanza>{};

  auto line_text = std::string{};
  auto line_no = 0;
  auto saw_version = false;
  auto cur = std::optional<Stanza>{};

  while (std::getline(is, line_text)) {
    ++line_no;
    if (not line_text.empty() && line_text.back() == '\r') { line_text.pop_back(); }
    auto trimmed = std::string{absl::StripAsciiWhitespace(line_text)};
    if (trimmed.empty() || trimmed.front() == '#') { continue; }

    auto tokens = std::vector<std::string>{};
    for (auto piece : absl::StrSplit(trimmed, absl::ByAnyChar(" \t"), absl::SkipEmpty())) {
      tokens.emplace_back(piece);
    }
    if (tokens.empty()) { continue; }
    const auto& key = tokens[0];

    if (not saw_version) {
      if (key != "version" || std::ssize(tokens) != 2 || tokens[1] != "1") {
        problems.push_back(absl::StrFormat(
            "line %d: a calibration configuration must start with 'version 1'", line_no));
        throw_problems(problems);
      }
      saw_version = true;
      continue;
    }

    if (key == "version") {
      problems.push_back(absl::StrFormat("line %d: duplicate 'version' line", line_no));
      continue;
    }

    if (key == "calibration") {
      if (cur.has_value()) {
        problems.push_back(absl::StrFormat(
            "line %d: calibration '%s' (line %d) was not closed with 'end'", line_no, cur->name, cur->line));
        cur.reset();
      }
      if (std::ssize(tokens) != 2) {
        problems.push_back(absl::StrFormat(
            "line %d: 'calibration' takes exactly one name, for example 'calibration root'", line_no));
        continue;
      }
      cur = Stanza{};
      cur->line = line_no;
      cur->name = tokens[1];
      continue;
    }

    if (key == "end") {
      if (not cur.has_value()) {
        problems.push_back(absl::StrFormat("line %d: 'end' without a matching 'calibration'", line_no));
        continue;
      }
      if (std::ssize(tokens) != 1) {
        problems.push_back(absl::StrFormat("line %d: 'end' takes no arguments", line_no));
      }
      stanzas.push_back(std::move(*cur));
      cur.reset();
      continue;
    }

    if (not cur.has_value()) {
      problems.push_back(absl::StrFormat(
          "line %d: '%s' appears outside a 'calibration ... end' block", line_no, key));
      continue;
    }

    if (key == "taxa") {
      if (std::ssize(tokens) < 2) {
        problems.push_back(absl::StrFormat(
            "line %d: 'taxa' needs either 'all' or one or more sample IDs", line_no));
        continue;
      }
      cur->saw_taxa = true;
      if (std::ssize(tokens) == 2 && tokens[1] == "all") {
        cur->all_tips = true;
      } else {
        for (auto i = 1; i != std::ssize(tokens); ++i) {
          if (tokens[i] == "all") {
            problems.push_back(absl::StrFormat(
                "line %d: 'all' must be the only entry of a 'taxa' line", line_no));
            continue;
          }
          cur->taxa.push_back(tokens[i]);
        }
      }
      continue;
    }

    if (std::ssize(tokens) != 2) {
      problems.push_back(absl::StrFormat(
          "line %d: key '%s' takes exactly one value", line_no, key));
      continue;
    }
    if (cur->keys.contains(key)) {
      problems.push_back(absl::StrFormat(
          "line %d: key '%s' is given twice in calibration '%s'", line_no, key, cur->name));
      continue;
    }
    cur->keys.emplace(key, Raw_value{tokens[1], line_no});
  }

  if (not saw_version) {
    problems.push_back("The calibration configuration is empty; it must start with 'version 1'");
    throw_problems(problems);
  }
  if (cur.has_value()) {
    problems.push_back(absl::StrFormat(
        "calibration '%s' (line %d) was not closed with 'end'", cur->name, cur->line));
  }

  auto result = Calibration_set{};
  auto seen_names = absl::flat_hash_set<std::string>{};
  for (auto& stanza : stanzas) {
    if (stanza.all_tips && not stanza.taxa.empty()) {
      problems.push_back(absl::StrFormat(
          "calibration '%s' (line %d): 'taxa all' cannot be combined with an explicit list of sample IDs",
          stanza.name, stanza.line));
      continue;
    }
    if (not seen_names.insert(stanza.name).second) {
      problems.push_back(absl::StrFormat(
          "calibration '%s' (line %d): duplicate calibration name", stanza.name, stanza.line));
      continue;
    }
    auto sorted = stanza.taxa;
    std::ranges::sort(sorted);
    auto duplicates = std::vector<std::string>{};
    for (auto i = 1; i < std::ssize(sorted); ++i) {
      if (sorted[i] == sorted[i - 1]) { duplicates.push_back(sorted[i]); }
    }
    if (not duplicates.empty()) {
      duplicates.erase(std::unique(duplicates.begin(), duplicates.end()), duplicates.end());
      problems.push_back(absl::StrFormat(
          "calibration '%s' (line %d): sample ID(s) listed more than once: %s",
          stanza.name, stanza.line, list_names(duplicates)));
      continue;
    }
    stanza.taxa = std::move(sorted);

    try {
      result.calibrations.push_back(stanza_to_calibration(stanza));
    } catch (const Calibration_error& e) {
      problems.push_back(e.what());
    }
  }

  if (result.calibrations.empty() && problems.empty()) {
    problems.push_back("The calibration configuration contains no calibration");
  }
  if (not problems.empty()) { throw_problems(problems); }
  return result;
}

auto write_calibration_config(const Calibration_set& set, std::ostream& os) -> void {
  // Canonical form: every calibration is written in the `years_bp` convention with an explicit
  // reference year, so that re-reading gives the same reference instant and the same density.  The
  // reference instant of every convention is 1 January of some year, which is what `years_bp` takes.
  os << "version 1\n";
  for (const auto& c : set.calibrations) {
    auto reference_year = int64_t{0};
    switch (c.units) {
      case Age_units::ka:
      case Age_units::years_bp: reference_year = c.reference_year.value(); break;
      case Age_units::cal_bp: reference_year = k_cal_bp_reference_year; break;
      case Age_units::ce:
      case Age_units::bce: reference_year = 2020; break;
    }
    os << "\ncalibration " << c.name << "\n";
    if (c.all_tips) {
      os << "  taxa all\n";
    } else {
      for (const auto& t : c.taxa) { os << "  taxa " << t << "\n"; }
    }
    if (not c.require_monophyly) { os << "  monophyly not_required\n"; }
    os << "  age_units years_bp\n";
    os << absl::StreamFormat("  age_reference_year %d\n", reference_year);
    os << "  distribution " << to_string(c.prior.distribution) << "\n";
    switch (c.prior.distribution) {
      case Calibration_distribution::normal:
        os << absl::StreamFormat("  mean %.17g\n", c.prior.mean);
        os << absl::StreamFormat("  stddev %.17g\n", c.prior.stddev);
        break;
      case Calibration_distribution::lognormal:
        os << absl::StreamFormat("  median %.17g\n", c.prior.median);
        os << absl::StreamFormat("  sigma %.17g\n", c.prior.sigma);
        break;
      case Calibration_distribution::uniform:
        break;
    }
    if (std::isfinite(c.prior.oldest_age)) {
      os << absl::StreamFormat("  oldest %.17g\n", c.prior.oldest_age);
    }
    if (std::isfinite(c.prior.youngest_age)) {
      os << absl::StreamFormat("  youngest %.17g\n", c.prior.youngest_age);
    }
    os << "end\n";
  }
}

auto calibration_config_to_string(const Calibration_set& set) -> std::string {
  auto oss = std::ostringstream{};
  write_calibration_config(set, oss);
  return oss.str();
}

// ---------------------------------------------------------------------------------------------
// Binding to tree nodes
// ---------------------------------------------------------------------------------------------

namespace {

struct Clade_counts {
  Node_vector<int> marked_below;
  Node_vector<int> tips_below;
};

auto count_clade(const Phylo_tree& tree, const absl::flat_hash_set<Node_index>& marked) -> Clade_counts {
  auto result = Clade_counts{};
  result.marked_below.assign(tree.size(), 0);
  result.tips_below.assign(tree.size(), 0);
  for (const auto& node : post_order_traversal(tree)) {
    if (tree.at(node).is_tip()) {
      result.tips_below[node] = 1;
      result.marked_below[node] = marked.contains(node) ? 1 : 0;
    } else {
      for (const auto& child : tree.at(node).children) {
        result.tips_below[node] += result.tips_below[child];
        result.marked_below[node] += result.marked_below[child];
      }
    }
  }
  return result;
}

auto is_strict_ancestor(const Phylo_tree& tree, Node_index ancestor, Node_index descendant) -> bool {
  auto node = tree.at(descendant).parent;
  while (node != k_no_node) {
    if (node == ancestor) { return true; }
    node = tree.at(node).parent;
  }
  return false;
}

auto min_tip_time_below(const Phylo_tree& tree, Node_index root_of_clade) -> double {
  auto result = +k_inf;
  auto stack = std::vector<Node_index>{root_of_clade};
  while (not stack.empty()) {
    auto node = stack.back();
    stack.pop_back();
    if (tree.at(node).is_tip()) {
      result = std::min(result, tree.at(node).t);
    } else {
      for (const auto& child : tree.at(node).children) { stack.push_back(child); }
    }
  }
  return result;
}

auto do_bind(const Calibration_set& set, const Phylo_tree& tree, bool check_times)
    -> std::vector<Node_index> {

  auto problems = std::vector<std::string>{};
  auto tip_index = absl::flat_hash_map<std::string, Node_index>{};
  auto num_tips = 0;
  for (const auto& node : index_order_traversal(tree)) {
    if (not tree.at(node).is_tip()) { continue; }
    ++num_tips;
    auto [it, inserted] = tip_index.emplace(tree.at(node).name, node);
    if (not inserted) {
      problems.push_back(absl::StrFormat(
          "The tree has more than one tip named '%s'; calibrations cannot be resolved", tree.at(node).name));
    }
  }

  auto nodes = std::vector<Node_index>(std::ssize(set.calibrations), k_no_node);

  for (auto i = 0; i != std::ssize(set.calibrations); ++i) {
    const auto& c = set.calibrations[i];

    if (c.all_tips) {
      nodes[i] = tree.root;
      continue;
    }

    auto marked = absl::flat_hash_set<Node_index>{};
    auto unknown = std::vector<std::string>{};
    for (const auto& name : c.taxa) {
      auto it = tip_index.find(name);
      if (it == tip_index.end()) { unknown.push_back(name); continue; }
      marked.insert(it->second);
    }
    if (not unknown.empty()) {
      problems.push_back(absl::StrFormat(
          "calibration '%s': %d sample ID(s) are not tips of the tree: %s.  Calibration targets are defined by "
          "sample IDs, so every ID must appear in the alignment and in the tree",
          c.name, std::ssize(unknown), list_names(unknown)));
      continue;
    }

    auto counts = count_clade(tree, marked);
    auto size = static_cast<int>(std::ssize(marked));
    auto mrca = k_no_node;
    auto best_tips = std::numeric_limits<int>::max();
    for (const auto& node : index_order_traversal(tree)) {
      if (counts.marked_below[node] == size && counts.tips_below[node] < best_tips) {
        mrca = node;
        best_tips = counts.tips_below[node];
      }
    }
    CHECK_NE(mrca, k_no_node);
    if (tree.at(mrca).is_tip()) {
      problems.push_back(absl::StrFormat(
          "calibration '%s': its taxon set resolves to a tip, not an ancestor", c.name));
      continue;
    }
    if (c.require_monophyly && counts.tips_below[mrca] != size) {
      auto extra = std::vector<std::string>{};
      auto stack = std::vector<Node_index>{mrca};
      while (not stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (tree.at(node).is_tip()) {
          if (not marked.contains(node)) { extra.push_back(tree.at(node).name); }
        } else {
          for (const auto& child : tree.at(node).children) { stack.push_back(child); }
        }
      }
      std::ranges::sort(extra);
      problems.push_back(absl::StrFormat(
          "calibration '%s': the %d listed sample(s) are not monophyletic in this tree.  Their common ancestor also "
          "has %d other sample(s) below it: %s.  Fix the taxon set or the input tree; Delphy will not calibrate the "
          "age of a paraphyletic group",
          c.name, size, std::ssize(extra), list_names(extra)));
      continue;
    }
    nodes[i] = mrca;
  }

  // Distinct targets.  A calibration that does not require monophyly may share its MRCA with another one, since
  // which node that is depends on the topology; the two densities then both apply to that node.
  for (auto i = 0; i != std::ssize(nodes); ++i) {
    if (nodes[i] == k_no_node || not set.calibrations[i].require_monophyly) { continue; }
    for (auto j = i + 1; j != std::ssize(nodes); ++j) {
      if (nodes[j] != nodes[i] || not set.calibrations[j].require_monophyly) { continue; }
      problems.push_back(absl::StrFormat(
          "calibrations '%s' and '%s' both resolve to the same node of the tree.  Give one calibration per node",
          set.calibrations[i].name, set.calibrations[j].name));
    }
  }

  // Mutual compatibility of the supports along the tree, and against the tips below each node
  for (auto i = 0; i != std::ssize(nodes); ++i) {
    if (nodes[i] == k_no_node) { continue; }
    const auto& ci = set.calibrations[i];
    for (auto j = 0; j != std::ssize(nodes); ++j) {
      if (i == j || nodes[j] == k_no_node) { continue; }
      if (not is_strict_ancestor(tree, nodes[i], nodes[j])) { continue; }
      const auto& cj = set.calibrations[j];
      // ci is an ancestor of cj, so we need t_i < t_j to be possible.
      if (not (ci.t_min() < cj.t_max())) {
        problems.push_back(absl::StrFormat(
            "calibrations '%s' and '%s' cannot both be satisfied: '%s' is an ancestor of '%s', so it must be older, "
            "but the oldest age allowed for '%s' (%g years) is not older than the youngest age allowed for '%s' "
            "(%g years)",
            ci.name, cj.name, ci.name, cj.name,
            ci.name, ci.prior.support_oldest_age(), cj.name, cj.prior.support_youngest_age()));
      }
    }
    auto min_tip_t = min_tip_time_below(tree, nodes[i]);
    if (not (ci.t_min() < min_tip_t)) {
      problems.push_back(absl::StrFormat(
          "calibration '%s' cannot be satisfied: its node must be older than every sample below it (the oldest is at "
          "%g days since 2020-01-01, i.e. age %g years on this calibration's scale), but the oldest age the prior "
          "allows is %g years",
          ci.name, min_tip_t, ci.age_of(min_tip_t), ci.prior.support_oldest_age()));
    }
  }

  if (check_times) {
    for (auto i = 0; i != std::ssize(nodes); ++i) {
      if (nodes[i] == k_no_node) { continue; }
      const auto& c = set.calibrations[i];
      auto t = tree.at(nodes[i]).t;
      if (not std::isfinite(c.log_density_at(t))) {
        problems.push_back(absl::StrFormat(
            "calibration '%s': the starting tree puts its node at age %g years, which is outside the hard bounds of "
            "the prior (%g to %g years).  The run would start at log posterior = -infinity.  Widen the bounds or "
            "supply a starting tree whose node ages are inside them; Delphy will not rescale the starting tree",
            c.name, c.age_of(t), c.prior.support_youngest_age(), c.prior.support_oldest_age()));
      }
    }
  }

  if (not problems.empty()) { throw_problems(problems); }
  return nodes;
}

}  // namespace

auto bind_calibrations(const Calibration_set& set, const Phylo_tree& tree) -> std::vector<Node_index> {
  return do_bind(set, tree, true);
}

auto bind_calibrations_ignoring_times(const Calibration_set& set, const Phylo_tree& tree)
    -> std::vector<Node_index> {
  return do_bind(set, tree, false);
}

auto calc_log_calibration_prior(
    const Calibration_set& set,
    const std::vector<Node_index>& nodes,
    const Phylo_tree& tree)
    -> double {
  CHECK_EQ(std::ssize(set.calibrations), std::ssize(nodes));
  auto result = 0.0;
  for (auto i = 0; i != std::ssize(nodes); ++i) {
    CHECK_NE(nodes[i], k_no_node);
    result += set.calibrations[i].log_density_at(tree.at(nodes[i]).t);
  }
  return result;
}

}  // namespace delphy
