#include "site_partition.h"

#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

#include "phylo_tree_calc.h"

namespace delphy {

namespace {

constexpr auto k_max_problems_shown = 25;

class Problems {
 public:
  auto add(std::string message) -> void { items_.push_back(std::move(message)); }
  auto empty() const -> bool { return items_.empty(); }

  auto throw_if_any(std::string_view header) const -> void {
    if (items_.empty()) { return; }
    auto text = absl::StrFormat("%s: %d problem%s found\n", header, items_.size(),
                                items_.size() == 1 ? "" : "s");
    auto shown = 0;
    for (const auto& item : items_) {
      if (shown == k_max_problems_shown) {
        absl::StrAppendFormat(&text, "  ... and %d more\n", std::ssize(items_) - shown);
        break;
      }
      absl::StrAppendFormat(&text, "  %d. %s\n", ++shown, item);
    }
    throw Site_partition_error{text};
  }

 private:
  std::vector<std::string> items_;
};

auto trimmed(std::string_view sv) -> std::string { return std::string{absl::StripAsciiWhitespace(sv)}; }

auto merge_intervals(std::vector<Site_interval> intervals) -> std::vector<Site_interval> {
  std::ranges::sort(intervals);
  auto merged = std::vector<Site_interval>{};
  for (const auto& iv : intervals) {
    if (not merged.empty() && iv.first <= merged.back().second) {
      merged.back().second = std::max(merged.back().second, iv.second);
    } else {
      merged.push_back(iv);
    }
  }
  return merged;
}

auto parse_site_list(std::string_view text, int line_number, const std::string& name, Problems& problems)
    -> std::vector<Site_interval> {
  auto intervals = std::vector<Site_interval>{};
  for (auto token : absl::StrSplit(text, ',')) {
    auto item = trimmed(token);
    if (item.empty()) { continue; }
    auto parts = std::vector<std::string>{absl::StrSplit(item, '-')};
    auto first = int64_t{}, last = int64_t{};
    auto ok = (parts.size() == 1 && absl::SimpleAtoi(parts[0], &first) && (last = first, true)) ||
              (parts.size() == 2 && absl::SimpleAtoi(trimmed(parts[0]), &first) &&
               absl::SimpleAtoi(trimmed(parts[1]), &last));
    if (not ok) {
      problems.add(absl::StrFormat(
          "Site partition line %d (partition '%s'): '%s' is not a site ('310') or a range ('303-315', "
          "1-based, inclusive)", line_number, name, item));
      continue;
    }
    if (first < 1 || last < first) {
      problems.add(absl::StrFormat(
          "Site partition line %d (partition '%s'): range %d-%d is empty or reversed; sites are 1-based "
          "and the first bound must not be larger than the second", line_number, name, first, last));
      continue;
    }
    intervals.push_back({static_cast<Site_index>(first - 1), static_cast<Site_index>(last)});
  }
  if (intervals.empty()) {
    problems.add(absl::StrFormat("Site partition line %d (partition '%s'): no site given", line_number, name));
  }
  return intervals;
}

auto parse_constant_sites(std::string_view text, int line_number, const std::string& name, Problems& problems)
    -> Seq_vector<Site_index> {
  auto counts = Seq_vector<Site_index>{-1, -1, -1, -1};
  for (auto token : absl::StrSplit(text, absl::ByAnyChar(", \t"))) {
    auto item = trimmed(token);
    if (item.empty()) { continue; }
    auto kv = std::vector<std::string>{absl::StrSplit(item, '=')};
    auto count = int64_t{};
    if (kv.size() != 2 || trimmed(kv[0]).size() != 1 || not absl::SimpleAtoi(trimmed(kv[1]), &count)) {
      problems.add(absl::StrFormat(
          "Site partition line %d (partition '%s'): constant_sites entry '%s' is not of the form 'A=1000'",
          line_number, name, item));
      continue;
    }
    auto letter = absl::AsciiStrToUpper(trimmed(kv[0]))[0];
    if (letter != 'A' && letter != 'C' && letter != 'G' && letter != 'T') {
      problems.add(absl::StrFormat("Site partition line %d (partition '%s'): '%c' is not a nucleotide",
                                   line_number, name, letter));
      continue;
    }
    auto a = char_to_real_seq_letter(letter);
    if (counts[a] >= 0) {
      problems.add(absl::StrFormat("Site partition line %d (partition '%s'): state %c is given twice",
                                   line_number, name, letter));
      continue;
    }
    if (count < 0) {
      problems.add(absl::StrFormat("Site partition line %d (partition '%s'): state %c has a negative count (%d)",
                                   line_number, name, letter, count));
      continue;
    }
    counts[a] = static_cast<Site_index>(count);
  }
  for (auto a : k_all_real_seq_letters) {
    if (counts[a] < 0) {
      problems.add(absl::StrFormat(
          "Site partition line %d (partition '%s'): constant_sites is missing state %c.  Give all four "
          "counts, for example 'A=4000,C=3000,G=3100,T=2400'.", line_number, name, to_char(a)));
      counts[a] = 0;
    }
  }
  return counts;
}

}  // namespace

auto Site_partition_config::total_constant_sites() const -> Site_index {
  auto total = Site_index{0};
  for (const auto& block : blocks) {
    for (auto a : k_all_real_seq_letters) { total += block.constant_sites[a]; }
  }
  return total;
}

auto Site_partition_config::constant_sites_per_block() const -> std::vector<Seq_vector<Site_index>> {
  auto any = std::ranges::any_of(blocks, [](const Site_partition_block& b) { return b.has_constant_sites; });
  if (not any) { return {}; }
  auto result = std::vector<Seq_vector<Site_index>>{};
  for (const auto& block : blocks) { result.push_back(block.constant_sites); }
  return result;
}

auto Site_partition::total_observed_sites() const -> Site_index {
  auto total = Site_index{0};
  for (auto n : num_observed_sites) { total += n; }
  return total;
}

auto read_site_partition_config(std::istream& is) -> Site_partition_config {
  auto problems = Problems{};
  auto config = Site_partition_config{};
  auto seen_version = false;
  auto seen_names = absl::flat_hash_map<std::string, int>{};  // lower-case name -> line
  auto cur = std::optional<Site_partition_block>{};
  auto line_number = 0;

  for (auto line = std::string{}; std::getline(is, line);) {
    ++line_number;
    if (not line.empty() && line.back() == '\r') { line.pop_back(); }
    if (auto hash = line.find('#'); hash != std::string::npos) { line.erase(hash); }
    auto text = trimmed(line);
    if (text.empty()) { continue; }

    auto space = text.find_first_of(" \t");
    auto key = text.substr(0, space);
    auto value = space == std::string::npos ? std::string{} : trimmed(text.substr(space));

    if (key == "version") {
      if (value != "1") {
        problems.add(absl::StrFormat("Site partition line %d: unsupported version '%s'; this Delphy reads "
                                     "version 1", line_number, value));
      }
      seen_version = true;
      continue;
    }
    if (key == "partition") {
      if (cur.has_value()) {
        problems.add(absl::StrFormat("Site partition line %d: partition '%s' starts before '%s' ended; every "
                                     "partition block ends with 'end'", line_number, value, cur->name));
        config.blocks.push_back(std::move(*cur));
        cur.reset();
      }
      if (value.empty()) {
        problems.add(absl::StrFormat("Site partition line %d: a partition needs a name", line_number));
        value = absl::StrFormat("partition_%d", std::ssize(config.blocks) + 1);
      }
      if (auto [it, inserted] = seen_names.emplace(absl::AsciiStrToLower(value), line_number); not inserted) {
        problems.add(absl::StrFormat("Site partition line %d: partition name '%s' repeats (up to letter case) "
                                     "line %d; names must be unique", line_number, value, it->second));
      }
      cur = Site_partition_block{};
      cur->name = value;
      cur->config_line = line_number;
      continue;
    }
    if (key == "end") {
      if (not cur.has_value()) {
        problems.add(absl::StrFormat("Site partition line %d: 'end' without a partition", line_number));
        continue;
      }
      if (cur->sites.empty()) {
        problems.add(absl::StrFormat("Site partition line %d: partition '%s' lists no site", line_number,
                                     cur->name));
      }
      cur->sites = merge_intervals(std::move(cur->sites));
      config.blocks.push_back(std::move(*cur));
      cur.reset();
      continue;
    }
    if (not cur.has_value()) {
      problems.add(absl::StrFormat("Site partition line %d: '%s' outside a partition block", line_number, key));
      continue;
    }
    if (key == "sites") {
      auto more = parse_site_list(value, line_number, cur->name, problems);
      cur->sites.insert(cur->sites.end(), more.begin(), more.end());
      continue;
    }
    if (key == "constant_sites") {
      if (cur->has_constant_sites) {
        problems.add(absl::StrFormat("Site partition line %d (partition '%s'): constant_sites is given twice",
                                     line_number, cur->name));
        continue;
      }
      cur->constant_sites = parse_constant_sites(value, line_number, cur->name, problems);
      cur->has_constant_sites = true;
      continue;
    }
    problems.add(absl::StrFormat("Site partition line %d: unknown key '%s'; a partition block has 'sites' and "
                                 "optionally 'constant_sites'", line_number, key));
  }

  if (cur.has_value()) {
    problems.add(absl::StrFormat("Partition '%s' (line %d) is not closed with 'end'", cur->name,
                                 cur->config_line));
    config.blocks.push_back(std::move(*cur));
  }
  if (not seen_version) {
    problems.add("The configuration must start with 'version 1'");
  }
  if (config.blocks.empty()) {
    problems.add("The configuration declares no partition");
  }
  {
    auto with = 0;
    for (const auto& block : config.blocks) {
      if (block.has_constant_sites) { ++with; }
    }
    if (with != 0 && with != std::ssize(config.blocks)) {
      problems.add(absl::StrFormat(
          "%d of %d partitions declare constant_sites.  Restored constant sites must be declared for every "
          "partition or for none: a global count cannot be split between partitions.",
          with, config.blocks.size()));
    }
  }
  problems.throw_if_any("Site partition configuration rejected");
  return config;
}

auto write_site_partition_config(const Site_partition_config& config, std::ostream& os) -> void {
  os << "version 1\n";
  for (const auto& block : config.blocks) {
    os << "\npartition " << block.name << "\n";
    auto sites = std::vector<std::string>{};
    for (const auto& [start, end] : block.sites) {
      sites.push_back(end - start == 1 ? absl::StrFormat("%d", start + 1)
                                       : absl::StrFormat("%d-%d", start + 1, end));
    }
    os << "  sites " << absl::StrJoin(sites, ",") << "\n";
    if (block.has_constant_sites) {
      os << absl::StreamFormat("  constant_sites A=%d,C=%d,G=%d,T=%d\n",
                               block.constant_sites[Real_seq_letter::A], block.constant_sites[Real_seq_letter::C],
                               block.constant_sites[Real_seq_letter::G], block.constant_sites[Real_seq_letter::T]);
    }
    os << "end\n";
  }
}

auto site_partition_config_to_string(const Site_partition_config& config) -> std::string {
  auto os = std::ostringstream{};
  write_site_partition_config(config, os);
  return os.str();
}

auto make_single_site_partition(const std::vector<bool>& site_is_observed) -> Site_partition {
  auto result = Site_partition{};
  auto L = static_cast<Site_index>(std::ssize(site_is_observed));
  result.names = {"all"};
  result.partition_for_site.assign(L, 0);
  result.num_sites = {L};
  auto observed = Site_index{0};
  for (auto l = Site_index{0}; l != L; ++l) {
    if (site_is_observed[l]) { ++observed; }
  }
  result.num_observed_sites = {observed};
  return result;
}

auto resolve_site_partition(
    const Site_partition_config& config,
    Site_index num_input_sites,
    const std::vector<bool>& site_is_observed)
    -> Site_partition {
  auto problems = Problems{};
  auto num_sites = static_cast<Site_index>(std::ssize(site_is_observed));
  auto num_partitions = config.size();

  auto owner = Site_vector<Partition_index>(num_sites, -1);
  for (auto p = Partition_index{0}; p != num_partitions; ++p) {
    const auto& block = config.blocks[p];
    for (const auto& [start, end] : block.sites) {
      if (end > num_input_sites) {
        problems.add(absl::StrFormat(
            "Partition '%s' (line %d) claims sites %d-%d, but the input reference has %d sites.  Partitions "
            "are defined in the coordinates of the input reference.",
            block.name, block.config_line, start + 1, end, num_input_sites));
        continue;
      }
      for (auto l = start; l != end; ++l) {
        if (owner[l] >= 0) {
          problems.add(absl::StrFormat(
              "Site %d is claimed by partition '%s' and by partition '%s'; partitions must not overlap",
              l + 1, config.blocks[owner[l]].name, block.name));
          continue;
        }
        owner[l] = p;
      }
    }
  }

  {
    auto unassigned = std::vector<Site_index>{};
    for (auto l = Site_index{0}; l != num_input_sites; ++l) {
      if (owner[l] < 0) { unassigned.push_back(l); }
    }
    if (not unassigned.empty()) {
      auto shown = std::vector<std::string>{};
      for (auto i = 0; i != std::min<int>(5, std::ssize(unassigned)); ++i) {
        shown.push_back(absl::StrFormat("%d", unassigned[i] + 1));
      }
      problems.add(absl::StrFormat(
          "%d of the %d input sites belong to no partition (first: %s%s).  Every modelled site must be in "
          "exactly one partition; sites that should not be modelled are excluded with a site mask or left out "
          "of the callable sites, not left out of the partitions.",
          unassigned.size(), num_input_sites, absl::StrJoin(shown, ", "),
          std::ssize(unassigned) > 5 ? ", ..." : ""));
    }
  }

  // The restored constant-site blocks follow the input sites, in configuration order.
  {
    auto declared = config.total_constant_sites();
    if (num_input_sites + declared != num_sites) {
      problems.add(absl::StrFormat(
          "The partitions declare %d restored constant site%s, which with the %d input sites make %d sites, "
          "but the analysed sequence has %d.  The declared counts must match the input that was loaded.",
          declared, declared == 1 ? "" : "s", num_input_sites, num_input_sites + declared, num_sites));
    } else {
      auto l = num_input_sites;
      for (auto p = Partition_index{0}; p != num_partitions; ++p) {
        for (auto a : k_all_real_seq_letters) {
          for (auto i = Site_index{0}; i != config.blocks[p].constant_sites[a]; ++i) { owner[l++] = p; }
        }
      }
    }
  }
  problems.throw_if_any("Site partition configuration rejected");

  auto result = Site_partition{};
  result.partition_for_site = std::move(owner);
  result.num_sites.assign(num_partitions, 0);
  result.num_observed_sites.assign(num_partitions, 0);
  for (const auto& block : config.blocks) { result.names.push_back(block.name); }
  for (auto l = Site_index{0}; l != num_sites; ++l) {
    auto p = result.partition_for_site[l];
    ++result.num_sites[p];
    if (site_is_observed[l]) { ++result.num_observed_sites[p]; }
  }

  for (auto p = Partition_index{0}; p != num_partitions; ++p) {
    if (result.num_observed_sites[p] == 0) {
      problems.add(absl::StrFormat(
          "Partition '%s' has %d site%s but not one of them is observed in any sample, so its relative rate "
          "is not identified by the data.  Remove the partition, or merge it with another one.",
          result.names[p], result.num_sites[p], result.num_sites[p] == 1 ? "" : "s"));
    }
  }
  problems.throw_if_any("Site partition configuration rejected");
  return result;
}

auto observed_sites_of_tree(const Phylo_tree& tree) -> std::vector<bool> {
  auto result = std::vector<bool>(tree.num_sites(), true);
  for (const auto& [start, end] : tree.at_root().missations.intervals) {
    for (auto l = start; l != end; ++l) { result[l] = false; }
  }
  return result;
}

auto weighted_mean_site_partition_rate(const Site_partition& partition, const std::vector<double>& rates)
    -> double {
  auto total_weight = 0.0;
  auto weighted = 0.0;
  for (auto p = Partition_index{0}; p != partition.num_partitions(); ++p) {
    auto weight = static_cast<double>(partition.num_observed_sites[p]);
    total_weight += weight;
    weighted += weight * rates[p];
  }
  return weighted / total_weight;
}

auto normalize_site_partition_rates(const Site_partition& partition, std::vector<double> rates)
    -> std::vector<double> {
  if (std::ssize(rates) != partition.num_partitions()) {
    throw Site_partition_error{absl::StrFormat(
        "%d relative rates were given for %d partitions", rates.size(), partition.num_partitions())};
  }
  for (auto p = Partition_index{0}; p != partition.num_partitions(); ++p) {
    if (not (std::isfinite(rates[p]) && rates[p] > 0.0)) {
      throw Site_partition_error{absl::StrFormat(
          "The relative rate of partition '%s' must be finite and positive, got %g", partition.names[p],
          rates[p])};
    }
  }
  auto mean = weighted_mean_site_partition_rate(partition, rates);
  if (not (std::isfinite(mean) && mean > 0.0)) {
    throw Site_partition_error{"The weighted mean of the relative rates is not finite and positive"};
  }
  for (auto& r : rates) { r /= mean; }
  return rates;
}

}  // namespace delphy
