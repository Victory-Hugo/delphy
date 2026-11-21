#include "human_input.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <ranges>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

#include "dates.h"
#include "sequence_utils.h"

namespace delphy {

namespace {

constexpr auto k_max_problems_shown = 25;
constexpr auto k_max_ids_listed = 8;

class Problems {
 public:
  auto add(std::string message) -> void { items_.push_back(std::move(message)); }
  auto empty() const -> bool { return items_.empty(); }
  auto size() const -> int { return static_cast<int>(items_.size()); }

  auto throw_if_any(std::string_view header) const -> void {
    if (items_.empty()) { return; }
    auto msg = absl::StrFormat("%s: %d problem%s found\n", header, items_.size(), items_.size() == 1 ? "" : "s");
    for (auto i = 0; i != std::min<int>(std::ssize(items_), k_max_problems_shown); ++i) {
      absl::StrAppendFormat(&msg, "  %d. %s\n", i + 1, items_[i]);
    }
    if (std::ssize(items_) > k_max_problems_shown) {
      absl::StrAppendFormat(&msg, "  ... and %d more\n", std::ssize(items_) - k_max_problems_shown);
    }
    throw Human_input_error(msg);
  }

 private:
  std::vector<std::string> items_;
};

auto list_ids(const std::vector<std::string>& ids) -> std::string {
  auto shown = std::min<int>(std::ssize(ids), k_max_ids_listed);
  auto result = absl::StrJoin(ids.begin(), ids.begin() + shown, ", ");
  if (std::ssize(ids) > shown) { absl::StrAppendFormat(&result, ", ... (%d in total)", std::ssize(ids)); }
  return result;
}

auto fnv1a64_hex(const Real_sequence& seq) -> std::string {
  auto h = uint64_t{0xcbf29ce484222325ULL};
  for (auto s : seq) {
    h ^= static_cast<uint64_t>(to_char(s));
    h *= 0x100000001b3ULL;
  }
  return absl::StrFormat("%016x", h);
}

// Characters that would corrupt Newick / NEXUS / BEAST output or the TSV and JSON files written alongside
auto check_sample_id(std::string_view id) -> std::optional<std::string> {
  if (id.empty()) { return "empty sample ID"; }
  for (auto c : id) {
    if (std::iscntrl(static_cast<unsigned char>(c)) || std::strchr("(),:;[]'\"\\", c) != nullptr) {
      return absl::StrFormat("sample ID '%s' contains the character '%c', which is not allowed "
                             "(control characters and ( ) , : ; [ ] ' \" \\ are rejected)", id, c);
    }
  }
  return std::nullopt;
}

auto first_token(std::string_view line) -> std::string {
  auto begin = line.find_first_not_of(" \t\r");
  if (begin == std::string_view::npos) { return {}; }
  auto end = line.find_first_of(" \t\r", begin);
  return std::string{line.substr(begin, end == std::string_view::npos ? std::string_view::npos : end - begin)};
}

//
// Raw parsed input (before dates, masking and consistency checks)
//

struct Raw_sample {
  std::string id;
  Tip_desc tip;
  Site_index num_iupac_ambiguous = -1;
  Site_index num_unknown = -1;
  Site_index num_gap = -1;
};

struct Raw_input {
  std::string format;
  std::string observation_mode;
  std::string reference_id;
  Real_sequence reference;
  std::vector<Raw_sample> samples;
};

struct Fasta_record {
  std::string id;
  Sequence sequence;
  Site_index num_iupac_ambiguous = 0;
  Site_index num_unknown = 0;
  Site_index num_gap = 0;
};

auto parse_fasta_records(std::istream& is, std::string_view what, Problems& problems) -> std::vector<Fasta_record> {
  auto records = std::vector<Fasta_record>{};
  auto in_record = false;
  auto bad_chars_in_record = 0;
  auto line_number = 0;

  for (auto line = std::string{}; std::getline(is, line);) {
    ++line_number;
    if (line.empty() || line[0] == ';') { continue; }

    if (line[0] == '>') {
      records.emplace_back();
      records.back().id = first_token(std::string_view{line}.substr(1));
      in_record = true;
      bad_chars_in_record = 0;
      continue;
    }

    if (not in_record) {
      problems.add(absl::StrFormat("%s line %d: sequence data before the first '>' header line", what, line_number));
      return records;
    }

    auto& rec = records.back();
    for (auto c : line) {
      if (std::isspace(static_cast<unsigned char>(c))) { continue; }
      switch (std::toupper(static_cast<unsigned char>(c))) {
        case 'A': case 'C': case 'G': case 'T':
          break;
        case 'N': case '?':
          ++rec.num_unknown; break;
        case '-': case '.':
          ++rec.num_gap; break;
        case 'R': case 'Y': case 'S': case 'W': case 'K': case 'M': case 'B': case 'D': case 'H': case 'V':
          ++rec.num_iupac_ambiguous; break;
        default:
          if (++bad_chars_in_record <= 3) {
            problems.add(absl::StrFormat(
                "%s, sample '%s', line %d: invalid sequence character '%c' at alignment column %d "
                "(accepted: A C G T, IUPAC ambiguity codes, N ? - .)",
                what, rec.id, line_number, c, std::ssize(rec.sequence) + 1));
          }
          continue;
      }
      rec.sequence.push_back(to_seq_letter(c));
    }
  }
  return records;
}

auto reference_from_record(const Fasta_record& rec, std::string_view what, Problems& problems) -> Real_sequence {
  auto result = Real_sequence{};
  if (rec.num_iupac_ambiguous + rec.num_unknown + rec.num_gap > 0) {
    problems.add(absl::StrFormat(
        "%s '%s' contains %d ambiguous, %d unknown and %d gap characters.  The reference must consist of A, C, G "
        "and T only; unknown bases cannot be recorded as reference states.",
        what, rec.id, rec.num_iupac_ambiguous, rec.num_unknown, rec.num_gap));
    return result;
  }
  result.reserve(rec.sequence.size());
  for (auto s : rec.sequence) { result.push_back(to_real_seq_letter(s)); }
  return result;
}

auto read_raw_fasta(std::istream& fasta_is, std::istream* reference_is, const Human_input_config& config)
    -> Raw_input {
  auto problems = Problems{};
  auto raw = Raw_input{};
  raw.format = "fasta";
  raw.observation_mode = "full_alignment";

  auto records = parse_fasta_records(fasta_is, "Input FASTA", problems);
  problems.throw_if_any("Input FASTA rejected");
  if (records.empty()) {
    throw Human_input_error("Input FASTA rejected: no sequences found");
  }

  if (reference_is != nullptr && config.reference_id.has_value()) {
    throw Human_input_error("Give the reference either as a separate FASTA file or as a record ID within the input "
                            "FASTA, not both");
  }
  if (reference_is == nullptr && not config.reference_id.has_value()) {
    throw Human_input_error("A reference sequence is required for FASTA input: give a separate reference FASTA "
                            "file (--v0-human-reference-fasta) or the ID of the reference record within the "
                            "input FASTA (--v0-human-reference-id)");
  }

  if (reference_is != nullptr) {
    auto ref_records = parse_fasta_records(*reference_is, "Reference FASTA", problems);
    problems.throw_if_any("Reference FASTA rejected");
    if (ref_records.size() != 1) {
      throw Human_input_error(absl::StrFormat(
          "Reference FASTA rejected: it must contain exactly one record, found %d", ref_records.size()));
    }
    raw.reference_id = ref_records[0].id;
    raw.reference = reference_from_record(ref_records[0], "Reference record", problems);
  } else {
    auto it = std::ranges::find(records, *config.reference_id, &Fasta_record::id);
    if (it == records.end()) {
      throw Human_input_error(absl::StrFormat(
          "Reference record '%s' not found in the input FASTA", *config.reference_id));
    }
    raw.reference_id = it->id;
    raw.reference = reference_from_record(*it, "Reference record", problems);
    records.erase(it);
  }
  problems.throw_if_any("Reference rejected");

  auto ignore_warnings = [](const std::string&, const Sequence_warning&) {};
  raw.samples.reserve(records.size());
  for (auto& rec : records) {
    auto sample = Raw_sample{};
    sample.id = rec.id;
    sample.num_iupac_ambiguous = rec.num_iupac_ambiguous;
    sample.num_unknown = rec.num_unknown;
    sample.num_gap = rec.num_gap;
    if (std::ssize(rec.sequence) != std::ssize(raw.reference)) {
      problems.add(absl::StrFormat(
          "Sample '%s' has %d sites but the reference '%s' has %d.  All sequences must be aligned to the reference "
          "and have its length (reference coordinates).",
          rec.id, std::ssize(rec.sequence), raw.reference_id, std::ssize(raw.reference)));
    } else {
      auto delta = calculate_delta_from_reference(rec.id, rec.sequence, raw.reference, ignore_warnings);
      sample.tip.seq_deltas = std::move(delta.seq_deltas);
      sample.tip.missations = std::move(delta.missations);
    }
    raw.samples.push_back(std::move(sample));
  }
  problems.throw_if_any("Input FASTA rejected");
  return raw;
}

auto read_raw_maple(std::istream& is, const Human_input_config& config) -> Raw_input {
  if (config.reference_id.has_value()) {
    throw Human_input_error("A reference record ID applies to FASTA input only; MAPLE files carry their own "
                            "full-length reference as the first record");
  }

  auto problems = Problems{};
  auto raw = Raw_input{};
  raw.format = "maple";
  raw.observation_mode = "reference_plus_deltas";

  auto line = std::string{};
  auto line_number = 1;
  if (not std::getline(is, line) || line.empty() || line[0] != '>') {
    throw Human_input_error("MAPLE input rejected: the first line must be '>' followed by the reference ID, "
                            "then the full-length reference sequence");
  }
  raw.reference_id = first_token(std::string_view{line}.substr(1));

  auto have_line = static_cast<bool>(std::getline(is, line));
  ++line_number;
  while (have_line && (line.empty() || line[0] != '>')) {
    for (auto c : line) {
      if (std::isspace(static_cast<unsigned char>(c))) { continue; }
      auto u = std::toupper(static_cast<unsigned char>(c));
      if (u == 'A' || u == 'C' || u == 'G' || u == 'T') {
        raw.reference.push_back(char_to_real_seq_letter(static_cast<char>(u)));
      } else {
        problems.add(absl::StrFormat(
            "MAPLE reference, line %d: character '%c' at reference site %d is not A, C, G or T.  The reference must "
            "be fully resolved; unknown bases cannot be recorded as reference states.",
            line_number, c, std::ssize(raw.reference) + 1));
        raw.reference.push_back(Real_seq_letter::A);
      }
    }
    have_line = static_cast<bool>(std::getline(is, line));
    ++line_number;
  }
  problems.throw_if_any("MAPLE input rejected");
  if (raw.reference.empty()) {
    throw Human_input_error("MAPLE input rejected: empty reference sequence");
  }
  auto L = static_cast<Site_index>(std::ssize(raw.reference));

  while (have_line) {
    // `line` is a '>' line here
    auto sample = Raw_sample{};
    sample.id = first_token(std::string_view{line}.substr(1));
    auto sample_line = line_number;
    auto mutated_sites = absl::flat_hash_set<Site_index>{};

    while ((have_line = static_cast<bool>(std::getline(is, line)), ++line_number, have_line)) {
      if (line.empty()) { continue; }
      if (line[0] == '>') { break; }

      auto ss = std::istringstream{line};
      auto c = char{};
      ss >> c;
      auto what = absl::StrFormat("MAPLE sample '%s', line %d", sample.id, line_number);
      if (not ss || std::toupper(static_cast<unsigned char>(c)) == 'U') {
        problems.add(absl::StrFormat("%s: cannot parse state letter in '%s'", what, line));
        continue;
      }
      auto s = Seq_letter{};
      try {
        s = to_seq_letter(c);
      } catch (const std::invalid_argument&) {
        problems.add(absl::StrFormat("%s: invalid state '%c'", what, c));
        continue;
      }

      if (is_ambiguous(s)) {
        auto start = Site_index{};
        ss >> start;
        auto len = Site_index{1};
        if (ss) { ss >> len; if (ss.fail() && not ss.eof()) { len = 0; } }  // length is optional; garbage is not
        --start;
        if (start < 0 || len < 1 || start + len > L) {
          problems.add(absl::StrFormat("%s: missing-data interval '%s' is outside the %d-site reference "
                                       "(expected: <letter> <1-based start> [<length>])", what, line, L));
          continue;
        }
        for (auto l = start; l != start + len; ++l) { sample.tip.missations.insert(Missation{l, raw.reference[l]}, raw.reference); }
      } else {
        auto pos = Site_index{};
        ss >> pos;
        --pos;
        if (not ss || pos < 0 || pos >= L) {
          problems.add(absl::StrFormat("%s: mutation '%s' has a bad or out-of-range site (reference has %d sites)",
                                       what, line, L));
          continue;
        }
        auto from = raw.reference[pos];
        auto to = to_real_seq_letter(s);
        if (from == to) {
          problems.add(absl::StrFormat("%s: '%s' equals the reference state at site %d; list differences only",
                                       what, line, pos + 1));
          continue;
        }
        if (not mutated_sites.insert(pos).second) {
          problems.add(absl::StrFormat("%s: site %d is listed more than once", what, pos + 1));
          continue;
        }
        sample.tip.seq_deltas.push_back({pos, from, to});
      }
    }

    for (const auto& d : sample.tip.seq_deltas) {
      if (sample.tip.missations.contains(d.site)) {
        problems.add(absl::StrFormat("MAPLE sample '%s' (line %d): site %d is both mutated and missing",
                                     sample.id, sample_line, d.site + 1));
        break;
      }
    }
    std::ranges::sort(sample.tip.seq_deltas, {}, &Seq_delta::site);
    raw.samples.push_back(std::move(sample));
  }
  problems.throw_if_any("MAPLE input rejected");
  return raw;
}

//
// Metadata and mask parsing
//

auto trimmed(std::string_view sv) -> std::string { return std::string{absl::StripAsciiWhitespace(sv)}; }

//
// Observation declarations: callable sites, per-sample callable sites and variant lists
//

auto parse_one_interval(std::string_view token, Site_index num_sites, std::string_view what, int line_number,
                        Problems& problems) -> std::optional<Site_interval> {
  auto text = trimmed(token);
  if (text.empty()) { return std::nullopt; }
  auto parts = std::vector<std::string>{absl::StrSplit(text, '-')};
  auto first = int64_t{}, last = int64_t{};
  auto ok = (parts.size() == 1 && absl::SimpleAtoi(parts[0], &first) && (last = first, true)) ||
            (parts.size() == 2 && absl::SimpleAtoi(trimmed(parts[0]), &first) &&
             absl::SimpleAtoi(trimmed(parts[1]), &last));
  if (not ok) {
    problems.add(absl::StrFormat("%s line %d: '%s' is not a site ('310') or a range ('303-315', 1-based, "
                                 "inclusive)", what, line_number, text));
    return std::nullopt;
  }
  if (first < 1 || last < first || last > num_sites) {
    problems.add(absl::StrFormat("%s line %d: range %d-%d is outside the valid sites 1-%d (or reversed)",
                                 what, line_number, first, last, num_sites));
    return std::nullopt;
  }
  return Site_interval{static_cast<Site_index>(first - 1), static_cast<Site_index>(last)};
}

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

auto read_interval_file(std::istream& is, Site_index num_sites, std::string_view what)
    -> std::vector<Site_interval> {
  auto problems = Problems{};
  auto intervals = std::vector<Site_interval>{};
  auto line_number = 0;
  for (auto line = std::string{}; std::getline(is, line);) {
    ++line_number;
    if (auto hash = line.find('#'); hash != std::string::npos) { line.erase(hash); }
    auto text = trimmed(line);
    if (text.empty()) { continue; }
    if (auto iv = parse_one_interval(text, num_sites, what, line_number, problems); iv.has_value()) {
      intervals.push_back(*iv);
    }
  }
  problems.throw_if_any(absl::StrFormat("%s rejected", what));
  return merge_intervals(std::move(intervals));
}

auto complement_intervals(const std::vector<Site_interval>& intervals, Site_index num_sites)
    -> std::vector<Site_interval> {
  auto result = std::vector<Site_interval>{};
  auto cur = Site_index{0};
  for (const auto& [start, end] : intervals) {
    if (start > cur) { result.push_back({cur, start}); }
    cur = std::max(cur, end);
  }
  if (cur < num_sites) { result.push_back({cur, num_sites}); }
  return result;
}

auto intersect_intervals(const std::vector<Site_interval>& a, const std::vector<Site_interval>& b)
    -> std::vector<Site_interval> {
  auto result = std::vector<Site_interval>{};
  auto i = 0, j = 0;
  while (i < std::ssize(a) && j < std::ssize(b)) {
    auto start = std::max(a[i].first, b[j].first);
    auto end = std::min(a[i].second, b[j].second);
    if (start < end) { result.push_back({start, end}); }
    if (a[i].second < b[j].second) { ++i; } else { ++j; }
  }
  return result;
}

auto num_sites_in(const std::vector<Site_interval>& intervals) -> Site_index {
  auto total = Site_index{0};
  for (const auto& [start, end] : intervals) { total += end - start; }
  return total;
}

auto is_in_intervals(const std::vector<Site_interval>& intervals, Site_index l) -> bool {
  auto it = std::ranges::upper_bound(intervals, l, {}, &Site_interval::first);
  return it != intervals.begin() && l < std::prev(it)->second;
}

// Makes the sites of `missing` missing in `tip`, because a callable declaration did not list them.  A call that
// differs from the reference inside those sites is a contradiction between the data and the declaration; its site
// is appended to `contradictions` and the caller rejects the input.  Input tips carry no missation from_states
// (no tip has a mutation and a missation at the same site), so the sites only enter the interval set.
auto apply_missing_intervals(
    Tip_desc& tip,
    const std::vector<Site_interval>& missing,
    std::vector<Site_index>& contradictions)
    -> void {
  if (missing.empty()) { return; }
  for (const auto& d : tip.seq_deltas) {
    if (is_in_intervals(missing, d.site)) { contradictions.push_back(d.site); }
  }
  std::erase_if(tip.seq_deltas, [&](const Seq_delta& d) { return is_in_intervals(missing, d.site); });
  for (const auto& iv : missing) { tip.missations.intervals.insert(iv); }
  for (auto it = tip.missations.from_states.begin(); it != tip.missations.from_states.end();) {
    it = is_in_intervals(missing, it->first) ? tip.missations.from_states.erase(it) : std::next(it);
  }
}

//
// Common processing
//

auto make_expected_length(const Human_input_config& config, Problems& problems) -> std::optional<Site_index> {
  switch (config.reference_standard) {
    case Reference_standard::rcrs:
    case Reference_standard::rsrs:
      if (config.expected_length.has_value() && *config.expected_length != k_rcrs_length) {
        problems.add(absl::StrFormat(
            "Reference standard %s has %d sites, but the expected length was set to %d.  Use the 'custom' "
            "standard for other references.", to_string(config.reference_standard), k_rcrs_length,
            *config.expected_length));
      }
      return k_rcrs_length;
    case Reference_standard::custom:
      if (not config.expected_length.has_value()) {
        problems.add("Reference standard 'custom' requires the expected number of sites to be declared "
                     "(--v0-human-expected-length), so that truncated or SNP-only input can be detected");
      }
      return config.expected_length;
  }
  return std::nullopt;
}

auto describe_date(const Sample_date_input& d) -> std::string {
  auto text = absl::StrCat(to_string(d.convention), " ", d.from);
  if (not d.to.empty()) { absl::StrAppend(&text, "..", d.to); }
  if (d.bp_reference_year.has_value()) { absl::StrAppend(&text, " (baseline ", *d.bp_reference_year, " CE)"); }
  return text;
}

auto finalize(Raw_input raw, const Human_input_config& config, const Sample_metadata* metadata,
              const Human_observation_inputs& observation) -> Human_input_result {
  auto problems = Problems{};
  auto L_in = static_cast<Site_index>(std::ssize(raw.reference));  // input reference, in reference coordinates
  auto L = L_in;                                                   // analysed sites, including appended ones

  if (not (config.max_missing_fraction > 0.0 && config.max_missing_fraction <= 1.0)) {
    problems.add(absl::StrFormat("max missing fraction must be in (0, 1], got %g", config.max_missing_fraction));
  }

  // Reference standard and length
  auto expected_length = make_expected_length(config, problems);
  if (expected_length.has_value() && L != *expected_length) {
    problems.add(absl::StrFormat(
        "Reference '%s' has %d sites but the declared standard (%s) has %d.  A trimmed, SNP-only or differently "
        "numbered alignment cannot be analysed as a full-length sequence; supply the full-length alignment in "
        "reference coordinates.",
        raw.reference_id, L, to_string(config.reference_standard), *expected_length));
  }

  // Sample IDs
  {
    auto seen = absl::flat_hash_map<std::string, std::string>{};  // lower-case ID -> first spelling
    for (const auto& s : raw.samples) {
      if (auto err = check_sample_id(s.id); err.has_value()) {
        problems.add(*err);
        continue;
      }
      auto [it, inserted] = seen.emplace(absl::AsciiStrToLower(s.id), s.id);
      if (not inserted) {
        problems.add(it->second == s.id
                         ? absl::StrFormat("Duplicate sample ID '%s'", s.id)
                         : absl::StrFormat("Sample IDs '%s' and '%s' differ only in letter case; IDs must be unique",
                                           it->second, s.id));
      }
    }
    if (raw.samples.size() < 2) {
      problems.add(absl::StrFormat("At least 2 samples are needed, found %d", raw.samples.size()));
    }
  }
  problems.throw_if_any("Human input rejected");

  // Mask and callable declarations, all in input reference coordinates
  auto mask = std::vector<Site_interval>{};
  if (observation.mask != nullptr) { mask = read_site_mask(*observation.mask, L_in); }
  auto non_callable = std::vector<Site_interval>{};
  if (observation.callable != nullptr) {
    auto callable = read_callable_sites(*observation.callable, L_in);
    if (callable.empty()) {
      problems.add("The callable-sites file lists no site.  At least one site must be declared callable.");
    }
    non_callable = complement_intervals(callable, L_in);
  }
  auto sample_callable = std::optional<Sample_callable_sites>{};
  if (observation.sample_callable != nullptr) {
    sample_callable = read_sample_callable_sites(*observation.sample_callable, L_in);
  }

  // Dates
  auto uniform_bounds = std::optional<Time_bounds>{};
  if (config.uniform_sample_date.has_value()) {
    try {
      uniform_bounds = convert_sample_date(
          {.convention = Date_convention::iso, .from = *config.uniform_sample_date, .to = {}, .bp_reference_year = {}});
    } catch (const std::invalid_argument& e) {
      problems.add(absl::StrFormat("Uniform sampling date: %s", e.what()));
    }
  }

  auto row_by_id = absl::flat_hash_map<std::string, const Sample_metadata_row*>{};
  if (metadata != nullptr) {
    for (const auto& row : metadata->rows) { row_by_id.emplace(row.sample_id, &row); }
    auto sample_ids = absl::flat_hash_set<std::string>{};
    for (const auto& s : raw.samples) { sample_ids.insert(s.id); }

    auto no_row = std::vector<std::string>{};
    for (const auto& s : raw.samples) {
      if (not row_by_id.contains(s.id)) { no_row.push_back(s.id); }
    }
    if (not no_row.empty()) {
      problems.add(absl::StrFormat("%d sample%s in the sequence file ha%s no row in the metadata (IDs must match "
                                   "exactly): %s", no_row.size(), no_row.size() == 1 ? "" : "s",
                                   no_row.size() == 1 ? "s" : "ve", list_ids(no_row)));
    }
    auto no_seq = std::vector<std::string>{};
    for (const auto& row : metadata->rows) {
      if (not sample_ids.contains(row.sample_id)) { no_seq.push_back(row.sample_id); }
    }
    if (not no_seq.empty()) {
      problems.add(absl::StrFormat("%d metadata row%s ha%s no sequence in the sequence file: %s",
                                   no_seq.size(), no_seq.size() == 1 ? "" : "s", no_seq.size() == 1 ? "s" : "ve",
                                   list_ids(no_seq)));
    }
  }

  struct Sample_time { Time_bounds bounds; std::string source; std::string convention; std::string text; };
  auto times = std::vector<Sample_time>{};
  times.reserve(raw.samples.size());
  auto undated = std::vector<std::string>{};
  for (const auto& s : raw.samples) {
    auto row = static_cast<const Sample_metadata_row*>(nullptr);
    if (auto it = row_by_id.find(s.id); it != row_by_id.end()) { row = it->second; }

    if (row != nullptr && row->date.has_value()) {
      try {
        times.push_back({convert_sample_date(*row->date), "metadata", std::string{to_string(row->date->convention)},
                         describe_date(*row->date)});
      } catch (const std::invalid_argument& e) {
        problems.add(absl::StrFormat("Metadata line %d (sample '%s'): %s", row->line, s.id, e.what()));
        times.push_back({});
      }
    } else if (uniform_bounds.has_value()) {
      times.push_back({*uniform_bounds, "uniform", "iso", absl::StrCat("iso ", *config.uniform_sample_date)});
    } else {
      undated.push_back(s.id);
      times.push_back({});
    }
  }
  if (not undated.empty()) {
    problems.add(absl::StrFormat(
        "%d sample%s ha%s no sampling date: %s.  Give each a date_convention / date_from in the metadata, "
        "or set a uniform sampling date explicitly.",
        undated.size(), undated.size() == 1 ? "" : "s", undated.size() == 1 ? "s" : "ve", list_ids(undated)));
  }
  problems.throw_if_any("Human input rejected");

  // Apply mask
  auto result = Human_input_result{};
  auto& audit = result.audit;
  audit.input_format = raw.format;
  audit.observation_mode = raw.observation_mode;
  if (not config.snp_only_constant_sites.empty()) { audit.observation_mode += "+completed_constant_sites"; }
  audit.reference_standard = std::string{to_string(config.reference_standard)};
  audit.reference_id = raw.reference_id;
  audit.reference_checksum_fnv1a64 = fnv1a64_hex(raw.reference);
  audit.num_samples = static_cast<int>(std::ssize(raw.samples));
  for (const auto& [a, b] : mask) { audit.num_masked_sites += b - a; }
  if (metadata != nullptr) { audit.ignored_metadata_columns = metadata->ignored_columns; }

  auto masked_override_counts = std::vector<Site_index>(raw.samples.size(), 0);
  if (not mask.empty()) {
    for (auto i = 0; i != std::ssize(raw.samples); ++i) {
      auto& tip = raw.samples[i].tip;
      auto in_mask = [&](Site_index l) {
        auto it = std::ranges::upper_bound(mask, l, {}, &Site_interval::first);
        return it != mask.begin() && l < std::prev(it)->second;
      };
      auto before = std::ssize(tip.seq_deltas);
      std::erase_if(tip.seq_deltas, [&](const Seq_delta& d) { return in_mask(d.site); });
      masked_override_counts[i] = static_cast<Site_index>(before - std::ssize(tip.seq_deltas));
      for (const auto& [a, b] : mask) {
        for (auto l = a; l != b; ++l) { tip.missations.insert(Missation{l, raw.reference[l]}, raw.reference); }
      }
    }
  }

  // Apply the callable declarations.  Sites that were not declared callable are missing: the input says nothing
  // about them, and an unobserved site is never a reference state.  A call that differs from the reference at
  // such a site means the data and the declaration disagree, which is an error (use the mask to exclude sites
  // deliberately).
  audit.callable_declared = observation.callable != nullptr;
  audit.sample_callable_declared = observation.sample_callable != nullptr;
  audit.num_non_callable_sites = num_sites_in(non_callable);
  auto declared_callable_sites = std::vector<Site_index>(raw.samples.size(), 0);
  {
    auto contradictions = std::vector<std::string>{};
    auto sample_callable_by_id = absl::flat_hash_map<std::string, const Sample_callable_row*>{};
    if (sample_callable.has_value()) {
      for (const auto& row : sample_callable->rows) { sample_callable_by_id.emplace(row.sample_id, &row); }
      auto no_row = std::vector<std::string>{};
      for (const auto& smp : raw.samples) {
        if (not sample_callable_by_id.contains(smp.id)) { no_row.push_back(smp.id); }
      }
      if (not no_row.empty()) {
        problems.add(absl::StrFormat(
            "%d sample%s ha%s no row in the per-sample callable-sites file: %s.  Every sample must declare which "
            "sites were examined in it; a sample examined everywhere declares the whole range (1-%d).",
            no_row.size(), no_row.size() == 1 ? "" : "s", no_row.size() == 1 ? "s" : "ve", list_ids(no_row), L_in));
      }
      auto sample_ids = absl::flat_hash_set<std::string>{};
      for (const auto& smp : raw.samples) { sample_ids.insert(smp.id); }
      auto no_sample = std::vector<std::string>{};
      for (const auto& row : sample_callable->rows) {
        if (not sample_ids.contains(row.sample_id)) { no_sample.push_back(row.sample_id); }
      }
      if (not no_sample.empty()) {
        problems.add(absl::StrFormat(
            "%d row%s of the per-sample callable-sites file ha%s no sample in the input: %s",
            no_sample.size(), no_sample.size() == 1 ? "" : "s", no_sample.size() == 1 ? "s" : "ve",
            list_ids(no_sample)));
      }
      problems.throw_if_any("Human input rejected");
    }

    auto outside_mask = complement_intervals(mask, L_in);
    auto global_callable = observation.callable != nullptr ? complement_intervals(non_callable, L_in)
                                                          : std::vector<Site_interval>{{0, L_in}};
    for (auto i = 0; i != std::ssize(raw.samples); ++i) {
      auto& smp = raw.samples[i];
      auto sites = std::vector<Site_index>{};
      apply_missing_intervals(smp.tip, non_callable, sites);
      auto declared = global_callable;
      if (sample_callable.has_value()) {
        const auto& row = *sample_callable_by_id.at(smp.id);
        apply_missing_intervals(smp.tip, complement_intervals(row.callable, L_in), sites);
        declared = intersect_intervals(declared, row.callable);
      }
      declared_callable_sites[i] = num_sites_in(intersect_intervals(declared, outside_mask));
      if (not sites.empty()) {
        std::ranges::sort(sites);
        sites.erase(std::ranges::unique(sites).begin(), sites.end());
        auto shown = std::vector<std::string>{};
        for (auto l : sites | std::views::take(5)) { shown.push_back(absl::StrFormat("%d", l + 1)); }
        contradictions.push_back(absl::StrFormat("%s (%d site%s: %s%s)", smp.id, sites.size(),
                                                 sites.size() == 1 ? "" : "s", absl::StrJoin(shown, ", "),
                                                 std::ssize(sites) > 5 ? ", ..." : ""));
      }
    }
    if (not contradictions.empty()) {
      problems.add(absl::StrFormat(
          "%d sample%s a base that differs from the reference at a site that the callable declaration does not "
          "list: %s.  The data and the declaration disagree: either the declaration is incomplete, or those calls "
          "must be removed.  To exclude sites deliberately, use the site mask instead of the callable declaration.",
          contradictions.size(), contradictions.size() == 1 ? " calls" : "s call", list_ids(contradictions)));
    }
    problems.throw_if_any("Human input rejected");
  }

  // SNP-only completion: restore the constant sites that the variability filter removed, as a block appended
  // after the input reference.  Only the number of sites of each state is needed, because sites are independent
  // and, within a partition, exchangeable; the restored sites are callable in every sample and carry no
  // reference coordinates.
  if (not config.snp_only_constant_sites.empty()) {
    auto total = Site_index{0};
    auto totals = Seq_vector<Site_index>{0, 0, 0, 0};
    for (auto i = 0; i != std::ssize(config.snp_only_constant_sites); ++i) {
      const auto& counts = config.snp_only_constant_sites[i];
      for (auto a : k_all_real_seq_letters) {
        if (counts[a] < 0) {
          problems.add(absl::StrFormat(
              "Declared number of constant sites with state %c is negative (%d)%s", to_char(a), counts[a],
              std::ssize(config.snp_only_constant_sites) > 1 ? absl::StrFormat(" in partition %d", i + 1) : ""));
        } else {
          total += counts[a];
          totals[a] += counts[a];
        }
      }
    }
    if (total <= 0) {
      problems.add("The declared constant-site counts are all zero.  Give the counts of the sites that the "
                   "variability filter removed, by state, or drop the declaration and supply full-length input.");
    }
    problems.throw_if_any("Human input rejected");
    raw.reference.reserve(L_in + total);
    for (const auto& counts : config.snp_only_constant_sites) {
      for (auto a : k_all_real_seq_letters) {
        raw.reference.insert(raw.reference.end(), counts[a], a);
      }
    }
    L = static_cast<Site_index>(std::ssize(raw.reference));
    audit.num_constant_sites_appended = totals;
    audit.num_constant_sites_appended_per_partition = config.snp_only_constant_sites;
  }
  audit.num_input_sites = L_in;
  audit.num_sites = L;

  // Missing-data fractions, site classes and duplicates
  auto n = static_cast<int>(std::ssize(raw.samples));
  auto diff_count = std::vector<int>(L, 0);
  auto diff_state = std::vector<Real_seq_letter>(L, Real_seq_letter::A);  // state of the first difference
  auto diff_states_agree = std::vector<char>(L, 1);  // every difference at this site has the same state
  auto missing_delta = std::vector<int>(L + 1, 0);
  auto high_missing = std::vector<std::string>{};
  auto dup_groups = absl::flat_hash_map<std::string, std::vector<int>>{};
  for (auto i = 0; i != n; ++i) {
    const auto& tip = raw.samples[i].tip;
    for (const auto& d : tip.seq_deltas) {
      if (diff_count[d.site] == 0) {
        diff_state[d.site] = d.to;
      } else if (diff_state[d.site] != d.to) {
        diff_states_agree[d.site] = 0;
      }
      ++diff_count[d.site];
    }
    for (const auto& [a, b] : tip.missations.intervals) { ++missing_delta[a]; --missing_delta[b]; }

    auto num_missing = static_cast<Site_index>(tip.missations.num_sites());
    auto fraction = static_cast<double>(num_missing) / L;
    if (fraction > config.max_missing_fraction) {
      high_missing.push_back(absl::StrFormat("%s (%.1f%%)", raw.samples[i].id, 100.0 * fraction));
    }

    auto key = std::string{};
    for (const auto& d : tip.seq_deltas) { absl::StrAppendFormat(&key, "%d%c;", d.site, to_char(d.to)); }
    key += '|';
    for (const auto& [a, b] : tip.missations.intervals) { absl::StrAppendFormat(&key, "%d-%d;", a, b); }
    dup_groups[std::move(key)].push_back(i);
  }
  if (not high_missing.empty()) {
    problems.add(absl::StrFormat(
        "%d sample%s exceed%s the maximum missing-data fraction of %.1f%%: %s.  Remove them from the input, or "
        "raise the limit explicitly if the missing data are expected.",
        high_missing.size(), high_missing.size() == 1 ? "" : "s", high_missing.size() == 1 ? "s" : "",
        100.0 * config.max_missing_fraction, list_ids(high_missing)));
  }

  auto cur_missing = 0;
  auto invariant_observed_in_input = Site_index{0};
  for (auto l = Site_index{0}; l != L; ++l) {
    cur_missing += missing_delta[l];
    auto num_observed = n - cur_missing;
    if (diff_count[l] > 0) {
      // A site that differs from the reference is reported as variable, as it always was, even when every
      // observed sample shares the same non-reference state: the reference base is a representation choice,
      // not an observation, so such a site is invariant data and is counted again under its own heading.
      ++audit.num_variable_sites;
      if (diff_states_agree[l] != 0 && diff_count[l] == num_observed) {
        ++audit.num_invariant_nonreference_sites;
      }
    } else if (num_observed == 0) {
      ++audit.num_never_observed_sites;
    } else {
      ++audit.num_invariant_observed_sites;
      if (l < L_in) { ++invariant_observed_in_input; }
    }
  }
  // The detector of variable-sites-only input: a filter that keeps the sites differing from the reference
  // leaves no site that is observed and equal to the reference everywhere.  Sites where every observed sample
  // shares one non-reference state do carry rate information, but such a site is exactly what that filter
  // keeps, so it does not clear the input of ascertainment.
  if (audit.num_invariant_observed_sites == 0) {
    problems.add(absl::StrFormat(
        "No site is observed as invariant (%d of %d sites differ from the reference or are never observed).  This looks like a "
        "SNP-only or variable-sites-only alignment, which lacks the invariant sites that rate and time estimation "
        "require.  Supply the full-length alignment, or a variant list with the callable sites it was called in, "
        "or, if the removed constant sites are known only as counts, declare them "
        "(--v0-human-snp-only-constant-sites A=..,C=..,G=..,T=..).  Conditioning the likelihood on a site being "
        "variable is not implemented: see doc/extended/observation.md.",
        audit.num_variable_sites + audit.num_never_observed_sites, L));
  }
  problems.throw_if_any("Human input rejected");

  // Duplicates
  {
    auto groups = std::vector<std::vector<int>>{};
    for (auto& [key, members] : dup_groups) {
      if (members.size() > 1) { groups.push_back(std::move(members)); }
    }
    std::ranges::sort(groups);  // deterministic: by first member in input order
    audit.duplicate_groups.reserve(groups.size());
    for (auto g = 0; g != std::ssize(groups); ++g) {
      auto ids = std::vector<std::string>{};
      for (auto i : groups[g]) { ids.push_back(raw.samples[i].id); }
      audit.duplicate_groups.push_back(std::move(ids));
    }
    audit.samples.resize(n);
    for (auto g = 0; g != std::ssize(groups); ++g) {
      for (auto i : groups[g]) { audit.samples[i].duplicate_group = g; }
    }
  }

  // Tips, per-sample audit, time summary
  result.maple.ref_sequence = raw.reference;
  result.maple.tip_descs.reserve(n);
  auto max_float_error = 0.0;
  auto max_abs_t = 0.0;
  auto earliest_mid = std::numeric_limits<double>::infinity();
  auto latest_mid = -std::numeric_limits<double>::infinity();
  audit.earliest_t_min = std::numeric_limits<double>::infinity();
  audit.latest_t_max = -std::numeric_limits<double>::infinity();
  auto any_iupac = 0, any_gap = 0;
  for (auto i = 0; i != n; ++i) {
    auto& rs = raw.samples[i];
    const auto& tm = times[i];

    auto& sa = audit.samples[i];
    sa.sample_id = rs.id;
    sa.date_source = tm.source;
    sa.date_convention = tm.convention;
    sa.date_text = tm.text;
    sa.t_min = tm.bounds.t_min;
    sa.t_max = tm.bounds.t_max;
    sa.num_differences = static_cast<Site_index>(std::ssize(rs.tip.seq_deltas));
    sa.num_missing = static_cast<Site_index>(rs.tip.missations.num_sites());
    sa.num_callable = L - sa.num_missing;
    sa.num_masked_overrides = masked_override_counts[i];
    sa.num_iupac_ambiguous = rs.num_iupac_ambiguous;
    sa.num_unknown = rs.num_unknown;
    sa.num_gap = rs.num_gap;
    sa.missing_fraction = static_cast<double>(sa.num_missing) / L;
    if (rs.num_iupac_ambiguous > 0) { ++any_iupac; }
    if (rs.num_gap > 0) { ++any_gap; }

    if (tm.source == "uniform") { ++audit.num_dates_from_uniform; } else { ++audit.num_dates_from_metadata; }
    audit.earliest_t_min = std::min(audit.earliest_t_min, tm.bounds.t_min);
    audit.latest_t_max = std::max(audit.latest_t_max, tm.bounds.t_max);
    auto mid = 0.5 * (tm.bounds.t_min + tm.bounds.t_max);
    earliest_mid = std::min(earliest_mid, mid);
    latest_mid = std::max(latest_mid, mid);
    max_float_error = std::max({max_float_error, float32_rounding_error(tm.bounds.t_min),
                                float32_rounding_error(tm.bounds.t_max)});
    max_abs_t = std::max({max_abs_t, std::abs(tm.bounds.t_min), std::abs(tm.bounds.t_max)});

    auto tip = std::move(rs.tip);
    tip.name = rs.id;
    tip.t_min = static_cast<float>(tm.bounds.t_min);
    tip.t_max = static_cast<float>(tm.bounds.t_max);
    result.maple.tip_descs.push_back(std::move(tip));
  }
  audit.midpoint_span_days = latest_mid - earliest_mid;
  {
    auto first = audit.samples.front();
    audit.no_time_signal = std::ranges::all_of(audit.samples, [&](const Human_sample_audit& s) {
      return s.t_min == first.t_min && s.t_max == first.t_max;
    });
  }

  // Warnings
  auto& w = audit.warnings;
  if (audit.no_time_signal) {
    w.push_back(absl::StrFormat(
        "NO TIME SIGNAL: all %d samples have the same sampling bounds [%s, %s]%s.  The data carry no information "
        "about the absolute time scale.  Node ages are meaningful only with an informative substitution-rate prior, "
        "a fixed rate, or an internal-node age calibration; they are not calibrated by these data.",
        n, to_iso_date(audit.earliest_t_min), to_iso_date(audit.latest_t_max),
        audit.num_dates_from_uniform == n ? " (uniform sampling date assigned by the user)" : ""));
  } else if (audit.num_dates_from_uniform > 0) {
    w.push_back(absl::StrFormat(
        "%d of %d samples received the uniform sampling date and therefore contribute no time-scale information.",
        audit.num_dates_from_uniform, n));
  }
  if (any_iupac > 0) {
    w.push_back(absl::StrFormat("%d sample%s contain IUPAC ambiguity codes; they are treated as missing data "
                                "(heteroplasmy and uncertain calls are not modelled)",
                                any_iupac, any_iupac == 1 ? "" : "s"));
  }
  if (any_gap > 0) {
    w.push_back(absl::StrFormat("%d sample%s contain gap characters; they are treated as missing data, not as "
                                "deletions", any_gap, any_gap == 1 ? "" : "s"));
  }
  if (audit.callable_declared || audit.sample_callable_declared) {
    auto samples_short = 0;
    auto sites_short = Site_index{0};
    for (auto i = 0; i != n; ++i) {
      auto missing_inside = declared_callable_sites[i] - (L_in - audit.samples[i].num_missing);
      if (missing_inside > 0) { ++samples_short; sites_short += missing_inside; }
    }
    w.push_back(absl::StrFormat(
        "Callable sites declared%s: %d of %d input sites are callable in no sample.  Sites that are not declared "
        "callable are missing, never reference states.",
        audit.sample_callable_declared ? " per sample" : "", audit.num_non_callable_sites, L_in));
    if (samples_short > 0) {
      w.push_back(absl::StrFormat(
          "%d sample%s data missing at %d site%s declared callable for %s (the call failed inside the examined "
          "region); those sites are treated as missing",
          samples_short, samples_short == 1 ? " has" : "s have", sites_short, sites_short == 1 ? "" : "s",
          samples_short == 1 ? "it" : "them"));
    }
  }
  if (not config.snp_only_constant_sites.empty() && invariant_observed_in_input > 0) {
    w.push_back(absl::StrFormat(
        "SNP-only completion: the input itself still contains %d site%s observed as invariant (not polymorphic). "
        " The declared counts must cover only the sites that the filter removed; sites that are in the input are "
        "counted from the input and must not appear in the declaration as well.",
        invariant_observed_in_input, invariant_observed_in_input == 1 ? "" : "s"));
  }
  if (not config.snp_only_constant_sites.empty()) {
    const auto& c = audit.num_constant_sites_appended;
    w.push_back(absl::StrFormat(
        "SNP-only completion: %d constant sites (A %d, C %d, G %d, T %d) were appended as sites %d-%d of the "
        "analysed sequence and are callable in every sample.  They carry no reference coordinates: output "
        "mutations at sites above %d refer to this block.  The counts are an input declaration; if the removed "
        "sites were not callable in every sample, supply full-length or variant-plus-callable input instead.",
        L - L_in, c[Real_seq_letter::A], c[Real_seq_letter::C], c[Real_seq_letter::G], c[Real_seq_letter::T],
        L_in + 1, L, L_in));
  }
  if (audit.num_never_observed_sites > 0) {
    w.push_back(absl::StrFormat("%d site%s never observed in any sample (missing or masked); they are excluded from "
                                "the likelihood and are not counted as invariant",
                                audit.num_never_observed_sites, audit.num_never_observed_sites == 1 ? " is" : "s are"));
  }
  if (not audit.duplicate_groups.empty()) {
    auto dups = 0;
    for (const auto& g : audit.duplicate_groups) { dups += static_cast<int>(std::ssize(g)); }
    w.push_back(absl::StrFormat("%d sample%s fall into %d group%s of identical processed sequences",
                                dups, dups == 1 ? "" : "s", audit.duplicate_groups.size(),
                                audit.duplicate_groups.size() == 1 ? "" : "s"));
  }
  if (max_float_error > 0.01) {
    w.push_back(absl::StrFormat(
        "Tip date bounds are stored as float32; the oldest bounds are rounded by up to %.2f day", max_float_error));
  }
  {
    auto f = static_cast<float>(max_abs_t);
    auto spacing = static_cast<double>(std::nextafter(f, std::numeric_limits<float>::infinity())) - f;
    if (spacing > 0.01) {
      w.push_back(absl::StrFormat(
          "Node and mutation times in .dphy files are stored as float32; at the oldest sample (|t| = %.0f days) "
          "the resolution is %.3g day.  Reloaded trees can violate time ordering for branches shorter than that.",
          max_abs_t, spacing));
    }
  }

  return result;
}

}  // namespace

auto to_string(Reference_standard standard) -> std::string_view {
  switch (standard) {
    case Reference_standard::rcrs: return "rCRS";
    case Reference_standard::rsrs: return "RSRS";
    case Reference_standard::custom: return "custom";
  }
  return "unknown";
}

auto parse_reference_standard(std::string_view name) -> Reference_standard {
  for (auto s : {Reference_standard::rcrs, Reference_standard::rsrs, Reference_standard::custom}) {
    if (name == to_string(s)) { return s; }
  }
  throw Human_input_error(absl::StrFormat(
      "Unknown reference standard '%s'.  Accepted values: rCRS, RSRS, custom.", name));
}

auto read_sample_metadata(std::istream& is) -> Sample_metadata {
  auto problems = Problems{};
  auto result = Sample_metadata{};

  auto header = std::vector<std::string>{};
  auto col = absl::flat_hash_map<std::string, int>{};
  auto line_number = 0;
  auto seen_ids = absl::flat_hash_map<std::string, int>{};  // lower-case ID -> line

  for (auto line = std::string{}; std::getline(is, line);) {
    ++line_number;
    if (not line.empty() && line.back() == '\r') { line.pop_back(); }
    if (absl::StripAsciiWhitespace(line).empty() || line[0] == '#') { continue; }

    auto fields = std::vector<std::string>{};
    for (auto f : absl::StrSplit(line, '\t')) { fields.push_back(trimmed(f)); }

    if (header.empty()) {
      header = std::move(fields);
      for (auto i = 0; i != std::ssize(header); ++i) {
        if (not col.emplace(header[i], i).second) {
          problems.add(absl::StrFormat("Metadata header (line %d): duplicate column '%s'", line_number, header[i]));
        }
      }
      if (not col.contains("sample_id")) {
        problems.add(absl::StrFormat(
            "Metadata header (line %d) has no 'sample_id' column.  Expected tab-separated columns: sample_id, "
            "date_convention, date_from, date_to, bp_reference_year.", line_number));
      }
      for (const auto& h : header) {
        if (h != "sample_id" && h != "date_convention" && h != "date_from" && h != "date_to" &&
            h != "bp_reference_year") {
          result.ignored_columns.push_back(h);
        }
      }
      problems.throw_if_any("Metadata rejected");
      continue;
    }

    if (fields.size() != header.size()) {
      problems.add(absl::StrFormat(
          "Metadata line %d has %d tab-separated fields but the header has %d.  Fields must be separated by tabs "
          "(not spaces) and empty cells must be left empty.", line_number, fields.size(), header.size()));
      continue;
    }
    auto get = [&](const char* name) -> std::string {
      auto it = col.find(name);
      return it == col.end() ? std::string{} : fields[it->second];
    };

    auto row = Sample_metadata_row{};
    row.line = line_number;
    row.sample_id = get("sample_id");
    if (row.sample_id.empty()) {
      problems.add(absl::StrFormat("Metadata line %d: empty sample_id", line_number));
      continue;
    }
    if (auto [it, inserted] = seen_ids.emplace(absl::AsciiStrToLower(row.sample_id), line_number); not inserted) {
      problems.add(absl::StrFormat("Metadata line %d: sample_id '%s' duplicates (up to letter case) line %d; "
                                   "sample IDs must be unique", line_number, row.sample_id, it->second));
      continue;
    }

    auto convention = get("date_convention");
    auto date_from = get("date_from");
    auto date_to = get("date_to");
    auto bp_ref = get("bp_reference_year");
    if (convention.empty()) {
      if (not date_from.empty() || not date_to.empty() || not bp_ref.empty()) {
        problems.add(absl::StrFormat(
            "Metadata line %d (sample '%s'): a date is given but date_convention is empty.  Name the convention "
            "(iso, ce, bce, bp, cal_bp); dates are never interpreted implicitly.", line_number, row.sample_id));
      }
    } else {
      auto d = Sample_date_input{};
      try {
        d.convention = parse_date_convention(convention);
      } catch (const std::invalid_argument& e) {
        problems.add(absl::StrFormat("Metadata line %d (sample '%s'): %s", line_number, row.sample_id, e.what()));
        continue;
      }
      d.from = date_from;
      d.to = date_to;
      if (not bp_ref.empty()) {
        auto year = int64_t{};
        if (not absl::SimpleAtoi(bp_ref, &year)) {
          problems.add(absl::StrFormat("Metadata line %d (sample '%s'): bp_reference_year '%s' is not an integer",
                                       line_number, row.sample_id, bp_ref));
          continue;
        }
        d.bp_reference_year = year;
      }
      row.date = std::move(d);
    }
    result.rows.push_back(std::move(row));
  }

  if (header.empty()) {
    problems.add("Metadata file is empty: a header line with a 'sample_id' column is required");
  }
  problems.throw_if_any("Metadata rejected");
  if (result.rows.empty()) {
    throw Human_input_error("Metadata rejected: no data rows");
  }
  return result;
}

auto read_site_mask(std::istream& is, Site_index num_sites) -> std::vector<Site_interval> {
  return read_interval_file(is, num_sites, "Site mask");
}

auto read_callable_sites(std::istream& is, Site_index num_sites) -> std::vector<Site_interval> {
  return read_interval_file(is, num_sites, "Callable sites");
}

auto read_sample_callable_sites(std::istream& is, Site_index num_sites) -> Sample_callable_sites {
  auto problems = Problems{};
  auto result = Sample_callable_sites{};
  auto seen = absl::flat_hash_map<std::string, int>{};  // lower-case ID -> line
  auto line_number = 0;
  for (auto line = std::string{}; std::getline(is, line);) {
    ++line_number;
    if (not line.empty() && line.back() == '\r') { line.pop_back(); }
    if (auto hash = line.find('#'); hash != std::string::npos) { line.erase(hash); }
    if (trimmed(line).empty()) { continue; }

    auto fields = std::vector<std::string>{};
    for (auto f : absl::StrSplit(line, '\t')) { fields.push_back(trimmed(f)); }
    if (fields.size() != 2) {
      problems.add(absl::StrFormat(
          "Per-sample callable sites line %d has %d tab-separated fields, expected 2: "
          "'<sample_id>\\t<sites>', where <sites> is a comma-separated list of 1-based inclusive sites or "
          "ranges ('1-16569,16800' or 'none')", line_number, fields.size()));
      continue;
    }
    auto row = Sample_callable_row{};
    row.line = line_number;
    row.sample_id = fields[0];
    if (row.sample_id.empty()) {
      problems.add(absl::StrFormat("Per-sample callable sites line %d: empty sample_id", line_number));
      continue;
    }
    if (auto [it, inserted] = seen.emplace(absl::AsciiStrToLower(row.sample_id), line_number); not inserted) {
      problems.add(absl::StrFormat(
          "Per-sample callable sites line %d: sample '%s' already has a row on line %d; declare every sample "
          "exactly once", line_number, row.sample_id, it->second));
      continue;
    }
    auto intervals = std::vector<Site_interval>{};
    if (absl::AsciiStrToLower(fields[1]) != "none") {
      for (auto token : absl::StrSplit(fields[1], ',')) {
        if (auto iv = parse_one_interval(token, num_sites, "Per-sample callable sites", line_number, problems);
            iv.has_value()) {
          intervals.push_back(*iv);
        }
      }
      if (intervals.empty()) {
        problems.add(absl::StrFormat(
            "Per-sample callable sites line %d (sample '%s'): no site given.  Write 'none' if nothing was "
            "examined in this sample.", line_number, row.sample_id));
      }
    }
    row.callable = merge_intervals(std::move(intervals));
    result.rows.push_back(std::move(row));
  }
  problems.throw_if_any("Per-sample callable sites rejected");
  if (result.rows.empty()) {
    throw Human_input_error("Per-sample callable sites rejected: the file declares no sample");
  }
  return result;
}

auto read_variant_table(std::istream& is, Site_index num_sites) -> Variant_table {
  auto problems = Problems{};
  auto result = Variant_table{};
  auto header = std::vector<std::string>{};
  auto col = absl::flat_hash_map<std::string, int>{};
  auto line_number = 0;

  for (auto line = std::string{}; std::getline(is, line);) {
    ++line_number;
    if (not line.empty() && line.back() == '\r') { line.pop_back(); }
    if (absl::StripAsciiWhitespace(line).empty() || line[0] == '#') { continue; }

    auto fields = std::vector<std::string>{};
    for (auto f : absl::StrSplit(line, '\t')) { fields.push_back(trimmed(f)); }

    if (header.empty()) {
      header = std::move(fields);
      for (auto i = 0; i != std::ssize(header); ++i) {
        if (not col.emplace(header[i], i).second) {
          problems.add(absl::StrFormat("Variant table header (line %d): duplicate column '%s'", line_number,
                                       header[i]));
        }
      }
      for (const auto* name : {"sample_id", "site", "base"}) {
        if (not col.contains(name)) {
          problems.add(absl::StrFormat(
              "Variant table header (line %d) has no '%s' column.  Required tab-separated columns: sample_id, "
              "site (1-based, reference coordinates), base (A, C, G or T).", line_number, name));
        }
      }
      for (const auto& h : header) {
        if (h != "sample_id" && h != "site" && h != "base") { result.ignored_columns.push_back(h); }
      }
      problems.throw_if_any("Variant table rejected");
      continue;
    }

    if (fields.size() != header.size()) {
      problems.add(absl::StrFormat(
          "Variant table line %d has %d tab-separated fields but the header has %d", line_number, fields.size(),
          header.size()));
      continue;
    }
    auto rec = Variant_record{};
    rec.line = line_number;
    rec.sample_id = fields[col.at("sample_id")];
    if (rec.sample_id.empty()) {
      problems.add(absl::StrFormat("Variant table line %d: empty sample_id", line_number));
      continue;
    }
    auto site_text = fields[col.at("site")];
    auto site = int64_t{};
    if (not absl::SimpleAtoi(site_text, &site)) {
      problems.add(absl::StrFormat("Variant table line %d (sample '%s'): site '%s' is not an integer",
                                   line_number, rec.sample_id, site_text));
      continue;
    }
    if (site < 1 || site > num_sites) {
      problems.add(absl::StrFormat(
          "Variant table line %d (sample '%s'): site %d is outside the reference sites 1-%d.  Variant sites are "
          "1-based positions in the coordinates of the reference supplied with the input.",
          line_number, rec.sample_id, site, num_sites));
      continue;
    }
    rec.site = static_cast<Site_index>(site - 1);

    auto base = absl::AsciiStrToUpper(fields[col.at("base")]);
    if (base.size() != 1) {
      problems.add(absl::StrFormat(
          "Variant table line %d (sample '%s'): base '%s' is not a single nucleotide.  Only substitutions of one "
          "unambiguous base are modelled: insertions, deletions, multi-nucleotide alleles and copy-number calls "
          "are not supported and must be removed from the input (the affected sites can be excluded with a site "
          "mask or left out of the callable sites).", line_number, rec.sample_id, fields[col.at("base")]));
      continue;
    }
    switch (base[0]) {
      case 'A': case 'C': case 'G': case 'T':
        rec.base = char_to_real_seq_letter(base[0]);
        break;
      case 'N': case '?': case '-': case '.':
        problems.add(absl::StrFormat(
            "Variant table line %d (sample '%s'): base '%c' means 'not called', which a variant record cannot "
            "express.  Leave the site out of this sample's callable sites instead.", line_number, rec.sample_id,
            base[0]));
        continue;
      case 'R': case 'Y': case 'S': case 'W': case 'K': case 'M': case 'B': case 'D': case 'H': case 'V':
        problems.add(absl::StrFormat(
            "Variant table line %d (sample '%s'): base '%c' is an IUPAC ambiguity code.  Mixed or uncertain calls "
            "(heteroplasmy, multiple copies) are not modelled: resolve the call, or leave the site out of this "
            "sample's callable sites.", line_number, rec.sample_id, base[0]));
        continue;
      default:
        problems.add(absl::StrFormat("Variant table line %d (sample '%s'): base '%c' is not a nucleotide",
                                     line_number, rec.sample_id, base[0]));
        continue;
    }
    result.records.push_back(std::move(rec));
  }

  if (header.empty()) {
    problems.add("Variant table is empty: a header line with sample_id, site and base columns is required");
  }
  problems.throw_if_any("Variant table rejected");
  return result;
}

auto parse_constant_site_counts(std::string_view spec) -> Seq_vector<Site_index> {
  auto counts = Seq_vector<Site_index>{-1, -1, -1, -1};
  auto problems = Problems{};
  for (auto part : absl::StrSplit(spec, ',')) {
    auto text = trimmed(part);
    if (text.empty()) { continue; }
    auto kv = std::vector<std::string>{absl::StrSplit(text, '=')};
    auto count = int64_t{};
    if (kv.size() != 2 || trimmed(kv[0]).size() != 1 || not absl::SimpleAtoi(trimmed(kv[1]), &count)) {
      problems.add(absl::StrFormat("Constant-site counts: '%s' is not of the form 'A=1000'", text));
      continue;
    }
    auto letter = absl::AsciiStrToUpper(trimmed(kv[0]))[0];
    if (letter != 'A' && letter != 'C' && letter != 'G' && letter != 'T') {
      problems.add(absl::StrFormat("Constant-site counts: '%c' is not a nucleotide", letter));
      continue;
    }
    auto a = char_to_real_seq_letter(letter);
    if (counts[a] >= 0) {
      problems.add(absl::StrFormat("Constant-site counts: state %c is given twice", letter));
      continue;
    }
    if (count < 0) {
      problems.add(absl::StrFormat("Constant-site counts: state %c has a negative count (%d)", letter, count));
      continue;
    }
    counts[a] = static_cast<Site_index>(count);
  }
  for (auto a : k_all_real_seq_letters) {
    if (counts[a] < 0) {
      problems.add(absl::StrFormat(
          "Constant-site counts: state %c is missing.  Give all four counts, for example "
          "'A=4000,C=3000,G=3100,T=2400'; the counts are the sites that the variability filter removed, by the "
          "state that every sample shared there.", to_char(a)));
    }
  }
  problems.throw_if_any("Constant-site counts rejected");
  return counts;
}

namespace {

// Variant list plus reference: the sample set comes from the metadata, so that samples without any variant are
// present and every sample has a declared date.  Nothing is assumed about sites that no record mentions; the
// caller's callable declaration decides which of them are reference states and which are missing.
auto read_raw_variants(
    std::istream& variants_is,
    std::istream& reference_is,
    const Sample_metadata& metadata,
    const Human_input_config& config)
    -> Raw_input {
  if (config.reference_id.has_value()) {
    throw Human_input_error("A reference record ID applies to FASTA input only; variant input takes the reference "
                            "from a separate FASTA file");
  }
  auto problems = Problems{};
  auto raw = Raw_input{};
  raw.format = "variants";
  raw.observation_mode = "variants_plus_callable";

  auto ref_records = parse_fasta_records(reference_is, "Reference FASTA", problems);
  problems.throw_if_any("Reference FASTA rejected");
  if (ref_records.size() != 1) {
    throw Human_input_error(absl::StrFormat(
        "Reference FASTA rejected: it must contain exactly one record, found %d", ref_records.size()));
  }
  raw.reference_id = ref_records[0].id;
  raw.reference = reference_from_record(ref_records[0], "Reference record", problems);
  problems.throw_if_any("Reference rejected");
  auto L = static_cast<Site_index>(std::ssize(raw.reference));

  auto variants = read_variant_table(variants_is, L);

  raw.samples.reserve(metadata.rows.size());
  auto index_of = absl::flat_hash_map<std::string, int>{};
  for (const auto& row : metadata.rows) {
    index_of.emplace(row.sample_id, static_cast<int>(std::ssize(raw.samples)));
    auto smp = Raw_sample{};
    smp.id = row.sample_id;
    raw.samples.push_back(std::move(smp));  // character counts stay at -1: this format carries no characters
  }

  auto seen = absl::flat_hash_map<std::pair<int, Site_index>, const Variant_record*>{};
  auto unknown_ids = std::vector<std::string>{};
  for (const auto& rec : variants.records) {
    auto it = index_of.find(rec.sample_id);
    if (it == index_of.end()) {
      unknown_ids.push_back(absl::StrFormat("'%s' (line %d)", rec.sample_id, rec.line));
      continue;
    }
    auto i = it->second;
    auto ref_base = raw.reference[rec.site];
    if (rec.base == ref_base) {
      problems.add(absl::StrFormat(
          "Variant table line %d (sample '%s'): the call %c at site %d is the reference base.  A variant record "
          "must differ from the reference; remove the record, or correct the reference.",
          rec.line, rec.sample_id, to_char(rec.base), rec.site + 1));
      continue;
    }
    if (auto [slot, inserted] = seen.emplace(std::pair{i, rec.site}, &rec); not inserted) {
      problems.add(slot->second->base == rec.base
                       ? absl::StrFormat("Variant table line %d (sample '%s'): site %d is already recorded on "
                                         "line %d with the same base; remove the duplicate record",
                                         rec.line, rec.sample_id, rec.site + 1, slot->second->line)
                       : absl::StrFormat("Variant table line %d (sample '%s'): site %d is recorded as %c here and "
                                         "as %c on line %d; a site has one state per sample",
                                         rec.line, rec.sample_id, rec.site + 1, to_char(rec.base),
                                         to_char(slot->second->base), slot->second->line));
      continue;
    }
    raw.samples[i].tip.seq_deltas.push_back(Seq_delta{rec.site, ref_base, rec.base});
  }
  if (not unknown_ids.empty()) {
    problems.add(absl::StrFormat(
        "%d variant record%s name a sample that has no metadata row: %s.  The sample set is the metadata table, "
        "so every sample must be listed there (IDs must match exactly).",
        unknown_ids.size(), unknown_ids.size() == 1 ? "s" : "s", absl::StrJoin(unknown_ids, ", ")));
  }
  problems.throw_if_any("Variant input rejected");

  for (auto& smp : raw.samples) {
    std::ranges::sort(smp.tip.seq_deltas, {}, &Seq_delta::site);
  }
  return raw;
}

}  // namespace

auto load_human_fasta(
    std::istream& fasta_is,
    std::istream* reference_is,
    const Sample_metadata* metadata,
    std::istream* mask_is,
    const Human_input_config& config)
    -> Human_input_result {
  return load_human_fasta(fasta_is, reference_is, metadata, Human_observation_inputs{.mask = mask_is}, config);
}

auto load_human_fasta(
    std::istream& fasta_is,
    std::istream* reference_is,
    const Sample_metadata* metadata,
    const Human_observation_inputs& observation,
    const Human_input_config& config)
    -> Human_input_result {
  return finalize(read_raw_fasta(fasta_is, reference_is, config), config, metadata, observation);
}

auto load_human_maple(
    std::istream& maple_is,
    const Sample_metadata* metadata,
    std::istream* mask_is,
    const Human_input_config& config)
    -> Human_input_result {
  return load_human_maple(maple_is, metadata, Human_observation_inputs{.mask = mask_is}, config);
}

auto load_human_maple(
    std::istream& maple_is,
    const Sample_metadata* metadata,
    const Human_observation_inputs& observation,
    const Human_input_config& config)
    -> Human_input_result {
  return finalize(read_raw_maple(maple_is, config), config, metadata, observation);
}

auto load_human_variants(
    std::istream& variants_is,
    std::istream& reference_is,
    const Sample_metadata& metadata,
    const Human_observation_inputs& observation,
    const Human_input_config& config)
    -> Human_input_result {
  if (observation.callable == nullptr && observation.sample_callable == nullptr) {
    throw Human_input_error(
        "Variant input requires a callable declaration: a variant list says nothing about the sites it does not "
        "mention, and an unobserved site is not a reference state.  Declare the sites that were examined with "
        "--v0-human-callable-file (same in every sample) or --v0-human-sample-callable-file (per sample).");
  }
  return finalize(read_raw_variants(variants_is, reference_is, metadata, config), config, &metadata, observation);
}

namespace {

auto json_string(std::string_view s) -> std::string {
  auto out = std::string{"\""};
  for (auto c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      case '\r': out += "\\r"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          absl::StrAppendFormat(&out, "\\u%04x", static_cast<unsigned>(c));
        } else {
          out += c;
        }
    }
  }
  out += '"';
  return out;
}

auto json_string_array(const std::vector<std::string>& v) -> std::string {
  auto parts = std::vector<std::string>{};
  for (const auto& s : v) { parts.push_back(json_string(s)); }
  return absl::StrCat("[", absl::StrJoin(parts, ", "), "]");
}

}  // namespace

auto write_human_audit_json(const Human_audit& a, std::ostream& os) -> void {
  os << "{\n";
  os << "  \"report_version\": " << a.report_version << ",\n";
  os << "  \"input_format\": " << json_string(a.input_format) << ",\n";
  os << "  \"observation_mode\": " << json_string(a.observation_mode) << ",\n";
  os << "  \"reference\": {\"standard\": " << json_string(a.reference_standard)
     << ", \"id\": " << json_string(a.reference_id)
     << ", \"num_sites\": " << a.num_sites
     << ", \"num_input_sites\": " << a.num_input_sites
     << ", \"checksum_fnv1a64\": " << json_string(a.reference_checksum_fnv1a64) << "},\n";
  os << "  \"num_samples\": " << a.num_samples << ",\n";
  os << "  \"sites\": {\"masked\": " << a.num_masked_sites
     << ", \"not_callable\": " << a.num_non_callable_sites
     << ", \"variable\": " << a.num_variable_sites
     << ", \"invariant_observed\": " << a.num_invariant_observed_sites
     << ", \"invariant_nonreference\": " << a.num_invariant_nonreference_sites
     << ", \"never_observed\": " << a.num_never_observed_sites << "},\n";
  os << "  \"callable_declaration\": {\"all_samples\": " << (a.callable_declared ? "true" : "false")
     << ", \"per_sample\": " << (a.sample_callable_declared ? "true" : "false") << "},\n";
  os << "  \"constant_sites_appended\": {\"A\": " << a.num_constant_sites_appended[Real_seq_letter::A]
     << ", \"C\": " << a.num_constant_sites_appended[Real_seq_letter::C]
     << ", \"G\": " << a.num_constant_sites_appended[Real_seq_letter::G]
     << ", \"T\": " << a.num_constant_sites_appended[Real_seq_letter::T] << "},\n";
  os << "  \"constant_sites_appended_per_partition\": [";
  for (auto i = 0; i != std::ssize(a.num_constant_sites_appended_per_partition); ++i) {
    const auto& c = a.num_constant_sites_appended_per_partition[i];
    os << (i == 0 ? "" : ", ")
       << absl::StreamFormat("{\"A\": %d, \"C\": %d, \"G\": %d, \"T\": %d}", c[Real_seq_letter::A],
                             c[Real_seq_letter::C], c[Real_seq_letter::G], c[Real_seq_letter::T]);
  }
  os << "],\n";
  os << absl::StreamFormat(
      "  \"time\": {\"dates_from_metadata\": %d, \"dates_from_uniform\": %d, \"earliest_t_min_days\": %.17g, "
      "\"latest_t_max_days\": %.17g, \"midpoint_span_days\": %.17g, \"no_time_signal\": %s},\n",
      a.num_dates_from_metadata, a.num_dates_from_uniform, a.earliest_t_min, a.latest_t_max, a.midpoint_span_days,
      a.no_time_signal ? "true" : "false");
  os << "  \"ignored_metadata_columns\": " << json_string_array(a.ignored_metadata_columns) << ",\n";
  os << "  \"warnings\": " << json_string_array(a.warnings) << ",\n";
  os << "  \"duplicate_groups\": [";
  for (auto g = 0; g != std::ssize(a.duplicate_groups); ++g) {
    os << (g == 0 ? "" : ", ") << json_string_array(a.duplicate_groups[g]);
  }
  os << "],\n";
  os << "  \"samples\": [\n";
  for (auto i = 0; i != std::ssize(a.samples); ++i) {
    const auto& s = a.samples[i];
    os << absl::StreamFormat(
        "    {\"sample_id\": %s, \"date_source\": %s, \"date\": %s, \"t_min_days\": %.17g, \"t_max_days\": %.17g, "
        "\"differences\": %d, \"missing\": %d, \"callable\": %d, \"missing_fraction\": %.6g, "
        "\"masked_overrides\": %d, "
        "\"iupac_ambiguous\": %d, \"unknown\": %d, \"gap\": %d, \"duplicate_group\": %d}%s\n",
        json_string(s.sample_id), json_string(s.date_source), json_string(s.date_text), s.t_min, s.t_max,
        s.num_differences, s.num_missing, s.num_callable, s.missing_fraction, s.num_masked_overrides,
        s.num_iupac_ambiguous,
        s.num_unknown, s.num_gap, s.duplicate_group, i + 1 == std::ssize(a.samples) ? "" : ",");
  }
  os << "  ]\n}\n";
}

auto write_human_audit_summary(const Human_audit& a, std::ostream& os) -> void {
  os << absl::StreamFormat(
      "Human input audit: %d samples x %d sites (%s, %s), reference '%s' (%s, checksum %s)\n",
      a.num_samples, a.num_sites, a.input_format, a.observation_mode, a.reference_id, a.reference_standard,
      a.reference_checksum_fnv1a64);
  os << absl::StreamFormat(
      "- sites: %d variable (of which %d invariant but not the reference base), %d invariant (observed), "
      "%d never observed, %d masked, %d not callable\n",
      a.num_variable_sites, a.num_invariant_nonreference_sites, a.num_invariant_observed_sites,
      a.num_never_observed_sites, a.num_masked_sites, a.num_non_callable_sites);
  os << absl::StreamFormat(
      "- sampling dates: %d from metadata, %d from the uniform date; sample-time midpoint span %.1f days\n",
      a.num_dates_from_metadata, a.num_dates_from_uniform, a.midpoint_span_days);
  for (const auto& w : a.warnings) { os << "WARNING: " << w << "\n"; }
}

}  // namespace delphy
