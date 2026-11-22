#include "clade_summary.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

namespace delphy {

namespace {

auto list_names(const std::vector<std::string>& names, int max_shown = 10) -> std::string {
  auto shown = std::vector<std::string>{};
  for (const auto& n : names) {
    if (std::ssize(shown) == max_shown) {
      shown.push_back(absl::StrFormat("... and %d more", std::ssize(names) - max_shown));
      break;
    }
    shown.push_back(absl::StrFormat("'%s'", n));
  }
  return absl::StrJoin(shown, ", ");
}

auto throw_problems(const std::vector<std::string>& problems) -> void {
  if (not problems.empty()) { throw Clade_spec_error{absl::StrJoin(problems, "\n")}; }
}

auto trim(std::string_view s) -> std::string_view {
  auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) { return {}; }
  auto last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Clade definition table
// ---------------------------------------------------------------------------------------------

auto read_clade_specs(std::istream& is) -> Clade_spec_set {
  auto problems = std::vector<std::string>{};
  auto result = Clade_spec_set{};
  auto seen_ids = absl::flat_hash_map<std::string, int>{};

  auto line = std::string{};
  auto line_number = 0;
  auto id_column = -1;
  auto taxa_column = -1;
  auto num_columns = 0;

  while (std::getline(is, line)) {
    ++line_number;
    auto content = trim(line);
    if (content.empty() || content.front() == '#') { continue; }

    auto fields = std::vector<std::string>{};
    for (auto f : absl::StrSplit(line, '\t')) { fields.push_back(std::string{trim(f)}); }

    if (id_column < 0) {  // Header
      num_columns = static_cast<int>(std::ssize(fields));
      for (auto i = 0; i != num_columns; ++i) {
        if (fields[i] == "clade_id") { id_column = i; }
        if (fields[i] == "taxa") { taxa_column = i; }
      }
      if (id_column < 0 || taxa_column < 0) {
        problems.push_back(absl::StrFormat(
            "line %d: the header must be a tab-separated line naming at least the columns 'clade_id' and "
            "'taxa'; found: %s", line_number, absl::StrJoin(fields, " | ")));
        throw_problems(problems);
      }
      continue;
    }

    if (std::ssize(fields) <= std::max(id_column, taxa_column)) {
      problems.push_back(absl::StrFormat(
          "line %d: expected at least %d tab-separated columns, found %d",
          line_number, std::max(id_column, taxa_column) + 1, std::ssize(fields)));
      continue;
    }

    auto spec = Clade_spec{};
    spec.id = fields[id_column];
    spec.config_line = line_number;
    if (spec.id.empty()) {
      problems.push_back(absl::StrFormat("line %d: empty clade_id", line_number));
      continue;
    }
    if (auto [it, inserted] = seen_ids.emplace(spec.id, line_number); not inserted) {
      problems.push_back(absl::StrFormat(
          "line %d: duplicate clade_id '%s' (first defined on line %d); clade IDs must be unique because "
          "they key every output table", line_number, spec.id, it->second));
      continue;
    }

    auto taxa_text = fields[taxa_column];
    auto taxa = std::vector<std::string>{};
    for (auto t : absl::StrSplit(taxa_text, absl::ByAnyChar(", \t"), absl::SkipEmpty())) {
      taxa.push_back(std::string{t});
    }
    if (taxa.empty()) {
      problems.push_back(absl::StrFormat(
          "line %d: clade '%s' lists no taxa; give sample IDs separated by commas or spaces, or the single "
          "word 'all' for the root of the tree", line_number, spec.id));
      continue;
    }
    if (std::ssize(taxa) == 1 && taxa.front() == "all") {
      spec.all_tips = true;
      result.specs.push_back(std::move(spec));
      continue;
    }
    if (std::ranges::find(taxa, "all") != taxa.end()) {
      problems.push_back(absl::StrFormat(
          "line %d: clade '%s' mixes 'all' with a list of sample IDs; 'all' must be the only entry",
          line_number, spec.id));
      continue;
    }
    std::ranges::sort(taxa);
    auto duplicates = std::vector<std::string>{};
    for (auto i = 1; i != std::ssize(taxa); ++i) {
      if (taxa[i] == taxa[i - 1]) { duplicates.push_back(taxa[i]); }
    }
    if (not duplicates.empty()) {
      problems.push_back(absl::StrFormat(
          "line %d: clade '%s' lists %d sample ID(s) more than once: %s",
          line_number, spec.id, std::ssize(duplicates), list_names(duplicates)));
      continue;
    }
    if (std::ssize(taxa) < 2) {
      problems.push_back(absl::StrFormat(
          "line %d: clade '%s' needs at least two sample IDs; the MRCA of a single sample is that sample",
          line_number, spec.id));
      continue;
    }
    spec.taxa = std::move(taxa);
    result.specs.push_back(std::move(spec));
  }

  if (id_column < 0) {
    problems.push_back("the clade definition table is empty: it needs a header line naming the columns "
                       "'clade_id' and 'taxa'");
  }
  if (result.specs.empty() && problems.empty()) {
    problems.push_back("the clade definition table defines no clades");
  }
  throw_problems(problems);
  return result;
}

auto clade_specs_from_calibrations(const Calibration_set& set) -> Clade_spec_set {
  auto result = Clade_spec_set{};
  for (const auto& c : set.calibrations) {
    auto spec = Clade_spec{};
    spec.id = c.name;
    spec.taxa = c.taxa;
    spec.all_tips = c.all_tips;
    spec.config_line = c.config_line;
    result.specs.push_back(std::move(spec));
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// Resolving clades in a posterior tree
// ---------------------------------------------------------------------------------------------

Clade_resolver::Clade_resolver(const Clade_spec_set& specs, const std::vector<std::string>& tip_names)
    : specs_{specs} {
  auto problems = std::vector<std::string>{};
  auto tip_index = absl::flat_hash_map<std::string, Node_index>{};
  for (auto i = 0; i != std::ssize(tip_names); ++i) {
    auto [it, inserted] = tip_index.emplace(tip_names[i], static_cast<Node_index>(i));
    if (not inserted) {
      problems.push_back(absl::StrFormat(
          "the posterior trees have more than one tip named '%s'; clades cannot be resolved by sample ID",
          tip_names[i]));
    }
  }
  num_tips_ = static_cast<int>(std::ssize(tip_names));

  clade_tips_.resize(specs_.specs.size());
  clade_is_all_tips_.resize(specs_.specs.size());
  for (auto c = 0; c != std::ssize(specs_.specs); ++c) {
    const auto& spec = specs_.specs[c];
    clade_is_all_tips_[c] = spec.all_tips;
    if (spec.all_tips) { continue; }
    auto unknown = std::vector<std::string>{};
    auto nodes = std::vector<Node_index>{};
    for (const auto& name : spec.taxa) {
      auto it = tip_index.find(name);
      if (it == tip_index.end()) { unknown.push_back(name); continue; }
      nodes.push_back(it->second);
    }
    if (not unknown.empty()) {
      problems.push_back(absl::StrFormat(
          "clade '%s' (line %d): %d sample ID(s) are not tips of the posterior trees: %s.  Clade targets are "
          "defined by sample ID, so every ID must appear in the analysis that produced these trees",
          spec.id, spec.config_line, std::ssize(unknown), list_names(unknown)));
      continue;
    }
    std::ranges::sort(nodes);
    clade_tips_[c] = std::move(nodes);
  }
  throw_problems(problems);
}

auto Clade_resolver::youngest_tip_time(const Phylo_tree& tree) -> double {
  auto result = -std::numeric_limits<double>::infinity();
  for (const auto& node : index_order_traversal(tree)) {
    if (tree.at(node).is_tip()) { result = std::max(result, tree.at(node).t); }
  }
  return result;
}

auto Clade_resolver::observe(const Phylo_tree& tree, std::vector<Observation>& out) const -> void {
  auto num_nodes = static_cast<Node_index>(std::ssize(tree));
  auto depth = Node_vector<int>(num_nodes, 0);
  auto tips_below = Node_vector<int>(num_nodes, 0);

  for (const auto& node : pre_order_traversal(tree)) {
    depth[node] = (node == tree.root) ? 0 : depth[tree.at(node).parent] + 1;
  }
  for (const auto& node : post_order_traversal(tree)) {
    if (tree.at(node).is_tip()) {
      tips_below[node] = 1;
    } else {
      tips_below[node] = 0;
      for (const auto& child : tree.at(node).children) { tips_below[node] += tips_below[child]; }
    }
  }

  // Lowest common ancestor by climbing from the deeper node.  This uses only the parent links, so it
  // is exact even when two node times are equal after rounding in the input file.
  auto lca = [&](Node_index a, Node_index b) {
    while (depth[a] > depth[b]) { a = tree.at(a).parent; }
    while (depth[b] > depth[a]) { b = tree.at(b).parent; }
    while (a != b) {
      a = tree.at(a).parent;
      b = tree.at(b).parent;
    }
    return a;
  };

  out.assign(specs_.specs.size(), Observation{});
  for (auto c = 0; c != std::ssize(specs_.specs); ++c) {
    if (clade_is_all_tips_[c]) {
      out[c].t = tree.at_root().t;
      out[c].num_tips_below = tips_below[tree.root];
      out[c].monophyletic = true;
      continue;
    }
    const auto& nodes = clade_tips_[c];
    CHECK_GE(std::ssize(nodes), 2);
    auto mrca = nodes.front();
    for (auto i = 1; i != std::ssize(nodes); ++i) { mrca = lca(mrca, nodes[i]); }
    out[c].t = tree.at(mrca).t;
    out[c].num_tips_below = tips_below[mrca];
    out[c].monophyletic = (tips_below[mrca] == std::ssize(nodes));
  }
}

// ---------------------------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------------------------

namespace {

auto calc_age_statistics(
    const std::vector<double>& values,
    const std::vector<int>& chain_of_value,
    int num_chains,
    const Summary_thresholds& thresholds)
    -> Age_statistics {

  auto result = Age_statistics{};
  result.credible_mass = thresholds.credible_mass;
  result.num_samples = static_cast<int>(std::ssize(values));
  if (result.num_samples == 0) { return result; }

  auto moments = calc_moments(values);
  result.mean = moments.mean;
  result.stddev = moments.stddev;

  auto sorted = values;
  std::ranges::sort(sorted);
  result.median = sample_quantile(sorted, 0.5);
  result.shortest_interval = shortest_credible_interval(sorted, thresholds.credible_mass);
  result.eti = equal_tailed_interval(sorted, thresholds.credible_mass);
  result.hdr = highest_density_region(sorted, thresholds.credible_mass, thresholds.hdr_num_bins);

  // Per-chain series, in the order the samples were read, for ESS and R-hat.
  auto chains = std::vector<std::vector<double>>(std::max(num_chains, 1));
  for (auto i = 0; i != std::ssize(values); ++i) {
    auto chain = chain_of_value.empty() ? 0 : chain_of_value[i];
    chains.at(chain).push_back(values[i]);
  }
  std::erase_if(chains, [](const std::vector<double>& c) { return c.empty(); });
  result.num_chains = static_cast<int>(std::ssize(chains));

  // The ESS of several independent chains is the sum of their ESS values.
  auto ess_total = 0.0;
  auto num_lags = 0;
  auto truncated = false;
  auto tau_weighted = 0.0;
  for (const auto& c : chains) {
    auto e = effective_sample_size(c);
    ess_total += e.ess;
    num_lags = std::max(num_lags, e.num_lags_used);
    truncated = truncated || e.truncated;
    tau_weighted += e.tau * std::ssize(c);
  }
  result.ess.ess = ess_total;
  result.ess.tau = tau_weighted / result.num_samples;
  result.ess.num_lags_used = num_lags;
  result.ess.truncated = truncated;

  result.split_rhat = split_rhat(chains);
  result.rank_split_rhat = rank_normalized_split_rhat(chains);
  return result;
}

auto add_diagnostic_warnings(
    const Age_statistics& stats,
    std::string_view prefix,
    const Summary_thresholds& thresholds,
    std::vector<std::string>& warnings)
    -> void {
  if (not stats.is_defined()) { return; }
  if (stats.ess.ess < thresholds.min_ess) {
    warnings.push_back(absl::StrFormat("%s_low_ess(%.0f<%.0f)", prefix, stats.ess.ess, thresholds.min_ess));
  }
  if (stats.ess.truncated) {
    warnings.push_back(absl::StrFormat("%s_ess_lag_truncated", prefix));
  }
  if (stats.rank_split_rhat > thresholds.max_rank_split_rhat) {
    warnings.push_back(absl::StrFormat(
        "%s_rank_split_rhat(%.4f>%.4f)", prefix, stats.rank_split_rhat, thresholds.max_rank_split_rhat));
  }
  if (stats.hdr.num_components > 1) {
    // The shortest contiguous interval bridges a low-density gap here, so it is not a highest
    // posterior density interval and must not be reported as one.
    warnings.push_back(absl::StrFormat(
        "%s_multimodal_interval_is_not_hpd(%d_hdr_components,%d_bins,gap=%.2f)",
        prefix, stats.hdr.num_components, stats.hdr.num_bins, stats.hdr.interval_gap_fraction));
  }
}

}  // namespace

auto summarize_clade(
    const Clade_samples& samples,
    const Clade_spec& spec,
    int num_chains,
    const Summary_thresholds& thresholds)
    -> Clade_summary {

  auto result = Clade_summary{};
  result.clade_id = samples.clade_id;
  result.num_taxa_specified = spec.all_tips ? 0 : spec.num_taxa();
  result.all_tips = spec.all_tips;
  result.num_samples_total = samples.num_samples();
  result.num_samples_per_chain.assign(std::max(num_chains, 1), 0);
  for (auto chain : samples.chain_of_sample) { result.num_samples_per_chain.at(chain) += 1; }

  if (result.num_samples_total == 0) {
    result.warnings.push_back("no_post_burn_in_samples");
    return result;
  }

  auto mono_values = std::vector<double>{};
  auto mono_chains = std::vector<int>{};
  auto tips_below = std::vector<int>{};
  for (auto i = 0; i != result.num_samples_total; ++i) {
    tips_below.push_back(samples.num_tips_below[i]);
    if (samples.monophyletic[i]) {
      ++result.num_monophyletic_samples;
      mono_values.push_back(samples.age_years[i]);
      mono_chains.push_back(samples.chain_of_sample.empty() ? 0 : samples.chain_of_sample[i]);
    }
  }
  result.posterior_support =
      static_cast<double>(result.num_monophyletic_samples) / result.num_samples_total;

  std::ranges::sort(tips_below);
  result.min_tips_below = tips_below.front();
  result.max_tips_below = tips_below.back();
  result.median_tips_below = tips_below[std::ssize(tips_below) / 2];

  result.mrca = calc_age_statistics(samples.age_years, samples.chain_of_sample, num_chains, thresholds);
  if (not mono_values.empty()) {
    result.clade = calc_age_statistics(mono_values, mono_chains, num_chains, thresholds);
  }

  add_diagnostic_warnings(result.mrca, "mrca", thresholds, result.warnings);
  if (result.num_monophyletic_samples == 0) {
    result.warnings.push_back("never_monophyletic");
  } else if (result.posterior_support < 1.0) {
    result.warnings.push_back(absl::StrFormat("not_always_monophyletic(support=%.4f)", result.posterior_support));
    add_diagnostic_warnings(result.clade, "clade", thresholds, result.warnings);
  }
  if (std::ssize(result.num_samples_per_chain) > 1) {
    for (auto i = 0; i != std::ssize(result.num_samples_per_chain); ++i) {
      if (result.num_samples_per_chain[i] == 0) {
        result.warnings.push_back(absl::StrFormat("chain_%d_contributed_no_samples", i));
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// Output tables
// ---------------------------------------------------------------------------------------------

namespace {

auto fmt(double x) -> std::string {
  if (std::isnan(x)) { return "NA"; }
  if (std::isinf(x)) { return x > 0 ? "inf" : "-inf"; }
  return absl::StrFormat("%.10g", x);
}

// Raw posterior samples are written with enough digits to round-trip every double exactly, so that
// re-summarizing an exported table reproduces the original summary bit for bit.
auto fmt_exact(double x) -> std::string {
  if (std::isnan(x)) { return "NA"; }
  if (std::isinf(x)) { return x > 0 ? "inf" : "-inf"; }
  return absl::StrFormat("%.17g", x);
}

auto fmt_interval_lo(const Credible_interval& i) -> std::string {
  return i.is_defined() ? fmt(i.lo) : std::string{"NA"}; }
auto fmt_interval_hi(const Credible_interval& i) -> std::string {
  return i.is_defined() ? fmt(i.hi) : std::string{"NA"}; }

auto fmt_rhat(double r) -> std::string { return r < 0.0 ? std::string{"NA"} : fmt(r); }

// The highest density region as a list of intervals, e.g. "1000-2000;5000-6000".
auto fmt_hdr(const Highest_density_region& hdr) -> std::string {
  if (hdr.components.empty()) { return "NA"; }
  auto parts = std::vector<std::string>{};
  for (const auto& c : hdr.components) { parts.push_back(absl::StrFormat("%s..%s", fmt(c.lo), fmt(c.hi))); }
  return absl::StrJoin(parts, ";");
}

auto write_provenance(const Summary_provenance& p, std::ostream& os) -> void {
  os << "# Delphy posterior clade age summary\n";
  for (const auto& f : p.input_files) { os << "# input_file\t" << f << "\n"; }
  os << "# burn_in\t" << p.burn_in_description << "\n";
  os << "# age_reference\t" << p.age_reference_description << "\n";
  os << "# time_anchor\t" << p.anchor_description << "\n";
  os << absl::StreamFormat("# credible_mass\t%.6g\n", p.credible_mass);
  os << absl::StreamFormat("# min_ess\t%.6g\n", p.min_ess);
  os << absl::StreamFormat("# max_rank_split_rhat\t%.6g\n", p.max_rank_split_rhat);
  for (const auto& n : p.notes) { os << "# note\t" << n << "\n"; }
}

const char* const k_summary_columns[] = {
  "clade_id",
  "clade_definition",
  "num_taxa_specified",
  "calibration_status",
  "posterior_support",
  "num_posterior_samples",
  "num_monophyletic_samples",
  "num_chains",
  "samples_per_chain",
  "age_units",
  "age_reference",
  "credible_mass",
  "mrca_age_mean",
  "mrca_age_median",
  "mrca_age_sd",
  "mrca_age_shortest_interval_lower",
  "mrca_age_shortest_interval_upper",
  "mrca_age_interval_is_hpd",
  "mrca_age_eti_lower",
  "mrca_age_eti_upper",
  "mrca_age_ess",
  "mrca_age_split_rhat",
  "mrca_age_rank_split_rhat",
  "mrca_age_hdr_components",
  "mrca_age_hdr_intervals",
  "clade_age_mean",
  "clade_age_median",
  "clade_age_sd",
  "clade_age_shortest_interval_lower",
  "clade_age_shortest_interval_upper",
  "clade_age_interval_is_hpd",
  "clade_age_eti_lower",
  "clade_age_eti_upper",
  "clade_age_ess",
  "clade_age_split_rhat",
  "clade_age_rank_split_rhat",
  "clade_age_hdr_components",
  "clade_age_hdr_intervals",
  "tips_below_mrca_min",
  "tips_below_mrca_median",
  "tips_below_mrca_max",
  "diagnostics",
};

}  // namespace

auto write_clade_summary_table(
    const std::vector<Clade_summary>& summaries,
    const Summary_provenance& provenance,
    std::ostream& os)
    -> void {
  write_provenance(provenance, os);
  os << "# mrca_age_*  : age of the MRCA of the listed samples in every post-burn-in tree, whether or not\n"
     << "#               the samples are monophyletic there (the quantity Mcc_node::t_mrca averages).\n"
     << "# clade_age_*  : the same age restricted to the trees in which they are monophyletic, i.e.\n"
     << "#               conditional on the clade existing (the quantity Mcc_node::t averages).\n"
     << "# shortest_interval_* : the shortest contiguous interval containing ceil(credible_mass * n)\n"
     << "#               samples.  It estimates the highest posterior density interval only when the\n"
     << "#               highest density region is a single interval, which is what interval_is_hpd says.\n"
     << "# interval_is_hpd : yes when hdr_components == 1, so the shortest interval above may be called\n"
     << "#               a highest posterior density interval; no when the posterior is multimodal and\n"
     << "#               that interval bridges a region of low density.  In that case read hdr_intervals.\n"
     << "# hdr_intervals : the estimated highest density region itself, as `lo..hi` parts separated by\n"
     << "#               `;`.  This is the set that the term HPD names; it need not be an interval.\n"
     << "# eti_*        : equal-tailed interval (quantiles), which is not a highest-density interval.\n";

  auto first = true;
  for (const auto* c : k_summary_columns) {
    if (not first) { os << "\t"; }
    first = false;
    os << c;
  }
  os << "\n";

  for (const auto& s : summaries) {
    auto per_chain = std::vector<std::string>{};
    for (auto n : s.num_samples_per_chain) { per_chain.push_back(absl::StrFormat("%d", n)); }
    auto fields = std::vector<std::string>{
      s.clade_id,
      s.all_tips ? std::string{"all_tips"} : absl::StrFormat("%d_named_taxa", s.num_taxa_specified),
      absl::StrFormat("%d", s.num_taxa_specified),
      s.calibration_status,
      fmt(s.posterior_support),
      absl::StrFormat("%d", s.num_samples_total),
      absl::StrFormat("%d", s.num_monophyletic_samples),
      absl::StrFormat("%d", std::ssize(s.num_samples_per_chain)),
      absl::StrJoin(per_chain, ","),
      "years",
      provenance.age_reference_description,
      fmt(s.mrca.credible_mass),
      fmt(s.mrca.mean), fmt(s.mrca.median), fmt(s.mrca.stddev),
      fmt_interval_lo(s.mrca.shortest_interval), fmt_interval_hi(s.mrca.shortest_interval),
      s.mrca.interval_is_hpd() ? "yes" : "no",
      fmt_interval_lo(s.mrca.eti), fmt_interval_hi(s.mrca.eti),
      fmt(s.mrca.ess.ess), fmt_rhat(s.mrca.split_rhat), fmt_rhat(s.mrca.rank_split_rhat),
      absl::StrFormat("%d", s.mrca.hdr.num_components),
      fmt_hdr(s.mrca.hdr),
    };
    if (s.clade.is_defined()) {
      fields.insert(fields.end(), {
        fmt(s.clade.mean), fmt(s.clade.median), fmt(s.clade.stddev),
        fmt_interval_lo(s.clade.shortest_interval), fmt_interval_hi(s.clade.shortest_interval),
        s.clade.interval_is_hpd() ? "yes" : "no",
        fmt_interval_lo(s.clade.eti), fmt_interval_hi(s.clade.eti),
        fmt(s.clade.ess.ess), fmt_rhat(s.clade.split_rhat), fmt_rhat(s.clade.rank_split_rhat),
        absl::StrFormat("%d", s.clade.hdr.num_components),
        fmt_hdr(s.clade.hdr),
      });
    } else {
      fields.insert(fields.end(),
                    {"NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "0", "NA"});
    }
    fields.insert(fields.end(), {
      absl::StrFormat("%d", s.min_tips_below),
      absl::StrFormat("%d", s.median_tips_below),
      absl::StrFormat("%d", s.max_tips_below),
      s.warnings.empty() ? std::string{"ok"} : absl::StrJoin(s.warnings, ";"),
    });
    CHECK_EQ(std::ssize(fields), std::ssize(k_summary_columns));
    os << absl::StrJoin(fields, "\t") << "\n";
  }
}

auto write_clade_samples_table(
    const std::vector<Clade_samples>& samples,
    const std::vector<std::int64_t>& states,
    const Summary_provenance& provenance,
    std::ostream& os)
    -> void {
  write_provenance(provenance, os);
  os << "# One row per (clade, post-burn-in tree).  age_years is the age of the MRCA of the clade's samples\n"
     << "# in that tree, on the time axis named by age_reference above.\n";
  os << "clade_id\tchain\tstate\tage_years\tmonophyletic\ttips_below_mrca\n";
  for (const auto& s : samples) {
    for (auto i = 0; i != s.num_samples(); ++i) {
      os << absl::StreamFormat("%s\t%d\t%d\t%s\t%d\t%d\n",
                               s.clade_id,
                               s.chain_of_sample.empty() ? 0 : s.chain_of_sample[i],
                               states.empty() ? std::int64_t{-1} : states[i],
                               fmt_exact(s.age_years[i]),
                               s.monophyletic[i] ? 1 : 0,
                               s.num_tips_below[i]);
    }
  }
}

auto read_clade_samples_table(std::istream& is) -> Clade_samples_file {
  auto problems = std::vector<std::string>{};
  auto result = Clade_samples_file{};
  auto by_id = std::map<std::string, int>{};
  auto header_seen = false;
  auto columns = absl::flat_hash_map<std::string, int>{};
  auto line = std::string{};
  auto line_number = 0;
  auto max_chain = 0;
  auto states_of_first_clade = std::vector<std::int64_t>{};

  while (std::getline(is, line)) {
    ++line_number;
    auto content = trim(line);
    if (content.empty()) { continue; }
    if (content.front() == '#') { result.comment_lines.push_back(std::string{content}); continue; }

    auto fields = std::vector<std::string>{};
    for (auto f : absl::StrSplit(line, '\t')) { fields.push_back(std::string{trim(f)}); }

    if (not header_seen) {
      for (auto i = 0; i != std::ssize(fields); ++i) { columns.emplace(fields[i], i); }
      for (const auto* needed : {"clade_id", "chain", "age_years", "monophyletic", "tips_below_mrca"}) {
        if (not columns.contains(needed)) {
          problems.push_back(absl::StrFormat(
              "line %d: the raw samples table must have a column '%s'", line_number, needed));
        }
      }
      throw_problems(problems);
      header_seen = true;
      continue;
    }

    auto field = [&](const char* name) -> const std::string& {
      return fields.at(columns.at(name)); };
    if (std::ssize(fields) < std::ssize(columns)) {
      problems.push_back(absl::StrFormat("line %d: expected %d columns, found %d",
                                         line_number, std::ssize(columns), std::ssize(fields)));
      continue;
    }

    auto id = field("clade_id");
    auto [it, inserted] = by_id.emplace(id, static_cast<int>(std::ssize(result.samples)));
    if (inserted) {
      auto s = Clade_samples{};
      s.clade_id = id;
      result.samples.push_back(std::move(s));
    }
    auto& s = result.samples[it->second];

    auto chain = 0;
    auto age = 0.0;
    auto mono = 0;
    auto tips = 0;
    auto state = std::int64_t{-1};
    auto ok = absl::SimpleAtoi(field("chain"), &chain)
        && absl::SimpleAtod(field("age_years"), &age)
        && absl::SimpleAtoi(field("monophyletic"), &mono)
        && absl::SimpleAtoi(field("tips_below_mrca"), &tips)
        && (not columns.contains("state") || absl::SimpleAtoi(field("state"), &state));
    if (not ok || chain < 0) {
      problems.push_back(absl::StrFormat("line %d: could not parse the numeric fields", line_number));
      continue;
    }
    max_chain = std::max(max_chain, chain);
    s.age_years.push_back(age);
    s.monophyletic.push_back(mono != 0 ? 1 : 0);
    s.num_tips_below.push_back(tips);
    s.chain_of_sample.push_back(chain);
    if (it->second == 0) { states_of_first_clade.push_back(state); }
  }

  if (not header_seen) { problems.push_back("the raw samples table has no header line"); }
  if (result.samples.empty() && problems.empty()) {
    problems.push_back("the raw samples table contains no rows");
  }
  for (const auto& s : result.samples) {
    if (s.num_samples() != result.samples.front().num_samples()) {
      problems.push_back(absl::StrFormat(
          "clade '%s' has %d rows but clade '%s' has %d; every clade must be observed in the same trees",
          s.clade_id, s.num_samples(), result.samples.front().clade_id,
          result.samples.front().num_samples()));
    }
  }
  throw_problems(problems);
  result.num_chains = max_chain + 1;
  result.states = std::move(states_of_first_clade);
  return result;
}

}  // namespace delphy
