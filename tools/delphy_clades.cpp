#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "cxxopts.hpp"

#include "beasty_input.h"
#include "calibration.h"
#include "clade_summary.h"
#include "human_dates.h"
#include "human_input.h"
#include "scratch_space.h"
#include "version.h"

namespace delphy {

namespace {

struct Burn_in_choice {
  enum class Mode { fraction, states, samples } mode = Mode::fraction;
  double fraction = 0.3;
  std::int64_t states = 0;
  int samples = 0;
};

// What one input trees file contributes.
struct Chain_input {
  std::string path;
  std::vector<std::int64_t> all_states;
  std::int64_t min_state_kept = 0;   // Trees with state < this are burn-in
  int num_burn_in_trees = 0;
  int num_kept_trees = 0;
};

auto open_in(const std::string& path) -> std::ifstream {
  auto is = std::ifstream{path};
  if (not is) {
    throw std::runtime_error{absl::StrFormat("cannot open '%s' for reading", path)};
  }
  return is;
}

auto open_out(const std::string& path) -> std::ofstream {
  auto os = std::ofstream{path};
  if (not os) {
    throw std::runtime_error{absl::StrFormat("cannot open '%s' for writing", path)};
  }
  return os;
}

// Decides which states of one chain are post-burn-in.
auto apply_burn_in(Chain_input& chain, const Burn_in_choice& choice, std::int64_t thin) -> void {
  auto states = std::vector<std::int64_t>{};
  for (auto s : chain.all_states) {
    if (thin > 1 && (s % thin) != 0) { continue; }
    states.push_back(s);
  }
  if (states.empty()) {
    chain.min_state_kept = std::numeric_limits<std::int64_t>::max();
    return;
  }
  switch (choice.mode) {
    case Burn_in_choice::Mode::fraction: {
      auto min_state = states.front();
      auto max_state = states.back();
      // Same rule as delphy_mcc's historical 30% burn-in: a fraction of the *state* range, and the
      // threshold state itself is discarded.
      auto cut = min_state + static_cast<std::int64_t>(
          std::floor(choice.fraction * static_cast<double>(max_state - min_state)));
      chain.min_state_kept = cut + 1;
      break;
    }
    case Burn_in_choice::Mode::states:
      chain.min_state_kept = choice.states + 1;
      break;
    case Burn_in_choice::Mode::samples: {
      auto k = std::min<int>(choice.samples, static_cast<int>(std::ssize(states)));
      chain.min_state_kept = (k == std::ssize(states))
          ? std::numeric_limits<std::int64_t>::max()
          : states[k];
      break;
    }
  }
  for (auto s : states) {
    if (s < chain.min_state_kept) { ++chain.num_burn_in_trees; } else { ++chain.num_kept_trees; }
  }
}

auto describe_burn_in(const Burn_in_choice& choice, std::int64_t thin, const std::vector<Chain_input>& chains)
    -> std::string {
  auto head = std::string{};
  switch (choice.mode) {
    case Burn_in_choice::Mode::fraction:
      head = absl::StrFormat("fraction %.6g of the state range of each chain", choice.fraction);
      break;
    case Burn_in_choice::Mode::states:
      head = absl::StrFormat("every state <= %d", choice.states);
      break;
    case Burn_in_choice::Mode::samples:
      head = absl::StrFormat("the first %d sampled trees of each chain", choice.samples);
      break;
  }
  auto per_chain = std::vector<std::string>{};
  for (const auto& c : chains) {
    per_chain.push_back(absl::StrFormat("%s: %d trees read, %d discarded, %d kept, first kept state %d",
                                        c.path, std::ssize(c.all_states), c.num_burn_in_trees,
                                        c.num_kept_trees, c.min_state_kept));
  }
  if (thin > 1) { head += absl::StrFormat("; thinned to states that are multiples of %d", thin); }
  return head + " [" + absl::StrJoin(per_chain, "; ") + "]";
}

// The instant that reported ages are measured back from.
struct Time_axis {
  bool anchored = false;
  std::string anchor_tip;
  std::string anchor_date_text;
  double anchor_t_abs = 0.0;       // Days since 2020-01-01
  Node_index anchor_tip_node = k_no_node;
  std::int64_t reference_year = 1950;
  double reference_t = 0.0;        // Days since 2020-01-01 of 1 Jan of reference_year
  std::string description;
};

auto choose_anchor(
    const Sample_metadata& metadata,
    const std::vector<std::string>& tip_names,
    const std::optional<std::string>& requested_tip)
    -> std::pair<std::string, Time_bounds> {

  auto tip_set = absl::flat_hash_set<std::string>{tip_names.begin(), tip_names.end()};
  auto exact = std::vector<std::pair<std::string, Time_bounds>>{};
  auto problems = std::vector<std::string>{};
  for (const auto& row : metadata.rows) {
    if (not row.date.has_value()) { continue; }
    if (not tip_set.contains(row.sample_id)) { continue; }
    auto bounds = Time_bounds{};
    try {
      bounds = convert_sample_date(*row.date);
    } catch (const std::exception& e) {
      problems.push_back(absl::StrFormat("metadata line %d (sample '%s'): %s",
                                         row.line, row.sample_id, e.what()));
      continue;
    }
    if (bounds.t_min == bounds.t_max) { exact.emplace_back(row.sample_id, bounds); }
  }
  if (not problems.empty()) { throw std::runtime_error{absl::StrJoin(problems, "\n")}; }

  if (requested_tip.has_value()) {
    for (const auto& [id, bounds] : exact) {
      if (id == *requested_tip) { return {id, bounds}; }
    }
    throw std::runtime_error{absl::StrFormat(
        "--anchor-tip '%s' is not a sample with an exactly known date in the metadata and a tip of the "
        "posterior trees.  The anchor fixes the absolute time axis, so it must be a sample whose date is a "
        "single instant (date_to empty or equal to date_from, and an ISO day rather than a month or a year)",
        *requested_tip)};
  }
  if (exact.empty()) {
    throw std::runtime_error{
        "none of the samples in --tip-dates has an exactly known sampling date, so there is nothing to anchor "
        "the absolute time axis to.  Either give one sample an exact ISO day, or drop --tip-dates and report "
        "ages relative to the most recent tip of each posterior tree"};
  }
  // Most recent first; ties broken by sample ID, so the choice does not depend on file order.
  std::ranges::sort(exact, [](const auto& a, const auto& b) {
    return a.second.t_max != b.second.t_max ? a.second.t_max > b.second.t_max : a.first < b.first; });
  return exact.front();
}

auto format_t_as_date(double t_days) -> std::string {
  // Years from 2020-01-01, using the 365-day year that the CLI uses everywhere else.
  return absl::StrFormat("%.6f days since 2020-01-01 (%.4f years)", t_days, t_days / 365.0);
}

struct Per_chain_row {
  std::string clade_id;
  int chain = 0;
  std::string path;
  Clade_summary summary;
};

auto write_per_chain_table(
    const std::vector<Per_chain_row>& rows,
    const Summary_provenance& provenance,
    std::ostream& os)
    -> void {
  os << "# Delphy posterior clade age summary, one row per (clade, chain)\n";
  os << "# age_reference\t" << provenance.age_reference_description << "\n";
  os << "clade_id\tchain\tinput_file\tnum_posterior_samples\tposterior_support\t"
     << "mrca_age_mean\tmrca_age_median\tmrca_age_sd\t"
     << "mrca_age_shortest_interval_lower\tmrca_age_shortest_interval_upper\tmrca_age_interval_is_hpd\t"
     << "mrca_age_eti_lower\tmrca_age_eti_upper\tmrca_age_ess\tmrca_age_split_rhat\tdiagnostics\n";
  for (const auto& r : rows) {
    const auto& m = r.summary.mrca;
    os << absl::StreamFormat("%s\t%d\t%s\t%d\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%s\t"
                             "%.10g\t%.10g\t%.10g\t",
                             r.clade_id, r.chain, r.path, r.summary.num_samples_total,
                             r.summary.posterior_support,
                             m.mean, m.median, m.stddev,
                             m.shortest_interval.lo, m.shortest_interval.hi,
                             m.interval_is_hpd() ? "yes" : "no",
                             m.eti.lo, m.eti.hi, m.ess.ess);
    if (m.split_rhat < 0.0) { os << "NA"; } else { os << absl::StreamFormat("%.10g", m.split_rhat); }
    os << "\t" << (r.summary.warnings.empty() ? std::string{"ok"} : absl::StrJoin(r.summary.warnings, ";"))
       << "\n";
  }
}

auto run(int argc, char** argv) -> int {
  auto options = cxxopts::Options{
      "delphy_clades",
      "Posterior tMRCA, HPD and convergence summaries of named clades from Delphy or BEAST trees files"};

  options.add_options("Generic")
      ("version", "print version string")
      ("h,help", "print usage")
      ;
  options.add_options("Input")
      ("trees", "BEAST-format posterior trees file.  Repeat the option once per MCMC chain; all chains must "
                "have the same taxa in the same order.",
       cxxopts::value<std::vector<std::string>>())
      ("in-samples", "Raw per-tree samples table written earlier by --out-samples, summarized again without "
                     "re-reading any tree.  Mutually exclusive with --trees.",
       cxxopts::value<std::string>())
      ("clades", "Clade definition table: tab-separated, with a header naming the columns 'clade_id' and "
                 "'taxa'.  'taxa' lists sample IDs separated by commas or spaces, or the single word 'all' "
                 "for the root of the tree.",
       cxxopts::value<std::string>())
      ("calibration-file", "A calibration configuration (see --v0-calibration-file).  Its calibrated clades "
                           "are summarized as well, and every clade's calibration status is reported.",
       cxxopts::value<std::string>())
      ;
  options.add_options("Burn-in and thinning")
      ("burn-in-fraction", "Discard the first fraction of the state range of each chain (default 0.3, which is "
                           "what delphy_mcc has always applied)",
       cxxopts::value<double>())
      ("burn-in-states", "Discard every sampled state <= this number", cxxopts::value<std::int64_t>())
      ("burn-in-samples", "Discard the first this many sampled trees of each chain", cxxopts::value<int>())
      ("thin", "Keep only states that are multiples of this number", cxxopts::value<std::int64_t>())
      ;
  options.add_options("Time axis")
      ("tip-dates", "Sample metadata table (the same one the run used) giving the sampling dates.  One exactly "
                    "dated sample anchors the absolute time axis; without this option ages are reported "
                    "relative to the most recent tip of each posterior tree, which is not a calendar age.",
       cxxopts::value<std::string>())
      ("uniform-tip-date", "ISO day (YYYY-MM-DD) shared by every sample, as an alternative to "
                           "--tip-dates when the analysis gave every sample one sampling date.  It must "
                           "be an exact day, not a month or a year.",
       cxxopts::value<std::string>())
      ("anchor-tip", "Sample ID to anchor the absolute time axis on (default: the most recent exactly dated "
                     "sample, ties broken by sample ID)",
       cxxopts::value<std::string>())
      ("age-reference-year", "Ages are reported in years before 1 January of this year CE, in astronomical "
                             "year numbering (default 1950, i.e. calibrated-radiocarbon BP).  Requires "
                             "--tip-dates.",
       cxxopts::value<std::int64_t>())
      ;
  options.add_options("Statistics")
      ("credible-mass", "Credible mass of the HPD and equal-tailed intervals (default 0.95)",
       cxxopts::value<double>())
      ("min-ess", "Report a diagnostic when the effective sample size is below this (default 200)",
       cxxopts::value<double>())
      ("max-rank-split-rhat", "Report a diagnostic when the rank-normalized split R-hat exceeds this "
                              "(default 1.01)",
       cxxopts::value<double>())
      ("hdr-bins", "Number of histogram bins for the multimodality diagnostic (default: from the sample size)",
       cxxopts::value<int>())
      ;
  options.add_options("Output")
      ("out-summary", "Summary table, one row per clade (default: standard output)",
       cxxopts::value<std::string>())
      ("out-samples", "Raw per-tree tMRCA samples", cxxopts::value<std::string>())
      ("out-per-chain", "Summary table with one row per (clade, chain)", cxxopts::value<std::string>())
      ("fail-on-diagnostics", "Exit with a non-zero status if any clade trips a diagnostic threshold",
       cxxopts::value<bool>()->default_value("false"))
      ;

  auto args = options.parse(argc, argv);

  if (args.count("version")) {
    std::cout << absl::StreamFormat("Delphy Version %s (build %d, commit %s)",
                                    k_delphy_version_string, k_delphy_build_number, k_delphy_commit_string)
              << "\n";
    std::cout << absl::StreamFormat("Delphy-Extended %s (extends upstream Delphy at commit %s)",
                                    k_delphy_extended_version_string, k_delphy_extended_upstream_commit)
              << "\n";
    return EXIT_SUCCESS;
  }
  if (args.count("help") || (args.count("trees") == 0 && args.count("in-samples") == 0)) {
    std::cerr << options.help() << "\n";
    std::cerr << "Ages are always reported in years of age.  The reference instant is named in the output.\n"
                 "mrca_age_* is the age of the MRCA of the named samples in every post-burn-in tree;\n"
                 "clade_age_* is the same age over only the trees in which they are monophyletic.\n";
    return args.count("help") ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (args.count("trees") && args.count("in-samples")) {
    std::cerr << "ERROR: --trees and --in-samples are mutually exclusive\n";
    return EXIT_FAILURE;
  }

  auto burn_in = Burn_in_choice{};
  auto num_burn_in_options = args.count("burn-in-fraction") + args.count("burn-in-states")
      + args.count("burn-in-samples");
  if (num_burn_in_options > 1) {
    std::cerr << "ERROR: give at most one of --burn-in-fraction, --burn-in-states, --burn-in-samples\n";
    return EXIT_FAILURE;
  }
  if (args.count("burn-in-fraction")) {
    burn_in.mode = Burn_in_choice::Mode::fraction;
    burn_in.fraction = args["burn-in-fraction"].as<double>();
    if (not (burn_in.fraction >= 0.0 && burn_in.fraction < 1.0)) {
      std::cerr << "ERROR: --burn-in-fraction must be in [0, 1)\n";
      return EXIT_FAILURE;
    }
  } else if (args.count("burn-in-states")) {
    burn_in.mode = Burn_in_choice::Mode::states;
    burn_in.states = args["burn-in-states"].as<std::int64_t>();
  } else if (args.count("burn-in-samples")) {
    burn_in.mode = Burn_in_choice::Mode::samples;
    burn_in.samples = args["burn-in-samples"].as<int>();
    if (burn_in.samples < 0) {
      std::cerr << "ERROR: --burn-in-samples must be >= 0\n";
      return EXIT_FAILURE;
    }
  }
  auto thin = args.count("thin") ? args["thin"].as<std::int64_t>() : std::int64_t{1};
  if (thin < 1) {
    std::cerr << "ERROR: --thin must be >= 1\n";
    return EXIT_FAILURE;
  }

  auto thresholds = Summary_thresholds{};
  if (args.count("credible-mass")) { thresholds.credible_mass = args["credible-mass"].as<double>(); }
  if (not (thresholds.credible_mass > 0.0 && thresholds.credible_mass < 1.0)) {
    std::cerr << "ERROR: --credible-mass must be in (0, 1)\n";
    return EXIT_FAILURE;
  }
  if (args.count("min-ess")) { thresholds.min_ess = args["min-ess"].as<double>(); }
  if (args.count("max-rank-split-rhat")) {
    thresholds.max_rank_split_rhat = args["max-rank-split-rhat"].as<double>();
  }
  if (args.count("hdr-bins")) { thresholds.hdr_num_bins = args["hdr-bins"].as<int>(); }

  // ---- Clade definitions -------------------------------------------------------------------
  auto specs = Clade_spec_set{};
  auto calibrations = Calibration_set{};
  if (args.count("calibration-file")) {
    auto is = open_in(args["calibration-file"].as<std::string>());
    calibrations = read_calibration_config(is);
  }
  if (args.count("clades")) {
    auto is = open_in(args["clades"].as<std::string>());
    specs = read_clade_specs(is);
  }
  {  // Calibrated clades are summarized too, unless the clade table already names the same set.
    auto from_cal = clade_specs_from_calibrations(calibrations);
    auto existing = absl::flat_hash_set<std::string>{};
    for (const auto& s : specs.specs) { existing.insert(s.id); }
    for (auto& s : from_cal.specs) {
      if (existing.contains(s.id)) { continue; }
      specs.specs.push_back(std::move(s));
    }
  }
  if (specs.empty()) {
    std::cerr << "ERROR: no clades to summarize.  Give --clades and/or --calibration-file\n";
    return EXIT_FAILURE;
  }

  // Calibration status per clade: a clade is "calibrated" when its taxon set is exactly a calibrated
  // set.  Comparison is on the sorted sample IDs, not on the clade's name.
  auto calibration_status = std::vector<std::string>(specs.specs.size(), "not_calibrated");
  for (auto i = 0; i != std::ssize(specs.specs); ++i) {
    for (const auto& c : calibrations.calibrations) {
      auto same = (c.all_tips && specs.specs[i].all_tips)
          || (not c.all_tips && not specs.specs[i].all_tips && c.taxa == specs.specs[i].taxa);
      if (same) {
        calibration_status[i] = absl::StrFormat("calibrated:%s", c.name);
        break;
      }
    }
  }
  if (calibrations.empty()) {
    for (auto& s : calibration_status) { s = "unknown"; }
  }

  auto provenance = Summary_provenance{};
  provenance.credible_mass = thresholds.credible_mass;
  provenance.min_ess = thresholds.min_ess;
  provenance.max_rank_split_rhat = thresholds.max_rank_split_rhat;
  if (calibrations.empty()) {
    provenance.notes.push_back(
        "no calibration configuration was given, so the calibration status of every clade is 'unknown'");
  }

  auto samples = std::vector<Clade_samples>{};
  auto states = std::vector<std::int64_t>{};
  auto num_chains = 0;
  auto chain_paths = std::vector<std::string>{};

  if (args.count("in-samples")) {
    // ---- Re-summarize an earlier export ----------------------------------------------------
    auto path = args["in-samples"].as<std::string>();
    auto is = open_in(path);
    auto file = read_clade_samples_table(is);
    auto by_id = std::map<std::string, const Clade_samples*>{};
    for (const auto& s : file.samples) { by_id.emplace(s.clade_id, &s); }
    for (const auto& spec : specs.specs) {
      auto it = by_id.find(spec.id);
      if (it == by_id.end()) {
        std::cerr << absl::StreamFormat(
            "ERROR: clade '%s' is not in '%s'.  A raw samples table can only be re-summarized for the "
            "clades it was written for\n", spec.id, path) << std::flush;
        return EXIT_FAILURE;
      }
      samples.push_back(*it->second);
    }
    states = file.states;
    num_chains = file.num_chains;
    chain_paths.push_back(path);
    provenance.input_files.push_back(path);
    provenance.burn_in_description = absl::StrFormat(
        "inherited from '%s' (re-summarized without re-reading the trees)", path);
    provenance.anchor_description = "inherited from the raw samples table";
    provenance.age_reference_description = "inherited from the raw samples table";
    for (const auto& c : file.comment_lines) {
      if (c.starts_with("# age_reference\t")) {
        provenance.age_reference_description = c.substr(std::strlen("# age_reference\t"));
      }
      if (c.starts_with("# time_anchor\t")) {
        provenance.anchor_description = c.substr(std::strlen("# time_anchor\t"));
      }
      if (c.starts_with("# burn_in\t")) {
        provenance.notes.push_back("original burn-in: " + c.substr(std::strlen("# burn_in\t")));
      }
    }
    if (num_burn_in_options > 0) {
      provenance.notes.push_back(
          "a burn-in option was given with --in-samples and ignored: the raw samples table is already "
          "post-burn-in");
    }

  } else {
    // ---- Stream the posterior trees --------------------------------------------------------
    auto paths = args["trees"].as<std::vector<std::string>>();
    auto chains = std::vector<Chain_input>{};
    auto tip_names = std::vector<std::string>{};

    for (const auto& path : paths) {
      auto chain = Chain_input{};
      chain.path = path;
      {  // Pass 1: state numbers only, so that a burn-in fraction can be applied before parsing trees
        auto is = open_in(path);
        auto stream = Beasty_trees_stream{is};
        if (tip_names.empty()) {
          tip_names = stream.tip_names();
        } else if (stream.tip_names() != tip_names) {
          std::cerr << absl::StreamFormat(
              "ERROR: '%s' does not have the same taxa, in the same order, as '%s'.  Clade membership is "
              "resolved by tip index, so every chain must come from the same alignment\n",
              path, paths.front()) << std::flush;
          return EXIT_FAILURE;
        }
        is.clear();
        is.seekg(0);
        chain.all_states = scan_beasty_tree_states(is);
      }
      if (chain.all_states.empty()) {
        std::cerr << absl::StreamFormat("ERROR: '%s' contains no trees\n", path) << std::flush;
        return EXIT_FAILURE;
      }
      apply_burn_in(chain, burn_in, thin);
      chains.push_back(std::move(chain));
    }
    num_chains = static_cast<int>(std::ssize(chains));

    auto resolver = Clade_resolver{specs, tip_names};

    samples.assign(specs.specs.size(), Clade_samples{});
    for (auto i = 0; i != std::ssize(specs.specs); ++i) { samples[i].clade_id = specs.specs[i].id; }

    if (args.count("tip-dates") && args.count("uniform-tip-date")) {
      std::cerr << "ERROR: --tip-dates and --uniform-tip-date are mutually exclusive\n";
      return EXIT_FAILURE;
    }

    auto time_axis = Time_axis{};
    if (args.count("tip-dates") || args.count("uniform-tip-date")) {
      auto metadata = Sample_metadata{};
      if (args.count("tip-dates")) {
        auto metadata_path = args["tip-dates"].as<std::string>();
        auto is = open_in(metadata_path);
        metadata = read_sample_metadata(is);
        provenance.input_files.push_back(metadata_path);
      } else {
        // One shared sampling date: synthesize the same metadata row for every tip
        auto date = Sample_date_input{};
        date.convention = Date_convention::iso;
        date.from = args["uniform-tip-date"].as<std::string>();
        for (auto i = 0; i != std::ssize(tip_names); ++i) {
          auto row = Sample_metadata_row{};
          row.line = i + 1;
          row.sample_id = tip_names[i];
          row.date = date;
          metadata.rows.push_back(std::move(row));
        }
        provenance.notes.push_back(absl::StrFormat(
            "every sample was given the sampling date %s with --uniform-tip-date", date.from));
      }
      auto requested = args.count("anchor-tip")
          ? std::optional<std::string>{args["anchor-tip"].as<std::string>()}
          : std::nullopt;
      auto [anchor_id, bounds] = choose_anchor(metadata, tip_names, requested);
      time_axis.anchored = true;
      time_axis.anchor_tip = anchor_id;
      time_axis.anchor_t_abs = bounds.t_min;
      time_axis.reference_year = args.count("age-reference-year")
          ? args["age-reference-year"].as<std::int64_t>()
          : std::int64_t{1950};
      time_axis.reference_t = astronomical_year_to_day(time_axis.reference_year);
      time_axis.description = absl::StrFormat("years_before_%d-01-01 (astronomical year numbering)",
                                              time_axis.reference_year);
      time_axis.anchor_date_text = absl::StrFormat(
          "sample '%s' at %s", anchor_id, format_t_as_date(bounds.t_min));
      for (auto i = 0; i != std::ssize(tip_names); ++i) {
        if (tip_names[i] == anchor_id) { time_axis.anchor_tip_node = static_cast<Node_index>(i); }
      }
      CHECK_NE(time_axis.anchor_tip_node, k_no_node);
    } else {
      if (args.count("age-reference-year") || args.count("anchor-tip")) {
        std::cerr << "ERROR: --age-reference-year and --anchor-tip need --tip-dates or "
                     "--uniform-tip-date: without sampling dates there is no absolute time axis to place "
                     "a reference year on\n";
        return EXIT_FAILURE;
      }
      time_axis.description = "years_before_the_most_recent_tip_of_each_tree";
      time_axis.anchor_date_text = "none (ages are relative, not calendar ages)";
      provenance.notes.push_back(
          "no sampling dates were given, so the reported ages are heights above the most recent tip of each "
          "posterior tree and are not calendar ages");
    }

    auto observations = std::vector<Clade_resolver::Observation>{};
    auto total_trees_read = std::int64_t{0};
    for (auto chain_index = 0; chain_index != num_chains; ++chain_index) {
      auto& chain = chains[chain_index];
      provenance.input_files.push_back(chain.path);
      chain_paths.push_back(chain.path);
      auto is = open_in(chain.path);
      auto stream = Beasty_trees_stream{is};
      stream.set_min_state(chain.min_state_kept);
      stream.set_every(thin);
      auto state = std::int64_t{};
      auto tree = Phylo_tree{};
      while (stream.next(state, tree)) {
        ++total_trees_read;
        resolver.observe(tree, observations);
        auto t_reference_in_tree = time_axis.anchored
            ? tree.at(time_axis.anchor_tip_node).t
                  + (time_axis.reference_t - time_axis.anchor_t_abs)
            : Clade_resolver::youngest_tip_time(tree);
        for (auto i = 0; i != std::ssize(observations); ++i) {
          samples[i].age_years.push_back((t_reference_in_tree - observations[i].t) / 365.0);
          samples[i].monophyletic.push_back(observations[i].monophyletic ? 1 : 0);
          samples[i].num_tips_below.push_back(observations[i].num_tips_below);
          samples[i].chain_of_sample.push_back(chain_index);
        }
        states.push_back(state);
      }
    }
    std::cerr << absl::StreamFormat("Summarized %d post-burn-in trees from %d chain(s) over %d clade(s)\n",
                                    total_trees_read, num_chains, specs.size());

    provenance.burn_in_description = describe_burn_in(burn_in, thin, chains);
    provenance.age_reference_description = time_axis.description;
    provenance.anchor_description = time_axis.anchor_date_text;
  }

  // ---- Summarize ---------------------------------------------------------------------------
  auto summaries = std::vector<Clade_summary>{};
  for (auto i = 0; i != std::ssize(specs.specs); ++i) {
    auto summary = summarize_clade(samples[i], specs.specs[i], num_chains, thresholds);
    summary.calibration_status = calibration_status[i];
    summaries.push_back(std::move(summary));
  }

  if (num_chains < 2) {
    provenance.notes.push_back(
        "only one chain was given: the R-hat reported is the split-chain statistic within that chain, which "
        "cannot detect a chain stuck in the wrong mode.  Run several chains before interpreting any interval");
  }

  auto any_diagnostic = false;
  for (const auto& s : summaries) {
    if (not s.warnings.empty()) { any_diagnostic = true; }
  }
  if (any_diagnostic) {
    std::cerr << "\nWARNING: at least one clade tripped a diagnostic threshold.  The pooled intervals below\n"
                 "are reported, but they are not trustworthy summaries of the posterior until these are\n"
                 "resolved:\n";
    for (const auto& s : summaries) {
      if (s.warnings.empty()) { continue; }
      std::cerr << absl::StreamFormat("  %-24s %s\n", s.clade_id, absl::StrJoin(s.warnings, "; "));
    }
    std::cerr << "\n";
  }

  auto multimodal = std::vector<std::string>{};
  for (const auto& s : summaries) {
    if (s.mrca.is_defined() && not s.mrca.interval_is_hpd()) { multimodal.push_back(s.clade_id); }
  }
  if (not multimodal.empty()) {
    std::cerr << absl::StreamFormat(
        "WARNING: the posterior age of %d clade(s) is multimodal: %s.\n"
        "  For these, the shortest contiguous interval reported in the shortest_interval_* columns is\n"
        "  NOT a highest posterior density interval: it bridges a region of low density between the\n"
        "  modes.  interval_is_hpd is 'no' for them and the hdr_intervals column gives the estimated\n"
        "  highest density region, which is a union of intervals.  Do not quote a single interval for\n"
        "  these clades, and check whether the multimodality is a real feature of the posterior or a\n"
        "  sign that the chains have not mixed (compare mrca_age_rank_split_rhat and the per-chain\n"
        "  table).\n\n",
        std::ssize(multimodal), absl::StrJoin(multimodal, ", "));
  }

  // ---- Write -------------------------------------------------------------------------------
  if (args.count("out-samples")) {
    auto os = open_out(args["out-samples"].as<std::string>());
    write_clade_samples_table(samples, states, provenance, os);
  }
  if (args.count("out-per-chain")) {
    auto rows = std::vector<Per_chain_row>{};
    for (auto i = 0; i != std::ssize(specs.specs); ++i) {
      for (auto chain = 0; chain != std::max(num_chains, 1); ++chain) {
        auto one = Clade_samples{};
        one.clade_id = samples[i].clade_id;
        for (auto k = 0; k != samples[i].num_samples(); ++k) {
          if (samples[i].chain_of_sample[k] != chain) { continue; }
          one.age_years.push_back(samples[i].age_years[k]);
          one.monophyletic.push_back(samples[i].monophyletic[k]);
          one.num_tips_below.push_back(samples[i].num_tips_below[k]);
          one.chain_of_sample.push_back(0);
        }
        auto row = Per_chain_row{};
        row.clade_id = one.clade_id;
        row.chain = chain;
        row.path = chain < std::ssize(chain_paths) ? chain_paths[chain] : std::string{"-"};
        row.summary = summarize_clade(one, specs.specs[i], 1, thresholds);
        rows.push_back(std::move(row));
      }
    }
    auto os = open_out(args["out-per-chain"].as<std::string>());
    write_per_chain_table(rows, provenance, os);
  }
  if (args.count("out-summary")) {
    auto os = open_out(args["out-summary"].as<std::string>());
    write_clade_summary_table(summaries, provenance, os);
  } else {
    write_clade_summary_table(summaries, provenance, std::cout);
  }

  if (any_diagnostic && args["fail-on-diagnostics"].as<bool>()) { return EXIT_FAILURE; }
  return EXIT_SUCCESS;
}

}  // namespace

}  // namespace delphy

auto main(int argc, char** argv) -> int {
  using namespace delphy;

  absl::InitializeLog();
  auto scope = Local_arena_scope{};

  try {
    return run(argc, argv);
  } catch (const Clade_spec_error& e) {
    std::cerr << "ERROR in the clade definitions:\n" << e.what() << "\n";
    return EXIT_FAILURE;
  } catch (const Calibration_error& e) {
    std::cerr << "ERROR in the calibration configuration:\n" << e.what() << "\n";
    return EXIT_FAILURE;
  } catch (const Human_input_error& e) {
    std::cerr << "ERROR in the sample metadata:\n" << e.what() << "\n";
    return EXIT_FAILURE;
  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << "ERROR on the command line: " << e.what() << "\n";
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
