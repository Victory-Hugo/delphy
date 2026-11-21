#include "cmdline.h"

#include <fstream>
#include <iostream>
#include <optional>

#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "cxxopts.hpp"

#include "beasty_input.h"
#include "calibration.h"
#include "calibration_init.h"
#include "dates.h"
#include "evo_subst_model.h"
#include "external_tree.h"
#include "human_input.h"
#include "io.h"
#include "phylo_tree.h"
#include "phylo_tree_calc.h"
#include "sequence_utils.h"
#include "utree.h"
#include "version.h"

namespace delphy {

bool delphy_invoked_via_cli{false};
std::vector<std::string> delphy_cli_args{};

// Repackage a set of equal-length sequences as a reference + diffs
auto fasta_to_maple(
    const std::vector<Fasta_entry>& in_fasta,
    const std::function<void(int,int)>& progress_hook,
    const std::function<void(const std::string&, const Sequence_warning&)>& warning_hook)
    -> Maple_file {
  
  auto result = Maple_file{};
  
  if (in_fasta.empty()) {
    throw std::runtime_error("Input has no sequences!");
  }
  auto L = std::ssize(in_fasta[0].sequence);
  for (auto i = 0; i != std::ssize(in_fasta); ++i) {
    if (std::ssize(in_fasta[i].sequence) != L) {
      throw std::runtime_error(absl::StrFormat(
          "Sequences have different lengths: '%s' has %d sites, while '%s' has %d.  Are these sequences aligned?",
          in_fasta[i].id, std::ssize(in_fasta[i].sequence),
          in_fasta[0].id, std::ssize(in_fasta[0].sequence)));
    }
  }

  auto total_seqs = std::ssize(in_fasta);
  progress_hook(0, total_seqs);
  
  // Build out "consensus" sequence from all samples, which we'll use to initially impute missing data
  // (helps prevent things like a site that's missing in almost all sequences being imputed as something
  // improbable, and then local topo/resampling moves have a very hard time untangling that error)
  auto consensus_sequence = deduce_consensus_sequence(
      in_fasta | std::views::transform([](const auto& e) { return e.sequence; }));

  // Arbitrary pick the consensus sequence as a reference
  result.ref_sequence = consensus_sequence;

  // Set up tips by converting every sequence into a list of mutations from the reference
  // (essentially, convert the FASTA to a MAPLE file)
  auto seqs_so_far = 0;
  for (auto& fasta_entry : in_fasta) {
    // Ignore sequences that we can't date correctly
    if (auto opt_t_range = extract_date_range_from_sequence_id(fasta_entry.id); not opt_t_range.has_value()) {
      warning_hook(fasta_entry.id, Sequence_warnings::No_valid_date{});
    } else {
      
      auto [t_min, t_max] = opt_t_range.value();
      auto delta = calculate_delta_from_reference(
          fasta_entry.id, fasta_entry.sequence, result.ref_sequence, warning_hook);

      result.tip_descs.push_back({
          .name = fasta_entry.id,
          .t_min = static_cast<float>(t_min),
          .t_max = static_cast<float>(t_max),
          .seq_deltas = std::move(delta.seq_deltas),
          .missations = std::move(delta.missations)});
    }

    ++seqs_so_far;
    progress_hook(seqs_so_far, total_seqs);
  }
  
  return result;
}

// Starting tree for free-topology calibrations; see build_calibration_constrained_initial_tree()
static auto build_free_topology_calibration_start(
    Phylo_tree guide,
    Maple_file maple,
    const Calibration_set& calibrations,
    const Init_time_scale& time_scale,
    std::optional<double> init_rate_per_site_per_year,
    int num_observed_sites)
    -> Phylo_tree {
  try {
    auto rate_per_site_per_day = std::optional<double>{};
    if (init_rate_per_site_per_year.has_value()) {
      rate_per_site_per_day = *init_rate_per_site_per_year / 365.0;
    } else if (time_scale.fallback_lambda.has_value() && num_observed_sites > 0) {
      rate_per_site_per_day = *time_scale.fallback_lambda / num_observed_sites;
    }
    auto summary = Constrained_init_summary{};
    auto tree = build_calibration_constrained_initial_tree(
        std::move(guide), std::move(maple.ref_sequence), std::move(maple.tip_descs), calibrations,
        rate_per_site_per_day, num_observed_sites, &summary);
    if (summary.rebuilt) {
      std::cerr << absl::StreamFormat(
          "- calibration constraints: rebuilt the initial topology with every calibrated sample set as a clade "
          "(%d parsimony mutations, timed at %.4g subst/site/year)\n",
          summary.num_parsimony_mutations, summary.rate_per_site_per_day * 365.0);
    } else {
      std::cerr << "- calibration constraints: the initial tree already has every calibrated sample set as a clade\n";
    }
    std::cerr << absl::StreamFormat(
        "- calibration constraints: %d inner node time(s) moved inside the hard age bounds\n", summary.num_nodes_moved);
    return tree;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    std::exit(EXIT_FAILURE);
  }
}

auto build_rough_initial_tree_from_maple(
    Maple_file&& in_maple,
    Init_method init_method,
    absl::BitGenRef bitgen,
    const std::function<void(int,int)>& progress_hook,
    const std::function<void(int,int,int)>& refined_tree_progress_hook,
    const std::function<void(int,int,int)>& spr_refine_progress_hook,
    const std::function<void(int,int,int,int,int)>& rooting_progress_hook,
    const std::function<void(const Rooting_info&)>& rooting_hook,
    const Init_time_scale& time_scale)
    -> Phylo_tree {

  if (in_maple.tip_descs.empty()) {
    throw std::runtime_error("Input has no sequences (after discarding those with warnings)!");
  }
  if (std::ssize(in_maple.tip_descs) == 1) {
    throw std::runtime_error("Input has only 1 sequence (after discarding those with warnings)!");
  }

  switch (init_method) {
    case Init_method::random:
      return build_random_tree(
          std::move(in_maple.ref_sequence), std::move(in_maple.tip_descs), bitgen, progress_hook);
    case Init_method::old_usher_like:
      return build_usher_like_tree(
          std::move(in_maple.ref_sequence), std::move(in_maple.tip_descs), bitgen, progress_hook);
    case Init_method::mp_plus_timing:
      return build_initial_phylo_tree(
          std::move(in_maple.ref_sequence), std::move(in_maple.tip_descs), bitgen,
          progress_hook, refined_tree_progress_hook, spr_refine_progress_hook,
          rooting_progress_hook, rooting_hook, time_scale);
  }
  CHECK(false) << "Unknown init method";
  return Phylo_tree{};  // unreachable; silences -Werror=return-type
}

// Mean of a Laplace(mu, s) distribution truncated to [a, b].
// Assumes a <= mu <= b (or a/b may be ±inf for unbounded cases).
static auto truncated_laplace_mean(double mu, double s, double a, double b) -> double {
  CHECK_GT(s, 0.0);
  CHECK_LE(a, b);
  CHECK_LE(a, mu);
  CHECK_LE(mu, b);

  auto p = (mu - a) / s;  // may be +inf
  auto q = (b - mu) / s;  // may be +inf

  auto result = 0.0;

  // Common special cases (avoid inf arithmetic)
  if (std::isinf(p) && std::isinf(q)) {
    result = mu;                                               // no bounds
  } else if (std::isinf(p)) {                                 // only upper bound
    auto eq = std::exp(-q);
    result = mu + (s / 2) * (-(q + 1) * eq) / (1 - eq / 2);
  } else if (std::isinf(q)) {                                 // only lower bound
    auto ep = std::exp(-p);
    result = mu + (s / 2) * ((p + 1) * ep) / (1 - ep / 2);
  } else if (p + q < 1e-4) {
    result = (a + b) / 2;                                      // Taylor fallback for tight bounds
  } else {
    auto ep = std::exp(-p);
    auto eq = std::exp(-q);
    result = mu + (s / 2) * ((1 + p) * ep - (1 + q) * eq) / (1 - (ep + eq) / 2);
  }

  CHECK_GE(result, a);
  CHECK_LE(result, b);
  return result;
}

auto process_args(int argc, char** argv) -> Processed_cmd_line {
  
  // Before anything else, record CLI invocation (e.g., for log files)
  delphy_invoked_via_cli = true;
  delphy_cli_args = std::vector<std::string>{argv, argv+argc};
  
  cxxopts::Options options("delphy", "Delphy - Fast and scalable Bayesian phylogenetic tree inference");

  options.add_options("Generic options")
      ("version", "print version string")
      ("h,help", "print usage")
      ("n,dry-run", "Dry run: process inputs, generate BEAST XMLs (if needed) but don't proceed with run")
      ;

  options.add_options("Version-0 options: unstable, temporary syntax for specifying Delphy's behaviour")
      ("v0-threads", "Number of threads to use (default: use all cores)",
       cxxopts::value<int>())
      ("v0-seed", "Initial random number seed (default: random)",
       cxxopts::value<uint32_t>())
      ("v0-paranoid", "Force full tree integrity checks even in release builds (useful for debugging; large performance cost)",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-in-fasta", "input FASTA file", cxxopts::value<std::string>())
      ("v0-in-maple", "input MAPLE file", cxxopts::value<std::string>())
      ("v0-init", "Tree initialization method: random, old-usher-like, mp-plus-timing (default: mp-plus-timing)",
       cxxopts::value<std::string>())
      ("v0-init-heuristic", "[deprecated, use --v0-init old-usher-like]",
       cxxopts::value<bool>())
      ("v0-init-random", "[deprecated, use --v0-init random]",
       cxxopts::value<bool>())
      ("v0-in-tree",
       "Rooted, strictly binary Newick tree to start from.  Tip names must match the sequence IDs exactly.  "
       "The rooted topology is kept as given; only node times and mutation histories are derived.  "
       "Combine with --v0-fix-topology to keep the topology fixed during MCMC.",
       cxxopts::value<std::string>())
      ("v0-in-tree-branch-lengths",
       "[--v0-in-tree] Meaning of the branch lengths in the input tree: ignore (use only the topology; node times "
       "come from mutation counts and an explicit substitution rate), years, days, or subs-per-site "
       "(converted to time with an explicit substitution rate).  Required if the tree has branch lengths.",
       cxxopts::value<std::string>())
      ("v0-fix-topology",
       "Do not change the rooted topology during MCMC.  Node times, mutation histories and all global "
       "parameters are still sampled.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-calibration-file",
       "Age calibrations on the MRCAs of named sample sets, as a prior of the sampled posterior.  Requires "
       "--v0-fix-topology, because an SPR move can change which node is the MRCA of a sample set, unless "
       "--v0-allow-free-topology-calibration is given.  See doc/extended/calibration.md for "
       "the file format.",
       cxxopts::value<std::string>())
      ("v0-allow-free-topology-calibration",
       "[--v0-calibration-file] Sample the topology together with the calibrations.  Every calibrated sample set "
       "other than 'taxa all' is constrained to stay monophyletic (trees where it is not have posterior density "
       "zero) unless its calibration says 'monophyly not_required', its MRCA is re-identified after every topology "
       "proposal, and its age prior enters every Metropolis-Hastings ratio.  Without --v0-in-tree, the starting "
       "tree is built from the alignment to satisfy every monophyly constraint and hard age bound.  Needs "
       "--v0-threads 1; not combinable with --v0-fix-topology.  Combinable with --v0-exact-coalescent-correction.  "
       "See doc/extended/calibration_free_topology.md and exact_free_topology.md.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-steps", "Total number of steps to run (default: ~100,000 per tip)",
       cxxopts::value<int64_t>())
      ("v0-out-log-file", "Filename for BEAST-like log file",
       cxxopts::value<std::string>())
      ("v0-log-every", "Steps between entries in screen & file logs (default: steps / 100)",
       cxxopts::value<int64_t>())
      ("v0-out-trees-file", "Filename for BEAST-like trees file",
       cxxopts::value<std::string>())
      ("v0-tree-every", "Steps between tree snapshots in tree log (default: steps / 100)",
       cxxopts::value<int64_t>())
      ("v0-out-trees-digits",
       "Significant digits of the branch lengths and mutation times written to the trees file (default 6, "
       "which is what every earlier version wrote).  A branch length of 140,000 years is quantized to 0.1 "
       "years at 6 digits, and the error accumulates along a root-to-tip path, so deep-time analyses that "
       "summarize node ages from the trees file should raise this (9 is enough for any human time scale).",
       cxxopts::value<int>())
      ("v0-out-delphy-file", "Filename for .dphy run file",
       cxxopts::value<std::string>())
      ("v0-out-delphy-metadata-file", "Filename for JSON metadata blob to append to .dphy run file",
       cxxopts::value<std::string>())
      ("v0-delphy-snapshot-every", "Steps between run snapshots in .dphy run file (default: steps / 100)",
       cxxopts::value<int64_t>())
      ("v0-site-rate-heterogeneity", "Enable site rate heterogeneity",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-fix-mutation-rate", "Fix mutation rate",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-init-mutation-rate",
       "Initial (or fixed) value of mutation rate, in subst / site / year.  "
       "If not specified and a proper Gamma prior is set, defaults to the prior mean; "
       "otherwise defaults to 1e-3.",
       cxxopts::value<double>()->default_value("1e-3"))
      ("v0-mu-prior-alpha",
       "Shape (alpha) parameter of the Gamma prior on the mutation rate mu: "
       "pi(mu) ~ mu^{alpha-1} exp[-beta mu].  Default alpha=1, beta=0 gives a uniform prior.",
       cxxopts::value<double>()->default_value("1.0"))
      ("v0-mu-prior-beta",
       "Rate (beta) parameter of the Gamma prior on the mutation rate mu: "
       "pi(mu) ~ mu^{alpha-1} exp[-beta mu].  Default alpha=1, beta=0 gives a uniform prior.  "
       "Units: years.",
       cxxopts::value<double>()->default_value("0.0"))
      ("v0-mu-prior-mean",
       "Mean of the Gamma prior on the mutation rate mu, in subst / site / year.  "
       "Must be specified together with --v0-mu-prior-stddev.  "
       "Mutually exclusive with --v0-mu-prior-alpha / --v0-mu-prior-beta.",
       cxxopts::value<double>())
      ("v0-mu-prior-stddev",
       "Standard deviation of the Gamma prior on the mutation rate mu, in subst / site / year.  "
       "Must be specified together with --v0-mu-prior-mean.  "
       "Mutually exclusive with --v0-mu-prior-alpha / --v0-mu-prior-beta.",
       cxxopts::value<double>())
      ("v0-out-beast-xml", "Filename for XML input file suitable for BEAST X or BEAST2 (see --v0-out-beast-version)",
       cxxopts::value<std::string>())
      ("v0-out-beast-version", "BEAST version to target for XML and output files.  Currently allowed values are '2.6.2', '2.7.7', and 'X-10.5.0'.",
       cxxopts::value<std::string>()->default_value("X-10.5.0"))
      ("v0-mpox-hack", "Enable mpox hack (very crude treatment of APOBEC3 mechanism)",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-target-coal-prior-cells", "Target number of cells to use in parallelized coalescent prior (coalescent prior resolution is adjusted if actual number is more than 33% away from target); higher is more accurate but more expensive",
       cxxopts::value<int>()->default_value("400"))
      ("v0-coal-prior-init-cells",
       "Number of cells to divide the initial tree span into when the coalescent prior's grid is "
       "first built.  The default of 0 keeps the historical behaviour of starting at a resolution of "
       "one day, which allocates one cell per day of tree depth before the first retarget and so "
       "costs tens of millions of cells on a tree tens of thousands of years deep.  Any positive "
       "value sizes that first grid from the tree instead.  In --v0-human-mode this defaults to "
       "--v0-target-coal-prior-cells.",
       cxxopts::value<int>()->default_value("0"))
      ("v0-coal-prior-grid-tolerance",
       "Maximum discretization excess, in nats, that the coalescent prior's grid may introduce.  "
       "The grid replaces the number of lineages k(t) by its average over each cell, which raises "
       "the coalescent log prior above the exact Kingman density of the same genealogy by "
       "sum_cells t_step Var_cell(k) / (2 Nbar_cell).  That excess is a function of the genealogy, "
       "so it does not cancel in Metropolis-Hastings ratios.  When this option is positive, the "
       "grid is refined -- from --v0-target-coal-prior-cells up to --v0-coal-prior-max-cells, "
       "doubling the cell count at each step -- until the measured excess falls at or below the "
       "tolerance.  The default of 0 disables the tolerance and keeps the plain span / cells rule.",
       cxxopts::value<double>()->default_value("0.0"))
      ("v0-coal-prior-trace-retargets",
       "Diagnostic.  Write one line to stderr whenever the coalescent prior's grid changes its cell width, "
       "with the change that imposes on the coalescent log prior of the current state.  That change is "
       "not part of any Metropolis-Hastings ratio.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-skip-fixed-tip-moves",
       "Leave the tip-date move out of the local move mix in every part of the tree whose tips all have a "
       "fixed sampling date (identical lower and upper date bound), where that move cannot change the "
       "state.  The other local moves are then chosen with proportionally larger probability.  The "
       "sampled distribution is unchanged.  Parts with any tip of uncertain date keep the full mix.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-local-moves-per-node-per-global-move",
       "Number of local moves per tree node between two global moves (clock rate, substitution "
       "model and population parameters).  The default of 50 is the historical value.  A smaller "
       "value updates the global parameters more often; the sampled distribution is unchanged.",
       cxxopts::value<int>()->default_value("50"))
      ("v0-pop-moves-per-global-move",
       "[pop-model == exponential] Number of population-size and growth-rate move pairs made in every "
       "global step.  The default of 50 is the historical value.  Each move recomputes the coalescent "
       "prior, so a smaller value makes global steps cheaper; the sampled distribution is unchanged.",
       cxxopts::value<int>()->default_value("50"))
      ("v0-root-displace-scale-factor",
       "Factor applied to the width of the Gaussian proposal for the time of the root, whose historical "
       "width is half the reciprocal of the mutation rate at the root (capped by the tree span).  The "
       "proposal stays symmetric, so the sampled distribution is unchanged.  Default 1.",
       cxxopts::value<double>()->default_value("1.0"))
      ("v0-coal-prior-fixed-t-step",
       "Pin the coalescent prior's cell width, in years, for the whole run.  The grid is then never "
       "retargeted from the sampled state, so the sampled posterior has one fixed target density "
       "(the cell prior on that grid) instead of a target that depends on the current tree span.  "
       "Incompatible with --v0-coal-prior-grid-tolerance.  The default of 0 keeps the retargeting rules.",
       cxxopts::value<double>()->default_value("0.0"))
      ("v0-exact-coalescent-correction",
       "[--v0-human-mode only, experimental] Target the exact Kingman coalescent prior instead of the cell prior "
       "on the coalescent grid: every batch of local moves is accepted with probability "
       "min(1, exp(delta' - delta)), delta = log P_exact - log P_cells of the genealogy, and rolled back "
       "otherwise, and the population moves use the exact density.  Requires "
       "--v0-coal-prior-fixed-t-step, the exponential, Skygrid or constant population model and samples with exact "
       "dates; a free topology (with or without --v0-allow-free-topology-calibration) needs --v0-threads 1; "
       "anything else is refused.  With the Skygrid model the "
       "smoothness, zero-mode and HMC moves all target the exact density, which is evaluated on the genealogy's own "
       "event intervals rather than on the grid.  See doc/extended/exact_coalescent.md and "
       "exact_skygrid.md and exact_free_topology.md.  Default: off.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-threshold-partition",
       "[--v0-human-mode only, experimental] Run the local moves on --v0-threads parts in parallel, cut by the "
       "threshold partition: before every batch, independently of the state, either a single part (probability "
       "p) or an integer m uniform on [a, b] x tips / threads is drawn, and the cut points are the clades of "
       "ceil(m / r) to m tips whose parent has more than m tips.  The root part is kept from changing the "
       "partition, so the batch kernel sees the same partition at both ends; this is what makes "
       "--v0-exact-coalescent-correction valid with a free topology on several threads.  Not combinable with age "
       "calibrations on a free topology.  See doc/extended/threshold_partition.md.  Default: off.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-threshold-partition-params",
       "[with --v0-threshold-partition] a,b,r,p as above.  Default: 0.5,1.0,4,0.02",
       cxxopts::value<std::string>())
      ("v0-global-scale-move",
       "[--v0-human-mode only, experimental] Add the joint time-scale move to every global step: all internal node ages and "
       "mutation ages, measured back from the common sampling date, are multiplied by s = exp(u), u uniform in "
       "[-window, +window], while the mutation rate is divided by s, the population size at the sampling date multiplied by s and "
       "the growth rate divided by s; the move is accepted with the full Metropolis-Hastings ratio (target likelihood, exact Kingman "
       "prior, calibration priors, priors on the rate and the population parameters, Jacobian).  It targets the slow rate-time ridge; "
       "the sampled posterior is unchanged.  Requires --v0-exact-coalescent-correction, --v0-fix-topology, the exponential population "
       "model with its reference time at the sampling date, and samples sharing one sampling date; anything else is refused.  "
       "See doc/extended/global_scale_move.md.  Default: off.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-global-scale-window",
       "[--v0-global-scale-move] Half width of the uniform proposal of log s.  Default 0.2.",
       cxxopts::value<double>()->default_value("0.2"))
      ("v0-global-scale-moves-per-global-move",
       "[--v0-global-scale-move] Number of joint time-scale proposals in every global step.  Default 1.",
       cxxopts::value<int>()->default_value("1"))
      ("v0-history-gibbs",
       "[--v0-human-mode only, experimental] Add the single-site mutational-history Gibbs updates to every batch of "
       "local moves.  Delphy's existing history proposal redraws every site of a path at once from a Jukes-Cantor "
       "process and corrects the whole draw with one Metropolis-Hastings decision, so its acceptance falls with the "
       "total number of substitution events and the explicit history freezes at high divergence.  These updates work "
       "one site at a time: the full conditional of one site's history on one branch, given the states at the two ends, "
       "is a continuous-time Markov bridge under that site's own rate matrix, which is drawn exactly, and the same holds "
       "for the state of one site at one node together with the histories of the incident branches.  Both are Gibbs "
       "steps, so they are always accepted and nothing about them degrades with sequence length or divergence.  The "
       "sampled posterior is unchanged.  See doc/extended/history_gibbs.md.  Default: off.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-history-gibbs-weight",
       "[--v0-history-gibbs] Weight of the history Gibbs updates in the local move mix, split evenly between the "
       "branch-history update and the node-state update.  For scale, the existing weights are 7.5 for each node-time "
       "move, 15 for the branch reform and 1 each for the subtree slide and the SPR.  Default 15.",
       cxxopts::value<double>()->default_value("15.0"))
      ("v0-history-gibbs-sites",
       "[--v0-history-gibbs] Number of sites one history Gibbs update visits.  The sites are drawn uniformly and "
       "independently of the current history, so any value leaves the update a valid random-scan Gibbs step.  Default 1.",
       cxxopts::value<int>()->default_value("1"))
      ("v0-site-history-refreshes",
       "[--v0-human-mode only, experimental] Number of sites whose whole mutational history over the whole tree is "
       "redrawn in every global step: Felsenstein pruning for the one site, then the root state from the root prior "
       "times the root partial likelihood, then every internal state from the root down, then an exact conditional "
       "bridge on every branch.  This is the update that mixes the root state.  It costs a pass over the tree per site "
       "and rebuilds the tree parts, so it belongs at a low rate.  Default 0, which is off.",
       cxxopts::value<int>()->default_value("0"))
      ("v0-marginal-node-time-move",
       "[--v0-human-mode only, experimental] Add a node-time move that integrates the mutational history of the "
       "branches incident on a node out, proposes the node time free of the mutation times that bound the existing "
       "node-time move, and draws the history back from its exact conditional.  The existing move takes its bounds "
       "from the nearest mutation above and below the node, so the node crosses at most one mutation gap per step and "
       "the step shrinks as the mutation density grows; once the history itself can move, that becomes the binding "
       "constraint on the tree's time scale.  The proposal density of the new history cancels against the history's "
       "own factor in the target, so the acceptance ratio involves only the local marginal likelihood, the coalescent "
       "and calibration priors and the Jacobian, and the sampled posterior is unchanged.  Needs --v0-history-gibbs to "
       "be of any use, and costs one pass over the sites per proposal.  See doc/extended/history_gibbs.md.  "
       "Default: off.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-marginal-node-time-weight",
       "[--v0-marginal-node-time-move] Weight of the move in the local mix.  Default 7.5, the same as each existing "
       "node-time move.",
       cxxopts::value<double>()->default_value("7.5"))
      ("v0-marginal-node-time-window",
       "[--v0-marginal-node-time-move] Half width of the uniform proposal of the log of the root's depth below its "
       "oldest child.  Default 0.5.",
       cxxopts::value<double>()->default_value("0.5"))
      ("v0-coal-prior-max-cells",
       "Largest number of coalescent-prior cells --v0-coal-prior-grid-tolerance may refine to.",
       cxxopts::value<int>()->default_value("4000000"))

      // Popultation model
      ("v0-pop-model", "Type of population model to use - exponential or skygrid",
       cxxopts::value<std::string>()->default_value("exponential"))
      
      // The following only make sense for an exponential population model
      ("v0-fix-final-pop-size", "[pop-model == exponential] Fix effective population size at time of last tip",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-init-final-pop-size", "[pop-model == exponential] Initial (or fixed) value of effective population size at time of last tip, in years (the effective population size includes a factor of the generation time).  "
       "Default: prior mean if an Inverse-Gamma prior with a defined mean is specified, otherwise 3.0 years.",
       cxxopts::value<double>()->default_value("3.0"))
      ("v0-fix-pop-growth-rate", "[pop-model == exponential] Fix effective population size growth rate",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-init-pop-growth-rate",
       "[pop-model == exponential] Initial (or fixed) value of effective population size growth rate, "
       "in e-foldings / year.  If not specified and a Laplace prior with non-default parameters is set, "
       "defaults to the truncated prior mean; otherwise defaults to 0.",
       cxxopts::value<double>()->default_value("0.0"))
      ("v0-pop-inv-n0-prior-alpha",
       "[pop-model == exponential] Alpha parameter of the Inverse-Gamma prior on the effective "
       "population size n0: pi(n0) ~ n0^{-(alpha+1)} exp[-beta/n0].  "
       "Default alpha=0, beta=0 gives the Jeffreys 1/x prior.",
       cxxopts::value<double>()->default_value("0.0"))
      ("v0-pop-inv-n0-prior-beta",
       "[pop-model == exponential] Beta parameter of the Inverse-Gamma prior on the effective "
       "population size n0: pi(n0) ~ n0^{-(alpha+1)} exp[-beta/n0].  "
       "Default alpha=0, beta=0 gives the Jeffreys 1/x prior.  Units: years.",
       cxxopts::value<double>()->default_value("0.0"))
      ("v0-pop-n0-prior-mean",
       "[pop-model == exponential] Mean of the Inverse-Gamma prior on effective population size "
       "n0, in years (mean = beta / (alpha - 1)).  "
       "Must be specified together with --v0-pop-n0-prior-stddev.  "
       "Mutually exclusive with --v0-pop-inv-n0-prior-alpha / --v0-pop-inv-n0-prior-beta.",
       cxxopts::value<double>())
      ("v0-pop-g-prior-mu",
       "[pop-model == exponential] Location (mu) of the Laplace prior on the growth rate g: "
       "pi(g) ~ exp(-|g - mu| / scale).  Units: e-foldings / year.",
       cxxopts::value<double>()->default_value("0.001"))
      ("v0-pop-g-prior-scale",
       "[pop-model == exponential] Scale of the Laplace prior on the growth rate g: "
       "pi(g) ~ exp(-|g - mu| / scale).  Units: e-foldings / year.",
       cxxopts::value<double>()->default_value("30.701135"))
      ("v0-pop-g-move-window",
       "[pop-model == exponential] Half-width, in e-foldings / year, of the uniform random-walk proposal "
       "for the growth rate g.  The default of 0 keeps the historical window of 1 per year, except in "
       "--v0-human-mode, where it becomes the smallest of 1 per year, the scale of the Laplace prior "
       "and the width of the allowed range of g, so that the proposal is on the scale of the prior "
       "(a window far wider than the allowed range is almost always rejected).  The window does not "
       "change the target distribution, only how fast the chain explores it.",
       cxxopts::value<double>()->default_value("0.0"))
      ("v0-pop-growth-rate-min",
       "[pop-model == exponential] Lower bound on the growth rate g, in e-foldings / year.  "
       "E.g., use 0 to constrain the population to be non-declining.  "
       "When g_min = mu = 0, the prior reduces to Exponential(rate = 1/scale).",
       cxxopts::value<double>())
      ("v0-pop-growth-rate-max",
       "[pop-model == exponential] Upper bound on the growth rate g, in e-foldings / year.",
       cxxopts::value<double>())
      ("v0-pop-g-prior-exponential-with-mean",
       "[pop-model == exponential] Shorthand for an Exponential prior on the growth rate g "
       "with the specified mean, in e-foldings / year.  "
       "If positive, enforces g >= 0 (growing population); if negative, enforces g <= 0 "
       "(declining population).  Equivalent to setting mu = 0, scale = |mean|, and the "
       "appropriate bound.  "
       "Mutually exclusive with --v0-pop-g-prior-mu, --v0-pop-g-prior-scale, "
       "--v0-pop-growth-rate-min, and --v0-pop-growth-rate-max.",
       cxxopts::value<double>())
      ("v0-pop-n0-prior-stddev",
       "[pop-model == exponential] Standard deviation of the Inverse-Gamma prior on effective "
       "population size n0, in years (var = beta^2 / ((alpha-1)^2 (alpha-2))).  "
       "Must be specified together with --v0-pop-n0-prior-mean.  "
       "Mutually exclusive with --v0-pop-inv-n0-prior-alpha / --v0-pop-inv-n0-prior-beta.",
       cxxopts::value<double>())
      ("v0-pop-min-pop",
       "[pop-model == exponential] Minimum effective population size for exponential model, in years "
       "(0 = disabled, default = 1 day = 1/365 years).  Effective population sizes below this minimum "
       "are clamped to this value.  This prevents numerical issues in deep trees.",
       cxxopts::value<double>()->default_value(absl::StrFormat("%.5g", 1.0 / 365.0)))

      // The following only make sense for a Skygrid population model
      ("v0-skygrid-type",
       "[pop-model == skygrid] Type of Skygrid model: \n"
       "* 'staircase' (default) = log N(t) is piecewise flat (standard Gill et al 2012)\n"
       "* 'log-linear' = log N(t) is piecewise linear and continuous\n",
       cxxopts::value<std::string>()->default_value("staircase"))
      ("v0-skygrid-num-parameters",
       "[pop-model == skygrid] Number of parameters used to parametrize N(t) (what BEAUTi calls 'Number of parameters' == one more than what Gill et al 2012 call `M`)",
       cxxopts::value<int>()->default_value("50"))
      ("v0-skygrid-cutoff",
       "[pop-model == skygrid] Time before time of last tip for final transition, in years (what BEAUTi calls 'Time of last transition point' == what Gill et al 2012 calls `x_M`).  Mutually exclusive with --v0-skygrid-first-knot-date / --v0-skygrid-last-knot-date.",
       cxxopts::value<double>()->default_value("0.0"))
      ("v0-skygrid-first-knot-date",
       "[pop-model == skygrid] Date (YYYY-MM-DD) of the first (oldest) Skygrid knot x_0.  "
       "Example: --v0-skygrid-first-knot-date 2013-12-01.  "
       "Must be specified together with --v0-skygrid-last-knot-date.  "
       "Mutually exclusive with --v0-skygrid-cutoff.",
       cxxopts::value<std::string>())
      ("v0-skygrid-last-knot-date",
       "[pop-model == skygrid] Date (YYYY-MM-DD) of the last (most recent) Skygrid knot x_M.  "
       "Example: --v0-skygrid-last-knot-date 2014-10-01.  "
       "Must be specified together with --v0-skygrid-first-knot-date.  "
       "Mutually exclusive with --v0-skygrid-cutoff.",
       cxxopts::value<std::string>())
      ("v0-skygrid-infer-prior-smoothness",
       "[pop-model == skygrid] Whether to fix the log-population curve's prior smoothness (Delphy default; see --v0-skygrid-prior-double-half-time) or infer it (BEAST default; see --v0-skygrid--tau-prior-alpha and --v0-skygrid-tau-prior-beta)",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-skygrid-prior-double-half-time",
       "[pop-model == skygrid] Typical timescale (in years) over which the a priori population curve fluctuates by a factor of 2 (default = 30 days = 0.0822 years).  Applies only if --v0-skygrid-infer-prior-smoothness is `false`.  More precisely, the log-population curve prior is a 1D random walk with diffusion constant D, such that after after a time T, the log-population is offset by a Gaussian random variable with mean zero and variance 2 D T.  Setting sqrt(2 D T) = log(2) at the 'double-half time' T yields D = log^2(2) / (2 T); at this diffusion rate, after one month, there's a 68% chance that the population has changed by a factor of 2 or less.  In Gill et al's 2012 formulation, this fixes the smoothness precision tau = 1 /(D dt), where dt = cutoff / (num_parameters - 1)",
       cxxopts::value<double>()->default_value("0.0822"))
      ("v0-skygrid-tau",
       "[pop-model == skygrid] Fixed value of Skygrid precision parameter `tau`; mutually exclusive with --v0-skygrid-prior-double-half-time",
       cxxopts::value<double>())
      ("v0-skygrid-tau-prior-alpha",
       "[pop-model == skygrid] When inferring the log-population curve's prior smoothness `tau`, the hyperprior on tau is ~ tau^{alpha - 1} exp[-beta tau]",
       cxxopts::value<double>()->default_value("0.001"))
      ("v0-skygrid-tau-prior-beta",
       "[pop-model == skygrid] When inferring the log-population curve's prior smoothness `tau`, the hyperprior on tau is ~ tau^{alpha - 1} exp[-beta tau]",
       cxxopts::value<double>()->default_value("0.001"))
      ("v0-skygrid-disable-low-pop-barrier",
       "[pop-model == skygrid] Disable quadratic penalty against effective population sizes dropping below a certain limit (--v0-skygrid-low-pop-barrier-loc, default=1 day to match resolution of tip dates).  Such low population sizes sometimes trigger numerical difficulties that cause some MCMC moves to grind to a halt.  Moreover, they cannot reliably be estimated from the data.  We recommend disabling this only if attempting to make an exact comparison to a run from another tool, like BEAST.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-skygrid-low-pop-barrier-loc",
       "[pop-model == skygrid] Minimum value of N(t), in years, below which we penalize smaller populations (default = 1 day = 1/365 years, to match resolution of tip dates)",
       cxxopts::value<double>()->default_value(absl::StrFormat("%.5g", 1.0 / 365.0)))
      ("v0-skygrid-low-pop-barrier-scale",
       "[pop-model == skygrid] Fraction by which N(t) must drop below minimum for the value of the low-pop barrier to reach a value of 1 nat (i.e., (log(N/N_min) / log(1-scale))^2 = 1 at N = N_min * (1-scale)).",
       cxxopts::value<double>()->default_value("0.30"))
      ("v0-skygrid-inv-nbar-prior-alpha",
       "[pop-model == skygrid] Alpha parameter of the Inverse-Gamma prior on N_bar, the "
       "geometric-mean effective population size across Skygrid knots "
       "(N_bar = exp(mean of log-population values gamma_k)).  "
       "pi(N_bar) ~ N_bar^{-(alpha+1)} exp[-beta/N_bar].  "
       "Default alpha=0, beta=0 gives a uniform prior on log(N_bar) (= Jeffreys 1/N_bar prior).",
       cxxopts::value<double>()->default_value("0.0"))
      ("v0-skygrid-inv-nbar-prior-beta",
       "[pop-model == skygrid] Beta parameter of the Inverse-Gamma prior on N_bar, the "
       "geometric-mean effective population size across Skygrid knots "
       "(N_bar = exp(mean of log-population values gamma_k)).  "
       "pi(N_bar) ~ N_bar^{-(alpha+1)} exp[-beta/N_bar].  "
       "Default alpha=0, beta=0 gives a uniform prior on log(N_bar) (= Jeffreys 1/N_bar prior).  "
       "Units: years.",
       cxxopts::value<double>()->default_value("0.0"))
      ("v0-skygrid-nbar-prior-mean",
       "[pop-model == skygrid] Mean of the Inverse-Gamma prior on N_bar, the geometric-mean "
       "effective population size across Skygrid knots, in years "
       "(mean = beta / (alpha - 1)).  "
       "Must be specified together with --v0-skygrid-nbar-prior-stddev.  "
       "Mutually exclusive with --v0-skygrid-inv-nbar-prior-alpha / "
       "--v0-skygrid-inv-nbar-prior-beta.",
       cxxopts::value<double>())
      ("v0-skygrid-nbar-prior-stddev",
       "[pop-model == skygrid] Standard deviation of the Inverse-Gamma prior on N_bar, the "
       "geometric-mean effective population size across Skygrid knots, in years "
       "(var = beta^2 / ((alpha-1)^2 (alpha-2))).  "
       "Must be specified together with --v0-skygrid-nbar-prior-mean.  "
       "Mutually exclusive with --v0-skygrid-inv-nbar-prior-alpha / "
       "--v0-skygrid-inv-nbar-prior-beta.",
       cxxopts::value<double>())
      ;

  options.add_options("Human uniparental (mtDNA) input: opt-in, replaces the date-in-sequence-ID convention")
      ("v0-human-mode",
       "Read human uniparental data: sampling dates come from --v0-human-metadata / "
       "--v0-human-uniform-sample-date instead of the sequence IDs, and the input is audited and "
       "validated strictly (nothing is dropped silently).  Requires --v0-in-fasta, --v0-in-maple or "
       "--v0-human-in-variants.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-human-reference-standard",
       "Reference coordinate system of the alignment: rCRS, RSRS (both 16569 sites) or custom "
       "(then --v0-human-expected-length is required).",
       cxxopts::value<std::string>())
      ("v0-human-expected-length",
       "Expected number of sites of the full-length reference (required for custom).",
       cxxopts::value<int>())
      ("v0-human-reference-fasta",
       "[FASTA input] FASTA file with the single reference sequence.",
       cxxopts::value<std::string>())
      ("v0-human-reference-id",
       "[FASTA input] ID of the record in the input FASTA that is the reference (it is not a sample).",
       cxxopts::value<std::string>())
      ("v0-human-metadata",
       "Tab-separated sample metadata (columns: sample_id, date_convention, date_from, date_to, "
       "bp_reference_year); matched to sequences by exact sample ID.",
       cxxopts::value<std::string>())
      ("v0-human-uniform-sample-date",
       "ISO date (YYYY-MM-DD, YYYY-MM or YYYY) given to every sample without its own date.  "
       "If all samples get the same date the data carry no time-scale information.",
       cxxopts::value<std::string>())
      ("v0-human-in-variants",
       "Variant-list input: tab-separated table with columns sample_id, site (1-based, reference "
       "coordinates) and base (A, C, G or T).  The sample set is --v0-human-metadata, the reference comes "
       "from --v0-human-reference-fasta, and the sites that were examined must be declared with "
       "--v0-human-callable-file or --v0-human-sample-callable-file: a site that is neither a variant nor "
       "declared callable is missing, not a reference state.",
       cxxopts::value<std::string>())
      ("v0-human-mask-file",
       "Sites to treat as missing in every sample, one per line: 310 or 303-315 (1-based, inclusive).",
       cxxopts::value<std::string>())
      ("v0-human-callable-file",
       "Sites that were examined in every sample, one per line: 310 or 303-315 (1-based, inclusive).  "
       "Every site outside the list is missing in every sample.  A call that differs from the reference at "
       "such a site is an error (use --v0-human-mask-file to exclude sites deliberately).",
       cxxopts::value<std::string>())
      ("v0-human-sample-callable-file",
       "Sites that were examined in each sample: one line per sample, '<sample_id><TAB><sites>', where "
       "<sites> is a comma-separated list of 1-based inclusive sites or ranges, or 'none'.  Every sample "
       "must appear exactly once; sites outside a sample's list are missing in that sample.",
       cxxopts::value<std::string>())
      ("v0-human-snp-only-constant-sites",
       "[variable-sites-only input] Counts of the constant sites that the variability filter removed, by "
       "the state every sample shared there: 'A=4000,C=3000,G=3100,T=2400'.  They are restored as a block "
       "of constant sites appended after the input reference, which makes the likelihood equal to the "
       "full-length one; the block is callable in every sample and carries no reference coordinates.  "
       "Without this declaration (or full-length input) variable-sites-only data are rejected.",
       cxxopts::value<std::string>())
      ("v0-human-max-missing-fraction",
       "Reject the input if any sample has a larger fraction of missing sites.",
       cxxopts::value<double>()->default_value("0.10"))
      ("v0-human-audit-report",
       "Filename for the JSON input audit report.",
       cxxopts::value<std::string>())
      ;

  options.add_options("Substitution model: opt-in, any reversible nucleotide model")
      ("v0-subst-model",
       "Nucleotide substitution model (see doc/extended/substitution_models.md).  Either a "
       "name (JC69, F81, K80, HKY85, TN93, TNe, K81, K81u, TPM2, TPM2u, TPM3, TPM3u, TIM, TIMe, TIM2, "
       "TIM2e, TIM3, TIM3e, TVM, TVMe, SYM, GTR and their aliases) or a six-digit exchangeability code "
       "in the order AC, AG, AT, CG, CT, GT, where equal digits constrain the corresponding "
       "exchangeabilities to be equal and the class of G-T carries the rate 1.  All 203 codes are "
       "supported.  A name or code may be followed by '+FQ' for equal frequencies or '+FO' for "
       "frequencies sampled in the posterior.  Default: HKY85, which is what Delphy has always run.",
       cxxopts::value<std::string>())
      ("v0-subst-initial-exchangeabilities",
       "Comma-separated initial relative exchangeability of every class of the model code, in class "
       "order, with the class of G-T at 1 (default: all 1).",
       cxxopts::value<std::string>())
      ("v0-subst-frequencies",
       "Comma-separated stationary frequencies of A, C, G and T.  With '+FO' they are the starting "
       "point of the sampled frequencies; otherwise they are held fixed.  Default: the frequencies of "
       "the reference sequence for a model with estimated frequencies, and 0.25 each for '+FQ'.",
       cxxopts::value<std::string>())
      ("v0-subst-fix-exchangeabilities",
       "Keep the relative exchangeabilities at their initial values instead of sampling them.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-subst-fix-frequencies",
       "Keep the stationary frequencies at their initial values instead of sampling them, which turns "
       "a model with estimated frequencies into the same model with fixed frequencies.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-subst-exchangeability-prior",
       "Three comma-separated numbers 'LOG_MEAN_TRANSITION,LOG_MEAN_OTHER,SD' of the log-normal prior "
       "on every free relative exchangeability: the mean of its log for a class of transitions only "
       "(A-G, C-T), the mean of its log for every other class, and the standard deviation of the log.  "
       "The default 1,0,1.25 is the prior Delphy has always used for HKY85's kappa.",
       cxxopts::value<std::string>())
      ;

  options.add_options("Site partitions: opt-in, named site sets with their own relative substitution rate")
      ("v0-site-partition-file",
       "Site-partition configuration (see doc/extended/site_partitions.md).  Named, "
       "non-overlapping sets of sites in the coordinates of the input reference, which share the timed "
       "tree, the HKY85 parameters and the global clock rate but have their own relative rate.  The "
       "relative rates are normalized so that their mean, weighted by the number of observed sites of "
       "each partition, is one.  Without this option a run has one partition covering every site.",
       cxxopts::value<std::string>())
      ("v0-site-partition-initial-rates",
       "[--v0-site-partition-file only] Comma-separated initial relative rates, one per partition in "
       "configuration order (default: all 1).  They are renormalized to the weighted mean of one.",
       cxxopts::value<std::string>())
      ("v0-site-partition-fix-rates",
       "[--v0-site-partition-file only] Keep the relative rates at their initial values instead of "
       "sampling them.",
       cxxopts::value<bool>()->default_value("false"))
      ("v0-site-partition-rate-prior-concentration",
       "[--v0-site-partition-file only] Concentration of the symmetric Dirichlet prior on the "
       "length-weighted rate shares; 1 (the default) is the uniform prior on the simplex.",
       cxxopts::value<double>()->default_value("1.0"))
      ;

  try {
    auto opts = options.parse(argc, argv);

    if (opts.count("version") || opts.count("help") || argc == 1) {
      std::cout << absl::StreamFormat("Delphy Version %s (build %d, commit %s)",
                                      k_delphy_version_string,
                                      k_delphy_build_number,
                                      k_delphy_commit_string) << "\n";
      std::cout << absl::StreamFormat("Delphy-Extended %s (extends upstream Delphy at commit %s)",
                                      k_delphy_extended_version_string,
                                      k_delphy_extended_upstream_commit) << "\n";
      
      if (opts.count("help") || argc == 1) {
        std::cout << options.help() << "\n";
      }
      
      std::exit(EXIT_SUCCESS);
    }

    // Validate options
    auto threads = int{};
    if (opts.count("v0-threads")) {
      threads = opts["v0-threads"].as<int>();
    } else {
      threads = std::max(1u, std::thread::hardware_concurrency());
    }
    if (threads < 1) {
      std::cerr << "ERROR: threads must be positive, got " << threads << "\n";
      std::exit(EXIT_FAILURE);
    }
    auto thread_pool = std::make_unique<ctpl::thread_pool>(threads);
    
    auto seed = uint32_t{};
    if (opts.count("v0-seed")) {
      seed = opts["v0-seed"].as<uint32_t>();
    } else {
      seed = std::random_device{}();
    }
    auto prng = std::mt19937{seed};
    std::cerr << "# Seed: " << seed << "\n";
    
    auto num_init_options =
        (opts.count("v0-init") > 0 ? 1 : 0) +
        (opts.count("v0-init-heuristic") > 0 ? 1 : 0) +
        (opts.count("v0-init-random") > 0 ? 1 : 0);
    if (num_init_options > 1) {
      std::cerr << "ERROR: The options --v0-init, --v0-init-heuristic, and --v0-init-random are mutually exclusive.  Pick one.\n";
      std::exit(EXIT_FAILURE);
    }
    auto init_method = Init_method::mp_plus_timing;
    if (opts.count("v0-init") > 0) {
      auto init_str = opts["v0-init"].as<std::string>();
      if (init_str == "random") {
        init_method = Init_method::random;
      } else if (init_str == "old-usher-like") {
        init_method = Init_method::old_usher_like;
      } else if (init_str == "mp-plus-timing") {
        init_method = Init_method::mp_plus_timing;
      } else {
        std::cerr << "ERROR: Unknown --v0-init value: " << init_str
                  << " (expected: random, old-usher-like, mp-plus-timing)\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (opts.count("v0-init-random") > 0) {
      init_method = Init_method::random;
    } else if (opts.count("v0-init-heuristic") > 0) {
      init_method = Init_method::old_usher_like;
    }

    auto tree = Phylo_tree{};
    if (opts.count("v0-in-fasta") > 0 && opts.count("v0-in-maple") > 0) {
      std::cerr << "ERROR: The options --v0-in-fasta and --v0-in-maple are mutually exclusive.  Pick one.\n";
      std::exit(EXIT_FAILURE);
    }

    // Site partitions.  The configuration is read before the input, because it can declare the constant
    // sites that a variability filter removed from each partition (see observation.md section
    // 5.2); it is resolved against the tree once the run exists.
    auto site_partition_config = Site_partition_config{};
    auto have_site_partitions = opts.count("v0-site-partition-file") > 0;
    if (have_site_partitions) {
      auto filename = opts["v0-site-partition-file"].as<std::string>();
      auto is = std::ifstream{filename};
      if (not is) {
        std::cerr << "ERROR: Could not read site-partition file " << filename << "\n";
        std::exit(EXIT_FAILURE);
      }
      try {
        site_partition_config = read_site_partition_config(is);
      } catch (const Site_partition_error& e) {
        std::cerr << "ERROR: " << e.what();
        std::exit(EXIT_FAILURE);
      }
    } else {
      for (const auto* name : {"v0-site-partition-initial-rates", "v0-site-partition-fix-rates",
                               "v0-site-partition-rate-prior-concentration"}) {
        if (opts.count(name) > 0) {
          std::cerr << "ERROR: --" << name << " is only valid together with --v0-site-partition-file\n";
          std::exit(EXIT_FAILURE);
        }
      }
    }


    auto maple_file = Maple_file{};  // Filled in directly with a MAPLE file, or indirectly with a FASTA file

    auto human_mode = opts["v0-human-mode"].as<bool>();
    auto human_no_time_signal = false;
    auto human_observed_sites = 0;  // sites that are observed in at least one sample
    if (not human_mode) {
      for (const auto* name : {"v0-human-reference-standard", "v0-human-expected-length", "v0-human-reference-fasta",
                               "v0-human-reference-id", "v0-human-metadata", "v0-human-uniform-sample-date",
                               "v0-human-mask-file", "v0-human-callable-file", "v0-human-sample-callable-file",
                               "v0-human-in-variants", "v0-human-snp-only-constant-sites",
                               "v0-human-max-missing-fraction", "v0-human-audit-report",
                               "v0-exact-coalescent-correction", "v0-threshold-partition",
                               "v0-threshold-partition-params", "v0-global-scale-move", "v0-global-scale-window",
                               "v0-global-scale-moves-per-global-move",
                               "v0-history-gibbs", "v0-history-gibbs-weight", "v0-history-gibbs-sites",
                               "v0-site-history-refreshes", "v0-marginal-node-time-move",
                               "v0-marginal-node-time-weight", "v0-marginal-node-time-window"}) {
        if (opts.count(name) > 0) {
          std::cerr << "ERROR: --" << name << " is only valid together with --v0-human-mode\n";
          std::exit(EXIT_FAILURE);
        }
      }
    } else {
      auto num_inputs = opts.count("v0-in-fasta") + opts.count("v0-in-maple") + opts.count("v0-human-in-variants");
      if (num_inputs == 0) {
        std::cerr << "ERROR: --v0-human-mode requires --v0-in-fasta, --v0-in-maple or --v0-human-in-variants\n";
        std::exit(EXIT_FAILURE);
      }
      if (num_inputs > 1) {
        std::cerr << "ERROR: --v0-in-fasta, --v0-in-maple and --v0-human-in-variants are mutually exclusive\n";
        std::exit(EXIT_FAILURE);
      }
      auto open_or_die = [](const std::string& filename, const char* what) {
        auto is = std::make_unique<std::ifstream>(filename);
        if (not *is) {
          std::cerr << "ERROR: Could not read " << what << " " << filename << "\n";
          std::exit(EXIT_FAILURE);
        }
        return is;
      };
      try {
        if (opts.count("v0-human-reference-standard") == 0) {
          throw Human_input_error("--v0-human-mode requires --v0-human-reference-standard (rCRS, RSRS or custom)");
        }
        auto config = Human_input_config{};
        config.reference_standard = parse_reference_standard(opts["v0-human-reference-standard"].as<std::string>());
        if (opts.count("v0-human-expected-length") > 0) {
          config.expected_length = opts["v0-human-expected-length"].as<int>();
        }
        if (opts.count("v0-human-reference-id") > 0) {
          config.reference_id = opts["v0-human-reference-id"].as<std::string>();
        }
        if (opts.count("v0-human-uniform-sample-date") > 0) {
          config.uniform_sample_date = opts["v0-human-uniform-sample-date"].as<std::string>();
        }
        config.max_missing_fraction = opts["v0-human-max-missing-fraction"].as<double>();
        if (not site_partition_config.constant_sites_per_block().empty()) {
          if (opts.count("v0-human-snp-only-constant-sites") > 0) {
            throw Human_input_error(
                "--v0-human-snp-only-constant-sites and the constant_sites declarations of the "
                "site-partition file are two ways to say the same thing.  With site partitions, declare the "
                "removed constant sites inside each partition block, so that every restored site belongs to "
                "a partition.");
          }
          if (opts["v0-mpox-hack"].as<bool>()) {
            throw Human_input_error("Site partitions cannot be combined with --v0-mpox-hack");
          }
          config.snp_only_constant_sites = site_partition_config.constant_sites_per_block();
        }
        if (opts.count("v0-human-snp-only-constant-sites") > 0) {
          if (have_site_partitions) {
            throw Human_input_error(
                "With a site-partition file, the removed constant sites are declared inside each partition "
                "block (constant_sites), not with --v0-human-snp-only-constant-sites: a global count cannot "
                "be split between partitions.");
          }
          if (opts["v0-mpox-hack"].as<bool>()) {
            throw Human_input_error(
                "--v0-human-snp-only-constant-sites cannot be combined with --v0-mpox-hack.  Restoring removed "
                "constant sites relies on sites being exchangeable within one partition, and the mpox hack "
                "assigns partitions from the sequence context of each site, which the restored block does not "
                "have.  Supply the full-length alignment instead.");
          }
          config.snp_only_constant_sites =
              {parse_constant_site_counts(opts["v0-human-snp-only-constant-sites"].as<std::string>())};
        }

        auto metadata = std::optional<Sample_metadata>{};
        if (opts.count("v0-human-metadata") > 0) {
          auto metadata_is = open_or_die(opts["v0-human-metadata"].as<std::string>(), "metadata file");
          metadata = read_sample_metadata(*metadata_is);
        }
        auto mask_is = std::unique_ptr<std::ifstream>{};
        if (opts.count("v0-human-mask-file") > 0) {
          mask_is = open_or_die(opts["v0-human-mask-file"].as<std::string>(), "mask file");
        }
        auto callable_is = std::unique_ptr<std::ifstream>{};
        if (opts.count("v0-human-callable-file") > 0) {
          callable_is = open_or_die(opts["v0-human-callable-file"].as<std::string>(), "callable-sites file");
        }
        auto sample_callable_is = std::unique_ptr<std::ifstream>{};
        if (opts.count("v0-human-sample-callable-file") > 0) {
          sample_callable_is = open_or_die(opts["v0-human-sample-callable-file"].as<std::string>(),
                                           "per-sample callable-sites file");
        }
        auto observation = Human_observation_inputs{.mask = mask_is.get(),
                                                    .callable = callable_is.get(),
                                                    .sample_callable = sample_callable_is.get()};
        auto* metadata_ptr = metadata.has_value() ? &*metadata : nullptr;

        auto result = Human_input_result{};
        if (opts.count("v0-human-in-variants") > 0) {
          if (metadata_ptr == nullptr) {
            throw Human_input_error("Variant input requires --v0-human-metadata: the metadata table is the "
                                    "sample set, so that samples without any variant are present");
          }
          if (opts.count("v0-human-reference-fasta") == 0) {
            throw Human_input_error("Variant input requires --v0-human-reference-fasta: variant sites are "
                                    "positions in the reference and the unvaried callable sites are its bases");
          }
          auto variants_is = open_or_die(opts["v0-human-in-variants"].as<std::string>(), "variant table");
          auto reference_is = open_or_die(opts["v0-human-reference-fasta"].as<std::string>(),
                                          "reference FASTA file");
          std::cerr << "Reading human variant input " << opts["v0-human-in-variants"].as<std::string>() << "\n";
          result = load_human_variants(*variants_is, *reference_is, *metadata_ptr, observation, config);
        } else if (opts.count("v0-in-fasta") > 0) {
          auto fasta_is = open_or_die(opts["v0-in-fasta"].as<std::string>(), "input FASTA file");
          auto reference_is = std::unique_ptr<std::ifstream>{};
          if (opts.count("v0-human-reference-fasta") > 0) {
            reference_is = open_or_die(opts["v0-human-reference-fasta"].as<std::string>(), "reference FASTA file");
          }
          std::cerr << "Reading human FASTA input " << opts["v0-in-fasta"].as<std::string>() << "\n";
          result = load_human_fasta(*fasta_is, reference_is.get(), metadata_ptr, observation, config);
        } else {
          if (opts.count("v0-human-reference-fasta") > 0) {
            throw Human_input_error("--v0-human-reference-fasta applies to FASTA input only");
          }
          auto maple_is = open_or_die(opts["v0-in-maple"].as<std::string>(), "input MAPLE file");
          std::cerr << "Reading human MAPLE input " << opts["v0-in-maple"].as<std::string>() << "\n";
          result = load_human_maple(*maple_is, metadata_ptr, observation, config);
        }

        write_human_audit_summary(result.audit, std::cerr);
        if (opts.count("v0-human-audit-report") > 0) {
          auto report_filename = opts["v0-human-audit-report"].as<std::string>();
          auto report_os = std::ofstream{report_filename};
          if (not report_os) {
            throw Human_input_error("Could not write audit report " + report_filename);
          }
          write_human_audit_json(result.audit, report_os);
        }
        human_no_time_signal = result.audit.no_time_signal;
        human_observed_sites = result.audit.num_sites - result.audit.num_never_observed_sites;
        maple_file = std::move(result.maple);
      } catch (const Human_input_error& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
      }
    }

    if (not human_mode && opts.count("v0-in-fasta") > 0) {
      auto in_fasta_filename = opts["v0-in-fasta"].as<std::string>();
      auto in_fasta_is = std::ifstream{in_fasta_filename};
      if (not in_fasta_is) {
        std::cerr << "ERROR: Could not read input FASTA file " << in_fasta_filename << "\n";
        std::exit(EXIT_FAILURE);
      }
      in_fasta_is.seekg(0, std::ios_base::end);
      auto total_bytes = in_fasta_is.tellg();
      in_fasta_is.seekg(0, std::ios_base::beg);
      
      std::cerr << "Reading fasta file " << in_fasta_filename << "\n";
      auto read_pct_reported = -1;
      auto in_fasta = read_fasta(
          in_fasta_is,
          [total_bytes, &read_pct_reported](int seqs_so_far, size_t bytes_so_far) {
            auto pct = total_bytes > 0 ? static_cast<int>(10 * bytes_so_far / total_bytes) * 10 : 0;
            if (pct > read_pct_reported) {
              read_pct_reported = pct;
              std::cerr << absl::StreamFormat("- read %d sequences (%d%%)\n", seqs_so_far, pct);
            }
          },
          default_sequence_warning_hook);
      in_fasta_is.close();

      std::cerr << "Analysing fasta file " << in_fasta_filename << "\n";
      auto analyse_pct_reported = -1;
      maple_file = fasta_to_maple(
          in_fasta,
          [&analyse_pct_reported](int seqs_so_far, int total_seqs) {
            auto pct = total_seqs > 0 ? (10 * seqs_so_far / total_seqs) * 10 : 0;
            if (pct > analyse_pct_reported) {
              analyse_pct_reported = pct;
              std::cerr << absl::StreamFormat("- analysed %d / %d sequences (%d%%)\n", seqs_so_far, total_seqs, pct);
            }
          },
          default_sequence_warning_hook);
    }
    if (not human_mode && opts.count("v0-in-maple") > 0) {
      auto in_maple_filename = opts["v0-in-maple"].as<std::string>();
      auto in_maple_is = std::ifstream{in_maple_filename};
      if (not in_maple_is) {
        std::cerr << "ERROR: Could not read input MAPLE file " << in_maple_filename << "\n";
        std::exit(EXIT_FAILURE);
      }
      in_maple_is.seekg(0, std::ios_base::end);
      auto total_bytes = in_maple_is.tellg();
      in_maple_is.seekg(0, std::ios_base::beg);
      
      std::cerr << "Reading MAPLE file " << in_maple_filename << "\n";
      auto read_pct_reported = -1;
      maple_file = read_maple(
          in_maple_is,
          [total_bytes, &read_pct_reported](int seqs_so_far, size_t bytes_so_far) {
            auto pct = total_bytes > 0 ? static_cast<int>(10 * bytes_so_far / total_bytes) * 10 : 0;
            if (pct > read_pct_reported) {
              read_pct_reported = pct;
              std::cerr << absl::StreamFormat("- read %d sequences (%d%%)\n", seqs_so_far, pct);
            }
          },
          default_sequence_warning_hook);
      in_maple_is.close();
    }
    
    if (human_mode && init_method != Init_method::mp_plus_timing) {
      std::cerr << "ERROR: --v0-human-mode supports only the default initialization (mp-plus-timing).  The "
                << "'random' and 'old-usher-like' methods place nodes with pathogen-scale constants "
                << "(about 1 day or 13 days per mutation).\n";
      std::exit(EXIT_FAILURE);
    }

    // Age calibrations.  They are read before the initial tree is built, because an informative internal-node
    // calibration is a source of time information in its own right and relaxes the same-date gate below.
    auto free_topology_calibration = opts["v0-allow-free-topology-calibration"].as<bool>();
    if (free_topology_calibration) {
      auto refuse = [](const std::string& why) {
        std::cerr << "ERROR: --v0-allow-free-topology-calibration " << why << "\n";
        std::exit(EXIT_FAILURE);
      };
      if (opts.count("v0-calibration-file") == 0) { refuse("requires --v0-calibration-file."); }
      if (opts["v0-fix-topology"].as<bool>()) {
        refuse("cannot be combined with --v0-fix-topology: it is the mode in which the topology is sampled.");
      }
      if (threads != 1) {
        refuse("needs --v0-threads 1: the monophyly constraints are only implemented for a single tree "
               "partition part.");
      }
    }
    auto calibration_set = Calibration_set{};
    if (opts.count("v0-calibration-file") > 0) {
      auto cal_filename = opts["v0-calibration-file"].as<std::string>();
      if (not opts["v0-fix-topology"].as<bool>() && not free_topology_calibration) {
        std::cerr << "ERROR: --v0-calibration-file requires --v0-fix-topology.  A calibration names the common "
                  << "ancestor of a set of samples; with a free topology an SPR move can change which node that is, "
                  << "so the calibration would stop referring to the intended ancestor.  Supply the rooted topology "
                  << "with --v0-in-tree and add --v0-fix-topology, or add --v0-allow-free-topology-calibration to "
                  << "sample the topology with every calibrated sample set constrained to be monophyletic.\n";
        std::exit(EXIT_FAILURE);
      }
      auto cal_is = std::ifstream{cal_filename};
      if (not cal_is) {
        std::cerr << "ERROR: Could not read calibration file '" << cal_filename << "'\n";
        std::exit(EXIT_FAILURE);
      }
      try {
        calibration_set = read_calibration_config(cal_is);
      } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
      }
      std::cerr << absl::StreamFormat("Read %d age calibration(s) from '%s'\n",
                                      calibration_set.size(), cal_filename);
    }
    auto have_calibrations = not calibration_set.empty();

    // In human mode, the initial tree time scale comes from the user-supplied substitution rate, never from
    // the pathogen constant (1 mutation / 30 days).  Overall rate = per-site rate * observed sites.
    auto time_scale = Init_time_scale{};
    if (human_mode) {
      time_scale.forbid_pathogen_fallback = true;
      auto rate_per_site_year = 0.0;
      if (opts.count("v0-init-mutation-rate") > 0) {
        rate_per_site_year = opts["v0-init-mutation-rate"].as<double>();
      } else if (opts.count("v0-mu-prior-mean") > 0) {
        rate_per_site_year = opts["v0-mu-prior-mean"].as<double>();
      } else if (opts.count("v0-mu-prior-alpha") > 0 && opts.count("v0-mu-prior-beta") > 0 &&
                 opts["v0-mu-prior-beta"].as<double>() > 0.0) {
        rate_per_site_year = opts["v0-mu-prior-alpha"].as<double>() / opts["v0-mu-prior-beta"].as<double>();
      }
      if (rate_per_site_year > 0.0 && human_observed_sites > 0) {
        time_scale.fallback_lambda = rate_per_site_year / 365.0 * human_observed_sites;
      }
      if (human_no_time_signal && not time_scale.fallback_lambda.has_value() && not have_calibrations) {
        std::cerr << "ERROR: All samples have the same sampling date, so the data do not determine the absolute time "
                  << "scale, and the default mutation-rate prior and initial rate are pathogen values.  Supply time "
                  << "information explicitly: a proper rate prior (--v0-mu-prior-mean and --v0-mu-prior-stddev, or "
                  << "--v0-mu-prior-alpha and --v0-mu-prior-beta > 0), a fixed rate (--v0-fix-mutation-rate with "
                  << "--v0-init-mutation-rate), or at least one internal-node age calibration "
                  << "(--v0-calibration-file with --v0-fix-topology).\n";
        std::exit(EXIT_FAILURE);
      }
      if (human_no_time_signal && not time_scale.fallback_lambda.has_value() && have_calibrations
          && opts.count("v0-in-tree") == 0) {
        std::cerr << "ERROR: With identical sampling dates, no rate information and only age calibrations, the "
                  << "starting tree cannot be timed: Delphy's initializer places nodes from mutation counts and needs "
                  << "a rate for that.  Supply the rooted, timed topology with --v0-in-tree and "
                  << "--v0-in-tree-branch-lengths years (or days), which is also what makes the calibration targets "
                  << "well defined.\n";
        std::exit(EXIT_FAILURE);
      }
    }

    // Optional user-supplied starting tree
    auto in_tree_filename = std::string{};
    auto in_tree_options = External_tree_options{};
    if (opts.count("v0-in-tree") > 0) {
      in_tree_filename = opts["v0-in-tree"].as<std::string>();
      if (opts.count("v0-init") > 0 || opts.count("v0-init-heuristic") > 0 || opts.count("v0-init-random") > 0) {
        std::cerr << "ERROR: --v0-in-tree cannot be combined with --v0-init, --v0-init-heuristic or "
                  << "--v0-init-random.\n";
        std::exit(EXIT_FAILURE);
      }
      if (opts.count("v0-in-tree-branch-lengths") > 0) {
        auto units_str = opts["v0-in-tree-branch-lengths"].as<std::string>();
        if (units_str == "ignore") { in_tree_options.units = Branch_length_units::ignore; }
        else if (units_str == "years") { in_tree_options.units = Branch_length_units::years; }
        else if (units_str == "days") { in_tree_options.units = Branch_length_units::days; }
        else if (units_str == "subs-per-site") { in_tree_options.units = Branch_length_units::subs_per_site; }
        else {
          std::cerr << "ERROR: Unknown --v0-in-tree-branch-lengths value: " << units_str
                    << " (expected ignore, years, days or subs-per-site)\n";
          std::exit(EXIT_FAILURE);
        }
      }
      auto rate_per_site_year = 0.0;
      if (opts.count("v0-init-mutation-rate") > 0) {
        rate_per_site_year = opts["v0-init-mutation-rate"].as<double>();
      } else if (opts.count("v0-mu-prior-mean") > 0) {
        rate_per_site_year = opts["v0-mu-prior-mean"].as<double>();
      } else if (opts.count("v0-mu-prior-alpha") > 0 && opts.count("v0-mu-prior-beta") > 0 &&
                 opts["v0-mu-prior-beta"].as<double>() > 0.0) {
        rate_per_site_year = opts["v0-mu-prior-alpha"].as<double>() / opts["v0-mu-prior-beta"].as<double>();
      }
      if (rate_per_site_year > 0.0) {
        in_tree_options.rate_per_site_per_day = rate_per_site_year / 365.0;
      }
      in_tree_options.num_observed_sites = human_mode ? human_observed_sites
                                                       : static_cast<Site_index>(std::ssize(maple_file.ref_sequence));
    } else if (opts.count("v0-in-tree-branch-lengths") > 0) {
      std::cerr << "ERROR: --v0-in-tree-branch-lengths requires --v0-in-tree.\n";
      std::exit(EXIT_FAILURE);
    }

    std::cerr << "Building initial tree...\n";
    auto last_pct_reported = -1;
    auto last_round_reported = 0;
    auto pct_progress_hook = [&last_pct_reported](const char* label, int tips_so_far, int total_tips) {
      auto pct = total_tips > 0 ? (10 * tips_so_far / total_tips) * 10 : 0;
      if (pct > last_pct_reported) {
        last_pct_reported = pct;
        std::cerr << absl::StreamFormat("- %s: %d / %d tips (%d%%)\n", label, tips_so_far, total_tips, pct);
      }
    };
    auto L = std::ssize(maple_file.ref_sequence);
    try {
      if (not in_tree_filename.empty()) {
        auto tree_is = std::ifstream{in_tree_filename};
        if (not tree_is) {
          std::cerr << "ERROR: Could not read tree file '" << in_tree_filename << "'\n";
          std::exit(EXIT_FAILURE);
        }
        auto ntree = Newick_tree{};
        try {
          ntree = read_newick_tree(tree_is);
        } catch (const std::exception& e) {
          std::cerr << "ERROR: Could not parse Newick tree '" << in_tree_filename << "': " << e.what() << "\n";
          std::exit(EXIT_FAILURE);
        }
        auto summary = External_tree_summary{};
        try {
          tree = build_phylo_tree_from_newick(
              ntree, std::move(maple_file.ref_sequence), std::move(maple_file.tip_descs), prng,
              in_tree_options, &summary);
        } catch (const std::exception& e) {
          std::cerr << "ERROR: Cannot use input tree '" << in_tree_filename << "': " << e.what() << "\n";
          std::exit(EXIT_FAILURE);
        }
        std::cerr << absl::StreamFormat(
            "- input tree: %d tips, %d parsimony mutations in the initial history, root at %s; "
            "branch lengths %s\n",
            summary.num_tips, summary.num_parsimony_mutations, to_iso_date(summary.root_t),
            summary.used_branch_lengths ? "used as time" : "not used");
      } else {
      // The alignment is needed again if the guide tree violates a monophyly constraint
      auto constrained_init_input = (free_topology_calibration && not calibration_set.empty())
          ? std::optional<Maple_file>{maple_file} : std::nullopt;
      tree = build_rough_initial_tree_from_maple(
          std::move(maple_file), init_method, prng,
          [&](int tips_so_far, int total_tips) {
            pct_progress_hook("guide tree", tips_so_far, total_tips);
          },
          [&](int round, int tips_so_far, int total_tips) {
            if (round != last_round_reported) {
              last_round_reported = round;
              last_pct_reported = -1;
            }
            pct_progress_hook(absl::StrFormat("refine round %d", round).c_str(), tips_so_far, total_tips);
          },
          [&, spr_last_pct = -1](int attempts_so_far, int max_attempts, int cur_deltas) mutable {
            auto pct = max_attempts > 0 ? (10 * attempts_so_far / max_attempts) * 10 : 0;
            if (pct > spr_last_pct) {
              spr_last_pct = pct;
              std::cerr << absl::StreamFormat(
                  "- SPR refine: %d / %d attempts (%d%%), %d deltas\n",
                  attempts_so_far, max_attempts, pct, cur_deltas);
            }
          },
          [&, root_last_done = 0](int /*substage_id*/, int substage, int num_substages,
                                  int nodes, int total) mutable {
            if (nodes >= total && substage != root_last_done) {
              root_last_done = substage;
              std::cerr << absl::StreamFormat("- rooting pass %d / %d done\n", substage, num_substages);
            }
          },
          [&](const Rooting_info& rooting_info) {
            auto method_str = (rooting_info.method == Rooting_method::regression)
                ? "root-to-tip regression" : "midpoint";
            std::cerr << absl::StreamFormat("- rooting: %s, R^2 = %.4f", method_str, rooting_info.r2)
                      << absl::StreamFormat(", lambda = %.4g mut/yr (%.2g * 10^-3 mut/site/yr)",
                                            rooting_info.lambda * 365.0,
                                            rooting_info.lambda * 365.0 / L * 1000)
                      << absl::StreamFormat(", t_MRCA = %s\n", to_iso_date(rooting_info.t_MRCA));
          },
          time_scale);
      if (constrained_init_input.has_value()) {
        tree = build_free_topology_calibration_start(
            std::move(tree), std::move(*constrained_init_input), calibration_set, time_scale,
            opts.count("v0-init-mutation-rate") > 0
                ? std::optional<double>{opts["v0-init-mutation-rate"].as<double>()} : std::nullopt,
            human_mode ? human_observed_sites : static_cast<int>(L));
      }
      }
    } catch (const std::runtime_error& e) {
      if (not human_mode) { throw; }
      std::cerr << "ERROR: " << e.what() << "\n";
      std::exit(EXIT_FAILURE);
    }
    
    assert_phylo_tree_integrity(tree);
    
    auto tip_count = (std::ssize(tree) + 1) / 2;
    std::cerr << absl::StreamFormat("Built initial tree with %d tips\n", tip_count);

    auto steps = 100'000 * tip_count;  // Useful default
    if (opts.count("v0-steps")) {
      steps = opts["v0-steps"].as<int64_t>();
    }
    if (steps < 0) {
      std::cerr << "ERROR: Number of steps must be non-negative, got " << steps << "\n";
      std::exit(EXIT_FAILURE);
    }

    auto log_filename = std::optional<std::string>{};
    if (opts.count("v0-out-log-file")) {
      log_filename = opts["v0-out-log-file"].as<std::string>();
    }

    auto log_every = std::max(1L, steps / 10000);
    if (opts.count("v0-log-every")) {
      log_every = opts["v0-log-every"].as<int64_t>();
    }
    if (log_every <= 0) {
      std::cerr << "ERROR: Number of steps between log entries must be positive, got " << log_every << "\n";
      std::exit(EXIT_FAILURE);
    }

    auto trees_filename = std::optional<std::string>{};
    if (opts.count("v0-out-trees-file")) {
      trees_filename = opts["v0-out-trees-file"].as<std::string>();
    }

    auto tree_every = std::max(1L, steps / 1000);
    if (opts.count("v0-tree-every")) {
      tree_every = opts["v0-tree-every"].as<int64_t>();
    }
    if (tree_every <= 0) {
      std::cerr << "ERROR: Number of steps between tree snapshots must be positive, got " << tree_every << "\n";
      std::exit(EXIT_FAILURE);
    }

    auto trees_significant_digits = 6;
    if (opts.count("v0-out-trees-digits")) {
      trees_significant_digits = opts["v0-out-trees-digits"].as<int>();
      if (trees_significant_digits < 1 || trees_significant_digits > 17) {
        std::cerr << "ERROR: --v0-out-trees-digits must be between 1 and 17 (17 significant digits "
                     "round-trip every double), got " << trees_significant_digits << "\n";
        std::exit(EXIT_FAILURE);
      }
    }

    auto delphy_output_filename = std::optional<std::string>{};
    if (opts.count("v0-out-delphy-file")) {
      delphy_output_filename = opts["v0-out-delphy-file"].as<std::string>();
    }

    auto delphy_output_metadata = std::optional<std::string>{};
    if (opts.count("v0-out-delphy-metadata-file")) {
      auto delphy_output_metadata_filename = opts["v0-out-delphy-metadata-file"].as<std::string>();
      auto fs = std::ifstream{delphy_output_metadata_filename};
      if (not fs) {
        std::cerr << "ERROR: Could not open .dphy metadata file " << delphy_output_metadata_filename << "\n";
        std::exit(EXIT_FAILURE);
      }

      auto sstr = std::ostringstream{};
      sstr << fs.rdbuf();
      delphy_output_metadata = std::move(sstr).str();
      std::cerr << *delphy_output_metadata << "\n";
    }

    auto delphy_snapshot_every = std::max(1L, steps / 1000);
    if (opts.count("v0-delphy-snapshot-every")) {
      delphy_snapshot_every = opts["v0-delphy-snapshot-every"].as<int64_t>();
    }
    if (delphy_snapshot_every <= 0) {
      std::cerr << "ERROR: Number of steps between .dphy run snapshots must be positive, got " << delphy_snapshot_every << "\n";
      std::exit(EXIT_FAILURE);
    }

    auto alpha_move_enabled = opts["v0-site-rate-heterogeneity"].as<bool>();

    auto fix_mutation_rate = opts["v0-fix-mutation-rate"].as<bool>();
    auto init_mu = opts["v0-init-mutation-rate"].as<double>();
    if (init_mu <= 0.0) {
      std::cerr << "ERROR: Initial mutation rate must be positive, got "
                << init_mu << " subst / site / year"  << "\n";
      std::exit(EXIT_FAILURE);
    }
    init_mu /= 365.0;  // convert to subst / site / day

    auto has_mu_prior_alpha_beta =
        (opts.count("v0-mu-prior-alpha") > 0 || opts.count("v0-mu-prior-beta") > 0);
    auto has_mu_prior_mean_stddev =
        (opts.count("v0-mu-prior-mean") > 0 || opts.count("v0-mu-prior-stddev") > 0);

    if (has_mu_prior_alpha_beta && has_mu_prior_mean_stddev) {
      std::cerr << "ERROR: --v0-mu-prior-alpha/beta and --v0-mu-prior-mean/stddev "
                << "are mutually exclusive\n";
      std::exit(EXIT_FAILURE);
    }

    auto mu_prior_alpha = 1.0;
    auto mu_prior_beta = 0.0;
    if (has_mu_prior_mean_stddev) {
      if (opts.count("v0-mu-prior-mean") == 0 || opts.count("v0-mu-prior-stddev") == 0) {
        std::cerr << "ERROR: --v0-mu-prior-mean and --v0-mu-prior-stddev "
                  << "must be specified together\n";
        std::exit(EXIT_FAILURE);
      }
      auto mean = opts["v0-mu-prior-mean"].as<double>();     // per year
      auto stddev = opts["v0-mu-prior-stddev"].as<double>(); // per year
      if (mean <= 0.0) {
        std::cerr << "ERROR: --v0-mu-prior-mean must be positive, got " << mean << "\n";
        std::exit(EXIT_FAILURE);
      }
      if (stddev <= 0.0) {
        std::cerr << "ERROR: --v0-mu-prior-stddev must be positive, got " << stddev << "\n";
        std::exit(EXIT_FAILURE);
      }
      mu_prior_alpha = std::pow(mean / stddev, 2);
      mu_prior_beta = mean * 365.0 / (stddev * stddev);  // convert per-year to per-day
    } else {
      mu_prior_alpha = opts["v0-mu-prior-alpha"].as<double>();
      auto beta_cli = opts["v0-mu-prior-beta"].as<double>();
      if (mu_prior_alpha < 0.0) {
        std::cerr << "ERROR: --v0-mu-prior-alpha must be non-negative, got " << mu_prior_alpha << "\n";
        std::exit(EXIT_FAILURE);
      }
      if (beta_cli < 0.0) {
        std::cerr << "ERROR: --v0-mu-prior-beta must be non-negative, got " << beta_cli << "\n";
        std::exit(EXIT_FAILURE);
      }
      mu_prior_beta = beta_cli * 365.0;  // convert per-year to per-day
    }

    // If the user didn't explicitly set an initial mu and the prior is proper,
    // use the prior mean as the initial value
    if (opts.count("v0-init-mutation-rate") == 0 && mu_prior_alpha > 0.0 && mu_prior_beta > 0.0) {
      init_mu = mu_prior_alpha / mu_prior_beta;  // already in per-day units
    }

    if (human_no_time_signal) {
      auto mu_is_fixed = fix_mutation_rate && opts.count("v0-init-mutation-rate") > 0;
      auto mu_prior_is_proper = mu_prior_alpha > 0.0 && mu_prior_beta > 0.0;
      // An informative internal-node age calibration determines the absolute time scale on its own: the node age
      // together with the mutations below it identifies the rate, which is then sampled jointly with the times.
      if (not (mu_is_fixed || mu_prior_is_proper || have_calibrations)) {
        std::cerr << "ERROR: All samples have the same sampling date, so the data do not determine the absolute time "
                  << "scale, and the default mutation-rate prior and initial rate are pathogen values.  Supply time "
                  << "information explicitly: a proper rate prior (--v0-mu-prior-mean and --v0-mu-prior-stddev, or "
                  << "--v0-mu-prior-alpha and --v0-mu-prior-beta > 0), a fixed rate (--v0-fix-mutation-rate with "
                  << "--v0-init-mutation-rate), or at least one internal-node age calibration "
                  << "(--v0-calibration-file with --v0-fix-topology).\n";
        std::exit(EXIT_FAILURE);
      }
    }

    auto mpox_hack_enabled = opts["v0-mpox-hack"].as<bool>();

    auto target_coal_prior_cells = opts["v0-target-coal-prior-cells"].as<int>();

    // Create and configure initial run.
    //
    // The coalescent prior's first grid is built while the Run is constructed (`repartition` seeds
    // the per-partition coalescent structures), so its cell width has to be decided here and not
    // afterwards.  At the historical one-day resolution that is one cell per day of tree depth,
    // which on a tree tens of thousands of years deep is tens of millions of cells and gigabytes of
    // allocation before the first MCMC step.
    auto t0 = calc_max_tip_time(tree);
    auto init_coal_prior_cells = opts["v0-coal-prior-init-cells"].as<int>();
    if (init_coal_prior_cells < 0) {
      std::cerr << "ERROR: --v0-coal-prior-init-cells must be non-negative, got "
                << init_coal_prior_cells << "\n";
      std::exit(EXIT_FAILURE);
    }
    if (human_mode && opts.count("v0-coal-prior-init-cells") == 0) {
      init_coal_prior_cells = target_coal_prior_cells;
    }
    auto init_coal_prior_t_step = 1.0;  // days; the default for every pathogen workflow
    if (init_coal_prior_cells > 0) {
      auto span = t0 - tree.at_root().t;
      if (span > 0.0) { init_coal_prior_t_step = span / init_coal_prior_cells; }
    }
    auto run = std::make_shared<Run>(*thread_pool, prng, seed, std::move(tree),
                                     init_coal_prior_t_step);
    run->set_paranoid(opts["v0-paranoid"].as<bool>());
    {
      auto per_node = opts["v0-local-moves-per-node-per-global-move"].as<int>();
      auto root_factor = opts["v0-root-displace-scale-factor"].as<double>();
      if (per_node < 1) {
        std::cerr << "ERROR: --v0-local-moves-per-node-per-global-move must be at least 1, got " << per_node << "\n";
        std::exit(EXIT_FAILURE);
      }
      if (not (root_factor > 0.0) || not std::isfinite(root_factor)) {
        std::cerr << "ERROR: --v0-root-displace-scale-factor must be positive and finite, got " << root_factor << "\n";
        std::exit(EXIT_FAILURE);
      }
      auto pop_moves = opts["v0-pop-moves-per-global-move"].as<int>();
      if (pop_moves < 1) {
        std::cerr << "ERROR: --v0-pop-moves-per-global-move must be at least 1, got " << pop_moves << "\n";
        std::exit(EXIT_FAILURE);
      }
      run->set_pop_moves_per_global_move(pop_moves);
      run->set_skip_fixed_tip_moves(opts["v0-skip-fixed-tip-moves"].as<bool>());
      run->set_root_displace_scale_factor(root_factor);
      if (opts.count("v0-local-moves-per-node-per-global-move") > 0) {
        run->set_local_moves_per_global_move(per_node * static_cast<int>(std::ssize(run->tree())));
      }
    }
    run->set_topology_moves_enabled(not opts["v0-fix-topology"].as<bool>());
    run->set_mpox_hack_enabled(mpox_hack_enabled);
    run->set_alpha_move_enabled(alpha_move_enabled);
    run->set_mu_move_enabled(not fix_mutation_rate);
    run->set_mu(init_mu);
    run->set_mu_prior_alpha(mu_prior_alpha);
    run->set_mu_prior_beta(mu_prior_beta);
    run->set_target_coal_prior_cells(target_coal_prior_cells);

    // The substitution model.  Reported before it is used, so that a run records the model it ran.
    {
      auto model = Subst_model{};
      auto spec = std::string{"HKY85"};
      auto explicit_model = opts.count("v0-subst-model") > 0;
      if (opts["v0-mpox-hack"].as<bool>()) {
        for (const auto* name : {"v0-subst-model", "v0-subst-initial-exchangeabilities",
                                 "v0-subst-frequencies", "v0-subst-exchangeability-prior",
                                 "v0-subst-fix-exchangeabilities", "v0-subst-fix-frequencies"}) {
          if (opts.count(name) > 0) {
            std::cerr << "ERROR: --" << name << " cannot be combined with --v0-mpox-hack, which "
                         "builds its own two rate matrices from the sequence context\n";
            std::exit(EXIT_FAILURE);
          }
        }
      }
      try {
        if (explicit_model) { spec = opts["v0-subst-model"].as<std::string>(); }
        model = parse_subst_model(spec);
        if (opts.count("v0-subst-initial-exchangeabilities") > 0) {
          auto rates = std::vector<double>{};
          for (auto token :
               absl::StrSplit(opts["v0-subst-initial-exchangeabilities"].as<std::string>(), ',')) {
            auto value = 0.0;
            if (not absl::SimpleAtod(absl::StripAsciiWhitespace(token), &value)) {
              throw std::invalid_argument(absl::StrFormat(
                  "--v0-subst-initial-exchangeabilities: '%s' is not a number",
                  std::string{absl::StripAsciiWhitespace(token)}));
            }
            rates.push_back(value);
          }
          if (std::ssize(rates) != model.grouping.num_classes()) {
            throw std::invalid_argument(absl::StrFormat(
                "--v0-subst-initial-exchangeabilities gives %d rates, but the model %s has %d "
                "exchangeability classes", rates.size(), model.grouping.code(),
                model.grouping.num_classes()));
          }
          model.class_rate = std::move(rates);
        }
        if (opts.count("v0-subst-frequencies") > 0) {
          auto freqs = std::vector<double>{};
          for (auto token : absl::StrSplit(opts["v0-subst-frequencies"].as<std::string>(), ',')) {
            auto value = 0.0;
            if (not absl::SimpleAtod(absl::StripAsciiWhitespace(token), &value)) {
              throw std::invalid_argument(absl::StrFormat(
                  "--v0-subst-frequencies: '%s' is not a number",
                  std::string{absl::StripAsciiWhitespace(token)}));
            }
            freqs.push_back(value);
          }
          if (std::ssize(freqs) != k_num_real_seq_letters) {
            throw std::invalid_argument(absl::StrFormat(
                "--v0-subst-frequencies needs %d numbers, one each for A, C, G and T, but %d were "
                "given", k_num_real_seq_letters, freqs.size()));
          }
          model.frequency_mode = Frequency_mode::fixed;
          for (auto i = 0; i != k_num_real_seq_letters; ++i) {
            model.pi_a[real_seq_letter_from_index(i)] = freqs[i];
          }
        } else if (model.frequency_mode != Frequency_mode::equal) {
          model.pi_a = run->hky_pi();  // the frequencies of the reference sequence, as before
        }
        if (opts["v0-subst-fix-frequencies"].as<bool>() &&
            model.frequency_mode == Frequency_mode::estimated) {
          model.frequency_mode = Frequency_mode::fixed;
        }
        if (opts.count("v0-subst-exchangeability-prior") > 0) {
          auto values = std::vector<double>{};
          for (auto token :
               absl::StrSplit(opts["v0-subst-exchangeability-prior"].as<std::string>(), ',')) {
            auto value = 0.0;
            if (not absl::SimpleAtod(absl::StripAsciiWhitespace(token), &value)) {
              throw std::invalid_argument(absl::StrFormat(
                  "--v0-subst-exchangeability-prior: '%s' is not a number",
                  std::string{absl::StripAsciiWhitespace(token)}));
            }
            values.push_back(value);
          }
          if (std::ssize(values) != 3) {
            throw std::invalid_argument(
                "--v0-subst-exchangeability-prior needs three numbers: the mean of the log for a "
                "class of transitions only, the mean of the log for every other class, and the "
                "standard deviation of the log");
          }
          run->set_subst_exchangeability_prior(values[0], values[1], values[2]);
        }
        run->set_subst_model(model);
        run->set_subst_exchangeability_move_enabled(
            not opts["v0-subst-fix-exchangeabilities"].as<bool>() &&
            model.grouping.num_free_rates() > 0);
        run->set_subst_frequencies_move_enabled(
            model.frequency_mode == Frequency_mode::estimated);
      } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
      }
      if (explicit_model || opts.count("v0-subst-frequencies") > 0 ||
          opts.count("v0-subst-initial-exchangeabilities") > 0 ||
          opts.count("v0-subst-exchangeability-prior") > 0 ||
          opts["v0-subst-fix-exchangeabilities"].as<bool>() ||
          opts["v0-subst-fix-frequencies"].as<bool>()) {
        const auto& m = run->subst_model();
        // With fixed frequencies the grouping has no published name of its own, so the names of the
        // same grouping with estimated frequencies are reported for orientation.
        auto names = subst_model_names_of(m.grouping, m.frequency_mode);
        if (names.empty() && m.frequency_mode == Frequency_mode::fixed) {
          names = subst_model_names_of(m.grouping, Frequency_mode::estimated);
        }
        std::cerr << absl::StreamFormat(
            "Substitution model: code %s%s, frequencies %s, %d free exchangeability %s, %d free "
            "frequencies\n",
            m.grouping.code(),
            names.empty() ? "" : absl::StrFormat(" (%s%s)", absl::StrJoin(names, " / "),
                m.frequency_mode == Frequency_mode::fixed ? ", with the frequencies fixed" : ""),
            to_string(m.frequency_mode), m.grouping.num_free_rates(),
            m.grouping.num_free_rates() == 1 ? "parameter" : "parameters",
            m.frequency_mode == Frequency_mode::estimated ? 3 : 0);
        for (auto k = 0; k != m.grouping.num_classes(); ++k) {
          auto members = std::string{};
          for (auto index : m.grouping.members_of(k)) {
            if (not members.empty()) { members += ", "; }
            members += exchangeability_name(index);
          }
          std::cerr << absl::StreamFormat(
              "- class %d (%s): initial relative rate %.6g%s\n", k, members, m.class_rate[k],
              k == m.grouping.reference_class() ? ", held at 1 as the relative scale" : "");
        }
        std::cerr << absl::StreamFormat(
            "- frequencies: A %.6g, C %.6g, G %.6g, T %.6g\n",
            m.pi_a[Real_seq_letter::A], m.pi_a[Real_seq_letter::C],
            m.pi_a[Real_seq_letter::G], m.pi_a[Real_seq_letter::T]);
        if (m.grouping.num_free_rates() > 0) {
          std::cerr << absl::StreamFormat(
              "- prior on every free relative exchangeability: log-normal, mean of the log %g for a "
              "class of transitions only and %g otherwise, standard deviation of the log %g\n",
              run->subst_exchangeability_prior_log_mean_transition(),
              run->subst_exchangeability_prior_log_mean_other(),
              run->subst_exchangeability_prior_sd());
        }
        if (not run->subst_exchangeability_move_enabled()) {
          std::cerr << "- the relative exchangeabilities are fixed at these values\n";
        }
        if (m.frequency_mode != Frequency_mode::estimated) {
          std::cerr << "- the frequencies are fixed at these values\n";
        }
      }
    }

    if (have_site_partitions) {
      try {
        run->set_site_partition(site_partition_config);
        if (opts.count("v0-site-partition-initial-rates") > 0) {
          auto rates = std::vector<double>{};
          for (auto token : absl::StrSplit(opts["v0-site-partition-initial-rates"].as<std::string>(), ',')) {
            auto value = 0.0;
            if (not absl::SimpleAtod(absl::StripAsciiWhitespace(token), &value)) {
              throw std::invalid_argument(absl::StrFormat(
                  "--v0-site-partition-initial-rates: '%s' is not a number",
                  std::string{absl::StripAsciiWhitespace(token)}));
            }
            rates.push_back(value);
          }
          if (std::ssize(rates) != run->site_partition().num_partitions()) {
            throw std::invalid_argument(absl::StrFormat(
                "--v0-site-partition-initial-rates gives %d rates for %d partitions", rates.size(),
                run->site_partition().num_partitions()));
          }
          run->set_site_partition_rates(std::move(rates));
        }
        run->set_site_partition_rate_prior_concentration(
            opts["v0-site-partition-rate-prior-concentration"].as<double>());
        run->set_site_partition_rate_move_enabled(not opts["v0-site-partition-fix-rates"].as<bool>());
      } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
      }
      const auto& partition = run->site_partition();
      std::cerr << absl::StreamFormat("Site partitions (%d), relative rates normalized to a mean of 1 "
                                      "weighted by observed sites:\n", partition.num_partitions());
      for (auto p = 0; p != partition.num_partitions(); ++p) {
        std::cerr << absl::StreamFormat("- %s: %d sites, %d observed in some sample, initial relative rate "
                                        "%.6g\n", partition.names[p], partition.num_sites[p],
                                        partition.num_observed_sites[p], run->site_partition_rates()[p]);
      }
      if (not run->site_partition_rate_move_enabled()) {
        std::cerr << "- the relative rates are fixed at these values (--v0-site-partition-fix-rates)\n";
      }
    }

    if (have_calibrations) {
      try {
        run->set_calibrations(calibration_set);
      } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
      }
      const auto& bound_tree = run->tree();
      const auto& nodes = run->calibration_nodes();
      std::cerr << "Age calibrations (prior on the MRCA age of each sample set):\n";
      for (auto i = 0; i != run->calibrations().size(); ++i) {
        const auto& c = run->calibrations().calibrations[i];
        std::cerr << absl::StreamFormat(
            "- %s: %s, %s prior, node %d, initial age %.4g years before %s\n",
            c.name,
            c.all_tips ? std::string{"all samples (the root of the tree)"}
                       : absl::StrFormat("%d samples", std::ssize(c.taxa)),
            to_string(c.prior.distribution), nodes[i], c.age_of(bound_tree.at(nodes[i]).t),
            to_iso_date(c.reference_t));
      }
      std::cerr << "  NOTE: a calibration constrains the MRCA of the SAMPLED lineages.  For 'taxa all' that is the "
                << "root of the supplied tree, which is not by itself the common ancestor of every lineage in the "
                << "population (for human mtDNA, not by itself the root of the whole mtDNA genealogy).\n";
      std::cerr << absl::StreamFormat("  log calibration prior at the initial tree: %.6g\n",
                                      run->log_calibration_prior());
    }

    // Population model
    auto has_exp_pop_model_parameters =
        (opts.count("v0-fix-final-pop-size") > 0) ||
        (opts.count("v0-init-final-pop-size") > 0) ||
        (opts.count("v0-fix-pop-growth-rate") > 0) ||
        (opts.count("v0-init-pop-growth-rate") > 0) ||
        (opts.count("v0-pop-inv-n0-prior-alpha") > 0) ||
        (opts.count("v0-pop-inv-n0-prior-beta") > 0) ||
        (opts.count("v0-pop-n0-prior-mean") > 0) ||
        (opts.count("v0-pop-n0-prior-stddev") > 0) ||
        (opts.count("v0-pop-g-prior-mu") > 0) ||
        (opts.count("v0-pop-g-prior-scale") > 0) ||
        (opts.count("v0-pop-growth-rate-min") > 0) ||
        (opts.count("v0-pop-growth-rate-max") > 0) ||
        (opts.count("v0-pop-g-prior-exponential-with-mean") > 0);
    auto has_skygrid_pop_model_parameters =
        (opts.count("v0-skygrid-type") > 0) ||
        (opts.count("v0-skygrid-num-parameters") > 0) ||
        (opts.count("v0-skygrid-cutoff") > 0) ||
        (opts.count("v0-skygrid-first-knot-date") > 0) ||
        (opts.count("v0-skygrid-last-knot-date") > 0) ||
        (opts.count("v0-skygrid-infer-prior-smoothness") > 0) ||
        (opts.count("v0-skygrid-prior-double-half-time") > 0) ||
        (opts.count("v0-skygrid-tau") > 0) ||
        (opts.count("v0-skygrid-tau-prior-alpha") > 0) ||
        (opts.count("v0-skygrid-tau-prior-beta") > 0) ||
        (opts.count("v0-skygrid-disable-low-pop-barrier") > 0) ||
        (opts.count("v0-skygrid-low-pop-barrier-loc") > 0) ||
        (opts.count("v0-skygrid-low-pop-barrier-scale") > 0) ||
        (opts.count("v0-skygrid-inv-nbar-prior-alpha") > 0) ||
        (opts.count("v0-skygrid-inv-nbar-prior-beta") > 0) ||
        (opts.count("v0-skygrid-nbar-prior-mean") > 0) ||
        (opts.count("v0-skygrid-nbar-prior-stddev") > 0);

    if (opts["v0-pop-model"].as<std::string>() == "exponential") {
      
      // Exponential population model
      
      if (has_skygrid_pop_model_parameters) {
        std::cerr << "ERROR: Cannot specify parameters for 'skygrid' model when pop-model is 'exponential'\n";
        std::exit(EXIT_FAILURE);
      }
      
      auto fix_final_pop_size = opts["v0-fix-final-pop-size"].as<bool>();
      auto init_final_pop_size = opts["v0-init-final-pop-size"].as<double>();
      if (init_final_pop_size <= 0.0) {
        std::cerr << "ERROR: Initial effective population size at time of last tip must be positive, got "
                  << init_final_pop_size << " years"  << "\n";
        std::exit(EXIT_FAILURE);
      }
      init_final_pop_size *= 365.0;  // convert to days

      auto fix_pop_growth_rate = opts["v0-fix-pop-growth-rate"].as<bool>();
      auto init_pop_growth_rate = opts["v0-init-pop-growth-rate"].as<double>();
      init_pop_growth_rate /= 365.0;  // convert to e-foldings / day

      // Inverse-Gamma prior on n0
      auto has_inv_n0_prior_alpha_beta =
          (opts.count("v0-pop-inv-n0-prior-alpha") > 0 || opts.count("v0-pop-inv-n0-prior-beta") > 0);
      auto has_n0_prior_mean_stddev =
          (opts.count("v0-pop-n0-prior-mean") > 0 || opts.count("v0-pop-n0-prior-stddev") > 0);

      if (has_inv_n0_prior_alpha_beta && has_n0_prior_mean_stddev) {
        std::cerr << "ERROR: --v0-pop-inv-n0-prior-alpha/beta and --v0-pop-n0-prior-mean/stddev "
                  << "are mutually exclusive\n";
        std::exit(EXIT_FAILURE);
      }

      if (has_n0_prior_mean_stddev) {
        if (opts.count("v0-pop-n0-prior-mean") == 0 || opts.count("v0-pop-n0-prior-stddev") == 0) {
          std::cerr << "ERROR: --v0-pop-n0-prior-mean and --v0-pop-n0-prior-stddev "
                    << "must be specified together\n";
          std::exit(EXIT_FAILURE);
        }
        auto mean = opts["v0-pop-n0-prior-mean"].as<double>();   // years
        auto stddev = opts["v0-pop-n0-prior-stddev"].as<double>(); // years
        if (mean <= 0.0) {
          std::cerr << "ERROR: --v0-pop-n0-prior-mean must be positive, got " << mean << "\n";
          std::exit(EXIT_FAILURE);
        }
        if (stddev <= 0.0) {
          std::cerr << "ERROR: --v0-pop-n0-prior-stddev must be positive, got " << stddev << "\n";
          std::exit(EXIT_FAILURE);
        }
        auto alpha = 2.0 + std::pow(mean / stddev, 2);
        auto beta_years = mean * (alpha - 1.0);
        run->set_pop_inv_n0_prior_alpha(alpha);
        run->set_pop_inv_n0_prior_beta(beta_years * 365.0);  // convert to days

        // Set initial n0 to prior mean
        if (opts.count("v0-init-final-pop-size") == 0) {
          init_final_pop_size = mean * 365.0;
        }
      } else {
        auto alpha = opts["v0-pop-inv-n0-prior-alpha"].as<double>();
        auto beta_cli = opts["v0-pop-inv-n0-prior-beta"].as<double>();
        if (alpha < 0.0) {
          std::cerr << "ERROR: --v0-pop-inv-n0-prior-alpha must be non-negative, got " << alpha << "\n";
          std::exit(EXIT_FAILURE);
        }
        if (beta_cli < 0.0) {
          std::cerr << "ERROR: --v0-pop-inv-n0-prior-beta must be non-negative, got " << beta_cli << "\n";
          std::exit(EXIT_FAILURE);
        }
        run->set_pop_inv_n0_prior_alpha(alpha);
        run->set_pop_inv_n0_prior_beta(beta_cli * 365.0);  // convert years to days

        // If the user didn't explicitly set an initial n0 and the prior mean is defined,
        // use the prior mean as the initial value
        if (opts.count("v0-init-final-pop-size") == 0 && alpha > 1.0 && beta_cli > 0.0) {
          init_final_pop_size = beta_cli / (alpha - 1.0) * 365.0;
        }
      }

      // Laplace prior on g, with optional bounds
      auto has_g_prior_direct =
          (opts.count("v0-pop-g-prior-mu") > 0) ||
          (opts.count("v0-pop-g-prior-scale") > 0) ||
          (opts.count("v0-pop-growth-rate-min") > 0) ||
          (opts.count("v0-pop-growth-rate-max") > 0);
      auto has_g_prior_exponential =
          (opts.count("v0-pop-g-prior-exponential-with-mean") > 0);

      if (has_g_prior_direct && has_g_prior_exponential) {
        std::cerr << "ERROR: --v0-pop-g-prior-exponential-with-mean is mutually exclusive with "
                  << "--v0-pop-g-prior-mu, --v0-pop-g-prior-scale, "
                  << "--v0-pop-growth-rate-min, and --v0-pop-growth-rate-max\n";
        std::exit(EXIT_FAILURE);
      }

      auto pop_g_prior_mu = opts["v0-pop-g-prior-mu"].as<double>() / 365.0;
      auto pop_g_prior_scale = opts["v0-pop-g-prior-scale"].as<double>() / 365.0;
      auto pop_g_min = -std::numeric_limits<double>::infinity();
      auto pop_g_max = +std::numeric_limits<double>::infinity();

      if (has_g_prior_exponential) {
        auto exp_mean = opts["v0-pop-g-prior-exponential-with-mean"].as<double>();
        if (exp_mean == 0.0) {
          std::cerr << "ERROR: --v0-pop-g-prior-exponential-with-mean must be nonzero\n";
          std::exit(EXIT_FAILURE);
        }
        pop_g_prior_mu = 0.0;
        pop_g_prior_scale = std::abs(exp_mean) / 365.0;
        if (exp_mean > 0.0) {
          pop_g_min = 0.0;
        } else {
          pop_g_max = 0.0;
        }
      } else {
        if (opts.count("v0-pop-growth-rate-min") > 0) {
          pop_g_min = opts["v0-pop-growth-rate-min"].as<double>() / 365.0;
        }
        if (opts.count("v0-pop-growth-rate-max") > 0) {
          pop_g_max = opts["v0-pop-growth-rate-max"].as<double>() / 365.0;
        }
      }

      if (pop_g_prior_scale <= 0.0) {
        std::cerr << "ERROR: --v0-pop-g-prior-scale must be positive, got "
                  << (pop_g_prior_scale * 365.0) << " e-foldings / year\n";
        std::exit(EXIT_FAILURE);
      }
      if (pop_g_min > pop_g_max) {
        std::cerr << "ERROR: --v0-pop-growth-rate-min (" << (pop_g_min * 365.0)
                  << ") must be <= --v0-pop-growth-rate-max (" << (pop_g_max * 365.0)
                  << ") e-foldings / year\n";
        std::exit(EXIT_FAILURE);
      }

      {
        auto window_per_year = opts["v0-pop-g-move-window"].as<double>();
        if (window_per_year < 0.0) {
          std::cerr << "ERROR: --v0-pop-g-move-window must be non-negative, got " << window_per_year << "\n";
          std::exit(EXIT_FAILURE);
        }
        if (window_per_year == 0.0) {
          window_per_year = 1.0;
          if (human_mode) {
            window_per_year = std::min(window_per_year, pop_g_prior_scale * 365.0);
            if (std::isfinite(pop_g_min) && std::isfinite(pop_g_max) && pop_g_max > pop_g_min) {
              window_per_year = std::min(window_per_year, (pop_g_max - pop_g_min) * 365.0);
            }
          }
        }
        run->set_pop_g_move_window(window_per_year / 365.0);

        // A window that changes the population size by a factor of e^1000 or more across the tree is
        // accepted in well under 1% of the proposals once the data constrain g, which is where the
        // chain stops mixing.  Only a warning: the window changes the speed of
        // the chain and not its target.
        if (human_mode and not fix_pop_growth_rate) {
          auto span_years = (calc_max_tip_time(run->tree()) - run->tree().at_root().t) / 365.0;
          if (span_years > 0.0 and window_per_year * span_years > 1000.0) {
            std::cerr << "WARNING: the growth-rate proposal window of " << window_per_year
                      << " e-foldings / year spans more than 1000 e-foldings over the initial tree depth of "
                      << span_years << " years, so proposals will rarely be accepted and the growth rate "
                      << "and population size will mix poorly.  Bound the growth rate with "
                      << "--v0-pop-growth-rate-min and --v0-pop-growth-rate-max, set "
                      << "--v0-pop-g-prior-scale, or set --v0-pop-g-move-window to about "
                      << (1.0 / span_years) << ".\n";
          }
        }
      }
      run->set_pop_g_prior_mu(pop_g_prior_mu);
      run->set_pop_g_prior_scale(pop_g_prior_scale);
      run->set_pop_g_min(pop_g_min);
      run->set_pop_g_max(pop_g_max);

      // If the user specified any g prior options but didn't explicitly set the initial growth rate,
      // override init_pop_growth_rate to the mean of the truncated prior.
      // This is only done when new options are explicitly specified, to preserve the behavior of
      // older versions of Delphy (which always used an initial value of 0, not the prior location mu).
      if ((has_g_prior_direct || has_g_prior_exponential) &&
          opts.count("v0-init-pop-growth-rate") == 0) {
        init_pop_growth_rate = truncated_laplace_mean(
            pop_g_prior_mu, pop_g_prior_scale, pop_g_min, pop_g_max);
      }

      if (init_pop_growth_rate < pop_g_min || init_pop_growth_rate > pop_g_max) {
        std::cerr << "ERROR: Initial growth rate " << (init_pop_growth_rate * 365.0)
                  << " e-foldings / year is outside bounds ["
                  << (pop_g_min * 365.0) << ", " << (pop_g_max * 365.0) << "]\n";
        std::exit(EXIT_FAILURE);
      }

      auto min_pop = opts["v0-pop-min-pop"].as<double>() * 365.0;
      run->set_pop_model(std::make_unique<Exp_pop_model>(t0, init_final_pop_size, init_pop_growth_rate, min_pop));
      run->set_final_pop_size_move_enabled(not fix_final_pop_size);
      run->set_pop_growth_rate_move_enabled(not fix_pop_growth_rate);

    } else if (opts["v0-pop-model"].as<std::string>() == "skygrid") {
      
      // Skygrid population model
      
      if (has_exp_pop_model_parameters) {
        std::cerr << "ERROR: Cannot specify parameters for 'exponential' model when pop-model is 'skygrid'\n";
        std::exit(EXIT_FAILURE);
      }

      auto skygrid_type = Skygrid_pop_model::Type::k_staircase;
      if (opts["v0-skygrid-type"].as<std::string>() == "staircase") {
        skygrid_type = Skygrid_pop_model::Type::k_staircase;
      } else if (opts["v0-skygrid-type"].as<std::string>() == "log-linear") {
        skygrid_type = Skygrid_pop_model::Type::k_log_linear;
      } else {
        std::cerr << "ERROR: Unknown Skygrid model type '" << opts["v0-skygrid-type"].as<std::string>() << "'\n";
        std::exit(EXIT_FAILURE);
      }

      auto num_parameters = opts["v0-skygrid-num-parameters"].as<int>();
      if (num_parameters < 2) {
        std::cerr << "ERROR: Number of Skygrid parameters must be at least 2\n";
        std::exit(EXIT_FAILURE);
      }

      auto has_first_knot_date = opts.count("v0-skygrid-first-knot-date") > 0;
      auto has_last_knot_date = opts.count("v0-skygrid-last-knot-date") > 0;
      auto has_cutoff = opts.count("v0-skygrid-cutoff") > 0;

      // Date options and cutoff are mutually exclusive
      if ((has_first_knot_date || has_last_knot_date) && has_cutoff) {
        std::cerr << "ERROR: --v0-skygrid-first-knot-date / --v0-skygrid-last-knot-date "
                  << "and --v0-skygrid-cutoff are mutually exclusive\n";
        std::exit(EXIT_FAILURE);
      }

      // Date options must be specified together
      if (has_first_knot_date != has_last_knot_date) {
        std::cerr << "ERROR: --v0-skygrid-first-knot-date and --v0-skygrid-last-knot-date "
                  << "must be specified together\n";
        std::exit(EXIT_FAILURE);
      }

      // Determine x_0 and x_M
      auto x_0 = 0.0;
      auto x_M = 0.0;
      if (has_first_knot_date) {
        x_0 = parse_iso_date(opts["v0-skygrid-first-knot-date"].as<std::string>());
        x_M = parse_iso_date(opts["v0-skygrid-last-knot-date"].as<std::string>());
      } else {
        auto skygrid_cutoff = opts["v0-skygrid-cutoff"].as<double>();  // years
        if (skygrid_cutoff < 0.0) {
          std::cerr << "ERROR: Skygrid cutoff must be positive\n";
          std::exit(EXIT_FAILURE);
        }
        x_M = t0;
        x_0 = t0 - skygrid_cutoff * 365.0;
      }

      if (x_0 >= x_M) {
        std::cerr << "ERROR: Skygrid first knot (" << to_iso_date(x_0) << ") must be before "
                  << "last knot (" << to_iso_date(x_M) << ")\n";
        std::exit(EXIT_FAILURE);
      }

      auto M = num_parameters - 1;
      auto x_k = std::vector<double>(M+1, 0.0);
      auto dt = (x_M - x_0) / M;
      for (auto k = 0; k <= M; ++k) {
        x_k[k] = x_0 + k * dt;
      }

      auto skygrid_tau = 0.0;
      auto infer_tau = opts["v0-skygrid-infer-prior-smoothness"].as<bool>();
      if (infer_tau) {
        // Infer a priori smoothness of log-population curve
        auto prior_alpha = opts["v0-skygrid-tau-prior-alpha"].as<double>();
        auto prior_beta = opts["v0-skygrid-tau-prior-beta"].as<double>();
        if (prior_alpha <= 0.0 || prior_beta <= 0.0) {
          std::cerr << "ERROR: Skygrid tau prior parameters must be positive\n";
          std::exit(EXIT_FAILURE);
        }
        
        run->set_skygrid_tau_prior_alpha(prior_alpha);
        run->set_skygrid_tau_prior_beta(prior_beta);

        skygrid_tau = prior_alpha / prior_beta;
        
      } else {
        // Fix a priori smoothness of log-population curve
        if (opts.count("v0-skygrid-tau") > 0 &&
            opts.count("v0-skygrid-prior-double-half-time") > 0) {
          std::cerr << "ERROR: Skygrid tau can be fixed either directly (--v0-skygrid-tau) or by specifying an effective 'double-half' time (--v0-skygrid-prior-double-half-time), but not both\n";
          std::exit(EXIT_FAILURE);
        }

        if (opts.count("v0-skygrid-tau") > 0) {
          skygrid_tau = opts["v0-skygrid-tau"].as<double>();
          if (skygrid_tau <= 0.0) {
            std::cerr << "ERROR: Skygrid tau parameter must be positive\n";
            std::exit(EXIT_FAILURE);
          }
          
        } else {
          if (human_mode && opts.count("v0-skygrid-prior-double-half-time") == 0) {
            std::cerr << "ERROR: --v0-human-mode with --v0-pop-model skygrid requires the prior "
                      << "smoothness of the log-population curve to be stated explicitly.  The "
                      << "default --v0-skygrid-prior-double-half-time of 0.0822 years (30 days) is "
                      << "a pathogen value: against a human knot spacing of thousands of years it "
                      << "makes the smoothness precision tau vanishingly small, which leaves the "
                      << "population curve essentially unsmoothed and free to overfit.  Supply "
                      << "--v0-skygrid-prior-double-half-time (a timescale comparable to the knot "
                      << "spacing, i.e. --v0-skygrid-cutoff divided by "
                      << "--v0-skygrid-num-parameters minus one), or --v0-skygrid-tau, or "
                      << "--v0-skygrid-infer-prior-smoothness true.\n";
            std::exit(EXIT_FAILURE);
          }
          auto double_half_time = opts["v0-skygrid-prior-double-half-time"].as<double>();
          double_half_time *= 365.0;  // convert to days
          if (double_half_time <= 0.0) {
            std::cerr << "ERROR: Skygrid prior 'double-half' time must be positive\n";
            std::exit(EXIT_FAILURE);
          }
          
          // Setting tau = 1 / (2 D dt), the prior for the log-population curve looks like
          // a 1D random walk with diffusion constant D.  Hence, on average, a starting
          // log-population changes after a time T by a root-mean-square deviation of
          // `sqrt(2 D T)`.  We parametrize D such that after T = double_half_time, the
          // rms deviation is log(2), i.e., population changes by up to a factor of ~2 in
          // the "double-half time" with 68% probability:
          //
          //   sqrt(2 D T) = log(2)  => D = log^2(2) / (2 T).
          //
          const auto D = std::pow(std::log(2.0), 2) / (2 * double_half_time);
          skygrid_tau = 1.0 / (2 * D * dt);
        }
      }

      // The smoothness precision also sets the HMC mass matrix.  `Run::skygrid_gammas_hmc_move`
      // uses m_k = tau [[k>0]] + tau [[k<M]] + c_k, where c_k counts the coalescences in interval k,
      // so when tau is far below 1 the mass of every interval that holds no coalescence collapses to
      // 2 tau, its inverse mass explodes, and the leapfrog integrator loses the step size it was
      // tuned for.  A tau small enough to leave the population curve unsmoothed therefore also
      // degrades the move that samples it.
      if (human_mode) {
        auto knot_stddev = 1.0 / std::sqrt(skygrid_tau);
        if (knot_stddev > std::log(1000.0)) {
          std::cerr << absl::StreamFormat(
              "WARNING: the Skygrid smoothness precision tau is %.4g, which lets the log-population "
              "curve change by a factor of %.3g between adjacent knots within one prior standard "
              "deviation.  At that precision the curve is effectively unsmoothed and free to "
              "overfit, and the HMC mass of any knot interval holding no coalescence falls to "
              "2 tau = %.4g, which costs the sampler its step size.  A smoothing timescale "
              "comparable to the knot spacing of %.4g years gives tau = %.4g.\n",
              skygrid_tau, std::exp(knot_stddev), 2 * skygrid_tau, dt / 365.0,
              1.0 / std::pow(std::log(2.0), 2));
        }
      }

      // Inverse-Gamma prior on N_bar = exp(gamma_bar)
      auto has_inv_nbar_prior_alpha_beta =
          (opts.count("v0-skygrid-inv-nbar-prior-alpha") > 0 || opts.count("v0-skygrid-inv-nbar-prior-beta") > 0);
      auto has_nbar_prior_mean_stddev =
          (opts.count("v0-skygrid-nbar-prior-mean") > 0 || opts.count("v0-skygrid-nbar-prior-stddev") > 0);

      if (has_inv_nbar_prior_alpha_beta && has_nbar_prior_mean_stddev) {
        std::cerr << "ERROR: --v0-skygrid-inv-nbar-prior-alpha/beta and --v0-skygrid-nbar-prior-mean/stddev "
                  << "are mutually exclusive\n";
        std::exit(EXIT_FAILURE);
      }

      auto inv_nbar_prior_alpha = 0.0;
      auto inv_nbar_prior_beta_days = 0.0;
      auto init_nbar_days = 3.0 * 365.0;  // Default: 3 years

      if (has_nbar_prior_mean_stddev) {
        if (opts.count("v0-skygrid-nbar-prior-mean") == 0 || opts.count("v0-skygrid-nbar-prior-stddev") == 0) {
          std::cerr << "ERROR: --v0-skygrid-nbar-prior-mean and --v0-skygrid-nbar-prior-stddev "
                    << "must be specified together\n";
          std::exit(EXIT_FAILURE);
        }
        auto mean = opts["v0-skygrid-nbar-prior-mean"].as<double>();   // years
        auto stddev = opts["v0-skygrid-nbar-prior-stddev"].as<double>(); // years
        if (mean <= 0.0) {
          std::cerr << "ERROR: --v0-skygrid-nbar-prior-mean must be positive, got " << mean << "\n";
          std::exit(EXIT_FAILURE);
        }
        if (stddev <= 0.0) {
          std::cerr << "ERROR: --v0-skygrid-nbar-prior-stddev must be positive, got " << stddev << "\n";
          std::exit(EXIT_FAILURE);
        }
        inv_nbar_prior_alpha = 2.0 + std::pow(mean / stddev, 2);
        auto beta_years = mean * (inv_nbar_prior_alpha - 1.0);
        inv_nbar_prior_beta_days = beta_years * 365.0;
        init_nbar_days = mean * 365.0;
      } else if (has_inv_nbar_prior_alpha_beta) {
        auto alpha = opts["v0-skygrid-inv-nbar-prior-alpha"].as<double>();
        auto beta_cli = opts["v0-skygrid-inv-nbar-prior-beta"].as<double>();
        if (alpha < 0.0) {
          std::cerr << "ERROR: --v0-skygrid-inv-nbar-prior-alpha must be non-negative, got " << alpha << "\n";
          std::exit(EXIT_FAILURE);
        }
        if (beta_cli < 0.0) {
          std::cerr << "ERROR: --v0-skygrid-inv-nbar-prior-beta must be non-negative, got " << beta_cli << "\n";
          std::exit(EXIT_FAILURE);
        }
        inv_nbar_prior_alpha = alpha;
        inv_nbar_prior_beta_days = beta_cli * 365.0;
        if (alpha > 1.0 && beta_cli > 0.0) {
          init_nbar_days = beta_cli / (alpha - 1.0) * 365.0;  // prior mean in days
        }
      }

      // At this point, skygrid_tau is set.  Sample a random trajectory with this
      // precision, then reset its mean to init_nbar_days (the `skygrid_gammas_zero_mode_gibbs_move`
      // Gibbs-samples the mean value of gamma, so we don't need to get this right at all here)
      auto gamma_k = std::vector<double>(M+1, 0.0);
      auto mean_gamma_k = 0.0;
      for (auto k = 1; k <= M; ++k) {
        gamma_k[k] = gamma_k[k-1] + absl::Gaussian(prng, 0.0, std::sqrt(1.0/skygrid_tau));
        mean_gamma_k += gamma_k[k];
      }
      mean_gamma_k /= M+1;
      for (auto k = 0; k <= M; ++k) {
        gamma_k[k] += (-mean_gamma_k) + std::log(init_nbar_days);
      }

      // Configure run
      run->set_pop_model(std::make_unique<Skygrid_pop_model>(std::move(x_k), std::move(gamma_k), skygrid_type));
      run->set_skygrid_tau(skygrid_tau);
      run->set_skygrid_tau_move_enabled(infer_tau);  // Prior configured above when infer_tau == true
      run->set_skygrid_inv_nbar_prior_alpha(inv_nbar_prior_alpha);
      run->set_skygrid_inv_nbar_prior_beta(inv_nbar_prior_beta_days);

      // Low-pop barrier
      if (opts["v0-skygrid-disable-low-pop-barrier"].as<bool>()) {
        run->set_skygrid_low_gamma_barrier_enabled(false);
      } else {
        auto low_pop_barrier_loc = opts["v0-skygrid-low-pop-barrier-loc"].as<double>();
        if (human_mode && opts.count("v0-skygrid-low-pop-barrier-loc") == 0) {
          std::cerr << "WARNING: the low-population barrier is at its default of 1 day, which was "
                    << "chosen to match the resolution of pathogen tip dates.  Every plausible "
                    << "human coalescent timescale is many orders of magnitude above it, so the "
                    << "barrier will never engage and will not catch a collapsing population "
                    << "curve.  Set --v0-skygrid-low-pop-barrier-loc to a floor that is meaningful "
                    << "on your time axis, in years, or pass "
                    << "--v0-skygrid-disable-low-pop-barrier true to state that you want none.\n";
        }
        low_pop_barrier_loc *= 365.0;  // convert to days
        auto low_gamma_barrier_loc = std::log(low_pop_barrier_loc);

        auto low_pop_barrier_scale = opts["v0-skygrid-low-pop-barrier-scale"].as<double>();
        auto low_gamma_barrier_scale = -std::log(1 - low_pop_barrier_scale);  // Convert to scale in gamma

        run->set_skygrid_low_gamma_barrier_enabled(true);
        run->set_skygrid_low_gamma_barrier_loc(low_gamma_barrier_loc);
        run->set_skygrid_low_gamma_barrier_scale(low_gamma_barrier_scale);
      }

    } else {
      std::cerr << "ERROR: Unknown population model '" << opts["v0-pop-model"].as<std::string>() << "'\n";
      std::exit(EXIT_FAILURE);
    }

    // Coalescent-prior grid.  This has to come after the population model is in place, because the
    // discretization excess depends on N(t) as well as on the genealogy.
    {
      auto grid_tolerance = opts["v0-coal-prior-grid-tolerance"].as<double>();
      auto grid_max_cells = opts["v0-coal-prior-max-cells"].as<int>();
      if (grid_tolerance < 0.0) {
        std::cerr << "ERROR: --v0-coal-prior-grid-tolerance must be non-negative, got "
                  << grid_tolerance << "\n";
        std::exit(EXIT_FAILURE);
      }
      if (grid_max_cells < target_coal_prior_cells) {
        std::cerr << "ERROR: --v0-coal-prior-max-cells (" << grid_max_cells << ") must be at least "
                  << "--v0-target-coal-prior-cells (" << target_coal_prior_cells << ")\n";
        std::exit(EXIT_FAILURE);
      }
      auto fixed_t_step_years = opts["v0-coal-prior-fixed-t-step"].as<double>();
      if (fixed_t_step_years < 0.0) {
        std::cerr << "ERROR: --v0-coal-prior-fixed-t-step must be non-negative, got "
                  << fixed_t_step_years << "\n";
        std::exit(EXIT_FAILURE);
      }
      if (fixed_t_step_years > 0.0 && grid_tolerance > 0.0) {
        std::cerr << "ERROR: --v0-coal-prior-fixed-t-step and --v0-coal-prior-grid-tolerance cannot be combined\n";
        std::exit(EXIT_FAILURE);
      }
      run->set_coal_prior_trace_retargets(opts["v0-coal-prior-trace-retargets"].as<bool>());
      run->set_coal_prior_grid_max_cells(grid_max_cells);
      run->set_coal_prior_grid_tolerance(grid_tolerance);

      auto report_degenerate_cells = [](const Coalescent_grid_excess& g) {
        if (g.num_degenerate_cells > 0) {
          std::cerr << absl::StreamFormat(
              "WARNING: the population size underflowed to zero in %d of %d coalescent-prior cells.  "
              "The sampler substitutes a floor of 1e-100 there and keeps going, so the reported "
              "discretization excess is a lower bound on the real one, and the coalescent prior in "
              "those cells is not the density it is meant to be.  This usually means the population "
              "model is free to run away: check the prior on its parameters.\n",
              g.num_degenerate_cells, g.num_cells);
        }
      };

      if (fixed_t_step_years > 0.0) {
        run->set_coal_prior_fixed_t_step(fixed_t_step_years * 365.0);
      }

      if (grid_tolerance > 0.0) {
        auto chosen = run->retarget_coal_prior_grid();
        std::cerr << absl::StreamFormat(
            "Coalescent prior grid: %d cells of %.6g years; discretization excess %.4g nats "
            "(tolerance %.4g)\n",
            chosen.num_cells, chosen.t_step / 365.0, chosen.excess, grid_tolerance);
        report_degenerate_cells(chosen);
        if (chosen.excess > grid_tolerance && chosen.num_degenerate_cells == 0) {
          std::cerr << absl::StreamFormat(
              "WARNING: the coalescent prior's discretization excess is %.4g nats at the cell "
              "cap of %d cells, above the requested tolerance of %.4g.  The sampled coalescent "
              "prior is this many nats above the exact Kingman density of the same genealogy, "
              "and the gap varies with the genealogy, so it biases node times.  Raise "
              "--v0-coal-prior-max-cells or reduce the number of samples.\n",
              chosen.excess, grid_max_cells, grid_tolerance);
        }
      }

      if (human_mode && grid_tolerance == 0.0) {
        auto measured = run->calc_coal_prior_grid_excess();
        std::cerr << absl::StreamFormat(
            "Coalescent prior grid: %d cells of %.6g years; discretization excess %.4g nats\n",
            measured.num_cells, measured.t_step / 365.0, measured.excess);
        report_degenerate_cells(measured);
        if (measured.excess > 1.0) {
          std::cerr << absl::StreamFormat(
              "WARNING: on the initial tree the coalescent prior exceeds the exact Kingman density "
              "of the same genealogy by %.4g nats because %d cells are too few for %d samples.  "
              "This excess depends on the genealogy, so it does not cancel in Metropolis-Hastings "
              "ratios and it biases node times.  Set --v0-coal-prior-grid-tolerance (for example "
              "0.1) to size the grid from the measured excess instead.\n",
              measured.excess, measured.num_cells, (std::ssize(run->tree()) + 1) / 2);
        }
      }
    }

    auto threshold_partition = human_mode && opts["v0-threshold-partition"].as<bool>();
    if (not threshold_partition && opts.count("v0-threshold-partition-params") > 0) {
      std::cerr << "ERROR: --v0-threshold-partition-params needs --v0-threshold-partition\n";
      std::exit(EXIT_FAILURE);
    }
    if (threshold_partition) {
      auto a = 0.5, b = 1.0, p = 0.02;
      auto r = 4;
      if (opts.count("v0-threshold-partition-params") > 0) {
        auto text = opts["v0-threshold-partition-params"].as<std::string>();
        auto fields = std::vector<std::string>(absl::StrSplit(text, ','));
        if (std::ssize(fields) != 4 || not absl::SimpleAtod(fields[0], &a) || not absl::SimpleAtod(fields[1], &b)
            || not absl::SimpleAtoi(fields[2], &r) || not absl::SimpleAtod(fields[3], &p)) {
          std::cerr << "ERROR: --v0-threshold-partition-params expects a,b,r,p (for example 0.5,1.0,4,0.02), got '"
                    << text << "'\n";
          std::exit(EXIT_FAILURE);
        }
      }
      if (run->topology_moves_enabled() && not run->calibrations().calibrations.empty()) {
        std::cerr << "ERROR: --v0-threshold-partition cannot be combined with age calibrations on a free topology\n";
        std::exit(EXIT_FAILURE);
      }
      try {
        run->set_threshold_partition_params(a, b, r, p);
        run->set_threshold_partitioning(true);
      } catch (const std::invalid_argument& e) {
        std::cerr << "ERROR: --v0-threshold-partition: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
      }
      run->set_num_parts(threads);
      std::cerr << absl::StreamFormat(
          "Threshold partitioning: ON (%d parts; m uniform on [%g, %g] x tips / parts, cut sizes [ceil(m/%d), m], "
          "single-part probability %g)\n", threads, a, b, r, p);
    }

    if (opts["v0-exact-coalescent-correction"].as<bool>()) {
      if (run->topology_moves_enabled() && threads != 1 && not threshold_partition) {
        std::cerr << "ERROR: --v0-exact-coalescent-correction with a free topology needs --v0-threads 1 or "
                  << "--v0-threshold-partition: with the default stencils the partition depends on the sampled "
                  << "topology.  Add --v0-threads 1, --v0-threshold-partition or --v0-fix-topology.\n";
        std::exit(EXIT_FAILURE);
      }
      try {
        run->set_exact_coalescent_correction(true);
      } catch (const std::invalid_argument& e) {
        std::cerr << "ERROR: --v0-exact-coalescent-correction: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
      }
      std::cerr << "Exact coalescent correction: ON (whole-batch Metropolis-Hastings step on log P_exact - log P_cells; "
                << "experimental, see doc/extended/exact_coalescent.md)\n";
    }

    if (opts["v0-global-scale-move"].as<bool>()) {
      if (not opts["v0-exact-coalescent-correction"].as<bool>()) {
        std::cerr << "ERROR: --v0-global-scale-move requires --v0-exact-coalescent-correction\n";
        std::exit(EXIT_FAILURE);
      }
      try {
        run->set_global_scale_move_window(opts["v0-global-scale-window"].as<double>());
        run->set_global_scale_moves_per_global_move(opts["v0-global-scale-moves-per-global-move"].as<int>());
        run->set_global_scale_move_enabled(true);
      } catch (const std::invalid_argument& e) {
        std::cerr << "ERROR: --v0-global-scale-move: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
      }
      std::cerr << "Joint time-scale move: ON (window " << opts["v0-global-scale-window"].as<double>() << ", "
                << opts["v0-global-scale-moves-per-global-move"].as<int>() << " proposal(s) per global step; experimental, see "
                << "doc/extended/global_scale_move.md)\n";
    } else if (opts.count("v0-global-scale-window") > 0 || opts.count("v0-global-scale-moves-per-global-move") > 0) {
      std::cerr << "ERROR: --v0-global-scale-window and --v0-global-scale-moves-per-global-move need --v0-global-scale-move\n";
      std::exit(EXIT_FAILURE);
    }

    if (opts["v0-history-gibbs"].as<bool>()) {
      try {
        run->set_history_gibbs_weight(opts["v0-history-gibbs-weight"].as<double>());
        run->set_history_gibbs_sites_per_move(opts["v0-history-gibbs-sites"].as<int>());
      } catch (const std::invalid_argument& e) {
        std::cerr << "ERROR: --v0-history-gibbs: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
      }
      std::cerr << "Single-site history Gibbs updates: ON (weight "
                << opts["v0-history-gibbs-weight"].as<double>() << ", "
                << opts["v0-history-gibbs-sites"].as<int>() << " site(s) per update; experimental, see "
                << "doc/extended/history_gibbs.md)\n";
    } else if (opts.count("v0-history-gibbs-weight") > 0 || opts.count("v0-history-gibbs-sites") > 0) {
      std::cerr << "ERROR: --v0-history-gibbs-weight and --v0-history-gibbs-sites need --v0-history-gibbs\n";
      std::exit(EXIT_FAILURE);
    }

    if (opts.count("v0-site-history-refreshes") > 0) {
      auto refreshes = opts["v0-site-history-refreshes"].as<int>();
      try {
        run->set_site_history_refreshes_per_global_move(refreshes);
      } catch (const std::invalid_argument& e) {
        std::cerr << "ERROR: --v0-site-history-refreshes: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
      }
      if (refreshes > 0) {
        std::cerr << "Whole-site history refresh: ON (" << refreshes
                  << " site(s) per global step; experimental, see doc/extended/history_gibbs.md)\n";
      }
    }

    if (opts["v0-marginal-node-time-move"].as<bool>()) {
      try {
        run->set_marginal_node_time_weight(opts["v0-marginal-node-time-weight"].as<double>());
        run->set_marginal_node_time_window(opts["v0-marginal-node-time-window"].as<double>());
      } catch (const std::invalid_argument& e) {
        std::cerr << "ERROR: --v0-marginal-node-time-move: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
      }
      std::cerr << "Marginal node-time move: ON (weight "
                << opts["v0-marginal-node-time-weight"].as<double>() << ", window "
                << opts["v0-marginal-node-time-window"].as<double>()
                << "; experimental, see doc/extended/history_gibbs.md)\n";
    } else if (opts.count("v0-marginal-node-time-weight") > 0
               || opts.count("v0-marginal-node-time-window") > 0) {
      std::cerr << "ERROR: --v0-marginal-node-time-weight and --v0-marginal-node-time-window need "
                << "--v0-marginal-node-time-move\n";
      std::exit(EXIT_FAILURE);
    }

    // Which version of BEAST to target for input XML and output log & trees files
    auto beast_version = opts["v0-out-beast-version"].as<std::string>();
    
    // Output BEAST input XML if needed
    if (opts.count("v0-out-beast-xml")) {
      auto beast_xml_filename = opts["v0-out-beast-xml"].as<std::string>();
      auto beast_xml_os = std::ofstream{beast_xml_filename};
      if (not beast_xml_os) {
        std::cerr << "ERROR: Could not write BEAST XML input file \'" << beast_xml_filename << "\'" << "\n";
        std::exit(EXIT_FAILURE);
      }

      // Very rough rule of thumb: 10 Delphy steps = 1 BEAST2 step
      export_beast_input(*run, beast_version, beast_xml_os, steps / 10, std::max(1L, log_every / 10), std::max(1L, tree_every / 10));
    }

    // Dry run => stop here
    if (opts.count("dry-run")) {
      std::exit(EXIT_SUCCESS);
    }
    
    return {
      .thread_pool = std::move(thread_pool),
      .run = std::move(run),
      .steps = steps,
      .log_filename = std::move(log_filename),
      .log_every = log_every,
      .trees_filename = std::move(trees_filename),
      .tree_every = tree_every,
      .trees_significant_digits = trees_significant_digits,
      .delphy_output_filename = std::move(delphy_output_filename),
      .delphy_output_metadata = std::move(delphy_output_metadata),
      .delphy_snapshot_every = delphy_snapshot_every,
      .beast_version = std::move(beast_version)
    };
    
  } catch (cxxopts::exceptions::exception& x) {
    std::cerr << "ERROR: " << x.what() << "\n" << options.help() << "\n";
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace delphy
