#include "run.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>

#include "distributions.h"
#include "tree_partitioning.h"
#include "phylo_tree_calc.h"
#include "dates.h"

#include "safe_gamma_math.h"

#include <numbers>

namespace delphy {

Run::Run(ctpl::thread_pool& thread_pool, std::mt19937 bitgen, uint32_t prng_seed, Phylo_tree tree,
         double init_coal_prior_t_step)
    : thread_pool_{&thread_pool},
      bitgen_{bitgen},
      prng_seed_{prng_seed},
      tree_{std::move(tree)},
      num_parts_{1},
      target_coal_prior_cells_{400},
      coal_prior_grid_max_cells_{4'000'000},
      coal_prior_grid_tolerance_{0.0},  // Default: no tolerance, keep the span/cells rule
      pop_model_{std::make_shared<Exp_pop_model>(calc_max_tip_time(tree_), 1000.0, 0.0, 1.0)},
      pop_inv_n0_prior_alpha_{0.0},  // Default: Jeffreys 1/x prior on n0
      pop_inv_n0_prior_beta_{0.0},
      pop_g_prior_mu_{0.001 / 365.0},       // BEAUti2 2.6.2 default, in e-foldings / day
      pop_g_prior_scale_{30.701135 / 365.0}, // BEAUti2 2.6.2 default, in e-foldings / day
      pop_g_min_{-std::numeric_limits<double>::infinity()},  // No lower bound by default
      pop_g_max_{+std::numeric_limits<double>::infinity()},   // No upper bound by default
      skygrid_tau_{1.0},  // Prior mean, Gill et al 2012 Eq. 15
      skygrid_tau_prior_alpha_{0.001},  // Prior, Gill et al 2012 discussion below Eq. 16
      skygrid_tau_prior_beta_{0.001},
      skygrid_low_gamma_barrier_loc_{std::log(1.0)},  // Pop sizes below 1 day are penalized
      skygrid_low_gamma_barrier_scale_{-std::log(0.70)},  // by 1 nat for every 30% drop
      skygrid_inv_nbar_prior_alpha_{0.0},  // Default: uniform prior on gamma_bar (= Jeffreys 1/N_bar)
      skygrid_inv_nbar_prior_beta_{0.0},
      alpha_{10.0},
      nu_(tree_.num_sites(), 1.0),
      mu_prior_alpha_{1.0},   // Default: uniform prior on mu
      mu_prior_beta_{0.0},
      evo_{make_single_partition_global_evo_model(tree_.num_sites())},
      coalescent_prior_{pop_model_, tree_.size(), calc_max_tip_time(tree_), init_coal_prior_t_step} {

  CHECK_GT(init_coal_prior_t_step, 0.0);

  // In Debug builds, record input tip sequences so that we can check that MCMC moves don't result
  // in incompatible real tip sequences
  save_original_sequences();
  
  // MCMC control
  step_ = 0;
  set_local_moves_per_global_move(-1);  // use default
  next_global_move_step_ = -1;
  next_partition_stencil_refresh_step_ = -1;
  
  // Initialize coalescent prior
  for (const auto& node : index_order_traversal(tree_)) {
    if (tree_.at(node).is_tip()) {
      coalescent_prior_.mark_as_tip(node);
    } else {
      coalescent_prior_.mark_as_coalescence(node);
    }
  }

  // One site partition covering every site, which is what every run without a configuration uses.
  site_partition_ = make_single_site_partition(observed_sites_of_tree(tree_));
  site_partition_rates_ = {1.0};

  // Pick a reasonable starting point for stationary frequencies of HKY model
  auto state_frequencies_of_ref_sequence = calc_cur_state_frequencies_of_ref_sequence();
  auto total_present_sites = static_cast<double>(estd::ranges::sum(state_frequencies_of_ref_sequence));
  auto est_pi = Seq_vector<double>{};
  for (auto a : k_all_real_seq_letters) {
    est_pi[a] = state_frequencies_of_ref_sequence[a] / total_present_sites;
  }
  auto crazy = false;
  for (auto a : k_all_real_seq_letters) {
    if (est_pi[a] < 0.01 || est_pi[a] > 0.99) {
      crazy = true;
    }
  }
  if (crazy) {
    // Just start with something uniform
    std::ranges::fill(est_pi, 0.25);
  }
  subst_model_ = parse_subst_model("HKY85");
  subst_model_.mu = 1e-3 / 365.0;
  subst_model_.pi_a = est_pi;

  repartition();

  invalidate_derived_quantities();
}

auto Run::refresh_partition_stencils() -> void {
  // Make 10 partition stencils very occasionally; during the run, we switch between these stencils randomly.
  // It's important for correctness that all partitions are equally likely to be chosen during a run.
  // If we change the set of partitions over which this choice takes place *slowly* (~ less than once
  // per effective sample), then it doesn't matter *how* those partitions are chosen.  But if we do it
  // more quickly, the sampling becomes biased.

  if (partition_stencils_valid_ && step_ < next_partition_stencil_refresh_step_) {
    return;
  }

  //std::cerr << "\n\n*****\nREFRESHING PARTITION STENCILS\n*****\n\n";

  partition_stencils_.clear();
  for (auto i = 0; i != 10; ++i) {
    partition_stencils_.push_back(generate_random_partition_stencil(tree_, num_parts_, bitgen_));
  }

  partition_stencils_valid_ = true;
  next_partition_stencil_refresh_step_ = step_ + 200 * local_moves_per_global_move_;
  force_repartition_ = true;
}

auto Run::set_threshold_partitioning(bool enabled) -> void {
  if (enabled && topology_moves_enabled_ && not calibrations_.empty()) {
    throw std::invalid_argument(
        "Threshold partitioning cannot be combined with age calibrations while the topology is sampled");
  }
  threshold_partitioning_ = enabled;
  force_repartition_ = true;
}

auto Run::set_threshold_partition_params(
    double min_fraction, double max_fraction, int min_ratio, double single_part_probability) -> void {
  if (not (min_fraction > 0.0 && min_fraction <= max_fraction && std::isfinite(max_fraction))) {
    throw std::invalid_argument(absl::StrFormat(
        "The threshold partition fractions must satisfy 0 < min <= max (got %g and %g)", min_fraction, max_fraction));
  }
  if (min_ratio < 1) {
    throw std::invalid_argument(absl::StrFormat(
        "The threshold partition size ratio must be at least 1 (got %d)", min_ratio));
  }
  if (not (single_part_probability >= 0.0 && single_part_probability <= 1.0)) {
    throw std::invalid_argument(absl::StrFormat(
        "The single-part probability must lie in [0, 1] (got %g)", single_part_probability));
  }
  threshold_partition_min_fraction_ = min_fraction;
  threshold_partition_max_fraction_ = max_fraction;
  threshold_partition_min_ratio_ = min_ratio;
  threshold_partition_single_part_probability_ = single_part_probability;
  force_repartition_ = true;
}

// The draw uses `bitgen_` only, never the state, so q(m | x) = q(m) and the auxiliary variable drops out of every
// acceptance ratio.
auto Run::draw_threshold_partition_stencil() -> std::vector<Partition_part_info> {
  ++threshold_partitions_drawn_;
  threshold_max_cut_tips_ = 0;
  threshold_min_cut_tips_ = 0;
  if (absl::Bernoulli(bitgen_, threshold_partition_single_part_probability_)) {
    ++threshold_partitions_single_part_;
    return {};
  }
  auto tips_per_part = static_cast<double>((std::ssize(tree_) + 1) / 2) / num_parts_;
  auto lo = std::max(2, static_cast<int>(std::ceil(threshold_partition_min_fraction_ * tips_per_part)));
  auto hi = std::max(lo, static_cast<int>(std::floor(threshold_partition_max_fraction_ * tips_per_part)));
  auto m = absl::Uniform(absl::IntervalClosed, bitgen_, lo, hi);
  threshold_max_cut_tips_ = m;
  threshold_min_cut_tips_ = std::max(2, (m + threshold_partition_min_ratio_ - 1) / threshold_partition_min_ratio_);
  return threshold_partition_stencil(tree_, threshold_max_cut_tips_, threshold_min_cut_tips_);
}

// Even when no node is a cut point, the root part must keep it that way: the partition is a function of m and the
// topology, and the batch kernel has to see the same one at both ends.
auto Run::constrain_threshold_root_part() -> void {
  if (threshold_max_cut_tips_ == 0) { return; }
  auto tips = calc_clade_tip_counts(tree_);
  const auto& root_part = tree_partition_.parts()[tree_partition_.root_part_index()];
  auto constraint = Threshold_partition_constraint{
    .max_cut_tips = threshold_max_cut_tips_,
    .min_cut_tips = threshold_min_cut_tips_,
    .tip_weights = Node_vector<int>(std::ssize(root_part), 0)};
  for (auto node : index_order_traversal(root_part)) {
    if (root_part.at(node).is_tip()) { constraint.tip_weights[node] = tips[root_part.at(node).orig_tree_index()]; }
  }
  subruns_[tree_partition_.root_part_index()].set_threshold_partition_constraint(std::move(constraint));
}

// Recomputes the threshold partition of the reassembled tree and compares it with the one the batch used.
auto Run::check_threshold_partition_kept() const -> void {
  if (not threshold_partitioning_active() || threshold_max_cut_tips_ == 0) { return; }
  auto expected = threshold_partition_stencil(tree_, threshold_max_cut_tips_, threshold_min_cut_tips_);
  auto expected_cuts = std::vector<Node_index>{};
  for (const auto& info : expected) { expected_cuts.push_back(info.cut_point); }
  auto actual_cuts = std::vector<Node_index>{};
  for (auto i = 0; i != std::ssize(tree_partition_.parts()); ++i) {
    if (i != tree_partition_.root_part_index()) { actual_cuts.push_back(tree_partition_.parts()[i].info().cut_point); }
  }
  std::ranges::sort(expected_cuts);
  std::ranges::sort(actual_cuts);
  CHECK(expected_cuts == actual_cuts) << "A local batch changed the threshold partition";
}

auto Run::repartition() -> void {
  if (threshold_partitioning_active()) {
    tree_partition_ = partition_tree(tree_, draw_threshold_partition_stencil());
    threshold_partition_parts_drawn_ += std::ssize(tree_partition_.parts());
    while (std::ssize(subbitgens_) < std::ssize(tree_partition_.parts())) {
      subbitgens_.emplace_back(bitgen_());
    }
  } else {
    // Ensure we have enough bitgens around
    while (std::ssize(subbitgens_) < num_parts_) {
      subbitgens_.emplace_back(bitgen_());
    }

    // Ensure partition stencils are in up to date
    refresh_partition_stencils();

    //std::cerr << "*** REPARTITION ***" << std::endl;
    auto partition_stencils_index =
        std::uniform_int_distribution<int>{0, int(std::ssize(partition_stencils_)-1)}(bitgen_);
    auto& partition_stencil = partition_stencils_.at(partition_stencils_index);
    tree_partition_ = partition_tree(tree_, partition_stencil);
  }
  check_free_topology_calibrations();

  // Before we propagate the reference sequence to subruns, normalize
  CHECK(tree_.at_root().missations.from_states.empty());
  normalize_root();

  assert_phylo_tree_integrity(tree_, paranoid_);
  
  // Make subruns
  subruns_.clear();
  for (auto& partition_part : tree_partition_.parts()) {
    auto subroot = partition_part.info().cut_point;
    auto subroot_seq_view = view_of_sequence_at(tree_, subroot);
    const auto& ref_seq = tree_.ref_sequence;

    auto subtree = Phylo_tree{partition_part.size()};
    subtree.ref_sequence = ref_seq;
    copy_topology(partition_part, subtree);

    // Missations in above subroot logically come before mutations above subroot,
    // so all their from_states match the reference sequence
    auto root_missation_map = Missation_map<>{};
    root_missation_map.intervals = reconstruct_missing_sites_at(tree_, subroot);
    root_missation_map.from_states.clear();
    
    auto root_mutations = Mutation_list<>{};
    for (const auto& [l, b] : subroot_seq_view.deltas()) {
      if (not root_missation_map.contains(l)) {
        root_mutations.push_back(Mutation{ref_seq[l], l, b, -std::numeric_limits<double>::max()});
      }
    }
    sort_mutations(root_mutations);

    const auto empty_mutation_list = Mutation_list<>{};
    auto original_sequences = Node_vector<Sequence_overlay>{};
    auto original_missing_sites_all = Node_vector<Interval_set<>>{};
    for (const auto& partition_node : index_order_traversal(partition_part)) {
      auto node = partition_part.at(partition_node).orig_tree_index();
      auto subtree_node = partition_node;
      const auto& mutation_list = (node == subroot) ? root_mutations : tree_.at(node).mutations;
      const auto& missation_map = (node == subroot) ? root_missation_map : tree_.at(node).missations;
      subtree.at(subtree_node).name = tree_.at(node).name;
      subtree.at(subtree_node).t = tree_.at(node).t;
      if (subtree.at(subtree_node).is_tip() && not tree_.at(node).is_tip()) {
        // "tip" in subtree is really just a frozen inner node, which we shouldn't move
        subtree.at(subtree_node).t_min = tree_.at(node).t;
        subtree.at(subtree_node).t_max = tree_.at(node).t;
      } else {
        subtree.at(subtree_node).t_min = tree_.at(node).t_min;
        subtree.at(subtree_node).t_max = tree_.at(node).t_max;
      }
      subtree.at(subtree_node).mutations = mutation_list;
      subtree.at(subtree_node).missations = missation_map;
      if (estd::is_debug_enabled) {
        original_sequences.push_back(original_sequences_[node]);
        original_missing_sites_all.push_back(original_missing_sites_all_[node]);
      }
    }

    assert_phylo_tree_integrity(subtree, paranoid_);
    auto& subrun = subruns_.emplace_back(
        absl::BitGenRef{subbitgens_[subruns_.size()]}, std::move(subtree), subroot == tree_.root, cur_shared_evo());
    if (estd::is_debug_enabled) {
      subrun.set_original_sequences(std::move(original_sequences), std::move(original_missing_sites_all));
    }
  }

  if (threshold_partitioning_active()) { constrain_threshold_root_part(); }

  // Start very_scalable_coalescent_prior from scratch
  reset_very_scalable_coalescent_parts();

  assign_calibrations_to_subruns();

  force_repartition_ = false;
}

// exp(gamma) overflows above gamma ~ 710 and underflows to zero below gamma ~ -745, and a cell or an
// event interval whose population is infinite or zero breaks both the cell prior and the exact
// density.  The bound leaves a decade of margin, so an interval of any width down to 1e-6 time units
// keeps a strictly positive, finite integral.  Moves that would leave this range abandon the
// proposal instead, which restricts the target to representable population curves exactly as the
// low-population barrier restricts it from below; see
// doc/human_uniparental/49_skygrid_acceptance_task31.md (development record, not in this repository), section 2.1.
constexpr auto k_max_representable_gamma = 700.0;

auto skygrid_gammas_are_representable(std::span<const double> gamma) -> bool {
  for (auto g : gamma) {
    if (not std::isfinite(g) || std::abs(g) > k_max_representable_gamma) { return false; }
  }
  return true;
}

auto pop_model_supports_exact_correction(const Pop_model& pop_model) -> bool {
  return typeid(pop_model) == typeid(Exp_pop_model)
      || typeid(pop_model) == typeid(Skygrid_pop_model)
      || typeid(pop_model) == typeid(Const_pop_model);
}

auto Run::set_exact_coalescent_correction(bool enabled) -> void {
  if (enabled) {
    auto refuse = [](const std::string& why) {
      throw std::invalid_argument("The exact coalescent correction is not supported here: " + why);
    };
    if (topology_moves_enabled_) {
      if (auto why = free_topology_exact_correction_problem(); not why.empty()) { refuse(why); }
    }
    const Pop_model& raw_pop_model = *pop_model_;
    if (not pop_model_supports_exact_correction(raw_pop_model)) {
      refuse("only the exponential, Skygrid and constant population models are supported");
    }
    if (not (coal_prior_fixed_t_step_ > 0.0)) {
      refuse("the coalescent prior grid must be pinned (--v0-coal-prior-fixed-t-step)");
    }
    if (mpox_hack_enabled_) { refuse("the mpox hack is not validated with the correction"); }
    // Sampling dates may differ between tips -- the exact Kingman density of a heterochronous
    // genealogy is the same formula with the tips as lineage-count changes, and the normalization
    // argument behind the correction does not use a shared date.  What is still refused is date
    // *uncertainty*: an uncertain date makes the tip time a sampled variable with its own proposal
    // and prior, which the whole-batch correction has not been validated against.
    for (const auto& node : index_order_traversal(tree_)) {
      if (tree_.at(node).is_tip() && tree_.at(node).t_min != tree_.at(node).t_max) {
        refuse("every sample must have an exact date, without date uncertainty");
      }
    }
  }
  exact_coalescent_correction_ = enabled;
  exact_batches_proposed_ = 0;
  exact_batches_accepted_ = 0;
  invalidate_derived_quantities();
}

// With a single tree partition part the stencil has no effect on the batch kernel, which is then reversible with
// respect to the cell-prior target whatever the topology.  Age calibrations are part of that target (their term
// enters every local MH ratio) and not of the correction, which depends on node times only; see
// doc/extended/exact_free_topology.md.
auto Run::free_topology_exact_correction_problem() const -> std::string {
  if (num_parts_ != 1 && not threshold_partitioning_) {
    return absl::StrFormat("a free topology needs a single tree partition part or threshold partitioning (got %d "
                           "parts; run with --v0-threads 1 or --v0-threshold-partition)", num_parts_);
  }
  return {};
}

auto Run::set_topology_moves_enabled(bool topology_moves_enabled) -> void {
  if (topology_moves_enabled && global_scale_move_enabled_) {
    throw std::invalid_argument(
        "Topology-changing moves cannot be enabled while the joint time-scale move is on: that move needs a fixed "
        "topology.  Disable the joint time-scale move first.");
  }
  if (topology_moves_enabled && exact_coalescent_correction_) {
    if (auto why = free_topology_exact_correction_problem(); not why.empty()) {
      throw std::invalid_argument(
          "Topology-changing moves cannot be enabled while the exact coalescent correction is on: " + why);
    }
  }
  if (topology_moves_enabled && threshold_partitioning_ && not calibrations_.empty()) {
    throw std::invalid_argument(
        "Topology-changing moves cannot be enabled with age calibrations while threshold partitioning is on");
  }
  topology_moves_enabled_ = topology_moves_enabled;
  // The subruns' calibrations carry the sample sets only while topology moves are enabled
  if (topology_moves_enabled && not calibrations_.empty()) { force_repartition_ = true; }
}

// With topology moves enabled, a calibration's target is re-identified from its sample set after every topology
// proposal, the sample set is constrained to stay monophyletic unless the calibration opts out (MRCA-age semantics),
// and the calibration term enters the MH ratio of every local move.  That is only implemented for a single tree
// partition part; see check_free_topology_calibrations.
auto Run::set_calibrations(Calibration_set calibrations) -> void {
  if (threshold_partitioning_ && topology_moves_enabled_ && not calibrations.empty()) {
    throw std::invalid_argument(
        "Age calibrations cannot be combined with threshold partitioning while the topology is sampled");
  }
  auto nodes = calibrations.empty()
      ? std::vector<Node_index>{}
      : bind_calibrations(calibrations, tree_);
  calibrations_ = std::move(calibrations);
  calibration_nodes_ = std::move(nodes);
  assign_calibrations_to_subruns();
  invalidate_derived_quantities();
}

// Gives every calibrated node to the one part that can move it, and sums the contribution of the
// calibrated nodes that no part can move during a local batch.  See
// doc/extended/calibration.md, section 6, for why this is exact.
auto Run::check_free_topology_calibrations() const -> void {
  if (topology_moves_enabled_ && exact_coalescent_correction_ && std::ssize(tree_partition_.parts()) != 1
      && not threshold_partitioning_active()) {
    throw std::invalid_argument(absl::StrFormat(
        "The exact coalescent correction with a free topology is only supported with a single tree partition part "
        "(got %d; run with --v0-threads 1).  With several parts the partition stencils depend on the sampled "
        "topology, which the correction does not cover.",
        std::ssize(tree_partition_.parts())));
  }
  if (not topology_moves_enabled_ || calibrations_.empty()) { return; }
  if (std::ssize(tree_partition_.parts()) != 1) {
    throw std::invalid_argument(absl::StrFormat(
        "Age calibrations with a free topology are only supported with a single tree partition part (got %d; "
        "run with --v0-threads 1).  With several parts a topology move inside one part could take a cut-point "
        "subtree into or out of a calibrated clade, which the parts cannot see.",
        std::ssize(tree_partition_.parts())));
  }
}

auto Run::assign_calibrations_to_subruns() -> void {
  if (calibrations_.empty()) {
    frozen_calibration_log_prior_ = 0.0;
    for (auto& subrun : subruns_) { subrun.set_calibrations({}); }
    return;
  }

  // Re-resolve the taxon sets against the current tree: the binding is defined by sample names, and a
  // silent drift of the target node must never pass unnoticed.
  calibration_nodes_ = bind_calibrations_ignoring_times(calibrations_, tree_);

  auto num_parts = std::ssize(tree_partition_.parts());
  auto local = std::vector<std::vector<Local_calibration>>(num_parts);
  auto assigned = std::vector<bool>(std::ssize(calibrations_.calibrations), false);

  for (auto p = 0; p != num_parts; ++p) {
    const auto& partition_part = tree_partition_.parts()[p];
    const auto& subtree = subruns_[p].tree();
    auto global_to_local = absl::flat_hash_map<Node_index, Node_index>{};
    for (const auto& subnode : index_order_traversal(subtree)) {
      global_to_local.emplace(partition_part.at(subnode).orig_tree_index(), subnode);
    }
    for (auto i = 0; i != std::ssize(calibration_nodes_); ++i) {
      auto it = global_to_local.find(calibration_nodes_[i]);
      if (it == global_to_local.end()) { continue; }
      auto local_node = it->second;
      // Movable in this part: an inner node that is either not the part's subroot, or is the run root.
      auto movable = subtree.at(local_node).is_inner_node()
          && (local_node != subtree.root || subruns_[p].includes_run_root());
      if (not movable) { continue; }
      CHECK(not assigned[i]) << "Calibration '" << calibrations_.calibrations[i].name
                             << "' is movable in more than one partition part";
      assigned[i] = true;
      const auto& c = calibrations_.calibrations[i];
      auto lc = Local_calibration{local_node, c.reference_t, c.prior};
      lc.all_tips = c.all_tips;
      lc.require_monophyly = c.require_monophyly;
      if (topology_moves_enabled_ && num_parts == 1 && not c.all_tips) {
        auto names = absl::flat_hash_set<std::string>(c.taxa.begin(), c.taxa.end());
        for (const auto& subnode : index_order_traversal(subtree)) {
          if (subtree.at(subnode).is_tip() && names.contains(tree_.at(partition_part.at(subnode).orig_tree_index()).name)) {
            lc.tips.push_back(subnode);
          }
        }
        CHECK_EQ(std::ssize(lc.tips), std::ssize(c.taxa));
      }
      local[p].push_back(std::move(lc));
    }
  }

  frozen_calibration_log_prior_ = 0.0;
  for (auto i = 0; i != std::ssize(calibration_nodes_); ++i) {
    if (assigned[i]) { continue; }
    frozen_calibration_log_prior_ +=
        calibrations_.calibrations[i].log_density_at(tree_.at(calibration_nodes_[i]).t);
  }

  for (auto p = 0; p != num_parts; ++p) {
    subruns_[p].set_calibrations(std::move(local[p]));
  }
}

auto Run::reassemble() -> void {
  CHECK_EQ(subruns_.size(), tree_partition_.parts().size());
  auto num_partitions = std::ssize(tree_partition_.parts());

  for (auto i = 0; i != num_partitions; ++i) {
    const auto& partition_part = tree_partition_.parts()[i];
    auto& subrun = subruns_[i];

    for (auto k = 0; k != k_num_move_kinds; ++k) {
      move_counts_[k].proposed += subrun.move_counts()[k].proposed;
      move_counts_[k].accepted += subrun.move_counts()[k].accepted;
      move_counts_[k].blocked_by_fixed_topology += subrun.move_counts()[k].blocked_by_fixed_topology;
      move_counts_[k].blocked_by_partition += subrun.move_counts()[k].blocked_by_partition;
    }
    subrun.reset_move_counts();
    history_gibbs_stats_ += subrun.history_gibbs_stats();
    subrun.reset_history_gibbs_stats();

    assert_phylo_tree_integrity(subrun.tree());
    CHECK(subrun.tree().at_root().missations.from_states.empty());

    // Transpose all the data in the subrun onto the main tree
    const auto& subtree = subrun.tree();
    for (const auto& subnode : post_order_traversal(subtree)) {
      auto node = partition_part.at(subnode).orig_tree_index();
      tree_.at(node).t = subtree.at(subnode).t;
      if (tree_.at(node).is_inner_node()) {
        coalescent_prior_.displace_coalescence(node, tree_.at(node).t);
      } else {
        coalescent_prior_.displace_tip(node, tree_.at(node).t);
      }
      if (subnode != subtree.root) {
        tree_.at(node).mutations = subtree.at(subnode).mutations;
        tree_.at(node).missations = subtree.at(subnode).missations;
      }

      // Transpose topology
      // Note: the test is not `tree_.at(node).is_inner_node()` to only handle cut points that are subroots, not subtips
      if (subtree.at(subnode).is_inner_node()) {
        auto subleft = subtree.at(subnode).left_child();
        auto subright = subtree.at(subnode).right_child();
        auto left = partition_part.at(subleft).orig_tree_index();
        auto right = partition_part.at(subright).orig_tree_index();
        tree_.at(node).children = {left, right};
        tree_.at(left).parent = node;
        tree_.at(right).parent = node;

        if (estd::is_debug_enabled) {
          CHECK_LE(tree_.at(node).t, tree_.at(left).t);
          CHECK_LE(tree_.at(node).t, tree_.at(right).t);
        }
      }
    }

    // If the subrun includes the root node, transpose the root index & deltas from ref sequence back
    if (subrun.includes_run_root()) {
      const auto& subtree = subrun.tree();
      auto subroot = subtree.root;
      auto subroot_in_main = partition_part.at(subroot).orig_tree_index();
      tree_.root = subroot_in_main;
      tree_.at_root().parent = k_no_node;
      tree_.at_root().mutations = subtree.at(subroot).mutations;
      tree_.at_root().missations = subtree.at(subroot).missations;
      DCHECK(tree_.ref_sequence == subtree.ref_sequence);  // subrun ref sequence unchanged
    }
  }

  assert_phylo_tree_integrity(tree_);

  // The local moves may have changed which nodes are the calibrated MRCAs.  Re-resolving the sample sets also
  // re-checks that every one of them is still monophyletic.
  if (topology_moves_enabled_ && not calibrations_.empty()) {
    calibration_nodes_ = bind_calibrations_ignoring_times(calibrations_, tree_);
  }

  invalidate_derived_quantities_same_reference();  // the parts never write the reference sequence of `tree_`
  check_threshold_partition_kept();
  check_global_and_local_totals_match();
}

auto Run::normalize_root() -> void {
  // Merge any mutations above the root into the reference sequence
  if (tree_.at_root().mutations.empty()) { return; }  // the reference sequence is already the root sequence
  // The cached per-partition state counts of the reference sequence follow the rewrite exactly
  auto& c = ref_site_stats_;
  auto counts_follow = ref_site_stats_inputs_unchanged();
  for (const auto& m : tree_.at_root().mutations) {
    --state_frequencies_of_ref_sequence_[m.from];
    ++state_frequencies_of_ref_sequence_[m.to];
    if (counts_follow) {
      auto beta = evo_.partition_for_site[m.site];
      --c.state_frequencies[beta][m.from];
      ++c.state_frequencies[beta][m.to];
    }
  }
  rereference_to_root_sequence(tree_);
  ++ref_version_;
  if (counts_follow) {
    c.ref_version = ref_version_;
    c.ntwiddle_valid = false;
  }
}

auto Run::push_global_params_to_subruns() -> void {
  // Drop the parts' references first, so that a changed model or table can be rewritten in place instead of copied
  for (auto& subrun : subruns_) { subrun.set_evo(std::shared_ptr<const Global_evo_model>{}); }
  auto shared_evo = cur_shared_evo();
  auto ref_cum_Q_l = cur_ref_cum_Q_l();
  const auto& state_frequencies = cur_ref_site_stats().state_frequencies;
  for (auto& subrun : subruns_) {
    subrun.set_t_max_tip(t_max_tip_);
    subrun.set_history_gibbs_weight(history_gibbs_weight_);
    subrun.set_history_gibbs_sites_per_move(history_gibbs_sites_per_move_);
    subrun.set_marginal_node_time_weight(marginal_node_time_weight_);
    subrun.set_marginal_node_time_window(marginal_node_time_window_);
    subrun.set_evo(shared_evo);
    DCHECK(subrun.tree().ref_sequence == tree_.ref_sequence);  // the tables below describe this reference
    subrun.set_ref_site_tables(ref_cum_Q_l, state_frequencies);
    subrun.set_only_displacing_inner_nodes(only_displacing_inner_nodes_);
    subrun.set_topology_moves_enabled(topology_moves_enabled_);
    subrun.set_skip_fixed_tip_moves(skip_fixed_tip_moves_);
    subrun.set_root_displace_scale_factor(root_displace_scale_factor_);
  }
  reset_very_scalable_coalescent_parts();  // TODO: Should only push kbars and k(t)?
}

auto Run::reset_very_scalable_coalescent_parts() -> void {
  // Make very scalable coalescent
  auto phylo_subtrees = std::vector<const Phylo_tree*>{};
  for (const auto& subrun : subruns_) {
    phylo_subtrees.push_back(&subrun.tree());
  }
  coalescent_prior_parts_ = make_very_scalable_coalescent_prior_parts(
      phylo_subtrees,
      tree_partition_.root_part_index(),
      pop_model_,
      subbitgens_,
      coalescent_prior_.t_step()  // NOTE: This picks up any changes to t_step midway
  );
  for (auto i = 0; i != std::ssize(phylo_subtrees); ++i) {
    subruns_[i].set_coalescent_prior_part(&coalescent_prior_parts_[i]);
  }
}

auto Run::set_coalescent_prior_t_step(double t_step) -> void {
  //std::cerr << absl::StreamFormat("\n\n*****\n\nResetting coalescent prior resolution to %g days\n\n*******\n\n", t_step);
  auto old_t_step = coalescent_prior_.t_step();
  auto log_before = coal_prior_trace_retargets_ ? calc_cur_log_coalescent_prior() : 0.0;
  coalescent_prior_.reset(t_step);
  log_coalescent_prior_ = calc_cur_log_coalescent_prior();
  if (exact_coalescent_correction_) { invalidate_derived_quantities(); }
  if (coal_prior_trace_retargets_) {
    std::cerr << absl::StreamFormat(
        "RETARGET step=%d t_step_years %.6g -> %.6g span_years=%.6g log_coalescent_prior %.6f -> %.6f delta=%+.6f\n",
        step_, old_t_step / 365.0, t_step / 365.0, (t_max_tip() - tree_.at_root().t) / 365.0,
        log_before, log_coalescent_prior_, log_coalescent_prior_ - log_before);
  }
  // Very scalable coalescent should pick up new t_step in reset_very_scalable_coalescent_parts()
}

auto Run::calc_coal_prior_grid_excess() const -> Coalescent_grid_excess {
  return calc_coalescent_grid_excess(
      coalescent_event_times_of(tree_), *pop_model_, t_max_tip(), coalescent_prior_.t_step());
}

auto Run::retarget_coal_prior_grid() -> Coalescent_grid_excess {
  if (exact_coalescent_correction_) {
    throw std::invalid_argument(
        "The coalescent grid cannot be retargeted from the sampled state while the exact coalescent correction is on; "
        "the grid must stay pinned");
  }
  auto events = coalescent_event_times_of(tree_);
  auto t_ref = t_max_tip();
  auto span = t_ref - tree_.at_root().t;
  // A tree that has collapsed to a point has no time axis to discretize; leave the grid alone
  // rather than divide by zero.
  if (not (span > 0.0)) { return calc_coal_prior_grid_excess(); }

  auto chosen = coal_prior_grid_tolerance_ > 0.0
      ? choose_coalescent_grid_t_step(events, *pop_model_, t_ref, coal_prior_grid_tolerance_,
                                      target_coal_prior_cells_, coal_prior_grid_max_cells_)
      : calc_coalescent_grid_excess(events, *pop_model_, t_ref, span / target_coal_prior_cells_);

  set_coalescent_prior_t_step(chosen.t_step);
  return chosen;
}

auto Run::recalc_derived_quantities() const -> void {
  // Derived quantities.  The statistics of the reference sequence that need a pass over every site come from
  // `ref_site_stats_`, which recomputes them only when one of their inputs has changed (see run.h).
  derive_evo();
  auto state_frequencies_per_partition = cur_ref_site_stats().state_frequencies;
  log_G_ = ref_cum_Q_l_is_current()
      ? calc_log_G_with(state_frequencies_per_partition, *ref_cum_Q_l_)
      : calc_log_G_with(state_frequencies_per_partition, calc_cum_Q_l_for_sequence(tree_.ref_sequence, evo_));
  Ttwiddle_beta_a_ = calc_cur_Ttwiddle_beta_a();
  num_muts_ = calc_cur_num_muts();
  num_muts_ab_ = calc_cur_num_muts_ab();
  log_coalescent_prior_ = calc_cur_log_coalescent_prior();
  log_coal_exact_correction_ = exact_coalescent_correction_
      ? calc_cur_log_coalescent_exact() - log_coalescent_prior_
      : 0.0;
  log_other_priors_ = calc_cur_log_other_priors();
  log_calibration_prior_ = calc_cur_log_calibration_prior();
  state_frequencies_of_ref_sequence_ = sum_over_partitions(state_frequencies_per_partition);
  t_max_tip_ = calc_max_tip_time(tree_);
  last_revalidation_step_ = step_;
}

auto Run::check_derived_quantities() const -> void {
  if (estd::is_debug_enabled) {
    // We maintain these incrementally except when we force a recalculation
    { auto expected = calc_cur_log_G();
      CHECK(std::abs(log_G_ - expected) < 1e-6)
          << log_G_ << " != " << expected; }
    { auto expected = calc_Ttwiddle_beta_a(tree_, evo_);  // from scratch, not from `ref_site_stats_`
      CHECK((taxicab_distance(Ttwiddle_beta_a_, expected)
             / static_cast<double>(tree_.num_sites())) < 1e-8)
          << absl::StrJoin(Ttwiddle_beta_a_, ",", absl::StreamFormatter())
          << " != "
          << absl::StrJoin(expected, ",", absl::StreamFormatter()); }
    CHECK_EQ(num_muts_, calc_cur_num_muts());
    CHECK_EQ(num_muts_ab_, calc_cur_num_muts_ab());
    // A cached statistic of the reference sequence must equal its recomputation from scratch, bit for bit
    if (ref_site_stats_inputs_unchanged()) {
      CHECK(ref_site_stats_.state_frequencies
            == calc_state_frequencies_per_partition_of(tree_.ref_sequence, evo_));
      if (ref_site_stats_.ntwiddle_valid && ref_site_stats_.evo_nu_l_version == evo_nu_l_version_) {
        CHECK(ref_site_stats_.ntwiddle_beta_a == calc_ref_ntwiddle_beta_a(tree_.ref_sequence, evo_));
      }
    }
    if (evo_nu_l_version_ == nu_version_) { CHECK(evo_.nu_l == nu_); }
    if (nu_sums_version_ == nu_version_) {
      auto sum_nu = 0.0;
      auto sum_log_nu = 0.0;
      for (auto nu_l : nu_) { sum_nu += nu_l; sum_log_nu += std::log(nu_l); }
      CHECK(sum_nu == nu_sum_ && sum_log_nu == nu_sum_log_);
    }
    { auto expected = calc_cur_log_coalescent_prior();
      CHECK(std::abs(log_coalescent_prior_ - expected) < 1e-8)
          << log_coalescent_prior_ << " != " << expected; }
    if (exact_coalescent_correction_) {
      auto expected = calc_cur_log_coalescent_exact() - calc_cur_log_coalescent_prior();
      CHECK(std::abs(log_coal_exact_correction_ - expected) < 1e-6)
          << log_coal_exact_correction_ << " != " << expected;
    }
    { auto expected = calc_cur_log_other_priors();
      CHECK(std::abs(log_other_priors_ - expected) < 1e-8)
          << log_other_priors_ << " != " << expected; }
    { auto expected = calc_cur_log_calibration_prior();
      CHECK(std::abs(log_calibration_prior_ - expected) < 1e-8)
          << log_calibration_prior_ << " != " << expected; }
    CHECK_EQ(state_frequencies_of_ref_sequence_, calc_cur_state_frequencies_of_ref_sequence());
  }
}

auto Run::check_global_and_local_totals_match() const -> void {
  if (estd::is_debug_enabled) {
    const auto& subrun_with_root = subruns_[tree_partition_.root_part_index()];
    auto sub_log_G = 0.0;
    auto sub_log_augmented_coalescent_prior = 0.0;
    auto sub_log_calibration_prior = frozen_calibration_log_prior_;
    for (const auto& subrun : subruns_) {
      sub_log_G += subrun.log_G();
      sub_log_augmented_coalescent_prior += subrun.log_augmented_coalescent_prior();
      sub_log_calibration_prior += subrun.log_calibration_prior();
    }
    CHECK(std::abs(log_calibration_prior() - sub_log_calibration_prior) < 1e-8)
        << log_calibration_prior() << " != " << sub_log_calibration_prior;
    CHECK(std::abs(log_G() - sub_log_G) < 1e-6)
        << log_G() << " != " << sub_log_G;
    auto log_augmented_coalescent_prior = calc_cur_log_augmented_coalescent_prior();
    CHECK(std::abs(log_augmented_coalescent_prior - sub_log_augmented_coalescent_prior) < 1e-6)
        << log_augmented_coalescent_prior << " != " << sub_log_augmented_coalescent_prior;
    { auto sub_counts = Seq_vector<int>{0};
      for (const auto& counts : subrun_with_root.state_frequencies_of_ref_sequence_per_partition()) {
        for (auto a : k_all_real_seq_letters) { sub_counts[a] += counts[a]; }
      }
      CHECK_EQ(state_frequencies_of_ref_sequence(), sub_counts); }
  }
}

auto Run::set_mpox_hack_enabled(bool mpox_hack_enabled) -> void {
  if (mpox_hack_enabled == mpox_hack_enabled_) { return; }

  mpox_hack_enabled_ = mpox_hack_enabled;
  if (mpox_hack_enabled) {
    // Set up two partitions as described in run.h
    auto first_tip = 0;
    CHECK(tree_.at(first_tip).is_tip());
    auto seq = view_of_sequence_at(tree_, first_tip);

    auto count_apobec_context = 0;
    for (auto l = Site_index{0}; l != tree_.num_sites(); ++l) {
      auto apobec_context =
          ((l != 0) &&
           (seq[l-1] == Real_seq_letter::T) &&
           ((seq[l] == Real_seq_letter::C) ||      // not yet mutated
            (seq[l] == Real_seq_letter::T))) ||    // already mutated
          ((l+1 != tree_.num_sites()) &&
           (seq[l+1] == Real_seq_letter::A) &&
           ((seq[l] == Real_seq_letter::G) ||      // not yet mutated
            (seq[l] == Real_seq_letter::A)));      // already mutated
      if (apobec_context) { ++count_apobec_context; }
      evo_.partition_for_site[l] = apobec_context ? 1 : 0;
    }
    ++site_layout_version_;
    std::cerr << count_apobec_context << " of " << tree_.num_sites() << " sites have APOBEC context\n";
    
    evo_.partition_evo_model = Partition_vector<Site_evo_model>(2, Site_evo_model{});

    mpox_mu_ = subst_model_.mu;
    mpox_mu_star_ = 0.0;
  } else {
    // Set up single partition
    for (auto l = Site_index{0}; l != tree_.num_sites(); ++l) {
      evo_.partition_for_site[l] = 0;
    }
    ++site_layout_version_;
    evo_.partition_evo_model = Partition_vector<Site_evo_model>(1, Site_evo_model{});
    subst_model_.mu = mpox_mu_;
  }
  invalidate_derived_quantities();
}

auto Run::derive_evo() const -> void {
  if (not mpox_hack_enabled_) {
    CHECK_EQ(evo_.num_partitions(), std::ssize(site_partition_rates_));
    // Every partition shares kappa and the state frequencies; its substitution rate is mu * r_p.
    auto site_evo = subst_model_.derive_site_evo_model();
    for (auto beta = Partition_index{0}; beta != evo_.num_partitions(); ++beta) {
      evo_.partition_evo_model[beta] = site_evo;
      evo_.partition_evo_model[beta].mu = subst_model_.mu * site_partition_rates_[beta];
    }
  } else {
    CHECK_EQ(evo_.num_partitions(), 2);

    // Partition 0 = Sites without APOBEC context
    auto& site_evo_0 = evo_.partition_evo_model[0];
    site_evo_0.mu = mpox_mu_;
    site_evo_0.pi_a = {0.25, 0.25, 0.25, 0.25};
    for (auto a : k_all_real_seq_letters) {
      for (auto b : k_all_real_seq_letters) {
        // Only polymerase-driven errors, following a simple Jukes-Cantor model
        site_evo_0.q_ab[a][b] = ((a == b) ? -1.0 : 1./3.);
      }
    }

    // Partition 1 = Sites with APOBEC context
    auto& site_evo_1 = evo_.partition_evo_model[1];

    // Polymerase-driven mutations are still there
    site_evo_1 = site_evo_0;
    site_evo_1.mu = mpox_mu_;
    auto rho = mpox_mu_star_ / mpox_mu_;
    
    // APOBEC-driven TC -> TT mutation
    site_evo_1.q_ab[Real_seq_letter::C][Real_seq_letter::T] += 2*rho;
    site_evo_1.q_ab[Real_seq_letter::C][Real_seq_letter::C] -= 2*rho;
    
    // APOBEC-driven GA -> AA mutation (== TC -> TT on the complementary strand)
    site_evo_1.q_ab[Real_seq_letter::G][Real_seq_letter::A] += 2*rho;
    site_evo_1.q_ab[Real_seq_letter::G][Real_seq_letter::G] -= 2*rho;
  }
  if (evo_nu_l_version_ != nu_version_) {
    evo_.nu_l = nu_;
    evo_nu_l_version_ = nu_version_;
  }
}

auto Run::calc_cur_log_G() const -> double {
  // From scratch, not from `ref_site_stats_`: this is the reference the incremental and cached values are checked against
  return calc_log_G_with(calc_state_frequencies_per_partition_of(tree_.ref_sequence, evo_),
                         calc_cum_Q_l_for_sequence(tree_.ref_sequence, evo_));
}

auto Run::calc_log_G_with(
    const Partition_vector<Seq_vector<int>>& state_frequencies_per_partition,
    const std::vector<double>& ref_cum_Q_l) const
    -> double {
  auto lambda_i = calc_lambda_i(tree_, evo_, ref_cum_Q_l);
  auto result = 0.0;
  result += calc_log_root_prior(tree_, evo_, state_frequencies_per_partition);
  result += calc_log_G_below_root(tree_, evo_, lambda_i, state_frequencies_per_partition);
  return result;
}

auto Run::calc_cur_Ttwiddle_beta_a() const -> Partition_vector<Seq_vector<double>> {
  return calc_Ttwiddle_beta_a(tree_, evo_, cur_ref_site_stats().ntwiddle_beta_a);
}

auto Run::ref_site_stats_inputs_unchanged() const -> bool {
  const auto& c = ref_site_stats_;
  return c.valid
      && c.ref_version == ref_version_
      && c.site_layout_version == site_layout_version_;
}

auto Run::cur_ref_site_stats() const -> const Ref_site_stats& {
  auto& c = ref_site_stats_;
  if (not ref_site_stats_inputs_unchanged()) {
    c.ref_version = ref_version_;
    c.site_layout_version = site_layout_version_;
    c.state_frequencies = calc_state_frequencies_per_partition_of(tree_.ref_sequence, evo_);
    c.ntwiddle_valid = false;
    c.valid = true;
  }
  if (not (c.ntwiddle_valid && c.evo_nu_l_version == evo_nu_l_version_)) {
    c.evo_nu_l_version = evo_nu_l_version_;
    if (evo_nu_l_all_ones()) {
      // Each nu-weighted count is then a sum of ones, which is exact in double precision below 2^53, so it equals the
      // count itself bit for bit
      c.ntwiddle_beta_a = Partition_vector<Seq_vector<double>>(evo_.num_partitions(), Seq_vector<double>{0.0});
      for (auto beta = Partition_index{0}; beta != evo_.num_partitions(); ++beta) {
        for (auto a : k_all_real_seq_letters) { c.ntwiddle_beta_a[beta][a] = c.state_frequencies[beta][a]; }
      }
    } else {
      c.ntwiddle_beta_a = calc_ref_ntwiddle_beta_a(tree_.ref_sequence, evo_);
    }
    c.ntwiddle_valid = true;
  }
  return c;
}

auto Run::evo_nu_l_all_ones() const -> bool {
  if (nu_l_all_ones_version_ != evo_nu_l_version_) {
    nu_l_all_ones_ = std::ranges::all_of(evo_.nu_l, [](double nu_l) { return nu_l == 1.0; });
    nu_l_all_ones_version_ = evo_nu_l_version_;
  }
  return nu_l_all_ones_;
}

static auto same_site_evo_models(const Partition_vector<Site_evo_model>& a, const Partition_vector<Site_evo_model>& b)
    -> bool {
  if (std::ssize(a) != std::ssize(b)) { return false; }
  for (auto beta = 0; beta != std::ssize(a); ++beta) {
    if (not (a[beta].mu == b[beta].mu && a[beta].pi_a == b[beta].pi_a && a[beta].q_ab == b[beta].q_ab)) {
      return false;
    }
  }
  return true;
}

auto Run::cur_shared_evo() const -> std::shared_ptr<const Global_evo_model> {
  if (not (shared_evo_ != nullptr
           && shared_evo_site_layout_version_ == site_layout_version_
           && shared_evo_nu_l_version_ == evo_nu_l_version_
           && same_site_evo_models(shared_evo_->partition_evo_model, evo_.partition_evo_model))) {
    if (shared_evo_ != nullptr && shared_evo_.use_count() == 1) {
      *shared_evo_ = evo_;  // no subrun reads the old copy: overwrite it in place
    } else {
      shared_evo_ = std::make_shared<Global_evo_model>(evo_);
    }
    shared_evo_site_layout_version_ = site_layout_version_;
    shared_evo_nu_l_version_ = evo_nu_l_version_;
  }
  if (estd::is_debug_enabled) {
    CHECK(shared_evo_->partition_for_site == evo_.partition_for_site);
    CHECK(shared_evo_->nu_l == evo_.nu_l);
  }
  return shared_evo_;
}

auto Run::ref_cum_Q_l_is_current() const -> bool {
  return ref_cum_Q_l_ != nullptr
      && ref_cum_Q_l_ref_version_ == ref_version_
      && ref_cum_Q_l_site_layout_version_ == site_layout_version_
      && ref_cum_Q_l_nu_l_version_ == evo_nu_l_version_
      && same_site_evo_models(ref_cum_Q_l_partition_models_, evo_.partition_evo_model);
}

auto Run::cur_ref_cum_Q_l() const -> std::shared_ptr<const std::vector<double>> {
  if (not ref_cum_Q_l_is_current()) {
    if (ref_cum_Q_l_ == nullptr || ref_cum_Q_l_.use_count() > 1) {  // a subrun still reads the old table
      ref_cum_Q_l_.reset();
      ref_cum_Q_l_ = std::make_shared<std::vector<double>>();
    }
    fill_cum_Q_l_for_sequence(tree_.ref_sequence, evo_, *ref_cum_Q_l_);
    ref_cum_Q_l_ref_version_ = ref_version_;
    ref_cum_Q_l_site_layout_version_ = site_layout_version_;
    ref_cum_Q_l_nu_l_version_ = evo_nu_l_version_;
    ref_cum_Q_l_partition_models_ = evo_.partition_evo_model;
  }
  return ref_cum_Q_l_;
}

auto Run::calc_cur_num_muts() const -> int {
  return calc_num_muts(tree_);
}

auto Run::calc_cur_num_muts_ab() const -> Seq_matrix<int> {
  return calc_num_muts_ab(tree_);
}

auto Run::calc_cur_log_coalescent_prior() const -> double {
  for (const auto& node : index_order_traversal(tree_)) {
    if (tree_.at(node).is_tip()) {
      coalescent_prior_.displace_tip(node, tree_.at(node).t);
    } else {
      coalescent_prior_.displace_coalescence(node, tree_.at(node).t);
    }
  }
  return coalescent_prior_.calc_log_prior();
}

auto Run::make_skygrid_exact_events() const -> Exact_coalescent_events {
  const Pop_model& raw_pop_model = *pop_model_;
  CHECK(typeid(raw_pop_model) == typeid(Skygrid_pop_model));
  const auto& skygrid_pop_model = static_cast<const Skygrid_pop_model&>(raw_pop_model);
  return Exact_coalescent_events{coalescent_event_times_of(tree_), skygrid_pop_model.x()};
}

auto Run::calc_cur_log_coalescent_exact() const -> double {
  return calc_exact_coalescent_log_prior(coalescent_event_times_of(tree_), *pop_model_);
}

auto Run::calc_cur_log_coalescent_target() const -> double {
  return exact_coalescent_correction_ ? calc_cur_log_coalescent_exact() : calc_cur_log_coalescent_prior();
}

auto Run::calc_cur_log_augmented_coalescent_prior() const -> double {
  auto result = 0.0;
  CHECK(not coalescent_prior_parts_.empty());
  for (const auto& part : coalescent_prior_parts_) {
    result += part.calc_partial_log_prior();
  }
  return result;
}

auto Run::calc_cur_state_frequencies_of_ref_sequence() const -> Seq_vector<int> {
  // Summed over the site partitions: the HKY state frequencies are shared, so the root prior only ever
  // needs the counts over all sites.  With one partition this is the historical value.
  return sum_over_partitions(calc_state_frequencies_per_partition_of(tree_.ref_sequence, evo_));
}

auto Run::sum_over_partitions(const Partition_vector<Seq_vector<int>>& per_partition) -> Seq_vector<int> {
  auto result = Seq_vector<int>{0};
  for (const auto& counts : per_partition) {
    for (auto a : k_all_real_seq_letters) { result[a] += counts[a]; }
  }
  return result;
}

auto Run::log_calibration_prior_from_parts() const -> double {
  auto result = frozen_calibration_log_prior_;
  for (const auto& subrun : subruns_) { result += subrun.log_calibration_prior(); }
  return result;
}

auto Run::calc_cur_log_calibration_prior() const -> double {
  if (calibrations_.empty()) { return 0.0; }
  return calc_log_calibration_prior(calibrations_, calibration_nodes_, tree_);
}

auto Run::calc_cur_log_other_priors() const -> double {
  auto log_prior = 0.0;

  // Site partitions: symmetric Dirichlet(c) prior on the length-weighted rate shares
  // x_p = L_p r_p / sum_j L_j.  With one partition x = (1) and the term is zero, so nothing changes for
  // runs without a partition configuration.
  if (site_partition_.num_partitions() > 1 && site_partition_rate_prior_concentration_ != 1.0) {
    auto total_weight = static_cast<double>(site_partition_.total_observed_sites());
    for (auto p = Partition_index{0}; p != site_partition_.num_partitions(); ++p) {
      auto x_p = site_partition_.num_observed_sites[p] * site_partition_rates_[p] / total_weight;
      log_prior += (site_partition_rate_prior_concentration_ - 1.0) * std::log(x_p);
    }
  }

  if (not mpox_hack_enabled_) {
    // Mu - Gamma(mu_prior_alpha_, mu_prior_beta_) prior
    log_prior += (mu_prior_alpha_ - 1) * std::log(mu()) - mu_prior_beta_ * mu();
  } else {
    // Mu - Gamma(mu_prior_alpha_, mu_prior_beta_) prior (mu* has a uniform prior)
    log_prior += (mu_prior_alpha_ - 1) * std::log(mpox_mu_) - mu_prior_beta_ * mpox_mu_;
  }

  // Alpha - Exponential prior with mean 1.0
  const auto mean_alpha = 1.0;
  log_prior += -alpha_ / mean_alpha - std::log(mean_alpha);
  
  // nu_l ~ Gamma(alpha, alpha) = (alpha^alpha / Gamma(alpha)) (\nu_l)^{\alpha - 1} exp(-\alpha nu_l)
  auto L = tree_.num_sites();
  log_prior += L * (alpha_ * std::log(alpha_) - std::lgamma(alpha_));
  if (nu_sums_version_ != nu_version_) {
    auto sum_nu = 0.0;
    auto sum_log_nu = 0.0;
    for (auto nu_l : nu_) {
      DCHECK_NE(nu_l, 0.0);
      sum_nu += nu_l;
      sum_log_nu += std::log(nu_l);
    }
    nu_sum_ = sum_nu;
    nu_sum_log_ = sum_log_nu;
    nu_sums_version_ = nu_version_;
  }
  log_prior += (alpha_ - 1) * nu_sum_log_ - alpha_ * nu_sum_;

  if (not mpox_hack_enabled_) {
    // State frequencies - uniform prior on the simplex when they are estimated, nothing otherwise.

    // Every free relative exchangeability s_k / s_GT has a log-normal prior,
    //  pi(s) = exp(-(log(s) - m_k)^2 / (2 sigma^2)) / (s sqrt(2 pi sigma^2)),
    // with m_k = 1 for a class of transitions only and 0 otherwise, and sigma = 1.25.  For HKY85 and
    // K80 the only free class is the transition class, so this is exactly the historical prior on
    // kappa.
    auto sigma = subst_exchangeability_prior_sd_;
    for (auto k = 0; k != subst_model_.grouping.num_classes(); ++k) {
      if (k == subst_model_.grouping.reference_class()) { continue; }
      auto s = subst_model_.class_rate[k];
      auto m = subst_exchangeability_prior_log_mean_of(k);
      log_prior += -std::pow(std::log(s) - m, 2) / (2 * sigma * sigma)
          - 0.5 * std::log(2 * M_PI * sigma * sigma) - std::log(s);
    }
  }

  const Pop_model& raw_pop_model = *pop_model_;
  if (typeid(raw_pop_model) == typeid(Exp_pop_model)) {
    const auto& exp_pop_model = static_cast<const Exp_pop_model&>(raw_pop_model);
    
    // pop_n0 - Inverse-Gamma(pop_inv_n0_prior_alpha_, pop_inv_n0_prior_beta_) prior
    log_prior += -(pop_inv_n0_prior_alpha_ + 1) * std::log(exp_pop_model.pop_at_t0())
                 - pop_inv_n0_prior_beta_ / exp_pop_model.pop_at_t0();
    
    // pop_g - Laplace(pop_g_prior_mu_, pop_g_prior_scale_) prior on g in [pop_g_min_, pop_g_max_]
    //   pi(g) ~ exp(-|g - mu| / scale) * I(g_min <= g <= g_max)
    log_prior += -std::abs(exp_pop_model.growth_rate() - pop_g_prior_mu_) / pop_g_prior_scale_
                 - std::log(2 * pop_g_prior_scale_);
    
  } else if (typeid(raw_pop_model) == typeid(Skygrid_pop_model)) {
    const auto tau = skygrid_tau_;
    const auto log_tau = std::log(tau);
    
    if (skygrid_tau_move_enabled_) {
      // tau - gamma prior (Gill et al 2012, Eq. 15)
      log_prior +=
          (skygrid_tau_prior_alpha_ - 1) * log_tau
          - skygrid_tau_prior_beta_ * tau;
    }
    
    log_prior += calc_cur_skygrid_gmrf_prior();
  }

  return log_prior;
}

auto Run::calc_cur_skygrid_gmrf_prior() const -> double {
  auto log_prior = 0.0;

  const Pop_model& raw_pop_model = *pop_model_;
  const auto& skygrid_pop_model = static_cast<const Skygrid_pop_model&>(raw_pop_model);

  const auto tau = skygrid_tau_;
  const auto log_tau = std::log(tau);
  const auto M = skygrid_pop_model.M();

  // InvGamma(alpha, beta) prior on N_bar = exp(gamma_bar)
  //   log pi(gamma_bar) = -alpha * gamma_bar - beta * exp(-gamma_bar) + const
  // Default alpha=0, beta=0: no contribution (uniform on gamma_bar).
  auto gamma_bar = skygrid_pop_model.gamma_bar();
  log_prior += -skygrid_inv_nbar_prior_alpha_ * gamma_bar
               - skygrid_inv_nbar_prior_beta_ * std::exp(-gamma_bar);

  // gamma - GMRF prior (Gill et al 2012, Eq. 13)
  //
  // Since time intervals are in principle not all the same, we really should
  // be using something like a diffusion law to condition how much consecutive
  // gamma's should vary, e.g., gamma_{i+1} - gamma_i ~ N(0, 2 D (x_{i+1} - x_i)).
  // We don't do that to not deviate from the standard Skygrid parametrization;
  // assuming equal time intervals of width dt = x_1 - x_0, we'd have
  //
  //    2 D dt = 1/tau => D = 1/(2 tau dt).

  const auto log_two_pi = std::log(2.0 * std::numbers::pi);
  for (auto k = 1; k <= M; ++k) {
    auto delta_gamma = skygrid_pop_model.gamma(k) - skygrid_pop_model.gamma(k-1);
    log_prior +=
          
        // Gaussian normalization: log(sqrt((tau/2) / pi)) = (1/2) (log(tau) - log(2 pi))
        0.5 * (log_tau - log_two_pi)

        // Gaussian exponent
        - 0.5 * std::pow(delta_gamma, 2) * tau;
  }

  if (skygrid_low_gamma_barrier_enabled_) {
    // Strong lower bound on gamma_k's
    //
    // Anything implying a coalescent timescale below N(t) = e^{skygrid_gamma_min} (with
    // default value of 1 day = the resolution of tip timestamps) is probably a numerical
    // instability.  These instabilities tend to drive the population curve to extremes that
    // our moveset finds it hard to recover from.  In realistic datasets, this prior should
    // almost always be exactly zero.
    //
    // The barrier on low gamma_k's is quadratic: for each gamma_k, dropping below
    // skygrid_gamma_min by skygrid_delta_gamma reduces the log-posterior by 1.
    //
    for (auto k = 0; k <= M; ++k) {
      if (skygrid_pop_model.gamma(k) < skygrid_low_gamma_barrier_loc_) {
        auto excess = skygrid_low_gamma_barrier_loc_ - skygrid_pop_model.gamma(k);
        log_prior -= std::pow(excess / skygrid_low_gamma_barrier_scale_, 2);
      }
    }
  }

  return log_prior;
}

static auto subrun_exec(
    int /*thread_id*/,
    Subrun& subrun,
    int64_t subrun_substeps)
    -> void {
  
  for (auto i = 0L; i != subrun_substeps; ++i) {
    subrun.mcmc_sub_iteration();
  }
}

// Do `substeps` local moves spread over the subruns, then one global move
auto Run::do_mcmc_steps(int substeps) -> void {
  auto target_step = step_ + substeps;

  // Ensure that changes to tree, such as from normalize_root() at the end of the last mcmc_iterations call
  // or any unexpected outside manipulation, are propagated to subruns.  With threshold partitioning, a repartition
  // that is about to happen before any local move makes this one redundant; skipping it only leaves out one draw of
  // the state-independent auxiliary variable.
  auto repartition_due = step_ >= next_global_move_step_ && (force_repartition_ || repartitioning_enabled_);
  if (not (threshold_partitioning_active() && repartition_due)) { repartition(); }
  
  while (step_ < target_step) {
    
    // Correct for accumulating round-off errors.  When global moves are due, they start by recomputing every derived
    // quantity from scratch (nothing in between reads them or draws random numbers), so validating here as well would
    // only repeat the same computation.
    if (step_ >= (last_revalidation_step_ + 1'000'000)) { invalidate_derived_quantities_same_reference(); }
    if (step_ < next_global_move_step_) { validate_derived_quantities(); }
    
    if (step_ >= next_global_move_step_) {
      auto should_repartition = force_repartition_ || repartitioning_enabled_;
      if (should_repartition) { repartition(); }
      run_global_moves();
      next_global_move_step_ = step_ + local_moves_per_global_move_;
    }

    CHECK_LT(step_, next_global_move_step_);
    auto steps_before_next_global_move = std::min(next_global_move_step_, target_step) - step_;
    if (steps_before_next_global_move > 0) {
      push_global_params_to_subruns();
      check_global_and_local_totals_match();
      if (exact_coalescent_correction_) {
        run_corrected_local_batch(steps_before_next_global_move);
      } else {
        run_local_moves(steps_before_next_global_move);
        reassemble();
      }
      step_ += steps_before_next_global_move;
    }
  }
  
  normalize_root();
  
  assert_cur_tip_sequences_compatible_with_original_ones();
  assert_phylo_tree_integrity(tree_, paranoid_);
}

auto Run::set_step(int64_t step) -> void {
  if (step < 0) {
    throw std::invalid_argument(absl::StrFormat(
        "Step should not be negative (got %d)", step));
  }
  step_ = step;
  next_global_move_step_ = -1;
  next_partition_stencil_refresh_step_ = -1;
}

auto Run::set_local_moves_per_global_move(int local_moves_per_global_move) -> void {
  if (local_moves_per_global_move == -1) {
    local_moves_per_global_move = 50 * std::ssize(tree_);  // A good default
  }
  if (local_moves_per_global_move < 1) {
    throw std::invalid_argument(absl::StrFormat(
        "There must be at least 1 local move per global move (got %d)", local_moves_per_global_move));
  }
  local_moves_per_global_move_ = local_moves_per_global_move;
  next_global_move_step_ = -1;
  next_partition_stencil_refresh_step_ = -1;
}

// One batch of local moves followed by the Metropolis-Hastings step of the exact coalescent correction.  The batch kernel is
// reversible with respect to the cell-prior target at fixed auxiliary variables (random scan over move types, independent
// parts), so the acceptance ratio needs only the change of delta = log P_exact - log P_cell.  On rejection the genealogy
// (tree, mutation histories, missations) is restored from the snapshot, the derived quantities are recomputed and the
// subruns are rebuilt from the restored tree; the random number generators are not restored.
auto Run::run_corrected_local_batch(int count) -> void {
  validate_derived_quantities();
  auto delta_before = log_coal_exact_correction_;
  // The batch never changes the reference sequence of `tree_` (see `reassemble`), so the snapshot leaves it out
  auto ref_sequence = std::move(tree_.ref_sequence);
  auto snapshot = tree_;
  tree_.ref_sequence = std::move(ref_sequence);

  run_local_moves(count);
  reassemble();
  validate_derived_quantities();  // recomputes the cell prior and delta of the proposed genealogy

  auto log_ratio = log_coal_exact_correction_ - delta_before;
  ++exact_batches_proposed_;
  if (log_ratio >= 0.0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_ratio)) {
    ++exact_batches_accepted_;
    return;
  }

  snapshot.ref_sequence = std::move(tree_.ref_sequence);
  tree_ = std::move(snapshot);
  ++ref_version_;
  invalidate_derived_quantities();
  repartition();
}

// The local moves of a batch are shared out in proportion to the part sizes, a function of the partition alone, so
// that every node is visited at the rate of a single-part run.  All parts go to the pool, largest first; the parts
// are independent and each has its own generator, so the schedule does not affect the result.
auto Run::run_threshold_partition_local_moves(int count) -> void {
  auto num_subruns = std::ssize(subruns_);
  auto total_nodes = int64_t{0};
  for (const auto& subrun : subruns_) { total_nodes += std::ssize(subrun.tree()); }
  auto substeps = std::vector<int64_t>(num_subruns, 0);
  auto assigned = int64_t{0};
  for (auto i = 0; i != num_subruns; ++i) {
    substeps[i] = int64_t{count} * std::ssize(subruns_[i].tree()) / total_nodes;
    assigned += substeps[i];
  }
  for (auto i = 0; assigned < count; i = (i + 1) % num_subruns) { ++substeps[i]; ++assigned; }

  auto order = std::vector<int>(num_subruns);
  std::iota(order.begin(), order.end(), 0);
  std::ranges::stable_sort(order, [&](int a, int b) { return substeps[a] > substeps[b]; });
  auto subrun_results = std::vector<std::future<void>>{};
  for (auto i : order) {
    if (substeps[i] == 0) { continue; }
    subrun_results.push_back(thread_pool_->push(subrun_exec, std::ref(subruns_[i]), substeps[i]));
  }
  for (auto& subrun_result : subrun_results) {
    subrun_result.get();
  }
}

auto Run::run_local_moves(int count) -> void {
  if (threshold_partitioning_active()) {
    run_threshold_partition_local_moves(count);
    return;
  }
  auto subcount = count / subruns_.size();
  auto subrun_count = std::ssize(subruns_);
  auto subrun_results = std::vector<std::future<void>>{};
  for (auto i = 1; i < subrun_count; ++i) {  // Note: start at 1
    subrun_results.push_back(thread_pool_->push(subrun_exec, std::ref(subruns_[i]), subcount));
  }
  subrun_exec(0, subruns_[0], count - (subrun_count-1) * subcount);  // Subrun 0 runs on this thread
  for (auto& subrun_result : subrun_results) {
    subrun_result.get();
  }
}

auto Run::run_global_moves() -> void {
  // Global moves.  The subruns are given the table of cumulative site rates again before the next batch; until then
  // nothing needs it kept, and some global moves allocate per-site arrays of their own.
  ref_cum_Q_l_.reset();
  invalidate_derived_quantities_same_reference();
  validate_derived_quantities();

  // Each group of global moves below is fully independent of the others and is practically a Gibbs sampling
  // of one of the global parameters, so there's no point in doing lots of these moves

  if (not mpox_hack_enabled_) {
    // 1. Gibbs sampling of mu
    // Depends on Ttwiddle_a, {q_a} and num_muts
    if (mu_move_enabled_) {
      mu_move();
      check_derived_quantities();
    }
    
    // 2. Pseudo-Gibbs sampling of evo model parameters
    // Depends only on num_muts_ab and counts of state in root sequence; after enough MCMC moves, the
    // resulting kappa & pi_a's are practically Gibbs sampled
    for (auto i = 0; i != 10; ++i) {
      if (subst_frequencies_move_enabled_) {
        subst_frequencies_move();
        check_derived_quantities();
      }
      if (subst_exchangeability_move_enabled_) {
        subst_exchangeability_moves();
        check_derived_quantities();
      }
    }

    // 2b. Relative rates of the site partitions (nothing happens with a single partition)
    if (site_partition_rate_move_enabled_) {
      site_partition_rate_moves();
    }
  } else {
    // 1 & 2. Gibbs sampling of mu & mu_star
    mpox_hack_moves();
    check_derived_quantities();
  }

  // 3. Pseudo-Gibbs sampling of alpha
  // On integrating out the nu_l's, p(alpha) depends only on {M_l} and {TTwiddle_l}.  After lots of little alpha moves,
  // alpha is practically Gibbs sampled.  Then the nu_l's really are Gibbs sampled
  if (alpha_move_enabled_) {
    alpha_moves();  // Gibbs samples all nu's as part of the alpha move
    check_derived_quantities();
  }

  // 4-pre. Adjust resolution of staircases used in coalescent prior if they're really out of whack
  auto min_t = tree_.at_root().t;
  auto max_t = t_max_tip();
  auto cur_t_step = coalescent_prior_.t_step();
  auto span = max_t - min_t;
  if (coal_prior_fixed_t_step_ > 0.0) {
    // The grid is pinned by the user; see set_coal_prior_fixed_t_step().
  } else if (coal_prior_grid_tolerance_ > 0.0) {
    // With a tolerance in force, the resolution is set by the measured discretization excess rather
    // than by the span alone, so the trigger watches the span instead of the cell width: refine
    // again once the tree has grown or shrunk enough that the last measurement is stale.
    auto ratio = span / coal_prior_grid_span_at_retarget_;
    if (coal_prior_grid_span_at_retarget_ <= 0.0 || ratio < 2./3. || ratio > 4./3.) {
      retarget_coal_prior_grid();
      coal_prior_grid_span_at_retarget_ = span;
    }
  } else {
    auto target_t_step = span / target_coal_prior_cells_;
    auto min_t_step = (1.0) / target_coal_prior_cells_;  // Avoid degenerate cases when tree collapses
    auto ratio_cur_to_target = cur_t_step / target_t_step;
    auto retarget =  // Don't do this unless really needed
        (cur_t_step > min_t_step) &&
        (ratio_cur_to_target < 2./3. || ratio_cur_to_target > 4./3.);
    if (retarget) {
      auto new_t_step = std::max(min_t_step, 0.5 * (cur_t_step + target_t_step)); // Avoid jumping to an extreme
      set_coalescent_prior_t_step(new_t_step);
    }
  }

  // 4. Pseudo-Gibbs sampling of population parameters
  // Depends only on tree topology; after enough MCMC moves, the
  // resulting population model parameters are practically Gibbs sampled
  const Pop_model& raw_pop_model = *pop_model_;
  
  if (typeid(raw_pop_model) == typeid(Exp_pop_model)) {
    for (auto i = 0; i != pop_moves_per_global_move_; ++i) {
      if (final_pop_size_move_enabled_) {
        pop_size_move();
        check_derived_quantities();
      }
      if (pop_growth_rate_move_enabled_) {
        pop_growth_rate_move();
        check_derived_quantities();
      }
    }
    
  } else if (typeid(raw_pop_model) == typeid(Skygrid_pop_model)) {
    // With HMC, each move likely samples each quantity quite well already
    if (skygrid_tau_move_enabled_) {
      skygrid_tau_move();
      check_derived_quantities();
    }
      
    skygrid_gammas_zero_mode_gibbs_move();
    check_derived_quantities();
      
    skygrid_gammas_hmc_move();
    check_derived_quantities();
  }

  // 5. Joint time-scale move (opt-in)
  if (global_scale_move_enabled_) {
    for (auto i = 0; i != global_scale_moves_per_global_move_; ++i) { global_scale_move(); }
  }

  // 6. Whole-site mutational-history refresh (opt-in)
  site_history_refresh_moves();
}


auto Run::resume_state() const -> Resume_state {
  auto state = Resume_state{};
  {
    auto oss = std::ostringstream{};
    oss << bitgen_;
    state.prng = oss.str();
  }
  for (const auto& gen : subbitgens_) {
    auto oss = std::ostringstream{};
    oss << gen;
    state.sub_prngs.push_back(oss.str());
  }
  if (partition_stencils_valid_) {
    for (const auto& stencil : partition_stencils_) {
      auto cut_points = std::vector<Node_index>{};
      for (const auto& info : stencil) { cut_points.push_back(info.cut_point); }
      state.stencils.push_back(std::move(cut_points));
    }
    state.stencil_refresh_step = next_partition_stencil_refresh_step_;
  }
  return state;
}

auto Run::restore_resume_state(const Resume_state& state) -> void {
  auto restore_gen = [](std::mt19937& gen, const std::string& text, const char* what) {
    auto iss = std::istringstream{text};
    auto restored = std::mt19937{};
    iss >> restored;
    if (iss.fail()) {
      throw std::invalid_argument(std::string{"The saved "} + what + " random number generator state cannot be read");
    }
    gen = restored;
  };
  if (state.prng.empty()) { return; }
  restore_gen(bitgen_, state.prng, "run");
  subbitgens_.clear();
  for (const auto& text : state.sub_prngs) {
    subbitgens_.emplace_back();
    restore_gen(subbitgens_.back(), text, "part");
  }
  if (not state.stencils.empty() && state.stencil_refresh_step >= 0) {
    auto stencils = std::vector<std::vector<Partition_part_info>>{};
    for (const auto& cut_points : state.stencils) {
      auto stencil = std::vector<Partition_part_info>{};
      for (auto cut_point : cut_points) {
        if (cut_point < 0 || cut_point >= std::ssize(tree_)) {
          throw std::invalid_argument(absl::StrFormat(
              "A saved partition stencil refers to node %d, but the tree has %d nodes", cut_point, std::ssize(tree_)));
        }
        stencil.push_back(Partition_part_info{cut_point});
      }
      stencils.push_back(std::move(stencil));
    }
    partition_stencils_ = std::move(stencils);
    partition_stencils_valid_ = true;
    next_partition_stencil_refresh_step_ = state.stencil_refresh_step;
  }
}

auto Run::set_global_scale_move_enabled(bool enabled) -> void {
  if (enabled) {
    global_scale_move_enabled_ = false;
    check_global_scale_move_supported();
  }
  global_scale_move_enabled_ = enabled;
}

auto Run::set_global_scale_move_window(double window) -> void {
  if (not (window > 0.0) || not std::isfinite(window)) {
    throw std::invalid_argument("The log-scale window of the joint time-scale move must be positive and finite");
  }
  global_scale_move_window_ = window;
}

auto Run::set_global_scale_moves_per_global_move(int count) -> void {
  if (count < 1) {
    throw std::invalid_argument("The joint time-scale move needs at least one proposal per global step");
  }
  global_scale_moves_per_global_move_ = count;
}

auto Run::check_global_scale_move_supported() const -> void {
  auto refuse = [](const std::string& why) {
    throw std::invalid_argument("The joint time-scale move is not supported here: " + why);
  };
  if (not exact_coalescent_correction_) { refuse("it needs the exact coalescent correction (the grid prior is not scale invariant)"); }
  if (topology_moves_enabled_) { refuse("the topology must be fixed"); }
  if (mpox_hack_enabled_) { refuse("the mpox hack is not supported"); }
  const Pop_model& raw_pop_model = *pop_model_;
  if (typeid(raw_pop_model) != typeid(Exp_pop_model)) { refuse("only the exponential population model is supported"); }
  const auto& exp_pop_model = static_cast<const Exp_pop_model&>(raw_pop_model);
  auto t_tips = calc_max_tip_time(tree_);
  for (const auto& node : index_order_traversal(tree_)) {
    if (tree_.at(node).is_tip()
        && (tree_.at(node).t != t_tips || tree_.at(node).t_min != tree_.at(node).t_max)) {
      refuse("all samples must share one sampling date without date uncertainty");
    }
  }
  if (exp_pop_model.t0() != t_tips) {
    refuse("the reference time t0 of the population model must be the sampling date");
  }
}

auto Run::global_scale_move() -> void {
  auto u = absl::Uniform(bitgen_, -global_scale_move_window_, +global_scale_move_window_);
  auto accept_draw = absl::Uniform(bitgen_, 0.0, 1.0);
  global_scale_move_with(u, accept_draw);
}

auto Run::global_scale_move_with(double u, double accept_draw) -> Global_scale_move_result {
  if (not std::isfinite(u)) {
    throw std::invalid_argument("The log scale of the joint time-scale move must be finite");
  }
  check_global_scale_move_supported();
  if (not tree_.at_root().mutations.empty()) {
    normalize_root();  // mutations above the root have no age; fold them into the reference sequence
    invalidate_derived_quantities();
  }
  validate_derived_quantities();
  auto result = Global_scale_move_result{};
  result.log_scale = u;

  const auto old_log_G = log_G_;
  const auto old_log_coal = log_coalescent_prior_ + log_coal_exact_correction_;
  const auto old_log_other = log_other_priors_;
  const auto old_log_calib = log_calibration_prior_;
  const auto old_target = old_log_G + old_log_coal + old_log_other + old_log_calib;

  const auto s = std::exp(u);
  const auto t_ref = calc_max_tip_time(tree_);
  auto n_internal = int64_t{0};
  auto n_mutations = int64_t{0};
  for (const auto& node : index_order_traversal(tree_)) {
    if (not tree_.at(node).is_tip()) { ++n_internal; }
    n_mutations += std::ssize(tree_.at(node).mutations);
  }
  result.log_jacobian = static_cast<double>(n_internal + n_mutations - 1) * u;

  const auto old_pop_model = std::static_pointer_cast<const Exp_pop_model>(pop_model_);
  const auto old_mu = mu();
  const auto new_g = old_pop_model->growth_rate() / s;
  ++global_scale_proposed_;
  if (new_g < pop_g_min_ || new_g > pop_g_max_) {
    result.log_alpha = -std::numeric_limits<double>::infinity();
    return result;
  }

  // Numerical range of the proposal: the scaled genealogy must stay representable and ordered (parent strictly older than child,
  // every mutation inside its branch), and its span must fit the pinned coalescent grid (at most `coal_prior_grid_max_cells_`
  // cells, as for the retargeting rules), because the cell prior that carries the whole-batch correction is evaluated on that grid.
  // A proposal outside this range is rejected before anything is changed; the range is a numerical limit of the implementation,
  // far outside any state a data set supports (it excludes root ages above `grid_max_cells * t_step`).
  {
    auto scaled = [&](double t) { return t_ref - s * (t_ref - t); };
    auto valid = std::isfinite(s) && s > 0.0
        && s * (t_ref - tree_.at_root().t) <= coal_prior_fixed_t_step_ * static_cast<double>(coal_prior_grid_max_cells_);
    for (const auto& node : index_order_traversal(tree_)) {
      if (not valid) { break; }
      const auto& n = tree_.at(node);
      if (n.parent == k_no_node) { continue; }
      auto t_parent = scaled(tree_.at(n.parent).t);
      auto t_child = n.is_tip() ? n.t : scaled(n.t);
      if (not (std::isfinite(t_parent) && std::isfinite(t_child) && t_parent < t_child)) { valid = false; break; }
      for (const auto& m : n.mutations) {
        auto t_mut = scaled(m.t);
        if (not (std::isfinite(t_mut) && t_mut >= t_parent && t_mut <= t_child)) { valid = false; break; }
      }
    }
    if (not valid) {
      result.log_alpha = -std::numeric_limits<double>::infinity();
      return result;
    }
  }

  auto snapshot = tree_;
  for (const auto& node : index_order_traversal(tree_)) {
    auto& n = tree_.at(node);
    if (not n.is_tip()) { n.t = t_ref - s * (t_ref - n.t); }
    for (auto& m : n.mutations) { m.t = t_ref - s * (t_ref - m.t); }
  }
  set_mu(old_mu / s);
  pop_model_ = std::make_shared<Exp_pop_model>(
      old_pop_model->t0(), s * old_pop_model->pop_at_t0(), new_g, old_pop_model->min_pop());
  coalescent_prior_.pop_model_changed(pop_model_);
  invalidate_derived_quantities();
  validate_derived_quantities();

  const auto new_log_coal = log_coalescent_prior_ + log_coal_exact_correction_;
  const auto new_target = log_G_ + new_log_coal + log_other_priors_ + log_calibration_prior_;
  result.delta_log_G = log_G_ - old_log_G;
  result.delta_log_coalescent = new_log_coal - old_log_coal;
  result.delta_log_other_priors = log_other_priors_ - old_log_other;
  result.delta_log_calibration = log_calibration_prior_ - old_log_calib;
  if (old_pop_model->min_pop() == 0.0 && std::isfinite(new_target)) {
    result.scale_free_residual = (result.delta_log_G + result.delta_log_coalescent + result.log_jacobian) + u;
    global_scale_max_abs_residual_ = std::max(global_scale_max_abs_residual_, std::abs(result.scale_free_residual));
  }
  result.log_alpha = std::isfinite(new_target)
      ? (new_target - old_target) + result.log_jacobian
      : -std::numeric_limits<double>::infinity();

  if (result.log_alpha >= 0.0 || accept_draw < std::exp(result.log_alpha)) {
    result.accepted = true;
    ++global_scale_accepted_;
    global_scale_abs_u_accepted_ += std::abs(u);
    repartition();
    return result;
  }

  tree_ = std::move(snapshot);
  ++ref_version_;
  set_mu(old_mu);
  pop_model_ = old_pop_model;
  coalescent_prior_.pop_model_changed(pop_model_);
  invalidate_derived_quantities();
  validate_derived_quantities();
  return result;
}

auto Run::set_history_gibbs_weight(double weight) -> void {
  if (not(weight >= 0.0) || not std::isfinite(weight)) {
    throw std::invalid_argument(absl::StrFormat(
        "The weight of the history Gibbs updates must be a finite non-negative number (got %g)", weight));
  }
  // The parts pick this up at the start of the next batch of local moves, exactly as they pick up
  // the other local-move settings; pushing it here would reset the coalescent parts mid-run.
  history_gibbs_weight_ = weight;
}

auto Run::set_history_gibbs_sites_per_move(int count) -> void {
  if (count < 1) {
    throw std::invalid_argument(absl::StrFormat(
        "A history Gibbs update must visit at least one site (got %d)", count));
  }
  history_gibbs_sites_per_move_ = count;
}

auto Run::set_marginal_node_time_weight(double weight) -> void {
  if (not(weight >= 0.0) || not std::isfinite(weight)) {
    throw std::invalid_argument(absl::StrFormat(
        "The weight of the marginal node-time move must be a finite non-negative number (got %g)",
        weight));
  }
  marginal_node_time_weight_ = weight;
}

auto Run::set_marginal_node_time_window(double window) -> void {
  if (not(window > 0.0) || not std::isfinite(window)) {
    throw std::invalid_argument(absl::StrFormat(
        "The window of the marginal node-time move must be a finite positive number (got %g)",
        window));
  }
  // A window at or above this bound lets one proposal change the root's depth by more than
  // `marginal_node_time_max_depth_factor`, where the move's grid guard would reject it.  That
  // rejection is asymmetric -- the reverse move from the rejected state is not rejected -- so such a
  // window would leave the kernel without detailed balance.  Refuse it rather than sample from the
  // wrong distribution.
  if (window >= max_marginal_node_time_window()) {
    throw std::invalid_argument(absl::StrFormat(
        "The window of the marginal node-time move must be smaller than %g, so that one proposal "
        "cannot change the root's depth by more than a factor of %g; a larger window would make the "
        "move's grid guard reject asymmetrically and break detailed balance (got %g)",
        max_marginal_node_time_window(), marginal_node_time_max_depth_factor, window));
  }
  marginal_node_time_window_ = window;
}

auto Run::set_site_history_refreshes_per_global_move(int count) -> void {
  if (count < 0) {
    throw std::invalid_argument(absl::StrFormat(
        "The number of whole-site history refreshes cannot be negative (got %d)", count));
  }
  site_history_refreshes_per_global_move_ = count;
}

// C3.  Redraws the whole history of a few sites over the whole tree: Felsenstein pruning for the one
// site, then the root state from the root prior times the root partial likelihood, then every
// internal state from the root down, then an exact conditional bridge on every branch.  Each refresh
// is a Gibbs step on the block of everything the site owns, so it is always accepted.
//
// This runs on the whole tree, so it has to happen between batches of local moves, and the parts have
// to be rebuilt from the changed tree afterwards.  The sites are drawn uniformly and independently of
// the current history.
auto Run::site_history_refresh_moves() -> void {
  if (site_history_refreshes_per_global_move_ <= 0) { return; }
  auto L = tree_.num_sites();
  if (L == 0) { return; }

  auto sampler = Site_history_sampler{evo_};
  for (auto i = 0; i != site_history_refreshes_per_global_move_; ++i) {
    auto l = absl::Uniform<Site_index>(absl::IntervalClosedOpen, bitgen_, 0, L);
    auto result = gibbs_refresh_site_history(tree_, l, sampler, bitgen_);
    ++history_gibbs_stats_.site_refreshes;
    history_gibbs_stats_.events_removed += result.num_old_events;
    history_gibbs_stats_.events_created += result.num_new_events;
    history_gibbs_stats_.node_states_changed += result.num_states_changed;
  }

  // The tree changed outside the parts, so the parts and every derived quantity are stale.
  invalidate_derived_quantities();
  repartition();
}

auto Run::set_site_partition(Site_partition_config config) -> void {
  if (mpox_hack_enabled_) {
    throw std::invalid_argument(
        "Site partitions cannot be combined with the mpox hack, which assigns its own two partitions from "
        "the sequence context of every site");
  }
  auto partition = resolve_site_partition(config, tree_.num_sites() - config.total_constant_sites(),
                                          observed_sites_of_tree(tree_));
  site_partition_config_ = std::move(config);
  site_partition_ = std::move(partition);
  site_partition_rates_.assign(site_partition_.num_partitions(), 1.0);
  evo_.partition_for_site = site_partition_.partition_for_site;
  ++site_layout_version_;
  evo_.partition_evo_model =
      Partition_vector<Site_evo_model>(site_partition_.num_partitions(), Site_evo_model{});
  invalidate_derived_quantities();
  validate_derived_quantities();  // the new partitions must be in `evo_` before anything reads it
}

auto Run::set_site_partition_rates(std::vector<double> rates) -> void {
  site_partition_rates_ = normalize_site_partition_rates(site_partition_, std::move(rates));
  invalidate_derived_quantities();
  validate_derived_quantities();
}

auto Run::set_site_partition_rate_prior_concentration(double concentration) -> void {
  if (not (std::isfinite(concentration) && concentration > 0.0)) {
    throw std::invalid_argument("The Dirichlet concentration of the site-partition rates must be finite "
                                "and positive");
  }
  site_partition_rate_prior_concentration_ = concentration;
  invalidate_derived_quantities();
}

auto Run::calc_cur_num_muts_per_partition() const -> std::vector<int> {
  auto result = std::vector<int>(evo_.num_partitions(), 0);
  for (const auto& node : index_order_traversal(tree_)) {
    if (node == tree_.root) { continue; }  // "mutations" above the root are deltas from the reference
    for (const auto& m : tree_.at(node).mutations) { ++result[evo_.partition_for_site[m.site]]; }
  }
  return result;
}

// One delta-exchange proposal on the length-weighted rate shares x_p = L_p r_p / sum_j L_j: x_p -> x_p + d,
// x_q -> x_q - d.  The shares live on the simplex, so the move has the P-1 free dimensions of the model and
// needs no Jacobian: it is a translation inside the simplex, and the proposal is symmetric because the pair
// and the sign are drawn uniformly.  Only the rates of p and q change --- the normalization is built into
// the parameterization rather than applied after the fact.
auto Run::site_partition_rate_move_with(
    Partition_index p, Partition_index q, double delta_x, double accept_draw)
    -> Site_partition_rate_move_result {
  validate_derived_quantities();
  auto num_muts_per_partition = calc_cur_num_muts_per_partition();

  auto result = Site_partition_rate_move_result{};
  result.p = p;
  result.q = q;
  result.delta_x = delta_x;
  result.log_alpha = -std::numeric_limits<double>::infinity();
  ++site_partition_rate_moves_proposed_;

  auto P = site_partition_.num_partitions();
  CHECK_GE(p, 0); CHECK_LT(p, P);
  CHECK_GE(q, 0); CHECK_LT(q, P);
  CHECK_NE(p, q);

  auto total_weight = static_cast<double>(site_partition_.total_observed_sites());
  auto weight = [&](Partition_index beta) {
    return static_cast<double>(site_partition_.num_observed_sites[beta]);
  };
  auto x_of = [&](Partition_index beta) {
    return weight(beta) * site_partition_rates_[beta] / total_weight;
  };

  auto old_x_p = x_of(p);
  auto old_x_q = x_of(q);
  auto new_x_p = old_x_p + delta_x;
  auto new_x_q = old_x_q - delta_x;
  if (not (new_x_p > 0.0 && new_x_p < 1.0 && new_x_q > 0.0 && new_x_q < 1.0)) { return result; }

  auto new_r_p = new_x_p * total_weight / weight(p);
  auto new_r_q = new_x_q * total_weight / weight(q);

  // log_G depends on the rate of partition beta through
  //   -mu r_beta * sum_a q_a Ttwiddle^beta_a + M_beta log(mu r_beta).
  auto mu = subst_model_.mu;
  auto exposure = [&](Partition_index beta) {
    auto total = 0.0;
    for (auto a : k_all_real_seq_letters) {
      total += evo_.partition_evo_model[beta].q_a(a) * Ttwiddle_beta_a_[beta][a];
    }
    return total;
  };
  auto delta_log_G = 0.0;
  for (auto [beta, old_r, new_r] : {std::tuple{p, site_partition_rates_[p], new_r_p},
                                    std::tuple{q, site_partition_rates_[q], new_r_q}}) {
    delta_log_G += -mu * (new_r - old_r) * exposure(beta)
        + num_muts_per_partition[beta] * std::log(new_r / old_r);
  }
  auto c = site_partition_rate_prior_concentration_;
  auto delta_log_other_priors = (c - 1.0) * (std::log(new_x_p / old_x_p) + std::log(new_x_q / old_x_q));

  result.delta_log_G = delta_log_G;
  result.delta_log_other_priors = delta_log_other_priors;
  result.log_alpha = delta_log_G + delta_log_other_priors;
  if (not std::isfinite(result.log_alpha)) { return result; }

  if (result.log_alpha >= 0.0 || accept_draw < std::exp(result.log_alpha)) {
    result.accepted = true;
    ++site_partition_rate_moves_accepted_;
    site_partition_rates_[p] = new_r_p;
    site_partition_rates_[q] = new_r_q;
    for (auto beta : {p, q}) {
      evo_.partition_evo_model[beta].mu = mu * site_partition_rates_[beta];
    }
    log_G_ += delta_log_G;
    log_other_priors_ += delta_log_other_priors;
  }
  return result;
}

auto Run::site_partition_rate_move() -> Site_partition_rate_move_result {
  auto P = site_partition_.num_partitions();
  CHECK_GE(P, 2);
  auto p = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0, P);
  auto q = p;
  while (q == p) { q = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0, P); }
  // A ladder of fixed step sizes, chosen uniformly: the shares of very unequal partitions differ by orders
  // of magnitude, and a single step size would be either always rejected or never moving.  The ladder does
  // not depend on the state, so the proposal stays symmetric.
  static constexpr double k_steps[] = {0.3, 0.1, 0.03, 0.01, 0.003, 0.001};
  auto step = k_steps[absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0,
                                    static_cast<int>(std::size(k_steps)))];
  auto delta_x = absl::Uniform(bitgen_, 0.0, step);
  auto accept_draw = absl::Uniform(bitgen_, 0.0, 1.0);
  return site_partition_rate_move_with(p, q, delta_x, accept_draw);
}

auto Run::site_partition_rate_moves() -> void {
  if (site_partition_.num_partitions() < 2) { return; }
  for (auto i = 0; i != 10; ++i) {
    site_partition_rate_move();
    check_derived_quantities();
  }
}

auto Run::mu_move() -> void {
  // We use a Gamma(mu_prior_alpha_, mu_prior_beta_) prior for mu.
  // The defaults alpha=1, beta=0 give the uniform prior used in LeMieux et al (2021).
  // For concreteness, we assume the mutation rates across all partitions are linked,
  // though the infrastructure is already here to allow for unlinked rates.
  //
  // We can Gibbs sample a new value of mu using a Gamma with
  // shape = M + mu_prior_alpha_ and rate = Ttwiddle + mu_prior_beta_,
  // where M = num_muts() and Ttwiddle = sum_{beta,a} q^[beta]_a * Ttwiddle^beta_a

  CHECK_GE(evo_.num_partitions(), 1);
  CHECK_EQ(evo_.num_partitions(), std::ssize(site_partition_rates_));
  for (auto beta = 0; beta != evo_.num_partitions(); ++beta) {
    CHECK_EQ(evo_.partition_evo_model[beta].mu, subst_model_.mu * site_partition_rates_[beta])
        << "The rate of a site partition must be mu times its relative rate";
  }

  // With the rate of partition beta equal to mu * r_beta, log_G depends on mu through
  //   -mu * sum_beta r_beta * sum_a q_a Ttwiddle^beta_a + M log(mu),
  // so mu is still Gibbs sampled, with the relative rates weighting the exposure.
  auto Ttwiddle = 0.0;
  for (auto beta = 0; beta != evo_.num_partitions(); ++beta) {
    const auto& evo_beta = evo_.partition_evo_model[beta];
    auto Ttwiddle_beta = 0.0;
    for (auto a : k_all_real_seq_letters) {
      Ttwiddle_beta += evo_beta.q_a(a) * Ttwiddle_beta_a_[beta][a];
    }
    Ttwiddle += site_partition_rates_[beta] * Ttwiddle_beta;
  }

  // WARNING: C++ calls "beta" the *scale* of the distribution, i.e., the reciprocal of the rate
  auto mu_dist = std::gamma_distribution<double>{
      num_muts_ + mu_prior_alpha_,
      1.0 / (Ttwiddle + mu_prior_beta_)};

  auto old_mu = subst_model_.mu;
  auto new_mu = mu_dist(bitgen_);
  subst_model_.mu = new_mu;
  for (auto beta = 0; beta != evo_.num_partitions(); ++beta) {
    evo_.partition_evo_model[beta] = subst_model_.derive_site_evo_model();
    evo_.partition_evo_model[beta].mu = new_mu * site_partition_rates_[beta];
  }
  log_G_ += -(new_mu - old_mu) * Ttwiddle + num_muts_ * std::log(new_mu / old_mu);
  log_other_priors_ += (mu_prior_alpha_ - 1) * std::log(new_mu / old_mu)
                       - mu_prior_beta_ * (new_mu - old_mu);
}

auto Run::mpox_hack_moves() -> void {
  // We use a Gamma(mu_prior_alpha_, mu_prior_beta_) prior for mu and a uniform prior for mu*.
  // The dependence of the posterior on mu and mu* conditional on everything else being fixed is thus:
  //
  // P ~ mu^{alpha-1} exp{-beta mu}
  //     * exp{-mu Ttwiddle + 2 mu* TTwiddle*} * (mu / 3)^(M-M*) * (mu / 3 + 2 mu*)^M*,
  //
  // where
  // - alpha = mu_prior_alpha_          // Shape parameter of Gamma prior on mu
  // - beta = mu_prior_beta_            // Rate parameter of Gamma prior on mu
  // - M^beta_ab = num_muts_beta_ab(),  // Number of a->b mutations in partition beta
  // - M = sum_{beta,a,b} M^beta_ab     // Total number of mutations
  // - M* = M^1_CT + M^1_GA             // APOBEC-like mutations in sites with APOBEC context
  // - Ttwiddle^beta_a = sum_{l in beta} nu^(l) T^(l)_a
  //                                    // Site-rate adjusted time spent by sites in partition beta in state a
  // - Ttwiddle = sum_{beta,a} Ttwiddle^beta_a  // Site-rate adjusted total branch length over all (N-pruned) site trees
  // - TTwiddle* = TTwiddle^1_C + Ttwiddle^1_G  // Site-rate adjusted time spent by APOBEC-context sites in C/G states
  //
  // Change variables to mu = mu and rho = mu* / mu.  The Jacobian |d(mu,rho)/d(mu,mu*)| is 1/mu, so
  //
  //  P(mu,rho) ~ e^{-(beta + Ttwiddle + 2 rho TTwiddle*) mu}
  //              * (mu / 3)^{M+alpha-1-1} * (1 + 6 rho)^M*.  // second -1 from Jacobian
  //
  //  It follows that
  //
  //         mu|rho ~ Gamma[M + alpha - 1, beta + Ttwiddle + 2 rho Ttwiddle*], and
  // (1 + 6 rho)|mu ~ Gamma[M* + 1, (mu / 3) * Ttwiddle*]  (subject to rho >= 0)
  //
  // Hence, we can Gibbs sample both mu and rho.

  // TODO: Always calculate this instead of M_ab
  auto M_beta_ab = calc_num_muts_beta_ab(tree_, evo_);
  auto M_ab = Seq_matrix<int>{0};
  for (auto beta = Partition_index{0}; beta != evo_.num_partitions(); ++beta) {
    for (auto a : k_all_real_seq_letters) {
      for (auto b : k_all_real_seq_letters) {
        M_ab[a][b] += M_beta_ab[beta][a][b];
      }
    }
  }
  auto M = num_muts_;  // == sum_{a,b} M_ab
  auto M_star =
      M_beta_ab[1][Real_seq_letter::C][Real_seq_letter::T] +
      M_beta_ab[1][Real_seq_letter::G][Real_seq_letter::A];

  static auto next_step_output = int64_t{-1};
  if (step_ >= next_step_output) {
    next_step_output = step_ + 1'000'000;
    
    std::cerr << "Mutations by type: \n";
    std::cerr << absl::StreamFormat(
        "- C>T = %d (of which %d = %g%% in APOBEC context)\n",
        M_ab[Real_seq_letter::C][Real_seq_letter::T],
        M_beta_ab[1][Real_seq_letter::C][Real_seq_letter::T],
        M_beta_ab[1][Real_seq_letter::C][Real_seq_letter::T] /
        double(M_ab[Real_seq_letter::C][Real_seq_letter::T]) * 100.0);
    std::cerr << absl::StreamFormat(
        "- G>A = %d (of which %d = %g%% in APOBEC context)\n",
        M_ab[Real_seq_letter::G][Real_seq_letter::A],
        M_beta_ab[1][Real_seq_letter::G][Real_seq_letter::A],
        M_beta_ab[1][Real_seq_letter::G][Real_seq_letter::A] /
        double(M_ab[Real_seq_letter::G][Real_seq_letter::A]) * 100.0);
    std::cerr << absl::StreamFormat("- A>C = %d\n", M_ab[Real_seq_letter::A][Real_seq_letter::C]);
    std::cerr << absl::StreamFormat("- A>G = %d\n", M_ab[Real_seq_letter::A][Real_seq_letter::G]);
    std::cerr << absl::StreamFormat("- A>T = %d\n", M_ab[Real_seq_letter::A][Real_seq_letter::T]);
    std::cerr << absl::StreamFormat("- C>A = %d\n", M_ab[Real_seq_letter::C][Real_seq_letter::A]);
    std::cerr << absl::StreamFormat("- C>G = %d\n", M_ab[Real_seq_letter::C][Real_seq_letter::G]);
    std::cerr << absl::StreamFormat("- G>C = %d\n", M_ab[Real_seq_letter::G][Real_seq_letter::C]);
    std::cerr << absl::StreamFormat("- G>T = %d\n", M_ab[Real_seq_letter::G][Real_seq_letter::T]);
    std::cerr << absl::StreamFormat("- T>A = %d\n", M_ab[Real_seq_letter::T][Real_seq_letter::A]);
    std::cerr << absl::StreamFormat("- T>C = %d\n", M_ab[Real_seq_letter::T][Real_seq_letter::C]);
    std::cerr << absl::StreamFormat("- T>G = %d\n", M_ab[Real_seq_letter::T][Real_seq_letter::G]);
  }
  
  auto Ttwiddle = 0.0;
  for (auto beta = Partition_index{0}; beta != evo_.num_partitions(); ++beta) {
    for (auto a : k_all_real_seq_letters) {
      Ttwiddle += Ttwiddle_beta_a_[beta][a];
    }
  }
  auto Ttwiddle_star =
      Ttwiddle_beta_a_[1][Real_seq_letter::C] +
      Ttwiddle_beta_a_[1][Real_seq_letter::G];

  // Do the above for 10 rounds to pseudo-Gibbs sample the tuple (mu, rho)
  for (auto i = 0; i != 10; ++i) {
    // WARNING: C++ calls "beta" the *scale* of the distribution, i.e., the reciprocal of the rate
    
    auto rho = mpox_mu_star_ / mpox_mu_;
    auto Ttwiddle_eff = Ttwiddle + 2 * rho * Ttwiddle_star;

    // mu
    if (mu_move_enabled_) {
      auto mu_dist = std::gamma_distribution<double>{
          M + mu_prior_alpha_ - 1.0,
          1.0 / (Ttwiddle_eff + mu_prior_beta_)};

      auto old_mu = mpox_mu_;
      auto new_mu = mu_dist(bitgen_);
      mpox_mu_ = new_mu;

      log_G_ += -(new_mu - old_mu) * Ttwiddle_eff + num_muts_ * std::log(new_mu / old_mu);
      log_other_priors_ += (mu_prior_alpha_ - 1) * std::log(new_mu / old_mu)
                           - mu_prior_beta_ * (new_mu - old_mu);
    }
      
    // rho
    if (Ttwiddle_star > 0.0) {
      // Use inverse transform sampling on k := 1 + 6 rho:
      //
      // p(k) ~ e^{-(mu Ttwiddle*/3) k} * k^M*, k >= 1
      // This is Gamma(M_star+1, mpox_mu_*Ttwiddle_star/3) truncated to [1, inf).
      auto k = safe_sample_truncated_gamma(
          M_star + 1, mpox_mu_*Ttwiddle_star/3, 1.0, std::numeric_limits<double>::infinity(),
          bitgen_);
      
      auto old_rho = rho;
      auto new_rho = (k - 1) / 6.0;
      mpox_mu_star_ = mpox_mu_ * new_rho;
      
      log_G_ += -2 * mpox_mu_ * (new_rho - old_rho) * Ttwiddle_star
          + M_star * std::log((1 + 6*new_rho) / (1 + 6*old_rho));
      log_other_priors_ += 0.0;
    }
  }

  // Update transition rate matrices
  derive_evo();
}

auto Run::set_subst_model(Subst_model model) -> void {
  model.mu = subst_model_.mu;
  model.validate();
  subst_model_ = std::move(model);
  invalidate_derived_quantities();
}

auto Run::hky_kappa() const -> double {
  CHECK(is_hky()) << "kappa is the transition rate of the HKY85 grouping 010010; the model is "
                  << subst_model_.grouping.code();
  return subst_model_.class_rate[subst_model_.grouping.class_of(1)];  // the class of A-G
}

auto Run::set_hky_kappa(double hky_kappa) -> void {
  CHECK(is_hky()) << "kappa is the transition rate of the HKY85 grouping 010010; the model is "
                  << subst_model_.grouping.code();
  subst_model_.class_rate[subst_model_.grouping.class_of(1)] = hky_kappa;
  invalidate_derived_quantities();
}

auto Run::set_subst_exchangeability_prior(
    double log_mean_transition, double log_mean_other, double sd) -> void {
  if (not (sd > 0.0) || not std::isfinite(sd)) {
    throw std::invalid_argument(absl::StrFormat(
        "The standard deviation of the log of the exchangeability prior is %g; it must be a finite "
        "positive number", sd));
  }
  if (not std::isfinite(log_mean_transition) || not std::isfinite(log_mean_other)) {
    throw std::invalid_argument("The mean of the log of the exchangeability prior must be finite");
  }
  subst_exchangeability_prior_log_mean_transition_ = log_mean_transition;
  subst_exchangeability_prior_log_mean_other_ = log_mean_other;
  subst_exchangeability_prior_sd_ = sd;
  invalidate_derived_quantities();
}

auto Run::subst_exchangeability_prior_log_mean_of(int k) const -> double {
  // A class of transitions only (A-G and C-T are exchangeabilities 1 and 4) carries the prior that
  // HKY85's kappa has always had; every other class is centered on the rate of G-T.
  auto all_transitions = true;
  for (auto index : subst_model_.grouping.members_of(k)) {
    if (index != 1 && index != 4) { all_transitions = false; }
  }
  return all_transitions ? subst_exchangeability_prior_log_mean_transition_
                         : subst_exchangeability_prior_log_mean_other_;
}

auto Run::subst_frequencies_move() -> void {
  // Adapting priors & moves from Gire et al 2014 BEAST file
  // - frequencies have a uniform prior
  // - frequencies moved by "delta-exchange", i.e., pick two frequencies, add d to one and -d to the other,
  //   where d is uniformly picked in [0, 0.01] (rejecting proposals of frequencies outside [0,1]).
  //
  // The delta exchange is a translation inside the simplex, so the proposal is symmetric and needs
  // neither a Hastings term nor a Jacobian.  Every site partition shares the frequencies, which is
  // why the mutation-type counts `num_muts_ab_` summed over the partitions are the right statistic.
  if (subst_model_.frequency_mode != Frequency_mode::estimated) { return; }
  validate_derived_quantities();

  // Make a frequency delta exchange
  auto freq_delta = 0.01;
  auto d = absl::Uniform(bitgen_, 0, freq_delta);
  auto ia = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0, std::ssize(k_all_real_seq_letters));
  auto ib = ia;
  while (ib == ia) {
    ib = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0, std::ssize(k_all_real_seq_letters));
  }
  auto move_a = *(k_all_real_seq_letters.begin() + ia);
  auto move_b = *(k_all_real_seq_letters.begin() + ib);

  auto old_model = subst_model_;
  auto new_model = old_model;
  new_model.pi_a[move_a] += d;
  if (new_model.pi_a[move_a] <= 0.0 || new_model.pi_a[move_a] >= 1.0) { return; }
  new_model.pi_a[move_b] -= d;
  if (new_model.pi_a[move_b] <= 0.0 || new_model.pi_a[move_b] >= 1.0) { return; }

  auto old_site_evo = old_model.derive_site_evo_model();  // == evo_.partition_evo_model[0];
  auto new_site_evo = new_model.derive_site_evo_model();

  auto delta_log_G = calc_delta_log_G_of_subst_model(old_site_evo, new_site_evo);
  if (not std::isfinite(delta_log_G)) { return; }

  auto log_p_acc = delta_log_G;
  if (log_p_acc > 0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_p_acc)) {
    // Accept
    apply_subst_model(std::move(new_model), new_site_evo);
    log_G_ += delta_log_G;
    log_other_priors_ += 0.0;
  } else {
    // Reject
  }
}

auto Run::subst_exchangeability_move(int free_class) -> void {
  // Adapting priors & moves from Gire et al 2014 BEAST file
  // - every free relative exchangeability has a log-normal prior; see
  //   subst_exchangeability_prior_log_mean_of
  // - the rate is scaled by a uniform factor from 0.75 to 1/0.75
  //    alpha(o -> n) = 1 / (o/0.75 - o*0.75)
  //    alpha(n -> o) = 1 / (n/0.75 - n*0.75)
  //    => alpha(n->o) / alpha(o->n) = o / n
  //
  // With the grouping 010010 and its single free class this is exactly the historical kappa move.
  DCHECK_NE(free_class, subst_model_.grouping.reference_class());
  validate_derived_quantities();

  auto scale_factor = 0.75;
  auto scale = absl::Uniform(bitgen_, scale_factor, 1.0 / scale_factor);

  auto old_model = subst_model_;
  auto new_model = old_model;
  new_model.class_rate[free_class] *= scale;

  auto old_site_evo = old_model.derive_site_evo_model();  // == evo_.partition_evo_model[0];
  auto new_site_evo = new_model.derive_site_evo_model();

  // log-normal prior:
  //  pi(s) ~ exp(-(log(s) - m)^2 / (2 sigma^2)) / s
  auto m = subst_exchangeability_prior_log_mean_of(free_class);
  auto sigma = subst_exchangeability_prior_sd_;
  auto old_s = old_model.class_rate[free_class];
  auto new_s = new_model.class_rate[free_class];
  auto alpha_n_to_o_over_o_to_n = old_s / new_s;
  auto prior_new_over_prior_old = std::exp(
      (-std::pow(std::log(new_s) - m, 2) - (-std::pow(std::log(old_s) - m, 2))) / (2 * sigma * sigma)
  ) * old_s / new_s;

  auto delta_log_G = calc_delta_log_G_of_subst_model(old_site_evo, new_site_evo);
  if (not std::isfinite(delta_log_G)) { return; }

  auto delta_log_other_priors = std::log(prior_new_over_prior_old);
  auto log_p_acc = delta_log_G + delta_log_other_priors + std::log(alpha_n_to_o_over_o_to_n);
  if (log_p_acc > 0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_p_acc)) {
    // Accept
    apply_subst_model(std::move(new_model), new_site_evo);
    log_G_ += delta_log_G;
    log_other_priors_ += delta_log_other_priors;
  } else {
    // Reject
  }
}

auto Run::subst_exchangeability_moves() -> void {
  // One proposal per free class, in class order.  With the grouping 010010 there is one free class,
  // so this is one historical kappa move and the stream of random numbers is unchanged.
  for (auto k = 0; k != subst_model_.grouping.num_classes(); ++k) {
    if (k == subst_model_.grouping.reference_class()) { continue; }
    subst_exchangeability_move(k);
  }
}

auto Run::calc_delta_log_G_of_subst_model(
    const Site_evo_model& old_site_evo,
    const Site_evo_model& new_site_evo) const -> double {

  // log_G depends on the rate matrix only through
  //   - the exposure  -sum_beta mu_beta sum_a q^beta_a Ttwiddle^beta_a,
  //   - the root state counts  sum_a n_a log pi_a, and
  //   - the mutation counts  sum_{a != b} M_ab log(mu_l nu_l q_ab),
  // so the whole difference is computed from the maintained statistics.  The factors of mu_l and
  // nu_l in the mutation term cancel because neither changes here.  A proposal that would put a
  // positive count on a zero rate has density zero and returns -infinity.
  auto delta_log_G = 0.0;

  // Branch lengths.  Every partition shares the matrix but has its own rate mu * r_beta.
  for (auto beta = 0; beta != evo_.num_partitions(); ++beta) {
    for (auto a : k_all_real_seq_letters) {
      delta_log_G -= new_site_evo.mu * site_partition_rates_[beta]
          * (new_site_evo.q_a(a) - old_site_evo.q_a(a)) * Ttwiddle_beta_a_[beta][a];
    }
  }

  // Root priors
  auto state_frequencies_at_root = state_frequencies_of_ref_sequence_;
  for (const auto& m : tree_.at_root().mutations) {
    --state_frequencies_at_root[m.from];
    ++state_frequencies_at_root[m.to];
  }
  // TODO: Treat gaps in one go
  for (const auto& [mi_site, mi_from] : tree_.at_root().missations.slow_elements(tree_.ref_sequence)) {
    --state_frequencies_at_root[mi_from];
  }
  for (auto a : k_all_real_seq_letters) {
    if (state_frequencies_at_root[a] > 0) {
      if (new_site_evo.pi_a[a] == 0) { return -std::numeric_limits<double>::infinity(); }
      delta_log_G += state_frequencies_at_root[a] * std::log(new_site_evo.pi_a[a] / old_site_evo.pi_a[a]);
    }
  }

  // Mutations
  for (auto a : k_all_real_seq_letters) {
    for (auto b : k_all_real_seq_letters) {
      if (a != b && num_muts_ab_[a][b] > 0) {
        if (new_site_evo.q_ab[a][b] == 0) { return -std::numeric_limits<double>::infinity(); }
        delta_log_G += num_muts_ab_[a][b] * std::log(new_site_evo.q_ab[a][b] / old_site_evo.q_ab[a][b]);
      }
    }
  }

  return delta_log_G;
}

auto Run::apply_subst_model(Subst_model model, const Site_evo_model& site_evo) -> void {
  subst_model_ = std::move(model);
  for (auto beta = 0; beta != evo_.num_partitions(); ++beta) {
    evo_.partition_evo_model[beta] = site_evo;
    evo_.partition_evo_model[beta].mu = site_evo.mu * site_partition_rates_[beta];
  }
}

auto Run::gibbs_sample_all_nus() -> void {
  validate_derived_quantities();
  // We can Gibbs sample a new value of nu_l using a Gamma with
  // shape alpha = M^(l) + alpha and rate beta = mu^(l) * TTwiddle^(l) + alpha

  auto Ttwiddle_l = calc_Ttwiddle_l(tree_, evo_);
  auto M_l = calc_num_muts_l(tree_);
  auto old_sum_nu = 0.0;
  auto new_sum_nu = 0.0;

  for (auto l = Site_index{0}; l != tree_.num_sites(); ++l) {
    // WARNING: C++ calls "beta" the *scale* of the distribution, i.e., the reciprocal of the rate
    auto nu_l_dist = std::gamma_distribution<double>{
      M_l[l] + alpha_,
      1.0 / (evo_.mu_l(l) * Ttwiddle_l[l] + alpha_)};
    
    auto old_nu_l = nu_[l];

    // CAREFUL! When M_l[l] == 0 and alpha is very low, nu_l_dist is very concentrated
    // very close to zero, to the point where random samples can come out as *exactly*
    // zero owing to roundoff.  Those zeros then appear in logs (e.g., below), which
    // wreaks havoc.  If nu_[l] is very close to zero, site l is effectively invariant,
    // and no mutation will be ascribed to it.  Hence, the contribution to log_G is
    // essentially zero, and it doesn't matter what the exact value is.  So we cop out
    // here and put a crazy but nonzero lower bound `eps` on nu_[l].  Technically, we
    // should then adjust the nu prior to distinguish between definite values above `eps`
    // and "anything below `eps`".  Since the distinction has no effect on the sampled
    // trees and remaining parameters, just on the reporting of the posterior values, we
    // purposely keep things simple and don't make the above correction.
    //
    // A more consistent approach would be to allow for some sites to be honest-to-god
    // invariant and *then* demand that the noninvariant sites have a relative site rate
    // above eps.  One day, one day...
    //
    // Thanks to David Pascall at the Cambridge Infectious Diseases centre for
    // reporting and helping debug this issue.
    auto new_nu_l = std::max(1e-50, nu_l_dist(bitgen_));  // eps = 1e-50
    nu_[l] = new_nu_l;
    
    auto log_new_over_old_nu_l = std::log(new_nu_l / old_nu_l);
    log_G_ += -evo_.mu_l(l) * (new_nu_l - old_nu_l) * Ttwiddle_l[l] + M_l[l] * log_new_over_old_nu_l;

    old_sum_nu += old_nu_l;
    new_sum_nu += new_nu_l;
    log_other_priors_ += (alpha_ - 1) * log_new_over_old_nu_l;
  }

  log_other_priors_ += -alpha_ * (new_sum_nu - old_sum_nu);
  ++nu_version_;

  evo_.nu_l = nu_;  // TODO: No need to store this twice & copy here
  evo_nu_l_version_ = nu_version_;
  Ttwiddle_beta_a_ = calc_cur_Ttwiddle_beta_a();
}

static auto calc_log_p_alpha(
    const Global_evo_model& evo,
    double alpha,
    const Node_vector<double>& Ttwiddle_l,
    const Node_vector<int>& M_l)
    -> double {
  
  // Excluding the prior pi(alpha), we have,
  //
  //  log p(alpha) = \sum_{l=0}^L (loggamma(M^(l) + alpha) - (M^(l) + alpha) log(mu^(l) Ttwiddle^(l) + alpha))
  //                          - L (loggamma(        alpha) -          alpha  log(                      alpha))

  auto L = std::ssize(M_l);
  auto result = 0.0;
  auto num_sites_with_muts = 0;
  for (auto l = Site_index{0}; l != L; ++l) {
    if (M_l[l] > 0) {
      ++num_sites_with_muts;
      result += std::lgamma(M_l[l] + alpha);
    }
    result -= (M_l[l] + alpha) * std::log(evo.mu_l(l) * Ttwiddle_l[l] + alpha);
  }
  result -= num_sites_with_muts * std::lgamma(alpha) - L * alpha * std::log(alpha);
  return result;
}

auto Run::calc_cur_log_p_alpha(double alpha) const -> double {
  validate_derived_quantities();
  auto Ttwiddle_l = calc_Ttwiddle_l(tree_, evo_);
  auto M_l = calc_num_muts_l(tree_);
  return calc_log_p_alpha(evo_, alpha, Ttwiddle_l, M_l);
}

auto Run::alpha_moves() -> void {
  validate_derived_quantities();
  auto Ttwiddle_l = calc_Ttwiddle_l(tree_, evo_);
  auto M_l = calc_num_muts_l(tree_);

  // exponential prior for alpha:
  //  pi(alpha) ~ exp(-alpha / mean_alpha)
  auto mean_alpha = 1.0;
  
  auto alpha_before_moves = alpha_;
  auto cur_log_p_alpha = calc_log_p_alpha(evo_, alpha_, Ttwiddle_l, M_l);
  for (auto sub_move = 0; sub_move != 10; ++sub_move) {
    // We follow LeMieux et al 2021 in using an exponential prior with mean 1.0
    // and scaling moves with scale factor 0.90 (see hky_move kappa moves for a detailed description)
    
    auto scale_factor = 0.90;
    auto old_alpha = alpha_;
    auto scale = absl::Uniform(bitgen_, scale_factor, 1 / scale_factor);
    auto new_alpha = scale * old_alpha;
    auto alpha_n_to_o_over_o_to_n = old_alpha / new_alpha;
    
    auto log_prior_new_over_prior_old = -(new_alpha - old_alpha) / mean_alpha;
    
    // Calculate change in `log p(alpha)` and acceptance probability
    auto new_log_p_alpha = calc_log_p_alpha(evo_, new_alpha, Ttwiddle_l, M_l);
    
    auto delta_log_posterior = log_prior_new_over_prior_old + new_log_p_alpha - cur_log_p_alpha;
    auto log_metropolis = delta_log_posterior + std::log(alpha_n_to_o_over_o_to_n);
    if (log_metropolis > 0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_metropolis)) {
      // Accept (no change to log_G)
      alpha_ = new_alpha;
      cur_log_p_alpha = new_log_p_alpha;
    }
  }
  
  // See calc_cur_log_other_priors() for discussion on alpha & nu_l priors
  auto L = tree_.num_sites();
  auto sum_nu = 0.0;
  auto sum_log_nu = 0.0;
  for (auto nu_l : nu_) {
    DCHECK_NE(nu_l, 0.0);
    sum_nu += nu_l;
    sum_log_nu += std::log(nu_l);
  }
  log_other_priors_ += 0.0
      + -(alpha_ - alpha_before_moves) / mean_alpha
      + L * (alpha_ * std::log(alpha_) - alpha_before_moves * std::log(alpha_before_moves))
      - L * (std::lgamma(alpha_) - std::lgamma(alpha_before_moves))
      + (alpha_ - alpha_before_moves) * sum_log_nu
      - (alpha_ - alpha_before_moves) * sum_nu;
  
  // IMPORTANT: Since alpha has changed, all the values of nu_l need to be Gibbs sampled!
  gibbs_sample_all_nus();
}

auto Run::pop_size_move() -> void {
  const Pop_model& raw_pop_model = *pop_model_;
  CHECK(typeid(raw_pop_model) == typeid(Exp_pop_model));
  auto exp_pop_model = std::static_pointer_cast<const Exp_pop_model>(pop_model_);
    
  // We use an Inverse-Gamma(alpha, beta) prior on n0:
  //   pi(n0) ~ n0^{-(alpha+1)} * exp(-beta / n0)
  // The default alpha=0, beta=0 gives the 1/x prior used in LeMieux et al (2021).
  // A scale operator with scale factor [0.75, 1/0.75] proposes new_n0 = scale * old_n0.
  //
  //   log pi(new) - log pi(old) = -(alpha + 1) * log(scale) - beta * (1/new_n0 - 1/old_n0)

  ++pop_size_proposed_;
  auto scale_factor = 0.75;
  auto old_n0 = exp_pop_model->pop_at_t0();
  auto scale = absl::Uniform(bitgen_, scale_factor, 1 / scale_factor);
  auto new_n0 = scale * old_n0;
  auto alpha_n_to_o_over_o_to_n = old_n0 / new_n0;

  auto log_prior_new_over_prior_old =
      -(pop_inv_n0_prior_alpha_ + 1) * std::log(scale)
      - pop_inv_n0_prior_beta_ * (1.0 / new_n0 - 1.0 / old_n0);
    
  auto old_log_coal = log_coalescent_prior_ + log_coal_exact_correction_;  // the target's coalescent term
  auto old_pop_model = exp_pop_model;
  pop_model_ = std::make_shared<Exp_pop_model>(old_pop_model->t0(), new_n0, old_pop_model->growth_rate(), old_pop_model->min_pop());
  coalescent_prior_.pop_model_changed(pop_model_);
  auto new_log_coal = calc_cur_log_coalescent_prior();
  auto new_correction = exact_coalescent_correction_ ? calc_cur_log_coalescent_exact() - new_log_coal : 0.0;
    
  auto log_metropolis =
      ((new_log_coal + new_correction) - old_log_coal) + log_prior_new_over_prior_old + std::log(alpha_n_to_o_over_o_to_n);
  if (log_metropolis > 0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_metropolis)) {
    // Accept
    ++pop_size_accepted_;
    log_coalescent_prior_ = new_log_coal;
    log_coal_exact_correction_ = new_correction;
    log_other_priors_ += log_prior_new_over_prior_old;
  } else {
    // Reject
    pop_model_ = old_pop_model;
    coalescent_prior_.pop_model_changed(pop_model_);
  }
}

auto Run::pop_growth_rate_move() -> void {
  const Pop_model& raw_pop_model = *pop_model_;
  CHECK(typeid(raw_pop_model) == typeid(Exp_pop_model));
  auto exp_pop_model = std::static_pointer_cast<const Exp_pop_model>(pop_model_);
    
  // Truncated Laplace prior on g with optional bounds [pop_g_min_, pop_g_max_]:
  //   pi(g) ~ exp(-|g - mu| / scale) * I(g_min <= g <= g_max)
  // By default (following LeMieux et al (2021)), mu = 0.001/365, scale = 30.701135/365,
  // and no bounds.
  // Random walk operator: uniform perturbation in [-delta, +delta], delta = 1.0 / 365.0.

  ++pop_growth_proposed_;
  auto window_size = pop_g_move_window_;
  auto old_g = exp_pop_model->growth_rate();
  CHECK(old_g >= pop_g_min_ && old_g <= pop_g_max_)
      << "Growth rate " << old_g << " outside bounds [" << pop_g_min_ << ", " << pop_g_max_ << "]";

  auto delta = absl::Uniform(bitgen_, -window_size, +window_size);
  auto new_g = old_g + delta;

  // Reject if outside bounds
  if (new_g < pop_g_min_ || new_g > pop_g_max_) { ++pop_growth_out_of_bounds_; return; }

  auto log_prior_new_over_prior_old =
      (std::abs(old_g - pop_g_prior_mu_) - std::abs(new_g - pop_g_prior_mu_)) / pop_g_prior_scale_;
    
  auto old_log_coal = log_coalescent_prior_ + log_coal_exact_correction_;  // the target's coalescent term
  auto old_pop_model = exp_pop_model;
  pop_model_ = std::make_shared<Exp_pop_model>(old_pop_model->t0(), old_pop_model->pop_at_t0(), new_g, old_pop_model->min_pop());
  coalescent_prior_.pop_model_changed(pop_model_);
  auto new_log_coal = calc_cur_log_coalescent_prior();
  auto new_correction = exact_coalescent_correction_ ? calc_cur_log_coalescent_exact() - new_log_coal : 0.0;
    
  auto log_metropolis = ((new_log_coal + new_correction) - old_log_coal) + log_prior_new_over_prior_old;
  if (log_metropolis > 0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_metropolis)) {
    // Accept
    ++pop_growth_accepted_;
    log_coalescent_prior_ = new_log_coal;
    log_coal_exact_correction_ = new_correction;
    log_other_priors_ += log_prior_new_over_prior_old;
  } else {
    // Reject
    pop_model_ = old_pop_model;
    coalescent_prior_.pop_model_changed(pop_model_);
  }
}

auto Run::skygrid_tau_move() -> void {
  const Pop_model& raw_pop_model = *pop_model_;
  CHECK(typeid(raw_pop_model) == typeid(Skygrid_pop_model));
  auto skygrid_pop_model = std::static_pointer_cast<const Skygrid_pop_model>(pop_model_);

  // If all other parameters are fixed, tau follows a Gamma distribution:
  //
  // P(tau) \propto tau^(alpha-1) exp(-beta tau)
  //                * tau^(M/2) exp(-tau sum_squared_delta_gammas/2)
  //
  // => tau ~ Gamma(alpha + M/2, beta + sum_squared_delta_gammas/2)
  //
  // We can Gibbs sample that directly

  auto M = skygrid_pop_model->M();
  
  auto sum_squared_delta_gammas = 0.0;
  for (auto k = 1; k <= M; ++k) {
    auto delta_gamma = skygrid_pop_model->gamma(k) - skygrid_pop_model->gamma(k-1);
    sum_squared_delta_gammas += std::pow(delta_gamma, 2);
  }

  auto post_alpha = skygrid_tau_prior_alpha_ + 0.5 * M;
  auto post_beta = skygrid_tau_prior_beta_ + 0.5 * sum_squared_delta_gammas;

  // WARNING: C++ calls "beta" the *scale* of the distribution, i.e., the reciprocal of the rate
  auto tau_dist = std::gamma_distribution<double>{
      post_alpha,
      1.0 / post_beta};
  
  auto old_tau = skygrid_tau_;
  auto new_tau = tau_dist(bitgen_);
  skygrid_tau_ = new_tau;
  
  log_other_priors_ +=
      (post_alpha - 1) * std::log(new_tau/old_tau)
      - post_beta * (new_tau - old_tau);
}

auto Run::skygrid_gammas_hmc_move() -> void {
  // This move uses Hamiltonian Monte Carlo (HMC) to perform effective collective moves of the
  // Skygrid population curve.  The curve is specified by a set of values gamma_k (0<=k<=M),
  // so that N(x_k) = exp(gamma_k) at predefined "knot" times x_k. At intermediate times, the
  // population curve is interpolated in some reasonable way; and before the first knot and
  // after the last knot, the curve is likewise extrapolated (see pop_model.h for exact
  // details).
  //
  // The dependence of the log-posterior on the {gamma_k} is complicated but tractable:
  //
  //   log P = const                              <-- terms independent of {gamma_k}
  //   
  //           - \sum_{c=0}^{C-1} (1/2) Delta k_c (k_c-1) / N_c       \              |
  //                                                                  +---  -U_coal  |
  //           - \sum_{i inner node} log(N(t_i))                      /              |
  //                                                                                 |
  //           - (tau/2) \sum_{k=1}^M (gamma_k - gamma_{k-1})^2       \              |
  //                                                                  |              |
  //           - \sum_{k=0}^M [[ gamma_k < gamma_min ]]               +---  -U_prior |
  //                 ((gamma_k - gamma_min) / Delta_gamma)^2          |              |
  //                                                                  |              |
  //           - alpha * gamma_bar - beta * exp(-gamma_bar)          /               |
  //
  // Here, c ranges over the C cells of width Delta in the scalable coalescent,
  //       k_c is the average number of active lineages in cell c,
  //       N_c is the average of N(t) over cell c,
  //       gamma_min is a minimum reasonable value for gamma_k,
  //       Delta_gamma is the scale at which deviations of gamma_k below gamma_min
  //          reduce log P by 1 in log P,
  //       gamma_bar = (1/(M+1)) sum_{k=0}^M gamma_k is the mean of the gamma_k's,
  //       alpha and beta are the parameters of the InvGamma prior on N_bar = exp(gamma_bar)
  //
  // A straightforward implementation of HMC (e.g., similar to Baele et al 2020
  // doi:10.12688/wellcomeopenres.15770.1, used in BEAST X for Skygrid with HMC) regards the
  // {gamma_k}'s as the "positions" of particles in a dynamical system, augments them with
  // momenta variable {p_k} of some arbitrary mass, then sets up a Hamiltonian and derives
  // equations of motion for those variables.  If every gamma_k gets assigned an equal mass,
  // the free system (U_coal = 0) looks like a vibrating string-of-beads with free ends whose
  // normal modes oscillate at very different frequencies: long modes (which are interesting
  // for N(t)!) are slow, while short modes (which are essentially noise on top of N(t)) are
  // much faster.  This is unfortunate because the HMC time step then needs to be small enough
  // to be a fraction of the period of the fastest mode.  That results in a lot of calculation
  // effort being spent on tracking noisy and irrelevant short-wavelength wiggles on top of
  // N(t).
  //
  // An alternative implementation that first Fourier-transforms the {gamma_k}'s into
  // normal mode amplitudes {A_s} and sets up a dynamical system in _those_ coordinates
  // can be configured with different per-mode "masses" such that all the normal modes
  // of the free system oscillate at the same frequency.  That would allow much larger
  // timesteps in the _free_ system: because the irrelevant short-wavelength vibrations
  // are slowed down while the important long-wavelength vibrations are sped up,
  // computational effort is focused where it matter most.  The details and code for such
  // an HMC setup are in the commit history.
  //
  // Sadly, while the above alternative frequency-space HMC seems ideal in the _free_ system
  // (U_coal = 0), it performs poorly once the effects of the actual coalescent prior are
  // included.  After lots of experimentation, the intuition is that while motion of the
  // {gamma_k}'s is dominated by U_prior in time intervals where there is little information,
  // it is dominated by the effects of U_coal in time intervals where there are lots of
  // coalescences and/or active branches.
  //
  // The above experiments suggested a very effective compromise: a real-space HMC, but with
  // different masses for different gamma_k's.  To get an intuition for what those masses
  // should be in the case of the (standard) staircase Skygrid population curve.  Following
  // the notation from Gill et al 2012, we rewrite log P as follows:
  //
  //   log P = const
  //
  //           - \sum_{k=0}^M SS_k exp(-gamma_k)                 <-- Note k=0 lower bound
  //
  //           - \sum_{k=0}^M c_k gamma_k                        <-- Note k=0 lower bound
  //
  //           - \sum_{k=1}^M (tau/2) (gamma_k - gamma_{k-1})^2. <-- Note k=1 lower bound
  //
  //           (--- terms from low-gamma_k guard and InvGamma prior on N_bar omitted ---)
  //
  // Here,
  //
  //   SS_k = int_{y_{k-1}}^{y_k} dt (1/2) k_c(t) (k_c(t) - 1), and
  //    c_k = \sum_{i inner} [[ y_{k-1} < t_i <= y_k ]],
  //
  // while
  //
  //          /   -infinity,         k < 0;      |  y_{k-1} < t <= y_k spans what is
  //    y_k = |   x_k,          0 <= k < M;      |  called "Interval k" in pop_model.h,
  //          \   +infinity,    M <  k.          |  but Interval M is extended to +infinity.
  //
  // Roughly speaking, c_k counts the number of coalescence event in interval k, while SS_k
  // counts something like an ``effective'' elapsed time in interval k.  So we expect the
  // average coalescence timescale (~ N(t)) in interval k to be something like SS_k / c_k
  // (and, in the absence of a GMRF prior, it would be exactly that).  We will see this
  // expectation realized below.
  //
  // Ideally, we'd then maximize `log P` with respect to {gamma_k}, calculate the Hessian at
  // that minimum, and apply a Gaussian approximation to identify the natural normal modes and
  // frequencies of the full system, and then implement HMC in terms of _those_ normal mode
  // amplitudes (indeed, this closely tracks what the ``block update'' move in the original
  // Gill et al 2012 paper does).  Actually realizing this idea is challenging and expensive.
  // Moreover, it would only work exactly for the staircase population curve.  So instead, we
  // recover most of the intuition yielded by the above approach with a much simpler
  // approximation that maintains a diagonal mass matrix.
  //
  // To proceed, consider just the dependence of log P on a particular gamma_k, we get
  //
  //   log P = const
  //           - SS_k exp(-gamma_k)
  //           - c_k gamma_k
  //           - (tau/2) (gamma_k - gamma_{k-1})^2 [[ k > 0 ]]
  //           - (tau/2) (gamma_k - gamma_{k+1})^2 [[ k < M ]].
  //           (--- terms from low-gamma_k guard and InvGamma prior on N_bar omitted ---)
  //
  // In the limit of large M, we can suppose that gamma_{k-1} =~ gamma_k =~ gamma_{k+1}, but
  // their small oscillations around these equal values will matter.  In that case, ignoring
  // the low-gamma_k guard and focusing again on a specific k,
  //
  //   log P =~ const
  //           - SS_k exp(-gamma_k)
  //           - c_k gamma_k
  //
  //  while
  //
  //   del(log P)/del gamma_k =~
  //           + SS_k exp(-gamma_k)
  //           - c_k
  //           - tau (gamma_k - gamma_{k-1}) [[ k > 0 ]]  \  these terms are terms
  //           - tau (gamma_k - gamma_{k+1}) [[ k < M ]]  /  are essentially zero
  //
  // and
  //
  //   del^2(log P)/del gamma_k^2 =~
  //           - SS_k exp(-gamma_k)
  //           - tau [[ k > 0 ]]                          \  independent of
  //           - tau [[ k < M ]]                          /  {gamma_k} !
  //
  // Neglecting the last two terms in del(log P)/del gamma_k places the maximum of log P at
  // a value gammahat_k satisfying:
  //
  //   exp(-gammahat_k) =~ c_k / SS_k
  //    =>  gammahat_k  =~ -log(c_k / SS_k).
  //
  // (Note: in the staircase model,
  //
  //   N(t in interval k) = exp(gamma_k) =~ SS_k/c_k,
  //
  // as alluded to above).
  //
  // Hence, we have our key result:
  //
  //   del^2(log P)/del gamma_k^2 =~ -(tau [[ k > 0 ]] + tau [[ k < M ]] + c_k).
  //
  // Compare this to a Gaussian distribution:
  //
  //   log P_G(gamma_k) = -(gamma_k - gammahat_k)^2 / (2 sigma^2),
  //   => del^2(log P_G)/del gamma_k^2 = -1 / sigma^2.
  //
  // Hence, the typical width of the posterior along the gamma_k direction near its
  // minimum is something like 1/sqrt(tau [[ k > 0 ]] + tau [[ k < M ]] + c_k).  This
  // isn't strictly true owing to the coupling between the neighbouring gamma_k's, and
  // it's very possible that the width of the posterior along other directions is much
  // larger, but this gives us a reasonable starting point with a physically pleasing
  // interpretation:
  //
  //   "In intervals where there is little data (c_k << tau), oscillations of the
  //    population curve are driven by the GMRF prior.  In intervals where there is
  //    a lot of data (c_k >> tau), oscillations of the population curve are driven
  //    by the coalescent prior."
  //
  // As we will see below, setting the mass m_k associated with variable gamma_k to
  //
  //    m_k = tau [[ k > 0 ]] + tau [[ k < M ]] + c_k
  //
  // results in all oscillations of the dynamical system having very similar angular
  // frequencies close to 1, which permits picking a natural time step and trajectory length.
  // Empirically, the above choice of m_k makes the population curve move slowly in intervals
  // with lots of data while allowing much larger and faster oscillations in intervals with
  // little data.  This behaviour makes the HMC moves quite effective at exploring the
  // posterior distribution of {gamma_k}.
  //
  // -------
  //
  // Now for the nitty-gritty.  We first introduce the (reduced) potential
  // energy U:
  //
  //    U := U_coal + U_prior.
  //
  // We augment this system with momentum variables {p_k} having kinetic energy K:
  //
  //    K := sum_{k=0}^M p_k^2 / (2 m_k),
  //
  // At equilibrium, we have:
  //
  //    p_k ~ N(0, m_k).
  //
  // With these energies in place, we obtain the following Hamiltonian H({gamma_k}, {p_k})
  // and associated equations of motion:
  //
  //    H = U + K,
  //    d gamma_k/dt = + del H / del p_k     = p_k / m_k,
  //    d p_k    /dt = - del H / del gamma_k = - del U / del gamma_k =: f_k;
  //
  // i.e.,
  //
  //    d^2 gamma_k/dt^2 = -(f_k / m_k).
  //
  // We integrate these equations of motion using a slightly nonstandard position Verlet
  // integrator over S steps of size dt:
  //
  //    gamma_k(t + dt/2) = gamma_k(t)        + (dt/2) (p_k(   t    ) / m_k),
  //        p_k(t + dt)   =     p_k(t)        +   dt    f_k(t + dt/2)
  //    gamma_k(t + dt)   = gamma_k(t + dt/2) + (dt/2) (p_k( t + dt ) / m_k).
  //
  // (Baele et al 2020 [and almost everyone else, for that matter] use a standard velocity
  // Verlet integrator, as follows:
  //
  //        p_k(t + dt/2) =     p_k(t)        + (dt/2) f_k(t);
  //    gamma_k(t + dt)   = gamma_k(t)        +   dt  (p_k(t + dt/2) / m_k);
  //        p_k(t + dt)   =     p_k(t + dt/2) + (dt/2) f_k(t + dt).
  //
  // The advantage of using the position Verlet integrator above is there is one fewer force
  // evaluation overall, which is a little bit helpful in the typically short HMC trajectories
  // we use.  Otherwise, velocity and position Verlet are both reversible and have very
  // similar energy conservation properties, with position Verlet possibly a little better
  // for large timestemp [see Tuckerman, Berne and Martyna 1992, doi: 10.1063/1.463137]).
  //
  // The two gaps remaining are:
  // (a) how to calculate f_k?
  // (b) how to choose m_k, dt and S?
  //
  // The second question is easier to answer given the above discussion.  As noted above,
  // near the maximum of log P at {gammahat_k}, for 0 < k < M, we'll have something like
  //
  //   d^2 (gamma_k - gammahat_k) / dt^2 = - (gamma_k - gammahat_k) * (2 tau + c_k) / m_k.
  //                                       \                                       /
  //                                        +-------------- ~ f_k ----------------+
  //
  // Hence, setting
  //
  //   m_k = tau [[ k > 0 ]] + tau [[ k < M ]] + c_k
  //
  // makes gamma_k behave as a harmonic oscillator about gammahat_k, with angular frequency 1.
  // This is not exactly what happens, but it's close enough to be a useful guide to tuning
  // masses and timesteps.
  //
  // For stability and to compensate for the approximate nature of the above argument, we'd
  // like the position Verlet integrator to trace out a full period in about 100 steps,
  // which suggests using:
  //
  //    dt =? 2 pi / 100.
  //
  // If we ran the position Verlet integrator for exactly one period = 100 steps, then to a
  // first approximation, all {gamma_k}'s would end at the same place where they started,
  // regardless of the starting momenta!  Moreover, while the above estimate for dt should
  // work reasonably well in most cases, we risk getting stuck in a region of parameter space
  // where forces are very high and the timestep needs to be substantially lower to improve
  // integration accuracy.
  //
  // We kill two birds with one stone by implementing the so-called Randomized HMC rule of
  // making the total integration time random with an exponential distribution (Bou-Rabee and
  // Maria Sanz-Serna 2017, doi:10.1214/16-AAP1255).  Concretely, instead of picking dt
  // deterministically, we draw a sample from an exponential distribution.  Then we sample for
  // exactly 25 steps; this is exactly 1/4 of a period if dt assumes its mean value and the
  // approximations above are accurate, where we expect the variables to differ most from
  // their initial values:
  //
  //    dt ~ Expo(100 / 2 pi),
  //     S = 25.
  //
  // Note that because this choice is symmetric between the forward and reverse moves, it does
  // not induce any correction terms in the acceptance probabilities below.
  //
  // Incidentally, note that setting the actual value of the angular frequencies to 1 is
  // irrelevant.  Had we chosen a different value for it and scaled {m_k} and dt accordingly,
  // the generated {gamma_k} trajectories would be unchanged.
  //
  // Now for forces.  We have
  //
  //   f_k := -(del U / del gamma_k),
  //
  // where
  //
  //  U = + \sum_{c=0}^{C-1} (1/2) Delta k_c (k_c-1) / N_c
  //      + \sum_{i inner node} log(N(t_i))
  //      + (tau/2) \sum_{k=1}^M (gamma_k - gamma_{k-1})^2
  //      + \sum_{k=0}^M [[ gamma_k < gamma_min ]] ((gamma_k - gamma_min) / Delta_gamma)^2
  //      + alpha * gamma_bar + beta * exp(-gamma_bar).
  //
  // Hence,
  //
  //   f_k = + \sum_{c=0}^{C-1} (1/2) Delta (k_c (k_c-1) / N_c^2) (del N_c / del gamma_k)
  //           - \sum_{i inner node} [del log(N(t)) / del gamma_k]_{t = t_i}
  //           - tau (  (gamma_k - gamma_{k-1}) [[ k > 0 ]]
  //                  + (gamma_k - gamma_{k+1}) [[ k < M ]] )
  //           - \sum_{k=0}^M [[ gamma_k < gamma_min ]] 2 (gamma_k - gamma_min)/Delta_gamma^2
  //           + (-alpha + beta * exp(-gamma_bar)) / (M+1).
  //
  //       = + \sum_{c=0}^{C-1} (1/2) Delta (k_c (k_c-1) / N_c) (del log(N_c) / del gamma_k)
  //           - \sum_{i inner node} [del log(N(t)) / del gamma_k]_{t = t_i}.
  //           - tau (  (gamma_k - gamma_{k-1}) [[ k > 0 ]]
  //                  + (gamma_k - gamma_{k+1}) [[ k < M ]] )
  //           - \sum_{k=0}^M [[ gamma_k < gamma_min ]] 2 (gamma_k - gamma_min)/Delta_gamma^2
  //           + (-alpha + beta * exp(-gamma_bar)) / (M+1).
  //
  // In the first sum, only a few terms are non-zero for a given k: these are the cells
  // that overlap with Pop_model::support_of_d_log_N_d_gamma.  The key factors
  // are then given by Pop_model::d_log_int_N_d_gamma
  // and Pop_model::d_log_N_d_gamma, respectively.

  // TODO: The code below could be significantly optimized:
  // - mutating population models in-place to remove allocations
  // - replacing `sum_{i inner} 1/N(t_i)` in the coalescent prior by `sum_c (1/N_c)^{n_c},
  //   where n_c is the number of tips in interval `k`.
  // - merging the position half-steps of consecutive steps
  // - identifying k_min = the largest k where x_k < t_root,
  //   evolving the pop curve just for k_min <= k <= M, then extending the pop curve down
  //   to k = 0 by Gibbs sampling.
 
  const Pop_model& raw_pop_model = *pop_model_;
  CHECK(typeid(raw_pop_model) == typeid(Skygrid_pop_model));
  auto old_pop_model = std::static_pointer_cast<const Skygrid_pop_model>(pop_model_);
  auto new_pop_model = old_pop_model;

  auto M = old_pop_model->M();

  auto Delta = coalescent_prior_.t_step();
  const auto& k_c = coalescent_prior_.k_bars();
  const auto& N_c = coalescent_prior_.popsize_bars();
  auto C = static_cast<int>(std::ssize(k_c));
  auto tau = skygrid_tau_;

  // With the exact Kingman correction on, U_coal is the exact density rather than the cell prior.
  // The genealogy does not change inside this move, so its event intervals are built once and the
  // whole trajectory -- values, forces and the final acceptance -- reads the exact density off them.
  // The cost of one evaluation is then linear in the number of nodes plus knots and does not depend
  // on the cell count of the coalescent grid at all, so the grid is refreshed only once, at the end.
  auto exact_events = std::optional<Exact_coalescent_events>{};
  if (exact_coalescent_correction_) { exact_events = make_skygrid_exact_events(); }

  // real-space "positions" and "momenta"
  auto gamma_k = std::vector<double>(M+1, 0.0);
  auto p_k = std::vector<double>(M+1, 0.0);

  // Count number of coalescences in each Skygrid interval
  // (intervals with more coalescences get assigned a higher mass, which slows them down)
  auto c_k = std::vector<double>(M+1, 0.0);
  for (const auto& node : index_order_traversal(tree_)) {
    if (not tree_.at(node).is_tip()) {
      auto t = tree_.at(node).t;
      auto node_k = std::clamp(old_pop_model->interval_containing_t(t), 0, M);
      c_k[node_k] += 1.0;  // Should be accurate up to 2^52 coalescences...
    }
  }

  // masses of each gamma_k that roughly equalize angular frequencies normal modes to "1"
  auto m_k = std::vector<double>(M+1, 0.0);
  auto inv_m_k = std::vector<double>(M+1, 0.0);       // inverse masses
  for (auto k = 0; k <= M; ++k) {
    m_k[k] = (k > 0 ? tau : 0.0) + (k < M ? tau : 0.0) + c_k[k];
    inv_m_k[k] = 1.0 / m_k[k];
  }

  // components of forces
  auto f_k = std::vector<double>(M+1, 0.0);
  
  // Helper functions to calculate intermediate quantitites
  auto calc_K = [&]() -> double {
    auto K = 0.0;
    for (auto k = 0; k <= M; ++k) {
      K += 0.5 * std::pow(p_k[k], 2) * inv_m_k[k];
    }
    return K;
  };
  auto calc_U_prior_from_gamma_k_s = [&]() -> double {
    auto U_prior = 0.0;
    for (auto k = 1; k <= M; ++k) {  // Note: k == 0 excluded!
      U_prior += std::pow(gamma_k[k] - gamma_k[k-1], 2);
    }
    U_prior *= 0.5 * tau;

    if (skygrid_low_gamma_barrier_enabled_) {
      // Strong lower bound on gamma_k's: See comments in calc_cur_log_other_priors().
      for (auto k = 0; k <= M; ++k) {
        if (gamma_k[k] < skygrid_low_gamma_barrier_loc_) {
          auto excess = skygrid_low_gamma_barrier_loc_ - gamma_k[k];
          U_prior += std::pow(excess / skygrid_low_gamma_barrier_scale_, 2);
        }
      }
    }

    // InvGamma(alpha, beta) prior on N_bar = exp(gamma_bar)
    //   U_invgamma = alpha * gamma_bar + beta * exp(-gamma_bar)
    {
      auto gamma_bar = 0.0;
      for (auto k = 0; k <= M; ++k) {
        gamma_bar += gamma_k[k];
      }
      gamma_bar /= (M + 1);
      U_prior += skygrid_inv_nbar_prior_alpha_ * gamma_bar
                 + skygrid_inv_nbar_prior_beta_ * std::exp(-gamma_bar);
    }

    return U_prior;
  };

  auto update_new_pop_model_from_gamma_k_s = [&]() -> void {
    new_pop_model = std::make_shared<Skygrid_pop_model>(
        old_pop_model->x(), gamma_k, old_pop_model->type());
  };

  // The coalescent grid's cell populations only have to be refreshed when the cell prior is the
  // target.  `grid_touched` records whether they were, so that an abandoned trajectory restores
  // them if and only if they changed.
  auto grid_touched = false;
  auto refresh_grid_for_new_pop_model = [&]() -> void {
    coalescent_prior_.pop_model_changed(new_pop_model);
    grid_touched = true;
  };

  // U_coal of the target: the exact Kingman density with the correction on, the cell prior without
  // it.  The cell prior reads the grid, so it needs `refresh_grid_for_new_pop_model` first; the
  // exact density reads the event intervals and does not touch the grid.
  auto calc_U_coal_from_new_pop_model = [&]() -> double {
    if (exact_events) { return -exact_events->calc_log_prior(*new_pop_model); }
    return -calc_cur_log_coalescent_prior();
  };

  // Calculate which tips and `k`s have non-zero d_log_N_d_gamma(t_tip, k)
  auto k_2_relevant_tips = std::vector<std::vector<Node_index>>{};
  for (auto k = 0; k <= M; ++k) {
    auto relevant_tips = std::vector<Node_index>{};
    auto [t_min, t_max] = new_pop_model->support_of_d_log_N_d_gamma(k);
    for (const auto& node : index_order_traversal(tree_)) {
      if (not tree_.at(node).is_tip()) {
        if (t_min <= tree_.at(node).t && tree_.at(node).t <= t_max) {
          relevant_tips.push_back(node);
        }
      }
    }
    k_2_relevant_tips.push_back(std::move(relevant_tips));
  }

  auto calc_f_k_s_from_gamma_k_s = [&]() -> void {

    auto t_min_coal = coalescent_prior_.cell_lbound(0);
    auto t_max_coal = coalescent_prior_.cell_ubound(C-1);

    if (exact_events) {
      // U_coal = -log P_exact, so the coalescent part of f_k = -del U / del gamma_k is exactly
      // + d(log P_exact) / d gamma_k, which covers both the rate integrals and the sum over the
      // coalescences of log N(t_i).  The two cell-based loops below replace it when the correction
      // is off; they are the same two terms evaluated on the grid instead of on the events.
      std::ranges::fill(f_k, 0.0);
      exact_events->add_d_log_prior_d_gamma(*new_pop_model, f_k);
    }

    for (auto k = 0; k <= M; ++k) {
      if (not exact_events) {
        f_k[k] = 0.0;

        // Careful: support can go down to -inf for k = 0, and up to +inf for k = M
        // (for staircase model, t_min = y_{k-1} and t_max = y_k, defined in the comment above)
        auto [t_min, t_max] = new_pop_model->support_of_d_log_N_d_gamma(k);

        // And careful: support can be fully outside range of coalescent prior cells
        auto c_min = (t_min < t_min_coal) ? 0   : std::clamp(coalescent_prior_.cell_for(t_min), 0, C-1);
        auto c_max = (t_max > t_max_coal) ? C-1 : std::clamp(coalescent_prior_.cell_for(t_max), 0, C-1);

        for (auto c = c_min; c <= c_max; ++c) {
          auto a = coalescent_prior_.cell_lbound(c);
          auto b = coalescent_prior_.cell_ubound(c);
          f_k[k] +=
              0.5 * Delta * k_c[c] * (k_c[c] - 1.0) / N_c[c]
              * new_pop_model->d_log_int_N_d_gamma(a, b, k);
        }

        // The loop below has the same effect as this one that's commented out,
        // but, across all knots k, executes in O(N) time instead of O(N*M) time.
        //
        // for (const auto& node : index_order_traversal(tree_)) {
        //   if (not tree_.at(node).is_tip()) {
        //     f_k[k] -= new_pop_model->d_log_N_d_gamma(tree_.at(node).t, k);
        //   }
        // }
        //
        for (const auto& node : k_2_relevant_tips[k]) {
          f_k[k] -= new_pop_model->d_log_N_d_gamma(tree_.at(node).t, k);
        }
      }

      if (k > 0) {
        f_k[k] -= tau * (gamma_k[k] - gamma_k[k-1]);
      }
      if (k < M) {
        f_k[k] -= tau * (gamma_k[k] - gamma_k[k+1]);
      }

      if (skygrid_low_gamma_barrier_enabled_) {
        // Strong lower bound on gamma_k's: See comments in calc_cur_log_other_priors().
        if (gamma_k[k] < skygrid_low_gamma_barrier_loc_) {
          auto excess = skygrid_low_gamma_barrier_loc_ - gamma_k[k];
          f_k[k] += 2 * excess / std::pow(skygrid_low_gamma_barrier_scale_, 2);
        }
      }
    }

    // InvGamma(alpha, beta) prior on N_bar = exp(gamma_bar): force on each gamma_k
    //   f_k_invgamma = (-alpha + beta * exp(-gamma_bar)) / (M+1)
    {
      auto gamma_bar = 0.0;
      for (auto k = 0; k <= M; ++k) {
        gamma_bar += gamma_k[k];
      }
      gamma_bar /= (M + 1);
      auto f_invgamma = (-skygrid_inv_nbar_prior_alpha_
                         + skygrid_inv_nbar_prior_beta_ * std::exp(-gamma_bar))
                        / (M + 1);
      for (auto k = 0; k <= M; ++k) {
        f_k[k] += f_invgamma;
      }
    }
  };

  
  // Start of Leapfrog integrator
  constexpr auto debug_hmc = false;

  // Initial "positions" and "momenta"
  for (auto k = 0; k <= M; ++k) {
    gamma_k[k] = old_pop_model->gamma(k);
    p_k[k] = absl::Gaussian(bitgen_, 0.0, std::sqrt(m_k[k]));
  }
  
  auto old_K = calc_K();
  auto old_U_prior = calc_U_prior_from_gamma_k_s();
  auto old_U_coal = -(log_coalescent_prior_ + log_coal_exact_correction_);  // the target's coalescent term
  auto old_H = old_K + old_U_prior + old_U_coal;

  if (debug_hmc) {
    // Debugging
    std::cerr << absl::StrFormat(
        "Initial:       K = %10.1f, U_prior = %10.1f, U_coal = %10.1f, H = %10.1f\n",
        old_K, old_U_prior, old_U_coal, old_H);
  }

  // Abandon the trajectory and leave the state as it was.  Used by the guards below; like them, it
  // is applied symmetrically between the forward and the reverse trajectory, so no trajectory that
  // completes can have passed through a guarded state and the acceptance region stays symmetric.
  auto abandon_trajectory = [&](const char* why) -> void {
    if (debug_hmc) {
      std::cerr << "Rejecting Skygrid HMC because " << why << "\n";
    }
    pop_model_ = old_pop_model;
    if (grid_touched) { coalescent_prior_.pop_model_changed(old_pop_model); }
  };

  // A knot value outside the representable range has to be caught before it reaches the population
  // model, which would otherwise fail a check and abort the process.  Such values are reachable: when
  // `tau` is small the GMRF prior itself admits excursions of order 1/sqrt(tau) in gamma_k.  The guards
  // are written so that a NaN fails them, which `x > bound` would not do.
  //
  // exp(gamma_k) overflows above gamma_k ~ 710 and underflows to zero below gamma_k ~ -745, and a cell
  // whose population is infinite or zero breaks the partitioned sampler: the auxiliary lineage counts
  // of `Very_scalable_coalescent_prior_part` are drawn with a standard deviation of
  // sqrt(N_c / (parts * t_step)) (`core/very_scalable_coalescent.cpp:205`), and that path -- unlike
  // `Scalable_coalescent_prior::pop_model_changed` -- has no floor for a vanishing population, so its
  // partial coalescent prior becomes a NaN.  Trajectories through such a curve are abandoned, on the
  // same symmetry argument as the kinetic-energy guard: no trajectory that completes has passed
  // through one, so the chain explores the posterior restricted to representable population curves,
  // exactly as the low-population barrier restricts it from below.  The bound leaves a decade of
  // margin, so a cell of any width down to 1e-6 time units keeps a strictly positive, finite integral.
  auto gammas_are_representable = [&]() -> bool { return skygrid_gammas_are_representable(gamma_k); };

  // Reject outright if initial momenta are outrageously high (see comments in analogous check mid-loop below)
  if (not (calc_K() <= (100.0 * (M+1)))) {
    abandon_trajectory("its initial K is too high or not finite");
    return;
  }

  // Position Verlet algorithm:
  //
  //    gamma_k(t + dt/2) = gamma_k(t)        + (dt/2) (p_k(   t    ) / m_k),
  //        p_k(t + dt)   =     p_k(t)        +   dt    f_k(t + dt/2)
  //    gamma_k(t + dt)   = gamma_k(t + dt/2) + (dt/2) (p_k( t + dt ) / m_k).
  //
  // For clarity, we don't try to merge the second position update of one step with
  // the first position update of the next step; this isn't the speed bottleneck
  const auto dt = absl::Exponential(bitgen_, 1.0 / (2 * std::numbers::pi / 100.0));

  // RHMC, mean 25 steps
  const auto num_steps = 25;

  for (auto step = 0; step != num_steps; ++step) {

    // "Position" half-step
    for (auto k = 0; k <= M; ++k) {
      gamma_k[k] += 0.5 * dt * p_k[k] * inv_m_k[k];
    }
    if (not gammas_are_representable()) {
      abandon_trajectory("a knot value left the representable range");
      return;
    }
    update_new_pop_model_from_gamma_k_s();
    if (not exact_events) { refresh_grid_for_new_pop_model(); }  // Updates N_c

    // Force calculation
    calc_f_k_s_from_gamma_k_s();

    if (skygrid_hmc_force_check_enabled_) {
      // Check the analytic forces against a central finite difference of the same potential.
      auto d_gamma = skygrid_hmc_force_check_step_;
      for (auto k = 0; k <= M; ++k) {
        auto old_gamma_k = gamma_k[k];

        gamma_k[k] = old_gamma_k + d_gamma;
        update_new_pop_model_from_gamma_k_s();
        if (not exact_events) { refresh_grid_for_new_pop_model(); }
        auto U_plus = calc_U_coal_from_new_pop_model() + calc_U_prior_from_gamma_k_s();

        gamma_k[k] = old_gamma_k - d_gamma;
        update_new_pop_model_from_gamma_k_s();
        if (not exact_events) { refresh_grid_for_new_pop_model(); }
        auto U_minus = calc_U_coal_from_new_pop_model() + calc_U_prior_from_gamma_k_s();

        auto slow_f_k = -(U_plus - U_minus) / (2 * d_gamma);
        if (std::isfinite(slow_f_k) && std::isfinite(f_k[k])) {
          auto scale = std::max({1.0, std::abs(f_k[k]), std::abs(slow_f_k)});
          auto discrepancy = std::abs(slow_f_k - f_k[k]) / scale;

          // The resolution of this comparison: a relative 1e-4 of the force's own scale, or the
          // round-off floor of the central difference, whichever is larger.  See the comment on
          // `skygrid_hmc_max_force_discrepancy_over_resolution`.
          auto fd_floor = std::numeric_limits<double>::epsilon()
              * std::max(std::abs(U_plus), std::abs(U_minus)) / d_gamma;
          auto resolution = std::max(1e-4 * scale, fd_floor);
          skygrid_hmc_max_force_over_resolution_ = std::max(
              skygrid_hmc_max_force_over_resolution_,
              std::abs(slow_f_k - f_k[k]) / resolution);

          if (discrepancy > skygrid_hmc_max_force_discrepancy_) {
            skygrid_hmc_max_force_discrepancy_ = discrepancy;
            skygrid_hmc_worst_force_analytic_ = f_k[k];
            skygrid_hmc_worst_force_numeric_ = slow_f_k;
            skygrid_hmc_worst_force_gamma_ = old_gamma_k;
            skygrid_hmc_worst_force_min_gamma_ = *std::ranges::min_element(gamma_k);
          }
          ++skygrid_hmc_force_checks_;
        }

        gamma_k[k] = old_gamma_k;
      }

      // Return to original gamma_k's
      update_new_pop_model_from_gamma_k_s();
      if (not exact_events) { refresh_grid_for_new_pop_model(); }
    }

    // "Momentum" full-step
    for (auto k = 0; k <= M; ++k) {
      p_k[k] += dt * f_k[k];
    }

    // Sanity check: if kinetic energy starts exploding, reject move prematurely
    // (numerics say this is simply going to crash).
    // At equilibrium, K =~ (M+1) / 2.
    // To ensure detailed balance, we also apply this rejection criteria at the very beginning;
    // thus, we _never_ propose trajectories where K > 10 (M+1) at any point.
    if (not (calc_K() <= (100.0 * (M+1)))) {
      abandon_trajectory("it is blowing up: K is too high or not finite");
      return;
    }

    // "Position" half-step
    for (auto k = 0; k <= M; ++k) {
      gamma_k[k] += 0.5 * dt * p_k[k] * inv_m_k[k];
    }

    if (not gammas_are_representable()) {
      abandon_trajectory("a knot value left the representable range");
      return;
    }

    if (debug_hmc) {
      // Debugging (all of the below can go away if debug trace is not needed)
      update_new_pop_model_from_gamma_k_s();
      if (not exact_events) { refresh_grid_for_new_pop_model(); }

      auto cur_K = calc_K();
      auto cur_U_prior = calc_U_prior_from_gamma_k_s();
      auto cur_U_coal = calc_U_coal_from_new_pop_model();
      auto cur_H = cur_K + cur_U_prior + cur_U_coal;
      
      std::cerr << absl::StrFormat(
          "Post step %3d: K = %10.1f, U_prior = %10.1f, U_coal = %10.1f, H = %10.1f\n",
          step, cur_K, cur_U_prior, cur_U_coal, cur_H);
    }
  }

  update_new_pop_model_from_gamma_k_s();
  refresh_grid_for_new_pop_model();

  auto new_K = calc_K();
  auto new_U_prior = calc_U_prior_from_gamma_k_s();
  auto new_log_coalescent_cells = calc_cur_log_coalescent_prior();
  auto new_U_coal = exact_events ? -exact_events->calc_log_prior(*new_pop_model)
                                 : -new_log_coalescent_cells;
  auto new_H = new_K + new_U_prior + new_U_coal;

  auto log_metropolis = old_H - new_H;  // HMC => this should be close to 0.0

  // Defend against horrible NaN's and overflow errors
  auto something_blew_up = false;
  for (auto k = 0; k <= M; ++k) {
    if (std::isnan(gamma_k[k])) {
      something_blew_up = true;
    }
  }
  if (std::isnan(log_metropolis) || not std::isfinite(new_U_coal)) {
    something_blew_up = true;
  }
  if (something_blew_up && debug_hmc) {
    std::cerr << "About to reject because a NaN appeared somewhere\n";
  }
  
  if (not something_blew_up && (log_metropolis > 0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_metropolis))) {
    // Accept
    if (debug_hmc) {
      std::cerr << "Accept!\n";
      for (auto k = 0; k <= M; ++k) {
        std::cerr << absl::StreamFormat("* gamma_k[%2d] = %10.5f  ( at x_k[%2d] = %s)\n",
                                        k, gamma_k[k], k, to_iso_date(new_pop_model->x(k)));
      }
    }
    pop_model_ = new_pop_model;
    log_coalescent_prior_ = new_log_coalescent_cells;
    if (exact_events) { log_coal_exact_correction_ = -new_U_coal - new_log_coalescent_cells; }
    log_other_priors_ += (-new_U_prior) - (-old_U_prior);
  } else {
    // Reject
    if (debug_hmc) {
      std::cerr << "Reject!\n";
    }
    pop_model_ = old_pop_model;
    coalescent_prior_.pop_model_changed(pop_model_);
  }
}

auto Run::skygrid_gammas_zero_mode_gibbs_move() -> void {
  //
  // Amazingly, the overall size of the population is something that can be Gibbs-sampled in
  // the Skygrid model.  First, we decompose {gamma_k} into a baseline `gamma_bar` plus
  // offsets {gamma'_k} which add up to 0:
  //
  //   gamma_bar := 1/(M+1) sum_{k=0}^M gamma_k,  and
  //   gamma'_k  := gamma_k - gamma_bar
  //
  // yields
  //
  //   gamma_k = gamma_bar + gamma'_k   and   sum_{k=0}^M gamma'_k = 0.
  //
  // The GMRF and low-gamma barrier terms in U_prior do not constrain `gamma_bar` at all.
  // (The optional InvGamma(alpha, beta) prior on N_bar = exp(gamma_bar) does constrain
  // gamma_bar, but it is conjugate with the coalescent likelihood, so Gibbs sampling is
  // preserved; see below.)  U_coal's dependence on `gamma_bar` is particularly simple.
  // This is best seen with a change of coordinates:
  //
  //    {gamma_k}  ->  {I_bar, gamma'_k | k < M}  (gamma'_M excluded as it's not independent).
  //
  // Here
  //
  //   I_bar := exp(-gamma_bar).
  //
  // We first factor out I_bar from N(t) and N_c, respectively:
  //
  //   Ntwiddle(t) := N(t) I_bar,
  //   Ntwiddle_c  := N_c  I_bar.
  //
  // Note that this makes Ntwiddle(t) and Ntwiddle_c a function purely of {gamma'_k | k < M},
  // and so they are both independent of I_bar.  Conversely, I_bar is by construction
  // independent of {gamma'_k | k < M}.
  //
  // We then write U_coal in terms of I_bar and {gamma'_k | k < M}:
  //
  //                     +---------------------- B ------------------------+
  //                    /                                                   \ |
  //   U_coal = + I_bar \sum_{c=0}^{C-1} (1/2) Delta k_c (k_c-1) / Ntwiddle_c
  //            - N_inner log(I_bar) + \sum_{i inner node} log(Ntwiddle(t_i)).
  //
  // Hence, the dependence of the posterior (~ exp(-U)) on I_bar is almost a straightforward
  // Gamma distribution!  Explicitly,
  //
  //   exp[-U_coal] = I_bar^{N_inner} exp[- B I_bar] * (factors independent of I_bar)
  //
  // That means that I_bar is amenable to Gibbs sampling.  The caveat is that there are
  // Jacobian factors that modify the base result, but not by much.  First, note that an
  // intermediate change from {gamma_k} to gamma_bar & {gamma'_k | k < M} has a constant
  // Jacobian, so that
  //
  //   P({gamma_k}) d{gamma_k} =
  //       (const.) * P({gamma_bar, gamma'_k | k < M}) d{gamma_bar, gamma'_k | k < M}.
  //
  // Hence, the only difficulty comes from switching from gamma_bar to I_bar.  We have
  //
  //           I_bar = exp[ -gamma_bar ]
  //   =>     dI_bar = - exp[ -gamma_bar ] dgamma_bar
  //   => dgamma_bar = - dI_bar / I_bar
  //
  // Thus, when switching from gamma_bar to I_bar, apart from a constant factor, we gain
  // an extra factor of 1/I_bar in the posterior.
  //
  // Additionally, an InvGamma(alpha, beta) prior on N_bar = exp(gamma_bar) contributes
  // (in I_bar coordinates) a factor I_bar^alpha * exp(-beta * I_bar), which is conjugate.
  //
  // Combining coalescent likelihood, Jacobian, and InvGamma prior:
  //
  //   P(I_bar) dI_bar = (const.) * I_bar^{N_inner + alpha - 1} exp[-(B + beta) I_bar] dI_bar.
  //
  // Hence, we can Gibbs sample I_bar from Gamma(shape = N_inner + alpha, rate = B + beta).
  // With the default alpha=0, beta=0, this reduces to Gamma(N_inner, B).
  
  const Pop_model& raw_pop_model = *pop_model_;
  CHECK(typeid(raw_pop_model) == typeid(Skygrid_pop_model));
  auto old_pop_model = std::static_pointer_cast<const Skygrid_pop_model>(pop_model_);

  auto M = old_pop_model->M();

  auto Delta = coalescent_prior_.t_step();
  const auto& k_c = coalescent_prior_.k_bars();
  const auto& N_c = coalescent_prior_.popsize_bars();
  auto C = std::ssize(k_c);
  
  auto gamma_bar = 0.0;
  for (auto k = 0; k <= M; ++k) {
    gamma_bar += old_pop_model->gamma(k);
  }
  gamma_bar /= (M+1);
  auto I_bar = std::exp(-gamma_bar);
  
  auto N_inner = (tree_.size() - 1) / 2;

  // B is the coefficient of I_bar in U_coal, which is the coalescent term's whole dependence on the
  // zero mode.  The derivation above holds term by term for the exact Kingman density as well as for
  // the cell prior, because a uniform shift of the curve rescales every rate integral by the same
  // factor and adds the same constant to every log N(t_i):
  //
  //   A := sum_j C(k_j, 2) int_{a_j}^{b_j} dt / N(t) = I_bar sum_j C(k_j, 2) int dt / Ntwiddle(t),
  //
  // so B = A / I_bar with A read off whichever coalescent term the target uses.  With the exact
  // correction on, the move is then a Gibbs step of the exact posterior's conditional, not of the
  // cell prior's.
  auto B = 0.0;
  if (exact_coalescent_correction_) {
    B = make_skygrid_exact_events().calc_intensity_term(*old_pop_model);
  } else {
    for (auto c = 0; c < C; ++c) {
      B += 0.5 * Delta * k_c[c] * (k_c[c] - 1.0) / N_c[c];
    }
  }
  B /= I_bar;

  // WARNING: C++ calls "beta" the *scale* of the distribution, i.e., 1/rate
  auto I_bar_dist = std::gamma_distribution<double>{
    static_cast<double>(N_inner) + skygrid_inv_nbar_prior_alpha_,
    1.0 / (B + skygrid_inv_nbar_prior_beta_)};
  
  auto old_I_bar = I_bar;
  auto new_I_bar = I_bar_dist(bitgen_);
  auto delta_gamma_bar = std::log(old_I_bar / new_I_bar); // -log(new_I_bar)-(-log(old_I_bar))

  auto new_gamma_k = std::vector<double>(M+1, 0.0);
  for (auto k = 0; k <= M; ++k) {
    new_gamma_k[k] = old_pop_model->gamma(k) + delta_gamma_bar;
  }
  
  // Strong lower bound on gamma_k's: See comments in calc_cur_log_other_priors().
  auto old_log_prior_bound = 0.0;
  auto new_log_prior_bound = 0.0;

  if (skygrid_low_gamma_barrier_enabled_) {
    for (auto k = 0; k <= M; ++k) {
      if (old_pop_model->gamma(k) < skygrid_low_gamma_barrier_loc_) {
        auto excess = skygrid_low_gamma_barrier_loc_ - old_pop_model->gamma(k);
        old_log_prior_bound -= std::pow(excess / skygrid_low_gamma_barrier_scale_, 2);
      }
      if (new_gamma_k[k] < skygrid_low_gamma_barrier_loc_) {
        auto excess = skygrid_low_gamma_barrier_loc_ - new_gamma_k[k];
        new_log_prior_bound -= std::pow(excess / skygrid_low_gamma_barrier_scale_, 2);
      }
    }
  }

  auto delta_log_posterior = new_log_prior_bound - old_log_prior_bound;  // Otherwise, this is a Gibbs move

  // Reject if anything becomes a NaN, or if the proposed curve is outside the representable range.
  // An extremely small draw of I_bar sends every knot above the overflow bound of exp, where neither
  // the cell prior nor the exact density can be evaluated; the shape parameter of the draw is
  // N_inner + alpha, so this has negligible probability, but it was the one path in the Skygrid
  // moveset without such a guard.  The restriction is the same one the HMC's guard imposes: the
  // chain explores the posterior restricted to representable curves.
  auto something_blew_up = not skygrid_gammas_are_representable(new_gamma_k);
  if (std::isnan(delta_log_posterior)) {
    something_blew_up = true;
  }
  if (something_blew_up) {
    if (false) {
      std::cerr << "Rejecting zero-mode Gibbs move because a NaN appeared somewhere\n";
    }
    return;
  }
  
  auto log_metropolis = delta_log_posterior;
  if (log_metropolis >= 0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_metropolis)) {
    // Accept
    pop_model_ = std::make_shared<Skygrid_pop_model>(old_pop_model->x(), new_gamma_k, old_pop_model->type());
    coalescent_prior_.pop_model_changed(pop_model_);
    
    log_coalescent_prior_ = calc_cur_log_coalescent_prior();
    if (exact_coalescent_correction_) {
      log_coal_exact_correction_ = calc_cur_log_coalescent_exact() - log_coalescent_prior_;
    }

    // InvGamma prior on N_bar: log pi = -alpha * gamma_bar - beta * exp(-gamma_bar)
    auto old_gamma_bar = 0.0;
    auto new_gamma_bar = 0.0;
    for (auto k = 0; k <= M; ++k) {
      old_gamma_bar += old_pop_model->gamma(k);
      new_gamma_bar += new_gamma_k[k];
    }
    old_gamma_bar /= (M + 1);
    new_gamma_bar /= (M + 1);
    auto old_inv_nbar_prior = -skygrid_inv_nbar_prior_alpha_ * old_gamma_bar
                              - skygrid_inv_nbar_prior_beta_ * std::exp(-old_gamma_bar);
    auto new_inv_nbar_prior = -skygrid_inv_nbar_prior_alpha_ * new_gamma_bar
                              - skygrid_inv_nbar_prior_beta_ * std::exp(-new_gamma_bar);

    log_other_priors_ += (new_inv_nbar_prior - old_inv_nbar_prior)
                       + (new_log_prior_bound - old_log_prior_bound);
  }
}

auto Run::save_original_sequences() -> void {
  if (estd::is_debug_enabled) {
    // Horribly inefficient, but this is only used for paranoid checks in Debug builds

    // Make a copy of the original ref sequence; all sequences are encoded as overlays atop this copy
    original_ref_sequence_ = tree_.ref_sequence;
    
    for (const auto& node : index_order_traversal(tree_)) {
      original_sequences_.push_back(view_of_sequence_at(tree_, node).rebase_to(original_ref_sequence_));
      auto missing_sites = Interval_set<>{};
      missing_sites = reconstruct_missing_sites_at(tree_, node);
      original_missing_sites_all_.push_back(std::move(missing_sites));
    }
  }
}

auto Run::assert_cur_tip_sequences_compatible_with_original_ones() -> void {
  assert_tip_sequences_compatible_with_original_ones(tree_, original_sequences_, original_missing_sites_all_);
}

}  // namespace delphy
