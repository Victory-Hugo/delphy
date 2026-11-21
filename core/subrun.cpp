#include "subrun.h"

#include <algorithm>

#include "distributions.h"
#include "phylo_tree_calc.h"
#include "spr_move.h"
#include "spr_study.h"

namespace delphy {

Subrun::Subrun(absl::BitGenRef bitgen, Phylo_tree tree, bool includes_run_root, Global_evo_model evo)
    : Subrun{bitgen, std::move(tree), includes_run_root, std::make_shared<const Global_evo_model>(std::move(evo))} {
}

Subrun::Subrun(absl::BitGenRef bitgen, Phylo_tree tree, bool includes_run_root,
               std::shared_ptr<const Global_evo_model> evo)
    : bitgen_{bitgen},
      tree_{std::move(tree)},
      includes_run_root_{includes_run_root},
      evo_{std::move(evo)} {
}

auto Subrun::recalc_derived_quantities() const -> void {
  // Derived quantities
  state_frequencies_of_ref_sequence_per_partition_ = given_state_frequencies_.has_value()
      ? *given_state_frequencies_
      : calc_cur_state_frequencies_of_ref_sequence_per_partition();
  ref_cum_Q_l_ = given_ref_cum_Q_l_ != nullptr
      ? given_ref_cum_Q_l_
      : std::make_shared<const std::vector<double>>(calc_cur_ref_cum_Q_l());
  lambda_i_ = calc_cur_lambda_i(*ref_cum_Q_l_);
  log_G_ = calc_cur_log_G(lambda_i_, state_frequencies_of_ref_sequence_per_partition_);
  num_sites_missing_at_every_node_ = calc_cur_num_sites_missing_at_every_node();
  log_augmented_coalescent_prior_ = calc_cur_log_augmented_coalescent_prior();
  log_calibration_prior_ = calc_cur_log_calibration_prior();
}

auto Subrun::check_derived_quantities() const -> void {
  if (estd::is_debug_enabled) {
    // We maintain these incrementally except when we force a recalculation
    { auto expected = calc_cur_log_augmented_coalescent_prior();
      CHECK(std::abs(log_augmented_coalescent_prior_ - expected) < 1e-5)
      << log_augmented_coalescent_prior_ << " != " << expected; }
    { auto expected = calc_cur_log_calibration_prior();
      CHECK(std::abs(log_calibration_prior_ - expected) < 1e-8)
      << log_calibration_prior_ << " != " << expected; }
    if (calibration_targets_follow_topology()) {
      auto nodes = std::vector<Node_index>{};
      CHECK(find_calibration_targets(nodes)) << "A calibrated sample set is not monophyletic";
      CHECK(save_calibration_nodes() == nodes) << "A calibration target no longer is the MRCA of its sample set";
      CHECK(std::isfinite(log_calibration_prior_)) << "A calibration target is outside its hard support";
    }
    if (threshold_constraint_) {
      auto tips = Node_vector<int>(std::ssize(tree_), 0);
      for (auto node : post_order_traversal(tree_)) {
        tips[node] = tree_.at(node).is_tip()
            ? threshold_constraint_->tip_weights[node]
            : tips[tree_.at(node).left_child()] + tips[tree_.at(node).right_child()];
      }
      CHECK(tips == clade_tips_) << "The clade sizes of the threshold partition are out of date";
      CHECK(satisfies_threshold_partition_constraint()) << "A move changed the threshold partition";
    }
    { auto expected = calc_cur_state_frequencies_of_ref_sequence_per_partition();
      for (auto beta = Partition_index{0}; beta != evo_->num_partitions(); ++beta) {
        CHECK_EQ(state_frequencies_of_ref_sequence_per_partition_[beta], expected[beta])
            << "Partition " << beta; } }
    { auto expected = calc_cur_ref_cum_Q_l();
      for (auto l = Site_index{0}; l <= tree_.num_sites(); ++l) {
        CHECK_LT(std::abs((*ref_cum_Q_l_)[l] - expected[l]), 1e-8)
            << "Site " << l << ", "
            << (*ref_cum_Q_l_)[l] << " != "
            << expected[l]; } }
    { auto expected = calc_cur_lambda_i(*ref_cum_Q_l_);
      for (const auto& node : index_order_traversal(tree_)) {
        CHECK((std::abs(lambda_i_[node] - expected[node])
               / static_cast<double>(tree_.num_sites())) < 1e-8)
            << "Node " << node << ", "
            << lambda_i_[node] << " != "
            << expected[node]; } }
    { auto expected = calc_cur_log_G(lambda_i_, state_frequencies_of_ref_sequence_per_partition_);
      CHECK(std::abs(log_G_ - expected) < 1e-6)
          << log_G_ << " != " << expected; }
    CHECK(num_sites_missing_at_every_node_ == calc_cur_num_sites_missing_at_every_node());
  }
}

auto Subrun::calc_cur_log_G(
    const Node_vector<double>& lambda_i,
    const Partition_vector<Seq_vector<int>>& state_frequencies_of_ref_sequence_per_partition) const
    -> double {
  auto result = 0.0;
  if (includes_run_root_) {
    result += calc_log_root_prior(tree_, *evo_, state_frequencies_of_ref_sequence_per_partition);
  }
  result += calc_log_G_below_root(tree_, *evo_, lambda_i, state_frequencies_of_ref_sequence_per_partition);
  return result;
}

auto Subrun::calc_cur_state_frequencies_of_ref_sequence_per_partition() const -> Partition_vector<Seq_vector<int>> {
  return calc_state_frequencies_per_partition_of(tree_.ref_sequence, *evo_);
}

auto Subrun::calc_cur_lambda_i(const std::vector<double>& ref_cum_Q_l) const -> Node_vector<double> {
  return calc_lambda_i(tree_, *evo_, ref_cum_Q_l);
}

auto Subrun::calc_cur_ref_cum_Q_l() const -> std::vector<double> {
  return calc_cum_Q_l_for_sequence(tree_.ref_sequence, *evo_);
}

auto Subrun::calc_cur_num_sites_missing_at_every_node() const -> Node_vector<int> {
  return calc_num_sites_missing_at_every_node(tree_);
}

auto Subrun::calc_cur_log_augmented_coalescent_prior() const -> double {
  if (coalescent_prior_part_ == nullptr) {
    return -1.0;
  } else {
    return coalescent_prior_part_->calc_partial_log_prior();
  }
}

auto Subrun::calc_cur_log_calibration_prior() const -> double {
  auto result = 0.0;
  for (const auto& c : calibrations_) {
    result += c.log_density_at(tree_.at(c.node).t);
  }
  return result;
}

auto Subrun::calc_log_calibration_prior_at(Node_index node, double t) const -> double {
  auto result = 0.0;
  for (const auto& c : calibrations_) {
    if (c.node == node) { result += c.log_density_at(t); }
  }
  return result;
}

auto Subrun::calc_delta_log_calibration_prior(Node_index node, double t_old, double t_new) const -> double {
  if (calibrations_.empty()) { return 0.0; }
  auto log_p_new = calc_log_calibration_prior_at(node, t_new);
  if (not std::isfinite(log_p_new)) { return -std::numeric_limits<double>::infinity(); }
  auto log_p_old = calc_log_calibration_prior_at(node, t_old);
  DCHECK(std::isfinite(log_p_old)) << "Current state is outside a calibration's hard support";
  return log_p_new - log_p_old;
}

auto Subrun::calibration_t_bounds(Node_index node) const -> std::pair<double, double> {
  auto t_lo = -std::numeric_limits<double>::infinity();
  auto t_hi = +std::numeric_limits<double>::infinity();
  for (const auto& c : calibrations_) {
    if (c.node != node) { continue; }
    // Older age == smaller time, so the oldest allowed age gives the lower bound on t.
    t_lo = std::max(t_lo, c.reference_t - 365.0 * c.prior.support_oldest_age());
    t_hi = std::min(t_hi, c.reference_t - 365.0 * c.prior.support_youngest_age());
  }
  return {t_lo, t_hi};
}

auto Subrun::find_calibration_targets(std::vector<Node_index>& nodes) const -> bool {
  nodes.assign(std::ssize(calibrations_), k_no_node);
  auto tips_below = Scratch_vector<int>(tree_.size(), 0);
  auto marked_below = Scratch_vector<int>(tree_.size(), 0);
  auto order = Scratch_vector<Node_index>{};
  for (const auto& node : post_order_traversal(tree_)) {
    order.push_back(node);
    if (tree_.at(node).is_tip()) {
      tips_below[node] = 1;
    } else {
      tips_below[node] = tips_below[tree_.at(node).left_child()] + tips_below[tree_.at(node).right_child()];
    }
  }

  for (auto i = 0; i != std::ssize(calibrations_); ++i) {
    const auto& c = calibrations_[i];
    if (c.all_tips) { nodes[i] = tree_.root; continue; }
    std::ranges::fill(marked_below, 0);
    for (auto tip : c.tips) { marked_below[tip] = 1; }
    auto size = static_cast<int>(std::ssize(c.tips));
    // In post order, the first node with every marked tip below it is their MRCA
    for (auto node : order) {
      if (tree_.at(node).is_inner_node()) {
        marked_below[node] = marked_below[tree_.at(node).left_child()] + marked_below[tree_.at(node).right_child()];
      }
      if (marked_below[node] == size) { nodes[i] = node; break; }
    }
    CHECK_NE(nodes[i], k_no_node);
    if (c.require_monophyly && tips_below[nodes[i]] != size) { return false; }
  }
  return true;
}

auto Subrun::rebind_calibrations() -> bool {
  auto nodes = std::vector<Node_index>{};
  auto ok = find_calibration_targets(nodes);
  if (ok) { restore_calibration_nodes(nodes); }
  return ok;
}

auto Subrun::calc_rebound_log_calibration_prior() -> double {
  if (not rebind_calibrations()) { return -std::numeric_limits<double>::infinity(); }
  auto result = calc_cur_log_calibration_prior();
  return std::isfinite(result) ? result : -std::numeric_limits<double>::infinity();
}

auto Subrun::save_calibration_nodes() const -> std::vector<Node_index> {
  auto result = std::vector<Node_index>{};
  result.reserve(std::ssize(calibrations_));
  for (const auto& c : calibrations_) { result.push_back(c.node); }
  return result;
}

auto Subrun::restore_calibration_nodes(const std::vector<Node_index>& nodes) -> void {
  CHECK_EQ(std::ssize(nodes), std::ssize(calibrations_));
  for (auto i = 0; i != std::ssize(nodes); ++i) { calibrations_[i].node = nodes[i]; }
}

auto Subrun::assert_cur_tip_sequences_compatible_with_original_ones() -> void {
  assert_tip_sequences_compatible_with_original_ones(tree_, original_sequences_, original_missing_sites_all_);
}

auto Subrun::mcmc_sub_iteration() -> void {
  // All moves can allocate Scratch_X containers quickly,
  // which will all be freed at the end of the move
  auto scope = Local_arena_scope{};
  
  validate_derived_quantities();

  auto run_move = [this](Move_kind kind, auto&& move) {
    move_accepted_ = false;
    move_blocked_ = false;
    move_blocked_by_partition_ = false;
    move();
    auto& counts = move_counts_[static_cast<int>(kind)];
    ++counts.proposed;
    if (move_accepted_) { ++counts.accepted; }
    if (move_blocked_) { ++counts.blocked_by_fixed_topology; }
    if (move_blocked_by_partition_) { ++counts.blocked_by_partition; }
  };

  if (only_displacing_inner_nodes_) {
    run_move(Move_kind::inner_node_displace, [this] { inner_node_displace_move(); });
  } else {
    // With topology moves disabled, the subtree slide stays in the mix: it keeps resampling node heights and the
    // mutation history around the slid node, and discards proposals that would hop over another node.  Only the
    // SPR move is dropped.
    // The tip move gets weight 7.5 unless it was asked to be skipped and no tip can move
    auto tip_weight = (skip_fixed_tip_moves_ && not has_movable_tip()) ? 0.0 : 7.5;
    auto inner_end = 7.5;                   // inner_node_displace: weight 7.5
    auto tip_end = inner_end + tip_weight;  // tip_displace: weight 7.5
    auto reform_end = tip_end + 15.0;       // branch_reform: weight 15.0
    auto slide_end = reform_end + 1.0;      // subtree_slide: weight 1.0
    auto spr_end = slide_end + (topology_moves_enabled() ? 1.0 : 0.0);  // spr1: weight 1.0
    // The single-site history Gibbs updates share one weight, half each.  At the default weight of
    // zero the total below, and so the draw from the random stream, is what it has always been.
    auto branch_gibbs_end = spr_end + 0.5 * history_gibbs_weight_;
    auto node_gibbs_end = spr_end + history_gibbs_weight_;
    auto total_weight = node_gibbs_end + marginal_node_time_weight_;
    auto r = absl::Uniform(bitgen_, 0.0, total_weight);
    if (r < inner_end) { run_move(Move_kind::inner_node_displace, [this] { inner_node_displace_move(); }); }
    else if (r < tip_end) { run_move(Move_kind::tip_displace, [this] { tip_displace_move(); }); }
    else if (r < reform_end) { run_move(Move_kind::branch_reform, [this] { branch_reform_move(); }); }
    else if (r < slide_end) { run_move(Move_kind::subtree_slide, [this] { subtree_slide_move(); }); }
    else if (r < spr_end) { run_move(Move_kind::spr1, [this] { spr1_move(); }); }
    else if (r < branch_gibbs_end) {
      run_move(Move_kind::branch_history_gibbs, [this] { branch_history_gibbs_move(); });
    } else if (r < node_gibbs_end) {
      run_move(Move_kind::node_state_gibbs, [this] { node_state_gibbs_move(); });
    } else {
      run_move(Move_kind::node_time_marginal, [this] { node_time_marginal_move(); });
    }
  }

  check_derived_quantities();
}

auto Subrun::has_movable_tip() const -> bool {
  if (has_movable_tip_ < 0) {
    has_movable_tip_ = 0;
    for (const auto& node : index_order_traversal(tree_)) {
      if (tree_.at(node).is_tip() && tree_.at(node).t_min != tree_.at(node).t_max) {
        has_movable_tip_ = 1;
        break;
      }
    }
  }
  return has_movable_tip_ == 1;
}

auto Subrun::pick_random_node() -> Node_index {
  DCHECK_GT(std::ssize(tree_), 0);
  return absl::Uniform<Node_index>(absl::IntervalClosedOpen, bitgen_, 0, std::ssize(tree_));
}

auto Subrun::pick_random_inner_node() -> Node_index {
  DCHECK_GT(std::ssize(tree_), 0);
  while (true) {
    auto result = pick_random_node();
    if (tree_.at(result).is_inner_node()) {
      return result;
    }
  }
}

auto Subrun::pick_random_tip() -> Node_index {
  DCHECK_GT(std::ssize(tree_), 0);
  while (true) {
    auto result = pick_random_node();
    if (tree_.at(result).is_tip()) {
      return result;
    }
  }
}

auto Subrun::inner_node_displace_move() -> void {
  if (std::ssize(tree_) < 1) return;

  // We only move the root of the subtree if it's the root of the whole tree
  auto node = pick_random_inner_node();
  if (node == tree_.root && not includes_run_root_) { return; }

  auto t_min = -std::numeric_limits<double>::infinity();
  if (node != tree_.root) {
    t_min = tree_.at_parent_of(node).t;
    for (const auto& m : tree_.at(node).mutations) {
      t_min = std::max(t_min, m.t);
    }
  }

  auto t_max = +std::numeric_limits<double>::infinity();
  for (const auto& child : tree_.at(node).children) {
    t_max = std::min(t_max, tree_.at(child).t);
    for (const auto& m : tree_.at(child).mutations) {
      t_max = std::min(t_max, m.t);
    }
  }

  // Restrict the proposal to the hard support of this node's age calibrations.  `t_min` and `t_max`
  // depend only on quantities this move leaves alone, and the hard support is constant, so the
  // proposal interval is the same for the forward and the reverse move and the Hastings term below
  // stays exact.  See doc/extended/calibration.md, section 2.3.
  if (not calibrations_.empty()) {
    auto [cal_t_min, cal_t_max] = calibration_t_bounds(node);
    t_min = std::max(t_min, cal_t_min);
    t_max = std::min(t_max, cal_t_max);
    if (not (t_min < t_max)) { return; }
  }

  auto lambda_at_node = lambda_i_[node];
  auto d_logG_dt = 0.0;
  if (node != tree_.root) {
    // parent branch lengthens
    d_logG_dt += -lambda_at_node;
  }
  for (const auto& child : tree_.at(node).children) {
    // child branches shorten (branch start of child is like branch end of node, adjusted for child missations)
    auto lambda_just_below_node = lambda_at_node + calc_delta_lambda_across_missations(
        *evo_, tree_.ref_sequence, *ref_cum_Q_l_, tree_.at(child).missations);
    d_logG_dt -= -lambda_just_below_node;
  }

   // Pick tree_.at(node).t as ~ exp(d_logG_dt * node.t), bounded by t_min and t_max.
   auto old_node_t = tree_.at(node).t;
  auto log_alpha_old_to_new_over_new_to_old = 0.0;
  auto new_node_t = old_node_t;
  if (node == tree_.root) {
    // Cap the proposal scale to the tree span so that we don't propose
    // absurdly distant root times when mu is tiny (making lambda_i
    // negligible and delta_scale enormous).  Such proposals would be
    // rejected by the coalescent prior anyway, but evaluating the prior
    // requires allocating a grid covering the proposed time range, which
    // can consume GiB of memory.
    // See plans/2026-03-26-01-cap-root-displace-proposal.md for details.
    auto tree_span = t_max_tip_ - t_max;
    CHECK_GE(tree_span, 0.0);
    auto delta_scale = std::min(root_displace_scale_factor_ * ((1 / lambda_i_.at(node)) / 2),  // 95% of the time, won't hop past more than 1 mutation (factor 1)
                                tree_span);
    new_node_t = old_node_t + absl::Gaussian(bitgen_, 0.0, delta_scale);
    if (new_node_t < t_min || new_node_t > t_max) { return; }
    log_alpha_old_to_new_over_new_to_old = 0.0;
  } else {
    auto t_dist = Bounded_exponential_distribution{d_logG_dt, t_min, t_max};
    new_node_t = t_dist(bitgen_);
    
    // The proposal probability being proportional to the likelihood, the MH ratio will be 1.0 (before considering the coalescent prior)
    log_alpha_old_to_new_over_new_to_old = d_logG_dt * (new_node_t - old_node_t);
  }

  // Forbid zero-length branches or mutation times that coincide with node times
  if (new_node_t == t_min || new_node_t == t_max) { return; }

  CHECK(t_min <= new_node_t) << lambda_at_node << ","
                             << d_logG_dt << "," << t_min << "," << t_max << "," << new_node_t;
  CHECK(new_node_t <= t_max);

  // See if coalescent prior agrees with us
  auto delta_log_G = d_logG_dt * (new_node_t - old_node_t);
  auto delta_log_prior =
      coalescent_prior_part_->calc_delta_partial_log_prior_after_displace_coalescence(old_node_t, new_node_t);
  auto delta_log_calibration_prior = calc_delta_log_calibration_prior(node, old_node_t, new_node_t);
  auto log_mh = delta_log_G + delta_log_prior + delta_log_calibration_prior
      - log_alpha_old_to_new_over_new_to_old;
  if (log_mh >= 0.0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_mh)) {
    // Accept
    move_accepted_ = true;
    coalescent_prior_part_->coalescence_displaced(old_node_t, new_node_t);
    tree_.at(node).t = new_node_t;

    auto dt = new_node_t - old_node_t;
    log_G_ += d_logG_dt * dt;
    log_augmented_coalescent_prior_ += delta_log_prior;
    log_calibration_prior_ += delta_log_calibration_prior;
  }
}

auto Subrun::tip_displace_move() -> void {
  if (std::ssize(tree_) < 1) return;

  auto node = pick_random_tip();
  CHECK_NE(node, tree_.root);
  // No calibration term here: a calibration target is the common ancestor of at least two samples, so
  // it is always an inner node of the whole tree.  A calibrated node that is a cut point appears in a
  // neighbouring part as a frozen "tip" with t_min == t_max, which returns on the next line.
  DCHECK(std::ranges::none_of(calibrations_, [node](const auto& c) { return c.node == node; }));
  if (tree_.at(node).t_min == tree_.at(node).t_max) { return; }  // No uncertainty!

  auto t_min = std::max(double{tree_.at(node).t_min}, tree_.at_parent_of(node).t);
  for (const auto& m : tree_.at(node).mutations) {
    t_min = std::max(t_min, m.t);
  }

  auto t_max = double{tree_.at(node).t_max};

  auto lambda_at_node = lambda_i_[node];
  auto d_logG_dt = 0.0;
  
  // parent branch lengthens
  d_logG_dt += -lambda_at_node;

  // Pick tree_.at(node).t as ~ exp(d_logG_dt * node.t), bounded by t_min and t_max.
  auto old_node_t = tree_.at(node).t;
  auto log_alpha_old_to_new_over_new_to_old = 0.0;
  
  auto t_dist = Bounded_exponential_distribution{d_logG_dt, t_min, t_max};
  auto new_node_t = t_dist(bitgen_);
  
  // The proposal probability being proportional to the likelihood, the MH ratio will be 1.0 (before considering the coalescent prior)
  log_alpha_old_to_new_over_new_to_old = d_logG_dt * (new_node_t - old_node_t);

  // Forbid zero-length branches or mutation times that coincide with node times
  if (new_node_t == t_min || new_node_t == t_max) { return; }

  CHECK(t_min <= new_node_t) << lambda_at_node << ","
                             << d_logG_dt << "," << t_min << "," << t_max << "," << new_node_t;
  CHECK(new_node_t <= t_max);

  // See if coalescent prior agrees with us
  auto delta_log_G = d_logG_dt * (new_node_t - old_node_t);
  auto delta_log_prior =
      coalescent_prior_part_->calc_delta_partial_log_prior_after_displace_tip(old_node_t, new_node_t);
  auto log_mh = delta_log_G + delta_log_prior - log_alpha_old_to_new_over_new_to_old;
  if (log_mh >= 0.0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_mh)) {
    // Accept
    move_accepted_ = true;
    coalescent_prior_part_->tip_displaced(old_node_t, new_node_t);
    tree_.at(node).t = new_node_t;

    auto dt = new_node_t - old_node_t;
    log_G_ += d_logG_dt * dt;
    log_augmented_coalescent_prior_ += delta_log_prior;
  }
}

auto Subrun::branch_reform_move() -> void {
  if (std::ssize(tree_) < 3) return;

  auto X = pick_random_node();
  if (X == tree_.root) { return; }  // The branch above the root never has (meaningful) mutations

  auto P = tree_.at(X).parent;
  auto S = tree_.at(P).sibling_of(X);
  auto t_X = tree_.at(X).t;
  auto t_P = tree_.at(P).t;

  if (P == tree_.root) {
    // Allow the mutations on the P-X and P-S branch to dance;
    // the logic below would artificially only reform the mutations on the P-X branch
    spr_move_core(X, {S, t_P}, 1.0);
    //return;  // Fall through instead
  }
  
  // Proposal
  auto& old_mutations = tree_.at(X).mutations;
  auto new_mutations = randomize_branch_mutation_times(tree_, X, bitgen_);

  // Accept/reject
  auto delta_log_G = 
      calc_branch_log_G(t_P, t_X, lambda_i_.at(X), *evo_, new_mutations)
      - calc_branch_log_G(t_P, t_X, lambda_i_.at(X), *evo_, old_mutations);
  auto log_mh = delta_log_G;
  auto accept = log_mh >= 0.0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_mh);
  
  if (accept) {
    move_accepted_ = true;
    tree_.at(X).mutations.assign(new_mutations.begin(), new_mutations.end());
    log_G_ += delta_log_G;
  }
}

auto Subrun::history_sampler() -> Site_history_sampler& {
  if (not history_sampler_.has_value()) { history_sampler_.emplace(*evo_); }
  return *history_sampler_;
}

// C1.  Redraws the explicit history of a few sites on one branch from the exact conditional CTMC
// bridge between the states the sites have at the two ends of the branch.  The branch and the sites
// are drawn uniformly, independently of the current history, so this is a random-scan Gibbs step and
// is always accepted; `delta_log_G` is bookkeeping, not an acceptance ratio.  Unlike the all-sites
// proposal, nothing here degrades as the sequence gets longer or more divergent.
auto Subrun::branch_history_gibbs_move() -> void {
  if (std::ssize(tree_) < 2) { return; }
  auto L = tree_.num_sites();
  if (L == 0) { return; }

  auto X = pick_random_node();
  if (X == tree_.root) { return; }  // the branch above the root carries no history
  auto P = tree_.at(X).parent;

  auto& sampler = history_sampler();
  // The sequence at the parent does not change during this move, so it is read out once and shared by
  // every site the move visits; walking to the root once per site would cost the same for one site and
  // k times as much for k of them.
  auto parent_sequence = view_of_sequence_at(tree_, tree_.node_loc(P));
  for (auto i = 0; i != history_gibbs_sites_per_move_; ++i) {
    auto l = absl::Uniform<Site_index>(absl::IntervalClosedOpen, bitgen_, 0, L);
    ++history_gibbs_stats_.branch_updates_proposed;
    auto state_at_parent = Real_seq_letter{parent_sequence[l]};
    auto result = gibbs_resample_branch_site_history(tree_, X, l, state_at_parent, sampler, bitgen_);
    if (not result.performed) { continue; }  // the site is missing here: the identity is the Gibbs step
    ++history_gibbs_stats_.branch_updates_performed;
    history_gibbs_stats_.events_removed += result.num_old_events;
    history_gibbs_stats_.events_created += result.num_new_events;
    log_G_ += result.delta_log_G;
    move_accepted_ = true;
  }
}

// C2.  Redraws the state of a few sites at one inner node from the full conditional given the states
// at the far ends of the incident branches, and then the conditional histories of those branches.
// This is what lets an ancestral state change without resampling the whole genome.
//
// The root of the part is only eligible when it is the root of the whole tree: the cut point that
// joins one part to another has to keep the sequence the other part was built from.
auto Subrun::node_state_gibbs_move() -> void {
  if (std::ssize(tree_) < 3) { return; }
  auto L = tree_.num_sites();
  if (L == 0) { return; }

  auto V = pick_random_inner_node();
  if (V == tree_.root && not includes_run_root_) { return; }

  auto& sampler = history_sampler();
  // As in the branch update: the sequence above `V` is fixed for the whole move.
  auto parent_sequence = (V == tree_.root)
      ? Sequence_overlay{tree_.ref_sequence}
      : view_of_sequence_at(tree_, tree_.node_loc(tree_.at(V).parent));
  for (auto i = 0; i != history_gibbs_sites_per_move_; ++i) {
    auto l = absl::Uniform<Site_index>(absl::IntervalClosedOpen, bitgen_, 0, L);
    ++history_gibbs_stats_.node_updates_proposed;
    auto state_at_parent = Real_seq_letter{parent_sequence[l]};
    auto result = gibbs_resample_node_site_state(tree_, V, l, state_at_parent, sampler, bitgen_);
    if (not result.performed) { continue; }
    ++history_gibbs_stats_.node_updates_performed;
    if (result.state_changed) { ++history_gibbs_stats_.node_states_changed; }
    history_gibbs_stats_.events_removed += result.num_old_events;
    history_gibbs_stats_.events_created += result.num_new_events;
    log_G_ += result.delta_log_G;
    lambda_i_[V] += result.delta_lambda_at_node;
    move_accepted_ = true;
  }
}


// Phase F.  A node-time move with the histories of the incident branches integrated out.
//
// `Subrun::inner_node_displace_move` bounds the node time by the mutation times on the incident
// branches, so a node can cross at most one mutation gap per step and the step shrinks as the
// mutation density grows.  This move instead integrates those histories out, proposes the node time
// against the local marginal likelihood alone, and then draws the histories back from their exact
// conditional at the new time.  The proposal density of the new history cancels against the history's
// own factor in the target, so the acceptance ratio is
//
//   (local marginal likelihood) x (coalescent prior) x (calibration prior) x (Hastings and Jacobian),
//
// with no reference to either history.  See doc/extended/history_gibbs.md, section 10.
auto Subrun::node_time_marginal_move() -> void {
  if (std::ssize(tree_) < 3) { return; }
  auto L = tree_.num_sites();
  if (L == 0) { return; }

  auto V = pick_random_inner_node();
  if (V == tree_.root && not includes_run_root_) { return; }

  // The support is bounded by the surrounding node times and by any calibration, and by nothing else:
  // the mutation times that bound the existing move are about to be discarded.
  auto t_min = (V == tree_.root) ? -std::numeric_limits<double>::infinity()
                                 : tree_.at_parent_of(V).t;
  auto t_max = +std::numeric_limits<double>::infinity();
  for (const auto& child : tree_.at(V).children) { t_max = std::min(t_max, tree_.at(child).t); }
  if (not calibrations_.empty()) {
    auto [cal_t_min, cal_t_max] = calibration_t_bounds(V);
    t_min = std::max(t_min, cal_t_min);
    t_max = std::min(t_max, cal_t_max);
  }
  if (not (t_min < t_max)) { return; }

  auto old_t = tree_.at(V).t;
  auto new_t = old_t;
  auto log_hastings = 0.0;  // log q(new -> old) - log q(old -> new), Jacobian included
  if (V == tree_.root) {
    // The root's depth below its oldest child is multiplied by exp(u), u uniform on [-w, w].  The
    // proposal is scale free, so it crosses the posterior of a deep root in a few steps whatever the
    // mutation density; the Jacobian of the reparametrization is exp(u).
    auto depth = t_max - old_t;
    if (not (depth > 0.0)) { return; }
    auto u = absl::Uniform(bitgen_, -marginal_node_time_window_, marginal_node_time_window_);
    auto new_depth = depth * std::exp(u);
    // Guard: evaluating the coalescent prior needs a grid covering the proposed range, and an absurd
    // proposal would allocate an absurd grid.  `Run::set_marginal_node_time_window` keeps the window
    // below `max_marginal_node_time_window()`, which makes this bound unreachable, so the guard never
    // rejects and the kernel stays reversible; it is kept as a safety net.
    if (not (new_depth > 0.0)
        || new_depth > marginal_node_time_max_depth_factor * std::max(depth, 1.0)) { return; }
    new_t = t_max - new_depth;
    if (new_t <= t_min) { return; }
    log_hastings = u;
  } else {
    if (not std::isfinite(t_min)) { return; }
    new_t = absl::Uniform(absl::IntervalOpenOpen, bitgen_, t_min, t_max);
    log_hastings = 0.0;  // uniform on an interval the move does not change
  }
  if (new_t <= t_min || new_t >= t_max) { return; }

  auto& sampler = history_sampler();
  // The sequences just outside the three branches do not depend on where `V` sits, so they are read
  // out once and shared by both evaluations and by the redraw.
  auto outside = read_incident_sequences(tree_, V);
  auto log_F_old = calc_local_marginal_log_likelihood(tree_, V, old_t, outside, sampler);
  auto log_F_new = calc_local_marginal_log_likelihood(tree_, V, new_t, outside, sampler);
  if (not std::isfinite(log_F_old) || not std::isfinite(log_F_new)) { return; }

  auto delta_log_prior =
      coalescent_prior_part_->calc_delta_partial_log_prior_after_displace_coalescence(old_t, new_t);
  auto delta_log_calibration_prior = calc_delta_log_calibration_prior(V, old_t, new_t);
  auto log_mh = (log_F_new - log_F_old) + delta_log_prior + delta_log_calibration_prior + log_hastings;
  if (not (log_mh >= 0.0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_mh))) { return; }

  // Accepted.  The three incident branches are the only part of the genealogy that changes, so their
  // contribution to log G is recomputed explicitly before and after.
  auto branch_log_G = [this](Node_index X) {
    return calc_branch_log_G(tree_.at_parent_of(X).t, tree_.at(X).t, lambda_i_[X], *evo_,
                             tree_.at(X).mutations);
  };
  auto before = 0.0;
  if (V != tree_.root) { before += branch_log_G(V); }
  for (const auto& child : tree_.at(V).children) { before += branch_log_G(child); }
  if (V == tree_.root) {
    before += calc_log_root_prior(tree_, *evo_, state_frequencies_of_ref_sequence_per_partition_);
  }

  coalescent_prior_part_->coalescence_displaced(old_t, new_t);
  tree_.at(V).t = new_t;
  auto refresh = gibbs_refresh_node_histories(tree_, V, outside, sampler, bitgen_);

  // `lambda_i` at `V` follows the new sequence there; at the children it is unchanged in exact
  // arithmetic, and is recomputed from `V` so that it stays exactly consistent.
  auto lambda_above_V = (V == tree_.root) ? ref_cum_Q_l_->back() : lambda_i_[tree_.at(V).parent];
  lambda_i_[V] = lambda_above_V + calc_delta_lambda_across_branch(
      *evo_, tree_.ref_sequence, *ref_cum_Q_l_, tree_.at(V).mutations, tree_.at(V).missations);
  for (const auto& child : tree_.at(V).children) {
    lambda_i_[child] = lambda_i_[V] + calc_delta_lambda_across_branch(
        *evo_, tree_.ref_sequence, *ref_cum_Q_l_, tree_.at(child).mutations,
        tree_.at(child).missations);
  }

  auto after = 0.0;
  if (V != tree_.root) { after += branch_log_G(V); }
  for (const auto& child : tree_.at(V).children) { after += branch_log_G(child); }
  if (V == tree_.root) {
    after += calc_log_root_prior(tree_, *evo_, state_frequencies_of_ref_sequence_per_partition_);
  }

  log_G_ += after - before;
  log_augmented_coalescent_prior_ += delta_log_prior;
  log_calibration_prior_ += delta_log_calibration_prior;
  history_gibbs_stats_.node_updates_proposed += L;
  history_gibbs_stats_.node_updates_performed += L;
  history_gibbs_stats_.node_states_changed += refresh.num_states_changed;
  history_gibbs_stats_.events_removed += refresh.num_old_events;
  history_gibbs_stats_.events_created += refresh.num_new_events;
  move_accepted_ = true;
}

// Enumerate all branches that straddle time tree and are at or below the branch ending at P,
// not at or below the branch ending at X.  The output iterator accumulates the Node_index's
// of the endpoint of the relevant branches.
template<typename Output_iterator>
static auto enumerate_descendant_branches_straddling(
    const Phylo_tree& tree,
    Node_index P,
    double t,
    Node_index X,
    Output_iterator out)
    -> Output_iterator {
  
  DCHECK(P == tree.root || tree.at_parent_of(P).t <= t);

  if (P == X) {
    // Skip the branch ending at X and all of its descendants
    return out;
  }

  if (t <= tree.at(P).t) {
    *out = P;
    ++out;
  } else if (tree.at(P).is_inner_node()) {
    for (const auto& child : tree.at(P).children) {
      out = enumerate_descendant_branches_straddling(tree, child, t, X, out);
    }
  }
  return out;
}

auto Subrun::subtree_slide_move() -> void {
  if (std::ssize(tree_) < 2) return;

  // Comments below purposely track those in SubtreeSlideOperator#doOperation() in BEAST

  // 1. choose a random node avoiding root
  auto X = pick_random_node();
  if (X == tree_.root) { return; }

  auto P = tree_.at(X).parent;
  auto S = tree_.at(P).sibling_of(X);  // "CiP"

  // 2. choose a delta to move
  // Cap the proposal scale to the tree span so that we don't propose
  // absurdly distant parent times when mu is tiny (making lambda_i
  // negligible and delta_scale enormous).  Such proposals would be
  // rejected by the coalescent prior anyway, but evaluating the prior
  // can consume GiB of memory.
  // See plans/2026-03-26-02-cap-subtree-slide-proposal.md for details.
  auto t_early = (P == tree_.root)
      ? std::min(tree_.at(X).t, tree_.at(S).t)
      : tree_.at(tree_.root).t;
  auto tree_span = t_max_tip_ - t_early;
  CHECK_GE(tree_span, 0.0);
  auto delta_scale = std::min((1 / lambda_i_.at(X)) / 2,  // 95% of the time, won't hop past more than 1 mutation
                              tree_span);
  auto delta_t = absl::Gaussian(bitgen_, 0.0, delta_scale);
  auto old_P_t = tree_.at(P).t;
  auto new_P_t = old_P_t + delta_t;

  // 3. if the move is up
  if (delta_t < 0.0) {

    // 3.1 if the topology will change
    if (P != tree_.root && new_P_t < tree_.at_parent_of(P).t) {
      if (not topology_moves_enabled()) { move_blocked_ = true; return; }
      // find new parent
      auto GG = tree_.at(P).parent;  // new grandparent after move
      auto SS = P;                  // new sibling after move
      while (new_P_t < tree_.at(GG).t) {
        SS = GG;
        GG = tree_.at(GG).parent;
        if (GG == k_no_node) {
          break;
        }
      }
      CHECK((GG == k_no_node || tree_.at(GG).t <= new_P_t) && new_P_t <= tree_.at(SS).t);

      // 3.1.1 if creating a new root... => tree rewired in spr_move_core
      // 3.1.2 no new root... => tree rewired in spr_move_core

      // 3.1.3 count the hypothetical sources of this destination
      auto branches = Scratch_vector<Node_index>{};
      enumerate_descendant_branches_straddling(tree_, SS, old_P_t, X, std::back_inserter(branches));

      auto alpha_old_to_new = 1.0;
      auto alpha_new_to_old = 1.0 / std::ssize(branches);

      spr_move_core(X, Phylo_tree_loc{SS, new_P_t}, alpha_new_to_old / alpha_old_to_new);
    } else {
      // just change the node height
      spr_move_core(X, Phylo_tree_loc{S, new_P_t}, 1.0);
    }

  } else {
    // 4 if we are sliding the subtree down.

    // 4.0 is it a valid move?
    if (new_P_t > tree_.at(X).t) {
      return;
    }

    // 4.1 will the move change the topology
    if (new_P_t > tree_.at(S).t) {
      if (not topology_moves_enabled()) { move_blocked_ = true; return; }

      auto branches = Scratch_vector<Node_index>{};
      enumerate_descendant_branches_straddling(tree_, P, new_P_t, X, std::back_inserter(branches));

      // if no valid destinations then return a failure
      if (branches.empty()) {
        return;
      }

      // pick a random parent/child destination edge uniformly from options
      auto branch_index = absl::Uniform(absl::IntervalClosedOpen, bitgen_, 0, std::ssize(branches));
      auto SS = branches[branch_index];

      // 4.1.1 if iP was root... => tree rewired in spr_move_core
      auto alpha_old_to_new = 1.0 / static_cast<double>(branches.size());
      auto alpha_new_to_old = 1.0;

      spr_move_core(X, Phylo_tree_loc{SS, new_P_t}, alpha_new_to_old / alpha_old_to_new);
    } else {
      // just change the node height
      spr_move_core(X, Phylo_tree_loc{S, new_P_t}, 1.0);
    }
  }
}

auto Subrun::wilson_balding_move() -> void {
  if (std::ssize(tree_) < 2) return;

  // Comments below purposely track those in WilsonBalding#proposeTree() in BEAST

  // choose a random node avoiding root
  auto X = Node_index{};
  do {
    X = pick_random_node();
  } while (tree_.root == X);
  auto P = tree_.at(X).parent;

  // choose another random node to insert X above
  auto SS = pick_random_node();  // new sibling of X after move
  auto GG = tree_.at(SS).parent;  // new grandparent of X after move
  
  // make sure that the target branch <GG, SS> is above the subtree being moved
  while ((GG != k_no_node && tree_.at(GG).t >= tree_.at(X).t) || (X == SS)) {
    SS = pick_random_node();
    GG = tree_.at(SS).parent;
  }

  // disallow moves that change the root
  if (SS == tree_.root || P == tree_.root) {
    return;
  }

  if (GG == P || SS == P || GG == X) {
    return;
  }

  auto S = tree_.at(P).sibling_of(X);
  auto G = tree_.at(P).parent;
  auto new_max_age = std::min(tree_.at(X).t, tree_.at(SS).t);
  auto new_range = new_max_age - tree_.at(GG).t;
  auto new_t_P = absl::Uniform(bitgen_, new_max_age - new_range, new_max_age);
  auto old_max_age = std::min(tree_.at(X).t, tree_.at(S).t);
  auto old_range = old_max_age - tree_.at(G).t;

  spr_move_core(X, Phylo_tree_loc{SS, new_t_P}, new_range / old_range);
}

auto Subrun::spr1_move() -> void {
  if (std::ssize(tree_) < 2) return;

  // An SPR move can change which node is the MRCA of a calibrated sample set.  The targets are re-identified on
  // the proposed tree and the calibration term enters the MH ratio below.
  CHECK(calibrations_.empty() || calibration_targets_follow_topology());

  auto chooser = absl::Uniform(bitgen_, 0.0, 1.0);
  auto limit =
      chooser < 0.01 ? std::numeric_limits<int>::max() :
      //chooser < 0.02 ? 2 :
      1;
  
  // Effective mu for all Jukes-Cantor-like calculations
  auto mu_JC = lambda_i_.at(tree_.root) / (tree_.num_sites() - num_sites_missing_at_every_node_.at(tree_.root));
  //std::cerr << absl::StreamFormat("mu_JC = %.3g x 10^3 / site / year\n",
  //                                mu_JC * 365.0 * 1e3);

  // Instead of picking it exactly in the case of a Jukes-Cantor model, we dampen the proposal
  // probability by an exponent `f` < 1.  See comments in `Spr_Study` constructor for details.
  // That way, we systematically somewhat under-propose the low-mutation regions found in the study,
  // which gives the MH criterion ample wiggle room to add corrections without accidentally
  // rejecting an otherwise great move
  auto annealing_factor = 0.8;

  // choose a random node avoiding root
  auto X = Node_index{};
  do {
    X = pick_random_node();
  } while (tree_.root == X);
  if (lambda_i_.at(X) == 0.0) return;  // All sites missing at X => SPR study undefined
  auto t_X = tree_.at(X).t;
  auto P = tree_.at(X).parent;
  auto old_t_P = tree_.at(P).t;
  auto old_S = tree_.at(P).sibling_of(X);
  auto old_G = tree_.at(P).parent;  // may be k_no_node

  auto pruning_changes_root = P == tree_.root;
  
  if (pruning_changes_root && not includes_run_root_) {
    // This move would change the root, and we can't do that here
    return;
  }

  // 0. Kick off SPR move by cataloguing and removing all of X's upstream mutations
  // FIXME: There *must* be a way to share the spr move core here
  auto spr = Spr_move{tree_, mu_JC, includes_run_root_,
    *evo_, lambda_i_, *ref_cum_Q_l_, num_sites_missing_at_every_node_};

  auto old_graft = spr.analyze_graft(X);
  spr.peel_graft(old_graft);
  if (pruning_changes_root) {
    CHECK(tree_.at(old_S).mutations.empty());  // This is part of what `peel_graft` does
    CHECK(tree_.at(P).mutations.empty() || tree_.at(P).mutations.back().t == -std::numeric_limits<double>::max());
  }
  
  // 1. Perform an SPR study around P
  auto old_min_muts = spr.count_min_mutations(old_graft);
  auto old_num_muts = spr.count_closed_mutations(old_graft);
  auto old_deltas_P_to_X = spr.summarize_closed_mutations(old_graft);
  auto missing_at_X = reconstruct_missing_sites_at(tree_, X);

  auto pre_builder = Spr_study_builder{tree_, X, t_X, missing_at_X};
  pre_builder.max_muts_from_start = limit; // Stop when crossing over more than 1 mutation (makes move local)
  pre_builder.seed_fill_from(old_S, 0, std::move(old_deltas_P_to_X), includes_run_root_);

  auto pre_study = Spr_study{std::move(pre_builder), lambda_i_.at(X), annealing_factor, t_X, t_max_tip_};

  
  // 2. Use SPR study to pick new nexus
  auto new_nexus_region = pre_study.pick_nexus_region(bitgen_);
  auto new_S = pre_study.candidate_regions[new_nexus_region].branch;
  CHECK_NE(new_S, P);
  auto new_t_P = pre_study.pick_time_in_region(new_nexus_region, bitgen_);
  auto log_alpha_old_to_new = pre_study.log_alpha_in_region(new_nexus_region, new_t_P);

  auto new_nexus_min_muts = pre_study.candidate_regions[new_nexus_region].min_muts;
  
  auto t_new_S = tree_.at(new_S).t;
  auto new_G = tree_.at(new_S).parent;  // may be k_no_node
  if (new_G == P) { new_G = old_G; }  // Sliding along the same branch
  auto t_new_G = (new_G == k_no_node) ? -std::numeric_limits<double>::max() : tree_.at(new_G).t;
  
  // Forbid zero-length branches
  if (new_t_P == t_X || new_t_P == t_new_S || new_t_P == t_new_G) {
    spr.apply_graft(old_graft);
    return;
  }

  
  // 3. Pre-flight SPR move
  spr.move(X, new_S, new_t_P);
  auto new_graft = spr.propose_new_graft(X, bitgen_);

  CHECK_EQ(tree_.at(X).parent, P);

  auto grafting_changes_root = P == tree_.root;
  CHECK(includes_run_root_ || not grafting_changes_root);
  if (grafting_changes_root) {
    CHECK(tree_.at(new_S).mutations.empty());  // No mutations should have been added by spr.move()
    CHECK(tree_.at(P).mutations.empty() || tree_.at(P).mutations.back().t == -std::numeric_limits<double>::max());
  }

  // 4. Perform an SPR study from the new position to calculate probability of reverse proposal
  auto new_min_muts = spr.count_min_mutations(new_graft);
  auto new_num_muts = spr.count_closed_mutations(new_graft);
  auto new_deltas_P_to_X = spr.summarize_closed_mutations(new_graft);
  auto post_builder = Spr_study_builder{tree_, X, t_X, missing_at_X};
  post_builder.max_muts_from_start = limit; // Stop when crossing over more than 1 mutation (makes move local)
  post_builder.seed_fill_from(new_S, 0, std::move(new_deltas_P_to_X), includes_run_root_);
  
  auto post_study = Spr_study{std::move(post_builder), lambda_i_.at(X), annealing_factor, t_X, t_max_tip_};

  auto old_nexus_region = post_study.find_region(old_S, old_t_P);
  CHECK_NE(old_nexus_region, -1);
  auto log_alpha_new_to_old = post_study.log_alpha_in_region(old_nexus_region, old_t_P);


  // 4bis. Sanity checking
  CHECK_EQ(new_min_muts, pre_study.candidate_regions[new_nexus_region].min_muts);
  CHECK_EQ(old_min_muts, post_study.candidate_regions[old_nexus_region].min_muts);
  
  
  // 5. Accept/reject?
  // Calibrations: the targets of the proposed tree, and zero density if a calibrated set stopped being
  // monophyletic or a target left its hard support.  Such a proposal is rejected, not redrawn.
  auto old_calibration_nodes = save_calibration_nodes();
  auto delta_log_calibration_prior = 0.0;
  if (not calibrations_.empty()) {
    delta_log_calibration_prior = calc_rebound_log_calibration_prior() - log_calibration_prior_;
  }
  auto delta_log_augmented_coalescent_prior =
      coalescent_prior_part_->calc_delta_partial_log_prior_after_displace_coalescence(
          old_t_P, new_t_P);
  auto log_mh = (new_graft.delta_log_G - new_graft.log_alpha_mut)
      - (old_graft.delta_log_G - old_graft.log_alpha_mut)
      + log_alpha_new_to_old - log_alpha_old_to_new
      + delta_log_augmented_coalescent_prior
      + delta_log_calibration_prior;

  refresh_clade_tips_after_regraft(P, old_S);
  auto partition_kept = threshold_partition_holds_after_regraft(P, old_S);
  if (not partition_kept) { move_blocked_by_partition_ = true; }
  auto accept = partition_kept && std::isfinite(delta_log_calibration_prior)
      && (log_mh >= 0.0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_mh));

  if (false && (pruning_changes_root || grafting_changes_root)) {
    std::cerr << absl::StreamFormat("old_t_P = %g, new_t_P = %g, accept = %s\n",
                                    old_t_P, new_t_P, accept ? "true" : "false");
  }
  
  if (false && (pruning_changes_root || grafting_changes_root) && not accept) { // && (new_num_muts < old_num_muts-5)) {
    auto mu = evo_->mu_l(0);
    auto L = tree_.num_sites();

    if (pruning_changes_root || grafting_changes_root) {
      std::cerr << "****** ";
    } else if (old_graft.branch_infos.size() > 1 || new_graft.branch_infos.size() > 1) {
      std::cerr << "++++++ ";
    }
    std::cerr << absl::StreamFormat("Odd rejection: old %d, new_min %d, new %d, t_n-t_o=%g, log_mh=%g\n",
                                    old_num_muts, new_nexus_min_muts, new_num_muts, new_t_P - old_t_P, log_mh);
    std::cerr << absl::StreamFormat(" - delta_log_augmented_coalescent_prior: %g\n",
                                    delta_log_augmented_coalescent_prior);
    std::cerr << absl::StreamFormat(" - new_graft.delta_log_G = %g, old_graft.delta_log_G = %g => \n",
                                    new_graft.delta_log_G, old_graft.delta_log_G);
    std::cerr << absl::StreamFormat("   log(G_n/G_0) = %g (cf JC expectation = %g)\n",
                                    new_graft.delta_log_G - old_graft.delta_log_G,
                                    -mu*L * (old_t_P - new_t_P)
                                    + (new_num_muts - old_num_muts)*std::log(mu / 3.0));
    std::cerr << absl::StreamFormat(" - new_graft.log_alpha_mut = %g, old_graft.log_alpha_mut = %g =>\n",
                                    new_graft.log_alpha_mut, old_graft.log_alpha_mut);
    std::cerr << absl::StreamFormat("   log(a_m(n->o)/a_m(o->n)) = %g (cf JC expectation = %g)\n",
                                    old_graft.log_alpha_mut - new_graft.log_alpha_mut,
                                    -old_num_muts*std::log(t_X - old_t_P) + new_num_muts*std::log(t_X - new_t_P));
    std::cerr << absl::StreamFormat(" - log_alpha_new_to_old = %g, log_alpha_old_to_new = %g =>\n",
                                    log_alpha_new_to_old, log_alpha_old_to_new);
    std::cerr << absl::StreamFormat("   log(a_SPR(n->o)/a_SPR(o->n)) = %g (cf JC expectation = %g)\n",
                                    log_alpha_new_to_old - log_alpha_old_to_new,
                                    -mu*L * (new_t_P - old_t_P)
                                    + old_num_muts*std::log(mu*(t_X - old_t_P)/3.0)
                                    - new_num_muts*std::log(mu*(t_X - new_t_P)/3.0));
    std::cerr << absl::StreamFormat("                                     (cf JC expectation with f = %g)\n",
                                    annealing_factor * (
                                        -mu*L * (new_t_P - old_t_P)
                                        + old_num_muts*std::log(mu*(t_X - old_t_P)/3.0)
                                        - new_num_muts*std::log(mu*(t_X - new_t_P)/3.0)));
  }
  
  if (accept) {
    move_accepted_ = true;
    spr.apply_graft(new_graft);
    log_G_ -= old_graft.delta_log_G;
    log_G_ += new_graft.delta_log_G;
    log_augmented_coalescent_prior_ += delta_log_augmented_coalescent_prior;
    log_calibration_prior_ += delta_log_calibration_prior;
    coalescent_prior_part_->coalescence_displaced(old_t_P, new_t_P);
  } else {
    spr.move(X, old_S, old_t_P);
    spr.apply_graft(old_graft);
    restore_calibration_nodes(old_calibration_nodes);
    refresh_clade_tips_after_regraft(P, new_S);
  }
}

// SPR = Subtree Pruning and Regrafting
// Prune the subtree rooted at node X and regraft it to just above SS at time new_P_t
// (which must be between tree.at_parent_of(SS).t and tree.at(SS).t, calculated *after* X is pruned).
// If the SPR move is accepted, then the nodes SS and X become siblings.
// The choice of this configuration has accumulated a Hastings ratio
// of alpha_new_to_old_over_alpha_old_to_new
auto Subrun::spr_move_core(
    Node_index X,
    Phylo_tree_loc new_nexus,
    double alpha_new_to_old_over_alpha_old_to_new)
    -> void {

  if (X == tree_.root) { return; }  // Really can't do this!
  if (not includes_run_root_) {
    if (tree_.at(X).parent == tree_.root || new_nexus.branch == tree_.root) {
      // This move would change the root, and we can't do that here
      return;
    }
  }

  auto t_X = tree_.at(X).t;
  auto P = tree_.at(X).parent;
  auto old_t_P = tree_.at(P).t;
  auto old_S = tree_.at(P).sibling_of(X);
  auto new_t_P = new_nexus.t;

  // Forbid zero-length branches
  if (new_t_P == t_X ||
      (new_t_P == tree_.at(new_nexus.branch).t) ||
      (P != tree_.root && new_t_P == tree_.at_parent_of(P).t)) {  // FIXME: This should look at G', not G!
    return;
  }

  // A proposal outside the hard support of an age calibration on P would be rejected below; return
  // before doing the work.  This leaves the state untouched, exactly as a rejection does.  When the targets
  // follow the topology, P may stop or start being a target, so the term is evaluated on the proposed tree.
  auto targets_follow_topology = calibration_targets_follow_topology();
  auto delta_log_calibration_prior = 0.0;
  if (not targets_follow_topology) {
    delta_log_calibration_prior = calc_delta_log_calibration_prior(P, old_t_P, new_t_P);
    if (delta_log_calibration_prior == -std::numeric_limits<double>::infinity()) { return; }
  }
  
  auto mu_JC = lambda_i_.at(tree_.root) / (tree_.num_sites() - num_sites_missing_at_every_node_.at(tree_.root));
  //std::cerr << absl::StreamFormat("mu_JC = %.3g x 10^3 / site / year\n",
  //                                mu_JC * 365.0 * 1e3);
  auto spr = Spr_move{tree_, mu_JC, includes_run_root_,
    *evo_, lambda_i_, *ref_cum_Q_l_, num_sites_missing_at_every_node_};

  auto old_graft = spr.analyze_graft(X);
  spr.peel_graft(old_graft);
  spr.move(X, new_nexus.branch, new_nexus.t);
  auto new_graft = spr.propose_new_graft(X, bitgen_);

  auto old_calibration_nodes = save_calibration_nodes();
  if (targets_follow_topology) {
    delta_log_calibration_prior = calc_rebound_log_calibration_prior() - log_calibration_prior_;
  }
  
  // Accept/reject?
  auto delta_log_augmented_coalescent_prior =
      coalescent_prior_part_->calc_delta_partial_log_prior_after_displace_coalescence(
          old_t_P, new_nexus.t);
  auto log_mh = (new_graft.delta_log_G - new_graft.log_alpha_mut)
      - (old_graft.delta_log_G - old_graft.log_alpha_mut)
      + std::log(alpha_new_to_old_over_alpha_old_to_new)
      + delta_log_augmented_coalescent_prior
      + delta_log_calibration_prior;

  refresh_clade_tips_after_regraft(P, old_S);
  auto partition_kept = threshold_partition_holds_after_regraft(P, old_S);
  if (not partition_kept) { move_blocked_by_partition_ = true; }
  auto accept = partition_kept && std::isfinite(delta_log_calibration_prior)
      && (log_mh >= 0.0 || absl::Uniform(bitgen_, 0.0, 1.0) < std::exp(log_mh));
  
  if (accept) {
    move_accepted_ = true;
    spr.apply_graft(new_graft);
    log_G_ -= old_graft.delta_log_G;
    log_G_ += new_graft.delta_log_G;
    log_augmented_coalescent_prior_ += delta_log_augmented_coalescent_prior;
    log_calibration_prior_ += delta_log_calibration_prior;
    coalescent_prior_part_->coalescence_displaced(old_t_P, new_nexus.t);
  } else {
    spr.move(X, old_S, old_t_P);
    spr.apply_graft(old_graft);
    restore_calibration_nodes(old_calibration_nodes);
    refresh_clade_tips_after_regraft(P, new_nexus.branch);
  }
}

auto Subrun::set_threshold_partition_constraint(std::optional<Threshold_partition_constraint> constraint) -> void {
  threshold_constraint_ = std::move(constraint);
  clade_tips_.clear();
  if (not threshold_constraint_) { return; }
  const auto& c = *threshold_constraint_;
  CHECK(includes_run_root_) << "Only the root part of a threshold partition is constrained";
  CHECK_GE(c.min_cut_tips, 2);
  CHECK_LE(c.min_cut_tips, c.max_cut_tips);
  CHECK_EQ(std::ssize(c.tip_weights), std::ssize(tree_));
  clade_tips_.assign(std::ssize(tree_), 0);
  for (auto node : post_order_traversal(tree_)) {
    if (tree_.at(node).is_tip()) {
      CHECK_GE(c.tip_weights[node], 1);
      clade_tips_[node] = c.tip_weights[node];
    } else {
      clade_tips_[node] = clade_tips_[tree_.at(node).left_child()] + clade_tips_[tree_.at(node).right_child()];
    }
  }
  CHECK(satisfies_threshold_partition_constraint()) << "The tree does not have the threshold partition it was cut by";
}

auto Subrun::threshold_partition_holds_at(Node_index node) const -> bool {
  if (node == tree_.root) { return true; }
  const auto& c = *threshold_constraint_;
  auto big_parent = clade_tips_[tree_.at(node).parent] > c.max_cut_tips;
  if (tree_.at(node).is_tip() && c.tip_weights[node] > 1) { return big_parent; }  // a cut point
  auto in_window = clade_tips_[node] >= c.min_cut_tips && clade_tips_[node] <= c.max_cut_tips;
  return not (in_window && big_parent);
}

auto Subrun::satisfies_threshold_partition_constraint() const -> bool {
  if (not threshold_constraint_) { return true; }
  for (auto node : index_order_traversal(tree_)) {
    if (not threshold_partition_holds_at(node)) { return false; }
  }
  return true;
}

// Only the nodes above P and above S have gained or lost tips.  The second walk recomputes, from where the two paths
// meet, every node that the first one computed from a child the second had not reached yet.
auto Subrun::refresh_clade_tips_after_regraft(Node_index P, Node_index S) -> void {
  if (not threshold_constraint_) { return; }
  auto refresh_above = [this](Node_index node) {
    for (; node != k_no_node; node = tree_.at(node).parent) {
      if (tree_.at(node).is_inner_node()) {
        clade_tips_[node] = clade_tips_[tree_.at(node).left_child()] + clade_tips_[tree_.at(node).right_child()];
      }
    }
  };
  refresh_above(P);
  refresh_above(tree_.at(S).parent);
}

// Whether a node is a cut point depends on its own size and on its parent's; both are unchanged except on the two paths
// above P and S and at the children of their nodes.
auto Subrun::threshold_partition_holds_after_regraft(Node_index P, Node_index S) const -> bool {
  if (not threshold_constraint_) { return true; }
  auto holds_above = [this](Node_index node) {
    for (; node != k_no_node; node = tree_.at(node).parent) {
      if (not threshold_partition_holds_at(node)) { return false; }
      for (auto child : tree_.at(node).children) {
        if (not threshold_partition_holds_at(child)) { return false; }
      }
    }
    return true;
  };
  return holds_above(P) && holds_above(S);
}

}  // namespace delphy
