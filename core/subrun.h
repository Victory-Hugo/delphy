#ifndef DELPHY_SUBRUN_H_
#define DELPHY_SUBRUN_H_

#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/random/bit_gen_ref.h"

#include "calibration.h"
#include "estd.h"
#include "evo_model.h"
#include "history_gibbs.h"
#include "phylo_tree.h"
#include "very_scalable_coalescent.h"
#include "sequence_overlay.h"
#include "scratch_space.h"

namespace delphy {

// Largest factor by which one proposal of the marginal node-time move may change the root's depth
// below its oldest child.  The move needs a bound because the coalescent prior is evaluated on a grid
// that has to cover the proposed range, and an absurd proposal would allocate an absurd grid; but a
// bound that a proposal can actually reach would reject asymmetrically and break the reversibility of
// the kernel, because the reverse move from the rejected state is not rejected.  The window is
// therefore restricted to `max_marginal_node_time_window()`, which makes the bound unreachable, and
// the bound stays as a safety net that never fires.  See doc/extended/history_gibbs.md,
// section 6b.
inline constexpr auto marginal_node_time_max_depth_factor = 1e3;

// Largest half width the uniform proposal of the log of the root's depth may be given.  At or above
// this value `marginal_node_time_max_depth_factor` becomes reachable.
inline auto max_marginal_node_time_window() -> double {
  return std::log(marginal_node_time_max_depth_factor);
}

// Local move types, for acceptance diagnostics
enum class Move_kind : int {
  inner_node_displace = 0,
  tip_displace,
  branch_reform,
  subtree_slide,
  spr1,
  branch_history_gibbs,
  node_state_gibbs,
  node_time_marginal,
  num_kinds
};
inline constexpr auto k_num_move_kinds = static_cast<int>(Move_kind::num_kinds);

// `proposed` counts every time the move was selected, including selections that ended before a
// Metropolis-Hastings decision (for example, because the proposal fell outside its support).
// `blocked_by_fixed_topology` counts proposals discarded because accepting them would change the topology.
// `blocked_by_partition` counts proposals discarded because they would change the threshold partition of the batch.
struct Move_counts {
  int64_t proposed = 0;
  int64_t accepted = 0;
  int64_t blocked_by_fixed_topology = 0;
  int64_t blocked_by_partition = 0;
};

// Keeps a threshold partition (`threshold_partition_stencil` in tree_partitioning.h) unchanged across the topology moves
// of its root part.  After a proposal, every node of the part must still be a cut point exactly when it was one: a cut
// point (a tip of the part standing for a clade) keeps a parent with more than `max_cut_tips` tips, and no other node
// gets between `min_cut_tips` and `max_cut_tips` tips below a parent with more than `max_cut_tips`.  A proposal that
// breaks this has zero density in the restricted target of the part and is rejected.  `tip_weights` gives, for every
// tip of the part, the number of tips of the whole tree it stands for: 1 at a sample, the clade size (at least
// `min_cut_tips` >= 2) at a cut point.  Parts below a cut point need no constraint.
struct Threshold_partition_constraint {
  int max_cut_tips = 0;
  int min_cut_tips = 0;
  Node_vector<int> tip_weights{};
};

// One age calibration, as seen by a partition part: the local index of the calibrated node together with
// the prior, by value, so that it stays valid while `Run` rebuilds its own calibration set.
//
// With topology moves enabled, `node` is not an identity: it is re-derived from the sample set (`tips`, or the
// root when `all_tips` is set) after every topology proposal, and a proposal on which the set is not monophyletic
// has posterior density zero.  See doc/extended/calibration_free_topology.md.
struct Local_calibration {
  Node_index node = k_no_node;
  double reference_t = 0.0;
  Age_prior prior{};
  std::vector<Node_index> tips{};  // Local indices of the calibrated samples; empty when `all_tips`
  bool all_tips = false;
  bool require_monophyly = true;

  auto age_of(double t) const -> double { return (reference_t - t) / 365.0; }
  auto log_density_at(double t) const -> double { return prior.log_density(age_of(t)); }
};

class Subrun {
 public:
  Subrun(absl::BitGenRef bitgen, Phylo_tree tree, bool includes_run_root, Global_evo_model evo);
  Subrun(absl::BitGenRef bitgen, Phylo_tree tree, bool includes_run_root, std::shared_ptr<const Global_evo_model> evo);

  auto mcmc_sub_iteration() -> void;

  auto tree() const -> const Phylo_tree& { return tree_; }
  auto includes_run_root() const -> bool { return includes_run_root_; }
  auto t_max_tip() const -> double { return t_max_tip_; }
  auto set_t_max_tip(double t_max_tip) -> void { t_max_tip_ = t_max_tip; }

  // Parameters
  auto evo() const -> const Global_evo_model& { return *evo_; }
  auto set_evo(const Global_evo_model& evo) -> void { set_evo(std::make_shared<const Global_evo_model>(evo)); }
  auto set_evo(std::shared_ptr<const Global_evo_model> evo) -> void {
    evo_ = std::move(evo), given_ref_cum_Q_l_.reset(), ref_cum_Q_l_.reset(), given_state_frequencies_.reset(),
    history_sampler_.reset(), invalidate_derived_quantities(); }

  // Supplies the two statistics of the reference sequence that take a pass over every site, computed by the caller
  // from the same reference sequence and the current `evo()`.  `set_evo` retracts them.
  auto set_ref_site_tables(std::shared_ptr<const std::vector<double>> ref_cum_Q_l,
                           Partition_vector<Seq_vector<int>> state_frequencies_per_partition) -> void {
    given_ref_cum_Q_l_ = std::move(ref_cum_Q_l), given_state_frequencies_ = std::move(state_frequencies_per_partition),
    invalidate_derived_quantities(); }
  auto coalescent_prior_part() -> Very_scalable_coalescent_prior_part& { return *coalescent_prior_part_; }
  auto set_coalescent_prior_part(Very_scalable_coalescent_prior_part* coalescent_prior_part) -> void {
    coalescent_prior_part_ = coalescent_prior_part, invalidate_derived_quantities();
  }

  // Move config
  auto only_displacing_inner_nodes() const -> bool { return only_displacing_inner_nodes_; }
  auto set_only_displacing_inner_nodes(bool val) -> void { only_displacing_inner_nodes_ = val; }
  // When set, `tip_displace` is left out of the move mix if no tip of this part can move in time
  // (every tip has t_min == t_max), because that move then returns before touching the state or the
  // random stream beyond the choice of the tip: it is the identity kernel.
  auto skip_fixed_tip_moves() const -> bool { return skip_fixed_tip_moves_; }
  auto set_skip_fixed_tip_moves(bool val) -> void { skip_fixed_tip_moves_ = val; }
  // Multiplies the width of the Gaussian proposal for the time of the root of the whole tree.
  auto root_displace_scale_factor() const -> double { return root_displace_scale_factor_; }
  auto set_root_displace_scale_factor(double val) -> void { root_displace_scale_factor_ = val; }
  auto topology_moves_enabled() const -> bool { return topology_moves_enabled_; }
  auto set_topology_moves_enabled(bool val) -> void { topology_moves_enabled_ = val; }

  // Weight of each single-site history Gibbs update in the local move mix, relative to the existing
  // weights of 7.5 for the node-time moves, 15 for the branch reform and 1 each for the subtree slide
  // and the SPR.  Zero, the default, leaves the mix and the random number stream exactly as they were.
  auto history_gibbs_weight() const -> double { return history_gibbs_weight_; }
  auto set_history_gibbs_weight(double val) -> void { history_gibbs_weight_ = val; }
  // How many sites one such update visits.  The sites are drawn uniformly and independently of the
  // current history, so the update stays a random-scan Gibbs step whatever this is.
  auto history_gibbs_sites_per_move() const -> int { return history_gibbs_sites_per_move_; }
  auto set_history_gibbs_sites_per_move(int val) -> void { history_gibbs_sites_per_move_ = val; }
  // Weight of the locally marginalized node-time move, which integrates the histories of the
  // branches incident on a node out, moves the node without reference to the mutation times that
  // otherwise bound it, and draws the histories back from their exact conditional.  Zero by default.
  auto marginal_node_time_weight() const -> double { return marginal_node_time_weight_; }
  auto set_marginal_node_time_weight(double val) -> void { marginal_node_time_weight_ = val; }
  // Half width of the uniform proposal of the log of the root's depth below its oldest child.
  auto marginal_node_time_window() const -> double { return marginal_node_time_window_; }
  auto set_marginal_node_time_window(double val) -> void { marginal_node_time_window_ = val; }

  // Age calibrations whose node can move inside this part.  Calibrated nodes that are cut points are
  // frozen for the whole local batch and are deliberately not listed here; `Run` keeps their constant
  // contribution.  See doc/extended/calibration.md, section 6.
  auto calibrations() const -> const std::vector<Local_calibration>& { return calibrations_; }
  // The calibration term depends only on node times, so it is kept up to date eagerly here and
  // incrementally inside the moves; it never needs the rest of the derived quantities.
  auto set_calibrations(std::vector<Local_calibration> calibrations) -> void {
    calibrations_ = std::move(calibrations);
    log_calibration_prior_ = calc_cur_log_calibration_prior();
  }

  // Set on the root part of a threshold partition, and only there; see `Threshold_partition_constraint`.
  auto set_threshold_partition_constraint(std::optional<Threshold_partition_constraint> constraint) -> void;
  auto has_threshold_partition_constraint() const -> bool { return threshold_constraint_.has_value(); }
  // Whether the current tree satisfies the constraint everywhere (true without one).  Checks every node.
  auto satisfies_threshold_partition_constraint() const -> bool;

  // Move diagnostics
  auto move_counts() const -> const std::array<Move_counts, k_num_move_kinds>& { return move_counts_; }
  auto reset_move_counts() -> void { move_counts_ = {}; }

  // Efficiency diagnostics of the history Gibbs updates: an acceptance rate says nothing about a
  // Gibbs step, so what is counted instead is how much history actually moved.
  auto history_gibbs_stats() const -> const History_gibbs_stats& { return history_gibbs_stats_; }
  auto reset_history_gibbs_stats() -> void { history_gibbs_stats_ = {}; }

  // Derived quantities
  auto log_G() const -> double {
    return validate_derived_quantities(), log_G_; }
  auto state_frequencies_of_ref_sequence_per_partition() const -> const Partition_vector<Seq_vector<int>>& {
    return validate_derived_quantities(), state_frequencies_of_ref_sequence_per_partition_; }
  auto lambda_i() const -> const Node_vector<double>& {
    return validate_derived_quantities(), lambda_i_; }
  auto ref_cum_Q_l() const -> const std::vector<double>& {
    return validate_derived_quantities(), *ref_cum_Q_l_;
  }
  auto num_sites_missing_at_every_node() const -> const Node_vector<int>& {
    return validate_derived_quantities(), num_sites_missing_at_every_node_; }
  auto log_augmented_coalescent_prior() const -> double {
    return validate_derived_quantities(), log_augmented_coalescent_prior_; }
  auto log_calibration_prior() const -> double { return log_calibration_prior_; }
  
  auto calc_cur_state_frequencies_of_ref_sequence_per_partition() const -> Partition_vector<Seq_vector<int>>;
  auto calc_cur_ref_cum_Q_l() const -> std::vector<double>;
  auto calc_cur_lambda_i(const std::vector<double>& ref_cum_Q_l) const -> Node_vector<double>;
  auto calc_cur_log_G(
      const Node_vector<double>& lambda_i,
      const Partition_vector<Seq_vector<int>>& state_frequencies_of_ref_sequence_per_partition) const
      -> double;
  auto calc_cur_num_sites_missing_at_every_node() const -> Node_vector<int>;
  auto calc_cur_log_augmented_coalescent_prior() const -> double;
  auto calc_cur_log_calibration_prior() const -> double;
  
  auto invalidate_derived_quantities() -> void { derived_quantities_valid_ = false; }
  auto validate_derived_quantities() const -> void {
    if (not derived_quantities_valid_) {
      recalc_derived_quantities();
      derived_quantities_valid_ = true;
    }
  }

  // Debug support
  auto set_original_sequences(
      Node_vector<Sequence_overlay> original_sequences,
      Node_vector<Interval_set<>> original_missing_sites_all)
      -> void {
    CHECK(estd::is_debug_enabled) << "We only track tip sequences in debug builds";
    original_sequences_ = std::move(original_sequences);
    original_missing_sites_all_ = std::move(original_missing_sites_all);
  }
  auto assert_cur_tip_sequences_compatible_with_original_ones() -> void;
  
 private:
  absl::BitGenRef bitgen_;
  Phylo_tree tree_;
  bool includes_run_root_;
  double t_max_tip_ = std::numeric_limits<double>::quiet_NaN();

  // Parameters
  std::shared_ptr<const Global_evo_model> evo_;
  std::shared_ptr<const std::vector<double>> given_ref_cum_Q_l_;
  std::optional<Partition_vector<Seq_vector<int>>> given_state_frequencies_;
  Very_scalable_coalescent_prior_part* coalescent_prior_part_;

  // Move config
  bool only_displacing_inner_nodes_ = false;
  bool topology_moves_enabled_ = true;
  bool skip_fixed_tip_moves_ = false;
  double root_displace_scale_factor_ = 1.0;
  double history_gibbs_weight_ = 0.0;
  int history_gibbs_sites_per_move_ = 1;
  double marginal_node_time_weight_ = 0.0;
  double marginal_node_time_window_ = 0.5;
  History_gibbs_stats history_gibbs_stats_{};
  std::optional<Site_history_sampler> history_sampler_;
  mutable int has_movable_tip_ = -1;  // -1 unknown, 0 no, 1 yes; tip bounds never change in a Subrun
  std::array<Move_counts, k_num_move_kinds> move_counts_{};
  std::vector<Local_calibration> calibrations_;
  std::optional<Threshold_partition_constraint> threshold_constraint_;
  Node_vector<int> clade_tips_;  // With a threshold constraint: tips of the whole tree below every node
  bool move_accepted_ = false;
  bool move_blocked_ = false;
  bool move_blocked_by_partition_ = false;

  // Derived quantities
  mutable bool derived_quantities_valid_{false};
  mutable double log_G_{std::numeric_limits<double>::quiet_NaN()};
  mutable Partition_vector<Seq_vector<int>> state_frequencies_of_ref_sequence_per_partition_{};
  mutable Node_vector<double> lambda_i_{};
  mutable std::shared_ptr<const std::vector<double>> ref_cum_Q_l_{};
  mutable Node_vector<int> num_sites_missing_at_every_node_{};
  mutable double log_augmented_coalescent_prior_{std::numeric_limits<double>::quiet_NaN()};
  mutable double log_calibration_prior_{0.0};
  
  auto recalc_derived_quantities() const -> void;
  auto check_derived_quantities() const -> void;

  // Debug support
  Node_vector<Sequence_overlay> original_sequences_;  // Only nonempty in debug builds!
  Node_vector<Interval_set<>> original_missing_sites_all_; // Only nonempty in debug builds!

  // Calibration support
  // Sum of the calibration log densities at `node` for the given node time; 0 if `node` is not calibrated.
  auto calc_log_calibration_prior_at(Node_index node, double t) const -> double;
  // log density difference of moving `node` from `t_old` to `t_new`; -infinity if `t_new` is outside a
  // hard bound.  The current state is always inside every support, so this is never inf - inf.
  auto calc_delta_log_calibration_prior(Node_index node, double t_old, double t_new) const -> double;
  // Intersection of the hard supports of the calibrations at `node`, as node times in days.
  auto calibration_t_bounds(Node_index node) const -> std::pair<double, double>;
  // Whether calibration targets must be re-identified after a topology proposal.
  auto calibration_targets_follow_topology() const -> bool {
    return topology_moves_enabled_ && not calibrations_.empty(); }
  // The MRCA of every calibration's sample set in the current tree, parallel to `calibrations_`.  Returns false
  // if some sample set is not monophyletic.
  auto find_calibration_targets(std::vector<Node_index>& nodes) const -> bool;
  // Same, storing the targets in `calibrations_`; leaves them unchanged and returns false if a set is not monophyletic.
  auto rebind_calibrations() -> bool;
  // Full calibration log density of the current tree with freshly identified targets; -infinity if a sample set
  // is not monophyletic or a target lies outside its hard support.  Updates the bindings as a side effect.
  auto calc_rebound_log_calibration_prior() -> double;
  auto save_calibration_nodes() const -> std::vector<Node_index>;
  auto restore_calibration_nodes(const std::vector<Node_index>& nodes) -> void;

  // Threshold partition support.  After the subtree above `X` has been regrafted, with `P` its (moved) parent and
  // `S` the node P was taken away from (or put back above), `refresh_clade_tips_after_regraft` brings `clade_tips_` up
  // to date and `threshold_partition_holds_after_regraft` says whether the constraint still holds.  Both do nothing,
  // and the latter returns true, without a constraint.
  auto refresh_clade_tips_after_regraft(Node_index P, Node_index S) -> void;
  auto threshold_partition_holds_after_regraft(Node_index P, Node_index S) const -> bool;
  auto threshold_partition_holds_at(Node_index node) const -> bool;

  // Move support
  auto pick_random_node() -> Node_index;
  auto pick_random_inner_node() -> Node_index;
  auto pick_random_tip() -> Node_index;

  // Moves
  auto inner_node_displace_move() -> void;
  auto has_movable_tip() const -> bool;
  auto tip_displace_move() -> void;
  auto branch_reform_move() -> void;
  auto subtree_slide_move() -> void;
  auto branch_history_gibbs_move() -> void;
  auto node_state_gibbs_move() -> void;
  auto node_time_marginal_move() -> void;
  // The per-partition bridge samplers for the current `evo_`, built on first use.
  auto history_sampler() -> Site_history_sampler&;
  auto wilson_balding_move() -> void;

  auto spr1_move() -> void;

  auto spr_move_core(
      Node_index X,
      Phylo_tree_loc new_nexus,
      double alpha_new_to_old_over_alpha_old_to_new)
      -> void;
};

}  // namespace delphy

#endif // DELPHY_SUBRUN_H_
