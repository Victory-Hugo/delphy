#ifndef DELPHY_RUN_H_
#define DELPHY_RUN_H_

#include <array>
#include <memory>
#include <random>
#include <stdexcept>
#include <typeinfo>

#include "ctpl_stl.h"

#include "calibration.h"
#include "coalescent_grid.h"
#include "exact_coalescent.h"
#include "evo_hky.h"
#include "evo_subst_model.h"
#include "phylo_tree.h"
#include "scalable_coalescent.h"
#include "site_partition.h"
#include "tree_partitioning.h"
#include "subrun.h"
#include "very_scalable_coalescent.h"
#include "sequence_overlay.h"

namespace delphy {

// Population models the exact Kingman correction covers.  The correction's mathematics needs only an
// exact `pop_at_time` and `intensity_integral`, which every model has; what it also needs is a
// population move that targets the exact density rather than the cell prior, and those exist for the
// exponential model (`pop_size_move`, `pop_growth_rate_move`) and for the Skygrid model
// (`skygrid_tau_move`, `skygrid_gammas_zero_mode_gibbs_move`, `skygrid_gammas_hmc_move`).  A constant
// population has no free parameter, so it is covered too.
auto pop_model_supports_exact_correction(const Pop_model& pop_model) -> bool;

class Run {
 public:
  // `init_coal_prior_t_step` is the cell width, in days, of the coalescent prior's first grid.  The
  // default of one day matches the resolution of pathogen tip dates and is what every existing
  // caller gets; it costs one cell per day of tree depth, which is tens of millions of cells on a
  // tree tens of thousands of years deep, allocated before the first MCMC step.  Deep-time callers
  // pass a width derived from the tree instead.
  Run(ctpl::thread_pool& thread_pool, std::mt19937 bitgen, uint32_t prng_seed, Phylo_tree tree,
      double init_coal_prior_t_step = 1.0);

  auto bitgen() -> absl::BitGenRef { return bitgen_; }
  // Non-const access may change the reference sequence, so it retires the cached statistics of the reference (see run.h)
  auto tree() -> Phylo_tree& { ++ref_version_; return tree_; }
  auto tree() const -> const Phylo_tree& { return tree_; }
  auto partition() const -> const Partition& { return tree_partition_; }

  auto step() const -> int64_t { return step_; }
  auto set_step(int64_t num_steps) -> void;
  auto local_moves_per_global_move() const -> int { return local_moves_per_global_move_; }
  auto set_local_moves_per_global_move(int local_moves_per_global_move) -> void;
  auto prng_seed() const -> uint32_t { return prng_seed_; }
  auto num_parts() const -> int { return num_parts_; }
  auto set_num_parts(int num_parts) -> void { num_parts_ = num_parts; partition_stencils_valid_ = false; }
  auto target_coal_prior_cells() const -> int { return target_coal_prior_cells_; }
  auto set_target_coal_prior_cells(int target_coal_prior_cells) -> void {
    target_coal_prior_cells_ = target_coal_prior_cells; partition_stencils_valid_ = false;
  }

  // Largest cell count the grid may be refined to when a tolerance is in force.  Ignored when
  // `coal_prior_grid_tolerance()` is zero.
  auto coal_prior_grid_max_cells() const -> int { return coal_prior_grid_max_cells_; }
  auto set_coal_prior_grid_max_cells(int max_cells) -> void { coal_prior_grid_max_cells_ = max_cells; }

  // When positive, the coalescent grid is retargeted to the coarsest cell width whose
  // discretization excess (see `coalescent_grid.h`) is at most this many nats, instead of simply
  // dividing the tree span into `target_coal_prior_cells()` cells.  Zero disables the tolerance and
  // restores the span/cells rule, which is the default everywhere.
  // Diagnostic: write one line to stderr at every change of the coalescent grid's cell width, giving the
  // change in the coalescent log prior that the change of grid imposes on the current state.  That change
  // is not part of any Metropolis-Hastings ratio, so its accumulated size measures how much the sampler
  // is moved by the grid itself rather than by its moves.  Off by default.
  auto coal_prior_trace_retargets() const -> bool { return coal_prior_trace_retargets_; }
  auto set_coal_prior_trace_retargets(bool trace) -> void { coal_prior_trace_retargets_ = trace; }

  // A positive fixed cell width, in days, pins the coalescent prior's grid for the whole run: the
  // grid is never retargeted from the sampled state, so the target density is one fixed function of
  // the genealogy and the population model.  Zero (the default) keeps the retargeting rules.
  // Opt-in efficiency settings of the local moves; the defaults reproduce the historical behaviour.
  auto skip_fixed_tip_moves() const -> bool { return skip_fixed_tip_moves_; }
  auto set_skip_fixed_tip_moves(bool val) -> void { skip_fixed_tip_moves_ = val; }
  // Number of (population size, growth rate) move pairs made in every global step of an exponential
  // population model; the historical value is 50.
  auto pop_moves_per_global_move() const -> int { return pop_moves_per_global_move_; }
  auto set_pop_moves_per_global_move(int val) -> void { pop_moves_per_global_move_ = val; }
  auto root_displace_scale_factor() const -> double { return root_displace_scale_factor_; }
  auto set_root_displace_scale_factor(double val) -> void { root_displace_scale_factor_ = val; }

  auto coal_prior_fixed_t_step() const -> double { return coal_prior_fixed_t_step_; }
  auto set_coal_prior_fixed_t_step(double t_step) -> void {
    if (exact_coalescent_correction_ && not (t_step > 0.0)) {
      throw std::invalid_argument(
          "The exact coalescent correction requires a pinned coalescent prior grid; "
          "disable the correction before releasing the grid");
    }
    coal_prior_fixed_t_step_ = t_step;
    if (t_step > 0.0) { set_coalescent_prior_t_step(t_step); }
  }

  // Exact Kingman coalescent prior through a whole-batch Metropolis-Hastings correction (experimental; off by default).
  // The partitioned sampler targets the cell prior P_cell on the coalescent grid.  With the correction on, every batch of
  // local moves between two global steps is accepted with probability min(1, exp(delta' - delta)), where
  // delta = log P_exact - log P_cell of the genealogy before (delta) and after (delta') the batch, and rolled back
  // otherwise; the population moves use the exact density.  The target is then the exact Kingman prior.
  // Supported configurations (anything else is refused with std::invalid_argument): fixed topology, tips with exact
  // dates (which may differ between tips), the exponential, Skygrid or constant population model, pinned grid; age
  // calibrations on the MRCAs of sample sets (root and non-root), with any number of partitions.
  // With a Skygrid curve the three population moves -- the smoothness Gibbs move, the zero-mode Gibbs move and the
  // HMC move on the knot values -- all target the exact density, which is evaluated on the genealogy's own event
  // intervals (`exact_coalescent.h`) rather than on the grid, so their cost does not grow with the cell count.
  // See doc/extended/exact_coalescent.md and
  // exact_skygrid.md.
  // Threshold partitioning (experimental; off by default).  When on and `num_parts()` > 1, every repartition draws an
  // auxiliary variable independently of the state: with probability `single_part_probability` the next batches use a
  // single part, and otherwise an integer m uniform on [min_fraction, max_fraction] x tips / num_parts (at least 2),
  // which cuts the tree into its threshold partition with bounds [ceil(m / min_ratio), m]
  // (`threshold_partition_stencil`).  The root part is constrained to keep that partition, and the parts below it
  // keep it whatever their moves, so the batch kernel uses the same partition at the start and at the end of every
  // batch.  This makes the exact coalescent correction valid with topology moves and several parts.  Local moves are
  // shared out in proportion to part sizes, and every part runs on the thread pool.
  // See doc/extended/threshold_partition.md.
  auto threshold_partitioning() const -> bool { return threshold_partitioning_; }
  auto set_threshold_partitioning(bool enabled) -> void;
  auto threshold_partition_min_fraction() const -> double { return threshold_partition_min_fraction_; }
  auto threshold_partition_max_fraction() const -> double { return threshold_partition_max_fraction_; }
  auto threshold_partition_min_ratio() const -> int { return threshold_partition_min_ratio_; }
  auto threshold_partition_single_part_probability() const -> double {
    return threshold_partition_single_part_probability_; }
  auto set_threshold_partition_params(
      double min_fraction, double max_fraction, int min_ratio, double single_part_probability) -> void;
  // Diagnostics (not serialized): partitions drawn, of which single-part draws, and the parts they had in total
  auto threshold_partitions_drawn() const -> int64_t { return threshold_partitions_drawn_; }
  auto threshold_partitions_single_part() const -> int64_t { return threshold_partitions_single_part_; }
  auto threshold_partition_parts_drawn() const -> int64_t { return threshold_partition_parts_drawn_; }

  auto exact_coalescent_correction() const -> bool { return exact_coalescent_correction_; }
  auto set_exact_coalescent_correction(bool enabled) -> void;
  auto exact_coalescent_batches_proposed() const -> int64_t { return exact_batches_proposed_; }
  auto exact_coalescent_batches_accepted() const -> int64_t { return exact_batches_accepted_; }
  // The coalescent term of the target posterior: the cell prior, or the exact density when the correction is on.
  auto calc_cur_log_coalescent_target() const -> double;

  // State that a checkpoint needs, beyond the genealogy and the parameters, to continue a run exactly as the uninterrupted run would:
  // the text forms of the standard Mersenne Twister state of the run and of every part, and the pool of partition stencils with the
  // step at which it is renewed.  A restored run draws the same numbers as the original from the same point.  `restore_resume_state`
  // must be called after set_step, set_num_parts and set_local_moves_per_global_move, which discard the stencil pool.
  struct Resume_state {
    std::string prng;
    std::vector<std::string> sub_prngs;
    std::vector<std::vector<Node_index>> stencils;  // cut points of each stencil
    int64_t stencil_refresh_step = -1;              // -1: not saved
  };
  auto resume_state() const -> Resume_state;
  auto restore_resume_state(const Resume_state& state) -> void;

  // Joint time-scale move (experimental; off by default).  Scales every internal node age and every mutation age,
  // measured back from the common sampling date T, by a factor s = exp(u), u ~ Uniform(-window, +window), and
  // transforms the rate and the population model so that the rate x time products and the population size
  // as a function of age scale together: mu -> mu / s, n0 -> s * n0, g -> g / s (the exponential model's n0 is
  // the size at its reference time t0, which must be the sampling date).  The map is a deterministic bijection
  // of (node ages, mutation ages, mu, n0, g); the move is accepted with probability min(1, exp(log_alpha)),
  //   log_alpha = log pi(new) - log pi(old) + (n_internal + n_mutations - 1) * u,
  // where pi is the full target (sequence likelihood, exact Kingman prior, calibration priors, priors on mu, n0
  // and g) and the last term is the log Jacobian of the map.  It runs in the global step, between batches of
  // local moves, and touches the coalescent only through the exact density, so it does not interact with the
  // whole-batch correction.  Requires the exact correction, a fixed topology, one sampling date and the
  // exponential population model with t0 at that date; anything else is refused with std::invalid_argument.
  // See doc/extended/global_scale_move.md.
  struct Global_scale_move_result {
    bool accepted = false;
    double log_scale = 0.0;
    double log_alpha = 0.0;            // log acceptance ratio including the Jacobian; -inf if rejected on support
    double delta_log_G = 0.0;
    double delta_log_coalescent = 0.0;  // exact Kingman density
    double delta_log_other_priors = 0.0;  // priors on mu, n0, g (and the unchanged rest)
    double delta_log_calibration = 0.0;
    double log_jacobian = 0.0;
    // (delta_log_G + delta_log_coalescent + log_jacobian) + log_scale; zero up to rounding when the population floor is off (the three terms
    // then add to -log s analytically).  NaN when min_pop > 0, where the closed form does not apply.
    double scale_free_residual = std::numeric_limits<double>::quiet_NaN();
  };
  auto global_scale_move_enabled() const -> bool { return global_scale_move_enabled_; }
  auto set_global_scale_move_enabled(bool enabled) -> void;
  auto global_scale_move_window() const -> double { return global_scale_move_window_; }
  auto set_global_scale_move_window(double window) -> void;
  // Number of proposals made in each global step
  auto global_scale_moves_per_global_move() const -> int { return global_scale_moves_per_global_move_; }
  auto set_global_scale_moves_per_global_move(int count) -> void;
  // Acceptance counts of the exponential population moves; a growth-rate proposal outside the allowed
  // range of g counts as proposed and as out of bounds
  auto pop_size_moves_proposed() const -> int64_t { return pop_size_proposed_; }
  auto pop_size_moves_accepted() const -> int64_t { return pop_size_accepted_; }
  auto pop_growth_rate_moves_proposed() const -> int64_t { return pop_growth_proposed_; }
  auto pop_growth_rate_moves_accepted() const -> int64_t { return pop_growth_accepted_; }
  auto pop_growth_rate_moves_out_of_bounds() const -> int64_t { return pop_growth_out_of_bounds_; }
  auto global_scale_moves_proposed() const -> int64_t { return global_scale_proposed_; }
  auto global_scale_moves_accepted() const -> int64_t { return global_scale_accepted_; }
  // Sum of |u| over accepted proposals: the distance travelled along the joint scale coordinate
  auto global_scale_abs_log_scale_accepted() const -> double { return global_scale_abs_u_accepted_; }
  // Largest |scale_free_residual| over all proposals so far (a rounding diagnostic; only proposals with min_pop = 0 count)
  auto global_scale_max_abs_residual() const -> double { return global_scale_max_abs_residual_; }
  // One proposal with a given log scale and a given uniform draw for the accept test (for tests); the
  // scheduled move draws both from the run's generator.
  auto global_scale_move_with(double log_scale, double uniform_draw) -> Global_scale_move_result;

  auto coal_prior_grid_tolerance() const -> double { return coal_prior_grid_tolerance_; }
  auto set_coal_prior_grid_tolerance(double tolerance) -> void { coal_prior_grid_tolerance_ = tolerance; }

  // Measured discretization excess of the current tree on the current grid, in nats.  This is the
  // amount by which the coalescent prior that the sampler actually uses exceeds the exact Kingman
  // density of the same genealogy; it is a diagnostic, not a correction.
  auto calc_coal_prior_grid_excess() const -> Coalescent_grid_excess;

  // Set the grid to the width the current tolerance (or, without one, `target_coal_prior_cells()`)
  // asks for, and return what was measured.  Safe to call before the first MCMC step: it is how a
  // deep-time run avoids ever materializing a grid at the default one-day resolution.
  auto retarget_coal_prior_grid() -> Coalescent_grid_excess;

  auto do_mcmc_steps(int steps) -> void;

  auto tree_modified() -> void { partition_stencils_valid_ = false; invalidate_derived_quantities(); }
  auto coalescent_prior() const -> const Scalable_coalescent_prior& { return coalescent_prior_; }
  auto coalescent_prior_t_step() const -> double { return coalescent_prior_.t_step(); }
  auto set_coalescent_prior_t_step(double t_step) -> void;
  auto mu() const -> double { return subst_model_.mu; }
  auto set_mu(double mu) -> void { subst_model_.mu = mpox_mu_ = mu, invalidate_derived_quantities(); }
  auto mu_prior_alpha() const -> double { return mu_prior_alpha_; }
  auto set_mu_prior_alpha(double mu_prior_alpha) -> void { mu_prior_alpha_ = mu_prior_alpha, invalidate_derived_quantities(); }
  auto mu_prior_beta() const -> double { return mu_prior_beta_; }
  auto set_mu_prior_beta(double mu_prior_beta) -> void { mu_prior_beta_ = mu_prior_beta, invalidate_derived_quantities(); }
  auto alpha() const -> double { return alpha_; }
  auto set_alpha(double alpha) -> void { alpha_ = alpha, invalidate_derived_quantities(); }
  auto nu() const -> const std::vector<double>& { return nu_; }
  auto set_nu(const std::vector<double>& nu) -> void { nu_ = nu, ++nu_version_, invalidate_derived_quantities(); }
  // The substitution model: one reversible CTMC shared by every site partition.  The default is
  // HKY85 with estimated frequencies, which is the model Delphy has always run.
  auto subst_model() const -> const Subst_model& { return subst_model_; }
  // Replaces the model, keeping mu.  Throws std::invalid_argument if the model is not valid.
  auto set_subst_model(Subst_model model) -> void;
  auto is_hky() const -> bool {
    return subst_model_.grouping == Exchangeability_grouping::from_code("010010"); }

  // kappa of the historical HKY parameterization: the relative rate of the transition class.  Only
  // meaningful while the grouping is 010010.
  auto hky_kappa() const -> double;
  auto set_hky_kappa(double hky_kappa) -> void;
  auto hky_pi() const -> Seq_vector<double> { return subst_model_.pi_a; }
  auto set_hky_pi(const Seq_vector<double>& hky_pi) -> void {
    subst_model_.pi_a = hky_pi, invalidate_derived_quantities(); }

  // Log-normal prior on every free relative exchangeability s_k / s_GT.  The mean of the log is
  // `subst_exchangeability_prior_log_mean_transition()` for a class whose every member is a
  // transition (A-G or C-T) and `..._other()` otherwise, and the standard deviation of the log is
  // `subst_exchangeability_prior_sd()`.  The defaults 1, 0 and 1.25 reproduce the historical prior on
  // kappa exactly for HKY85 and K80, whose single free class is the transition class.
  auto subst_exchangeability_prior_log_mean_transition() const -> double {
    return subst_exchangeability_prior_log_mean_transition_; }
  auto subst_exchangeability_prior_log_mean_other() const -> double {
    return subst_exchangeability_prior_log_mean_other_; }
  auto subst_exchangeability_prior_sd() const -> double { return subst_exchangeability_prior_sd_; }
  auto set_subst_exchangeability_prior(double log_mean_transition, double log_mean_other, double sd)
      -> void;
  // The mean of the log of the prior on the free class `k` of the current grouping.
  auto subst_exchangeability_prior_log_mean_of(int k) const -> double;
  auto pop_model() const -> const Pop_model& { return *pop_model_; }
  auto set_pop_model(std::shared_ptr<const Pop_model> pop_model) -> void {
    if (exact_coalescent_correction_ && not pop_model_supports_exact_correction(*pop_model)) {
      throw std::invalid_argument(
          "The exact coalescent correction supports only the exponential and Skygrid population models");
    }
    pop_model_ = std::move(pop_model);
    coalescent_prior_.pop_model_changed(pop_model_);
    invalidate_derived_quantities(); }
  auto pop_inv_n0_prior_alpha() const -> double { return pop_inv_n0_prior_alpha_; }
  auto set_pop_inv_n0_prior_alpha(double pop_inv_n0_prior_alpha) -> void {
    pop_inv_n0_prior_alpha_ = pop_inv_n0_prior_alpha, invalidate_derived_quantities(); }
  auto pop_inv_n0_prior_beta() const -> double { return pop_inv_n0_prior_beta_; }
  auto set_pop_inv_n0_prior_beta(double pop_inv_n0_prior_beta) -> void {
    pop_inv_n0_prior_beta_ = pop_inv_n0_prior_beta, invalidate_derived_quantities(); }
  auto pop_g_prior_mu() const -> double { return pop_g_prior_mu_; }
  auto set_pop_g_prior_mu(double pop_g_prior_mu) -> void {
    pop_g_prior_mu_ = pop_g_prior_mu, invalidate_derived_quantities(); }
  auto pop_g_prior_scale() const -> double { return pop_g_prior_scale_; }
  auto set_pop_g_prior_scale(double pop_g_prior_scale) -> void {
    pop_g_prior_scale_ = pop_g_prior_scale, invalidate_derived_quantities(); }
  // Half-width, in e-foldings per day, of the uniform random-walk proposal for the growth rate.  The
  // historical value of 1 per year suits pathogen growth rates of tens per year; a human-scale
  // growth rate lives within 1e-4 per year, where that window is accepted about once in 1e4 tries.
  auto pop_g_move_window() const -> double { return pop_g_move_window_; }
  auto set_pop_g_move_window(double window) -> void { pop_g_move_window_ = window; }
  auto pop_g_min() const -> double { return pop_g_min_; }
  auto set_pop_g_min(double pop_g_min) -> void {
    pop_g_min_ = pop_g_min, invalidate_derived_quantities(); }
  auto pop_g_max() const -> double { return pop_g_max_; }
  auto set_pop_g_max(double pop_g_max) -> void {
    pop_g_max_ = pop_g_max, invalidate_derived_quantities(); }
  auto skygrid_tau() const -> double { return skygrid_tau_; }
  auto set_skygrid_tau(double skygrid_tau) -> void { skygrid_tau_ = skygrid_tau, invalidate_derived_quantities(); }
  auto skygrid_tau_prior_alpha() const -> double { return skygrid_tau_prior_alpha_; }
  auto set_skygrid_tau_prior_alpha(double skygrid_tau_prior_alpha) -> void { skygrid_tau_prior_alpha_ = skygrid_tau_prior_alpha, invalidate_derived_quantities(); }
  auto skygrid_tau_prior_beta() const -> double { return skygrid_tau_prior_beta_; }
  auto set_skygrid_tau_prior_beta(double skygrid_tau_prior_beta) -> void { skygrid_tau_prior_beta_ = skygrid_tau_prior_beta, invalidate_derived_quantities(); }
  auto skygrid_low_gamma_barrier_loc() const -> double { return skygrid_low_gamma_barrier_loc_; }
  auto set_skygrid_low_gamma_barrier_loc(double skygrid_low_gamma_barrier_loc) -> void { skygrid_low_gamma_barrier_loc_ = skygrid_low_gamma_barrier_loc, invalidate_derived_quantities(); }
  auto skygrid_low_gamma_barrier_scale() const -> double { return skygrid_low_gamma_barrier_scale_; }
  auto set_skygrid_low_gamma_barrier_scale(double skygrid_low_gamma_barrier_scale) -> void { skygrid_low_gamma_barrier_scale_ = skygrid_low_gamma_barrier_scale, invalidate_derived_quantities(); }
  auto skygrid_inv_nbar_prior_alpha() const -> double { return skygrid_inv_nbar_prior_alpha_; }
  auto set_skygrid_inv_nbar_prior_alpha(double skygrid_inv_nbar_prior_alpha) -> void {
    skygrid_inv_nbar_prior_alpha_ = skygrid_inv_nbar_prior_alpha, invalidate_derived_quantities(); }
  auto skygrid_inv_nbar_prior_beta() const -> double { return skygrid_inv_nbar_prior_beta_; }
  auto set_skygrid_inv_nbar_prior_beta(double skygrid_inv_nbar_prior_beta) -> void {
    skygrid_inv_nbar_prior_beta_ = skygrid_inv_nbar_prior_beta, invalidate_derived_quantities(); }

  auto t_max_tip() const -> double {
    return validate_derived_quantities(), t_max_tip_; }
  auto evo() const -> const Global_evo_model& {
    return validate_derived_quantities(), evo_; }
  auto pi_a() const -> const Seq_vector<double>& {
    return validate_derived_quantities(), evo_.partition_evo_model[0].pi_a; }
  auto q_ab() const -> const Seq_matrix<double>& {
    return validate_derived_quantities(), evo_.partition_evo_model[0].q_ab; }
  auto log_posterior() const -> double {
    return validate_derived_quantities(),
        log_G_ + log_coalescent_prior_ + log_coal_exact_correction_ + log_other_priors_ + log_calibration_prior_; }
  auto log_G() const -> double {
    return validate_derived_quantities(), log_G_; }
  auto Ttwiddle_beta_a() const -> const Partition_vector<Seq_vector<double>>& {
    return validate_derived_quantities(), Ttwiddle_beta_a_; }
  auto num_muts() const -> int {
    return validate_derived_quantities(), num_muts_; }
  auto num_muts_ab() const -> const Seq_matrix<int>& {
    return validate_derived_quantities(), num_muts_ab_; }
  auto state_frequencies_of_ref_sequence() const -> const Seq_vector<int>& {
    return validate_derived_quantities(), state_frequencies_of_ref_sequence_; }
  auto log_coalescent_prior() const -> double {
    return validate_derived_quantities(), log_coalescent_prior_ + log_coal_exact_correction_; }
  auto log_other_priors() const -> double {
    return validate_derived_quantities(), log_other_priors_; }

  // Age calibrations on the MRCAs of named sample sets.  Empty by default, so nothing changes for
  // existing workflows.  See doc/extended/calibration.md.
  //
  // Calibrations are defined by taxon sets; `set_calibrations` resolves each one to the node index of
  // the MRCA of its taxon set in the current tree and throws Calibration_error if the configuration
  // cannot be satisfied.  With topology moves enabled, every calibrated sample set is constrained to stay
  // monophyletic, its target is re-identified after every topology proposal, and only a single tree
  // partition part is supported (`repartition` throws otherwise).  See
  // doc/extended/calibration_free_topology.md.
  auto set_calibrations(Calibration_set calibrations) -> void;
  auto calibrations() const -> const Calibration_set& { return calibrations_; }
  auto has_calibrations() const -> bool { return not calibrations_.empty(); }
  // Parallel to calibrations().calibrations: the node index of each calibration's target.
  auto calibration_nodes() const -> const std::vector<Node_index>& { return calibration_nodes_; }
  auto log_calibration_prior() const -> double {
    return validate_derived_quantities(), log_calibration_prior_; }
  // The same quantity as maintained incrementally by the partition parts, plus the contribution of the
  // calibrated nodes that the current partitioning freezes.  It equals log_calibration_prior() exactly
  // when the incremental bookkeeping inside the moves is correct, so tests compare the two after a
  // batch of local moves.  Only meaningful between a batch of local moves and the next repartitioning.
  auto log_calibration_prior_from_parts() const -> double;

  // Site partitions: named, non-overlapping sets of sites that share the timed tree, the HKY parameters
  // and the global clock rate, but have their own relative substitution rate (see site_partition.h and
  // doc/extended/site_partitions.md).  Every run without a configuration has exactly one
  // partition covering every site, which reproduces the historical behaviour.
  //
  // `set_site_partition` resolves the configuration against the current tree, which fixes the weights
  // L_p (sites observed in at least one sample) for the whole run, and starts every relative rate at 1.
  // It throws std::invalid_argument for configurations this sampler does not support (the mpox hack owns
  // the partitioning when it is on) and Site_partition_error for configurations the data reject.
  auto set_site_partition(Site_partition_config config) -> void;
  auto site_partition_config() const -> const Site_partition_config& { return site_partition_config_; }
  auto site_partition() const -> const Site_partition& { return site_partition_; }
  auto has_site_partitions() const -> bool { return site_partition_.num_partitions() > 1; }
  // Normalized relative rates, parallel to site_partition().names: sum_p L_p r_p / sum_p L_p == 1.
  auto site_partition_rates() const -> const std::vector<double>& { return site_partition_rates_; }
  auto set_site_partition_rates(std::vector<double> rates) -> void;
  auto site_partition_rate_move_enabled() const -> bool { return site_partition_rate_move_enabled_; }
  auto set_site_partition_rate_move_enabled(bool enabled) -> void {
    site_partition_rate_move_enabled_ = enabled; }
  // Concentration of the symmetric Dirichlet prior on the length-weighted rate shares
  // x_p = L_p r_p / sum_j L_j; 1 (the default) is the uniform prior on the simplex.
  auto site_partition_rate_prior_concentration() const -> double {
    return site_partition_rate_prior_concentration_; }
  auto set_site_partition_rate_prior_concentration(double concentration) -> void;
  auto site_partition_rate_moves_proposed() const -> int64_t { return site_partition_rate_moves_proposed_; }
  auto site_partition_rate_moves_accepted() const -> int64_t { return site_partition_rate_moves_accepted_; }
  // One delta-exchange proposal on a pair of rate shares, exposed for the tests.  Returns the log
  // acceptance ratio of the proposal that was made (-infinity when the proposal left the simplex).
  struct Site_partition_rate_move_result {
    bool accepted = false;
    Partition_index p = -1;
    Partition_index q = -1;
    double delta_x = 0.0;
    double log_alpha = 0.0;
    double delta_log_G = 0.0;
    double delta_log_other_priors = 0.0;
  };
  auto site_partition_rate_move() -> Site_partition_rate_move_result;
  // The site rates nu_l are independent draws from Gamma(alpha, alpha) in the shape-rate
  // parameterization, so their prior mean is one and `mu` keeps the whole rate scale.  Conditional on
  // the mutational history the posterior of nu_l is Gamma(M_l + alpha, mu_l Ttwiddle_l + alpha), which
  // `gibbs_sample_all_nus` draws, and `calc_cur_log_p_alpha` is the density of alpha with every nu_l
  // integrated out, up to a constant: the target of the alpha move.  Exposed for the tests.
  auto gibbs_sample_all_nus() -> void;
  auto calc_cur_log_p_alpha(double alpha) const -> double;
  auto alpha_moves() -> void;
  // One scaling proposal on the free exchangeability class `free_class`, one delta exchange on the
  // frequencies, and one proposal per free class, exposed for the tests.
  auto subst_exchangeability_move(int free_class) -> void;
  auto subst_exchangeability_moves() -> void;
  auto subst_frequencies_move() -> void;
  // The difference in log_G between two rate matrices, from the maintained statistics alone.
  auto calc_delta_log_G_of_subst_model(
      const Site_evo_model& old_site_evo, const Site_evo_model& new_site_evo) const -> double;
  auto site_partition_rate_move_with(Partition_index p, Partition_index q, double delta_x, double accept_draw)
      -> Site_partition_rate_move_result;
  // Mutations per site partition, recomputed from the tree (as the mpox moves do with their own counts).
  auto calc_cur_num_muts_per_partition() const -> std::vector<int>;

  auto calc_cur_log_G() const -> double;
  auto calc_log_G_with(const Partition_vector<Seq_vector<int>>& state_frequencies_per_partition,
                       const std::vector<double>& ref_cum_Q_l) const -> double;
  static auto sum_over_partitions(const Partition_vector<Seq_vector<int>>& per_partition) -> Seq_vector<int>;
  auto calc_cur_Ttwiddle_beta_a() const -> Partition_vector<Seq_vector<double>>;
  auto calc_cur_log_coalescent_prior() const -> double;
  auto calc_cur_log_augmented_coalescent_prior() const -> double;
  auto calc_cur_num_muts() const -> int;
  auto calc_cur_num_muts_ab() const -> Seq_matrix<int>;
  auto calc_cur_state_frequencies_of_ref_sequence() const -> Seq_vector<int>;
  auto calc_cur_log_other_priors() const -> double;
  auto calc_cur_log_calibration_prior() const -> double;
  auto calc_cur_skygrid_gmrf_prior() const -> double;

  // Mpox hack.  TODO: We really need something much more generic for supporting different
  // forms of posteriors with different sets of parameters.
  //
  // When we turn on "mpox_hack", the evolution model `evo_` switches to 2 site partitions
  // with the following transition rate matrices:
  //
  //               A    C    G    T                           A    C    G    T
  //
  //           /  -1   1/3  1/3  1/3  \                   /   0    0    0    0  \  .
  //           |                      |                   |                     |  .
  //           |  1/3  -1   1/3  1/3  |                   |   0   -2    0    2  |  .
  //  Q_0 = mu |                      |,  Q_1 = Q_0 + mu* |                     |  .
  //           |  1/3  1/3  -1   1/3  |                   |   2    0   -2    0  |  .
  //           |                      |                   |                     |  .
  //           \  1/3  1/3  1/3  -1   /                   \   0    0    0    0  /  .
  //
  // The sites in partition 1 are those with "APOBEC context", i.e., C or T preceded by
  // a T and G or A followed by an A in the sequence of the first tip.  All other sites
  // are in partition 0.  The idea is to use a simple JC model (given how few mutations
  // we see, no point in trying something more sophisticated), and to approximate
  // the much faster (mu*) APOBEC-mediated mutation of 5'-TC-3' and 5'-GA-3' dimers to TT and AA.
  // To model this without introducing couplings between evolution at different sites,
  // we're assuming that mutations are so rare that we'll never see two mutations
  // in adjacent sites, so it's unlikely for a site to change between having or lacking
  // "APOBEC context".  Merging TC and GA into a single partition
  // keeps things simpler still; you could instead imagine 4 different partitions
  // according to whether a site is (1) preceded or not by a T and/or (2) followed or
  // not by an A.
  //
  // This idea is a slight adaptation of what Andrew Rambaut discussed on a Zoom call with us
  // on 23 May 2023, and which they have implemented in BEAST for their mpox analyses
  // (O'Toole et al, Science, Nov 2023).  We're targetting trees with tips only from after the
  // spillover into humans, so there's no attempt to identify a breakpoint along the tree
  // above which only Q_0 applies.
  //
  // The factors of 2 in Q_1 are chosen to match the convention used by O'Toole et al in
  // reporting APOBEC mutation rates (the effective rate at which APOBEC changes would
  // be observed if 50% of sites with APOBEC context were unmutated and 50% were mutated)
  //
  auto mpox_hack_enabled() const -> bool { return mpox_hack_enabled_; }
  auto set_mpox_hack_enabled(bool mpox_hack_enabled) -> void;
  auto mpox_mu() const -> double { return mpox_mu_; }
  auto set_mpox_mu(double mpox_mu) -> void { mpox_mu_ = mpox_mu, invalidate_derived_quantities(); }
  auto mpox_mu_star() const -> double { return mpox_mu_star_; }
  auto set_mpox_mu_star(double mpox_mu_star) -> void { mpox_mu_star_ = mpox_mu_star, invalidate_derived_quantities(); }
  
  auto invalidate_derived_quantities() -> void { derived_quantities_valid_ = false; ++ref_version_; }
  auto validate_derived_quantities() const -> void {
    if (not derived_quantities_valid_) {
      recalc_derived_quantities();
      derived_quantities_valid_ = true;
    }
  }

  // Selectively suppress some moves for demo purposes
  auto only_displacing_inner_nodes() const -> bool { return only_displacing_inner_nodes_; }
  auto set_only_displacing_inner_nodes(bool only_displacing_inner_nodes) -> void {
    only_displacing_inner_nodes_ = only_displacing_inner_nodes; }
  auto topology_moves_enabled() const -> bool { return topology_moves_enabled_; }
  auto set_topology_moves_enabled(bool topology_moves_enabled) -> void;
  // Cumulative local-move diagnostics since this Run object was created (not serialized)
  auto move_counts() const -> const std::array<Move_counts, k_num_move_kinds>& { return move_counts_; }

  // The single-site mutational-history Gibbs updates, for data divergent enough that the all-sites
  // history proposal has stopped being accepted.  All three are off by default and none of them
  // changes the default move mix, the default schedule or the random number stream.
  // See doc/extended/history_gibbs.md.
  //
  // Weight of the two local updates (C1 and C2, half each) in the local move mix.
  auto history_gibbs_weight() const -> double { return history_gibbs_weight_; }
  auto set_history_gibbs_weight(double weight) -> void;
  // Sites visited by one local update.
  auto history_gibbs_sites_per_move() const -> int { return history_gibbs_sites_per_move_; }
  auto set_history_gibbs_sites_per_move(int count) -> void;
  // Sites whose whole history is redrawn by Felsenstein pruning (C3) in every global step.
  auto site_history_refreshes_per_global_move() const -> int {
    return site_history_refreshes_per_global_move_; }
  auto set_site_history_refreshes_per_global_move(int count) -> void;
  // The locally marginalized node-time move of Phase F: it integrates the histories of the branches
  // incident on a node out, proposes the node time free of the mutation times that otherwise bound
  // it, and draws the histories back from their exact conditional.  Off by default.
  auto marginal_node_time_weight() const -> double { return marginal_node_time_weight_; }
  auto set_marginal_node_time_weight(double weight) -> void;
  auto marginal_node_time_window() const -> double { return marginal_node_time_window_; }
  auto set_marginal_node_time_window(double window) -> void;
  // Cumulative diagnostics of all three updates (not serialized)
  auto history_gibbs_stats() const -> const History_gibbs_stats& { return history_gibbs_stats_; }

  auto repartitioning_enabled() const -> bool { return repartitioning_enabled_; }
  auto set_repartitioning_enabled(bool repartitioning_enabled) -> void {
    repartitioning_enabled_ = repartitioning_enabled; }

  // Other moves and features are enabled selectively depending on config
  auto alpha_move_enabled() const -> bool { return alpha_move_enabled_; }
  auto set_alpha_move_enabled(bool enabled) -> void {
    alpha_move_enabled_ = enabled; }
  auto mu_move_enabled() const -> bool { return mu_move_enabled_; }
  auto set_mu_move_enabled(bool enabled) -> void {
    mu_move_enabled_ = enabled; }
  // Whether the exchangeabilities and the frequencies of the substitution model are sampled.  Both
  // default to on, which is what Delphy has always done with HKY85's kappa and frequencies; the
  // frequency move does nothing unless the frequency mode is `estimated`.
  auto subst_exchangeability_move_enabled() const -> bool {
    return subst_exchangeability_move_enabled_; }
  auto set_subst_exchangeability_move_enabled(bool enabled) -> void {
    subst_exchangeability_move_enabled_ = enabled; }
  auto subst_frequencies_move_enabled() const -> bool { return subst_frequencies_move_enabled_; }
  auto set_subst_frequencies_move_enabled(bool enabled) -> void {
    subst_frequencies_move_enabled_ = enabled; }
  auto final_pop_size_move_enabled() const -> bool { return final_pop_size_move_enabled_; }
  auto set_final_pop_size_move_enabled(bool enabled) -> void {
    final_pop_size_move_enabled_ = enabled; }
  auto pop_growth_rate_move_enabled() const -> bool { return pop_growth_rate_move_enabled_; }
  auto set_pop_growth_rate_move_enabled(bool enabled) -> void {
    pop_growth_rate_move_enabled_ = enabled; }
  auto skygrid_tau_move_enabled() const -> bool { return skygrid_tau_move_enabled_; }
  auto set_skygrid_tau_move_enabled(bool enabled) -> void {
    skygrid_tau_move_enabled_ = enabled; }
  auto skygrid_low_gamma_barrier_enabled() const -> bool { return skygrid_low_gamma_barrier_enabled_; }
  auto set_skygrid_low_gamma_barrier_enabled(bool enabled) -> void {
    skygrid_low_gamma_barrier_enabled_ = enabled;
  }

  // Diagnostic for validating the Skygrid HMC move.  When enabled, every trajectory step compares
  // its analytic forces against a central finite difference of the very potential the move
  // evaluates, and records the largest relative discrepancy seen since the last reset.  Off by
  // default; it draws no random numbers and leaves the state it inspects unchanged, so the chain is
  // identical with it on, but it costs O(M) extra posterior evaluations per step.
  auto skygrid_hmc_force_check_enabled() const -> bool { return skygrid_hmc_force_check_enabled_; }
  auto set_skygrid_hmc_force_check_enabled(bool enabled) -> void {
    skygrid_hmc_force_check_enabled_ = enabled; }
  auto skygrid_hmc_force_check_step() const -> double { return skygrid_hmc_force_check_step_; }
  auto set_skygrid_hmc_force_check_step(double d_gamma) -> void {
    skygrid_hmc_force_check_step_ = d_gamma; }
  // Largest |f_analytic - f_numeric| / max(1, |f_analytic|, |f_numeric|) over every knot and every
  // trajectory step since the last reset.  Zero if the check has never run.
  auto skygrid_hmc_max_force_discrepancy() const -> double { return skygrid_hmc_max_force_discrepancy_; }
  // The same discrepancy divided by what the comparison can resolve, which is the larger of 1e-4 of
  // the force's own scale and the round-off floor of the central difference itself,
  // `eps max(|U_plus|, |U_minus|) / d_gamma`.  A central difference of a potential of magnitude |U|
  // simply cannot resolve a derivative to better than that, and at deep-time spans |U| reaches 1e8,
  // where the floor is 1e-4 absolute and a fixed relative gate measures the floor rather than the
  // gradient.  A value below 1 means every check agreed to within its own resolution.
  auto skygrid_hmc_max_force_discrepancy_over_resolution() const -> double {
    return skygrid_hmc_max_force_over_resolution_; }
  auto reset_skygrid_hmc_force_check() -> void {
    skygrid_hmc_max_force_discrepancy_ = 0.0, skygrid_hmc_force_checks_ = 0,
    skygrid_hmc_max_force_over_resolution_ = 0.0,
    skygrid_hmc_worst_force_analytic_ = 0.0, skygrid_hmc_worst_force_numeric_ = 0.0,
    skygrid_hmc_worst_force_gamma_ = 0.0, skygrid_hmc_worst_force_min_gamma_ = 0.0; }
  auto skygrid_hmc_force_checks() const -> int64_t { return skygrid_hmc_force_checks_; }
  // The state at the worst discrepancy: the two force values, the knot value differentiated, and the
  // smallest knot value of the curve at that point (populations below exp(-745) underflow to zero).
  auto skygrid_hmc_worst_force_analytic() const -> double { return skygrid_hmc_worst_force_analytic_; }
  auto skygrid_hmc_worst_force_numeric() const -> double { return skygrid_hmc_worst_force_numeric_; }
  auto skygrid_hmc_worst_force_gamma() const -> double { return skygrid_hmc_worst_force_gamma_; }
  auto skygrid_hmc_worst_force_min_gamma() const -> double { return skygrid_hmc_worst_force_min_gamma_; }

  // When paranoid mode is on, force full tree integrity checks even in release builds.
  // Profiling showed these checks consume ~23% of runtime; they have not failed for many months
  // of intense use, so they are off by default in release builds (always on in debug builds).
  auto paranoid() const -> bool { return paranoid_; }
  auto set_paranoid(bool paranoid) -> void { paranoid_ = paranoid; }

 private:
  ctpl::thread_pool* thread_pool_;
  std::mt19937 bitgen_;
  uint32_t prng_seed_;
  Phylo_tree tree_;

  int64_t step_;
  int local_moves_per_global_move_;
  int64_t next_global_move_step_;
  
  Partition tree_partition_;
  int num_parts_;
  int target_coal_prior_cells_;
  int coal_prior_grid_max_cells_;
  double coal_prior_grid_tolerance_;
  bool coal_prior_trace_retargets_ = false;
  double coal_prior_fixed_t_step_ = 0.0;
  bool skip_fixed_tip_moves_ = false;
  int pop_moves_per_global_move_ = 50;
  double root_displace_scale_factor_ = 1.0;
  double coal_prior_grid_span_at_retarget_ = 0.0;
  std::vector<std::vector<Partition_part_info>> partition_stencils_;
  bool threshold_partitioning_ = false;
  double threshold_partition_min_fraction_ = 0.5;
  double threshold_partition_max_fraction_ = 1.0;
  int threshold_partition_min_ratio_ = 4;
  double threshold_partition_single_part_probability_ = 0.02;
  // Bounds of the threshold partition in use; 0 when the current draw is a single part without constraint
  int threshold_max_cut_tips_ = 0;
  int threshold_min_cut_tips_ = 0;
  int64_t threshold_partitions_drawn_ = 0;
  int64_t threshold_partitions_single_part_ = 0;
  int64_t threshold_partition_parts_drawn_ = 0;
  mutable bool partition_stencils_valid_ = false;
  mutable int64_t next_partition_stencil_refresh_step_;
  bool force_repartition_ = false;
  std::vector<std::mt19937> subbitgens_;
  std::vector<Subrun> subruns_;
  std::array<Move_counts, k_num_move_kinds> move_counts_{};

  // Calibrations, their target nodes in `tree_`, and the part of the calibration prior contributed by
  // calibrated nodes that no part can move during the current local batch (cut points).
  Calibration_set calibrations_;
  std::vector<Node_index> calibration_nodes_;
  double frozen_calibration_log_prior_ = 0.0;

  Real_sequence original_ref_sequence_;                    // Only nonempty in debug builds!
  Node_vector<Sequence_overlay> original_sequences_;       // Only nonempty in debug builds!
  Node_vector<Interval_set<>> original_missing_sites_all_; // Only nonempty in debug builds!
  auto save_original_sequences() -> void;
  auto assert_cur_tip_sequences_compatible_with_original_ones() -> void;

  bool paranoid_ = false;
  bool only_displacing_inner_nodes_ = false;
  bool topology_moves_enabled_ = true;
  bool repartitioning_enabled_ = true;
  bool alpha_move_enabled_ = false;
  bool site_partition_rate_move_enabled_ = true;
  double site_partition_rate_prior_concentration_ = 1.0;
  int64_t site_partition_rate_moves_proposed_ = 0;
  int64_t site_partition_rate_moves_accepted_ = 0;
  Site_partition_config site_partition_config_{};  // empty unless the user gave one
  Site_partition site_partition_{};
  std::vector<double> site_partition_rates_{};
  bool mu_move_enabled_ = true;
  bool subst_exchangeability_move_enabled_ = true;
  bool subst_frequencies_move_enabled_ = true;
  bool final_pop_size_move_enabled_ = true;
  bool pop_growth_rate_move_enabled_ = true;
  bool skygrid_tau_move_enabled_ = false;  // tau can either be fixed, or inferred with a gamma prior (as in Gill et al 2012, Eq. 15)
  bool skygrid_low_gamma_barrier_enabled_ = true;
  bool skygrid_hmc_force_check_enabled_ = false;
  double skygrid_hmc_force_check_step_ = 1e-4;
  mutable double skygrid_hmc_max_force_discrepancy_ = 0.0;
  mutable double skygrid_hmc_max_force_over_resolution_ = 0.0;
  mutable int64_t skygrid_hmc_force_checks_ = 0;
  mutable double skygrid_hmc_worst_force_analytic_ = 0.0;
  mutable double skygrid_hmc_worst_force_numeric_ = 0.0;
  mutable double skygrid_hmc_worst_force_gamma_ = 0.0;
  mutable double skygrid_hmc_worst_force_min_gamma_ = 0.0;

  // Parameters
  std::shared_ptr<const Pop_model> pop_model_;
  double pop_inv_n0_prior_alpha_;
  double pop_inv_n0_prior_beta_;
  double pop_g_prior_mu_;
  double pop_g_prior_scale_;
  double pop_g_move_window_ = 1.0 / 365.0;
  double pop_g_min_;
  double pop_g_max_;
  double skygrid_tau_;
  double skygrid_tau_prior_alpha_;
  double skygrid_tau_prior_beta_;
  double skygrid_low_gamma_barrier_loc_;
  double skygrid_low_gamma_barrier_scale_;
  double skygrid_inv_nbar_prior_alpha_;
  double skygrid_inv_nbar_prior_beta_;
  double alpha_;
  Site_vector<double> nu_;
  Subst_model subst_model_;
  double subst_exchangeability_prior_log_mean_transition_ = 1.0;
  double subst_exchangeability_prior_log_mean_other_ = 0.0;
  double subst_exchangeability_prior_sd_ = 1.25;
  double mu_prior_alpha_;
  double mu_prior_beta_;

  bool mpox_hack_enabled_ = false;
  double mpox_mu_ = 0.0;
  double mpox_mu_star_ = 0.0;

  // Derived quantities
  mutable bool derived_quantities_valid_ = false;
  mutable int64_t last_revalidation_step_ = 0;
  mutable double t_max_tip_ = std::numeric_limits<double>::quiet_NaN();
  mutable Global_evo_model evo_;
  mutable double log_G_ = std::numeric_limits<double>::quiet_NaN();
  mutable Partition_vector<Seq_vector<double>> Ttwiddle_beta_a_ = {};
  mutable int num_muts_ = -1;
  mutable double log_coalescent_prior_ = std::numeric_limits<double>::quiet_NaN();
  // log P_exact - log P_cell of the current genealogy when the exact correction is on, exactly 0.0 otherwise
  mutable double log_coal_exact_correction_ = 0.0;
  bool exact_coalescent_correction_ = false;
  bool global_scale_move_enabled_ = false;
  double global_scale_move_window_ = 0.1;
  int global_scale_moves_per_global_move_ = 1;
  double history_gibbs_weight_ = 0.0;
  int history_gibbs_sites_per_move_ = 1;
  int site_history_refreshes_per_global_move_ = 0;
  double marginal_node_time_weight_ = 0.0;
  double marginal_node_time_window_ = 0.5;
  History_gibbs_stats history_gibbs_stats_{};
  int64_t pop_size_proposed_ = 0;
  int64_t pop_size_accepted_ = 0;
  int64_t pop_growth_proposed_ = 0;
  int64_t pop_growth_accepted_ = 0;
  int64_t pop_growth_out_of_bounds_ = 0;
  int64_t global_scale_proposed_ = 0;
  int64_t global_scale_accepted_ = 0;
  double global_scale_abs_u_accepted_ = 0.0;
  double global_scale_max_abs_residual_ = 0.0;
  int64_t exact_batches_proposed_ = 0;
  int64_t exact_batches_accepted_ = 0;
  mutable double log_other_priors_ = std::numeric_limits<double>::quiet_NaN();
  mutable double log_calibration_prior_ = 0.0;
  mutable Seq_matrix<int> num_muts_ab_ = {};
  mutable Seq_vector<int> state_frequencies_of_ref_sequence_ = {};

  // Statistics of the reference sequence that take a pass over every site.  They depend only on the reference sequence, the
  // site partitions and the site rates nu_l.  On long references these passes dominate a recomputation of the derived
  // quantities, which happens several times per global step, so they are kept here and recomputed only when one of their
  // inputs has changed; a recomputation runs the same code on the same inputs, so the values are the same bit for bit.
  //
  // Every input carries a version counter.  `ref_version_` is bumped wherever the reference sequence of `tree_` can change:
  // `normalize_root`, every restoration of `tree_` from a snapshot, every non-const call of `tree()` and every call of the
  // public `invalidate_derived_quantities()` (so code outside Run that edits the tree and then invalidates, as it must
  // already, gets a full recomputation exactly as before).  Inside Run, the places that invalidate the derived
  // quantities without being able to change the reference sequence use `invalidate_derived_quantities_same_reference()`.
  struct Ref_site_stats {
    // `state_frequencies` depends on the reference sequence and the site partitions only; `ntwiddle_beta_a` on the site
    // rates too, so a new draw of the site rates recomputes only the latter.
    bool valid = false;
    int64_t ref_version = -1;
    int64_t site_layout_version = -1;
    Partition_vector<Seq_vector<int>> state_frequencies;
    bool ntwiddle_valid = false;
    int64_t evo_nu_l_version = -1;
    Partition_vector<Seq_vector<double>> ntwiddle_beta_a;
  };
  mutable Ref_site_stats ref_site_stats_;
  mutable int64_t ref_version_ = 0;   // bumped whenever the reference sequence of `tree_` may have changed
  int64_t nu_version_ = 0;            // bumped whenever `nu_` changes
  int64_t site_layout_version_ = 0;   // bumped whenever `evo_.partition_for_site` changes
  mutable int64_t evo_nu_l_version_ = -1;  // the `nu_version_` that `evo_.nu_l` was last copied from
  mutable int64_t nu_sums_version_ = -1;   // the `nu_version_` that the two sums below were computed for
  mutable double nu_sum_ = 0.0;
  mutable double nu_sum_log_ = 0.0;
  auto cur_ref_site_stats() const -> const Ref_site_stats&;
  auto ref_site_stats_inputs_unchanged() const -> bool;
  mutable int64_t nu_l_all_ones_version_ = -2;  // the `evo_nu_l_version_` that `nu_l_all_ones_` describes
  mutable bool nu_l_all_ones_ = false;
  auto evo_nu_l_all_ones() const -> bool;

  // The subruns read `evo_` through one immutable copy that they all share (`shared_evo_`), and the cumulative
  // per-site rates of the reference sequence through one shared table (`ref_cum_Q_l_`), instead of each holding and
  // recomputing its own.  Each is renewed only when one of its inputs differs from those it was made from: the
  // per-site arrays of `evo_` are keyed on `site_layout_version_` and `evo_nu_l_version_`, the per-partition models
  // are compared by value, and the table is also keyed on `ref_version_`.  Both are recomputed by the same code on
  // equal inputs, so their values are the same bit for bit as a recomputation.  The copy is only renewed when the
  // subruns are given their parameters, after they have released the old one, so that it can be rewritten in place.
  mutable std::shared_ptr<Global_evo_model> shared_evo_;
  mutable int64_t shared_evo_site_layout_version_ = -1;
  mutable int64_t shared_evo_nu_l_version_ = -1;
  mutable std::shared_ptr<std::vector<double>> ref_cum_Q_l_;
  mutable int64_t ref_cum_Q_l_ref_version_ = -1;
  mutable int64_t ref_cum_Q_l_site_layout_version_ = -1;
  mutable int64_t ref_cum_Q_l_nu_l_version_ = -1;
  mutable Partition_vector<Site_evo_model> ref_cum_Q_l_partition_models_;
  auto cur_shared_evo() const -> std::shared_ptr<const Global_evo_model>;
  auto cur_ref_cum_Q_l() const -> std::shared_ptr<const std::vector<double>>;
  auto ref_cum_Q_l_is_current() const -> bool;
  auto invalidate_derived_quantities_same_reference() -> void { derived_quantities_valid_ = false; }

  mutable Scalable_coalescent_prior coalescent_prior_;
  mutable std::vector<Very_scalable_coalescent_prior_part> coalescent_prior_parts_;

  auto refresh_partition_stencils() -> void;
  auto threshold_partitioning_active() const -> bool { return threshold_partitioning_ && num_parts_ > 1; }
  auto draw_threshold_partition_stencil() -> std::vector<Partition_part_info>;
  auto constrain_threshold_root_part() -> void;
  auto check_threshold_partition_kept() const -> void;
  auto run_threshold_partition_local_moves(int count) -> void;
  auto repartition() -> void;
  auto reassemble() -> void;
  auto normalize_root() -> void;
  auto push_tree_to_subruns() -> void;
  auto push_global_params_to_subruns() -> void;
  auto reset_very_scalable_coalescent_parts() -> void;
  auto assign_calibrations_to_subruns() -> void;
  auto check_free_topology_calibrations() const -> void;
  // Empty if the exact coalescent correction supports topology moves in the current configuration, else why not
  auto free_topology_exact_correction_problem() const -> std::string;
  
  auto recalc_derived_quantities() const -> void;
  auto check_derived_quantities() const -> void;
  auto check_global_and_local_totals_match() const -> void;

  auto derive_evo() const -> void;

  auto run_local_moves(int count) -> void;
  auto run_corrected_local_batch(int count) -> void;
  auto calc_cur_log_coalescent_exact() const -> double;
  // The exact Kingman density's population moves need the genealogy reduced to its event intervals,
  // cut at the Skygrid knots.  Built once per move; the genealogy does not change inside one.
  auto make_skygrid_exact_events() const -> Exact_coalescent_events;
  auto run_global_moves() -> void;
  
  auto mu_move() -> void;
  auto mpox_hack_moves() -> void;
  auto apply_subst_model(Subst_model model, const Site_evo_model& site_evo) -> void;
  auto site_partition_rate_moves() -> void;
  auto pop_size_move() -> void;
  auto pop_growth_rate_move() -> void;
  auto global_scale_move() -> void;
  auto site_history_refresh_moves() -> void;
  auto check_global_scale_move_supported() const -> void;
  auto skygrid_tau_move() -> void;
  auto skygrid_gammas_hmc_move() -> void;
  auto skygrid_gammas_zero_mode_gibbs_move() -> void;
};

}  // namespace delphy

#endif // DELPHY_RUN_H_
