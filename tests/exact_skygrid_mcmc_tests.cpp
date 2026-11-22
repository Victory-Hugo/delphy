// Scientific acceptance tests for the exact Kingman coalescent target under a Skygrid population
// model.
//
// `exact_skygrid_tests.cpp` establishes that the exact density and its gradient are right.  This file
// establishes that the sampler targets that density: that every Skygrid population move is a move of
// the exact posterior, that the target no longer depends on the coalescent grid, that the
// incrementally maintained log posterior survives the moves across tree partitions and a
// checkpoint, and that the marginal of the Skygrid parameters under the whole moveset is exactly
// their prior.
//
// Why the last statement is exact here and was only approximate before: with the exact correction on,
// the coalescent term is the Kingman density, whose integral over the node times of a fixed ranked
// topology is a constant independent of the population curve.  So with a flat sequence likelihood the
// marginal of the Skygrid parameters is their prior, with no normalization left over.  The cell prior
// on a grid has no such property; its normalization over genealogies depends on the curve, which is
// the caveat the previous Skygrid recovery test had to carry (see
// doc/human_uniparental/49_skygrid_acceptance_task31.md (development record, not in this repository), part 1, kind 4).

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <sstream>
#include <vector>

#include <absl/strings/str_format.h>
#include <boost/math/special_functions/gamma.hpp>

#include "ctpl_stl.h"

#include "calibration.h"
#include "coalescent_grid.h"
#include "exact_coalescent.h"
#include "phylo_tree.h"
#include "pop_model.h"
#include "run.h"

namespace delphy {

namespace {

using Type = Skygrid_pop_model::Type;

// Kingman genealogy with every tip at t = 0 and one invariant site, as in
// `skygrid_acceptance_tests.cpp` and `exact_coalescent_correction_tests.cpp`, so that the three sets
// of results are comparable.
auto make_kingman_tree(int num_tips, std::mt19937_64& g, double n0) -> Phylo_tree {
  auto tree = Phylo_tree{2 * num_tips - 1};
  tree.ref_sequence = Real_sequence(1, Real_seq_letter::A);
  auto active = std::vector<Node_index>{};
  for (auto i = 0; i != num_tips; ++i) {
    tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = 0.0;
    tree.at(i).name = "S" + std::to_string(i);
    active.push_back(i);
  }
  auto age = 0.0;
  auto next = num_tips;
  while (std::ssize(active) > 1) {
    auto k = static_cast<double>(std::ssize(active));
    age += std::exponential_distribution<double>{k * (k - 1) / 2.0 / n0}(g);
    auto i = g() % active.size(); auto a = active[i]; active[i] = active.back(); active.pop_back();
    auto j = g() % active.size(); auto b = active[j]; active[j] = active.back(); active.pop_back();
    auto& node = tree.at(next);
    node.t = -age;
    node.t_min = -std::numeric_limits<float>::max();
    node.t_max = +std::numeric_limits<float>::max();
    node.name = "n" + std::to_string(next);
    node.children = {a, b};
    tree.at(a).parent = next;
    tree.at(b).parent = next;
    active.push_back(next++);
  }
  tree.root = next - 1;
  tree.at(tree.root).parent = k_no_node;
  return tree;
}

struct Skygrid_case {
  std::string name;
  std::vector<double> x;
  std::vector<double> gamma;
};

auto uniform_knots(double lo, double hi, int M) -> std::vector<double> {
  auto x = std::vector<double>{};
  for (auto k = 0; k <= M; ++k) { x.push_back(lo + (hi - lo) * k / M); }
  return x;
}

// Curves the sampler has to cope with: flat, monotone, oscillating, a steep one, non-uniform knots
// and the smallest possible curve.  The population scale is matched to the genealogies below so that
// the exact density is finite and the moves are not permanently rejected.
auto skygrid_cases() -> std::vector<Skygrid_case> {
  auto cases = std::vector<Skygrid_case>{};
  cases.push_back({"flat-5", uniform_knots(-4000.0, 0.0, 5), std::vector<double>(6, std::log(400.0))});
  cases.push_back({"monotone-5", uniform_knots(-4000.0, 0.0, 5),
                   {std::log(100.0), std::log(200.0), std::log(400.0), std::log(800.0),
                    std::log(1600.0), std::log(3200.0)}});
  cases.push_back({"oscillating-6", uniform_knots(-4000.0, 0.0, 6),
                   {6.0, 4.5, 7.0, 4.0, 7.5, 5.0, 6.0}});
  cases.push_back({"steep-3", uniform_knots(-4000.0, 0.0, 3), {9.0, 3.0, 9.0, 3.0}});
  cases.push_back({"non-uniform-4", {-20000.0, -8000.0, -2000.0, -400.0, 0.0},
                   {7.0, 5.0, 8.0, 4.0, 6.5}});
  cases.push_back({"two-knots", {-4000.0, 0.0}, {5.0, 7.0}});
  return cases;
}

auto type_name(Type type) -> const char* {
  return type == Type::k_staircase ? "staircase" : "log-linear";
}

// Tip names below `node`, as the calibration config needs them.
auto tips_under(const Phylo_tree& tree, Node_index node) -> std::vector<std::string> {
  auto out = std::vector<std::string>{};
  auto stack = std::vector<Node_index>{node};
  while (not stack.empty()) {
    auto v = stack.back();
    stack.pop_back();
    if (tree.at(v).is_tip()) { out.push_back(tree.at(v).name); continue; }
    for (auto c : tree.at(v).children) { stack.push_back(c); }
  }
  return out;
}

// The non-root inner node with the most tips below it
auto largest_non_root_node(const Phylo_tree& tree) -> Node_index {
  auto best = k_no_node;
  auto best_size = 0;
  for (auto v = Node_index{0}; v != std::ssize(tree); ++v) {
    if (tree.at(v).is_tip() || v == tree.root) { continue; }
    auto size = static_cast<int>(tips_under(tree, v).size());
    if (size > best_size) { best_size = size; best = v; }
  }
  return best;
}

// Age calibrations on the root and on the largest non-root clade, lognormal with a median 15% above
// the true age and hard bounds at 0.4 and 4 times it; the same shape
// `exact_coalescent_correction_tests.cpp` uses, so the two sets of results are comparable.
auto calibration_config_for(const Phylo_tree& tree, Node_index node) -> std::string {
  auto os = std::ostringstream{};
  auto block = [&](const std::string& name, const std::string& taxa, double age_days) {
    auto age_years = age_days / 365.0;
    os << "calibration " << name << "\n  taxa " << taxa << "\n  age_units years_bp\n"
       << "  age_reference_year 2020\n  distribution lognormal\n  median " << 1.15 * age_years
       << "\n  sigma 0.2\n  oldest " << 4.0 * age_years << "\n  youngest " << 0.4 * age_years
       << "\nend\n";
  };
  os << "version 1\n\n";
  block("root", "all", -tree.at(tree.root).t);
  auto taxa = std::string{};
  for (const auto& name : tips_under(tree, node)) { taxa += name + " "; }
  block("clade", taxa, -tree.at(node).t);
  return os.str();
}

// A Skygrid run with the exact Kingman correction on: the grid is pinned (the correction requires
// it), the topology is fixed, the sequence likelihood is flat and every tip shares one date.
auto make_exact_skygrid_run(
    ctpl::thread_pool& pool,
    const Phylo_tree& tree,
    const Skygrid_case& c,
    Type type,
    double tau,
    uint32_t seed,
    int parts = 1,
    double t_step = 0.0)
    -> std::unique_ptr<Run> {
  auto span = -tree.at_root().t;
  if (t_step <= 0.0) { t_step = std::max(1.0, span / 200.0); }
  auto run = std::make_unique<Run>(pool, std::mt19937{seed}, seed, tree, t_step);
  run->set_pop_model(std::make_shared<Skygrid_pop_model>(c.x, c.gamma, type));
  run->set_skygrid_tau(tau);
  run->set_mu(1e-13);
  run->set_mu_move_enabled(false);
  run->set_alpha_move_enabled(false);
  run->set_topology_moves_enabled(false);
  run->set_num_parts(parts);
  run->set_local_moves_per_global_move(10);
  run->set_coal_prior_fixed_t_step(t_step);
  run->set_exact_coalescent_correction(true);
  return run;
}

// The exact Kingman density of the run's current state, computed from the reference implementation
// in `coalescent_grid.h` rather than from anything the sampler maintains.
auto reference_exact_log_prior(const Run& run) -> double {
  return calc_exact_coalescent_log_prior(coalescent_event_times_of(run.tree()), run.pop_model());
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// 1. The correction accepts a Skygrid model, and still refuses what it refused before
// ---------------------------------------------------------------------------------------------

TEST(Exact_skygrid_mcmc_test, the_correction_accepts_a_skygrid_model) {
  auto pool = ctpl::thread_pool{1};
  auto g = std::mt19937_64{1};
  auto tree = make_kingman_tree(8, g, 400.0);
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    for (const auto& c : skygrid_cases()) {
      EXPECT_NO_THROW({ auto run = make_exact_skygrid_run(pool, tree, c, type, 1.0, 7); })
          << c.name << " " << type_name(type);
    }
  }
}

TEST(Exact_skygrid_mcmc_test, the_correction_still_refuses_an_unpinned_grid_and_a_free_topology_with_several_parts) {
  auto pool = ctpl::thread_pool{1};
  auto g = std::mt19937_64{1};
  auto tree = make_kingman_tree(8, g, 400.0);
  auto cases = skygrid_cases();
  const auto& c = cases.front();

  {  // No pinned grid
    auto run = std::make_unique<delphy::Run>(pool, std::mt19937{7}, 7u, tree, 20.0);
    run->set_pop_model(std::make_shared<Skygrid_pop_model>(c.x, c.gamma, Type::k_staircase));
    run->set_topology_moves_enabled(false);
    EXPECT_THROW(run->set_exact_coalescent_correction(true), std::invalid_argument);
  }
  {  // Free topology with several tree partition parts
    auto run = std::make_unique<delphy::Run>(pool, std::mt19937{7}, 7u, tree, 20.0);
    run->set_num_parts(2);
    run->set_pop_model(std::make_shared<Skygrid_pop_model>(c.x, c.gamma, Type::k_staircase));
    run->set_coal_prior_fixed_t_step(20.0);
    run->set_topology_moves_enabled(true);
    EXPECT_THROW(run->set_exact_coalescent_correction(true), std::invalid_argument);
  }
  {  // The grid may not be released once the correction is on
    auto run = make_exact_skygrid_run(pool, tree, c, Type::k_staircase, 1.0, 7);
    EXPECT_THROW(run->set_coal_prior_fixed_t_step(0.0), std::invalid_argument);
  }
}

// ---------------------------------------------------------------------------------------------
// 2. The coalescent term of the target is the exact density, before and after sampling
// ---------------------------------------------------------------------------------------------

TEST(Exact_skygrid_mcmc_test, the_targets_coalescent_term_is_the_exact_density) {
  auto pool = ctpl::thread_pool{2};
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    for (const auto& c : skygrid_cases()) {
      auto g = std::mt19937_64{5};
      auto tree = make_kingman_tree(20, g, 400.0);
      auto run = make_exact_skygrid_run(pool, tree, c, type, 1.0, 11);

      auto expected = reference_exact_log_prior(*run);
      EXPECT_NEAR(run->calc_cur_log_coalescent_target(), expected, 1e-8 * std::max(1.0, std::abs(expected)))
          << c.name << " " << type_name(type) << " (initial)";

      run->do_mcmc_steps(2000);
      expected = reference_exact_log_prior(*run);
      EXPECT_NEAR(run->calc_cur_log_coalescent_target(), expected, 1e-6 * std::max(1.0, std::abs(expected)))
          << c.name << " " << type_name(type) << " (after 2,000 steps)";
    }
  }
}

// The whole point of the exact target: it is a function of the genealogy and the population curve
// alone, so pinning the grid at any width gives the same target.  Under the cell prior the same
// comparison differs by the discretization excess, which is what `calc_coal_prior_grid_excess`
// measures and what biased every previous Skygrid run.
TEST(Exact_skygrid_mcmc_test, the_exact_target_does_not_depend_on_the_grid) {
  auto pool = ctpl::thread_pool{2};
  auto g = std::mt19937_64{5};
  auto tree = make_kingman_tree(20, g, 400.0);

  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    for (const auto& c : skygrid_cases()) {
      auto reference = 0.0;
      auto cell_priors = std::vector<double>{};
      auto first = true;
      for (auto t_step : {500.0, 100.0, 20.0, 4.0}) {
        auto run = make_exact_skygrid_run(pool, tree, c, type, 1.0, 11, 1, t_step);
        auto target = run->calc_cur_log_coalescent_target();
        cell_priors.push_back(run->calc_coal_prior_grid_excess().excess);
        if (first) { reference = target; first = false; }
        EXPECT_NEAR(target, reference, 1e-8 * std::max(1.0, std::abs(reference)))
            << c.name << " " << type_name(type) << " t_step=" << t_step;
      }
      // ... while the cell prior the correction starts from does depend on it, so the comparison is
      // not vacuous
      EXPECT_GT(std::abs(cell_priors.front() - cell_priors.back()), 1e-6)
          << c.name << " " << type_name(type) << ": the grid excess did not change with the grid";
    }
  }
}

// ---------------------------------------------------------------------------------------------
// 3. Every Skygrid population move targets the exact density
// ---------------------------------------------------------------------------------------------

// The HMC forces must be the gradient of the potential the move evaluates, which with the correction
// on is the exact density plus the Skygrid prior.  The diagnostic compares them against a central
// finite difference of that same potential at every knot at every trajectory step.
TEST(Exact_skygrid_mcmc_test, the_hmc_forces_are_the_gradient_of_the_exact_potential) {
  auto pool = ctpl::thread_pool{2};
  auto worst = 0.0;
  auto checks = int64_t{0};

  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    for (const auto& c : skygrid_cases()) {
      auto g = std::mt19937_64{5};
      auto tree = make_kingman_tree(20, g, 400.0);
      auto run = make_exact_skygrid_run(pool, tree, c, type, 1.0, 23);
      run->set_skygrid_tau_move_enabled(true);
      run->set_skygrid_hmc_force_check_step(1e-5);
      run->set_skygrid_hmc_force_check_enabled(true);
      run->do_mcmc_steps(200);

      EXPECT_GT(run->skygrid_hmc_force_checks(), 0) << c.name << " " << type_name(type);
      // Gated on the discrepancy measured in units of what the central difference can resolve: at
      // these spans the potential reaches 1e8, where a fixed relative gate would be measuring the
      // finite difference's own round-off and not the gradient.
      EXPECT_LT(run->skygrid_hmc_max_force_discrepancy_over_resolution(), 1.0)
          << c.name << " " << type_name(type)
          << ": analytic " << run->skygrid_hmc_worst_force_analytic()
          << " vs numeric " << run->skygrid_hmc_worst_force_numeric()
          << " at gamma = " << run->skygrid_hmc_worst_force_gamma()
          << "; worst relative discrepancy " << run->skygrid_hmc_max_force_discrepancy();
      worst = std::max(worst, run->skygrid_hmc_max_force_discrepancy_over_resolution());
      checks += run->skygrid_hmc_force_checks();
    }
  }
  std::cerr << absl::StreamFormat(
      "EXACT HMC FORCES worst discrepancy %.3g of the comparison's own resolution, over %d checks\n",
      worst, checks);
}

// The zero-mode Gibbs move draws I_bar = exp(-gamma_bar) from Gamma(N_inner + alpha, rate B + beta).
// With the correction on, B must be the exact density's rate-integral term divided by I_bar, not the
// grid's.  If it were the grid's, the move would sample the conditional of the wrong target, which
// would show up as a dependence of the move's own conditional on the cell width.  This checks the
// identity the move rests on against the exact density directly, at the state the run is in.
TEST(Exact_skygrid_mcmc_test, the_zero_mode_conditional_is_the_exact_densitys) {
  auto pool = ctpl::thread_pool{2};
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    for (const auto& c : skygrid_cases()) {
      auto g = std::mt19937_64{5};
      auto tree = make_kingman_tree(20, g, 400.0);
      auto run = make_exact_skygrid_run(pool, tree, c, type, 1.0, 29);
      run->do_mcmc_steps(500);

      const auto& pm = static_cast<const Skygrid_pop_model&>(run->pop_model());
      auto events = Exact_coalescent_events{coalescent_event_times_of(run->tree()), pm.x()};
      auto n_inner = static_cast<double>((std::ssize(run->tree()) - 1) / 2);
      auto i_bar = std::exp(-pm.gamma_bar());
      auto b = events.calc_intensity_term(pm) / i_bar;

      // log P_exact as a function of the zero mode, with everything else held fixed:
      //   log P_exact(I_bar) = N_inner log I_bar - B I_bar + const.
      // Predicted at four shifts from the fitted constant at the current state.
      auto base = events.calc_log_prior(pm);
      auto konst = base - (n_inner * std::log(i_bar) - b * i_bar);
      for (auto shift : {-2.0, -0.5, 0.5, 2.0}) {
        auto shifted_gamma = pm.gamma();
        for (auto& v : shifted_gamma) { v += shift; }
        auto shifted = Skygrid_pop_model{pm.x(), shifted_gamma, type};
        auto shifted_i_bar = std::exp(-shifted.gamma_bar());
        auto predicted = n_inner * std::log(shifted_i_bar) - b * shifted_i_bar + konst;
        auto got = events.calc_log_prior(shifted);
        EXPECT_NEAR(got, predicted, 1e-7 * std::max(1.0, std::abs(predicted)))
            << c.name << " " << type_name(type) << " shift " << shift;
      }
    }
  }
}

// The smoothness `tau` does not enter the coalescent term at all, whichever target is in force, so
// its full conditional is the same Gamma under the exact density as under the cell prior and the
// Gibbs move needs no change.  Both halves of that statement are checked here against the posterior
// the run actually evaluates, with the exact correction on: the coalescent term does not move with
// `tau` at all, and the whole log posterior's dependence on `tau` is exactly
//   (alpha - 1 + M/2) log tau - (beta + SS/2) tau,
// whose conditional is the Gamma(alpha + M/2, rate beta + SS/2) that `skygrid_tau_move` samples.
TEST(Exact_skygrid_mcmc_test, the_full_conditional_of_tau_is_unchanged_by_the_exact_target) {
  auto pool = ctpl::thread_pool{2};
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    for (const auto& c : skygrid_cases()) {
      auto g = std::mt19937_64{5};
      auto tree = make_kingman_tree(12, g, 400.0);
      auto run = make_exact_skygrid_run(pool, tree, c, type, 1.0, 31);
      constexpr auto alpha = 3.0;
      constexpr auto beta = 3.0;
      run->set_skygrid_tau_prior_alpha(alpha);
      run->set_skygrid_tau_prior_beta(beta);
      run->set_skygrid_tau_move_enabled(true);
      run->do_mcmc_steps(200);

      const auto& pm = static_cast<const Skygrid_pop_model&>(run->pop_model());
      auto M = pm.M();
      auto sum_sq = 0.0;
      for (auto k = 1; k <= M; ++k) {
        auto d = pm.gamma(k) - pm.gamma(k-1);
        sum_sq += d * d;
      }
      auto log_conditional = [&](double tau) {
        return (alpha - 1.0 + 0.5 * M) * std::log(tau) - (beta + 0.5 * sum_sq) * tau;
      };

      auto base_tau = run->skygrid_tau();
      auto base_posterior = run->log_posterior();
      auto base_coalescent = run->calc_cur_log_coalescent_target();
      for (auto tau : {0.17, 0.5, 1.0, 2.3, 11.0}) {
        run->set_skygrid_tau(tau);
        EXPECT_EQ(run->calc_cur_log_coalescent_target(), base_coalescent)
            << c.name << " " << type_name(type) << " tau=" << tau;
        auto got = run->log_posterior() - base_posterior;
        auto expected = log_conditional(tau) - log_conditional(base_tau);
        EXPECT_NEAR(got, expected, 1e-8 * std::max(1.0, std::abs(expected)))
            << c.name << " " << type_name(type) << " tau=" << tau;
      }
      run->set_skygrid_tau(base_tau);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// 4. Incremental bookkeeping, tree partitions and resume
// ---------------------------------------------------------------------------------------------

// Every incrementally maintained term of the log posterior, including the exact correction itself,
// has to agree with a full recomputation after the moves have run -- with the genealogy assembled
// from 1, 2 and 4 tree partitions.
TEST(Exact_skygrid_mcmc_test, the_incremental_log_posterior_survives_the_exact_skygrid_moves) {
  auto pool = ctpl::thread_pool{4};
  for (auto parts : {1, 2, 4}) {
    for (auto type : {Type::k_staircase, Type::k_log_linear}) {
      for (const auto& c : skygrid_cases()) {
        auto g = std::mt19937_64{3};
        auto tree = make_kingman_tree(40, g, 400.0);
        auto run = make_exact_skygrid_run(pool, tree, c, type, 2.0, 13, parts);
        run->set_skygrid_tau_move_enabled(true);
        run->do_mcmc_steps(2000);

        auto incremental = run->log_posterior();
        run->invalidate_derived_quantities();
        auto recomputed = run->log_posterior();
        EXPECT_NEAR(incremental, recomputed, 1e-6 * std::max(1.0, std::abs(recomputed)))
            << c.name << " " << type_name(type) << " parts=" << parts;

        // ... and the coalescent term of that posterior is still the exact density
        auto expected = reference_exact_log_prior(*run);
        EXPECT_NEAR(run->calc_cur_log_coalescent_target(), expected,
                    1e-6 * std::max(1.0, std::abs(expected)))
            << c.name << " " << type_name(type) << " parts=" << parts;

        // The batch correction must be doing something: all-accept or all-reject would mean the
        // correction is not engaging with the genealogy at all
        EXPECT_GT(run->exact_coalescent_batches_proposed(), 0)
            << c.name << " " << type_name(type) << " parts=" << parts;
      }
    }
  }
}

// A run restored from a checkpoint is on the same target and keeps sampling consistently, with the
// exact correction on and a Skygrid curve.  Delphy's checkpoint carries the tree, the parameters and
// the generator state; the grid and the correction are re-supplied by the caller, exactly as
// `joint_model_mcmc_tests.cpp` does for the exponential model.
TEST(Exact_skygrid_mcmc_test, a_resumed_run_is_on_the_same_target_and_stays_consistent) {
  auto pool = ctpl::thread_pool{2};
  auto cases = skygrid_cases();
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    const auto& c = cases[2];  // the oscillating curve
    auto g = std::mt19937_64{3};
    auto tree = make_kingman_tree(20, g, 400.0);
    auto t_step = std::max(1.0, -tree.at_root().t / 200.0);

    auto reference = make_exact_skygrid_run(pool, tree, c, type, 1.0, 17, 1, t_step);
    reference->set_skygrid_tau_move_enabled(true);
    // The substitution model's own moves are on by default; a checkpoint of them is
    // `joint_model_mcmc_tests.cpp`'s subject, not this one, so they are held fixed here and the
    // comparison isolates the coalescent target and the Skygrid parameters.
    reference->set_subst_exchangeability_move_enabled(false);
    reference->set_subst_frequencies_move_enabled(false);
    reference->set_site_partition_rate_move_enabled(false);
    reference->do_mcmc_steps(1000);

    auto state = reference->resume_state();
    const auto& saved_pm = static_cast<const Skygrid_pop_model&>(reference->pop_model());
    auto saved_x = saved_pm.x();
    auto saved_gamma = saved_pm.gamma();

    auto resumed = std::make_unique<delphy::Run>(
        pool, std::mt19937{17}, 17u, reference->tree(), t_step);
    resumed->set_pop_model(std::make_shared<Skygrid_pop_model>(saved_x, saved_gamma, type));
    resumed->set_skygrid_tau(reference->skygrid_tau());
    resumed->set_mu(reference->mu());
    resumed->set_mu_move_enabled(false);
    resumed->set_alpha_move_enabled(false);
    resumed->set_topology_moves_enabled(false);
    resumed->set_local_moves_per_global_move(10);
    resumed->set_coal_prior_fixed_t_step(t_step);
    resumed->set_exact_coalescent_correction(true);
    resumed->set_skygrid_tau_move_enabled(true);
    resumed->set_subst_exchangeability_move_enabled(false);
    resumed->set_subst_frequencies_move_enabled(false);
    resumed->set_site_partition_rate_move_enabled(false);
    resumed->set_step(reference->step());
    resumed->restore_resume_state(state);

    // Same state, same target
    EXPECT_EQ(resumed->step(), reference->step()) << type_name(type);
    EXPECT_NEAR(resumed->calc_cur_log_coalescent_target(),
                reference->calc_cur_log_coalescent_target(),
                1e-9 * std::max(1.0, std::abs(reference->calc_cur_log_coalescent_target())))
        << type_name(type);
    EXPECT_NEAR(resumed->log_posterior(), reference->log_posterior(),
                1e-9 * std::max(1.0, std::abs(reference->log_posterior())))
        << type_name(type);

    // ... and sampling on from there keeps the incremental bookkeeping and the exact target intact
    resumed->do_mcmc_steps(1000);
    auto incremental = resumed->log_posterior();
    resumed->invalidate_derived_quantities();
    EXPECT_NEAR(incremental, resumed->log_posterior(),
                1e-6 * std::max(1.0, std::abs(incremental))) << type_name(type);
    auto expected = reference_exact_log_prior(*resumed);
    EXPECT_NEAR(resumed->calc_cur_log_coalescent_target(), expected,
                1e-6 * std::max(1.0, std::abs(expected))) << type_name(type);
  }
}

// Rollback.  A rejected batch has to leave the genealogy, the mutation histories and every derived
// quantity exactly as they were, and the only way to check that from outside is to rebuild a Run from
// the current genealogy and compare.  The grid is left coarse so that batches really are rejected: on
// a fine surrogate almost every batch is accepted and the rollback path is never taken.
TEST(Exact_skygrid_mcmc_test, a_rejected_batch_leaves_a_consistent_state) {
  auto pool = ctpl::thread_pool{4};
  auto cases = skygrid_cases();
  for (auto parts : {1, 2}) {
    for (auto type : {Type::k_staircase, Type::k_log_linear}) {
      const auto& c = cases[2];   // the oscillating curve, whose cell prior is furthest from exact
      auto g = std::mt19937_64{23};
      auto tree = make_kingman_tree(30, g, 400.0);
      auto span = -tree.at_root().t;
      auto coarse = std::max(1.0, span / 8.0);   // eight cells: the surrogate is deliberately bad
      auto run = make_exact_skygrid_run(pool, tree, c, type, 1.0, 41, parts, coarse);
      run->set_local_moves_per_global_move(40);

      for (auto batch = 0; batch != 60; ++batch) {
        run->do_mcmc_steps(40);
        auto expected = reference_exact_log_prior(*run);
        ASSERT_NEAR(run->calc_cur_log_coalescent_target(), expected,
                    1e-6 * std::max(1.0, std::abs(expected)))
            << type_name(type) << " parts=" << parts << " batch " << batch;

        // A Run rebuilt from the current genealogy and curve agrees, whether the last batch was
        // accepted or rolled back
        const auto& pm = static_cast<const Skygrid_pop_model&>(run->pop_model());
        auto fresh = make_exact_skygrid_run(pool, run->tree(), c, type, 1.0, 41, parts, coarse);
        fresh->set_pop_model(std::make_shared<Skygrid_pop_model>(pm.x(), pm.gamma(), type));
        // The substitution model's own moves are on, and the batch correction does not roll them
        // back -- they are global moves, outside the batch -- so the rebuilt run is given the
        // current values, exactly as `exact_coalescent_correction_tests.cpp` does
        fresh->set_hky_kappa(run->hky_kappa());
        fresh->set_hky_pi(run->hky_pi());
        ASSERT_NEAR(fresh->log_G(), run->log_G(), 1e-6 * std::max(1.0, std::abs(run->log_G())))
            << type_name(type) << " parts=" << parts << " batch " << batch;
        ASSERT_NEAR(fresh->calc_cur_log_coalescent_target(), run->calc_cur_log_coalescent_target(),
                    1e-6 * std::max(1.0, std::abs(run->calc_cur_log_coalescent_target())))
            << type_name(type) << " parts=" << parts << " batch " << batch;
      }

      auto proposed = run->exact_coalescent_batches_proposed();
      auto accepted = run->exact_coalescent_batches_accepted();
      std::cerr << absl::StreamFormat("EXACT ROLLBACK %s parts=%d: %d of %d batches accepted\n",
                                      type_name(type), parts, accepted, proposed);
      EXPECT_GT(proposed, 0) << type_name(type) << " parts=" << parts;
      EXPECT_GT(accepted, 0) << type_name(type) << " parts=" << parts;
      EXPECT_LT(accepted, proposed)
          << type_name(type) << " parts=" << parts << ": no batch was rolled back, so this test "
          << "did not exercise the rollback path";
    }
  }
}

// Age calibrations are a factor of the target that the local moves already carry, so the whole-batch
// correction's ratio is unchanged by them: with rho proportional to L P_cells cal and pi proportional
// to L P_exact cal, the ratio is still exp(delta' - delta).  What has to hold in the implementation is
// that the incremental bookkeeping survives, that the coalescent term stays the exact density, and
// that the calibration prior is still the one a fresh run computes -- across partitions, since the
// calibrated node is assigned to a part.
TEST(Exact_skygrid_mcmc_test, calibrations_hold_with_the_exact_skygrid_target) {
  auto pool = ctpl::thread_pool{4};
  auto cases = skygrid_cases();
  for (auto parts : {1, 2, 4}) {
    for (auto type : {Type::k_staircase, Type::k_log_linear}) {
      for (const auto& c : {cases[0], cases[2]}) {   // flat and oscillating
        auto g = std::mt19937_64{19};
        auto tree = make_kingman_tree(20, g, 400.0);
        auto config = calibration_config_for(tree, largest_non_root_node(tree));

        auto run = make_exact_skygrid_run(pool, tree, c, type, 1.0, 37, parts);
        auto is = std::istringstream{config};
        run->set_calibrations(read_calibration_config(is));
        EXPECT_TRUE(run->exact_coalescent_correction()) << c.name << " " << type_name(type);
        EXPECT_NE(run->log_calibration_prior(), 0.0) << c.name << " " << type_name(type);

        run->do_mcmc_steps(2000);

        auto incremental = run->log_posterior();
        run->invalidate_derived_quantities();
        EXPECT_NEAR(incremental, run->log_posterior(), 1e-6 * std::max(1.0, std::abs(incremental)))
            << c.name << " " << type_name(type) << " parts=" << parts;

        auto expected = reference_exact_log_prior(*run);
        EXPECT_NEAR(run->calc_cur_log_coalescent_target(), expected,
                    1e-6 * std::max(1.0, std::abs(expected)))
            << c.name << " " << type_name(type) << " parts=" << parts;

        // A run rebuilt from the sampled genealogy agrees on the calibration prior, which is the
        // check that the partitioned bookkeeping of the calibrated node is intact
        auto fresh = make_exact_skygrid_run(pool, run->tree(), c, type, 1.0, 37, parts);
        auto is2 = std::istringstream{config};
        fresh->set_calibrations(read_calibration_config(is2));
        const auto& pm = static_cast<const Skygrid_pop_model&>(run->pop_model());
        fresh->set_pop_model(std::make_shared<Skygrid_pop_model>(pm.x(), pm.gamma(), type));
        EXPECT_NEAR(fresh->log_calibration_prior(), run->log_calibration_prior(),
                    1e-8 * std::max(1.0, std::abs(run->log_calibration_prior())))
            << c.name << " " << type_name(type) << " parts=" << parts;
        EXPECT_NEAR(fresh->calc_cur_log_coalescent_target(), run->calc_cur_log_coalescent_target(),
                    1e-6 * std::max(1.0, std::abs(run->calc_cur_log_coalescent_target())))
            << c.name << " " << type_name(type) << " parts=" << parts;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// 5. The target distribution of the whole moveset
// ---------------------------------------------------------------------------------------------

// With the sequence likelihood flat in the node times, the topology fixed and the zero mode given a
// proper prior, the marginal of the Skygrid parameters under the full posterior is exactly their
// prior.  The reason is specific to the exact target: the Kingman density of the coalescence times,
// at a fixed ranked topology and fixed sampling times, integrates to a constant that does not depend
// on the population curve, so nothing is left over to tilt the marginal.  The prior factorizes in
// closed form as
//
//   I_bar := exp(-gamma_bar) ~ Gamma(alpha, rate beta)        [the InverseGamma prior on N_bar]
//   delta_k := gamma_k - gamma_{k-1} ~ iid N(0, 1/tau),  independent of I_bar,
//
// so a wrong acceptance ratio in any of the three population moves, a missing Jacobian in the
// zero-mode Gibbs move, a gradient that is not the gradient of the potential, or a batch correction
// that does not restore the exact target would all move those marginals.
//
// Under the cell prior the same argument holds only approximately, because the cell prior's
// normalization over genealogies depends on the curve; the previous Skygrid recovery test had to pin
// a fine grid for that reason.  Here the grid is deliberately left COARSE: it is only the surrogate
// the local moves run against, and the correction restores the exact target whatever it is.  A coarse
// grid that still recovers the prior is the clearest evidence that the target no longer depends on
// the discretization.
namespace {

constexpr auto k_recovery_tips = 12;
constexpr auto k_recovery_nbar_alpha = 4.0;
constexpr auto k_recovery_nbar_beta = 1.2e6;   // days; E[I_bar] = alpha / beta
constexpr auto k_recovery_knots = 6;           // M = 5

// z-score of the mean of `x` against `expected`, with the standard error from batch means.
auto batch_mean_z(const std::vector<double>& x, double expected, int num_batches = 20) -> double {
  auto size = static_cast<int>(x.size()) / num_batches;
  auto means = std::vector<double>{};
  for (auto b = 0; b != num_batches; ++b) {
    auto sum = 0.0;
    for (auto i = 0; i != size; ++i) { sum += x[b * size + i]; }
    means.push_back(sum / size);
  }
  auto grand = 0.0;
  for (auto m : means) { grand += m; }
  grand /= num_batches;
  auto var = 0.0;
  for (auto m : means) { var += (m - grand) * (m - grand); }
  var /= (num_batches - 1);
  return (grand - expected) / std::sqrt(var / num_batches);
}

// A chain that starts from an exact draw of its target stays exactly at its target: after any number
// of moves of a kernel that leaves the target invariant, the state is again an exact draw.  So
// independent replicates, each started from an independent exact draw, end in independent exact draws,
// and the end states can be compared with the known prior directly, whatever the mixing time.  A
// wrong acceptance ratio, a missing Jacobian, a gradient that is not the gradient of the potential or
// a batch correction that does not restore the exact target moves the end states away from it.
//
// Each end state is reduced to probability integral transforms that are exactly uniform under the
// target: u = F(tau) for the Gamma prior of tau (when it is sampled), u = F(I_bar) for its Gamma
// prior, and u = Phi(delta_k sqrt(tau)) for each knot difference, which is N(0, 1) and independent of
// tau.  For each u the mean and the mean of (u - 1/2)^2 are compared with 1/2 and 1/12 using their
// exact variances, 1/12 and 1/180; the second statistic is the one a dispersion deficit would move.
// The statistics are bounded, so the normal approximation of their z-scores is accurate far into the
// tails, unlike a batch-means z of a heavy-tailed time average (the previous design: for delta_k^2
// with an inferred smoothness that z was biased by -0.4 to -0.8 even from a stationary start, and
// about 5% of correct chains crossed |z| = 4; see doc/extended/exact_skygrid.md, section 7).
//
// The gate |z| < 4.5 is a Bonferroni bound over every z-score of the recovery tests in this file
// (8 configurations, at most 14 each): the probability that a correct sampler fails one of them in a
// given build is below 1e-3.

constexpr auto k_tau_alpha = 3.0;        // inferred smoothness: tau ~ Gamma(3, rate 3), E[1/tau] = 3/2
constexpr auto k_tau_beta = 3.0;
constexpr auto k_recovery_t_step = 5000.0;   // 400 cells over the knot range: a coarse grid
constexpr auto k_recovery_gate = 4.5;
constexpr auto k_recovery_moves = 120;       // global moves per replicate (60 MCMC steps each)
constexpr auto k_recovery_replicates = 160;
constexpr auto k_heterochronous_tip_time = -2.0e5;

struct Recovery_config {
  Type type = Type::k_staircase;
  bool free_tau = false;        // tau ~ Gamma(k_tau_alpha, rate k_tau_beta); otherwise tau = 1
  bool heterochronous = false;  // two tips, the second at k_heterochronous_tip_time
  int parts = 1;
  // The priors the chain runs with.  They equal the ones the end states are scored against, except in
  // the validation driver, which changes them to measure the power of the statistics.
  double kernel_tau_beta = k_tau_beta;
  double kernel_nbar_beta = k_recovery_nbar_beta;
};

auto recovery_knots() -> std::vector<double> {
  auto x = std::vector<double>{};
  for (auto k = 0; k != k_recovery_knots; ++k) { x.push_back(-2.0e6 + 2.0e6 * k / (k_recovery_knots - 1)); }
  return x;
}

// The genealogy of tips at `tip_times` under `pop`, by inverting the cumulative pair hazard exactly
// with `intensity_integral` between sampling times.
auto kingman_tree_under(const Pop_model& pop, const std::vector<double>& tip_times, std::mt19937_64& g)
    -> Phylo_tree {
  auto num_tips = static_cast<int>(std::ssize(tip_times));
  auto tree = Phylo_tree{2 * num_tips - 1};
  tree.ref_sequence = Real_sequence(1, Real_seq_letter::A);
  auto order = std::vector<int>{};
  for (auto i = 0; i != num_tips; ++i) {
    tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = static_cast<float>(tip_times[i]);
    tree.at(i).name = "S" + std::to_string(i);
    order.push_back(i);
  }
  std::ranges::sort(order, [&](int a, int b) { return tip_times[a] > tip_times[b]; });
  auto active = std::vector<Node_index>{};
  auto next_tip = 0;
  auto t = tip_times[order[0]];
  auto join_tips = [&] {
    while (next_tip != num_tips && tip_times[order[next_tip]] >= t) { active.push_back(order[next_tip++]); }
  };
  join_tips();
  auto next = num_tips;
  while (next_tip != num_tips || std::ssize(active) > 1) {
    auto k = static_cast<double>(std::ssize(active));
    auto t_stop = next_tip != num_tips ? tip_times[order[next_tip]] : -std::numeric_limits<double>::infinity();
    if (k < 2) { t = t_stop; join_tips(); continue; }
    auto target = std::exponential_distribution<double>{1.0}(g) / (k * (k - 1) / 2.0);
    if (std::isfinite(t_stop) && pop.intensity_integral(t_stop, t) <= target) { t = t_stop; join_tips(); continue; }
    auto width = pop.pop_at_time(t) * target;
    auto lo = t - width;
    while (pop.intensity_integral(lo, t) < target) { width *= 2.0; lo = t - width; }
    auto hi = t;
    for (auto it = 0; it != 200 && hi - lo > 1e-9 * std::max(1.0, std::abs(lo)); ++it) {
      auto mid = 0.5 * (lo + hi);
      if (pop.intensity_integral(mid, t) < target) { hi = mid; } else { lo = mid; }
    }
    t = 0.5 * (lo + hi);
    auto i = g() % active.size(); auto a = active[i]; active[i] = active.back(); active.pop_back();
    auto j = g() % active.size(); auto b = active[j]; active[j] = active.back(); active.pop_back();
    auto& node = tree.at(next);
    node.t = t;
    node.t_min = -std::numeric_limits<float>::max();
    node.t_max = +std::numeric_limits<float>::max();
    node.name = "n" + std::to_string(next);
    node.children = {a, b};
    tree.at(a).parent = next;
    tree.at(b).parent = next;
    active.push_back(next++);
  }
  tree.root = next - 1;
  tree.at(tree.root).parent = k_no_node;
  return tree;
}

struct Recovery_state {
  double tau = 1.0;
  std::vector<double> gamma;
  Phylo_tree tree;
};

// An exact draw from the target: tau from its prior (or 1), the knot differences given tau, I_bar from
// its prior, and the genealogy given the curve.  With isochronous tips (or two tips) the topology
// carries no information about the curve, so the drawn topology may then be held fixed.
auto draw_stationary_state(const Recovery_config& c, std::mt19937_64& g) -> Recovery_state {
  auto state = Recovery_state{};
  if (c.free_tau) { state.tau = std::gamma_distribution<double>{k_tau_alpha, 1.0 / k_tau_beta}(g); }
  auto i_bar = std::gamma_distribution<double>{k_recovery_nbar_alpha, 1.0 / k_recovery_nbar_beta}(g);
  auto normal = std::normal_distribution<double>{0.0, 1.0 / std::sqrt(state.tau)};
  auto walk = std::vector<double>(k_recovery_knots, 0.0);
  for (auto k = 1; k != k_recovery_knots; ++k) { walk[k] = walk[k-1] + normal(g); }
  auto mean = 0.0;
  for (auto v : walk) { mean += v; }
  mean /= k_recovery_knots;
  for (auto k = 0; k != k_recovery_knots; ++k) { state.gamma.push_back(walk[k] - mean - std::log(i_bar)); }
  auto curve = Skygrid_pop_model{recovery_knots(), state.gamma, c.type};
  auto tip_times = c.heterochronous ? std::vector<double>{0.0, k_heterochronous_tip_time}
                                    : std::vector<double>(k_recovery_tips, 0.0);
  state.tree = kingman_tree_under(curve, tip_times, g);
  return state;
}

auto make_recovery_run(ctpl::thread_pool& pool, const Recovery_config& c, const Recovery_state& state,
                       uint32_t seed) -> std::unique_ptr<delphy::Run> {
  auto run = std::make_unique<delphy::Run>(pool, std::mt19937{seed}, seed, state.tree, k_recovery_t_step);
  run->set_pop_model(std::make_shared<Skygrid_pop_model>(recovery_knots(), state.gamma, c.type));
  run->set_topology_moves_enabled(false);
  run->set_coal_prior_fixed_t_step(k_recovery_t_step);
  run->set_num_parts(c.parts);
  run->set_local_moves_per_global_move(60);
  run->set_mu(1e-14);  // the single site carries no mutation, so the likelihood is flat in the times
  run->set_mu_move_enabled(false);
  run->set_alpha_move_enabled(false);
  run->set_skygrid_tau(state.tau);
  run->set_skygrid_tau_move_enabled(c.free_tau);
  if (c.free_tau) {
    run->set_skygrid_tau_prior_alpha(k_tau_alpha);
    run->set_skygrid_tau_prior_beta(c.kernel_tau_beta);
  }
  run->set_skygrid_low_gamma_barrier_enabled(false);  // the barrier would distort the prior
  run->set_skygrid_inv_nbar_prior_alpha(k_recovery_nbar_alpha);
  run->set_skygrid_inv_nbar_prior_beta(c.kernel_nbar_beta);
  run->set_exact_coalescent_correction(true);
  return run;
}

struct Recovery_end_state {
  double tau = 1.0;
  double i_bar = 0.0;
  std::vector<double> delta;
};

struct Recovery_result {
  std::vector<Recovery_end_state> end_states;
  double batch_acceptance = 0.0;   // acceptance rate of the exact correction's whole-batch step
  double worst_grid_excess = 0.0;  // largest |log P_cells - log P_exact| over the end states, in nats
};

auto run_stationary_replicates(const Recovery_config& c, uint32_t seed_base, int replicates, int moves)
    -> Recovery_result {
  auto pool = ctpl::thread_pool{c.parts};
  auto result = Recovery_result{};
  auto proposed = int64_t{0};
  auto accepted = int64_t{0};
  for (auto r = 0; r != replicates; ++r) {
    auto seed = seed_base + 7919u * static_cast<uint32_t>(r);
    auto g = std::mt19937_64{seed};
    auto run = make_recovery_run(pool, c, draw_stationary_state(c, g), seed);
    run->do_mcmc_steps(60 * moves);
    const auto& pm = static_cast<const Skygrid_pop_model&>(run->pop_model());
    auto end = Recovery_end_state{.tau = run->skygrid_tau(), .i_bar = std::exp(-pm.gamma_bar()), .delta = {}};
    for (auto k = 1; k != k_recovery_knots; ++k) { end.delta.push_back(pm.gamma(k) - pm.gamma(k-1)); }
    result.end_states.push_back(std::move(end));
    proposed += run->exact_coalescent_batches_proposed();
    accepted += run->exact_coalescent_batches_accepted();
    result.worst_grid_excess = std::max(result.worst_grid_excess, std::abs(run->calc_coal_prior_grid_excess().excess));
  }
  result.batch_acceptance = proposed > 0 ? static_cast<double>(accepted) / static_cast<double>(proposed) : 0.0;
  return result;
}

struct Pit_score {
  std::string name;
  double z_mean = 0.0;    // mean of u against 1/2
  double z_spread = 0.0;  // mean of (u - 1/2)^2 against 1/12
};

auto score_end_states(const Recovery_config& c, const Recovery_result& r) -> std::vector<Pit_score> {
  auto columns = std::vector<std::pair<std::string, std::vector<double>>>{};
  auto add = [&](const std::string& name, double u) {
    auto it = std::ranges::find_if(columns, [&](const auto& col) { return col.first == name; });
    if (it == columns.end()) { columns.push_back({name, {}}); it = columns.end() - 1; }
    it->second.push_back(u);
  };
  for (const auto& e : r.end_states) {
    if (c.free_tau) { add("tau", boost::math::gamma_p(k_tau_alpha, k_tau_beta * e.tau)); }
    add("I_bar", boost::math::gamma_p(k_recovery_nbar_alpha, k_recovery_nbar_beta * e.i_bar));
    for (auto k = 0; k != std::ssize(e.delta); ++k) {
      add(absl::StrFormat("delta_%d", k + 1), 0.5 * std::erfc(-e.delta[k] * std::sqrt(e.tau) / std::sqrt(2.0)));
    }
  }
  auto scores = std::vector<Pit_score>{};
  for (const auto& [name, u] : columns) {
    auto n = static_cast<double>(std::ssize(u));
    auto mean = 0.0;
    auto spread = 0.0;
    for (auto v : u) { mean += v; spread += (v - 0.5) * (v - 0.5); }
    mean /= n;
    spread /= n;
    scores.push_back({name, (mean - 0.5) / std::sqrt(1.0 / (12.0 * n)),
                      (spread - 1.0 / 12.0) / std::sqrt(1.0 / (180.0 * n))});
  }
  return scores;
}

auto expect_recovered(const Recovery_config& c, const Recovery_result& r, const std::string& label) -> void {
  std::cerr << absl::StreamFormat("EXACT RECOVERY %s: %d replicates, batch acceptance %.3f, grid excess up to %.3g nats\n",
                                  label, std::ssize(r.end_states), r.batch_acceptance, r.worst_grid_excess);
  for (const auto& s : score_end_states(c, r)) {
    std::cerr << absl::StreamFormat("EXACT RECOVERY %s %s z(mean)=%+.2f z(spread)=%+.2f\n",
                                    label, s.name, s.z_mean, s.z_spread);
    EXPECT_LT(std::abs(s.z_mean), k_recovery_gate) << label << " " << s.name << " location";
    EXPECT_LT(std::abs(s.z_spread), k_recovery_gate) << label << " " << s.name << " dispersion";
  }
}

}  // namespace

// The headline statement, on a deliberately coarse grid (400 cells over the knot range): both
// variants recover the prior exactly.
TEST(Exact_skygrid_mcmc_test, the_marginal_of_the_skygrid_parameters_is_their_prior_on_a_coarse_grid) {
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    auto c = Recovery_config{.type = type};
    auto r = run_stationary_replicates(c, 20261118u, k_recovery_replicates, k_recovery_moves);
    expect_recovered(c, r, absl::StrFormat("coarse %s", type_name(type)));
    // The grid really is coarse: if the cell prior were the target, this is the bias it would carry
    EXPECT_GT(r.worst_grid_excess, 1e-3)
        << type_name(type) << ": the grid was not coarse enough for the comparison to mean anything";
  }
}

// The same statement with the genealogy assembled from 2 and 4 tree partitions, which is where the
// auxiliary lineage counts of `Very_scalable_coalescent_prior_part` enter the batch kernel the
// correction sits on top of.
TEST(Exact_skygrid_mcmc_test, the_marginal_is_still_the_prior_across_tree_partitions) {
  for (auto parts : {2, 4}) {
    auto c = Recovery_config{.type = Type::k_staircase, .parts = parts};
    auto r = run_stationary_replicates(c, 20261119u, k_recovery_replicates, k_recovery_moves);
    expect_recovered(c, r, absl::StrFormat("parts=%d staircase", parts));
  }
}

// The same statement with the smoothness inferred, tau ~ Gamma(3, rate 3).  This is the arm in which
// the earlier campaigns and an earlier version of this test reported a deficit in the second moments
// of the knot differences.  Independent replicates from an exact start showed no deficit: their time
// averages of delta_k^2 agree with E[1/tau] = 3/2 to within |z| < 2.2 in every arm, and to 0.5% over
// 64 chains of 200,000 global moves.  What was low was the statistic: a batch-means z of a
// heavy-tailed, right-skewed time average is biased towards negative values.  See
// doc/extended/exact_skygrid.md, section 7.
TEST(Exact_skygrid_mcmc_test, the_marginal_is_the_prior_with_an_inferred_smoothness) {
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    auto c = Recovery_config{.type = type, .free_tau = true};
    auto r = run_stationary_replicates(c, 20261122u, k_recovery_replicates, k_recovery_moves);
    expect_recovered(c, r, absl::StrFormat("inferred tau %s", type_name(type)));
  }
}

// Heterochronous tips need a different argument, and the difference is worth stating precisely
// because the obvious test is invalid.  The coalescent density here is the density of the labelled
// ranked genealogy, topology and times together, so integrating it over the times compatible with one
// topology gives that topology's probability P(R | theta).  The prior-recovery argument -- flat
// likelihood, topology fixed, marginal of theta must equal its prior -- needs that integral to be a
// constant in the population curve.  With one sampling date it is, because the isochronous Kingman
// process draws its ranked topology uniformly and independently of the demography.  With serial
// sampling it is not, so fixing the topology and asking for the prior back measures
// prior(theta) x P(R | theta), which is not the prior; the deviation is a property of the question.
// (Measured, at 12 heterochronous tips: the second moments of the most recent two differences sit at
// z = -9 to -13 while every first moment is inside the gate, for both variants.)
//
// This does NOT make every heterochronous validation invalid.  An SBC that draws the genealogy from
// the coalescent under the drawn parameters and hands the analysis the realized topology as data is
// conditioning on an observation, and Delphy's fixed-topology posterior is then exactly
// p(theta | R, sequences); that is calibrated for serial sampling too.  What is not calibrated is an
// analysis that fixes the topology by design and whose data were drawn from the conditional given it,
// which is missing the factor 1 / P(R | theta).  See doc/extended/exact_skygrid.md,
// section 1.7.
//
// Two valid statements take the place of a heterochronous prior-recovery test.  The first: with TWO
// tips there is only one labelled topology, so the integral is 1 whatever the sampling dates, and
// prior recovery is exact again.
TEST(Exact_skygrid_mcmc_test, the_marginal_is_the_prior_with_two_heterochronous_tips) {
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    auto c = Recovery_config{.type = type, .heterochronous = true};
    auto r = run_stationary_replicates(c, 20261120u, k_recovery_replicates, k_recovery_moves);
    expect_recovered(c, r, absl::StrFormat("two heterochronous tips %s", type_name(type)));
  }
}

// The second valid statement, and the one that covers the node times themselves: with a constant
// population there are no population moves at all, so the sampler's stationary distribution is
// exactly the exact Kingman density of the node times, restricted to the fixed topology's region and
// normalized over it.  That normalization is computed here by quadrature, independently of anything
// the sampler maintains, and the sampled moments are compared with it.  Serial sampling is included:
// this is where the heterochronous relaxation of the correction's support is actually tested.
namespace {

// Tree with the fixed labelled topology ((0, 1), 2) and given tip and internal times.
auto make_three_tip_tree(double s0, double s1, double s2, double t_root, double t_inner)
    -> Phylo_tree {
  auto tree = Phylo_tree{5};
  tree.ref_sequence = Real_sequence(1, Real_seq_letter::A);
  auto set_tip = [&](Node_index i, double t) {
    tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = static_cast<float>(t);
    tree.at(i).name = "S" + std::to_string(i);
  };
  set_tip(0, s0);
  set_tip(1, s1);
  set_tip(2, s2);
  // Node 3 = (0, 1); node 4 = root = (3, 2)
  tree.at(3).t = t_inner;
  tree.at(3).t_min = -std::numeric_limits<float>::max();
  tree.at(3).t_max = +std::numeric_limits<float>::max();
  tree.at(3).name = "n3";
  tree.at(3).children = {0, 1};
  tree.at(0).parent = 3;
  tree.at(1).parent = 3;
  tree.at(4).t = t_root;
  tree.at(4).t_min = -std::numeric_limits<float>::max();
  tree.at(4).t_max = +std::numeric_limits<float>::max();
  tree.at(4).name = "n4";
  tree.at(4).children = {3, 2};
  tree.at(3).parent = 4;
  tree.at(2).parent = 4;
  tree.root = 4;
  tree.at(4).parent = k_no_node;
  return tree;
}

// E[t_root] and E[t_inner] under the exact Kingman density of the topology ((0,1),2), by quadrature
// over the valid region {t_root <= t_inner <= min(s0, s1), t_root <= s2}.
struct Three_tip_moments { double e_root = 0.0; double e_inner = 0.0; };

auto three_tip_moments_by_quadrature(
    double s0, double s1, double s2, const Pop_model& pop_model, double t_lo, int n)
    -> Three_tip_moments {

  auto inner_hi = std::min(s0, s1);
  auto root_hi = std::min(inner_hi, s2);
  auto mass = 0.0L;
  auto sum_root = 0.0L;
  auto sum_inner = 0.0L;
  for (auto i = 0; i != n; ++i) {
    auto t_root = t_lo + (root_hi - t_lo) * (i + 0.5) / n;
    for (auto j = 0; j != n; ++j) {
      auto t_inner = t_root + (inner_hi - t_root) * (j + 0.5) / n;
      auto events = Coalescent_event_times{};
      events.tip_times = {s0, s1, s2};
      events.coalescence_times = {t_root, t_inner};
      auto w = std::exp(calc_exact_coalescent_log_prior(events, pop_model))
          * ((root_hi - t_lo) / n) * ((inner_hi - t_root) / n);
      mass += w;
      sum_root += w * t_root;
      sum_inner += w * t_inner;
    }
  }
  return {static_cast<double>(sum_root / mass), static_cast<double>(sum_inner / mass)};
}

}  // namespace

TEST(Exact_skygrid_mcmc_test, the_node_time_marginals_match_quadrature_with_serial_sampling) {
  auto pool = ctpl::thread_pool{1};
  constexpr auto n_pop = 400.0;

  struct Arm { std::string name; double s0, s1, s2; };
  auto arms = std::vector<Arm>{
    {"isochronous", 0.0, 0.0, 0.0},
    {"one late tip", 0.0, 0.0, -300.0},
    {"three dates", 0.0, -120.0, -400.0},
  };

  for (const auto& arm : arms) {
    auto inner_hi = std::min(arm.s0, arm.s1);
    auto root_hi = std::min(inner_hi, arm.s2);
    auto tree = make_three_tip_tree(arm.s0, arm.s1, arm.s2, root_hi - 2.0 * n_pop,
                                    inner_hi - n_pop);
    auto pop_model = std::make_shared<Const_pop_model>(n_pop);

    // The quadrature's lower bound: far enough below the region that the remaining mass is
    // negligible, since the density falls off like exp(-|t| / N)
    auto t_lo = root_hi - 40.0 * n_pop;
    auto reference = three_tip_moments_by_quadrature(arm.s0, arm.s1, arm.s2, *pop_model, t_lo, 900);

    auto t_step = n_pop / 4.0;
    auto run = std::make_unique<delphy::Run>(pool, std::mt19937{20261121u}, 20261121u, tree, t_step);
    run->set_pop_model(pop_model);
    run->set_topology_moves_enabled(false);
    run->set_coal_prior_fixed_t_step(t_step);
    run->set_local_moves_per_global_move(20);
    run->set_mu(1e-14);
    run->set_mu_move_enabled(false);
    run->set_alpha_move_enabled(false);
    run->set_subst_exchangeability_move_enabled(false);
    run->set_subst_frequencies_move_enabled(false);
    run->set_site_partition_rate_move_enabled(false);
    run->set_exact_coalescent_correction(true);

    auto roots = std::vector<double>{};
    auto inners = std::vector<double>{};
    constexpr auto k_sweeps = 200'000;
    for (auto b = 0; b != k_sweeps; ++b) {
      run->do_mcmc_steps(20);
      if (b < k_sweeps / 20) { continue; }
      roots.push_back(run->tree().at(4).t);
      inners.push_back(run->tree().at(3).t);
    }

    auto z_root = batch_mean_z(roots, reference.e_root, 40);
    auto z_inner = batch_mean_z(inners, reference.e_inner, 40);
    std::cerr << absl::StreamFormat(
        "EXACT NODE TIMES %s E[t_root] quadrature %.4f z=%+.2f | E[t_inner] quadrature %.4f z=%+.2f"
        " | batch acceptance %.3f\n",
        arm.name, reference.e_root, z_root, reference.e_inner, z_inner,
        static_cast<double>(run->exact_coalescent_batches_accepted())
            / std::max<int64_t>(1, run->exact_coalescent_batches_proposed()));
    EXPECT_LT(std::abs(z_root), 4.0) << arm.name << " E[t_root]";
    EXPECT_LT(std::abs(z_inner), 4.0) << arm.name << " E[t_inner]";
  }
}

// ---------------------------------------------------------------------------------------------
// Validation driver for the prior-recovery campaigns (not part of the CI suite).
//
// Run with --gtest_also_run_disabled_tests --gtest_filter='*DISABLED_prior_recovery_driver*' and
// the environment variables below.  Each replicate is an independent chain; with
// DELPHY_VAL_START=stationary its initial state is an exact draw from the joint target (tau, the
// knot differences, I_bar, and the genealogy given the curve), so the expected time average of
// every statistic equals its prior expectation at any chain length, burn-in or not.
//
//   DELPHY_VAL_OUT        output file (one line per replicate, then the thinned traces)
//   DELPHY_VAL_TYPE       staircase | log-linear
//   DELPHY_VAL_TAU        fixed | free  (free: tau ~ Gamma(3, rate 3), so E[delta^2] = 3/2)
//   DELPHY_VAL_START      flat | test-dispersed | stationary
//   DELPHY_VAL_REPS       number of replicates
//   DELPHY_VAL_FIRST      index of the first replicate (replicate r uses seed 1000003 * (r + 1))
//   DELPHY_VAL_MOVES      global moves per replicate (60 MCMC steps each)
//   DELPHY_VAL_THIN       thinning of the trace output (0 = no traces)
namespace {

auto env_or(const char* name, const char* fallback) -> std::string {
  auto v = std::getenv(name);
  return v != nullptr ? std::string{v} : std::string{fallback};
}

// The genealogy of `num_tips` isochronous tips at t = 0 under `pop`, by inverting the cumulative
// pair hazard exactly with `intensity_integral`.
auto kingman_tree_under(const Pop_model& pop, int num_tips, std::mt19937_64& g) -> Phylo_tree {
  auto tree = Phylo_tree{2 * num_tips - 1};
  tree.ref_sequence = Real_sequence(1, Real_seq_letter::A);
  auto active = std::vector<Node_index>{};
  for (auto i = 0; i != num_tips; ++i) {
    tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = 0.0;
    tree.at(i).name = "S" + std::to_string(i);
    active.push_back(i);
  }
  auto t = 0.0;
  auto next = num_tips;
  while (std::ssize(active) > 1) {
    auto k = static_cast<double>(std::ssize(active));
    auto target = std::exponential_distribution<double>{1.0}(g) / (k * (k - 1) / 2.0);
    // Find t_new < t with intensity_integral(t_new, t) = target
    auto width = pop.pop_at_time(t) * target;
    auto lo = t - width;
    while (pop.intensity_integral(lo, t) < target) { width *= 2.0; lo = t - width; }
    auto hi = t;
    for (auto it = 0; it != 200 && hi - lo > 1e-9 * std::max(1.0, std::abs(lo)); ++it) {
      auto mid = 0.5 * (lo + hi);
      if (pop.intensity_integral(mid, t) < target) { hi = mid; } else { lo = mid; }
    }
    t = 0.5 * (lo + hi);
    auto i = g() % active.size(); auto a = active[i]; active[i] = active.back(); active.pop_back();
    auto j = g() % active.size(); auto b = active[j]; active[j] = active.back(); active.pop_back();
    auto& node = tree.at(next);
    node.t = t;
    node.t_min = -std::numeric_limits<float>::max();
    node.t_max = +std::numeric_limits<float>::max();
    node.name = "n" + std::to_string(next);
    node.children = {a, b};
    tree.at(a).parent = next;
    tree.at(b).parent = next;
    active.push_back(next++);
  }
  tree.root = next - 1;
  tree.at(tree.root).parent = k_no_node;
  return tree;
}

}  // namespace

TEST(Exact_skygrid_validation, DISABLED_prior_recovery_driver) {
  auto out_path = env_or("DELPHY_VAL_OUT", "");
  ASSERT_FALSE(out_path.empty()) << "DELPHY_VAL_OUT is required";
  auto type_name_in = env_or("DELPHY_VAL_TYPE", "log-linear");
  auto type = type_name_in == "staircase" ? Type::k_staircase : Type::k_log_linear;
  auto free_tau = env_or("DELPHY_VAL_TAU", "free") == "free";
  auto start = env_or("DELPHY_VAL_START", "stationary");
  ASSERT_TRUE(start == "flat" || start == "test-dispersed" || start == "stationary") << start;
  auto reps = std::stoi(env_or("DELPHY_VAL_REPS", "1"));
  auto first = std::stoi(env_or("DELPHY_VAL_FIRST", "0"));
  auto moves = std::stoi(env_or("DELPHY_VAL_MOVES", "20000"));
  auto thin = std::stoi(env_or("DELPHY_VAL_THIN", "10"));
  constexpr auto tau_alpha = 3.0;
  constexpr auto tau_beta = 3.0;
  constexpr auto fixed_tau = 1.0;
  constexpr auto t_step = 5000.0;
  const auto M = k_recovery_knots - 1;

  auto out = std::ofstream{out_path};
  ASSERT_TRUE(out) << out_path;
  out << "# type=" << type_name_in << " tau=" << (free_tau ? "free" : "fixed") << " start=" << start
      << " moves=" << moves << " thin=" << thin << "\n";

  auto x = std::vector<double>{};
  for (auto k = 0; k != k_recovery_knots; ++k) { x.push_back(-2.0e6 + 2.0e6 * k / (k_recovery_knots - 1)); }

  for (auto r = first; r != first + reps; ++r) {
    auto seed = static_cast<uint32_t>(1000003u * static_cast<uint32_t>(r + 1));
    auto g = std::mt19937_64{seed};

    auto tau0 = fixed_tau;
    auto gamma = std::vector<double>(k_recovery_knots, std::log(4.0e5));
    if (start == "stationary") {
      if (free_tau) { tau0 = std::gamma_distribution<double>{tau_alpha, 1.0 / tau_beta}(g); }
      auto i_bar = std::gamma_distribution<double>{k_recovery_nbar_alpha, 1.0 / k_recovery_nbar_beta}(g);
      auto normal = std::normal_distribution<double>{0.0, 1.0 / std::sqrt(tau0)};
      auto walk = std::vector<double>(k_recovery_knots, 0.0);
      for (auto k = 1; k != k_recovery_knots; ++k) { walk[k] = walk[k-1] + normal(g); }
      auto mean = 0.0;
      for (auto v : walk) { mean += v; }
      mean /= k_recovery_knots;
      for (auto k = 0; k != k_recovery_knots; ++k) { gamma[k] = walk[k] - mean - std::log(i_bar); }
    } else if (start == "test-dispersed") {
      // The start of a_dispersed_start_removes_the_dispersion_deficit: a walk drawn with tau = 1,
      // whatever the tau prior, on the constant-population genealogy below
      auto s = std::mt19937_64{seed ^ 0x5eedu};
      auto i_bar = std::gamma_distribution<double>{k_recovery_nbar_alpha, 1.0 / k_recovery_nbar_beta}(s);
      auto normal = std::normal_distribution<double>{0.0, 1.0};
      auto walk = std::vector<double>(k_recovery_knots, 0.0);
      for (auto k = 1; k != k_recovery_knots; ++k) { walk[k] = walk[k-1] + normal(s); }
      auto mean = 0.0;
      for (auto v : walk) { mean += v; }
      mean /= k_recovery_knots;
      for (auto k = 0; k != k_recovery_knots; ++k) { gamma[k] = walk[k] - mean - std::log(i_bar); }
    }
    auto curve = std::make_shared<Skygrid_pop_model>(x, gamma, type);
    auto tree = start == "stationary" ? kingman_tree_under(*curve, k_recovery_tips, g)
                                      : make_kingman_tree(k_recovery_tips, g, 4.0e5);

    auto pool = ctpl::thread_pool{1};
    auto run = std::make_unique<delphy::Run>(pool, std::mt19937{seed}, seed, tree, t_step);
    run->set_pop_model(curve);
    run->set_topology_moves_enabled(false);
    run->set_coal_prior_fixed_t_step(t_step);
    run->set_num_parts(1);
    run->set_local_moves_per_global_move(60);
    run->set_mu(1e-14);
    run->set_mu_move_enabled(false);
    run->set_alpha_move_enabled(false);
    run->set_skygrid_tau(tau0);
    run->set_skygrid_tau_move_enabled(free_tau);
    if (free_tau) {
      run->set_skygrid_tau_prior_alpha(tau_alpha);
      run->set_skygrid_tau_prior_beta(tau_beta);
    }
    run->set_skygrid_low_gamma_barrier_enabled(false);
    run->set_skygrid_inv_nbar_prior_alpha(k_recovery_nbar_alpha);
    run->set_skygrid_inv_nbar_prior_beta(k_recovery_nbar_beta);
    run->set_exact_coalescent_correction(true);

    // Statistics: tau, 1/tau, log I_bar, I_bar, delta_k, delta_k^2
    auto sums = std::vector<double>(4 + 2 * M, 0.0);
    auto trace = std::vector<double>{};
    auto record = [&](bool into_sums, bool into_trace) {
      const auto& pm = static_cast<const Skygrid_pop_model&>(run->pop_model());
      auto v = std::vector<double>{run->skygrid_tau(), 1.0 / run->skygrid_tau(), -pm.gamma_bar(),
                                   std::exp(-pm.gamma_bar())};
      for (auto k = 1; k <= M; ++k) { v.push_back(pm.gamma(k) - pm.gamma(k-1)); }
      for (auto k = 1; k <= M; ++k) { v.push_back(std::pow(pm.gamma(k) - pm.gamma(k-1), 2)); }
      if (into_sums) { for (auto i = 0; i != std::ssize(v); ++i) { sums[i] += v[i]; } }
      if (into_trace) { trace.insert(trace.end(), v.begin(), v.end()); }
    };
    for (auto b = 0; b != moves; ++b) {
      run->do_mcmc_steps(60);
      record(true, thin > 0 && b % thin == 0);
    }
    out << "REP " << r << " " << seed << " " << run->exact_coalescent_batches_accepted() << " "
        << run->exact_coalescent_batches_proposed();
    for (auto v : sums) { out << " " << absl::StrFormat("%.17g", v / moves); }
    out << "\n";
    if (thin > 0) {
      out << "TRACE " << r;
      for (auto v : trace) { out << " " << absl::StrFormat("%.9g", v); }
      out << "\n";
    }
    out.flush();
  }
}

// Validation driver for the replicate design of the recovery tests (not part of the CI suite): it
// evaluates the statistics of `expect_recovered` for many independent seed bases, to measure their
// false-alarm rate, and with DELPHY_VAL_KERNEL_TAU_BETA or DELPHY_VAL_KERNEL_NBAR_BETA set, for a
// chain whose prior differs from the one the end states are scored against, to measure their power.
//
//   DELPHY_VAL_OUT, DELPHY_VAL_TYPE, DELPHY_VAL_TAU (fixed | free), DELPHY_VAL_PARTS,
//   DELPHY_VAL_HETEROCHRONOUS (0 | 1), DELPHY_VAL_FIRST, DELPHY_VAL_REPS (seed bases),
//   DELPHY_VAL_REPLICATES, DELPHY_VAL_MOVES, DELPHY_VAL_KERNEL_TAU_BETA, DELPHY_VAL_KERNEL_NBAR_BETA
TEST(Exact_skygrid_validation, DISABLED_recovery_statistic_driver) {
  auto out_path = env_or("DELPHY_VAL_OUT", "");
  ASSERT_FALSE(out_path.empty()) << "DELPHY_VAL_OUT is required";
  auto c = Recovery_config{};
  c.type = env_or("DELPHY_VAL_TYPE", "staircase") == "staircase" ? Type::k_staircase : Type::k_log_linear;
  c.free_tau = env_or("DELPHY_VAL_TAU", "fixed") == "free";
  c.parts = std::stoi(env_or("DELPHY_VAL_PARTS", "1"));
  c.heterochronous = env_or("DELPHY_VAL_HETEROCHRONOUS", "0") == "1";
  c.kernel_tau_beta = std::stod(env_or("DELPHY_VAL_KERNEL_TAU_BETA", "3.0"));
  c.kernel_nbar_beta = std::stod(env_or("DELPHY_VAL_KERNEL_NBAR_BETA", "1.2e6"));
  auto first = std::stoi(env_or("DELPHY_VAL_FIRST", "0"));
  auto reps = std::stoi(env_or("DELPHY_VAL_REPS", "1"));
  auto replicates = std::stoi(env_or("DELPHY_VAL_REPLICATES", std::to_string(k_recovery_replicates).c_str()));
  auto moves = std::stoi(env_or("DELPHY_VAL_MOVES", std::to_string(k_recovery_moves).c_str()));
  auto out = std::ofstream{out_path};
  ASSERT_TRUE(out) << out_path;
  for (auto b = first; b != first + reps; ++b) {
    auto seed_base = static_cast<uint32_t>(990001u * static_cast<uint32_t>(b + 1));
    auto r = run_stationary_replicates(c, seed_base, replicates, moves);
    out << "BASE " << b << " " << absl::StrFormat("%.4f", r.batch_acceptance);
    for (const auto& s : score_end_states(c, r)) {
      out << " " << s.name << " " << absl::StrFormat("%.4f %.4f", s.z_mean, s.z_spread);
    }
    out << "\n";
    out.flush();
  }
}

}  // namespace delphy
