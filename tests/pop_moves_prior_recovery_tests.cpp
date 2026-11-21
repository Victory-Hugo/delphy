#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "absl/strings/str_format.h"
#include "ctpl_stl.h"

#include "coalescent_grid.h"
#include "phylo_tree_calc.h"
#include "pop_model.h"
#include "run.h"

// The exponential population moves have no test of their target distribution of their own.  With no
// information in the sequences, the topology fixed, the exact Kingman density as the coalescent term
// and a growth rate that cannot be negative (so that the coalescent process always completes and the
// tree density integrates to one whatever the population parameters are), the marginal distribution
// of (n0, g) under the full posterior is exactly their prior.  A wrong acceptance ratio in either
// move, such as a missing Hastings factor of the scale proposal, shifts that marginal.

namespace delphy {

namespace {

constexpr auto k_num_tips = 12;
constexpr auto k_n0_alpha = 4.0;
constexpr auto k_n0_beta = 1.2e6;             // days; prior mean beta / (alpha - 1) = 4e5 days
constexpr auto k_g_scale = 1.0e-6;            // e-foldings per day
constexpr auto k_g_max = 3.0e-6;

auto kingman_tree(uint64_t seed) -> Phylo_tree {
  constexpr auto n0 = 4.0e5;
  auto rng = std::mt19937_64{seed};
  auto tree = Phylo_tree{2 * k_num_tips - 1};
  tree.ref_sequence = Real_sequence(1, Real_seq_letter::A);
  for (auto i = 0; i != k_num_tips; ++i) {
    tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = 0.0;
    tree.at(i).name = "S" + std::to_string(i);
  }
  auto active = std::vector<Node_index>{};
  for (auto i = 0; i != k_num_tips; ++i) { active.push_back(i); }
  auto age = 0.0;
  auto next = Node_index{k_num_tips};
  while (std::ssize(active) > 1) {
    auto k = static_cast<double>(std::ssize(active));
    age += std::exponential_distribution<double>{k * (k - 1) / 2.0 / n0}(rng);
    auto i = rng() % active.size(); auto x = active[i]; active[i] = active.back(); active.pop_back();
    auto j = rng() % active.size(); auto y = active[j]; active[j] = active.back(); active.pop_back();
    auto& node = tree.at(next);
    node.t = -age;
    node.t_min = -std::numeric_limits<float>::max();
    node.t_max = +std::numeric_limits<float>::max();
    node.name = "n" + std::to_string(next);
    node.children = {x, y};
    tree.at(x).parent = next;
    tree.at(y).parent = next;
    active.push_back(next++);
  }
  tree.root = next - 1;
  tree.at(tree.root).parent = k_no_node;
  return tree;
}

auto make_prior_only_run(ctpl::thread_pool& pool, uint32_t seed, bool exact, double min_pop = 1.0)
    -> std::shared_ptr<Run> {
  constexpr auto t_step_days = 200.0 * 365.0;
  auto run = std::make_shared<Run>(pool, std::mt19937{seed}, seed, kingman_tree(seed), t_step_days);
  run->set_topology_moves_enabled(false);
  run->set_coal_prior_fixed_t_step(t_step_days);
  run->set_local_moves_per_global_move(60);
  run->set_pop_moves_per_global_move(10);
  run->set_mu(1e-14);  // the single site carries no mutation, so its likelihood is flat in the times
  run->set_mu_move_enabled(false);
  run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, 4.0e5, 1.0e-6, min_pop));
  run->set_pop_inv_n0_prior_alpha(k_n0_alpha);
  run->set_pop_inv_n0_prior_beta(k_n0_beta);
  run->set_pop_g_prior_mu(0.0);
  run->set_pop_g_prior_scale(k_g_scale);
  run->set_pop_g_min(0.0);
  run->set_pop_g_max(k_g_max);
  run->set_pop_g_move_window(1.0e-6);
  if (exact) { run->set_exact_coalescent_correction(true); }
  return run;
}

// z-score of the mean of `x` against `expected`, with the standard error taken from batch means.
auto batch_mean_z(const std::vector<double>& x, double expected, int num_batches = 20) -> double {
  auto size = static_cast<int>(x.size()) / num_batches;
  auto means = std::vector<double>{};
  for (auto b = 0; b != num_batches; ++b) {
    auto s = 0.0;
    for (auto i = 0; i != size; ++i) { s += x[b * size + i]; }
    means.push_back(s / size);
  }
  auto grand = 0.0;
  for (auto m : means) { grand += m; }
  grand /= num_batches;
  auto var = 0.0;
  for (auto m : means) { var += (m - grand) * (m - grand); }
  var /= (num_batches - 1);
  return (grand - expected) / std::sqrt(var / num_batches);
}

auto expect_prior_recovered(uint32_t seed, double min_pop) -> void {
  auto pool = ctpl::thread_pool{2};
  auto run = make_prior_only_run(pool, seed, true, min_pop);

  auto n0s = std::vector<double>{};
  auto log_n0s = std::vector<double>{};
  auto gs = std::vector<double>{};
  for (auto b = 0; b != 12'000; ++b) {
    run->do_mcmc_steps(60);
    if (b < 500) { continue; }  // burn-in
    const auto& pm = static_cast<const Exp_pop_model&>(run->pop_model());
    n0s.push_back(pm.pop_at_t0());
    log_n0s.push_back(std::log(pm.pop_at_t0()));
    gs.push_back(pm.growth_rate());
  }

  // Prior moments: Inverse-Gamma(alpha, beta) for n0; Exponential(scale) truncated to [0, g_max] for g.
  constexpr auto euler_gamma = 0.57721566490153286;
  auto psi_alpha = 1.0 + 1.0 / 2 + 1.0 / 3 - euler_gamma;  // digamma(4)
  auto expected_log_n0 = std::log(k_n0_beta) - psi_alpha;
  auto expected_n0 = k_n0_beta / (k_n0_alpha - 1.0);
  auto tail = std::exp(-k_g_max / k_g_scale);
  auto expected_g = k_g_scale - k_g_max * tail / (1.0 - tail);

  EXPECT_LT(std::abs(batch_mean_z(log_n0s, expected_log_n0)), 4.0) << "mean of log n0";
  EXPECT_LT(std::abs(batch_mean_z(n0s, expected_n0)), 4.0) << "mean of n0";
  EXPECT_LT(std::abs(batch_mean_z(gs, expected_g)), 4.0) << "mean of g";

  // Both moves must have moved and been rejected, otherwise nothing was tested.
  EXPECT_GT(run->pop_size_moves_accepted(), run->pop_size_moves_proposed() / 20);
  EXPECT_LT(run->pop_size_moves_accepted(), run->pop_size_moves_proposed());
  EXPECT_GT(run->pop_growth_rate_moves_accepted(), run->pop_growth_rate_moves_proposed() / 20);
  EXPECT_LT(run->pop_growth_rate_moves_accepted(), run->pop_growth_rate_moves_proposed());
  EXPECT_GT(run->pop_growth_rate_moves_out_of_bounds(), 0);
}

}  // namespace

TEST(Pop_moves_prior_recovery_test, the_marginal_of_n0_and_g_is_their_prior_under_the_exact_density) {
  expect_prior_recovered(20261116, 1.0);
}

// The generator and the analysis of the simulation-based calibration use a population floor of 100 years.
// With n0 around a thousand years and g up to 3e-6 per day the floor is reached in a good share of the
// genealogies, so this is the case in which the floor enters the exact density and the population moves.
TEST(Pop_moves_prior_recovery_test, the_marginal_is_still_the_prior_when_the_population_floor_is_active) {
  expect_prior_recovered(20261119, 100.0 * 365.0);
}

TEST(Pop_moves_prior_recovery_test, a_window_far_wider_than_the_posterior_is_still_correct_and_rarely_accepted) {
  auto pool = ctpl::thread_pool{2};
  auto run = make_prior_only_run(pool, 20261117, true);
  // The window is the knob that decides the efficiency and not the target: widening it to 1e4 times
  // the support leaves the sampled distribution alone and drives the acceptance of g to almost zero.
  run->set_pop_g_move_window(k_g_max * 1.0e4);
  for (auto b = 0; b != 2'000; ++b) { run->do_mcmc_steps(60); }
  auto accepted = static_cast<double>(run->pop_growth_rate_moves_accepted());
  auto proposed = static_cast<double>(run->pop_growth_rate_moves_proposed());
  EXPECT_LT(accepted / proposed, 1e-3);
  EXPECT_GT(run->pop_growth_rate_moves_out_of_bounds(), 0.99 * proposed);
}

// The coalescent density must stay a number, never plus infinity, for every growth rate a proposal
// can reach: the default window of one e-folding per year is a hundred thousand times the posterior
// width of a human-scale growth rate, so the chain does visit such values.
TEST(Pop_moves_prior_recovery_test, extreme_growth_rates_keep_the_coalescent_density_finite_until_the_exponential_overflows) {
  auto tree = kingman_tree(20261118);
  auto events = coalescent_event_times_of(tree);
  auto t_ref = 0.0;
  auto span = 0.0;  // days from the oldest coalescence to the tips
  for (auto t : events.coalescence_times) { span = std::max(span, t_ref - t); }
  for (auto g : {-1.0, -1e-1, -1e-2, -1e-3, -1e-4, -1e-6, 0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0}) {
    for (auto min_pop : {1.0, 100.0 * 365.0}) {
      for (auto n0 : {1.0e3, 4.0e5, 1.0e9}) {
        auto pm2 = Exp_pop_model{0.0, n0, g, min_pop};
        auto exact = calc_exact_coalescent_log_prior(events, pm2);
        auto cells = calc_cell_coalescent_log_prior(events, pm2, t_ref, 200.0 * 365.0);
        // The cell prior stays a number everywhere.  The exact density overflows for a growth rate that
        // is strongly negative (population size growing exponentially into the past, e^{|g| span} past
        // the double range) and returns NaN there, which the Metropolis-Hastings tests read as a
        // rejection, the same decision as the true density, which is hopelessly small.  Above that
        // edge it has to be finite.
        EXPECT_FALSE(std::isnan(cells)) << "cells, g=" << g << " n0=" << n0 << " min_pop=" << min_pop;
        EXPECT_NE(cells, std::numeric_limits<double>::infinity()) << "g=" << g << " n0=" << n0;
        EXPECT_NE(exact, std::numeric_limits<double>::infinity()) << "g=" << g << " n0=" << n0;
        if (g * span >= -300.0) {
          EXPECT_TRUE(std::isfinite(exact)) << "exact, g=" << g << " n0=" << n0 << " min_pop=" << min_pop;
        }
      }
    }
  }
}

}  // namespace delphy
