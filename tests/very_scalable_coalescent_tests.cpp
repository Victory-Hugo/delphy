#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <cmath>
#include <optional>
#include <random>

#include "tree_partitioning.h"
#include "very_scalable_coalescent.h"

namespace delphy {

TEST(Very_scalable_coalescent_test, cells) {
  const auto t_ref = 8.0;
  const auto t_step = 2.0;

  using namespace Very_scalable_coalescent_prior;
  
  EXPECT_THAT(cell_for(7.5, t_ref, t_step), testing::Eq(0));
  EXPECT_THAT(cell_for(6.1, t_ref, t_step), testing::Eq(0));
  EXPECT_THAT(cell_for(5.9, t_ref, t_step), testing::Eq(1));
  EXPECT_THAT(cell_for(2.5, t_ref, t_step), testing::Eq(2));
  EXPECT_THAT(cell_for(3.5, t_ref, t_step), testing::Eq(2));

  EXPECT_THAT(cell_lbound(0, t_ref, t_step), testing::DoubleNear(6.0, 1e-6));
  EXPECT_THAT(cell_ubound(0, t_ref, t_step), testing::DoubleNear(8.0, 1e-6));
  EXPECT_THAT(cell_lbound(1, t_ref, t_step), testing::DoubleNear(4.0, 1e-6));
  EXPECT_THAT(cell_ubound(1, t_ref, t_step), testing::DoubleNear(6.0, 1e-6));
}

TEST(Very_scalable_coalescent_test, add_interval_single_whole_cell) {
  const auto t_ref = 8.0;
  const auto t_step = 2.0;

  using namespace Very_scalable_coalescent_prior;

  auto cells = std::vector<double>(4, 0.0);

  //                                     (6,8] (4,6] (2,4] (0,2]
  EXPECT_THAT(cells, testing::ElementsAre(0.0,  0.0,  0.0,  0.0));
  add_interval(2.0, 4.0, +2.0, cells, t_ref, t_step);
  EXPECT_THAT(cells, testing::ElementsAre(0.0,  0.0,  2.0,  0.0));
}

TEST(Very_scalable_coalescent_test, add_interval_two_whole_cells) {
  const auto t_ref = 8.0;
  const auto t_step = 2.0;

  using namespace Very_scalable_coalescent_prior;

  auto cells = std::vector<double>(4, 0.0);

  //                                     (6,8] (4,6] (2,4] (0,2]
  EXPECT_THAT(cells, testing::ElementsAre(0.0,  0.0,  0.0,  0.0));
  add_interval(2.0, 6.0, +2.0, cells, t_ref, t_step);
  EXPECT_THAT(cells, testing::ElementsAre(0.0,  2.0,  2.0,  0.0));
}

TEST(Very_scalable_coalescent_test, add_interval_single_partial_cell) {
  const auto t_ref = 8.0;
  const auto t_step = 2.0;

  using namespace Very_scalable_coalescent_prior;

  auto cells = std::vector<double>(4, 0.0);

  //                                     (6,8] (4,6] (2,4] (0,2]
  EXPECT_THAT(cells, testing::ElementsAre(0.0,  0.0,  0.0,  0.0));
  add_interval(3.0, 3.5, +2.0, cells, t_ref, t_step);
  EXPECT_THAT(cells, testing::ElementsAre(0.0,  0.0,  0.5, 0.0));
}

TEST(Very_scalable_coalescent_test, add_interval_two_partial_cells) {
  const auto t_ref = 8.0;
  const auto t_step = 2.0;

  using namespace Very_scalable_coalescent_prior;

  auto cells = std::vector<double>(4, 0.0);

  //                                     (6,8] (4,6] (2,4] (0,2]
  EXPECT_THAT(cells, testing::ElementsAre(0.0,  0.0,  0.0,  0.0));
  add_interval(3.5, 5.0, +2.0, cells, t_ref, t_step);
  EXPECT_THAT(cells, testing::ElementsAre(0.0,  1.0,  0.5,  0.0));
}

TEST(Very_scalable_coalescent_test, add_interval_two_partial_cells_and_one_whole_cell) {
  const auto t_ref = 8.0;
  const auto t_step = 2.0;

  using namespace Very_scalable_coalescent_prior;

  auto cells = std::vector<double>(4, 0.0);

  //                                     (6,8] (4,6] (2,4] (0,2]
  EXPECT_THAT(cells, testing::ElementsAre(0.0,  0.0,  0.0,  0.0));
  add_interval(3.5, 7.0, +2.0, cells, t_ref, t_step);
  EXPECT_THAT(cells, testing::ElementsAre(1.0,  2.0,  0.5,  0.0));
}

TEST(Very_scalable_coalescent_test, delta_log_prior) {
  auto n0 = 20.0;   // N_e * rho, in days
  auto g = 0.1;     // growing by a factor of `e` every 10 days

  auto pop_model = std::make_shared<Exp_pop_model>(0.0, n0, g, 0.0);

  auto num_tips = 10;
  auto num_nodes = 2*num_tips - 1;

  // Kludge: Very_scalable_coalescent_prior_part doesn't actually use the subtree topology,
  // just the node times and whether or not they are inner nodes
  auto tree = Phylo_tree{num_nodes};
  auto t_ref = 0.0;
  for (auto i = 0; i != num_tips - 1; ++i) {
    tree.at(i).children = {1,2};  // "Inner node"
    tree.at(i).t = t_ref;
  }
  for (auto i = num_tips - 1; i != num_nodes; ++i) {
    tree.at(i).children = {};     // "Tip"
    tree.at(i).t = t_ref;
  }
  
  auto bitgen = std::mt19937(12345);
  auto t_step = 0.17841;
  auto prior = Very_scalable_coalescent_prior_part{
    pop_model, tree, bitgen, true, t_ref, t_step,
    {8.5, 4.7},  // k_bar_p
    {-0.5672, +6.141897},  // k_twiddle_bar_p
    {+1.8963, -3.527041},  // k_twiddle_bar
    {pop_model->pop_integral(t_ref - 1*t_step, t_ref - 0*t_step),
     pop_model->pop_integral(t_ref - 2*t_step, t_ref - 1*t_step)
    },  // popsize_bar
    {4, 2}   // num_active_parts
  };

  auto ts = std::vector{-50.0, -40.0, -40.0, -30.0, -30.0, -30.0, -30.0, -20.0, -10.0};
  auto prng = std::mt19937{std::random_device{}()};
  std::ranges::shuffle(ts, prng);
  auto log_prior_now = prior.calc_partial_log_prior();
  for (auto i = 0; i != num_tips - 1; ++i) {
    auto delta = prior.calc_delta_partial_log_prior_after_displace_coalescence(t_ref, ts[i]);
    tree.at(i).t = ts[i];
    prior.coalescence_displaced(t_ref, ts[i]);
    auto new_log_prior = prior.calc_partial_log_prior();
    auto expected_delta = new_log_prior - log_prior_now;

    EXPECT_NEAR(delta, expected_delta, 1e-8);
    log_prior_now = new_log_prior;
  }
  for (auto i = 0; i != num_tips - 1; ++i) {
    prior.coalescence_displaced(ts[i], t_ref);
    tree.at(i).t = t_ref;
  }

  for (auto i = 0; i != num_tips - 1; ++i) {
    prior.coalescence_displaced(t_ref, -60.0);
    tree.at(i).t = t_ref;
  }
  auto i = num_tips - 1;
  for (const auto& t : {-45.0, -35.0, -30.0, -30.0, -25.0}) {
    tree.at(i).t = t;
    prior.tip_displaced(t_ref, t);
    ++i;
  }

  log_prior_now = prior.calc_partial_log_prior();
  for (auto i = 0; i != num_tips - 1; ++i) {
    auto delta = prior.calc_delta_partial_log_prior_after_displace_coalescence(t_ref, ts[i]);
    tree.at(i).t = ts[i];
    prior.coalescence_displaced(t_ref, ts[i]);
    auto new_log_prior = prior.calc_partial_log_prior();

    if (abs(new_log_prior - (log_prior_now + delta)) > 1e-8) {
      prior.coalescence_displaced(ts[i], -60.0);
      tree.at(i).t = -60.0;
      delta = prior.calc_delta_partial_log_prior_after_displace_coalescence(-60.0, ts[i]);
      prior.coalescence_displaced(-60.0, ts[i]);
      tree.at(i).t = ts[i];
      new_log_prior = prior.calc_partial_log_prior();
      EXPECT_NEAR(new_log_prior, log_prior_now + delta, 1e-8);
    }
    log_prior_now = new_log_prior;
  }
}


// Regression for an abort at human depths.  `calc_delta_partial_log_prior_on_add_interval` used to
// assert that the part of a cell an interval covers has non-negative width.  A cell bound is
// recomputed as `t_ref - t_step * cell`, while the cell index comes from `floor((t_ref - t)/t_step)`,
// and the two roundings need not agree: when `(t_ref - t)/t_step` rounds up to exactly an integer
// `k`, the index is `k` but `t_step * k` can exceed `t_ref - t`, so the width comes out negative by a
// few units in the last place.  The values below are chosen so that this happens at a tree depth of
// 122 ka: the width is -7.45e-9 days.  Observed in a free exponential run on 1,000 samples with a
// 256,000-cell grid, which aborted with `Check failed: dt_cell_end >= 0.0 (-3.72529e-09 vs. 0)`.
// The widths are now clamped to the cell instead of asserted.
TEST(Very_scalable_coalescent_test, a_cell_bound_that_rounds_past_a_node_time_does_not_abort_the_delta) {
  using namespace Very_scalable_coalescent_prior;

  auto t_ref = 0.0;
  auto t_step = 144.6;  // days; the cell width a fine grid gives on a tree of this depth
  auto k = 308'226;
  auto root_t = -std::nextafter(t_step * k, 0.0);  // 122,108 years before `t_ref`
  ASSERT_EQ(cell_for(root_t, t_ref, t_step), k);
  ASSERT_LT(cell_ubound(k, t_ref, t_step) - root_t, 0.0);  // The width that used to abort

  // Two tips at `t_ref` and one coalescence at that depth.
  auto tree = Phylo_tree{3};
  tree.ref_sequence = Real_sequence(4, Real_seq_letter::A);
  tree.root = 0;
  tree.at(0).t = root_t;
  tree.at(0).t_min = -std::numeric_limits<float>::max();
  tree.at(0).t_max = +std::numeric_limits<float>::max();
  tree.at(0).children = {1, 2};
  for (auto tip : {1, 2}) {
    tree.at(tip).parent = 0;
    tree.at(tip).t = t_ref;
    tree.at(tip).t_min = static_cast<float>(t_ref);
    tree.at(tip).t_max = static_cast<float>(t_ref);
  }

  auto pop_model = std::make_shared<Const_pop_model>(75'000 * 365.0);
  auto prngs = std::vector<std::mt19937>{std::mt19937{1u}};
  auto subtrees = std::vector<const Phylo_tree*>{&tree};
  auto parts = make_very_scalable_coalescent_prior_parts(subtrees, 0, pop_model, prngs, t_step);
  ASSERT_EQ(std::ssize(parts), 1);

  // Moving the root forward in time makes `root_t` the lower end of the crossed interval, which is
  // the end whose width used to be asserted.
  auto delta = parts[0].calc_delta_partial_log_prior_after_displace_coalescence(root_t, -100.0);
  EXPECT_TRUE(std::isfinite(delta));
}


// The auxiliary-variable construction.  In a cell with m active parts, cell-averaged lineage counts k_p (K = sum k_p),
// auxiliary variables z_p (Z = sum z_p), cell width t and averaged population size N, the part priors add up to
//
//   sum_p A_p = -(t/N) [ (m/2) sum_p k_p^2 - (m sum_p z_p k_p - Z K + K/2) ]
//             = -t K (K - 1) / (2 N)                     (the cell prior)
//               - sum_p (z_p - k_p + K/m)^2 / (2 s^2)    (log of the conditional density of z_p, without its constant;
//                                                         s^2 = N / (m t) is the variance the sampler draws with)
//               + (m t / 2N) sum_p z_p^2                 (a function of z alone)
//
// so that, at fixed z, the part priors are the cell prior times p(z | k) up to a factor that does not depend on k.
// The test builds the parts through their public constructor with chosen k and z and checks the identity for several k.
TEST(Very_scalable_coalescent_test, part_priors_add_up_to_the_cell_prior_times_the_conditional_density_of_the_auxiliary_variables) {
  auto pop_model = std::make_shared<Exp_pop_model>(0.0, 400.0, 0.002, 0.0);
  auto t_ref = 0.0;
  auto t_step = 6.5;
  auto num_cells = 3;
  auto num_parts = 3;

  auto tree = Phylo_tree{1};  // only tips: the event term of the prior vanishes and the cell terms stay
  tree.at(0).children = {};
  tree.at(0).t = t_ref;

  auto popsize_bar = std::vector<double>{};
  for (auto c = 0; c != num_cells; ++c) {
    popsize_bar.push_back(pop_model->pop_integral(t_ref - (c + 1) * t_step, t_ref - c * t_step) / t_step);
  }
  auto num_active_parts = std::vector<int>(num_cells, num_parts);

  auto prng = std::mt19937{777};
  auto uniform = [&](double lo, double hi) { return std::uniform_real_distribution<double>{lo, hi}(prng); };

  // z fixed for all trials
  auto z = std::vector<std::vector<double>>(num_parts, std::vector<double>(num_cells));
  auto z_bar = std::vector<double>(num_cells, 0.0);
  for (auto p = 0; p != num_parts; ++p) {
    for (auto c = 0; c != num_cells; ++c) { z[p][c] = uniform(-15.0, 15.0); z_bar[c] += z[p][c]; }
  }

  auto previous_difference = std::optional<double>{};
  for (auto trial = 0; trial != 6; ++trial) {
    auto k = std::vector<std::vector<double>>(num_parts, std::vector<double>(num_cells));
    for (auto p = 0; p != num_parts; ++p) {
      for (auto c = 0; c != num_cells; ++c) { k[p][c] = uniform(0.0, 40.0); }
    }

    auto sum_of_parts = 0.0;
    for (auto p = 0; p != num_parts; ++p) {
      auto part = Very_scalable_coalescent_prior_part{
          pop_model, tree, prng, p == 0, t_ref, t_step, k[p], z[p], z_bar, popsize_bar, num_active_parts};
      sum_of_parts += part.calc_partial_log_prior();
    }

    auto expected = 0.0;  // cell prior + log conditional density (no constant) + z-only term
    auto z_only = 0.0;
    for (auto c = 0; c != num_cells; ++c) {
      auto K = 0.0;
      for (auto p = 0; p != num_parts; ++p) { K += k[p][c]; }
      auto m = static_cast<double>(num_parts);
      auto s2 = popsize_bar[c] / (m * t_step);
      expected += -t_step * K * (K - 1) / (2 * popsize_bar[c]);
      for (auto p = 0; p != num_parts; ++p) {
        expected += -std::pow(z[p][c] - k[p][c] + K / m, 2) / (2 * s2);
        z_only += (m * t_step / (2 * popsize_bar[c])) * z[p][c] * z[p][c];
      }
    }
    EXPECT_NEAR(sum_of_parts, expected + z_only, 1e-8 * (1.0 + std::abs(sum_of_parts))) << "trial " << trial;
    if (previous_difference) { EXPECT_NEAR(sum_of_parts - expected, *previous_difference, 1e-8 * (1.0 + std::abs(z_only))); }
    previous_difference = sum_of_parts - expected;
  }
}

// The partition stencil is drawn from the topology alone, never from node times, so that in a run with a fixed topology
// the stencil is independent of everything the sampler changes.
TEST(Very_scalable_coalescent_test, the_partition_stencil_depends_on_the_topology_and_the_generator_only) {
  auto num_tips = 60;
  auto make_tree = [&](double time_scale, uint32_t seed) {
    auto tree = Phylo_tree{2 * num_tips - 1};
    auto active = std::vector<Node_index>{};
    for (auto i = 0; i != num_tips; ++i) { tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = 0.0; active.push_back(i); }
    auto topology_prng = std::mt19937{12345};              // same topology for every call
    auto times_prng = std::mt19937{seed};
    auto next = num_tips;
    auto t = 0.0;
    while (std::ssize(active) > 1) {
      t -= time_scale * std::exponential_distribution<double>{1.0}(times_prng);
      auto i = topology_prng() % active.size(); auto x = active[i]; active.erase(active.begin() + i);
      auto j = topology_prng() % active.size(); auto y = active[j]; active.erase(active.begin() + j);
      tree.at(next).t = t; tree.at(next).children = {x, y};
      tree.at(x).parent = next; tree.at(y).parent = next;
      active.push_back(next++);
    }
    tree.root = next - 1;
    return tree;
  };

  auto tree_a = make_tree(1.0, 1);
  auto tree_b = make_tree(37.0, 2);  // same topology, different node times
  auto num_nonempty = 0;
  for (auto parts : {2, 3, 5}) {
    for (auto seed : {1u, 2u, 3u, 4u}) {
      auto bitgen_a = std::mt19937{seed};
      auto bitgen_b = std::mt19937{seed};
      auto stencil_a = generate_random_partition_stencil(tree_a, parts, bitgen_a);
      auto stencil_b = generate_random_partition_stencil(tree_b, parts, bitgen_b);
      ASSERT_EQ(std::ssize(stencil_a), std::ssize(stencil_b));
      if (not stencil_a.empty()) { ++num_nonempty; }
      for (auto i = 0; i != std::ssize(stencil_a); ++i) { EXPECT_EQ(stencil_a[i].cut_point, stencil_b[i].cut_point); }
    }
  }
  EXPECT_GT(num_nonempty, 3);  // the comparison is not vacuous
}

}  // namespace delphy
