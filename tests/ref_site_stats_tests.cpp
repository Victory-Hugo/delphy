#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>

#include "ctpl_stl.h"

#include "phylo_tree.h"
#include "phylo_tree_calc.h"
#include "pop_model.h"
#include "run.h"

namespace delphy {

namespace {

constexpr auto k_n0 = 1000.0;  // days
constexpr auto k_sites = 150;
constexpr auto k_mu = 2e-4;    // per site per day

// Kingman tree, all tips at t = 0, Jukes-Cantor mutations along the branches, including the branch above the root, so
// that the first normalization of the root rewrites the reference sequence.
auto make_tree(int num_tips, uint64_t seed) -> Phylo_tree {
  auto g = std::mt19937_64{seed};
  auto tree = Phylo_tree{2 * num_tips - 1};
  tree.ref_sequence = Real_sequence(k_sites, Real_seq_letter::A);
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
    age += std::exponential_distribution<double>{k * (k - 1) / 2.0 / k_n0}(g);
    auto i = g() % active.size(); auto x = active[i]; active[i] = active.back(); active.pop_back();
    auto j = g() % active.size(); auto y = active[j]; active[j] = active.back(); active.pop_back();
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

  auto root_state = std::vector<int>(k_sites, 0);
  for (auto site = 0; site != 10; ++site) {  // mutations above the root, folded into the reference by normalize_root
    auto to = 1 + static_cast<int>(g() % 3);
    tree.at(tree.root).mutations.push_back(
        Mutation{Real_seq_letter::A, site, static_cast<Real_seq_letter>(to), tree.at(tree.root).t - 1.0 - site});
    root_state[site] = to;
  }
  auto stack = std::vector<std::pair<Node_index, std::vector<int>>>{{tree.root, root_state}};
  while (not stack.empty()) {
    auto [p, state] = std::move(stack.back()); stack.pop_back();
    for (auto c : tree.at(p).children) {
      auto t0 = tree.at(p).t, t1 = tree.at(c).t;
      auto st = state;
      auto m = std::poisson_distribution<int>{k_sites * k_mu * (t1 - t0)}(g);
      auto times = std::vector<double>(m);
      for (auto& x : times) { x = t0 + (t1 - t0) * std::uniform_real_distribution<double>{0.0, 1.0}(g); }
      std::ranges::sort(times);
      for (auto tm : times) {
        auto site = static_cast<int>(g() % k_sites);
        auto to = (st[site] + 1 + static_cast<int>(g() % 3)) % 4;
        tree.at(c).mutations.push_back(
            Mutation{static_cast<Real_seq_letter>(st[site]), site, static_cast<Real_seq_letter>(to), tm});
        st[site] = to;
      }
      stack.push_back({c, std::move(st)});
    }
  }
  return tree;
}

// The grid is deliberately coarse: the cell prior then differs enough from the exact density that whole batches are
// rejected and `tree_` is restored from its snapshot, which is one of the paths the cache has to follow.
auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, bool alpha) -> std::unique_ptr<Run> {
  auto t_step = k_n0;
  auto run = std::make_unique<Run>(pool, std::mt19937{seed}, seed, tree, t_step);
  run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, k_n0, 0.0, 0.0));
  run->set_mu(k_mu);
  run->set_alpha_move_enabled(alpha);
  run->set_local_moves_per_global_move(150);
  run->set_coal_prior_fixed_t_step(t_step);
  run->set_exact_coalescent_correction(true);  // whole-batch rejections restore `tree_` from a snapshot
  return run;
}

// Everything Run derives from the reference sequence through `ref_site_stats_`, compared with a recomputation from
// scratch.  Reads go through a const reference, because a non-const `tree()` retires the cache and would make the
// comparison vacuous.
auto expect_cached_statistics_exact(const Run& run) -> void {
  const auto& tree = run.tree();
  const auto& evo = run.evo();
  EXPECT_EQ(run.calc_cur_Ttwiddle_beta_a(), calc_Ttwiddle_beta_a(tree, evo));
  EXPECT_EQ(run.state_frequencies_of_ref_sequence(), run.calc_cur_state_frequencies_of_ref_sequence());
  auto scale = std::max(1.0, std::abs(run.log_G()));
  EXPECT_LT(std::abs(run.log_G() - run.calc_cur_log_G()), 1e-9 * scale);
}

auto expect_same_state(const Run& a, const Run& b) -> void {
  EXPECT_EQ(a.log_posterior(), b.log_posterior());
  EXPECT_EQ(a.log_G(), b.log_G());
  const auto& ta = a.tree();
  const auto& tb = b.tree();
  ASSERT_EQ(ta.root, tb.root);
  EXPECT_EQ(ta.ref_sequence, tb.ref_sequence);
  for (auto node = 0; node != std::ssize(ta); ++node) {
    EXPECT_EQ(ta.at(node).t, tb.at(node).t) << "node " << node;
    EXPECT_EQ(ta.at(node).parent, tb.at(node).parent) << "node " << node;
    ASSERT_EQ(std::ssize(ta.at(node).mutations), std::ssize(tb.at(node).mutations)) << "node " << node;
    auto ma = ta.at(node).mutations.begin();
    for (const auto& m : tb.at(node).mutations) {
      EXPECT_TRUE(ma->site == m.site && ma->from == m.from && ma->to == m.to && ma->t == m.t) << "node " << node;
      ++ma;
    }
  }
}

// Two chains with the same seeds, one of which retires every cached statistic of the reference sequence (and the
// table of cumulative site rates that the parts share) before each chunk.  Retiring them draws no random numbers, so
// the chains must stay identical bit for bit if the cached values equal a recomputation.
auto expect_caching_does_not_change_the_chain(int num_parts, bool exact, bool alpha) -> void {
  auto pool = ctpl::thread_pool{std::max(1, num_parts - 1)};
  auto tree = make_tree(14, 31);
  auto make = [&]() {
    auto run = std::make_unique<Run>(pool, std::mt19937{41}, 41, tree, k_n0);
    run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, k_n0, 0.0, 0.0));
    run->set_mu(k_mu);
    run->set_alpha_move_enabled(alpha);
    run->set_num_parts(num_parts);
    run->set_local_moves_per_global_move(150);
    run->set_coal_prior_fixed_t_step(k_n0);
    if (exact) { run->set_exact_coalescent_correction(true); }
    return run;
  };
  auto cached = make();
  auto recomputed = make();
  for (auto chunk = 0; chunk != 40; ++chunk) {
    cached->do_mcmc_steps(70);  // not a multiple of the batch length, so batches are cut at chunk boundaries
    recomputed->invalidate_derived_quantities();
    recomputed->do_mcmc_steps(70);
    expect_same_state(*cached, *recomputed);
    expect_cached_statistics_exact(*cached);
  }
}

}  // namespace

TEST(Ref_site_stats_test, shared_tables_do_not_change_the_chain_single_part_exact_correction) {
  expect_caching_does_not_change_the_chain(1, /*exact=*/true, /*alpha=*/true);
}

TEST(Ref_site_stats_test, shared_tables_do_not_change_the_chain_several_parts) {
  expect_caching_does_not_change_the_chain(3, /*exact=*/false, /*alpha=*/true);
}

TEST(Ref_site_stats_test, cached_statistics_equal_a_recomputation_through_moves_rejections_and_reroots) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(12, 17);
  auto run = make_run(pool, tree, 23, /*alpha=*/true);  // the alpha move redraws every nu_l in each global step
  for (auto chunk = 0; chunk != 60; ++chunk) {
    run->do_mcmc_steps(300);  // ends with normalize_root, which rewrites the reference sequence
    expect_cached_statistics_exact(*run);
  }
  EXPECT_GT(run->exact_coalescent_batches_proposed(), run->exact_coalescent_batches_accepted());  // some batches were rolled back
}

TEST(Ref_site_stats_test, cached_statistics_equal_a_recomputation_with_unit_site_rates) {
  // Without site-rate heterogeneity every nu_l is 1: the nu-weighted counts are then taken from the state counts, and
  // the state counts follow each rewrite of the reference sequence incrementally
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(12, 19);
  auto run = make_run(pool, tree, 29, /*alpha=*/false);
  for (auto chunk = 0; chunk != 60; ++chunk) {
    run->do_mcmc_steps(300);
    expect_cached_statistics_exact(*run);
  }
}

TEST(Ref_site_stats_test, setting_the_site_rates_retires_the_cached_statistics) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(10, 5);
  auto run = make_run(pool, tree, 7, /*alpha=*/false);
  run->do_mcmc_steps(300);
  expect_cached_statistics_exact(*run);

  // Only the site rates change below, so the prior must move by exactly their part of it,
  //   (alpha - 1) sum_l log nu_l - alpha sum_l nu_l.
  auto sums = [](const std::vector<double>& nu) {
    auto s = 0.0, s_log = 0.0;
    for (auto x : nu) { s += x; s_log += std::log(x); }
    return std::pair{s, s_log};
  };
  auto [s_old, s_log_old] = sums(run->nu());
  auto prior_old = run->calc_cur_log_other_priors();

  auto g = std::mt19937_64{11};
  auto nu = std::vector<double>(run->nu().size());
  for (auto& x : nu) { x = std::gamma_distribution<double>{2.0, 0.5}(g); }
  run->set_nu(nu);
  expect_cached_statistics_exact(*run);

  auto [s_new, s_log_new] = sums(nu);
  auto alpha = run->alpha();
  auto expected_change = (alpha - 1) * (s_log_new - s_log_old) - alpha * (s_new - s_old);
  auto prior_new = run->calc_cur_log_other_priors();
  EXPECT_NEAR(prior_new - prior_old, expected_change, 1e-9 * std::max(1.0, std::abs(prior_old)));

  run->do_mcmc_steps(300);
  expect_cached_statistics_exact(*run);
}

TEST(Ref_site_stats_test, editing_the_reference_through_the_tree_accessor_retires_the_cached_statistics) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(10, 9);
  auto run = make_run(pool, tree, 3, /*alpha=*/false);
  run->do_mcmc_steps(300);
  expect_cached_statistics_exact(*run);

  // An outside edit that changes the reference sequence and nothing the data constrain: put a mutation from -> to above
  // the root at one site and undo it at the very start of both branches below the root, then fold the root mutation into
  // the reference as normalize_root does.  Every tip sequence stays the same; the reference changes at that site.
  {
    auto& t = run->tree();
    ASSERT_TRUE(t.at_root().mutations.empty());
    auto site = 3;
    auto from = t.ref_sequence[site];
    auto to = static_cast<Real_seq_letter>((static_cast<int>(from) + 1) % 4);
    t.at_root().mutations.push_back(Mutation{from, site, to, t.at_root().t - 1.0});
    for (auto c : t.at_root().children) {
      auto t_m = t.at_root().t + 1e-6 * (t.at(c).t - t.at_root().t);
      auto& muts = t.at(c).mutations;
      muts.insert(muts.begin(), Mutation{to, site, from, t_m});
    }
    rereference_to_root_sequence(t);
    ASSERT_EQ(t.ref_sequence[site], to);
  }
  run->tree_modified();
  expect_cached_statistics_exact(*run);
  run->do_mcmc_steps(300);
  expect_cached_statistics_exact(*run);
}

}  // namespace delphy
