#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "ctpl_stl.h"

#include "api.h"
#include "calibration.h"
#include "evo_subst_model.h"
#include "phylo_tree_calc.h"
#include "pop_model.h"
#include "run.h"
#include "site_partition.h"

// Every module of this extension is validated on its own: the substitution models and the site-rate
// Gamma, the site partitions and their relative rates, the internal-node calibrations, the exact
// Kingman correction, the observation semantics, and the Run/Subrun tree partitioning with its
// repartitioning and its checkpoints.  These tests are about what none of those covers: that the
// whole stack is **one** posterior.  The increment the moves maintain must still equal a full
// recomputation of every term when all of them are switched on together, through accepted moves,
// rejected moves and rollbacks, across repartitionings, on one and on several threads, and a
// checkpoint taken in that state must restore the same posterior.

namespace delphy {

namespace {

constexpr auto k_n0_days = 400'000.0;   // a deep, human-like coalescent timescale, in days
constexpr auto k_mu_site_day = 2.0e-6;  // substitutions per site per day

// A Kingman tree over `num_tips` isochronous tips with Jukes-Cantor mutations at rate `mu`, and a
// leading region made missing in two of the tips, which is what gives the observation semantics
// something to do in the combined run.
auto simulate_tree(int num_tips, Site_index num_sites, uint64_t seed, double mu,
                   Site_index missing_sites) -> Phylo_tree {
  auto g = std::mt19937_64{seed};
  auto tree = Phylo_tree{2 * num_tips - 1};
  tree.ref_sequence = Real_sequence(num_sites, Real_seq_letter::A);
  for (auto i = 0; i != num_tips; ++i) {
    tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = 0.0;
    tree.at(i).name = "S" + std::to_string(i);
  }

  auto active = std::vector<Node_index>{};
  for (auto i = 0; i != num_tips; ++i) { active.push_back(i); }
  auto age = 0.0;
  auto next = num_tips;
  while (std::ssize(active) > 1) {
    auto k = static_cast<double>(std::ssize(active));
    age += std::exponential_distribution<double>{k * (k - 1) / 2.0 / k_n0_days}(g);
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

  auto stack = std::vector<std::pair<Node_index, std::vector<int>>>{
      {tree.root, std::vector<int>(num_sites, 0)}};
  while (not stack.empty()) {
    auto [p, state] = std::move(stack.back()); stack.pop_back();
    for (auto c : tree.at(p).children) {
      auto t0 = tree.at(p).t, t1 = tree.at(c).t;
      auto st = state;
      auto times = std::vector<std::pair<double, Site_index>>{};
      for (auto l = Site_index{0}; l != num_sites; ++l) {
        auto m = std::poisson_distribution<int>{mu * (t1 - t0)}(g);
        for (auto i = 0; i != m; ++i) {
          times.push_back({t0 + (t1 - t0) * std::uniform_real_distribution<double>{0.0, 1.0}(g), l});
        }
      }
      std::ranges::sort(times);
      for (auto [tm, site] : times) {
        auto to = (st[site] + 1 + static_cast<int>(g() % 3)) % 4;
        tree.at(c).mutations.push_back(Mutation{static_cast<Real_seq_letter>(st[site]), site,
                                                static_cast<Real_seq_letter>(to), tm});
        st[site] = to;
      }
      stack.push_back({c, std::move(st)});
    }
  }

  if (missing_sites > 0) {
    for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
      std::erase_if(tree.at(node).mutations,
                    [&](const Mutation& m) { return m.site < missing_sites; });
    }
    tree.at(0).missations.intervals.insert(Site_interval{0, missing_sites});
    tree.at(1).missations.intervals.insert(Site_interval{0, missing_sites / 2});
  }
  return tree;
}

// The tip names below a node, which is how a calibration names its target: a stable taxon set, never
// a node index.
auto tip_names_below(const Phylo_tree& tree, Node_index node) -> std::vector<std::string> {
  auto out = std::vector<std::string>{};
  auto stack = std::vector<Node_index>{node};
  while (not stack.empty()) {
    auto n = stack.back(); stack.pop_back();
    if (tree.at(n).is_tip()) { out.push_back(tree.at(n).name); continue; }
    for (auto c : tree.at(n).children) { stack.push_back(c); }
  }
  std::ranges::sort(out);
  return out;
}

// The deepest internal node with at least `min_tips` tips below it, excluding the root.
auto some_inner_clade(const Phylo_tree& tree, int min_tips) -> Node_index {
  auto best = k_no_node;
  auto best_tips = std::numeric_limits<int>::max();
  for (auto n = Node_index{0}; n != std::ssize(tree); ++n) {
    if (tree.at(n).is_tip() || n == tree.root) { continue; }
    auto tips = static_cast<int>(std::ssize(tip_names_below(tree, n)));
    if (tips >= min_tips && tips < best_tips) { best = n; best_tips = tips; }
  }
  return best;
}

auto two_block_partition(Site_index num_sites) -> Site_partition_config {
  auto half = num_sites / 2;
  auto text = absl::StrFormat("version 1\n\npartition first\n  sites 1-%d\nend\n\n"
                              "partition second\n  sites %d-%d\nend\n",
                              half, half + 1, num_sites);
  auto is = std::istringstream{text};
  return read_site_partition_config(is);
}

// Two calibrations stated as correct measurement likelihoods: the dispersion is a fixed fraction of
// the coalescent timescale and of the clade's tip count, never a function of the realized age, and
// the target is named by its taxon set.
auto two_calibrations(const Phylo_tree& tree, Node_index inner) -> Calibration_set {
  auto root_age_years = -tree.at_root().t / 365.0;
  auto inner_age_years = -tree.at(inner).t / 365.0;
  auto names = tip_names_below(tree, inner);
  auto text = absl::StrFormat(
      "version 1\n"
      "calibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution lognormal\n  median %.12g\n  sigma 0.2\nend\n"
      "calibration inner\n  taxa %s\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n",
      root_age_years, absl::StrJoin(names, " "), inner_age_years,
      std::max(inner_age_years / 4.0, 1.0));
  auto is = std::istringstream{text};
  return read_calibration_config(is);
}

auto gtr_model(double mu) -> Subst_model {
  auto model = Subst_model{};
  model.mu = mu;
  model.grouping = Exchangeability_grouping::from_code("012345");
  model.class_rate = {1.4, 4.2, 0.7, 1.1, 5.1, 1.0};
  model.frequency_mode = Frequency_mode::estimated;
  model.pi_a[Real_seq_letter::A] = 0.31;
  model.pi_a[Real_seq_letter::C] = 0.29;
  model.pi_a[Real_seq_letter::G] = 0.14;
  model.pi_a[Real_seq_letter::T] = 0.26;
  model.validate();
  return model;
}

// Everything at once: a general reversible matrix with its exchangeabilities and frequencies
// sampled, continuous Gamma site rates with the shape sampled, two site partitions with a sampled
// relative rate, two internal calibrations, the exact Kingman correction on a pinned grid, missing
// data in two tips, and the topology fixed, which is the regime the calibrations require.
auto make_joint_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, int parts,
                    int moves_per_global_move, const Calibration_set& calibrations,
                    bool free_population = false)
    -> std::shared_ptr<Run> {
  constexpr auto t_step_days = 200.0 * 365.0;
  auto run = std::make_shared<delphy::Run>(pool, std::mt19937{seed}, seed, tree, t_step_days);
  run->set_topology_moves_enabled(false);
  run->set_coal_prior_fixed_t_step(t_step_days);
  run->set_num_parts(parts);
  run->set_local_moves_per_global_move(moves_per_global_move);

  run->set_subst_model(gtr_model(k_mu_site_day));
  run->set_subst_exchangeability_move_enabled(true);
  run->set_subst_frequencies_move_enabled(true);

  run->set_mu(k_mu_site_day);
  run->set_mu_prior_alpha(1.0);
  run->set_mu_prior_beta(0.0);

  run->set_alpha(1.0);
  run->set_alpha_move_enabled(true);

  run->set_site_partition(two_block_partition(tree.num_sites()));
  run->set_site_partition_rates({0.6, 1.4});
  run->set_site_partition_rate_move_enabled(true);
  run->set_site_partition_rate_prior_concentration(2.0);

  run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, k_n0_days, 0.0, 1.0));
  run->set_final_pop_size_move_enabled(free_population);
  run->set_pop_growth_rate_move_enabled(free_population);
  if (free_population) {
    // A proper prior on n0 and a growth rate that cannot be negative, on the scale of this tree
    run->set_pop_inv_n0_prior_alpha(4.0);
    run->set_pop_inv_n0_prior_beta(1.2e6);
    run->set_pop_g_prior_mu(0.0);
    run->set_pop_g_prior_scale(1.0e-6);
    run->set_pop_g_min(0.0);
    run->set_pop_g_max(3.0e-6);
    run->set_pop_g_move_window(1.0e-6);
    run->set_pop_moves_per_global_move(5);
  }
  run->set_exact_coalescent_correction(true);

  run->set_calibrations(calibrations);
  return run;
}

// Every term of the posterior, recomputed from the current state with no use of the increments the
// moves maintain.
struct Full_posterior {
  double log_G{};
  double log_coalescent{};
  double log_other{};
  double log_calibration{};

  auto total() const -> double { return log_G + log_coalescent + log_other + log_calibration; }
};

auto recompute(const delphy::Run& run) -> Full_posterior {
  // With the exact correction on, the coalescent term of the target is the exact Kingman density and
  // not the cell prior on the grid, which is what `calc_cur_log_coalescent_target` returns and what
  // `log_coalescent_prior` carries incrementally.
  return {run.calc_cur_log_G(), run.calc_cur_log_coalescent_target(),
          run.calc_cur_log_other_priors(), run.calc_cur_log_calibration_prior()};
}

auto expect_consistent(const delphy::Run& run, std::string_view where) -> void {
  auto full = recompute(run);
  auto tol = [](double x) { return 1e-6 * std::max(1.0, std::abs(x)); };
  EXPECT_NEAR(run.log_G(), full.log_G, tol(full.log_G)) << where << ": log_G";
  EXPECT_NEAR(run.log_coalescent_prior(), full.log_coalescent, tol(full.log_coalescent))
      << where << ": coalescent prior";
  EXPECT_NEAR(run.log_other_priors(), full.log_other, tol(full.log_other))
      << where << ": other priors";
  EXPECT_NEAR(run.log_calibration_prior(), full.log_calibration, tol(full.log_calibration))
      << where << ": calibration prior";
  EXPECT_NEAR(run.log_calibration_prior_from_parts(), run.log_calibration_prior(),
              tol(run.log_calibration_prior()))
      << where << ": the partition-local calibration bookkeeping disagrees with the global value";
  EXPECT_NEAR(run.log_posterior(), full.total(), tol(full.total())) << where << ": log_posterior";
  EXPECT_TRUE(std::isfinite(run.log_posterior())) << where << ": the posterior left its support";
  assert_phylo_tree_integrity(run.tree(), true);
}

class Joint_model_mcmc_test : public testing::TestWithParam<int> {
 protected:
  ctpl::thread_pool pool_{4};
  auto parts() const -> int { return GetParam(); }
};

}  // namespace

// The whole stack is one posterior: with every model switched on together, the incremental value the
// moves carry equals a full recomputation of each of its four terms, batch after batch, on one part
// and on four.  Forced repartitioning every 200 local moves makes the calibrated nodes change part
// and alternate between movable and frozen while this holds.
TEST_P(Joint_model_mcmc_test, the_incremental_posterior_matches_a_full_recomputation) {
  constexpr auto num_sites = Site_index{240};
  auto tree = simulate_tree(16, num_sites, 90'210, k_mu_site_day, 24);
  auto inner = some_inner_clade(tree, 4);
  ASSERT_NE(inner, k_no_node);
  auto calibrations = two_calibrations(tree, inner);

  auto run = make_joint_run(pool_, tree, 31, parts(), 200, calibrations);
  expect_consistent(*run, "before any step");

  auto root_t_values = std::vector<double>{};
  auto alpha_values = std::vector<double>{};
  auto rate_values = std::vector<double>{};
  for (auto batch = 0; batch != 10; ++batch) {
    run->do_mcmc_steps(20'000);
    expect_consistent(*run, absl::StrFormat("batch %d", batch));
    root_t_values.push_back(run->tree().at_root().t);
    alpha_values.push_back(run->alpha());
    rate_values.push_back(run->site_partition_rates()[0]);
  }

  // The test proves nothing unless every sampled direction actually moved.
  auto moved = [](const std::vector<double>& v) {
    for (auto x : v) { if (x != v.front()) { return true; } }
    return false;
  };
  EXPECT_TRUE(moved(root_t_values)) << "the node times never moved";
  EXPECT_TRUE(moved(alpha_values)) << "the Gamma shape never moved";
  EXPECT_TRUE(moved(rate_values)) << "the partition relative rate never moved";
  EXPECT_GT(run->exact_coalescent_batches_proposed(), 0);
  EXPECT_GT(run->exact_coalescent_batches_accepted(), 0);

  // Rejections and rollbacks are part of what is being checked, so at least one move kind must have
  // rejected something while the bookkeeping stayed consistent.
  auto rejected = int64_t{0};
  for (const auto& counts : run->move_counts()) { rejected += counts.proposed - counts.accepted; }
  EXPECT_GT(rejected, 0) << "no move was ever rejected, so no rollback was exercised";
}

// The population moves belong to the same posterior: with n0 and g sampled next to the calibrations,
// the site partitions, the substitution model and the exact correction, the increments their accepted
// and rejected proposals leave behind equal a full recomputation, on one part and on four.
TEST_P(Joint_model_mcmc_test, the_population_moves_keep_the_incremental_posterior_consistent) {
  constexpr auto num_sites = Site_index{240};
  auto tree = simulate_tree(16, num_sites, 90'210, k_mu_site_day, 24);
  auto inner = some_inner_clade(tree, 4);
  ASSERT_NE(inner, k_no_node);
  auto calibrations = two_calibrations(tree, inner);

  auto run = make_joint_run(pool_, tree, 37, parts(), 200, calibrations, true);
  expect_consistent(*run, "before any step");

  auto n0_values = std::vector<double>{};
  auto g_values = std::vector<double>{};
  for (auto batch = 0; batch != 10; ++batch) {
    run->do_mcmc_steps(20'000);
    expect_consistent(*run, absl::StrFormat("batch %d", batch));
    const auto& pm = static_cast<const Exp_pop_model&>(run->pop_model());
    n0_values.push_back(pm.pop_at_t0());
    g_values.push_back(pm.growth_rate());
    EXPECT_GE(pm.growth_rate(), 0.0);
    EXPECT_LE(pm.growth_rate(), 3.0e-6);
  }
  auto moved = [](const std::vector<double>& v) {
    for (auto x : v) { if (x != v.front()) { return true; } }
    return false;
  };
  EXPECT_TRUE(moved(n0_values)) << "n0 never moved";
  EXPECT_TRUE(moved(g_values)) << "g never moved";
  EXPECT_GT(run->pop_size_moves_accepted(), 0);
  EXPECT_LT(run->pop_size_moves_accepted(), run->pop_size_moves_proposed());
  EXPECT_GT(run->pop_growth_rate_moves_accepted(), 0);
  EXPECT_LT(run->pop_growth_rate_moves_accepted(), run->pop_growth_rate_moves_proposed());
}

INSTANTIATE_TEST_SUITE_P(Parts, Joint_model_mcmc_test, testing::Values(1, 4),
                         [](const testing::TestParamInfo<int>& info) {
                           return std::to_string(info.param) + "parts";
                         });

// A repartitioning is a change of the Run/Subrun tree partition and not of the site partition or of
// any model: the posterior must come back bit for bit, and the run must go on consistently with a
// different number of parts.
TEST(Joint_model_run_test, repartitioning_leaves_the_joint_posterior_unchanged) {
  auto pool = ctpl::thread_pool{4};
  constexpr auto num_sites = Site_index{200};
  auto tree = simulate_tree(14, num_sites, 55'301, k_mu_site_day, 20);
  auto inner = some_inner_clade(tree, 4);
  ASSERT_NE(inner, k_no_node);
  auto calibrations = two_calibrations(tree, inner);

  auto run = make_joint_run(pool, tree, 77, 2, -1, calibrations);
  run->do_mcmc_steps(40'000);
  expect_consistent(*run, "before repartitioning");
  auto before = run->log_posterior();

  for (auto parts : {1, 4, 2, 8}) {
    run->set_num_parts(parts);
    EXPECT_EQ(run->num_parts(), parts);
    EXPECT_NEAR(run->log_posterior(), before, 1e-9 * std::max(1.0, std::abs(before)))
        << "repartitioning to " << parts << " parts changed the posterior";
    expect_consistent(*run, absl::StrFormat("after repartitioning to %d parts", parts));
    run->do_mcmc_steps(20'000);
    expect_consistent(*run, absl::StrFormat("after 20,000 steps on %d parts", parts));
    before = run->log_posterior();
  }
}

// A checkpoint taken with the whole stack on must restore the same posterior, the same models and
// the same calibrations, and the resumed run must keep sampling consistently.
TEST(Joint_model_run_test, a_checkpoint_round_trip_keeps_the_joint_posterior) {
  auto pool = ctpl::thread_pool{2};
  constexpr auto num_sites = Site_index{200};
  auto tree = simulate_tree(12, num_sites, 11'203, k_mu_site_day, 16);
  auto inner = some_inner_clade(tree, 4);
  ASSERT_NE(inner, k_no_node);
  auto calibrations = two_calibrations(tree, inner);

  auto run = make_joint_run(pool, tree, 91, 2, -1, calibrations);
  run->do_mcmc_steps(40'000);
  expect_consistent(*run, "before the checkpoint");

  auto params = run_to_api_params(*run);
  auto resumed = std::make_shared<delphy::Run>(pool, std::mt19937{5}, 5, run->tree(),
                                       run->coalescent_prior_t_step());
  resumed->set_num_parts(2);
  resumed->set_coal_prior_fixed_t_step(run->coal_prior_fixed_t_step());
  apply_api_params_to_run(params.data(), *resumed);

  EXPECT_EQ(resumed->subst_model().grouping.code(), run->subst_model().grouping.code());
  EXPECT_EQ(resumed->site_partition().num_partitions(), run->site_partition().num_partitions());
  EXPECT_EQ(std::ssize(resumed->calibrations().calibrations),
            std::ssize(run->calibrations().calibrations));
  EXPECT_NEAR(resumed->alpha(), run->alpha(), 1e-12);
  for (auto p = 0; p != std::ssize(run->site_partition_rates()); ++p) {
    EXPECT_NEAR(resumed->site_partition_rates()[p], run->site_partition_rates()[p], 1e-12);
  }
  EXPECT_NEAR(resumed->log_G(), run->log_G(), 1e-9 * std::max(1.0, std::abs(run->log_G())));
  EXPECT_NEAR(resumed->log_calibration_prior(), run->log_calibration_prior(),
              1e-9 * std::max(1.0, std::abs(run->log_calibration_prior())));

  // The grid and the exact correction are not carried by the parameter blob, which is what
  // `Restore_test` records; re-supplying them puts the resumed run on the same target.
  resumed->set_topology_moves_enabled(false);
  resumed->set_exact_coalescent_correction(true);
  resumed->set_alpha_move_enabled(true);
  resumed->set_subst_exchangeability_move_enabled(true);
  resumed->set_subst_frequencies_move_enabled(true);
  resumed->set_site_partition_rate_move_enabled(true);
  expect_consistent(*resumed, "after the restore");

  resumed->do_mcmc_steps(40'000);
  expect_consistent(*resumed, "after resuming");
}

}  // namespace delphy
