#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <sstream>

#include "ctpl_stl.h"

#include "api.h"
#include "api_generated.h"
#include "calibration.h"
#include "estd.h"
#include "phylo_tree.h"
#include "pop_model.h"
#include "run.h"

namespace delphy {

namespace {

constexpr auto k_n0 = 1000.0;  // days
constexpr auto k_sites = 120;
constexpr auto k_mu = 2e-4;    // per site per day

// Kingman tree, all tips at t = 0, Jukes-Cantor mutations along the branches (rate k_mu per site per day)
auto make_tree_with_mutations(int num_tips, uint64_t seed) -> Phylo_tree {
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

  // Mutations: site-wise current state, Jukes-Cantor
  auto stack = std::vector<std::pair<Node_index, std::vector<int>>>{{tree.root, std::vector<int>(k_sites, 0)}};
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
        tree.at(c).mutations.push_back(Mutation{static_cast<Real_seq_letter>(st[site]), site, static_cast<Real_seq_letter>(to), tm});
        st[site] = to;
      }
      stack.push_back({c, std::move(st)});
    }
  }
  return tree;
}

// An exact-correction run with the move available; no global parameter move changes the setup of a test
auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, int parts = 1, int batch = 200) -> std::unique_ptr<Run> {
  auto t_step = 1.0 * k_n0 / 20.0;
  auto run = std::make_unique<delphy::Run>(pool, std::mt19937{seed}, seed, tree, t_step);
  run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, k_n0, 5e-4, 0.0));
  run->set_mu(k_mu);
  run->set_hky_kappa(1.0);
  run->set_hky_pi({0.25, 0.25, 0.25, 0.25});
  run->set_mu_prior_alpha(20.0);
  run->set_mu_prior_beta(20.0 / k_mu);
  run->set_pop_g_prior_mu(0.0);
  run->set_pop_g_prior_scale(2e-4);
  run->set_pop_g_min(0.0);
  run->set_pop_g_max(2e-3);
  run->set_alpha_move_enabled(false);
  run->set_num_parts(parts);
  run->set_local_moves_per_global_move(batch);
  run->set_topology_moves_enabled(false);
  run->set_coal_prior_fixed_t_step(t_step);
  run->set_exact_coalescent_correction(true);
  return run;
}

auto tips_under(const Phylo_tree& tree, Node_index node) -> std::vector<std::string> {
  auto out = std::vector<std::string>{};
  auto stack = std::vector<Node_index>{node};
  while (not stack.empty()) {
    auto v = stack.back(); stack.pop_back();
    if (tree.at(v).is_tip()) { out.push_back(tree.at(v).name); continue; }
    for (auto c : tree.at(v).children) { stack.push_back(c); }
  }
  std::ranges::sort(out);
  return out;
}

// Calibration on the root and on the larger child of the root: lognormal priors around the true ages
auto make_calibrations(const Phylo_tree& tree) -> Calibration_set {
  auto cfg = std::ostringstream{};
  cfg.precision(17);
  cfg << "version 1\n\n";
  auto add = [&](const std::string& name, Node_index node) {
    auto age_years = -tree.at(node).t / 365.0;
    cfg << "calibration " << name << "\n  taxa ";
    if (node == tree.root) { cfg << "all"; } else { for (const auto& s : tips_under(tree, node)) { cfg << s << " "; } }
    cfg << "\n  age_units years_bp\n  age_reference_year 2020\n  distribution lognormal\n  median " << age_years * 1.1
        << "\n  sigma 0.25\n  oldest " << age_years * 6 << "\n  youngest " << age_years * 0.2 << "\nend\n";
  };
  add("root", tree.root);
  auto kids = tree.at(tree.root).children;
  if (not tree.at(kids[0]).is_tip()) { add("clade", kids[0]); }
  auto is = std::istringstream{cfg.str()};
  return read_calibration_config(is);
}

auto max_time_diff(const Phylo_tree& a, const Phylo_tree& b) -> double {
  auto d = 0.0;
  for (Node_index i = 0; i != std::ssize(a); ++i) {
    d = std::max(d, std::abs(a.at(i).t - b.at(i).t));
    if (a.at(i).mutations.size() != b.at(i).mutations.size()) { return std::numeric_limits<double>::infinity(); }
    for (size_t k = 0; k != a.at(i).mutations.size(); ++k) { d = std::max(d, std::abs(a.at(i).mutations[k].t - b.at(i).mutations[k].t)); }
  }
  return d;
}

auto exactly_equal(const Phylo_tree& a, const Phylo_tree& b) -> bool {
  if (a.root != b.root || std::ssize(a) != std::ssize(b)) { return false; }
  for (Node_index i = 0; i != std::ssize(a); ++i) { if (not (a.at(i) == b.at(i))) { return false; } }
  return true;
}

auto scaled_run(ctpl::thread_pool& pool, const Run& source, double u, uint32_t seed) -> std::unique_ptr<Run> {
  auto tree = source.tree();
  auto s = std::exp(u);
  for (Node_index v = 0; v != std::ssize(tree); ++v) {
    if (not tree.at(v).is_tip()) { tree.at(v).t *= s; }
    for (auto& m : tree.at(v).mutations) { m.t *= s; }
  }
  auto run = make_run(pool, tree, seed);
  const auto& pm = static_cast<const Exp_pop_model&>(source.pop_model());
  run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, pm.pop_at_t0() * s, pm.growth_rate() / s, 0.0));
  run->set_mu(source.mu() / s);
  run->set_hky_kappa(source.hky_kappa());
  run->set_hky_pi({source.hky_pi()[Real_seq_letter::A], source.hky_pi()[Real_seq_letter::C], source.hky_pi()[Real_seq_letter::G], source.hky_pi()[Real_seq_letter::T]});
  if (source.has_calibrations()) { run->set_calibrations(source.calibrations()); }
  return run;
}

}  // namespace

class Global_scale_move_test : public ::testing::TestWithParam<bool> {  // parameter: with calibrations
 protected:
  ctpl::thread_pool pool_{1};
};

TEST_P(Global_scale_move_test, ratio_matches_fresh_run_and_identities_hold) {
  auto tree = make_tree_with_mutations(12, 17);
  auto run = make_run(pool_, tree, 5);
  if (GetParam()) { run->set_calibrations(make_calibrations(tree)); }
  run->do_mcmc_steps(200);  // normalizes the root and builds the parts
  run->set_global_scale_move_enabled(true);

  auto n_mut = 0L;
  for (Node_index v = 0; v != std::ssize(run->tree()); ++v) { n_mut += std::ssize(run->tree().at(v).mutations); }
  auto n_internal = 11;

  for (auto u : {-0.3, -0.05, 0.02, 0.2}) {
    auto before = make_run(pool_, run->tree(), 6);
    before->set_pop_model(std::make_shared<Exp_pop_model>(static_cast<const Exp_pop_model&>(run->pop_model())));
    before->set_mu(run->mu());
    if (GetParam()) { before->set_calibrations(run->calibrations()); }
    auto target_before = before->log_posterior();
    auto fresh = scaled_run(pool_, *before, u, 7);
    auto expected = (fresh->log_posterior() - target_before) + (n_internal + n_mut - 1) * u;

    auto moved = scaled_run(pool_, *before, 0.0, 8);  // copy of `before`, independent object
    moved->set_global_scale_move_enabled(true);
    auto result = moved->global_scale_move_with(u, 0.0);
    EXPECT_TRUE(result.accepted);
    EXPECT_NEAR(result.log_alpha, expected, 1e-6 * (1.0 + std::abs(expected))) << "u=" << u;
    EXPECT_NEAR(result.delta_log_G, -n_mut * u, 1e-6 * (1.0 + n_mut * std::abs(u)));
    EXPECT_NEAR(result.delta_log_coalescent, -n_internal * u, 1e-6);
    EXPECT_LT(std::abs(result.scale_free_residual), 1e-9);
    EXPECT_NEAR(moved->log_posterior(), fresh->log_posterior(), 1e-6 * (1.0 + std::abs(fresh->log_posterior())));
    EXPECT_LT(max_time_diff(moved->tree(), fresh->tree()), 1e-7 * k_n0 * 10);
    EXPECT_NEAR(moved->mu(), fresh->mu(), 1e-12);

    // the inverse move restores the state and has the opposite ratio
    auto back = moved->global_scale_move_with(-u, 0.0);
    EXPECT_NEAR(back.log_alpha, -result.log_alpha, 1e-6 * (1.0 + std::abs(result.log_alpha)));
    EXPECT_LT(max_time_diff(moved->tree(), before->tree()), 1e-7 * k_n0 * 10);
    EXPECT_NEAR(moved->log_posterior(), target_before, 1e-6 * (1.0 + std::abs(target_before)));
  }
}

// Independent reconstruction of the acceptance ratio: every term is recomputed from the explicit state with closed-form
// formulas written here (Jukes-Cantor likelihood of the explicit mutation history, Kingman density, priors), without any function
// of Run, and the sum is compared with the ratio that Run reports.
namespace {

auto total_branch_length(const Phylo_tree& tree) -> double {
  auto total = 0.0;
  for (Node_index v = 0; v != std::ssize(tree); ++v) {
    if (tree.at(v).parent != k_no_node) { total += tree.at(v).t - tree.at(tree.at(v).parent).t; }
  }
  return total;
}

auto count_mutations(const Phylo_tree& tree) -> long {
  auto n = 0L;
  for (Node_index v = 0; v != std::ssize(tree); ++v) { n += std::ssize(tree.at(v).mutations); }
  return n;
}

// log of the Kingman density of the inner node ages (up to a constant): N(a) = n0 exp(-g a), tips at age 0
auto independent_log_kingman(const Phylo_tree& tree, double n0, double g, int num_tips) -> double {
  auto ages = std::vector<double>{};
  for (Node_index v = 0; v != std::ssize(tree); ++v) { if (not tree.at(v).is_tip()) { ages.push_back(-tree.at(v).t); } }
  std::ranges::sort(ages);
  auto lambda = [&](double a) { return std::abs(g * a) < 1e-12 ? a / n0 : std::expm1(g * a) / (g * n0); };
  auto result = 0.0;
  auto m = static_cast<int>(ages.size());
  for (auto j = 1; j <= m; ++j) { result -= (num_tips - j) * lambda(ages[j - 1]); result += g * ages[j - 1] - std::log(n0); }
  return result;
}

auto independent_log_calibration(const Phylo_tree& tree, const Calibration_set& cals, const std::vector<Node_index>& nodes) -> double {
  auto result = 0.0;
  for (size_t i = 0; i != nodes.size(); ++i) {
    const auto& prior = cals.calibrations[i].prior;
    auto age_years = -tree.at(nodes[i]).t / 365.0;
    if (age_years < prior.youngest_age || age_years > prior.oldest_age) { return -std::numeric_limits<double>::infinity(); }
    auto z = std::log(age_years / prior.median) / prior.sigma;
    result += -std::log(age_years) - 0.5 * z * z;
  }
  return result;
}

}  // namespace

TEST_P(Global_scale_move_test, ratio_equals_an_independent_reconstruction_of_the_full_posterior_difference) {
  auto tree = make_tree_with_mutations(14, 61);
  auto run = make_run(pool_, tree, 5);
  if (GetParam()) { run->set_calibrations(make_calibrations(tree)); }
  run->do_mcmc_steps(200);
  run->set_global_scale_move_enabled(true);
  const auto num_tips = 14;
  auto before = run->tree();
  const auto& pm = static_cast<const Exp_pop_model&>(run->pop_model());
  auto mu0 = run->mu(), n00 = pm.pop_at_t0(), g0 = pm.growth_rate();
  auto m = count_mutations(before);
  auto cal_nodes = run->calibration_nodes();
  auto cals = run->calibrations();

  for (auto u : {-0.4, -0.1, 0.03, 0.17, 0.35}) {
    auto probe = make_run(pool_, before, 6);
    probe->set_pop_model(std::make_shared<Exp_pop_model>(0.0, n00, g0, 0.0));
    probe->set_mu(mu0);
    if (GetParam()) { probe->set_calibrations(cals); }
    probe->set_global_scale_move_enabled(true);
    auto result = probe->global_scale_move_with(u, 0.0);
    ASSERT_TRUE(result.accepted || std::isinf(result.log_alpha));
    if (not result.accepted) { continue; }
    const auto& after = probe->tree();
    const auto& pm1 = static_cast<const Exp_pop_model&>(probe->pop_model());
    auto mu1 = probe->mu(), n01 = pm1.pop_at_t0(), g1 = pm1.growth_rate();
    constexpr auto L = static_cast<double>(k_sites);

    // Jukes-Cantor likelihood of the explicit history with kappa = 1, uniform frequencies: rate mu per site, factor mu/3 per event
    auto d_log_G = m * std::log(mu1 / mu0) - L * (mu1 * total_branch_length(after) - mu0 * total_branch_length(before));
    auto d_log_kingman = independent_log_kingman(after, n01, g1, num_tips) - independent_log_kingman(before, n00, g0, num_tips);
    constexpr auto a_mu = 20.0, a_n0 = 0.0, g_scale = 2e-4;
    auto b_mu = 20.0 / k_mu;
    auto d_priors = (a_mu - 1) * std::log(mu1 / mu0) - b_mu * (mu1 - mu0)
        - (a_n0 + 1) * std::log(n01 / n00)
        - (std::abs(g1 - 0.0) - std::abs(g0 - 0.0)) / g_scale;
    auto d_cal = GetParam() ? independent_log_calibration(after, cals, cal_nodes) - independent_log_calibration(before, cals, cal_nodes) : 0.0;
    auto n_internal = num_tips - 1;
    auto expected = d_log_G + d_log_kingman + d_priors + d_cal + (n_internal + m - 1) * u;
    EXPECT_NEAR(result.log_alpha, expected, 1e-7 * (1.0 + std::abs(expected))) << "u=" << u << " calibrations=" << GetParam();
    EXPECT_NEAR(result.delta_log_G, d_log_G, 1e-7 * (1.0 + std::abs(d_log_G)));
    EXPECT_NEAR(result.delta_log_coalescent, d_log_kingman, 1e-7 * (1.0 + std::abs(d_log_kingman)));
    EXPECT_NEAR(result.delta_log_calibration, d_cal, 1e-8 * (1.0 + std::abs(d_cal)));
  }
}

TEST_P(Global_scale_move_test, rejection_restores_the_state_exactly) {
  auto tree = make_tree_with_mutations(12, 19);
  auto run = make_run(pool_, tree, 9);
  if (GetParam()) { run->set_calibrations(make_calibrations(tree)); }
  run->do_mcmc_steps(200);
  run->set_global_scale_move_enabled(true);
  auto tree_before = run->tree();
  auto mu_before = run->mu();
  const auto& pm = static_cast<const Exp_pop_model&>(run->pop_model());
  auto n0_before = pm.pop_at_t0(), g_before = pm.growth_rate();
  auto lp_before = run->log_posterior();
  auto rejected = 0;
  for (auto u : {0.4, -0.4, 0.8, -0.8}) {
    auto result = run->global_scale_move_with(u, 2.0);  // draw 2: accepted only if log_alpha >= 0
    if (not result.accepted) {
      ++rejected;
      EXPECT_TRUE(exactly_equal(run->tree(), tree_before));
      EXPECT_EQ(run->mu(), mu_before);
      const auto& pm2 = static_cast<const Exp_pop_model&>(run->pop_model());
      EXPECT_EQ(pm2.pop_at_t0(), n0_before);
      EXPECT_EQ(pm2.growth_rate(), g_before);
      EXPECT_NEAR(run->log_posterior(), lp_before, 1e-9 * (1.0 + std::abs(lp_before)));
    } else {
      break;  // the state moved; the remaining proposals would compare against a different state
    }
  }
  EXPECT_GE(rejected, 1);
}

INSTANTIATE_TEST_SUITE_P(Calibrations, Global_scale_move_test, ::testing::Values(false, true));

TEST(Global_scale_move, extreme_and_illegal_proposals_leave_a_valid_state) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree_with_mutations(10, 23);
  auto run = make_run(pool, tree, 3);
  run->do_mcmc_steps(200);
  run->set_global_scale_move_enabled(true);
  auto tree_before = run->tree();
  auto lp_before = run->log_posterior();
  for (auto u : {40.0, -40.0, 700.0, -700.0}) {
    auto result = run->global_scale_move_with(u, 0.5);
    EXPECT_FALSE(result.accepted) << u;
    EXPECT_TRUE(result.log_alpha == -std::numeric_limits<double>::infinity() || result.log_alpha < -50.0) << u;
    EXPECT_TRUE(exactly_equal(run->tree(), tree_before)) << u;
    EXPECT_NEAR(run->log_posterior(), lp_before, 1e-9 * (1.0 + std::abs(lp_before)));
  }
  EXPECT_THROW(run->global_scale_move_with(std::numeric_limits<double>::quiet_NaN(), 0.5), std::invalid_argument);
  EXPECT_THROW(run->global_scale_move_with(std::numeric_limits<double>::infinity(), 0.5), std::invalid_argument);
  EXPECT_THROW(run->set_global_scale_move_window(0.0), std::invalid_argument);
  EXPECT_THROW(run->set_global_scale_move_window(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
  EXPECT_THROW(run->set_global_scale_moves_per_global_move(0), std::invalid_argument);
}

TEST(Global_scale_move, unsupported_configurations_are_refused) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree_with_mutations(10, 29);

  {  // no exact correction
    auto run = std::make_unique<delphy::Run>(pool, std::mt19937{1}, 1, tree, 50.0);
    run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, k_n0, 0.0, 0.0));
    run->set_topology_moves_enabled(false);
    run->set_coal_prior_fixed_t_step(50.0);
    EXPECT_THROW(run->set_global_scale_move_enabled(true), std::invalid_argument);
  }
  {  // reference time of the population model away from the sampling date
    auto run = make_run(pool, tree, 1);
    run->set_pop_model(std::make_shared<Exp_pop_model>(5.0, k_n0, 0.0, 0.0));
    EXPECT_THROW(run->set_global_scale_move_enabled(true), std::invalid_argument);
  }
  {  // Skygrid: the exact correction now supports it, but the joint time-scale move does not -- its
     // map transforms (n0, g) of an exponential curve, and no corresponding map of a Skygrid curve's
     // knot values has been derived, so the move is refused with or without the correction
    auto skygrid = [] {
      return std::make_shared<Skygrid_pop_model>(std::vector<double>{-3000.0, -1000.0}, std::vector<double>{6.0, 6.5},
                                                 Skygrid_pop_model::Type::k_staircase);
    };
    auto with_exact = make_run(pool, tree, 1);
    with_exact->set_pop_model(skygrid());
    EXPECT_THROW(with_exact->set_global_scale_move_enabled(true), std::invalid_argument);
    auto without_exact = std::make_unique<delphy::Run>(pool, std::mt19937{1}, 1, tree, 50.0);
    without_exact->set_pop_model(skygrid());
    without_exact->set_topology_moves_enabled(false);
    EXPECT_THROW(without_exact->set_global_scale_move_enabled(true), std::invalid_argument);
  }
  {  // topology moves cannot be turned on once the move is enabled together with the exact correction
    auto run = make_run(pool, tree, 1);
    run->set_global_scale_move_enabled(true);
    EXPECT_THROW(run->set_topology_moves_enabled(true), std::invalid_argument);
  }
  {  // samples at different dates: the exact correction accepts them, but the joint time-scale move
     // does not -- it measures every age back from one common sampling date
    auto dated = tree;
    dated.at(0).t = dated.at(0).t_min = dated.at(0).t_max = -30.0;
    std::erase_if(dated.at(0).mutations, [](const Mutation& m) { return m.t > -30.0; });  // keep the branch consistent with the earlier tip time
    auto run = std::make_unique<delphy::Run>(pool, std::mt19937{1}, 1, dated, 50.0);
    run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, k_n0, 0.0, 0.0));
    run->set_topology_moves_enabled(false);
    run->set_coal_prior_fixed_t_step(50.0);
    EXPECT_NO_THROW(run->set_exact_coalescent_correction(true));
    EXPECT_THROW(run->set_global_scale_move_enabled(true), std::invalid_argument);
  }
  {  // disabling the exact correction while the move is on is refused at the next proposal
    auto run = make_run(pool, tree, 1);
    run->set_global_scale_move_enabled(true);
    run->set_exact_coalescent_correction(false);
    EXPECT_THROW(run->global_scale_move_with(0.1, 0.5), std::invalid_argument);
  }
}

TEST(Global_scale_move, off_by_default_draws_nothing_and_changes_nothing) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree_with_mutations(12, 31);
  auto a = make_run(pool, tree, 4);
  auto b = make_run(pool, tree, 4);
  b->set_global_scale_move_enabled(true);
  b->set_global_scale_move_enabled(false);  // switched on and off again: must leave no trace
  for (auto i = 0; i != 6; ++i) { a->do_mcmc_steps(200); b->do_mcmc_steps(200); }
  EXPECT_TRUE(exactly_equal(a->tree(), b->tree()));
  EXPECT_EQ(a->mu(), b->mu());
  EXPECT_EQ(a->global_scale_moves_proposed(), 0);
  EXPECT_EQ(b->global_scale_moves_proposed(), 0);
}

TEST(Global_scale_move, scheduled_moves_run_with_the_window_and_count_and_keep_a_consistent_state) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree_with_mutations(12, 37);
  auto run = make_run(pool, tree, 4);
  run->set_calibrations(make_calibrations(tree));
  run->set_global_scale_move_window(0.15);
  run->set_global_scale_moves_per_global_move(4);
  run->set_global_scale_move_enabled(true);
  for (auto i = 0; i != 30; ++i) { run->do_mcmc_steps(200); }
  EXPECT_EQ(run->global_scale_moves_proposed(), 4 * 30);
  EXPECT_GT(run->global_scale_moves_accepted(), 0);
  auto fresh = scaled_run(pool, *run, 0.0, 99);
  EXPECT_NEAR(fresh->log_posterior(), run->log_posterior(), 1e-8 * (1.0 + std::abs(run->log_posterior())));
}

// ---------------------------------------------------------------------------------------------------------------------------
// Serialization and resume

namespace {

auto round_trip(ctpl::thread_pool& pool, const Run& source, uint32_t seed, int parts) -> std::unique_ptr<Run> {
  auto params = run_to_api_params(source);
  auto target = std::make_unique<delphy::Run>(pool, std::mt19937{seed}, seed, source.tree(), source.coalescent_prior_t_step());
  target->set_num_parts(parts);
  target->set_topology_moves_enabled(false);
  apply_api_params_to_run(params.data(), *target);
  return target;
}

}  // namespace

TEST(Global_scale_move, configuration_round_trips_through_the_api_params) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree_with_mutations(12, 41);
  auto run = make_run(pool, tree, 4);
  run->set_calibrations(make_calibrations(tree));
  run->set_global_scale_move_window(0.07);
  run->set_global_scale_moves_per_global_move(3);
  run->set_global_scale_move_enabled(true);
  run->do_mcmc_steps(200);

  auto restored = round_trip(pool, *run, 77, 1);
  EXPECT_TRUE(restored->exact_coalescent_correction());
  EXPECT_EQ(restored->coal_prior_fixed_t_step(), run->coal_prior_fixed_t_step());
  EXPECT_TRUE(restored->global_scale_move_enabled());
  EXPECT_EQ(restored->global_scale_move_window(), 0.07);
  EXPECT_EQ(restored->global_scale_moves_per_global_move(), 3);
  EXPECT_NEAR(restored->log_posterior(), run->log_posterior(), 1e-9 * (1.0 + std::abs(run->log_posterior())));
}

TEST(Global_scale_move, files_of_runs_without_the_features_carry_no_new_fields) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree_with_mutations(12, 43);
  auto run = std::make_unique<delphy::Run>(pool, std::mt19937{4}, 4, tree, 50.0);
  run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, k_n0, 0.0, 0.0));
  run->set_topology_moves_enabled(false);
  run->do_mcmc_steps(200);
  auto params = run_to_api_params(*run);
  auto api_params = flatbuffers::GetSizePrefixedRoot<api::Params>(params.data());
  EXPECT_FALSE(api_params->exact_coalescent_correction());
  EXPECT_FALSE(api_params->global_scale_move_enabled());
  EXPECT_EQ(api_params->prng_state(), nullptr);
  EXPECT_EQ(api_params->sub_prng_states(), nullptr);
  EXPECT_EQ(api_params->partition_stencil_cut_points(), nullptr);

  // an old file (no new fields) loaded into a run that has the features on switches them off
  auto other = make_run(pool, tree, 5);
  other->set_global_scale_move_enabled(true);
  apply_api_params_to_run(params.data(), *other);
  EXPECT_FALSE(other->global_scale_move_enabled());
  EXPECT_FALSE(other->exact_coalescent_correction());
}

TEST(Global_scale_move, inconsistent_saved_configurations_are_refused) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree_with_mutations(12, 47);
  auto run = make_run(pool, tree, 4);
  run->set_global_scale_move_enabled(true);
  run->do_mcmc_steps(200);
  auto params = run_to_api_params(*run);

  // a file with the move on but the exact correction off
  auto bad = std::vector<uint8_t>(params.data(), params.data() + params.size());
  auto* mutable_params = flatbuffers::GetMutableSizePrefixedRoot<api::Params>(bad.data());
  ASSERT_TRUE(reinterpret_cast<flatbuffers::Table*>(mutable_params)->SetField<uint8_t>(api::Params::VT_EXACT_COALESCENT_CORRECTION, 0, 0));
  auto target = make_run(pool, tree, 5);
  EXPECT_THROW(apply_api_params_to_run(bad.data(), *target), std::invalid_argument);
}

class Resume_test : public ::testing::TestWithParam<int> {  // parameter: number of parts
 protected:
  ctpl::thread_pool pool_{4};
};

TEST_P(Resume_test, a_resumed_run_continues_exactly_like_the_uninterrupted_run) {
  auto parts = GetParam();
  auto tree = make_tree_with_mutations(14, 53);
  auto setup = [&](uint32_t seed) {
    auto r = make_run(pool_, tree, seed, parts, 300);
    r->set_calibrations(make_calibrations(tree));
    r->set_global_scale_move_window(0.2);
    r->set_global_scale_moves_per_global_move(2);
    r->set_global_scale_move_enabled(true);
    return r;
  };

  auto reference = setup(11);
  auto trace = std::vector<Phylo_tree>{};
  auto mus = std::vector<double>{};
  auto accepted = std::vector<int64_t>{};
  for (auto i = 0; i != 12; ++i) {
    reference->do_mcmc_steps(300);
    trace.push_back(reference->tree());
    mus.push_back(reference->mu());
    accepted.push_back(reference->global_scale_moves_accepted());
  }

  EXPECT_GT(reference->global_scale_moves_accepted(), 0);
  EXPECT_LT(reference->global_scale_moves_accepted(), reference->global_scale_moves_proposed());
  auto first = setup(11);
  for (auto i = 0; i != 5; ++i) { first->do_mcmc_steps(300); }
  auto params = run_to_api_params(*first);

  auto resumed = std::make_unique<delphy::Run>(pool_, std::mt19937{999}, 999, first->tree(), first->coalescent_prior_t_step());
  resumed->set_num_parts(parts);
  apply_api_params_to_run(params.data(), *resumed);
  EXPECT_EQ(resumed->step(), first->step());
  EXPECT_TRUE(resumed->global_scale_move_enabled());
  for (auto i = 5; i != 12; ++i) {
    resumed->do_mcmc_steps(300);
    EXPECT_TRUE(exactly_equal(resumed->tree(), trace[i])) << "step block " << i;
    EXPECT_EQ(resumed->mu(), mus[i]) << "step block " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Parts, Resume_test, ::testing::Values(1, 2));

}  // namespace delphy
