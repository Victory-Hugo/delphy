// Tests for the exact coalescent correction with a free topology (single tree partition part).
//
// The reference genealogies here come from a forward Kingman simulation that shares no code with the
// production coalescent: the pairwise hazard 1 / N(t) is integrated numerically from `Pop_model::pop_at_time`,
// which is the definition of the population curve, not part of the density being tested.  All numbers are
// test inputs.

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "ctpl_stl.h"

#include "api.h"
#include "calibration.h"
#include "coalescent_grid.h"
#include "estd.h"
#include "phylo_tree.h"
#include "pop_model.h"
#include "run.h"

namespace delphy {

namespace {

using Type = Skygrid_pop_model::Type;

constexpr auto k_n0 = 1000.0;  // days

// Integrates the pair coalescence hazard backwards from `t` until `budget` is used up or `t_stop` is reached.
// Returns the time reached and leaves the unused budget in `budget`.
auto integrate_hazard_back(const Pop_model& pop, double t, double t_stop, double pairs, double& budget) -> double {
  while (t > t_stop) {
    auto n = pop.pop_at_time(t);
    auto dt = std::min(t - t_stop, n / pairs / 400.0);
    auto h = pairs / pop.pop_at_time(t - 0.5 * dt);   // midpoint rule
    if (h * dt >= budget) { return t - budget / h; }
    budget -= h * dt;
    t -= dt;
  }
  return t_stop;
}

// A Kingman genealogy for tips sampled at `tip_times` under the curve `pop`, no mutations (one invariant site)
auto ref_kingman_tree(const std::vector<double>& tip_times, const Pop_model& pop, std::mt19937_64& g) -> Phylo_tree {
  auto num_tips = std::ssize(tip_times);
  auto tree = Phylo_tree{static_cast<int>(2 * num_tips - 1)};
  tree.ref_sequence = Real_sequence(1, Real_seq_letter::A);
  auto order = std::vector<int>{};
  for (auto i = 0; i != num_tips; ++i) {
    tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = tip_times[i];
    tree.at(i).name = "S" + std::to_string(i);
    order.push_back(i);
  }
  std::ranges::sort(order, [&](int a, int b) { return tip_times[a] > tip_times[b]; });
  auto active = std::vector<Node_index>{};
  auto next_tip = 0;
  auto t = tip_times[order[0]];
  while (next_tip != num_tips && tip_times[order[next_tip]] == t) { active.push_back(order[next_tip++]); }
  auto next = static_cast<Node_index>(num_tips);
  while (next_tip != num_tips || std::ssize(active) > 1) {
    auto budget = std::exponential_distribution<double>{1.0}(g);
    while (true) {
      auto k = static_cast<double>(std::ssize(active));
      auto t_stop = next_tip != num_tips ? tip_times[order[next_tip]] : -1e300;
      if (k < 2) {   // nothing can coalesce before the next sample joins
        t = t_stop;
      } else {
        t = integrate_hazard_back(pop, t, t_stop, k * (k - 1) / 2.0, budget);
        if (t > t_stop) { break; }
      }
      while (next_tip != num_tips && tip_times[order[next_tip]] == t) { active.push_back(order[next_tip++]); }
    }
    auto i = g() % active.size(); auto x = active[i]; active[i] = active.back(); active.pop_back();
    auto j = g() % active.size(); auto y = active[j]; active[j] = active.back(); active.pop_back();
    auto& node = tree.at(next);
    node.t = t;
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

auto topology_signature(const Phylo_tree& tree, Node_index node) -> std::string {
  if (tree.at(node).is_tip()) { return tree.at(node).name; }
  auto a = topology_signature(tree, tree.at(node).left_child());
  auto b = topology_signature(tree, tree.at(node).right_child());
  if (b < a) { std::swap(a, b); }
  return absl::StrCat("(", a, ",", b, ")");
}
auto topology_signature(const Phylo_tree& tree) -> std::string { return topology_signature(tree, tree.root); }

struct Summary {
  std::string topology;
  double root_age;    // relative to the youngest tip
  double first_age;   // age of the youngest coalescence
};

auto summarize(const Phylo_tree& tree) -> Summary {
  auto t_ref = -std::numeric_limits<double>::max();
  auto first = std::numeric_limits<double>::max();
  for (const auto& node : index_order_traversal(tree)) {
    if (tree.at(node).is_tip()) { t_ref = std::max(t_ref, tree.at(node).t); }
  }
  for (const auto& node : index_order_traversal(tree)) {
    if (tree.at(node).is_inner_node()) { first = std::min(first, t_ref - tree.at(node).t); }
  }
  return {topology_signature(tree), t_ref - tree.at_root().t, first};
}

// Configuration of a test chain
struct Chain_config {
  std::shared_ptr<Pop_model> pop = {};
  double t_step = 0.0;
  bool correct = true;
  bool topology = true;
  int parts = 1;
  int batch = 5;
  // Skygrid only: smoothness (fixed) and the Gamma(alpha, beta) prior on 1 / N_bar, proper so that joint draws exist
  double tau = 1.0;
  double inv_nbar_alpha = 0.0;
  double inv_nbar_beta = 0.0;
  Calibration_set calibrations = {};
  bool threshold = false;               // threshold partitioning on `parts` parts
  double single_part_probability = 0.02;
};

auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, const Chain_config& c)
    -> std::unique_ptr<Run> {
  auto run = std::make_unique<Run>(pool, std::mt19937{seed}, seed, tree, c.t_step);
  run->set_pop_model(c.pop);
  run->set_mu(1e-13);
  run->set_mu_move_enabled(false);
  run->set_alpha_move_enabled(false);
  run->set_final_pop_size_move_enabled(false);
  run->set_pop_growth_rate_move_enabled(false);
  run->set_skygrid_tau(c.tau);
  run->set_skygrid_inv_nbar_prior_alpha(c.inv_nbar_alpha);
  run->set_skygrid_inv_nbar_prior_beta(c.inv_nbar_beta);
  run->set_skygrid_low_gamma_barrier_enabled(false);
  run->set_num_parts(c.parts);
  if (c.threshold) {
    run->set_threshold_partition_params(0.5, 1.0, 4, c.single_part_probability);
    run->set_threshold_partitioning(true);
  }
  run->set_local_moves_per_global_move(c.batch);
  run->set_topology_moves_enabled(c.topology);
  run->set_coal_prior_fixed_t_step(c.t_step);
  if (not c.calibrations.empty()) { run->set_calibrations(c.calibrations); }
  if (c.correct) { run->set_exact_coalescent_correction(true); }
  return run;
}

// A Skygrid curve drawn from the prior of `make_run`: 1 / N_bar ~ Gamma(alpha, beta), increments ~ Normal(0, 1/tau)
auto ref_skygrid_draw(const std::vector<double>& x, Type type, const Chain_config& c, std::mt19937_64& g)
    -> std::shared_ptr<Skygrid_pop_model> {
  auto u = std::gamma_distribution<double>{c.inv_nbar_alpha, 1.0 / c.inv_nbar_beta}(g);
  auto gamma = std::vector<double>(x.size(), 0.0);
  for (auto k = 1; k != std::ssize(x); ++k) {
    gamma[k] = gamma[k - 1] + std::normal_distribution<double>{0.0, 1.0 / std::sqrt(c.tau)}(g);
  }
  auto mean = 0.0; for (auto v : gamma) { mean += v; } mean /= std::ssize(gamma);
  for (auto& v : gamma) { v += -std::log(u) - mean; }
  return std::make_shared<Skygrid_pop_model>(x, gamma, type);
}

// Pearson statistic of the homogeneity of two samples of categories, and its degrees of freedom
auto two_sample_chi2(const std::map<std::string, int>& a, const std::map<std::string, int>& b) -> std::pair<double, int> {
  auto na = 0.0, nb = 0.0;
  for (const auto& [k, v] : a) { na += v; }
  for (const auto& [k, v] : b) { nb += v; }
  auto keys = std::map<std::string, int>{};
  for (const auto& [k, v] : a) { keys[k] += v; }
  for (const auto& [k, v] : b) { keys[k] += v; }
  auto chi2 = 0.0;
  for (const auto& [k, total] : keys) {
    auto oa = a.contains(k) ? a.at(k) : 0;
    auto ob = b.contains(k) ? b.at(k) : 0;
    auto ea = total * na / (na + nb), eb = total * nb / (na + nb);
    chi2 += (oa - ea) * (oa - ea) / ea + (ob - eb) * (ob - eb) / eb;
  }
  return {chi2, static_cast<int>(std::ssize(keys)) - 1};
}

// Upper 1e-4 quantile of chi-squared with `df` degrees of freedom (Wilson-Hilferty), a deliberately strict gate
// for a unit test that runs on every build
auto chi2_gate(int df) -> double {
  auto z = 3.719;
  auto h = 2.0 / (9.0 * df);
  return df * std::pow(1.0 - h + z * std::sqrt(h), 3);
}

auto z_of_means(const std::vector<double>& a, const std::vector<double>& b) -> double {
  auto moments = [](const std::vector<double>& v) {
    auto m = 0.0; for (auto x : v) { m += x; } m /= std::ssize(v);
    auto s = 0.0; for (auto x : v) { s += (x - m) * (x - m); } s /= (std::ssize(v) - 1);
    return std::pair{m, s / std::ssize(v)};
  };
  auto [ma, va] = moments(a);
  auto [mb, vb] = moments(b);
  return (ma - mb) / std::sqrt(va + vb);
}

struct Stationarity {
  std::pair<double, int> topology_chi2;
  double z_root_age;
  double z_first_age;
  int64_t topology_changes;
  double batch_acceptance;
};

// `chains` chains, each started from an independent reference draw and run for `iterations` batches; their
// endpoints are compared with a second, independent set of reference draws.  For a kernel that leaves the
// target invariant the endpoints are exact draws.
auto run_stationarity(int chains, int iterations, const std::vector<double>& tip_times, const Chain_config& c,
                      uint64_t seed, const std::vector<double>& skygrid_x = {}, Type type = Type::k_staircase)
    -> Stationarity {
  auto pool = ctpl::thread_pool{1};
  auto g = std::mt19937_64{seed};
  auto ref_topologies = std::map<std::string, int>{};
  auto mcmc_topologies = std::map<std::string, int>{};
  auto ref_root = std::vector<double>{}, mcmc_root = std::vector<double>{};
  auto ref_first = std::vector<double>{}, mcmc_first = std::vector<double>{};
  auto changes = int64_t{0};
  auto proposed = int64_t{0}, accepted = int64_t{0};
  for (auto i = 0; i != chains; ++i) {
    auto config = c;
    if (not skygrid_x.empty()) { config.pop = ref_skygrid_draw(skygrid_x, type, c, g); }
    auto ref = summarize(ref_kingman_tree(tip_times, *config.pop, g));
    ++ref_topologies[ref.topology];
    ref_root.push_back(ref.root_age);
    ref_first.push_back(ref.first_age);

    if (not skygrid_x.empty()) { config.pop = ref_skygrid_draw(skygrid_x, type, c, g); }
    auto run = make_run(pool, ref_kingman_tree(tip_times, *config.pop, g), static_cast<uint32_t>(g()), config);
    auto last = topology_signature(run->tree());
    for (auto it = 0; it != iterations; ++it) {
      run->do_mcmc_steps(c.batch);
      auto now = topology_signature(run->tree());
      if (now != last) { ++changes; last = now; }
    }
    auto end = summarize(run->tree());
    ++mcmc_topologies[end.topology];
    mcmc_root.push_back(end.root_age);
    mcmc_first.push_back(end.first_age);
    proposed += run->exact_coalescent_batches_proposed();
    accepted += run->exact_coalescent_batches_accepted();
  }
  return {two_sample_chi2(mcmc_topologies, ref_topologies), z_of_means(mcmc_root, ref_root),
          z_of_means(mcmc_first, ref_first), changes,
          proposed > 0 ? static_cast<double>(accepted) / proposed : 1.0};
}

auto print(const std::string& label, const Stationarity& s) -> void {
  std::cerr << absl::StreamFormat(
      "EXACT-FREE %s: topology chi2 %.1f on %d df (gate %.1f), z(root age) %+.2f, z(first age) %+.2f, "
      "%d topology changes, batch acceptance %.3f\n",
      label, s.topology_chi2.first, s.topology_chi2.second, chi2_gate(s.topology_chi2.second),
      s.z_root_age, s.z_first_age, s.topology_changes, s.batch_acceptance);
}

auto staircase_knots() -> std::vector<double> { return {-1.5 * k_n0, -1.0 * k_n0, -0.5 * k_n0, -0.2 * k_n0, 0.0}; }

}  // namespace

// ---------------------------------------------------------------------------------------------
// 1. What is accepted and what is still refused
// ---------------------------------------------------------------------------------------------

TEST(Exact_free_topology_test, a_free_topology_is_accepted_with_one_part_only) {
  auto pool = ctpl::thread_pool{2};
  auto g = std::mt19937_64{1};
  auto pop = std::make_shared<Const_pop_model>(k_n0);
  auto tree = ref_kingman_tree(std::vector<double>(8, 0.0), *pop, g);
  auto c = Chain_config{.pop = pop, .t_step = k_n0, .correct = false};

  {  // topology moves first, then the correction
    auto run = make_run(pool, tree, 7, c);
    EXPECT_NO_THROW(run->set_exact_coalescent_correction(true));
    EXPECT_NO_THROW(run->do_mcmc_steps(50));
  }
  {  // the correction first (fixed topology), then topology moves
    auto fixed = c;
    fixed.topology = false;
    auto run = make_run(pool, tree, 7, fixed);
    run->set_exact_coalescent_correction(true);
    EXPECT_NO_THROW(run->set_topology_moves_enabled(true));
  }
  {  // several parts are refused, in either order and when the parts change afterwards
    auto two = c;
    two.parts = 2;
    auto run = make_run(pool, tree, 7, two);
    EXPECT_THROW(run->set_exact_coalescent_correction(true), std::invalid_argument);
    EXPECT_FALSE(run->exact_coalescent_correction());

    auto fixed = two;
    fixed.topology = false;
    auto run2 = make_run(pool, tree, 7, fixed);
    run2->set_exact_coalescent_correction(true);
    EXPECT_THROW(run2->set_topology_moves_enabled(true), std::invalid_argument);

    auto run3 = make_run(pool, ref_kingman_tree(std::vector<double>(60, 0.0), *pop, g), 7, c);
    run3->set_exact_coalescent_correction(true);
    run3->set_num_parts(2);
    EXPECT_THROW(run3->do_mcmc_steps(50), std::invalid_argument);
  }
  {  // calibrations with a free topology combine with the correction in either order, still on one part only
    auto config = std::string{
        "version 1\n\ncalibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
        "  distribution uniform\n  youngest 0.1\n  oldest 100\nend\n"};
    auto is = std::istringstream{config};
    auto calibrations = read_calibration_config(is);

    auto run = make_run(pool, tree, 7, c);
    run->set_exact_coalescent_correction(true);
    EXPECT_NO_THROW(run->set_calibrations(calibrations));
    EXPECT_NO_THROW(run->do_mcmc_steps(50));

    auto run2 = make_run(pool, tree, 7, c);
    run2->set_calibrations(calibrations);
    EXPECT_NO_THROW(run2->set_exact_coalescent_correction(true));
    EXPECT_NO_THROW(run2->do_mcmc_steps(50));

    auto run3 = make_run(pool, ref_kingman_tree(std::vector<double>(60, 0.0), *pop, g), 7, c);
    run3->set_calibrations(calibrations);
    run3->set_exact_coalescent_correction(true);
    run3->set_num_parts(2);
    EXPECT_THROW(run3->do_mcmc_steps(50), std::invalid_argument);
  }
}

// ---------------------------------------------------------------------------------------------
// 2. Bookkeeping through topology changes and rolled-back batches
// ---------------------------------------------------------------------------------------------

TEST(Exact_free_topology_test, the_coalescent_term_stays_exact_through_topology_changes_and_rollbacks) {
  auto pool = ctpl::thread_pool{1};
  auto g = std::mt19937_64{3};
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    auto c = Chain_config{.t_step = 2.0 * k_n0, .batch = 40, .tau = 4.0};
    c.pop = std::make_shared<Skygrid_pop_model>(
        staircase_knots(), std::vector<double>{std::log(400.0), std::log(2000.0), std::log(300.0),
                                                std::log(1500.0), std::log(800.0)}, type);
    auto tip_times = std::vector<double>{0.0, 0.0, 0.0, -30.0, -30.0, -200.0, -450.0, -450.0, -700.0, -900.0};
    auto run = make_run(pool, ref_kingman_tree(tip_times, *c.pop, g), 11, c);
    auto exact = [](const delphy::Run& r) {
      return calc_exact_coalescent_log_prior(coalescent_event_times_of(r.tree()), r.pop_model());
    };
    auto topologies = std::map<std::string, int>{};
    for (auto i = 0; i != 150; ++i) {
      run->do_mcmc_steps(c.batch);
      ++topologies[topology_signature(run->tree())];
      ASSERT_NEAR(run->log_coalescent_prior(), exact(*run), 1e-8) << "batch " << i;

      // A Run rebuilt from the current state agrees on every derived quantity
      auto fresh_config = c;
      fresh_config.pop = std::make_shared<Skygrid_pop_model>(
          static_cast<const Skygrid_pop_model&>(run->pop_model()));
      auto fresh = make_run(pool, run->tree(), 5, fresh_config);
      fresh->set_hky_kappa(run->hky_kappa());
      fresh->set_hky_pi(run->hky_pi());
      ASSERT_NEAR(fresh->log_G(), run->log_G(), 1e-6 * std::max(1.0, std::abs(run->log_G())));
      ASSERT_NEAR(fresh->log_coalescent_prior(), run->log_coalescent_prior(), 1e-8);
      ASSERT_NEAR(fresh->log_other_priors(), run->log_other_priors(), 1e-8);
    }
    EXPECT_GT(std::ssize(topologies), 5) << "the topology hardly moved";
    EXPECT_GT(run->exact_coalescent_batches_accepted(), 0);
    EXPECT_LT(run->exact_coalescent_batches_accepted(), run->exact_coalescent_batches_proposed())
        << "no batch was rolled back, so the rollback path was not exercised";
  }
}

// ---------------------------------------------------------------------------------------------
// 3. Chains started from exact draws stay on the exact target: topology and node ages
// ---------------------------------------------------------------------------------------------

TEST(Exact_free_topology_test, four_tips_same_date_constant_population) {
  auto chains = estd::is_debug_enabled ? 300 : 3000;
  auto c = Chain_config{.pop = std::make_shared<Const_pop_model>(k_n0), .t_step = 2.0 * k_n0};
  auto tips = std::vector<double>(4, 0.0);
  auto exact = run_stationarity(chains, 40, tips, c, 101);
  print("4 tips, constant, corrected", exact);
  EXPECT_LT(exact.topology_chi2.first, chi2_gate(exact.topology_chi2.second));
  EXPECT_LT(std::abs(exact.z_root_age), 4.0);
  EXPECT_LT(std::abs(exact.z_first_age), 4.0);
  EXPECT_GT(exact.topology_changes, chains / 2);

  // The same grid without the correction is biased in the node ages, which shows the test has power
  if (not estd::is_debug_enabled) {
    c.correct = false;
    auto cells = run_stationarity(chains, 40, tips, c, 102);
    print("4 tips, constant, cell prior", cells);
    EXPECT_GT(std::max(std::abs(cells.z_root_age), std::abs(cells.z_first_age)), 6.0);
  }
}

TEST(Exact_free_topology_test, four_tips_different_dates_constant_population) {
  auto chains = estd::is_debug_enabled ? 300 : 3000;
  auto c = Chain_config{.pop = std::make_shared<Const_pop_model>(k_n0), .t_step = 1.5 * k_n0};
  auto exact = run_stationarity(chains, 40, {0.0, 0.0, -300.0, -800.0}, c, 103);
  print("4 tips at 3 dates, constant, corrected", exact);
  EXPECT_LT(exact.topology_chi2.first, chi2_gate(exact.topology_chi2.second));
  EXPECT_LT(std::abs(exact.z_root_age), 4.0);
  EXPECT_LT(std::abs(exact.z_first_age), 4.0);
}

TEST(Exact_free_topology_test, five_tips_skygrid_with_the_curve_sampled) {
  auto chains = estd::is_debug_enabled ? 200 : 2000;
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    auto c = Chain_config{.t_step = 1.5 * k_n0, .tau = 2.0, .inv_nbar_alpha = 4.0, .inv_nbar_beta = 3.0 * k_n0};
    auto s = run_stationarity(chains, 40, {0.0, 0.0, 0.0, -250.0, -600.0}, c, 104, staircase_knots(), type);
    print(type == Type::k_staircase ? "5 tips, staircase Skygrid" : "5 tips, log-linear Skygrid", s);
    EXPECT_LT(s.topology_chi2.first, chi2_gate(s.topology_chi2.second));
    EXPECT_LT(std::abs(s.z_root_age), 4.0);
    EXPECT_LT(std::abs(s.z_first_age), 4.0);
  }
}

// ---------------------------------------------------------------------------------------------
// 4. Age calibrations whose targets follow the topology, with and without monophyly
// ---------------------------------------------------------------------------------------------
//
// The target is  Kingman(genealogy) x prod_c f_c(age of MRCA(A_c)) x prod_{c requires monophyly} 1{A_c is a clade}.
// Its exact draws come from rejection sampling of the forward Kingman simulation, with the calibration densities,
// MRCAs and clade checks written out again below (no data, so the likelihood is flat).

namespace {

enum class Ref_density { uniform, normal, lognormal };

// One calibration, as the reference sees it.  Ages are in years before 2020-01-01; `taxa` empty means the root.
struct Ref_calibration {
  std::string name = {};
  std::vector<std::string> taxa = {};
  bool require_monophyly = true;
  Ref_density density = Ref_density::uniform;
  double a = 0.0;  // uniform: youngest; normal: mean; lognormal: median
  double b = 0.0;  // uniform: oldest; normal: stddev; lognormal: sigma
  double youngest = -std::numeric_limits<double>::infinity();  // extra hard bounds (normal, lognormal)
  double oldest = +std::numeric_limits<double>::infinity();

  // Unnormalized log density and its maximum, for rejection sampling
  auto log_f(double age) const -> double {
    auto lo = density == Ref_density::uniform ? a : youngest;
    auto hi = density == Ref_density::uniform ? b : oldest;
    if (not (age >= lo && age <= hi)) { return -std::numeric_limits<double>::infinity(); }
    switch (density) {
      case Ref_density::uniform: return 0.0;
      case Ref_density::normal: return -0.5 * (age - a) * (age - a) / (b * b);
      case Ref_density::lognormal: {
        if (not (age > 0.0)) { return -std::numeric_limits<double>::infinity(); }
        auto y = std::log(age) - std::log(a);
        return -0.5 * y * y / (b * b) - std::log(age);
      }
    }
    return 0.0;
  }
  auto log_f_max() const -> double {
    return density == Ref_density::lognormal ? 0.5 * b * b - std::log(a) : 0.0;
  }

  auto stanza() const -> std::string {
    auto out = absl::StrCat("\ncalibration ", name, "\n  taxa ", taxa.empty() ? "all" : absl::StrJoin(taxa, " "), "\n");
    if (not require_monophyly) { absl::StrAppend(&out, "  monophyly not_required\n"); }
    absl::StrAppend(&out, "  age_units years_bp\n  age_reference_year 2020\n");
    switch (density) {
      case Ref_density::uniform:
        absl::StrAppend(&out, absl::StrFormat("  distribution uniform\n  youngest %.17g\n  oldest %.17g\n", a, b));
        break;
      case Ref_density::normal:
        absl::StrAppend(&out, absl::StrFormat("  distribution normal\n  mean %.17g\n  stddev %.17g\n", a, b));
        break;
      case Ref_density::lognormal:
        absl::StrAppend(&out, absl::StrFormat("  distribution lognormal\n  median %.17g\n  sigma %.17g\n", a, b));
        break;
    }
    if (density != Ref_density::uniform && std::isfinite(youngest)) {
      absl::StrAppend(&out, absl::StrFormat("  youngest %.17g\n", youngest));
    }
    if (density != Ref_density::uniform && std::isfinite(oldest)) {
      absl::StrAppend(&out, absl::StrFormat("  oldest %.17g\n", oldest));
    }
    return out + "end\n";
  }
};

auto parse_ref_calibrations(const std::vector<Ref_calibration>& cals) -> Calibration_set {
  auto text = std::string{"version 1\n"};
  for (const auto& c : cals) { text += c.stanza(); }
  auto is = std::istringstream{text};
  auto set = read_calibration_config(is);
  for (const auto& c : set.calibrations) { CHECK_EQ(c.reference_t, 0.0); }
  return set;
}

// The MRCA of `taxa` (the root if empty) and whether `taxa` is a clade, from the tip names below every node
auto ref_mrca(const Phylo_tree& tree, const std::vector<std::string>& taxa) -> std::pair<Node_index, bool> {
  if (taxa.empty()) { return {tree.root, true}; }
  auto best = k_no_node;
  auto best_size = std::numeric_limits<int>::max();
  std::function<std::pair<int, int>(Node_index)> visit = [&](Node_index node) -> std::pair<int, int> {
    auto marked = 0, all = 0;
    if (tree.at(node).is_tip()) {
      all = 1;
      marked = std::ranges::count(taxa, tree.at(node).name) > 0 ? 1 : 0;
    } else {
      for (auto child : tree.at(node).children) {
        auto [m, a] = visit(child);
        marked += m; all += a;
      }
    }
    if (marked == std::ssize(taxa) && all < best_size) { best = node; best_size = all; }
    return {marked, all};
  };
  visit(tree.root);
  return {best, best_size == std::ssize(taxa)};
}

// log of the calibration factor of the target, up to a constant (-infinity on zero density)
auto ref_log_weight(const Phylo_tree& tree, const std::vector<Ref_calibration>& cals) -> double {
  auto result = 0.0;
  for (const auto& c : cals) {
    auto [mrca, clade] = ref_mrca(tree, c.taxa);
    if (c.require_monophyly && not clade) { return -std::numeric_limits<double>::infinity(); }
    result += c.log_f(-tree.at(mrca).t / 365.0) - c.log_f_max();
  }
  return result;
}

// An exact draw of the genealogy, and of the Skygrid curve too if `skygrid_x` is given (the curve from its prior,
// as in ref_skygrid_draw, redrawn on every attempt, so that accepted pairs follow the joint calibrated target)
auto ref_calibrated_draw(const std::vector<double>& tip_times, const Chain_config& c,
                         const std::vector<Ref_calibration>& cals, std::mt19937_64& g,
                         const std::vector<double>& skygrid_x = {}, Type type = Type::k_staircase)
    -> std::pair<Phylo_tree, std::shared_ptr<Pop_model>> {
  for (auto attempt = 0; attempt != 10'000'000; ++attempt) {
    auto pop = skygrid_x.empty() ? c.pop : ref_skygrid_draw(skygrid_x, type, c, g);
    auto tree = ref_kingman_tree(tip_times, *pop, g);
    if (std::log(std::uniform_real_distribution<double>{0.0, 1.0}(g)) < ref_log_weight(tree, cals)) {
      return {std::move(tree), pop};
    }
  }
  CHECK(false) << "the calibrated target has almost no mass under the Kingman prior";
  return {};
}

struct Calibrated_stationarity {
  Stationarity base;
  std::vector<double> z_calibrated_age;   // one per calibration: age of the MRCA of its set
  std::vector<int64_t> not_a_clade;       // per calibration: states after a batch in which its set is not a clade
  int64_t states = 0;
};

// As run_stationarity, with every state after every batch checked against the reference: the calibration term of
// the Run equals the production densities at the independently found MRCAs, every set that requires monophyly is a
// clade, and the state has nonzero reference density.
auto run_calibrated_stationarity(int chains, int iterations, const std::vector<double>& tip_times, Chain_config c,
                                 const std::vector<Ref_calibration>& cals, uint64_t seed,
                                 const std::vector<double>& skygrid_x = {}, Type type = Type::k_staircase,
                                 const std::vector<Ref_calibration>* chain_cals = nullptr)
    -> Calibrated_stationarity {
  // `chain_cals`, if given, is what the chains run with (same sample sets, in the same order); for the controls
  const auto& run_cals = chain_cals != nullptr ? *chain_cals : cals;
  c.calibrations = parse_ref_calibrations(run_cals);
  auto pool = ctpl::thread_pool{1};
  auto g = std::mt19937_64{seed};
  auto ref_topologies = std::map<std::string, int>{}, mcmc_topologies = std::map<std::string, int>{};
  auto ref_root = std::vector<double>{}, mcmc_root = std::vector<double>{};
  auto ref_first = std::vector<double>{}, mcmc_first = std::vector<double>{};
  auto num_cals = std::ssize(cals);
  auto ref_age = std::vector<std::vector<double>>(num_cals), mcmc_age = std::vector<std::vector<double>>(num_cals);
  auto result = Calibrated_stationarity{};
  result.not_a_clade.assign(num_cals, 0);
  auto changes = int64_t{0}, proposed = int64_t{0}, accepted = int64_t{0};
  auto record = [&](const Phylo_tree& tree, std::map<std::string, int>& topologies, std::vector<double>& root,
                    std::vector<double>& first, std::vector<std::vector<double>>& age) {
    auto s = summarize(tree);
    ++topologies[s.topology];
    root.push_back(s.root_age);
    first.push_back(s.first_age);
    for (auto i = 0; i != num_cals; ++i) { age[i].push_back(-tree.at(ref_mrca(tree, cals[i].taxa).first).t); }
  };
  for (auto chain = 0; chain != chains; ++chain) {
    record(ref_calibrated_draw(tip_times, c, cals, g, skygrid_x, type).first, ref_topologies, ref_root, ref_first,
           ref_age);

    auto config = c;
    auto [start, pop] = ref_calibrated_draw(tip_times, c, cals, g, skygrid_x, type);
    config.pop = pop;
    auto run = make_run(pool, start, static_cast<uint32_t>(g()), config);
    auto last = topology_signature(run->tree());
    for (auto it = 0; it != iterations; ++it) {
      run->do_mcmc_steps(c.batch);
      const auto& tree = run->tree();
      auto now = topology_signature(tree);
      if (now != last) { ++changes; last = now; }

      auto expected = 0.0;
      for (auto i = 0; i != num_cals; ++i) {
        auto [mrca, clade] = ref_mrca(tree, run_cals[i].taxa);
        if (run_cals[i].require_monophyly) {
          CHECK(clade) << "calibration " << run_cals[i].name << " left its clade in " << now;
        }
        if (not clade) { ++result.not_a_clade[i]; }
        expected += c.calibrations.calibrations[i].log_density_at(tree.at(mrca).t);
      }
      ++result.states;
      CHECK(std::isfinite(ref_log_weight(tree, run_cals))) << "a state of zero density in " << now;
      CHECK(std::abs(run->log_calibration_prior() - expected) < 1e-8)
          << run->log_calibration_prior() << " != " << expected << " in " << now;
    }
    record(run->tree(), mcmc_topologies, mcmc_root, mcmc_first, mcmc_age);
    proposed += run->exact_coalescent_batches_proposed();
    accepted += run->exact_coalescent_batches_accepted();
  }
  result.base = {two_sample_chi2(mcmc_topologies, ref_topologies), z_of_means(mcmc_root, ref_root),
                 z_of_means(mcmc_first, ref_first), changes,
                 proposed > 0 ? static_cast<double>(accepted) / proposed : 1.0};
  for (auto i = 0; i != num_cals; ++i) { result.z_calibrated_age.push_back(z_of_means(mcmc_age[i], ref_age[i])); }
  return result;
}

auto print(const std::string& label, const Calibrated_stationarity& s) -> void {
  print(label, s.base);
  auto zs = std::vector<std::string>{};
  for (auto z : s.z_calibrated_age) { zs.push_back(absl::StrFormat("%+.2f", z)); }
  std::cerr << absl::StreamFormat("EXACT-FREE %s: z(calibrated MRCA ages) %s, states where the set is not a clade %s "
                                  "of %d\n", label, absl::StrJoin(zs, " "), absl::StrJoin(s.not_a_clade, " "), s.states);
}

auto expect_stationary(const Calibrated_stationarity& s) -> void {
  EXPECT_LT(s.base.topology_chi2.first, chi2_gate(s.base.topology_chi2.second));
  EXPECT_LT(std::abs(s.base.z_root_age), 4.0);
  EXPECT_LT(std::abs(s.base.z_first_age), 4.0);
  for (auto z : s.z_calibrated_age) { EXPECT_LT(std::abs(z), 4.0); }
}

auto calibrated_chains() -> int { return estd::is_debug_enabled ? 300 : 3000; }

}  // namespace

TEST(Exact_free_topology_calibration_test, the_monophyly_key_is_parsed_written_and_bound) {
  auto cal = Ref_calibration{.name = "pair", .taxa = {"S0", "S2"}, .require_monophyly = false,
                             .density = Ref_density::uniform, .a = 0.1, .b = 20.0};
  auto set = parse_ref_calibrations({cal});
  ASSERT_EQ(set.size(), 1);
  EXPECT_FALSE(set.calibrations[0].require_monophyly);
  auto text = calibration_config_to_string(set);
  EXPECT_NE(text.find("monophyly not_required"), std::string::npos);
  auto is = std::istringstream{text};
  EXPECT_FALSE(read_calibration_config(is).calibrations[0].require_monophyly);

  // The default is unchanged and is not written, so existing canonical texts stay identical
  cal.require_monophyly = true;
  auto required = parse_ref_calibrations({cal});
  EXPECT_TRUE(required.calibrations[0].require_monophyly);
  EXPECT_EQ(calibration_config_to_string(required).find("monophyly"), std::string::npos);

  auto bad = [](const std::string& body) {
    auto is = std::istringstream{"version 1\n\ncalibration c\n" + body +
                                 "  age_units years_bp\n  age_reference_year 2020\n  distribution uniform\n"
                                 "  youngest 0.1\n  oldest 2\nend\n"};
    EXPECT_THROW(read_calibration_config(is), Calibration_error) << body;
  };
  bad("  taxa S0 S1\n  monophyly maybe\n");
  bad("  taxa all\n  monophyly not_required\n");
  bad("  taxa S0 S1\n  monophyly required\n  monophyly not_required\n");

  // ((S0,S1),(S2,S3)): {S0,S2} is not a clade; its MRCA is the root
  auto tree = Phylo_tree{7};
  tree.ref_sequence = Real_sequence(1, Real_seq_letter::A);
  for (auto i = 0; i != 4; ++i) { tree.at(i).name = absl::StrCat("S", i); tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = 0.0; }
  auto join = [&](Node_index p, Node_index l, Node_index r, double t) {
    tree.at(p).children = {l, r}; tree.at(p).t = t; tree.at(l).parent = p; tree.at(r).parent = p;
  };
  join(4, 0, 1, -300.0); join(5, 2, 3, -400.0); join(6, 4, 5, -1000.0);
  tree.root = 6;
  tree.at(6).parent = k_no_node;
  EXPECT_EQ(bind_calibrations(set, tree), std::vector<Node_index>{6});
  EXPECT_THROW(bind_calibrations(required, tree), Calibration_error);

  // Two age-only sets may share their MRCA; two sets that require monophyly may not
  auto other = cal;
  other.name = "other"; other.taxa = {"S1", "S3"}; other.require_monophyly = false;
  cal.require_monophyly = false;
  EXPECT_EQ(bind_calibrations(parse_ref_calibrations({cal, other}), tree), (std::vector<Node_index>{6, 6}));
  auto root = Ref_calibration{.name = "root", .density = Ref_density::uniform, .a = 0.1, .b = 20.0};
  auto root2 = root;
  root2.name = "root2"; root2.taxa = {"S0", "S1", "S2", "S3"};
  EXPECT_THROW(bind_calibrations(parse_ref_calibrations({root, root2}), tree), Calibration_error);
}

TEST(Exact_free_topology_calibration_test, conflicting_and_out_of_support_calibrations_are_refused) {
  auto pool = ctpl::thread_pool{1};
  auto g = std::mt19937_64{41};
  auto pop = std::make_shared<Const_pop_model>(k_n0);
  auto c = Chain_config{.pop = pop, .t_step = 2.0 * k_n0};
  auto run = make_run(pool, ref_kingman_tree(std::vector<double>(4, 0.0), *pop, g), 3, c);

  // Nested sets whose supports cannot be ordered (the parent must be older than its child)
  auto inner = Ref_calibration{.name = "inner", .taxa = {"S0", "S1"}, .density = Ref_density::uniform,
                               .a = 5.0, .b = 9.0};
  auto outer = Ref_calibration{.name = "outer", .taxa = {"S0", "S1", "S2"}, .density = Ref_density::uniform,
                               .a = 0.1, .b = 4.0};
  EXPECT_ANY_THROW(run->set_calibrations(parse_ref_calibrations({inner, outer})));

  // A starting tree outside a hard bound is refused, not rescaled
  auto root = Ref_calibration{.name = "root", .density = Ref_density::uniform, .a = 500.0, .b = 900.0};
  EXPECT_ANY_THROW(run->set_calibrations(parse_ref_calibrations({root})));
  EXPECT_TRUE(run->calibrations().empty());
}

TEST(Exact_free_topology_calibration_test, four_tips_root_and_clade_calibrations) {
  auto c = Chain_config{.pop = std::make_shared<Const_pop_model>(k_n0), .t_step = 2.0 * k_n0};
  auto cals = std::vector<Ref_calibration>{
    {.name = "root", .density = Ref_density::uniform, .a = 2.0, .b = 7.0},
    {.name = "pair", .taxa = {"S0", "S1"}, .density = Ref_density::normal, .a = 1.0, .b = 0.6, .youngest = 0.05}};
  auto s = run_calibrated_stationarity(calibrated_chains(), 40, std::vector<double>(4, 0.0), c, cals, 201);
  print("4 tips, root + clade", s);
  expect_stationary(s);
  // The clade constraint leaves 3 rooted topologies, so fewer changes than in the unconstrained tests
  EXPECT_GT(s.base.topology_changes, calibrated_chains() / 10);
  EXPECT_LT(s.base.batch_acceptance, 1.0);

  // Control: chains that keep the clade constraint but drop the age prior on the pair are detected
  if (not estd::is_debug_enabled) {
    auto dropped = cals;
    dropped[1].density = Ref_density::uniform;
    dropped[1].a = 0.0;
    dropped[1].b = 1e3;
    auto control = run_calibrated_stationarity(calibrated_chains(), 40, std::vector<double>(4, 0.0), c, cals, 206, {},
                                               Type::k_staircase, &dropped);
    print("4 tips, root + clade, chains without the pair's age prior", control);
    auto worst = std::max({std::abs(control.base.z_root_age), std::abs(control.base.z_first_age),
                           std::abs(control.z_calibrated_age[0]), std::abs(control.z_calibrated_age[1])});
    EXPECT_GT(worst, 6.0);
  }
}

TEST(Exact_free_topology_calibration_test, four_tips_mrca_ages_without_monophyly) {
  // Two crossing age-only sets.  In some topologies they share their MRCA, which then carries both densities.
  auto c = Chain_config{.pop = std::make_shared<Const_pop_model>(k_n0), .t_step = 2.0 * k_n0};
  auto cals = std::vector<Ref_calibration>{
    {.name = "a", .taxa = {"S0", "S1"}, .require_monophyly = false, .density = Ref_density::uniform,
     .a = 0.1, .b = 5.0},
    {.name = "b", .taxa = {"S0", "S2"}, .require_monophyly = false, .density = Ref_density::lognormal,
     .a = 3.0, .b = 0.5, .youngest = 1.0}};
  auto s = run_calibrated_stationarity(calibrated_chains(), 40, {0.0, 0.0, -200.0, -500.0}, c, cals, 202);
  print("4 tips, MRCA ages without monophyly", s);
  expect_stationary(s);
  for (auto i = 0; i != 2; ++i) {  // each set's MRCA identity really changes: sometimes a clade, sometimes not
    EXPECT_GT(s.not_a_clade[i], s.states / 50) << cals[i].name;
    EXPECT_LT(s.not_a_clade[i], s.states - s.states / 50) << cals[i].name;
  }
}

TEST(Exact_free_topology_calibration_test, five_tips_disjoint_and_nested_calibrations) {
  auto c = Chain_config{.pop = std::make_shared<Const_pop_model>(k_n0), .t_step = 1.5 * k_n0};
  auto tips = std::vector<double>{0.0, 0.0, 0.0, -250.0, -600.0};
  {
    auto cals = std::vector<Ref_calibration>{
      {.name = "left", .taxa = {"S0", "S1"}, .density = Ref_density::uniform, .a = 0.1, .b = 2.5},
      {.name = "right", .taxa = {"S2", "S3"}, .density = Ref_density::normal, .a = 2.0, .b = 0.7, .youngest = 0.7}};
    auto s = run_calibrated_stationarity(calibrated_chains(), 40, tips, c, cals, 203);
    print("5 tips, disjoint clades", s);
    expect_stationary(s);
  }
  {
    auto cals = std::vector<Ref_calibration>{
      {.name = "inner", .taxa = {"S0", "S1"}, .density = Ref_density::uniform, .a = 0.05, .b = 3.0},
      {.name = "outer", .taxa = {"S0", "S1", "S2"}, .density = Ref_density::lognormal, .a = 2.5, .b = 0.5},
      {.name = "loose", .taxa = {"S2", "S4"}, .require_monophyly = false, .density = Ref_density::normal,
       .a = 4.0, .b = 1.0},
      {.name = "root", .density = Ref_density::uniform, .a = 2.5, .b = 12.0}};
    auto s = run_calibrated_stationarity(calibrated_chains(), 40, tips, c, cals, 204);
    print("5 tips, nested clades + age-only set + root", s);
    expect_stationary(s);
  }
}

TEST(Exact_free_topology_calibration_test, five_tips_skygrid_curve_sampled_with_calibrations) {
  auto c = Chain_config{.t_step = 1.5 * k_n0, .tau = 2.0, .inv_nbar_alpha = 4.0, .inv_nbar_beta = 3.0 * k_n0};
  auto chains = estd::is_debug_enabled ? 200 : 2000;
  auto cals = std::vector<Ref_calibration>{
    {.name = "pair", .taxa = {"S0", "S1"}, .density = Ref_density::normal, .a = 0.8, .b = 0.5, .youngest = 0.02},
    {.name = "root", .density = Ref_density::lognormal, .a = 4.0, .b = 0.5}};
  auto tips = std::vector<double>{0.0, 0.0, 0.0, -250.0, -600.0};
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    auto s = run_calibrated_stationarity(chains, 40, tips, c, cals, 205, staircase_knots(), type);
    print(type == Type::k_staircase ? "5 tips, staircase Skygrid, calibrations"
                                    : "5 tips, log-linear Skygrid, calibrations", s);
    expect_stationary(s);
  }
}

TEST(Exact_free_topology_calibration_test, bookkeeping_through_rollbacks_and_a_checkpoint) {
  auto pool = ctpl::thread_pool{1};
  auto g = std::mt19937_64{43};
  auto c = Chain_config{.t_step = 2.0 * k_n0, .batch = 40, .tau = 4.0};
  c.pop = std::make_shared<Skygrid_pop_model>(
      staircase_knots(), std::vector<double>{std::log(400.0), std::log(2000.0), std::log(300.0),
                                              std::log(1500.0), std::log(800.0)}, Type::k_log_linear);
  auto tip_times = std::vector<double>{0.0, 0.0, 0.0, -30.0, -30.0, -200.0, -450.0, -450.0, -700.0, -900.0};
  auto cals = std::vector<Ref_calibration>{
    {.name = "clade", .taxa = {"S0", "S1", "S2"}, .density = Ref_density::normal, .a = 1.5, .b = 1.0,
     .youngest = 0.01},
    {.name = "loose", .taxa = {"S3", "S8"}, .require_monophyly = false, .density = Ref_density::lognormal,
     .a = 4.0, .b = 0.6},
    {.name = "root", .density = Ref_density::uniform, .a = 2.6, .b = 40.0}};
  c.calibrations = parse_ref_calibrations(cals);
  auto run = make_run(pool, ref_calibrated_draw(tip_times, c, cals, g).first, 13, c);
  auto exact_coal = [](const delphy::Run& r) {
    return calc_exact_coalescent_log_prior(coalescent_event_times_of(r.tree()), r.pop_model());
  };
  auto check = [&](const delphy::Run& r, int i) {
    auto expected = 0.0;
    for (auto k = 0; k != std::ssize(cals); ++k) {
      auto [mrca, clade] = ref_mrca(r.tree(), cals[k].taxa);
      ASSERT_TRUE(clade || not cals[k].require_monophyly) << "batch " << i;
      expected += c.calibrations.calibrations[k].log_density_at(r.tree().at(mrca).t);
    }
    ASSERT_NEAR(r.log_calibration_prior(), expected, 1e-8) << "batch " << i;
    ASSERT_NEAR(r.log_coalescent_prior(), exact_coal(r), 1e-8) << "batch " << i;
  };
  auto topologies = std::map<std::string, int>{};
  for (auto i = 0; i != 150; ++i) {
    run->do_mcmc_steps(c.batch);
    ++topologies[topology_signature(run->tree())];
    check(*run, i);
  }
  EXPECT_GT(std::ssize(topologies), 5) << "the topology hardly moved";
  EXPECT_LT(run->exact_coalescent_batches_accepted(), run->exact_coalescent_batches_proposed())
      << "no batch was rolled back, so the rollback path was not exercised";

  // A checkpoint carries the calibrations (with the monophyly setting), the free topology and the correction
  auto api_tree = phylo_tree_to_api_tree(run->tree());
  auto api_tree_info = phylo_tree_to_api_tree_info(run->tree());
  auto api_params = run_to_api_params(*run);
  auto restored = std::make_unique<delphy::Run>(
      pool, std::mt19937{17}, 17, api_tree_and_tree_info_to_phylo_tree(api_tree.data(), api_tree_info.data()));
  apply_api_params_to_run(api_params.data(), *restored);
  restored->set_num_parts(1);
  ASSERT_TRUE(restored->exact_coalescent_correction());
  ASSERT_TRUE(restored->topology_moves_enabled());
  ASSERT_EQ(restored->calibrations().size(), std::ssize(cals));
  EXPECT_FALSE(restored->calibrations().calibrations[1].require_monophyly);
  // Same tolerance as the other checkpoint tests: the saved state is rounded at the 1e-9 relative level
  EXPECT_NEAR(restored->log_calibration_prior(), run->log_calibration_prior(),
              1e-6 * std::max(1.0, std::abs(run->log_calibration_prior())));
  EXPECT_NEAR(restored->log_coalescent_prior(), run->log_coalescent_prior(),
              1e-6 * std::max(1.0, std::abs(run->log_coalescent_prior())));
  for (auto i = 0; i != 50; ++i) {
    restored->do_mcmc_steps(c.batch);
    check(*restored, 150 + i);
  }
}

// ---------------------------------------------------------------------------------------------
// 5. Threshold partitioning: several parts with a free topology
// ---------------------------------------------------------------------------------------------

namespace {

// Builds a tree from a nested list of tip indices written as a Newick-like string of digits, e.g. "((0,1),2)".
// All tips at time 0, inner nodes 1 day older than their oldest child.
auto shape_tree(const std::string& shape, int num_tips) -> Phylo_tree {
  auto tree = Phylo_tree{2 * num_tips - 1};
  tree.ref_sequence = Real_sequence(1, Real_seq_letter::A);
  for (auto i = 0; i != num_tips; ++i) {
    tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = 0.0;
    tree.at(i).name = "S" + std::to_string(i);
  }
  auto next = static_cast<Node_index>(num_tips);
  auto pos = std::size_t{0};
  std::function<Node_index()> parse = [&]() -> Node_index {
    if (shape[pos] == '(') {
      ++pos;
      auto a = parse();
      ++pos;  // ','
      auto b = parse();
      ++pos;  // ')'
      auto node = next++;
      tree.at(node).children = {a, b};
      tree.at(a).parent = node;
      tree.at(b).parent = node;
      tree.at(node).t = std::min(tree.at(a).t, tree.at(b).t) - 1.0;
      tree.at(node).t_min = -std::numeric_limits<float>::max();
      tree.at(node).t_max = +std::numeric_limits<float>::max();
      return node;
    }
    auto end = shape.find_first_of(",)", pos);
    auto tip = std::stoi(shape.substr(pos, end - pos));
    pos = end;
    return tip;
  };
  tree.root = parse();
  tree.at(tree.root).parent = k_no_node;
  return tree;
}

// The tip sets of the cut points, as sorted name lists joined by '+'
auto cut_clades(const Phylo_tree& tree, int max_cut_tips, int min_cut_tips) -> std::vector<std::string> {
  auto result = std::vector<std::string>{};
  for (const auto& info : threshold_partition_stencil(tree, max_cut_tips, min_cut_tips)) {
    auto names = std::vector<std::string>{};
    auto stack = std::vector<Node_index>{info.cut_point};
    while (not stack.empty()) {
      auto node = stack.back();
      stack.pop_back();
      if (tree.at(node).is_tip()) { names.push_back(tree.at(node).name); }
      for (auto child : tree.at(node).children) { stack.push_back(child); }
    }
    std::ranges::sort(names);
    result.push_back(absl::StrJoin(names, "+"));
  }
  std::ranges::sort(result);
  return result;
}

// Summaries of a larger genealogy whose full topology has too many values for a two-sample test
struct Shape_summary {
  std::string root_split;  // tips below the smaller child of the root
  std::string cherries;    // number of cherries
  std::string clade_01;    // whether S0 and S1 form a cherry
  double root_age;
  double first_age;
  double length;           // total branch length
};

auto summarize_shape(const Phylo_tree& tree) -> Shape_summary {
  auto tips = calc_clade_tip_counts(tree);
  auto t_ref = -std::numeric_limits<double>::max();
  auto first = std::numeric_limits<double>::max();
  auto length = 0.0;
  auto cherries = 0;
  for (const auto& node : index_order_traversal(tree)) {
    if (tree.at(node).is_tip()) { t_ref = std::max(t_ref, tree.at(node).t); }
  }
  for (const auto& node : index_order_traversal(tree)) {
    if (node != tree.root) { length += tree.at(node).t - tree.at_parent_of(node).t; }
    if (tree.at(node).is_inner_node()) {
      first = std::min(first, t_ref - tree.at(node).t);
      if (tips[node] == 2) { ++cherries; }
    }
  }
  auto small = std::min(tips[tree.at_root().left_child()], tips[tree.at_root().right_child()]);
  auto pair = tree.at(0).parent == tree.at(1).parent;
  return {std::to_string(small), std::to_string(cherries), pair ? "yes" : "no",
          t_ref - tree.at_root().t, first, length};
}

struct Shape_stationarity {
  std::pair<double, int> root_split_chi2;
  std::pair<double, int> cherries_chi2;
  std::pair<double, int> clade_01_chi2;
  double z_root_age;
  double z_first_age;
  double z_length;
  int64_t topology_changes;
  double batch_acceptance;
  double mean_parts;
  int64_t blocked_by_partition;
};

// As `run_stationarity`, for summaries that stay testable with many tips, on a pool of `threads` threads
auto run_shape_stationarity(int chains, int iterations, const std::vector<double>& tip_times, const Chain_config& c,
                            uint64_t seed, int threads, const std::vector<double>& skygrid_x = {},
                            Type type = Type::k_staircase)
    -> Shape_stationarity {
  auto pool = ctpl::thread_pool{threads};
  auto g = std::mt19937_64{seed};
  auto ref_split = std::map<std::string, int>{}, mcmc_split = std::map<std::string, int>{};
  auto ref_cherries = std::map<std::string, int>{}, mcmc_cherries = std::map<std::string, int>{};
  auto ref_pair = std::map<std::string, int>{}, mcmc_pair = std::map<std::string, int>{};
  auto ref_root = std::vector<double>{}, mcmc_root = std::vector<double>{};
  auto ref_first = std::vector<double>{}, mcmc_first = std::vector<double>{};
  auto ref_length = std::vector<double>{}, mcmc_length = std::vector<double>{};
  auto changes = int64_t{0};
  auto proposed = int64_t{0}, accepted = int64_t{0};
  auto drawn = int64_t{0}, parts = int64_t{0}, blocked = int64_t{0};
  for (auto i = 0; i != chains; ++i) {
    auto config = c;
    if (not skygrid_x.empty()) { config.pop = ref_skygrid_draw(skygrid_x, type, c, g); }
    auto ref = summarize_shape(ref_kingman_tree(tip_times, *config.pop, g));
    ++ref_split[ref.root_split];
    ++ref_cherries[ref.cherries];
    ++ref_pair[ref.clade_01];
    ref_root.push_back(ref.root_age);
    ref_first.push_back(ref.first_age);
    ref_length.push_back(ref.length);

    if (not skygrid_x.empty()) { config.pop = ref_skygrid_draw(skygrid_x, type, c, g); }
    auto run = make_run(pool, ref_kingman_tree(tip_times, *config.pop, g), static_cast<uint32_t>(g()), config);
    auto last = topology_signature(run->tree());
    for (auto it = 0; it != iterations; ++it) {
      run->do_mcmc_steps(c.batch);
      auto now = topology_signature(run->tree());
      if (now != last) { ++changes; last = now; }
    }
    auto end = summarize_shape(run->tree());
    ++mcmc_split[end.root_split];
    ++mcmc_cherries[end.cherries];
    ++mcmc_pair[end.clade_01];
    mcmc_root.push_back(end.root_age);
    mcmc_first.push_back(end.first_age);
    mcmc_length.push_back(end.length);
    proposed += run->exact_coalescent_batches_proposed();
    accepted += run->exact_coalescent_batches_accepted();
    drawn += run->threshold_partitions_drawn();
    parts += run->threshold_partition_parts_drawn();
    for (const auto& counts : run->move_counts()) { blocked += counts.blocked_by_partition; }
  }
  return {two_sample_chi2(mcmc_split, ref_split), two_sample_chi2(mcmc_cherries, ref_cherries),
          two_sample_chi2(mcmc_pair, ref_pair), z_of_means(mcmc_root, ref_root), z_of_means(mcmc_first, ref_first),
          z_of_means(mcmc_length, ref_length), changes,
          proposed > 0 ? static_cast<double>(accepted) / proposed : 1.0,
          drawn > 0 ? static_cast<double>(parts) / drawn : 1.0, blocked};
}

auto print(const std::string& label, const Shape_stationarity& s) -> void {
  std::cerr << absl::StreamFormat(
      "EXACT-THRESHOLD %s: chi2 root split %.1f/%d, cherries %.1f/%d, S0S1 cherry %.1f/%d; z(root age) %+.2f, "
      "z(first age) %+.2f, z(length) %+.2f; %d topology changes, batch acceptance %.3f, mean parts %.2f, "
      "%d proposals blocked by the partition\n",
      label, s.root_split_chi2.first, s.root_split_chi2.second, s.cherries_chi2.first, s.cherries_chi2.second,
      s.clade_01_chi2.first, s.clade_01_chi2.second, s.z_root_age, s.z_first_age, s.z_length, s.topology_changes,
      s.batch_acceptance, s.mean_parts, s.blocked_by_partition);
}

auto expect_stationary(const Shape_stationarity& s) -> void {
  EXPECT_LT(s.root_split_chi2.first, chi2_gate(s.root_split_chi2.second));
  EXPECT_LT(s.cherries_chi2.first, chi2_gate(s.cherries_chi2.second));
  EXPECT_LT(s.clade_01_chi2.first, chi2_gate(s.clade_01_chi2.second));
  EXPECT_LT(std::abs(s.z_root_age), 4.0);
  EXPECT_LT(std::abs(s.z_first_age), 4.0);
  EXPECT_LT(std::abs(s.z_length), 4.0);
}

}  // namespace

TEST(Exact_threshold_partition_test, the_cut_points_are_the_maximal_clades_in_the_window) {
  auto balanced = shape_tree("(((0,1),(2,3)),((4,5),(6,7)))", 8);
  EXPECT_EQ(cut_clades(balanced, 2, 2), (std::vector<std::string>{"S0+S1", "S2+S3", "S4+S5", "S6+S7"}));
  EXPECT_EQ(cut_clades(balanced, 3, 2), (std::vector<std::string>{"S0+S1", "S2+S3", "S4+S5", "S6+S7"}));
  EXPECT_EQ(cut_clades(balanced, 4, 2), (std::vector<std::string>{"S0+S1+S2+S3", "S4+S5+S6+S7"}));
  EXPECT_EQ(cut_clades(balanced, 4, 3), (std::vector<std::string>{"S0+S1+S2+S3", "S4+S5+S6+S7"}));
  EXPECT_TRUE(cut_clades(balanced, 4, 5).empty());   // empty window
  EXPECT_TRUE(cut_clades(balanced, 8, 2).empty());   // no parent has more than 8 tips

  // A caterpillar has one cut point at most, and small clades hanging off the spine stay in the root part
  auto caterpillar = shape_tree("(((((0,1),2),3),4),5)", 6);
  EXPECT_EQ(cut_clades(caterpillar, 2, 2), (std::vector<std::string>{"S0+S1"}));
  EXPECT_EQ(cut_clades(caterpillar, 4, 2), (std::vector<std::string>{"S0+S1+S2+S3"}));
  EXPECT_TRUE(cut_clades(caterpillar, 4, 5).empty());

  // Mixed: the 3-tip clade is a cut point at m = 3; at m = 2 its cherry is, and the sibling tip is not (too small)
  auto mixed = shape_tree("(((0,1),2),((3,4),(5,(6,7))))", 8);
  EXPECT_EQ(cut_clades(mixed, 3, 2), (std::vector<std::string>{"S0+S1+S2", "S3+S4", "S5+S6+S7"}));
  EXPECT_EQ(cut_clades(mixed, 2, 2), (std::vector<std::string>{"S0+S1", "S3+S4", "S6+S7"}));
  EXPECT_EQ(cut_clades(mixed, 5, 3), (std::vector<std::string>{"S0+S1+S2", "S3+S4+S5+S6+S7"}));

  // The parts of the partition: no cut point lies below another, and the root part holds everything else
  auto partition = partition_tree(mixed, threshold_partition_stencil(mixed, 3, 2));
  ASSERT_EQ(std::ssize(partition.parts()), 4);
  auto total_inner = 0;
  for (const auto& part : partition.parts()) {
    for (auto node : index_order_traversal(part)) {
      if (part.at(node).is_inner_node()) { ++total_inner; }
    }
  }
  EXPECT_EQ(total_inner, 7);  // every inner node belongs to exactly one part
}

TEST(Exact_threshold_partition_test, free_topologies_on_several_parts_are_accepted_and_keep_their_partition) {
  auto pool = ctpl::thread_pool{3};
  auto g = std::mt19937_64{21};
  auto pop = std::make_shared<Const_pop_model>(k_n0);
  auto tree = ref_kingman_tree(std::vector<double>(40, 0.0), *pop, g);
  auto c = Chain_config{.pop = pop, .t_step = k_n0, .parts = 3, .batch = 60, .threshold = true};

  {  // accepted in either order; `reassemble` checks after every batch that the partition is the one drawn
    auto run = make_run(pool, tree, 7, c);
    EXPECT_TRUE(run->exact_coalescent_correction());
    EXPECT_NO_THROW(run->do_mcmc_steps(60 * 40));
    EXPECT_GT(run->threshold_partitions_drawn(), 30);
    EXPECT_GT(run->threshold_partition_parts_drawn(), 2 * run->threshold_partitions_drawn());
    auto blocked = int64_t{0};
    for (const auto& counts : run->move_counts()) { blocked += counts.blocked_by_partition; }
    EXPECT_GT(blocked, 0) << "the root-part constraint was never exercised";

    auto fixed = c;
    fixed.topology = false;
    auto run2 = make_run(pool, tree, 7, fixed);
    EXPECT_NO_THROW(run2->set_topology_moves_enabled(true));
    EXPECT_NO_THROW(run2->do_mcmc_steps(600));
  }
  {  // without threshold partitioning several parts stay refused
    auto stencil = c;
    stencil.threshold = false;
    EXPECT_THROW(make_run(pool, tree, 7, stencil), std::invalid_argument);
  }
  {  // age calibrations on a free topology are refused, in any order
    auto config = std::string{
        "version 1\n\ncalibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
        "  distribution uniform\n  youngest 0.1\n  oldest 100\nend\n"};
    auto is = std::istringstream{config};
    auto calibrations = read_calibration_config(is);
    auto with = c;
    with.calibrations = calibrations;
    EXPECT_THROW(make_run(pool, tree, 7, with), std::invalid_argument);

    auto run = make_run(pool, tree, 7, c);
    EXPECT_THROW(run->set_calibrations(calibrations), std::invalid_argument);

    auto fixed = c;
    fixed.topology = false;
    fixed.calibrations = calibrations;
    auto run2 = make_run(pool, tree, 7, fixed);  // a fixed topology is fine
    EXPECT_NO_THROW(run2->do_mcmc_steps(300));
    EXPECT_THROW(run2->set_topology_moves_enabled(true), std::invalid_argument);
  }
  {  // invalid parameters
    auto run = make_run(pool, tree, 7, c);
    EXPECT_THROW(run->set_threshold_partition_params(0.0, 1.0, 4, 0.0), std::invalid_argument);
    EXPECT_THROW(run->set_threshold_partition_params(0.8, 0.5, 4, 0.0), std::invalid_argument);
    EXPECT_THROW(run->set_threshold_partition_params(0.5, 1.0, 0, 0.0), std::invalid_argument);
    EXPECT_THROW(run->set_threshold_partition_params(0.5, 1.0, 4, 1.5), std::invalid_argument);
  }
}

TEST(Exact_threshold_partition_test, the_coalescent_term_stays_exact_through_parallel_batches_and_rollbacks) {
  auto pool = ctpl::thread_pool{4};
  auto g = std::mt19937_64{23};
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    auto c = Chain_config{.t_step = 2.0 * k_n0, .parts = 3, .batch = 80, .tau = 4.0, .threshold = true};
    c.pop = std::make_shared<Skygrid_pop_model>(
        staircase_knots(), std::vector<double>{std::log(400.0), std::log(2000.0), std::log(300.0),
                                                std::log(1500.0), std::log(800.0)}, type);
    auto tip_times = std::vector<double>{};
    for (auto i = 0; i != 30; ++i) { tip_times.push_back(-30.0 * (i % 7)); }
    auto run = make_run(pool, ref_kingman_tree(tip_times, *c.pop, g), 13, c);
    auto exact = [](const delphy::Run& r) {
      return calc_exact_coalescent_log_prior(coalescent_event_times_of(r.tree()), r.pop_model());
    };
    auto topologies = std::map<std::string, int>{};
    for (auto i = 0; i != 120; ++i) {
      run->do_mcmc_steps(c.batch);
      ++topologies[topology_signature(run->tree())];
      ASSERT_NEAR(run->log_coalescent_prior(), exact(*run), 1e-8) << "batch " << i;

      auto fresh_config = c;
      fresh_config.threshold = false;
      fresh_config.parts = 1;
      fresh_config.pop = std::make_shared<Skygrid_pop_model>(
          static_cast<const Skygrid_pop_model&>(run->pop_model()));
      auto fresh = make_run(pool, run->tree(), 5, fresh_config);
      fresh->set_hky_kappa(run->hky_kappa());
      fresh->set_hky_pi(run->hky_pi());
      ASSERT_NEAR(fresh->log_G(), run->log_G(), 1e-6 * std::max(1.0, std::abs(run->log_G())));
      ASSERT_NEAR(fresh->log_coalescent_prior(), run->log_coalescent_prior(), 1e-8);
      ASSERT_NEAR(fresh->log_other_priors(), run->log_other_priors(), 1e-8);
    }
    EXPECT_GT(std::ssize(topologies), 20) << "the topology hardly moved";
    EXPECT_GT(run->exact_coalescent_batches_accepted(), 0);
    EXPECT_LT(run->exact_coalescent_batches_accepted(), run->exact_coalescent_batches_proposed())
        << "no batch was rolled back, so the rollback path was not exercised";
    EXPECT_GT(run->threshold_partition_parts_drawn(), 2 * run->threshold_partitions_drawn());
  }
}

TEST(Exact_threshold_partition_test, the_number_of_threads_does_not_change_the_chain) {
  auto g = std::mt19937_64{25};
  auto pop = std::make_shared<Const_pop_model>(k_n0);
  auto tree = ref_kingman_tree(std::vector<double>(50, 0.0), *pop, g);
  auto c = Chain_config{.pop = pop, .t_step = k_n0, .parts = 4, .batch = 100, .threshold = true};
  auto states = std::vector<std::string>{};
  for (auto threads : {1, 2, 5}) {
    auto pool = ctpl::thread_pool{threads};
    auto run = make_run(pool, tree, 9, c);
    run->do_mcmc_steps(100 * 30);
    auto state = topology_signature(run->tree());
    for (const auto& node : index_order_traversal(run->tree())) {
      state += absl::StrFormat(" %.17g", run->tree().at(node).t);
    }
    states.push_back(state);
  }
  EXPECT_EQ(states[0], states[1]);
  EXPECT_EQ(states[0], states[2]);
}

namespace {

// Mutations drawn on every branch of `tree` at `rate` per day (Jukes-Cantor), on a reference of `L` sites
auto add_random_mutations(Phylo_tree& tree, int L, double rate, std::mt19937_64& g) -> void {
  tree.ref_sequence = Real_sequence(L, Real_seq_letter::A);
  for (auto l = 0; l != L; ++l) { tree.ref_sequence[l] = static_cast<Real_seq_letter>(g() % 4); }
  auto seqs = Node_vector<Real_sequence>(std::ssize(tree));
  seqs[tree.root] = tree.ref_sequence;
  auto stack = std::vector<Node_index>{tree.root};
  while (not stack.empty()) {
    auto parent = stack.back();
    stack.pop_back();
    for (auto child : tree.at(parent).children) {
      auto seq = seqs[parent];
      auto t0 = tree.at(parent).t, t1 = tree.at(child).t;
      auto k = std::poisson_distribution<int>{rate * (t1 - t0)}(g);
      auto times = std::vector<double>{};
      for (auto i = 0; i != k; ++i) { times.push_back(std::uniform_real_distribution<double>{t0, t1}(g)); }
      std::ranges::sort(times);
      auto& ms = tree.at(child).mutations;
      for (auto t : times) {
        if (not (t > t0 && t < t1)) { continue; }
        auto l = static_cast<Site_index>(g() % L);
        auto from = seq[l];
        auto to = static_cast<Real_seq_letter>((static_cast<int>(from) + 1 + g() % 3) % 4);
        ms.push_back(Mutation{from, l, to, t});
        seq[l] = to;
      }
      seqs[child] = std::move(seq);
      stack.push_back(child);
    }
  }
}

}  // namespace

TEST(Exact_threshold_partition_test, parallel_batches_with_mutation_histories_keep_every_derived_quantity) {
  auto pool = ctpl::thread_pool{4};
  auto g = std::mt19937_64{27};
  auto pop = std::make_shared<Const_pop_model>(k_n0);
  auto tree = ref_kingman_tree(std::vector<double>(48, 0.0), *pop, g);
  auto L = 300;
  auto rate = 3.0 / k_n0;  // about three mutations per branch of mean length
  add_random_mutations(tree, L, rate, g);
  assert_phylo_tree_integrity(tree);
  auto c = Chain_config{.pop = pop, .t_step = k_n0, .parts = 4, .batch = 100, .threshold = true};
  auto run = make_run(pool, tree, 17, c);
  run->set_mu(rate / L);
  auto topologies = std::map<std::string, int>{};
  for (auto i = 0; i != 60; ++i) {
    run->do_mcmc_steps(c.batch);
    ++topologies[topology_signature(run->tree())];
    auto fresh_config = c;
    fresh_config.threshold = false;
    fresh_config.parts = 1;
    auto fresh = make_run(pool, run->tree(), 5, fresh_config);
    fresh->set_mu(run->mu());
    fresh->set_hky_kappa(run->hky_kappa());
    fresh->set_hky_pi(run->hky_pi());
    ASSERT_NEAR(fresh->log_G(), run->log_G(), 1e-6 * std::max(1.0, std::abs(run->log_G()))) << "batch " << i;
    ASSERT_NEAR(fresh->log_coalescent_prior(), run->log_coalescent_prior(), 1e-8) << "batch " << i;
  }
  auto spr = run->move_counts()[static_cast<int>(Move_kind::spr1)];
  EXPECT_GT(spr.accepted, 0);
  EXPECT_GT(std::ssize(topologies), 10);
  EXPECT_LT(run->exact_coalescent_batches_accepted(), run->exact_coalescent_batches_proposed());
}

TEST(Exact_threshold_partition_test, eight_tips_same_date_constant_population_two_parts) {
  auto chains = estd::is_debug_enabled ? 200 : 2000;
  auto c = Chain_config{.pop = std::make_shared<Const_pop_model>(k_n0), .t_step = 2.0 * k_n0, .parts = 2,
                        .batch = 20, .threshold = true};
  auto tips = std::vector<double>(8, 0.0);
  auto s = run_shape_stationarity(chains, 30, tips, c, 201, 2);
  print("8 tips, constant, 2 parts", s);
  expect_stationary(s);
  EXPECT_GT(s.mean_parts, 1.5);
  EXPECT_GT(s.blocked_by_partition, 0);

  // The same grid without the correction is biased in the node ages, which shows the test has power
  if (not estd::is_debug_enabled) {
    c.correct = false;
    auto cells = run_shape_stationarity(chains, 30, tips, c, 202, 2);
    print("8 tips, constant, 2 parts, cell prior", cells);
    EXPECT_GT(std::max({std::abs(cells.z_root_age), std::abs(cells.z_first_age), std::abs(cells.z_length)}), 6.0);
  }
}

TEST(Exact_threshold_partition_test, sixteen_tips_at_several_dates_four_parts) {
  auto chains = estd::is_debug_enabled ? 150 : 1500;
  auto c = Chain_config{.pop = std::make_shared<Const_pop_model>(k_n0), .t_step = 1.5 * k_n0, .parts = 4,
                        .batch = 40, .threshold = true};
  auto tips = std::vector<double>{};
  for (auto i = 0; i != 16; ++i) { tips.push_back(-100.0 * (i % 5)); }
  auto s = run_shape_stationarity(chains, 30, tips, c, 203, 4);
  print("16 tips at 5 dates, constant, 4 parts", s);
  expect_stationary(s);
  EXPECT_GT(s.mean_parts, 2.0);
}

TEST(Exact_threshold_partition_test, twelve_tips_skygrid_with_the_curve_sampled_three_parts) {
  auto chains = estd::is_debug_enabled ? 150 : 1500;
  for (auto type : {Type::k_staircase, Type::k_log_linear}) {
    auto c = Chain_config{.t_step = 1.5 * k_n0, .parts = 3, .batch = 30, .tau = 2.0, .inv_nbar_alpha = 4.0,
                          .inv_nbar_beta = 3.0 * k_n0, .threshold = true};
    auto tips = std::vector<double>{0.0, 0.0, 0.0, 0.0, -100.0, -100.0, -250.0, -250.0, -400.0, -600.0, -600.0, -800.0};
    auto s = run_shape_stationarity(chains, 30, tips, c, 204, 3, staircase_knots(), type);
    print(type == Type::k_staircase ? "12 tips, staircase Skygrid, 3 parts" : "12 tips, log-linear Skygrid, 3 parts",
          s);
    expect_stationary(s);
  }
}

}  // namespace delphy
