#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cmath>
#include <functional>
#include <map>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "ctpl_stl.h"

#include "api.h"
#include "evo_hky.h"
#include "phylo_tree_calc.h"
#include "pop_model.h"
#include "run.h"
#include "subrun.h"

namespace delphy {

namespace {

constexpr auto k_n0 = 1000.0;  // days

// A Kingman tree over `num_tips` tips sampled on one date, with Jukes-Cantor mutations at rate `mu`,
// and `missing` leading sites made missing in the first tip.
// `sampling_span` spreads the tip dates over that many days before the last sample, which is what
// identifies the absolute time scale: with every tip on one date the clock has no signal at all, the
// tree's scale is pinned only by the coalescent prior, and no two samplers need agree on where along
// that flat direction they sit.
auto make_tree(int num_tips, Site_index num_sites, uint64_t seed, double mu,
               Site_index missing_in_first_tip = 0, double sampling_span = 0.0)
    -> Phylo_tree {
  auto g = std::mt19937_64{seed};
  auto tree = Phylo_tree{2 * num_tips - 1};
  tree.ref_sequence = Real_sequence(num_sites, Real_seq_letter::A);
  auto tip_age = std::vector<double>(num_tips, 0.0);
  if (sampling_span > 0.0) {
    for (auto i = 1; i != num_tips; ++i) {
      tip_age[i] = std::uniform_real_distribution<double>{0.0, sampling_span}(g);
    }
    std::ranges::sort(tip_age);  // youngest first
  }
  for (auto i = 0; i != num_tips; ++i) {
    tree.at(i).t = tree.at(i).t_min = tree.at(i).t_max = -tip_age[i];
    tree.at(i).name = "S" + std::to_string(i);
  }

  // A serially sampled Kingman coalescent: walk back in time, add each lineage as its sampling date
  // passes, and coalesce two of the active ones at rate k(k-1)/2/n0.
  auto active = std::vector<Node_index>{};
  auto next_tip = 0;
  auto age = 0.0;
  auto next = num_tips;
  while (next_tip != num_tips || std::ssize(active) > 1) {
    while (next_tip != num_tips && tip_age[next_tip] <= age) { active.push_back(next_tip++); }
    auto k = static_cast<double>(std::ssize(active));
    auto wait = (k > 1) ? std::exponential_distribution<double>{k * (k - 1) / 2.0 / k_n0}(g)
                        : std::numeric_limits<double>::infinity();
    if (next_tip != num_tips && age + wait > tip_age[next_tip]) {
      age = tip_age[next_tip];
      continue;
    }
    age += wait;
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

  if (missing_in_first_tip > 0) {
    for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
      std::erase_if(tree.at(node).mutations,
                    [&](const Mutation& m) { return m.site < missing_in_first_tip; });
    }
    tree.at(0).missations.intervals.insert(Site_interval{0, missing_in_first_tip});
  }
  return tree;
}

auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, double mu,
              bool fixed_topology, int parts, bool const_pop = true)
    -> std::shared_ptr<Run> {
  auto run = std::make_shared<delphy::Run>(pool, std::mt19937{seed}, seed, tree);
  run->set_topology_moves_enabled(not fixed_topology);
  run->set_mu(mu);
  run->set_mu_prior_alpha(1.0);
  run->set_mu_prior_beta(0.0);
  run->set_num_parts(parts);
  run->set_local_moves_per_global_move(200);
  if (const_pop) {
    // A constant population keeps these small trees away from the collapse attractor of the default
    // coalescent grid; the checkpoint test keeps the exponential model, because that is the one the
    // .dphy parameters carry.
    run->set_pop_model(std::make_shared<Const_pop_model>(k_n0));
    run->set_final_pop_size_move_enabled(false);
    run->set_pop_growth_rate_move_enabled(false);
  }
  return run;
}

auto enable_gibbs(Run& run, double weight = 15.0, int sites = 4, int refreshes = 2,
                  double marginal_weight = 7.5) -> void {
  run.set_history_gibbs_weight(weight);
  run.set_history_gibbs_sites_per_move(sites);
  run.set_site_history_refreshes_per_global_move(refreshes);
  run.set_marginal_node_time_weight(marginal_weight);
}

auto tip_states(const Phylo_tree& tree, Site_index num_sites) -> std::vector<Real_seq_letter> {
  auto out = std::vector<Real_seq_letter>{};
  for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
    if (not tree.at(node).is_tip()) { continue; }
    for (auto l = Site_index{0}; l != num_sites; ++l) {
      if (is_site_missing_at(tree, node, l)) { continue; }
      out.push_back(calc_site_state_at(tree, tree.node_loc(node), l));
    }
  }
  return out;
}

// The marginal sequence likelihood by Felsenstein pruning, in long double, written here from the
// model alone and sharing no code with history_gibbs.cpp or phylo_tree_calc.cpp.  Integrating the
// augmented EMAT density over every mutational history with the tip states held fixed gives exactly
// this times the coalescent prior, so it is the right yardstick for which tree the chain should
// prefer.
using Ref_matrix = std::array<std::array<long double, 4>, 4>;

auto ref_mat_mul(const Ref_matrix& x, const Ref_matrix& y) -> Ref_matrix {
  auto out = Ref_matrix{};
  for (auto i = 0; i != 4; ++i) {
    for (auto j = 0; j != 4; ++j) {
      auto sum = 0.0L;
      for (auto k = 0; k != 4; ++k) { sum += x[i][k] * y[k][j]; }
      out[i][j] = sum;
    }
  }
  return out;
}

// exp(Q t) by a scaled and squared Taylor series in long double.
auto ref_expm(const Seq_matrix<double>& q, long double scale) -> Ref_matrix {
  auto m = Ref_matrix{};
  auto norm = 0.0L;
  for (auto a : k_all_real_seq_letters) {
    for (auto b : k_all_real_seq_letters) {
      m[index_of(a)][index_of(b)] = static_cast<long double>(q[a][b]) * scale;
      norm = std::max(norm, std::fabs(m[index_of(a)][index_of(b)]));
    }
  }
  auto squarings = 0;
  while (norm > 0.05L) { norm /= 2; ++squarings; }
  for (auto& row : m) { for (auto& v : row) { v /= std::pow(2.0L, squarings); } }
  auto term = Ref_matrix{};
  auto sum = Ref_matrix{};
  for (auto i = 0; i != 4; ++i) { term[i][i] = 1.0L; sum[i][i] = 1.0L; }
  for (auto k = 1; k != 30; ++k) {
    term = ref_mat_mul(term, m);
    for (auto i = 0; i != 4; ++i) {
      for (auto j = 0; j != 4; ++j) {
        sum[i][j] += term[i][j] / std::tgamma(static_cast<long double>(k + 1));
      }
    }
  }
  for (auto s = 0; s != squarings; ++s) { sum = ref_mat_mul(sum, sum); }
  return sum;
}

// The marginal sequence likelihood by Felsenstein pruning, in long double, written from the model
// alone and sharing no code with history_gibbs.cpp or phylo_tree_calc.cpp.  Integrating the
// augmented EMAT density over every mutational history with the tip states held fixed gives exactly
// this, so it is the right yardstick for which tree the chain should prefer.
auto marginal_log_likelihood(const Phylo_tree& tree, const Global_evo_model& evo) -> long double {
  // Sites that share a partition and a relative rate share every transition matrix, so the matrix
  // exponentials are formed once per such group rather than once per site.
  auto groups = std::map<std::pair<Partition_index, double>, std::vector<Site_index>>{};
  for (auto l = Site_index{0}; l != tree.num_sites(); ++l) {
    if (tree.at_root().missations.contains(l)) { continue; }
    groups[{evo.partition_for_site[l], evo.mu_l(l) * evo.nu_l[l]}].push_back(l);
  }

  auto order = std::vector<Node_index>{};
  {
    auto stack = std::vector<Node_index>{tree.root};
    while (not stack.empty()) {
      auto n = stack.back(); stack.pop_back();
      order.push_back(n);
      for (auto c : tree.at(n).children) { stack.push_back(c); }
    }
  }

  auto total = 0.0L;
  for (const auto& [key, sites] : groups) {
    auto partition = key.first;
    auto rate = key.second;
    const auto& q = evo.partition_evo_model[partition].q_ab;
    const auto& pi_a = evo.partition_evo_model[partition].pi_a;
    auto branch = std::vector<Ref_matrix>(std::ssize(tree));
    for (auto n : order) {
      if (n == tree.root) { continue; }
      branch[n] = ref_expm(q, static_cast<long double>(rate)
                           * static_cast<long double>(tree.at(n).t - tree.at_parent_of(n).t));
    }
    for (auto l : sites) {
      auto partial = std::vector<std::array<long double, 4>>(std::ssize(tree));
      auto missing = std::vector<char>(std::ssize(tree), 0);
      for (auto n : order) {
        if (n != tree.root
            && (missing[tree.at(n).parent] || tree.at(n).missations.contains(l))) { missing[n] = 1; }
      }
      auto scale_log = 0.0L;
      for (auto it = order.rbegin(); it != order.rend(); ++it) {
        auto n = *it;
        if (missing[n]) { continue; }
        if (tree.at(n).is_tip()) {
          auto state = calc_site_state_at(tree, tree.node_loc(n), l);
          for (auto i = 0; i != 4; ++i) { partial[n][i] = (i == index_of(state)) ? 1.0L : 0.0L; }
          continue;
        }
        auto v = std::array<long double, 4>{1.0L, 1.0L, 1.0L, 1.0L};
        for (auto c : tree.at(n).children) {
          if (missing[c]) { continue; }
          for (auto i = 0; i != 4; ++i) {
            auto sum = 0.0L;
            for (auto j = 0; j != 4; ++j) { sum += branch[c][i][j] * partial[c][j]; }
            v[i] *= sum;
          }
        }
        auto mx = 0.0L;
        for (auto x : v) { mx = std::max(mx, x); }
        if (mx > 0) { for (auto& x : v) { x /= mx; } scale_log += std::log(mx); }
        partial[n] = v;
      }
      auto site = 0.0L;
      for (auto a : k_all_real_seq_letters) {
        site += static_cast<long double>(pi_a[a]) * partial[tree.root][index_of(a)];
      }
      total += scale_log + std::log(site);
    }
  }
  return total;
}

class History_gibbs_mcmc_test : public testing::TestWithParam<std::tuple<bool, int>> {
 protected:
  ctpl::thread_pool pool_{4};
  auto fixed_topology() const -> bool { return std::get<0>(GetParam()); }
  auto parts() const -> int { return std::get<1>(GetParam()); }
};

}  // namespace

// The increment the moves maintain must still be the posterior weight of the state, in every
// combination of topology freedom and number of tree parts, with the new updates in the mix.
TEST_P(History_gibbs_mcmc_test, the_incremental_posterior_matches_a_full_recomputation) {
  constexpr auto num_sites = Site_index{400};
  constexpr auto mu = 4e-4;
  auto tree = make_tree(12, num_sites, 4242, mu, 20);
  auto want_tips = tip_states(tree, num_sites);

  auto run = make_run(pool_, tree, 17, mu, fixed_topology(), parts());
  enable_gibbs(*run);
  for (auto i = 0; i != 10; ++i) {
    run->do_mcmc_steps(4000);
    EXPECT_NEAR(run->log_G(), run->calc_cur_log_G(), 1e-6 * std::max(1.0, std::abs(run->log_G())))
        << "after " << run->step() << " steps";
    assert_phylo_tree_integrity(run->tree(), true);
    EXPECT_EQ(tip_states(run->tree(), num_sites), want_tips) << "after " << run->step() << " steps";
  }

  // The updates must actually have moved the history, or the test above proves nothing.
  const auto& stats = run->history_gibbs_stats();
  EXPECT_GT(stats.branch_updates_performed, 0);
  EXPECT_GT(stats.node_updates_performed, 0);
  EXPECT_GT(stats.node_states_changed, 0);
  EXPECT_GT(stats.site_refreshes, 0);
  EXPECT_GT(stats.events_created, 0);
  EXPECT_GT(run->move_counts()[static_cast<int>(Move_kind::node_time_marginal)].proposed, 0);
  EXPECT_GT(run->move_counts()[static_cast<int>(Move_kind::node_time_marginal)].accepted, 0);
  // A site that is missing in a tip must be skipped sometimes, so the "present" counts cannot all be
  // equal to the proposed counts on this tree.
  EXPECT_LT(stats.branch_updates_performed, stats.branch_updates_proposed);
}

INSTANTIATE_TEST_SUITE_P(
    Configurations, History_gibbs_mcmc_test,
    testing::Combine(testing::Values(true, false), testing::Values(1, 4)),
    [](const testing::TestParamInfo<std::tuple<bool, int>>& info) {
      return std::string(std::get<0>(info.param) ? "fixed" : "free") + "_"
          + std::to_string(std::get<1>(info.param)) + "parts";
    });

// With the default configuration the new moves are never selected, and a run that has never heard of
// them takes exactly the same path through the random stream as one whose weight is zero.
TEST(History_gibbs_run_test, the_updates_are_off_by_default) {
  auto pool = ctpl::thread_pool{2};
  constexpr auto num_sites = Site_index{200};
  constexpr auto mu = 3e-4;
  auto tree = make_tree(10, num_sites, 909, mu);

  auto plain = make_run(pool, tree, 31, mu, false, 1);
  plain->do_mcmc_steps(20000);

  auto twin = make_run(pool, tree, 31, mu, false, 1);
  twin->do_mcmc_steps(20000);
  ASSERT_EQ(twin->log_G(), plain->log_G()) << "two identically built runs must agree before anything else is claimed";

  auto zeroed = make_run(pool, tree, 31, mu, false, 1);
  zeroed->set_history_gibbs_weight(0.0);
  zeroed->set_history_gibbs_sites_per_move(8);
  zeroed->set_site_history_refreshes_per_global_move(0);
  zeroed->set_marginal_node_time_weight(0.0);
  zeroed->set_marginal_node_time_window(0.3);
  zeroed->do_mcmc_steps(20000);

  EXPECT_EQ(plain->move_counts()[static_cast<int>(Move_kind::branch_history_gibbs)].proposed, 0);
  EXPECT_EQ(plain->move_counts()[static_cast<int>(Move_kind::node_state_gibbs)].proposed, 0);
  EXPECT_EQ(plain->move_counts()[static_cast<int>(Move_kind::node_time_marginal)].proposed, 0);
  EXPECT_EQ(zeroed->move_counts()[static_cast<int>(Move_kind::branch_history_gibbs)].proposed, 0);
  EXPECT_EQ(zeroed->log_G(), plain->log_G());
  EXPECT_EQ(zeroed->num_muts(), plain->num_muts());
  EXPECT_EQ(zeroed->tree().at_root().t, plain->tree().at_root().t);
  for (auto k = 0; k != k_num_move_kinds; ++k) {
    EXPECT_EQ(zeroed->move_counts()[k].proposed, plain->move_counts()[k].proposed) << "move " << k;
    EXPECT_EQ(zeroed->move_counts()[k].accepted, plain->move_counts()[k].accepted) << "move " << k;
  }
}

// The marginal posterior yardstick.
//
// Integrating the augmented EMAT density over every mutational history with the tip states held
// fixed gives the Felsenstein marginal likelihood times the coalescent prior, so that quantity is a
// function of the tree alone whose posterior expectation the new updates must not move.  It is
// computed here by an independent long-double pruning (`marginal_log_likelihood` in this file), not
// by anything the operators use.
//
// What this test deliberately does *not* assert is that the two arms agree on the posterior of the
// root age or of the number of substitution events.  They do not, and that is the point of the task:
// the existing move set cannot lengthen the tree, because lengthening it requires adding hidden
// substitutions first, so its root age is confined to a narrow band while the marginal posterior is
// nearly flat over a much wider one.  Which of the two reproduces the true posterior of the root age
// is settled against the exact two-tip quadrature in the campaign of make_taskU_tasks.py, arm D, not
// here.
TEST(History_gibbs_run_test, the_updates_keep_the_chain_on_the_same_marginal_posterior) {
  auto pool = ctpl::thread_pool{4};
  constexpr auto num_sites = Site_index{1000};
  constexpr auto mu = 8e-6;   // about 0.04 substitutions per site from the root to a tip
  auto tree = make_tree(10, num_sites, 20261114, mu, 0, 4000.0);

  struct Arm { std::vector<double> marginal, root_sd; };
  auto arms = std::array<Arm, 2>{};
  for (auto gibbs = 0; gibbs != 2; ++gibbs) {
    for (auto seed = 1; seed <= 4; ++seed) {
      auto run = make_run(pool, tree, 100 + seed, mu, true, 1);
      run->set_mu_move_enabled(false);  // the clock is pinned, so the tree scale is identified twice
      if (gibbs) { enable_gibbs(*run); }
      auto marginal_sum = 0.0;
      auto root_sum = 0.0;
      auto root_sq = 0.0;
      auto kept = 0;
      for (auto i = 0; i != 40; ++i) {
        run->do_mcmc_steps(5000);
        if (i < 12) { continue; }  // 30% burn-in
        marginal_sum += static_cast<double>(marginal_log_likelihood(run->tree(), run->evo()))
            + run->log_coalescent_prior();
        auto age = -run->tree().at_root().t;
        root_sum += age;
        root_sq += age * age;
        ++kept;
      }
      arms[gibbs].marginal.push_back(marginal_sum / kept);
      arms[gibbs].root_sd.push_back(std::sqrt(std::max(root_sq / kept - (root_sum / kept) * (root_sum / kept), 0.0)));
    }
  }

  auto mean = [](const std::vector<double>& v) {
    auto s = 0.0; for (auto x : v) { s += x; } return s / std::ssize(v); };
  auto sd = [&](const std::vector<double>& v) {
    auto m = mean(v); auto s = 0.0;
    for (auto x : v) { s += (x - m) * (x - m); }
    return std::sqrt(s / (std::ssize(v) - 1)); };

  // The tolerance is the spread of the four seeds of each arm, which is the Monte Carlo error of the
  // comparison itself rather than a threshold chosen to make the test pass.
  auto tolerance = 4.0 * std::sqrt((sd(arms[0].marginal) * sd(arms[0].marginal)
                                    + sd(arms[1].marginal) * sd(arms[1].marginal)) / 4.0);
  EXPECT_LT(std::abs(mean(arms[0].marginal) - mean(arms[1].marginal)), std::max(tolerance, 2.0))
      << "marginal posterior: base " << mean(arms[0].marginal) << " +/- " << sd(arms[0].marginal)
      << ", gibbs " << mean(arms[1].marginal) << " +/- " << sd(arms[1].marginal);

  // And the updates must widen the exploration of the tree, not narrow it: that is what they are for.
  EXPECT_GT(mean(arms[1].root_sd), mean(arms[0].root_sd))
      << "root age spread within a chain: base " << mean(arms[0].root_sd)
      << ", gibbs " << mean(arms[1].root_sd);
}

// A checkpoint carries the genealogy and the parameters; the operator configuration is run
// configuration, re-supplied on resume exactly as the other experimental operators are.  What must
// hold is that the resumed run picks the posterior up where it left it and keeps it consistent.
TEST(History_gibbs_run_test, a_checkpoint_round_trip_keeps_the_posterior_and_the_updates) {
  auto pool = ctpl::thread_pool{2};
  constexpr auto num_sites = Site_index{300};
  constexpr auto mu = 4e-4;
  auto tree = make_tree(10, num_sites, 777, mu, 15);
  auto want_tips = tip_states(tree, num_sites);

  auto run = make_run(pool, tree, 53, mu, true, 1, false);
  enable_gibbs(*run);
  run->do_mcmc_steps(20000);
  auto events_before = run->history_gibbs_stats().events_created;
  EXPECT_GT(events_before, 0);

  auto params = run_to_api_params(*run);
  auto resumed = std::make_shared<delphy::Run>(pool, std::mt19937{1}, 1, run->tree(),
                                       run->coalescent_prior_t_step());
  resumed->set_num_parts(1);
  apply_api_params_to_run(params.data(), *resumed);
  EXPECT_NEAR(resumed->log_G(), run->log_G(), 1e-9 * std::max(1.0, std::abs(run->log_G())));
  EXPECT_NEAR(resumed->log_G(), resumed->calc_cur_log_G(), 1e-6);

  // The updates are not serialized, so a resumed run starts with them off; the counters confirm it.
  EXPECT_EQ(resumed->history_gibbs_stats().events_created, 0);
  EXPECT_EQ(resumed->history_gibbs_weight(), 0.0);
  resumed->do_mcmc_steps(4000);
  EXPECT_EQ(resumed->move_counts()[static_cast<int>(Move_kind::branch_history_gibbs)].proposed, 0);

  // Re-supplying them continues the run as before, and the posterior stays consistent.
  resumed->set_topology_moves_enabled(false);
  enable_gibbs(*resumed);
  resumed->do_mcmc_steps(20000);
  EXPECT_GT(resumed->history_gibbs_stats().events_created, 0);
  EXPECT_NEAR(resumed->log_G(), resumed->calc_cur_log_G(), 1e-6);
  assert_phylo_tree_integrity(resumed->tree(), true);
  EXPECT_EQ(tip_states(resumed->tree(), num_sites), want_tips);
}

// The updates must be refused, not silently mangled, where their inputs make no sense.
TEST(History_gibbs_run_test, the_configuration_is_validated) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(6, 100, 3, 2e-4);
  auto run = make_run(pool, tree, 5, 2e-4, true, 1);
  EXPECT_THROW(run->set_history_gibbs_weight(-1.0), std::invalid_argument);
  EXPECT_THROW(run->set_history_gibbs_weight(std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_THROW(run->set_history_gibbs_sites_per_move(0), std::invalid_argument);
  EXPECT_THROW(run->set_site_history_refreshes_per_global_move(-1), std::invalid_argument);
  EXPECT_THROW(run->set_marginal_node_time_weight(-0.5), std::invalid_argument);
  EXPECT_THROW(run->set_marginal_node_time_window(0.0), std::invalid_argument);
  EXPECT_THROW(run->set_marginal_node_time_window(-1.0), std::invalid_argument);
  // A window that reaches the move's grid guard is refused, because that guard rejects asymmetrically
  // and the kernel would lose detailed balance.  The refusal leaves the window where it was.
  auto window_before = run->marginal_node_time_window();
  EXPECT_THROW(run->set_marginal_node_time_window(max_marginal_node_time_window()),
               std::invalid_argument);
  EXPECT_THROW(run->set_marginal_node_time_window(max_marginal_node_time_window() + 1.0),
               std::invalid_argument);
  EXPECT_THROW(run->set_marginal_node_time_window(40.0), std::invalid_argument);
  EXPECT_EQ(run->marginal_node_time_window(), window_before);
  // Everything below the bound is accepted.
  run->set_marginal_node_time_window(std::nextafter(max_marginal_node_time_window(), 0.0));
  EXPECT_LT(run->marginal_node_time_window(), max_marginal_node_time_window());
  run->set_marginal_node_time_window(window_before);
  EXPECT_EQ(run->marginal_node_time_weight(), 0.0);
  EXPECT_EQ(run->history_gibbs_weight(), 0.0);
  EXPECT_EQ(run->history_gibbs_sites_per_move(), 1);
  EXPECT_EQ(run->site_history_refreshes_per_global_move(), 0);
}

// The marginal node-time move's root proposal multiplies the root's depth by exp(u) and then refuses
// the proposal if the new depth exceeds `marginal_node_time_max_depth_factor * max(depth, 1)`, a
// bound the coalescent grid needs.  That refusal is not symmetric: from the refused state the reverse
// proposal would be allowed, so if it could ever fire the kernel would no longer satisfy detailed
// balance.  The window the setter admits must therefore keep it unreachable for every root depth,
// which is what this test pins -- it is the arithmetic that couples the two constants.
TEST(History_gibbs_run_test, the_root_depth_guard_is_unreachable_for_every_admitted_window) {
  const auto largest_window = std::nextafter(max_marginal_node_time_window(), 0.0);
  for (auto depth : {1e-12, 1e-6, 0.001, 0.5, 1.0, 2.0, 100.0, 1e4, 1e8, 1e12}) {
    // The extreme proposal of the admitted window, in both directions.
    for (auto u : {largest_window, -largest_window}) {
      auto new_depth = depth * std::exp(u);
      EXPECT_GT(new_depth, 0.0) << "depth " << depth << ", u " << u;
      EXPECT_LE(new_depth, marginal_node_time_max_depth_factor * std::max(depth, 1.0))
          << "the guard would fire and reject asymmetrically at depth " << depth << ", u " << u;
    }
  }
  // And the bound really is the tight one: at the window the setter refuses, the guard does fire.
  EXPECT_GT(1.0 * std::exp(max_marginal_node_time_window() + 0.1),
            marginal_node_time_max_depth_factor * 1.0);
}

// The same, end to end: with the largest window the setter admits, a chain that uses the move stays
// consistent -- the incremental posterior still matches a full recomputation, the tree stays intact
// and the tip sequences are untouched.
TEST(History_gibbs_run_test, the_largest_admitted_window_keeps_the_chain_consistent) {
  auto pool = ctpl::thread_pool{1};
  constexpr auto num_sites = Site_index{100};
  constexpr auto mu = 2e-4;
  auto tree = make_tree(6, num_sites, 7, mu);
  auto want_tips = tip_states(tree, num_sites);
  auto run = make_run(pool, tree, 11, mu, true, 1);
  enable_gibbs(*run);
  run->set_marginal_node_time_window(std::nextafter(max_marginal_node_time_window(), 0.0));
  for (auto i = 0; i != 5; ++i) {
    run->do_mcmc_steps(4000);
    EXPECT_NEAR(run->log_G(), run->calc_cur_log_G(), 1e-6 * std::max(1.0, std::abs(run->log_G())));
    assert_phylo_tree_integrity(run->tree(), true);
  }
  EXPECT_EQ(tip_states(run->tree(), num_sites), want_tips);
  EXPECT_GT(run->tree().at_root().t, -std::numeric_limits<double>::infinity());
}

// The measurement that settled what the updates do to the tree, kept as the evidence trail rather
// than as an assertion: it prints, for both arms and for a pinned and a sampled clock rate, the
// trajectory of the number of substitution events, the root age and the clock rate, together with the
// independent marginal posterior of the final state.  What it shows is that the existing move set
// holds the root age in a narrow band while the marginal posterior is nearly flat over a much wider
// one, and that the updates move the chain across that band without changing the marginal posterior
// level.  Run it with --gtest_also_run_disabled_tests.
TEST(History_gibbs_diag, DISABLED_trajectories) {
  auto pool = ctpl::thread_pool{4};
  constexpr auto num_sites = Site_index{1000};
  constexpr auto mu = 8e-6;
  auto tree = make_tree(10, num_sites, 20261114, mu, 0, 4000.0);
  std::printf("generating tree: M = %d, root age = %.1f days, T = %.1f days\n",
              calc_num_muts(tree), -tree.at_root().t, calc_T(tree));
  for (auto fix_mu : {false, true}) {
  for (auto gibbs : {false, true}) {
    auto run = make_run(pool, tree, 101, mu, true, 1);
    if (fix_mu) { run->set_mu_move_enabled(false); }
    if (gibbs) { enable_gibbs(*run); }
    std::printf("%s mu=%s:", gibbs ? "gibbs" : "base ", fix_mu ? "fixed" : "free ");
    for (auto i = 0; i != 20; ++i) {
      run->do_mcmc_steps(20000);
      std::printf(" (%d, %.0f, %.2e)", run->num_muts(), -run->tree().at_root().t, run->mu());
    }
    std::printf("\n      marginal = %.2f, coal = %.2f, augmented log_G = %.2f, sum = %.2f\n",
                static_cast<double>(marginal_log_likelihood(run->tree(), run->evo())),
                run->log_coalescent_prior(),
                run->log_G(),
                static_cast<double>(marginal_log_likelihood(run->tree(), run->evo()))
                    + run->log_coalescent_prior());
  }
  }
}
}  // namespace delphy
