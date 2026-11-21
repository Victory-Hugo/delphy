#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ctpl_stl.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "api.h"
#include "external_tree.h"
#include "human_input.h"
#include "evo_hky.h"
#include "independent_seq_density.h"
#include "phylo_tree_calc.h"
#include "pop_model.h"
#include "run.h"
#include "site_partition.h"

namespace delphy {

namespace {

constexpr auto k_mu = 2e-4;    // per site per day
constexpr auto k_n0 = 1000.0;  // days

auto parse_config(const std::string& text) -> Site_partition_config {
  auto is = std::istringstream{text};
  return read_site_partition_config(is);
}

template<typename F>
auto expect_error(F&& f, const std::string& fragment) -> void {
  try {
    f();
    ADD_FAILURE() << "expected Site_partition_error containing: " << fragment;
  } catch (const Site_partition_error& e) {
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr(fragment));
  }
}

// A Kingman tree over `num_tips` tips at one date, with Jukes-Cantor mutations whose rate is `rate(site)`
// times k_mu, and `missing` sites made missing in the first tip.
auto make_tree(int num_tips, Site_index num_sites, uint64_t seed,
               const std::function<double(Site_index)>& rate = [](Site_index) { return 1.0; },
               std::vector<Site_interval> missing_in_first_tip = {})
    -> Phylo_tree {
  auto g = std::mt19937_64{seed};
  auto tree = Phylo_tree{2 * num_tips - 1};
  tree.ref_sequence = Real_sequence(num_sites, Real_seq_letter::A);
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

  auto stack = std::vector<std::pair<Node_index, std::vector<int>>>{{tree.root, std::vector<int>(num_sites, 0)}};
  while (not stack.empty()) {
    auto [p, state] = std::move(stack.back()); stack.pop_back();
    for (auto c : tree.at(p).children) {
      auto t0 = tree.at(p).t, t1 = tree.at(c).t;
      auto st = state;
      auto times = std::vector<std::pair<double, Site_index>>{};
      for (auto l = Site_index{0}; l != num_sites; ++l) {
        auto m = std::poisson_distribution<int>{rate(l) * k_mu * (t1 - t0)}(g);
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

  for (const auto& iv : missing_in_first_tip) {
    // Mutations at a missing site would break the tree invariants, so they go first
    for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
      std::erase_if(tree.at(node).mutations,
                    [&](const Mutation& m) { return m.site >= iv.first && m.site < iv.second; });
    }
    tree.at(0).missations.intervals.insert(iv);
  }
  return tree;
}

auto make_evo(const Site_partition& partition, const std::vector<double>& rates, double mu = k_mu)
    -> Global_evo_model {
  auto evo = make_global_evo_model(partition.partition_for_site);
  auto site_evo = Hky_model{.mu = mu, .kappa = 2.5, .pi_a = {0.31, 0.19, 0.17, 0.33}}.derive_site_evo_model();
  for (auto p = Partition_index{0}; p != partition.num_partitions(); ++p) {
    evo.partition_evo_model[p] = site_evo;
    evo.partition_evo_model[p].mu = mu * rates[p];
  }
  evo.nu_l.assign(std::ssize(partition.partition_for_site), 1.0);
  return evo;
}

// The run's own evolutionary model with the per-partition rates replaced, so that a comparison against
// the independent density uses exactly the kappa, frequencies and site rates the run is using.
auto with_rates(const Global_evo_model& evo, double mu, const std::vector<double>& rates)
    -> Global_evo_model {
  auto result = evo;
  for (auto p = Partition_index{0}; p != result.num_partitions(); ++p) {
    result.partition_evo_model[p].mu = mu * rates[p];
  }
  return result;
}

auto delphy_log_G(const Phylo_tree& tree, const Global_evo_model& evo) -> double {
  return calc_log_root_prior(tree, evo) + calc_log_G_below_root(tree, evo);
}

auto two_partition_config(Site_index num_sites) -> Site_partition_config {
  return parse_config(absl::StrFormat("version 1\n\npartition left\n  sites 1-%d\nend\n\n"
                                      "partition right\n  sites %d-%d\nend\n",
                                      num_sites / 2, num_sites / 2 + 1, num_sites));
}

// `const_pop` replaces the default exponential population model by a constant one and switches the
// population moves off, which keeps these small trees away from the collapse attractor of the default
// coalescent grid.  The checkpoint tests keep the exponential model, because that is the one the .dphy
// parameters carry (a constant model is serialized as "no model").
auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, bool fixed_topology,
              int parts = 1, bool const_pop = true) -> std::shared_ptr<delphy::Run> {
  auto run = std::make_shared<delphy::Run>(pool, std::mt19937{seed}, seed, tree);
  run->set_topology_moves_enabled(not fixed_topology);
  run->set_mu(k_mu);
  run->set_mu_prior_alpha(1.0);
  run->set_mu_prior_beta(0.0);
  run->set_num_parts(parts);
  run->set_local_moves_per_global_move(200);
  if (const_pop) {
    run->set_pop_model(std::make_shared<Const_pop_model>(k_n0));
    run->set_final_pop_size_move_enabled(false);
    run->set_pop_growth_rate_move_enabled(false);
  }
  return run;
}

}  // namespace

//
// 1. The configuration file
//

TEST(Site_partition_config_test, parses_contiguous_and_scattered_partitions) {
  auto config = parse_config(R"(
version 1

# the first and the last third
partition outer
  sites 1-100
  sites 201-300
end

partition middle
  sites 101-200
end
)");
  ASSERT_EQ(config.size(), 2);
  EXPECT_EQ(config.blocks[0].name, "outer");
  EXPECT_THAT(config.blocks[0].sites, testing::ElementsAre(Site_interval{0, 100}, Site_interval{200, 300}));
  EXPECT_EQ(config.blocks[1].name, "middle");
  EXPECT_THAT(config.blocks[1].sites, testing::ElementsAre(Site_interval{100, 200}));
  EXPECT_FALSE(config.blocks[0].has_constant_sites);
  EXPECT_TRUE(config.constant_sites_per_block().empty());
}

TEST(Site_partition_config_test, single_sites_and_merged_ranges) {
  auto config = parse_config("version 1\n\npartition a\n  sites 5,1-3,2-4,9\nend\n");
  EXPECT_THAT(config.blocks[0].sites, testing::ElementsAre(Site_interval{0, 5}, Site_interval{8, 9}));
}

TEST(Site_partition_config_test, constant_sites_are_declared_per_partition) {
  auto config = parse_config("version 1\n\npartition a\n  sites 1-10\n  constant_sites A=5,C=4,G=3,T=2\nend\n"
                             "\npartition b\n  sites 11-20\n  constant_sites A=1,C=1,G=1,T=1\nend\n");
  ASSERT_EQ(config.size(), 2);
  EXPECT_EQ(config.total_constant_sites(), 18);
  auto per_block = config.constant_sites_per_block();
  ASSERT_EQ(std::ssize(per_block), 2);
  EXPECT_EQ(per_block[0][Real_seq_letter::A], 5);
  EXPECT_EQ(per_block[1][Real_seq_letter::T], 1);
}

TEST(Site_partition_config_test, errors) {
  expect_error([] { parse_config("partition a\n  sites 1-10\nend\n"); }, "must start with 'version 1'");
  expect_error([] { parse_config("version 2\n\npartition a\n  sites 1\nend\n"); }, "unsupported version");
  expect_error([] { parse_config("version 1\n"); }, "declares no partition");
  expect_error([] { parse_config("version 1\n\npartition a\nend\n"); }, "lists no site");
  expect_error([] { parse_config("version 1\n\npartition a\n  sites 1\n"); }, "not closed with 'end'");
  expect_error([] { parse_config("version 1\n\npartition a\n  sites 1\nend\n\npartition A\n  sites 2\nend\n"); },
               "repeats (up to letter case)");
  expect_error([] { parse_config("version 1\n\npartition a\n  sites 0-5\nend\n"); }, "empty or reversed");
  expect_error([] { parse_config("version 1\n\npartition a\n  sites 9-5\nend\n"); }, "empty or reversed");
  expect_error([] { parse_config("version 1\n\npartition a\n  sites x\nend\n"); }, "is not a site");
  expect_error([] { parse_config("version 1\n\npartition a\n  parts 3\n  sites 1\nend\n"); },
               "unknown key 'parts'");
  expect_error([] { parse_config("version 1\n  sites 1\n"); }, "outside a partition block");
  expect_error([] { parse_config("version 1\n\nend\n"); }, "'end' without a partition");
  expect_error([] { parse_config("version 1\n\npartition a\n  sites 1\n  constant_sites A=1,C=1,G=1\nend\n"); },
               "constant_sites is missing state T");
  expect_error([] { parse_config("version 1\n\npartition a\n  sites 1\n  constant_sites A=-1,C=1,G=1,T=1\nend\n"); },
               "negative count");
  expect_error([] {
    parse_config("version 1\n\npartition a\n  sites 1\n  constant_sites A=1,C=1,G=1,T=1\nend\n"
                 "\npartition b\n  sites 2\nend\n");
  }, "Restored constant sites must be declared for every partition or for none");
}

TEST(Site_partition_config_test, the_canonical_text_round_trips) {
  auto text = "version 1\n\npartition outer\n  sites 1-100,201-300\n  constant_sites A=0,C=0,G=1,T=0\nend\n"
              "\npartition middle\n  sites 101-200\n  constant_sites A=1,C=2,G=3,T=4\nend\n";
  auto config = parse_config(text);
  EXPECT_EQ(site_partition_config_to_string(config), text);
  auto again = parse_config(site_partition_config_to_string(config));
  EXPECT_EQ(site_partition_config_to_string(again), text);
}

//
// 2. Resolution against the analysed sequence
//

TEST(Site_partition_test, resolution_assigns_every_site_and_counts_the_weights) {
  auto observed = std::vector<bool>(300, true);
  for (auto l = 280; l != 300; ++l) { observed[l] = false; }  // never observed: not part of a weight
  auto config = parse_config("version 1\n\npartition outer\n  sites 1-100\n  sites 201-300\nend\n"
                             "\npartition middle\n  sites 101-200\nend\n");
  auto partition = resolve_site_partition(config, 300, observed);

  EXPECT_THAT(partition.names, testing::ElementsAre("outer", "middle"));
  EXPECT_THAT(partition.num_sites, testing::ElementsAre(200, 100));
  EXPECT_THAT(partition.num_observed_sites, testing::ElementsAre(180, 100));
  EXPECT_EQ(partition.total_observed_sites(), 280);
  EXPECT_EQ(partition.partition_for_site[0], 0);
  EXPECT_EQ(partition.partition_for_site[150], 1);
  EXPECT_EQ(partition.partition_for_site[299], 0);
}

TEST(Site_partition_test, resolution_errors) {
  auto observed = std::vector<bool>(100, true);
  expect_error([&] {
    resolve_site_partition(parse_config("version 1\n\npartition a\n  sites 1-50\nend\n"), 100, observed);
  }, "belong to no partition");
  expect_error([&] {
    resolve_site_partition(parse_config("version 1\n\npartition a\n  sites 1-60\nend\n"
                                        "\npartition b\n  sites 50-100\nend\n"), 100, observed);
  }, "must not overlap");
  expect_error([&] {
    resolve_site_partition(parse_config("version 1\n\npartition a\n  sites 1-200\nend\n"), 100, observed);
  }, "but the input reference has 100 sites");
  {
    auto some_unobserved = std::vector<bool>(100, true);
    for (auto l = 50; l != 100; ++l) { some_unobserved[l] = false; }
    expect_error([&] {
      resolve_site_partition(parse_config("version 1\n\npartition a\n  sites 1-50\nend\n"
                                          "\npartition b\n  sites 51-100\nend\n"), 100, some_unobserved);
    }, "not one of them is observed in any sample");
  }
  expect_error([&] {
    resolve_site_partition(parse_config("version 1\n\npartition a\n  sites 1-100\n"
                                        "  constant_sites A=1,C=0,G=0,T=0\nend\n"), 100, observed);
  }, "but the analysed sequence has 100");
}

TEST(Site_partition_test, restored_constant_sites_belong_to_their_own_partition) {
  // 20 input sites, then partition a's block (2 sites) and partition b's block (3 sites)
  auto observed = std::vector<bool>(25, true);
  auto config = parse_config("version 1\n\npartition a\n  sites 1-10\n  constant_sites A=1,C=1,G=0,T=0\nend\n"
                             "\npartition b\n  sites 11-20\n  constant_sites A=0,C=0,G=2,T=1\nend\n");
  auto partition = resolve_site_partition(config, 20, observed);
  EXPECT_THAT(partition.num_sites, testing::ElementsAre(12, 13));
  // The appended sites follow the input sites, partition by partition
  EXPECT_EQ(partition.partition_for_site[20], 0);
  EXPECT_EQ(partition.partition_for_site[21], 0);
  EXPECT_EQ(partition.partition_for_site[22], 1);
  EXPECT_EQ(partition.partition_for_site[24], 1);
}

TEST(Site_partition_test, weights_come_from_the_tree) {
  auto tree = make_tree(6, 40, 3, [](Site_index) { return 1.0; }, {{30, 40}});
  // Sites 31-40 are missing in the first tip only, so they are still observed elsewhere
  auto observed = observed_sites_of_tree(tree);
  EXPECT_EQ(std::ranges::count(observed, true), 40);

  auto all_missing = make_tree(6, 40, 3);
  for (auto node = Node_index{0}; node != std::ssize(all_missing); ++node) {
    std::erase_if(all_missing.at(node).mutations, [](const Mutation& m) { return m.site >= 30; });
  }
  for (auto i = 0; i != 6; ++i) { all_missing.at(i).missations.intervals.insert({30, 40}); }
  fix_up_missations(all_missing);
  auto observed2 = observed_sites_of_tree(all_missing);
  EXPECT_EQ(std::ranges::count(observed2, true), 30);
}

TEST(Site_partition_test, normalization) {
  auto observed = std::vector<bool>(100, true);
  auto config = parse_config("version 1\n\npartition a\n  sites 1-25\nend\n"
                             "\npartition b\n  sites 26-100\nend\n");
  auto partition = resolve_site_partition(config, 100, observed);

  auto rates = normalize_site_partition_rates(partition, {2.0, 0.5});
  EXPECT_NEAR(weighted_mean_site_partition_rate(partition, rates), 1.0, 1e-15);
  EXPECT_NEAR(rates[0] / rates[1], 4.0, 1e-12);          // the ratio is what the user asked for
  EXPECT_NEAR(25 * rates[0] + 75 * rates[1], 100.0, 1e-12);
  // Scaling every rate by the same factor is the same model
  EXPECT_THAT(normalize_site_partition_rates(partition, {20.0, 5.0}),
              testing::Pointwise(testing::DoubleNear(1e-12), rates));

  expect_error([&] { normalize_site_partition_rates(partition, {1.0}); }, "for 2 partitions");
  expect_error([&] { normalize_site_partition_rates(partition, {1.0, 0.0}); }, "finite and positive");
  expect_error([&] { normalize_site_partition_rates(partition, {1.0, -1.0}); }, "finite and positive");
}

//
// 3. The likelihood
//

TEST(Site_partition_test, equal_rates_reproduce_the_single_partition_likelihood) {
  auto tree = make_tree(10, 200, 17);
  auto observed = observed_sites_of_tree(tree);
  auto single = make_single_site_partition(observed);
  auto two = resolve_site_partition(two_partition_config(200), 200, observed);

  auto log_G_single = delphy_log_G(tree, make_evo(single, {1.0}));
  auto log_G_two = delphy_log_G(tree, make_evo(two, {1.0, 1.0}));
  EXPECT_NEAR(log_G_single, log_G_two, 1e-12);
  EXPECT_NEAR(log_G_two, independent_log_G(tree, make_evo(two, {1.0, 1.0})), 1e-9);
}

TEST(Site_partition_test, unequal_rates_match_an_independent_per_site_computation) {
  auto observed = std::vector<bool>(200, true);
  auto two = resolve_site_partition(two_partition_config(200), 200, observed);
  auto rate_of = [&](Site_index l) { return l < 100 ? 0.4 : 1.6; };
  auto tree = make_tree(10, 200, 23, rate_of);

  for (auto rates : {std::vector<double>{0.4, 1.6}, std::vector<double>{1.6, 0.4},
                     std::vector<double>{0.01, 1.99}}) {
    auto evo = make_evo(two, rates);
    EXPECT_NEAR(delphy_log_G(tree, evo), independent_log_G(tree, evo), 1e-8)
        << rates[0] << " " << rates[1];
  }
}

TEST(Site_partition_test, scattered_partitions_match_an_independent_per_site_computation) {
  auto observed = std::vector<bool>(120, true);
  auto config = parse_config("version 1\n\npartition even\n  sites 1-20,41-60,81-100\nend\n"
                             "\npartition odd\n  sites 21-40,61-80,101-120\nend\n");
  auto partition = resolve_site_partition(config, 120, observed);
  auto rate_of = [&](Site_index l) { return partition.partition_for_site[l] == 0 ? 0.5 : 1.5; };
  auto tree = make_tree(8, 120, 29, rate_of);
  auto evo = make_evo(partition, {0.5, 1.5});
  EXPECT_NEAR(delphy_log_G(tree, evo), independent_log_G(tree, evo), 1e-8);
}

TEST(Site_partition_test, a_partition_without_mutations_still_contributes) {
  auto observed = std::vector<bool>(100, true);
  auto config = parse_config("version 1\n\npartition quiet\n  sites 1-50\nend\n"
                             "\npartition loud\n  sites 51-100\nend\n");
  auto partition = resolve_site_partition(config, 100, observed);
  auto tree = make_tree(8, 100, 31, [](Site_index l) { return l < 50 ? 0.0 : 2.0; });
  auto quiet_muts = 0;
  for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
    for (const auto& m : tree.at(node).mutations) {
      if (m.site < 50) { ++quiet_muts; }
    }
  }
  ASSERT_EQ(quiet_muts, 0);  // the silent partition really has no mutation

  // Its sites still carry the exposure term, so the likelihood depends on its rate
  auto evo_a = make_evo(partition, {0.5, 1.5});
  auto evo_b = make_evo(partition, {0.1, 1.9});
  EXPECT_NEAR(delphy_log_G(tree, evo_a), independent_log_G(tree, evo_a), 1e-8);
  EXPECT_NEAR(delphy_log_G(tree, evo_b), independent_log_G(tree, evo_b), 1e-8);
  EXPECT_GT(delphy_log_G(tree, evo_b), delphy_log_G(tree, evo_a));  // a quieter partition is more likely
}

//
// 4. The Metropolis-Hastings update of the relative rates
//

namespace {

// The run of the tests below: 2 partitions over `num_sites` sites, mutations simulated with the given
// per-partition rates, and the site partition set on the run.
struct Partitioned_run {
  Phylo_tree tree;
  Site_partition partition;
  std::shared_ptr<delphy::Run> run;
};

auto make_partitioned_run(ctpl::thread_pool& pool, int num_tips, Site_index num_sites, uint64_t seed,
                          const std::vector<double>& true_rates, bool fixed_topology = true, int parts = 1,
                          bool const_pop = true)
    -> Partitioned_run {
  auto half = num_sites / 2;
  auto rate_of = [&](Site_index l) { return l < half ? true_rates[0] : true_rates[1]; };
  auto result = Partitioned_run{};
  result.tree = make_tree(num_tips, num_sites, seed, rate_of);
  result.run = make_run(pool, result.tree, static_cast<uint32_t>(seed), fixed_topology, parts, const_pop);
  result.run->set_site_partition(two_partition_config(num_sites));
  result.partition = result.run->site_partition();
  return result;
}

}  // namespace

class Site_partition_mcmc_test : public testing::TestWithParam<bool> {  // parameter: fixed topology
 protected:
  ctpl::thread_pool pool_{2};
};

TEST(Site_partition_rate_move_test, the_delta_matches_an_independent_computation) {
  auto pool = ctpl::thread_pool{1};
  auto pr = make_partitioned_run(pool, 10, 200, 41, {0.5, 1.5});
  auto& run = *pr.run;
  ASSERT_EQ(run.site_partition().num_partitions(), 2);

  for (auto delta_x : {0.05, -0.05, 0.2, -0.3, 0.001}) {
    for (auto accept_draw : {0.0, 1.0}) {
      auto old_rates = run.site_partition_rates();
      auto old_log_G = run.calc_cur_log_G();
      auto old_log_other = run.calc_cur_log_other_priors();
      auto old_evo = with_rates(run.evo(), run.mu(), old_rates);
      ASSERT_NEAR(old_log_G, independent_log_G(run.tree(), old_evo), 1e-7);

      auto result = run.site_partition_rate_move_with(0, 1, delta_x, accept_draw);
      ASSERT_TRUE(std::isfinite(result.log_alpha)) << delta_x;

      // The proposed rates, recomputed from the definition x_p = L_p r_p / sum_j L_j
      auto total = static_cast<double>(run.site_partition().total_observed_sites());
      auto proposed = old_rates;
      for (auto [p, sign] : {std::pair{0, +1.0}, std::pair{1, -1.0}}) {
        auto x = run.site_partition().num_observed_sites[p] * old_rates[p] / total + sign * delta_x;
        proposed[p] = x * total / run.site_partition().num_observed_sites[p];
      }
      auto new_evo = with_rates(run.evo(), run.mu(), proposed);
      auto expected_delta_log_G =
          independent_log_G(run.tree(), new_evo) - independent_log_G(run.tree(), old_evo);
      EXPECT_NEAR(result.delta_log_G, expected_delta_log_G, 1e-6 * std::max(1.0, std::abs(old_log_G)))
          << "delta_x " << delta_x;

      if (result.accepted) {
        EXPECT_THAT(run.site_partition_rates(), testing::Pointwise(testing::DoubleNear(1e-12), proposed));
        EXPECT_NEAR(run.log_G(), old_log_G + result.delta_log_G, 1e-9 * std::max(1.0, std::abs(old_log_G)));
        EXPECT_NEAR(run.log_G(), run.calc_cur_log_G(), 1e-6);
        EXPECT_NEAR(run.log_other_priors(), old_log_other + result.delta_log_other_priors, 1e-9);
        EXPECT_NEAR(weighted_mean_site_partition_rate(run.site_partition(), run.site_partition_rates()),
                    1.0, 1e-12);
      } else {
        // Rolled back completely: rates, likelihood and priors are what they were
        EXPECT_THAT(run.site_partition_rates(), testing::Pointwise(testing::DoubleNear(1e-15), old_rates));
        EXPECT_NEAR(run.log_G(), old_log_G, 1e-9);
        EXPECT_NEAR(run.log_G(), run.calc_cur_log_G(), 1e-6);
        EXPECT_NEAR(run.log_other_priors(), old_log_other, 1e-9);
      }
    }
  }
  EXPECT_GT(run.site_partition_rate_moves_proposed(), 0);
}

TEST(Site_partition_rate_move_test, a_proposal_that_leaves_the_simplex_is_rejected) {
  auto pool = ctpl::thread_pool{1};
  auto pr = make_partitioned_run(pool, 8, 100, 43, {1.0, 1.0});
  auto& run = *pr.run;
  auto old_rates = run.site_partition_rates();
  auto result = run.site_partition_rate_move_with(0, 1, 0.9, 0.0);  // x_q would go negative
  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.log_alpha, -std::numeric_limits<double>::infinity());
  EXPECT_THAT(run.site_partition_rates(), testing::Pointwise(testing::DoubleNear(1e-15), old_rates));
}

TEST(Site_partition_rate_move_test, the_dirichlet_prior_enters_the_acceptance_ratio) {
  auto pool = ctpl::thread_pool{1};
  auto pr = make_partitioned_run(pool, 8, 100, 47, {1.0, 1.0});
  auto& run = *pr.run;
  run.set_site_partition_rate_prior_concentration(3.0);
  auto total = static_cast<double>(run.site_partition().total_observed_sites());
  auto old_rates = run.site_partition_rates();
  auto x_of = [&](int p, const std::vector<double>& rates) {
    return run.site_partition().num_observed_sites[p] * rates[p] / total;
  };
  auto delta_x = 0.1;
  auto result = run.site_partition_rate_move_with(0, 1, delta_x, 1.0);
  auto expected = 2.0 * (std::log((x_of(0, old_rates) + delta_x) / x_of(0, old_rates))
                         + std::log((x_of(1, old_rates) - delta_x) / x_of(1, old_rates)));
  EXPECT_NEAR(result.delta_log_other_priors, expected, 1e-12);
  // The prior term is part of the maintained posterior, so a fresh computation agrees
  EXPECT_NEAR(run.log_other_priors(), run.calc_cur_log_other_priors(), 1e-9);
}

TEST(Site_partition_rate_move_test, the_prior_is_flat_by_default) {
  auto pool = ctpl::thread_pool{1};
  auto pr = make_partitioned_run(pool, 8, 100, 53, {1.0, 1.0});
  auto& run = *pr.run;
  EXPECT_EQ(run.site_partition_rate_prior_concentration(), 1.0);
  auto result = run.site_partition_rate_move_with(0, 1, 0.07, 1.0);
  EXPECT_EQ(result.delta_log_other_priors, 0.0);
  EXPECT_NEAR(run.log_other_priors(), run.calc_cur_log_other_priors(), 1e-9);
}

//
// 5. The rates reach the posterior through every move
//

TEST_P(Site_partition_mcmc_test, incremental_and_fresh_posteriors_agree_over_long_runs) {
  for (auto parts : {1, 2}) {
    auto pr = make_partitioned_run(pool_, 20, 300, 59, {0.4, 1.6}, GetParam(), parts);
    auto& run = *pr.run;
    for (auto batch = 0; batch != 8; ++batch) {
      run.do_mcmc_steps(2000);
      auto fresh = run.calc_cur_log_G();
      EXPECT_NEAR(run.log_G(), fresh, 1e-6 * std::max(1.0, std::abs(fresh)))
          << "parts " << parts << " batch " << batch;
      EXPECT_NEAR(run.log_other_priors(), run.calc_cur_log_other_priors(), 1e-6);
      EXPECT_NEAR(weighted_mean_site_partition_rate(run.site_partition(), run.site_partition_rates()),
                  1.0, 1e-9);
      EXPECT_NEAR(fresh, independent_log_G(run.tree(), run.evo()),
                  1e-6 * std::max(1.0, std::abs(fresh)))
          << "parts " << parts << " batch " << batch;
      assert_phylo_tree_integrity(run.tree(), true);
    }
    EXPECT_GT(run.site_partition_rate_moves_accepted(), 0);
    EXPECT_LT(run.site_partition_rate_moves_accepted(), run.site_partition_rate_moves_proposed());
  }
}

TEST_P(Site_partition_mcmc_test, fixed_rates_stay_fixed) {
  auto pr = make_partitioned_run(pool_, 12, 200, 61, {0.5, 1.5}, GetParam());
  auto& run = *pr.run;
  run.set_site_partition_rates({0.5, 1.5});
  run.set_site_partition_rate_move_enabled(false);
  auto rates = run.site_partition_rates();
  run.do_mcmc_steps(4000);
  EXPECT_THAT(run.site_partition_rates(), testing::Pointwise(testing::DoubleNear(1e-15), rates));
  EXPECT_EQ(run.site_partition_rate_moves_proposed(), 0);
  EXPECT_NEAR(run.log_G(), run.calc_cur_log_G(), 1e-6);
}

TEST_P(Site_partition_mcmc_test, the_clock_rate_keeps_its_meaning) {
  // mu is the mean rate per observed site: with the rates fixed and unequal, the Gibbs draw of mu must
  // still agree with a fresh posterior, and the implied per-partition rates are mu * r_p.
  auto pr = make_partitioned_run(pool_, 16, 400, 67, {0.25, 1.75}, GetParam());
  auto& run = *pr.run;
  run.do_mcmc_steps(6000);
  const auto& evo = run.evo();
  for (auto p = 0; p != run.site_partition().num_partitions(); ++p) {
    EXPECT_NEAR(evo.partition_evo_model[p].mu, run.mu() * run.site_partition_rates()[p], 1e-18);
  }
  EXPECT_NEAR(weighted_mean_site_partition_rate(run.site_partition(), run.site_partition_rates()), 1.0,
              1e-9);
  EXPECT_NEAR(run.log_G(), run.calc_cur_log_G(), 1e-6);
}

TEST_P(Site_partition_mcmc_test, a_checkpoint_restores_the_partition_and_its_rates) {
  auto pr = make_partitioned_run(pool_, 12, 200, 71, {0.5, 1.5}, GetParam(), 1, false);
  auto& run = *pr.run;
  run.do_mcmc_steps(4000);

  auto params = run_to_api_params(run);
  auto resumed = std::make_shared<delphy::Run>(pool_, std::mt19937{1}, 1, run.tree(),
                                               run.coalescent_prior_t_step());
  resumed->set_num_parts(1);
  apply_api_params_to_run(params.data(), *resumed);

  ASSERT_EQ(resumed->site_partition().num_partitions(), 2);
  EXPECT_THAT(resumed->site_partition().names, testing::ElementsAreArray(run.site_partition().names));
  EXPECT_THAT(resumed->site_partition().num_observed_sites,
              testing::ElementsAreArray(run.site_partition().num_observed_sites));
  EXPECT_THAT(resumed->site_partition_rates(),
              testing::Pointwise(testing::DoubleNear(1e-12), run.site_partition_rates()));
  EXPECT_EQ(resumed->site_partition_rate_prior_concentration(), run.site_partition_rate_prior_concentration());
  EXPECT_EQ(resumed->site_partition_rate_move_enabled(), run.site_partition_rate_move_enabled());
  EXPECT_EQ(resumed->mu(), run.mu());
  EXPECT_NEAR(resumed->log_G(), run.log_G(), 1e-9 * std::max(1.0, std::abs(run.log_G())));
  EXPECT_NEAR(resumed->log_G(), resumed->calc_cur_log_G(), 1e-6);

  resumed->do_mcmc_steps(2000);
  EXPECT_NEAR(resumed->log_G(), resumed->calc_cur_log_G(), 1e-6);
}

TEST(Site_partition_test, a_file_without_the_fields_restores_a_single_partition) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(8, 100, 73);
  auto run = make_run(pool, tree, 7, true, 1, false);
  run->do_mcmc_steps(1000);
  auto params = run_to_api_params(*run);  // no site partitions: the fields are not written
  auto resumed = std::make_shared<delphy::Run>(pool, std::mt19937{1}, 1, run->tree(),
                                               run->coalescent_prior_t_step());
  apply_api_params_to_run(params.data(), *resumed);
  EXPECT_FALSE(resumed->has_site_partitions());
  EXPECT_EQ(resumed->site_partition().num_partitions(), 1);
  EXPECT_THAT(resumed->site_partition_rates(), testing::ElementsAre(1.0));
  EXPECT_NEAR(resumed->log_G(), run->log_G(), 1e-9 * std::max(1.0, std::abs(run->log_G())));
}

TEST(Site_partition_test, the_mpox_hack_and_site_partitions_are_mutually_exclusive) {
  auto pool = ctpl::thread_pool{1};
  auto tree = make_tree(8, 100, 79);
  auto run = make_run(pool, tree, 11, true);
  run->set_mpox_hack_enabled(true);
  EXPECT_THROW(run->set_site_partition(two_partition_config(100)), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(Topology, Site_partition_mcmc_test, testing::Values(true, false),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "fixed" : "free";
                         });

//
// 6. Integration with the observation models
//

namespace {

constexpr auto k_obs_newick = "(((S0:400,S1:400):600,S2:1000):900,S3:1900);";

auto build_tree_from_newick(Maple_file maple) -> Phylo_tree {
  auto is = std::istringstream{k_obs_newick};
  auto ntree = read_newick_tree(is);
  auto prng = std::mt19937{7};
  auto options = External_tree_options{};
  options.units = Branch_length_units::days;
  return build_phylo_tree_from_newick(ntree, std::move(maple.ref_sequence), std::move(maple.tip_descs),
                                      prng, options);
}

auto metadata_for(const std::vector<std::string>& ids) -> Sample_metadata {
  auto text = std::string{"sample_id\tdate_convention\tdate_from\n"};
  for (auto i = 0; i != std::ssize(ids); ++i) {
    absl::StrAppendFormat(&text, "%s\tiso\t20%02d-01-01\n", ids[i], 10 + i);
  }
  auto is = std::istringstream{text};
  return read_sample_metadata(is);
}

auto load_alignment(const std::string& reference, const std::vector<std::string>& ids,
                    const std::vector<std::string>& sequences,
                    std::vector<Seq_vector<Site_index>> constant_sites = {})
    -> Human_input_result {
  auto fasta = std::string{};
  for (auto i = 0; i != std::ssize(ids); ++i) {
    absl::StrAppend(&fasta, ">", ids[i], "\n", sequences[i], "\n");
  }
  auto fasta_is = std::istringstream{fasta};
  auto ref_is = std::istringstream{absl::StrCat(">REF\n", reference, "\n")};
  auto metadata = metadata_for(ids);
  auto config = Human_input_config{};
  config.reference_standard = Reference_standard::custom;
  config.expected_length = static_cast<Site_index>(std::ssize(reference));
  config.max_missing_fraction = 1.0;
  config.snp_only_constant_sites = std::move(constant_sites);
  return load_human_fasta(fasta_is, &ref_is, &metadata, nullptr, config);
}

// Per analysed site: the partition's name followed by the state of every sample ('-' for missing).  Two
// inputs that give the same multiset describe the same observation, partition by partition.
auto partition_site_patterns(const Human_input_result& result, const Site_partition& partition)
    -> std::vector<std::string> {
  const auto& maple = result.maple;
  auto L = static_cast<Site_index>(std::ssize(maple.ref_sequence));
  auto n = static_cast<int>(std::ssize(maple.tip_descs));
  auto patterns = std::vector<std::string>(L);
  for (auto l = Site_index{0}; l != L; ++l) {
    patterns[l] = absl::StrCat(partition.names[partition.partition_for_site[l]], ":",
                               std::string(n, to_char(maple.ref_sequence[l])));
  }
  for (auto i = 0; i != n; ++i) {
    auto offset = static_cast<int>(std::ssize(patterns[0])) - n;
    for (const auto& [start, end] : maple.tip_descs[i].missations.intervals) {
      for (auto l = start; l != end; ++l) {
        patterns[l][patterns[l].size() - n + i] = '-';
      }
    }
    for (const auto& d : maple.tip_descs[i].seq_deltas) {
      patterns[d.site][patterns[d.site].size() - n + i] = to_char(d.to);
    }
    (void) offset;
  }
  return patterns;
}

}  // namespace

TEST(Site_partition_observation_test, constant_sites_restored_per_partition_match_the_full_alignment) {
  // 20 sites: partition "left" is 1-10, partition "right" is 11-20.  Each has variable sites and sites
  // that are constant across the samples (which a variability filter would remove).
  auto ids = std::vector<std::string>{"S0", "S1", "S2", "S3"};
  auto reference = std::string{"ACGTACGTACGTACGTACGT"};
  auto sequences = std::vector<std::string>(4, reference);
  auto set = [&](int sample, int site_1based, char c) { sequences[sample][site_1based - 1] = c; };
  set(1, 2, 'A');  set(2, 2, 'A');   // left: variable
  set(3, 5, 'C');                    // left: variable
  set(0, 13, 'T'); set(1, 13, 'T');  // right: variable (site 13 of the reference is A)
  set(2, 18, 'A');                   // right: variable (site 18 of the reference is C)

  auto full = load_alignment(reference, ids, sequences);
  auto full_partition = resolve_site_partition(
      parse_config("version 1\n\npartition left\n  sites 1-10\nend\n\npartition right\n  sites 11-20\nend\n"),
      20, observed_sites_of_tree(build_tree_from_newick(full.maple)));

  // The same data filtered down to the variable sites of each partition, with the removed constant sites
  // declared per partition.  The kept sites of "left" come first, then those of "right".
  auto variable_in = [&](Site_index from, Site_index to) {
    auto kept = std::vector<Site_index>{};
    auto counts = Seq_vector<Site_index>{0, 0, 0, 0};
    for (auto l = from; l != to; ++l) {
      auto varies = false;
      for (const auto& seq : sequences) {
        if (seq[l] != reference[l]) { varies = true; }
      }
      if (varies) { kept.push_back(l); } else { ++counts[char_to_real_seq_letter(reference[l])]; }
    }
    return std::pair{kept, counts};
  };
  auto [left_kept, left_counts] = variable_in(0, 10);
  auto [right_kept, right_counts] = variable_in(10, 20);
  auto kept = left_kept;
  kept.insert(kept.end(), right_kept.begin(), right_kept.end());

  auto snp_reference = std::string{};
  auto snp_sequences = std::vector<std::string>(4);
  for (auto l : kept) {
    snp_reference += reference[l];
    for (auto i = 0; i != 4; ++i) { snp_sequences[i] += sequences[i][l]; }
  }
  auto completed = load_alignment(snp_reference, ids, snp_sequences, {left_counts, right_counts});
  auto completed_partition = resolve_site_partition(
      parse_config(absl::StrFormat(
          "version 1\n\npartition left\n  sites 1-%d\n  constant_sites A=%d,C=%d,G=%d,T=%d\nend\n"
          "\npartition right\n  sites %d-%d\n  constant_sites A=%d,C=%d,G=%d,T=%d\nend\n",
          left_kept.size(), left_counts[Real_seq_letter::A], left_counts[Real_seq_letter::C],
          left_counts[Real_seq_letter::G], left_counts[Real_seq_letter::T],
          left_kept.size() + 1, kept.size(), right_counts[Real_seq_letter::A],
          right_counts[Real_seq_letter::C], right_counts[Real_seq_letter::G],
          right_counts[Real_seq_letter::T])),
      static_cast<Site_index>(std::ssize(kept)),
      observed_sites_of_tree(build_tree_from_newick(completed.maple)));

  EXPECT_EQ(completed.audit.num_sites, 20);
  EXPECT_EQ(completed.audit.num_input_sites, std::ssize(kept));
  ASSERT_EQ(completed_partition.num_partitions(), 2);
  EXPECT_THAT(completed_partition.num_sites, testing::ElementsAreArray(full_partition.num_sites));
  EXPECT_THAT(completed_partition.num_observed_sites,
              testing::ElementsAreArray(full_partition.num_observed_sites));

  // Every restored site sits in the partition it was removed from: the two inputs have the same multiset
  // of (partition, observed states) pairs.
  auto a = partition_site_patterns(completed, completed_partition);
  auto b = partition_site_patterns(full, full_partition);
  std::ranges::sort(a);
  std::ranges::sort(b);
  EXPECT_THAT(a, testing::Eq(b));

  // Hence the same likelihood on the same tree, for any relative rates
  auto full_tree = build_tree_from_newick(full.maple);
  auto completed_tree = build_tree_from_newick(completed.maple);
  for (auto rates : {std::vector<double>{1.0, 1.0}, std::vector<double>{0.5, 1.5},
                     std::vector<double>{1.8, 0.2}}) {
    auto full_evo = make_evo(full_partition, rates);
    auto completed_evo = make_evo(completed_partition, rates);
    EXPECT_NEAR(delphy_log_G(full_tree, full_evo), delphy_log_G(completed_tree, completed_evo), 1e-12)
        << rates[0] << " " << rates[1];
    EXPECT_NEAR(delphy_log_G(completed_tree, completed_evo),
                independent_log_G(completed_tree, completed_evo), 1e-9);
  }
}

TEST(Site_partition_observation_test, per_sample_missing_sites_keep_their_partition) {
  auto ids = std::vector<std::string>{"S0", "S1", "S2", "S3"};
  auto reference = std::string{"ACGTACGTACGTACGTACGT"};
  auto sequences = std::vector<std::string>(4, reference);
  sequences[0][14] = 'N';                                  // missing in one sample only
  for (auto& seq : sequences) { seq[19] = 'N'; }           // never observed
  sequences[1][2] = 'T';
  sequences[3][16] = 'A';

  auto loaded = load_alignment(reference, ids, sequences);
  auto tree = build_tree_from_newick(loaded.maple);
  auto observed = observed_sites_of_tree(tree);
  EXPECT_FALSE(observed[19]);
  EXPECT_TRUE(observed[14]);

  auto partition = resolve_site_partition(
      parse_config("version 1\n\npartition left\n  sites 1-10\nend\n\npartition right\n  sites 11-20\nend\n"),
      20, observed);
  // The never-observed site is in "right" and does not count towards its weight
  EXPECT_THAT(partition.num_sites, testing::ElementsAre(10, 10));
  EXPECT_THAT(partition.num_observed_sites, testing::ElementsAre(10, 9));

  auto evo = make_evo(partition, {0.5, 1.5});
  EXPECT_NEAR(delphy_log_G(tree, evo), independent_log_G(tree, evo), 1e-9);
}

//
// 7. The rate move samples its exact conditional distribution
//
// With the genealogy, the mutation history and every other parameter frozen, the conditional density of
// the length-weighted share x = x_0 (and x_1 = 1 - x_0) is known in closed form up to a constant:
//
//   log p(x) = M_0 log(x) + M_1 log(1 - x) - mu L [E_0 x / L_0 + E_1 (1 - x) / L_1]
//              + (c - 1) [log x + log(1 - x)],
//
// where M_p is the number of mutations of partition p, E_p = sum_a q_a Ttwiddle^p_a its exposure, L_p its
// weight and c the Dirichlet concentration.  A long chain of rate moves alone must reproduce that density;
// the test compares the empirical distribution with twenty equal-probability bins of the exact one.

TEST(Site_partition_rate_move_test, samples_the_exact_conditional_of_the_rate_shares) {
  auto pool = ctpl::thread_pool{1};
  auto pr = make_partitioned_run(pool, 10, 200, 83, {0.5, 1.5});
  auto& run = *pr.run;
  for (auto concentration : {1.0, 4.0}) {
    run.set_site_partition_rate_prior_concentration(concentration);
    run.set_site_partition_rates({1.0, 1.0});

    const auto& partition = run.site_partition();
    auto total = static_cast<double>(partition.total_observed_sites());
    auto L0 = static_cast<double>(partition.num_observed_sites[0]);
    auto L1 = static_cast<double>(partition.num_observed_sites[1]);
    auto muts = run.calc_cur_num_muts_per_partition();
    auto exposure = std::vector<double>(2, 0.0);
    for (auto p = Partition_index{0}; p != 2; ++p) {
      for (auto a : k_all_real_seq_letters) {
        exposure[p] += run.evo().partition_evo_model[p].q_a(a) * run.Ttwiddle_beta_a()[p][a];
      }
    }
    auto mu = run.mu();
    auto log_target = [&](double x) {
      if (not (x > 0.0 && x < 1.0)) { return -std::numeric_limits<double>::infinity(); }
      return muts[0] * std::log(x) + muts[1] * std::log(1.0 - x)
          - mu * total * (exposure[0] * x / L0 + exposure[1] * (1.0 - x) / L1)
          + (concentration - 1.0) * (std::log(x) + std::log(1.0 - x));
    };

    // The exact cumulative distribution on a fine grid, from which the bin edges are read off
    constexpr auto k_grid = 200001;
    auto density = std::vector<double>(k_grid, 0.0);
    auto max_log = -std::numeric_limits<double>::infinity();
    for (auto i = 1; i != k_grid - 1; ++i) {
      max_log = std::max(max_log, log_target(static_cast<double>(i) / (k_grid - 1)));
    }
    for (auto i = 1; i != k_grid - 1; ++i) {
      density[i] = std::exp(log_target(static_cast<double>(i) / (k_grid - 1)) - max_log);
    }
    auto cdf = std::vector<double>(k_grid, 0.0);
    for (auto i = 1; i != k_grid; ++i) { cdf[i] = cdf[i - 1] + density[i]; }
    for (auto& c : cdf) { c /= cdf.back(); }
    constexpr auto k_bins = 20;
    auto edges = std::vector<double>{};
    {
      auto i = 0;
      for (auto b = 1; b != k_bins; ++b) {
        auto target = static_cast<double>(b) / k_bins;
        while (i < k_grid - 1 && cdf[i] < target) { ++i; }
        edges.push_back(static_cast<double>(i) / (k_grid - 1));
      }
    }

    // A long chain of rate moves alone, thinned so that the samples are practically independent
    constexpr auto k_keep = 6000;
    constexpr auto k_thin = 40;
    auto counts = std::vector<int>(k_bins, 0);
    for (auto i = 0; i != k_keep * k_thin; ++i) {
      run.site_partition_rate_move();
      if (i % k_thin == 0) {
        auto x = partition.num_observed_sites[0] * run.site_partition_rates()[0] / total;
        auto bin = static_cast<int>(std::ranges::upper_bound(edges, x) - edges.begin());
        ++counts[bin];
      }
    }

    auto expected = static_cast<double>(k_keep) / k_bins;
    auto chi2 = 0.0;
    for (auto c : counts) { chi2 += (c - expected) * (c - expected) / expected; }
    // 19 degrees of freedom: the 0.1% critical value is 43.8
    EXPECT_LT(chi2, 43.8) << "concentration " << concentration << ", counts "
                          << absl::StrJoin(counts, ",");
    EXPECT_NEAR(weighted_mean_site_partition_rate(partition, run.site_partition_rates()), 1.0, 1e-9);
  }
}

}  // namespace delphy
