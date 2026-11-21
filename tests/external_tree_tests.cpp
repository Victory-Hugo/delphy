// Tests for external (user-supplied) rooted Newick trees and fixed-topology MCMC.
//
// All data here are synthetic: a heterochronous coalescent tree with Poisson substitutions on a short
// sequence.  Rates and time scales are simulation truth for the tests only.

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "ctpl_stl.h"

#include "api.h"
#include "beasty_input.h"
#include "beasty_output.h"
#include "estd.h"
#include "external_tree.h"
#include "newick.h"
#include "phylo_tree.h"
#include "run.h"

namespace delphy {

namespace {

constexpr auto k_sites = 300;
constexpr auto k_mu_site_day = 4e-6;     // simulation truth
constexpr auto k_coal_days = 300.0;      // pairwise coalescent timescale of the simulation

// Debug builds re-verify every incremental quantity after every move, which makes MCMC steps much slower
auto scaled_steps(int64_t steps) -> int64_t { return estd::is_debug_enabled ? std::max<int64_t>(steps / 25, 20'000) : steps; }

struct Sim_node {
  std::string name;
  double t = 0.0;  // days
  int left = -1;
  int right = -1;
};

struct Sim_data {
  std::vector<Sim_node> nodes;  // Tips first, then inner nodes in creation order; root last
  Real_sequence ref_sequence;
  std::vector<Tip_desc> tip_descs;
  int true_events = 0;
};

auto letter(int i) -> Real_seq_letter { return real_seq_letter_from_index(i); }

auto simulate(int num_tips, uint32_t seed, bool with_missing = true, bool tips_share_date = false) -> Sim_data {
  auto rng = std::mt19937{seed};
  auto uni = [&] { return std::uniform_real_distribution<double>{0.0, 1.0}(rng); };
  auto sim = Sim_data{};

  // Tips with sampling times in [-400, 0] days (or all 0)
  for (auto i = 0; i != num_tips; ++i) {
    sim.nodes.push_back(Sim_node{absl::StrFormat("S%03d", i), tips_share_date ? 0.0 : -400.0 * uni()});
  }
  auto order = std::vector<int>(num_tips);
  for (auto i = 0; i != num_tips; ++i) { order[i] = i; }
  std::ranges::sort(order, [&](int a, int b) { return sim.nodes[a].t > sim.nodes[b].t; });  // newest first

  auto active = std::vector<int>{};
  auto next = 0;
  auto t = sim.nodes[order[0]].t;
  active.push_back(order[next++]);
  while (next < num_tips || active.size() > 1) {
    auto k = static_cast<double>(active.size());
    auto wait = k >= 2 ? -std::log(1.0 - uni()) / (k * (k - 1) / 2.0 / k_coal_days) : 1e300;
    auto t_sample = next < num_tips ? sim.nodes[order[next]].t : -1e300;
    if (t - wait > t_sample) {
      t -= wait;
      auto i = static_cast<int>(rng() % active.size());
      auto a = active[i];
      active.erase(active.begin() + i);
      auto j = static_cast<int>(rng() % active.size());
      auto b = active[j];
      active.erase(active.begin() + j);
      sim.nodes.push_back(Sim_node{"", t, a, b});
      active.push_back(static_cast<int>(sim.nodes.size()) - 1);
    } else {
      t = t_sample;
      active.push_back(order[next++]);
    }
  }

  // Sequences
  sim.ref_sequence.resize(k_sites);
  for (auto& r : sim.ref_sequence) { r = letter(rng() % 4); }
  auto root = static_cast<int>(sim.nodes.size()) - 1;
  auto seqs = std::vector<Real_sequence>(sim.nodes.size());
  seqs[root] = sim.ref_sequence;
  for (auto d = 0; d != 3; ++d) {  // The root differs from the reference
    auto s = static_cast<int>(rng() % k_sites);
    seqs[root][s] = letter((index_of(seqs[root][s]) + 1 + rng() % 3) % 4);
  }
  for (auto n = root; n >= 0; --n) {
    for (auto c : {sim.nodes[n].left, sim.nodes[n].right}) {
      if (c < 0) { continue; }
      seqs[c] = seqs[n];
      auto expected = k_mu_site_day * k_sites * (sim.nodes[c].t - sim.nodes[n].t);
      auto num = std::poisson_distribution<int>{expected}(rng);
      for (auto m = 0; m != num; ++m) {
        auto s = static_cast<int>(rng() % k_sites);
        seqs[c][s] = letter((index_of(seqs[c][s]) + 1 + rng() % 3) % 4);
        ++sim.true_events;
      }
    }
  }

  for (auto i = 0; i != num_tips; ++i) {
    auto td = Tip_desc{};
    td.name = sim.nodes[i].name;
    td.t_min = td.t_max = static_cast<float>(sim.nodes[i].t);
    auto miss_a = -1, miss_b = -1;
    if (with_missing && i % 4 == 1) {
      miss_a = static_cast<int>(rng() % (k_sites - 40));
      miss_b = miss_a + 10 + static_cast<int>(rng() % 30);
      for (auto s = miss_a; s != miss_b; ++s) { td.missations.insert(Missation{s, sim.ref_sequence[s]}, sim.ref_sequence); }
    }
    for (auto s = 0; s != k_sites; ++s) {
      if (s >= miss_a && s < miss_b) { continue; }
      if (seqs[i][s] != sim.ref_sequence[s]) { td.seq_deltas.push_back(Seq_delta{s, sim.ref_sequence[s], seqs[i][s]}); }
    }
    sim.tip_descs.push_back(std::move(td));
  }
  return sim;
}

auto newick_of(const Sim_data& sim, int node, double scale, bool with_lengths = true) -> std::string {
  const auto& n = sim.nodes[node];
  if (n.left < 0) { return newick_label(n.name); }
  auto child = [&](int c) {
    auto s = newick_of(sim, c, scale, with_lengths);
    if (with_lengths) { absl::StrAppend(&s, ":", absl::StrFormat("%.9g", (sim.nodes[c].t - n.t) * scale)); }
    return s;
  };
  return absl::StrCat("(", child(n.left), ",", child(n.right), ")");
}

auto root_of(const Sim_data& sim) -> int { return static_cast<int>(sim.nodes.size()) - 1; }

auto parse(const std::string& s) -> Newick_tree {
  auto is = std::istringstream{s};
  return read_newick_tree(is);
}

auto sim_newick(const Sim_data& sim, double scale = 1.0, bool with_lengths = true) -> Newick_tree {
  return parse(newick_of(sim, root_of(sim), scale, with_lengths) + ";");
}

// Rooted clades as sorted lists of tip names
auto rooted_clades(const Phylo_tree& tree) -> std::set<std::vector<std::string>> {
  auto clades = std::map<Node_index, std::vector<std::string>>{};
  auto result = std::set<std::vector<std::string>>{};
  for (const auto& node : post_order_traversal(tree)) {
    auto& names = clades[node];
    if (tree.at(node).is_tip()) {
      names.push_back(tree.at(node).name);
    } else {
      for (auto c : tree.at(node).children) {
        names.insert(names.end(), clades[c].begin(), clades[c].end());
        clades.erase(c);
      }
      std::ranges::sort(names);
      result.insert(names);
    }
  }
  return result;
}

auto rooted_clades(const Newick_tree& tree) -> std::set<std::vector<std::string>> {
  auto clades = std::map<Node_index, std::vector<std::string>>{};
  auto result = std::set<std::vector<std::string>>{};
  for (const auto& node : post_order_traversal(tree)) {
    auto& names = clades[node];
    if (tree.at(node).is_tip()) {
      names.push_back(tree.at(node).name);
    } else {
      for (auto c : tree.at(node).children) {
        names.insert(names.end(), clades[c].begin(), clades[c].end());
        clades.erase(c);
      }
      std::ranges::sort(names);
      result.insert(names);
    }
  }
  return result;
}

auto child_pairs(const Phylo_tree& tree) -> std::set<std::pair<std::string, std::string>> {
  // (parent clade, child clade) for every branch, each clade written as its sorted tip names joined by '|'
  auto tips_below = std::map<Node_index, std::vector<std::string>>{};
  auto keys = std::map<Node_index, std::string>{};
  for (const auto& node : post_order_traversal(tree)) {
    auto& names = tips_below[node];
    if (tree.at(node).is_tip()) {
      names.push_back(tree.at(node).name);
    } else {
      for (auto c : tree.at(node).children) { names.insert(names.end(), tips_below[c].begin(), tips_below[c].end()); }
      std::ranges::sort(names);
    }
    keys[node] = absl::StrJoin(names, "|");
  }
  auto result = std::set<std::pair<std::string, std::string>>{};
  for (auto node = Node_index{0}; node != tree.size(); ++node) {
    if (node != tree.root) { result.insert({keys[tree.at(node).parent], keys[node]}); }
  }
  return result;
}

auto build(const Sim_data& sim, const Newick_tree& nt, Branch_length_units units, std::optional<double> rate = {},
           External_tree_summary* summary = nullptr) -> Phylo_tree {
  auto bitgen = std::mt19937{1};
  auto options = External_tree_options{};
  options.units = units;
  options.rate_per_site_per_day = rate;
  options.num_observed_sites = k_sites;
  return build_phylo_tree_from_newick(nt, sim.ref_sequence, sim.tip_descs, bitgen, options, summary);
}

auto expect_invalid(const std::function<void()>& f, const std::string& needle) -> void {
  try {
    f();
    ADD_FAILURE() << "expected an error mentioning: " << needle;
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr(needle));
  }
}

auto min_branch_gap(const Phylo_tree& tree) -> double {
  auto gap = std::numeric_limits<double>::infinity();
  for (auto node = Node_index{0}; node != tree.size(); ++node) {
    if (node != tree.root) { gap = std::min(gap, tree.at(node).t - tree.at_parent_of(node).t); }
  }
  return gap;
}

}  // namespace

// Import
// ======

TEST(External_tree_test, imports_topology_exactly_and_tip_order_does_not_matter) {
  for (auto seed : {1u, 2u, 3u}) {
    auto sim = simulate(30, seed);
    auto nt = sim_newick(sim);
    std::ranges::reverse(sim.tip_descs);  // Sequence order differs from Newick order
    auto tree = build(sim, nt, Branch_length_units::days);
    EXPECT_EQ(rooted_clades(tree), rooted_clades(nt));
    EXPECT_EQ(std::ssize(tree), 59);
    for (auto node = Node_index{0}; node != 30; ++node) { EXPECT_EQ(tree.at(node).name, sim.tip_descs[node].name); }
    for (auto node = Node_index{0}; node != tree.size(); ++node) {
      if (node != tree.root) { EXPECT_GT(tree.at(node).t, tree.at_parent_of(node).t); }
    }
    assert_phylo_tree_integrity(tree, true);
    assert_phylo_tree_matches_tip_descs(tree, sim.ref_sequence, sim.tip_descs, true);
  }
}

TEST(External_tree_test, time_scale_follows_declared_units) {
  auto sim = simulate(20, 7);
  auto as_days = build(sim, sim_newick(sim), Branch_length_units::days);
  auto as_years = build(sim, sim_newick(sim, 1.0 / 365.0), Branch_length_units::years);
  EXPECT_NEAR(as_days.at_root().t, as_years.at_root().t, 1e-6);
  EXPECT_NEAR(as_days.at_root().t, sim.nodes[root_of(sim)].t, 1e-3 * 39);  // inner nodes follow the true heights

  auto as_subs = build(sim, sim_newick(sim, k_mu_site_day), Branch_length_units::subs_per_site, k_mu_site_day);
  EXPECT_NEAR(as_subs.at_root().t, as_days.at_root().t, 1e-4);
}

TEST(External_tree_test, rejects_malformed_trees) {
  auto sim = simulate(10, 11);
  auto nt = sim_newick(sim);
  auto days = Branch_length_units::days;

  // Unrooted (trifurcating root) and multifurcations
  expect_invalid([&] { build(sim, parse("(S000,S001,S002,S003,S004,S005,S006,S007,S008,S009);"), days); },
                 "rooted and strictly binary");
  expect_invalid([&] { build(sim, parse("((S000,S001,S002),(S003,S004),(S005,S006),(S007,S008,S009));"), days); },
                 "rooted and strictly binary");
  expect_invalid([&] { build(sim, parse("(((S000,S001)),S002);"), days); }, "single child");

  // Tip sets
  auto without = [&](const std::string& drop, const std::string& add) {
    auto s = newick_of(sim, root_of(sim), 1.0);
    auto pos = s.find(drop);
    s.replace(pos, drop.size(), add);
    return s + ";";
  };
  expect_invalid([&] { build(sim, parse(without("S003:", "S002:")), days); }, "duplicate tip names: S002");
  expect_invalid([&] { build(sim, parse(without("S003:", "S777:")), days); }, "S777");
  expect_invalid([&] { build(sim, parse(without("S003:", "S777:")), days); }, "missing from the tree: S003");
  expect_invalid([&] { build(sim, parse(without("S003:", ":")), days); }, "without a name");

  // Branch lengths
  expect_invalid([&] { build(sim, nt, Branch_length_units::undeclared); }, "meaning was not declared");
  expect_invalid([&] { build(sim, sim_newick(sim, 1.0, false), days); }, "has no length");
  auto some_branch = nt.at(nt.root).children[0];
  auto neg = nt;
  neg.at(some_branch).branch_length = -1.0;
  neg.at(some_branch).has_branch_length = true;
  expect_invalid([&] { build(sim, neg, days); }, "negative length");
  auto nonfinite = nt;
  nonfinite.at(some_branch).branch_length = std::numeric_limits<double>::quiet_NaN();
  expect_invalid([&] { build(sim, nonfinite, days); }, "non-finite length");
  nonfinite.at(some_branch).branch_length = std::numeric_limits<double>::infinity();
  expect_invalid([&] { build(sim, nonfinite, days); }, "non-finite length");

  // A time scale needs an explicit rate when branch lengths are not times
  expect_invalid([&] { build(sim, nt, Branch_length_units::ignore); }, "explicit substitution rate");
  expect_invalid([&] { build(sim, nt, Branch_length_units::subs_per_site); }, "explicit substitution rate");
  expect_invalid([&] { build(sim, nt, Branch_length_units::ignore, 0.0); }, "explicit substitution rate");
}

TEST(External_tree_test, unlabelled_trees_are_accepted_when_topology_only) {
  auto sim = simulate(25, 5);
  auto summary = External_tree_summary{};
  auto tree = build(sim, sim_newick(sim, 1.0, false), Branch_length_units::ignore, k_mu_site_day, &summary);
  EXPECT_FALSE(summary.used_branch_lengths);
  EXPECT_EQ(rooted_clades(tree), rooted_clades(sim_newick(sim, 1.0, false)));
  EXPECT_GT(min_branch_gap(tree), 0.0);
}

TEST(External_tree_test, zero_length_branches_are_legal_and_get_a_positive_gap) {
  auto sim = simulate(12, 9);
  auto nt = sim_newick(sim);
  for (auto node = Node_index{0}; node != nt.size(); ++node) {
    if (node != nt.root) { nt.at(node).branch_length = 0.0; }
  }
  auto tree = build(sim, nt, Branch_length_units::days);
  EXPECT_EQ(rooted_clades(tree), rooted_clades(nt));
  EXPECT_GT(min_branch_gap(tree), 0.0);
  EXPECT_NEAR(min_branch_gap(tree), External_tree_options{}.min_branch_days, 1e-12);
}

// Initial mutation history
// ========================

TEST(External_tree_test, parsimony_history_is_consistent_with_sequences_missing_data_and_root_state) {
  for (auto seed : {1u, 2u, 3u, 4u, 5u, 6u}) {
    auto sim = simulate(40, seed);
    auto summary = External_tree_summary{};
    auto tree = build(sim, sim_newick(sim), Branch_length_units::days, {}, &summary);
    assert_phylo_tree_integrity(tree, true);
    assert_phylo_tree_matches_tip_descs(tree, sim.ref_sequence, sim.tip_descs, true);
    EXPECT_LE(summary.num_parsimony_mutations, sim.true_events + 3 * 2) << "parsimony cannot exceed the true "
        "history by more than what missing data allow";
    EXPECT_GT(summary.num_parsimony_mutations, 0);
  }
}

TEST(External_tree_test, identical_sequences_get_zero_mutation_branches_and_homoplasy_is_kept) {
  // Four tips: ((A,B),(C,D)) with A=B and C=D sharing a recurrent change at site 2
  auto ref = Real_sequence(10, Real_seq_letter::A);
  auto tips = std::vector<Tip_desc>(4);
  auto names = std::vector<std::string>{"A", "B", "C", "D"};
  for (auto i = 0; i != 4; ++i) {
    tips[i].name = names[i];
    tips[i].t_min = tips[i].t_max = 0.0f;
  }
  tips[0].seq_deltas = {Seq_delta{2, Real_seq_letter::A, Real_seq_letter::G}};
  tips[1].seq_deltas = {Seq_delta{2, Real_seq_letter::A, Real_seq_letter::G}};
  tips[2].seq_deltas = {Seq_delta{2, Real_seq_letter::A, Real_seq_letter::G}, Seq_delta{5, Real_seq_letter::A, Real_seq_letter::T}};
  tips[3].seq_deltas = {Seq_delta{2, Real_seq_letter::A, Real_seq_letter::G}, Seq_delta{5, Real_seq_letter::A, Real_seq_letter::T}};
  auto bitgen = std::mt19937{1};
  auto options = External_tree_options{};
  options.units = Branch_length_units::days;
  auto tree = build_phylo_tree_from_newick(parse("((A:10,B:10):20,(C:10,D:10):20);"), ref, tips, bitgen, options);
  // Site 2 is G everywhere (root state G, absorbed into the reference); site 5 changes once on the (C,D) branch
  EXPECT_EQ(tree.ref_sequence[2], Real_seq_letter::G);
  auto total = 0;
  for (auto n = Node_index{0}; n != tree.size(); ++n) { total += static_cast<int>(std::ssize(tree.at(n).mutations)); }
  EXPECT_EQ(total, 1);

  // Without the shared state at the root, homoplasy appears as two independent changes: ((A,C),(B,D))
  tips[0].seq_deltas = {Seq_delta{2, Real_seq_letter::A, Real_seq_letter::G}};
  tips[1].seq_deltas = {};
  tips[2].seq_deltas = {Seq_delta{2, Real_seq_letter::A, Real_seq_letter::G}};
  tips[3].seq_deltas = {};
  auto tree2 = build_phylo_tree_from_newick(parse("((A:10,B:10):20,(C:10,D:10):20);"), ref, tips, bitgen, options);
  auto total2 = 0;
  for (auto n = Node_index{0}; n != tree2.size(); ++n) { total2 += static_cast<int>(std::ssize(tree2.at(n).mutations)); }
  EXPECT_EQ(total2, 2);  // One independent change on each of the A and C tip branches
}

// Fixed-topology MCMC
// ===================

namespace {

struct Mcmc_outcome {
  std::set<std::vector<std::string>> clades;
  std::set<std::pair<std::string, std::string>> relations;
  std::vector<double> inner_times;
  double min_gap = 0.0;
  std::array<Move_counts, k_num_move_kinds> counts{};
  int num_fixed_topology_flag = 0;
};

auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, bool fix_topology, int num_parts,
              int local_moves_per_global_move) -> std::shared_ptr<Run> {
  auto run = std::make_shared<Run>(pool, std::mt19937{seed}, seed, tree);
  run->set_mu(k_mu_site_day);
  run->set_topology_moves_enabled(not fix_topology);
  run->set_num_parts(num_parts);
  run->set_local_moves_per_global_move(local_moves_per_global_move);
  run->set_mu_prior_alpha(1.0);
  return run;
}

auto inner_times(const Phylo_tree& tree) -> std::vector<double> {
  auto result = std::vector<double>{};
  for (auto node = Node_index{0}; node != tree.size(); ++node) {
    if (tree.at(node).is_inner_node()) { result.push_back(tree.at(node).t); }
  }
  return result;
}

auto run_and_check_fixed_topology(int num_tips, uint32_t data_seed, uint32_t chain_seed, int threads, int num_parts,
                                  int local_moves, int64_t steps, int checks) -> void {
  auto sim = simulate(num_tips, data_seed);
  auto start = build(sim, sim_newick(sim), Branch_length_units::days);
  auto start_clades = rooted_clades(start);
  auto start_relations = child_pairs(start);
  auto start_times = inner_times(start);

  auto pool = ctpl::thread_pool{threads};
  auto run = make_run(pool, start, chain_seed, true, num_parts, local_moves);
  auto changed_any = std::vector<bool>(start_times.size(), false);
  for (auto c = 0; c != checks; ++c) {
    run->do_mcmc_steps(scaled_steps(steps) / checks);
    const auto& tree = run->tree();
    ASSERT_EQ(rooted_clades(tree), start_clades) << "clades changed after check " << c;
    ASSERT_EQ(child_pairs(tree), start_relations) << "parent-child relations changed after check " << c;
    // The sampled sequences are still the input ones, node times stay ordered
    assert_phylo_tree_integrity(tree, true);
    ASSERT_GE(min_branch_gap(tree), 0.0);
    auto times = inner_times(tree);
    ASSERT_EQ(times.size(), start_times.size());
    for (size_t i = 0; i != times.size(); ++i) {
      ASSERT_TRUE(std::isfinite(times[i]));
      if (times[i] != start_times[i]) { changed_any[i] = true; }
    }
  }
  auto num_changed = std::ranges::count(changed_any, true);
  EXPECT_GE(num_changed, static_cast<std::ptrdiff_t>(start_times.size()) * 9 / 10)
      << "node times should be sampled, not frozen";

  const auto& counts = run->move_counts();
  EXPECT_EQ(counts[static_cast<int>(Move_kind::spr1)].proposed, 0) << "SPR must not be selected";
  EXPECT_GT(counts[static_cast<int>(Move_kind::inner_node_displace)].accepted, 0);
  EXPECT_GT(counts[static_cast<int>(Move_kind::branch_reform)].accepted, 0);
  EXPECT_GT(counts[static_cast<int>(Move_kind::subtree_slide)].proposed, 0);
  EXPECT_GT(counts[static_cast<int>(Move_kind::subtree_slide)].accepted, 0) << "topology-preserving slides must be accepted";
  EXPECT_GT(counts[static_cast<int>(Move_kind::subtree_slide)].blocked_by_fixed_topology, 0)
      << "proposals that would hop over a node must be discarded";
  EXPECT_FALSE(run->topology_moves_enabled());
}

}  // namespace

TEST(Fixed_topology_test, single_thread_n20_and_n100_keep_rooted_clades_and_sample_node_times) {
  for (auto chain_seed : {11u, 12u, 13u}) {
    run_and_check_fixed_topology(20, chain_seed, chain_seed + 100, 1, 1, -1, 400'000, 4);
  }
  for (auto chain_seed : {21u, 22u}) {
    run_and_check_fixed_topology(100, chain_seed, chain_seed + 100, 1, 1, -1, 600'000, 3);
  }
}

TEST(Fixed_topology_test, multithreaded_partitions_and_repartitioning_keep_the_topology) {
  // Several partitions and frequent global moves force repartitioning of the same tree many times
  for (auto chain_seed : {31u, 32u}) {
    run_and_check_fixed_topology(100, chain_seed, chain_seed + 100, 4, 4, 200, 600'000, 6);
  }
  run_and_check_fixed_topology(40, 33, 133, 3, 3, 50, 300'000, 6);
}

TEST(Fixed_topology_test, free_topology_control_does_change_the_topology) {
  // Guards the tests above against vacuity: the same data with topology moves enabled gets rearranged.
  auto changed = 0;
  for (auto data_seed : {41u, 42u, 43u}) {
    auto sim = simulate(30, data_seed);
    auto start = build(sim, sim_newick(sim), Branch_length_units::days);
    auto pool = ctpl::thread_pool{1};
    auto run = make_run(pool, start, data_seed, false, 1, -1);
    run->do_mcmc_steps(estd::is_debug_enabled ? 150'000 : 600'000);
    if (rooted_clades(run->tree()) != rooted_clades(start)) { ++changed; }
    EXPECT_GT(run->move_counts()[static_cast<int>(Move_kind::spr1)].proposed, 0);
    EXPECT_EQ(run->move_counts()[static_cast<int>(Move_kind::subtree_slide)].blocked_by_fixed_topology, 0);
  }
  EXPECT_GT(changed, 0);
}

TEST(Fixed_topology_test, zero_mutation_branches_and_duplicate_haplotypes_do_not_rearrange_the_tree) {
  // Very low rate: nearly all branches have zero mutations and many tips are identical
  auto sim = simulate(60, 51, false);
  auto start = build(sim, sim_newick(sim), Branch_length_units::days);
  auto summary = External_tree_summary{};
  build(sim, sim_newick(sim), Branch_length_units::days, {}, &summary);
  auto num_branches_without_mutations = 0;
  for (auto node = Node_index{0}; node != start.size(); ++node) {
    if (node != start.root && start.at(node).mutations.empty()) { ++num_branches_without_mutations; }
  }
  EXPECT_GT(num_branches_without_mutations, 40);

  auto start_clades = rooted_clades(start);
  auto pool = ctpl::thread_pool{1};
  auto run = make_run(pool, start, 5, true, 1, -1);
  for (auto i = 0; i != 5; ++i) {
    run->do_mcmc_steps(scaled_steps(200'000));
    ASSERT_EQ(rooted_clades(run->tree()), start_clades);
  }
}

TEST(Fixed_topology_test, same_sampling_date_tips_keep_the_topology_and_times_still_move) {
  auto sim = simulate(30, 61, true, true);
  auto start = build(sim, sim_newick(sim), Branch_length_units::days);
  auto start_clades = rooted_clades(start);
  auto pool = ctpl::thread_pool{1};
  auto run = make_run(pool, start, 6, true, 1, -1);
  run->do_mcmc_steps(scaled_steps(500'000));
  EXPECT_EQ(rooted_clades(run->tree()), start_clades);
  EXPECT_NE(inner_times(run->tree()), inner_times(start));
}

TEST(Fixed_topology_test, setting_survives_params_round_trip_and_resumed_runs_stay_fixed) {
  auto sim = simulate(40, 71);
  auto start = build(sim, sim_newick(sim), Branch_length_units::days);
  auto start_clades = rooted_clades(start);
  auto pool = ctpl::thread_pool{2};

  auto run = make_run(pool, start, 7, true, 2, 100);
  run->do_mcmc_steps(scaled_steps(200'000));
  ASSERT_EQ(rooted_clades(run->tree()), start_clades);

  // Save: tree + params, like a .dphy snapshot
  auto tree_fb = phylo_tree_to_api_tree(run->tree());
  auto info_fb = phylo_tree_to_api_tree_info(run->tree());
  auto params_fb = run_to_api_params(*run);

  // Restore into a fresh Run whose own default is "topology moves enabled"
  auto reloaded_tree = api_tree_and_tree_info_to_phylo_tree(tree_fb.data(), info_fb.data());
  auto resumed = std::make_shared<delphy::Run>(pool, std::mt19937{8}, 8, std::move(reloaded_tree));
  EXPECT_TRUE(resumed->topology_moves_enabled());
  apply_api_params_to_run(params_fb.data(), *resumed);
  EXPECT_FALSE(resumed->topology_moves_enabled());

  resumed->do_mcmc_steps(scaled_steps(300'000));
  EXPECT_EQ(rooted_clades(resumed->tree()), start_clades);
  EXPECT_EQ(resumed->move_counts()[static_cast<int>(Move_kind::spr1)].proposed, 0);

  // And the opposite direction: a free-topology run restores as free
  auto free_run = make_run(pool, start, 9, false, 1, -1);
  auto free_params = run_to_api_params(*free_run);
  auto other = std::make_shared<delphy::Run>(pool, std::mt19937{10}, 10, start);
  other->set_topology_moves_enabled(false);
  apply_api_params_to_run(free_params.data(), *other);
  EXPECT_TRUE(other->topology_moves_enabled());
}

// Deep-time float32 behavior of imported trees
// ============================================

namespace {

auto check_deep_time_float32(double shift_years) -> void {
  // Input branch lengths in years: a few legal zero-length branches, a few very short ones, and ordinary ones.
  auto sim = simulate(60, 81, false);
  auto scale_to_years = 150000.0 / 365.0 / 600.0;  // stretch the tree to a root depth of roughly 100-200 kyr
  auto nt = sim_newick(sim, scale_to_years);
  auto zero_length = std::set<Node_index>{};
  auto short_length = std::set<Node_index>{};
  for (auto node = Node_index{0}; node != nt.size(); ++node) {
    if (node == nt.root) { continue; }
    if (node % 11 == 0) { nt.at(node).branch_length = 0.0; zero_length.insert(node); }
    else if (node % 13 == 0) { nt.at(node).branch_length = 1e-4 / 365.0; short_length.insert(node); }  // 1e-4 days
  }
  // Move the whole tree into the past by shifting all tips
  for (auto& td : sim.tip_descs) { td.t_min = td.t_max = static_cast<float>(-shift_years * 365.0 + td.t_min); }
  auto tree = build(sim, nt, Branch_length_units::years);

  // In memory (double), the ordering is strict whatever the input said
  EXPECT_GE(min_branch_gap(tree), External_tree_options{}.min_branch_days * 0.99);
  EXPECT_LT(tree.at_root().t, -shift_years * 365.0);

  // After the float32 round trip: ties may appear on branches that are short compared to float32 spacing; they must
  // never invert
  auto tree_fb = phylo_tree_to_api_tree(tree);
  auto info_fb = phylo_tree_to_api_tree_info(tree);
  auto reloaded = api_tree_and_tree_info_to_phylo_tree(tree_fb.data(), info_fb.data());
  auto ties_on_long_branches = 0, ties_total = 0, inversions = 0;
  auto max_time_error = 0.0;
  for (auto node = Node_index{0}; node != tree.size(); ++node) {
    max_time_error = std::max(max_time_error, std::abs(tree.at(node).t - reloaded.at(node).t));
    if (node == tree.root) { continue; }
    auto before_gap = tree.at(node).t - tree.at_parent_of(node).t;
    auto after_gap = reloaded.at(node).t - reloaded.at_parent_of(node).t;
    if (after_gap < 0.0) { ++inversions; }
    if (after_gap == 0.0) {
      ++ties_total;
      // Spacing of float32 near the node time
      auto t = static_cast<float>(tree.at(node).t);
      auto spacing = static_cast<double>(std::nextafter(t, std::numeric_limits<float>::infinity())) - t;
      if (before_gap > 2.0 * spacing) { ++ties_on_long_branches; }
    }
  }
  EXPECT_EQ(inversions, 0);
  EXPECT_EQ(ties_on_long_branches, 0) << "a tie on a branch much longer than float32 spacing would be a bug";
  EXPECT_GT(ties_total, 0) << "legal zero-length/short branches do tie at this depth: documented limitation";
  EXPECT_LE(max_time_error, 8.0);  // half the float32 spacing at 200 kyr is 4 days; allow rounding of tips too
  std::cerr << absl::StreamFormat(
      "[deep time] depth %.0f years, %d zero-length + %d very short input branches, %d float32 ties, "
      "max time error %.2f days\n",
      -tree.at_root().t / 365.0, std::ssize(zero_length), std::ssize(short_length), ties_total, max_time_error);
}

}  // namespace

TEST(External_tree_test, deep_time_float32_ties_are_distinguished_from_double_precision_collapse) {
  for (auto shift_years : {50'000.0, 100'000.0, 140'000.0, 200'000.0}) {
    check_deep_time_float32(shift_years);
  }
}

// Sample IDs with Newick metacharacters
// =====================================

TEST(External_tree_test, special_character_ids_survive_import_mcmc_and_trees_output) {
  const auto names = std::vector<std::string>{
      "123", "007", "2A", "A@B", "semi;colon", "dq\"uote", "it's", "a''b", "tilde~bang!q?", "with space",
      "(paren)", "a,b", "x:y", "[bracket]", "Ünïcødé-样本", std::string(300, 'L')};
  for (auto fix_topology : {true, false}) {
    auto sim = simulate(std::ssize(names), 7);
    for (auto i = 0; i != std::ssize(names); ++i) {
      sim.nodes[i].name = names[i];
      sim.tip_descs[i].name = names[i];
    }
    auto nt = sim_newick(sim);
    std::ranges::reverse(sim.tip_descs);
    auto start = build(sim, nt, Branch_length_units::days);
    EXPECT_EQ(rooted_clades(start), rooted_clades(nt));
    assert_phylo_tree_matches_tip_descs(start, sim.ref_sequence, sim.tip_descs, true);

    auto pool = ctpl::thread_pool{2};
    auto run = make_run(pool, start, 5, fix_topology, 2, 100);
    run->do_mcmc_steps(estd::is_debug_enabled ? 20'000 : 100'000);
    assert_phylo_tree_matches_tip_descs(run->tree(), sim.ref_sequence, sim.tip_descs, true);
    if (fix_topology) { EXPECT_EQ(rooted_clades(run->tree()), rooted_clades(start)); }

    auto oss = std::ostringstream{};
    auto out = Beasty_trees_output{&oss, "2.6.2"};
    out.output_headers(*run);
    out.output_tree(*run);
    out.output_footers(*run);

    auto is = std::istringstream{oss.str()};
    auto stream = Beasty_trees_stream{is};
    auto expected = std::vector<std::string>{};
    for (auto node = Node_index{0}; node != run->tree().size(); ++node) {
      if (run->tree().at(node).is_tip()) { expected.push_back(run->tree().at(node).name); }
    }
    EXPECT_EQ(stream.tip_names(), expected);
    auto state = int64_t{};
    auto read_back = Phylo_tree{};
    ASSERT_TRUE(stream.next(state, read_back));
    EXPECT_EQ(rooted_clades(read_back), rooted_clades(run->tree()));
  }
}

}  // namespace delphy
