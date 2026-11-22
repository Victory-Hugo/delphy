// Tests for posterior clade tMRCA summaries.
//
// The clade resolution is checked against an independent reference implementation that walks the
// parent links from scratch, and against the exact-match logic of mcc_tree.cpp, which uses random
// 64-bit clade fingerprints rather than exact sample sets.  Every age here is a test input.

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

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "ctpl_stl.h"

#include "beasty_input.h"
#include "beasty_output.h"
#include "clade_summary.h"
#include "estd.h"
#include "external_tree.h"
#include "mcc_tree.h"
#include "newick.h"
#include "phylo_tree.h"
#include "pop_model.h"
#include "run.h"

namespace delphy {

namespace {

constexpr auto k_sites = 400;
constexpr auto k_mu_site_day = 2e-6;  // test input
constexpr auto k_coal_days = 500.0;   // test input

auto scaled_steps(int64_t steps) -> int64_t {
  return estd::is_debug_enabled ? std::max<int64_t>(steps / 25, 20'000) : steps;
}

// ---------------------------------------------------------------------------------------------
// Hand-built trees: a rooted Newick with named tips and branch lengths in days.  Tip node indices
// follow the order the tips appear in the text, which is what a .trees translate table does.
// ---------------------------------------------------------------------------------------------

struct Named_tree {
  Phylo_tree tree;
  std::vector<std::string> tip_names;  // Indexed by tip node index
};

auto tree_from_newick(const std::string& text) -> Named_tree {
  auto is = std::istringstream{text};
  auto nt = read_newick_tree(is);

  auto num_nodes = static_cast<Node_index>(std::ssize(nt));
  auto num_tips = (num_nodes + 1) / 2;
  auto n_to_p = Node_vector<Node_index>(num_nodes, k_no_node);
  auto next_tip = Node_index{0};
  auto next_inner = num_tips;
  for (const auto& n : pre_order_traversal(nt)) {
    n_to_p[n] = nt.at(n).is_tip() ? next_tip++ : next_inner++;
  }

  auto result = Named_tree{};
  result.tree = Phylo_tree{num_nodes};
  result.tip_names.assign(num_tips, {});
  for (const auto& n : pre_order_traversal(nt)) {
    auto p = n_to_p[n];
    if (n == nt.root) {
      result.tree.root = p;
      result.tree.at(p).t = 0.0;
    } else {
      result.tree.at(p).parent = n_to_p[nt.at(n).parent];
      result.tree.at(p).t = result.tree.at_parent_of(p).t + nt.at(n).branch_length;
    }
    if (nt.at(n).is_tip()) {
      result.tree.at(p).name = nt.at(n).name;
      result.tip_names[p] = nt.at(n).name;
      result.tree.at(p).t_min = result.tree.at(p).t_max = static_cast<float>(result.tree.at(p).t);
    } else {
      CHECK_EQ(std::ssize(nt.at(n).children), 2);
      result.tree.at(p).children = {n_to_p[nt.at(n).children[0]], n_to_p[nt.at(n).children[1]]};
      result.tree.at(p).t_min = -std::numeric_limits<float>::max();
      result.tree.at(p).t_max = +std::numeric_limits<float>::max();
    }
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// Independent reference: the MRCA of a set of tip names, found by intersecting ancestor paths.
// ---------------------------------------------------------------------------------------------

auto ref_ancestors(const Phylo_tree& tree, Node_index node) -> std::vector<Node_index> {
  auto result = std::vector<Node_index>{};
  for (auto n = node; n != k_no_node; n = tree.at(n).parent) { result.push_back(n); }
  return result;
}

auto ref_depth(const Phylo_tree& tree, Node_index node) -> int {
  return static_cast<int>(std::ssize(ref_ancestors(tree, node))) - 1;
}

auto ref_tips_below(const Phylo_tree& tree, Node_index node) -> std::set<std::string> {
  auto result = std::set<std::string>{};
  auto stack = std::vector<Node_index>{node};
  while (not stack.empty()) {
    auto n = stack.back();
    stack.pop_back();
    if (tree.at(n).is_tip()) {
      result.insert(tree.at(n).name);
    } else {
      for (auto c : tree.at(n).children) { stack.push_back(c); }
    }
  }
  return result;
}

auto ref_mrca(const Phylo_tree& tree, const std::set<std::string>& names) -> Node_index {
  auto common = std::set<Node_index>{};
  auto first = true;
  for (const auto& node : index_order_traversal(tree)) {
    if (not tree.at(node).is_tip()) { continue; }
    if (not names.contains(tree.at(node).name)) { continue; }
    auto anc = ref_ancestors(tree, node);
    auto as_set = std::set<Node_index>{anc.begin(), anc.end()};
    if (first) {
      common = as_set;
      first = false;
    } else {
      auto next = std::set<Node_index>{};
      std::ranges::set_intersection(common, as_set, std::inserter(next, next.end()));
      common = std::move(next);
    }
  }
  CHECK(not common.empty());
  auto best = k_no_node;
  for (auto n : common) {
    if (best == k_no_node || ref_depth(tree, n) > ref_depth(tree, best)) { best = n; }
  }
  return best;
}

auto parse_specs(const std::string& text) -> Clade_spec_set {
  auto is = std::istringstream{text};
  return read_clade_specs(is);
}

auto expect_spec_rejected(const std::string& text, const std::string& needle) -> void {
  try {
    parse_specs(text);
    ADD_FAILURE() << "expected a rejection mentioning: " << needle;
  } catch (const Clade_spec_error& e) {
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr(needle)) << e.what();
  }
}

// ---------------------------------------------------------------------------------------------
// Synthetic data, as in calibration_tests.cpp, so that a real Run can be driven.
// ---------------------------------------------------------------------------------------------

struct Sim_node {
  std::string name;
  double t = 0.0;
  int left = -1;
  int right = -1;
};
struct Sim_data {
  std::vector<Sim_node> nodes;
  Real_sequence ref_sequence;
  std::vector<Tip_desc> tip_descs;
};

auto letter(int i) -> Real_seq_letter { return real_seq_letter_from_index(i); }

auto simulate(int num_tips, uint32_t seed, double scale_days, double tip_t, double mu_site_day) -> Sim_data {
  auto rng = std::mt19937{seed};
  auto uni = [&] { return std::uniform_real_distribution<double>{0.0, 1.0}(rng); };
  auto sim = Sim_data{};
  for (auto i = 0; i != num_tips; ++i) {
    sim.nodes.push_back(Sim_node{absl::StrFormat("S%03d", i), tip_t});
  }
  auto active = std::vector<int>{};
  for (auto i = 0; i != num_tips; ++i) { active.push_back(i); }
  auto t = tip_t;
  while (active.size() > 1) {
    auto k = static_cast<double>(active.size());
    t -= -std::log(1.0 - uni()) / (k * (k - 1) / 2.0 / (k_coal_days * scale_days));
    auto i = static_cast<int>(rng() % active.size());
    auto a = active[i];
    active.erase(active.begin() + i);
    auto j = static_cast<int>(rng() % active.size());
    auto b = active[j];
    active.erase(active.begin() + j);
    sim.nodes.push_back(Sim_node{"", t, a, b});
    active.push_back(static_cast<int>(sim.nodes.size()) - 1);
  }
  sim.ref_sequence.resize(k_sites);
  for (auto& r : sim.ref_sequence) { r = letter(static_cast<int>(rng() % 4)); }
  auto root = static_cast<int>(sim.nodes.size()) - 1;
  auto seqs = std::vector<Real_sequence>(sim.nodes.size());
  seqs[root] = sim.ref_sequence;
  for (auto n = root; n >= 0; --n) {
    for (auto c : {sim.nodes[n].left, sim.nodes[n].right}) {
      if (c < 0) { continue; }
      seqs[c] = seqs[n];
      auto expected = mu_site_day * k_sites * (sim.nodes[c].t - sim.nodes[n].t);
      auto num = std::poisson_distribution<int>{expected}(rng);
      for (auto m = 0; m != num; ++m) {
        auto s = static_cast<int>(rng() % k_sites);
        seqs[c][s] = letter((index_of(seqs[c][s]) + 1 + static_cast<int>(rng() % 3)) % 4);
      }
    }
  }
  for (auto i = 0; i != num_tips; ++i) {
    auto td = Tip_desc{};
    td.name = sim.nodes[i].name;
    td.t_min = td.t_max = static_cast<float>(sim.nodes[i].t);
    for (auto s = 0; s != k_sites; ++s) {
      if (seqs[i][s] != sim.ref_sequence[s]) {
        td.seq_deltas.push_back(Seq_delta{s, sim.ref_sequence[s], seqs[i][s]});
      }
    }
    sim.tip_descs.push_back(std::move(td));
  }
  return sim;
}

auto newick_of(const Sim_data& sim, int node) -> std::string {
  const auto& n = sim.nodes[node];
  if (n.left < 0) { return n.name; }
  auto child = [&](int c) {
    return absl::StrCat(newick_of(sim, c), ":", absl::StrFormat("%.12g", sim.nodes[c].t - sim.nodes[node].t));
  };
  return absl::StrCat("(", child(n.left), ",", child(n.right), ")");
}

auto sim_tree(const Sim_data& sim) -> Phylo_tree {
  auto text = newick_of(sim, static_cast<int>(sim.nodes.size()) - 1) + ";";
  auto is = std::istringstream{text};
  auto nt = read_newick_tree(is);
  auto bitgen = std::mt19937{7};
  auto options = External_tree_options{};
  options.units = Branch_length_units::days;
  options.num_observed_sites = k_sites;
  return build_phylo_tree_from_newick(nt, sim.ref_sequence, sim.tip_descs, bitgen, options);
}

auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, bool fix_topology)
    -> std::shared_ptr<Run> {
  auto run = std::make_shared<Run>(pool, std::mt19937{seed}, seed, tree);
  run->set_topology_moves_enabled(not fix_topology);
  run->set_mu(k_mu_site_day);
  run->set_mu_prior_alpha(1.0);
  run->set_mu_prior_beta(0.0);
  run->set_num_parts(1);
  run->set_pop_model(std::make_shared<Const_pop_model>(k_coal_days));
  run->set_final_pop_size_move_enabled(false);
  run->set_pop_growth_rate_move_enabled(false);
  run->set_target_coal_prior_cells(4000);
  return run;
}

// Writes a .trees file by sampling `num_trees` states of `run`, using the production writer.
auto write_trees_file(Run& run, int num_trees, int64_t steps_between, int significant_digits)
    -> std::string {
  auto oss = std::ostringstream{};
  auto out = Beasty_trees_output{&oss, "2.6.2"};
  out.set_significant_digits(significant_digits);
  out.output_headers(run);
  for (auto i = 0; i != num_trees; ++i) {
    out.output_tree(run);
    if (i + 1 != num_trees) { run.do_mcmc_steps(steps_between); }
  }
  out.output_footers(run);
  return std::move(oss).str();
}

// The same file format, written straight from trees, so that deep-time precision can be measured
// without a Run.  A Run builds its coalescent prior on a one-day grid, which costs tens of gigabytes
// for a 200,000-year genealogy; the Newick body is written by the production
// writer either way.
auto write_trees_file_from_trees(
    const std::vector<std::pair<int64_t, const Phylo_tree*>>& trees, int significant_digits)
    -> std::string {
  CHECK(not trees.empty());
  const auto& first = *trees.front().second;
  auto node_to_tip = beasty_tip_indices(first);
  auto tip_nodes = std::vector<Node_index>{};
  for (const auto& node : index_order_traversal(first)) {
    if (first.at(node).is_tip()) { tip_nodes.push_back(node); }
  }

  auto oss = std::ostringstream{};
  oss << "#NEXUS\n"
      << "\n"
      << "Begin taxa;\n"
      << "    Dimensions ntax=" << std::ssize(tip_nodes) << ";\n"
      << "        Taxlabels\n";
  for (auto node : tip_nodes) { oss << "            " << first.at(node).name << "\n"; }
  oss << "            ;\n"
      << "End;\n"
      << "Begin trees;\n"
      << "    Translate\n";
  auto first_tip = true;
  for (auto node : tip_nodes) {
    if (not first_tip) { oss << ",\n"; }
    first_tip = false;
    oss << absl::StrFormat("        %4d %s", node_to_tip[node], first.at(node).name);
  }
  oss << "\n;\n";
  for (const auto& [state, tree] : trees) {
    oss << "tree STATE_" << state << " = ";
    write_beasty_newick_tree(*tree, node_to_tip, significant_digits, oss);
    oss << "\n";
  }
  oss << "End;";
  return std::move(oss).str();
}

// Node ages of one tree, in years before its most recent tip.
auto ages_of(const Phylo_tree& tree, const std::vector<std::set<std::string>>& clades)
    -> std::vector<double> {
  auto youngest = Clade_resolver::youngest_tip_time(tree);
  auto result = std::vector<double>{};
  for (const auto& names : clades) {
    auto mrca = names.empty() ? tree.root : ref_mrca(tree, names);
    result.push_back((youngest - tree.at(mrca).t) / 365.0);
  }
  return result;
}

}  // namespace

// Clade definition table
// ======================

TEST(Clade_spec_test, parses_a_table) {
  auto set = parse_specs(
      "# a comment\n"
      "clade_id\ttaxa\tnote\n"
      "root\tall\tthe root\n"
      "M\tS002, S000 S001\tnote ignored\n"
      "\n");
  ASSERT_EQ(set.size(), 2);
  EXPECT_EQ(set.specs[0].id, "root");
  EXPECT_TRUE(set.specs[0].all_tips);
  EXPECT_TRUE(set.specs[0].taxa.empty());
  EXPECT_EQ(set.specs[1].id, "M");
  EXPECT_FALSE(set.specs[1].all_tips);
  // Sorted and unique, so that the same set written in any order is the same set
  EXPECT_THAT(set.specs[1].taxa, testing::ElementsAre("S000", "S001", "S002"));
  EXPECT_EQ(set.specs[1].config_line, 4);
}

TEST(Clade_spec_test, column_order_does_not_matter) {
  auto set = parse_specs("note\ttaxa\tclade_id\nx\tA,B\tAB\n");
  ASSERT_EQ(set.size(), 1);
  EXPECT_EQ(set.specs[0].id, "AB");
  EXPECT_THAT(set.specs[0].taxa, testing::ElementsAre("A", "B"));
}

TEST(Clade_spec_test, rejections) {
  expect_spec_rejected("clade\ttaxa\nA\tX,Y\n", "'clade_id' and 'taxa'");
  expect_spec_rejected("", "needs a header line");
  expect_spec_rejected("clade_id\ttaxa\n", "defines no clades");
  expect_spec_rejected("clade_id\ttaxa\nAB\tA,B\nAB\tC,D\n", "duplicate clade_id 'AB'");
  expect_spec_rejected("clade_id\ttaxa\n\tA,B\n", "empty clade_id");
  expect_spec_rejected("clade_id\ttaxa\nAB\t\n", "lists no taxa");
  expect_spec_rejected("clade_id\ttaxa\nAB\tA\n", "at least two sample IDs");
  expect_spec_rejected("clade_id\ttaxa\nAB\tA,A,B\n", "more than once");
  expect_spec_rejected("clade_id\ttaxa\nAB\tall,A\n", "mixes 'all'");
  expect_spec_rejected("clade_id\ttaxa\nAB\n", "expected at least 2 tab-separated columns");
}

TEST(Clade_spec_test, every_problem_is_reported_together) {
  try {
    parse_specs("clade_id\ttaxa\nA\tX\nB\tY,Y,Z\nA\tP,Q\n");
    ADD_FAILURE() << "expected a rejection";
  } catch (const Clade_spec_error& e) {
    auto text = std::string{e.what()};
    EXPECT_THAT(text, testing::HasSubstr("at least two sample IDs"));
    EXPECT_THAT(text, testing::HasSubstr("more than once"));
    EXPECT_THAT(text, testing::HasSubstr("duplicate clade_id"));
  }
}

TEST(Clade_spec_test, clades_from_a_calibration_configuration) {
  auto cal_text = std::string{R"(
version 1

calibration root
  taxa all
  age_units ka
  age_reference_year 1950
  distribution normal
  mean 140
  stddev 10
end

calibration M
  taxa S001 S000
  age_units cal_bp
  distribution uniform
  oldest 120000
  youngest 80000
end
)"};
  auto cal_is = std::istringstream{cal_text};
  auto set = clade_specs_from_calibrations(read_calibration_config(cal_is));
  ASSERT_EQ(set.size(), 2);
  EXPECT_EQ(set.specs[0].id, "root");
  EXPECT_TRUE(set.specs[0].all_tips);
  EXPECT_EQ(set.specs[1].id, "M");
  EXPECT_THAT(set.specs[1].taxa, testing::ElementsAre("S000", "S001"));
}

// Resolving clades in a tree
// ==========================

TEST(Clade_resolver_test, resolves_against_an_independent_reference) {
  auto nt = tree_from_newick("((((A:3,B:3):3,C:6):4,(D:7,E:7):3):4,F:14);");
  auto specs = parse_specs(
      "clade_id\ttaxa\n"
      "root\tall\n"
      "M\tA,B,C,D,E\n"
      "N\tA,B,C\n"
      "AB\tA,B\n"
      "DE\tD,E\n"
      "AD\tA,D\n");  // Paraphyletic: its MRCA is the M node
  auto resolver = Clade_resolver{specs, nt.tip_names};
  auto obs = std::vector<Clade_resolver::Observation>{};
  resolver.observe(nt.tree, obs);
  ASSERT_EQ(std::ssize(obs), 6);

  const auto expected = std::vector<std::pair<std::set<std::string>, int>>{
    {{}, 6}, {{"A","B","C","D","E"}, 5}, {{"A","B","C"}, 3}, {{"A","B"}, 2}, {{"D","E"}, 2},
    {{"A","D"}, 5}};
  for (auto i = 0; i != 6; ++i) {
    auto mrca = expected[i].first.empty() ? nt.tree.root : ref_mrca(nt.tree, expected[i].first);
    EXPECT_EQ(obs[i].t, nt.tree.at(mrca).t) << "clade " << specs.specs[i].id;
    EXPECT_EQ(obs[i].num_tips_below, expected[i].second) << "clade " << specs.specs[i].id;
    EXPECT_EQ(obs[i].num_tips_below, std::ssize(ref_tips_below(nt.tree, mrca)));
  }
  // Monophyly: everything but the paraphyletic AD
  EXPECT_TRUE(obs[0].monophyletic);
  EXPECT_TRUE(obs[1].monophyletic);
  EXPECT_TRUE(obs[2].monophyletic);
  EXPECT_TRUE(obs[3].monophyletic);
  EXPECT_TRUE(obs[4].monophyletic);
  EXPECT_FALSE(obs[5].monophyletic);
}

TEST(Clade_resolver_test, times_are_the_node_times_of_the_tree) {
  auto nt = tree_from_newick("((((A:3,B:3):3,C:6):4,(D:7,E:7):3):4,F:14);");
  auto specs = parse_specs("clade_id\ttaxa\nroot\tall\nAB\tA,B\nDE\tD,E\n");
  auto resolver = Clade_resolver{specs, nt.tip_names};
  auto obs = std::vector<Clade_resolver::Observation>{};
  resolver.observe(nt.tree, obs);
  // Root at 0, tips at 14; (A,B) at 4 + 4 + 3 = 11, (D,E) at 4 + 3 = 7
  EXPECT_DOUBLE_EQ(obs[0].t, 0.0);
  EXPECT_DOUBLE_EQ(obs[1].t, 11.0);
  EXPECT_DOUBLE_EQ(obs[2].t, 7.0);
  EXPECT_DOUBLE_EQ(Clade_resolver::youngest_tip_time(nt.tree), 14.0);
}

TEST(Clade_resolver_test, order_of_the_taxa_does_not_matter) {
  auto nt = tree_from_newick("((((A:3,B:3):3,C:6):4,(D:7,E:7):3):4,F:14);");
  auto a = parse_specs("clade_id\ttaxa\nX\tA,B,C\n");
  auto b = parse_specs("clade_id\ttaxa\nX\tC,A,B\n");
  auto obs_a = std::vector<Clade_resolver::Observation>{};
  auto obs_b = std::vector<Clade_resolver::Observation>{};
  Clade_resolver{a, nt.tip_names}.observe(nt.tree, obs_a);
  Clade_resolver{b, nt.tip_names}.observe(nt.tree, obs_b);
  EXPECT_EQ(obs_a[0].t, obs_b[0].t);
  EXPECT_EQ(obs_a[0].num_tips_below, obs_b[0].num_tips_below);
}

TEST(Clade_resolver_test, rejects_unknown_sample_ids_and_duplicate_tips) {
  auto nt = tree_from_newick("((A:1,B:1):1,C:2);");
  auto specs = parse_specs("clade_id\ttaxa\nX\tA,ZZZ,QQQ\n");
  try {
    Clade_resolver{specs, nt.tip_names};
    ADD_FAILURE() << "expected a rejection";
  } catch (const Clade_spec_error& e) {
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr("not tips of the posterior trees"));
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr("'QQQ'"));
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr("'ZZZ'"));
  }
  auto good = parse_specs("clade_id\ttaxa\nX\tA,B\n");
  try {
    Clade_resolver{good, std::vector<std::string>{"A", "B", "A"}};
    ADD_FAILURE() << "expected a rejection";
  } catch (const Clade_spec_error& e) {
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr("more than one tip named 'A'"));
  }
}

TEST(Clade_resolver_test, works_when_two_node_times_are_equal) {
  // Rounding in a .trees file can make a parent and its child print the same time.  Resolution uses
  // the parent links only, so it still finds the right node.
  auto nt = tree_from_newick("(((A:1,B:1):0,C:1):1,D:2);");
  auto specs = parse_specs("clade_id\ttaxa\nAB\tA,B\nABC\tA,B,C\n");
  auto resolver = Clade_resolver{specs, nt.tip_names};
  auto obs = std::vector<Clade_resolver::Observation>{};
  resolver.observe(nt.tree, obs);
  EXPECT_EQ(obs[0].num_tips_below, 2);
  EXPECT_TRUE(obs[0].monophyletic);
  EXPECT_EQ(obs[1].num_tips_below, 3);
  EXPECT_TRUE(obs[1].monophyletic);
  EXPECT_DOUBLE_EQ(obs[0].t, obs[1].t);  // The two nodes really are at the same time here
}

// Exact sets versus the MCC tree's random fingerprints
// ====================================================

TEST(Clade_resolver_test, agrees_with_the_mcc_trees_exact_match_logic) {
  // mcc_tree.cpp decides monophyly by comparing random 64-bit clade fingerprints, which can collide.
  // Clade_resolver compares the sorted sample sets exactly.  On a posterior sample where no collision
  // happened, the two must agree on every node of the MCC tree and in every base tree.
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(20, 11, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto run = make_run(pool, tree, 11, /*fix_topology=*/false);

  auto trees = std::vector<Phylo_tree>{};
  for (auto i = 0; i != 20; ++i) {
    run->do_mcmc_steps(scaled_steps(200'000));
    trees.push_back(run->tree());
  }
  auto base_trees = std::vector<Phylo_tree*>{};
  for (auto& t : trees) { base_trees.push_back(&t); }
  auto bitgen = absl::BitGen{};
  auto mcc = derive_mcc_tree(base_trees, bitgen);

  // The tip names, by tip node index (identical in every base tree)
  auto tip_names = std::vector<std::string>{};
  for (const auto& node : index_order_traversal(trees.front())) {
    if (trees.front().at(node).is_tip()) { tip_names.push_back(trees.front().at(node).name); }
  }

  auto num_inner_checked = 0;
  for (const auto& mcc_node : index_order_traversal(mcc)) {
    if (mcc.at(mcc_node).is_tip()) { continue; }
    auto names = ref_tips_below(trees[mcc.master_base_tree_index()],
                               corresponding_node_to(mcc, mcc_node, mcc.master_base_tree_index()));
    if (std::ssize(names) < 2) { continue; }
    ++num_inner_checked;

    auto spec_text = absl::StrCat("clade_id\ttaxa\nX\t", absl::StrJoin(names, ","), "\n");
    auto specs = parse_specs(spec_text);
    auto resolver = Clade_resolver{specs, tip_names};
    auto obs = std::vector<Clade_resolver::Observation>{};

    auto num_exact = 0;
    auto sum_t = 0.0;
    auto sum_t_mrca = 0.0;
    for (auto i = 0; i != std::ssize(trees); ++i) {
      resolver.observe(trees[i], obs);
      const auto& info = mcc.at(mcc_node).corresponding_node_infos().at(i);
      EXPECT_EQ(obs[0].monophyletic, info.is_monophyletic_in_base_tree)
          << "clade of " << std::ssize(names) << " tips, base tree " << i;
      EXPECT_EQ(obs[0].t, trees[i].at(info.node_in_base_tree).t);
      sum_t_mrca += obs[0].t;
      if (obs[0].monophyletic) {
        sum_t += obs[0].t;
        ++num_exact;
      }
    }
    // The MCC's two averages are reproduced exactly from the exact-set resolution.
    EXPECT_NEAR(mcc.at(mcc_node).posterior_support(),
                static_cast<double>(num_exact) / std::ssize(trees), 1e-12);
    EXPECT_NEAR(mcc.at(mcc_node).t(), sum_t / num_exact, 1e-9);
    EXPECT_NEAR(mcc.at(mcc_node).t_mrca(), sum_t_mrca / std::ssize(trees), 1e-9);
  }
  EXPECT_GT(num_inner_checked, 5);
}

// t versus t_mrca: fixed and free topology
// ========================================

TEST(Clade_summary_test, fixed_topology_makes_the_two_quantities_identical) {
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(20, 21, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto run = make_run(pool, tree, 21, /*fix_topology=*/true);

  auto tip_names = std::vector<std::string>{};
  for (const auto& node : index_order_traversal(tree)) {
    if (tree.at(node).is_tip()) { tip_names.push_back(tree.at(node).name); }
  }
  // A clade of the starting tree, which the fixed topology preserves in every sampled tree
  auto some_inner = k_no_node;
  for (const auto& node : index_order_traversal(tree)) {
    if (tree.at(node).is_inner_node() && node != tree.root
        && std::ssize(ref_tips_below(tree, node)) >= 3) {
      some_inner = node;
      break;
    }
  }
  ASSERT_NE(some_inner, k_no_node);
  auto names = ref_tips_below(tree, some_inner);
  auto specs = parse_specs(absl::StrCat("clade_id\ttaxa\nroot\tall\nC\t", absl::StrJoin(names, ","), "\n"));
  auto resolver = Clade_resolver{specs, tip_names};

  auto samples = std::vector<Clade_samples>(2);
  samples[0].clade_id = "root";
  samples[1].clade_id = "C";
  auto obs = std::vector<Clade_resolver::Observation>{};
  for (auto i = 0; i != 40; ++i) {
    run->do_mcmc_steps(scaled_steps(100'000));
    resolver.observe(run->tree(), obs);
    auto youngest = Clade_resolver::youngest_tip_time(run->tree());
    for (auto c = 0; c != 2; ++c) {
      samples[c].age_years.push_back((youngest - obs[c].t) / 365.0);
      samples[c].monophyletic.push_back(obs[c].monophyletic ? 1 : 0);
      samples[c].num_tips_below.push_back(obs[c].num_tips_below);
      samples[c].chain_of_sample.push_back(0);
    }
  }
  auto thresholds = Summary_thresholds{};
  for (auto c = 0; c != 2; ++c) {
    auto s = summarize_clade(samples[c], specs.specs[c], 1, thresholds);
    EXPECT_EQ(s.posterior_support, 1.0) << s.clade_id;
    EXPECT_EQ(s.num_monophyletic_samples, s.num_samples_total);
    // With a fixed topology the clade exists in every tree, so conditioning on it changes nothing
    EXPECT_EQ(s.mrca.mean, s.clade.mean) << s.clade_id;
    EXPECT_EQ(s.mrca.median, s.clade.median) << s.clade_id;
    EXPECT_EQ(s.mrca.shortest_interval.lo, s.clade.shortest_interval.lo) << s.clade_id;
    EXPECT_EQ(s.mrca.shortest_interval.hi, s.clade.shortest_interval.hi) << s.clade_id;
    EXPECT_EQ(s.min_tips_below, s.max_tips_below);
  }
}

TEST(Clade_summary_test, free_topology_keeps_the_two_quantities_apart) {
  // Two trees, built by hand so that the arithmetic is exact.  {A,B} is a clade in the first tree at
  // age 1 year (365 days below the youngest tip) and is paraphyletic in the second, where the MRCA of
  // A and B is the root's child, at age 3 years.
  auto specs = parse_specs("clade_id\ttaxa\nAB\tA,B\n");
  auto t1 = tree_from_newick("(((A:365,B:365):730,C:1095):365,D:1460);");
  auto t2 = tree_from_newick("(((A:365,C:365):730,B:1095):365,D:1460);");
  // Tip order differs between the two Newick strings, so resolve each with its own tip names; a real
  // .trees file has one translate table for all of its trees.
  auto obs1 = std::vector<Clade_resolver::Observation>{};
  auto obs2 = std::vector<Clade_resolver::Observation>{};
  Clade_resolver{specs, t1.tip_names}.observe(t1.tree, obs1);
  Clade_resolver{specs, t2.tip_names}.observe(t2.tree, obs2);

  EXPECT_TRUE(obs1[0].monophyletic);
  EXPECT_EQ(obs1[0].num_tips_below, 2);
  EXPECT_FALSE(obs2[0].monophyletic);
  EXPECT_EQ(obs2[0].num_tips_below, 3);

  auto samples = Clade_samples{};
  samples.clade_id = "AB";
  // Three trees of shape 1 (age 1 year) and one of shape 2 (age 3 years)
  for (auto i = 0; i != 3; ++i) {
    samples.age_years.push_back((Clade_resolver::youngest_tip_time(t1.tree) - obs1[0].t) / 365.0);
    samples.monophyletic.push_back(1);
    samples.num_tips_below.push_back(2);
    samples.chain_of_sample.push_back(0);
  }
  samples.age_years.push_back((Clade_resolver::youngest_tip_time(t2.tree) - obs2[0].t) / 365.0);
  samples.monophyletic.push_back(0);
  samples.num_tips_below.push_back(3);
  samples.chain_of_sample.push_back(0);

  auto s = summarize_clade(samples, specs.specs[0], 1, Summary_thresholds{});
  EXPECT_EQ(s.num_samples_total, 4);
  EXPECT_EQ(s.num_monophyletic_samples, 3);
  EXPECT_DOUBLE_EQ(s.posterior_support, 0.75);
  // Unconditional: (1 + 1 + 1 + 3) / 4 = 1.5 years.  Conditional on monophyly: exactly 1 year.
  EXPECT_DOUBLE_EQ(s.mrca.mean, 1.5);
  EXPECT_DOUBLE_EQ(s.clade.mean, 1.0);
  EXPECT_EQ(s.mrca.num_samples, 4);
  EXPECT_EQ(s.clade.num_samples, 3);
  EXPECT_EQ(s.min_tips_below, 2);
  EXPECT_EQ(s.max_tips_below, 3);
  EXPECT_THAT(s.warnings, testing::Contains(testing::HasSubstr("not_always_monophyletic")));
}

TEST(Clade_summary_test, a_set_that_is_never_monophyletic_has_no_clade_age) {
  auto specs = parse_specs("clade_id\ttaxa\nAD\tA,D\n");
  auto samples = Clade_samples{};
  samples.clade_id = "AD";
  for (auto i = 0; i != 10; ++i) {
    samples.age_years.push_back(5.0 + i);
    samples.monophyletic.push_back(0);
    samples.num_tips_below.push_back(4);
    samples.chain_of_sample.push_back(0);
  }
  auto s = summarize_clade(samples, specs.specs[0], 1, Summary_thresholds{});
  EXPECT_EQ(s.posterior_support, 0.0);
  EXPECT_FALSE(s.clade.is_defined());
  EXPECT_TRUE(s.mrca.is_defined());
  EXPECT_THAT(s.warnings, testing::Contains("never_monophyletic"));
}

// Summary arithmetic and diagnostics
// ==================================

TEST(Clade_summary_test, statistics_of_a_hand_checked_sample) {
  auto specs = parse_specs("clade_id\ttaxa\nX\tA,B\n");
  auto samples = Clade_samples{};
  samples.clade_id = "X";
  for (auto v : {1.0, 2.0, 3.0, 4.0, 100.0}) {
    samples.age_years.push_back(v);
    samples.monophyletic.push_back(1);
    samples.num_tips_below.push_back(2);
    samples.chain_of_sample.push_back(0);
  }
  auto thresholds = Summary_thresholds{};
  thresholds.credible_mass = 0.6;
  auto s = summarize_clade(samples, specs.specs[0], 1, thresholds);
  EXPECT_EQ(s.num_samples_total, 5);
  EXPECT_DOUBLE_EQ(s.mrca.mean, 22.0);
  EXPECT_DOUBLE_EQ(s.mrca.median, 3.0);
  EXPECT_NEAR(s.mrca.stddev, std::sqrt(1902.5), 1e-9);  // sum of squares 7610 over n - 1 = 4
  EXPECT_DOUBLE_EQ(s.mrca.shortest_interval.lo, 1.0);
  EXPECT_DOUBLE_EQ(s.mrca.shortest_interval.hi, 3.0);
  EXPECT_TRUE(s.mrca.interval_is_hpd());
  EXPECT_EQ(s.min_tips_below, 2);
  EXPECT_EQ(s.median_tips_below, 2);
  EXPECT_EQ(s.max_tips_below, 2);
}

TEST(Clade_summary_test, per_chain_sample_counts_and_pooled_ess) {
  auto specs = parse_specs("clade_id\ttaxa\nX\tA,B\n");
  auto prng = std::mt19937{3};
  auto normal = std::normal_distribution<double>{100.0, 5.0};
  auto samples = Clade_samples{};
  samples.clade_id = "X";
  for (auto chain = 0; chain != 3; ++chain) {
    for (auto i = 0; i != 1000 + 100 * chain; ++i) {
      samples.age_years.push_back(normal(prng));
      samples.monophyletic.push_back(1);
      samples.num_tips_below.push_back(2);
      samples.chain_of_sample.push_back(chain);
    }
  }
  auto s = summarize_clade(samples, specs.specs[0], 3, Summary_thresholds{});
  EXPECT_THAT(s.num_samples_per_chain, testing::ElementsAre(1000, 1100, 1200));
  EXPECT_EQ(s.mrca.num_chains, 3);
  EXPECT_EQ(s.mrca.num_samples, 3300);
  // Independent draws: the pooled ESS is close to the total number of samples
  EXPECT_GT(s.mrca.ess.ess, 0.7 * 3300);
  EXPECT_NEAR(s.mrca.rank_split_rhat, 1.0, 0.02);
  EXPECT_THAT(s.warnings, testing::IsEmpty());
}

TEST(Clade_summary_test, diagnostics_fire_on_a_low_ess_and_on_disagreeing_chains) {
  auto specs = parse_specs("clade_id\ttaxa\nX\tA,B\n");
  auto samples = Clade_samples{};
  samples.clade_id = "X";
  // Two chains sitting in different places: a pooled 95% interval here would be meaningless
  for (auto chain = 0; chain != 2; ++chain) {
    auto v = 0.0;
    auto prng = std::mt19937{static_cast<uint32_t>(chain + 1)};
    auto normal = std::normal_distribution<double>{0.0, 1.0};
    for (auto i = 0; i != 500; ++i) {
      v = 0.99 * v + 0.1 * normal(prng);  // strongly autocorrelated
      samples.age_years.push_back(v + 50.0 * chain);
      samples.monophyletic.push_back(1);
      samples.num_tips_below.push_back(2);
      samples.chain_of_sample.push_back(chain);
    }
  }
  auto s = summarize_clade(samples, specs.specs[0], 2, Summary_thresholds{});
  EXPECT_LT(s.mrca.ess.ess, 200.0);
  EXPECT_GT(s.mrca.rank_split_rhat, 1.01);
  EXPECT_THAT(s.warnings, testing::Contains(testing::HasSubstr("mrca_low_ess")));
  EXPECT_THAT(s.warnings, testing::Contains(testing::HasSubstr("mrca_rank_split_rhat")));
  EXPECT_GT(s.mrca.hdr.num_components, 1);  // and it says the pooled posterior is bimodal
  EXPECT_FALSE(s.mrca.interval_is_hpd());   // so the shortest interval is not an HPD interval
  EXPECT_THAT(s.warnings,
              testing::Contains(testing::HasSubstr("mrca_multimodal_interval_is_not_hpd")));
}

TEST(Clade_summary_test, an_empty_sample_is_reported_not_crashed) {
  auto specs = parse_specs("clade_id\ttaxa\nX\tA,B\n");
  auto samples = Clade_samples{};
  samples.clade_id = "X";
  auto s = summarize_clade(samples, specs.specs[0], 1, Summary_thresholds{});
  EXPECT_EQ(s.num_samples_total, 0);
  EXPECT_FALSE(s.mrca.is_defined());
  EXPECT_THAT(s.warnings, testing::Contains("no_post_burn_in_samples"));
}

// Tables
// ======

TEST(Clade_summary_test, raw_samples_table_round_trip) {
  auto samples = std::vector<Clade_samples>(2);
  samples[0].clade_id = "root";
  samples[1].clade_id = "AB";
  auto states = std::vector<int64_t>{};
  for (auto i = 0; i != 7; ++i) {
    states.push_back(1000 * (i + 1));
    samples[0].age_years.push_back(140000.0 + i * 0.1);  // not exactly representable: tests the format
    samples[0].monophyletic.push_back(1);
    samples[0].num_tips_below.push_back(6);
    samples[0].chain_of_sample.push_back(i % 2);
    samples[1].age_years.push_back(30000.0 - i / 3.0);
    samples[1].monophyletic.push_back(i % 3 == 0 ? 1 : 0);
    samples[1].num_tips_below.push_back(i % 3 == 0 ? 2 : 3);
    samples[1].chain_of_sample.push_back(i % 2);
  }
  auto provenance = Summary_provenance{};
  provenance.age_reference_description = "years_before_1950-01-01 (astronomical year numbering)";
  provenance.burn_in_description = "fraction 0.3";
  provenance.anchor_description = "sample 'X'";

  auto oss = std::ostringstream{};
  write_clade_samples_table(samples, states, provenance, oss);
  auto iss = std::istringstream{oss.str()};
  auto back = read_clade_samples_table(iss);

  ASSERT_EQ(std::ssize(back.samples), 2);
  EXPECT_EQ(back.num_chains, 2);
  EXPECT_THAT(back.states, testing::ElementsAreArray(states));
  for (auto c = 0; c != 2; ++c) {
    EXPECT_EQ(back.samples[c].clade_id, samples[c].clade_id);
    ASSERT_EQ(back.samples[c].num_samples(), samples[c].num_samples());
    for (auto i = 0; i != samples[c].num_samples(); ++i) {
      EXPECT_DOUBLE_EQ(back.samples[c].age_years[i], samples[c].age_years[i]);
      EXPECT_EQ(back.samples[c].monophyletic[i], samples[c].monophyletic[i]);
      EXPECT_EQ(back.samples[c].num_tips_below[i], samples[c].num_tips_below[i]);
      EXPECT_EQ(back.samples[c].chain_of_sample[i], samples[c].chain_of_sample[i]);
    }
  }
  // The provenance comments survive, so a re-summary can state where the numbers came from
  EXPECT_THAT(back.comment_lines,
              testing::Contains(testing::HasSubstr("years_before_1950-01-01")));
}

TEST(Clade_summary_test, raw_samples_table_state_column) {
  auto header = std::string{"clade_id\tchain\tstate\tage_years\tmonophyletic\ttips_below_mrca\n"};

  auto good = std::istringstream{header + "AB\t0\t-1\t100.5\t1\t2\n" + "AB\t0\t2000\t101.5\t0\t3\n"};
  auto back = read_clade_samples_table(good);
  EXPECT_THAT(back.states, testing::ElementsAre(-1, 2000));

  auto without_state = std::istringstream{
    "clade_id\tchain\tage_years\tmonophyletic\ttips_below_mrca\nAB\t0\t100.5\t1\t2\n"};
  EXPECT_THAT(read_clade_samples_table(without_state).states, testing::ElementsAre(-1));

  // A malformed state is an error, not a silent -1
  auto bad = std::istringstream{header + "AB\t0\tten\t100.5\t1\t2\n"};
  try {
    read_clade_samples_table(bad);
    FAIL() << "a non-numeric state was accepted";
  } catch (const Clade_spec_error& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr("line 2: could not parse the numeric fields"));
  }
}

TEST(Clade_summary_test, summary_table_names_the_conventions_and_both_quantities) {
  auto specs = parse_specs("clade_id\ttaxa\nAB\tA,B\n");
  auto samples = Clade_samples{};
  samples.clade_id = "AB";
  for (auto i = 0; i != 100; ++i) {
    samples.age_years.push_back(1000.0 + i);
    samples.monophyletic.push_back(i < 60 ? 1 : 0);
    samples.num_tips_below.push_back(i < 60 ? 2 : 3);
    samples.chain_of_sample.push_back(0);
  }
  auto s = summarize_clade(samples, specs.specs[0], 1, Summary_thresholds{});
  s.calibration_status = "calibrated:AB";
  auto provenance = Summary_provenance{};
  provenance.age_reference_description = "years_before_the_most_recent_tip_of_each_tree";
  auto oss = std::ostringstream{};
  write_clade_summary_table({s}, provenance, oss);
  auto text = std::move(oss).str();
  EXPECT_THAT(text, testing::HasSubstr("years_before_the_most_recent_tip_of_each_tree"));
  EXPECT_THAT(text, testing::HasSubstr("mrca_age_shortest_interval_lower"));
  EXPECT_THAT(text, testing::HasSubstr("clade_age_shortest_interval_lower"));
  EXPECT_THAT(text, testing::HasSubstr("mrca_age_interval_is_hpd"));
  EXPECT_THAT(text, testing::HasSubstr("mrca_age_hdr_intervals"));
  EXPECT_THAT(text, testing::HasSubstr("mrca_age_eti_lower"));
  EXPECT_THAT(text, testing::HasSubstr("calibrated:AB"));
  EXPECT_THAT(text, testing::HasSubstr("equal-tailed interval (quantiles), which is not a "
                                       "highest-density interval"));
  EXPECT_THAT(text, testing::HasSubstr("This is the set that the term HPD names"));
  // Every data row has as many fields as the header
  auto header_fields = 0;
  auto row_fields = 0;
  auto iss = std::istringstream{text};
  auto line = std::string{};
  while (std::getline(iss, line)) {
    if (line.starts_with("#")) { continue; }
    auto n = 1 + static_cast<int>(std::ranges::count(line, '\t'));
    if (header_fields == 0) { header_fields = n; } else { row_fields = n; }
  }
  EXPECT_GT(header_fields, 30);
  EXPECT_EQ(row_fields, header_fields);
}

// Streaming the posterior trees
// =============================

TEST(Beasty_trees_stream_test, streams_the_same_trees_that_read_beasty_trees_returns) {
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(12, 31, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto run = make_run(pool, tree, 31, /*fix_topology=*/false);
  auto text = write_trees_file(*run, 6, scaled_steps(50'000), 6);

  auto iss1 = std::istringstream{text};
  auto all = read_beasty_trees(iss1, 0);
  EXPECT_EQ(std::ssize(all), 6);

  auto iss2 = std::istringstream{text};
  auto stream = Beasty_trees_stream{iss2};
  EXPECT_EQ(stream.num_tips(), 12);
  auto state = int64_t{};
  auto one = Phylo_tree{};
  auto seen = 0;
  while (stream.next(state, one)) {
    ++seen;
    ASSERT_TRUE(all.contains(state));
    const auto& expected = all.at(state);
    ASSERT_EQ(std::ssize(one), std::ssize(expected));
    for (const auto& node : index_order_traversal(one)) {
      EXPECT_EQ(one.at(node).parent, expected.at(node).parent);
      EXPECT_DOUBLE_EQ(one.at(node).t, expected.at(node).t);
      EXPECT_EQ(one.at(node).name, expected.at(node).name);
    }
  }
  EXPECT_EQ(seen, 6);

  // The tip names the stream reports are the tip node indices of the trees it hands back
  auto iss3 = std::istringstream{text};
  auto stream3 = Beasty_trees_stream{iss3};
  auto names = stream3.tip_names();
  ASSERT_TRUE(stream3.next(state, one));
  for (auto i = 0; i != std::ssize(names); ++i) { EXPECT_EQ(one.at(i).name, names[i]); }
}

TEST(Beasty_trees_stream_test, trees_read_back_satisfy_the_tree_integrity_invariants) {
  // newick_to_phylo_tree() used to leave the tip date bounds at their deliberately illegal defaults
  // (+42 / -42), so every tree read from a trees file failed assert_phylo_tree_integrity in a Debug
  // or --v0-paranoid build.  The bounds are now the exactly known tip times of the file.
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(10, 61, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto run = make_run(pool, tree, 61, /*fix_topology=*/true);
  auto text = write_trees_file(*run, 3, scaled_steps(20'000), 9);

  auto is = std::istringstream{text};
  auto stream = Beasty_trees_stream{is};
  auto state = int64_t{};
  auto back = Phylo_tree{};
  auto seen = 0;
  while (stream.next(state, back)) {
    ++seen;
    assert_phylo_tree_integrity(back, /*force=*/true);  // CHECK-fails if the invariants are broken
    for (const auto& node : index_order_traversal(back)) {
      if (back.at(node).is_tip()) {
        EXPECT_LE(back.at(node).t_min, back.at(node).t);
        EXPECT_LE(back.at(node).t, back.at(node).t_max);
      } else {
        EXPECT_EQ(back.at(node).t_min, -std::numeric_limits<float>::max());
        EXPECT_EQ(back.at(node).t_max, +std::numeric_limits<float>::max());
      }
    }
  }
  EXPECT_EQ(seen, 3);

  // Deep time: the float bounds must still bracket the double time, where the spacing is days
  auto probe = simulate(10, 62, 1.0, 0.0, k_mu_site_day);
  auto scale = 200'000.0 * 365.0 / -probe.nodes.back().t;
  auto deep_sim = simulate(10, 62, scale, 0.0, k_mu_site_day / scale);
  auto deep = sim_tree(deep_sim);
  auto deep_text = write_trees_file_from_trees({{0, &deep}}, 12);
  auto deep_is = std::istringstream{deep_text};
  auto deep_stream = Beasty_trees_stream{deep_is};
  ASSERT_TRUE(deep_stream.next(state, back));
  assert_phylo_tree_integrity(back, /*force=*/true);
  for (const auto& node : index_order_traversal(back)) {
    if (not back.at(node).is_tip()) { continue; }
    EXPECT_LE(back.at(node).t_min, back.at(node).t);
    EXPECT_LE(back.at(node).t, back.at(node).t_max);
  }
}

TEST(Beasty_trees_stream_test, scan_states_and_burn_in_and_thinning) {
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(8, 41, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto run = make_run(pool, tree, 41, /*fix_topology=*/true);
  auto every = scaled_steps(20'000);
  auto text = write_trees_file(*run, 10, every, 6);

  auto iss = std::istringstream{text};
  auto states = scan_beasty_tree_states(iss);
  ASSERT_EQ(std::ssize(states), 10);
  EXPECT_EQ(states.front(), 0);
  EXPECT_EQ(states.back(), 9 * every);
  EXPECT_TRUE(std::ranges::is_sorted(states));

  auto count_kept = [&](int64_t min_state, int64_t thin) {
    auto is = std::istringstream{text};
    auto s = Beasty_trees_stream{is};
    s.set_min_state(min_state);
    s.set_every(thin);
    auto state = int64_t{};
    auto t = Phylo_tree{};
    auto n = 0;
    while (s.next(state, t)) {
      EXPECT_GE(state, min_state);
      EXPECT_EQ(state % thin, 0);
      ++n;
    }
    return n;
  };
  EXPECT_EQ(count_kept(0, 1), 10);
  EXPECT_EQ(count_kept(3 * every + 1, 1), 6);
  EXPECT_EQ(count_kept(0, 2 * every), 5);
  EXPECT_EQ(count_kept(10 * every, 1), 0);
}

// Deep-time serialization precision
// =================================

TEST(Clade_summary_test, trees_file_precision_at_human_depths) {
  // The .trees format stores branch lengths in years, so a node age is the sum of the branch lengths
  // on its root-to-tip path.  At 6 significant digits a 140,000-year branch is quantized to 0.1
  // years, and the error accumulates along the path.  This measures the error at four depths and
  // checks that --v0-out-trees-digits removes it.
  for (auto depth_years : {50'000.0, 100'000.0, 140'000.0, 200'000.0}) {
    // Place the simulated genealogy so that its root sits at the target depth
    auto probe = simulate(20, 51, 1.0, 0.0, k_mu_site_day);
    auto scale = depth_years * 365.0 / -probe.nodes.back().t;
    auto sim = simulate(20, 51, scale, 0.0, k_mu_site_day / scale);
    auto tree = sim_tree(sim);
    EXPECT_NEAR(-tree.at_root().t / 365.0, depth_years, 0.05 * depth_years);

    auto tip_names = std::vector<std::string>{};
    for (const auto& node : index_order_traversal(tree)) {
      if (tree.at(node).is_tip()) { tip_names.push_back(tree.at(node).name); }
    }
    // Every inner clade of the tree, plus the root
    auto clades = std::vector<std::set<std::string>>{{}};
    auto spec_lines = std::string{"clade_id\ttaxa\nroot\tall\n"};
    auto next_id = 0;
    for (const auto& node : index_order_traversal(tree)) {
      if (not tree.at(node).is_inner_node() || node == tree.root) { continue; }
      auto names = ref_tips_below(tree, node);
      if (std::ssize(names) < 2) { continue; }
      clades.push_back(names);
      spec_lines += absl::StrCat("C", next_id++, "\t", absl::StrJoin(names, ","), "\n");
    }
    auto specs = parse_specs(spec_lines);
    auto resolver = Clade_resolver{specs, tip_names};
    auto exact = ages_of(tree, clades);

    for (auto digits : {6, 9, 12}) {
      auto text = write_trees_file_from_trees({{0, &tree}}, digits);
      auto is = std::istringstream{text};
      auto stream = Beasty_trees_stream{is};
      auto state = int64_t{};
      auto back = Phylo_tree{};
      ASSERT_TRUE(stream.next(state, back));
      auto obs = std::vector<Clade_resolver::Observation>{};
      resolver.observe(back, obs);
      auto youngest = Clade_resolver::youngest_tip_time(back);
      auto max_error = 0.0;
      for (auto i = 0; i != std::ssize(obs); ++i) {
        max_error = std::max(max_error, std::abs((youngest - obs[i].t) / 365.0 - exact[i]));
      }
      std::cerr << absl::StreamFormat(
          "trees precision: depth %.0f yr, %d digits -> max clade age error %.6g yr\n",
          depth_years, digits, max_error);
      // Measured bounds, not guesses: 6 digits loses a fraction of a year at these depths, 9 digits
      // loses less than a day, and 12 digits is exact well below a second.
      if (digits == 6) {
        EXPECT_LT(max_error, 5.0) << "depth " << depth_years;
      } else if (digits == 9) {
        EXPECT_LT(max_error, 1.0 / 365.0) << "depth " << depth_years;
      } else {
        EXPECT_LT(max_error, 1e-6) << "depth " << depth_years;
      }
    }
  }
}

}  // namespace delphy
