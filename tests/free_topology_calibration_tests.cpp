// Tests for age calibrations with a free topology: constraint-aware starting trees, the monophyly
// constraint and the calibration term in the topology moves.
//
// All numbers here are test inputs (simulation truth or hand-picked windows), never defaults.

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "ctpl_stl.h"

#include "api.h"
#include "calibration.h"
#include "calibration_init.h"
#include "estd.h"
#include "external_tree.h"
#include "newick.h"
#include "phylo_tree.h"
#include "pop_model.h"
#include "run.h"

namespace delphy {

namespace {

constexpr auto k_sites = 400;
constexpr auto k_coal_days = 500.0;  // Constant population of the tests: pair coalescence rate 1 / k_coal_days

auto parse_config(const std::string& text) -> Calibration_set {
  auto is = std::istringstream{text};
  return read_calibration_config(is);
}

// A calibration stanza on `taxa` (a space-separated list, or "all") with a uniform window of ages in years
// before 2020-01-01, i.e., before t = 0
auto uniform_stanza(const std::string& name, const std::string& taxa, double youngest, double oldest) -> std::string {
  return absl::StrFormat(
      "calibration %s\n  taxa %s\n  age_units years_bp\n  age_reference_year 2020\n  distribution uniform\n"
      "  youngest %.12g\n  oldest %.12g\nend\n", name, taxa, youngest, oldest);
}

auto normal_stanza(const std::string& name, const std::string& taxa, double mean, double stddev) -> std::string {
  return absl::StrFormat(
      "calibration %s\n  taxa %s\n  age_units years_bp\n  age_reference_year 2020\n  distribution normal\n"
      "  mean %.12g\n  stddev %.12g\nend\n", name, taxa, mean, stddev);
}

auto letter(int i) -> Real_seq_letter { return real_seq_letter_from_index(i); }

struct Data {
  Real_sequence ref_sequence;
  std::vector<Tip_desc> tip_descs;
  std::string newick;  // The true genealogy, branch lengths in days
};

// A Kingman genealogy (all tips at t = 0) with Poisson substitutions at `mu_site_day`
auto simulate(int num_tips, uint32_t seed, double mu_site_day) -> Data {
  auto rng = std::mt19937{seed};
  auto uni = [&] { return std::uniform_real_distribution<double>{0.0, 1.0}(rng); };
  struct Node { double t; int left; int right; };
  auto nodes = std::vector<Node>(num_tips, Node{0.0, -1, -1});
  auto active = std::vector<int>{};
  for (auto i = 0; i != num_tips; ++i) { active.push_back(i); }
  auto t = 0.0;
  while (std::ssize(active) > 1) {
    auto k = static_cast<double>(std::ssize(active));
    t -= -std::log(1.0 - uni()) / (k * (k - 1) / 2.0 / k_coal_days);
    auto i = static_cast<int>(rng() % active.size());
    auto a = active[i];
    active.erase(active.begin() + i);
    auto j = static_cast<int>(rng() % active.size());
    auto b = active[j];
    active.erase(active.begin() + j);
    nodes.push_back(Node{t, a, b});
    active.push_back(static_cast<int>(std::ssize(nodes)) - 1);
  }
  auto root = static_cast<int>(std::ssize(nodes)) - 1;

  auto data = Data{};
  data.ref_sequence.resize(k_sites);
  for (auto& r : data.ref_sequence) { r = letter(static_cast<int>(rng() % 4)); }
  auto seqs = std::vector<Real_sequence>(nodes.size());
  seqs[root] = data.ref_sequence;
  for (auto n = root; n >= num_tips; --n) {
    for (auto c : {nodes[n].left, nodes[n].right}) {
      seqs[c] = seqs[n];
      auto num = std::poisson_distribution<int>{mu_site_day * k_sites * (nodes[c].t - nodes[n].t)}(rng);
      for (auto m = 0; m != num; ++m) {
        auto s = static_cast<int>(rng() % k_sites);
        seqs[c][s] = letter((index_of(seqs[c][s]) + 1 + static_cast<int>(rng() % 3)) % 4);
      }
    }
  }
  for (auto i = 0; i != num_tips; ++i) {
    auto td = Tip_desc{};
    td.name = absl::StrFormat("S%03d", i);
    td.t_min = td.t_max = 0.0f;
    for (auto s = 0; s != k_sites; ++s) {
      if (seqs[i][s] != data.ref_sequence[s]) { td.seq_deltas.push_back(Seq_delta{s, data.ref_sequence[s], seqs[i][s]}); }
    }
    data.tip_descs.push_back(std::move(td));
  }
  auto newick_of = std::function<std::string(int)>{};
  newick_of = [&](int n) -> std::string {
    if (nodes[n].left < 0) { return absl::StrFormat("S%03d", n); }
    auto child = [&](int c) { return absl::StrFormat("%s:%.12g", newick_of(c), nodes[c].t - nodes[n].t); };
    return absl::StrCat("(", child(nodes[n].left), ",", child(nodes[n].right), ")");
  };
  data.newick = newick_of(root) + ";";
  return data;
}

auto tree_from_newick(const std::string& newick, const Data& data) -> Phylo_tree {
  auto is = std::istringstream{newick};
  auto nt = read_newick_tree(is);
  auto bitgen = std::mt19937{7};
  auto options = External_tree_options{};
  options.units = Branch_length_units::days;
  options.num_observed_sites = k_sites;
  return build_phylo_tree_from_newick(nt, data.ref_sequence, data.tip_descs, bitgen, options);
}

auto tip_names_below(const Phylo_tree& tree, Node_index node) -> std::set<std::string> {
  auto result = std::set<std::string>{};
  auto stack = std::vector<Node_index>{node};
  while (not stack.empty()) {
    auto n = stack.back();
    stack.pop_back();
    if (tree.at(n).is_tip()) { result.insert(tree.at(n).name); }
    else { for (auto c : tree.at(n).children) { stack.push_back(c); } }
  }
  return result;
}

// Independent of the production code: is `names` exactly the tip set below some node?
auto ref_is_clade(const Phylo_tree& tree, const std::vector<std::string>& names) -> bool {
  auto wanted = std::set<std::string>(names.begin(), names.end());
  for (const auto& node : index_order_traversal(tree)) {
    if (tip_names_below(tree, node) == wanted) { return true; }
  }
  return false;
}

auto ref_node_of(const Phylo_tree& tree, const Calibration& c) -> Node_index {
  if (c.all_tips) { return tree.root; }
  auto wanted = std::set<std::string>(c.taxa.begin(), c.taxa.end());
  for (const auto& node : index_order_traversal(tree)) {
    if (tip_names_below(tree, node) == wanted) { return node; }
  }
  return k_no_node;
}

// sum_c log f_c(age_c), resolving each target from its tip names; -infinity if a set is not a clade
auto ref_log_calibration_prior(const Calibration_set& set, const Phylo_tree& tree) -> double {
  auto result = 0.0;
  for (const auto& c : set.calibrations) {
    auto node = ref_node_of(tree, c);
    if (node == k_no_node) { return -std::numeric_limits<double>::infinity(); }
    auto age = (c.reference_t - tree.at(node).t) / 365.0;
    if (age < c.prior.youngest_age || age > c.prior.oldest_age) { return -std::numeric_limits<double>::infinity(); }
    switch (c.prior.distribution) {
      case Calibration_distribution::normal: {
        auto z = (age - c.prior.mean) / c.prior.stddev;
        result += -std::log(c.prior.stddev) - 0.5 * std::log(2.0 * std::acos(-1.0)) - 0.5 * z * z;
        break;
      }
      case Calibration_distribution::uniform:
        result += -std::log(c.prior.oldest_age - c.prior.youngest_age);
        break;
      case Calibration_distribution::lognormal:
        ADD_FAILURE() << "not used here";
        break;
    }
  }
  return result;
}

// Canonical text of the rooted topology, for detecting topology changes
auto topology_signature(const Phylo_tree& tree, Node_index node) -> std::string {
  if (tree.at(node).is_tip()) { return tree.at(node).name; }
  auto a = topology_signature(tree, tree.at(node).left_child());
  auto b = topology_signature(tree, tree.at(node).right_child());
  if (b < a) { std::swap(a, b); }
  return absl::StrCat("(", a, ",", b, ")");
}

auto tip_names_of(const Data& data) -> std::vector<std::string> {
  auto result = std::vector<std::string>{};
  for (const auto& td : data.tip_descs) { result.push_back(td.name); }
  return result;
}

auto constrained_start(const Data& data, const std::string& guide_newick, const Calibration_set& set,
                       Constrained_init_summary* summary = nullptr) -> Phylo_tree {
  return build_calibration_constrained_initial_tree(
      tree_from_newick(guide_newick, data), data.ref_sequence, data.tip_descs, set, std::nullopt, k_sites, summary);
}

// `const_pop`: a fixed constant population of k_coal_days instead of the default exponential model.  The
// checkpoint format stores only the exponential and Skygrid models.
auto make_free_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, double mu,
                   bool const_pop = true) -> std::shared_ptr<Run> {
  auto run = std::make_shared<Run>(pool, std::mt19937{seed}, seed, tree);
  run->set_topology_moves_enabled(true);
  run->set_mu(mu);
  run->set_mu_move_enabled(false);
  run->set_num_parts(1);
  if (const_pop) {
    run->set_pop_model(std::make_shared<Const_pop_model>(k_coal_days));
    run->set_final_pop_size_move_enabled(false);
    run->set_pop_growth_rate_move_enabled(false);
  }
  run->set_target_coal_prior_cells(4000);
  return run;
}

auto scaled_steps(int64_t steps) -> int64_t {
  return estd::is_debug_enabled ? std::max<int64_t>(steps / 25, 20'000) : steps;
}

// A fixed 8-tip guide topology, so that the constraints below are known not to be clades of it
const auto k_guide_8 = std::string{
  "((((S000:10,S001:10):10,S002:20):10,S003:30):10,(((S004:10,S005:10):10,S006:20):10,S007:30):10);"};

auto expect_valid_constrained_start(const Data& data, const Calibration_set& set, const Phylo_tree& tree) -> void {
  assert_phylo_tree_integrity(tree, true);
  assert_phylo_tree_matches_tip_descs(tree, data.ref_sequence, data.tip_descs, true);
  EXPECT_EQ(tree.size(), 2 * std::ssize(data.tip_descs) - 1);
  for (const auto& c : set.calibrations) {
    if (not c.all_tips) { EXPECT_TRUE(ref_is_clade(tree, c.taxa)) << c.name; }
  }
  EXPECT_TRUE(std::isfinite(ref_log_calibration_prior(set, tree)));
  EXPECT_NO_THROW(bind_calibrations(set, tree));  // Including the hard bounds at the current times
}

}  // namespace

// Test A: starting trees from the alignment, without a user tree
// ===============================================================

TEST(Free_topology_calibration_init_test, root_only_keeps_the_guide_tree) {
  auto data = simulate(8, 1, 2e-6);
  auto set = parse_config("version 1\n" + normal_stanza("root", "all", 0.2, 0.05));
  auto summary = Constrained_init_summary{};
  auto guide = tree_from_newick(k_guide_8, data);
  auto tree = constrained_start(data, k_guide_8, set, &summary);
  EXPECT_FALSE(summary.rebuilt);
  EXPECT_EQ(summary.num_nodes_moved, 0);  // A normal prior has no hard bound
  EXPECT_EQ(topology_signature(tree, tree.root), topology_signature(guide, guide.root));
  expect_valid_constrained_start(data, set, tree);
}

TEST(Free_topology_calibration_init_test, one_inner_clade_is_imposed_on_the_guide) {
  auto data = simulate(8, 2, 2e-6);
  auto set = parse_config("version 1\n" + normal_stanza("ae", "S000 S004", 0.1, 0.05));
  auto summary = Constrained_init_summary{};
  auto tree = constrained_start(data, k_guide_8, set, &summary);
  EXPECT_TRUE(summary.rebuilt);
  expect_valid_constrained_start(data, set, tree);
  // Outside the constraint, the guide's arrangement is kept, with the clade where its representative S000 was
  EXPECT_TRUE(ref_is_clade(tree, {"S000", "S001", "S004"}));
  EXPECT_TRUE(ref_is_clade(tree, {"S000", "S001", "S002", "S004"}));
  EXPECT_TRUE(ref_is_clade(tree, {"S005", "S006"}));
}

TEST(Free_topology_calibration_init_test, nested_clades) {
  auto data = simulate(8, 3, 2e-6);
  auto set = parse_config("version 1\n" + normal_stanza("outer", "S000 S001 S004 S005", 0.2, 0.05)
                          + normal_stanza("inner", "S000 S004", 0.1, 0.05) + normal_stanza("root", "all", 0.3, 0.1));
  auto tree = constrained_start(data, k_guide_8, set);
  expect_valid_constrained_start(data, set, tree);
}

TEST(Free_topology_calibration_init_test, disjoint_clades) {
  auto data = simulate(8, 4, 2e-6);
  auto set = parse_config("version 1\n" + normal_stanza("a", "S000 S004", 0.1, 0.05)
                          + normal_stanza("b", "S001 S002 S007", 0.1, 0.05));
  auto tree = constrained_start(data, k_guide_8, set);
  expect_valid_constrained_start(data, set, tree);
}

TEST(Free_topology_calibration_init_test, hard_bounds_move_the_node_times_and_carry_the_mutations) {
  auto data = simulate(8, 5, 2e-5);
  // The guide puts every node within 40 days of the tips; both windows are far older and nested
  auto set = parse_config("version 1\n" + uniform_stanza("inner", "S000 S004", 10.0, 12.0)
                          + uniform_stanza("root", "all", 20.0, 21.0));
  auto summary = Constrained_init_summary{};
  auto tree = constrained_start(data, k_guide_8, set, &summary);
  EXPECT_GT(summary.num_nodes_moved, 0);
  expect_valid_constrained_start(data, set, tree);
  auto root_age = -tree.at_root().t / 365.0;
  EXPECT_GT(root_age, 20.0);
  EXPECT_LT(root_age, 21.0);
}

TEST(Free_topology_calibration_init_test, crossing_sets_are_rejected) {
  auto data = simulate(8, 6, 2e-6);
  auto set = parse_config("version 1\n" + normal_stanza("a", "S000 S001 S002", 0.1, 0.05)
                          + normal_stanza("b", "S002 S003", 0.1, 0.05));
  try {
    constrained_start(data, k_guide_8, set);
    FAIL() << "crossing sets accepted";
  } catch (const Calibration_error& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr("nested or disjoint"));
  }
}

TEST(Free_topology_calibration_init_test, other_unsatisfiable_sets_are_rejected) {
  auto data = simulate(8, 7, 2e-6);
  auto names = tip_names_of(data);
  auto reject = [&](const std::string& config, const std::string& needle) {
    try {
      check_calibration_sets_compatible(parse_config("version 1\n" + config), names);
      ADD_FAILURE() << "accepted: " << config;
    } catch (const Calibration_error& e) {
      EXPECT_THAT(e.what(), testing::HasSubstr(needle));
    }
  };
  reject(normal_stanza("a", "S000 S999", 0.1, 0.05), "not in the alignment");
  reject(normal_stanza("a", "S000 S001", 0.1, 0.05) + normal_stanza("b", "S001 S000", 0.2, 0.05), "same samples");
  reject(normal_stanza("a", "all", 0.1, 0.05)
         + normal_stanza("b", "S000 S001 S002 S003 S004 S005 S006 S007", 0.2, 0.05), "name the root");
}

TEST(Free_topology_calibration_init_test, incompatible_hard_bounds_are_rejected) {
  auto data = simulate(8, 8, 2e-6);
  // The clade must be older than 5 years but its ancestor, the root, younger than 2 years
  auto set = parse_config("version 1\n" + uniform_stanza("inner", "S000 S004", 5.0, 6.0)
                          + uniform_stanza("root", "all", 1.0, 2.0));
  EXPECT_THROW(constrained_start(data, k_guide_8, set), Calibration_error);
}

// Test B: the target of the topology moves
// ========================================

namespace {

// Exact reference for 4 isochronous tips with no data, a constant population (pair rate 1/N), the
// constraint that {S000, S001} is a clade and a uniform prior on its age over [a, b] years.
//
// Under Kingman every ranked history has probability 1/18, independently of the coalescence times
// t1 < t2 < t3.  The clade {S000,S001} exists in 4 of them: as the first event in the balanced topology
// (age t1), in the two caterpillars ((S000,S001),S002),S003 and ((S000,S001),S003),S002 (age t1), and as
// the second event of the balanced topology after (S002,S003) (age t2).  With w_i = E[1{t_i in window}]:
//   P(balanced) = (w1 + w2) / (3 w1 + w2),   P(each caterpillar) = w1 / (3 w1 + w2),
//   E[age] = (3 E[t1 1{..}] + E[t2 1{..}]) / (3 w1 + w2).
// t1 ~ Exp(6/N); t2 = t1 + Exp(3/N), whose density is the hypoexponential one.  Integrated numerically.
struct Four_tip_reference {
  double p_balanced = 0.0;
  double p_caterpillar = 0.0;
  double mean_age_years = 0.0;
};

auto four_tip_reference(double a_years, double b_years) -> Four_tip_reference {
  auto l1 = 6.0 / k_coal_days;
  auto l2 = 3.0 / k_coal_days;
  auto f1 = [&](double t) { return l1 * std::exp(-l1 * t); };
  auto f2 = [&](double t) { return l1 * l2 / (l1 - l2) * (std::exp(-l2 * t) - std::exp(-l1 * t)); };
  auto lo = 365.0 * a_years;
  auto hi = 365.0 * b_years;
  auto n = 200'000;  // Simpson
  auto h = (hi - lo) / n;
  auto w1 = 0.0, w2 = 0.0, m1 = 0.0, m2 = 0.0;
  for (auto i = 0; i <= n; ++i) {
    auto t = lo + i * h;
    auto c = (i == 0 || i == n) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
    w1 += c * f1(t); w2 += c * f2(t); m1 += c * t * f1(t); m2 += c * t * f2(t);
  }
  auto z = 3 * w1 + w2;
  return {(w1 + w2) / z, w1 / z, (3 * m1 + m2) / z / 365.0};
}

struct Four_tip_estimate {
  double p_balanced = 0.0, se_balanced = 0.0;
  double p_cat_2 = 0.0, se_cat_2 = 0.0;   // ((S000,S001),S002),S003
  double mean_age = 0.0, se_age = 0.0;
  int64_t topology_changes = 0;
};

auto four_tip_chain(const Calibration_set& set, uint32_t seed, int64_t steps, int samples) -> Four_tip_estimate {
  auto data = Data{};
  data.ref_sequence.assign(k_sites, Real_seq_letter::A);
  for (auto i = 0; i != 4; ++i) {
    auto td = Tip_desc{};
    td.name = absl::StrFormat("S%03d", i);
    td.t_min = td.t_max = 0.0f;
    data.tip_descs.push_back(std::move(td));
  }
  auto pool = ctpl::thread_pool{1};
  // The guide is a caterpillar that violates the constraint; the start is built from it
  auto tree = constrained_start(data, "(((S000:1,S002:1):1,S001:2):1,S003:3);", set);
  auto run = make_free_run(pool, tree, seed, 1e-12);  // Essentially no data
  run->set_calibrations(set);

  auto bal = std::vector<double>{};
  auto cat2 = std::vector<double>{};
  auto age = std::vector<double>{};
  auto last = std::string{};
  auto est = Four_tip_estimate{};
  auto per = std::max<int64_t>(steps / samples, 1);
  for (auto s = 0; s != samples; ++s) {
    run->do_mcmc_steps(static_cast<int>(per));
    const auto& t = run->tree();
    auto node = ref_node_of(t, set.calibrations[0]);
    EXPECT_NE(node, k_no_node) << "the constrained set stopped being a clade";
    if (node == k_no_node) { break; }
    auto sib = t.at(t.at(node).parent).sibling_of(node);
    bal.push_back(t.at(sib).is_inner_node() ? 1.0 : 0.0);
    cat2.push_back((t.at(sib).is_tip() && t.at(sib).name == "S002") ? 1.0 : 0.0);
    age.push_back(set.calibrations[0].age_of(t.at(node).t));
    auto sig = topology_signature(t, t.root);
    if (sig != last) { ++est.topology_changes; last = sig; }
  }
  // Mean and batch-means standard error over 25 batches
  auto mean_se = [](const std::vector<double>& x) -> std::pair<double, double> {
    auto nb = 25;
    auto len = std::ssize(x) / nb;
    auto means = std::vector<double>{};
    for (auto b = 0; b != nb; ++b) {
      auto sum = 0.0;
      for (auto i = b * len; i != (b + 1) * len; ++i) { sum += x[i]; }
      means.push_back(sum / len);
    }
    auto m = 0.0;
    for (auto v : means) { m += v; }
    m /= nb;
    auto v2 = 0.0;
    for (auto v : means) { v2 += (v - m) * (v - m); }
    return {m, std::sqrt(v2 / (nb - 1) / nb)};
  };
  std::tie(est.p_balanced, est.se_balanced) = mean_se(bal);
  std::tie(est.p_cat_2, est.se_cat_2) = mean_se(cat2);
  std::tie(est.mean_age, est.se_age) = mean_se(age);
  return est;
}

auto check_four_tip(double a_years, double b_years, uint32_t seed) -> void {
  auto set = parse_config("version 1\n" + uniform_stanza("ab", "S000 S001", a_years, b_years));
  auto ref = four_tip_reference(a_years, b_years);
  auto est = four_tip_chain(set, seed, scaled_steps(4'000'000), 20'000);
  std::cerr << absl::StreamFormat(
      "window [%g, %g] y: P(balanced) %.4f +- %.4f (exact %.4f); P(cat S002) %.4f +- %.4f (exact %.4f); "
      "mean age %.4f +- %.4f y (exact %.4f); %d topology changes\n",
      a_years, b_years, est.p_balanced, est.se_balanced, ref.p_balanced, est.p_cat_2, est.se_cat_2,
      ref.p_caterpillar, est.mean_age, est.se_age, ref.mean_age_years, est.topology_changes);
  EXPECT_GT(est.topology_changes, 100);
  // 4.5 standard errors plus a floor for the coalescent grid's discretization
  EXPECT_NEAR(est.p_balanced, ref.p_balanced, 4.5 * est.se_balanced + 0.01);
  EXPECT_NEAR(est.p_cat_2, ref.p_caterpillar, 4.5 * est.se_cat_2 + 0.01);
  EXPECT_NEAR(est.mean_age, ref.mean_age_years, 4.5 * est.se_age + 0.01 * ref.mean_age_years);
}

}  // namespace

TEST(Free_topology_calibration_mcmc_test, four_tips_monophyly_only_matches_the_exact_topology_distribution) {
  // A window so wide that the age prior is flat over every reachable age: only the constraint acts
  check_four_tip(0.0, 1000.0, 11);
}

TEST(Free_topology_calibration_mcmc_test, four_tips_age_prior_reweights_the_topologies_exactly) {
  // The window favours the second coalescence, which only the balanced topology can put at the clade
  check_four_tip(0.5, 1.5, 12);
}

namespace {

struct Invariant_result {
  int topology_changes = 0;
  int root_clade_changes = 0;
  int batches = 0;
  int64_t spr_proposed = 0, spr_accepted = 0, slide_accepted = 0;
};

// Runs a calibrated free-topology chain and checks after every batch, against independent recomputations, that
// every calibrated set is a clade, that the run's targets and calibration term match them, that the parts agree
// with the global value, that every hard bound holds and that the tips are unchanged.
auto run_with_invariants(Run& run, const Calibration_set& set, const Data& data, int64_t steps, int batches)
    -> Invariant_result {
  auto result = Invariant_result{};
  auto tips = std::set<std::string>{};
  for (const auto& td : data.tip_descs) { tips.insert(td.name); }
  auto last_sig = topology_signature(run.tree(), run.tree().root);
  auto last_root_split = tip_names_below(run.tree(), run.tree().at_root().left_child());
  for (auto b = 0; b != batches; ++b) {
    run.do_mcmc_steps(static_cast<int>(std::max<int64_t>(steps / batches, 1)));
    ++result.batches;
    const auto& tree = run.tree();
    assert_phylo_tree_integrity(tree, true);
    assert_phylo_tree_matches_tip_descs(tree, data.ref_sequence, data.tip_descs, true);
    EXPECT_EQ(tip_names_below(tree, tree.root), tips);
    for (auto i = 0; i != set.size(); ++i) {
      const auto& c = set.calibrations[i];
      auto node = ref_node_of(tree, c);
      EXPECT_NE(node, k_no_node) << "batch " << b << ": '" << c.name << "' is not a clade";
      if (node == k_no_node) { return result; }
      EXPECT_EQ(run.calibration_nodes()[i], node) << "batch " << b << ": stale target of '" << c.name << "'";
      EXPECT_TRUE(std::isfinite(c.log_density_at(tree.at(node).t))) << "batch " << b << ": hard bound broken";
    }
    auto ref = ref_log_calibration_prior(set, tree);
    EXPECT_NEAR(run.log_calibration_prior(), ref, 1e-8 * std::max(1.0, std::abs(ref))) << "batch " << b;
    EXPECT_NEAR(run.log_calibration_prior_from_parts(), ref, 1e-7 * std::max(1.0, std::abs(ref))) << "batch " << b;
    EXPECT_TRUE(std::isfinite(run.log_posterior())) << "batch " << b;

    auto sig = topology_signature(tree, tree.root);
    if (sig != last_sig) { ++result.topology_changes; last_sig = sig; }
    auto root_split = tip_names_below(tree, tree.at_root().left_child());
    auto other = tip_names_below(tree, tree.at_root().right_child());
    if (root_split != last_root_split && other != last_root_split) { ++result.root_clade_changes; }
    last_root_split = root_split;
  }
  const auto& counts = run.move_counts();
  result.spr_proposed = counts[static_cast<int>(Move_kind::spr1)].proposed;
  result.spr_accepted = counts[static_cast<int>(Move_kind::spr1)].accepted;
  result.slide_accepted = counts[static_cast<int>(Move_kind::subtree_slide)].accepted;
  return result;
}

}  // namespace

TEST(Free_topology_calibration_mcmc_test, eight_tips_keep_every_invariant_while_the_topology_and_root_move) {
  // Low divergence, so that the data leave the topology uncertain and the root moves
  auto data = simulate(8, 21, 2e-7);
  auto set = parse_config("version 1\n"
                          + normal_stanza("outer", "S000 S001 S004 S005", 2.0, 0.5)
                          + uniform_stanza("inner", "S000 S004", 0.5, 3.0)
                          + normal_stanza("root", "all", 4.0, 1.0));
  auto pool = ctpl::thread_pool{1};
  auto run = make_free_run(pool, constrained_start(data, k_guide_8, set), 22, 2e-7);
  run->set_calibrations(set);
  auto result = run_with_invariants(*run, set, data, scaled_steps(2'000'000), 200);
  std::cerr << absl::StreamFormat("%d topology changes, %d root-split changes in %d batches; SPR %d/%d accepted\n",
                                  result.topology_changes, result.root_clade_changes, result.batches,
                                  result.spr_accepted, result.spr_proposed);
  EXPECT_GT(result.topology_changes, 10);
  EXPECT_GT(result.root_clade_changes, 0);
  EXPECT_GT(result.spr_accepted, 0);
}

TEST(Free_topology_calibration_mcmc_test, the_constraint_binds_where_the_unconstrained_chain_breaks_it) {
  // Control for the test above: without the constraint, the same chain does leave the set {S000, S004}
  auto data = simulate(8, 21, 2e-7);
  auto set = parse_config("version 1\n" + uniform_stanza("inner", "S000 S004", 0.0, 1000.0));
  auto pool = ctpl::thread_pool{1};
  auto start = constrained_start(data, k_guide_8, set);

  auto free_run = make_free_run(pool, start, 23, 2e-7);
  auto broken = 0;
  for (auto b = 0; b != 100; ++b) {
    free_run->do_mcmc_steps(static_cast<int>(scaled_steps(1'000'000) / 100));
    if (not ref_is_clade(free_run->tree(), {"S000", "S004"})) { ++broken; }
  }
  EXPECT_GT(broken, 0) << "the control never left the clade, so the constrained run below tests nothing";

  auto constrained_run = make_free_run(pool, start, 23, 2e-7);
  constrained_run->set_calibrations(set);
  auto result = run_with_invariants(*constrained_run, set, data, scaled_steps(1'000'000), 100);
  EXPECT_GT(result.topology_changes, 0);
}

TEST(Free_topology_calibration_mcmc_test, a_checkpoint_restores_the_free_topology_calibrations) {
  auto data = simulate(8, 31, 2e-7);
  auto set = parse_config("version 1\n" + normal_stanza("inner", "S000 S004", 1.0, 0.5)
                          + normal_stanza("root", "all", 4.0, 1.0));
  auto pool = ctpl::thread_pool{1};
  auto source = make_free_run(pool, constrained_start(data, k_guide_8, set), 32, 2e-7, false);
  source->set_calibrations(set);
  source->do_mcmc_steps(static_cast<int>(scaled_steps(200'000)));

  auto api_tree = phylo_tree_to_api_tree(source->tree());
  auto api_tree_info = phylo_tree_to_api_tree_info(source->tree());
  auto api_params = run_to_api_params(*source);
  auto restored_tree = api_tree_and_tree_info_to_phylo_tree(api_tree.data(), api_tree_info.data());
  auto restored = std::make_shared<delphy::Run>(pool, std::mt19937{33}, 33, restored_tree);
  apply_api_params_to_run(api_params.data(), *restored);
  restored->set_num_parts(1);

  ASSERT_TRUE(restored->has_calibrations());
  EXPECT_TRUE(restored->topology_moves_enabled());
  EXPECT_NEAR(restored->log_calibration_prior(), source->log_calibration_prior(),
              1e-6 * std::max(1.0, std::abs(source->log_calibration_prior())));
  auto result = run_with_invariants(*restored, set, data, scaled_steps(400'000), 40);
  EXPECT_EQ(result.batches, 40);
}

}  // namespace delphy
