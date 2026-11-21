// Tests for internal-node (MRCA) age calibrations.
//
// All numbers here are test inputs: the simulated substitution rate, the coalescent time scale and
// every calibration age are simulation truth for these tests only, never defaults.
//
// Reference log densities were computed independently with mpmath at 40 digits; the exact expressions
// are repeated in the comments next to each constant.

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
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "ctpl_stl.h"

#include "api.h"
#include "calibration.h"
#include "dates.h"
#include "estd.h"
#include "external_tree.h"
#include "human_dates.h"
#include "newick.h"
#include "phylo_tree.h"
#include "pop_model.h"
#include "run.h"

namespace delphy {

namespace {

constexpr auto k_sites = 400;
constexpr auto k_mu_site_day = 2e-6;  // simulation truth
constexpr auto k_coal_days = 500.0;   // pairwise coalescent timescale of the simulation

auto scaled_steps(int64_t steps) -> int64_t {
  return estd::is_debug_enabled ? std::max<int64_t>(steps / 25, 20'000) : steps;
}

auto parse_config(const std::string& text) -> Calibration_set {
  auto is = std::istringstream{text};
  return read_calibration_config(is);
}

auto expect_rejected(const std::string& text, const std::string& needle) -> void {
  try {
    parse_config(text);
    ADD_FAILURE() << "expected a rejection mentioning: " << needle;
  } catch (const Calibration_error& e) {
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr(needle)) << e.what();
  }
}

auto expect_bind_rejected(
    const Calibration_set& set, const Phylo_tree& tree, const std::string& needle) -> void {
  try {
    bind_calibrations(set, tree);
    ADD_FAILURE() << "expected a rejection mentioning: " << needle;
  } catch (const Calibration_error& e) {
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr(needle)) << e.what();
  }
}

// ---------------------------------------------------------------------------------------------
// An independent reference implementation, used to check the production code rather than itself.
// It resolves taxon sets by name from scratch and evaluates the densities from their definitions.
// ---------------------------------------------------------------------------------------------

auto ref_log_normal_density(double a, double m, double s) -> double {
  return -std::log(s) - 0.5 * std::log(2.0 * std::acos(-1.0)) - 0.5 * (a - m) * (a - m) / (s * s);
}

auto ref_log_lognormal_density(double a, double median, double sigma) -> double {
  if (a <= 0.0) { return -std::numeric_limits<double>::infinity(); }
  auto z = (std::log(a) - std::log(median)) / sigma;
  return -std::log(a) - std::log(sigma) - 0.5 * std::log(2.0 * std::acos(-1.0)) - 0.5 * z * z;
}

auto ref_mrca_of_names(const Phylo_tree& tree, const std::vector<std::string>& names) -> Node_index {
  auto wanted = std::set<std::string>{names.begin(), names.end()};
  auto below = std::map<Node_index, std::set<std::string>>{};
  auto best = k_no_node;
  auto best_size = std::numeric_limits<std::size_t>::max();
  for (const auto& node : post_order_traversal(tree)) {
    auto& s = below[node];
    if (tree.at(node).is_tip()) {
      s.insert(tree.at(node).name);
    } else {
      for (auto c : tree.at(node).children) { s.insert(below[c].begin(), below[c].end()); }
    }
    auto contains_all = std::ranges::all_of(wanted, [&](const auto& n) { return s.contains(n); });
    if (contains_all && s.size() < best_size) { best = node; best_size = s.size(); }
  }
  return best;
}

// Recomputes sum_c log f_c(age_c) from the tree, resolving names and evaluating densities without
// calling any of the production calibration code.
auto ref_log_calibration_prior(const Calibration_set& set, const Phylo_tree& tree) -> double {
  auto result = 0.0;
  for (const auto& c : set.calibrations) {
    auto node = c.all_tips ? tree.root : ref_mrca_of_names(tree, c.taxa);
    auto age = (c.reference_t - tree.at(node).t) / 365.0;
    if (age < c.prior.youngest_age || age > c.prior.oldest_age) {
      return -std::numeric_limits<double>::infinity();
    }
    switch (c.prior.distribution) {
      case Calibration_distribution::normal:
        result += ref_log_normal_density(age, c.prior.mean, c.prior.stddev);
        break;
      case Calibration_distribution::lognormal:
        result += ref_log_lognormal_density(age, c.prior.median, c.prior.sigma);
        break;
      case Calibration_distribution::uniform:
        result += -std::log(c.prior.oldest_age - c.prior.youngest_age);
        break;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// Synthetic data on a known rooted topology
// ---------------------------------------------------------------------------------------------

struct Sim_node {
  std::string name;
  double t = 0.0;
  int left = -1;
  int right = -1;
};

struct Sim_data {
  std::vector<Sim_node> nodes;  // tips first, inner nodes in creation order, root last
  Real_sequence ref_sequence;
  std::vector<Tip_desc> tip_descs;
};

auto letter(int i) -> Real_seq_letter { return real_seq_letter_from_index(i); }

// A coalescent genealogy with all tips at `tip_t` and Poisson substitutions.  `scale_days` stretches
// the whole genealogy, so the same shape can be put at pathogen or at deep-human depth.
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

// Tip names below a node, sorted
auto tips_below(const Phylo_tree& tree, Node_index node) -> std::vector<std::string> {
  auto result = std::vector<std::string>{};
  auto stack = std::vector<Node_index>{node};
  while (not stack.empty()) {
    auto n = stack.back();
    stack.pop_back();
    if (tree.at(n).is_tip()) {
      result.push_back(tree.at(n).name);
    } else {
      for (auto c : tree.at(n).children) { stack.push_back(c); }
    }
  }
  std::ranges::sort(result);
  return result;
}

// A clade of the given tree with at least `min_tips` tips, excluding the root
auto some_clade(const Phylo_tree& tree, int min_tips) -> std::pair<Node_index, std::vector<std::string>> {
  for (const auto& node : post_order_traversal(tree)) {
    if (node == tree.root || tree.at(node).is_tip()) { continue; }
    auto names = tips_below(tree, node);
    if (std::ssize(names) >= min_tips) { return {node, names}; }
  }
  return {k_no_node, {}};
}

// `fixed_pop` replaces the default free exponential population model by a constant one and switches
// the population moves off.  That removes the collapse attractor of the default 400-cell coalescent
// grid with a free population size, which otherwise dominates any comparison
// of two chains on these small trees and has nothing to do with calibrations.
auto make_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed, int num_parts,
              int local_moves_per_global_move, bool fixed_pop = false) -> std::shared_ptr<Run> {
  auto run = std::make_shared<Run>(pool, std::mt19937{seed}, seed, tree);
  run->set_topology_moves_enabled(false);
  run->set_mu(k_mu_site_day);
  run->set_mu_prior_alpha(1.0);
  run->set_mu_prior_beta(0.0);
  run->set_num_parts(num_parts);
  run->set_local_moves_per_global_move(local_moves_per_global_move);
  if (fixed_pop) {
    run->set_pop_model(std::make_shared<Const_pop_model>(k_coal_days));
    run->set_final_pop_size_move_enabled(false);
    run->set_pop_growth_rate_move_enabled(false);
    run->set_target_coal_prior_cells(4000);
  }
  return run;
}

}  // namespace

// Configuration file: parsing, units and rejections
// =================================================

TEST(Calibration_config_test, parses_a_minimal_configuration) {
  auto set = parse_config(R"(
version 1

# A lognormal prior on the age of the MRCA of every sample
calibration root
  taxa all
  age_units ka
  age_reference_year 1950
  distribution lognormal
  median 140
  sigma 0.2
end
)");
  ASSERT_EQ(set.size(), 1);
  const auto& c = set.calibrations[0];
  EXPECT_EQ(c.name, "root");
  EXPECT_TRUE(c.all_tips);
  EXPECT_TRUE(c.taxa.empty());
  EXPECT_EQ(c.units, Age_units::ka);
  ASSERT_TRUE(c.reference_year.has_value());
  EXPECT_EQ(*c.reference_year, 1950);
  EXPECT_EQ(c.reference_t, astronomical_year_to_day(1950));
  EXPECT_EQ(c.prior.distribution, Calibration_distribution::lognormal);
  EXPECT_DOUBLE_EQ(c.prior.median, 140'000.0);  // ka -> years
  EXPECT_DOUBLE_EQ(c.prior.sigma, 0.2);
}

TEST(Calibration_config_test, parses_several_calibrations_and_taxon_lists) {
  auto set = parse_config(R"(
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
  taxa S001 S002
  taxa S003
  age_units cal_bp
  distribution normal
  mean 50000
  stddev 5000
  oldest 90000
  youngest 20000
end
calibration N
  taxa S004 S005
  age_units ka
  age_reference_year 2020
  distribution uniform
  oldest 70
  youngest 40
end
)");
  ASSERT_EQ(set.size(), 3);
  EXPECT_DOUBLE_EQ(set.calibrations[0].prior.mean, 140'000.0);
  EXPECT_DOUBLE_EQ(set.calibrations[0].prior.stddev, 10'000.0);  // ka scales too
  EXPECT_THAT(set.calibrations[1].taxa, testing::ElementsAre("S001", "S002", "S003"));
  EXPECT_EQ(set.calibrations[1].reference_t, astronomical_year_to_day(k_cal_bp_reference_year));
  EXPECT_DOUBLE_EQ(set.calibrations[1].prior.oldest_age, 90'000.0);
  EXPECT_DOUBLE_EQ(set.calibrations[1].prior.youngest_age, 20'000.0);
  EXPECT_EQ(set.calibrations[2].prior.distribution, Calibration_distribution::uniform);
  EXPECT_DOUBLE_EQ(set.calibrations[2].prior.oldest_age, 70'000.0);
  EXPECT_DOUBLE_EQ(set.calibrations[2].prior.youngest_age, 40'000.0);
  EXPECT_EQ(set.calibrations[2].reference_t, 0.0);  // 2020-01-01
}

TEST(Calibration_config_test, calendar_conventions_convert_exactly) {
  // 1 CE and 1 BCE are adjacent years, and 1 BCE (astronomical year 0) is a leap year
  auto set = parse_config(R"(
version 1
calibration ce_one
  taxa S000 S001
  age_units ce
  distribution normal
  mean 1
  stddev 10
end
calibration bce_one
  taxa S002 S003
  age_units bce
  distribution normal
  mean 1
  stddev 10
end
)");
  ASSERT_EQ(set.size(), 2);
  const auto& ce = set.calibrations[0];
  const auto& bce = set.calibrations[1];
  EXPECT_EQ(ce.reference_t, 0.0);
  EXPECT_EQ(bce.reference_t, 0.0);
  // age(1 BCE) - age(1 CE) == 366 days == 366/365 years
  EXPECT_NEAR(bce.prior.mean - ce.prior.mean, 366.0 / 365.0, 1e-12);
  // The node time implied by each mean is exactly 1 January of the corresponding year
  EXPECT_NEAR(ce.time_of_age(ce.prior.mean), astronomical_year_to_day(1), 1e-9);
  EXPECT_NEAR(bce.time_of_age(bce.prior.mean), astronomical_year_to_day(0), 1e-9);
  EXPECT_DOUBLE_EQ(ce.prior.stddev, 10.0);
}

TEST(Calibration_config_test, bce_uniform_bounds_keep_their_meaning) {
  // In BCE, larger numbers are older, so `oldest` is the larger number
  auto set = parse_config(R"(
version 1
calibration c
  taxa S000 S001
  age_units bce
  distribution uniform
  oldest 3000
  youngest 1000
end
)");
  const auto& c = set.calibrations[0];
  EXPECT_GT(c.prior.oldest_age, c.prior.youngest_age);
  EXPECT_NEAR(c.time_of_age(c.prior.oldest_age), astronomical_year_to_day(1 - 3000), 1e-6);
  EXPECT_NEAR(c.time_of_age(c.prior.youngest_age), astronomical_year_to_day(1 - 1000), 1e-6);
}

TEST(Calibration_config_test, years_bp_reference_year_shifts_the_whole_scale) {
  auto a = parse_config(R"(
version 1
calibration c
  taxa S000 S001
  age_units years_bp
  age_reference_year 1950
  distribution normal
  mean 10000
  stddev 500
end
)");
  auto b = parse_config(R"(
version 1
calibration c
  taxa S000 S001
  age_units years_bp
  age_reference_year 2000
  distribution normal
  mean 10000
  stddev 500
end
)");
  // The same stated age maps to node times 50 calendar years apart
  auto t_a = a.calibrations[0].time_of_age(10000);
  auto t_b = b.calibrations[0].time_of_age(10000);
  EXPECT_NEAR(t_b - t_a, astronomical_year_to_day(2000) - astronomical_year_to_day(1950), 1e-9);
}

TEST(Calibration_config_test, cal_bp_is_years_bp_with_a_1950_reference) {
  auto set = parse_config(R"(
version 1
calibration c14
  taxa S000 S001
  age_units cal_bp
  distribution normal
  mean 10000
  stddev 500
end
calibration modern_ref
  taxa S002 S003
  age_units years_bp
  age_reference_year 2020
  distribution normal
  mean 10000
  stddev 500
end
)");
  ASSERT_EQ(set.size(), 2);
  const auto& cal_bp = set.calibrations[0];
  const auto& ref_2020 = set.calibrations[1];
  EXPECT_EQ(cal_bp.reference_t, astronomical_year_to_day(k_cal_bp_reference_year));
  EXPECT_EQ(ref_2020.reference_t, 0.0);
  // 1950-01-01 to 2020-01-01 is 70 years with 17 leap days, i.e. 25,567 days, so the same node time
  // is 25567/365 years younger on the cal BP scale than on a 2020 scale
  auto t = -3.0e6;
  EXPECT_NEAR(ref_2020.age_of(t) - cal_bp.age_of(t), 25567.0 / 365.0, 1e-9);
  EXPECT_EQ(astronomical_year_to_day(2020) - astronomical_year_to_day(1950), 25567.0);
  // The same stated age therefore maps to node times 25,567 days apart
  EXPECT_NEAR(ref_2020.time_of_age(10000) - cal_bp.time_of_age(10000), 25567.0, 1e-6);
}

TEST(Calibration_config_test, canonical_form_round_trips) {
  auto original = parse_config(R"(
version 1
calibration root
  taxa all
  age_units ka
  age_reference_year 1950
  distribution lognormal
  median 140
  sigma 0.2
  oldest 400
  youngest 60
end
calibration bounded
  taxa S000 S001 S002
  age_units bce
  distribution uniform
  oldest 3000
  youngest 1000
end
)");
  auto round_tripped = parse_config(calibration_config_to_string(original));
  ASSERT_EQ(round_tripped.size(), original.size());
  for (auto i = 0; i != original.size(); ++i) {
    const auto& a = original.calibrations[i];
    const auto& b = round_tripped.calibrations[i];
    EXPECT_EQ(a.name, b.name);
    EXPECT_EQ(a.all_tips, b.all_tips);
    EXPECT_EQ(a.taxa, b.taxa);
    EXPECT_DOUBLE_EQ(a.reference_t, b.reference_t);
    EXPECT_EQ(a.prior.distribution, b.prior.distribution);
    EXPECT_DOUBLE_EQ(a.prior.mean, b.prior.mean);
    EXPECT_DOUBLE_EQ(a.prior.stddev, b.prior.stddev);
    EXPECT_DOUBLE_EQ(a.prior.median, b.prior.median);
    EXPECT_DOUBLE_EQ(a.prior.sigma, b.prior.sigma);
    EXPECT_DOUBLE_EQ(a.prior.oldest_age, b.prior.oldest_age);
    EXPECT_DOUBLE_EQ(a.prior.youngest_age, b.prior.youngest_age);
    // Identical densities at an arbitrary node time
    EXPECT_DOUBLE_EQ(a.log_density_at(-3.0e6), b.log_density_at(-3.0e6));
  }
}

TEST(Calibration_config_test, rejects_malformed_configurations) {
  auto stanza = [](const std::string& body) {
    return absl::StrCat("version 1\ncalibration c\n", body, "end\n");
  };
  auto ok_body = std::string{
    "  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
    "  distribution normal\n  mean 140\n  stddev 10\n"};

  expect_rejected("", "must start with 'version 1'");
  expect_rejected("calibration c\n  taxa all\nend\n", "must start with 'version 1'");
  expect_rejected("version 2\n", "must start with 'version 1'");
  expect_rejected("version 1\n", "contains no calibration");
  expect_rejected("version 1\n  taxa all\n", "outside a 'calibration ... end' block");
  expect_rejected("version 1\ncalibration c\n" + ok_body, "not closed with 'end'");
  expect_rejected("version 1\nend\n", "without a matching 'calibration'");
  expect_rejected("version 1\ncalibration a b\n  taxa all\nend\n", "takes exactly one name");

  expect_rejected(stanza("  age_units ka\n  age_reference_year 1950\n  distribution normal\n"
                         "  mean 140\n  stddev 10\n"), "missing required key 'taxa'");
  expect_rejected(stanza("  taxa S000\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution normal\n  mean 140\n  stddev 10\n"),
                  "at least 2 sample IDs");
  expect_rejected(stanza("  taxa S000 S000\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution normal\n  mean 140\n  stddev 10\n"),
                  "listed more than once");
  expect_rejected(stanza("  taxa all\n  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution normal\n  mean 140\n  stddev 10\n"),
                  "cannot be combined with an explicit list");

  expect_rejected(stanza("  taxa S000 S001\n  distribution normal\n  mean 140\n  stddev 10\n"),
                  "missing required key 'age_units'");
  expect_rejected(stanza("  taxa S000 S001\n  age_units kyr\n  distribution normal\n"
                         "  mean 140\n  stddev 10\n"), "Unknown age_units");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  distribution normal\n"
                         "  mean 140\n  stddev 10\n"), "missing required key 'age_reference_year'");
  expect_rejected(stanza("  taxa S000 S001\n  age_units cal_bp\n  age_reference_year 1950\n"
                         "  distribution normal\n  mean 140\n  stddev 10\n"),
                  "'age_reference_year' is not allowed here");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  age_reference_year 0\n"
                         "  distribution normal\n  mean 140\n  stddev 10\n"), "there is no year 0");

  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution gamma\n  mean 140\n  stddev 10\n"), "Unknown distribution");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution normal\n  mean 140\n"), "missing required key 'stddev'");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution normal\n  mean 140\n  stddev 0\n"), "stddev must be > 0");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution lognormal\n  median 140\n  sigma 0\n"), "sigma must be > 0");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution lognormal\n  median -1\n  sigma 0.2\n"), "median must be > 0");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ce\n  distribution lognormal\n"
                         "  median 1500\n  sigma 0.2\n"), "needs an age convention");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution uniform\n  oldest 40\n  youngest 70\n"),
                  "strictly older than 'youngest'");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution normal\n  mean 140\n  stddev 10\n  median 7\n"),
                  "belongs to distribution lognormal");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution normal\n  mean 140\n  stddev 10\n  colour blue\n"),
                  "unknown key");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution normal\n  mean 140\n  mean 150\n  stddev 10\n"),
                  "is given twice");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ka\n  age_reference_year 1950\n"
                         "  distribution normal\n  mean abc\n  stddev 10\n"), "is not a number");
  expect_rejected(stanza("  taxa S000 S001\n  age_units ce\n  distribution normal\n"
                         "  mean 1500.5\n  stddev 10\n"), "is not a whole year");
  expect_rejected(stanza("  taxa S000 S001\n  age_units bce\n  distribution normal\n"
                         "  mean 0\n  stddev 10\n"), "there is no year 0");

  expect_rejected("version 1\ncalibration c\n" + ok_body + "end\ncalibration c\n" + ok_body + "end\n",
                  "duplicate calibration name");
}

TEST(Calibration_config_test, reports_every_problem_at_once) {
  try {
    parse_config(
        "version 1\n"
        "calibration a\n  taxa S000\n  age_units ka\n  age_reference_year 1950\n"
        "  distribution normal\n  mean 140\n  stddev 10\nend\n"
        "calibration b\n  taxa S001 S002\n  age_units ka\n  age_reference_year 1950\n"
        "  distribution normal\n  mean 140\n  stddev -1\nend\n");
    ADD_FAILURE() << "expected a rejection";
  } catch (const Calibration_error& e) {
    auto what = std::string{e.what()};
    EXPECT_THAT(what, testing::HasSubstr("2 problems found"));
    EXPECT_THAT(what, testing::HasSubstr("at least 2 sample IDs"));
    EXPECT_THAT(what, testing::HasSubstr("stddev must be > 0"));
  }
}

// Log densities
// =============

TEST(Calibration_density_test, matches_independent_reference_values) {
  // mpmath, 40 digits: -log(s) - log(sqrt(2 pi)) - (a-m)^2/(2 s^2)
  auto normal = Age_prior{};
  normal.distribution = Calibration_distribution::normal;
  normal.mean = 140'000.0;
  normal.stddev = 10'000.0;
  EXPECT_NEAR(normal.log_density(140'000.0), -10.129278905180855478, 1e-12);
  EXPECT_NEAR(normal.log_density(155'000.0), -11.254278905180855478, 1e-12);
  EXPECT_NEAR(normal.log_density(120'500.0), -12.030528905180855478, 1e-12);

  // mpmath: -log(a) - log(S) - log(sqrt(2 pi)) - (log a - log M)^2/(2 S^2)
  auto lognormal = Age_prior{};
  lognormal.distribution = Calibration_distribution::lognormal;
  lognormal.median = 140'000.0;
  lognormal.sigma = 0.2;
  EXPECT_NEAR(lognormal.log_density(140'000.0), -11.158898322362013773, 1e-12);
  EXPECT_NEAR(lognormal.log_density(100'000.0), -12.237595660951819484, 1e-12);

  auto small = Age_prior{};
  small.distribution = Calibration_distribution::lognormal;
  small.median = 50.0;
  small.sigma = 0.5;
  EXPECT_NEAR(small.log_density(73.0), -4.8026790655544049197, 1e-12);

  auto uniform = Age_prior{};
  uniform.distribution = Calibration_distribution::uniform;
  uniform.oldest_age = 70'000.0;
  uniform.youngest_age = 40'000.0;
  EXPECT_NEAR(uniform.log_density(55'000.0), -10.308952660644292427, 1e-12);  // -log(30000)
  EXPECT_NEAR(uniform.log_density(40'000.0), -10.308952660644292427, 1e-12);  // closed interval
  EXPECT_NEAR(uniform.log_density(70'000.0), -10.308952660644292427, 1e-12);

  auto calendar = Age_prior{};
  calendar.distribution = Calibration_distribution::normal;
  calendar.mean = 1500.0;
  calendar.stddev = 250.0;
  EXPECT_NEAR(calendar.log_density(1234.5), -7.004321451066919175, 1e-12);
}

TEST(Calibration_density_test, hard_bounds_and_edge_cases) {
  auto inf = std::numeric_limits<double>::infinity();

  auto uniform = Age_prior{};
  uniform.distribution = Calibration_distribution::uniform;
  uniform.oldest_age = 70'000.0;
  uniform.youngest_age = 40'000.0;
  EXPECT_EQ(uniform.log_density(39'999.0), -inf);
  EXPECT_EQ(uniform.log_density(70'001.0), -inf);
  EXPECT_EQ(uniform.log_density(inf), -inf);
  EXPECT_EQ(uniform.log_density(std::nan("")), -inf);

  auto bounded_normal = Age_prior{};
  bounded_normal.distribution = Calibration_distribution::normal;
  bounded_normal.mean = 140'000.0;
  bounded_normal.stddev = 10'000.0;
  bounded_normal.oldest_age = 160'000.0;
  bounded_normal.youngest_age = 120'000.0;
  EXPECT_TRUE(std::isfinite(bounded_normal.log_density(120'000.0)));
  EXPECT_EQ(bounded_normal.log_density(119'999.0), -inf);
  EXPECT_EQ(bounded_normal.log_density(160'001.0), -inf);
  // Truncation is not renormalized, so values inside are unchanged
  auto plain = bounded_normal;
  plain.oldest_age = inf;
  plain.youngest_age = -inf;
  EXPECT_DOUBLE_EQ(bounded_normal.log_density(140'000.0), plain.log_density(140'000.0));

  // A lognormal has no support at age <= 0 even without hard bounds
  auto lognormal = Age_prior{};
  lognormal.distribution = Calibration_distribution::lognormal;
  lognormal.median = 140'000.0;
  lognormal.sigma = 0.2;
  EXPECT_EQ(lognormal.log_density(0.0), -inf);
  EXPECT_EQ(lognormal.log_density(-1.0), -inf);
  EXPECT_DOUBLE_EQ(lognormal.support_youngest_age(), 0.0);

  // A normal keeps unbounded support unless hard bounds are given
  EXPECT_EQ(plain.support_youngest_age(), -inf);
  EXPECT_EQ(plain.support_oldest_age(), inf);
}

TEST(Calibration_density_test, age_and_time_are_exact_inverses_at_deep_time) {
  auto c = Calibration{};
  c.reference_t = astronomical_year_to_day(1950);
  // 140 ka: about -5.1e7 days, where float32 spacing is 4 days.  The calibration arithmetic is double
  // throughout, so the round trip is exact to well below a day.
  auto age = 140'000.0;
  auto t = c.time_of_age(age);
  EXPECT_LT(t, -5.0e7);
  EXPECT_NEAR(c.age_of(t), age, 1e-9);
  // One year of age is exactly 365 days of node time, at any depth
  EXPECT_NEAR(c.time_of_age(age) - c.time_of_age(age + 1.0), 365.0, 1e-6);
}

// Binding to tree nodes
// =====================

TEST(Calibration_binding_test, binds_monophyletic_sets_and_the_root) {
  auto sim = simulate(24, 11, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto [clade_node, clade_names] = some_clade(tree, 3);
  ASSERT_NE(clade_node, k_no_node);

  auto text = absl::StrCat(
      "version 1\n"
      "calibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean 1000\n  stddev 400\nend\n"
      "calibration clade\n  taxa ", absl::StrJoin(clade_names, " "),
      "\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean 300\n  stddev 200\nend\n");
  auto set = parse_config(text);
  auto nodes = bind_calibrations(set, tree);
  ASSERT_EQ(std::ssize(nodes), 2);
  EXPECT_EQ(nodes[0], tree.root);
  EXPECT_EQ(nodes[1], clade_node);
  EXPECT_EQ(nodes[1], ref_mrca_of_names(tree, clade_names));

  // The order in which the taxa are listed does not matter
  auto shuffled = clade_names;
  std::ranges::reverse(shuffled);
  auto set2 = parse_config(absl::StrCat(
      "version 1\ncalibration clade\n  taxa ", absl::StrJoin(shuffled, " "),
      "\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean 300\n  stddev 200\nend\n"));
  EXPECT_EQ(bind_calibrations(set2, tree)[0], clade_node);
}

TEST(Calibration_binding_test, rejects_unknown_taxa_and_paraphyletic_sets) {
  auto sim = simulate(24, 12, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);

  auto set_unknown = parse_config(
      "version 1\ncalibration c\n  taxa S000 GHOST\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean 300\n  stddev 200\nend\n");
  expect_bind_rejected(set_unknown, tree, "not tips of the tree");

  // Take a clade and drop one of its tips: the MRCA then has an extra tip below it
  auto [clade_node, clade_names] = some_clade(tree, 3);
  ASSERT_NE(clade_node, k_no_node);
  auto dropped = clade_names;
  auto removed = dropped.back();
  dropped.pop_back();
  auto set_para = parse_config(absl::StrCat(
      "version 1\ncalibration c\n  taxa ", absl::StrJoin(dropped, " "),
      "\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean 300\n  stddev 200\nend\n"));
  try {
    bind_calibrations(set_para, tree);
    ADD_FAILURE() << "expected a monophyly rejection";
  } catch (const Calibration_error& e) {
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr("not monophyletic"));
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr(removed));
  }
}

TEST(Calibration_binding_test, rejects_two_calibrations_on_the_same_node) {
  auto sim = simulate(24, 13, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto all = tips_below(tree, tree.root);
  auto set = parse_config(absl::StrCat(
      "version 1\n"
      "calibration a\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean 1000\n  stddev 400\nend\n"
      "calibration b\n  taxa ", absl::StrJoin(all, " "),
      "\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean 1000\n  stddev 400\nend\n"));
  expect_bind_rejected(set, tree, "both resolve to the same node");
}

TEST(Calibration_binding_test, rejects_incompatible_nested_ranges) {
  auto sim = simulate(24, 14, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto [clade_node, clade_names] = some_clade(tree, 3);
  ASSERT_NE(clade_node, k_no_node);

  // The root must be older than the nested clade, but its uniform window is entirely younger
  auto set = parse_config(absl::StrCat(
      "version 1\n"
      "calibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution uniform\n  oldest 400\n  youngest 100\nend\n"
      "calibration inner\n  taxa ", absl::StrJoin(clade_names, " "),
      "\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution uniform\n  oldest 2000\n  youngest 600\nend\n"));
  expect_bind_rejected(set, tree, "cannot both be satisfied");

  // Nested windows that do overlap in the right direction are accepted
  auto ok = parse_config(absl::StrCat(
      "version 1\n"
      "calibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution uniform\n  oldest 20000\n  youngest 100\nend\n"
      "calibration inner\n  taxa ", absl::StrJoin(clade_names, " "),
      "\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution uniform\n  oldest 19000\n  youngest 50\nend\n"));
  EXPECT_NO_THROW(bind_calibrations_ignoring_times(ok, tree));
}

TEST(Calibration_binding_test, rejects_a_window_younger_than_the_samples_below_it) {
  auto sim = simulate(24, 15, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  // Tips are at t = 0, i.e. age 0 on a 2020 reference, so a node cannot be younger than age 0
  auto set = parse_config(
      "version 1\ncalibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution uniform\n  oldest -1\n  youngest -5\nend\n");
  expect_bind_rejected(set, tree, "must be older than every sample below it");
}

TEST(Calibration_binding_test, rejects_a_starting_tree_outside_the_hard_bounds) {
  auto sim = simulate(24, 16, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto root_age_years = -tree.at_root().t / 365.0;
  // A window far older than the starting root age
  auto set = parse_config(absl::StrFormat(
      "version 1\ncalibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution uniform\n  oldest %g\n  youngest %g\nend\n",
      root_age_years * 1000.0, root_age_years * 100.0));
  expect_bind_rejected(set, tree, "outside the hard bounds");
  // Without the time check the same configuration binds
  EXPECT_NO_THROW(bind_calibrations_ignoring_times(set, tree));
}

// Posterior bookkeeping
// =====================

TEST(Calibration_posterior_test, log_posterior_includes_the_calibration_term) {
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(20, 21, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);

  auto uncalibrated = make_run(pool, tree, 31, 1, -1);
  EXPECT_FALSE(uncalibrated->has_calibrations());
  EXPECT_EQ(uncalibrated->log_calibration_prior(), 0.0);
  auto base_posterior = uncalibrated->log_posterior();

  auto root_age_years = -tree.at_root().t / 365.0;
  auto set = parse_config(absl::StrFormat(
      "version 1\ncalibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n",
      root_age_years, root_age_years / 4));
  auto run = make_run(pool, tree, 31, 1, -1);
  run->set_calibrations(set);
  ASSERT_TRUE(run->has_calibrations());
  ASSERT_EQ(run->calibration_nodes().size(), 1u);
  EXPECT_EQ(run->calibration_nodes()[0], tree.root);

  auto expected = ref_log_calibration_prior(set, run->tree());
  EXPECT_NEAR(run->log_calibration_prior(), expected, 1e-10);
  EXPECT_NEAR(run->log_posterior() - base_posterior, expected, 1e-8);
  EXPECT_NEAR(run->log_calibration_prior_from_parts(), expected, 1e-10);
}

TEST(Calibration_posterior_test, calibrations_with_a_free_topology_need_a_single_part) {
  auto pool = ctpl::thread_pool{2};
  auto sim = simulate(20, 22, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto root_age_years = -tree.at_root().t / 365.0;
  auto set = parse_config(absl::StrFormat(
      "version 1\ncalibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n", root_age_years, root_age_years / 4));

  // Calibrations and topology moves can be combined in either order
  auto free_run = make_run(pool, tree, 32, 1, -1);
  free_run->set_topology_moves_enabled(true);
  EXPECT_NO_THROW(free_run->set_calibrations(set));
  EXPECT_NO_THROW(free_run->do_mcmc_steps(1000));

  auto fixed_run = make_run(pool, tree, 32, 1, -1);
  fixed_run->set_calibrations(set);
  EXPECT_NO_THROW(fixed_run->set_topology_moves_enabled(true));

  // ... but only on a single tree partition part
  auto multi_run = make_run(pool, tree, 32, 2, -1);
  multi_run->set_topology_moves_enabled(true);
  multi_run->set_calibrations(set);
  EXPECT_THROW(multi_run->do_mcmc_steps(1000), std::invalid_argument);
  multi_run->set_topology_moves_enabled(false);
  EXPECT_NO_THROW(multi_run->do_mcmc_steps(1000));
}

namespace {

// Drives an MCMC run with calibrations and checks, after every batch, that
//   (a) the globally recomputed calibration prior equals a fully independent recomputation, and
//   (b) the value maintained incrementally by the partition parts equals the global one.
// (b) is the check that the deltas inside the moves are right on accept, reject and rollback paths:
// the parts only ever update their value by deltas.
struct Mcmc_check_result {
  double min_root_age = 0.0;
  double max_root_age = 0.0;
  double last_mu = 0.0;
  // Means over the second half of the batches, as a crude burn-in
  double mean_mu = 0.0;
  double mean_root_age = 0.0;
  int num_batches = 0;
  int num_root_age_changes = 0;
  // One sample of each quantity per batch
  std::vector<double> root_age_samples;
  std::vector<double> mu_samples;
  // Per batch: did each calibrated node's time change since the previous batch?
  std::vector<int> num_changes_per_calibration;
};

auto run_and_check(Run& run, const Calibration_set& set, int64_t steps, int batches) -> Mcmc_check_result {
  auto result = Mcmc_check_result{};
  result.min_root_age = +std::numeric_limits<double>::infinity();
  result.max_root_age = -std::numeric_limits<double>::infinity();
  auto per_batch = std::max<int64_t>(steps / batches, 1);
  auto last_root_t = run.tree().at_root().t;
  result.num_changes_per_calibration.assign(set.calibrations.size(), 0);
  auto last_cal_t = std::vector<double>{};
  for (auto i = 0; i != std::ssize(run.calibration_nodes()); ++i) {
    last_cal_t.push_back(run.tree().at(run.calibration_nodes()[i]).t);
  }
  for (auto b = 0; b != batches; ++b) {
    run.do_mcmc_steps(static_cast<int>(per_batch));
    ++result.num_batches;

    auto global = run.log_calibration_prior();
    auto independent = ref_log_calibration_prior(set, run.tree());
    EXPECT_TRUE(std::isfinite(global)) << "calibration prior left its support";
    EXPECT_NEAR(global, independent, 1e-8 * std::max(1.0, std::abs(independent)))
        << "batch " << b << ": incremental/global value disagrees with an independent recomputation";
    EXPECT_NEAR(run.log_calibration_prior_from_parts(), global, 1e-7 * std::max(1.0, std::abs(global)))
        << "batch " << b << ": partition-local bookkeeping disagrees with the global value";

    auto root_t = run.tree().at_root().t;
    if (root_t != last_root_t) { ++result.num_root_age_changes; }
    last_root_t = root_t;
    auto age = -root_t / 365.0;
    result.min_root_age = std::min(result.min_root_age, age);
    result.max_root_age = std::max(result.max_root_age, age);
    result.last_mu = run.mu();
    result.root_age_samples.push_back(age);
    result.mu_samples.push_back(run.mu());
    for (auto i = 0; i != std::ssize(run.calibration_nodes()); ++i) {
      auto t_i = run.tree().at(run.calibration_nodes()[i]).t;
      if (t_i != last_cal_t[i]) { ++result.num_changes_per_calibration[i]; }
      last_cal_t[i] = t_i;
    }
    if (b >= batches / 2) {
      result.mean_mu += run.mu();
      result.mean_root_age += age;
    }
  }
  auto counted = batches - batches / 2;
  result.mean_mu /= counted;
  result.mean_root_age /= counted;
  return result;
}

}  // namespace

TEST(Calibration_mcmc_test, incremental_and_independent_values_agree_single_part) {
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(20, 23, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto root_age_years = -tree.at_root().t / 365.0;
  auto [clade_node, clade_names] = some_clade(tree, 3);
  ASSERT_NE(clade_node, k_no_node);
  auto clade_age_years = -tree.at(clade_node).t / 365.0;

  auto set = parse_config(absl::StrFormat(
      "version 1\n"
      "calibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution lognormal\n  median %.12g\n  sigma 0.3\nend\n"
      "calibration inner\n  taxa %s\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n",
      root_age_years, absl::StrJoin(clade_names, " "), clade_age_years,
      std::max(clade_age_years / 3, 1.0)));

  auto run = make_run(pool, tree, 41, 1, -1);
  run->set_calibrations(set);
  auto result = run_and_check(*run, set, scaled_steps(400'000), 8);
  EXPECT_EQ(result.num_batches, 8);
  EXPECT_GT(result.num_root_age_changes, 0) << "the root age never moved, so nothing was tested";
}

TEST(Calibration_mcmc_test, incremental_and_independent_values_agree_multi_part) {
  auto pool = ctpl::thread_pool{4};
  auto sim = simulate(60, 24, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto root_age_years = -tree.at_root().t / 365.0;
  auto [clade_node, clade_names] = some_clade(tree, 4);
  ASSERT_NE(clade_node, k_no_node);
  auto clade_age_years = -tree.at(clade_node).t / 365.0;

  auto set = parse_config(absl::StrFormat(
      "version 1\n"
      "calibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n"
      "calibration inner\n  taxa %s\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n",
      root_age_years, root_age_years / 3, absl::StrJoin(clade_names, " "), clade_age_years,
      std::max(clade_age_years / 3, 1.0)));

  // Forced repartitioning every 200 local moves, so the calibrated nodes land in different parts and
  // alternate between movable and frozen
  auto run = make_run(pool, tree, 42, 4, 200);
  run->set_calibrations(set);
  auto result = run_and_check(*run, set, scaled_steps(600'000), 10);
  EXPECT_EQ(result.num_batches, 10);
  EXPECT_GT(result.num_root_age_changes, 0);
}

TEST(Calibration_mcmc_test, bounded_uniform_calibration_keeps_the_node_inside_its_window) {
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(20, 25, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto root_age_years = -tree.at_root().t / 365.0;
  auto oldest = root_age_years * 1.2;
  auto youngest = root_age_years * 0.8;
  auto set = parse_config(absl::StrFormat(
      "version 1\ncalibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution uniform\n  oldest %.12g\n  youngest %.12g\nend\n", oldest, youngest));

  auto run = make_run(pool, tree, 43, 1, -1);
  run->set_calibrations(set);
  auto result = run_and_check(*run, set, scaled_steps(300'000), 10);
  EXPECT_GE(result.min_root_age, youngest - 1e-6);
  EXPECT_LE(result.max_root_age, oldest + 1e-6);
  EXPECT_GT(result.num_root_age_changes, 0);
}

TEST(Calibration_mcmc_test, same_date_samples_identify_time_and_rate_jointly) {
  // The acceptance scenario: every sample shares one sampling date, the rooted topology is given and
  // fixed, there is no external rate information at all (improper uniform prior on mu, pathogen
  // initial value), and only a root age calibration.  Rate and node times must then be inferred
  // jointly from the posterior, and a different calibration must move both.
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(30, 26, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto true_root_age_days = -tree.at_root().t;

  auto make_calibrated_run = [&](double factor, uint32_t seed) {
    auto target_age_years = factor * true_root_age_days / 365.0;
    auto set = parse_config(absl::StrFormat(
        "version 1\ncalibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
        "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n",
        target_age_years, target_age_years * 0.05));
    auto run = make_run(pool, tree, seed, 1, -1);
    // No rate information: improper uniform prior on mu and the pathogen default initial rate
    run->set_mu(1e-3 / 365.0);
    run->set_mu_prior_alpha(1.0);
    run->set_mu_prior_beta(0.0);
    run->set_calibrations(set);
    auto result = run_and_check(*run, set, scaled_steps(2'000'000), 10);
    return std::pair{result, target_age_years};
  };

  auto [tight, target_1x] = make_calibrated_run(1.0, 51);
  auto [stretched, target_3x] = make_calibrated_run(3.0, 52);

  // Each chain's root age sits near its own calibration, not near the other's
  EXPECT_NEAR(tight.mean_root_age, target_1x, 0.5 * target_1x);
  EXPECT_NEAR(stretched.mean_root_age, target_3x, 0.5 * target_3x);
  EXPECT_GT(stretched.min_root_age, tight.max_root_age)
      << "a three times older calibration did not move the node ages";

  // The rate compensates: a three times older tree needs a roughly three times slower clock.
  // mu and the node times are strongly correlated and there is no joint rescaling operator, so the
  // bounds here are loose on purpose; the direction and the order of magnitude are what is being
  // checked.
  EXPECT_GT(tight.mean_mu, 0.0);
  EXPECT_GT(stretched.mean_mu, 0.0);
  EXPECT_LT(stretched.mean_mu, tight.mean_mu)
      << "the clock rate did not respond to the calibration, so rate and time were not sampled jointly";
  auto ratio = tight.mean_mu / stretched.mean_mu;
  EXPECT_GT(ratio, 1.3);
  EXPECT_LT(ratio, 8.0);
  // Neither chain used the pathogen initial rate
  EXPECT_NE(tight.mean_mu, 1e-3 / 365.0);
}

TEST(Calibration_mcmc_test, works_at_deep_human_time_scales) {
  // The same machinery with a genealogy about 140,000 years deep, where node times are about
  // -5e7 days and the .dphy float32 spacing is 4 days.  The calibration arithmetic is
  // double throughout.
  auto pool = ctpl::thread_pool{1};
  auto deep_mu = 1.7e-8 / 365.0;  // substitutions / site / day, a test input
  auto scale = 140'000.0 * 365.0 / k_coal_days;
  auto sim = simulate(20, 27, scale, 0.0, deep_mu);
  auto tree = sim_tree(sim);
  auto root_age_years = -tree.at_root().t / 365.0;
  ASSERT_GT(root_age_years, 10'000.0);
  ASSERT_LT(tree.at_root().t, -1.0e7);

  auto set = parse_config(absl::StrFormat(
      "version 1\ncalibration root\n  taxa all\n  age_units ka\n  age_reference_year 1950\n"
      "  distribution lognormal\n  median %.12g\n  sigma 0.15\nend\n", root_age_years / 1000.0));
  auto run = make_run(pool, tree, 44, 1, -1);
  run->set_mu(deep_mu);
  run->set_calibrations(set);
  auto result = run_and_check(*run, set, scaled_steps(200'000), 4);
  EXPECT_GT(result.num_root_age_changes, 0);
  EXPECT_GT(result.min_root_age, 1000.0);
}

// Serialization
// =============

TEST(Calibration_api_test, params_round_trip_restores_the_calibrations_exactly) {
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(24, 28, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto root_age_years = -tree.at_root().t / 365.0;
  auto [clade_node, clade_names] = some_clade(tree, 3);
  ASSERT_NE(clade_node, k_no_node);

  auto set = parse_config(absl::StrFormat(
      "version 1\n"
      "calibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution lognormal\n  median %.12g\n  sigma 0.25\nend\n"
      "calibration inner\n  taxa %s\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\n  oldest %.12g\n  youngest 0\nend\n",
      root_age_years, absl::StrJoin(clade_names, " "),
      -tree.at(clade_node).t / 365.0, std::max(-tree.at(clade_node).t / 365.0 / 3, 1.0),
      root_age_years * 10));

  auto source = make_run(pool, tree, 45, 1, -1);
  source->set_calibrations(set);
  auto params = run_to_api_params(*source);

  auto target = make_run(pool, tree, 46, 1, -1);
  apply_api_params_to_run(params.data(), *target);

  ASSERT_TRUE(target->has_calibrations());
  ASSERT_EQ(target->calibrations().size(), source->calibrations().size());
  EXPECT_EQ(target->calibration_nodes(), source->calibration_nodes());
  for (auto i = 0; i != source->calibrations().size(); ++i) {
    const auto& a = source->calibrations().calibrations[i];
    const auto& b = target->calibrations().calibrations[i];
    EXPECT_EQ(a.name, b.name);
    EXPECT_EQ(a.all_tips, b.all_tips);
    EXPECT_EQ(a.taxa, b.taxa);
    EXPECT_DOUBLE_EQ(a.reference_t, b.reference_t);
    EXPECT_EQ(a.prior.distribution, b.prior.distribution);
    EXPECT_DOUBLE_EQ(a.log_density_at(tree.at(source->calibration_nodes()[i]).t),
                     b.log_density_at(tree.at(target->calibration_nodes()[i]).t));
  }
  EXPECT_DOUBLE_EQ(target->log_calibration_prior(), source->log_calibration_prior());
  EXPECT_NEAR(target->log_calibration_prior(), ref_log_calibration_prior(set, target->tree()), 1e-10);
  EXPECT_FALSE(target->topology_moves_enabled());

  // The restored run keeps sampling the calibrated posterior
  auto result = run_and_check(*target, set, scaled_steps(100'000), 2);
  EXPECT_EQ(result.num_batches, 2);
}

TEST(Calibration_api_test, params_without_the_field_mean_no_calibrations) {
  // This is what every .dphy written before the calibration_config field looks like: the field is
  // absent, which must mean "no calibrations" rather than "keep whatever the target run had".
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(20, 29, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto root_age_years = -tree.at_root().t / 365.0;
  auto set = parse_config(absl::StrFormat(
      "version 1\ncalibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n", root_age_years, root_age_years / 4));

  auto uncalibrated = make_run(pool, tree, 47, 1, -1);
  auto params = run_to_api_params(*uncalibrated);

  auto calibrated = make_run(pool, tree, 48, 1, -1);
  calibrated->set_calibrations(set);
  ASSERT_TRUE(calibrated->has_calibrations());

  apply_api_params_to_run(params.data(), *calibrated);
  EXPECT_FALSE(calibrated->has_calibrations());
  EXPECT_EQ(calibrated->log_calibration_prior(), 0.0);
  EXPECT_TRUE(calibrated->calibration_nodes().empty());
}

TEST(Calibration_api_test, restoring_into_an_incompatible_tree_is_an_error) {
  auto pool = ctpl::thread_pool{1};
  auto sim_a = simulate(20, 30, 1.0, 0.0, k_mu_site_day);
  auto tree_a = sim_tree(sim_a);
  auto [clade_node, clade_names] = some_clade(tree_a, 3);
  ASSERT_NE(clade_node, k_no_node);
  auto set = parse_config(absl::StrCat(
      "version 1\ncalibration inner\n  taxa ", absl::StrJoin(clade_names, " "),
      "\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean 1000\n  stddev 100000\nend\n"));
  auto source = make_run(pool, tree_a, 49, 1, -1);
  source->set_calibrations(set);
  auto params = run_to_api_params(*source);

  // A tree over the same samples but with a different topology, in which the set is not monophyletic
  auto sim_b = simulate(20, 31, 1.0, 0.0, k_mu_site_day);
  auto tree_b = sim_tree(sim_b);
  auto target = make_run(pool, tree_b, 50, 1, -1);
  auto threw = false;
  try {
    apply_api_params_to_run(params.data(), *target);
  } catch (const Calibration_error&) {
    threw = true;
  }
  if (not threw) {
    // The random topology happened to contain the same clade; then restoring must be exact instead
    EXPECT_TRUE(target->has_calibrations());
    EXPECT_EQ(tips_below(target->tree(), target->calibration_nodes()[0]), clade_names);
  }
  EXPECT_FALSE(target->topology_moves_enabled());
}


// Normalization, parameterization and ordering
// ============================================

namespace {

// Trapezoidal integral of exp(log_density) over [lo, hi]
auto integrate_density(const Age_prior& prior, double lo, double hi, int steps) -> double {
  auto h = (hi - lo) / steps;
  auto total = 0.0;
  for (auto i = 0; i <= steps; ++i) {
    auto a = lo + i * h;
    auto d = std::exp(prior.log_density(a));
    total += ((i == 0 || i == steps) ? 0.5 : 1.0) * d;
  }
  return total * h;
}

}  // namespace

TEST(Calibration_density_test, densities_are_normalized_on_their_support) {
  auto normal = Age_prior{};
  normal.distribution = Calibration_distribution::normal;
  normal.mean = 140'000.0;
  normal.stddev = 10'000.0;
  // +/- 10 sd, 200k steps
  EXPECT_NEAR(integrate_density(normal, 40'000.0, 240'000.0, 200'000), 1.0, 1e-6);

  auto lognormal = Age_prior{};
  lognormal.distribution = Calibration_distribution::lognormal;
  lognormal.median = 140'000.0;
  lognormal.sigma = 0.2;
  EXPECT_NEAR(integrate_density(lognormal, 1.0, 1'500'000.0, 1'500'000), 1.0, 1e-5);

  auto uniform = Age_prior{};
  uniform.distribution = Calibration_distribution::uniform;
  uniform.oldest_age = 70'000.0;
  uniform.youngest_age = 40'000.0;
  EXPECT_NEAR(integrate_density(uniform, 40'000.0, 70'000.0, 300'000), 1.0, 1e-9);
}

TEST(Calibration_density_test, hard_bounds_truncate_without_renormalizing) {
  // Documented convention: a hard bound removes support but the remaining density keeps its values,
  // because the truncation normalizer does not depend on the state and cancels in every MH ratio.
  // The integral over a truncated normal is therefore the probability mass of that window, not 1.
  auto truncated = Age_prior{};
  truncated.distribution = Calibration_distribution::normal;
  truncated.mean = 140'000.0;
  truncated.stddev = 10'000.0;
  truncated.oldest_age = 150'000.0;   // +1 sd
  truncated.youngest_age = 130'000.0;  // -1 sd
  // P(|Z| <= 1) = 0.682689492137086
  EXPECT_NEAR(integrate_density(truncated, 130'000.0, 150'000.0, 200'000), 0.682689492137086, 1e-7);
}

TEST(Calibration_density_test, lognormal_parameterization_is_median_and_log_sd) {
  // `median` is the median of the age, and `sigma` is the standard deviation of log(age).
  auto prior = Age_prior{};
  prior.distribution = Calibration_distribution::lognormal;
  prior.median = 140'000.0;
  prior.sigma = 0.3;

  // Half the mass lies below the median
  EXPECT_NEAR(integrate_density(prior, 1.0, 140'000.0, 1'400'000), 0.5, 1e-5);

  // The mean of the age is median * exp(sigma^2 / 2), not the median
  auto h = 2'000'000.0 / 2'000'000;
  auto mean = 0.0;
  for (auto i = 1; i <= 2'000'000; ++i) {
    auto a = i * h;
    mean += a * std::exp(prior.log_density(a)) * h;
  }
  EXPECT_NEAR(mean, 140'000.0 * std::exp(0.3 * 0.3 / 2), 1e-5 * mean);

  // log(age) is normal with mean log(median) and sd sigma: the +/- 1 sigma quantiles are
  // median * exp(-sigma) and median * exp(sigma)
  EXPECT_NEAR(integrate_density(prior, 1.0, 140'000.0 * std::exp(-0.3), 1'400'000),
              0.158655253931457, 1e-5);
  EXPECT_NEAR(integrate_density(prior, 1.0, 140'000.0 * std::exp(0.3), 1'400'000),
              0.841344746068543, 1e-5);

  // The density of the age at the median matches 1 / (median * sigma * sqrt(2 pi))
  EXPECT_NEAR(std::exp(prior.log_density(140'000.0)),
              1.0 / (140'000.0 * 0.3 * std::sqrt(2.0 * std::acos(-1.0))), 1e-18);
}

TEST(Calibration_binding_test, ancestor_age_ordering_is_checked_in_both_directions) {
  auto sim = simulate(40, 17, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  // A nested chain root > outer > inner
  auto outer = k_no_node;
  auto inner = k_no_node;
  for (const auto& node : post_order_traversal(tree)) {
    if (node == tree.root || tree.at(node).is_tip()) { continue; }
    for (auto c : tree.at(node).children) {
      if (tree.at(c).is_inner_node() && std::ssize(tips_below(tree, c)) >= 2) {
        outer = node;
        inner = c;
      }
    }
    if (outer != k_no_node) { break; }
  }
  ASSERT_NE(inner, k_no_node);
  auto outer_names = tips_below(tree, outer);
  auto inner_names = tips_below(tree, inner);

  auto cfg = [&](double outer_oldest, double outer_youngest, double inner_oldest, double inner_youngest) {
    return parse_config(absl::StrFormat(
        "version 1\n"
        "calibration outer\n  taxa %s\n  age_units years_bp\n  age_reference_year 2020\n"
        "  distribution uniform\n  oldest %.10g\n  youngest %.10g\nend\n"
        "calibration inner\n  taxa %s\n  age_units years_bp\n  age_reference_year 2020\n"
        "  distribution uniform\n  oldest %.10g\n  youngest %.10g\nend\n",
        absl::StrJoin(outer_names, " "), outer_oldest, outer_youngest,
        absl::StrJoin(inner_names, " "), inner_oldest, inner_youngest));
  };

  // outer must be older than inner: disjoint windows the wrong way round are rejected
  expect_bind_rejected(cfg(200, 100, 5000, 2000), tree, "cannot both be satisfied");
  // Overlapping windows are accepted (the ordering is enforced by the tree, the prior only has to
  // leave it possible)
  EXPECT_NO_THROW(bind_calibrations_ignoring_times(cfg(5000, 100, 4000, 50), tree));
  // Disjoint windows the right way round are accepted
  EXPECT_NO_THROW(bind_calibrations_ignoring_times(cfg(9000, 5000, 4000, 1000), tree));
  // Touching windows are rejected: sup(outer) must be strictly greater than inf(inner)
  expect_bind_rejected(cfg(4000, 2000, 9000, 4000), tree, "cannot both be satisfied");
}

// Multi-partition behaviour
// =========================

TEST(Calibration_mcmc_test, calibrated_nodes_keep_moving_across_repartitionings) {
  // With four parts and forced repartitioning, a calibrated node is a movable inner node in some
  // batches and a frozen cut point in others.  It must still move often, and the full calibration
  // prior must agree with an independent recomputation in every batch (checked by run_and_check).
  auto pool = ctpl::thread_pool{4};
  auto sim = simulate(80, 18, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto root_age_years = -tree.at_root().t / 365.0;
  auto [clade_node, clade_names] = some_clade(tree, 5);
  ASSERT_NE(clade_node, k_no_node);
  auto clade_age_years = -tree.at(clade_node).t / 365.0;

  auto set = parse_config(absl::StrFormat(
      "version 1\n"
      "calibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n"
      "calibration inner\n  taxa %s\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n",
      root_age_years, root_age_years / 4, absl::StrJoin(clade_names, " "),
      clade_age_years, std::max(clade_age_years / 4, 1.0)));

  auto run = make_run(pool, tree, 61, 4, 100);
  run->set_calibrations(set);
  auto batches = 40;
  auto result = run_and_check(*run, set, scaled_steps(1'200'000), batches);
  ASSERT_EQ(std::ssize(result.num_changes_per_calibration), 2);
  for (auto i = 0; i != 2; ++i) {
    EXPECT_GT(result.num_changes_per_calibration[i], batches / 2)
        << "calibrated node " << i << " moved in only " << result.num_changes_per_calibration[i]
        << " of " << batches << " batches, so cut-point freezing is stalling it";
  }
}

TEST(Calibration_mcmc_test, single_and_multi_partition_posteriors_agree) {
  // The partition-local bookkeeping is exact, so the same calibrated posterior must be sampled with
  // one part and with four.  The comparison is a Monte Carlo one: the two posterior means of the root
  // age must agree to within the Monte Carlo error implied by the spread of the samples.
  auto sim = simulate(60, 19, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto root_age_years = -tree.at_root().t / 365.0;
  auto set = parse_config(absl::StrFormat(
      "version 1\ncalibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n",
      root_age_years, root_age_years / 5));

  auto stats = [](const std::vector<double>& xs) {
    auto half = std::ssize(xs) / 2;  // crude burn-in
    auto mean = 0.0;
    for (auto i = half; i != std::ssize(xs); ++i) { mean += xs[i]; }
    mean /= (std::ssize(xs) - half);
    auto var = 0.0;
    for (auto i = half; i != std::ssize(xs); ++i) { var += (xs[i] - mean) * (xs[i] - mean); }
    var /= (std::ssize(xs) - half - 1);
    return std::pair{mean, std::sqrt(var / (std::ssize(xs) - half))};  // mean, standard error
  };

  auto batches = 40;
  auto pool1 = ctpl::thread_pool{1};
  auto run1 = make_run(pool1, tree, 62, 1, -1, true);
  run1->set_calibrations(set);
  auto r1 = run_and_check(*run1, set, scaled_steps(1'600'000), batches);

  auto pool4 = ctpl::thread_pool{4};
  auto run4 = make_run(pool4, tree, 63, 4, 200, true);
  run4->set_calibrations(set);
  auto r4 = run_and_check(*run4, set, scaled_steps(1'600'000), batches);

  auto [m1, se1] = stats(r1.root_age_samples);
  auto [m4, se4] = stats(r4.root_age_samples);
  auto se = std::sqrt(se1 * se1 + se4 * se4);
  // The samples within a chain are correlated, so the nominal standard error understates the real
  // one; 4 nominal standard errors is a deliberately generous but still falsifiable band.
  EXPECT_LT(std::abs(m1 - m4), 4.0 * se + 0.02 * root_age_years)
      << "one part: " << m1 << " +/- " << se1 << ", four parts: " << m4 << " +/- " << se4;

  // The clock rate is only compared in Release builds: a Debug build runs 1/25 of the steps (see
  // scaled_steps) and mu, which moves only once per global move and is strongly correlated with the
  // node times, has then not left its starting value, so the comparison would test burn-in, not the
  // partition bookkeeping.
  if (not estd::is_debug_enabled) {
    auto [mu1, muse1] = stats(r1.mu_samples);
    auto [mu4, muse4] = stats(r4.mu_samples);
    EXPECT_LT(std::abs(mu1 - mu4), 4.0 * std::sqrt(muse1 * muse1 + muse4 * muse4) + 0.05 * mu1)
        << "one part: " << mu1 << ", four parts: " << mu4;
  }
}

// Checkpoint and resume
// =====================

TEST(Calibration_api_test, checkpoint_round_trip_through_tree_and_params_then_resume) {
  // This is the data path a .dphy snapshot uses: the tree goes through phylo_tree_to_api_tree /
  // ..._tree_info and the parameters through run_to_api_params.  The restored run must carry the same
  // calibrations, bound to the same nodes, with the same calibration prior, and must keep sampling.
  auto pool = ctpl::thread_pool{1};
  auto sim = simulate(30, 20, 1.0, 0.0, k_mu_site_day);
  auto tree = sim_tree(sim);
  auto root_age_years = -tree.at_root().t / 365.0;
  auto [clade_node, clade_names] = some_clade(tree, 3);
  ASSERT_NE(clade_node, k_no_node);
  auto set = parse_config(absl::StrFormat(
      "version 1\n"
      "calibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution lognormal\n  median %.12g\n  sigma 0.3\nend\n"
      "calibration inner\n  taxa %s\n  age_units years_bp\n  age_reference_year 2020\n"
      "  distribution normal\n  mean %.12g\n  stddev %.12g\nend\n",
      root_age_years, absl::StrJoin(clade_names, " "),
      -tree.at(clade_node).t / 365.0, std::max(-tree.at(clade_node).t / 365.0 / 3, 1.0)));

  auto source = make_run(pool, tree, 71, 1, -1);
  source->set_calibrations(set);
  source->do_mcmc_steps(static_cast<int>(scaled_steps(200'000)));
  auto before_calibration_prior = source->log_calibration_prior();
  auto before_posterior = source->log_posterior();
  auto before_step = source->step();
  auto before_mu = source->mu();

  // Save
  auto api_tree = phylo_tree_to_api_tree(source->tree());
  auto api_tree_info = phylo_tree_to_api_tree_info(source->tree());
  auto api_params = run_to_api_params(*source);

  // Restore into a fresh Run
  auto restored_tree = api_tree_and_tree_info_to_phylo_tree(api_tree.data(), api_tree_info.data());
  auto restored = std::make_shared<delphy::Run>(pool, std::mt19937{71}, 71, restored_tree);
  restored->set_topology_moves_enabled(false);
  apply_api_params_to_run(api_params.data(), *restored);

  ASSERT_TRUE(restored->has_calibrations());
  EXPECT_EQ(restored->calibrations().size(), source->calibrations().size());
  EXPECT_EQ(restored->step(), before_step);
  EXPECT_DOUBLE_EQ(restored->mu(), before_mu);
  for (auto i = 0; i != source->calibrations().size(); ++i) {
    const auto& a = source->calibrations().calibrations[i];
    const auto& b = restored->calibrations().calibrations[i];
    EXPECT_EQ(a.name, b.name);
    EXPECT_EQ(a.taxa, b.taxa);
    EXPECT_EQ(a.all_tips, b.all_tips);
    EXPECT_DOUBLE_EQ(a.reference_t, b.reference_t);
    // The calibration target is the same clade, named by its tip set rather than by an index
    EXPECT_EQ(tips_below(restored->tree(), restored->calibration_nodes()[i]),
              tips_below(source->tree(), source->calibration_nodes()[i]));
  }
  // Node times are stored as float32 in this format: at this depth the spacing is far
  // below a day, so the calibration prior agrees to many digits rather than exactly.
  EXPECT_NEAR(restored->log_calibration_prior(), before_calibration_prior,
              1e-6 * std::max(1.0, std::abs(before_calibration_prior)));
  // log_G and the coalescent prior react to the float32 rounding of every node and mutation time, so
  // the whole posterior agrees to about 1e-5 relative rather than to the 1e-6 of the calibration term
  EXPECT_NEAR(restored->log_posterior(), before_posterior,
              1e-4 * std::max(1.0, std::abs(before_posterior)));
  EXPECT_NEAR(restored->log_calibration_prior(),
              ref_log_calibration_prior(set, restored->tree()), 1e-10);

  // Resuming continues to sample the calibrated posterior, with the bookkeeping still consistent
  auto result = run_and_check(*restored, set, scaled_steps(200'000), 4);
  EXPECT_EQ(result.num_batches, 4);
  EXPECT_GT(restored->step(), before_step);
}

// Local move mix: opt-in skipping of the tip-date move
// =============================================================

namespace {

struct Move_mix {
  int64_t inner = 0, tip = 0, tip_accepted = 0, reform = 0, slide = 0, total = 0;
};

auto run_move_mix(Sim_data sim, bool skip, bool movable_tip, uint32_t seed, int num_parts) -> Move_mix {
  if (movable_tip) {
    sim.tip_descs[0].t_min -= 5.0f;  // a tip with an uncertain sampling date
    sim.tip_descs[0].t_max += 5.0f;
  }
  auto tree = sim_tree(sim);
  auto pool = ctpl::thread_pool{2};
  auto run = make_run(pool, tree, seed, num_parts, 400, /*fixed_pop=*/true);
  run->set_skip_fixed_tip_moves(skip);
  run->do_mcmc_steps(20'000);
  const auto& c = run->move_counts();
  auto at = [&](Move_kind k) { return c[static_cast<int>(k)]; };
  auto m = Move_mix{};
  m.inner = at(Move_kind::inner_node_displace).proposed;
  m.tip = at(Move_kind::tip_displace).proposed;
  m.tip_accepted = at(Move_kind::tip_displace).accepted;
  m.reform = at(Move_kind::branch_reform).proposed;
  m.slide = at(Move_kind::subtree_slide).proposed;
  m.total = m.inner + m.tip + m.reform + m.slide;
  return m;
}

}  // namespace

TEST(Move_efficiency_test, the_tip_move_is_the_identity_when_every_tip_date_is_fixed) {
  auto sim = simulate(12, 5, 1.0, 0.0, k_mu_site_day);
  auto m = run_move_mix(sim, /*skip=*/false, /*movable_tip=*/false, 21, 1);
  EXPECT_GT(m.tip, 0) << "by default the tip move is still scheduled";
  EXPECT_EQ(m.tip_accepted, 0) << "and cannot change anything when every tip has a fixed date";
}

TEST(Move_efficiency_test, skipping_removes_only_the_tip_move_and_renormalizes_the_others) {
  auto sim = simulate(12, 5, 1.0, 0.0, k_mu_site_day);
  for (auto num_parts : {1, 2}) {
    auto off = run_move_mix(sim, false, false, 31, num_parts);
    auto on = run_move_mix(sim, true, false, 31, num_parts);
    EXPECT_GT(off.tip, 0) << num_parts << " parts";
    EXPECT_EQ(on.tip, 0) << num_parts << " parts";
    // Topology moves are off, so the weights are 7.5 / 7.5 / 15 / 1 and, without the tip move, 7.5 / 15 / 1
    auto frac = [](int64_t x, int64_t t) { return static_cast<double>(x) / static_cast<double>(t); };
    EXPECT_NEAR(frac(off.inner, off.total), 7.5 / 31.0, 0.02) << num_parts << " parts";
    EXPECT_NEAR(frac(on.inner, on.total), 7.5 / 23.5, 0.02) << num_parts << " parts";
    EXPECT_NEAR(frac(on.reform, on.total), 15.0 / 23.5, 0.02) << num_parts << " parts";
    EXPECT_NEAR(frac(on.slide, on.total), 1.0 / 23.5, 0.01) << num_parts << " parts";
  }
}

TEST(Move_efficiency_test, a_tip_with_an_uncertain_date_keeps_the_tip_move_even_when_skipping_is_requested) {
  auto sim = simulate(12, 5, 1.0, 0.0, k_mu_site_day);
  auto on = run_move_mix(sim, /*skip=*/true, /*movable_tip=*/true, 41, 1);
  EXPECT_GT(on.tip, 0);
  EXPECT_NEAR(static_cast<double>(on.inner) / static_cast<double>(on.total), 7.5 / 31.0, 0.02);
}

// Saved state and resumed run: what a restored run carries of the local move mix settings
// ==============================================================================

namespace {

// A run with a human-style configuration: free exponential population under proper priors, the local-move
// interval of the local move mix, two parts, a pinned grid and every efficiency option away from its default.
auto make_configured_run(ctpl::thread_pool& pool, const Phylo_tree& tree, uint32_t seed) -> std::shared_ptr<Run> {
  auto run = make_run(pool, tree, seed, 2, 7 * static_cast<int>(std::ssize(tree)));
  run->set_pop_inv_n0_prior_alpha(3.0);
  run->set_pop_inv_n0_prior_beta(2.0 * k_coal_days);  // mean k_coal_days, matching the simulated timescale
  run->set_pop_g_prior_mu(0.0);
  run->set_pop_g_prior_scale(1e-5 / 365.0);
  run->set_pop_g_min(0.0);
  run->set_pop_g_max(1e-4 / 365.0);
  run->set_pop_model(std::make_shared<Exp_pop_model>(0.0, k_coal_days, 1e-6 / 365.0, 100.0));
  run->set_coal_prior_fixed_t_step(2.0);
  run->set_pop_moves_per_global_move(2);
  run->set_skip_fixed_tip_moves(true);
  run->set_root_displace_scale_factor(3.0);
  run->set_pop_g_move_window(1e-5 / 365.0);
  return run;
}

struct Saved_state {
  flatbuffers::DetachedBuffer tree, tree_info, params;
};

auto save_state(const Run& run) -> Saved_state {
  return {phylo_tree_to_api_tree(run.tree()), phylo_tree_to_api_tree_info(run.tree()), run_to_api_params(run)};
}

auto restore_into_fresh_run(ctpl::thread_pool& pool, const Saved_state& saved, uint32_t seed) -> std::shared_ptr<Run> {
  auto restored_tree = api_tree_and_tree_info_to_phylo_tree(saved.tree.data(), saved.tree_info.data());
  auto restored = std::make_shared<Run>(pool, std::mt19937{seed}, seed, restored_tree);
  restored->set_topology_moves_enabled(false);
  return restored;
}

}  // namespace

TEST(Restore_test, a_restored_run_carries_the_population_priors_the_schedule_and_the_parts) {
  auto pool = ctpl::thread_pool{2};
  auto tree = sim_tree(simulate(20, 61, 1.0, 0.0, k_mu_site_day));
  auto source = make_configured_run(pool, tree, 81);
  source->do_mcmc_steps(static_cast<int>(scaled_steps(100'000)));
  auto saved = save_state(*source);
  auto restored = restore_into_fresh_run(pool, saved, 82);
  restored->set_coal_prior_fixed_t_step(2.0);  // the grid is not stored; the caller pins it again
  apply_api_params_to_run(saved.params.data(), *restored);

  EXPECT_EQ(restored->step(), source->step());
  EXPECT_EQ(restored->local_moves_per_global_move(), source->local_moves_per_global_move());
  EXPECT_EQ(restored->num_parts(), source->num_parts());
  EXPECT_DOUBLE_EQ(restored->mu(), source->mu());
  EXPECT_DOUBLE_EQ(restored->hky_kappa(), source->hky_kappa());
  EXPECT_DOUBLE_EQ(restored->pop_inv_n0_prior_alpha(), source->pop_inv_n0_prior_alpha());
  EXPECT_DOUBLE_EQ(restored->pop_inv_n0_prior_beta(), source->pop_inv_n0_prior_beta());
  EXPECT_DOUBLE_EQ(restored->pop_g_prior_mu(), source->pop_g_prior_mu());
  EXPECT_DOUBLE_EQ(restored->pop_g_prior_scale(), source->pop_g_prior_scale());
  EXPECT_DOUBLE_EQ(restored->pop_g_min(), source->pop_g_min());
  EXPECT_DOUBLE_EQ(restored->pop_g_max(), source->pop_g_max());
  const auto& a = static_cast<const Exp_pop_model&>(source->pop_model());
  const auto& b = static_cast<const Exp_pop_model&>(restored->pop_model());
  EXPECT_DOUBLE_EQ(b.pop_at_t0(), a.pop_at_t0());
  EXPECT_DOUBLE_EQ(b.growth_rate(), a.growth_rate());
  EXPECT_DOUBLE_EQ(b.min_pop(), a.min_pop());
  EXPECT_EQ(restored->final_pop_size_move_enabled(), source->final_pop_size_move_enabled());
  EXPECT_EQ(restored->pop_growth_rate_move_enabled(), source->pop_growth_rate_move_enabled());
  EXPECT_EQ(restored->mu_move_enabled(), source->mu_move_enabled());
  // Node and mutation times are stored as float32; with the same grid pinned on both sides the
  // whole posterior agrees to that rounding
  EXPECT_NEAR(restored->log_posterior(), source->log_posterior(),
              1e-4 * std::max(1.0, std::abs(source->log_posterior())));
  EXPECT_NEAR(restored->log_coalescent_prior(), source->log_coalescent_prior(),
              1e-4 * std::max(1.0, std::abs(source->log_coalescent_prior())));
}

TEST(Restore_test, a_restored_run_does_not_carry_the_grid_or_the_efficiency_options) {
  auto pool = ctpl::thread_pool{2};
  auto tree = sim_tree(simulate(20, 61, 1.0, 0.0, k_mu_site_day));
  auto source = make_configured_run(pool, tree, 81);
  auto saved = save_state(*source);
  auto restored = restore_into_fresh_run(pool, saved, 82);
  apply_api_params_to_run(saved.params.data(), *restored);

  // Documented limitation (api.cpp: "Make more things settable, e.g. coalescent prior staircase
  // resolution"): none of these is stored, so a run restored without them silently reverts to the defaults.
  EXPECT_EQ(restored->coal_prior_fixed_t_step(), 0.0);
  EXPECT_EQ(source->coal_prior_fixed_t_step(), 2.0);
  EXPECT_EQ(restored->pop_moves_per_global_move(), 50);
  EXPECT_EQ(source->pop_moves_per_global_move(), 2);
  EXPECT_FALSE(restored->skip_fixed_tip_moves());
  EXPECT_TRUE(source->skip_fixed_tip_moves());
  EXPECT_DOUBLE_EQ(restored->root_displace_scale_factor(), 1.0);
  EXPECT_DOUBLE_EQ(source->root_displace_scale_factor(), 3.0);
  EXPECT_DOUBLE_EQ(restored->pop_g_move_window(), 1.0 / 365.0);
  EXPECT_DOUBLE_EQ(source->pop_g_move_window(), 1e-5 / 365.0);
  // The coalescent prior then differs from the source's, because it is evaluated on another grid: the
  // restored run does not sample the same target density
  EXPECT_GT(std::abs(restored->log_coalescent_prior() - source->log_coalescent_prior()), 1e-6);
}

TEST(Restore_test, settings_made_on_the_new_run_survive_the_restore_and_pinning_the_grid_restores_the_target) {
  auto pool = ctpl::thread_pool{2};
  auto tree = sim_tree(simulate(20, 61, 1.0, 0.0, k_mu_site_day));
  auto source = make_configured_run(pool, tree, 81);
  source->do_mcmc_steps(static_cast<int>(scaled_steps(50'000)));
  auto saved = save_state(*source);
  auto restored = restore_into_fresh_run(pool, saved, 82);
  restored->set_coal_prior_fixed_t_step(source->coal_prior_fixed_t_step());
  restored->set_pop_moves_per_global_move(source->pop_moves_per_global_move());
  restored->set_skip_fixed_tip_moves(source->skip_fixed_tip_moves());
  restored->set_root_displace_scale_factor(source->root_displace_scale_factor());
  restored->set_pop_g_move_window(source->pop_g_move_window());
  apply_api_params_to_run(saved.params.data(), *restored);
  EXPECT_EQ(restored->pop_moves_per_global_move(), 2);
  EXPECT_TRUE(restored->skip_fixed_tip_moves());
  EXPECT_DOUBLE_EQ(restored->root_displace_scale_factor(), 3.0);
  EXPECT_DOUBLE_EQ(restored->pop_g_move_window(), 1e-5 / 365.0);
  EXPECT_EQ(restored->coal_prior_fixed_t_step(), 2.0);
  EXPECT_NEAR(restored->log_coalescent_prior(), source->log_coalescent_prior(),
              1e-4 * std::max(1.0, std::abs(source->log_coalescent_prior())));
}

TEST(Restore_test, a_resumed_run_with_other_efficiency_options_keeps_sampling_with_consistent_bookkeeping) {
  auto pool = ctpl::thread_pool{2};
  auto tree = sim_tree(simulate(20, 61, 1.0, 0.0, k_mu_site_day));
  auto source = make_configured_run(pool, tree, 81);
  source->do_mcmc_steps(static_cast<int>(scaled_steps(50'000)));
  auto saved = save_state(*source);
  auto restored = restore_into_fresh_run(pool, saved, 83);
  restored->set_coal_prior_fixed_t_step(2.0);
  apply_api_params_to_run(saved.params.data(), *restored);
  // Other options than the source's are legal: they change the schedule, not the target
  restored->set_pop_moves_per_global_move(5);
  restored->set_skip_fixed_tip_moves(false);
  restored->set_root_displace_scale_factor(0.5);
  restored->do_mcmc_steps(static_cast<int>(scaled_steps(100'000)));  // checks derived quantities in Debug
  const auto& c = restored->move_counts();
  EXPECT_GT(c[static_cast<int>(Move_kind::tip_displace)].proposed, 0) << "the tip move is back in the mix";
  EXPECT_GT(c[static_cast<int>(Move_kind::inner_node_displace)].accepted, 0);
  EXPECT_TRUE(std::isfinite(restored->log_posterior()));
}

}  // namespace delphy
