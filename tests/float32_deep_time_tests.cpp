// Measurements and checks of float32 time storage at human deep-time scales.
//
// In memory, node and mutation times are double.  Tip bounds (Tip_desc / Phylo_node t_min, t_max) are float, and the
// .dphy / api representation stores node times, mutation times and tip bounds as float32.  Times are days since
// 2020-01-01, so 50,000 to 200,000 years before present are about -1.8e7 to -7.3e7 days.

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "absl/strings/str_format.h"

#include "api.h"
#include "phylo_tree.h"
#include "utree.h"

namespace delphy {

namespace {

constexpr auto k_days_per_year = 365.0;

auto float32_spacing_days(double t_days) -> double {
  auto f = static_cast<float>(t_days);
  return static_cast<double>(std::nextafter(f, std::numeric_limits<float>::infinity())) - static_cast<double>(f);
}

// Modern tips (t = 0) derived from earlier tips by 1-3 substitutions, in `num_sites` sites.
auto make_modern_tips(int num_tips, int num_sites, uint32_t seed, Real_sequence& ref) -> std::vector<Tip_desc> {
  auto rng = std::mt19937{seed};
  ref.assign(num_sites, Real_seq_letter::A);
  for (auto& r : ref) { r = static_cast<Real_seq_letter>(rng() % 4); }
  auto seqs = std::vector<Real_sequence>{};
  auto tips = std::vector<Tip_desc>{};
  for (auto i = 0; i != num_tips; ++i) {
    auto seq = (i == 0) ? ref : seqs[rng() % i];
    for (auto m = 0, n = 1 + static_cast<int>(rng() % 3); m != n; ++m) {
      auto site = static_cast<int>(rng() % num_sites);
      auto old = seq[site];
      do { seq[site] = static_cast<Real_seq_letter>(rng() % 4); } while (seq[site] == old);
    }
    seqs.push_back(seq);
    auto td = Tip_desc{};
    td.name = absl::StrFormat("T%04d", i);
    td.t_min = 0.0f;
    td.t_max = 0.0f;
    for (auto s = 0; s != num_sites; ++s) {
      if (seq[s] != ref[s]) { td.seq_deltas.push_back(Seq_delta{s, ref[s], seq[s]}); }
    }
    tips.push_back(std::move(td));
  }
  return tips;
}

auto build_tree_with_root_depth(double target_years, int num_tips, uint32_t seed) -> Phylo_tree {
  auto ref = Real_sequence{};
  auto tips = make_modern_tips(num_tips, 400, seed, ref);
  auto depth_for = [&](double lambda) {
    auto bitgen = std::mt19937{seed};
    auto tree = build_initial_phylo_tree(
        ref, tips, bitgen, [](int, int) {}, [](int, int, int) {}, [](int, int, int) {},
        [](int, int, int, int, int) {}, [](const Rooting_info&) {},
        Init_time_scale{.fallback_lambda = lambda, .forbid_pathogen_fallback = true});
    return tree;
  };
  auto lambda = 1e-3;
  auto tree = depth_for(lambda);
  auto depth_years = -tree.at_root().t / k_days_per_year;
  lambda *= depth_years / target_years;  // root depth is inversely proportional to the rate
  return depth_for(lambda);
}

struct Roundtrip_stats {
  double max_abs_t_days = 0.0;
  double max_time_error_days = 0.0;
  int ties_with_parent = 0;        // child time == parent time after the round trip, but not before
  int inversions = 0;              // child time < parent time after the round trip, beyond the integrity slack
  int mutation_ties = 0;
  int mutation_inversions = 0;
  double min_branch_before_days = std::numeric_limits<double>::infinity();
};

auto roundtrip_stats(const Phylo_tree& tree) -> Roundtrip_stats {
  auto tree_fb = phylo_tree_to_api_tree(tree);
  auto info_fb = phylo_tree_to_api_tree_info(tree);
  auto reloaded = api_tree_and_tree_info_to_phylo_tree(tree_fb.data(), info_fb.data());
  auto s = Roundtrip_stats{};
  EXPECT_EQ(std::ssize(reloaded), std::ssize(tree));
  assert_phylo_tree_integrity(reloaded);  // aborts if the reloaded tree violates the 1e-2 day ordering slack
  for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
    const auto& before = tree.at(node);
    const auto& after = reloaded.at(node);
    s.max_abs_t_days = std::max(s.max_abs_t_days, std::abs(before.t));
    s.max_time_error_days = std::max(s.max_time_error_days, std::abs(after.t - before.t));
    if (before.parent == k_no_node) { continue; }
    auto t_parent_before = tree.at(before.parent).t;
    auto t_parent_after = reloaded.at(after.parent).t;
    s.min_branch_before_days = std::min(s.min_branch_before_days, before.t - t_parent_before);
    if (after.t == t_parent_after && before.t != t_parent_before) { ++s.ties_with_parent; }
    if (after.t < t_parent_after - 1e-2) { ++s.inversions; }
    auto t_prev = t_parent_after;
    for (const auto& m : after.mutations) {
      if (m.t == t_prev) { ++s.mutation_ties; }
      if (m.t < t_prev - 1e-2) { ++s.mutation_inversions; }
      t_prev = m.t;
    }
    if (not after.mutations.empty() && after.t < t_prev - 1e-2) { ++s.mutation_inversions; }
  }
  return s;
}

}  // namespace

TEST(Float32_deep_time_test, float32_spacing_at_deep_times) {
  // The spacing doubles at every power of two.
  struct Row { double years; double expected_spacing_days; };
  for (auto row : {Row{10'000, 0.25}, Row{50'000, 2.0}, Row{100'000, 4.0}, Row{140'000, 4.0}, Row{200'000, 8.0},
                   Row{300'000, 8.0}}) {
    auto t = -row.years * k_days_per_year;
    auto spacing = float32_spacing_days(t);
    std::cout << absl::StreamFormat("[float32] %7.0f years before 2020: t = %.0f d, float32 spacing = %g d, "
                                    "max rounding error = %g d\n", row.years, t, spacing, spacing / 2);
    EXPECT_EQ(spacing, row.expected_spacing_days) << row.years;
  }
}

TEST(Float32_deep_time_test, float32_rounding_is_monotone_so_close_times_tie_but_never_invert) {
  auto rng = std::mt19937_64{3};
  for (auto years : {50'000.0, 100'000.0, 140'000.0, 200'000.0}) {
    auto t0 = -years * k_days_per_year;
    auto ties = 0;
    constexpr auto n = 100000;
    for (auto i = 0; i != n; ++i) {
      auto gap = std::uniform_real_distribution<double>{0.0, 20.0}(rng);
      auto a = t0 + std::uniform_real_distribution<double>{0.0, 1000.0}(rng);
      auto b = a + gap;
      EXPECT_LE(static_cast<float>(a), static_cast<float>(b));
      if (static_cast<float>(a) == static_cast<float>(b)) { ++ties; }
    }
    std::cout << absl::StreamFormat("[float32] %7.0f years: %.1f%% of node pairs 0-20 days apart collapse to equal float32 times\n",
                                    years, 100.0 * ties / n);
  }
}

TEST(Float32_deep_time_test, dphy_roundtrip_of_trees_with_deep_roots) {
  for (auto target_years : {50'000.0, 100'000.0, 140'000.0, 200'000.0}) {
    auto tree = build_tree_with_root_depth(target_years, 60, 17);
    auto s = roundtrip_stats(tree);
    auto depth_years = -tree.at_root().t / k_days_per_year;
    std::cout << absl::StreamFormat(
        "[float32] root %.0f years: max |t| = %.0f d, max time error = %.3f d (float32 spacing there = %g d), "
        "min branch before = %.3f d, branches tied with parent after = %d, inversions = %d, "
        "mutation-time ties = %d, mutation inversions = %d\n",
        depth_years, s.max_abs_t_days, s.max_time_error_days, float32_spacing_days(s.max_abs_t_days),
        s.min_branch_before_days, s.ties_with_parent, s.inversions, s.mutation_ties, s.mutation_inversions);
    EXPECT_NEAR(depth_years, target_years, 0.05 * target_years);
    EXPECT_LE(s.max_time_error_days, float32_spacing_days(s.max_abs_t_days) / 2 + 1e-9);
    EXPECT_EQ(s.inversions, 0);
    EXPECT_EQ(s.mutation_inversions, 0);
  }
}

}  // namespace delphy
