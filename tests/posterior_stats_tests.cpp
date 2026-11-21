// Tests for the posterior summary statistics.
//
// Reference values are computed independently: the quantile and interval references come from
// numpy/scipy on the same deterministic samples (the expressions are repeated in the comments), and
// the effective sample size and R-hat references come from their closed forms.  Nothing here is
// checked against another part of Delphy.

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "boost/math/distributions/normal.hpp"

#include "posterior_stats.h"

namespace delphy {

namespace {

auto sorted(std::vector<double> v) -> std::vector<double> {
  std::ranges::sort(v);
  return v;
}

// Deterministic sample of a distribution: the inverse CDF at the n midpoints (i - 0.5) / n.  This
// makes every interval below an exact property of a known sample, with no Monte Carlo noise.
auto normal_grid(int n) -> std::vector<double> {
  auto d = boost::math::normal_distribution<double>{};
  auto v = std::vector<double>{};
  for (auto i = 1; i <= n; ++i) { v.push_back(boost::math::quantile(d, (i - 0.5) / n)); }
  return v;
}

auto exponential_grid(int n) -> std::vector<double> {
  auto v = std::vector<double>{};
  for (auto i = 1; i <= n; ++i) { v.push_back(-std::log(1.0 - (i - 0.5) / n)); }
  return v;
}

auto count_in(const std::vector<double>& xs, Credible_interval ci) -> int {
  auto n = 0;
  for (auto x : xs) {
    if (x >= ci.lo && x <= ci.hi) { ++n; }
  }
  return n;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Quantiles and equal-tailed intervals
// ---------------------------------------------------------------------------------------------

TEST(Posterior_stats_test, type_7_quantiles_match_numpy) {
  // numpy.quantile([1,2,3,4,100], [0.025, 0.5, 0.975]) == [1.1, 3.0, 90.4]
  auto x = sorted({1.0, 2.0, 3.0, 4.0, 100.0});
  EXPECT_NEAR(sample_quantile(x, 0.025), 1.1, 1e-12);
  EXPECT_NEAR(sample_quantile(x, 0.5), 3.0, 1e-12);
  EXPECT_NEAR(sample_quantile(x, 0.975), 90.4, 1e-12);
  EXPECT_EQ(sample_quantile(x, 0.0), 1.0);
  EXPECT_EQ(sample_quantile(x, 1.0), 100.0);
}

TEST(Posterior_stats_test, quantiles_of_degenerate_samples) {
  EXPECT_TRUE(std::isnan(sample_quantile(std::vector<double>{}, 0.5)));
  auto one = std::vector<double>{7.0};
  EXPECT_EQ(sample_quantile(one, 0.0), 7.0);
  EXPECT_EQ(sample_quantile(one, 0.5), 7.0);
  EXPECT_EQ(sample_quantile(one, 1.0), 7.0);
}

TEST(Posterior_stats_test, equal_tailed_interval_is_the_pair_of_quantiles) {
  auto x = sorted({1.0, 2.0, 3.0, 4.0, 100.0});
  auto ci = equal_tailed_interval(x, 0.95);
  EXPECT_NEAR(ci.lo, 1.1, 1e-12);
  EXPECT_NEAR(ci.hi, 90.4, 1e-12);
}

// ---------------------------------------------------------------------------------------------
// Shortest credible intervals, and how they relate to the highest density region
// ---------------------------------------------------------------------------------------------

TEST(Posterior_stats_test, shortest_interval_is_the_shortest_window_of_ceil_mass_times_n_samples) {
  // n = 5, mass = 0.6 -> k = 3.  Windows: [1,3] width 2, [2,4] width 2, [3,100] width 97.
  // The first shortest window wins.
  auto x = sorted({1.0, 2.0, 3.0, 4.0, 100.0});
  auto ci = shortest_credible_interval(x, 0.6);
  EXPECT_EQ(ci.lo, 1.0);
  EXPECT_EQ(ci.hi, 3.0);
  EXPECT_GE(count_in(x, ci), 3);
}

TEST(Posterior_stats_test, shortest_interval_contains_at_least_the_requested_mass) {
  for (auto n : {5, 17, 100, 1001}) {
    auto x = normal_grid(n);
    for (auto mass : {0.5, 0.9, 0.95, 0.99}) {
      auto ci = shortest_credible_interval(x, mass);
      auto required = static_cast<int>(std::ceil(mass * n));
      EXPECT_GE(count_in(x, ci), required) << "n=" << n << " mass=" << mass;
    }
  }
}

TEST(Posterior_stats_test, shortest_interval_is_never_wider_than_the_equal_tailed_interval) {
  // Holds whenever the interval does not have to contain every sample, i.e. ceil(mass * n) < n.
  for (auto n : {101, 1001, 20001}) {
    for (const auto& x : {normal_grid(n), exponential_grid(n)}) {
      auto interval = shortest_credible_interval(x, 0.95);
      auto eti = equal_tailed_interval(x, 0.95);
      EXPECT_LE(interval.width(), eti.width() + 1e-9) << "n=" << n;
    }
  }
}

TEST(Posterior_stats_test, shortest_interval_degenerates_to_the_full_range_when_the_sample_is_too_small) {
  // With n = 11 and 95% credibility, ceil(0.95 * 11) = 11: every sample must be inside, so the only
  // interval is the full range.  The equal-tailed interval interpolates between order statistics and
  // is therefore narrower.  This is a property of 11 samples, not a defect, and it is why the
  // summary tables report the sample size next to every interval.
  auto x = normal_grid(11);
  auto interval = shortest_credible_interval(x, 0.95);
  EXPECT_EQ(interval.lo, x.front());
  EXPECT_EQ(interval.hi, x.back());
  EXPECT_GT(interval.width(), equal_tailed_interval(x, 0.95).width());
}

TEST(Posterior_stats_test, shortest_interval_equals_the_equal_tailed_interval_for_a_symmetric_sample) {
  // Both intervals of a symmetric unimodal sample are the central one.  scipy:
  // norm.ppf((i-0.5)/20001) at i = 500.025 -> +/-1.9595578021 for both intervals.
  auto x = normal_grid(20001);
  auto interval = shortest_credible_interval(x, 0.95);
  auto eti = equal_tailed_interval(x, 0.95);
  EXPECT_NEAR(interval.lo, -1.9595578021, 1e-6);
  EXPECT_NEAR(interval.hi, +1.9595578021, 1e-6);
  EXPECT_NEAR(eti.lo, -1.9595578021, 1e-6);
  EXPECT_NEAR(eti.hi, +1.9595578021, 1e-6);
}

TEST(Posterior_stats_test, shortest_interval_differs_from_the_equal_tailed_interval_for_a_skewed_sample) {
  // A unit exponential sample.  The shortest interval of a monotone density runs from the lower end
  // to the 95% quantile; the equal-tailed interval cuts both tails.  Reference values from numpy on
  // the same grid: shortest = [0.0000249991, 2.9952823973], eti = [0.0253421660, 3.6879299525].
  auto x = exponential_grid(20001);
  auto interval = shortest_credible_interval(x, 0.95);
  auto eti = equal_tailed_interval(x, 0.95);
  EXPECT_NEAR(interval.lo, 0.0000249991, 1e-8);
  EXPECT_NEAR(interval.hi, 2.9952823973, 1e-7);
  EXPECT_NEAR(eti.lo, 0.0253421660, 1e-8);
  EXPECT_NEAR(eti.hi, 3.6879299525, 1e-7);
  EXPECT_LT(interval.width(), eti.width());
  // This is exactly the mistake the task forbids: naming the 2.5%-97.5% quantiles a highest-density
  // interval would move the upper end by 23% here.
  EXPECT_GT(std::abs(eti.hi - interval.hi) / interval.hi, 0.2);
}

TEST(Posterior_stats_test, intervals_of_degenerate_samples) {
  auto empty = std::vector<double>{};
  EXPECT_FALSE(shortest_credible_interval(empty, 0.95).is_defined());
  EXPECT_FALSE(equal_tailed_interval(empty, 0.95).is_defined());
  auto one = std::vector<double>{3.5};
  auto ci = shortest_credible_interval(one, 0.95);
  EXPECT_EQ(ci.lo, 3.5);
  EXPECT_EQ(ci.hi, 3.5);
  auto constant = std::vector<double>(100, 2.0);
  auto ci2 = shortest_credible_interval(constant, 0.95);
  EXPECT_EQ(ci2.lo, 2.0);
  EXPECT_EQ(ci2.hi, 2.0);
}

// ---------------------------------------------------------------------------------------------
// Multimodality diagnostic
// ---------------------------------------------------------------------------------------------

TEST(Posterior_stats_test, hdr_of_a_unimodal_sample_is_a_single_interval) {
  for (auto n : {200, 2000, 20000}) {
    auto x = normal_grid(n);
    auto hdr = highest_density_region(x, 0.95);
    EXPECT_EQ(hdr.num_components, 1) << "n=" << n;
    EXPECT_NEAR(hdr.interval_gap_fraction, 0.0, 1e-9) << "n=" << n;
  }
}

TEST(Posterior_stats_test, hdr_does_not_cry_multimodal_on_noisy_unimodal_samples) {
  // A finite random sample of a unimodal density has Poisson noise in its histogram, which a naive
  // bin-by-bin highest-density rule breaks into many pieces.  This is the regression test for that.
  auto prng = std::mt19937{12345};
  auto normal = std::normal_distribution<double>{0.0, 1.0};
  for (auto n : {500, 5600, 20000}) {
    auto x = std::vector<double>{};
    for (auto i = 0; i != n; ++i) { x.push_back(normal(prng)); }
    auto hdr = highest_density_region(sorted(x), 0.95);
    EXPECT_EQ(hdr.num_components, 1) << "n=" << n << " bins=" << hdr.num_bins;
  }
}

TEST(Posterior_stats_test, hdr_finds_two_well_separated_modes) {
  // Half the mass near 0, half near 20, with nothing in between.  A single interval spanning both
  // modes is 95% credible and almost entirely empty in the middle.  The highest density region -- the
  // set that "HPD" actually names -- has two components here, and reporting the shortest contiguous
  // interval as if it were an HPD interval is exactly the error this distinction prevents.
  auto x = std::vector<double>{};
  auto left = normal_grid(5000);
  for (auto v : left) { x.push_back(v); }
  for (auto v : left) { x.push_back(v + 20.0); }
  auto s = sorted(x);
  auto hdr = highest_density_region(s, 0.95);
  EXPECT_EQ(hdr.num_components, 2);
  EXPECT_GT(hdr.interval_gap_fraction, 0.5);
  auto interval = shortest_credible_interval(s, 0.95);
  EXPECT_GT(interval.width(), 18.0);  // Spans the empty middle, so it is not the HPD region
  // The two HDR components together are much narrower than the interval that bridges them
  auto hdr_width = 0.0;
  for (const auto& c : hdr.components) { hdr_width += c.width(); }
  EXPECT_LT(hdr_width, 0.5 * interval.width());
}

TEST(Posterior_stats_test, hdr_counts_a_tiny_satellite_mode_as_minor) {
  auto x = normal_grid(10000);
  for (auto i = 0; i != 50; ++i) { x.push_back(40.0 + i * 1e-3); }  // 0.5% of the mass, far away
  auto hdr = highest_density_region(sorted(x), 0.95);
  EXPECT_EQ(hdr.num_components, 1);
}

TEST(Posterior_stats_test, hdr_bin_count_is_reported_and_configurable) {
  auto x = normal_grid(10000);
  EXPECT_EQ(highest_density_region(x, 0.95).num_bins, 100);  // round(sqrt(10000)) clamped to 100
  EXPECT_EQ(highest_density_region(x, 0.95, 37).num_bins, 37);
  EXPECT_EQ(highest_density_region(normal_grid(49), 0.95).num_bins, 10);  // clamped up from 7
}

// ---------------------------------------------------------------------------------------------
// Effective sample size
// ---------------------------------------------------------------------------------------------

TEST(Posterior_stats_test, ess_of_independent_draws_is_about_n) {
  auto prng = std::mt19937{777};
  auto normal = std::normal_distribution<double>{0.0, 1.0};
  auto x = std::vector<double>{};
  for (auto i = 0; i != 20000; ++i) { x.push_back(normal(prng)); }
  auto e = effective_sample_size(x);
  EXPECT_FALSE(e.truncated);
  EXPECT_GT(e.ess, 0.8 * 20000);
  EXPECT_LE(e.ess, 20000.0);
  EXPECT_NEAR(e.tau, 1.0, 0.25);
}

TEST(Posterior_stats_test, ess_of_an_ar1_series_matches_its_closed_form) {
  // For an AR(1) series with lag-1 correlation rho, tau = (1 + rho) / (1 - rho).
  auto prng = std::mt19937{4242};
  auto normal = std::normal_distribution<double>{0.0, 1.0};
  for (auto rho : {0.5, 0.8, 0.95}) {
    auto n = 200000;
    auto x = std::vector<double>{};
    auto v = 0.0;
    for (auto i = 0; i != n; ++i) {
      v = rho * v + std::sqrt(1.0 - rho * rho) * normal(prng);
      x.push_back(v);
    }
    auto expected_tau = (1.0 + rho) / (1.0 - rho);
    auto e = effective_sample_size(x);
    EXPECT_NEAR(e.tau / expected_tau, 1.0, 0.2) << "rho=" << rho << " tau=" << e.tau;
    EXPECT_NEAR(e.ess / (n / expected_tau), 1.0, 0.25) << "rho=" << rho;
  }
}

TEST(Posterior_stats_test, ess_of_degenerate_series) {
  EXPECT_EQ(effective_sample_size(std::vector<double>{}).ess, 0.0);
  EXPECT_EQ(effective_sample_size(std::vector<double>{1.0, 2.0, 3.0}).ess, 0.0);  // n < 4
  auto constant = std::vector<double>(100, 5.0);
  EXPECT_EQ(effective_sample_size(constant).ess, 100.0);
  EXPECT_EQ(effective_sample_size(constant).tau, 1.0);
}

TEST(Posterior_stats_test, ess_reports_truncation_of_the_positive_sequence) {
  // A series with a very long correlation time, summarized with a deliberately tight lag budget.
  auto x = std::vector<double>{};
  for (auto i = 0; i != 4000; ++i) { x.push_back(std::sin(i * 2.0 * std::acos(-1.0) / 100000.0)); }
  auto e = effective_sample_size(x, 50);
  EXPECT_TRUE(e.truncated);
  EXPECT_EQ(e.num_lags_used, 50);
}

// ---------------------------------------------------------------------------------------------
// Split R-hat
// ---------------------------------------------------------------------------------------------

TEST(Posterior_stats_test, rhat_of_chains_from_the_same_distribution_is_about_one) {
  auto prng = std::mt19937{99};
  auto normal = std::normal_distribution<double>{0.0, 1.0};
  auto chains = std::vector<std::vector<double>>{};
  for (auto c = 0; c != 4; ++c) {
    auto x = std::vector<double>{};
    for (auto i = 0; i != 5000; ++i) { x.push_back(normal(prng)); }
    chains.push_back(std::move(x));
  }
  EXPECT_NEAR(split_rhat(chains), 1.0, 0.01);
  EXPECT_NEAR(rank_normalized_split_rhat(chains), 1.0, 0.01);
}

TEST(Posterior_stats_test, rhat_detects_chains_in_different_places) {
  auto prng = std::mt19937{100};
  auto normal = std::normal_distribution<double>{0.0, 1.0};
  auto chains = std::vector<std::vector<double>>{};
  for (auto c = 0; c != 4; ++c) {
    auto x = std::vector<double>{};
    for (auto i = 0; i != 5000; ++i) { x.push_back(normal(prng) + 3.0 * c); }
    chains.push_back(std::move(x));
  }
  EXPECT_GT(split_rhat(chains), 1.5);
  EXPECT_GT(rank_normalized_split_rhat(chains), 1.5);
}

TEST(Posterior_stats_test, rank_normalization_tames_a_heavy_tailed_target) {
  // Cauchy draws have no finite variance, so the plain statistic is driven by whichever chain drew
  // the largest outlier.  The rank-normalized statistic is the one to judge convergence by.
  auto prng = std::mt19937{2024};
  auto uniform = std::uniform_real_distribution<double>{0.0, 1.0};
  auto chains = std::vector<std::vector<double>>{};
  for (auto c = 0; c != 4; ++c) {
    auto x = std::vector<double>{};
    for (auto i = 0; i != 4000; ++i) {
      x.push_back(std::tan(std::acos(-1.0) * (uniform(prng) - 0.5)));
    }
    chains.push_back(std::move(x));
  }
  EXPECT_NEAR(rank_normalized_split_rhat(chains), 1.0, 0.02);
}

TEST(Posterior_stats_test, rhat_of_a_single_chain_is_the_split_statistic) {
  auto prng = std::mt19937{5};
  auto normal = std::normal_distribution<double>{0.0, 1.0};
  auto one = std::vector<double>{};
  for (auto i = 0; i != 10000; ++i) { one.push_back(normal(prng)); }
  auto chains = std::vector<std::vector<double>>{one};
  EXPECT_NEAR(rank_normalized_split_rhat(chains), 1.0, 0.02);

  // A chain that drifts: the two halves disagree and the split statistic says so.
  auto drifting = std::vector<double>{};
  for (auto i = 0; i != 10000; ++i) { drifting.push_back(normal(prng) + i * 1e-3); }
  auto drifting_chains = std::vector<std::vector<double>>{drifting};
  EXPECT_GT(rank_normalized_split_rhat(drifting_chains), 1.5);
}

TEST(Posterior_stats_test, rhat_is_undefined_for_too_few_samples) {
  auto chains = std::vector<std::vector<double>>{{1.0, 2.0, 3.0}};
  EXPECT_LT(split_rhat(chains), 0.0);
  EXPECT_LT(rank_normalized_split_rhat(chains), 0.0);
  auto none = std::vector<std::vector<double>>{};
  EXPECT_LT(split_rhat(none), 0.0);
}

TEST(Posterior_stats_test, moments) {
  auto x = std::vector<double>{1.0, 2.0, 3.0, 4.0};
  auto m = calc_moments(x);
  EXPECT_EQ(m.n, 4);
  EXPECT_NEAR(m.mean, 2.5, 1e-12);
  EXPECT_NEAR(m.stddev, std::sqrt(5.0 / 3.0), 1e-12);  // sample standard deviation
  EXPECT_EQ(calc_moments(std::vector<double>{9.0}).stddev, 0.0);
}

}  // namespace delphy
