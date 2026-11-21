#include "posterior_stats.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "absl/log/check.h"
#include "boost/math/distributions/normal.hpp"

namespace delphy {

namespace {

auto check_mass(double mass) -> void {
  CHECK_GT(mass, 0.0);
  CHECK_LT(mass, 1.0);
}

// Number of samples that an interval of the given credible mass must contain.
auto num_samples_in_interval(int n, double mass) -> int {
  auto k = static_cast<int>(std::ceil(mass * n));
  return std::clamp(k, 1, n);
}

}  // namespace

auto sample_quantile(std::span<const double> sorted_samples, double p) -> double {
  auto n = static_cast<int>(std::ssize(sorted_samples));
  if (n == 0) { return std::numeric_limits<double>::quiet_NaN(); }
  if (n == 1) { return sorted_samples[0]; }
  auto h = (n - 1) * std::clamp(p, 0.0, 1.0);
  auto lo = static_cast<int>(std::floor(h));
  auto hi = std::min(lo + 1, n - 1);
  auto frac = h - lo;
  return sorted_samples[lo] + frac * (sorted_samples[hi] - sorted_samples[lo]);
}

auto equal_tailed_interval(std::span<const double> sorted_samples, double mass) -> Credible_interval {
  check_mass(mass);
  if (sorted_samples.empty()) {
    return {+std::numeric_limits<double>::quiet_NaN(), -std::numeric_limits<double>::quiet_NaN()};
  }
  return {sample_quantile(sorted_samples, 0.5 * (1.0 - mass)),
          sample_quantile(sorted_samples, 0.5 * (1.0 + mass))};
}

auto shortest_credible_interval(std::span<const double> sorted_samples, double mass)
    -> Credible_interval {
  check_mass(mass);
  auto n = static_cast<int>(std::ssize(sorted_samples));
  if (n == 0) {
    return {+std::numeric_limits<double>::quiet_NaN(), -std::numeric_limits<double>::quiet_NaN()};
  }
  auto k = num_samples_in_interval(n, mass);
  auto best_i = 0;
  auto best_width = std::numeric_limits<double>::infinity();
  for (auto i = 0; i + k <= n; ++i) {
    auto width = sorted_samples[i + k - 1] - sorted_samples[i];
    if (width < best_width) {
      best_width = width;
      best_i = i;
    }
  }
  return {sorted_samples[best_i], sorted_samples[best_i + k - 1]};
}

auto highest_density_region(std::span<const double> sorted_samples, double mass, int num_bins)
    -> Highest_density_region {
  check_mass(mass);
  auto result = Highest_density_region{};
  auto n = static_cast<int>(std::ssize(sorted_samples));
  if (n == 0) { return result; }

  if (num_bins <= 0) {
    num_bins = std::clamp(static_cast<int>(std::lround(std::sqrt(static_cast<double>(n)))), 10, 100);
  }
  result.num_bins = num_bins;

  auto lo = sorted_samples.front();
  auto hi = sorted_samples.back();
  if (not (hi > lo)) {  // Constant sample: one component, no gap
    result.num_components = 1;
    result.components.push_back({lo, hi});
    return result;
  }
  auto interval = shortest_credible_interval(sorted_samples, mass);

  auto bin_width = (hi - lo) / num_bins;
  auto counts = std::vector<int>(num_bins, 0);
  for (auto x : sorted_samples) {
    auto b = static_cast<int>((x - lo) / bin_width);
    counts[std::clamp(b, 0, num_bins - 1)] += 1;
  }

  // 1-2-1 smoothing, used only to rank the bins: a single bin of Poisson noise in a tail cannot
  // outrank a genuinely dense region, and cannot become a mode of its own.
  auto smoothed = std::vector<double>(num_bins, 0.0);
  for (auto b = 0; b != num_bins; ++b) {
    auto left = counts[b == 0 ? 0 : b - 1];
    auto right = counts[b == num_bins - 1 ? num_bins - 1 : b + 1];
    smoothed[b] = 0.25 * left + 0.5 * counts[b] + 0.25 * right;
  }

  auto order = std::vector<int>(num_bins);
  std::iota(begin(order), end(order), 0);
  // Highest smoothed density first; ties by bin index, so the result does not depend on the sort.
  std::ranges::sort(order, [&](int a, int b) {
    return smoothed[a] != smoothed[b] ? smoothed[a] > smoothed[b] : a < b; });

  auto target = num_samples_in_interval(n, mass);
  auto selected = std::vector<char>(num_bins, 0);
  auto so_far = 0;
  for (auto b : order) {
    if (so_far >= target) { break; }
    selected[b] = 1;
    so_far += counts[b];
  }

  // Maximal runs of selected bins, bridging gaps of at most k_max_bridged_gap_bins unselected bins.
  struct Run { int first; int last; int mass; };
  auto runs = std::vector<Run>{};
  for (auto b = 0; b != num_bins; ++b) {
    if (not selected[b]) { continue; }
    auto last = b;
    auto run_mass = counts[b];
    while (true) {
      auto next = last + 1;
      auto gap = 0;
      while (next != num_bins && not selected[next] && gap < k_max_bridged_gap_bins) {
        ++next;
        ++gap;
      }
      if (next == num_bins || not selected[next]) { break; }
      for (auto k = last + 1; k <= next; ++k) { run_mass += counts[k]; }
      last = next;
    }
    runs.push_back({b, last, run_mass});
    b = last;
  }

  auto minor_cutoff = k_minor_component_fraction * target;
  for (const auto& r : runs) {
    if (r.mass < minor_cutoff) {
      ++result.num_minor_components;
      continue;
    }
    result.components.push_back({lo + r.first * bin_width, lo + (r.last + 1) * bin_width});
  }
  result.num_components = static_cast<int>(std::ssize(result.components));
  if (result.num_components == 0 && not runs.empty()) {
    // Every run was minor: report the largest one so that the result is never empty.
    auto largest = std::ranges::max_element(runs, {}, [](const Run& r) { return r.mass; });
    result.components.push_back({lo + largest->first * bin_width, lo + (largest->last + 1) * bin_width});
    result.num_components = 1;
    --result.num_minor_components;
  }

  // How much of the shortest contiguous interval falls between two major components.  Only the
  // interior gaps count: the ends of the outermost components are bin edges, and the small mismatch
  // between a bin edge and the sample value at the end of the interval is a grid artifact, not a gap.
  if (interval.width() > 0.0 && std::ssize(result.components) > 1) {
    auto gap = 0.0;
    for (auto i = 1; i != std::ssize(result.components); ++i) {
      auto a = std::max(result.components[i - 1].hi, interval.lo);
      auto b = std::min(result.components[i].lo, interval.hi);
      if (b > a) { gap += b - a; }
    }
    result.interval_gap_fraction = std::clamp(gap / interval.width(), 0.0, 1.0);
  }
  return result;
}

auto calc_moments(std::span<const double> samples) -> Moments {
  auto result = Moments{};
  result.n = static_cast<int>(std::ssize(samples));
  if (result.n == 0) {
    result.mean = std::numeric_limits<double>::quiet_NaN();
    result.stddev = std::numeric_limits<double>::quiet_NaN();
    return result;
  }
  auto sum = 0.0;
  for (auto x : samples) { sum += x; }
  result.mean = sum / result.n;
  if (result.n < 2) { return result; }
  auto sum_sq = 0.0;
  for (auto x : samples) {
    auto d = x - result.mean;
    sum_sq += d * d;
  }
  result.stddev = std::sqrt(sum_sq / (result.n - 1));
  return result;
}

auto effective_sample_size(std::span<const double> samples, int max_lag) -> Ess_result {
  auto result = Ess_result{};
  auto n = static_cast<int>(std::ssize(samples));
  if (n < 4) { return result; }

  auto moments = calc_moments(samples);
  auto var0 = 0.0;
  for (auto x : samples) {
    auto d = x - moments.mean;
    var0 += d * d;
  }
  var0 /= n;
  if (not (var0 > 0.0)) {  // Constant sample: every draw carries the same information
    result.ess = n;
    result.tau = 1.0;
    return result;
  }

  if (max_lag <= 0) { max_lag = std::min(n - 1, 5000); }
  max_lag = std::min(max_lag, n - 1);

  auto autocorr = [&](int lag) {
    auto acc = 0.0;
    for (auto i = 0; i + lag != n; ++i) {
      acc += (samples[i] - moments.mean) * (samples[i + lag] - moments.mean);
    }
    return (acc / n) / var0;
  };

  auto sum_gamma = 0.0;
  auto prev_gamma = std::numeric_limits<double>::infinity();
  auto terminated = false;
  auto lag = 0;
  for (; lag + 1 <= max_lag; lag += 2) {
    auto gamma = autocorr(lag) + autocorr(lag + 1);
    if (gamma < 0.0) { terminated = true; break; }
    gamma = std::min(gamma, prev_gamma);  // Initial *monotone* positive sequence
    prev_gamma = gamma;
    sum_gamma += gamma;
  }
  result.num_lags_used = lag;
  result.truncated = not terminated;
  result.tau = std::max(-1.0 + 2.0 * sum_gamma, 1.0);
  result.ess = n / result.tau;
  return result;
}

namespace {

// Splits every chain in two halves of equal length and trims all halves to the shortest one.
auto make_half_chains(std::span<const std::vector<double>> chains) -> std::vector<std::vector<double>> {
  auto halves = std::vector<std::vector<double>>{};
  auto m = std::numeric_limits<int>::max();
  for (const auto& c : chains) {
    auto h = static_cast<int>(std::ssize(c)) / 2;
    m = std::min(m, h);
  }
  if (chains.empty() || m < 2) { return halves; }
  for (const auto& c : chains) {
    auto h = static_cast<int>(std::ssize(c)) / 2;
    halves.emplace_back(c.begin(), c.begin() + m);
    halves.emplace_back(c.begin() + h, c.begin() + h + m);
  }
  return halves;
}

auto rhat_of_half_chains(const std::vector<std::vector<double>>& halves) -> double {
  if (std::ssize(halves) < 2) { return -1.0; }
  auto m = static_cast<int>(std::ssize(halves.front()));
  auto num_halves = static_cast<int>(std::ssize(halves));
  auto means = std::vector<double>{};
  auto within = 0.0;
  for (const auto& h : halves) {
    auto mom = calc_moments(h);
    means.push_back(mom.mean);
    within += mom.stddev * mom.stddev;
  }
  within /= num_halves;
  auto mean_of_means = calc_moments(means);
  auto between = m * mean_of_means.stddev * mean_of_means.stddev;
  if (not (within > 0.0)) { return between > 0.0 ? std::numeric_limits<double>::infinity() : 1.0; }
  return std::sqrt(((m - 1.0) / m * within + between / m) / within);
}

}  // namespace

auto split_rhat(std::span<const std::vector<double>> chains) -> double {
  auto halves = make_half_chains(chains);
  return rhat_of_half_chains(halves);
}

auto rank_normalized_split_rhat(std::span<const std::vector<double>> chains) -> double {
  auto halves = make_half_chains(chains);
  if (std::ssize(halves) < 2) { return -1.0; }

  // Pool every draw, rank it (average rank over ties), then take its normal score.
  struct Entry { double value; int half; int index; };
  auto pooled = std::vector<Entry>{};
  for (auto h = 0; h != std::ssize(halves); ++h) {
    for (auto i = 0; i != std::ssize(halves[h]); ++i) {
      pooled.push_back({halves[h][i], static_cast<int>(h), static_cast<int>(i)});
    }
  }
  std::ranges::sort(pooled, [](const Entry& a, const Entry& b) { return a.value < b.value; });

  auto S = static_cast<double>(std::ssize(pooled));
  auto normal = boost::math::normal_distribution<double>{};
  auto scored = halves;  // Same shape
  for (auto i = 0; i != std::ssize(pooled);) {
    auto j = i;
    while (j != std::ssize(pooled) && pooled[j].value == pooled[i].value) { ++j; }
    auto average_rank = 0.5 * ((i + 1) + j);  // Ranks are 1-based
    // Blom's plotting position, as in Vehtari et al. (2021)
    auto z = boost::math::quantile(normal, (average_rank - 0.375) / (S + 0.25));
    for (auto k = i; k != j; ++k) { scored[pooled[k].half][pooled[k].index] = z; }
    i = j;
  }
  return rhat_of_half_chains(scored);
}

}  // namespace delphy
