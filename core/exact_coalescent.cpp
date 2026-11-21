#include "exact_coalescent.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <absl/log/check.h>
#include <absl/strings/str_format.h>

#include "estd.h"

namespace delphy {

namespace {

// `Skygrid_pop_model::interval_containing_t`, evaluated on a bare knot array so that the intervals
// can be built before any population model is in hand.  Interval 0 is t <= x_0, interval k is
// x_{k-1} < t <= x_k for 1 <= k <= M, and interval M+1 is x_M < t.
auto interval_containing(std::span<const double> x, double t) -> int {
  auto it = std::ranges::lower_bound(x, t);
  if (it == std::ranges::begin(x)) { return 0; }
  if (it == std::ranges::end(x)) { return static_cast<int>(std::ssize(x)); }  // M+1
  return static_cast<int>(std::ranges::distance(std::ranges::begin(x), it));
}

// The knots whose value the population curve reads in interval `kk`.  At most two, so the gradient
// over all knots is accumulated in one pass over the intervals.
struct Relevant_knots {
  int lo;  // first knot, always valid
  int hi;  // second knot, or `lo` when only one enters
};

auto relevant_knots_of(const Skygrid_pop_model& pop_model, int kk) -> Relevant_knots {
  auto M = pop_model.M();
  if (kk <= 0) { return {0, 0}; }
  if (kk > M) { return {M, M}; }
  switch (pop_model.type()) {
    case Skygrid_pop_model::Type::k_staircase:
      return {kk, kk};
    case Skygrid_pop_model::Type::k_log_linear:
      return {kk - 1, kk};
    default:
      CHECK(false) << "unrecognized type " << static_cast<int>(pop_model.type());
  }
}

}  // namespace

Exact_coalescent_events::Exact_coalescent_events(const Coalescent_event_times& events) {
  build(events, {});
}

Exact_coalescent_events::Exact_coalescent_events(
    const Coalescent_event_times& events,
    std::span<const double> x) {
  for (auto i = 1; i < std::ssize(x); ++i) {
    if (not (x[i-1] < x[i])) {
      throw std::invalid_argument(absl::StrFormat(
          "Exact_coalescent_events needs strictly increasing knot times, but x[%d] = %g >= x[%d] = %g",
          i-1, x[i-1], i, x[i]));
    }
  }
  build(events, x);
}

auto Exact_coalescent_events::build(
    const Coalescent_event_times& events,
    std::span<const double> x)
    -> void {

  x_.assign(std::ranges::begin(x), std::ranges::end(x));

  auto changes = merged_lineage_changes(events);
  intervals_.reserve(std::size(changes) + std::size(x));
  coalescence_times_.reserve(std::size(events.coalescence_times));

  // Above the root there is one lineage and nothing to integrate, so the first interval starts at
  // the first event.  The weight of [prev, t) is C(k, 2) with k the count after every change at or
  // before `prev`, which is the convention of `calc_exact_coalescent_log_prior`.
  auto k = 1;
  auto prev = 0.0;
  auto have_prev = false;
  for (const auto& change : changes) {
    if (have_prev && k >= 2 && prev < change.t) {
      auto weight = 0.5 * static_cast<double>(k) * (k - 1);

      // Subdivide at every knot strictly inside (prev, change.t), so that each piece lies inside
      // one interval of the population curve and its intensity integral depends on at most two
      // knot values.
      auto lo = prev;
      auto first_knot = std::ranges::upper_bound(x, prev);
      for (auto it = first_knot; it != std::ranges::end(x) && *it < change.t; ++it) {
        intervals_.push_back({.a = lo, .b = *it, .weight = weight,
                              .knot_interval = interval_containing(x, *it)});
        lo = *it;
      }
      intervals_.push_back({.a = lo, .b = change.t, .weight = weight,
                            .knot_interval = interval_containing(x, change.t)});
    }
    prev = change.t;
    have_prev = true;
    k += change.delta_k;
    if (change.delta_k > 0) { coalescence_times_.push_back(change.t); }
  }
}

auto Exact_coalescent_events::calc_intensity_term(const Pop_model& pop_model) const -> double {
  auto result = 0.0;
  for (const auto& interval : intervals_) {
    result += interval.weight * pop_model.intensity_integral(interval.a, interval.b);
  }
  return result;
}

auto Exact_coalescent_events::calc_log_pop_term(const Pop_model& pop_model) const -> double {
  auto result = 0.0;
  for (auto t : coalescence_times_) {
    result += std::log(pop_model.pop_at_time(t));
  }
  return result;
}

auto Exact_coalescent_events::add_d_log_prior_d_gamma(
    const Skygrid_pop_model& pop_model,
    std::span<double> out) const
    -> void {

  auto M = pop_model.M();
  CHECK_EQ(std::ssize(out), M + 1);
  CHECK(x_ == pop_model.x())
      << "Exact_coalescent_events was built for different knot times than the population curve's";

  // d(-A)/d gamma_k = - sum_j w_j I_j d(log I_j)/d gamma_k, with I_j the intensity integral of
  // piece j.  Only the knots the piece's interval reads contribute.
  for (const auto& interval : intervals_) {
    auto intensity = pop_model.intensity_integral(interval.a, interval.b);
    if (intensity == 0.0) { continue; }
    auto [lo, hi] = relevant_knots_of(pop_model, interval.knot_interval);
    auto d_lo = pop_model.d_log_intensity_integral_d_gamma(interval.a, interval.b, lo);
    out[lo] -= interval.weight * intensity * d_lo;
    if (hi != lo) {
      auto d_hi = pop_model.d_log_intensity_integral_d_gamma(interval.a, interval.b, hi);
      out[hi] -= interval.weight * intensity * d_hi;
      if (estd::is_debug_enabled) {
        // Shifting the whole curve up by delta divides the intensity integral by exp(delta), so the
        // derivatives of its logarithm over the knots that enter it must sum to -1.
        CHECK(std::abs(d_lo + d_hi + 1.0) < 1e-8) << d_lo << " + " << d_hi << " != -1";
      }
    } else if (estd::is_debug_enabled) {
      CHECK(std::abs(d_lo + 1.0) < 1e-8) << d_lo << " != -1";
    }
  }

  // d(-S)/d gamma_k = - sum_i d(log N(t_i))/d gamma_k.
  for (auto t : coalescence_times_) {
    auto [lo, hi] = relevant_knots_of(pop_model, interval_containing(x_, t));
    out[lo] -= pop_model.d_log_N_d_gamma(t, lo);
    if (hi != lo) { out[hi] -= pop_model.d_log_N_d_gamma(t, hi); }
  }
}

auto Exact_coalescent_events::calc_staircase_sufficient_statistics(
    const Skygrid_pop_model& pop_model) const
    -> Staircase_sufficient_statistics {

  if (pop_model.type() != Skygrid_pop_model::Type::k_staircase) {
    throw std::invalid_argument(
        "Sufficient statistics of this form exist only for the staircase Skygrid variant; the "
        "log-linear variant's intensity integral is not a function of one knot value");
  }
  CHECK(x_ == pop_model.x())
      << "Exact_coalescent_events was built for different knot times than the population curve's";

  auto M = pop_model.M();
  auto result = Staircase_sufficient_statistics{
      .ss = std::vector<double>(M + 1, 0.0), .c = std::vector<double>(M + 1, 0.0)};
  for (const auto& interval : intervals_) {
    auto k = std::clamp(interval.knot_interval, 0, M);
    result.ss[k] += interval.weight * (interval.b - interval.a);
  }
  for (auto t : coalescence_times_) {
    auto k = std::clamp(interval_containing(x_, t), 0, M);
    result.c[k] += 1.0;
  }
  return result;
}

}  // namespace delphy
