// safe_gamma_math.h

#ifndef DELPHY_SAFE_GAMMA_MATH_H_
#define DELPHY_SAFE_GAMMA_MATH_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "absl/log/check.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"

#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/gamma.hpp>

namespace delphy {

// On Emscripten/WASM, `long double` is declared as IEEE binary128 (128-bit, max
// exponent ~11356), but the underlying math functions (`expl`, `logl`, `powl`,
// etc.) are thin wrappers around the `double` versions, which overflow/underflow
// at ~710.  Boost's `gamma_q`/`gamma_q_inv` promote `double` arguments to
// `long double` by default, then rely on those math functions to handle the full
// range of the promoted type.  This causes intermediate calculations to overflow
// to `inf` or underflow to 0, resulting in `nan` (e.g., `inf * 0`).
//
// The `promote_double<false>` policy keeps the entire computation in `double`
// precision.  Boost's `double` code path uses `lanczos13m53` (53-bit precision) with
// overflow guards that handle all parameter ranges correctly, matching the actual
// capabilities of the math library on all platforms (x86-64, ARM64, WASM).
//
// This is a *loss of precision*, not correctness: the difference between
// `double` and `long double` results is < 1e-12 relative error.

namespace internal {

using gamma_policy = boost::math::policies::policy<
    boost::math::policies::promote_double<false>>;

}  // namespace internal

// Returns the regularized upper incomplete gamma function Q(a, x) = Γ(a, x) / Γ(a)
// with the `no-promote` policy to avoid Emscripten/WASM overflow issues.
// Q(a, x) = P(X > x) where X ~ Gamma(a, 1).
inline auto safe_gamma_q(double a, double x) -> double {
  auto result = boost::math::gamma_q(a, x, internal::gamma_policy{});
  CHECK(not std::isnan(result)) << "safe_gamma_q(" << a << ", " << x << ") returned nan";
  CHECK(not std::isinf(result)) << "safe_gamma_q(" << a << ", " << x << ") returned inf";
  return result;
}

// Returns the inverse regularized upper incomplete gamma function:
// x such that Q(a, x) = Q, with the `no-promote` policy.
// Note: gamma_q_inv(a, 0) = +inf is a valid boundary case.
// We handle Q=0 explicitly because Boost's gamma_q_inv throws an overflow error.
inline auto safe_gamma_q_inv(double a, double Q) -> double {
  CHECK_GE(Q, 0.0) << "safe_gamma_q_inv: Q must be >= 0, got " << Q;
  CHECK_LE(Q, 1.0) << "safe_gamma_q_inv: Q must be <= 1, got " << Q;

  // Handle Q=0 explicitly: gamma_q_inv(a, 0) = +inf.
  // Boost's gamma_q_inv throws on Q=0, so we short-circuit it.
  if (Q == 0.0) {
    return std::numeric_limits<double>::infinity();
  }

  double result;
  try {
    result = boost::math::gamma_q_inv(a, Q, internal::gamma_policy{});
  } catch (const std::exception& e) {
    std::cerr << "gamma_q_inv threw: " << e.what()
              << "\n  a=" << a << ", Q=" << Q << "\n";
    throw;
  }
  CHECK(not std::isnan(result)) << "safe_gamma_q_inv(" << a << ", " << Q << ") returned nan";
  CHECK(not std::isinf(result)) << "safe_gamma_q_inv(" << a << ", " << Q << ") returned inf";
  return result;
}

// Returns log(P(x_min < X < x_max)) where X ~ Gamma(shape=a, rate=1).
// This is log(Q(a, x_min) - Q(a, x_max)).
// Requires x_min < x_max (otherwise the result is -infinity or invalid).
//
// For x_max well below the bulk of the distribution, Q(a, x_min) and Q(a, x_max) are both
// close to 1 and their difference cancels catastrophically (even to exactly 0).  In that regime
// the difference is evaluated in log space from the lower-gamma series
//   P(a, x) = x^a e^{-x} / Gamma(a+1) * sum_{k>=0} x^k / ((a+1)...(a+k)),
// which converges for x < a + 1 and does not underflow for the integrals of interest.
namespace internal {

inline auto log_gamma_p_series(double a, double x) -> double {
  auto term = 1.0;
  auto sum = 1.0;
  for (auto k = 1; k < 10000; ++k) {
    term *= x / (a + k);
    sum += term;
    if (term < 1e-17 * sum) { break; }
  }
  return a * std::log(x) - x - std::lgamma(a + 1.0) + std::log(sum);
}

}  // namespace internal

inline auto safe_log_gamma_integral(double a, double x_min, double x_max) -> double {
  CHECK_LT(x_min, x_max) << "safe_log_gamma_integral: x_min must be < x_max, got "
                          << x_min << " >= " << x_max;
  auto Q_hi = safe_gamma_q(a, x_min);
  auto Q_lo = safe_gamma_q(a, x_max);
  if (Q_hi - Q_lo < 1e-3 * Q_hi && x_max < a + 1.0) {
    auto log_p_hi = internal::log_gamma_p_series(a, x_max);
    if (x_min <= 0.0) { return log_p_hi; }
    auto log_p_lo = internal::log_gamma_p_series(a, x_min);
    return log_p_hi + std::log1p(-std::exp(log_p_lo - log_p_hi));
  }
  CHECK_GE(Q_hi, Q_lo) << "safe_log_gamma_integral: Q(a, x_min) >= Q(a, x_max) violated: "
                       << Q_hi << " < " << Q_lo;
  return std::log(Q_hi - Q_lo);
}

// Samples from Gamma(shape=alpha, rate=beta) truncated to [lo, hi].
// Uses inverse-CDF sampling via the upper incomplete gamma function.
//
// Algorithm:
//   Let Y ~ Gamma(alpha, 1) (unit-rate Gamma).
//   We want X = Y / beta in [lo, hi], i.e., Y in [beta*lo, beta*hi].
//   P(Y > y) = Q(alpha, y), so:
//     - The truncation interval for Y is [y_lo, y_hi] = [beta*lo, beta*hi]
//     - Q(y_lo) = P(Y > y_lo), Q(y_hi) = P(Y > y_hi)
//     - Since Q is decreasing, Q_hi := Q(y_lo) > Q_lo := Q(y_hi)
//     - Sample Q ~ Uniform(Q_lo, Q_hi)
//     - Then Y = Q^{-1}(alpha, Q) ~ Gamma(alpha, 1) truncated to [y_lo, y_hi]
//     - Return X = Y / beta
//
// We use `IntervalOpenClosed` for Q to avoid Q = Q_lo ≈ 0 (when hi is
// large or infinite), which would make `gamma_q_inv` return `inf`.  At
// the upper end, Q = Q_hi is safe since `gamma_q_inv(a, Q_hi)` returns
// y_lo, giving X = lo.
inline auto safe_sample_truncated_gamma(
    double alpha, double beta, double lo, double hi, absl::BitGenRef bitgen) -> double {
  CHECK_GT(alpha, 0.0) << "safe_sample_truncated_gamma: alpha must be > 0";
  CHECK_GT(beta, 0.0) << "safe_sample_truncated_gamma: beta must be > 0";
  CHECK_LT(lo, hi) << "safe_sample_truncated_gamma: lo must be < hi";

  // Transform to unit-rate Gamma(alpha, 1)
  auto y_lo = beta * lo;
  auto y_hi = beta * hi;

  // Q is decreasing, so Q_hi = Q(y_lo) > Q_lo = Q(y_hi)
  auto Q_hi = safe_gamma_q(alpha, y_lo);
  auto Q_lo = safe_gamma_q(alpha, y_hi);

  if (Q_hi - Q_lo < 1e-3 * Q_hi && y_hi < alpha + 1.0) {
    // Both Q values are close to 1, so inverting Q would lose all precision.  Invert the lower
    // incomplete gamma function in log space instead, by bisection.
    auto log_p_hi = internal::log_gamma_p_series(alpha, y_hi);
    auto log_p_lo = (y_lo > 0.0) ? internal::log_gamma_p_series(alpha, y_lo)
                                 : -std::numeric_limits<double>::infinity();
    auto u = absl::Uniform(absl::IntervalClosedClosed, bitgen, 0.0, 1.0);
    auto log_target = log_p_hi + std::log((1.0 - u) * std::exp(log_p_lo - log_p_hi) + u);
    auto a = y_lo;
    auto b = y_hi;
    for (auto i = 0; i < 200 && b - a > 1e-15 * b; ++i) {
      auto mid = 0.5 * (a + b);
      if (internal::log_gamma_p_series(alpha, mid) < log_target) { a = mid; } else { b = mid; }
    }
    return std::clamp(0.5 * (a + b) / beta, lo, hi);
  }

  CHECK_LT(Q_lo, Q_hi) << "safe_sample_truncated_gamma: Q range is degenerate: "
                       << "Q_lo=" << Q_lo << " >= Q_hi=" << Q_hi;

  // Sample Q uniformly in (Q_lo, Q_hi]
  auto rand_Q = absl::Uniform(absl::IntervalOpenClosed, bitgen, Q_lo, Q_hi);

  // Invert: Y = Q^{-1}(alpha, rand_Q) ~ Gamma(alpha, 1) in [y_lo, y_hi]
  auto y = safe_gamma_q_inv(alpha, rand_Q);

  // Transform back to Gamma(alpha, beta)
  auto x = y / beta;
  CHECK_GE(x, lo - 1e-6) << "safe_sample_truncated_gamma: sample x " << x << " < lo " << lo;
  CHECK_LE(x, hi + 1e-6) << "safe_sample_truncated_gamma: sample x " << x << " > hi " << hi;
  x = std::clamp(x, lo, hi);
  return x;
}

}  // namespace delphy

#endif  // DELPHY_SAFE_GAMMA_MATH_H_
