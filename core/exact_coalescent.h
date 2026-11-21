#ifndef DELPHY_EXACT_COALESCENT_H_
#define DELPHY_EXACT_COALESCENT_H_

#include <span>
#include <vector>

#include "coalescent_grid.h"
#include "pop_model.h"

namespace delphy {

// Event-driven evaluation of the exact Kingman coalescent density of a genealogy,
//
//   log P_exact = - sum_{coalescences i} log N(t_i) - sum_j C(k_j, 2) int_{a_j}^{b_j} dt / N(t),
//
// where the intervals [a_j, b_j) are delimited by the genealogy's own events -- the sampling times
// and the coalescence times -- and k_j is the number of lineages alive over the interval.  This is
// the same decomposition BEAST's Skygrid likelihood uses: the real event times, subdivided at the
// population curve's knots so that each piece lies inside one interval of the curve.
//
// Delphy's samplers do not evaluate this density; they evaluate the cell prior of
// `Scalable_coalescent_prior` on a fixed grid of equal-width cells, whose difference from the exact
// density is measured by `calc_coalescent_grid_excess` and is a functional of the genealogy
// (`coalescent_grid.h`).  The exact Kingman correction of `Run` makes the exact density the target
// anyway: the local moves run against the cell prior and a whole batch of them is accepted or
// rejected on the change of log P_exact - log P_cells, while the population moves use the exact
// density directly.  Those population moves are what this class is for.
//
// Two properties make it cheap enough to sit inside an HMC trajectory.
//
// 1. The decomposition depends on the genealogy alone.  A population move leaves the genealogy
//    fixed, so the intervals and their weights are built once per move and reused for every curve
//    the trajectory visits.  The cost of one density or gradient evaluation is then linear in the
//    number of intervals, which is the number of nodes plus the number of knots, and is independent
//    of the time span and of the cell count of the coalescent grid.
//
// 2. Because every piece lies inside one interval of the Skygrid curve, the piece's intensity
//    integral depends on at most two knot values -- one for the staircase variant, the two ends of
//    the interval for the log-linear variant -- so the whole gradient is accumulated in one pass.
//
// For the staircase variant the density collapses onto a finite sufficient statistic per Skygrid
// interval, exactly as in Gill et al (2012) and in BEAST:
//
//   log P_exact = - sum_k c_k gamma_k - sum_k SS_k exp(-gamma_k),
//
//   c_k  = number of coalescences in interval k,
//   SS_k = sum over the pieces of interval k of C(k_j, 2) (b_j - a_j).
//
// `calc_staircase_sufficient_statistics` returns those statistics.  It is an independent reference
// for the general path above, not a second production path: the samplers always go through the
// general path, so there is one implementation of the mathematics in the sampler.
class Exact_coalescent_events {
 public:
  // One interval between consecutive events of the genealogy, possibly subdivided at a knot.
  struct Interval {
    double a = 0.0;       // Lower time bound, in days
    double b = 0.0;       // Upper time bound, in days
    double weight = 0.0;  // C(k, 2) = k (k - 1) / 2 of the lineages alive over [a, b)
    int knot_interval = 0;  // `Skygrid_pop_model::interval_containing_t(b)`, or 0 without knots
  };

  // Intervals delimited by the genealogy's events only.  Good enough for any population model whose
  // `intensity_integral` is exact over an arbitrary range, which is every model Delphy has.
  explicit Exact_coalescent_events(const Coalescent_event_times& events);

  // The same, with every interval further subdivided at the knot times `x`, which must be strictly
  // increasing.  Needed for the Skygrid gradient, which is a sum over pieces of one interval each.
  Exact_coalescent_events(const Coalescent_event_times& events, std::span<const double> x);

  auto intervals() const -> const std::vector<Interval>& { return intervals_; }
  auto coalescence_times() const -> const std::vector<double>& { return coalescence_times_; }

  // A := sum_j C(k_j, 2) int_{a_j}^{b_j} dt / N(t) >= 0, the integral term of -log P_exact.
  auto calc_intensity_term(const Pop_model& pop_model) const -> double;
  // S := sum_{coalescences i} log N(t_i), the other term of -log P_exact.
  auto calc_log_pop_term(const Pop_model& pop_model) const -> double;
  // log P_exact = -A - S.
  auto calc_log_prior(const Pop_model& pop_model) const -> double {
    return -calc_intensity_term(pop_model) - calc_log_pop_term(pop_model);
  }

  // Adds d(log P_exact) / d(gamma_k) to `out[k]` for every knot k of `pop_model`, which must have
  // been constructed with that model's knot times.  `out` has M+1 entries.
  auto add_d_log_prior_d_gamma(const Skygrid_pop_model& pop_model, std::span<double> out) const -> void;

  // Sufficient statistics of the staircase variant, indexed by knot 0..M: `ss[k]` and `c[k]` of the
  // formula above, with interval 0 (before the first knot) folded into k = 0 and interval M+1
  // (after the last knot) into k = M, as the curve's own extrapolation rule requires.
  struct Staircase_sufficient_statistics {
    std::vector<double> ss;  // SS_k
    std::vector<double> c;   // c_k
  };
  auto calc_staircase_sufficient_statistics(const Skygrid_pop_model& pop_model) const
      -> Staircase_sufficient_statistics;

 private:
  std::vector<Interval> intervals_;
  std::vector<double> coalescence_times_;
  std::vector<double> x_;  // Knot times the intervals were cut at; empty when there were none

  auto build(const Coalescent_event_times& events, std::span<const double> x) -> void;
};

}  // namespace delphy

#endif  // DELPHY_EXACT_COALESCENT_H_
