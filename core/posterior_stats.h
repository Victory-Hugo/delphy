#ifndef DELPHY_POSTERIOR_STATS_H_
#define DELPHY_POSTERIOR_STATS_H_

#include <span>
#include <vector>

namespace delphy {

// Summary statistics of MCMC samples of a scalar quantity.
//
// Everything here works on plain arrays of doubles and knows nothing about trees, so it can be
// checked against independent reference calculations.  See doc/extended/clade_summary.md
// for the definitions and the conventions chosen where more than one is in use.

// A closed interval of the sampled quantity.  `lo > hi` means "not defined for this sample".
struct Credible_interval {
  double lo;
  double hi;

  auto width() const -> double { return hi - lo; }
  auto is_defined() const -> bool { return lo <= hi; }
};

// Type 7 sample quantile (the default of R's quantile() and of numpy.quantile): linear interpolation
// between the order statistics at (n - 1) * p.  `sorted_samples` must be sorted ascending.
auto sample_quantile(std::span<const double> sorted_samples, double p) -> double;

// Equal-tailed interval: the quantiles at (1 - mass) / 2 and (1 + mass) / 2.  This is *not* a highest
// posterior density interval and is never reported as one.
auto equal_tailed_interval(std::span<const double> sorted_samples, double mass) -> Credible_interval;

// Shortest contiguous interval spanned by k = ceil(mass * n) of the n samples, i.e. the shortest
// *interval* that contains at least `mass` of the sampled mass.  Computed exactly for the empirical
// distribution.
//
// This is NOT the same object as a highest posterior density region.  An HPD region is the set
// { x : p(x) >= c } with the smallest c that carries `mass`; it is an interval only when the posterior
// is unimodal, and only then does the shortest contiguous interval estimate it.  For a multimodal
// posterior the shortest interval bridges the low-density gap between the modes and is wider than the
// HPD region, which is disconnected.  Use highest_density_region() to get that set, and never call
// this interval an HPD interval without checking that the region has one component.
auto shortest_credible_interval(std::span<const double> sorted_samples, double mass)
    -> Credible_interval;

// Histogram estimate of the highest density region (HDR) at the given mass -- the set the term "HPD"
// actually names: the bins with the largest counts, taken in order, until their total count reaches
// ceil(mass * n), reported as the intervals spanned by maximal runs of adjacent selected bins.  One
// component means the HDR is an interval and the shortest contiguous interval estimates it; more than
// one means the posterior is multimodal and no single interval describes it.
//
// This is a diagnostic, not an estimator: the number of components depends on the bin count, which is
// reported alongside it.
struct Highest_density_region {
  int num_bins = 0;
  int num_components = 0;        // Major components only; > 1 is the multimodality signal
  int num_minor_components = 0;  // Components holding less than k_minor_component_fraction of the mass
  std::vector<Credible_interval> components;  // Major components, in increasing order
  // Fraction of the width of the shortest contiguous interval that falls in a gap *between* two major
  // components, i.e. the part of that interval which the sample says is a region of low density.
  // Exactly 0 for a single component, so it never reports a grid artifact as a gap.
  double interval_gap_fraction = 0.0;
};

// A histogram of n samples has Poisson noise of order sqrt(count) per bin, which on its own breaks the
// tails of a perfectly unimodal sample into isolated bins.  Three rules keep that noise out of the
// multimodality signal, and all three are stated here rather than buried in the code:
//
//   * bins are ranked by a 1-2-1 smoothed count, so a single noisy bin cannot become its own mode;
//   * runs of selected bins separated by at most k_max_bridged_gap_bins unselected bins are merged;
//   * a component holding less than k_minor_component_fraction of the selected mass is counted as
//     minor and does not make the sample multimodal.
inline constexpr int k_max_bridged_gap_bins = 2;
inline constexpr double k_minor_component_fraction = 0.05;

// `num_bins <= 0` uses round(sqrt(n)), clamped to [10, 100].
auto highest_density_region(std::span<const double> sorted_samples, double mass, int num_bins = 0)
    -> Highest_density_region;

// Effective sample size by Geyer's initial monotone positive sequence, as used by Tracer:
//
//   ESS = n / tau,  tau = -1 + 2 * sum_m Gamma_m
//
// where Gamma_m = rho_{2m} + rho_{2m+1} is truncated at the first negative term and made
// non-increasing.  A constant sample returns n.  Fewer than 4 samples returns 0.
//
// Autocovariances are computed directly, up to `max_lag` lags (0 picks a default).  If the positive
// sequence has not terminated by then, the sum is truncated there and `truncated` is set: the returned
// ESS is then an upper bound.
struct Ess_result {
  double ess = 0.0;
  double tau = 1.0;  // Integrated autocorrelation time, >= 1
  int num_lags_used = 0;
  bool truncated = false;
};
auto effective_sample_size(std::span<const double> samples, int max_lag = 0) -> Ess_result;

// Split R-hat over one or more chains: every chain is split in two halves and the usual
// between-over-within variance ratio is formed over the 2 * num_chains half-chains.  Returns a
// negative value when there are too few samples to compute it (fewer than 2 samples per half).
//
// `rank_normalized` first replaces every draw by the normal score of its rank among the pooled draws
// of all half-chains (Vehtari et al. 2021), which makes the statistic insensitive to the tails and
// well defined for distributions without finite variance.  Both are reported: the rank-normalized one
// is the one to judge convergence by.
auto split_rhat(std::span<const std::vector<double>> chains) -> double;
auto rank_normalized_split_rhat(std::span<const std::vector<double>> chains) -> double;

// Mean, standard deviation (sample, n - 1 denominator) and median of unsorted samples.
struct Moments {
  int n = 0;
  double mean = 0.0;
  double stddev = 0.0;
};
auto calc_moments(std::span<const double> samples) -> Moments;

}  // namespace delphy

#endif  // DELPHY_POSTERIOR_STATS_H_
