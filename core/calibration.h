#ifndef DELPHY_CALIBRATION_H_
#define DELPHY_CALIBRATION_H_

#include <iosfwd>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "phylo_tree.h"

namespace delphy {

// Prior densities on the age of the most recent common ancestor of a named set of samples.
//
// A calibration is a term of the posterior that Delphy samples: it enters every Metropolis-Hastings
// ratio of every move that can change the age of a calibrated node.  Rescaling the times of a
// finished tree is not a calibration and is not what this does.
//
// A calibration is defined by a stable set of sample IDs, never by a node index.  See
// doc/extended/calibration.md for the mathematical design, the operator coverage
// audit and the conventions below.

// Thrown for every rejected calibration configuration.  what() lists the problems, one per line.
struct Calibration_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Age conventions.  There is no default: every calibration names one.
//
//   ka         Thousands of years of age before `age_reference_year` (1 ka = 1000 * 365 days)
//   years_bp   Years of age before `age_reference_year`
//   cal_bp     Calibrated radiocarbon years of age; the reference is fixed at 1950 CE
//   ce         Calendar year CE (>= 1); larger values are younger
//   bce        Calendar year BCE (>= 1; 1 BCE is astronomical year 0); larger values are older
enum class Age_units { ka, years_bp, cal_bp, ce, bce };

auto to_string(Age_units units) -> std::string_view;
auto parse_age_units(std::string_view name) -> Age_units;  // throws Calibration_error

enum class Calibration_distribution { normal, lognormal, uniform };

auto to_string(Calibration_distribution distribution) -> std::string_view;
auto parse_calibration_distribution(std::string_view name) -> Calibration_distribution;  // throws

// A density on a node's age, in years before the calibration's reference instant.  All parameters are
// already converted to years of age, so this struct carries no unit ambiguity.
//
// `oldest_age` / `youngest_age` are the support of a `uniform` prior and hard truncation bounds of a
// `normal` or `lognormal` one.  Outside the support, log_density() is -infinity.  Truncation is not
// renormalized: the normalizer is independent of the state and so cancels in every MH ratio (see the
// design document).
struct Age_prior {
  Calibration_distribution distribution = Calibration_distribution::normal;

  double mean = 0.0;      // `normal`: mean age, in years
  double stddev = 0.0;    // `normal`: standard deviation, in years, > 0
  double median = 0.0;    // `lognormal`: median age, in years, > 0
  double sigma = 0.0;     // `lognormal`: standard deviation of log(age), dimensionless, > 0

  double oldest_age = +std::numeric_limits<double>::infinity();
  double youngest_age = -std::numeric_limits<double>::infinity();

  // log f(age), with age in years.  -infinity outside the support.
  auto log_density(double age_years) const -> double;

  // Support of the density, in years of age, after hard bounds.
  auto support_youngest_age() const -> double;
  auto support_oldest_age() const -> double;
};

struct Calibration {
  std::string name;

  // Sorted, unique sample IDs.  Empty exactly when `all_tips` is set.
  std::vector<std::string> taxa;
  bool all_tips = false;

  // With `true` (the default) the sample set must be a clade: a tree in which it is not has density zero.  With
  // `false` the prior is on the age of the MRCA of the set, whatever else lies below that MRCA.
  bool require_monophyly = true;

  Age_units units = Age_units::ka;
  std::optional<int64_t> reference_year;  // Present exactly for `ka` and `years_bp`

  // Days since 2020-01-01 of the instant that ages are measured back from.
  double reference_t = 0.0;

  Age_prior prior;

  int config_line = 0;  // 1-based line of the `calibration` keyword, for error messages

  // Node age in years, given the node's time in days since 2020-01-01.
  auto age_of(double t) const -> double { return (reference_t - t) / 365.0; }
  // Inverse of age_of().
  auto time_of_age(double age_years) const -> double { return reference_t - 365.0 * age_years; }

  auto log_density_at(double t) const -> double { return prior.log_density(age_of(t)); }

  // Closed interval of node times allowed by the hard support, in days since 2020-01-01.
  auto t_min() const -> double { return time_of_age(prior.support_oldest_age()); }
  auto t_max() const -> double { return time_of_age(prior.support_youngest_age()); }
};

struct Calibration_set {
  std::vector<Calibration> calibrations;

  auto empty() const -> bool { return calibrations.empty(); }
  auto size() const -> int { return static_cast<int>(std::ssize(calibrations)); }
};

// Parses a calibration configuration file.  Throws Calibration_error listing every problem found.
// Format (see doc/extended/calibration.md and the example configuration):
//
//   version 1
//
//   calibration root
//     taxa all
//     age_units ka
//     age_reference_year 1950
//     distribution lognormal
//     median 140
//     sigma 0.2
//   end
//
// `#` starts a comment, blank lines are ignored, keys are separated from values by whitespace.
// `taxa` may be repeated and lists sample IDs; `taxa all` means every tip of the tree.  The optional key
// `monophyly` is `required` (the default) or `not_required`; see Calibration::require_monophyly.
auto read_calibration_config(std::istream& is) -> Calibration_set;

// Canonical text of a configuration: re-reading this gives the same Calibration_set.  Used to store
// the configuration inside .dphy files.
auto write_calibration_config(const Calibration_set& set, std::ostream& os) -> void;
auto calibration_config_to_string(const Calibration_set& set) -> std::string;

// Resolves every calibration to the node index of the MRCA of its taxon set and checks everything
// that depends on the tree: taxon membership, monophyly (for the calibrations that require it), distinct
// targets (between calibrations that require monophyly), mutual compatibility of
// the age constraints, and that the tree's current node times are inside every hard support.
//
// Throws Calibration_error listing every problem found.  The returned vector is parallel to
// `set.calibrations`.
auto bind_calibrations(const Calibration_set& set, const Phylo_tree& tree) -> std::vector<Node_index>;

// Same, without the requirement that the current node times lie inside the hard supports.  Used when
// the times are about to be set (and by the tests).
auto bind_calibrations_ignoring_times(const Calibration_set& set, const Phylo_tree& tree)
    -> std::vector<Node_index>;

// sum_c log f_c(age_c(t_node(c))) over all calibrations.
auto calc_log_calibration_prior(
    const Calibration_set& set,
    const std::vector<Node_index>& nodes,
    const Phylo_tree& tree)
    -> double;

}  // namespace delphy

#endif  // DELPHY_CALIBRATION_H_
