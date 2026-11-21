#ifndef DELPHY_EVO_SUBST_MODEL_H_
#define DELPHY_EVO_SUBST_MODEL_H_

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "evo_model.h"
#include "sequence.h"

namespace delphy {

// A four-state, time-homogeneous, reversible nucleotide substitution model.
//
// Six symmetric exchangeabilities s_ab = s_ba > 0 and stationary frequencies pi_a > 0 summing to one
// give the unnormalized rate matrix
//
//   Qt_ab = s_ab pi_b  (a != b),   Qt_aa = -sum_{b != a} Qt_ab,
//
// which is normalized by c = -sum_a pi_a Qt_aa so that Q = Qt / c has mean substitution rate
// -sum_a pi_a Q_aa = 1.  The strict clock rate mu then carries the whole absolute rate scale, and the
// relative site rates nu_l only describe heterogeneity.  Every named nucleotide model in this family
// is a constraint on (s, pi) rather than a separate likelihood.

// The six exchangeabilities in the fixed order A-C, A-G, A-T, C-G, C-T, G-T of IQ-TREE's six-digit
// model codes.
inline constexpr int k_num_exchangeabilities = 6;

struct Exchangeability_pair {
  Real_seq_letter a;
  Real_seq_letter b;
};

auto exchangeability_pairs() -> std::span<const Exchangeability_pair>;

// Name of exchangeability `index`, e.g. "AG" for index 1.
auto exchangeability_name(int index) -> std::string_view;

// The index of the exchangeability whose rate is held at one, i.e. G-T.
inline constexpr int k_reference_exchangeability = 5;

// A partition of the six exchangeabilities into equality classes, held in canonical form: the class
// of exchangeability 0 is 0, and the class of exchangeability i is at most one more than the largest
// class index among the exchangeabilities before it.  Canonical codes are exactly the restricted
// growth strings of length six, of which there are Bell(6) = 203.  Two codes that differ only in
// which digits name which class, e.g. "101101" and "010010", denote the same grouping and
// canonicalize to the same code.
class Exchangeability_grouping {
 public:
  // The single-class grouping "000000" (JC69 / F81).
  Exchangeability_grouping();

  // Reads a six-digit code.  Throws std::invalid_argument with an explanation on a code of the wrong
  // length or with a character that is not a digit.
  static auto from_code(std::string_view code) -> Exchangeability_grouping;

  // Canonical six-digit code of this grouping.
  auto code() const -> std::string;

  auto num_classes() const -> int { return num_classes_; }

  // The class of exchangeability `index`, in [0, num_classes()).
  auto class_of(int index) const -> int;

  // The class whose rate is held at one: the class of G-T.
  auto reference_class() const -> int { return class_of(k_reference_exchangeability); }

  // Number of free relative exchangeabilities, i.e. num_classes() - 1.
  auto num_free_rates() const -> int { return num_classes_ - 1; }

  // Indices of the exchangeabilities in class `k`, in increasing order.
  auto members_of(int k) const -> std::vector<int>;

  auto operator==(const Exchangeability_grouping& that) const -> bool {
    return class_of_ == that.class_of_;
  }

 private:
  std::array<int, k_num_exchangeabilities> class_of_{};
  int num_classes_{1};
};

auto operator<<(std::ostream& os, const Exchangeability_grouping& g) -> std::ostream&;

// All 203 canonical groupings, in the order the restricted growth strings are generated.
auto all_exchangeability_groupings() -> std::span<const Exchangeability_grouping>;

// How the stationary frequencies are determined.  `equal` holds pi at (1/4, 1/4, 1/4, 1/4),
// `fixed` holds pi at a user-supplied vector, and `estimated` samples pi in the posterior.  IQ-TREE's
// `+FO` optimizes the frequencies by maximum likelihood; `estimated` here is a Bayesian posterior
// over pi under an explicit prior, which is a different statement about the same parameters.
enum class Frequency_mode { equal, fixed, estimated };

auto to_string(Frequency_mode mode) -> std::string_view;
auto frequency_mode_from_string(std::string_view name) -> Frequency_mode;

struct Subst_model {
  double mu{0.0};
  Exchangeability_grouping grouping{};
  // One relative rate per class of `grouping`; the reference class is held at 1.0.
  std::vector<double> class_rate{1.0};
  Frequency_mode frequency_mode{Frequency_mode::equal};
  Seq_vector<double> pi_a{1.0 / k_num_real_seq_letters};

  // The exchangeability of index `index`, i.e. the rate of its class.
  auto exchangeability(int index) const -> double;

  auto derive_site_evo_model() const -> Site_evo_model;

  // Resets `class_rate` to all ones for the current grouping.
  auto reset_class_rates() -> void;

  // Throws std::invalid_argument if the rates or the frequencies are out of their support, if the
  // reference class is not held at one, or if the number of rates does not match the grouping.
  auto validate() const -> void;
};

// Builds the rate matrix from a symmetric exchangeability matrix with zero diagonal.  The arithmetic
// is the one `Hky_model::derive_site_evo_model` has always used, so an HKY exchangeability matrix
// reproduces the historical matrix bit for bit.
auto derive_site_evo_model_from_exchangeabilities(
    double mu,
    const Seq_matrix<double>& r_ab,
    const Seq_vector<double>& pi_a)
    -> Site_evo_model;

// A named model of IQ-TREE's table: a grouping together with a constraint on the frequencies.  The
// same six-digit code appears under several names because the frequency constraint differs.
struct Subst_model_alias {
  std::string_view name;
  std::string_view code;
  Frequency_mode frequencies;
};

auto subst_model_aliases() -> std::span<const Subst_model_alias>;

// Looks up an alias by name, ignoring case.  Returns nullptr if there is no such name.
auto find_subst_model_alias(std::string_view name) -> const Subst_model_alias*;

// Reads a model specification: either a name from the alias table, or a six-digit code, either of
// them optionally followed by "+FQ" (equal frequencies) or "+FO" (estimated frequencies).  Throws
// std::invalid_argument with an explanation on anything else.
auto parse_subst_model(std::string_view spec) -> Subst_model;

// A one-line summary of the free relative exchangeabilities for progress output, e.g.
// "kappa = 3.21, " for HKY85 or "rate(AG) = 3.21, rate(CT) = 2.90, " for TN93.
auto subst_model_summary(const Subst_model& model) -> std::string;

// The names of the aliases that denote `grouping` with `mode`, for reporting; empty if none does.
auto subst_model_names_of(const Exchangeability_grouping& grouping, Frequency_mode mode)
    -> std::vector<std::string_view>;

}  // namespace delphy

#endif // DELPHY_EVO_SUBST_MODEL_H_
