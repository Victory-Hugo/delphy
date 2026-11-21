#include "evo_subst_model.h"

#include <algorithm>
#include <optional>
#include <ostream>
#include <stdexcept>

#include <Eigen/Dense>

#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

namespace delphy {

namespace {

using enum Real_seq_letter;

constexpr auto k_pairs = std::array<Exchangeability_pair, k_num_exchangeabilities>{{
    {A, C}, {A, G}, {A, T}, {C, G}, {C, T}, {G, T}}};

constexpr auto k_pair_names = std::array<std::string_view, k_num_exchangeabilities>{
    "AC", "AG", "AT", "CG", "CT", "GT"};

// IQ-TREE's table of named DNA models.  The code is the canonical six-digit grouping and the
// frequency entry is the constraint the name carries.  `fixed` never appears here: a name with fixed
// frequencies is the `equal` entry with the frequencies supplied separately.
constexpr auto k_aliases = std::array<Subst_model_alias, 22>{{
    {"JC",    "000000", Frequency_mode::equal},
    {"JC69",  "000000", Frequency_mode::equal},
    {"F81",   "000000", Frequency_mode::estimated},
    {"K80",   "010010", Frequency_mode::equal},
    {"K2P",   "010010", Frequency_mode::equal},
    {"HKY",   "010010", Frequency_mode::estimated},
    {"HKY85", "010010", Frequency_mode::estimated},
    {"TN",    "010020", Frequency_mode::estimated},
    {"TN93",  "010020", Frequency_mode::estimated},
    {"TNe",   "010020", Frequency_mode::equal},
    {"K81",   "012210", Frequency_mode::equal},
    {"K3P",   "012210", Frequency_mode::equal},
    {"K81u",  "012210", Frequency_mode::estimated},
    {"TPM2",  "010212", Frequency_mode::equal},
    {"TPM2u", "010212", Frequency_mode::estimated},
    {"TPM3",  "012012", Frequency_mode::equal},
    {"TPM3u", "012012", Frequency_mode::estimated},
    {"TIM",   "012230", Frequency_mode::estimated},
    {"TIMe",  "012230", Frequency_mode::equal},
    {"TVM",   "012314", Frequency_mode::estimated},
    {"TVMe",  "012314", Frequency_mode::equal},
    {"GTR",   "012345", Frequency_mode::estimated}}};

// The remaining names of the table, kept separate only to stay inside one initializer.
constexpr auto k_more_aliases = std::array<Subst_model_alias, 5>{{
    {"TIM2",  "010232", Frequency_mode::estimated},
    {"TIM2e", "010232", Frequency_mode::equal},
    {"TIM3",  "012032", Frequency_mode::estimated},
    {"TIM3e", "012032", Frequency_mode::equal},
    {"SYM",   "012345", Frequency_mode::equal}}};

auto build_alias_table() -> const std::vector<Subst_model_alias>& {
  static const auto table = [] {
    auto out = std::vector<Subst_model_alias>{};
    out.insert(out.end(), k_aliases.begin(), k_aliases.end());
    out.insert(out.end(), k_more_aliases.begin(), k_more_aliases.end());
    return out;
  }();
  return table;
}

auto build_all_groupings() -> const std::vector<Exchangeability_grouping>& {
  static const auto all = [] {
    // Every canonical code is a restricted growth string: a[0] = 0 and a[i] <= 1 + max(a[0..i-1]).
    auto out = std::vector<Exchangeability_grouping>{};
    auto digits = std::array<int, k_num_exchangeabilities>{};
    auto recurse = [&](auto&& self, int index, int next_class) -> void {
      if (index == k_num_exchangeabilities) {
        auto code = std::string{};
        for (auto d : digits) { code += static_cast<char>('0' + d); }
        out.push_back(Exchangeability_grouping::from_code(code));
        return;
      }
      for (auto k = 0; k <= next_class; ++k) {
        digits[index] = k;
        self(self, index + 1, std::max(next_class, k + 1));
      }
    };
    recurse(recurse, 0, 0);
    return out;
  }();
  return all;
}

}  // namespace

auto exchangeability_pairs() -> std::span<const Exchangeability_pair> { return k_pairs; }

auto exchangeability_name(int index) -> std::string_view {
  if (index < 0 || index >= k_num_exchangeabilities) {
    throw std::invalid_argument(absl::StrFormat(
        "Exchangeability index %d is out of range: it must be between 0 and %d",
        index, k_num_exchangeabilities - 1));
  }
  return k_pair_names[index];
}

Exchangeability_grouping::Exchangeability_grouping() : class_of_{}, num_classes_{1} {}

auto Exchangeability_grouping::from_code(std::string_view code) -> Exchangeability_grouping {
  if (std::ssize(code) != k_num_exchangeabilities) {
    throw std::invalid_argument(absl::StrFormat(
        "A substitution model code must have exactly %d digits, one per exchangeability in the order "
        "%s; '%s' has %d characters",
        k_num_exchangeabilities, absl::StrJoin(k_pair_names, ", "), code, std::ssize(code)));
  }
  auto first_seen = std::array<int, 10>{};
  first_seen.fill(-1);
  auto result = Exchangeability_grouping{};
  auto next_class = 0;
  for (auto i = 0; i != k_num_exchangeabilities; ++i) {
    auto c = code[i];
    if (c < '0' || c > '9') {
      throw std::invalid_argument(absl::StrFormat(
          "A substitution model code must consist of digits; '%s' has '%c' at position %d",
          code, c, i + 1));
    }
    auto digit = c - '0';
    if (first_seen[digit] < 0) {
      first_seen[digit] = next_class;
      ++next_class;
    }
    result.class_of_[i] = first_seen[digit];
  }
  result.num_classes_ = next_class;
  return result;
}

auto Exchangeability_grouping::code() const -> std::string {
  auto out = std::string{};
  for (auto k : class_of_) { out += static_cast<char>('0' + k); }
  return out;
}

auto Exchangeability_grouping::class_of(int index) const -> int {
  if (index < 0 || index >= k_num_exchangeabilities) {
    throw std::invalid_argument(absl::StrFormat(
        "Exchangeability index %d is out of range: it must be between 0 and %d",
        index, k_num_exchangeabilities - 1));
  }
  return class_of_[index];
}

auto Exchangeability_grouping::members_of(int k) const -> std::vector<int> {
  auto out = std::vector<int>{};
  for (auto i = 0; i != k_num_exchangeabilities; ++i) {
    if (class_of_[i] == k) { out.push_back(i); }
  }
  return out;
}

auto operator<<(std::ostream& os, const Exchangeability_grouping& g) -> std::ostream& {
  return os << g.code();
}

auto all_exchangeability_groupings() -> std::span<const Exchangeability_grouping> {
  return build_all_groupings();
}

auto to_string(Frequency_mode mode) -> std::string_view {
  switch (mode) {
    case Frequency_mode::equal: return "equal";
    case Frequency_mode::fixed: return "fixed";
    case Frequency_mode::estimated: return "estimated";
  }
  throw std::invalid_argument("Unknown frequency mode");
}

auto frequency_mode_from_string(std::string_view name) -> Frequency_mode {
  auto lower = absl::AsciiStrToLower(name);
  if (lower == "equal") { return Frequency_mode::equal; }
  if (lower == "fixed") { return Frequency_mode::fixed; }
  if (lower == "estimated") { return Frequency_mode::estimated; }
  throw std::invalid_argument(absl::StrFormat(
      "Unknown frequency mode '%s': it must be 'equal', 'fixed' or 'estimated'", name));
}

auto Subst_model::exchangeability(int index) const -> double {
  auto k = grouping.class_of(index);
  if (k >= std::ssize(class_rate)) {
    throw std::invalid_argument(absl::StrFormat(
        "The model code %s has %d classes but only %d relative rates were given",
        grouping.code(), grouping.num_classes(), std::ssize(class_rate)));
  }
  return class_rate[k];
}

auto Subst_model::reset_class_rates() -> void {
  class_rate.assign(grouping.num_classes(), 1.0);
}

auto Subst_model::validate() const -> void {
  if (std::ssize(class_rate) != grouping.num_classes()) {
    throw std::invalid_argument(absl::StrFormat(
        "The model code %s has %d exchangeability classes, so it needs %d relative rates, but %d "
        "were given",
        grouping.code(), grouping.num_classes(), grouping.num_classes(), std::ssize(class_rate)));
  }
  for (auto k = 0; k != std::ssize(class_rate); ++k) {
    if (not (class_rate[k] > 0.0) || not std::isfinite(class_rate[k])) {
      throw std::invalid_argument(absl::StrFormat(
          "Relative exchangeability %d of the model code %s is %g; every exchangeability must be a "
          "finite positive number",
          k, grouping.code(), class_rate[k]));
    }
  }
  if (class_rate[grouping.reference_class()] != 1.0) {
    throw std::invalid_argument(absl::StrFormat(
        "The exchangeability class of G-T carries the relative scale and must be held at 1, but "
        "class %d of the model code %s is %g",
        grouping.reference_class(), grouping.code(), class_rate[grouping.reference_class()]));
  }
  auto sum = 0.0;
  for (auto a : k_all_real_seq_letters) {
    if (not (pi_a[a] > 0.0) || not (pi_a[a] < 1.0)) {
      throw std::invalid_argument(absl::StrFormat(
          "The stationary frequency of %c is %g; every frequency must lie strictly between 0 and 1",
          to_char(a), pi_a[a]));
    }
    sum += pi_a[a];
  }
  if (std::abs(sum - 1.0) > 1e-9) {
    throw std::invalid_argument(absl::StrFormat(
        "The stationary frequencies sum to %.12g; they must sum to 1", sum));
  }
  if (not (mu >= 0.0) || not std::isfinite(mu)) {
    throw std::invalid_argument(absl::StrFormat(
        "The substitution rate is %g; it must be a finite non-negative number", mu));
  }
}

auto derive_site_evo_model_from_exchangeabilities(
    double mu,
    const Seq_matrix<double>& r_ab,
    const Seq_vector<double>& pi_a)
    -> Site_evo_model {

  auto r = Eigen::Matrix4d{
      { r_ab[A][A], r_ab[A][C], r_ab[A][G], r_ab[A][T] },
      { r_ab[C][A], r_ab[C][C], r_ab[C][G], r_ab[C][T] },
      { r_ab[G][A], r_ab[G][C], r_ab[G][G], r_ab[G][T] },
      { r_ab[T][A], r_ab[T][C], r_ab[T][G], r_ab[T][T] }
  };
  auto real_pi = Eigen::Vector4d{ pi_a[A], pi_a[C], pi_a[G], pi_a[T] };

  auto R = real_pi.transpose() * r * real_pi;
  auto q = Eigen::Matrix4d{};
  for (auto a = 0; a != k_num_real_seq_letters; ++a) {
    q(a, a) = 0.0;
    for (auto b = 0; b != k_num_real_seq_letters; ++b) {
      if (a != b) {
        q(a, b) = r(a, b) / R * real_pi(b);
        q(a, a) -= q(a, b);
      }
    }
  };

  auto q_ab = Seq_matrix<double>{};
  for (auto a : k_all_real_seq_letters) {
    for (auto b : k_all_real_seq_letters) {
      q_ab[a][b] = q(index_of(a), index_of(b));
    }
  }

  return Site_evo_model{.mu = mu, .pi_a = pi_a, .q_ab = q_ab};
}

auto Subst_model::derive_site_evo_model() const -> Site_evo_model {
  auto r_ab = Seq_matrix<double>{0.0};
  for (auto i = 0; i != k_num_exchangeabilities; ++i) {
    const auto& [a, b] = k_pairs[i];
    auto value = exchangeability(i);
    r_ab[a][b] = value;
    r_ab[b][a] = value;
  }
  return derive_site_evo_model_from_exchangeabilities(mu, r_ab, pi_a);
}

auto subst_model_aliases() -> std::span<const Subst_model_alias> { return build_alias_table(); }

auto find_subst_model_alias(std::string_view name) -> const Subst_model_alias* {
  auto lower = absl::AsciiStrToLower(name);
  for (const auto& alias : build_alias_table()) {
    if (absl::AsciiStrToLower(alias.name) == lower) { return &alias; }
  }
  return nullptr;
}

auto subst_model_summary(const Subst_model& model) -> std::string {
  const auto& grouping = model.grouping;
  if (grouping == Exchangeability_grouping::from_code("010010")) {
    return absl::StrFormat("kappa = %.3g, ", model.class_rate[grouping.class_of(1)]);
  }
  auto out = std::string{};
  for (auto k = 0; k != grouping.num_classes(); ++k) {
    if (k == grouping.reference_class()) { continue; }
    auto members = std::string{};
    for (auto index : grouping.members_of(k)) {
      if (not members.empty()) { members += '/'; }
      members += k_pair_names[index];
    }
    out += absl::StrFormat("rate(%s) = %.3g, ", members, model.class_rate[k]);
  }
  return out;
}

auto subst_model_names_of(const Exchangeability_grouping& grouping, Frequency_mode mode)
    -> std::vector<std::string_view> {
  auto out = std::vector<std::string_view>{};
  for (const auto& alias : build_alias_table()) {
    if (alias.frequencies == mode &&
        Exchangeability_grouping::from_code(alias.code) == grouping) {
      out.push_back(alias.name);
    }
  }
  return out;
}

auto parse_subst_model(std::string_view spec) -> Subst_model {
  if (spec.empty()) {
    throw std::invalid_argument(
        "An empty substitution model specification: give a model name such as HKY85 or GTR, or a "
        "six-digit exchangeability code such as 010010, optionally followed by '+FQ' for equal "
        "frequencies or '+FO' for estimated frequencies");
  }
  auto base = spec;
  auto frequency_override = std::optional<Frequency_mode>{};
  if (auto plus = spec.find('+'); plus != std::string_view::npos) {
    base = spec.substr(0, plus);
    auto suffix = absl::AsciiStrToUpper(spec.substr(plus));
    if (suffix == "+FQ") {
      frequency_override = Frequency_mode::equal;
    } else if (suffix == "+FO") {
      frequency_override = Frequency_mode::estimated;
    } else {
      throw std::invalid_argument(absl::StrFormat(
          "Unknown model suffix '%s' in '%s': only '+FQ' (equal frequencies) and '+FO' (estimated "
          "frequencies) are understood.  Site rate heterogeneity is configured separately",
          spec.substr(plus), spec));
    }
  }

  auto result = Subst_model{};
  if (const auto* alias = find_subst_model_alias(base); alias != nullptr) {
    result.grouping = Exchangeability_grouping::from_code(alias->code);
    result.frequency_mode = alias->frequencies;
  } else if (auto digits = std::ranges::count_if(base, [](char c) { return c >= '0' && c <= '9'; });
             digits == std::ssize(base) ||
             (digits > 0 && std::ssize(base) == k_num_exchangeabilities)) {
    // A string of digits, and anything of the right length that contains a digit, is meant as a code;
    // from_code then explains exactly what is wrong with it.
    result.grouping = Exchangeability_grouping::from_code(base);
    result.frequency_mode = Frequency_mode::estimated;
  } else {
    throw std::invalid_argument(absl::StrFormat(
        "Unknown substitution model '%s': give a model name (one of %s) or a six-digit "
        "exchangeability code such as 010010, optionally followed by '+FQ' or '+FO'",
        base, absl::StrJoin(build_alias_table(), ", ",
                            [](std::string* out, const Subst_model_alias& a) { *out += a.name; })));
  }
  if (frequency_override.has_value()) { result.frequency_mode = *frequency_override; }
  result.reset_class_rates();
  if (result.frequency_mode == Frequency_mode::equal) {
    result.pi_a = Seq_vector<double>{1.0 / k_num_real_seq_letters};
  }
  return result;
}

}  // namespace delphy
