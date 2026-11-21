#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

#include "evo_hky.h"
#include "evo_subst_model.h"

namespace delphy {

namespace {

using enum Real_seq_letter;

using Mat = std::array<std::array<long double, 4>, 4>;

auto mat_mul(const Mat& x, const Mat& y) -> Mat {
  auto out = Mat{};
  for (auto i = 0; i != 4; ++i)
    for (auto j = 0; j != 4; ++j)
      for (auto k = 0; k != 4; ++k) out[i][j] += x[i][k] * y[k][j];
  return out;
}

// exp(m) by scaling and squaring of a 30-term Taylor series in long double; shares no code with Delphy.
auto mat_exp(const Mat& m) -> Mat {
  auto norm = 0.0L;
  for (const auto& row : m) for (auto v : row) norm = std::max(norm, std::fabs(v));
  auto squarings = 0;
  while (norm > 0.1L) { norm /= 2; ++squarings; }
  auto scaled = m;
  for (auto& row : scaled) for (auto& v : row) v /= std::pow(2.0L, squarings);
  auto term = Mat{};
  auto sum = Mat{};
  for (auto i = 0; i != 4; ++i) { term[i][i] = 1.0L; sum[i][i] = 1.0L; }
  for (auto n = 1; n != 31; ++n) {
    term = mat_mul(term, scaled);
    for (auto i = 0; i != 4; ++i)
      for (auto j = 0; j != 4; ++j) { term[i][j] /= n; sum[i][j] += term[i][j]; }
  }
  for (auto s = 0; s != squarings; ++s) sum = mat_mul(sum, sum);
  return sum;
}

auto transition_matrix(const Site_evo_model& evo, double t) -> Mat {
  auto m = Mat{};
  for (auto a : k_all_real_seq_letters)
    for (auto b : k_all_real_seq_letters) m[index_of(a)][index_of(b)] = evo.q_ab[a][b] * evo.mu * t;
  return mat_exp(m);
}

auto make_model(std::string_view code, std::vector<double> rates, std::array<double, 4> pi,
                double mu = 1.0) -> Subst_model {
  auto model = Subst_model{};
  model.mu = mu;
  model.grouping = Exchangeability_grouping::from_code(code);
  model.class_rate = std::move(rates);
  model.frequency_mode = Frequency_mode::estimated;
  model.pi_a[A] = pi[0];
  model.pi_a[C] = pi[1];
  model.pi_a[G] = pi[2];
  model.pi_a[T] = pi[3];
  return model;
}

// Random positive rates with the reference class held at one, and random frequencies.
auto random_model(const Exchangeability_grouping& grouping, std::mt19937& engine) -> Subst_model {
  auto model = Subst_model{};
  model.mu = 1.0;
  model.grouping = grouping;
  model.frequency_mode = Frequency_mode::estimated;
  auto rate = std::uniform_real_distribution<double>{0.1, 6.0};
  model.class_rate.assign(grouping.num_classes(), 1.0);
  for (auto k = 0; k != grouping.num_classes(); ++k) {
    if (k != grouping.reference_class()) { model.class_rate[k] = rate(engine); }
  }
  auto weight = std::uniform_real_distribution<double>{0.5, 2.0};
  auto total = 0.0;
  for (auto a : k_all_real_seq_letters) { model.pi_a[a] = weight(engine); total += model.pi_a[a]; }
  for (auto a : k_all_real_seq_letters) { model.pi_a[a] /= total; }
  return model;
}

}  // namespace

TEST(Exchangeability_grouping_test, there_are_exactly_203_canonical_groupings) {
  auto all = all_exchangeability_groupings();
  EXPECT_EQ(std::ssize(all), 203);  // Bell(6)

  auto codes = std::set<std::string>{};
  for (const auto& g : all) {
    codes.insert(g.code());
    // Canonical form: a restricted growth string.
    auto high = -1;
    for (auto i = 0; i != k_num_exchangeabilities; ++i) {
      EXPECT_LE(g.class_of(i), high + 1) << g.code();
      high = std::max(high, g.class_of(i));
    }
    EXPECT_EQ(g.num_classes(), high + 1) << g.code();
    EXPECT_EQ(g.num_free_rates(), g.num_classes() - 1) << g.code();
  }
  EXPECT_EQ(std::ssize(codes), 203) << "the groupings must be pairwise distinct";
}

TEST(Exchangeability_grouping_test, every_grouping_of_six_elements_canonicalizes_into_the_table) {
  // All 10^6 strings over the digits 0-8 of length six would be slow; the 46656 strings over six
  // digits already cover every partition of the six exchangeabilities.
  auto codes = std::set<std::string>{};
  auto digits = std::array<char, k_num_exchangeabilities>{};
  auto recurse = [&](auto&& self, int index) -> void {
    if (index == k_num_exchangeabilities) {
      codes.insert(Exchangeability_grouping::from_code(
          std::string_view{digits.data(), digits.size()}).code());
      return;
    }
    for (auto d = '0'; d <= '5'; ++d) { digits[index] = d; self(self, index + 1); }
  };
  recurse(recurse, 0);
  EXPECT_EQ(std::ssize(codes), 203);

  auto table = std::set<std::string>{};
  for (const auto& g : all_exchangeability_groupings()) { table.insert(g.code()); }
  EXPECT_EQ(codes, table);
}

TEST(Exchangeability_grouping_test, codes_that_differ_only_in_the_digit_names_are_the_same_grouping) {
  EXPECT_EQ(Exchangeability_grouping::from_code("101101").code(), "010010");
  EXPECT_EQ(Exchangeability_grouping::from_code("010010").code(), "010010");
  EXPECT_EQ(Exchangeability_grouping::from_code("989898").code(), "010101");
  EXPECT_EQ(Exchangeability_grouping::from_code("543210").code(), "012345");
  EXPECT_EQ(Exchangeability_grouping::from_code("111111").code(), "000000");
  EXPECT_TRUE(Exchangeability_grouping::from_code("101101") ==
              Exchangeability_grouping::from_code("010010"));
}

TEST(Exchangeability_grouping_test, invalid_codes_are_refused_with_an_explanation) {
  EXPECT_THROW(Exchangeability_grouping::from_code("01001"), std::invalid_argument);
  EXPECT_THROW(Exchangeability_grouping::from_code("0100100"), std::invalid_argument);
  EXPECT_THROW(Exchangeability_grouping::from_code(""), std::invalid_argument);
  EXPECT_THROW(Exchangeability_grouping::from_code("01x010"), std::invalid_argument);
  EXPECT_THROW(Exchangeability_grouping::from_code("01-010"), std::invalid_argument);
  try {
    Exchangeability_grouping::from_code("0100");
    FAIL() << "expected a refusal";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr("exactly 6 digits"));
    EXPECT_THAT(std::string{e.what()}, testing::HasSubstr("AC, AG, AT, CG, CT, GT"));
  }
}

TEST(Exchangeability_grouping_test, the_reference_class_is_the_class_of_G_T) {
  EXPECT_EQ(exchangeability_name(k_reference_exchangeability), "GT");
  for (const auto& g : all_exchangeability_groupings()) {
    EXPECT_EQ(g.reference_class(), g.class_of(5)) << g.code();
  }
  EXPECT_EQ(Exchangeability_grouping::from_code("012345").reference_class(), 5);
  EXPECT_EQ(Exchangeability_grouping::from_code("010010").reference_class(), 0);
  EXPECT_EQ(Exchangeability_grouping::from_code("000000").reference_class(), 0);
}

TEST(Subst_model_test, every_grouping_gives_a_valid_reversible_normalized_rate_matrix) {
  auto engine = std::mt19937{20260101};
  for (const auto& grouping : all_exchangeability_groupings()) {
    for (auto repeat = 0; repeat != 3; ++repeat) {
      auto model = random_model(grouping, engine);
      ASSERT_NO_THROW(model.validate()) << grouping.code();
      auto evo = model.derive_site_evo_model();

      for (auto a : k_all_real_seq_letters) {
        auto row = 0.0;
        for (auto b : k_all_real_seq_letters) {
          if (a != b) {
            EXPECT_GT(evo.q_ab[a][b], 0.0) << grouping.code() << " " << a << b;
          }
          row += evo.q_ab[a][b];
        }
        EXPECT_NEAR(row, 0.0, 1e-12) << "row sum " << grouping.code();
        EXPECT_LT(evo.q_ab[a][a], 0.0) << grouping.code();
      }

      // Detailed balance: pi_a Q_ab == pi_b Q_ba.
      for (auto a : k_all_real_seq_letters) {
        for (auto b : k_all_real_seq_letters) {
          EXPECT_NEAR(model.pi_a[a] * evo.q_ab[a][b], model.pi_a[b] * evo.q_ab[b][a], 1e-13)
              << grouping.code() << " " << a << b;
        }
      }

      // Stationarity: pi^T Q == 0.
      for (auto b : k_all_real_seq_letters) {
        auto column = 0.0;
        for (auto a : k_all_real_seq_letters) { column += model.pi_a[a] * evo.q_ab[a][b]; }
        EXPECT_NEAR(column, 0.0, 1e-13) << grouping.code();
      }

      // Mean substitution rate one: -sum_a pi_a Q_aa == 1.
      auto mean_rate = 0.0;
      for (auto a : k_all_real_seq_letters) { mean_rate -= model.pi_a[a] * evo.q_ab[a][a]; }
      EXPECT_NEAR(mean_rate, 1.0, 1e-13) << grouping.code();
    }
  }
}

TEST(Subst_model_test, the_exchangeabilities_of_a_class_are_equal_and_G_T_is_one) {
  auto engine = std::mt19937{777};
  for (const auto& grouping : all_exchangeability_groupings()) {
    auto model = random_model(grouping, engine);
    EXPECT_EQ(model.exchangeability(k_reference_exchangeability), 1.0) << grouping.code();
    for (auto i = 0; i != k_num_exchangeabilities; ++i) {
      for (auto j = 0; j != k_num_exchangeabilities; ++j) {
        if (grouping.class_of(i) == grouping.class_of(j)) {
          EXPECT_EQ(model.exchangeability(i), model.exchangeability(j)) << grouping.code();
        }
      }
    }
    // Q_ab / (s_ab pi_b) is the same constant for every off-diagonal entry.
    auto evo = model.derive_site_evo_model();
    auto scale = 0.0;
    for (auto i = 0; i != k_num_exchangeabilities; ++i) {
      auto [a, b] = exchangeability_pairs()[i];
      auto here = evo.q_ab[a][b] / (model.exchangeability(i) * model.pi_a[b]);
      if (scale == 0.0) { scale = here; }
      EXPECT_NEAR(here, scale, 1e-12 * scale) << grouping.code();
      auto back = evo.q_ab[b][a] / (model.exchangeability(i) * model.pi_a[a]);
      EXPECT_NEAR(back, scale, 1e-12 * scale) << grouping.code();
    }
  }
}

TEST(Subst_model_test, HKY85_reproduces_the_historical_HKY_matrix_bit_for_bit) {
  for (auto kappa : {0.5, 1.0, 2.0, 3.7, 10.0}) {
    for (auto pi : std::vector<std::array<double, 4>>{
             {0.25, 0.25, 0.25, 0.25}, {0.3, 0.2, 0.2, 0.3}, {0.1, 0.4, 0.35, 0.15}}) {
      auto hky = Hky_model{};
      hky.mu = 1.5e-5;
      hky.kappa = kappa;
      hky.pi_a[A] = pi[0]; hky.pi_a[C] = pi[1]; hky.pi_a[G] = pi[2]; hky.pi_a[T] = pi[3];

      // 010010: class 0 holds the four transversions (and G-T, the reference), class 1 the transitions.
      auto generic = make_model("010010", {1.0, kappa}, pi, hky.mu);
      auto a_evo = hky.derive_site_evo_model();
      auto b_evo = generic.derive_site_evo_model();
      EXPECT_EQ(a_evo.mu, b_evo.mu);
      for (auto x : k_all_real_seq_letters) {
        EXPECT_EQ(a_evo.pi_a[x], b_evo.pi_a[x]);
        for (auto y : k_all_real_seq_letters) {
          EXPECT_EQ(a_evo.q_ab[x][y], b_evo.q_ab[x][y])
              << "kappa " << kappa << " " << x << y;  // bit-for-bit, not just close
        }
      }
    }
  }
}

TEST(Subst_model_test, nested_models_agree_exactly_under_the_nesting_constraint) {
  auto equal = std::array<double, 4>{0.25, 0.25, 0.25, 0.25};
  auto uneven = std::array<double, 4>{0.3, 0.2, 0.2, 0.3};

  // JC69 == HKY85 with kappa 1 and equal frequencies == GTR with all rates 1 and equal frequencies.
  auto jc = make_model("000000", {1.0}, equal);
  auto hky_as_jc = make_model("010010", {1.0, 1.0}, equal);
  auto gtr_as_jc = make_model("012345", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, equal);
  auto jc_evo = jc.derive_site_evo_model();
  for (const auto& other : {hky_as_jc, gtr_as_jc}) {
    auto evo = other.derive_site_evo_model();
    for (auto a : k_all_real_seq_letters)
      for (auto b : k_all_real_seq_letters) EXPECT_NEAR(jc_evo.q_ab[a][b], evo.q_ab[a][b], 1e-15);
  }
  // JC69's off-diagonal rate is 1/3.
  EXPECT_NEAR(jc_evo.q_ab[A][C], 1.0 / 3.0, 1e-15);

  // HKY85 == TN93 with the two transition rates equal == GTR with the HKY constraint.
  auto kappa = 3.25;
  auto hky = make_model("010010", {1.0, kappa}, uneven);
  auto tn93_as_hky = make_model("010020", {1.0, kappa, kappa}, uneven);
  auto gtr_as_hky = make_model("012345", {1.0, kappa, 1.0, 1.0, kappa, 1.0}, uneven);
  auto hky_evo = hky.derive_site_evo_model();
  for (const auto& other : {tn93_as_hky, gtr_as_hky}) {
    auto evo = other.derive_site_evo_model();
    for (auto a : k_all_real_seq_letters)
      for (auto b : k_all_real_seq_letters) EXPECT_NEAR(hky_evo.q_ab[a][b], evo.q_ab[a][b], 1e-15);
  }

  // TN93 with distinct transition rates == GTR under the same constraint.
  auto tn93 = make_model("010020", {1.0, 2.5, 4.5}, uneven);
  auto gtr_as_tn93 = make_model("012345", {1.0, 2.5, 1.0, 1.0, 4.5, 1.0}, uneven);
  auto tn_evo = tn93.derive_site_evo_model();
  auto gtr_evo = gtr_as_tn93.derive_site_evo_model();
  for (auto a : k_all_real_seq_letters)
    for (auto b : k_all_real_seq_letters) EXPECT_NEAR(tn_evo.q_ab[a][b], gtr_evo.q_ab[a][b], 1e-15);
}

TEST(Subst_model_test, every_grouping_is_a_constrained_GTR) {
  // A model is the GTR whose six exchangeabilities repeat the class rates; the matrices must agree.
  auto engine = std::mt19937{4242};
  for (const auto& grouping : all_exchangeability_groupings()) {
    auto model = random_model(grouping, engine);
    auto gtr = model;
    gtr.grouping = Exchangeability_grouping::from_code("012345");
    gtr.class_rate.assign(k_num_exchangeabilities, 1.0);
    for (auto i = 0; i != k_num_exchangeabilities; ++i) {
      gtr.class_rate[i] = model.exchangeability(i);
    }
    // Rescaling all six exchangeabilities by the reference rate leaves Q unchanged, which is how the
    // GTR form holds G-T at one.
    auto reference = gtr.class_rate[k_reference_exchangeability];
    for (auto& r : gtr.class_rate) { r /= reference; }
    ASSERT_NO_THROW(gtr.validate()) << grouping.code();
    auto a_evo = model.derive_site_evo_model();
    auto b_evo = gtr.derive_site_evo_model();
    for (auto a : k_all_real_seq_letters)
      for (auto b : k_all_real_seq_letters)
        EXPECT_NEAR(a_evo.q_ab[a][b], b_evo.q_ab[a][b], 1e-13) << grouping.code();
  }
}

TEST(Subst_model_test, scaling_every_exchangeability_leaves_the_normalized_matrix_unchanged) {
  auto engine = std::mt19937{99};
  for (const auto& grouping : all_exchangeability_groupings()) {
    auto model = random_model(grouping, engine);
    auto scaled = model;
    for (auto& r : scaled.class_rate) { r *= 7.5; }
    auto a_evo = model.derive_site_evo_model();
    auto b_evo = scaled.derive_site_evo_model();
    for (auto a : k_all_real_seq_letters)
      for (auto b : k_all_real_seq_letters)
        EXPECT_NEAR(a_evo.q_ab[a][b], b_evo.q_ab[a][b], 1e-13) << grouping.code();
  }
}

TEST(Subst_model_test, transition_probabilities_are_stochastic_and_approach_the_stationary_vector) {
  auto engine = std::mt19937{31337};
  for (const auto& grouping : all_exchangeability_groupings()) {
    auto model = random_model(grouping, engine);
    auto evo = model.derive_site_evo_model();
    for (auto t : {1e-6, 0.01, 0.5, 2.0}) {
      auto p = transition_matrix(evo, t);
      for (auto a = 0; a != 4; ++a) {
        auto row = 0.0L;
        for (auto b = 0; b != 4; ++b) {
          EXPECT_GE(p[a][b], 0.0L) << grouping.code();
          EXPECT_LE(p[a][b], 1.0L) << grouping.code();
          row += p[a][b];
        }
        EXPECT_NEAR(static_cast<double>(row), 1.0, 1e-12) << grouping.code();
      }
      // Reversibility of the transition matrix: pi_a P_ab == pi_b P_ba.
      for (auto a : k_all_real_seq_letters) {
        for (auto b : k_all_real_seq_letters) {
          EXPECT_NEAR(model.pi_a[a] * static_cast<double>(p[index_of(a)][index_of(b)]),
                      model.pi_a[b] * static_cast<double>(p[index_of(b)][index_of(a)]), 1e-12)
              << grouping.code();
        }
      }
    }
    // Long time: every row tends to pi.
    auto far = transition_matrix(evo, 2000.0);
    for (auto a : k_all_real_seq_letters) {
      for (auto b : k_all_real_seq_letters) {
        EXPECT_NEAR(static_cast<double>(far[index_of(a)][index_of(b)]), model.pi_a[b], 1e-6)
            << grouping.code();
      }
    }
  }
}

TEST(Subst_model_test, the_expected_number_of_substitutions_over_time_t_is_mu_t) {
  // -sum_a pi_a Q_aa == 1 means the expected number of events per site over t is mu t; checked here
  // against the derivative of the transition matrix at zero.
  auto engine = std::mt19937{5};
  for (const auto& grouping : all_exchangeability_groupings()) {
    auto model = random_model(grouping, engine);
    model.mu = 0.037;
    auto evo = model.derive_site_evo_model();
    auto expected = 0.0;
    for (auto a : k_all_real_seq_letters) { expected += model.pi_a[a] * evo.mu * evo.q_a(a); }
    EXPECT_NEAR(expected, model.mu, 1e-14) << grouping.code();
  }
}

TEST(Subst_model_test, invalid_parameters_are_refused_with_an_explanation) {
  auto base = make_model("012345", {1.0, 2.0, 3.0, 4.0, 5.0, 1.0}, {0.25, 0.25, 0.25, 0.25});
  EXPECT_NO_THROW(base.validate());

  auto wrong_count = base;
  wrong_count.class_rate = {1.0, 2.0};
  EXPECT_THROW(wrong_count.validate(), std::invalid_argument);

  auto negative = base;
  negative.class_rate[1] = -1.0;
  EXPECT_THROW(negative.validate(), std::invalid_argument);

  auto zero = base;
  zero.class_rate[1] = 0.0;
  EXPECT_THROW(zero.validate(), std::invalid_argument);

  auto unpinned = base;
  unpinned.class_rate[k_reference_exchangeability] = 2.0;
  EXPECT_THROW(unpinned.validate(), std::invalid_argument);

  auto bad_freqs = base;
  bad_freqs.pi_a[A] = 0.0;
  EXPECT_THROW(bad_freqs.validate(), std::invalid_argument);

  auto unnormalized = base;
  unnormalized.pi_a[A] = 0.3;
  EXPECT_THROW(unnormalized.validate(), std::invalid_argument);

  auto bad_mu = base;
  bad_mu.mu = -1.0;
  EXPECT_THROW(bad_mu.validate(), std::invalid_argument);
}

TEST(Subst_model_test, the_alias_table_matches_the_published_codes_and_degrees_of_freedom) {
  struct Expected { std::string_view name; std::string_view code; Frequency_mode freqs; int df; };
  // IQ-TREE's DNA model table: the code and the number of free parameters (exchangeabilities plus
  // frequencies).
  const auto table = std::vector<Expected>{
      {"JC",    "000000", Frequency_mode::equal,     0},
      {"JC69",  "000000", Frequency_mode::equal,     0},
      {"F81",   "000000", Frequency_mode::estimated, 3},
      {"K80",   "010010", Frequency_mode::equal,     1},
      {"K2P",   "010010", Frequency_mode::equal,     1},
      {"HKY",   "010010", Frequency_mode::estimated, 4},
      {"HKY85", "010010", Frequency_mode::estimated, 4},
      {"TN",    "010020", Frequency_mode::estimated, 5},
      {"TN93",  "010020", Frequency_mode::estimated, 5},
      {"TNe",   "010020", Frequency_mode::equal,     2},
      {"K81",   "012210", Frequency_mode::equal,     2},
      {"K3P",   "012210", Frequency_mode::equal,     2},
      {"K81u",  "012210", Frequency_mode::estimated, 5},
      {"TPM2",  "010212", Frequency_mode::equal,     2},
      {"TPM2u", "010212", Frequency_mode::estimated, 5},
      {"TPM3",  "012012", Frequency_mode::equal,     2},
      {"TPM3u", "012012", Frequency_mode::estimated, 5},
      {"TIM",   "012230", Frequency_mode::estimated, 6},
      {"TIMe",  "012230", Frequency_mode::equal,     3},
      {"TIM2",  "010232", Frequency_mode::estimated, 6},
      {"TIM2e", "010232", Frequency_mode::equal,     3},
      {"TIM3",  "012032", Frequency_mode::estimated, 6},
      {"TIM3e", "012032", Frequency_mode::equal,     3},
      {"TVM",   "012314", Frequency_mode::estimated, 7},
      {"TVMe",  "012314", Frequency_mode::equal,     4},
      {"SYM",   "012345", Frequency_mode::equal,     5},
      {"GTR",   "012345", Frequency_mode::estimated, 8}};

  for (const auto& expected : table) {
    const auto* alias = find_subst_model_alias(expected.name);
    ASSERT_NE(alias, nullptr) << expected.name;
    EXPECT_EQ(alias->frequencies, expected.freqs) << expected.name;
    auto grouping = Exchangeability_grouping::from_code(alias->code);
    EXPECT_EQ(grouping.code(), expected.code) << expected.name;
    auto free_frequencies = expected.freqs == Frequency_mode::estimated ? 3 : 0;
    EXPECT_EQ(grouping.num_free_rates() + free_frequencies, expected.df) << expected.name;
  }
  EXPECT_EQ(std::ssize(subst_model_aliases()), std::ssize(table));
  EXPECT_EQ(find_subst_model_alias("hky85"), find_subst_model_alias("HKY85"));
  EXPECT_EQ(find_subst_model_alias("not-a-model"), nullptr);
}

TEST(Subst_model_test, model_specifications_are_parsed_and_bad_ones_refused) {
  auto hky = parse_subst_model("HKY85");
  EXPECT_EQ(hky.grouping.code(), "010010");
  EXPECT_EQ(hky.frequency_mode, Frequency_mode::estimated);
  EXPECT_EQ(std::ssize(hky.class_rate), 2);
  EXPECT_EQ(hky.class_rate[0], 1.0);
  EXPECT_EQ(hky.class_rate[1], 1.0);

  EXPECT_EQ(parse_subst_model("JC69").grouping.code(), "000000");
  EXPECT_EQ(parse_subst_model("JC69").frequency_mode, Frequency_mode::equal);
  EXPECT_EQ(parse_subst_model("GTR").grouping.code(), "012345");
  EXPECT_EQ(parse_subst_model("SYM").frequency_mode, Frequency_mode::equal);

  // A bare code defaults to estimated frequencies; the suffixes override it.
  EXPECT_EQ(parse_subst_model("010020").grouping.code(), "010020");
  EXPECT_EQ(parse_subst_model("010020").frequency_mode, Frequency_mode::estimated);
  EXPECT_EQ(parse_subst_model("101101").grouping.code(), "010010");
  EXPECT_EQ(parse_subst_model("HKY85+FQ").frequency_mode, Frequency_mode::equal);
  EXPECT_EQ(parse_subst_model("JC69+FO").frequency_mode, Frequency_mode::estimated);
  EXPECT_EQ(parse_subst_model("012345+fq").frequency_mode, Frequency_mode::equal);
  for (auto a : k_all_real_seq_letters) { EXPECT_EQ(parse_subst_model("GTR+FQ").pi_a[a], 0.25); }

  EXPECT_THROW(parse_subst_model(""), std::invalid_argument);
  EXPECT_THROW(parse_subst_model("NOSUCH"), std::invalid_argument);
  EXPECT_THROW(parse_subst_model("01001"), std::invalid_argument);
  EXPECT_THROW(parse_subst_model("HKY85+G4"), std::invalid_argument);
  EXPECT_THROW(parse_subst_model("HKY85+I"), std::invalid_argument);
  EXPECT_THROW(parse_subst_model("UNREST"), std::invalid_argument);
}

TEST(Subst_model_test, a_grouping_can_be_named_back) {
  auto hky = Exchangeability_grouping::from_code("010010");
  EXPECT_THAT(subst_model_names_of(hky, Frequency_mode::estimated),
              testing::UnorderedElementsAre("HKY", "HKY85"));
  EXPECT_THAT(subst_model_names_of(hky, Frequency_mode::equal),
              testing::UnorderedElementsAre("K80", "K2P"));
  EXPECT_THAT(subst_model_names_of(Exchangeability_grouping::from_code("012345"),
                                   Frequency_mode::equal),
              testing::ElementsAre("SYM"));
  // Most of the 203 groupings have no published name, which is not an error.
  EXPECT_TRUE(subst_model_names_of(Exchangeability_grouping::from_code("001122"),
                                   Frequency_mode::estimated).empty());
}

}  // namespace delphy
