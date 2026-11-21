#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/random/random.h"

#include "ctmc_bridge.h"

namespace delphy {

namespace {

using enum Real_seq_letter;

// Independent reference arithmetic
// ================================
// None of what follows shares code with ctmc_bridge.cpp: the matrix exponential is a scaled and
// squared Taylor series in long double, the reference sampler is forward Gillespie simulation with
// rejection on the end state, and the reference expected number of substitutions is Simpson
// quadrature of the convolution integral.  The two routes to the matrix exponential (series in the
// uniformized chain versus series in Q itself) therefore agree only if both are right.

using Mat = std::array<std::array<long double, 4>, 4>;

auto mat_mul(const Mat& x, const Mat& y) -> Mat {
  auto out = Mat{};
  for (auto i = 0; i != 4; ++i) {
    for (auto j = 0; j != 4; ++j) {
      auto sum = 0.0L;
      for (auto k = 0; k != 4; ++k) { sum += x[i][k] * y[k][j]; }
      out[i][j] = sum;
    }
  }
  return out;
}

auto ref_mat_exp(const Mat& m) -> Mat {
  auto norm = 0.0L;
  for (const auto& row : m) { for (auto v : row) { norm = std::max(norm, std::fabs(v)); } }
  auto squarings = 0;
  while (norm > 0.05L) { norm /= 2; ++squarings; }
  auto scaled = m;
  for (auto& row : scaled) { for (auto& v : row) { v /= std::pow(2.0L, squarings); } }

  auto term = Mat{};
  auto sum = Mat{};
  for (auto i = 0; i != 4; ++i) { term[i][i] = 1.0L; sum[i][i] = 1.0L; }
  for (auto k = 1; k != 40; ++k) {
    term = mat_mul(term, scaled);
    for (auto i = 0; i != 4; ++i) {
      for (auto j = 0; j != 4; ++j) { sum[i][j] += term[i][j] / std::tgamma(static_cast<long double>(k + 1)); }
    }
  }
  for (auto s = 0; s != squarings; ++s) { sum = mat_mul(sum, sum); }
  return sum;
}

auto to_ref_mat(const Seq_matrix<double>& q, double scale) -> Mat {
  auto out = Mat{};
  for (auto a : k_all_real_seq_letters) {
    for (auto b : k_all_real_seq_letters) {
      out[index_of(a)][index_of(b)] = static_cast<long double>(q[a][b]) * static_cast<long double>(scale);
    }
  }
  return out;
}

// E[number of substitutions | X(0) = a, X(tau) = b] by Simpson quadrature of
//   (1 / P_ab(tau)) sum_{i != j} Q_ij int_0^tau P_ai(s) P_jb(tau - s) ds.
auto ref_expected_num_subs(const Seq_matrix<double>& q, double tau, Real_seq_letter a, Real_seq_letter b)
    -> double {
  const auto num_panels = 2000;  // even, so Simpson applies
  auto p_full = ref_mat_exp(to_ref_mat(q, tau));
  auto integral = Mat{};  // integral[i][j] holds int P_ai(s) P_jb(tau-s) ds
  auto h = static_cast<long double>(tau) / num_panels;
  for (auto n = 0; n <= num_panels; ++n) {
    auto s = h * n;
    auto weight = (n == 0 || n == num_panels) ? 1.0L : ((n % 2 == 1) ? 4.0L : 2.0L);
    auto p_s = ref_mat_exp(to_ref_mat(q, static_cast<double>(s)));
    auto p_rest = ref_mat_exp(to_ref_mat(q, static_cast<double>(static_cast<long double>(tau) - s)));
    for (auto i = 0; i != 4; ++i) {
      for (auto j = 0; j != 4; ++j) {
        integral[i][j] += weight * p_s[index_of(a)][i] * p_rest[j][index_of(b)];
      }
    }
  }
  auto total = 0.0L;
  for (auto i = 0; i != 4; ++i) {
    for (auto j = 0; j != 4; ++j) {
      if (i == j) { continue; }
      total += static_cast<long double>(q[real_seq_letter_from_index(i)][real_seq_letter_from_index(j)])
          * integral[i][j] * h / 3;
    }
  }
  return static_cast<double>(total / p_full[index_of(a)][index_of(b)]);
}

// Forward Gillespie simulation of the chain on [0, tau], rejected unless it ends in `b`.  Exact, and
// built on nothing but the escape rates and the jump probabilities.
auto ref_sample_bridge(
    const Seq_matrix<double>& q,
    Real_seq_letter a,
    Real_seq_letter b,
    double tau,
    std::mt19937_64& bitgen)
    -> std::vector<Ctmc_bridge_event> {

  while (true) {
    auto path = std::vector<Ctmc_bridge_event>{};
    auto state = a;
    auto t = 0.0;
    while (true) {
      auto escape = -q[state][state];
      if (not(escape > 0.0)) { t = tau + 1; break; }
      t += std::exponential_distribution<double>{escape}(bitgen);
      if (t >= tau) { break; }
      auto u = std::uniform_real_distribution<double>{0.0, escape}(bitgen);
      auto cum = 0.0;
      auto next = state;
      for (auto s : k_all_real_seq_letters) {
        if (s == state) { continue; }
        cum += q[state][s];
        if (cum >= u) { next = s; break; }
      }
      state = next;
      path.push_back(Ctmc_bridge_event{t, state});
    }
    if (state == b) { return path; }
  }
}

// Model matrices
// ==============

// A reversible rate matrix from six exchangeabilities (order AC, AG, AT, CG, CT, GT) and
// frequencies, with no normalization: Q_ab = s_ab pi_b.
auto make_rate_matrix(const std::array<double, 6>& s, const Seq_vector<double>& pi) -> Seq_matrix<double> {
  auto pairs = std::array<std::pair<Real_seq_letter, Real_seq_letter>, 6>{
      {{A, C}, {A, G}, {A, T}, {C, G}, {C, T}, {G, T}}};
  auto q = Seq_matrix<double>{};
  for (auto i = 0; i != 6; ++i) {
    auto [x, y] = pairs[i];
    q[x][y] = s[i] * pi[y];
    q[y][x] = s[i] * pi[x];
  }
  for (auto x : k_all_real_seq_letters) {
    auto sum = 0.0;
    for (auto y : k_all_real_seq_letters) { if (x != y) { sum += q[x][y]; } }
    q[x][x] = -sum;
  }
  return q;
}

auto jc_matrix() -> Seq_matrix<double> {
  // Escape rate one from every state, so that tau is the expected number of substitutions.
  return make_rate_matrix({4.0 / 3, 4.0 / 3, 4.0 / 3, 4.0 / 3, 4.0 / 3, 4.0 / 3}, Seq_vector<double>{0.25});
}

auto hky_matrix() -> Seq_matrix<double> {
  return make_rate_matrix({1, 4.5, 1, 1, 4.5, 1}, Seq_vector<double>{0.35, 0.15, 0.2, 0.3});
}

auto gtr_matrix() -> Seq_matrix<double> {
  return make_rate_matrix({0.4, 3.1, 0.7, 1.3, 2.6, 1.0}, Seq_vector<double>{0.4, 0.1, 0.25, 0.25});
}

// Replays a sampled path and returns the state it ends in, checking the invariants of the result.
auto replay(const std::vector<Ctmc_bridge_event>& path, Real_seq_letter a, double T) -> Real_seq_letter {
  auto state = a;
  auto prev_t = 0.0;
  for (const auto& e : path) {
    EXPECT_GT(e.t, 0.0);
    EXPECT_LT(e.t, T);
    EXPECT_GE(e.t, prev_t);
    EXPECT_NE(e.to, state);
    prev_t = e.t;
    state = e.to;
  }
  return state;
}

}  // namespace

TEST(Ctmc_bridge_test, transition_probs_match_independent_matrix_exponential) {
  for (const auto& q : {jc_matrix(), hky_matrix(), gtr_matrix()}) {
    auto sampler = Ctmc_bridge_sampler{q};
    for (auto tau : {0.0, 1e-9, 1e-4, 0.01, 0.3, 1.0, 3.7, 25.0, 200.0}) {
      auto got = sampler.calc_transition_probs(tau);
      auto want = ref_mat_exp(to_ref_mat(q, tau));
      for (auto a : k_all_real_seq_letters) {
        auto row_sum = 0.0;
        for (auto b : k_all_real_seq_letters) {
          EXPECT_NEAR(got[a][b], static_cast<double>(want[index_of(a)][index_of(b)]), 1e-12)
              << "tau = " << tau << ", a = " << to_char(a) << ", b = " << to_char(b);
          row_sum += got[a][b];
        }
        EXPECT_NEAR(row_sum, 1.0, 1e-12) << "tau = " << tau;
      }
    }
  }
}

TEST(Ctmc_bridge_test, rejects_matrices_that_are_not_rate_matrices) {
  auto q = jc_matrix();
  auto bad_row_sum = q;
  bad_row_sum[A][A] += 0.5;
  EXPECT_THROW(Ctmc_bridge_sampler{bad_row_sum}, std::invalid_argument);

  auto negative_rate = q;
  negative_rate[A][C] = -1.0;
  negative_rate[A][A] = -(negative_rate[A][C] + negative_rate[A][G] + negative_rate[A][T]);
  EXPECT_THROW(Ctmc_bridge_sampler{negative_rate}, std::invalid_argument);

  auto not_finite = q;
  not_finite[A][C] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(Ctmc_bridge_sampler{not_finite}, std::invalid_argument);
}

TEST(Ctmc_bridge_test, zero_duration_and_zero_rate_admit_only_the_empty_path) {
  auto sampler = Ctmc_bridge_sampler{gtr_matrix()};
  auto bitgen = absl::BitGen{};
  auto path = std::vector<Ctmc_bridge_event>{};

  sampler.sample(C, C, 0.0, 1.0, bitgen, path);
  EXPECT_TRUE(path.empty());
  sampler.sample(C, C, 1.0, 0.0, bitgen, path);
  EXPECT_TRUE(path.empty());
  EXPECT_THROW(sampler.sample(C, G, 0.0, 1.0, bitgen, path), std::domain_error);
  EXPECT_THROW(sampler.sample(C, G, 1.0, 0.0, bitgen, path), std::domain_error);
  EXPECT_THROW(sampler.sample(C, C, -1.0, 1.0, bitgen, path), std::domain_error);
  EXPECT_THROW(sampler.sample(C, C, 1.0, -1.0, bitgen, path), std::domain_error);

  auto zero = Seq_matrix<double>{};
  auto zero_sampler = Ctmc_bridge_sampler{zero};
  EXPECT_EQ(zero_sampler.omega(), 0.0);
  zero_sampler.sample(A, A, 1.0, 1.0, bitgen, path);
  EXPECT_TRUE(path.empty());
  EXPECT_THROW(zero_sampler.sample(A, T, 1.0, 1.0, bitgen, path), std::domain_error);
}

TEST(Ctmc_bridge_test, rejects_transitions_no_path_can_make) {
  // Only A <-> C and G <-> T are allowed, so the two pairs are separate communicating classes.
  auto q = Seq_matrix<double>{};
  q[A][C] = 1.0; q[A][A] = -1.0;
  q[C][A] = 1.0; q[C][C] = -1.0;
  q[G][T] = 1.0; q[G][G] = -1.0;
  q[T][G] = 1.0; q[T][T] = -1.0;
  auto sampler = Ctmc_bridge_sampler{q};
  auto bitgen = absl::BitGen{};
  auto path = std::vector<Ctmc_bridge_event>{};
  EXPECT_THROW(sampler.sample(A, G, 1.0, 1.0, bitgen, path), std::domain_error);
  EXPECT_NO_THROW(sampler.sample(A, C, 1.0, 1.0, bitgen, path));
}

TEST(Ctmc_bridge_test, paths_respect_the_endpoints_and_the_branch) {
  auto q = gtr_matrix();
  auto sampler = Ctmc_bridge_sampler{q};
  auto bitgen = absl::BitGen{};
  // A branch whose duration and rate scale differ from one, so that the two are not interchangeable
  // in the returned times.
  auto T = 7.5;
  auto rate_scale = 0.4;
  for (auto a : k_all_real_seq_letters) {
    for (auto b : k_all_real_seq_letters) {
      for (auto rep = 0; rep != 200; ++rep) {
        auto path = std::vector<Ctmc_bridge_event>{};
        sampler.sample(a, b, T, rate_scale, bitgen, path);
        EXPECT_EQ(replay(path, a, T), b) << "a = " << to_char(a) << ", b = " << to_char(b);
      }
    }
  }
}

TEST(Ctmc_bridge_test, appends_without_clearing) {
  auto sampler = Ctmc_bridge_sampler{jc_matrix()};
  auto bitgen = absl::BitGen{};
  auto path = std::vector<Ctmc_bridge_event>{Ctmc_bridge_event{-1.0, A}};
  sampler.sample(A, C, 1.0, 1.0, bitgen, path);
  ASSERT_GE(std::ssize(path), 2);
  EXPECT_EQ(path[0].t, -1.0);
}

TEST(Ctmc_bridge_test, mean_number_of_substitutions_matches_independent_quadrature) {
  struct Case { Seq_matrix<double> q; double tau; Real_seq_letter a; Real_seq_letter b; };
  auto cases = std::vector<Case>{
      // Equal endpoints at low rate: the hidden substitutions come in pairs and are rare.
      {jc_matrix(), 0.05, A, A},
      // Equal endpoints at high rate: most of the mass is on two or more hidden substitutions.
      {jc_matrix(), 1.5, A, A},
      {gtr_matrix(), 2.5, G, G},
      // Different endpoints, from the parsimonious single substitution to multiple hits.
      {jc_matrix(), 0.05, A, C},
      {gtr_matrix(), 0.3, A, T},
      {hky_matrix(), 1.2, C, T},
      {gtr_matrix(), 4.0, C, A},
  };

  auto bitgen = absl::BitGen{};
  for (const auto& c : cases) {
    auto want = ref_expected_num_subs(c.q, c.tau, c.a, c.b);
    auto sampler = Ctmc_bridge_sampler{c.q};
    const auto num_reps = 200000;
    auto sum = 0.0;
    auto sum_sq = 0.0;
    for (auto rep = 0; rep != num_reps; ++rep) {
      auto path = std::vector<Ctmc_bridge_event>{};
      sampler.sample(c.a, c.b, 1.0, c.tau, bitgen, path);
      auto n = static_cast<double>(std::ssize(path));
      sum += n;
      sum_sq += n * n;
    }
    auto mean = sum / num_reps;
    auto variance = std::max(sum_sq / num_reps - mean * mean, 1e-12);
    auto stderr_mean = std::sqrt(variance / num_reps);
    EXPECT_LT(std::abs(mean - want), 5 * stderr_mean + 1e-9)
        << "tau = " << c.tau << ", " << to_char(c.a) << " to " << to_char(c.b)
        << ": sampled " << mean << " +/- " << stderr_mean << ", quadrature " << want;
  }
}

TEST(Ctmc_bridge_test, path_distribution_matches_independent_rejection_sampling) {
  struct Case { Seq_matrix<double> q; double tau; Real_seq_letter a; Real_seq_letter b; };
  auto cases = std::vector<Case>{
      {jc_matrix(), 0.8, A, A},
      {jc_matrix(), 0.8, A, G},
      {gtr_matrix(), 1.6, C, C},
      {gtr_matrix(), 1.6, G, A},
      {hky_matrix(), 2.4, T, C},
  };

  const auto num_reps = 120000;
  const auto max_bin = 5;  // counts of max_bin or more share the last bin

  for (const auto& c : cases) {
    auto sampler = Ctmc_bridge_sampler{c.q};
    auto bitgen_got = std::mt19937_64{12345};
    auto bitgen_want = std::mt19937_64{54321};

    auto got_counts = std::vector<double>(max_bin + 1, 0.0);
    auto want_counts = std::vector<double>(max_bin + 1, 0.0);
    // Mean time of the first substitution, and the frequency of each substitution type.
    auto got_first_t = 0.0;
    auto want_first_t = 0.0;
    auto got_first_n = 0;
    auto want_first_n = 0;
    auto got_types = Seq_matrix<double>{};
    auto want_types = Seq_matrix<double>{};

    auto tally = [&](const std::vector<Ctmc_bridge_event>& path, std::vector<double>& counts,
                     double& first_t, int& first_n, Seq_matrix<double>& types, Real_seq_letter a) {
      counts[std::min<int>(static_cast<int>(std::ssize(path)), max_bin)] += 1;
      if (not path.empty()) { first_t += path.front().t; ++first_n; }
      auto state = a;
      for (const auto& e : path) { types[state][e.to] += 1; state = e.to; }
    };

    for (auto rep = 0; rep != num_reps; ++rep) {
      auto path = std::vector<Ctmc_bridge_event>{};
      sampler.sample(c.a, c.b, 1.0, c.tau, bitgen_got, path);
      tally(path, got_counts, got_first_t, got_first_n, got_types, c.a);

      // The reference works on the time axis [0, tau]; rescale to [0, 1] to compare the times.
      auto ref_path = ref_sample_bridge(c.q, c.a, c.b, c.tau, bitgen_want);
      for (auto& e : ref_path) { e.t /= c.tau; }
      tally(ref_path, want_counts, want_first_t, want_first_n, want_types, c.a);
    }

    // Pearson chi-square between the two empirical distributions of the number of substitutions.
    auto chi_square = 0.0;
    auto degrees = 0;
    for (auto k = 0; k <= max_bin; ++k) {
      auto pooled = got_counts[k] + want_counts[k];
      if (pooled < 50) { continue; }
      auto expected = pooled / 2;
      chi_square += (got_counts[k] - expected) * (got_counts[k] - expected) / expected;
      chi_square += (want_counts[k] - expected) * (want_counts[k] - expected) / expected;
      ++degrees;
    }
    ASSERT_GE(degrees, 2) << "tau = " << c.tau;
    // The 1e-4 upper quantile of chi-square with `degrees` degrees of freedom is below this bound
    // for every `degrees` the cases reach, so a passing test is not a tight fit by luck.
    EXPECT_LT(chi_square, 5.0 * degrees + 20.0)
        << "tau = " << c.tau << ", " << to_char(c.a) << " to " << to_char(c.b)
        << ": chi-square " << chi_square << " on " << degrees << " bins";

    // The mean time of the first substitution, to three standard errors of the difference.
    ASSERT_GT(got_first_n, 1000);
    ASSERT_GT(want_first_n, 1000);
    auto got_mean_t = got_first_t / got_first_n;
    auto want_mean_t = want_first_t / want_first_n;
    // The times lie in [0, 1], so 1/4 bounds their variance.
    auto t_tolerance = 4.0 * std::sqrt(0.25 * (1.0 / got_first_n + 1.0 / want_first_n));
    EXPECT_LT(std::abs(got_mean_t - want_mean_t), t_tolerance)
        << "tau = " << c.tau << ": first substitution at " << got_mean_t << " against " << want_mean_t;

    // Each substitution type, as a share of all substitutions.
    auto got_total = 0.0;
    auto want_total = 0.0;
    for (auto x : k_all_real_seq_letters) {
      for (auto y : k_all_real_seq_letters) { got_total += got_types[x][y]; want_total += want_types[x][y]; }
    }
    ASSERT_GT(got_total, 1000);
    for (auto x : k_all_real_seq_letters) {
      for (auto y : k_all_real_seq_letters) {
        if (x == y) { continue; }
        auto got_share = got_types[x][y] / got_total;
        auto want_share = want_types[x][y] / want_total;
        auto share_tolerance = 5.0 * std::sqrt(std::max(want_share, 1e-4) / std::min(got_total, want_total));
        EXPECT_LT(std::abs(got_share - want_share), share_tolerance)
            << "tau = " << c.tau << ", type " << to_char(x) << to_char(y) << ": " << got_share
            << " against " << want_share;
      }
    }
  }
}

TEST(Ctmc_bridge_test, handles_extreme_durations_and_rates) {
  auto sampler = Ctmc_bridge_sampler{gtr_matrix()};
  auto bitgen = absl::BitGen{};

  // A branch so short that the conditional distribution is concentrated on the parsimonious path.
  auto path = std::vector<Ctmc_bridge_event>{};
  for (auto rep = 0; rep != 2000; ++rep) {
    path.clear();
    sampler.sample(A, C, 1e-12, 1.0, bitgen, path);
    EXPECT_EQ(std::ssize(path), 1);
    EXPECT_EQ(replay(path, A, 1e-12), C);
  }

  // A branch long enough that the endpoints are nearly independent, and the number of substitutions
  // is large.  The stationary frequencies and the mean count must still come out right.
  auto long_tau = 60.0;
  auto want = ref_expected_num_subs(gtr_matrix(), long_tau, A, T);
  auto sum = 0.0;
  const auto num_reps = 20000;
  for (auto rep = 0; rep != num_reps; ++rep) {
    path.clear();
    sampler.sample(A, T, 1.0, long_tau, bitgen, path);
    EXPECT_EQ(replay(path, A, 1.0), T);
    sum += static_cast<double>(std::ssize(path));
  }
  EXPECT_NEAR(sum / num_reps, want, 0.05 * want);
}

TEST(Ctmc_bridge_test, equal_endpoints_recover_hidden_multiple_substitutions) {
  // The move must be able to put substitutions on a branch whose endpoints agree, which is the
  // mechanism the all-sites proposal cannot reach at high divergence.
  auto sampler = Ctmc_bridge_sampler{jc_matrix()};
  auto bitgen = absl::BitGen{};
  auto num_with_events = 0;
  auto max_events = 0;
  const auto num_reps = 20000;
  for (auto rep = 0; rep != num_reps; ++rep) {
    auto path = std::vector<Ctmc_bridge_event>{};
    sampler.sample(A, A, 1.0, 1.0, bitgen, path);
    EXPECT_EQ(replay(path, A, 1.0), A);
    EXPECT_NE(std::ssize(path), 1);  // one substitution cannot return to the start state
    if (not path.empty()) { ++num_with_events; }
    max_events = std::max<int>(max_events, static_cast<int>(std::ssize(path)));
  }
  // At tau = 1 under this matrix, P_AA(1) = 1/4 + 3/4 exp(-4/3) = 0.447698, so the probability of no
  // substitution at all is exp(-1) / 0.447698 = 0.82172.
  EXPECT_NEAR(static_cast<double>(num_with_events) / num_reps, 1.0 - 0.82172, 0.01);
  EXPECT_GE(max_events, 4);
}

}  // namespace delphy
