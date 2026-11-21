#include "gtest/gtest.h"

#include <array>
#include <cmath>

#include "evo_hky.h"

namespace delphy {

namespace {

using Mat = std::array<std::array<long double, 4>, 4>;

auto mat_mul(const Mat& x, const Mat& y) -> Mat {
  auto out = Mat{};
  for (auto i = 0; i != 4; ++i)
    for (auto j = 0; j != 4; ++j)
      for (auto k = 0; k != 4; ++k) out[i][j] += x[i][k] * y[k][j];
  return out;
}

// exp(m) by scaling and squaring of a 30-term Taylor series, in long double; shares no code with Delphy.
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

auto transition_matrix(const Hky_model& hky, double t) -> Mat {
  auto evo = hky.derive_site_evo_model();
  auto m = Mat{};
  for (auto a : k_all_real_seq_letters)
    for (auto b : k_all_real_seq_letters) m[static_cast<int>(a)][static_cast<int>(b)] = evo.q_ab[a][b] * evo.mu * t;
  return mat_exp(m);
}

auto make_hky(double mu, double kappa, std::array<double, 4> pi) -> Hky_model {
  auto hky = Hky_model{};
  hky.mu = mu;
  hky.kappa = kappa;
  hky.pi_a[Real_seq_letter::A] = pi[0];
  hky.pi_a[Real_seq_letter::C] = pi[1];
  hky.pi_a[Real_seq_letter::G] = pi[2];
  hky.pi_a[Real_seq_letter::T] = pi[3];
  return hky;
}

}  // namespace

// Reference values: scipy.linalg.expm of the rate matrix built from the textbook definition
// (tests/cli/simulate_hky_joint.py, validated against the closed-form HKY probabilities).
TEST(Hky_reference_test, rate_matrix_is_normalized_stationary_and_reversible) {
  auto hky = make_hky(1.0, 3.0, {0.31, 0.23, 0.17, 0.29});
  auto evo = hky.derive_site_evo_model();
  auto mean_rate = 0.0;
  for (auto a : k_all_real_seq_letters) {
    auto row = 0.0;
    for (auto b : k_all_real_seq_letters) row += evo.q_ab[a][b];
    EXPECT_NEAR(row, 0.0, 1e-14);
    mean_rate += hky.pi_a[a] * evo.q_a(a);
    for (auto b : k_all_real_seq_letters)
      EXPECT_NEAR(hky.pi_a[a] * evo.q_ab[a][b], hky.pi_a[b] * evo.q_ab[b][a], 1e-14);
  }
  EXPECT_NEAR(mean_rate, 1.0, 1e-14);
}

TEST(Hky_reference_test, transition_probabilities_match_scipy_expm) {
  struct Case { double mu; double kappa; std::array<double, 4> pi; double t; Mat p; };
  auto cases = std::vector<Case>{
      {0.1, 1.0, {0.25, 0.25, 0.25, 0.25}, 3.0,
       {{{0.752740034526729, 0.082419988491090, 0.082419988491090, 0.082419988491090},
         {0.082419988491090, 0.752740034526730, 0.082419988491090, 0.082419988491090},
         {0.082419988491090, 0.082419988491090, 0.752740034526729, 0.082419988491090},
         {0.082419988491090, 0.082419988491090, 0.082419988491090, 0.752740034526730}}}},
      {0.4, 3.0, {0.31, 0.23, 0.17, 0.29}, 2.0,
       {{{0.581405743920161, 0.110899450737447, 0.167865063108219, 0.139829742234173},
         {0.149473172733081, 0.485596836274105, 0.081969159240722, 0.282960831752092},
         {0.306106879785576, 0.110899450737447, 0.443163927242804, 0.139829742234173},
         {0.149473172733081, 0.224417211389590, 0.081969159240722, 0.544140456636607}}}},
      {2.5, 8.5, {0.1, 0.2, 0.3, 0.4}, 1.0,
       {{{0.162411145693469, 0.130973700833890, 0.444667751804862, 0.261947401667780},
         {0.065486850416945, 0.247935424153835, 0.196460551250835, 0.490117174178386},
         {0.148222583934954, 0.130973700833890, 0.458856313563376, 0.261947401667780},
         {0.065486850416945, 0.245058587089193, 0.196460551250835, 0.492994011243028}}}},
  };
  for (const auto& c : cases) {
    auto p = transition_matrix(make_hky(c.mu, c.kappa, c.pi), c.t);
    for (auto i = 0; i != 4; ++i)
      for (auto j = 0; j != 4; ++j) EXPECT_NEAR(static_cast<double>(p[i][j]), static_cast<double>(c.p[i][j]), 1e-12);
  }
}

// Delphy's rate is per site per DAY and its times are in days: a rate of mu_year / 365 over
// 365 * t_years days is the same transition matrix as mu_year over t_years years.
TEST(Hky_reference_test, per_day_rate_over_days_equals_per_year_rate_over_years) {
  auto pi = std::array<double, 4>{0.31, 0.23, 0.17, 0.29};
  auto mu_year = 1.7e-8;
  auto p = transition_matrix(make_hky(mu_year / 365.0, 3.0, pi), 40'000.0 * 365.0);
  auto ref = Mat{{{0.99942408895926926, 0.0001286247643162661, 0.00028510722575487778, 0.00016217905065963982},
                  {0.00017336381277409767, 0.99924521357717266, 9.50704779728923e-05, 0.00048635213208035637},
                  {0.00051990141167065939, 0.00012862476431626613, 0.99918929477335339, 0.00016217905065963982},
                  {0.0001733638127740977, 0.00038572755302924841, 9.5070477972892327e-05, 0.9993458381562238}}};
  for (auto i = 0; i != 4; ++i)
    for (auto j = 0; j != 4; ++j) EXPECT_NEAR(static_cast<double>(p[i][j]), static_cast<double>(ref[i][j]), 1e-13);

  auto p_long = transition_matrix(make_hky(mu_year / 365.0, 3.0, pi), 1'200'000.0 * 365.0);
  auto ref_long = Mat{{{0.9829512405002373, 0.0038276152733004326, 0.0083950206209964951, 0.0048261236054657607},
                       {0.0051589597161875392, 0.97769736946227082, 0.0028291069411351031, 0.014314563880406512},
                       {0.015308567014758313, 0.0038276152733004308, 0.97603769410647545, 0.004826123605465759},
                       {0.0051589597161875384, 0.011352929974115512, 0.0028291069411351022, 0.98065900336856182}}};
  for (auto i = 0; i != 4; ++i)
    for (auto j = 0; j != 4; ++j) EXPECT_NEAR(static_cast<double>(p_long[i][j]), static_cast<double>(ref_long[i][j]), 1e-12);
}

}  // namespace delphy
