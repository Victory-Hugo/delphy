#include "ctmc_bridge.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "absl/random/distributions.h"
#include "absl/strings/str_format.h"

namespace delphy {

namespace {

// Relative tolerance on the row sums of a rate matrix.
constexpr auto k_rate_matrix_tolerance = 1e-9;

// The Poisson series is stopped past its mode once the mass it has left to distribute can no longer
// change the quantity being accumulated at this relative precision, or once that remaining mass, or
// the individual terms, have underflowed to zero.  The conditional distribution of the number of
// jumps has mass above every finite bound; what is dropped here is bounded by the remaining Poisson
// mass divided by the transition probability, which this test holds below the relative tolerance
// rather than at a fixed index.
constexpr auto k_series_relative_tolerance = 1e-13;

}  // namespace

Ctmc_bridge_sampler::Ctmc_bridge_sampler(const Seq_matrix<double>& rate_ab)
    : rate_ab_{rate_ab}, r_ab_{}, omega_{0.0} {

  auto scale = 0.0;
  for (auto a : k_all_real_seq_letters) {
    for (auto b : k_all_real_seq_letters) {
      auto q = rate_ab_[a][b];
      if (not std::isfinite(q)) {
        throw std::invalid_argument(absl::StrFormat(
            "Rate matrix entry (%c,%c) is not finite", to_char(a), to_char(b)));
      }
      if (a != b && q < 0.0) {
        throw std::invalid_argument(absl::StrFormat(
            "Rate matrix entry (%c,%c) is negative (%g)", to_char(a), to_char(b), q));
      }
      scale = std::max(scale, std::abs(q));
    }
  }
  for (auto a : k_all_real_seq_letters) {
    auto row_sum = 0.0;
    for (auto b : k_all_real_seq_letters) {
      row_sum += rate_ab_[a][b];
    }
    if (std::abs(row_sum) > k_rate_matrix_tolerance * std::max(scale, 1.0)) {
      throw std::invalid_argument(absl::StrFormat(
          "Row %c of the rate matrix sums to %g instead of zero", to_char(a), row_sum));
    }
    omega_ = std::max(omega_, -rate_ab_[a][a]);
  }

  // R = I + Q/Omega.  With Omega equal to the largest escape rate, R is stochastic: its off-diagonal
  // entries are non-negative, its diagonal is 1 + Q_aa/Omega >= 0, and its rows sum to one.  A
  // vanishing Omega leaves R at the identity, which is the correct uniformization of a zero matrix.
  for (auto a : k_all_real_seq_letters) {
    for (auto b : k_all_real_seq_letters) {
      r_ab_[a][b] = (a == b ? 1.0 : 0.0) + (omega_ > 0.0 ? rate_ab_[a][b] / omega_ : 0.0);
    }
    // Clean up the rounding of the diagonal so that R stays a probability matrix exactly.
    r_ab_[a][a] = std::max(r_ab_[a][a], 0.0);
  }
}

auto Ctmc_bridge_sampler::calc_transition_probs(double tau) const -> Seq_matrix<double> {
  if (not(tau >= 0.0) || not std::isfinite(tau)) {
    throw std::domain_error(absl::StrFormat("Effective duration %g is not a finite non-negative number", tau));
  }

  auto result = Seq_matrix<double>{};
  for (auto a : k_all_real_seq_letters) {
    result[a][a] = 1.0;
  }
  auto omega_tau = omega_ * tau;
  if (not(omega_tau > 0.0)) {
    return result;
  }

  // P = sum_k Pois(k; Omega tau) R^k, accumulated with the running power of R.
  auto power = Seq_matrix<double>{};
  for (auto a : k_all_real_seq_letters) {
    power[a][a] = 1.0;
  }
  auto accum = Seq_matrix<double>{};
  auto log_omega_tau = std::log(omega_tau);
  auto poisson_mass = 0.0;
  for (auto k = 0; k <= k_max_jumps; ++k) {
    auto p_k = std::exp(-omega_tau + k * log_omega_tau - std::lgamma(k + 1.0));
    poisson_mass += p_k;
    auto min_accum = std::numeric_limits<double>::infinity();
    for (auto a : k_all_real_seq_letters) {
      for (auto b : k_all_real_seq_letters) {
        accum[a][b] += p_k * power[a][b];
        min_accum = std::min(min_accum, accum[a][b]);
      }
    }
    auto tail_bound = std::max(0.0, 1.0 - poisson_mass);
    if (k >= omega_tau
        && (p_k == 0.0 || tail_bound == 0.0 || tail_bound <= k_series_relative_tolerance * min_accum)) {
      return accum;
    }
    auto next = Seq_matrix<double>{};
    for (auto a : k_all_real_seq_letters) {
      for (auto b : k_all_real_seq_letters) {
        auto sum = 0.0;
        for (auto c : k_all_real_seq_letters) {
          sum += power[a][c] * r_ab_[c][b];
        }
        next[a][b] = sum;
      }
    }
    power = next;
  }

  throw std::domain_error(absl::StrFormat(
      "The uniformized chain needs more than %d jumps at Omega*tau = %g; the rate or the duration is"
      " out of the supported range",
      k_max_jumps, omega_tau));
}

auto Ctmc_bridge_sampler::calc_transition_prob(Real_seq_letter a, Real_seq_letter b, double tau) const
    -> double {
  return calc_transition_probs(tau)[a][b];
}

auto Ctmc_bridge_sampler::build_jump_weights(Real_seq_letter a, Real_seq_letter b, double tau)
    -> double {

  weights_.clear();
  auto omega_tau = omega_ * tau;

  // The forward row vector e_a^T R^k.
  auto row = Seq_vector<double>{0.0};
  row[a] = 1.0;

  auto log_omega_tau = std::log(omega_tau);
  auto poisson_mass = 0.0;
  auto total = 0.0;
  for (auto k = 0; k <= k_max_jumps; ++k) {
    auto p_k = std::exp(-omega_tau + k * log_omega_tau - std::lgamma(k + 1.0));
    poisson_mass += p_k;
    auto w_k = p_k * row[b];
    weights_.push_back(w_k);
    total += w_k;
    auto tail_bound = std::max(0.0, 1.0 - poisson_mass);
    if (k >= omega_tau
        && (p_k == 0.0 || tail_bound == 0.0 || tail_bound <= k_series_relative_tolerance * total)) {
      return total;
    }
    auto next = Seq_vector<double>{0.0};
    for (auto s : k_all_real_seq_letters) {
      auto v = row[s];
      if (v == 0.0) { continue; }
      for (auto t : k_all_real_seq_letters) {
        next[t] += v * r_ab_[s][t];
      }
    }
    row = next;
  }

  throw std::domain_error(absl::StrFormat(
      "The uniformized chain needs more than %d jumps at Omega*tau = %g; the rate or the duration is"
      " out of the supported range",
      k_max_jumps, omega_tau));
}

auto Ctmc_bridge_sampler::sample(
    Real_seq_letter a,
    Real_seq_letter b,
    double T,
    double rate_scale,
    absl::BitGenRef bitgen,
    std::vector<Ctmc_bridge_event>& out)
    -> void {

  if (not(T >= 0.0) || not std::isfinite(T)) {
    throw std::domain_error(absl::StrFormat("Branch duration %g is not a finite non-negative number", T));
  }
  if (not(rate_scale >= 0.0) || not std::isfinite(rate_scale)) {
    throw std::domain_error(absl::StrFormat("Rate scale %g is not a finite non-negative number", rate_scale));
  }

  auto tau = rate_scale * T;
  if (not(omega_ * tau > 0.0)) {
    // No time and no rate: the only path is the empty one, and it exists only between equal states.
    if (a != b) {
      throw std::domain_error(absl::StrFormat(
          "No path can take %c to %c across a branch of effective duration zero", to_char(a), to_char(b)));
    }
    return;
  }

  auto total = build_jump_weights(a, b, tau);
  if (not(total > 0.0)) {
    throw std::domain_error(absl::StrFormat(
        "The transition %c to %c has probability zero at Omega*tau = %g under this rate matrix",
        to_char(a), to_char(b), omega_ * tau));
  }

  // Draw K from the normalized weights.  Normalizing by the computed sum rather than by a separately
  // computed [exp(Q tau)]_ab keeps the draw consistent with the weights to the last bit.
  auto target = absl::Uniform(absl::IntervalOpenOpen, bitgen, 0.0, total);
  auto num_jumps = -1;
  auto cumulative = 0.0;
  for (auto k = 0; k != std::ssize(weights_); ++k) {
    cumulative += weights_[k];
    if (cumulative >= target) {
      num_jumps = k;
      break;
    }
  }
  if (num_jumps < 0) {
    // Rounding can leave the cumulative sum just below `target`.  The leftover belongs to the
    // largest index that carries weight.
    num_jumps = static_cast<int>(std::ssize(weights_)) - 1;
    while (num_jumps > 0 && weights_[num_jumps] == 0.0) { --num_jumps; }
  }

  if (num_jumps == 0) {
    // The endpoints must agree; `build_jump_weights` gives weight zero to k = 0 otherwise.
    return;
  }

  // Backward column vectors c_j = R^j e_b, needed to walk the discrete-time bridge forwards.
  backward_.clear();
  backward_.resize(num_jumps);
  auto column = Seq_vector<double>{0.0};
  column[b] = 1.0;
  backward_[0] = column;
  for (auto j = 1; j != num_jumps; ++j) {
    auto next = Seq_vector<double>{0.0};
    for (auto s : k_all_real_seq_letters) {
      auto sum = 0.0;
      for (auto t : k_all_real_seq_letters) {
        sum += r_ab_[s][t] * column[t];
      }
      next[s] = sum;
    }
    backward_[j] = next;
    column = next;
  }

  // The k jump times are the order statistics of k independent uniforms on the branch.
  times_.clear();
  for (auto i = 0; i != num_jumps; ++i) {
    times_.push_back(absl::Uniform(absl::IntervalOpenOpen, bitgen, 0.0, T));
  }
  std::ranges::sort(times_);

  // Walk the discrete bridge: P(X_i = s | X_{i-1} = r, X_k = b) proportional to R_rs (R^{k-i})_sb.
  auto prev = a;
  for (auto i = 1; i <= num_jumps; ++i) {
    auto next = b;
    if (i != num_jumps) {
      const auto& c = backward_[num_jumps - i];
      auto weight = Seq_vector<double>{0.0};
      auto weight_total = 0.0;
      for (auto s : k_all_real_seq_letters) {
        auto w = r_ab_[prev][s] * c[s];
        weight[s] = w;
        weight_total += w;
      }
      // weight_total > 0 because the bridge was drawn with positive weight at this k.
      auto u = absl::Uniform(absl::IntervalOpenOpen, bitgen, 0.0, weight_total);
      auto cum = 0.0;
      next = Real_seq_letter::T;
      for (auto s : k_all_real_seq_letters) {
        cum += weight[s];
        if (cum >= u) { next = s; break; }
      }
    }
    if (next != prev) {
      out.push_back(Ctmc_bridge_event{times_[i - 1], next});
      prev = next;
    }
  }

  // The removal of the self-jumps leaves the path at `b`, by construction of the bridge.
}

}  // namespace delphy
