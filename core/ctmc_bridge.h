#ifndef DELPHY_CTMC_BRIDGE_H_
#define DELPHY_CTMC_BRIDGE_H_

#include <vector>

#include "absl/random/bit_gen_ref.h"

#include "sequence.h"

namespace delphy {

// A substitution event of a sampled path: the chain enters state `to` at time `t`, measured on the
// time axis the caller asked for.
struct Ctmc_bridge_event {
  double t;
  Real_seq_letter to;

  auto operator<=>(const Ctmc_bridge_event& that) const = default;
};

// Exact endpoint-conditioned sampling of a four-state continuous-time Markov chain, by the
// uniformization construction of Hobolth and Stone (2009), Simulation from endpoint-conditioned
// continuous-time Markov chains, doi:10.1214/09-AOAS247.
//
// With Omega >= max_a (-Q_aa) and R = I + Q/Omega a stochastic matrix, a CTMC with rate matrix Q is
// the discrete chain R observed at the jumps of a Poisson process of rate Omega, some of those
// jumps being self-jumps of R.  Conditioning on the endpoints gives
//
//   P(K = k | X(0) = a, X(tau) = b) = Pois(k; Omega tau) (R^k)_ab / [exp(Q tau)]_ab,
//
// after which the k intermediate states are a discrete-time Markov bridge of R and the k jump times
// are the order statistics of k independent uniforms on the interval.  Removing the self-jumps of R
// leaves a path drawn exactly from P(H | X(0) = a, X(tau) = b, Q); no approximation and no
// truncation of the distribution of K is involved.
//
// The sampler is built once per rate matrix and then used at many durations and rate scales: the
// path of a site whose effective rate matrix is `rate_scale * rate_ab` over a branch of duration `T`
// is sampled by passing that `T` and that `rate_scale`, which only enter as the product
// tau = rate_scale * T, so one sampler serves every site of a partition whatever its relative rate.
class Ctmc_bridge_sampler {
 public:
  // `rate_ab` must have non-negative off-diagonal entries and rows summing to zero.  Throws
  // std::invalid_argument otherwise.  A rate matrix that is identically zero is accepted; it admits
  // only the empty path between equal endpoints.
  explicit Ctmc_bridge_sampler(const Seq_matrix<double>& rate_ab);

  auto rate_ab() const -> const Seq_matrix<double>& { return rate_ab_; }

  // The uniformization constant, max_a (-Q_aa).
  auto omega() const -> double { return omega_; }

  // exp(rate_ab * tau), by the same series.  Rows sum to one to within rounding.
  auto calc_transition_probs(double tau) const -> Seq_matrix<double>;

  // [exp(rate_ab * tau)]_ab.
  auto calc_transition_prob(Real_seq_letter a, Real_seq_letter b, double tau) const -> double;

  // Appends to `out`, in increasing time, the substitution events of a path drawn from
  // P(H | X(0) = a, X(T) = b) under the rate matrix `rate_scale * rate_ab()`.  Every event time is
  // strictly inside (0, T) and consecutive events never repeat a state, so the path starts at `a`
  // and ends at `b`.  `out` is appended to, not cleared.
  //
  // Throws std::domain_error if `T` or `rate_scale` is negative or not finite, or if the
  // conditioning event has probability zero -- a != b with tau == 0, a != b that no sequence of
  // allowed substitutions connects, or a duration so long that the number of uniformized jumps
  // exceeds `k_max_jumps`.
  auto sample(
      Real_seq_letter a,
      Real_seq_letter b,
      double T,
      double rate_scale,
      absl::BitGenRef bitgen,
      std::vector<Ctmc_bridge_event>& out)
      -> void;

  // Hard ceiling on the number of uniformized jumps.  The conditional distribution of K has mass
  // beyond any finite bound, but its mass above Omega tau + O(sqrt(Omega tau)) underflows to zero in
  // double precision long before this; the ceiling exists so that a nonsensical rate or duration
  // raises instead of exhausting memory.
  static constexpr int k_max_jumps = 1 << 22;

 private:
  // Fills `weights_` with w_k = Pois(k; tau) (R^k)_ab for k = 0, 1, ... up to the point where the
  // remaining Poisson mass has underflowed, and returns sum_k w_k, which is [exp(Q tau)]_ab.
  auto build_jump_weights(Real_seq_letter a, Real_seq_letter b, double tau) -> double;

  Seq_matrix<double> rate_ab_;
  Seq_matrix<double> r_ab_;  // I + Q/Omega
  double omega_;

  // Reused across calls to keep the move free of allocation in steady state.
  std::vector<double> weights_;
  std::vector<Seq_vector<double>> backward_;
  std::vector<double> times_;
};

}  // namespace delphy

#endif // DELPHY_CTMC_BRIDGE_H_
