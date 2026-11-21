#include "evo_hky.h"

#include "evo_subst_model.h"

namespace delphy {

auto Hky_model::derive_site_evo_model() const -> Site_evo_model {
  using enum Real_seq_letter;

  // HKY85 is the exchangeability grouping 010010: kappa for the transitions A<->G and C<->T, 1 for
  // the four transversions.  The matrix is built by the shared reversible-model builder, so every
  // model in the family goes through one implementation.
  auto r_ab = Seq_matrix<double>{0.0};
  for (auto [a, b] : {std::pair{A, C}, std::pair{A, T}, std::pair{C, G}, std::pair{G, T}}) {
    r_ab[a][b] = 1.0;
    r_ab[b][a] = 1.0;
  }
  for (auto [a, b] : {std::pair{A, G}, std::pair{C, T}}) {
    r_ab[a][b] = kappa;
    r_ab[b][a] = kappa;
  }
  return derive_site_evo_model_from_exchangeabilities(mu, r_ab, pi_a);
}

}  // namespace delphy
