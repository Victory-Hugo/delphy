#ifndef DELPHY_TESTS_INDEPENDENT_SEQ_DENSITY_H_
#define DELPHY_TESTS_INDEPENDENT_SEQ_DENSITY_H_

#include <cmath>
#include <utility>
#include <vector>

#include "evo_model.h"
#include "phylo_tree.h"

namespace delphy {

//
// Likelihood: an independent per-site implementation of the explicit-mutation-annotated-tree density
//
// It shares no code with calc_log_root_prior / calc_log_G_below_root: every site is walked separately, the
// state is tracked explicitly along each branch, and a branch whose endpoint declares the site missing is
// skipped together with its whole subtree (a missing site contributes nothing, which is the marginalization
// over the unobserved state).
//
inline auto independent_log_G(const Phylo_tree& tree, const Global_evo_model& evo) -> double {
  auto total = 0.0;
  for (auto l = Site_index{0}; l != tree.num_sites(); ++l) {
    if (tree.at_root().missations.contains(l)) { continue; }
    auto root_state = tree.ref_sequence[l];
    for (const auto& m : tree.at_root().mutations) {
      if (m.site == l) { root_state = m.to; }
    }
    total += std::log(evo.pi_l_a(l, root_state));

    auto stack = std::vector<std::pair<Node_index, Real_seq_letter>>{{tree.root, root_state}};
    while (not stack.empty()) {
      auto [p, state] = stack.back();
      stack.pop_back();
      for (auto c : tree.at(p).children) {
        if (tree.at(c).missations.contains(l)) { continue; }
        auto cur = state;
        auto t = tree.at(p).t;
        for (const auto& m : tree.at(c).mutations) {
          if (m.site != l) { continue; }
          total -= evo.mu_l(l) * evo.nu_l[l] * evo.q_l_a(l, cur) * (m.t - t);
          total += std::log(evo.mu_l(l) * evo.nu_l[l] * evo.q_l_ab(l, cur, m.to));
          cur = m.to;
          t = m.t;
        }
        total -= evo.mu_l(l) * evo.nu_l[l] * evo.q_l_a(l, cur) * (tree.at(c).t - t);
        stack.push_back({c, cur});
      }
    }
  }
  return total;
}

}  // namespace delphy

#endif  // DELPHY_TESTS_INDEPENDENT_SEQ_DENSITY_H_
