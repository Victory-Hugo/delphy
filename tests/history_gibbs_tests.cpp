#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <vector>

#include "absl/random/random.h"

#include "history_gibbs.h"
#include "phylo_tree_calc.h"

namespace delphy {

namespace {

inline constexpr auto rA = Real_seq_letter::A;
inline constexpr auto rC = Real_seq_letter::C;
inline constexpr auto rG = Real_seq_letter::G;
inline constexpr auto rT = Real_seq_letter::T;

// Independent reference arithmetic, sharing no code with ctmc_bridge.cpp or history_gibbs.cpp.
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

auto ref_mat_exp(const Seq_matrix<double>& q, double scale) -> Mat {
  auto m = Mat{};
  auto norm = 0.0L;
  for (auto a : k_all_real_seq_letters) {
    for (auto b : k_all_real_seq_letters) {
      m[index_of(a)][index_of(b)] = static_cast<long double>(q[a][b]) * static_cast<long double>(scale);
      norm = std::max(norm, std::fabs(m[index_of(a)][index_of(b)]));
    }
  }
  auto squarings = 0;
  while (norm > 0.05L) { norm /= 2; ++squarings; }
  for (auto& row : m) { for (auto& v : row) { v /= std::pow(2.0L, squarings); } }

  auto term = Mat{};
  auto sum = Mat{};
  for (auto i = 0; i != 4; ++i) { term[i][i] = 1.0L; sum[i][i] = 1.0L; }
  for (auto k = 1; k != 40; ++k) {
    term = mat_mul(term, m);
    for (auto i = 0; i != 4; ++i) {
      for (auto j = 0; j != 4; ++j) { sum[i][j] += term[i][j] / std::tgamma(static_cast<long double>(k + 1)); }
    }
  }
  for (auto s = 0; s != squarings; ++s) { sum = mat_mul(sum, sum); }
  return sum;
}

// A reversible rate matrix from exchangeabilities (AC, AG, AT, CG, CT, GT) and frequencies.
auto make_rate_matrix(const std::array<double, 6>& s, const Seq_vector<double>& pi) -> Seq_matrix<double> {
  auto pairs = std::array<std::pair<Real_seq_letter, Real_seq_letter>, 6>{
      {{rA, rC}, {rA, rG}, {rA, rT}, {rC, rG}, {rC, rT}, {rG, rT}}};
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

// The full posterior weight of the genealogy as Delphy computes it, from scratch.
auto full_log_G(const Phylo_tree& tree, const Global_evo_model& evo) -> double {
  return calc_log_root_prior(tree, evo) + calc_log_G_below_root(tree, evo);
}

auto site_states_at_tips(const Phylo_tree& tree, Site_index num_sites) -> std::vector<Real_seq_letter> {
  auto out = std::vector<Real_seq_letter>{};
  for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
    if (not tree.at(node).is_tip()) { continue; }
    for (auto l = Site_index{0}; l != num_sites; ++l) {
      if (is_site_missing_at(tree, node, l)) { continue; }
      out.push_back(calc_site_state_at(tree, tree.node_loc(node), l));
    }
  }
  return out;
}

auto count_site_mutations(const Phylo_tree& tree, Site_index l) -> int {
  auto n = 0;
  for (auto node = Node_index{0}; node != std::ssize(tree); ++node) {
    if (node == tree.root) { continue; }
    for (const auto& m : tree.at(node).mutations) { if (m.site == l) { ++n; } }
  }
  return n;
}

auto root_site_state(const Phylo_tree& tree, Site_index l) -> Real_seq_letter {
  auto state = tree.ref_sequence[l];
  for (const auto& m : tree.at(tree.root).mutations) { if (m.site == l) { state = m.to; } }
  return state;
}

// Tree fixtures
// =============
//
//   r (root, t = -2) --- x (t = -0.5) --- a (t = 1.0)
//                     |                \ b (t = 0.5)
//                     \ c (t = 1.5)
//
// Two sites in one partition, so that a site-by-site update has to leave the other site alone.
class History_gibbs_test : public testing::Test {
 protected:
  static constexpr Node_index r = 0;
  static constexpr Node_index x = 1;
  static constexpr Node_index a = 2;
  static constexpr Node_index b = 3;
  static constexpr Node_index c = 4;
  static constexpr Site_index num_sites = 2;

  Phylo_tree tree{5};
  Global_evo_model evo = make_global_evo_model(Site_vector<Partition_index>{0, 0});

  History_gibbs_test() {
    evo.nu_l = {0.8, 1.2};
    evo.partition_evo_model[0] = {
        .mu = 0.35,
        .pi_a = {0.4, 0.1, 0.25, 0.25},
        .q_ab = make_rate_matrix({0.4, 3.1, 0.7, 1.3, 2.6, 1.0}, Seq_vector<double>{0.4, 0.1, 0.25, 0.25}),
    };

    tree.root = r;
    tree.ref_sequence = Real_sequence{rA, rC};

    auto set_node = [this](Node_index node, const char* name, Node_index parent, double t, bool inner) {
      tree.at(node).name = name;
      tree.at(node).parent = parent;
      tree.at(node).t = t;
      tree.at(node).t_min = inner ? -std::numeric_limits<float>::max() : static_cast<float>(t);
      tree.at(node).t_max = inner ? +std::numeric_limits<float>::max() : static_cast<float>(t);
    };
    set_node(r, "r", k_no_node, -2.0, true);
    set_node(x, "x", r, -0.5, true);
    set_node(a, "a", x, 1.0, false);
    set_node(b, "b", x, 0.5, false);
    set_node(c, "c", r, 1.5, false);
    tree.at(r).children = {x, c};
    tree.at(x).children = {a, b};

    // A parsimonious starting history: one substitution at site 0 on the branch to `a`, one at site 1
    // on the branch to `c`.
    tree.at(a).mutations = {Mutation{rA, 0, rG, 0.25}};
    tree.at(c).mutations = {Mutation{rC, 1, rT, 0.5}};

    assert_phylo_tree_integrity(tree, true);
  }
};

}  // namespace

TEST_F(History_gibbs_test, branch_update_keeps_the_endpoints_and_the_rest_of_the_tree) {
  auto sampler = Site_history_sampler{evo};
  auto bitgen = std::mt19937_64{20261114};
  auto want_tips = site_states_at_tips(tree, num_sites);

  for (auto rep = 0; rep != 2000; ++rep) {
    for (auto branch : {x, a, b, c}) {
      for (auto l = Site_index{0}; l != num_sites; ++l) {
        auto state_at_parent = calc_site_state_at(tree, tree.node_loc(tree.at(branch).parent), l);
        auto before = full_log_G(tree, evo);
        auto other_site_before = count_site_mutations(tree, 1 - l);
        auto result = gibbs_resample_branch_site_history(tree, branch, l, state_at_parent, sampler, bitgen);
        ASSERT_TRUE(result.performed);
        auto after = full_log_G(tree, evo);
        // The increment the move reports must be the change in the whole posterior weight.
        EXPECT_NEAR(result.delta_log_G, after - before, 1e-9)
            << "branch " << branch << ", site " << l;
        // Nothing but the chosen site on the chosen branch may change.
        EXPECT_EQ(count_site_mutations(tree, 1 - l), other_site_before);
        assert_phylo_tree_integrity(tree, true);
        EXPECT_EQ(site_states_at_tips(tree, num_sites), want_tips);
      }
    }
  }
}

TEST_F(History_gibbs_test, branch_update_reproduces_the_conditional_bridge) {
  // With the endpoints of the branch to `a` fixed at A and G, the number of substitutions the update
  // writes must follow the endpoint-conditioned CTMC, which an independent rejection sampler gives.
  auto sampler = Site_history_sampler{evo};
  auto bitgen = std::mt19937_64{777};
  auto l = Site_index{0};
  auto t_X = tree.at(a).t;
  auto t_P = tree.at(x).t;
  auto rate = evo.mu_l(l) * evo.nu_l[l];
  const auto& q = evo.partition_evo_model[0].q_ab;

  const auto num_reps = 200000;
  auto got = std::map<int, int>{};
  for (auto rep = 0; rep != num_reps; ++rep) {
    auto result = gibbs_resample_branch_site_history(tree, a, l, rA, sampler, bitgen);
    ASSERT_TRUE(result.performed);
    ++got[result.num_new_events];
    EXPECT_EQ(calc_site_state_at(tree, tree.node_loc(a), l), rG);
  }

  // Independent reference: forward Gillespie simulation with rejection on the end state.
  auto ref_bitgen = std::mt19937_64{999};
  auto want = std::map<int, int>{};
  for (auto rep = 0; rep != num_reps; ++rep) {
    while (true) {
      auto state = rA;
      auto t = t_P;
      auto n = 0;
      while (true) {
        auto escape = rate * (-q[state][state]);
        t += std::exponential_distribution<double>{escape}(ref_bitgen);
        if (t >= t_X) { break; }
        auto u = std::uniform_real_distribution<double>{0.0, -q[state][state]}(ref_bitgen);
        auto cum = 0.0;
        auto next = state;
        for (auto s : k_all_real_seq_letters) {
          if (s == state) { continue; }
          cum += q[state][s];
          if (cum >= u) { next = s; break; }
        }
        state = next;
        ++n;
      }
      if (state == rG) { ++want[n]; break; }
    }
  }

  auto chi_square = 0.0;
  auto degrees = 0;
  for (auto n = 1; n <= 5; ++n) {
    auto pooled = static_cast<double>(got[n] + want[n]);
    if (pooled < 50) { continue; }
    auto expected = pooled / 2;
    chi_square += (got[n] - expected) * (got[n] - expected) / expected;
    chi_square += (want[n] - expected) * (want[n] - expected) / expected;
    ++degrees;
  }
  ASSERT_GE(degrees, 2);
  EXPECT_LT(chi_square, 5.0 * degrees + 20.0) << "chi-square " << chi_square << " on " << degrees << " bins";
}

TEST_F(History_gibbs_test, node_state_update_matches_the_independent_conditional) {
  // The conditional of the state at `x` given the states at `r`, `a` and `b` is
  // P_rx(s_r, s) P_xa(s, s_a) P_xb(s, s_b), which an independent matrix exponential gives.
  auto sampler = Site_history_sampler{evo};
  auto bitgen = std::mt19937_64{4242};
  auto l = Site_index{0};
  const auto& q = evo.partition_evo_model[0].q_ab;
  auto rate = evo.mu_l(l) * evo.nu_l[l];

  auto s_r = root_site_state(tree, l);
  auto s_a = calc_site_state_at(tree, tree.node_loc(a), l);
  auto s_b = calc_site_state_at(tree, tree.node_loc(b), l);

  auto p_rx = ref_mat_exp(q, rate * (tree.at(x).t - tree.at(r).t));
  auto p_xa = ref_mat_exp(q, rate * (tree.at(a).t - tree.at(x).t));
  auto p_xb = ref_mat_exp(q, rate * (tree.at(b).t - tree.at(x).t));
  auto want = Seq_vector<double>{};
  auto want_total = 0.0;
  for (auto s : k_all_real_seq_letters) {
    want[s] = static_cast<double>(
        p_rx[index_of(s_r)][index_of(s)] * p_xa[index_of(s)][index_of(s_a)]
        * p_xb[index_of(s)][index_of(s_b)]);
    want_total += want[s];
  }

  const auto num_reps = 200000;
  auto got = Seq_vector<int>{0};
  auto want_tips = site_states_at_tips(tree, num_sites);
  for (auto rep = 0; rep != num_reps; ++rep) {
    auto before = full_log_G(tree, evo);
    auto result = gibbs_resample_node_site_state(tree, x, l, root_site_state(tree, l), sampler, bitgen);
    ASSERT_TRUE(result.performed);
    auto after = full_log_G(tree, evo);
    EXPECT_NEAR(result.delta_log_G, after - before, 1e-9);
    assert_phylo_tree_integrity(tree, true);
    EXPECT_EQ(site_states_at_tips(tree, num_sites), want_tips);
    ++got[calc_site_state_at(tree, tree.node_loc(x), l)];
    // The states at the far ends of the incident branches must not have moved.
    EXPECT_EQ(root_site_state(tree, l), s_r);
    EXPECT_EQ(calc_site_state_at(tree, tree.node_loc(a), l), s_a);
    EXPECT_EQ(calc_site_state_at(tree, tree.node_loc(b), l), s_b);
  }

  for (auto s : k_all_real_seq_letters) {
    auto p = want[s] / want_total;
    auto observed = static_cast<double>(got[s]) / num_reps;
    auto tolerance = 5.0 * std::sqrt(std::max(p, 1e-5) * (1 - p) / num_reps);
    EXPECT_LT(std::abs(observed - p), tolerance)
        << "state " << to_char(s) << ": sampled " << observed << ", conditional " << p;
  }
}

TEST_F(History_gibbs_test, node_state_update_at_the_root_uses_the_root_prior) {
  auto sampler = Site_history_sampler{evo};
  auto bitgen = std::mt19937_64{31337};
  auto l = Site_index{1};
  const auto& q = evo.partition_evo_model[0].q_ab;
  const auto& pi_a = evo.partition_evo_model[0].pi_a;
  auto rate = evo.mu_l(l) * evo.nu_l[l];

  auto s_x = calc_site_state_at(tree, tree.node_loc(x), l);
  auto s_c = calc_site_state_at(tree, tree.node_loc(c), l);
  auto p_rx = ref_mat_exp(q, rate * (tree.at(x).t - tree.at(r).t));
  auto p_rc = ref_mat_exp(q, rate * (tree.at(c).t - tree.at(r).t));
  auto want = Seq_vector<double>{};
  auto want_total = 0.0;
  for (auto s : k_all_real_seq_letters) {
    want[s] = pi_a[s] * static_cast<double>(
        p_rx[index_of(s)][index_of(s_x)] * p_rc[index_of(s)][index_of(s_c)]);
    want_total += want[s];
  }

  const auto num_reps = 200000;
  auto got = Seq_vector<int>{0};
  auto want_tips = site_states_at_tips(tree, num_sites);
  for (auto rep = 0; rep != num_reps; ++rep) {
    auto before = full_log_G(tree, evo);
    auto result = gibbs_resample_node_site_state(tree, r, l, tree.ref_sequence[l], sampler, bitgen);
    ASSERT_TRUE(result.performed);
    auto after = full_log_G(tree, evo);
    EXPECT_NEAR(result.delta_log_G, after - before, 1e-9);
    assert_phylo_tree_integrity(tree, true);
    EXPECT_EQ(site_states_at_tips(tree, num_sites), want_tips);
    ++got[root_site_state(tree, l)];
    EXPECT_EQ(calc_site_state_at(tree, tree.node_loc(x), l), s_x);
    EXPECT_EQ(calc_site_state_at(tree, tree.node_loc(c), l), s_c);
  }

  for (auto s : k_all_real_seq_letters) {
    auto p = want[s] / want_total;
    auto observed = static_cast<double>(got[s]) / num_reps;
    auto tolerance = 5.0 * std::sqrt(std::max(p, 1e-5) * (1 - p) / num_reps);
    EXPECT_LT(std::abs(observed - p), tolerance)
        << "state " << to_char(s) << ": sampled " << observed << ", root conditional " << p;
  }
}

TEST_F(History_gibbs_test, node_state_update_reports_the_change_in_lambda) {
  auto sampler = Site_history_sampler{evo};
  auto bitgen = std::mt19937_64{5150};
  for (auto rep = 0; rep != 500; ++rep) {
    for (auto node : {r, x}) {
      for (auto l = Site_index{0}; l != num_sites; ++l) {
        auto before = calc_lambda_at_node(tree, node, evo);
        auto other_before = std::vector<double>{};
        for (auto other : {r, x, a, b, c}) {
          if (other != node) { other_before.push_back(calc_lambda_at_node(tree, other, evo)); }
        }
        auto result = gibbs_resample_node_site_state(tree, node, l,
            node == r ? tree.ref_sequence[l] : calc_site_state_at(tree, tree.node_loc(tree.at(node).parent), l),
            sampler, bitgen);
        ASSERT_TRUE(result.performed);
        EXPECT_NEAR(calc_lambda_at_node(tree, node, evo), before + result.delta_lambda_at_node, 1e-12);
        auto i = 0;
        for (auto other : {r, x, a, b, c}) {
          if (other != node) {
            EXPECT_NEAR(calc_lambda_at_node(tree, other, evo), other_before[i], 1e-12) << other;
            ++i;
          }
        }
      }
    }
  }
}

TEST_F(History_gibbs_test, site_refresh_matches_the_independent_joint_posterior) {
  // With the topology, the times and the parameters fixed, the joint posterior of the states at `r`
  // and `x` at one site is pi(s_r) P_rx(s_r, s_x) P_xa(s_x, s_a) P_xb(s_x, s_b) P_rc(s_r, s_c), which
  // an independent matrix exponential gives by enumeration over the sixteen pairs.
  auto sampler = Site_history_sampler{evo};
  auto bitgen = std::mt19937_64{86420};
  auto l = Site_index{0};
  const auto& q = evo.partition_evo_model[0].q_ab;
  const auto& pi_a = evo.partition_evo_model[0].pi_a;
  auto rate = evo.mu_l(l) * evo.nu_l[l];

  auto s_a = calc_site_state_at(tree, tree.node_loc(a), l);
  auto s_b = calc_site_state_at(tree, tree.node_loc(b), l);
  auto s_c = calc_site_state_at(tree, tree.node_loc(c), l);
  auto p_rx = ref_mat_exp(q, rate * (tree.at(x).t - tree.at(r).t));
  auto p_rc = ref_mat_exp(q, rate * (tree.at(c).t - tree.at(r).t));
  auto p_xa = ref_mat_exp(q, rate * (tree.at(a).t - tree.at(x).t));
  auto p_xb = ref_mat_exp(q, rate * (tree.at(b).t - tree.at(x).t));

  auto want = std::vector<double>(16, 0.0);
  auto want_total = 0.0;
  for (auto s_r : k_all_real_seq_letters) {
    for (auto s_x : k_all_real_seq_letters) {
      auto w = pi_a[s_r] * static_cast<double>(
          p_rx[index_of(s_r)][index_of(s_x)] * p_rc[index_of(s_r)][index_of(s_c)]
          * p_xa[index_of(s_x)][index_of(s_a)] * p_xb[index_of(s_x)][index_of(s_b)]);
      want[index_of(s_r) * 4 + index_of(s_x)] = w;
      want_total += w;
    }
  }

  const auto num_reps = 200000;
  auto got = std::vector<int>(16, 0);
  auto want_tips = site_states_at_tips(tree, num_sites);
  auto other_site_before = count_site_mutations(tree, 1);
  for (auto rep = 0; rep != num_reps; ++rep) {
    gibbs_refresh_site_history(tree, l, sampler, bitgen);
    assert_phylo_tree_integrity(tree, true);
    EXPECT_EQ(site_states_at_tips(tree, num_sites), want_tips);
    EXPECT_EQ(count_site_mutations(tree, 1), other_site_before);
    auto s_r = root_site_state(tree, l);
    auto s_x = calc_site_state_at(tree, tree.node_loc(x), l);
    ++got[index_of(s_r) * 4 + index_of(s_x)];
  }

  for (auto i = 0; i != 16; ++i) {
    auto p = want[i] / want_total;
    auto observed = static_cast<double>(got[i]) / num_reps;
    auto tolerance = 5.0 * std::sqrt(std::max(p, 1e-5) * (1 - p) / num_reps);
    EXPECT_LT(std::abs(observed - p), tolerance)
        << "pair " << i << ": sampled " << observed << ", joint posterior " << p;
  }
}

TEST_F(History_gibbs_test, missing_sites_are_left_alone_and_their_from_states_follow) {
  // Site 1 goes missing below `c`, so the branch to `c` carries no history for it, and site 1 is
  // missing at the tip `c`.
  tree.at(c).mutations = {};
  tree.at(c).missations.insert(Missation{1, rC}, tree.ref_sequence);
  assert_phylo_tree_integrity(tree, true);

  auto sampler = Site_history_sampler{evo};
  auto bitgen = std::mt19937_64{13579};
  auto want_tips = site_states_at_tips(tree, num_sites);

  for (auto rep = 0; rep != 2000; ++rep) {
    // The branch update at the missing site on the missing branch is the identity.
    auto result = gibbs_resample_branch_site_history(tree, c, 1, rC, sampler, bitgen);
    EXPECT_FALSE(result.performed);
    EXPECT_EQ(result.delta_log_G, 0.0);
    EXPECT_TRUE(tree.at(c).mutations.empty());

    // Resampling the root state at the missing site has to carry the state entering the missing
    // region with it, so that the posterior weight stays consistent.
    auto before = full_log_G(tree, evo);
    auto root_result = gibbs_resample_node_site_state(tree, r, 1, tree.ref_sequence[1], sampler, bitgen);
    ASSERT_TRUE(root_result.performed);
    EXPECT_NEAR(root_result.delta_log_G, full_log_G(tree, evo) - before, 1e-9);
    EXPECT_EQ(tree.at(c).missations.get_from_state(1, tree.ref_sequence), root_site_state(tree, 1));
    assert_phylo_tree_integrity(tree, true);
    EXPECT_EQ(site_states_at_tips(tree, num_sites), want_tips);

    // The whole-site refresh must also respect the missing subtree.
    before = full_log_G(tree, evo);
    gibbs_refresh_site_history(tree, 1, sampler, bitgen);
    EXPECT_TRUE(tree.at(c).mutations.empty());
    EXPECT_EQ(tree.at(c).missations.get_from_state(1, tree.ref_sequence), root_site_state(tree, 1));
    assert_phylo_tree_integrity(tree, true);
    EXPECT_EQ(site_states_at_tips(tree, num_sites), want_tips);
  }
}

TEST_F(History_gibbs_test, hidden_substitutions_appear_where_the_endpoints_agree) {
  // Site 1 is C everywhere in the starting history except below `c`; the updates must still be able
  // to put pairs of substitutions on the branches, which is what the all-sites proposal cannot do at
  // high divergence.
  auto sampler = Site_history_sampler{evo};
  auto bitgen = std::mt19937_64{24680};
  auto max_events = 0;
  auto num_with_hidden = 0;
  const auto num_reps = 20000;
  for (auto rep = 0; rep != num_reps; ++rep) {
    auto result = gibbs_resample_branch_site_history(tree, b, 0, rA, sampler, bitgen);
    ASSERT_TRUE(result.performed);
    EXPECT_NE(result.num_new_events, 1);  // the endpoints of this branch agree
    if (result.num_new_events > 0) { ++num_with_hidden; }
    max_events = std::max(max_events, result.num_new_events);
  }
  EXPECT_GT(num_with_hidden, 0);
  EXPECT_GE(max_events, 2);
}

TEST_F(History_gibbs_test, a_chain_of_updates_recovers_the_independent_posterior_of_the_event_count) {
  // Running the three updates together must reproduce the exact conditional expectation of the
  // number of substitutions at a site, summed over the internal states with their posterior weights.
  // Everything on the right-hand side comes from the independent matrix exponential and a Simpson
  // quadrature of the convolution integral.
  auto l = Site_index{0};
  const auto& q = evo.partition_evo_model[0].q_ab;
  const auto& pi_a = evo.partition_evo_model[0].pi_a;
  auto rate = evo.mu_l(l) * evo.nu_l[l];

  auto s_a = calc_site_state_at(tree, tree.node_loc(a), l);
  auto s_b = calc_site_state_at(tree, tree.node_loc(b), l);
  auto s_c = calc_site_state_at(tree, tree.node_loc(c), l);

  auto branch_tau = std::map<Node_index, double>{
      {x, rate * (tree.at(x).t - tree.at(r).t)},
      {a, rate * (tree.at(a).t - tree.at(x).t)},
      {b, rate * (tree.at(b).t - tree.at(x).t)},
      {c, rate * (tree.at(c).t - tree.at(r).t)},
  };

  // E[N | endpoints] on one branch, by Simpson quadrature.
  auto expected_on_branch = [&](double tau, Real_seq_letter from, Real_seq_letter to) -> double {
    const auto num_panels = 400;
    auto p_full = ref_mat_exp(q, tau);
    auto integral = Mat{};
    auto h = static_cast<long double>(tau) / num_panels;
    for (auto n = 0; n <= num_panels; ++n) {
      auto s = h * n;
      auto weight = (n == 0 || n == num_panels) ? 1.0L : ((n % 2 == 1) ? 4.0L : 2.0L);
      auto p_s = ref_mat_exp(q, static_cast<double>(s));
      auto p_rest = ref_mat_exp(q, static_cast<double>(static_cast<long double>(tau) - s));
      for (auto i = 0; i != 4; ++i) {
        for (auto j = 0; j != 4; ++j) {
          integral[i][j] += weight * p_s[index_of(from)][i] * p_rest[j][index_of(to)];
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
    return static_cast<double>(total / p_full[index_of(from)][index_of(to)]);
  };

  auto p_rx = ref_mat_exp(q, branch_tau[x]);
  auto p_rc = ref_mat_exp(q, branch_tau[c]);
  auto p_xa = ref_mat_exp(q, branch_tau[a]);
  auto p_xb = ref_mat_exp(q, branch_tau[b]);

  auto want = 0.0;
  auto total_weight = 0.0;
  for (auto s_r : k_all_real_seq_letters) {
    for (auto s_x : k_all_real_seq_letters) {
      auto w = pi_a[s_r] * static_cast<double>(
          p_rx[index_of(s_r)][index_of(s_x)] * p_rc[index_of(s_r)][index_of(s_c)]
          * p_xa[index_of(s_x)][index_of(s_a)] * p_xb[index_of(s_x)][index_of(s_b)]);
      if (w <= 0.0) { continue; }
      auto n = expected_on_branch(branch_tau[x], s_r, s_x)
          + expected_on_branch(branch_tau[c], s_r, s_c)
          + expected_on_branch(branch_tau[a], s_x, s_a)
          + expected_on_branch(branch_tau[b], s_x, s_b);
      want += w * n;
      total_weight += w;
    }
  }
  want /= total_weight;

  auto sampler = Site_history_sampler{evo};
  auto bitgen = std::mt19937_64{192837465};
  const auto num_reps = 400000;
  auto sum = 0.0;
  auto sum_sq = 0.0;
  for (auto rep = 0; rep != num_reps; ++rep) {
    // One sweep of the three updates, in the order a random scan would visit them.
    auto branch = std::array<Node_index, 4>{x, a, b, c}[rep % 4];
    auto state_at_parent = calc_site_state_at(tree, tree.node_loc(tree.at(branch).parent), l);
    gibbs_resample_branch_site_history(tree, branch, l, state_at_parent, sampler, bitgen);
    gibbs_resample_node_site_state(
        tree, (rep % 2 == 0) ? x : r, l,
        (rep % 2 == 0) ? root_site_state(tree, l) : tree.ref_sequence[l], sampler, bitgen);
    if (rep % 50 == 0) { gibbs_refresh_site_history(tree, l, sampler, bitgen); }
    auto n = static_cast<double>(count_site_mutations(tree, l));
    sum += n;
    sum_sq += n * n;
  }
  auto mean = sum / num_reps;
  auto variance = std::max(sum_sq / num_reps - mean * mean, 1e-12);
  // The sweeps are correlated, so allow for an effective sample size a hundred times smaller.
  auto stderr_mean = std::sqrt(variance * 100.0 / num_reps);
  EXPECT_LT(std::abs(mean - want), 4 * stderr_mean)
      << "chain gives " << mean << " +/- " << stderr_mean << ", independent reference " << want;
}


TEST_F(History_gibbs_test, the_local_marginal_likelihood_matches_an_independent_calculation) {
  // log F(V) must equal sum over sites of log( sum_s w(s) prod_children P(s, s_child) ), with w the
  // transition across the parent branch or the root prior, from an independent matrix exponential.
  auto sampler = Site_history_sampler{evo};
  const auto& q = evo.partition_evo_model[0].q_ab;
  const auto& pi_a = evo.partition_evo_model[0].pi_a;

  auto independent_at = [&](Node_index V, double t_V) {
    auto total = 0.0L;
    auto is_root = (V == r);
    for (auto l = Site_index{0}; l != num_sites; ++l) {
      auto rate = evo.mu_l(l) * evo.nu_l[l];
      auto weight = std::array<long double, 4>{};
      if (is_root) {
        for (auto s : k_all_real_seq_letters) { weight[index_of(s)] = pi_a[s]; }
      } else {
        auto p_parent = ref_mat_exp(q, rate * (t_V - tree.at(tree.at(V).parent).t));
        auto s_parent = calc_site_state_at(tree, tree.node_loc(tree.at(V).parent), l);
        for (auto s : k_all_real_seq_letters) {
          weight[index_of(s)] = p_parent[index_of(s_parent)][index_of(s)];
        }
      }
      for (auto C : tree.at(V).children) {
        auto p_child = ref_mat_exp(q, rate * (tree.at(C).t - t_V));
        auto s_child = calc_site_state_at(tree, tree.node_loc(C), l);
        for (auto s : k_all_real_seq_letters) {
          weight[index_of(s)] *= p_child[index_of(s)][index_of(s_child)];
        }
      }
      auto sum = 0.0L;
      for (auto w : weight) { sum += w; }
      total += std::log(sum);
    }
    return static_cast<double>(total);
  };

  // The node's own time, and two others inside the interval the move may propose in: for `x` that is
  // (t_r, min(t_a, t_b)) = (-2, 0.5), and for the root it is everything above its oldest child, -0.5.
  auto times = std::map<Node_index, std::vector<double>>{
      {x, {tree.at(x).t, -1.5, 0.0}},
      {r, {tree.at(r).t, -5.0, -0.75}}};
  for (auto V : {x, r}) {
    for (auto t : times[V]) {
      auto got = calc_local_marginal_log_likelihood(tree, V, t, read_incident_sequences(tree, V), sampler);
      EXPECT_NEAR(got, independent_at(V, t), 1e-10)
          << "node " << V << " at t = " << t;
    }
  }
}

TEST_F(History_gibbs_test, the_node_refresh_keeps_the_tree_legal_and_the_far_endpoints_fixed) {
  auto sampler = Site_history_sampler{evo};
  auto bitgen = std::mt19937_64{606060};
  auto want_tips = site_states_at_tips(tree, num_sites);

  for (auto rep = 0; rep != 500; ++rep) {
    for (auto V : {x, r}) {
      auto s_far = std::vector<Real_seq_letter>{};
      for (auto l = Site_index{0}; l != num_sites; ++l) {
        for (auto C : tree.at(V).children) { s_far.push_back(calc_site_state_at(tree, tree.node_loc(C), l)); }
        if (V != r) { s_far.push_back(calc_site_state_at(tree, tree.node_loc(tree.at(V).parent), l)); }
      }
      auto result = gibbs_refresh_node_histories(tree, V, read_incident_sequences(tree, V), sampler, bitgen);
      EXPECT_GE(result.num_new_events, 0);
      assert_phylo_tree_integrity(tree, true);
      EXPECT_EQ(site_states_at_tips(tree, num_sites), want_tips);
      auto s_far_after = std::vector<Real_seq_letter>{};
      for (auto l = Site_index{0}; l != num_sites; ++l) {
        for (auto C : tree.at(V).children) {
          s_far_after.push_back(calc_site_state_at(tree, tree.node_loc(C), l));
        }
        if (V != r) {
          s_far_after.push_back(calc_site_state_at(tree, tree.node_loc(tree.at(V).parent), l));
        }
      }
      EXPECT_EQ(s_far, s_far_after) << "node " << V;
    }
  }
}

TEST_F(History_gibbs_test, the_node_refresh_reproduces_the_conditional_state_at_the_node) {
  // The refresh draws the state at the node from the same conditional as the single-site update, so
  // its empirical distribution must match the independent calculation of that conditional.
  auto sampler = Site_history_sampler{evo};
  auto bitgen = std::mt19937_64{70707};
  auto l = Site_index{0};
  const auto& q = evo.partition_evo_model[0].q_ab;
  auto rate = evo.mu_l(l) * evo.nu_l[l];

  auto s_r = root_site_state(tree, l);
  auto s_a = calc_site_state_at(tree, tree.node_loc(a), l);
  auto s_b = calc_site_state_at(tree, tree.node_loc(b), l);
  auto p_rx = ref_mat_exp(q, rate * (tree.at(x).t - tree.at(r).t));
  auto p_xa = ref_mat_exp(q, rate * (tree.at(a).t - tree.at(x).t));
  auto p_xb = ref_mat_exp(q, rate * (tree.at(b).t - tree.at(x).t));
  auto want = Seq_vector<double>{};
  auto want_total = 0.0;
  for (auto s : k_all_real_seq_letters) {
    want[s] = static_cast<double>(p_rx[index_of(s_r)][index_of(s)] * p_xa[index_of(s)][index_of(s_a)]
                                  * p_xb[index_of(s)][index_of(s_b)]);
    want_total += want[s];
  }

  const auto num_reps = 100000;
  auto got = Seq_vector<int>{0};
  for (auto rep = 0; rep != num_reps; ++rep) {
    gibbs_refresh_node_histories(tree, x, read_incident_sequences(tree, x), sampler, bitgen);
    ++got[calc_site_state_at(tree, tree.node_loc(x), l)];
    EXPECT_EQ(root_site_state(tree, l), s_r);
    EXPECT_EQ(calc_site_state_at(tree, tree.node_loc(a), l), s_a);
  }
  for (auto s : k_all_real_seq_letters) {
    auto p = want[s] / want_total;
    auto observed = static_cast<double>(got[s]) / num_reps;
    auto tolerance = 5.0 * std::sqrt(std::max(p, 1e-5) * (1 - p) / num_reps);
    EXPECT_LT(std::abs(observed - p), tolerance)
        << "state " << to_char(s) << ": refreshed " << observed << ", conditional " << p;
  }
}

}  // namespace delphy
