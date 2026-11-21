#include "history_gibbs.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "absl/log/check.h"
#include "absl/random/distributions.h"
#include "absl/strings/str_format.h"

#include "phylo_tree_calc.h"

namespace delphy {

namespace {

// Buffers reused across calls so that a Gibbs update allocates nothing in steady state.  One set per
// thread, and one set per role, so that the whole-site refresh and the branch update it calls never
// share a buffer.
struct Branch_write_scratch {
  std::vector<Mutation> old_site_muts;
  std::vector<Mutation> new_site_muts;
  std::vector<Ctmc_bridge_event> events;
};

// Removes every mutation at site `l` from `muts`, which stays sorted, and copies them in order into
// `site_muts`.
auto split_off_site_mutations(
    Mutation_list<>& muts,
    Site_index l,
    std::vector<Mutation>& site_muts)
    -> void {

  site_muts.clear();
  auto out = muts.begin();
  for (auto it = muts.begin(); it != muts.end(); ++it) {
    if (it->site == l) {
      site_muts.push_back(*it);
    } else {
      *out = *it;
      ++out;
    }
  }
  muts.erase(out, muts.end());
}

// Turns the events of a bridge across [t_P, t_X] into mutations at site `l` and merges them into the
// sorted `muts`.  `start_state` is the state of the site at `t_P`.
auto merge_in_site_history(
    Mutation_list<>& muts,
    Site_index l,
    double t_P,
    Real_seq_letter start_state,
    std::span<const Ctmc_bridge_event> events,
    std::vector<Mutation>& new_site_muts)
    -> void {

  new_site_muts.clear();
  auto state = start_state;
  for (const auto& e : events) {
    new_site_muts.push_back(Mutation{state, l, e.to, t_P + e.t});
    state = e.to;
  }

  auto split = std::ssize(muts);
  muts.insert(muts.end(), new_site_muts.begin(), new_site_muts.end());
  std::inplace_merge(muts.begin(), muts.begin() + split, muts.end(), mutations_on_branch_less);
}

// The state of site `l` at the root: the reference sequence as changed by the mutations Delphy keeps
// above the root.
auto calc_root_site_state(const Phylo_tree& tree, Site_index l) -> Real_seq_letter {
  auto state = tree.ref_sequence[l];
  for (const auto& m : tree.at_root().mutations) {
    if (m.site == l) { state = m.to; }
  }
  return state;
}

// Records `state` as the state of site `l` at the root, by rewriting the mutations above the root.
auto set_root_site_state(Phylo_tree& tree, Site_index l, Real_seq_letter state) -> void {
  auto site_muts = std::vector<Mutation>{};
  split_off_site_mutations(tree.at_root().mutations, l, site_muts);
  if (state != tree.ref_sequence[l]) {
    auto& muts = tree.at_root().mutations;
    auto split = std::ssize(muts);
    // Delphy places the mutations above the root at -DBL_MAX, logically before everything else on
    // the tree, and the SPR code relies on that ordering; see `Run::repartition`.
    muts.push_back(Mutation{tree.ref_sequence[l], l, state, -std::numeric_limits<double>::max()});
    std::inplace_merge(muts.begin(), muts.begin() + split, muts.end(), mutations_on_branch_less);
  }
}

// The state of site `l` at `node`, given the state at its parent.  Cheaper than walking to the root.
auto calc_site_state_below_branch(
    const Phylo_tree& tree,
    Node_index node,
    Site_index l,
    Real_seq_letter state_at_parent)
    -> Real_seq_letter {
  auto state = state_at_parent;
  for (const auto& m : tree.at(node).mutations) {
    if (m.site == l) { state = m.to; }
  }
  return state;
}

// Picks a state from non-negative weights that are not all zero.
auto pick_weighted(const Seq_vector<double>& weight, absl::BitGenRef bitgen) -> Real_seq_letter {
  auto total = 0.0;
  for (auto s : k_all_real_seq_letters) { total += weight[s]; }
  if (not(total > 0.0)) {
    throw std::domain_error("Every state of the conditional distribution has probability zero");
  }
  auto u = absl::Uniform(absl::IntervalOpenOpen, bitgen, 0.0, total);
  auto cum = 0.0;
  for (auto s : k_all_real_seq_letters) {
    cum += weight[s];
    if (cum >= u) { return s; }
  }
  return Real_seq_letter::T;
}

// Rewrites the history of site `l` on the branch ending at `X` as an exact conditional bridge between
// the given endpoint states, and returns the change in the branch's contribution to log G.
auto rewrite_branch_site_history(
    Phylo_tree& tree,
    Branch_index X,
    Site_index l,
    Real_seq_letter old_start_state,
    Real_seq_letter new_start_state,
    Real_seq_letter new_end_state,
    Site_history_sampler& sampler,
    absl::BitGenRef bitgen,
    Branch_write_scratch& scratch,
    int& num_old_events,
    int& num_new_events)
    -> double {

  auto t_P = tree.at_parent_of(X).t;
  auto t_X = tree.at(X).t;
  const auto& evo = sampler.evo();

  split_off_site_mutations(tree.at(X).mutations, l, scratch.old_site_muts);
  DCHECK(scratch.old_site_muts.empty() || scratch.old_site_muts.front().from == old_start_state);
  auto old_log_G = calc_site_branch_log_G(t_P, t_X, old_start_state, evo, l, scratch.old_site_muts);

  scratch.events.clear();
  sampler.sample_bridge(l, new_start_state, new_end_state, t_X - t_P, bitgen, scratch.events);
  merge_in_site_history(tree.at(X).mutations, l, t_P, new_start_state, scratch.events,
                        scratch.new_site_muts);
  auto new_log_G = calc_site_branch_log_G(t_P, t_X, new_start_state, evo, l, scratch.new_site_muts);

  num_old_events = static_cast<int>(std::ssize(scratch.old_site_muts));
  num_new_events = static_cast<int>(std::ssize(scratch.new_site_muts));
  return new_log_G - old_log_G;
}

}  // namespace

Site_history_sampler::Site_history_sampler(const Global_evo_model& evo) : evo_{evo}, per_partition_{} {
  per_partition_.reserve(evo.num_partitions());
  for (auto beta = Partition_index{0}; beta != evo.num_partitions(); ++beta) {
    per_partition_.emplace_back(evo.partition_evo_model[beta].q_ab);
  }
}

auto Site_history_sampler::rate_scale_of_site(Site_index l) const -> double {
  return evo_.mu_l(l) * evo_.nu_l[l];
}

auto Site_history_sampler::calc_transition_probs(Site_index l, double T) const -> Seq_matrix<double> {
  const auto& sampler = per_partition_[evo_.partition_for_site[l]];
  return sampler.calc_transition_probs(rate_scale_of_site(l) * T);
}

auto Site_history_sampler::sample_bridge(
    Site_index l,
    Real_seq_letter a,
    Real_seq_letter b,
    double T,
    absl::BitGenRef bitgen,
    std::vector<Ctmc_bridge_event>& out)
    -> void {
  auto& sampler = per_partition_[evo_.partition_for_site[l]];
  sampler.sample(a, b, T, rate_scale_of_site(l), bitgen, out);
}

auto calc_site_branch_log_G(
    double t_P,
    double t_X,
    Real_seq_letter start_state,
    const Global_evo_model& evo,
    Site_index l,
    std::span<const Mutation> site_muts)
    -> double {

  auto rate = evo.mu_l(l) * evo.nu_l[l];
  auto result = 0.0;
  auto state = start_state;
  auto t_prev = t_P;
  for (const auto& m : site_muts) {
    DCHECK_EQ(m.site, l);
    DCHECK(m.from == state);
    result -= rate * evo.q_l_a(l, state) * (m.t - t_prev);
    result += std::log(rate * evo.q_l_ab(l, m.from, m.to));
    state = m.to;
    t_prev = m.t;
  }
  result -= rate * evo.q_l_a(l, state) * (t_X - t_prev);
  return result;
}

auto gibbs_resample_branch_site_history(
    Phylo_tree& tree,
    Branch_index X,
    Site_index l,
    Real_seq_letter state_at_parent,
    Site_history_sampler& sampler,
    absl::BitGenRef bitgen)
    -> History_gibbs_result {

  static thread_local auto scratch = Branch_write_scratch{};

  auto result = History_gibbs_result{};
  CHECK_NE(X, tree.root);
  if (is_site_missing_at(tree, X, l)) { return result; }

  auto state_at_X = calc_site_state_below_branch(tree, X, l, state_at_parent);
  result.delta_log_G = rewrite_branch_site_history(
      tree, X, l, state_at_parent, state_at_parent, state_at_X, sampler, bitgen, scratch,
      result.num_old_events, result.num_new_events);
  result.performed = true;
  return result;
}

auto gibbs_resample_node_site_state(
    Phylo_tree& tree,
    Node_index V,
    Site_index l,
    Real_seq_letter state_at_parent,
    Site_history_sampler& sampler,
    absl::BitGenRef bitgen)
    -> History_gibbs_result {

  static thread_local auto scratch = Branch_write_scratch{};

  auto result = History_gibbs_result{};
  CHECK(tree.at(V).is_inner_node());
  if (is_site_missing_at(tree, V, l)) { return result; }

  const auto& evo = sampler.evo();
  auto is_root = (V == tree.root);
  auto t_V = tree.at(V).t;

  // The state of the site at `V` now, and at the far end of every incident branch.  Those far
  // endpoints are what the update conditions on.
  auto old_state_V = Real_seq_letter::A;
  if (is_root) {
    old_state_V = calc_root_site_state(tree, l);
  } else {
    DCHECK(state_at_parent == calc_site_state_at(tree, tree.node_loc(tree.at(V).parent), l));
    old_state_V = calc_site_state_below_branch(tree, V, l, state_at_parent);
  }

  // Weight of each candidate state: the root prior or the transition across the parent branch, times
  // the transition across each child branch whose subtree still has the site.
  auto weight = Seq_vector<double>{1.0};
  if (is_root) {
    const auto& pi_a = evo.partition_evo_model[evo.partition_for_site[l]].pi_a;
    for (auto s : k_all_real_seq_letters) { weight[s] = pi_a[s]; }
  } else {
    auto p = sampler.calc_transition_probs(l, t_V - tree.at_parent_of(V).t);
    for (auto s : k_all_real_seq_letters) { weight[s] = p[state_at_parent][s]; }
  }

  auto child_state = std::array<Real_seq_letter, 2>{};
  auto child_present = std::array<bool, 2>{};
  for (auto i = 0; i != 2; ++i) {
    auto C = tree.at(V).children[i];
    child_present[i] = not tree.at(C).missations.contains(l);
    if (not child_present[i]) { continue; }
    child_state[i] = calc_site_state_below_branch(tree, C, l, old_state_V);
    auto p = sampler.calc_transition_probs(l, tree.at(C).t - t_V);
    for (auto s : k_all_real_seq_letters) { weight[s] *= p[s][child_state[i]]; }
  }

  auto weight_total = 0.0;
  for (auto s : k_all_real_seq_letters) { weight_total += weight[s]; }
  if (not(weight_total > 0.0)) {
    // Only reachable if a zero-length incident branch makes every candidate state impossible.  The
    // identity is then the only update that keeps the state legal.
    return result;
  }
  auto new_state_V = pick_weighted(weight, bitgen);

  // Rewrite the histories of the incident branches, and the state at `V` itself.
  auto delta_log_G = 0.0;
  if (is_root) {
    const auto& pi_a = evo.partition_evo_model[evo.partition_for_site[l]].pi_a;
    delta_log_G += std::log(pi_a[new_state_V]) - std::log(pi_a[old_state_V]);
    set_root_site_state(tree, l, new_state_V);
  } else {
    auto num_old = 0;
    auto num_new = 0;
    delta_log_G += rewrite_branch_site_history(
        tree, V, l, state_at_parent, state_at_parent, new_state_V, sampler, bitgen, scratch,
        num_old, num_new);
    result.num_old_events += num_old;
    result.num_new_events += num_new;
  }

  for (auto i = 0; i != 2; ++i) {
    auto C = tree.at(V).children[i];
    if (not child_present[i]) {
      // The site is missing below this child.  No history lives on the branch; only the state
      // entering the missing region has to follow the new state at `V`, which is what keeps
      // `lambda_i` at and below `C` unchanged.
      tree.at(C).missations.set_from_state(l, new_state_V, tree.ref_sequence);
      continue;
    }
    auto num_old = 0;
    auto num_new = 0;
    delta_log_G += rewrite_branch_site_history(
        tree, C, l, old_state_V, new_state_V, child_state[i], sampler, bitgen, scratch,
        num_old, num_new);
    result.num_old_events += num_old;
    result.num_new_events += num_new;
  }

  auto rate = evo.mu_l(l) * evo.nu_l[l];
  result.performed = true;
  result.state_changed = (new_state_V != old_state_V);
  result.delta_log_G = delta_log_G;
  result.delta_lambda_at_node = rate * (evo.q_l_a(l, new_state_V) - evo.q_l_a(l, old_state_V));
  return result;
}

auto gibbs_refresh_site_history(
    Phylo_tree& tree,
    Site_index l,
    Site_history_sampler& sampler,
    absl::BitGenRef bitgen)
    -> Site_refresh_result {

  static thread_local auto scratch = Branch_write_scratch{};

  auto result = Site_refresh_result{};
  const auto& evo = sampler.evo();
  auto num_nodes = std::ssize(tree);

  // The site may be missing at the root itself, in which case it is missing everywhere.
  if (tree.at_root().missations.contains(l)) { return result; }

  // Pass one, from the root down: the state the site has now at every node, and which subtrees have
  // lost it.  `missing[node]` is true as soon as an ancestor of `node`, or `node` itself, declares
  // the site missing; the subtree below such a node carries no history for the site.
  auto old_state = Node_vector<Real_seq_letter>(num_nodes, Real_seq_letter::A);
  auto missing = Node_vector<char>(num_nodes, 0);
  auto transition = Node_vector<Seq_matrix<double>>(num_nodes);
  old_state[tree.root] = calc_root_site_state(tree, l);
  for (const auto& node : pre_order_traversal(tree)) {
    if (node != tree.root) {
      auto parent = tree.at(node).parent;
      if (missing[parent] || tree.at(node).missations.contains(l)) {
        missing[node] = 1;
      } else {
        old_state[node] = calc_site_state_below_branch(tree, node, l, old_state[parent]);
        transition[node] = sampler.calc_transition_probs(l, tree.at(node).t - tree.at(parent).t);
      }
    }
  }

  // Pass two, from the tips up: Felsenstein pruning for the single site.  The partials are rescaled
  // at every node, which is harmless because only their ratios are used.
  auto partial = Node_vector<Seq_vector<double>>(num_nodes, Seq_vector<double>{1.0});
  for (const auto& node : post_order_traversal(tree)) {
    if (missing[node]) { continue; }
    if (tree.at(node).is_tip()) {
      // The state of a tip is data: the only term the site contributes is the one that agrees with it.
      for (auto s : k_all_real_seq_letters) { partial[node][s] = (s == old_state[node]) ? 1.0 : 0.0; }
      continue;
    }
    auto value = Seq_vector<double>{1.0};
    for (auto C : tree.at(node).children) {
      if (missing[C]) { continue; }
      for (auto s : k_all_real_seq_letters) {
        auto sum = 0.0;
        for (auto s_c : k_all_real_seq_letters) { sum += transition[C][s][s_c] * partial[C][s_c]; }
        value[s] *= sum;
      }
    }
    auto scale = 0.0;
    for (auto s : k_all_real_seq_letters) { scale = std::max(scale, value[s]); }
    if (scale > 0.0) {
      for (auto s : k_all_real_seq_letters) { value[s] /= scale; }
    }
    partial[node] = value;
  }

  // Draw the root state from the root prior times the root partial likelihood, then every internal
  // state from the root down, then the history of every branch.
  auto new_state = Node_vector<Real_seq_letter>(num_nodes, Real_seq_letter::A);
  {
    const auto& pi_a = evo.partition_evo_model[evo.partition_for_site[l]].pi_a;
    auto weight = Seq_vector<double>{};
    auto total = 0.0;
    for (auto s : k_all_real_seq_letters) { weight[s] = pi_a[s] * partial[tree.root][s]; total += weight[s]; }
    if (not(total > 0.0)) { return result; }
    new_state[tree.root] = pick_weighted(weight, bitgen);
  }
  result.root_state_changed = (new_state[tree.root] != old_state[tree.root]);
  if (result.root_state_changed) { ++result.num_states_changed; }
  set_root_site_state(tree, l, new_state[tree.root]);

  for (const auto& node : pre_order_traversal(tree)) {
    if (missing[node]) { continue; }
    for (auto C : tree.at(node).children) {
      if (missing[C]) {
        tree.at(C).missations.set_from_state(l, new_state[node], tree.ref_sequence);
        continue;
      }
      if (tree.at(C).is_tip()) {
        new_state[C] = old_state[C];  // data
      } else {
        auto weight = Seq_vector<double>{};
        for (auto s : k_all_real_seq_letters) {
          weight[s] = transition[C][new_state[node]][s] * partial[C][s];
        }
        new_state[C] = pick_weighted(weight, bitgen);
        if (new_state[C] != old_state[C]) { ++result.num_states_changed; }
      }
      auto num_old = 0;
      auto num_new = 0;
      rewrite_branch_site_history(tree, C, l, old_state[node], new_state[node], new_state[C],
                                  sampler, bitgen, scratch, num_old, num_new);
      result.num_old_events += num_old;
      result.num_new_events += num_new;
    }
  }

  return result;
}


namespace {

// The transition probabilities of the three branches incident on a node, for one group of sites: the
// sites of a partition that share a relative rate share every matrix, so the uniformization series is
// run once per group rather than once per site.  Without site-rate heterogeneity there is one group
// per partition.
struct Incident_transitions {
  Seq_matrix<double> parent;    // across the branch above the node
  Seq_matrix<double> child[2];  // across each child branch
};

auto make_incident_transitions(
    const Phylo_tree& tree,
    Node_index V,
    double t_V,
    Site_index l,
    Site_history_sampler& sampler,
    bool is_root)
    -> Incident_transitions {

  auto out = Incident_transitions{};
  if (not is_root) {
    out.parent = sampler.calc_transition_probs(l, t_V - tree.at_parent_of(V).t);
  }
  for (auto i = 0; i != 2; ++i) {
    auto C = tree.at(V).children[i];
    out.child[i] = sampler.calc_transition_probs(l, tree.at(C).t - t_V);
  }
  return out;
}

}  // namespace

auto read_incident_sequences(const Phylo_tree& tree, Node_index V) -> Incident_sequences {
  CHECK(tree.at(V).is_inner_node());
  return Incident_sequences{
      .parent = (V == tree.root)
          ? Sequence_overlay{tree.ref_sequence}
          : view_of_sequence_at(tree, tree.node_loc(tree.at(V).parent)),
      .child = {{view_of_sequence_at(tree, tree.node_loc(tree.at(V).children[0])),
                 view_of_sequence_at(tree, tree.node_loc(tree.at(V).children[1]))}},
  };
}

auto calc_local_marginal_log_likelihood(
    const Phylo_tree& tree,
    Node_index V,
    double t_V,
    const Incident_sequences& outside,
    Site_history_sampler& sampler)
    -> double {

  CHECK(tree.at(V).is_inner_node());
  const auto& evo = sampler.evo();
  auto is_root = (V == tree.root);
  const auto& parent_sequence = outside.parent;
  const auto& child_sequence = outside.child;
  // One set of matrices per group of sites that share a partition and a relative rate.
  auto cache = absl::flat_hash_map<std::pair<Partition_index, double>, Incident_transitions>{};

  auto total = 0.0;
  for (auto l = Site_index{0}; l != tree.num_sites(); ++l) {
    if (is_site_missing_at(tree, V, l)) { continue; }
    auto key = std::pair{evo.partition_for_site[l], evo.nu_l[l]};
    auto it = cache.find(key);
    if (it == cache.end()) {
      it = cache.emplace(key, make_incident_transitions(tree, V, t_V, l, sampler, is_root)).first;
    }
    const auto& p = it->second;

    auto weight = Seq_vector<double>{};
    if (is_root) {
      const auto& pi_a = evo.partition_evo_model[evo.partition_for_site[l]].pi_a;
      for (auto s : k_all_real_seq_letters) { weight[s] = pi_a[s]; }
    } else {
      auto s_parent = Real_seq_letter{parent_sequence[l]};
      for (auto s : k_all_real_seq_letters) { weight[s] = p.parent[s_parent][s]; }
    }
    for (auto i = 0; i != 2; ++i) {
      auto C = tree.at(V).children[i];
      if (tree.at(C).missations.contains(l)) { continue; }
      auto s_child = Real_seq_letter{child_sequence[i][l]};
      for (auto s : k_all_real_seq_letters) { weight[s] *= p.child[i][s][s_child]; }
    }
    auto sum = 0.0;
    for (auto s : k_all_real_seq_letters) { sum += weight[s]; }
    if (not(sum > 0.0)) { return -std::numeric_limits<double>::infinity(); }
    total += std::log(sum);
  }
  return total;
}

auto gibbs_refresh_node_histories(
    Phylo_tree& tree,
    Node_index V,
    const Incident_sequences& outside,
    Site_history_sampler& sampler,
    absl::BitGenRef bitgen)
    -> Node_refresh_result {

  static thread_local auto scratch = Branch_write_scratch{};

  auto result = Node_refresh_result{};
  CHECK(tree.at(V).is_inner_node());
  const auto& evo = sampler.evo();
  auto is_root = (V == tree.root);
  auto t_V = tree.at(V).t;
  const auto& parent_sequence = outside.parent;
  const auto& child_sequence = outside.child;

  // The old histories of the incident branches may sit outside the branches now, so they are dropped
  // wholesale rather than edited site by site; the states at the far ends are what carry over.
  auto cache = absl::flat_hash_map<std::pair<Partition_index, double>, Incident_transitions>{};

  // Count the events that are about to go, then clear the three branches.
  auto incident = std::array<Branch_index, 3>{V, tree.at(V).children[0], tree.at(V).children[1]};
  auto first = is_root ? 1 : 0;
  for (auto k = first; k != 3; ++k) {
    result.num_old_events += static_cast<int>(std::ssize(tree.at(incident[k]).mutations));
    tree.at(incident[k]).mutations.clear();
  }
  for (auto l = Site_index{0}; l != tree.num_sites(); ++l) {
    if (is_site_missing_at(tree, V, l)) { continue; }
    auto key = std::pair{evo.partition_for_site[l], evo.nu_l[l]};
    auto it = cache.find(key);
    if (it == cache.end()) {
      it = cache.emplace(key, make_incident_transitions(tree, V, t_V, l, sampler, is_root)).first;
    }
    const auto& p = it->second;

    auto s_parent = Real_seq_letter{parent_sequence[l]};
    auto weight = Seq_vector<double>{};
    if (is_root) {
      const auto& pi_a = evo.partition_evo_model[evo.partition_for_site[l]].pi_a;
      for (auto s : k_all_real_seq_letters) { weight[s] = pi_a[s]; }
    } else {
      for (auto s : k_all_real_seq_letters) { weight[s] = p.parent[s_parent][s]; }
    }
    auto child_state = std::array<Real_seq_letter, 2>{};
    auto child_present = std::array<bool, 2>{};
    for (auto i = 0; i != 2; ++i) {
      auto C = tree.at(V).children[i];
      child_present[i] = not tree.at(C).missations.contains(l);
      if (not child_present[i]) { continue; }
      child_state[i] = Real_seq_letter{child_sequence[i][l]};
      for (auto s : k_all_real_seq_letters) { weight[s] *= p.child[i][s][child_state[i]]; }
    }
    auto sum = 0.0;
    for (auto s : k_all_real_seq_letters) { sum += weight[s]; }
    // The caller only moves the node to a time where the local marginal likelihood is positive, so
    // every site has a reachable state; a site whose conditional has collapsed numerically keeps the
    // state it had, which leaves the sequence legal.
    auto old_state = is_root ? calc_root_site_state(tree, l) : s_parent;
    auto new_state = (sum > 0.0) ? pick_weighted(weight, bitgen) : old_state;
    if (is_root) {
      if (new_state != calc_root_site_state(tree, l)) { ++result.num_states_changed; }
      set_root_site_state(tree, l, new_state);
    } else {
      auto num_old = 0;
      auto num_new = 0;
      // The branch was cleared, so there is no old history to score here; the whole of the three
      // branches' contribution is recomputed by the caller.
      rewrite_branch_site_history(tree, V, l, new_state, s_parent, new_state, sampler, bitgen,
                                  scratch, num_old, num_new);
      result.num_new_events += num_new;
    }
    for (auto i = 0; i != 2; ++i) {
      auto C = tree.at(V).children[i];
      if (not child_present[i]) {
        tree.at(C).missations.set_from_state(l, new_state, tree.ref_sequence);
        continue;
      }
      auto num_old = 0;
      auto num_new = 0;
      rewrite_branch_site_history(tree, C, l, new_state, new_state, child_state[i], sampler, bitgen,
                                 scratch, num_old, num_new);
      result.num_new_events += num_new;
    }
  }

  return result;
}

}  // namespace delphy
