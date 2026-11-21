#include "calibration_init.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "external_tree.h"

namespace delphy {

namespace {

constexpr auto k_inf = std::numeric_limits<double>::infinity();

auto is_subset(const std::vector<int>& a, const std::vector<int>& b) -> bool {  // Both sorted
  return std::ranges::includes(b, a);
}

auto intersects(const std::vector<int>& a, const std::vector<int>& b) -> bool {  // Both sorted
  auto i = 0;
  auto j = 0;
  while (i != std::ssize(a) && j != std::ssize(b)) {
    if (a[i] == b[j]) { return true; }
    if (a[i] < b[j]) { ++i; } else { ++j; }
  }
  return false;
}

auto throw_problems(const std::vector<std::string>& problems) -> void {
  throw Calibration_error(absl::StrCat(
      "The age calibrations cannot all hold on one tree:\n  ", absl::StrJoin(problems, "\n  ")));
}

// Sorted tip indices of every calibrated set other than `taxa all` and sets that list every tip, which both
// name the root.  Indices are into `tip_names`.
struct Clade_constraint {
  int calibration = 0;
  std::vector<int> tips;
};

auto clade_constraints(const Calibration_set& set, const std::vector<std::string>& tip_names)
    -> std::vector<Clade_constraint> {
  auto index_of = absl::flat_hash_map<std::string, int>{};
  for (auto i = 0; i != std::ssize(tip_names); ++i) { index_of.emplace(tip_names[i], i); }
  auto result = std::vector<Clade_constraint>{};
  for (auto i = 0; i != set.size(); ++i) {
    const auto& c = set.calibrations[i];
    if (c.all_tips || not c.require_monophyly) { continue; }
    auto cc = Clade_constraint{i, {}};
    for (const auto& name : c.taxa) {
      auto it = index_of.find(name);
      CHECK(it != index_of.end()) << "Unknown sample " << name;
      cc.tips.push_back(it->second);
    }
    std::ranges::sort(cc.tips);
    if (std::ssize(cc.tips) == std::ssize(tip_names)) { continue; }
    result.push_back(std::move(cc));
  }
  return result;
}

}  // namespace

auto check_calibration_sets_compatible(const Calibration_set& set, const std::vector<std::string>& tip_names)
    -> void {
  auto problems = std::vector<std::string>{};
  auto known = absl::flat_hash_set<std::string>{};
  for (const auto& name : tip_names) {
    if (not known.insert(name).second) {
      problems.push_back(absl::StrFormat("More than one sample is named '%s'; calibrations cannot be resolved", name));
    }
  }

  auto root_calibrations = std::vector<std::string>{};
  for (const auto& c : set.calibrations) {
    if (c.all_tips) { root_calibrations.push_back(c.name); continue; }
    auto unknown = std::vector<std::string>{};
    for (const auto& name : c.taxa) {
      if (not known.contains(name)) { unknown.push_back(name); }
    }
    if (not unknown.empty()) {
      problems.push_back(absl::StrFormat(
          "calibration '%s' (line %d): %d sample ID(s) are not in the alignment: %s",
          c.name, c.config_line, std::ssize(unknown), absl::StrJoin(unknown, ", ")));
    } else if (std::ssize(c.taxa) < 2) {
      problems.push_back(absl::StrFormat(
          "calibration '%s' (line %d): a calibrated sample set needs at least 2 samples; a single sample is a tip, "
          "not an ancestor", c.name, c.config_line));
    } else if (std::ssize(c.taxa) == std::ssize(tip_names) && c.require_monophyly) {
      root_calibrations.push_back(c.name);
    }
  }
  if (std::ssize(root_calibrations) > 1) {
    problems.push_back(absl::StrFormat(
        "calibrations %s all name the root (every sample).  Give one calibration per node",
        absl::StrJoin(root_calibrations, ", ")));
  }
  if (not problems.empty()) { throw_problems(problems); }

  auto constraints = clade_constraints(set, tip_names);
  for (auto i = 0; i != std::ssize(constraints); ++i) {
    for (auto j = i + 1; j != std::ssize(constraints); ++j) {
      const auto& a = constraints[i];
      const auto& b = constraints[j];
      const auto& ca = set.calibrations[a.calibration];
      const auto& cb = set.calibrations[b.calibration];
      if (a.tips == b.tips) {
        problems.push_back(absl::StrFormat(
            "calibrations '%s' and '%s' list the same samples, so they name the same node.  Give one calibration "
            "per node", ca.name, cb.name));
      } else if (intersects(a.tips, b.tips) && not is_subset(a.tips, b.tips) && not is_subset(b.tips, a.tips)) {
        problems.push_back(absl::StrFormat(
            "calibrations '%s' and '%s' share some samples but neither set contains the other, so both cannot be "
            "monophyletic in one tree.  Calibrated sample sets must be nested or disjoint", ca.name, cb.name));
      }
    }
  }
  if (not problems.empty()) { throw_problems(problems); }
}

auto calibrated_sets_are_monophyletic(const Calibration_set& set, const Phylo_tree& tree) -> bool {
  auto tip_of = absl::flat_hash_map<std::string, Node_index>{};
  for (const auto& node : index_order_traversal(tree)) {
    if (tree.at(node).is_tip()) { tip_of.emplace(tree.at(node).name, node); }
  }
  auto tips_below = Node_vector<int>(tree.size(), 0);
  auto order = std::vector<Node_index>{};
  for (const auto& node : post_order_traversal(tree)) {
    order.push_back(node);
    tips_below[node] = tree.at(node).is_tip() ? 1
        : tips_below[tree.at(node).left_child()] + tips_below[tree.at(node).right_child()];
  }
  auto marked_below = Node_vector<int>(tree.size(), 0);
  for (const auto& c : set.calibrations) {
    if (c.all_tips || not c.require_monophyly) { continue; }
    std::ranges::fill(marked_below, 0);
    for (const auto& name : c.taxa) {
      auto it = tip_of.find(name);
      if (it == tip_of.end()) { return false; }
      marked_below[it->second] = 1;
    }
    auto size = static_cast<int>(std::ssize(c.taxa));
    for (auto node : order) {
      if (tree.at(node).is_inner_node()) {
        marked_below[node] = marked_below[tree.at(node).left_child()] + marked_below[tree.at(node).right_child()];
      }
      if (marked_below[node] == size) {
        if (tips_below[node] != size) { return false; }
        break;
      }
    }
  }
  return true;
}

auto build_calibration_constrained_topology(const Phylo_tree& guide, const Calibration_set& set) -> Newick_tree {
  // Tip indices below are indices into `tip_nodes`, i.e., the guide's tips in index order
  auto tip_nodes = std::vector<Node_index>{};
  auto tip_names = std::vector<std::string>{};
  auto tip_number = Node_vector<int>(guide.size(), -1);
  for (const auto& node : index_order_traversal(guide)) {
    if (guide.at(node).is_tip()) {
      tip_number[node] = static_cast<int>(std::ssize(tip_nodes));
      tip_nodes.push_back(node);
      tip_names.push_back(guide.at(node).name);
    }
  }
  check_calibration_sets_compatible(set, tip_names);
  auto num_tips = static_cast<int>(std::ssize(tip_nodes));

  // The laminar family of clades: constraint g's parent is the smallest constraint strictly containing it;
  // -1 is the whole tree
  auto constraints = clade_constraints(set, tip_names);
  std::ranges::sort(constraints, [](const auto& a, const auto& b) { return std::ssize(a.tips) > std::ssize(b.tips); });
  auto num_groups = static_cast<int>(std::ssize(constraints));
  auto group_parent = std::vector<int>(num_groups, -1);
  for (auto g = 0; g != num_groups; ++g) {
    for (auto h = g - 1; h >= 0; --h) {  // Larger or equal sizes come first; the last container found is smallest
      if (std::ssize(constraints[h].tips) > std::ssize(constraints[g].tips)
          && is_subset(constraints[g].tips, constraints[h].tips)) {
        group_parent[g] = h;
        break;
      }
    }
  }
  // Innermost group of each tip
  auto tip_group = std::vector<int>(num_tips, -1);
  for (auto g = 0; g != num_groups; ++g) {  // Larger groups first, so smaller ones overwrite
    for (auto tip : constraints[g].tips) { tip_group[tip] = g; }
  }

  auto result = Newick_tree{};
  auto add_inner = [&result](Node_index a, Node_index b) -> Node_index {
    auto n = result.add_node();
    result.at(n).children = {a, b};
    result.at(a).parent = n;
    result.at(b).parent = n;
    return n;
  };

  // Builds the clade of group g (-1 for the whole tree) and returns its node in `result`
  auto build = std::function<Node_index(int)>{};
  build = [&](int g) -> Node_index {
    // Items directly inside g, keyed by a representative tip
    auto item_of_rep = absl::flat_hash_map<int, Node_index>{};
    for (auto tip = 0; tip != num_tips; ++tip) {
      // The child group of g that contains tip, if any
      auto h = tip_group[tip];
      while (h != -1 && group_parent[h] != g) { h = group_parent[h]; }
      auto inside_g = (g == -1) || tip_group[tip] == g || h != -1;
      if (not inside_g) { continue; }
      if (h == -1) {
        auto leaf = result.add_node();
        result.at(leaf).name = tip_names[tip];
        item_of_rep.emplace(tip, leaf);
      } else if (constraints[h].tips.front() == tip) {
        item_of_rep.emplace(tip, build(h));
      }
    }
    CHECK_GE(std::ssize(item_of_rep), 2);

    // Topology induced by the guide on the representatives
    auto built = Node_vector<Node_index>(guide.size(), k_no_node);
    for (const auto& node : post_order_traversal(guide)) {
      if (guide.at(node).is_tip()) {
        auto it = item_of_rep.find(tip_number[node]);
        if (it != item_of_rep.end()) { built[node] = it->second; }
      } else {
        auto a = built[guide.at(node).left_child()];
        auto b = built[guide.at(node).right_child()];
        built[node] = (a != k_no_node && b != k_no_node) ? add_inner(a, b) : (a != k_no_node ? a : b);
      }
    }
    CHECK_NE(built[guide.root], k_no_node);
    return built[guide.root];
  };

  result.root = build(-1);
  result.at(result.root).parent = k_no_node;
  CHECK_EQ(result.size(), 2 * num_tips - 1);
  return result;
}

auto fit_node_times_to_calibration_bounds(
    Phylo_tree& tree,
    const Calibration_set& set,
    double min_branch_days)
    -> int {

  auto nodes = bind_calibrations_ignoring_times(set, tree);

  // Hard support of each node, inset so that the result is strictly inside it
  auto lo = Node_vector<double>(tree.size(), -k_inf);
  auto hi = Node_vector<double>(tree.size(), +k_inf);
  for (auto i = 0; i != set.size(); ++i) {
    const auto& c = set.calibrations[i];
    auto t_lo = c.t_min();
    auto t_hi = c.t_max();
    auto inset = (std::isfinite(t_lo) && std::isfinite(t_hi))
        ? std::min(min_branch_days, 1e-3 * (t_hi - t_lo)) : min_branch_days;
    lo[nodes[i]] = std::max(lo[nodes[i]], t_lo + inset);
    hi[nodes[i]] = std::min(hi[nodes[i]], t_hi - inset);
  }

  // Latest time each node may have, given the nodes below it
  auto latest = Node_vector<double>(tree.size(), +k_inf);
  for (const auto& node : post_order_traversal(tree)) {
    if (tree.at(node).is_tip()) {
      latest[node] = tree.at(node).t;
    } else {
      auto t = hi[node];
      for (auto child : tree.at(node).children) { t = std::min(t, latest[child] - min_branch_days); }
      latest[node] = t;
    }
  }

  auto old_t = Node_vector<double>(tree.size());
  for (const auto& node : index_order_traversal(tree)) { old_t[node] = tree.at(node).t; }

  auto problems = std::vector<std::string>{};
  auto num_moved = 0;
  for (const auto& node : pre_order_traversal(tree)) {
    if (tree.at(node).is_tip()) { continue; }
    auto t = std::max(old_t[node], lo[node]);
    if (node != tree.root) { t = std::max(t, tree.at_parent_of(node).t + min_branch_days); }
    t = std::min(t, latest[node]);
    if (t < lo[node]) {
      problems.push_back(absl::StrFormat(
          "no starting time satisfies the hard bounds at the node at %g days since 2020-01-01 together with the "
          "calibrations and samples below it", old_t[node]));
    }
    if (t != old_t[node]) { ++num_moved; }
    tree.at(node).t = t;
  }
  for (auto i = 0; i != set.size(); ++i) {
    const auto& c = set.calibrations[i];
    if (not std::isfinite(c.log_density_at(tree.at(nodes[i]).t))) {
      problems.push_back(absl::StrFormat(
          "calibration '%s': its hard bounds (%g to %g years) cannot be met together with the samples and the other "
          "calibrations below it", c.name, c.prior.support_youngest_age(), c.prior.support_oldest_age()));
    }
  }
  if (not problems.empty()) { throw_problems(problems); }

  // Carry the mutation times along
  for (const auto& node : index_order_traversal(tree)) {
    auto& n = tree.at(node);
    if (node == tree.root) {
      auto shift = n.t - old_t[node];
      for (auto& m : n.mutations) { m.t += shift; }
      continue;
    }
    auto old_t_P = old_t[n.parent];
    auto old_t_X = old_t[node];
    auto new_t_P = tree.at(n.parent).t;
    auto new_t_X = n.t;
    if (old_t_P == new_t_P && old_t_X == new_t_X) { continue; }
    for (auto& m : n.mutations) {
      auto f = (m.t - old_t_P) / (old_t_X - old_t_P);
      m.t = std::clamp(new_t_P + f * (new_t_X - new_t_P), new_t_P, new_t_X);
    }
  }

  return num_moved;
}

auto build_calibration_constrained_initial_tree(
    Phylo_tree guide,
    Real_sequence ref_sequence,
    std::vector<Tip_desc> tip_descs,
    const Calibration_set& set,
    std::optional<double> rate_per_site_per_day,
    int num_observed_sites,
    Constrained_init_summary* summary)
    -> Phylo_tree {

  auto local_summary = Constrained_init_summary{};
  auto tip_names = std::vector<std::string>{};
  for (const auto& node : index_order_traversal(guide)) {
    if (guide.at(node).is_tip()) { tip_names.push_back(guide.at(node).name); }
  }
  check_calibration_sets_compatible(set, tip_names);

  auto tree = std::move(guide);
  if (not calibrated_sets_are_monophyletic(set, tree)) {
    auto rate = 0.0;
    if (rate_per_site_per_day.has_value()) {
      rate = *rate_per_site_per_day;
    } else {
      auto num_muts = 0.0;
      auto total_days = 0.0;
      for (const auto& node : index_order_traversal(tree)) {
        if (node == tree.root) { continue; }
        num_muts += static_cast<double>(std::ssize(tree.at(node).mutations));
        total_days += tree.at(node).t - tree.at_parent_of(node).t;
      }
      // With no mutations at all the rate does not enter the node times
      rate = (num_muts > 0.0 && total_days > 0.0) ? num_muts / total_days / num_observed_sites : 1.0;
    }
    if (not (std::isfinite(rate) && rate > 0.0)) {
      throw std::invalid_argument(absl::StrFormat(
          "Cannot time the calibration-constrained starting tree: the substitution rate %g per site per day is not "
          "positive and finite.  Give an explicit rate with --v0-init-mutation-rate", rate));
    }
    auto ntree = build_calibration_constrained_topology(tree, set);
    auto options = External_tree_options{};
    options.units = Branch_length_units::ignore;
    options.rate_per_site_per_day = rate;
    options.num_observed_sites = num_observed_sites;
    auto ext_summary = External_tree_summary{};
    auto unused_bitgen = std::mt19937{0};  // The builder is deterministic
    tree = build_phylo_tree_from_newick(
        ntree, std::move(ref_sequence), std::move(tip_descs), unused_bitgen, options, &ext_summary);
    local_summary.rebuilt = true;
    local_summary.num_parsimony_mutations = ext_summary.num_parsimony_mutations;
    local_summary.rate_per_site_per_day = rate;
  }

  local_summary.num_nodes_moved = fit_node_times_to_calibration_bounds(tree, set);
  assert_phylo_tree_integrity(tree, true);
  if (summary != nullptr) { *summary = local_summary; }
  return tree;
}

}  // namespace delphy
