#include "external_tree.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <istream>
#include <limits>
#include <stdexcept>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "phylo_tree_calc.h"

namespace delphy {

auto read_newick_tree(std::istream& is) -> Newick_tree {
  auto parser = Newick_parser{is};
  return parser.parse_tree();
}

auto newick_tree_has_branch_lengths(const Newick_tree& tree) -> bool {
  for (auto node = Node_index{0}; node != tree.size(); ++node) {
    if (node != tree.root && tree.at(node).has_branch_length) { return true; }
  }
  return false;
}

namespace {

constexpr auto k_max_listed = 10;

auto list_names(const std::vector<std::string>& names) -> std::string {
  auto shown = std::vector<std::string>(
      names.begin(), names.begin() + std::min<std::ptrdiff_t>(std::ssize(names), k_max_listed));
  auto result = absl::StrJoin(shown, ", ");
  if (std::ssize(names) > k_max_listed) {
    absl::StrAppend(&result, absl::StrFormat(", ... (%d in total)", std::ssize(names)));
  }
  return result;
}

auto describe_node(const Newick_tree& ntree, Node_index node) -> std::string {
  if (not ntree.at(node).name.empty()) { return absl::StrFormat("node '%s'", ntree.at(node).name); }
  return absl::StrFormat("unnamed node #%d", node);
}

struct Branch_change {
  Node_index node;  // Branch ending at this node (in Phylo_tree indices)
  Site_index site;
  Real_seq_letter from;
  Real_seq_letter to;
};

}  // namespace

auto build_phylo_tree_from_newick(
    const Newick_tree& ntree,
    Real_sequence ref_sequence,
    std::vector<Tip_desc> tip_descs,
    absl::BitGenRef /*bitgen*/,
    const External_tree_options& options,
    External_tree_summary* summary)
    -> Phylo_tree {

  const auto num_tips = static_cast<int>(std::ssize(tip_descs));
  const auto num_sites = static_cast<Site_index>(std::ssize(ref_sequence));
  if (num_tips < 2) {
    throw std::invalid_argument("An input tree needs at least 2 sequences");
  }
  if (ntree.root == k_no_node || ntree.size() == 0) {
    throw std::invalid_argument("The input tree is empty");
  }

  // 1. Structure: rooted and strictly bifurcating
  for (auto node = Node_index{0}; node != ntree.size(); ++node) {
    auto num_children = std::ssize(ntree.at(node).children);
    if (num_children == 1) {
      throw std::invalid_argument(absl::StrFormat(
          "The input tree has a node with a single child (%s); remove unary nodes", describe_node(ntree, node)));
    }
    if (num_children > 2) {
      throw std::invalid_argument(absl::StrFormat(
          "The input tree must be rooted and strictly binary, but %s has %d children (an unrooted tree has a "
          "trifurcating root).  Root the tree and resolve polytomies before passing it in; Delphy will not "
          "reroot or resolve it",
          describe_node(ntree, node), num_children));
    }
  }

  // 2. Tip names match sequence names one to one
  auto seq_index = absl::flat_hash_map<std::string, int>{};
  for (auto i = 0; i != num_tips; ++i) {
    seq_index.emplace(tip_descs[i].name, i);
  }
  auto tree_tip_names = absl::flat_hash_set<std::string>{};
  auto duplicate_tips = std::vector<std::string>{};
  auto extra_tips = std::vector<std::string>{};
  auto unnamed_tips = 0;
  for (auto node = Node_index{0}; node != ntree.size(); ++node) {
    if (not ntree.at(node).is_tip()) { continue; }
    const auto& name = ntree.at(node).name;
    if (name.empty()) { ++unnamed_tips; continue; }
    if (not tree_tip_names.insert(name).second) { duplicate_tips.push_back(name); continue; }
    if (not seq_index.contains(name)) { extra_tips.push_back(name); }
  }
  auto missing_tips = std::vector<std::string>{};
  for (const auto& td : tip_descs) {
    if (not tree_tip_names.contains(td.name)) { missing_tips.push_back(td.name); }
  }
  if (unnamed_tips > 0) {
    throw std::invalid_argument(absl::StrFormat(
        "The input tree has %d tips without a name; every tip must be labelled with a sequence ID", unnamed_tips));
  }
  if (not duplicate_tips.empty()) {
    throw std::invalid_argument(absl::StrCat(
        "The input tree has duplicate tip names: ", list_names(duplicate_tips)));
  }
  if (not extra_tips.empty() || not missing_tips.empty()) {
    auto msg = std::string{"The tips of the input tree do not match the input sequences."};
    if (not extra_tips.empty()) {
      absl::StrAppend(&msg, absl::StrFormat(
          "  %d tip(s) in the tree have no sequence (or the sequence was discarded): %s.",
          extra_tips.size(), list_names(extra_tips)));
    }
    if (not missing_tips.empty()) {
      absl::StrAppend(&msg, absl::StrFormat(
          "  %d sequence(s) are missing from the tree: %s.", missing_tips.size(), list_names(missing_tips)));
    }
    throw std::invalid_argument(msg);
  }

  // 3. Branch lengths
  const auto has_lengths = newick_tree_has_branch_lengths(ntree);
  if (has_lengths && options.units == Branch_length_units::undeclared) {
    throw std::invalid_argument(
        "The input tree has branch lengths but their meaning was not declared.  State whether they are in "
        "years, days or substitutions per site, or ask for them to be ignored (topology only)");
  }
  for (auto node = Node_index{0}; node != ntree.size(); ++node) {
    if (node == ntree.root) { continue; }
    const auto& nn = ntree.at(node);
    if (not nn.has_branch_length) { continue; }
    if (not std::isfinite(nn.branch_length)) {
      throw std::invalid_argument(absl::StrFormat(
          "The branch above %s has a non-finite length (%g)", describe_node(ntree, node), nn.branch_length));
    }
    if (nn.branch_length < 0.0) {
      throw std::invalid_argument(absl::StrFormat(
          "The branch above %s has a negative length (%g)", describe_node(ntree, node), nn.branch_length));
    }
  }
  const auto uses_lengths = options.units == Branch_length_units::years ||
                            options.units == Branch_length_units::days ||
                            options.units == Branch_length_units::subs_per_site;
  if (uses_lengths) {
    for (auto node = Node_index{0}; node != ntree.size(); ++node) {
      if (node != ntree.root && not ntree.at(node).has_branch_length) {
        throw std::invalid_argument(absl::StrFormat(
            "The branch above %s has no length, but branch lengths were declared to be meaningful",
            describe_node(ntree, node)));
      }
    }
  }
  if ((options.units == Branch_length_units::subs_per_site || options.units == Branch_length_units::ignore) &&
      not (options.rate_per_site_per_day.has_value() && *options.rate_per_site_per_day > 0.0)) {
    throw std::invalid_argument(
        "Converting genetic distances or mutation counts into time needs an explicit substitution rate "
        "(an initial rate or a rate prior); none was given");
  }
  if (options.units == Branch_length_units::ignore && options.num_observed_sites <= 0) {
    throw std::invalid_argument("The number of observed sites must be positive");
  }

  // 4. Number nodes: tips keep the sequence order, inner nodes follow in post-order, root last
  auto pnode_of = std::vector<Node_index>(ntree.size(), k_no_node);
  auto next_inner = Node_index{num_tips};
  for (const auto& nnode : post_order_traversal(ntree)) {
    if (ntree.at(nnode).is_tip()) {
      pnode_of[nnode] = seq_index.at(ntree.at(nnode).name);
    } else {
      pnode_of[nnode] = next_inner++;
    }
  }
  if (next_inner != 2 * num_tips - 1) {
    throw std::invalid_argument("The input tree is not a strictly binary tree over the input sequences");
  }

  auto ptree = Phylo_tree{2 * num_tips - 1};
  ptree.ref_sequence = ref_sequence;
  ptree.root = pnode_of[ntree.root];
  auto len_days = std::vector<double>(2 * num_tips - 1, 0.0);  // Branch length in days, by Phylo_tree node
  for (const auto& nnode : post_order_traversal(ntree)) {
    auto pn = pnode_of[nnode];
    auto& node = ptree.at(pn);
    node.parent = nnode == ntree.root ? k_no_node : pnode_of[ntree.at(nnode).parent];
    if (ntree.at(nnode).is_tip()) {
      node.children = {};
      node.name = tip_descs[pn].name;
      node.t_min = tip_descs[pn].t_min;
      node.t_max = tip_descs[pn].t_max;
      node.missations = tip_descs[pn].missations;
    } else {
      const auto& c = ntree.at(nnode).children;
      node.children = {pnode_of[c[0]], pnode_of[c[1]]};
      node.t_min = -std::numeric_limits<float>::max();
      node.t_max = +std::numeric_limits<float>::max();
    }
    if (nnode != ntree.root && uses_lengths) {
      auto len = ntree.at(nnode).branch_length;
      switch (options.units) {
        case Branch_length_units::years: len_days[pn] = len * 365.0; break;
        case Branch_length_units::days: len_days[pn] = len; break;
        case Branch_length_units::subs_per_site: len_days[pn] = len / *options.rate_per_site_per_day; break;
        default: break;
      }
    }
  }

  // 5. Fitch parsimony over the variable sites.  Missing data are wildcards.
  auto order = std::vector<Node_index>{};  // Inner nodes, children before parents
  order.reserve(num_tips - 1);
  for (auto n = Node_index{num_tips}; n != ptree.size(); ++n) { order.push_back(n); }

  auto tip_changes_at = absl::flat_hash_map<Site_index, std::vector<std::pair<int, Real_seq_letter>>>{};
  auto missing_count = std::vector<int>(num_sites + 1, 0);  // Difference array, then prefix sums
  for (auto i = 0; i != num_tips; ++i) {
    for (const auto& d : tip_descs[i].seq_deltas) {
      if (d.site < 0 || d.site >= num_sites) {
        throw std::invalid_argument(absl::StrFormat(
            "In sequence %s, mutation %s is outside the reference sequence", tip_descs[i].name,
            absl::FormatStreamed(d)));
      }
      tip_changes_at[d.site].emplace_back(i, d.to);
    }
    for (const auto& [mi_start, mi_end] : tip_descs[i].missations.intervals) {
      if (mi_start < 0 || mi_end > num_sites || mi_start > mi_end) {
        throw std::invalid_argument(absl::StrFormat(
            "In sequence %s, the missing-data interval [%d,%d) is outside the reference sequence",
            tip_descs[i].name, mi_start + 1, mi_end + 1));
      }
      ++missing_count[mi_start];
      --missing_count[mi_end];
    }
  }
  auto variable_sites = std::vector<Site_index>{};
  {
    auto running = 0;
    for (auto l = Site_index{0}; l != num_sites; ++l) {
      running += missing_count[l];
      if (tip_changes_at.contains(l) || (running > 0 && running < num_tips)) { variable_sites.push_back(l); }
    }
  }

  auto changes = std::vector<Branch_change>{};
  auto mask = std::vector<std::uint8_t>(ptree.size());
  auto state = std::vector<std::uint8_t>(ptree.size());
  for (auto l : variable_sites) {
    auto ref_bit = static_cast<std::uint8_t>(1u << index_of(ref_sequence[l]));
    for (auto i = 0; i != num_tips; ++i) {
      mask[i] = tip_descs[i].missations.contains(l) ? std::uint8_t{0xF} : ref_bit;
    }
    if (auto it = tip_changes_at.find(l); it != tip_changes_at.end()) {
      for (const auto& [i, to] : it->second) {
        if (mask[i] != 0xF) { mask[i] = static_cast<std::uint8_t>(1u << index_of(to)); }
      }
    }
    for (auto n : order) {
      auto a = mask[ptree.at(n).children[0]];
      auto b = mask[ptree.at(n).children[1]];
      mask[n] = (a & b) != 0 ? static_cast<std::uint8_t>(a & b) : static_cast<std::uint8_t>(a | b);
    }
    auto root_state = (mask[ptree.root] & ref_bit) != 0 ? ref_bit : static_cast<std::uint8_t>(mask[ptree.root] & -mask[ptree.root]);
    state[ptree.root] = root_state;
    if (root_state != ref_bit) {
      changes.push_back({ptree.root, l, ref_sequence[l],
                         real_seq_letter_from_index(std::countr_zero(static_cast<unsigned>(root_state)))});
    }
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      auto p = *it;
      for (auto c : ptree.at(p).children) {
        if ((mask[c] & state[p]) != 0) {
          state[c] = state[p];
        } else {
          state[c] = static_cast<std::uint8_t>(mask[c] & -mask[c]);
          changes.push_back({c, l,
              real_seq_letter_from_index(std::countr_zero(static_cast<unsigned>(state[p]))),
              real_seq_letter_from_index(std::countr_zero(static_cast<unsigned>(state[c])))});
        }
      }
    }
  }

  // 6. Node times
  auto num_changes_above = std::vector<int>(ptree.size(), 0);
  for (const auto& ch : changes) { ++num_changes_above[ch.node]; }
  if (options.units == Branch_length_units::ignore) {
    for (auto n = Node_index{0}; n != ptree.size(); ++n) {
      if (n == ptree.root) { continue; }
      len_days[n] = num_changes_above[n] / (*options.rate_per_site_per_day * options.num_observed_sites);
    }
  }
  for (auto i = 0; i != num_tips; ++i) {
    auto& node = ptree.at(i);
    node.t = std::clamp(0.5 * (static_cast<double>(node.t_min) + static_cast<double>(node.t_max)),
                        static_cast<double>(node.t_min), static_cast<double>(node.t_max));
  }
  for (auto n : order) {
    auto t = +std::numeric_limits<double>::infinity();
    for (auto c : ptree.at(n).children) {
      t = std::min(t, ptree.at(c).t - std::max(len_days[c], options.min_branch_days));
    }
    if (not std::isfinite(t)) {
      throw std::invalid_argument("The branch lengths produce non-finite node times; check the units and rate");
    }
    ptree.at(n).t = t;
  }

  // 7. Mutations and missations
  for (const auto& ch : changes) {
    ptree.at(ch.node).mutations.push_back(Mutation{ch.from, ch.site, ch.to, ptree.at(ch.node).t});
  }
  for (auto n = Node_index{0}; n != ptree.size(); ++n) {
    sort_mutations(ptree.at(n).mutations);
  }
  fix_up_missations(ptree);

  // Spread mutation times uniformly over their branch, without a random draw: deterministic initial state.
  for (auto n = Node_index{0}; n != ptree.size(); ++n) {
    if (n == ptree.root || ptree.at(n).mutations.empty()) { continue; }
    auto t_P = ptree.at_parent_of(n).t;
    auto t_X = ptree.at(n).t;
    auto k = static_cast<double>(std::ssize(ptree.at(n).mutations));
    auto i = 0;
    for (auto& m : ptree.at(n).mutations) {
      ++i;
      m.t = t_P + (t_X - t_P) * (static_cast<double>(i) / (k + 1.0));
    }
  }
  rereference_to_root_sequence(ptree);

  assert_phylo_tree_integrity(ptree, true);
  assert_phylo_tree_matches_tip_descs(ptree, ref_sequence, tip_descs, true);

  if (summary != nullptr) {
    summary->num_tips = num_tips;
    summary->num_parsimony_mutations = 0;
    for (auto n = Node_index{0}; n != ptree.size(); ++n) {
      summary->num_parsimony_mutations += static_cast<int>(std::ssize(ptree.at(n).mutations));
    }
    summary->root_t = ptree.at_root().t;
    summary->used_branch_lengths = uses_lengths;
  }
  return ptree;
}

}  // namespace delphy
