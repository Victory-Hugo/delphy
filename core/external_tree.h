#ifndef DELPHY_EXTERNAL_TREE_H_
#define DELPHY_EXTERNAL_TREE_H_

#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

#include "newick.h"
#include "phylo_tree.h"

namespace delphy {

// Meaning of the branch lengths in a user-supplied Newick tree.
enum class Branch_length_units {
  undeclared,     // The caller did not say; this is an error if the tree has any branch length
  ignore,         // Only the rooted topology is used; node times come from parsimony mutation counts
  years,          // Calendar time
  days,           // Calendar time
  subs_per_site,  // Genetic distance; converted to time with the supplied rate
};

struct External_tree_options {
  Branch_length_units units = Branch_length_units::undeclared;

  // Substitution rate per site per day used to turn genetic distances into time
  // (required for `ignore` and `subs_per_site`).
  std::optional<double> rate_per_site_per_day;

  // Number of sites that count towards mutation numbers when `units == ignore`.
  Site_index num_observed_sites = 0;

  // Smallest time gap enforced between a node and its parent (days), so that zero-length input
  // branches do not produce coincident node times.
  double min_branch_days = 1e-3;
};

struct External_tree_summary {
  int num_tips = 0;
  int num_parsimony_mutations = 0;  // Mutations on the initial history (Fitch parsimony)
  double root_t = 0.0;
  bool used_branch_lengths = false;
};

// Parse a Newick tree from a stream.  Throws std::runtime_error on malformed input.
auto read_newick_tree(std::istream& is) -> Newick_tree;

// Whether any non-root node of the tree carries an explicit branch length.
auto newick_tree_has_branch_lengths(const Newick_tree& tree) -> bool;

// Convert a user-supplied rooted, strictly bifurcating Newick tree into an initial Phylo_tree whose
// tips are the given sequences.  The rooted topology is preserved exactly: tip names are matched
// one-to-one against `tip_descs` (duplicates, missing and extra tips are errors), no rerooting or
// rearrangement happens, and only the node times and the mutation history are derived:
//
//  - Mutation history: Fitch parsimony on the given topology, treating missing data as wildcards.
//    This is only an initial state for MCMC, not the posterior model.
//  - Node times: tips sit at the middle of their date range; each inner node is placed
//    `max(branch length, min_branch_days)` before its earliest child, with branch lengths in
//    calendar time as declared by `options.units`.
//
// Throws std::invalid_argument with a message describing how to fix the input.
auto build_phylo_tree_from_newick(
    const Newick_tree& ntree,
    Real_sequence ref_sequence,
    std::vector<Tip_desc> tip_descs,
    absl::BitGenRef bitgen,
    const External_tree_options& options,
    External_tree_summary* summary = nullptr)
    -> Phylo_tree;

}  // namespace delphy

#endif  // DELPHY_EXTERNAL_TREE_H_
