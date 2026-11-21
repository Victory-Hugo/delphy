#ifndef DELPHY_CALIBRATION_INIT_H_
#define DELPHY_CALIBRATION_INIT_H_

#include <optional>
#include <string>
#include <vector>

#include "calibration.h"
#include "newick.h"
#include "phylo_tree.h"
#include "sequence.h"

namespace delphy {

// Starting trees for age calibrations with a free topology, where every calibrated sample set that requires
// monophyly (Calibration::require_monophyly) must be a clade; the other sets impose nothing on the topology.
// See doc/extended/calibration_free_topology.md.

// Checks that every calibrated sample set names known, distinct samples and that the sets that require monophyly
// can all be monophyletic at once: any two of them must be nested or disjoint.  Throws Calibration_error listing
// every problem found.
auto check_calibration_sets_compatible(const Calibration_set& set, const std::vector<std::string>& tip_names)
    -> void;

// Whether every calibrated sample set that requires monophyly (other than `taxa all`) is monophyletic in `tree`.
auto calibrated_sets_are_monophyletic(const Calibration_set& set, const Phylo_tree& tree) -> bool;

// A rooted binary topology over the tips of `guide` in which every calibrated sample set is a clade.  Inside
// each calibrated clade (and at the top level), the arrangement of its maximal calibrated subclades and loose
// samples is the topology that `guide` induces on one representative sample of each.  If `guide` already
// satisfies every constraint, the result is the topology of `guide`.  Requires
// check_calibration_sets_compatible() to pass.
auto build_calibration_constrained_topology(const Phylo_tree& guide, const Calibration_set& set) -> Newick_tree;

// Moves inner node times of `tree` as little as this simple projection allows so that every calibrated node
// (resolved by sample set) lies strictly inside its hard support and every parent stays at least
// `min_branch_days` older than its children.  Tip times never move.  Mutation times on a branch whose end
// points moved are mapped linearly onto the new branch, which keeps their order and the mutation history
// unchanged.  Returns the number of inner nodes moved.  Throws Calibration_error if no such times exist.
auto fit_node_times_to_calibration_bounds(
    Phylo_tree& tree,
    const Calibration_set& set,
    double min_branch_days = 1e-3)
    -> int;

struct Constrained_init_summary {
  bool rebuilt = false;                // The guide violated a constraint and the topology was rebuilt
  int num_parsimony_mutations = 0;     // Of the rebuilt history
  double rate_per_site_per_day = 0.0;  // Used to time the rebuilt history
  int num_nodes_moved = 0;             // By fit_node_times_to_calibration_bounds()
};

// The starting tree of a free-topology calibrated run.  If `guide` (a data-driven tree over `tip_descs`)
// already has every calibrated sample set as a clade, it is kept.  Otherwise the constrained topology it
// induces (build_calibration_constrained_topology()) gets a parsimony mutation history and node times from
// mutation counts at `rate_per_site_per_day` (if absent, the rate implied by the guide's own timing), exactly
// as an input tree with `--v0-in-tree-branch-lengths ignore`.  Either way the node times are then moved inside
// every hard age bound.  Throws Calibration_error or std::invalid_argument with a message for the user.
auto build_calibration_constrained_initial_tree(
    Phylo_tree guide,
    Real_sequence ref_sequence,
    std::vector<Tip_desc> tip_descs,
    const Calibration_set& set,
    std::optional<double> rate_per_site_per_day,
    int num_observed_sites,
    Constrained_init_summary* summary = nullptr)
    -> Phylo_tree;

}  // namespace delphy

#endif  // DELPHY_CALIBRATION_INIT_H_
