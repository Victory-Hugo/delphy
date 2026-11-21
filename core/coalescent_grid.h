#ifndef DELPHY_COALESCENT_GRID_H_
#define DELPHY_COALESCENT_GRID_H_

#include <vector>

#include "phylo_tree.h"
#include "pop_model.h"

namespace delphy {

// Delphy's coalescent prior (`Scalable_coalescent_prior` and `Very_scalable_coalescent_prior_part`)
// does not evaluate the Kingman density
//
//   log P = - int dt k(t) (k(t) - 1) / (2 N(t)) - sum_{coalescences i} log N(t_i)
//
// directly.  It partitions the time axis into cells of fixed width `t_step` and, in each cell c,
// replaces k(t) by its time average kbar_c and 1/N(t) by the reciprocal 1/Nbar_c of the time average
// of N(t):
//
//   log P_cells = - sum_c t_step kbar_c (kbar_c - 1) / (2 Nbar_c) - sum_i log N(t_i).
//
// This is what makes the prior cheap to update incrementally: moving one node changes kbar only in
// the cells the node crosses.  It is also an approximation.  The `sum_i log N(t_i)` term is shared,
// so the whole difference sits in the integral, and it has two distinct sources.  Write <.>_c for
// the time average over cell c, and let
//
//   J_c := int_c dt / N(t)        (what `Pop_model::intensity_integral` returns for the cell)
//
// so that the exact and the cell-wise contributions of cell c to the integral are
//
//   exact_c = (1/2) int_c dt (k(t)^2 - k(t)) / N(t),
//   cells_c = (1/2) t_step (kbar_c^2 - kbar_c) / Nbar_c.
//
// Then, splitting the difference at the intermediate quantity (1/2) J_c <k^2 - k>_c,
//
//   cells_c - exact_c = [ (1/2) J_c (<k^2-k>_c - (kbar_c^2 - kbar_c)) ]            (I)   <- sign-definite
//                     - [ (1/2) (J_c <(k^2-k)/N>_c t_step / J_c - J_c <k^2-k>_c) ] (II)
//                     - [ (1/2) (kbar_c^2 - kbar_c) (J_c - t_step / Nbar_c) ].     (III)
//
// Term (I) equals (1/2) J_c Var_c(k), because <k^2> - <k>^2 = Var(k), and is therefore always
// non-negative: the cell average of a convex function of k is below the average of the function.
// Terms (II) and (III) vanish when N(t) is constant across the cell, in which case
//
//   cells_c - exact_c = t_step Var_c(k) / (2 Nbar_c)   [N(t) constant on cell c]        (*)
//
// and the whole difference is non-negative.  Term (III) is also sign-definite on its own, since the
// arithmetic-harmonic mean inequality gives J_c >= t_step / Nbar_c, but term (II) -- the correlation
// between k(t) and 1/N(t) inside the cell -- has no fixed sign.  So (*) is an identity only under
// the stated condition, and must not be assumed for a free exponential model or for a Skygrid curve
// whose knots do not coincide with the grid: there the cell prior can fall either side of the exact
// density, and only a direct comparison settles it.
//
// For that reason `Coalescent_grid_excess::excess` below is the *measured* difference
// `log P_cells - log P_exact`, evaluated for whatever population model is supplied, and the
// variance term (*) is reported alongside it as the leading contribution.  Both are exact: the
// population model's own `intensity_integral` is used for the exact side, so no quadrature enters.
//
// Two consequences matter for inference.  First, the difference is a functional of the genealogy,
// not an additive constant, so it does not cancel in Metropolis-Hastings ratios.  Second, Var_c(k)
// is large wherever many coalescences fall inside one cell, which for samples that share a sampling
// date is the most recent part of the tree, where the coalescence rate is k(k-1)/2N with k close to
// the number of tips.  The cell count needed for a given accuracy therefore grows with the number
// of tips, and is not a property of the time span alone.
struct Coalescent_grid_excess {
  double excess = 0.0;           // Nats.  Measured log P_cells - log P_exact.  Sign not guaranteed.
  double variance_term = 0.0;    // Nats.  sum_c J_c Var_c(k) / 2 >= 0; equals `excess` under (*).
  double t_step = 0.0;           // Cell width used, in days
  int num_cells = 0;             // Cells spanned by the genealogy
  double worst_cell_excess = 0.0;  // Largest single-cell contribution to `excess`, by magnitude
  double worst_cell_t_lo = 0.0;    // Lower time bound of that cell, in days

  // Cells in which the population size underflowed to zero, or the exact rate integral overflowed,
  // and no comparison is possible.  `Scalable_coalescent_prior` keeps going in that situation by
  // substituting a floor of 1e-100 for the cell's population size, so the sampler does not stop, but
  // the two sides of the comparison are then not measuring the same thing.  A non-zero count makes
  // `excess` a lower bound on the real difference and has to be reported, not hidden.
  int num_degenerate_cells = 0;
};

// Tip and coalescence times of a genealogy, in Delphy's internal time units (days since 2020-01-01).
// Tip times are the sampled times `t`, not the date bounds.
struct Coalescent_event_times {
  std::vector<double> tip_times;
  std::vector<double> coalescence_times;
};

auto coalescent_event_times_of(const Phylo_tree& tree) -> Coalescent_event_times;

// One change in the number of active lineages, as seen walking forwards in time.
struct Lineage_change {
  double t;
  int delta_k;  // +1 at a coalescence (one lineage becomes two), -1 at a tip
};

// The tip and coalescence times merged into one list of lineage-count changes, sorted by time, with
// coalescences before tips at the same time: a lineage must exist before it can be sampled, which is
// the tie-breaking rule of `Coalescent_prior::calc_log_prior`.  Shared so that everything evaluating
// a coalescent density walks the same events in the same order.
auto merged_lineage_changes(const Coalescent_event_times& events) -> std::vector<Lineage_change>;

// Measured difference between the cell-based coalescent prior and the exact Kingman density of the
// same genealogy under the same population model.  Cells are aligned so that a boundary falls
// exactly on `t_ref`, as in `Scalable_coalescent_prior`, and run backwards over the genealogy.
// `t_ref` should be the latest tip time, and `t_step` must be positive.
auto calc_coalescent_grid_excess(
    const Coalescent_event_times& events,
    const Pop_model& pop_model,
    double t_ref,
    double t_step)
    -> Coalescent_grid_excess;

// The exact Kingman log density of a genealogy, for cross-checking a cell-based prior.
// Costs O(n log n) in the number of nodes, so it is a reference, not a sampler component.
auto calc_exact_coalescent_log_prior(
    const Coalescent_event_times& events,
    const Pop_model& pop_model)
    -> double;

// The same quantity evaluated the way Delphy's samplers do, on a grid of fixed-width cells.
// Provided so that the two can be compared without constructing a `Scalable_coalescent_prior`.
auto calc_cell_coalescent_log_prior(
    const Coalescent_event_times& events,
    const Pop_model& pop_model,
    double t_ref,
    double t_step)
    -> double;

// Exact change in the coalescent log prior when one coalescence moves from `old_t` to `new_t`, with
// every other node held fixed.  Moving a coalescence back in time adds one lineage over the interval
// it crosses, and since C(k+1,2) - C(k,2) = k, the exact change in the integral term is an integral
// of k(t)/N(t) over that interval and nothing else:
//
//   delta = -/+ int_{min}^{max} dt k(t) / N(t) - log( N(new_t) / N(old_t) ),
//
// where k(t) is the lineage count with the coalescence at the later of its two positions, and the
// sign is minus when lineages are added.  The identity holds for any move that leaves a genealogy,
// that is, one after which every lineage count is still positive; it says nothing about a proposal
// that would place a coalescence above all of the tips.
// This is the quantity a Metropolis-Hastings ratio actually needs, and it can be evaluated exactly
// in time proportional to the number of events inside the interval, with memory proportional to the
// number of nodes and independent of the time span.  It is the reference against which the cell
// prior's incremental update is judged, and the basis of the event-driven scheme discussed in
// doc/extended/coalescent_grid.md.
auto calc_exact_delta_log_prior_after_displace_coalescence(
    const Coalescent_event_times& events,
    const Pop_model& pop_model,
    double old_t,
    double new_t)
    -> double;

// Coarsest cell width, among `min_cells`, 2 `min_cells`, 4 `min_cells`, ... up to `max_cells`, whose
// discretization excess has magnitude at most `tolerance_nats`.  Returns the width for `max_cells`
// if none qualifies; the caller is expected to report the excess that remains.  The search is over a
// fixed geometric ladder rather than a bisection because the excess is not guaranteed to be monotone
// in the cell count for every genealogy and population model, and a ladder gives a reproducible
// answer either way.
auto choose_coalescent_grid_t_step(
    const Coalescent_event_times& events,
    const Pop_model& pop_model,
    double t_ref,
    double tolerance_nats,
    int min_cells,
    int max_cells)
    -> Coalescent_grid_excess;

}  // namespace delphy

#endif  // DELPHY_COALESCENT_GRID_H_
