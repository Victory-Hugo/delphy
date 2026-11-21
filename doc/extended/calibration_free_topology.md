# Age calibrations with a free topology and monophyly constraints

This extends the calibrations of `calibration.md`, which required a fixed topology, to runs that
sample the topology.  It is opt-in: `--v0-allow-free-topology-calibration`.  Without that option every
existing workflow, including fixed-topology calibration, behaves exactly as before (byte-identical output
for the same seed).

## 1. Target

    log pi(state) = log G (sequence likelihood)
                  + log coalescent prior (the existing cell prior on the coalescent grid)
                  + log priors on the global parameters
                  + sum_c log f_c(age of MRCA(A_c))
                  + sum_{c requires monophyly} log 1{A_c is a clade}

`A_c` is the sample set of calibration `c` (`taxa all` names the root and is always a clade).  `f_c` are the
existing Normal, Lognormal and Uniform densities with their hard bounds.  The two semantics are selected per
calibration with the optional key `monophyly`:

- `monophyly required` (the default, and the only behaviour of files written before the key existed): a tree
  in which `A_c` is not a clade has density zero, and `f_c` is the prior on the age of that clade.
- `monophyly not_required`: `f_c` is the prior on the age of MRCA(A_c), whatever else lies below it.  The
  set imposes nothing on the topology; two such sets may cross, and may share their MRCA in some
  topologies, in which case both densities apply to that node.  Not allowed with `taxa all`.

The canonical text written into `.dphy` files contains `monophyly not_required` only for the second kind, so
the stored configuration of every existing calibration is byte-identical.

The calibration file format is unchanged.  A target is always identified by its sample IDs, never by a
node index: node indices are re-derived from the sample sets after every topology proposal.

## 2. Moves

All topology changes in the local move mix go through two functions:

- `Subrun::spr1_move` (the SPR move), and
- `Subrun::spr_move_core`, used by the subtree slide (both its topology-changing and its height-only
  branches) and by the branch reform move when P is the root.

`inner_node_displace`, `tip_displace`, `node_time_marginal` and the non-root `branch_reform` path never
change the topology.  The history Gibbs moves change neither topology nor node times.  The Wilson-Balding
move is not in the mix.  At the Run level, only the joint time-scale move changes node times, and it
requires the exact coalescent correction, which is refused with a free topology.

In both topology functions, when the calibration targets follow the topology (topology moves enabled and
at least one calibration), after the proposed tree is built:

1. every target is re-identified as the MRCA of its sample set on the proposed tree, and every set is
   checked to be a clade (`Subrun::find_calibration_targets`);
2. the full calibration log density of the proposed tree is computed, -infinity if a set is not a clade
   or a target is outside its hard support;
3. `delta = new full value - current value` is added to the existing log MH ratio, which keeps its
   proposal ratio, SPR-study terms and mutation-history terms unchanged;
4. an infinite `delta` is a rejection (no redraw, so the proposal kernel and its Hastings ratio are
   those of the unconstrained move);
5. on rejection the tree and mutation history are restored by the existing rollback, and the target
   bindings are restored from a saved copy.

This is the standard MH construction for a target restricted to a subset: proposals outside the subset
are rejected, so detailed balance with respect to the restricted target holds whenever the unrestricted
kernel is reversible.  The time moves use the bindings, which are current after every accepted topology
move, so they need no change.

With topology moves disabled the old code path is used unchanged (the pre-check on P's hard support,
then the incremental delta on P).

## 3. Partitioning

Only a single tree partition part is supported (`--v0-threads 1`).  With several parts, a topology move
inside one part could move a cut-point subtree into or out of a calibrated clade (or change an MRCA), which
that part cannot see.  `Run::repartition` throws `std::invalid_argument` if there is more than one part, and the CLI
refuses the option without `--v0-threads 1`.

After `Run::reassemble`, the Run's own bindings are re-derived with `bind_calibrations_ignoring_times`,
which also re-checks that every set is still a clade.

## 4. Starting tree

Without `--v0-in-tree`, `build_calibration_constrained_initial_tree` (`core/calibration_init.h`):

1. checks that every ID is a sample of the alignment, that each set has at least 2 samples, that no two
   calibrations that require monophyly name the same set or both the root, and that any two such sets are
   nested or disjoint (crossing sets are rejected with the pair named); sets with `monophyly not_required`
   are not constraints and are not part of these checks;
2. builds the existing data-driven tree (default `mp-plus-timing`); if every set is already a clade there,
   that tree is kept;
3. otherwise builds a binary topology in which every set is a clade: inside each calibrated clade, and at
   the top, the maximal calibrated subclades and loose samples are arranged as the data-driven tree
   arranges one representative sample of each.  The mutation history comes from Fitch parsimony and the
   node times from mutation counts, exactly as for `--v0-in-tree --v0-in-tree-branch-lengths ignore`,
   at the rate of `--v0-init-mutation-rate` if given, else the rate implied by the data-driven tree's
   own timing;
4. moves inner node times (never tip times) so that every target is strictly inside its hard support and
   every parent stays older than its children, mapping each moved branch's mutation times linearly onto
   the new branch, and fails with a message naming the calibration if that is impossible.

The start only has to be a valid state; it does not change the target.

With `--v0-in-tree`, the given tree is used as is: every set must already be a clade of it and every
target inside its hard support, as in the fixed-topology mode.

## 5. Supported and unsupported combinations

Supported: one tree partition part; the existing strict clock (fixed or sampled rate); the existing
Normal / Lognormal / Uniform calibrations and hard bounds; `taxa all` plus any laminar family of
monophyletic sets; the existing coalescent cell prior with the exponential, constant or Skygrid
population model; resuming from a saved state (the topology flag and the calibrations are stored).

Without `--v0-exact-coalescent-correction` the coalescent term is the cell prior on the coalescent grid, a
numerical approximation of the Kingman density.  With it, the exact Kingman density is the target; the
calibration term stays in the local moves and the correction depends on node times only, so the two compose
without counting any factor twice (see `exact_free_topology.md`, section 3).  The joint
time-scale move needs a fixed topology and is refused.

Not supported and refused: several tree partition parts; `--v0-fix-topology` together with the option;
crossing sample sets that both require monophyly.

## 6. Validation (Release and Debug builds)

`tests/free_topology_calibration_tests.cpp`:

- starting trees for root-only, one inner clade, nested, disjoint, hard-bound fitting, and the rejections
  (crossing sets, unknown IDs, duplicate sets, two root calibrations, incompatible hard bounds);
- exact reference: 4 samples without data, a constant population, the constraint that {S000, S001} is a
  clade and a uniform prior on its age.  Every ranked history has probability 1/18 independently of the
  times, so the topology probabilities and the mean clade age have closed forms in terms of the first two
  coalescence times (see the test).  The chain is compared with them for a flat window (constraint only:
  0.5 / 0.25 / 0.25) and for a window that reweights the topologies (balanced 0.729);
- an 8-sample chain with nested calibrations and a root calibration, checked after every batch against
  independent recomputations (targets, clade membership, hard bounds, calibration term, part totals,
  tip sequences), with the topology and the root split changing; and a control showing that the same
  chain without the constraint leaves the clade;
- a checkpoint round trip that resumes sampling.

In Debug builds `Subrun::check_derived_quantities` also verifies, after every local move, that the
bindings equal a fresh re-identification and that every set is a clade.
