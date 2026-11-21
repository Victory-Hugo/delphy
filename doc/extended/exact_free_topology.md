# Exact Kingman coalescent with a free topology

This extends the exact coalescent correction of `exact_coalescent.md` and `exact_skygrid.md`,
which required a fixed topology, to runs that sample the topology (stage A), and combines it with the
age calibrations of `calibration_free_topology.md` (stage B).  Both remain opt-in:
`--v0-exact-coalescent-correction` together with a free topology, and optionally
`--v0-allow-free-topology-calibration`.  Only a single tree partition part is supported
(`--v0-threads 1`); see section 5.

## 1. Targets

A state is x = (T, t, H, theta): rooted topology, inner node times, mutation histories, global parameters.

    pi_cell(x) ∝ G(x) · P_cell(t; theta, grid) · p(theta) · C(T, t)
    pi(x)      ∝ G(x) · P_exact(t; theta)      · p(theta) · C(T, t)  =  pi_cell(x) · exp(Delta(x))

    Delta(x) = log P_exact(t; theta) - log P_cell(t; theta, grid)
    C(T, t)  = prod_c f_c(age of MRCA_T(A_c)) · prod_{c requires monophyly} 1{A_c is a clade of T}

G is the sequence likelihood, P_exact the Kingman density of the node and tip times under the population curve,
P_cell the existing cell prior on the pinned coalescent grid.  Both coalescent densities depend on the
multiset of coalescence and sampling times only, never on which lineages coalesce, so Delta does not depend on T
given t, and it does not contain C.

## 2. Stage A: the whole-batch correction is exact with a free topology on one part

Between two global steps Delphy runs a batch of n local moves.  Each local move picks a move type at random
(random scan) and is a Metropolis-Hastings move with respect to pi_cell at fixed theta: node-time moves,
SPR (`Subrun::spr1_move`), subtree slide and branch reform (`Subrun::spr_move_core`), and the history moves.
With one part, the partition stencil and the auxiliary variables of the partitioning do not enter any local
move, so nothing in the batch depends on how the tree was partitioned.

- Each move type is reversible with respect to pi_cell, so their random mixture P is reversible (self-adjoint
  in L2(pi_cell)).
- The batch kernel is K = P^n, which is self-adjoint because P is:  pi_cell(x) K(x, y) = pi_cell(y) K(y, x).
- The batch is then accepted with probability alpha(x, y) = min(1, exp(Delta(y) - Delta(x))), and otherwise the
  genealogy is restored.  For x != y:

      pi(x) K(x, y) alpha(x, y) = pi_cell(x) K(x, y) min(exp(Delta(x)), exp(Delta(y))),

  which is symmetric in (x, y).  So the batch step is reversible with respect to pi.

Global moves (rate, population parameters, Skygrid smoothness, zero mode and HMC) run between batches and
already target pi with the exact density (stage of `exact_skygrid.md`); C does not depend on theta.
The joint time-scale move needs a fixed topology and is refused with topology moves enabled, in either order
of the setters.

Rollback (`Run::run_corrected_local_batch`): the tree, histories and missations are restored from a snapshot,
the derived quantities are recomputed and the parts are rebuilt with `Run::repartition`, which also re-derives
every calibration target from its sample set.  The random number generators are not restored; the
acceptance draw is independent of the batch, so this does not change the kernel.

Known point of the existing code, tested empirically rather than proven here: `spr1_move` computes the JC
rate factor and `Spr_move` from the pre-move `lambda_i_`; a root-changing SPR that changes the root sequence
could make the reverse proposal density differ from what the Hastings ratio uses.  The stage-A2 data arms
(section 4.2), which include root changes, show no measurable bias.

## 3. Stage B: calibrations with targets that follow the topology

Two minimal options were compared.

1. **Calibration term in every local MH ratio** (C in pi_cell).  This is what doc 51 already implements for
   the cell-prior target: after every topology proposal the targets are re-identified from the sample sets,
   the full calibration density of the proposed tree is computed, and a state of zero density (a required
   set that is not a clade, or a target outside its hard support) is rejected without redrawing.  The node-time
   moves use the current bindings, and the hard support restricts their proposal windows exactly.
2. **Calibration term in the whole-batch ratio** (local moves target pi_cell / C).  The inner chain would
   then wander through states where a required set is not a clade, which have zero density under pi; every
   such batch end is rejected, and the rate of such rejections grows with the batch length.  It would also
   need new code in every local move to stop applying the calibration term.

Option 1 is used.  Because C is part of pi_cell, section 2 applies unchanged: the local kernel is reversible
with respect to pi_cell including C, Delta does not contain C, and the batch ratio exp(Delta' - Delta)
carries the coalescent correction only.  No calibration factor is counted twice, and none is applied only
after the batch.

Semantics (per calibration, key `monophyly` in the calibration file, see doc 51, section 1):

- `required` (default): prior on the age of the clade A_c, and density zero when A_c is not a clade.
- `not_required`: prior on the age of MRCA(A_c); the set imposes nothing on the topology, sets may cross, and
  when two sets share their MRCA both densities apply to that node.

Code changes: `Calibration::require_monophyly` with its parser and canonical writer (written only when
`not_required`, so existing stored configurations are unchanged); `bind_calibrations` skips the clade check
and the distinct-target check for such calibrations; `Local_calibration::require_monophyly` and
`Subrun::find_calibration_targets`; the starting-tree builder treats only the required sets as constraints;
the refusal of calibrations together with the exact correction and a free topology is removed (Run setters
and CLI).

## 4. Validation

### 4.1 Unit tests (`tests/exact_free_topology_tests.cpp`)

The reference genealogies come from a forward Kingman simulation that shares no code with the production
coalescent (the hazard 1/N(t) is integrated numerically from `Pop_model::pop_at_time`).  With calibrations the
exact draws are obtained by rejection: a Kingman draw (and, for Skygrid, a curve drawn from its prior on every
attempt) is accepted with probability C / max C, with the densities, MRCAs and clade checks written out again in
the test.  Chains start from an exact draw and run 40 batches; their end points are compared with independent
exact draws (two-sample chi-squared on the rooted topology at the 1e-4 level, z of the root age, the first
coalescence age and every calibrated MRCA age, |z| < 4).  After every batch the Run's calibration term is
compared with the production densities at the independently found MRCAs (1e-8), every required set is checked
to be a clade and the state to have nonzero reference density.

| Test | Chains | Result (Release) |
|---|---|---|
| 4 tips, root uniform + clade {S0,S1} normal with hard bound | 3000 | chi2 0.2 / 2 df, z root -0.23, z ages -0.23 +0.91 |
| control: same, chains without the pair's age prior | 3000 | z first age -7.08 (detected) |
| 4 tips at 3 dates, two crossing age-only sets (shared MRCA in some topologies) | 3000 | chi2 11.3 / 14 df, z ages -0.23 +1.52; each set not a clade in 53% and 87% of states |
| 5 tips at 3 dates, two disjoint clades | 3000 | chi2 4.7 / 2 df, z ages -2.15 -0.78 |
| 5 tips, nested clades + an age-only set + root | 3000 | chi2 2.0 / 2 df, z ages -0.06 +0.60 +2.70 +1.90 |
| 5 tips, Skygrid curve sampled (staircase, log-linear), clade + root lognormal | 2000 each | chi2 7.6 and 13.8 / 14 df, all |z| < 2.1 |
| 10 tips, log-linear Skygrid, clade + age-only + root, 150 batches with rollbacks, then a checkpoint round trip and 50 more batches | 1 | coalescent term exact (1e-8), calibration term equals the independent recomputation (1e-8) after every batch |
| parser, writer and binding of `monophyly`; conflicting nested supports and a start outside a hard bound refused | - | pass |

The stage-A tests in the same file (constant population at one and three dates, Skygrid with the curve
sampled, bookkeeping through rollbacks, refusals) are unchanged and pass; the uncorrected control of the
4-tip test is biased (z of the first coalescence age +8.25).

### 4.2 Stage A2: Delphy with data against an independent posterior (campaign cluster)

Driver `tests/human_input/run_exact_free_topology_task31.sh`, references
`tests/prototypes/exact_free_topology_reference_task31.py` (importance sampling from the Kingman prior, JC69
pruning, 4e7 draws, no shared code), scoring `tests/prototypes/compare_exact_free_topology_task31.py` with the
rule fixed before any chain was read: PASS iff R-hat < 1.01, bulk and tail ESS >= 400, |z| < 4 for every
topology with reference probability >= 0.01 and for the root-age mean.  JC69 at the simulated rate (fixed),
constant timescale 20,000 years (fixed), 4 chains x 4e8 steps per arm, 20% burn-in.

| Data set | Arm | R-hat | bulk / tail ESS | z root age | worst topology \|z\| | Verdict |
|---|---|---|---|---|---|---|
| A4 (4 tips, one date) | exact, coarse grid | 0.9999 | 15669 / 16061 | -1.09 | 1.40 | PASS |
| | exact, fine grid | 1.0001 | 15825 / 15902 | -1.00 | 0.78 | PASS |
| | cell prior, coarse (control) | 0.9999 | 15396 / 15331 | -4.16 | 1.47 | FAIL |
| A5m (5 tips, recurrent sites) | exact, coarse | 1.0000 | 15748 / 15076 | -0.67 | 0.02 | PASS |
| | exact, fine | 1.0001 | 15424 / 14824 | +0.90 | 0.03 | PASS |
| | cell prior, coarse (control) | 1.0000 | 15571 / 16046 | +0.06 | 0.00 | PASS |
| A5h (5 tips, 2 ancient, 105 topologies) | exact, coarse | 1.0001 | 15398 / 14999 | -0.20 | 2.77 | PASS |
| | exact, fine | 1.0002 | 15645 / 15228 | -1.72 | 3.72 | PASS |
| | cell prior, coarse (control) | 1.0001 | 15807 / 15860 | +3.08 | 1.09 | PASS |

All six exact arms pass.  The control is discriminating only on A4: on A5m the data dominate the coalescent
term, and on A5h the shift (+3.08) stays below the gate.  The largest exact-arm deviation (A5h fine, one
topology at |z| = 3.72) is inside the gate; about 60 z values enter the exact arms, and 20 batch means per
chain may understate the Monte Carlo error.

### 4.3 CLI

`tests/cli/run_exact_correction_cli_tests.sh`: 14 checks, including a free-topology run with the
correction and an age-only calibration on one thread, and its refusal on two threads.
`tests/cli/run_exact_skygrid_cli_tests.sh`: 8 checks.

## 5. Stage C: several tree partition parts (closed, not supported)

Stage C was closed without an implementation.  The limit stays, is enforced by `Run::repartition` and the CLI,
and is stated as:

    Exact Kingman Skygrid with free topology currently requires one tree partition.

### 5.1 Why detailed balance cannot be shown for the current mechanism

The fixed-topology proof treats the partition stencil z as an auxiliary variable drawn from q(z | x) before
each batch.  With a fixed partition, each part's local moves are reversible with respect to pi_cell.  With
several parts, the step "draw z, then run K_z" is reversible with respect to pi_cell only if
q(z | x) = q(z | y) for every y the batch can reach.  Otherwise the acceptance ratio would need the factor
q(z | y) / q(z | x).  Three properties of the current code break this once the topology is free.

1. **The stencil depends on the state.**  `generate_random_partition_stencil` (`core/tree_partitioning.h`)
   walks the tree in a randomized post order, accumulates subtree sizes and cuts, with probability 1/2, where
   a size threshold is reached.  An SPR inside a part changes subtree sizes, so q(z | T') != q(z | T) in
   general.  q(z | T) is the marginal probability of that randomized greedy procedure.  It has no closed form
   and is too costly to enumerate, so the missing factor cannot be added to the ratio.
2. **Stencils are cached and refreshed.**  `Run::refresh_partition_stencils` generates 10 stencils from the
   tree of the moment, stores them as node indices, and draws one of them for about 200 batches before it
   generates new ones.  Once the topology has changed, the same node indices name different subtrees, so z is
   not even a draw from q(z | current tree).  Its distribution depends on the history of the chain.  The
   auxiliary-variable argument requires z to be drawn from a fixed conditional of the current state.
3. **Calibrations cross part boundaries.**  A topology move inside one part can take a cut-point subtree into
   or out of a calibrated sample set.  It can also change the MRCA of the set or whether the set is a clade,
   and the part cannot see either effect.  A part therefore cannot evaluate the calibration term of its own
   proposals, and the calibration factor would be wrong in the local MH ratios.  The same reason already
   restricts free-topology calibrations to one part (doc 51, section 3).

With one part none of this matters: the stencil does not enter any move, and every move sees the whole tree.
That is the case proven in section 2.

### 5.2 What was deliberately not done

- No A4/A5h comparison of several parts against one part was run on the campaign cluster.  Such a run could only bound a bias.
  It could not establish correctness, and it would not justify removing the limit.
- The partitioning algorithm is unchanged, and the limit is not relaxed.

### 5.3 Follow-up research task

The exact Kingman Skygrid with a free topology on several tree partition parts is left as a separate research
task.  Candidate directions:

- a partitioning distribution that does not depend on the state, or one whose dependence can be evaluated;
- fixed part boundaries kept over a batch, with q(z | T) depending only on quantities that the within-part
  moves preserve (the sample set of each part and the nesting of the parts);
- extended-state Metropolis-Hastings, with z part of the target and moves on z that have their own
  acceptance ratio.

None of these is assumed to work.  In particular, fixed boundaries may make some topologies unreachable, and
they need their own argument.  Any candidate must establish all of the following before it is implemented:

1. detailed balance (or invariance) of the combined kernel with respect to the exact target, including the
   whole-batch correction;
2. irreducibility on the topology space: every rooted topology of positive posterior density must stay
   reachable when parts cannot exchange subtrees within a batch;
3. correctness of calibrations whose sample sets span several parts, including MRCA identity changes and the
   monophyly constraint.

Validation would then use 2 and 4 parts against the one-part target and the independent references of
section 4.  The measured runtime, peak memory and ESS per hour must not be worse than the one-part run.
Adopting any such scheme is a change of the sampler.  It needs explicit approval under the project's rules
on new MCMC machinery.

## 6. Other limits

- Every sample needs an exact date (as for the correction with a fixed topology).
- The joint time-scale move is not available with a free topology.
- Not yet run for this extension: BEAST comparison, simulation-based calibration over independent data sets,
  the 8 to 20 tip and larger tiers, and the pathogen regression of the final binary.
