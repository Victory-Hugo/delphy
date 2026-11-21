# Internal-node (MRCA) age calibrations: mathematical design

This document is the design that the implementation follows; section 5 is the operator coverage audit
that the task requires before coding.

Note: this design requires a fixed topology.  Calibrations with a sampled topology, where every
calibrated sample set is constrained to be monophyletic, are described in
`calibration_free_topology.md` (opt-in with `--v0-allow-free-topology-calibration`).

## 1. Scope and the one sentence that matters

A calibration is a prior density on the **age of the most recent common ancestor of a named set of
sample IDs**.  It is a term of the posterior that Delphy samples, it enters every
Metropolis-Hastings ratio of every move that can change the age of a calibrated node, and it is
never applied by rescaling a finished tree.

In scope for this task: single-threaded and multi-partition runs with a **fixed rooted topology**
(`--v0-fix-topology`), normal / lognormal / bounded-uniform age priors with optional hard bounds,
`ka` / `years_bp` / `cal_bp` / `ce` / `bce` age conventions, joint sampling of the strict-clock rate
and node times with no external rate information, and exact agreement between the global and the
partition-local posterior.

Out of scope (stated, not silently skipped): free-topology calibration (the MRCA node can change
identity under SPR -- Task 3.3), strictly fixed ("sigma = 0") node ages, calibrations on tip ages
(that is sampling-date input, Task 1.2), and a CLI resume from `.dphy` (the CLI has none).

## 2. Posterior

Delphy's posterior (`core/run.h: log_posterior()`) gains a fourth term:

```
log P = log G + log P_coal + log P_other + log P_cal
```

with

```
log P_cal = sum_{c in C} log f_c( a_c( t_{n(c)} ) )
```

where

* `C` is the set of calibrations;
* `T_c` is the stable set of sample IDs of calibration `c`;
* `n(c)` is the MRCA of `T_c` in the tree.  Under a fixed topology `n(c)` is determined once, at
  initialization, and the taxon set -- never a node index -- is the definition.  `T_c` must be
  monophyletic (section 4);
* `t_{n(c)}` is that node's time in days since 2020-01-01;
* `a_c(t) = (t_ref(c) - t) / 365` is the node's **age in years** before the calibration's reference
  instant `t_ref(c)` (days since 2020-01-01), with 1 year = 365 days, as everywhere else in the CLI;
* `f_c` is a density on that age.

The map `t -> a` is affine with constant Jacobian `|da/dt| = 1/365`.  A constant factor is
irrelevant to every MH ratio, so `log P_cal` is reported as the sum of log densities **on age in
years**, and the state-independent `log(1/365)` per calibration is omitted.  Hard bounds truncate
the support without renormalizing, for the same reason.  Both omissions are documented here because
they make the reported `log_calibration_prior` a log density up to a fixed additive constant, which
is exactly the convention Delphy already uses for its Gamma prior on `mu`
(`run.cpp: calc_cur_log_other_priors`).

### 2.1 Densities

With `a` the age in years, all parameters converted to years of age (section 3):

| `distribution` | parameters | `log f(a)` | support |
|---|---|---|---|
| `normal` | `mean m`, `stddev s > 0` | `-log(s) - log(sqrt(2 pi)) - (a-m)^2 / (2 s^2)` | `(-inf, +inf)` |
| `lognormal` | `median M > 0`, `sigma S > 0` | `-log(a) - log(S) - log(sqrt(2 pi)) - (log a - log M)^2 / (2 S^2)` | `(0, +inf)` |
| `uniform` | `oldest a_hi`, `youngest a_lo`, `a_lo < a_hi` | `-log(a_hi - a_lo)` | `[a_lo, a_hi]` |

`oldest` / `youngest` are optional for `normal` and `lognormal`, where they act as hard bounds, and
required for `uniform`, where they are the support.  Outside the support `log f = -inf`.

A degenerate `stddev = 0` or `sigma = 0` is **rejected**, not approximated: a point mass on a node
age needs its own time operator (the current proposals have zero probability of hitting a point),
and squeezing it into a density would silently produce an unsamplable posterior.

### 2.2 The delta used by the samplers

For a move that changes node `n` from `t_old` to `t_new`:

```
delta log P_cal = sum_{c : n(c) = n} [ log f_c(a_c(t_new)) - log f_c(a_c(t_old)) ]
```

and the move's acceptance probability becomes

```
log_mh = delta log G + delta log P_coal + delta log P_cal - log( q(old->new) / q(new->old) )
```

The current state is always inside every support (validated at initialization, and no move that
leaves the support is ever accepted), so `log f_c(a_c(t_old))` is always finite and the delta is
never `inf - inf`.  `delta log P_cal` is computed as `-inf` whenever the proposed state is outside
the support, without evaluating the difference.

### 2.3 Proposal support clipping

`inner_node_displace_move` draws a non-root node time from a bounded exponential on `[t_min, t_max]`
and a root time from a Gaussian.  `t_min` and `t_max` are functions of the parent time, the child
times and the mutation times on the adjacent branches, none of which this move changes; so they are
identical for the forward and the reverse proposal, which is what makes the existing
`log_alpha_old_to_new_over_new_to_old = d_logG_dt * (t_new - t_old)` correct.

The hard support of a calibration is a constant interval in `t`.  Intersecting `[t_min, t_max]` with
it therefore changes the proposal in the same way in both directions, and the existing Hastings term
stays exact.  The implementation does this clipping for the bounded-exponential proposal (otherwise a
narrow bounded-uniform calibration would reject almost every proposal); the root's Gaussian proposal
is symmetric and out-of-support proposals are simply rejected.

### 2.4 Rate and time are sampled jointly

`log P_cal` depends on node times only, never on `mu`.  Delphy's `mu` move
(`run.cpp: mu_move`) Gibbs-samples `mu ~ Gamma(M + alpha_mu, Ttwiddle + beta_mu)` from the complete
conditional of `log G` and the `mu` prior; since the calibration term is independent of `mu`, that
move remains an exact Gibbs step of the calibrated posterior and needs no change.  Node times move
under the local moves, which now carry `delta log P_cal`.  The chain therefore samples the joint
posterior of `(mu, tree)` and the absolute time scale is identified by the calibration rather than by
tip-date spread or by a rate prior.  Nothing is rescaled after the fact.

The cost is statistical, not mathematical: `mu` and the node times are strongly correlated under a
calibration, and this Gibbs-plus-local-moves scheme has no joint rate/time scaling operator, so
mixing in that direction is by random walk.  It is measured, not assumed (section 7 of the report).

## 3. Age conventions and units

The config file states one convention per calibration; there is no default.

| `age_units` | parameter values are | `t_ref` | `age_reference_year` |
|---|---|---|---|
| `ka` | thousands of years of age (1 ka = 1000 x 365 days) | 1 Jan of `age_reference_year` | required |
| `years_bp` | years of age | 1 Jan of `age_reference_year` | required |
| `cal_bp` | calibrated radiocarbon years of age | 1 Jan 1950 CE | forbidden (implied 1950) |
| `ce` | calendar year CE (>= 1) | 2020-01-01 (`t = 0`) | forbidden |
| `bce` | calendar year BCE (>= 1; 1 BCE is astronomical year 0) | 2020-01-01 (`t = 0`) | forbidden |

Calendar conversions reuse `astronomical_year_to_day` from `core/human_dates.h` (Task 1.2), which
works in the proleptic Gregorian calendar with astronomical year numbering, so there is no year 0
and the leap-year rules hold for negative years.

Location parameters (`mean`, `median`, `oldest`, `youngest`) are converted value by value to years of
age; dispersion parameters (`stddev`) are converted by scale only (`ka` -> x1000, otherwise x1);
`sigma` of the lognormal is dimensionless.  For `ce` / `bce`, where larger values are *younger*, the
conversion flips the order, which is why the bounds are named `oldest` / `youngest` instead of
lower / upper: the names mean the same thing in every convention.  `lognormal` is rejected with
`ce` / `bce` units, because a lognormal on a calendar year has no meaning.

## 4. Binding a calibration to a node, and what is rejected

Validation happens in two layers.  Every problem found is reported together, with the line number of
the offending stanza, and nothing is guessed or silently repaired.

Config only (at parse time):

1. `version 1` header required; unknown keys, duplicate keys, missing required keys are errors.
2. Duplicate calibration names are errors.
3. `stddev > 0`, `sigma > 0`, `median > 0`, `oldest > youngest` (in age), finite values only.
4. `age_reference_year` present exactly for `ka` / `years_bp`.
5. At least two taxa, or `taxa all`; duplicate IDs inside one set are errors.

Against the tree (at initialization, and re-verified at every repartition):

6. Every named sample ID is a tip of the tree; unknown IDs are listed.
7. `taxa all` binds to the tree root.  **The root of the input tree is the MRCA of the sampled
   lineages, not by assumption the MRCA of all human mtDNA**; the CLI prints this as a note and the
   name of the calibration is only a label.  Nothing in Delphy knows what "root", "M" or "N" mean.
8. `T_c` must be **monophyletic**: the MRCA of `T_c` must have exactly `T_c` below it.  Otherwise the
   input is rejected and the extra tips are listed.  (The alternative semantics -- calibrating the
   age of the MRCA of a non-monophyletic set -- is well defined but is not what a haplogroup
   calibration means, and under a fixed topology it is always a sign of a mismatch between the
   calibration file and the tree.)  A calibration with the optional key `monophyly not_required`
   opts into that alternative semantics explicitly: it binds to the MRCA of `T_c` whatever else lies
   below it.  See `calibration_free_topology.md`, section 1.
9. Two calibrations that require monophyly may not bind to the same node.
10. The MRCA must be an inner node, so a one-taxon "clade" is rejected by rule 5; tip ages are input
    data (Task 1.2), not calibrations.
11. **Compatibility.**  For every pair where `n(c1)` is a strict ancestor of `n(c2)`, the supports
    must admit `age(c1) > age(c2)`, i.e. `sup supp(c1) > inf supp(c2)`; otherwise the two intervals
    can never be satisfied at once and the input is rejected naming both calibrations.  For every
    calibration, `sup supp(c)` must exceed the age of the youngest tip in its clade, and the age must
    be able to be positive.
12. The initial tree must already lie inside every hard support; otherwise the run would start at
    `log P = -inf`.  This is an error that names the node, its initial age and the bound.  Delphy does
    not move the initial tree into the support by itself.

Items 6-12 are re-checked, cheaply, every time the tree is repartitioned, so a drift of the binding
cannot pass unnoticed.

## 5. Operator coverage audit

Every code path that can change a node time, from `core/subrun.cpp` and `core/run.cpp`.  "Needs
delta" means the calibration term can change and the MH ratio must carry it.

Line numbers are those of the implementation commit.

| Operator | Location | Times it changes | Needs delta | Treatment |
|---|---|---|---|---|
| `inner_node_displace_move`, non-root | `subrun.cpp:200`, MH at `286` | one inner node | **yes** | `delta log P_cal` added to `log_mh`; `[t_min,t_max]` clipped to the node's hard support at `subrun.cpp:228` (see 2.3) |
| `inner_node_displace_move`, root (root part only) | same function, Gaussian branch | run root | **yes** | same delta, no clipping of the Gaussian (symmetric proposal; out-of-support draws are rejected) |
| `tip_displace_move` | `subrun.cpp:301` | one tip | no | a calibration target is an inner node by construction (rule 10), and frozen cut-point "tips" have `t_min == t_max` and return early.  A `DCHECK` asserts that no calibration binds to a tip |
| `branch_reform_move`, non-root path | `subrun.cpp:359` | none (mutation times only) | no | proven zero |
| `branch_reform_move`, root path | calls `spr_move_core(X, {S, t_P}, 1)` | none: regrafts at the same time | no (delta is identically 0) | covered by the generic code in `spr_move_core` |
| `subtree_slide_move`, no hop | `subrun.cpp:425` -> `spr_move_core(X,{S,new_P_t},1)` | `t_P` | **yes** | `spr_move_core:794` computes the delta for `P`, `816` puts it in `log_mh`, `826` updates the running value |
| `subtree_slide_move`, hop | `subrun.cpp:425` (two `spr_move_core(X,{SS,...})` calls) | `t_P` + topology | n/a | blocked in fixed-topology mode (Task 2.1); calibration mode requires fixed topology |
| `spr1_move` | `subrun.cpp:567` | `t_P` + topology | n/a | not selected when topology moves are off; a `CHECK` at the top of the move refuses to run it at all if calibrations are present |
| `wilson_balding_move` | `subrun.cpp:525` | -- | n/a | defined, never called by `mcmc_sub_iteration` |
| global moves: `mu_move`, `mpox_hack_moves`, `hky_frequencies_move`, `hky_kappa_move`, `alpha_moves`, `gibbs_sample_all_nus`, `pop_size_move`, `pop_growth_rate_move`, `skygrid_tau_move`, `skygrid_gammas_*` | `run.cpp:run_global_moves` and below | none | no | proven zero: no global move changes a node time.  `mu_move` stays an exact Gibbs step (2.4) |
| coalescent grid retarget (`set_coalescent_prior_t_step`) | `run.cpp:run_global_moves` | none | no | proven zero |
| `Run::repartition` (`run.cpp:110`), `reassemble`, `normalize_root` | `run.cpp` | none | no | rebinding and re-validation happen in `assign_calibrations_to_subruns` (`run.cpp:225`), called from `repartition` at `run.cpp:192` |

## 6. Global and partition-local consistency

`Run` keeps `log_calibration_prior_` as a derived quantity, recomputed from the whole tree and the
bound nodes.  Each `Subrun` keeps the partial sum over the calibrations assigned to it and maintains
it incrementally inside the moves.

Assignment, performed in `Run::repartition()` after the subruns are built: a calibration is assigned
to the one part in which its node is a **movable** inner node, that is, an inner node of that part's
subtree which is either not the subtree root or is the run root in the root part.  Otherwise the
node is a cut point and is frozen for the whole local batch:

* a non-root subroot is never moved (`subrun.cpp:161`, `spr_move_core:689`, `spr1_move:527`), and
* the same node appears in its parent part as a "tip" with `t_min == t_max == t`, where
  `tip_displace_move` returns before proposing (`subrun.cpp:243`).

A calibration is therefore assigned to at most one part, and the sum of the frozen contributions is
constant during a batch.  This is exact, not an approximation, so multi-partition runs are supported
rather than refused.

`Subrun` keeps its partial sum outside the lazily validated block: it is set eagerly by
`set_calibrations` and then changed only by the deltas inside the moves, so its value is evidence
about the moves and not a by-product of a recomputation.  Three checks compare it with independent
recomputations:

* `Subrun::check_derived_quantities`, which in a Debug build runs **after every single local move**,
  compares the incrementally maintained value with `calc_cur_log_calibration_prior()` recomputed from
  the part's own subtree.  This is what covers the accept, reject and rollback paths.
* `Run::check_global_and_local_totals_match` (Debug) checks
  `log_calibration_prior_ == sum_parts subrun.log_calibration_prior() + frozen_calibration_log_prior_`.
* `Run::log_calibration_prior_from_parts()` exposes the same sum so that Release-build tests can
  compare it with the global value and with a fully independent recomputation that resolves the taxon
  sets by name from scratch.

These are the checks that turn "a prior was added somewhere" into "the sampler uses it everywhere".

## 7. Persistence

`.dphy` / the API `Params` table gains one optional field, `calibration_config: string (id: 48)`,
holding the canonical text of the calibration configuration.  FlatBuffers ignores unknown fields and
defaults absent ones, so:

* a `.dphy` written before this change loads with no calibrations, which is the correct meaning;
* a calibrated run round-trips its calibrations exactly, and `apply_api_params_to_run` re-binds them
  against the restored tree and **throws** if they no longer validate, so a calibration can never be
  silently dropped or silently re-pointed at a different node.

The format version is not bumped and no existing field changes meaning; the `.dphy` reader for older
files is covered by a regression test.

## 8. Known limitations carried into the report

* Free topology is refused, not approximated.
* No joint rate/time scaling operator: mixing of the `(mu, times)` direction is a random walk.
* `.dphy` stores node times as float32 (R-02): at 140 ka the spacing is 4 days, far below any
  calibration width used here, but a resumed run's node times differ from the saved ones by up to
  half a spacing.  The calibration density itself is computed in `double` throughout; ages are formed
  as `(t_ref - t) / 365` in `double` with no float32 intermediate.
* The 400-cell default coalescent grid biases node times at N >= 1000 (Task 1.3 N-01); calibration
  does not fix it and the validation runs therefore set the grid explicitly.
