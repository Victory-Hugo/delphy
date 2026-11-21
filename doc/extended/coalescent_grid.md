# Coalescent prior at deep time scales: audit and design

This document is the mathematical and source-level audit of Delphy's coalescent prior at human time
scales, and the design of the changes made in this task.  The measured results are in
`10_coalescent_report.md`.

Scope boundary: this task may fix numerical errors, unit errors and memory allocation problems in
the coalescent machinery that have explicit evidence behind them.  It does not add MCMC operators,
does not change Skygrid's model or its moves, and does not rewrite the free-topology search.

---

## 1. Four different grids, and why they must not be conflated

Four distinct discretizations appear in and around the coalescent prior.  Only the first is a
numerical approximation; refining the others changes the model or only the picture drawn for a user.

| | What it is | Where | Controlled by | Refining it |
|---|---|---|---|---|
| **G1 Coalescent integration grid** | Cells of fixed width `t_step` over the tree span, in which the lineage count k(t) and the population size N(t) are each replaced by their time average | `core/scalable_coalescent.cpp` (`k_bars_`, `popsize_bars_`); `core/very_scalable_coalescent.cpp` (`k_bar_p_`, `k_twiddle_bar_p_`, `popsize_bar_`) | `--v0-target-coal-prior-cells`; initial width now `--v0-coal-prior-init-cells`; retargeted in `Run::run_global_moves` step 4-pre (`core/run.cpp`) | **Reduces a numerical error.** The target is the exact Kingman density |
| **G2 Population-history parameter grid** | Skygrid knot times `x_0 < ... < x_M` and the log-population values `gamma_k` at them | `core/pop_model.h` `Skygrid_pop_model`; built in `core/cmdline.cpp` | `--v0-skygrid-num-parameters`, `--v0-skygrid-cutoff` or `--v0-skygrid-first-knot-date` / `--v0-skygrid-last-knot-date` | **Changes the model.** More knots is a different, more flexible prior, not a more accurate one |
| **G3 Skygrid smoothing prior** | The Gaussian Markov random field on the `gamma_k`, with precision `tau` | `Run::calc_cur_log_skygrid_prior` (`core/run.cpp`); `tau` derived in `core/cmdline.cpp` | `--v0-skygrid-prior-double-half-time`, `--v0-skygrid-tau`, or `--v0-skygrid-infer-prior-smoothness` with its hyperprior | **Changes the model.** `tau` is tied to G2: the same smoothing timescale implies a different `tau` at a different knot spacing |
| **G4 Transient and display grids** | The grid built while the `Run` is constructed, before any MCMC step; and the curve rendered for the API | `Run::Run` -> `repartition` -> `reset_very_scalable_coalescent_parts`; `render_population_curve` (API and WASM only) | was hard-coded to one day; now `--v0-coal-prior-init-cells`; the display grid's cell count comes from the API caller | **Only cost.** The transient grid is discarded at the first retarget; the display grid never enters the posterior |

G1 and G4 are the same data structure at two moments in its life, which is exactly why the
hard-coded one-day initial width was expensive: see section 6.

## 2. Units

Delphy's internal time is days since 2020-01-01 and its internal substitution rate is per site per
day.  The coalescent timescale follows the time axis.

| Quantity | Internal unit | CLI unit | Conversion site |
|---|---|---|---|
| Node and tip times | days | ISO dates, `cal BP`, `ka` | `core/dates.cpp`, `core/human_dates.cpp` |
| `Pop_model::pop_at_time`, i.e. `N(t) rho` | days | years | `core/cmdline.cpp` multiplies every population-size option by 365 |
| `--v0-init-final-pop-size`, `--v0-pop-min-pop` | -- | years | `core/cmdline.cpp` (`* 365.0`) |
| `--v0-skygrid-low-pop-barrier-loc`, `--v0-skygrid-nbar-prior-mean/stddev`, `--v0-skygrid-inv-nbar-prior-beta` | -- | years | `core/cmdline.cpp` (`* 365.0`) |
| `--v0-pop-g-prior-mu`, `--v0-pop-g-prior-scale`, growth rate | e-foldings / day | e-foldings / year | `core/cmdline.cpp` (`/ 365.0`) |
| `--v0-skygrid-cutoff` | days | years | `core/cmdline.cpp` (`* 365.0`) |
| `--v0-skygrid-prior-double-half-time` | days, then absorbed into the dimensionless `tau` | years | `core/cmdline.cpp` (`* 365.0`) |
| `tau` | dimensionless | dimensionless | see section 7 |
| Log columns `ePopSize`, `pop_at_time` | -- | years (`/ 365.0` on output) | `core/beasty_output.cpp` |

`Pop_model::pop_at_time` returns `N(t) rho`, the product of an effective population size and a
generation time, and it carries the dimension of time.  It is **not** a census population size and
not an effective number of individuals.  Turning it into an effective number of individuals requires
a generation time, a ploidy and, for mtDNA or chrY, a sex-specific inheritance factor; none of these
exists anywhere in Delphy, and nothing in this task introduces one.  Log columns that call it an
"effective population size" remain mislabelled; that is risk R-09 and belongs to the reporting work,
not here.

The only place the unit of the coalescent timescale is visible in the density itself is the
`- sum_i log N(t_i)` term, once per coalescence.  Rescaling the time axis and the timescale together
by a factor `s` therefore shifts the log density by exactly `-(n-1) log s` and changes nothing else;
`Coalescent_grid_test.rescaling_time_and_the_coalescent_timescale_together_shifts_the_density_by_a_known_amount`
pins this.

A separate unit audit of every simulation script in Tasks 1.3, 2.1, 2.2 and 2.3 was carried out in
Task 2.3; its conclusion (that `--v0-init-final-pop-size` is in years and enters the actual
coalescent prior, not merely the initialization) is recorded in `clade_summary.md`
section 10 and is unchanged here.

## 3. What the cell-based prior computes, and how it differs from the exact density

The Kingman density of a timed genealogy with a population-size function `N(t)` is

```
log P = - int dt k(t) (k(t) - 1) / (2 N(t))  -  sum_{coalescences i} log N(t_i),
```

where `k(t)` is the number of lineages at time `t`.  `core/coalescent.cpp` (`Coalescent_prior`)
evaluates this exactly, interval by interval between successive events, using the population model's
own `intensity_integral`.  It is used only by tests; the samplers do not call it.

The samplers instead partition the time axis into cells of fixed width `t_step` and, in cell `c`,
replace `k(t)` by its time average `kbar_c` and `1/N(t)` by the reciprocal of the time average of
`N(t)`:

```
log P_cells = - sum_c t_step kbar_c (kbar_c - 1) / (2 Nbar_c)  -  sum_i log N(t_i).
```

The second term is identical in the two expressions, so the whole difference sits in the integral.
Writing `<.>_c` for the time average over cell `c` and `J_c = int_c dt / N(t)` for what
`intensity_integral` returns on the cell, the cell and exact contributions are

```
cells_c = (1/2) t_step (kbar_c^2 - kbar_c) / Nbar_c,
exact_c = (1/2) int_c dt (k^2 - k) / N.
```

Inserting the intermediate quantity `(1/2) J_c <k^2 - k>_c` splits the difference into three pieces:

```
cells_c - exact_c =   (1/2) J_c ( <k^2 - k>_c - (kbar_c^2 - kbar_c) )                     (I)
                    - (1/2) ( int_c dt (k^2-k)/N  -  J_c <k^2 - k>_c )                    (II)
                    - (1/2) (kbar_c^2 - kbar_c) ( J_c - t_step / Nbar_c ).                (III)
```

* **(I) is the variance term.** Because `<k^2> - <k>^2 = Var(k)`, term (I) equals
  `(1/2) J_c Var_c(k)`, and it is always non-negative: averaging `k` before applying the convex map
  `k -> k(k-1)` loses exactly the within-cell variance.
* **(III) is the mean-inequality term.** The arithmetic-harmonic mean inequality gives
  `J_c >= t_step / Nbar_c`, so (III) is non-positive and its subtraction is non-negative.
* **(II) is the correlation term**: it measures how `k(t)` and `1/N(t)` co-vary inside the cell. It
  **has no fixed sign**.

Terms (II) and (III) both vanish when `N(t)` is constant across the cell, and only then:

```
log P_cells - log P_exact = sum_c t_step Var_c(k) / (2 Nbar_c)      [N(t) cell-wise constant]   (*)
```

**(\*) must not be assumed for a general population model.** For a free exponential model, or for a
Skygrid curve whose knots do not coincide with the grid, `N(t)` varies inside cells; the cell prior
can then fall on either side of the exact density, and only a direct comparison settles it.  For
that reason the diagnostic added in this task reports the *measured* difference
`log P_cells - log P_exact`, evaluated with whatever population model is in force, and carries the
variance term alongside it as the leading contribution.  Both sides are exact: the population
model's own `intensity_integral` supplies the exact integral, so no quadrature enters.
`Coalescent_grid_test.the_variance_closed_form_is_exact_for_a_constant_population_only` and
`...the_measured_excess_matches_the_production_cell_prior_for_every_pop_model` check both halves of
this statement against the production `Scalable_coalescent_prior`, at 2, 5, 20, 200 and 1,000 tips
and at 37, 400, 4,000 and 40,000 cells, for a constant model, a free exponential model and both
Skygrid variants.

Two consequences matter.  First, the difference is a functional of the genealogy, not an additive
constant, so it does not cancel in a Metropolis-Hastings ratio: it is a distortion of the sampled
posterior, not an offset of the reported log prior.  Second, `Var_c(k)` is large wherever many
coalescences fall inside one cell.  For samples that share a sampling date that is the most recent
part of the tree, where the coalescence rate is `k(k-1)/2N` with `k` close to the number of tips.
The accuracy of the grid therefore depends on the number of tips and on how concentrated the
sampling dates are, and **not** on the time span alone -- a point the existing
`--v0-target-coal-prior-cells` help text does not make.

### 3.1 Why the existing tests did not catch this

`tests/scalable_coalescent_tests.cpp` compares the cell prior with a hand-written expected value at
`t_step = 1.0` with every node time an integer multiple of 10, so every event lands exactly on a
cell boundary and `Var_c(k)` is identically zero.  Its second test uses a non-aligned
`t_step = 0.17841` but only checks that the incremental update agrees with a full recomputation --
self-consistency, not accuracy.  `tests/very_scalable_coalescent_tests.cpp` likewise checks
partitioned-versus-global consistency.  No test anywhere compared the cell prior with the exact
density for events that fall inside cells, which is the only configuration in which the
approximation is visible.

## 4. The criterion that matters: the error of a proposal's log-prior difference

The absolute error of one tree is the wrong yardstick.  A Metropolis-Hastings ratio consumes the
*difference* of the log prior between the current and proposed genealogy, and most of the
discretization error is common to the two and cancels.  The quantity that distorts the sampler is

```
residual = [log P_cells(proposed) - log P_cells(current)] - [log P_exact(proposed) - log P_exact(current)].
```

The exact delta has a closed form.  Moving one coalescence changes `k(t)` by exactly one over the
interval it crosses, and `C(k+1,2) - C(k,2) = k`, so

```
delta log P = -/+ int_{min}^{max} dt k_lo(t) / N(t)  -  log( N(new_t) / N(old_t) ),
```

where `k_lo(t)` is the lineage count with the coalescence at the later of its two positions and the
sign is minus when lineages are added.  This holds for any move that leaves a genealogy, that is,
one after which every lineage count is still positive.  It is implemented as
`calc_exact_delta_log_prior_after_displace_coalescence` and checked against recomputing the exact
density from scratch, for all four population models, in
`Coalescent_grid_test.the_exact_proposal_delta_agrees_with_recomputing_the_exact_density`.

Measured on Kingman genealogies with a constant timescale of 75,000 years and all tips at one date,
one proposal per internal node of the size a local node displacement makes (uniform over 2% of the
tree span), the residual is one and a half to two orders of magnitude below the absolute error:

| tips | cells | absolute error (nats) | max residual | rms residual | median \|delta\| | max residual / absolute |
|---|---|---|---|---|---|---|
| 100 | 400 | 0.26 | 1.8e-2 | 7.8e-3 | 0.60 | 6.9e-2 |
| 100 | 4,000 | 0.0089 | 4.1e-4 | 1.5e-4 | 0.60 | 4.6e-2 |
| 100 | 40,000 | 0.00058 | 2.7e-5 | 7.0e-6 | 0.60 | 4.4e-2 |
| 1,000 | 400 | 110.1 | 4.5e-1 | 2.5e-1 | 1.47 | 4.1e-3 |
| 1,000 | 4,000 | 1.97 | 1.3e-2 | 5.3e-3 | 1.47 | 6.5e-3 |
| 1,000 | 40,000 | 0.025 | 2.0e-4 | 6.0e-5 | 1.47 | 8.0e-3 |

This changes the answer to "how many cells does human mode need".  Measured by the absolute error,
the cell count needed to stay under 0.1 nats grows roughly as the square of the number of tips --
about 14,000 cells at 1,000 tips and about 100,000 at 3,000.  Measured by the residual of a
proposal's delta, with the threshold set at 1% of a typical proposal's own magnitude, the
requirement is close to **linear** in the number of tips:

| tips | median \|delta\| (nats) | rms residual at 400 cells | at 4,000 | at 40,000 | cells for 1% | cells / tips |
|---|---|---|---|---|---|---|
| 100 | 0.60 | 7.8e-3 | 1.5e-4 | 7.0e-6 | 244 | 2.4 |
| 300 | 0.83 | 1.8e-2 | 2.1e-4 | 5.2e-6 | 394 | 1.3 |
| 1,000 | 1.47 | 2.5e-1 | 5.3e-3 | 6.0e-5 | 2,373 | 2.4 |
| 3,000 | 1.71 | 3.3e-1 | 1.2e-2 | 1.2e-4 | 3,385 | 1.1 |

So Delphy's default of 400 cells is genuinely inadequate at a thousand same-date tips -- an rms
distortion of 0.25 nats per proposal, about 17% of a typical proposal's magnitude, is enough to
change acceptance probabilities materially -- while a few thousand cells is already enough, and the
quadratic growth implied by the absolute-error criterion is an artifact of using the wrong yardstick.
`Coalescent_grid_test.the_residual_error_of_a_proposal_delta_is_far_below_the_absolute_error` pins
the ratio.

**Rule for human mode, with its range of validity**: for `n` up to 3,000 samples sharing a sampling
date, `--v0-target-coal-prior-cells` at least `max(400, 3 n)`, and in any case enough that the
measured absolute excess that Delphy now prints is below about 1 nat.  The rule was fitted to
per-proposal residuals measured up to n = 3,000 and **fails at n = 10,000**, where it gives 30,000
cells and a startup excess of 319.6 nats (report section 10.1).  Beyond 3,000 samples use the
printed excess, or `--v0-coal-prior-grid-tolerance`, and not the formula.  Heterochronous sampling
flattens `k(t)` and makes the requirement milder.

## 5. An exact, event-driven alternative, and why it is not implemented here

The grid exists for one reason: incremental updates.  `add_interval` touches only the cells a node
crosses, so a local move costs `O(span / t_step)` in the worst case but `O(1)` cells in practice, and
the partitioned sampler can decouple its parts by augmenting each cell with a Gaussian variable
(`k_twiddle_bar_p`), which is what makes `Very_scalable_coalescent_prior_part` sum to the global
prior.  Both properties are built on cells.

An exact scheme is nevertheless possible, and the delta formula of section 4 is its basis.  What a
move needs is `int_{min}^{max} dt k_lo(t) / N(t)` -- an integral of the exact step function `k(t)`
weighted by `1/N(t)`, over an interval bounded by two event times.  Keyed on **event times** rather
than on fixed cells, that is a range-add, range-sum problem over `n` ordered keys: a balanced search
tree or a Fenwick tree over the sorted event times, each node carrying the accumulated
`intensity_integral` of its subtree, supports moving one coalescence and reading the integral in
`O(log n)` time with `O(n)` memory, **independent of the time span**.  Its cost would not grow with
tree depth at all, and the discretization error would be exactly zero.

Two things stop it being the right change for this task:

1. It replaces the mathematical core of the partitioned sampler, not just an implementation detail.
   The auxiliary-variable construction in `Very_scalable_coalescent_prior_part::calc_partial_log_prior`
   is defined per cell; an event-keyed structure needs a different decoupling argument, and getting
   that wrong is exactly the class of error this project ranks above everything else.
2. Section 4 shows it is not needed at the sample sizes in scope.  A few thousand cells already
   drives the per-proposal residual to `5e-3` nats at a thousand tips, and the cost of cells is now
   negligible (section 6): 40,000 cells is 1.4 MB of grid.  At 10,000 tips the rule of section 4 does
   not hold (it leaves 319.6 nats of excess at 30,000 cells), the required count was not determined,
   and the claim here is only that grid memory is not what limits it.

The honest conclusion is that the near-quadratic growth of the *absolute* error does not translate
into a scaling problem for the sampler, so the exact scheme is a future option with a clear design,
not a blocker.  `calc_exact_delta_log_prior_after_displace_coalescence` is kept as the reference the
design would be validated against.

## 6. Memory: every path that scaled with the time span

The coalescent grid was the only structure in Delphy whose size depended on the tree's time span,
and it did so through four allocations, all at the same cell count:

| Structure | Bytes per cell | Where |
|---|---|---|
| `Scalable_coalescent_prior::k_bars_`, `popsize_bars_` | 2 x 8 = 16 | `core/scalable_coalescent.h` |
| `Very_scalable_coalescent_prior_part::k_bar_p_`, `k_twiddle_bar_p_`, `k_twiddle_bar_`, `popsize_bar_` | 4 x 8 = 32 | `core/very_scalable_coalescent.h` |
| `Very_scalable_coalescent_prior_part::num_active_parts_` | 4 | same |
| Per-part copies: `k_twiddle_bar`, `popsize_bar`, `num_active_parts` are passed **by value** to every part | 20 per part | `make_very_scalable_coalescent_prior_parts` |

About 52 bytes per cell for a single partition, plus a 20-byte-per-cell copy per partition.  The cell
count was `span / t_step`, and `t_step` started at **one day**, hard-coded in `Run`'s member
initializer list (`core/run.cpp`: `coalescent_prior_{..., 1.0}`).  One cell per day of tree depth is
13 million cells at 35 ka and 129 million at 355 ka.

The width could not be corrected after construction, because `Run`'s constructor itself calls
`repartition()`, which calls `reset_very_scalable_coalescent_parts()` and builds the per-partition
grids at whatever `t_step` the prior currently has.  The fix is therefore a constructor parameter,
`Run(..., double init_coal_prior_t_step = 1.0)`, whose default reproduces the previous behaviour
exactly, with `core/cmdline.cpp` deriving a width from the tree span and `--v0-coal-prior-init-cells`
before the `Run` is created.  In `--v0-human-mode` that option defaults to
`--v0-target-coal-prior-cells`; everywhere else it defaults to 0, which keeps the one-day grid and
leaves every pathogen workflow byte-identical.

The remaining span-dependent path is `Very_scalable_coalescent_prior_part::ensure_space`, which grows
the root partition's grid when a root move proposes an older root.  With `t_step` tied to the span it
is bounded, and the existing warning at more than 10,000 new cells at once is the right tripwire;
nothing here changes it.

## 7. Skygrid at human time scales

### 7.1 The smoothing precision is tied to the knot spacing

With `--v0-skygrid-infer-prior-smoothness false` (Delphy's default), `core/cmdline.cpp` converts a
"double-half time" `T` into the Gill et al. (2012) precision `tau` by treating `log N(t)` as a random
walk with diffusion constant `D` chosen so that the root-mean-square change over `T` is `log 2`:

```
D = log^2(2) / (2 T),      tau = 1 / (2 D dt) = T / (log^2(2) dt),      dt = cutoff / (num_parameters - 1).
```

`tau` is dimensionless and depends only on the **ratio** of the smoothing timescale to the knot
spacing.  That is the whole problem with the default.  `T = 0.0822` years (30 days) was chosen for
pathogens, where `dt` is also of order weeks.  On a human axis with a 400 ka cutoff and 41 knots,
`dt = 10,000` years and

```
tau = 0.0822 / (0.4805 x 10,000) = 1.7e-5,
```

a prior standard deviation of `1/sqrt(tau) = 240` log units between adjacent knots.  That is not
smoothing at all: each `gamma_k` is effectively free, and the population curve will follow whatever
the data and the coalescent term happen to suggest, knot by knot.  A smoothing timescale comparable
to the knot spacing, `T = dt`, gives `tau = 1/log^2(2) = 2.08`, i.e. a prior standard deviation of
0.69 log units -- a factor of two per knot interval, which is what the parametrization was meant to
express.

Because silently inheriting a viral default here is exactly what this project forbids,
`--v0-human-mode` combined with `--v0-pop-model skygrid` now **refuses to run** unless the smoothness
is stated: `--v0-skygrid-prior-double-half-time`, or `--v0-skygrid-tau`, or
`--v0-skygrid-infer-prior-smoothness true`.  The error message gives the knot-spacing rule.

### 7.2 The low-population barrier is inert on a human axis

`--v0-skygrid-low-pop-barrier-loc` defaults to one day, "to match resolution of tip dates".  Every
plausible human coalescent timescale is between `10^3` and `10^5` years, which is six to eight orders
of magnitude above the barrier, so the barrier never engages and cannot catch a collapsing curve.
In `--v0-human-mode` Delphy now warns when the default is left in place and points at the two ways to
state an intention: a floor that is meaningful on the time axis in use, or
`--v0-skygrid-disable-low-pop-barrier true`.

### 7.3 The cutoff has no usable default

`--v0-skygrid-cutoff` defaults to 0.0, which makes `x_0 == x_M` and is rejected with "first knot must
be before last knot".  That is already a hard failure rather than a silent viral default, so it needs
no change; it is noted here because it means a human Skygrid run must always set the knot range
explicitly.  `doc/extended/examples/skygrid/` records a worked configuration.

### 7.4 Deep-time numerics of `Skygrid_pop_model`

`log_int_N_core` biases the integral by the largest `gamma` that enters it and works in `expm1` form
within each interval, so it is well conditioned at human depths.  Its `pop_integral` and
`intensity_integral` are checked against composite Simpson quadrature in long double at 50, 100, 140,
200 and 352 ka for both the staircase and the log-linear variant, and the open-ended first and last
intervals are checked to reproduce a constant population exactly.  One caveat for anyone extending
those tests: a staircase population is discontinuous at a knot, so quadrature is not a valid
reference across one, and the test windows are placed strictly inside a knot interval.

## 8. Changes made in this task

| File | Change | Why |
|---|---|---|
| `core/coalescent_grid.{h,cpp}` | New. The decomposition of section 3 as a measurement; the exact Kingman density and the exact proposal delta of section 4; a grid chooser | The distortion has to be measurable before it can be managed, and the exact reference has to exist before the approximation can be judged |
| `core/run.{h,cpp}` | `Run` takes `init_coal_prior_t_step` (default 1.0, unchanged behaviour); `coal_prior_grid_tolerance`, `coal_prior_grid_max_cells`, `calc_coal_prior_grid_excess`, `retarget_coal_prior_grid`; step 4-pre honours a tolerance when one is set | Section 6 needs the width at construction time; the tolerance rule needs a place in the retarget path |
| `core/cmdline.{h,cpp}` | `--v0-coal-prior-init-cells`, `--v0-coal-prior-grid-tolerance`, `--v0-coal-prior-max-cells`; the grid is reported and warned about in human mode; human-mode guards on the Skygrid smoothness and population floor | Opt-in, with pathogen defaults untouched |
| `tests/coalescent_grid_tests.cpp` | New. Hand calculations, the production-code comparison for four population models, the condition on the closed form, the proposal-delta reference and residual, the chooser, deep-time `Pop_model` quadrature checks | Section 3.1: the gap the existing suite left |
| `tests/human_input/run_coalescent_grid_study.sh`, `run_coalescent_models.sh`, `run_coalescent_scale.sh`, `summarize_coalescent_runs.py` | New campaign scripts | The measurements in `10_coalescent_report.md` |
| `tests/human_input/summarize_clade_validation.py` | `--modes` so arm labels other than `fixed`/`free` can be summarized | Reuse rather than a second coverage implementation |

Default behaviour with no new option set is unchanged, and the pathogen regression confirms it
byte for byte.

## 9. Out of scope, and deliberately not done

* No new MCMC operator, in particular no joint rescaling of times and rate.
* No change to Skygrid's model, its moves, its HMC or its `tau` inference.
* No change to free-topology search.
* No change to the `.dphy` or API float32 time representation (risk R-02).
* No fix for the `int` overflow in `to_linear_year` (risk R-18): it changes existing log output.
* No relabelling of the "effective population size" log columns (risk R-09).
* No event-driven coalescent prior (section 5), which would replace the partitioned sampler's core.
