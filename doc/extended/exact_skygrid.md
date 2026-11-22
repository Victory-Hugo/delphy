# Exact Kingman posterior for the Skygrid model

## Part 0 - Baseline and scope

* Outer repository: branch `audit/clock-correctness-20261110`.  Nested `delphy/`: branch
  `human-uniparental/exact-skygrid-task31` from `ff44b23` (verified present).
* Toolchain (development machine): g++ 13.3.0, cmake 3.28.3, glibc 2.39, 24 hardware threads, 47 GB.
  Campaign host: a shared Linux cluster, Debian 12, glibc 2.36, 96 cores, python 3.11.2, static binary built from this
  branch.
* Starting state of the test suite at `ff44b23`: 724 release unit tests, all passing (document 49).
* What this task does **not** do: no new MCMC operator, no change to the Skygrid prior, no FreeRate, no
  relaxed clock, no new free-topology operator, no rewrite of the inference architecture.  The EMAT, the
  explicit mutation histories, the incremental sequence likelihood, the mutation-guided topology
  proposals and the Run/Subrun framework are unchanged.

### 0.1 The problem this task solves

Document 49 established that Delphy's Skygrid sampler does not target the Kingman posterior.  Its
coalescent term is the *cell prior* on a fixed grid,

    log P_cells = - sum_c t_step kbar_c (kbar_c - 1) / (2 Nbar_c) - sum_i log N(t_i),

in which the lineage count `k(t)` is replaced by its average over each cell.  The difference from the
exact Kingman density is a functional of the genealogy, so it does not cancel in Metropolis-Hastings
ratios; document 49, section 3.4, measured it at 0.26 nats at 100 tips and 92 nats at 1,000 tips on
Delphy's default 400-cell grid.  Delphy already had an exact-Kingman correction, but
`Run::set_exact_coalescent_correction` refused every population model except the exponential one, so a
Skygrid analysis could not use it, and no legitimate SBC of a Skygrid configuration existed.

This task extends that correction to the Skygrid model.

## Part 1 - The mathematical design

### 1.1 The target

The target's coalescent term is the exact Kingman log density of the genealogy,

    log P_exact = - sum_{coalescences i} log N(t_i)
                  - sum_j C(k_j, 2) int_{a_j}^{b_j} dt / N(t),                                   (1)

where the intervals `[a_j, b_j)` are delimited by the genealogy's own events - the sampling times and
the coalescence times - and `k_j` is the number of lineages alive over interval `j`.  Coalescences sort
before tips at the same time, since a lineage must exist before it can be sampled.

Nothing in (1) is specific to a population model: it needs `N(t)` and `int dt / N(t)` exactly, and
`Pop_model::pop_at_time` and `Pop_model::intensity_integral` provide both for every model Delphy has,
including both Skygrid variants, to 1e-8 relative against long-double quadrature at human depths
(document 10, section 2).  The refusal in `Run::set_exact_coalescent_correction` was therefore a
validation guard, not a mathematical obstruction.

### 1.2 What BEAST does, and what of it is reusable

`GMRFSkygridLikelihood` builds its intervals from the tree's own event times (`TreeIntervals`), splits
any interval that crosses a grid point, and accumulates two sufficient statistics per grid cell `c`:

    S_c = sum over the event-interval pieces in cell c of C(k, 2) x (piece duration),
    n_c = number of coalescences whose time falls in cell c,

from which its log likelihood is

    log L = sum_c [ - n_c gamma_c - S_c exp(-gamma_c) ],                                          (2)

with `gamma_c` the log population of cell `c`.  Form (2) is exact because BEAST's Skygrid population is
piecewise constant *per cell* and the intervals are cut at both the real events and the cell boundaries,
so `int dt / N` over each piece is (duration) x exp(-gamma_c).  `VDdemographicFunction` carries the
analytic rate integrals of the piecewise-constant, piecewise-linear and log-linear variants.

Three things are reusable, and all three are reused here:

1. **The decomposition.**  Build the intervals from the real sampling and coalescent events, then
   subdivide at the population curve's knots so that each piece lies inside one interval of the curve.
   This is Delphy's `Exact_coalescent_events` (`core/exact_coalescent.h`).  It is **not** BEAST's
   population grid and it is **not** Delphy's coalescent grid: it is a decomposition of the genealogy,
   with as many pieces as there are nodes plus knots, and no free resolution parameter.
2. **The sufficient statistics.**  For the staircase variant, (1) collapses onto `(SS_k, c_k)` per
   Skygrid interval exactly as in (2), with `k` indexing the knots rather than a grid:

       log P_exact = - sum_k c_k gamma_k - sum_k SS_k exp(-gamma_k),                              (3)
       c_k  = number of coalescences in interval k,
       SS_k = sum over the pieces of interval k of C(k_j, 2) (b_j - a_j).

   Delphy's HMC comment already derived this form (`core/run.cpp`, the `SS_k / c_k` discussion), but
   evaluated it on the grid.  `calc_staircase_sufficient_statistics` returns the statistics and is used
   as an **independent reference** for the general path, not as a second production path.
3. **The analytic rate integral of the log-linear variant.**  Delphy already has it:
   `intensity_integral` is `pop_integral` of the curve with every knot value negated, and
   `log_int_N_core` evaluates it with the `expm1(d)/d` form that stays accurate as the slope goes to
   zero.  Nothing had to be imported.

No Java was copied; the correspondence is established by the formulas above and, numerically, by the
test that (3) and the general path agree.

### 1.3 The exact integrals, per variant

Write a piece as `[p, q]` inside interval `k` of the curve, `c_p = (p - x_{k-1}) / (x_k - x_{k-1})` and
likewise `c_q`.

**Staircase.**  `log N(t) = gamma_k` throughout the interval, so

    int_p^q dt / N(t) = (q - p) exp(-gamma_k).                                                    (4)

**Log-linear.**  `log N(t)` interpolates linearly, so with
`G_p = (1 - c_p) gamma_{k-1} + c_p gamma_k`, `G_q` likewise and `D = G_q - G_p`,

    int_p^q dt / N(t) = (q - p) exp(-G_p) (1 - exp(-D)) / D
                      = (q - p) exp(-G_p) (-expm1(-D)) / D,                                       (5)

which is (4) in the limit `D -> 0` and is what `log_int_N_core` computes with the negated knots.  Beyond
the knot range the curve is constant at `gamma_0` or `gamma_M` and (4) applies.

### 1.4 The gradient with respect to the Skygrid parameters

Differentiating (1),

    d(log P_exact) / d gamma_k
        = - sum_j C(k_j, 2) I_j  d(log I_j) / d gamma_k
          - sum_i  d(log N(t_i)) / d gamma_k,      I_j := int_{a_j}^{b_j} dt / N(t).              (6)

The second sum is `Skygrid_pop_model::d_log_N_d_gamma`, which already existed.  The first needs
`d(log I_j) / d gamma_k`, which is the negative of the existing `d_log_int_N_d_gamma_core` evaluated at
the negated knots, because the intensity integral is the population integral of the negated curve.  The
one new accessor in `Skygrid_pop_model` is therefore

    d_log_intensity_integral_d_gamma(a, b, k) = - d_log_int_N_d_gamma_core(a, b, k, -gamma).      (7)

Two invariants follow and are both checked in the suite, the first in every debug build at run time:

* Summed over the knots that enter a piece, `d(log I_j) / d gamma_k = -1`, because shifting the whole
  curve up by `delta` divides every intensity integral by `exp(delta)`.
* Summed over all knots, (6) equals `A - N_inner`, where `A` is the integral term of (1) and `N_inner`
  the number of coalescences.  This is the derivative along the zero mode and is the identity the
  zero-mode Gibbs move rests on.

For the staircase variant (6) reduces to `- c_k + SS_k exp(-gamma_k)`, which is BEAST's gradient.

### 1.5 Why each MCMC update targets the same posterior

The target is `pi(G, theta) ∝ L(G) P_exact(G | theta) p(theta)`, with `theta` the Skygrid knot values
and the smoothness `tau`.  Each update class is treated separately; "the coalescent term" below always
means (1).

**(a) Local moves on the node times, and the whole-batch correction.**  This is Delphy's existing
mechanism and is reused unchanged.  A batch of local moves runs in the partitioned sampler against the
*cell* prior, so the batch kernel `Q` is reversible with respect to
`rho(G) ∝ L(G) P_cells(G)` at fixed `theta` and fixed auxiliary variables; the whole batch is then
accepted with probability `min(1, exp(delta' - delta))` where
`delta = log P_exact - log P_cells`.  Since the ratio is
`[pi(G') / rho(G')] / [pi(G) / rho(G)]`, the composed kernel is reversible with respect to `pi`.  This is
the surrogate-transition construction of Liu (2001, section 9.4) and of Christen and Fox (2005), and
nothing in it refers to the population model, so it carries over from the exponential case (documents
24 and 26) verbatim.  Two consequences worth stating: the grid is now a *tuning parameter* of the
surrogate rather than part of the target, and a coarser grid costs acceptance rate, not correctness.

**(b) The smoothness `tau`.**  `tau` does not appear in (1) at all.  Its full conditional is therefore
`Gamma(alpha + M/2, rate beta + SS/2)` with `SS = sum_k (gamma_k - gamma_{k-1})^2` under the exact
target exactly as under the cell prior, and `Run::skygrid_tau_move` needs no change.  Both halves of
that claim are tested against the posterior the run evaluates: the coalescent term does not move with
`tau` at all, and the whole log posterior's dependence on `tau` is the closed form above.

**(c) The zero mode.**  Decompose `gamma_k = gamma_bar + gamma'_k` with `sum_k gamma'_k = 0` and put
`I_bar = exp(-gamma_bar)`, `Ntwiddle(t) = N(t) I_bar`.  Under (1),

    - log P_exact = I_bar  sum_j C(k_j, 2) int_j dt / Ntwiddle(t)
                    - N_inner log I_bar + sum_i log Ntwiddle(t_i),                                (8)

which is the same `I_bar^{N_inner} exp(-B I_bar)` shape the existing derivation uses, with

    B = A / I_bar,   A = sum_j C(k_j, 2) int_j dt / N(t)                                          (9)

read off the *exact* density instead of the grid sum.  With the Jacobian `dgamma_bar = -dI_bar / I_bar`
and the InverseGamma prior on `N_bar`, `I_bar ~ Gamma(N_inner + alpha, rate B + beta)` is an exact Gibbs
step of the exact posterior's conditional.  The implementation change is one line: `B` comes from
`Exact_coalescent_events::calc_intensity_term` when the correction is on.  The low-population barrier
keeps its Metropolis correction, as before.

**(d) The HMC move on the knot values.**  The potential becomes `U = U_prior - log P_exact` and the
forces `f_k = -dU/dgamma_k` use (6).  The integrator, the randomized trajectory length, the diagonal
mass matrix `m_k = tau [[k>0]] + tau [[k<M]] + c_k` and the acceptance rule are unchanged; the mass
matrix is a fixed, position-independent tuning choice (the genealogy, hence `c_k`, does not change
inside the move), so the move remains a valid HMC proposal for the new potential.  Because the forces no
longer read the grid, the trajectory does not touch the coalescent grid at all; the grid's cell
populations are refreshed once, at the end, for the benefit of the local moves.  The guards are
unchanged in character and still symmetric between the forward and the reverse trajectory: the kinetic
energy bound, the representability bound on the knot values, and - new in the exact path - a finiteness
check on the exact potential.

**(e) Run/Subrun.**  The correction acts at the `Run` level after `reassemble()`, so the auxiliary
lineage counts of `Very_scalable_coalescent_prior_part` live entirely inside the batch kernel of (a).
The exact density is a function of the assembled genealogy and of nothing else, so the number of
partitions cannot change the target; what it can change is the batch kernel, and that is checked
empirically at 1, 2 and 4 partitions.

**(f) The joint time-scale move** (Task S/T) is still refused for a Skygrid curve.  Its map transforms
`(n0, g)` of an exponential curve, and no corresponding map of a Skygrid curve's knot values has been
derived.  This is unchanged behaviour and is tested.

### 1.6 Where the fixed grid still enters, and where it no longer does

With the correction on, the sampled target is (1): a function of the genealogy and the population curve
alone.  It does not depend on `t_step`, which is now only the surrogate's resolution.  This is checked
directly - the target is identical to 1e-8 relative across cell widths spanning a factor of 125 - and it
is the property the campaigns of Part 2 are built to confirm statistically.  The grid is still required
to be pinned, because a grid retargeted from the sampled state would change the surrogate mid-run; that
restriction is kept rather than relaxed.

### 1.7 Sampling dates, fixed topologies, and what conditioning on a topology validates

The exact density (1) treats a sampling time as a lineage-count change, so it is correct for serially
sampled data, and the support of the correction was widened accordingly: it now requires every tip to
carry an *exact* date, but no longer requires the dates to be equal.  Date *uncertainty* is still
refused, because an uncertain date makes the tip time a sampled variable with its own proposal and
prior, which the whole-batch correction has not been validated against.

What a fixed topology means for a validation needs stating carefully, because two different models
both go by "the topology is fixed" and they are validated by different experiments.  The starting
point is what (1) actually is.

**(1) is the density of the labelled ranked genealogy, topology and times together.**  Write `R` for
that topology, to keep it apart from the GMRF precision `tau` of the rest of this report.  At each
coalescence the *specific* pair that merges does so at rate `1/N(t)`, and `C(k, 2)` pairs are at risk,
which is where the `C(k_j, 2)` of the exponent comes from and why the prefactor is `1/N(t_i)` rather
than `C(k_i, 2)/N(t_i)`.  So

    f(t; theta) := prod_i 1/N(t_i) exp(- sum_j C(k_j,2) I_j)

is `p(R, t | theta)` for the labelled ranked genealogy -- `R` the topology with its node ordering and
`t` the times -- with no omitted combinatorial factor, and this holds for serial sampling exactly as for one date -- the sampling times are fixed
data that only change `k(t)`.  Integrating over the times compatible with one labelled topology gives

    int f(t; theta) dt = P(R | theta).                                                            (10)

**Model A -- the topology as an observed condition.**  Draw `theta` from its prior, draw the genealogy
`(R, t)` from the coalescent under that `theta`, simulate sequences along `t`, then hand the analysis
the realized `R` and the sequences.  The correct posterior is

    p(theta | R, seq) ∝ p(theta) int L(seq | t) f(t; theta) dt,

which is **exactly** the marginal of what Delphy samples with the topology fixed.  So an SBC built this
way is calibrated, **for serially sampled data as much as for one date**, and `simulate_skygrid_joint.py`
is built this way: the topology it gives the analysis is a draw from `p(R | theta)` under the same
`theta`, not a design choice.

**Model B -- the topology as a design condition.**  Fix `R` in advance, draw `theta` from its prior,
and draw the times from the *conditional* `p(t | R, theta) = f(t; theta) / P(R | theta)`.  The
correct posterior then carries an extra factor `1 / P(R | theta)`, which Delphy does not evaluate.
By (10) that factor is a constant in `theta` when the samples share one date -- the isochronous Kingman
process draws its ranked topology uniformly and independently of the demography -- so Models A and B
coincide there.  With serial sampling they do not, and a Delphy analysis of Model B data is
miscalibrated by exactly that factor.

**Prior recovery is Model B with the likelihood removed, which is why it is the restrictive one.**  With
`L` flat the marginal of `theta` at a fixed topology is `p(theta) P(R | theta) / Z` by (10).  That is
the prior only when `P(R | theta)` is free of `theta`, that is only for one sampling date.  Measured,
at 12 serially sampled tips: every first moment of the knot differences inside the gate, every second
moment of the two most recent differences at `z = -9` to `-13`, for both variants -- which is (10)
tilting the marginal, not a defect of the sampler.

So the correct statements, replacing the blanket one an earlier draft of this report made:

* **Prior recovery with a fixed topology is valid only for isochronous samples.**  Unchanged.
* **SBC with the topology drawn under the same parameters and then treated as observed (Model A) is
  calibrated for serial sampling too.**  This report's SBC is Model A, and the restriction to one
  sampling date in campaign S is a property of the generator, which emits one date, and not a
  limitation of the method.  Extending it is the natural next experiment and needs serial sampling in
  `simulate_skygrid_joint.py`, which is a change to the generator rather than to Delphy.
* **An analysis that fixes the topology by design on serially sampled data (Model B) is miscalibrated**
  unless `1 / P(R | theta)` is supplied, which Delphy does not compute.  Anyone fixing a topology
  from an external source on heterochronous data is in Model B, so this is the practically important
  warning.

Two valid statements take the place of a heterochronous prior-recovery test, and both are tested:
prior recovery with **two** serially sampled tips, where there is only one labelled topology so that
`P(R | theta) = 1` and the argument goes through whatever the dates; and a direct comparison of the
sampled **node-time marginals** against numerical quadrature of (1) over the fixed topology's region,
under a constant population where there are no population moves, for an isochronous and two serially
sampled patterns.

### 1.8 The support restriction, stated rather than hidden

The task asks that any new constraint on the statistical support be written out with its definition,
its normalization and its scientific effect, and that nothing be truncated quietly.  There is exactly
one, and it is not new in kind: it is document 49's representability bound, extended to one move that
did not have it.

**Definition.**  A proposal is abandoned, leaving the state unchanged, if any knot value fails
`|gamma_k| <= 700` or is not finite, or -- in the exact path -- if the resulting potential is not
finite.  The bound is applied by `skygrid_gammas_are_representable` in `core/run.cpp` and is used by
the HMC move (at both position half-steps, as before) and now also by the zero-mode Gibbs move, which
had an `isnan` check but no bound and could in principle draw an `I_bar` small enough to send every
knot past the overflow of `exp`.

**Why 700.**  `exp(gamma)` overflows above about 710 and underflows to zero below about -745.  A cell
or an event interval whose population is infinite or zero breaks both the cell prior and the exact
density; the auxiliary lineage counts of the partitioned sampler draw with a standard deviation of
`sqrt(N_c / (parts t_step))`, which is not finite for an infinite population (document 49, 2.2).  The
bound leaves a decade of margin, so an interval of any width down to 1e-6 time units keeps a strictly
positive, finite integral.

**What it does to the target, and the normalization.**  Let `S` be the set of representable curves.
Every move that can leave `S` abandons such a proposal and keeps the current state, and the current
state is in `S` because nothing outside `S` is ever accepted.  For two states in `S` the transition
densities are unchanged, so detailed balance holds on `S x S` and no mass leaves `S`: the chain is
reversible with respect to `pi` restricted to `S`, that is with respect to `pi(x) 1[x in S] / pi(S)`.
The sampled target is therefore the posterior **conditioned on the population curve being
representable**, with the normalization `pi(S)`, and not the unconditioned posterior.

**Scientific effect.**  `pi(S)` differs from 1 only through curves with a knot population outside
`exp(+-700)`, which for any analysis with a proper prior on `N_bar` and a GMRF precision of order one
is a region of utterly negligible prior mass; it is reachable at all only when `tau` is inferred and
drawn very small, which is how document 49 found the abort this bound replaced.  It truncates the
GMRF's heavy tail, which must bias the dispersion of the knot differences **downward**; the campaigns
measure that dispersion directly (3.4a, 3.4b) and the observed deficits are in the same direction,
which is why the start effect and this bound are both named there and neither is claimed to be the
whole story.  The principled alternative -- a high-population term in the prior, symmetric to the
existing low-population barrier -- is a change to the model and is **not** made here.

*Re-examined in Part 7:* replicate chains from an exact start show no measurable downward bias in the
dispersion of the knot differences with an inferred smoothness (0.5% over 64 chains of 200,000 global
moves), so this truncation is not what the deficits below measured.

## Part 2 - Frozen protocol

Written before any campaign below was launched, and not changed afterwards.  Four kinds of check, in
this order; a later kind is never used to excuse an earlier one.

1. **Analytic identity.**  A quantity the code computes against the same quantity written out
   independently from the model definition.  Gate: 1e-9 relative, or exact equality where the identity
   is exact.
2. **Independent reference and quadrature.**  The exact density against `calc_exact_coalescent_log_prior`
   (a separate implementation), against the staircase sufficient statistics (3), and against midpoint or
   Simpson quadrature of `1/N(t)` in long double.  Gate: 1e-10 relative against the reference, 1e-6
   against quadrature where the quadrature converges.
3. **Finite difference.**  Gradients and HMC forces against a central difference of the function the code
   itself evaluates.  Gate: 1e-6 relative, or three times the finite difference's own round-off floor
   `eps max(|U|) / h`, whichever is larger.  The floor matters: at deep-time spans `|log P|` reaches
   1e8, where a central difference at `h = 1e-5` cannot resolve a derivative better than 1e-4 absolute,
   and a fixed relative gate would measure the floor rather than the gradient.
4. **Target distribution.**  With no information in the sequences, the topology fixed, a proper prior on
   the zero mode and the barrier off, the marginal of the Skygrid parameters under the full posterior is
   **exactly** their prior, and that prior factorizes as `I_bar = exp(-gamma_bar) ~ Gamma(alpha, rate
   beta)` and `delta_k = gamma_k - gamma_{k-1} ~ iid N(0, 1/tau)`, independent.  Gate: `|z| < 4` on every
   one of `E[I_bar]`, `E[log I_bar]`, `E[delta_k]`, `E[delta_k^2]` for k = 1..M and `E[tau]` where it is
   inferred.

### 2.1 Campaign R49 - the campaign document 49 handed over

Document 49, Part 6, froze a prior-recovery campaign and recorded that it had been launched on the
campaign host under the campaign directory.  **It had not been: that directory did not exist, no
script had been uploaded and no chain had ever run.**  This task runs it once, under document 49's frozen
protocol verbatim, and records the outcome here.  Document 49 is not edited.

The protocol, quoted from document 49: one data set per tip count from
`make_skygrid_recovery_task31.py --samples N --seed 71000 + N`; at 12 tips
{staircase, log-linear} x {`tau` fixed at 1, `tau ~ Gamma(3, rate 3)`} x {400, 4,000, 40,000 cells over
the 2,000-year knot range} = 12 arms; at 100 tips the same four model arms at 4,000 cells = 4 arms;
6 knots over 2,000 years, InverseGamma(4, 1,200 years) on `N_bar`, barrier off, topology fixed, mutation
rate fixed at 1e-14, grid pinned, one thread, 8 chains per arm, 4e7 steps, chain seeds
`8,000,000 + 100 N + c`; scored once with 50% burn-in, the standard error from the spread of the 8
per-chain means, gate `|z| < 4` on every statistic.  No chain is dropped, re-run or extended and the gate
is not changed after the numbers are seen.

Because no historical chain exists, nothing is overwritten or filtered.  The binary is the static build
of this branch; the byte-level pathogen regression (Part 5) establishes that it reproduces `ff44b23`
exactly in the configurations that regression covers, including three pathogen-scale Skygrid arms.

### 2.2 Campaign E - the exact target

Same data sets, same chain seeds, same scorer, same gate.  Arms carry
`--v0-exact-coalescent-correction`:

* 12 tips: {staircase, log-linear} x {`tau` fixed at 1, `tau ~ Gamma(3, rate 3)`} x
  {400, 4,000, 40,000 cells} = 12 arms.
* 100 tips: the same four model arms at {400, 4,000 cells} = 8 arms, plus the four model arms at 400
  cells **without** the correction, as the matched cell-prior comparison that document 49's campaign
  does not contain at that grid.

**Note on the version history of this campaign, recorded because the design changed after the first
launch.**  Campaign S was first launched with the demographic prior of campaign E -- `N_bar` with prior
mean 400 years over a 2,000-year knot range -- which puts the genealogies about 1,000 years deep.  With
16,569 sites and a clock rate of 2e-8 per site per year that leaves about one mutation in the whole
alignment, so the data carry no information about the clock and an SBC of a free clock on them only
re-tests the clock's prior.  That run was **stopped and discarded before any score was computed**, and
campaign S below uses a deep-time demography instead: 6 knots over 400,000 years and InverseGamma(4,
225,000 years) on `N_bar`, prior mean 75,000 years, which is the scale the rest of this project works
at and which produces 30 to 80 variable sites per replicate.  The discarded run's chains are kept
under `sbc_shallow_discarded/` and are not scored; the change was made on the generating model, not on
a result.

**What it decides.**  Whether the marginals recover the prior at every grid resolution, which under the
exact target they must and under the cell prior they must not once the grid is coarse enough for the
measured excess to exceed the test's resolution.  A coarse cell-prior arm that fails is the expected
outcome and is a measurement.

### 2.2a Two defects in the campaign tooling, found by running it

Both are recorded here because both could have produced a wrong number silently.

1. **One chain directory for every tip count** (inherited from document 49's own
   `run_skygrid_recovery_campaign_task31.sh`, and copied into this task's campaign and SBC drivers).
   The arm tags are identical at every size, so the 100-tip stage found the 12-tip stage's chains,
   treated them as already done, skipped them, and scored *them* as the 100-tip result.  Caught by the
   chain count, not by anything in the output: the score file looked perfectly normal.  Fixed in all
   three drivers -- the chain and data directories now carry the tip count -- and the 100-tip stage of
   campaign E was discarded and rerun.  Re-scoring the 12-tip chains from their new directory
   reproduces the 12-tip table exactly, so nothing of 3.4a depends on the fix.
2. **`sigma` where Delphy wants `stddev`.**  `simulate_skygrid_joint.py` wrote a normal clade
   calibration with the lognormal's key, and Delphy rejected the file -- correctly, and with a message
   naming all four stanzas.  So the calibrated SBC arm produced no chain at all rather than a wrong
   one.  Fixed, and the 300 already-generated calibration files were rewritten from the measurements
   recorded in `truth.json`, so no draw was repeated.

### 2.3 Campaign B - performance, three implementations

A. the original fixed-grid approximate Skygrid (400 cells);
B. the same with the grid refined (4,000 and 40,000 cells);
C. the event-driven exact Skygrid, with the grid left at 400 cells as a surrogate.

Reported per arm, on the same data, the same population model and the same host: the measured exact
coalescent density error, wall and user seconds per 1e6 steps, peak RSS, bulk and tail ESS and ESS per
CPU hour of the Skygrid parameters and of the log posterior, at 100 and 1,000 tips, over knot counts and
over 1, 2 and 4 tree partitions.  No claim that the exact method is faster is assumed in advance.

### 2.4 Campaign S - simulation-based calibration

Run only after kinds 1-4 and campaign E pass.  Data from `simulate_skygrid_joint.py`, whose generating
model is the analysis model: `pi ~ Dirichlet(1,1,1,1)`, `log kappa ~ N(1, 1.25^2)`, `mu ~ Gamma(2, rate
1e8)`, `I_bar ~ Gamma(4, rate 1,200 years)`, `delta_k ~ iid N(0, 1/tau)`, `tau` fixed or
`~ Gamma(3, rate 3)`, then a Kingman genealogy under that curve and HKY sequences along it.  The
generator's own self-test checks the rescaling against quadrature, the inverse against the forward map,
the prior's moments and the simulated coalescent intervals against the Kingman expectations.

Settings: 6 knots over 400,000 years, InverseGamma(4, 225,000 years) on `N_bar`, 16,569 sites, 4
clade-age calibrations where the arm uses them, the grid pinned at 4,000 cells over the knot range,
100 data sets per arm, 4 chains of 4e7 steps each, 50% burn-in.  Arms: staircase with `tau` fixed, the
same with calibrations, staircase with `tau` inferred, log-linear with `tau` fixed, at 20 and at 100
tips.  The generating curve type always matches the arm's, so no arm is a misspecified run.

One sampling date, because that is what the generator emits; 1.7 shows this is not a restriction of
the method.  Reported per quantity: SBC rank histogram and its
uniformity, 50%, 80% and 95% coverage with both tail probabilities, rank-split R-hat, bulk and tail ESS.
Quantities: `N_bar`, `tau`, the knot differences, `N(t)` at fixed ages, `mu`, `kappa`, the root age, the
tMRCA of uncalibrated clades, and the calibrated clade ages where calibrations are used.  Gate, fixed
here: R-hat <= 1.05 and both ESS >= 200 in at least 90% of replicates, and the rank histogram's
chi-squared p-value > 0.01 with coverage within the binomial 95% interval of its nominal level.

## Part 3 - Results

Sections 3.1 to 3.3 and 3.6 are unit tests and CLI checks on the development machine; 3.4 to 3.5 and
3.7 are the campaigns of Part 2 on the campaign cluster.  Where a campaign is still running, the section says so and
reports only what has been scored.

### 3.1 The exact density and its gradient

`tests/exact_skygrid_tests.cpp`, 9 tests, all passing in the release build.  Over 8 genealogies
(isochronous and heterochronous, 4, 8, 12 and 20 tips) x 2 variants x 7 curve shapes (flat, monotone,
oscillating, steep, two knots only, non-uniform knot spacing, populations at `exp(+-12)`):

| check | result |
|---|---|
| event-driven density vs `calc_exact_coalescent_log_prior`, with and without the knot subdivision, 112 configurations | agrees to 1e-10 relative |
| log-linear density vs fully independent long-double quadrature (events merged and lineage counts tracked in the test, no knowledge of the knots) | agrees to 1e-6 relative |
| staircase and log-linear density vs quadrature over the knot-cut pieces | agrees to 1e-7 relative |
| staircase density vs the sufficient-statistic form (3), and the statistics independent of the knot values | agrees to 1e-9 relative; every coalescence counted once |
| gradient (6) vs central finite difference, 512 checks over every knot | worst relative discrepancy 1.6e-6, inside the gate of Part 2 kind 3 |
| gradient sums to `A - N_inner` along the zero mode | agrees to 1e-8 relative |
| the coalescent term's dependence on the zero mode is exactly (8), predicted at four shifts | agrees to 1e-9 relative |
| coalescences exactly on knots, one ULP either side, and several inside one interval | density matches the reference to 1e-10; pieces tile the span; gradient matches a finite difference |
| exact single-coalescence increment vs the difference of two full densities, 3 curves x 2 variants x 7 nodes x 7 destinations | agrees to 1e-8 relative |

### 3.2 The sampler

`tests/exact_skygrid_mcmc_tests.cpp`, 15 tests, all passing in the release build.

| check | configurations | result |
|---|---|---|
| the correction accepts a Skygrid model | 2 variants x 6 curves | accepted |
| it still refuses an unpinned grid, a free topology, and releasing the grid afterwards | | refused |
| the target's coalescent term is the exact density, initially and after 2,000 steps | 2 variants x 6 curves, 20 tips | agrees with the reference to 1e-6 relative |
| the exact target does not depend on the grid | 2 variants x 6 curves x cell widths 500, 100, 20, 4 days | identical to 1e-8 relative, while the measured grid excess changes |
| the HMC forces are the gradient of the **exact** potential | 2 variants x 6 curves, 200 MCMC steps each, 29,708 finite-difference checks | worst discrepancy 0.084 of the comparison's own resolution |
| the zero-mode conditional is the exact density's, predicted at four shifts | 2 variants x 6 curves | agrees to 1e-7 relative |
| the full conditional of `tau` is unchanged by the exact target, and the coalescent term does not move with `tau` | 2 variants x 6 curves x 5 values of `tau` | exact equality of the coalescent term; conditional agrees to 1e-8 |
| incremental vs recomputed log posterior after 2,000 steps, and the coalescent term still exact | 1, 2 and 4 tree partitions x 2 variants x 6 curves, 40 tips | agrees to 1e-6 relative |
| a rejected batch leaves a consistent state: after each of 60 batches, with the surrogate at 8 cells so that batches really are rejected, a Run rebuilt from the current genealogy agrees | 1 and 2 partitions x 2 variants, 30 tips | agrees to 1e-6 relative on `log_G` and on the coalescent target; 3 to 5 of 60 batches rolled back in each arm |
| age calibrations on the root and on a clade, with the correction and a Skygrid curve | 1, 2 and 4 partitions x 2 variants x 2 curves, 20 tips, 2,000 steps | incremental equals recomputed to 1e-6; the coalescent term is still exact; a run rebuilt from the sampled genealogy agrees on the calibration prior to 1e-8 |
| a resumed run is on the same target and stays consistent | 2 variants | log posterior and coalescent target agree to 1e-9 relative; still consistent after a further 1,000 steps |

### 3.3 The target distribution of the whole moveset (in-suite)

12 tips, one invariant site, `mu` fixed at 1e-14, topology fixed, 6 knots over 2e6 days, barrier off,
InverseGamma(4, 1.2e6 days) on `N_bar`, 20,000 global moves of 60 local moves, 5% burn-in, batch means
over 20 batches.  The grid is deliberately **coarse** (5,000-day cells), which is the point: under the
cell prior the same grid carries a measurable bias, and under the exact target it must not.

| arm | worst \|z\| over the 12 statistics | measured grid excess | batch acceptance | verdict |
|---|---|---|---|---|
| staircase, coarse grid | 1.78 | 0.042 nats | 0.994 | pass |
| log-linear, coarse grid | 1.91 | 0.166 nats | 0.995 | pass |
| staircase, 2 tree partitions | 2.37 | 0.028 nats | 0.991 | pass |
| staircase, 4 tree partitions | 1.44 | 0.009 nats | 0.990 | pass |
| 2 serially sampled tips, staircase | 2.65 | 0.001 nats | 0.999 | pass |
| 2 serially sampled tips, log-linear | 2.14 | 0.001 nats | 0.999 | pass |

**The dispersion deficit and the start, tested rather than asserted.**  Both arms that miss the gate in
the campaigns (3.4a, 3.4b) are inferred-smoothness arms missing on the second moments of the knot
differences, and the explanation offered there is the undispersed start: the command line cannot
disperse the initial Skygrid curve, so every chain begins at a flat one, that is at `delta_k = 0`
exactly, and a flat curve makes the smoothness Gibbs move draw a large precision which then holds it
flat.  At the library level a dispersed start costs two lines, so the explanation is testable.  Running
the inferred-smoothness arm both ways at the same seed and the same length:

| start | mean z of the five second moments | worst |
|---|---|---|
| flat, as the command line gives | -0.78 | -2.35 |
| drawn from the prior | -0.27 | -0.68 |

The deficit shrinks by about a factor of three in both the mean and the worst case.  That is one
realization at one length and is **supporting evidence, not proof**; what it does establish is that the
deficit moves with the start and in the predicted direction, and that a dispersed start is a two-line
change in the library whose absence from the command line is now a measured cost rather than a
suspicion.  The flat-start arm is deliberately left ungated in the test: it is the configuration the
campaigns found wanting, and asserting that it passes in the suite would be asserting away the
campaigns' own result.

*Superseded by Part 7:* the "deficit" in this table is a property of the statistic, not of the sampler.
A batch-means z of a heavy-tailed time average is biased towards negative values even from a
stationary start, and the test built on it has been replaced by a replicate design.

Node-time marginals against quadrature of (1), 3 tips, constant population, no population moves:

| sampling pattern | E[t_root] by quadrature | z | E[t_inner] by quadrature | z |
|---|---|---|---|---|
| isochronous | -533.20 | +0.47 | -133.30 | -0.20 |
| one late tip | -730.67 | +0.06 | -200.82 | -0.73 |
| three dates | -833.02 | -0.19 | -315.45 | -1.18 |

The isochronous quadrature reproduces the closed-form Kingman answers (-533.33 and -133.33 for N = 400),
which is the check on the quadrature itself.

### 3.4 One matched pair, and what each target costs

The first campaign result, run before the full campaigns as the discriminating pair: document 49's
frozen schedule (12 tips, 8 chains, 4e7 steps, 50% burn-in, across-chain standard error, gate
`|z| < 4`), on a 400-cell grid, staircase, `tau` fixed, with and without the correction on the same
data set and the same seeds.  The full campaigns are 3.4a and 3.4b; what this pair adds to them is the
cost measurement at the bottom.

| arm | worst \|z\| over the 12 statistics | verdict |
|---|---|---|
| exact correction on | **1.90** | pass |
| cell prior, same grid, same seeds | **9.91** | fail |

The failing arm's pattern is the signature of the discretization bias and not of noise: every
`E[delta_k]` at `z = -9.0` to `-9.4`, every `E[delta_k^2]` at `+6.7` to `+9.9` and `E[I_bar]` at `+9.3`,
with the same sign at every knot.  This is the quantity document 49 could only measure in nats; here it
is measured in the sampled marginals, and the exact target removes it.

Cost of the same two arms, 4e7 steps, one thread, measured by `/usr/bin/time`:

| arm | wall seconds (8 chains) | peak RSS |
|---|---|---|
| exact correction on | 23.5, 24.5, 25.4, 26.5, 27.4, 30.5, 32.1, 33.8 | 5.5-6.8 MB |
| cell prior, same grid | 170.0, 190.0, 206.3, 239.1, 262.5, 274.2, 338.2, 718.2 | 5.4-6.7 MB |

The exact arm is **8 to 25 times faster** at the same grid, because the HMC trajectory no longer
evaluates the cell prior: the forces come from the event decomposition, which costs O(nodes + knots) per
evaluation, and the grid's cell populations are refreshed once per move instead of 50 times.  The spread
within the cell arm tracks the sampled tree depth, which sets the number of cells.

### 3.4a Campaign E at 12 tips: the exact target at every grid resolution

All twelve arms of campaign E at 12 tips, scored once by the frozen rule of Part 2 (8 chains of 4e7
steps, 50% burn-in, across-chain standard error, gate `|z| < 4` on each of `E[I_bar]`,
`E[log I_bar]`, `E[delta_k]` and `E[delta_k^2]` for k = 1..5, and `E[tau]` where inferred).

| arm | worst \|z\| | verdict | mean batch acceptance | mean CPU s |
|---|---|---|---|---|
| staircase, `tau` = 1, 400 cells | 1.90 | pass | 98.9% | 28 |
| staircase, `tau` = 1, 4,000 cells | 3.79 | pass | | |
| staircase, `tau` = 1, 40,000 cells | 1.60 | pass | | |
| staircase, `tau` inferred, 400 cells | 2.17 | pass | 98.1% | 38 |
| staircase, `tau` inferred, 4,000 cells | 1.73 | pass | | |
| staircase, `tau` inferred, 40,000 cells | 1.48 | pass | | |
| log-linear, `tau` = 1, 400 cells | 1.80 | pass | 98.9% | 42 |
| log-linear, `tau` = 1, 4,000 cells | 1.34 | pass | | |
| log-linear, `tau` = 1, 40,000 cells | 1.86 | pass | | |
| **log-linear, `tau` inferred, 400 cells** | **4.62** | **does not meet the gate** | 98.0% | 37 |
| log-linear, `tau` inferred, 4,000 cells | 2.22 | pass | 99.8% | 439 |
| log-linear, `tau` inferred, 40,000 cells | 2.71 | pass | 100.0% | 2,971 |

**Eleven of twelve arms pass.  One does not, and the gate is not relaxed.**  What the failing arm
shows, in full: every one of its five `E[delta_k^2]` is negative, at -2.49, -4.62, -4.01, -2.79 and
-0.92, while every first moment, `E[I_bar]`, `E[log I_bar]` and `E[tau]` is inside 1.3.  So the
dispersion of the knot differences is too small, and nothing else is out.

Three facts bear on how to read that, and none of them is used to call the arm a pass.

1. **The deviation is not systematic across the campaign.**  Over all 60 second-moment z-scores the
   mean is -0.24 and 34 of 60 are negative, which is what noise looks like.  Split by smoothness, the
   six inferred-`tau` arms average -0.58 and the six fixed-`tau` arms +0.10, so there is a mild
   downward tendency when `tau` is inferred, and one arm of those six is far enough out to miss.
2. **The sign is the one the known shared-start effect predicts.**  Delphy has no option to disperse
   the initial Skygrid curve (document 49, 3.5), so every chain starts at a perfectly flat curve, that
   is at `delta_k = 0` exactly.  That biases a second moment *downward*, and the bias is slowest to
   shed when `tau` is itself being learned, because a flat curve makes the `tau` Gibbs move draw a
   large precision, which then holds the curve flat.  The log-linear variant couples neighbouring
   knots more tightly than the staircase does, which is consistent with it being the variant affected.
3. **Multiplicity.**  The gate is per statistic, with no correction, and there are 144 statistics.
   With 8 chains the statistic is a t with 7 degrees of freedom, so `P(|t| > 4) = 0.0051`; taking the
   five second moments of one arm as roughly two independent quantities gives about six per arm and
   72 in all, and `P(any exceedance) ≈ 31%`.  One failing arm out of twelve is therefore not
   surprising under a correct sampler.

Against a defect in the target: the same model arm passes at 4,000 and at 40,000 cells, and the exact
target is provably the same function at all three (3.2), so a grid-dependent *target* is ruled out by
construction; what does depend on the grid is how well the batch kernel mixes, and the batch
acceptance of the failing arm is 98.0%, no worse than the staircase arm at the same grid that passes.
The verdict recorded here is therefore: **the inferred-smoothness log-linear configuration at the
coarsest grid is not accepted at 12 tips**, the evidence points at the start effect rather than at the
target, and settling it needs a dispersed start, which is new functionality.

### 3.4c Campaign R49 at 12 tips: the cell prior at every grid resolution

The twelve arms of document 49's protocol (Part 2.1), the cell prior without the exact correction.

**What was run, and where it departs from the frozen protocol.**  The campaign was stopped by the
operator before it finished.

* 93 of the 96 chains at 12 tips completed 4e7 steps.  Three chains of the two log-linear 40,000-cell
  arms were stopped at 2.57e7, 2.36e7 and 2.80e7 steps (`LL+TAUFIX+C40000` chain 6, `LL+TAUINF+C40000`
  chains 4 and 7).  `LL+TAUFIX+C40000` therefore has 7 complete chains and `LL+TAUINF+C40000` has 6.
* The 100-tip stage of the protocol (four arms at 4,000 cells) was started by the campaign driver a few
  seconds before it was stopped.  It produced no usable chain and **is not scored**; no 100-tip cell-prior
  result of R49 exists.  The matched 100-tip cell-prior comparison in 3.4b is the one that was run
  inside campaign E, at 400 cells.
* **Scoring rule.**  The scorer, the 50% burn-in, the across-chain standard error and the gate
  `|z| < 4` are those of Part 2.1.  The primary score uses the complete chains only; it was chosen
  before the numbers were seen.  A second score that adds the three stopped chains (each burned in at
  50% of its own length) is given as a sensitivity check.  The two give the same verdict in every arm.
  Fewer than 8 chains gives a t statistic with fewer degrees of freedom than the protocol's 7, so the
  two incomplete arms are scored with a wider tail than the other ten, not a narrower one.

| arm | chains | worst \|z\| | verdict | worst statistic | mean CPU s | mean RSS MB |
|---|---|---|---|---|---|---|
| staircase, `tau` = 1, 400 cells | 8 | 9.91 | does not meet the gate | `E[delta_4^2]` | 360 | 6 |
| staircase, `tau` = 1, 4,000 cells | 8 | 2.43 | pass | `E[delta_1]` | 7,435 | 24 |
| staircase, `tau` = 1, 40,000 cells | 8 | 2.91 | pass | `E[delta_5]` | 40,298 | 116 |
| staircase, `tau` inferred, 400 cells | 8 | 1,399.13 | does not meet the gate | `E[tau]` | 114 | 5 |
| staircase, `tau` inferred, 4,000 cells | 8 | 66.75 | does not meet the gate | `E[delta_2^2]` | 2,108 | 9 |
| staircase, `tau` inferred, 40,000 cells | 8 | 2.80 | pass | `E[delta_1]` | 25,233 | 64 |
| log-linear, `tau` = 1, 400 cells | 8 | 11.56 | does not meet the gate | `E[delta_5]` | 381 | 6 |
| log-linear, `tau` = 1, 4,000 cells | 8 | 2.68 | pass | `E[delta_1]` | 7,943 | 29 |
| log-linear, `tau` = 1, 40,000 cells | 7 | 2.02 | pass | `E[delta_1^2]` | 57,825 | 129 |
| log-linear, `tau` inferred, 400 cells | 8 | 1,171.37 | does not meet the gate | `E[tau]` | 184 | 5 |
| log-linear, `tau` inferred, 4,000 cells | 8 | 2,329.38 | does not meet the gate | `E[tau]` | 2,868 | 9 |
| log-linear, `tau` inferred, 40,000 cells | 6 | 1.72 | pass | `E[I_bar]` | 46,392 | 101 |

With the three stopped chains added, the worst `|z|` of the two incomplete arms is 1.40 and 1.59; every
other row is unchanged.  CPU seconds and peak resident memory are the means of the completed chains
(`/usr/bin/time`, user time, one thread), so for the two incomplete arms they are lower bounds.

**Six of twelve arms pass and six do not, and the gate is not relaxed.**  The pattern is by grid
resolution, not by model variant:

* all four 40,000-cell arms pass;
* every 400-cell arm fails, by 9.9 to 1,399 in `|z|`;
* at 4,000 cells the two fixed-`tau` arms pass and both inferred-`tau` arms fail, by 66.75 and 2,329.38.

When `tau` is inferred the failures are in `E[tau]` itself (-1,399, -1,171, -2,329) and in the first
moments of the knot differences; the sampled smoothness is far from its prior.  This is the direction
and the size of the grid-dependent bias that 3.4b measures at 100 tips and that campaign E removes
(3.4a, every arm at every grid within the gate except the one recorded there); it is a measurement of
the cell prior, not a defect of the exact target.  No separate analysis of the mechanism was made for
these arms.

The price of passing is the one 3.5a records: the 40,000-cell cell-prior chains cost 25,233 to 57,825
CPU seconds each at 12 tips, against 2,088 for the exact target at the same grid, and they were the
only chains in this task that did not finish within a day.

### 3.4b Campaign E at 100 tips: the paired comparison, and the size of what is removed

The twelve arms of campaign E at 100 tips, scored once by the same frozen rule.  The last four are the
matched cell-prior arms: same data set, same chain seeds, same grid, the correction switched off.

| arm | worst \|z\| | verdict |
|---|---|---|
| staircase, `tau` = 1, 400 cells, **exact** | 1.99 | pass |
| staircase, `tau` = 1, 4,000 cells, **exact** | 1.46 | pass |
| **staircase, `tau` inferred, 400 cells, exact** | **4.36** | **does not meet the gate** |
| staircase, `tau` inferred, 4,000 cells, **exact** | 0.59 | pass |
| log-linear, `tau` = 1, 400 cells, **exact** | 1.93 | pass |
| log-linear, `tau` = 1, 4,000 cells, **exact** | 1.30 | pass |
| log-linear, `tau` inferred, 400 cells, **exact** | 1.14 | pass |
| log-linear, `tau` inferred, 4,000 cells, **exact** | 0.93 | pass |
| staircase, `tau` = 1, 400 cells, *cell prior* | **459.0** | fail |
| staircase, `tau` inferred, 400 cells, *cell prior* | **1,603.9** | fail |
| log-linear, `tau` = 1, 400 cells, *cell prior* | **822.4** | fail |
| log-linear, `tau` inferred, 400 cells, *cell prior* | **20,873.5** | fail |

**This is the result the task was set to obtain.**  At 100 tips on a grid of the kind Delphy would
build by default, the cell prior misses the gate by two to four orders of magnitude, and the exact
target meets it.  The cell arms' pattern is the discretization bias, not noise: in the staircase,
fixed-`tau` arm every `E[delta_k]` is negative at -459 to -106, every `E[delta_k^2]` positive at +233
to +55, `E[I_bar]` at +178 and `E[log I_bar]` at +298, with the deviation largest in the most recent
knot interval, which is where the lineage count varies fastest within a cell and where
`coalescent_grid.h` predicts the variance term to be largest.  Document 49 measured this excess at 0.26
nats at 100 tips; here it is measured where it matters, in the sampled marginals, and it is enormous.

**The exception, reported as such.**  The staircase arm with `tau` inferred at 400 cells misses at
`|z| = 4.36`.  Its pattern is the same one the 12-tip campaign showed in the corresponding place:
every `E[delta_k^2]` negative (-3.43, -2.93, -4.36, -1.58, -2.87) while every first moment,
`E[I_bar]`, `E[log I_bar]` and `E[tau]` is inside 1.4.  So across the two sizes, both arms that miss
the gate are inferred-smoothness arms at the coarsest grid, both miss on the dispersion of the knot
differences and only there, and the sign is the one the undispersed start predicts (3.4a).  The two
inferred-smoothness arms at 4,000 cells pass at 0.59 and 0.93, the lowest worst-\|z\| of any arm in the
campaign.  The verdict stands: **the inferred-smoothness configuration at the coarsest grid is not
accepted**, at either size, and a dispersed start is what would settle it.

### 3.5 Cost of the three implementations, 100 tips

One data set from `simulate_skygrid_joint.py`: 100 tips, 16,569 sites, a free strict clock under the
Gamma(2, rate 1e8) prior the rate was drawn from, HKY, the topology fixed, 6 knots over 2,000 years,
`tau` fixed at 1.  Four chains of 1e7 steps per arm, one thread unless the arm says otherwise, on the campaign cluster.
`excess` is what Delphy's own `calc_coal_prior_grid_excess` reported for the initial tree, in nats: it
is the distance between the *cell* prior on that grid and the Kingman density, and for an EXACT arm it
describes the surrogate, not the target.

| arm | CPU s per 1e6 steps | peak RSS | grid excess (nats) | batch acceptance |
|---|---|---|---|---|
| A: staircase, cell prior, 400 cells | 0.441 | 6.0 MB | 0.2135 | - |
| B: staircase, cell prior, 4,000 cells | 2.372 | 7.7 MB | 0.006698 | - |
| B: staircase, cell prior, 40,000 cells | 23.608 | 32.3 MB | 0.0004881 | - |
| C: staircase, **exact**, 400 cells | **0.412** | 6.0 MB | (target exact) | 0.695 |
| C: staircase, **exact**, 4,000 cells | 0.601 | 8.0 MB | (target exact) | 0.987 |
| C: staircase, **exact**, 40,000 cells | 2.476 | 21.0 MB | (target exact) | 1.000 |
| A: log-linear, cell prior, 400 cells | 0.655 | 6.0 MB | 0.2313 | - |
| C: log-linear, **exact**, 400 cells | 0.478 | 6.0 MB | (target exact) | 0.544 |
| cell prior, 4,000 cells, 21 knots | 1.988 | 6.7 MB | 0.000181 | - |
| **exact**, 4,000 cells, 21 knots | 0.579 | 7.1 MB | (target exact) | 0.997 |
| cell prior, 4,000 cells, 51 knots | 1.537 | 6.4 MB | 0.000113 | - |
| **exact**, 4,000 cells, 51 knots | 0.717 | 6.6 MB | (target exact) | 0.997 |
| cell prior, 4,000 cells, 2 partitions | 2.127 | 7.5 MB | 0.006698 | - |
| **exact**, 4,000 cells, 2 partitions | 1.054 | 8.4 MB | (target exact) | 0.977 |
| cell prior, 4,000 cells, 4 partitions | 2.070 | 9.2 MB | 0.006698 | - |
| **exact**, 4,000 cells, 4 partitions | 1.265 | 11.9 MB | (target exact) | 0.930 |

Three things to read off it, and one trade-off that has to be stated.

1. **The exact target is cheaper than the approximation it replaces, at every grid.**  At the same
   cell width the exact arm costs 0.93x to 0.73x of the cell arm, because the HMC trajectory no longer
   evaluates the cell prior; the saving grows with the cell count, reaching 9.5x at 40,000 cells.
2. **Against equal accuracy the gap is far larger.**  To bring the *cell* target within 0.0005 nats of
   the Kingman density needs 40,000 cells and 23.6 CPU s per 1e6 steps; the exact target is exact at
   400 cells for 0.412, which is **57x cheaper**, with a fifth of the peak RSS.
3. **Knot count costs the cell path and not the exact path.**  Refining from 6 to 51 knots leaves the
   exact arm at 0.58-0.72 CPU s per 1e6 steps, because its force evaluation is linear in nodes plus
   knots; the cell arm pays for a force loop over the cells in each knot's support.

**The trade-off.**  The grid is now the surrogate's resolution, and a coarse surrogate lowers the
whole-batch acceptance rate: 0.695 at 400 cells against 0.987 at 4,000 and 1.000 at 40,000 for the
staircase, and 0.544 at 400 cells for the log-linear curve, whose cell prior is further from the exact
density.  A rejected batch discards a whole batch of local moves, so the cheapest arm is not
necessarily the best-mixing one; the measured ESS per CPU hour of the log posterior at 100 tips is
10,900 for the cell prior at 400 cells against 5,300 for the exact target at the same grid and 5,900 at
4,000 cells.  Those ESS figures are **not** a like-for-like comparison, because the two arms sample
different targets and the cell arm's target is the biased one; they are reported so that the cost of
correctness is on record.  The honest summary is that the exact target costs nothing in CPU time and
some batch rejections, and that `--v0-coal-prior-fixed-t-step` should be set from the batch acceptance
rate rather than from the excess once the correction is on.

**Convergence caveat.**  At 100 tips and 1e7 steps the rank-split R-hat of the log posterior and of the
most recent knot is 1.16-1.78 in every arm, exact or not, and the bulk ESS of those columns is below
20.  That is a statement about how far a 100-tip Skygrid run is from converged at 1e7 steps, not about
the correction; the SBC campaign reports the gate fraction rather than assuming convergence.

### 3.5b The effect of the mutation-history density

One data set per rate, 100 tips, 16,569 sites, a 173.6 ka genealogy drawn under a Skygrid curve with
prior mean `N_bar` 75 ka, the rate pinned at the generating value so that the arms differ only in the
mutation history.  CPU seconds per 1e6 steps, two chains of 1e7 steps each.

| clock rate | variable sites | mutations in the state | cell 400 | cell 4,000 | cell 40,000 | **exact** 400 | **exact** 4,000 | **exact** 40,000 |
|---|---|---|---|---|---|---|---|---|
| 2e-9 | 97 | 77 | 0.539 | 1.300 | 8.680 | **0.508** | **0.654** | **2.381** |
| 2e-8 | 749 | 749 | 0.850 | 1.477 | 7.215 | **0.855** | **1.025** | **2.325** |
| 2e-7 | 5,788 | 7,354 | 7.575 | 7.829 | 12.434 | **7.391** | **7.481** | **8.674** |

The mutation-history density **dilutes the comparison rather than changing its direction**.  A hundred
fold increase in the rate takes the explicit mutation history from 77 events to 7,354, and the
sequence likelihood then dominates the step cost: at the highest density both targets cost about 7.4
to 7.6 CPU seconds per 1e6 steps at the coarse grid, and the choice of coalescent target is worth a
few per cent rather than a factor.  At the fine grid the gap survives at every density -- 3.6x, 3.1x
and 1.4x at 40,000 cells -- because the grid work is per step and does not shrink when the likelihood
work grows.  Peak RSS moves from 6.1 to 9.9 MB across the whole table, so nothing here is
memory-bound.

The practical reading for a real analysis: on a dense alignment the exact target is close to free, and
on a sparse one it is where the saving is.  Either way the reason to switch it on is 3.4b, not the
cost.

### 3.5a Cost of the two targets across tip counts, from the campaigns themselves

Mean CPU seconds for one chain of 4e7 steps, from the recovery campaigns' own `/usr/bin/time` records,
at 12 tips (campaigns R49 and E, the same data set, the same seeds, one thread) and from the benchmark
at 100 and 1,000 tips.  The two targets are not interchangeable -- the cell prior's is the biased one --
so this is the price of the grid, not a like-for-like speed-up.

| grid | 12 tips, cell prior | 12 tips, exact | ratio | 100 tips, ratio | 1,000 tips, ratio |
|---|---|---|---|---|---|
| 400 cells | 360 | 28 | **12.9x** | 1.07x | 0.93x |
| 4,000 cells | 4,768 | 253 | **18.8x** | 3.9x | 0.87x |
| 40,000 cells | 25,233 to 57,825 by arm (3.4c; lower bounds, incomplete arms) | 2,088 | not pooled | 9.5x | 0.90x |

The ratio falls with the tip count, and the reason is structural rather than incidental: the exact
evaluation costs one pass over the event decomposition, whose size grows with the number of nodes,
while the cell evaluation costs one pass over the grid, whose size is fixed by the pinned width.  At 12
tips the decomposition has about 25 pieces against 400 to 40,000 cells; at 1,000 tips it has about
2,000, and the two meet.  So the exact target is very much cheaper on small trees with a fine grid,
about even on large trees, and never more than about 15% dearer in anything measured here -- while
removing a bias that at 100 tips moves the sampled marginals by hundreds of standard errors (3.4b).

One more number worth recording, because it is what document 49's prescription costs: it recommended
40,000 cells to bring the *approximate* target within 0.02 nats at 1,000 tips.  At 12 tips a
4,000-cell cell-prior chain of this campaign already costs 4,768 CPU seconds against the exact
target's 253 at the same grid, and the 40,000-cell cell-prior arms are the only chains in any campaign
here that have not finished within a day.

### 3.6 Engineering

| check | result |
|---|---|
| release unit suite, whole tree | **749 / 749 pass** (724 at `ff44b23` plus the 25 new tests) |
| debug build, the 17 fast exact-Skygrid tests | **17 / 17 pass**, 42 s; `check_derived_quantities` and the new debug-only invariant (the intensity-integral gradient summing to -1 over the knots that enter a piece) run after every move |
| debug build, the multi-partition incremental test and the node-time quadrature test | **2 / 2 pass**, 15 min; `check_global_and_local_totals_match` therefore ran after every Skygrid move of a 1-, 2- and 4-partition run |
| ASan (`detect_leaks=1`), the same 17 tests | **17 / 17 pass, no sanitizer report** |
| ASan (`detect_leaks=1`), the multi-partition, calibration and node-time tests | **3 / 3 pass, no sanitizer report**, 18 min; this is where the thread pool and the partitioned auxiliary variables are exercised under the sanitizer |
| coalescent and Skygrid CLI behaviour (`run_coalescent_cli_tests.sh`) | **30 / 30 pass**, unchanged |
| exact Skygrid CLI behaviour (`run_exact_skygrid_cli_tests.sh`, new) | **8 / 8 pass**, 9 s |
| pathogen byte regression against a binary built from `ff44b23` | **14 / 14 `IDENTICAL`** (.log and .trees; 1 and 4 threads; default and `old-usher-like` initialization; and the three pathogen-scale Skygrid configurations), 60 samples of 1,200 sites, 300,000 steps each |
| provenance of the campaign binary | the static binary used for every campaign produces a byte-identical log to one built from the committed source, so the campaigns and the repository are the same program |

The new CLI suite is the end-to-end statement, and it is two-sided.  For each of the two variants it
runs the shipped binary twice, once with the correction and once without, writes the trees and the log,
and has `check_exact_skygrid_log_prior.py` recompute both candidate coalescent terms from those files
alone -- merging the events itself, tracking the lineage counts itself and writing out the Skygrid rate
integral itself from the model definition.  The grid is deliberately coarse (20 cells over the knot
range), so the two candidates are far apart and "which one does this run report" has an unambiguous
answer.  Measured: with the correction on the log matches the **exact Kingman density** to 1.2e-5
relative while the cell prior is 0.090 away, a separation of 7,800x; with the correction off the log
matches the **cell prior** to 1.8e-5 while the exact density is 0.099 away.  Both variants behave the
same way.  The residual 1e-5 is the finite precision of the branch lengths in the trees file.


Only the two prior-recovery tests of 3.3 are release-only, and they are Monte Carlo runs whose subject
is a sampled distribution rather than an invariant, so a debug build would re-check what the other
tests already cover at 300 times the cost.

The pathogen regression is the backward-compatibility statement: the default pathogen workflow and the
three pathogen-scale Skygrid configurations are byte-identical to the baseline, so the new code path is
reached only when `--v0-exact-coalescent-correction` is given.

### 3.7 Simulation-based calibration

#### 3.7.0 What was checked before any arm was scored

Three things were verified about the campaign's execution, and one of them was wrong.

**(a) The generator did not receive `--draw-tau`.**  The campaign driver built the generator's tau
arguments into a bash array and exported it.  **A bash array cannot be exported**: `export ARR` marks
nothing, so inside `gen_one`, which runs under `xargs bash -c`, `"${TAU_ARGS[@]}"` expanded to
nothing, and with `set -u` bash 4.4 and later do not complain about an unset array in that form.  The
generator therefore fell back to its own default of a fixed `tau = 1`, while the analysis arm inferred
`tau` under a Gamma(3, rate 3) prior.  Evidence, from the 100 truth files of that group: every one
records `skygrid_tau = 1.0` exactly, `skygrid_tau_prior = null` and `fixed: ["skygrid_tau"]`.  That is
a misspecified SBC -- the generating model is not the analysis model -- so **the inferred-smoothness
group is discarded and rerun**, under `sbc_shallow_discarded`'s sibling
`sbc_sc_drawn_misspecified_discarded`; nothing valid is overwritten.  The rerun records
`skygrid_tau_prior = {alpha: 3, beta: 3}`, no `fixed` entry, and 100 draws from 0.145 to 2.912 with
median 0.914, which is Gamma(3, rate 3).

The fixed-smoothness groups are **not** affected, and this is checked rather than assumed: the
argument that went missing was `--tau 1.0`, and the generator's default is `tau = 1.0` by the same code
path, consuming the same random numbers.  Generating one data set both ways gives byte-identical
`samples.fasta`, `reference.fasta`, `metadata.tsv`, `true_tree.nwk`, `tips.tsv`, `calibration.cal`,
`mask.txt` and `truth.json`.  So the arm scored below stands as run.

The driver now passes the smoothness as scalars and, after each data set, re-reads `truth.json` and
**aborts** if `skygrid_tau_prior` is not what the arm asked for.  A dropped argument can no longer
become a misspecified campaign silently.

**(b) Every data set has its full complement of chains, at the planned length.**  For the arm scored
below: 100 data sets, 4 chains each, every chain 1,002 logged states with the last at state
40,000,000, which is the planned 4e7 steps.  The scorer now checks this itself and prints the count of
exclusions; it reports "every data set found has the full complement of 4 chains".

**(c) The scorer's treatment of failure.**  Three ways a replicate can fail, all now counted rather
than absorbed:

* a data set with fewer than `--require-chains` readable chains is **excluded** from the rank
  statistics and the exclusions are listed by seed.  This is an execution failure, not a
  posterior-dependent filter, so excluding does not bend the ranks -- but an SBC over 60 of 100
  replicates is a different experiment from one over 100 and the output must say which it is.
* a **non-finite rank-split R-hat** -- what one surviving chain or a column that never moved gives --
  now **fails** the gate.  Before this it passed, because the test was written
  `not isfinite(rhat) or rhat <= max_rhat`, which reads the absence of a diagnostic as the absence of
  a problem.
* bulk or tail ESS below `--min-ess` fails the gate, as before.

The gate is reported and never applied to the primary statistics: discarding replicates by their own
effective sample size selects on the posterior and would bend the rank distribution.  The scorer prints
the gated-subset uniformity and coverage beside the full ones, labelled as a diagnostic.

**(d) The ranks were not computed over adequately decorrelated draws, and the rule was tightened.**
The rank of the truth among *correlated* draws is not uniform: in the limit of perfect correlation
every draw is the same number and the rank is 0 or L, so under-thinning pushes the rank histogram
towards its ends.  The first rule thinned the pooled chains to about the **bulk** effective sample
size.  Two things were wrong with it.  It used the bulk figure alone, where the rank is a
tail-sensitive statistic and the tail ESS can be the smaller of the two; and it thinned the
*concatenation*, so the retained draws of one chain sat at a different phase from the next and a lag
could straddle a chain boundary.

The fix is to thin chain by chain with a stride set from the integrated autocorrelation time
`tau = N / min(bulk ESS, tail ESS)`, and to check the result rather than trust the arithmetic: the
scorer now reports the mean within-chain **lag-1 autocorrelation of the retained draws**.  Measured on
the 100 replicates of the arm below, over the 19 quantities:

| thinning rule | stride | residual lag-1, median over replicates | residual lag-1, worst replicate | retained draws, median |
|---|---|---|---|---|
| `ess` (stride `ceil(tau)`, ~ESS draws) | 2 to 30 | -0.005 to **+0.119** | up to **0.37** | 68 to 1,004 |
| `tau2` (stride `ceil(2 tau)`) | 3 to 60 | **-0.047 to +0.033** | up to 0.43 in one replicate | 34 to 668 |

So the original rule left a median residual correlation of up to 0.12 and a worst case of 0.37: **not
adequately decorrelated**, exactly as suspected.  The conservative rule brings every median inside
+-0.05.  The worst individual replicates still reach 0.2 to 0.43 under either rule, and those are the
replicates whose ESS estimate is itself built on a handful of draws, so the stride they get is
unreliable -- which is the convergence gate's message again, not a separate problem.

Both rules are kept and both scores are stored (`*_thin_ess.json`, `*_thin_tau2.json`); `tau2` is the
primary one from here on.  Adopting it after seeing the residual correlations is a change to the
analysis made on evidence about the chains, not about the conclusions, and it has to be declared as
post hoc -- which is why the next subsection reports what changes under each rule.  Nothing in the
frozen gate of Part 2.4 is altered.

#### 3.7.1 The four arms scored at 20 tips

All four have 100 data sets with the full complement of 4 chains of 4e7 steps, 1,002 logged states
each, 50% burn-in, and ranks over the draws the `tau2` rule retains (3.7.0(d)).  "Passed" means every
part of the frozen protocol: uniform ranks at the 0.01 level on both tests, all three coverage figures
inside their binomial 95% intervals, and the convergence gate in at least 90% of replicates.

| arm | passed | not passed | why they are not |
|---|---|---|---|
| staircase, `tau` fixed, no calibrations | **7 / 19** | 12 | 10 miss the convergence gate; `gamma_0`, `gamma_4` and `delta_2` each have one coverage figure outside its interval |
| staircase, `tau` fixed, **4 clade calibrations** | **17 / 19** | 2 | `delta_2` and `delta_5` only, on 95% coverage, at 0.880 and 0.900 against [0.907, 0.993] |
| log-linear, `tau` fixed, no calibrations | **8 / 19** | 11 | 6 miss the gate; `clock.rate` and `gamma_4` miss one coverage figure; `delta_3`'s rank histogram misses uniformity at chi-squared p = 0.0095 |
| staircase, **`tau` inferred**, no calibrations (the rerun arm) | **11 / 20** | 9 | 8 miss the gate; `gamma_2`'s 50% coverage is 0.640 against [0.402, 0.598] |

**The inferred-smoothness arm scores twenty quantities rather than nineteen, and the twentieth is the
end-to-end proof that the `--draw-tau` fix took effect**: the generator now records a prior on the
smoothness, so the scorer scores `skygrid.precision` against the value it was drawn with, which it
could not do while the argument was being dropped (3.7.0(a)).

**`skygrid.precision` passes every criterion.**  Rank uniformity at chi-squared p 0.72 and KS p 0.68,
coverage 0.420 / 0.770 / 0.970 with all three inside their intervals, the convergence gate met in 99%
of replicates, a median bulk ESS of 1,811 and ranks taken over a median of 668 retained draws with a
minimum of 64 and a residual lag-1 autocorrelation of +0.005.  So **the GMRF precision's own posterior
is calibrated and well mixed**, which is the free-smoothness half of the task's "fixed and free Skygrid
smoothness" and is a configuration document 49 had to leave experimental.

Inferring the smoothness does not make the rest worse: 11 of 20 pass against 7 of 19 with it fixed, and
the quantities that gain are the knot differences, whose gate fractions rise to 0.88-1.00.  The ones
that still miss are the same three plus their neighbours -- the clock rate at a gate fraction of 0.35,
the root age at 0.34 and the oldest knot at 0.36 -- which is the rate-time ridge again.

**The calibrated arm is the result that matters, and it is the configuration the task asked about:
Skygrid with a free strict clock and internal node calibrations.**  Seventeen of nineteen quantities
pass every criterion, and the two that do not miss only their 95% coverage by 2 to 3 points.  Every
gate fraction is 0.96 or better, including the two that were the worst without calibrations: the clock
rate goes from a gate fraction of 0.27 and a median bulk ESS of 72 to 0.97 and 1,213, and the root age
from 0.26 and 69 to 0.96 and 1,036.  The retained draws follow: a median of 668 with a minimum of 40,
against a median of 34 and a minimum of 4 in the uncalibrated arm.

The mechanism is not mysterious and it is the same one 3.7.2 of the uncalibrated arm describes.  With
no calibration and one sampling date the data carry no information about the absolute time scale, so
the clock rate, the tree depth and the oldest part of the population curve slide along a ridge that
the moveset crosses slowly; four clade-age measurements pin that ridge, and everything that was slow
becomes fast.  So the limiting factor in the uncalibrated arms is identified, and it is not the exact
target.

**Where each arm's ranks came from.**  Median and minimum retained draws, and the residual lag-1
autocorrelation of those draws, for the quantities at the two ends of the range:

| arm | `clock.rate` | `rootHeight` | `gamma_5` | the base frequencies |
|---|---|---|---|---|
| staircase, no calibrations | 34 / 4, resid -0.047 | 36 / 4, resid -0.038 | 40 / 8, resid -0.034 | 668 / 668, resid -0.010 to -0.002 |
| staircase, **calibrated** | **504 / 72**, resid +0.006 | **504 / 40**, resid +0.001 | **668 / 288**, resid -0.002 | 668 / 668, resid -0.014 to -0.003 |
| log-linear, no calibrations | 32 / 4, resid -0.040 | 30 / 4, resid -0.033 | 44 / 4, resid -0.026 | 668 / 504, resid -0.005 to +0.004 |

Every median residual is inside +-0.05 in all three arms, so the thinning did its job; the worst
single replicate reaches 0.29 to 0.43, and those are replicates whose own ESS estimate rests on a
handful of draws.  The calibrated arm is the cleanest on this measure too, with every residual maximum
at or below 0.29 and most at or below 0.13.

**The one genuine rank-uniformity miss.**  `delta_3` of the log-linear arm has a chi-squared p-value of
0.0095 against the pre-set 0.01.  It is recorded as a miss and the criterion is not relaxed.  For
context and not as an excuse: across the three arms there are 19 quantities x 2 uniformity tests x 3
arms = 114 tests at the 0.01 level, so 1.14 exceedances are expected, and this is one.  Its
Kolmogorov-Smirnov p-value is 0.62 and its coverage is 0.520 / 0.860 / 0.940, all inside their
intervals, so the chi-squared statistic is the only thing that moved.

#### 3.7.2 Node ages

`delphy_clades` and `summarize_sbc.py`, the validated path, with the clade table built by
`--select all` so that nothing reads a simulated age.  The data set is the unit; the uncalibrated arm:

| clades | central 50% | central 80% | central 95% | below 2.5% | above 97.5% | KS p | verdict |
|---|---|---|---|---|---|---|---|
| marked calibrated | 0.510 +- 0.042 | 0.833 +- 0.033 | 0.950 +- 0.018 | 0.030 +- 0.013 | 0.020 +- 0.012 | 0.123 | meets its criterion |
| the rest | 0.504 +- 0.025 | 0.817 +- 0.021 | 0.958 +- 0.010 | 0.017 +- 0.005 | 0.026 +- 0.009 | 0.354 | meets its criterion |

In that arm no calibration file was supplied, so both groups are uncalibrated targets and the split is
only the clade table's marking.  `summarize_sbc.py` applies its own per-clade gate, so these lines are
not a statement about the clades that failed it, and they are read with the same reservation as 3.7.1.

#### 3.7.3 What the 20-tip stage does and does not establish

**It does not establish that the Skygrid model passes SBC without calibrations.**  In the three
uncalibrated arms 7, 8 and 11 of nineteen or twenty global quantities meet every part of the frozen
protocol; the rest do not, almost all of them because the convergence gate is missed, and the worst of
them rest on 30 to 44 effectively independent draws against the 200 the protocol asked for.  Under the
protocol's own rule those are **not accepted**, and a uniform-looking rank histogram does not change
it: the protocol asked for 200 draws precisely so that the test would have power.

**It does establish, for the configuration the task asked about, rather more.**  With a free strict
clock and four internal node calibrations, 17 of 19 quantities pass every criterion, the two misses
being 95% coverage figures 2 to 3 points low on two knot differences, and every convergence gate is
met in at least 96% of replicates.  That is the joint posterior of the Skygrid curve, its smoothness
normalization, the clock rate, the substitution model and the node ages, checked against 100 draws
from the prior predictive of the same model, and it holds.

**And nothing is detected as miscalibrated anywhere.**  Of 77 rank-uniformity tests across the four
arms one falls below 0.01, where 1.54 are expected; of 231 coverage figures, 223 lie inside their
binomial 95% intervals, where about 220 are expected.  A wrong target would be expected to show first
in the quantities with the largest effective sample sizes, and in all three arms those are the ones
that pass.

**Why 4e7 steps is not enough without calibrations, in one number.**  For the clock rate of the
uncalibrated staircase arm the integrated autocorrelation time is `2004 / 72 = 28` logged states, and
one state is 40,000 MCMC steps, so about 1.1e6 steps.  A 4e7-step chain holds at most 36 independent
draws of that quantity and 144 over four chains; the measured bulk ESS of 72 is already within a
factor of two of that ceiling.  Logging more often raises the count towards 144 and no further;
reaching 200 needs three to six times the steps, a move along the rate-time ridge, or -- as the
calibrated arm shows -- information about the time scale.

#### 3.7.4 The four arms at 100 tips

The same four arms, 100 data sets x 4 chains of 4e7 steps each, scored by the scorer and the frozen
protocol of 3.7.1 with the `tau2` thinning rule.  The raw score files are kept in
`evidence_sbc/` (`*_sbc_scores_n100.txt`); the 20-tip calibrated arm's valid score is
`evidence_sbc/sc_fixed_CAL_n20_thin_tau2.json`.

| arm | meet every part | miss only the convergence gate | fail | rank-uniformity p < 0.01 | gate met: clock rate / root height / `gamma_5` |
|---|---|---|---|---|---|
| staircase, free clock, **4 clade calibrations** | 14 of 19 | 3 (clock rate, root height, `gamma_5`) | 2 (`delta_2` 95% coverage 0.89; `gamma_0` 50% coverage 0.40) | none (smallest p 0.2023) | 0.39 / 0.32 / 0.58 |
| staircase, free clock, no calibrations | 6 of 19 | 9 | 4 (clock rate, `gamma_4`, `gamma_5`, root height) | clock rate, root height, `gamma_5` | 0.02 / 0.02 / 0.03 |
| log-linear, free clock, no calibrations | 6 of 19 | 9 | 4 (same four) | clock rate, root height, `gamma_5` | 0.00 / 0.00 / 0.00 |
| staircase, free clock, **free smoothness**, no calibrations | 6 of 20 | 7 | 7 (clock rate, `delta_2`, `delta_4`, `delta_5`, `gamma_5`, root height, `skygrid.precision`) | clock rate, root height | 0.02 / 0.02 / 0.04 |

The three columns of verdicts are the scorer's own: "miss only the convergence gate" means the rank
histogram is uniform and the coverage is inside its intervals but the gate is not met.

Node ages, through `delphy_clades` and `summarize_sbc.py`, by the criterion fixed in advance (95%
coverage and each tail within two cluster standard errors, KS p above 0.01); the group names are those
of 3.7.2.

| arm | marked calibrated | the rest |
|---|---|---|
| staircase, **calibrated** | 95%: 0.943 +- 0.015, tails 0.027 / 0.030, KS 0.390: meets | 95%: 0.948 +- 0.004: meets |
| staircase, uncalibrated | 95%: 0.957 +- 0.017, lower tail 0.010 +- 0.006, KS 0.534: **does not meet** | 95%: 0.947 +- 0.008: meets |
| log-linear, uncalibrated | 95%: 0.977 +- 0.013, 50%: 0.613 +- 0.042, KS 0.096: **does not meet** | 95%: 0.943 +- 0.009: meets |
| staircase, free smoothness | 95%: 0.907 +- 0.027, tails 0.047 / 0.047, KS 0.390: meets | 95%: 0.939 +- 0.011: meets |

**What this establishes.**

* **With four internal calibrations the 100-tip arm does not fail any rank test.**  The smallest
  uniformity p-value is 0.2023.  14 of 19 quantities meet every part of the protocol and the three that
  miss only the gate are the rate-time-ridge quantities, met in 32% to 58% of replicates against 2% to
  4% without calibrations.  Two quantities fail on coverage and are recorded as misses: `delta_2` at
  95% (0.89) and `gamma_0` at 50% (0.40, which is two standard errors below nominal).  The node ages
  meet their criterion in both groups.
* **Without calibrations all three arms have rank histograms that fail uniformity in the clock rate and
  the root height, and two also in `gamma_5`**, with p-values down to 0.0007 and below the printed
  precision.  These are the quantities whose convergence gate is met in at most 4% of replicates.  At
  20 tips the same quantities were under-converged but their rank histograms were uniform (3.7.1); at
  100 tips the under-convergence is severe enough to show in the ranks.  The free-smoothness arm also
  fails on `skygrid.precision`, whose 80% coverage is 0.72.
* So the statement of 3.7.3 that nothing is detected as miscalibrated holds for the 20-tip stage and
  **does not hold at 100 tips**: it fails in the uncalibrated arms by rank uniformity, and in all four
  arms by at least one coverage figure.

**What it does not establish.**  The uncalibrated 100-tip arms are **not accepted**, and the data here
do not separate non-convergence from a defect of the target in them: the failing quantities are the
ones that did not converge, and no longer or better-started chain was run.  What argues against a
defect of the target is that the same model with calibrations, which removes the rate-time ridge,
fails no rank test at the same tip count and chain length.  That is an argument, not a test.  The
node-age failures of two uncalibrated arms are in the clades marked calibrated, which in those arms
carry no calibration; they are recorded and are read with the same reservation as 3.7.1, since
`summarize_sbc.py` applies its own per-clade convergence gate.

## Part 4 - Answers to the questions the task asks

**1. What of BEAST's exact Skygrid coalescent calculation is reusable in Delphy?**

Three things, all three reused and listed with their Delphy counterparts in 1.2: the interval
decomposition (real sampling and coalescent events, subdivided at the curve's knots), the sufficient
statistics `(S_c, n_c)` that make the staircase density a function of `M+1` numbers, and the analytic
rate integrals of the piecewise-constant and log-linear curves.  Delphy already had the third
(`intensity_integral` is `pop_integral` of the negated curve) and had already derived the second in the
HMC's own comment, but evaluated it on the grid rather than on the events.  What was genuinely missing
was the first, and it is the whole of the new code: `Exact_coalescent_events`.  No Java was copied; the
correspondence is established by the formulas and by a test that the sufficient-statistic form and the
general path agree to 1e-9.

What is **not** reusable is BEAST's architecture: its population grid is the model's own resolution,
while Delphy's coalescent grid is an auxiliary structure of the partitioned sampler.  The event
decomposition here is neither - it has no resolution parameter at all.

**2. Was the exact Kingman prior successfully extended to the Skygrid staircase and log-linear
variants?**

Yes, and with the evidence in 3.1 to 3.4 rather than by assertion.  The density agrees with two
independent references and with long-double quadrature; the gradient agrees with a central finite
difference at 512 probes and the HMC forces at 29,708; the target is provably independent of the
coalescent grid and measured to be so to 1e-8; and at the frozen campaign schedule the sampled
marginals recover the Skygrid prior on a grid where the cell prior demonstrably does not.  Both
variants are covered everywhere except the sufficient-statistic reference, which exists only for the
staircase.

**3. Do the Skygrid HMC, the `tau` Gibbs move, the zero-mode Gibbs move and the local tree updates all
target the correct joint posterior?**

Each is argued separately in 1.5 and tested separately in 3.2.  `tau` does not enter the coalescent term
at all, so its conditional is unchanged and the move needed no change - tested by exact equality of the
coalescent term across five values of `tau` and by the closed form of the whole posterior's
`tau`-dependence.  The zero mode's conditional is a Gamma whose rate is the exact density's
rate-integral term, which is an identity of (8) and is tested at four shifts.  The HMC's forces are the
gradient of the exact potential to within 0.084 of what the comparison can resolve.  The local moves
keep Delphy's existing whole-batch correction, whose validity argument does not mention the population
model.  The joint statement - all four together, on one target - is the prior-recovery result of 3.3 and
3.4, which would move if any one of them were wrong.

**4. Does the exact coalescent posterior stay correct across several Run/Subrun partitions?**

Yes, as far as the available tests can show.  The exact density is a function of the assembled genealogy
alone, so partitioning cannot change the target; what it can change is the batch kernel the correction
sits on.  Measured: the incrementally maintained posterior equals a full recomputation and the
coalescent term is still the exact density after 2,000 steps at 1, 2 and 4 partitions over 2 variants
and 6 curves; and the prior recovery passes at 2 and 4 partitions.  In debug builds
`check_derived_quantities` and `check_global_and_local_totals_match` run after every Skygrid move, so
the global total and the sum over parts are compared at every step of those runs.

**5. Does joint inference of Skygrid with a free strict clock and internal calibrations pass SBC?**

**With the internal calibrations the task names: yes, for 17 of the 19 quantities, and the two misses
are named.  Without them: no.**  Three arms at 20 tips, each 100 data sets x 4 chains of 4e7 steps,
the generating model equal to the analysis model (3.7.1):

| arm | quantities meeting every part of the frozen protocol |
|---|---|
| staircase, free clock, **4 clade calibrations** | **17 / 19**; `delta_2` and `delta_5` miss only their 95% coverage, at 0.880 and 0.900 |
| staircase, free clock, no calibrations | 7 / 19; ten miss the convergence gate |
| log-linear, free clock, no calibrations | 8 / 19; six miss the gate, and `delta_3`'s rank histogram misses uniformity at p = 0.0095 |
| staircase, free clock, **free smoothness**, no calibrations | 11 / 20, and the twentieth quantity is the GMRF precision itself, which passes every criterion with its gate met in 99% of replicates |

The calibrated arm covers the Skygrid curve, its knot differences, the geometric mean population, the
free strict clock rate, the substitution model and the root age, with every convergence gate met in at
least 96% of replicates; the node ages pass their own criterion through the `delphy_clades` path.  The
difference between the arms is the rate-time ridge: with one sampling date and no calibration the data
say nothing about the absolute time scale, the clock rate's integrated autocorrelation time is about
1.1e6 steps, and a 4e7-step chain cannot hold the 200 effective draws the protocol asks for.  Four
clade-age measurements remove that, taking the clock rate's median bulk ESS from 72 to 1,213 and its
gate fraction from 0.27 to 0.97.  So the uncalibrated arms are **not accepted**, and the reason is
identified and is not the exact target.

Nothing is detected as miscalibrated in any arm: one of 57 rank-uniformity tests falls below 0.01
where 1.14 are expected, and 164 of 171 coverage figures lie inside their binomial 95% intervals where
about 163 are expected.  **That statement is for the 20-tip stage.  At 100 tips the calibrated arm fails no rank test but
misses two coverage figures, and the uncalibrated arms fail the clock-rate and root-height rank
histograms (3.7.4); the uncalibrated arms are not accepted, and the data do not separate
non-convergence from a defect of the target in them.**

Two defects in the campaign's execution were found and fixed before any of this was scored, both
reported in full in 3.7.0: the inferred-smoothness group's generator never received `--draw-tau`,
because a bash array cannot be exported, which made that group a misspecified SBC -- it is set aside
and rerun; and the ranks were originally taken over draws thinned only to the bulk effective sample
size, which left a median residual lag-1 autocorrelation of up to 0.12 and a worst case of 0.37, so
the rule was tightened to twice the integrated autocorrelation time computed from the smaller of the
bulk and tail ESS, and both scores are kept.  The rerun inferred-smoothness arm and all four arms at
100 tips have finished and are scored in 3.7.4.  What can be said independently of the campaign:

**6. How do speed, memory and ESS per CPU hour compare with the numerical-grid approximation?**

3.5 and 3.6.  In one line: at 100 tips the exact target is 57 times cheaper than the grid refinement
that would bring the approximate target to the same accuracy, and slightly cheaper than the coarse grid
it replaces; at 1,000 tips it is cost-neutral, 0.9 to 1.1 times the cell prior's cost at the same grid,
because the event decomposition then has more pieces than the coarse grid has cells.  No claim that it
is always faster is made, and the measurement was not presumed: the 1,000-tip arms are the ones where
it is not.  Memory is unchanged at the same grid.  ESS per CPU hour is reported but is not a
like-for-like comparison, because the two arms sample different targets; the cost of correctness shows
up as batch rejections at a coarse grid, and the practical advice is to set the grid from the batch
acceptance rate rather than from the measured excess.

**7. Which models now have formal scientific acceptance, and which remain experimental?**

Part 5.

**8. What proven obstacles remain before large-scale general Bayesian phylogenetics?**

* **The topology must be fixed.**  The correction refuses topology moves, because the partition
  stencils then depend on the sampled topology.  This is the single largest restriction and it is
  unchanged by this task.
* **Serially sampled data have no campaign behind them yet, and prior recovery cannot supply one.**
  1.7 proves that prior recovery at a fixed topology measures `p(theta) P(R | theta)` and so is valid
  only for one sampling date, and that an analysis which fixes a topology *by design* on serial data
  is miscalibrated by `1 / P(R | theta)`, which Delphy does not compute.  What is available is an SBC
  conditioning on a topology drawn under the same parameters, which 1.7 shows is calibrated for serial
  sampling; it needs serial sampling in `simulate_skygrid_joint.py` and has not been run.
* **Date uncertainty is still refused** by the correction.
* **Mixing, not the target, is now the limit at 100 tips and beyond.**  At 1e7 steps the rank-split
  R-hat of the log posterior and of the most recent knot is 1.16 to 1.78 in every arm (3.5), and Delphy
  still has no option to disperse the initial Skygrid curve, so chains of one data set start from the
  same flat curve (document 49, 3.5).  Both are efficiency problems and neither is addressed here.
* **A pinned fine grid still grows with the tree.**  With `tau` inferred the tree can wander to several
  times the knot range, and a grid pinned at 0.05 years then has to be extended to tens of thousands of
  cells; in exact mode that cost is pure surrogate overhead, which suggests that the surrogate should be
  allowed to be coarse, but the rule that it must be pinned is kept.
* **The high-population guard is still a proposal, not a change.**  The representability bound of
  document 49, 2.1, is kept and is now also applied to the zero-mode Gibbs move; it truncates the
  prior's support in a region the arithmetic cannot represent.  Replacing it with a prior term remains a
  model change needing approval.

## Part 5 - Model support matrix

**Passed** - validated against an independent reference in this task, at the gates of Part 2:

| item | evidence |
|---|---|
| The exact Kingman density of a genealogy under both Skygrid variants, event-driven | 3.1 |
| Its gradient with respect to the knot values, and the two invariants it must satisfy | 3.1 |
| The staircase sufficient-statistic form, i.e. the correspondence with BEAST's likelihood | 3.1 |
| The exact single-coalescence increment, as the reference a local move's ratio needs | 3.1 |
| The exact target's independence of the coalescent grid | 3.2 |
| The HMC forces as the gradient of the exact potential | 3.2 |
| The zero-mode Gibbs move's conditional under the exact density | 3.2 |
| The `tau` Gibbs move's conditional being unchanged by the exact target | 3.2 |
| Incremental versus full posterior, and the coalescent term, at 1, 2 and 4 partitions | 3.2 |
| Checkpoint and resume under the exact Skygrid target | 3.2 |
| Target distribution of the whole moveset, coarse grid, both variants, 12 tips | 3.3, 3.4 |
| Target distribution at 2 and 4 tree partitions | 3.3 |
| Target distribution with two serially sampled tips | 3.3 |
| Node-time marginals against quadrature, isochronous and two serial patterns, constant population | 3.3 |
| Backward compatibility of the pathogen defaults and of the three pathogen-scale Skygrid arms | 3.6 |
| **Simulation-based calibration of the joint posterior -- Skygrid curve, knot differences, `N_bar`, free strict clock, HKY parameters and node ages -- with four internal clade calibrations, 20 tips, 100 data sets x 4 chains** | 17 of 19 quantities meet every part of the frozen protocol, every convergence gate met in >= 96% of replicates; the two misses are 95% coverage figures 2-3 points low (3.7.1) |
| **The GMRF precision `tau` under its own prior**: rank uniformity, coverage and convergence of `skygrid.precision`, 20 tips, 100 data sets x 4 chains | passes every criterion; gate met in 99% of replicates, median bulk ESS 1,811 (3.7.1) |

**Not passed** - tested and did not meet a pre-set gate:

| item | what happened |
|---|---|
| **SBC of the Skygrid model with a free strict clock and NO calibrations, 20 tips** | 7 of 19 quantities (staircase) and 8 of 19 (log-linear) meet every part of the frozen protocol; the rest miss the convergence gate, whose cause is the rate-time ridge and is identified.  Not accepted (3.7.1, 3.7.3) |
| One rank histogram in 57 uniformity tests: `delta_3` of the log-linear arm | chi-squared p = 0.0095 against a pre-set 0.01; its KS p is 0.62 and its coverage is inside every interval.  Recorded as a miss; 1.14 exceedances are expected over the three arms (3.7.1) |
| **Campaign R49 (cell prior, no exact correction), 12 tips** | 6 of 12 arms meet `\|z\| < 4`: all four 40,000-cell arms and the two fixed-`tau` arms at 4,000 cells; all four 400-cell arms and both inferred-`tau` arms at 4,000 cells do not (worst 1,399 and 2,329).  The campaign was stopped at 93 of 96 chains and its 100-tip stage never ran (3.4c) |
| **SBC at 100 tips, calibrated arm** | no rank test fails (smallest p 0.2023), 14 of 19 quantities meet every part, 3 miss only the gate; `delta_2` (95%: 0.89) and `gamma_0` (50%: 0.40) miss on coverage (3.7.4) |
| **SBC at 100 tips, three uncalibrated arms** | the clock-rate and root-height rank histograms fail uniformity in all three, `gamma_5`'s in two; gates met in at most 4% of replicates.  Not accepted; non-convergence is not separated from a defect of the target (3.7.4) |
| Campaign E's **inferred-smoothness arm at the coarsest grid**, at 12 and at 100 tips | misses `\|z\| < 4` on the dispersion of the knot differences, at 4.62 and 4.36; the same arm passes at the finer grids, and a dispersed start shrinks the deficit threefold (3.3, 3.4a, 3.4b) |
| Campaign E's **cell-prior arms at 100 tips**, which are the control | miss the gate by two to four orders of magnitude.  This is the measurement the task was set to make, not a defect (3.4b) |

**Experimental** - usable, with a stated caveat, not accepted:

| item | caveat |
|---|---|
| The exact correction with **more than about 100 tips** | the target is right, but at 1e7 steps the chains are not converged (3.5); report R-hat and ESS |
| A **coarse surrogate grid** under the correction | the target is unaffected, but the whole-batch acceptance rate falls to 0.70 at 400 cells and 0.54 for a log-linear curve, which costs tree mixing (3.5) |
| Skygrid with **serially sampled tips** | the sampler accepts it, the density and the gradient are validated heterochronously (3.1) and the node-time marginals match quadrature under two serial patterns (3.3), but no campaign has been run on serially sampled data.  Prior recovery is not available there (1.7); an SBC conditioning on a topology drawn under the same parameters would be, and needs serial sampling in the generator |
| Skygrid with `--v0-skygrid-infer-prior-smoothness true` | document 49's caveats stand; in addition a small `tau` lets the tree wander and a pinned fine grid then grows to tens of thousands of cells |

**Not tested** - no evidence either way from this task:

The exact Skygrid target with a free topology (refused); with date uncertainty (refused); with site
partitions or relative rates; with HKY85, TN93 or GTR beyond the HKY the SBC campaign uses; with
continuous Gamma site rates; with a callable mask or missing data; at more than 1,000 tips for cost or
more than 100 tips for calibration; on any real data set.  The joint time-scale move remains refused for
a Skygrid curve and is unchanged.

## Part 6 - Hand-over

* Branch `human-uniparental/exact-skygrid-task31`, from `ff44b23`; not pushed; `main`, the Task Z branch
  and the Skygrid acceptance branch untouched.  Document 49 is not edited.
* Source changes: `core/exact_coalescent.{h,cpp}` (new), `core/pop_model.{h,cpp}` (one gradient
  accessor), `core/coalescent_grid.{h,cpp}` (the merged event list shared rather than duplicated),
  `core/run.{h,cpp}` (the support of the correction, the two population moves' use of the exact
  density, the zero-mode finiteness guard, the resolution-aware force diagnostic),
  `core/cmdline.cpp` (help text).  No new MCMC operator, no new prior term, no change to the EMAT, the
  explicit mutation histories, the incremental likelihood, the mutation-guided topology proposals or the
  Run/Subrun framework.
* New tests: `tests/exact_skygrid_tests.cpp` (9) and `tests/exact_skygrid_mcmc_tests.cpp` (15).  Two
  existing refusal tests were updated to the new, correct expectations.
* New tooling: `tests/human_input/simulate_skygrid_joint.py` (with a `selftest` subcommand),
  `run_exact_skygrid_recovery_{task31.sh,campaign_task31.sh}`,
  `run_exact_skygrid_bench_{task31.sh,campaign_task31.sh}`, `score_exact_skygrid_bench_task31.py`,
  `run_exact_skygrid_sbc_{task31.sh,campaign_task31.sh}`, `score_exact_skygrid_sbc_task31.py`.
* Compute: every unit-test result on the development machine; every campaign on the campaign cluster under
  the campaign directory, with the static binary built from this branch and the baseline binary
  built from `ff44b23` for the regression.
* Campaign state at hand-over: R49 was stopped by the operator at 93 of 96 chains at 12 tips and its
  100-tip stage did not run (3.4c); every other campaign finished.  The score files are in
  `evidence_r49/` (R49, complete chains only and with the three stopped chains) and `evidence_sbc/`
  (SBC arms); the chains themselves remain on the campaign cluster under the campaign directory, and the
  R49 scoring directory is the campaign directory.  No process of this task is running
  on the campaign cluster.

### Entry conditions for the next task

1. **Before a free topology with Skygrid**: the correction's refusal of topology moves is the blocking
   item, and it is about the partition stencils, not about the population model.
2. **Before any heterochronous Skygrid result is reported as validated**: a validation design that does
   not condition on the simulated topology (1.7).
3. **Before a 1,000-tip Skygrid analysis**: a mixing story.  The target is right at that size; the
   chains are not converged at the step counts measured here.
4. **A dispersed start for the Skygrid curve on the command line**, which is a two-line change in the
   library (3.3) and is the one concrete thing that would settle the two arms that miss their gate.
   It is a new option rather than a model change, and it is not added here because it was not in scope.
   *Re-examined in Part 7:* the in-suite deficit this item rests on is explained by the statistic;
   a dispersed start is not needed for correctness.
5. **Carried forward unchanged from document 49**: the high-population guard is a proposal, not a
   change; and `Very_scalable_coalescent_prior_part` still has no floor for a vanishing cell population
   where `Scalable_coalescent_prior` has one, so the single-part and multi-part surrogates differ in
   that corner.  Under the correction a difference in the surrogate costs acceptance rate rather than
   correctness, which lowers the priority but does not close the item.

## Part 7 - The dispersion deficit re-examined (2026-11-22)

### 7.1 What was observed

The in-suite test `a_dispersed_start_removes_the_dispersion_deficit` and the coarse-grid recovery test
failed in the WASM build of the CI on `|z| < 4`, and a sweep over 12 seeds on two standard libraries
gave second-moment z-scores of the knot differences that were negative in 19 of 24 runs of the
dispersed-start arm.  Both tests ran one chain of 20,000 global moves per configuration and scored the
time average of each statistic with a batch-means z (20 batches after a 5% burn-in).

### 7.2 Three hypotheses and the experiment that separates them

A, the target is wrong; B, a transition kernel does not leave the target invariant; C, both are right
and the deficit comes from mixing, the start or the error estimate.  A and B are tested without any
assumption about mixing by starting each chain from an **exact draw of its target**: tau from its
Gamma(3, rate 3) prior (or 1), the knot differences given tau, I_bar from its Gamma prior, and the
genealogy given the curve by exact inversion of the cumulative pair hazard.  With isochronous tips the
ranked topology is independent of the curve, so the drawn topology can be held fixed.  If the kernel
leaves the target invariant, every state of such a chain is an exact draw, and the time average of
each statistic has exactly its prior expectation at any chain length; independent replicates then give
a t-test whose validity does not depend on the mixing time.

The driver is the disabled test `Exact_skygrid_validation.DISABLED_prior_recovery_driver`
(`tests/exact_skygrid_mcmc_tests.cpp`); the runs, scripts and reports are in the campaign directory
`87-Delphy-exact-skygrid-dispersion` (phase 1).  Arms: staircase and log-linear, fixed and inferred
tau, three starts (exact, the old test's dispersed start, flat), 200 replicates of 20,000 global moves;
plus 64 replicates of 200,000 global moves for the inferred-tau arms from the exact and the dispersed
start.

### 7.3 Results

* **No bias.**  Over 16 arms and about 180 statistics the across-replicate z of the time averages is
  inside `|z| < 2.2`.  In the four long arms with inferred tau the mean of `delta_k^2` is 1.4925 to
  1.5052 against `E[1/tau] = 3/2` (`|z| <= 1.2`), split R-hat is at most 1.001 and the ESS per chain is
  1,400 to 15,000.  The flat and the dispersed starts give the same answers; the largest start effect
  at 20,000 moves is `z = -2.3` (staircase, inferred tau, flat start).
* **The statistic is biased.**  On the same exact-start chains the batch-means z of `delta_k^2` has a
  mean of -0.4 to -0.8, also at 200,000 moves, and 4.5% to 9.5% of single chains of 20,000 moves with
  inferred tau cross `|z| >= 4` on some statistic (1% to 2% with tau fixed; 3% to 6% at 200,000
  moves).  The reason is the distribution: with tau ~ Gamma(3, rate 3), `delta_k^2` has a finite
  variance but an infinite fourth moment, so its time average is right-skewed and its batch variance
  is small exactly when the average is low.  The five differences share tau, so their z-scores are
  strongly correlated and the 19 negative runs of 24 are far fewer than 24 independent signs.
* **Joint behaviour.**  The correlation of `log tau` and `log sum delta_k^2` is -0.66 per chain, as the
  Gibbs coupling implies; there is no slowly mixing region.  The batch acceptance of the correction is
  0.987 to 0.993.

Verdict: C.  The target and the kernels are right; the deficit reported by the test was a property of
its statistic.  The campaign deficits of 3.4a and 3.4b were scored with the same kind of statistic and
are consistent with this explanation, but those chains were not re-scored.

### 7.4 The replacement design

The four recovery tests (`the_marginal_of_the_skygrid_parameters_is_their_prior_on_a_coarse_grid`,
`the_marginal_is_still_the_prior_across_tree_partitions`,
`the_marginal_is_the_prior_with_an_inferred_smoothness`, which replaces the dispersed-start test, and
`the_marginal_is_the_prior_with_two_heterochronous_tips`) now run 160 independent replicates of 120
global moves per configuration, each from an exact draw of the target, and score the **end states**,
which are independent exact draws if the kernel is right.  Each end state is reduced to probability
integral transforms that are exactly uniform under the target: `F(tau)`, `F(I_bar)` and
`Phi(delta_k sqrt(tau))`.  For each the mean and the mean of `(u - 1/2)^2` are compared with 1/2 and
1/12 using their exact variances 1/12 and 1/180.  The statistics are bounded, so the normal
approximation is accurate in the tails.  The gate `|z| < 4.5` is a Bonferroni bound over the at most
104 z-scores of the four tests: a correct sampler fails one of them with probability below 1e-3 per
build.

Measured on c2 (phase 2): over 200 independent seed bases per configuration (1,600 simulated runs of
the configurations) the z-scores have mean -0.01 to +0.06, standard deviation 0.98 to 1.02 and
`P(|z| > 3)` of 0.0017 to 0.0039 (0.0027 for a normal), and **no run fails**.  Power, measured by
running the chains with a prior different from the one the end states are scored against (50 runs
each): a Gamma(3, rate 2) tau prior instead of rate 3 fails every run, rate 2.5 one run in five; an
N_bar prior scale 1.5 times too large fails every run, 1.2 times one run in three to five.  A wrong
tau prior is seen through `F(tau)` only, because `delta_k sqrt(tau)` is N(0, 1) under any tau prior; a
kernel error in the curve moves would move the `delta_k` statistics.  The four tests take about 25 s
on one c2 core.

### 7.5 What is not changed, and what remains open

No model, prior, move or acceptance ratio was changed.  The other batch-means recovery tests
(`skygrid_acceptance_tests.cpp`, `pop_moves_prior_recovery_tests.cpp`, the node-time quadrature test)
use the same kind of statistic with tau fixed or with lighter tails and were not changed; their
false-alarm rates were not measured.  The SBC and the campaign arms of Parts 3 and 4 keep their
recorded status.

