# Whole-batch exact coalescent correction: staged validation

This file is written stage by stage; Section 1 is stage 1, Section 2 is stage 2 (the audit), later sections are added with
their stages.  Evidence classes: **[P]** proof from the source, **[N]** independent numerical test, **[R]** test on the real kernel,
**[M]** measurement, **[U]** unverified.

## 1. Stage 1: R-34 (integer overflow in the exact-prior reference functions)  [N]

Fixed in `core/coalescent_grid.cpp` (`calc_exact_coalescent_log_prior`) and `core/coalescent.cpp` (`Coalescent_prior::calc_log_prior`):
`(k * (k - 1)) / 2` is now `0.5 * static_cast<double>(k) * (k - 1)`.  The delta functions multiply an `int` by a `double` and were not affected.
The test helper `kingman_events` in `tests/coalescent_grid_tests.cpp` had the same overflow in its rate (`k * (k - 1) / 2.0`), which made its
generated coalescence times fall after the sampling time from 46,342 tips on (this made the first version of the regression test abort in
`Coalescent_prior::calc_log_prior`); fixed as well.

Verification (commit `609acf4`):

* New unit test `Coalescent_grid_test.the_exact_prior_does_not_overflow_the_pair_count_at_tens_of_thousands_of_tips`: N = 46,341, 46,342,
  50,000, 52,930, constant and exponential populations, against a reference that shares no code with the functions under test (summation by
  parts, H = sum_j (n - j) Lambda(a_j), in long double), tolerance 1e-9 relative.  **It fails on the old code** from N = 46,342 (difference
  0.303 nats at 46,342, thousands of nats at 52,930; N = 46,341 passes on both) and passes on the fixed code.
* `tests/prototypes/check_exact_overflow.py`: genealogies drawn in python, reference by the interval-by-interval closed forms (float64) and by
  summation by parts (longdouble), against `calc_exact_coalescent_log_prior` and `Coalescent_prior::calc_log_prior`: worst relative
  difference 8.5e-13 over the four sizes, g = 0 and g = 1e-5 per year.
* Release suite 539 of 539 passed; Debug/ASan `Coalescent*`, `Scalable*`, `Very_scalable*`: 28 of 28 passed.

## 2. Stage 2: audit of the real batch kernel  [P unless stated]

Scope of the claims: **fixed topology, all tips at one date without date uncertainty, exponential population model, pinned grid, no calibrations**
(the configuration of the first experimental version, Section 3).  Where a statement is weaker or fails outside that scope, it says so.

### A1. The conditional update of the Gaussian variables is the conditional of J(x, z | theta)

In a cell c with m = `num_active_parts[c]`, width t, averaged population size N (`popsize_bar`), per-part cell-averaged lineage count k_p and K = sum k_p, the
sampler draws, independently for every (cell, part) with the part active in the cell, z ~ Normal(k_p - K/m, N / (m t))
(`core/very_scalable_coalescent.cpp:199-211`, mean `k_bar_p[i] - k_bar[i] / num_active_parts[i]`, standard deviation `sqrt(popsize_bar[i] / (num_active_parts[i] * t_step))`).
`k_bar` is the sum of the parts' `k_bar_p` (lines 175-185), so K is the total lineage count.  Cells a part is not active in get z = 0 and are excluded from m.
Define J(x, z | theta) = P_cell(x | theta) prod p(z_cp | k_cp).  Each factor integrates to one and N depends on theta and the grid only, so the x-marginal of J is P_cell exactly,
and the conditional of z given x is the Gaussian above.  The x-dependent part of the code's per-cell prior (`calc_partial_log_prior`, line 356, and the delta function)

    A(k, z) = -(t/N) [ (m/2) sum_p k_p^2 - (m sum_p z_p k_p - Z K + K/2) ]

satisfies A = cell(k) + log p(z | k) (without its constant) + (m t / 2N) sum_p z_p^2, an identity whose last term does not depend on k: **at fixed z, exp(A) is proportional to J(x | z)** (algebra in `22_...md`, 1.1).
Test: `Very_scalable_coalescent_test.part_priors_add_up_to_the_cell_prior_times_the_conditional_density_of_the_auxiliary_variables` builds parts through the public
constructor with chosen k and z and checks the identity to 1e-8 for six different k at fixed z (passes).  [N for the implementation, P for the algebra]

The z draw happens in `Run::push_global_params_to_subruns` (`core/run.cpp:359-369`, via `reset_very_scalable_coalescent_parts`, 371) before **every** local batch, from the current tree and the current population model,
so the order global moves -> z | x, theta -> local batch is the Gibbs scheme of J.  When the root moves older than the cells allocated at the start of the batch, `ensure_space` (`very_scalable_coalescent.cpp:260-293`)
appends cells in which only the root part is active (m = 1, k_p = 1) and draws a new z ~ Normal(0, N/t) for them; for m = 1 the part prior reduces to the cell prior and does not depend on z, so this extra draw has no effect on any acceptance ratio.

### A2. The part ranges and partition are fixed during a batch

Each part's active cell range comes from `t_min`, `t_max` of its nodes at batch start (`very_scalable_coalescent.cpp:116-132`): tips contribute their (frozen) `t_min`, `t_max`, inner nodes their current time, and the root part always extends to the earliest tracked cell.
In a part, `t_max` is attained at tips (leaves are the youngest nodes), and `t_min` at the part's root, which is a cut-point node: frozen for non-root parts (the root of a non-root part is only moved when the part includes the root of the whole tree, `Subrun::inner_node_displace_move`, `core/subrun.cpp:224`);
for the root part, the range is "everything older than the youngest frozen tip", which does not depend on where the root is.  Local moves keep every inner node between its children and its parent, so no node leaves its part's range, and the sets m_c are the same at the start and at the end of a batch.
With a fixed topology, no move changes which nodes belong to which part, so the stencil gives the same partition at x and at x' (`partition_tree` is deterministic given the cut points, `core/tree_partitioning.h:196`).

### A3. Each local move is reversible with respect to its target at fixed z

Read from `core/subrun.cpp` (fixed topology, so `spr1_move` is not in the mix and topology-changing slides are rejected by `move_blocked_`):

| move | proposal | acceptance |
|---|---|---|
| `inner_node_displace_move` (219) | non-root: `Bounded_exponential_distribution` with rate d logG/dt = -lambda_node + sum over children of lambda just below the node, bounds from parent, children and mutation times; none depends on the node's own time.  Root of the whole tree: Gaussian, scale `min(factor/(2 lambda_root), t_max_tip - t_max)`, independent of the root's own time | `delta_log_G + delta_log_prior - log_alpha`, log_alpha = (d logG/dt) dt, zero for the symmetric root proposal |
| `tip_displace_move` (320) | same construction; no effect for tips without date uncertainty | same |
| `branch_reform_move` (378) | uniform resampling of the mutation times on one branch (symmetric) | `delta_log_G` |
| `subtree_slide_move` (444), fixed topology | Gaussian in the time of P, scale `min(1/(2 lambda_X), t_max_tip - t_early)` with t_early the root time (P not the root) or min(t_X, t_S) (P the root): neither changes when P's time changes; proposals that would change the topology are blocked | `delta_log_G + delta_log_prior` (ratio 1.0 for height-only changes in `spr_move_core`) |

`delta_log_prior` is the exact change of the part prior A at the current z and grid (the incremental formula is tested against the full recomputation in `Very_scalable_coalescent_test.delta_log_prior`).  Early returns (bounds, zero-length branches, blocked moves) are rejections.
Every move is therefore a Metropolis-Hastings step for its target pi_A(x | z) proportional to P_cell(x) p(z | x) L(x) (likelihood, mutation history included), hence reversible with respect to it.
The acceptance rules are read from the source; the validity of the proposals (e.g. the Hastings terms of the bounded exponential) is inherited from earlier validation (Tasks D, E, F) and supported numerically: Task K and Section 4 below
run these moves from exact joint draws.  Supplementary test in this stage: inner-node displacement alone (`ONLY_INNER=1`, fixed topology, N = 20): spacing chi-squared 20.6 and 16.8 on 19 for the corrected chains (1 and 2 parts), 246.4 on 19 uncorrected (`results_taskL/stage2_inner_only_stationarity.log`).  [N]

### A4. The batch kernel (random scan, several parts, several steps) is reversible

`Subrun::mcmc_sub_iteration` (139) draws the move type with fixed weights at every step (inner 7.5, tip 7.5 or 0 depending on a constant property of the part, reform 15, slide 1).  One step is a mixture of reversible kernels, hence reversible; a power of a reversible kernel is reversible.
`Run::run_local_moves` (825) gives each subrun `count / #subruns` steps (remainder to subrun 0), a function of the batch length only.
At fixed z the part priors A_p depend on the part's own k_p and on z only (the cross terms are linear in k_p), the boundary nodes are frozen and each part's likelihood is local, so the parts are independent given z and the product of their batch kernels is reversible.
No systematic scan over move types, no state-dependent step counts or weights exist in `Subrun` or `Run`.  Composition in a fixed order would break this and break the correction (exact toy of Task K: stationary error 7e-2 against 1e-15, `batch_order_toy.py`).

### A5. The stencil update and the fixed grid do not break invariance (in the restricted configuration)

* **Stencil.**  `generate_random_partition_stencil` (`core/tree_partitioning.h:139`) walks `randomized_post_order_traversal(tree, bitgen)` and uses only the children structure and subtree sizes: **the topology, the node indices and the generator, never node times.**  Test: `Very_scalable_coalescent_test.the_partition_stencil_depends_on_the_topology_and_the_generator_only` (same topology with node times differing by a factor 37 gives identical stencils for the same generator state).  [N]
  With a fixed topology the stencil list is therefore generated from information that no sampled variable changes; refresh by step count (`Run::refresh_partition_stencils`, `core/run.cpp:92`) or after a rejection is a state-independent choice of the kernel, and a uniformly chosen stencil gives a mixture of kernels each leaving the target invariant.
  **This argument does not hold with a free topology**, where the stencil depends on subtree sizes, i.e. on the state: that is the existing "changes slowly" approximation and is why the first version excludes free topology.
* **Grid.**  With a pinned grid (`Run::set_coal_prior_fixed_t_step`) `t_step` is a constant, and `t_ref` is the tip date, constant for same-date samples.  The corrected target does not depend on the grid at all (the grid enters pi* only through delta, which cancels the cell prior).  Automatic retargeting (`run_global_moves`, step 4-pre) is a function of the state and is excluded.

### A6. The whole-batch acceptance ratio  [P]

With Q_B the batch kernel (A1-A5: reversible with respect to pi_A(. | z) at fixed z, fixed partition, fixed grid) and pi*(x, z) proportional to pi_A(x | z) exp(delta(x, theta)), delta = log P_exact - log P_cell,
the Metropolis-Hastings ratio of the proposal x -> x' is

    pi*(x') Q_B(x', x) / (pi*(x) Q_B(x, x')) = exp(delta(x') - delta(x)) * [pi_A(x') Q_B(x', x)] / [pi_A(x) Q_B(x, x')] = exp(delta(x') - delta(x)),

because the bracket is 1 by reversibility.  delta does not depend on z, so the conditional z | x, theta of pi* is still the Gaussian of A1 and the refresh needs no change.  The x-marginal of pi* is P_cell exp(delta) = P_exact.
Population parameters theta: the marginal over z of pi* given x is P_exact(x | theta); a Metropolis-Hastings move on theta with the exact prior ratio (z is redrawn afterwards) targets the correct conditional.  Moves that do not involve the coalescent prior (mu, kappa, pi, alpha) are unchanged.

### A7. Rejection restores the state completely

The state of a batch is the `Phylo_tree` `tree_` (topology, times, mutation lists, missation maps) and, for the global parameters, values that are not modified inside the batch (they change in the preceding global step).  `Run` has no other state the target depends on: the derived quantities (log_G, sums of mutation counts, the global cell prior `coalescent_prior_`, the Gaussian variables and the parts) are recomputed from `tree_`; the subruns hold copies of the subtrees and must be rebuilt (`repartition()`) after a rollback because they carry the rejected batch.
The random number generators are deliberately **not** restored (restoring them would replay the rejected batch).  Tested in Task K (800 forced rejections on trees with mutations; exact equality of trees and caches) and re-tested at stage 3 on the implementation.  [R]

### A8. What is not proven, and is therefore excluded or flagged

| item | status |
|---|---|
| Free topology (stencil depends on the state; SPR moves) | not proven; excluded from the first version |
| Automatic grid retargeting | not proven; excluded (pinned grid required) |
| Skygrid population (HMC gradient of the exact hazard; barrier terms) | not derived; excluded |
| Heterochronous tips, tip-date uncertainty | tip moves and the cell ranges change; not analysed; excluded |
| Calibrations | contained in pi_A formally, not tested with the correction; excluded |
| Validity of each move's proposal (annealed SPR studies, Hastings terms) | inherited from earlier tasks and numerical tests, not re-proved |
| The remaining accumulated floating-point error of cell bookkeeping | not analysed; the cell value is recomputed from the tree at every batch end |

**Verdict of the audit.**  For the restricted configuration the conditions required by the correction hold: the Gaussian update is the exact conditional (A1); the batch kernel is reversible (A3, A4); the stencil and grid choices are state-independent (A2, A5); the acceptance ratio needs only delta (A6); and a rejection can be undone completely by restoring `tree_` and rebuilding the parts (A7).  Stage 3 proceeds under these restrictions.

## 3. Stage 3: the experimental `Run` option  [R, N]

`--v0-exact-coalescent-correction` (CLI, `--v0-human-mode` only, default off; write `=true` / `=false` because cxxopts boolean options take an
implicit true, so `--flag false` turns the option ON) and `Run::set_exact_coalescent_correction(bool)`.

**What it does** (`core/run.cpp`, `core/run.h`):

* `Run::run_corrected_local_batch`: snapshot of `tree_`; `run_local_moves`; `reassemble`; recompute the derived quantities, which include
  `log_coal_exact_correction_` = log P_exact - log P_cell of the proposed genealogy (`calc_exact_coalescent_log_prior`, the fixed R-34 function); accept with
  probability min(1, exp(delta' - delta)) using the `Run` generator; on rejection restore `tree_`, `invalidate_derived_quantities()` and `repartition()`
  (the subruns carry the rejected batch).  The generators are not restored.  Counters `exact_coalescent_batches_proposed()` and `_accepted()`.
* The population-size and growth-rate moves compare `log_coalescent_prior_ + log_coal_exact_correction_`, the target's coalescent term, before and after the
  proposal, with the exact density evaluated at the proposed parameters.
* `log_coalescent_prior()` and `log_posterior()` report the target (cell prior plus the correction); the BEAST-style log uses `calc_cur_log_coalescent_target()`.
  With the option off the correction term is exactly 0.0, so every value, and the order of the floating-point operations that matter, is unchanged.

**Refused configurations** (`std::invalid_argument`, CLI: `ERROR: --v0-exact-coalescent-correction: ...` with what to change): topology moves enabled (and enabling them later),
a grid that is not pinned (and releasing it later, retargeting it, or changing the population model to a non-exponential one), age calibrations, the mpox hack,
samples that do not share one date or have date uncertainty.  Outside `--v0-human-mode` the flag is refused.

**Regression results.**

| check | result |
|---|---|
| Release unit tests | 545 of 545 passed (539 before this task plus 6 new) |
| Debug/ASan unit tests | 545 of 545 passed (988 s) |
| new tests `Exact_coalescent_correction_test` (4): off by default bit for bit; refusals (free topology, unpinned grid, Const and Skygrid population, heterochronous tips, tip date uncertainty, and the changes that would break a supported configuration); reported coalescent term equals the exact density after every batch, 1 and 2 parts, with rollbacks, and a Run rebuilt from the genealogy agrees on every derived quantity; the grid bias of the first coalescence age (z of the corrected chains below 4, of the uncorrected above 6, 500 chains) | passed in Release and Debug/ASan |
| pathogen byte identity, `run_pathogen_regression.sh` (1 and 4 threads, default and old-usher-like initialization, three Skygrid configurations; `.log` and `.trees`) against a binary built from the same sources without the stage 3 files | 14 of 14 IDENTICAL, outputs non-empty (16,742 bytes of log body for the first); the two binaries differ |
| CLI scripts `run_coalescent_cli_tests.sh` (30), `run_cli_scale_tests.sh`, `run_external_tree_cli_tests.sh`, `run_calibration_cli_tests.sh`, `run_clade_summary_cli_tests.sh`, new `run_exact_correction_cli_tests.sh` | 0 failures |

A first pathogen comparison in this stage compared the base binary with itself (the release binary had been rebuilt with the stage 3 files stashed) and reported 14 of 14 identical;
that result was discarded, the binary was rebuilt with the stage 3 sources, the two were checked to differ, and the comparison above was run again.

## 4. Stage 4a: posterior validation against an independent reference sampler, N = 8 and N = 20  [N, R]

### 4.1 Reference sampler (`tests/prototypes/ref_sampler.cpp`)

Shares no code with Delphy.  Model: fixed topology, all tips at one date, inner node ages a_v (days before the sampling date, child age < parent age);
**exact Kingman density** with N(a) = n0 exp(-g a), log P = -sum_j (n - j) Lambda(a_(j)) - sum_v log N(a_v), Lambda(a) = (exp(g a) - 1) / (g n0);
likelihood by Felsenstein pruning with HKY (r = kappa on transitions, R = sum pi_a r_ab pi_b, q_ab = r_ab pi_b / R, rate mu per site per day, root sequence from pi),
summing over all mutation histories (Delphy samples them explicitly); priors mu ~ Gamma, n0 ~ InvGamma, g ~ Laplace on [g_min, g_max], log kappa ~ Normal(1, 1.25^2), pi uniform.
Moves: single node ages (reflecting uniform window, incremental pruning along the path to the root), log mu, log n0, g, log kappa, delta-exchange on pi, and joint
time-scale moves (ages and mu / s; and additionally n0 x s, g / s) with Jacobian s^(m-1).
**Check of the likelihood and the prior** (`check_ref_likelihood.py`, scipy `expm`, brute force): log-likelihood and Kingman log prior agree with the sampler's to nine decimals on both data sets
(N = 8: -805.711210132 and -57.460361482; N = 20: -1034.154551310 and -153.365870748).
The reference chains are also checked against Delphy with an accurate grid and no correction (Section 4.3): three independent routes to the exact posterior.

### 4.2 Design

* Data: `batch_correction_real dataset N 300 5e-5 SEED`: one Kingman genealogy (n0 = 1,000 days), HKY mutations (kappa = 2, pi = (0.3, 0.2, 0.2, 0.3), 300 sites, mu = 5e-5 per site per day):
  N = 8 (277 mutation events, root 1,423 days), N = 20 (309, 1,459), N = 100 (350, 1,797).  The same file feeds both samplers; Delphy starts from the true genealogy and mutation histories
  rescaled by 0.5, 1, 2 (the reference starts from the same rescaled ages), with kappa = 1, pi uniform, so the three chains start dispersed.
* **Priors (proper, informative): mu ~ Gamma(50, mean 5e-5), n0 ~ InvGamma(10, 9000) (mean 1,000 days), g ~ Laplace(5e-4, 2e-4) on [0, 2e-3] per day.**
  A first N = 8 attempt with the improper prior on n0 (`1/n0`) and weak priors on mu and g failed for a reason unrelated to the correction: every uncorrected Delphy chain (coarse and fine grid) fell into an
  absorbing degenerate state (n0 about 1e-287 days, g at its bound, node ages about 1e-14 days, mu about 2.5e-3 per day) and one corrected chain spent 0.3% of its draws near it, so the
  Delphy chains had ESS of 3 to 5 and R-hat up to 2.2 (`results_taskL/stage4_n8_improper_prior_comparison.txt`).  This is the known collapse of the free exponential model under the
  default improper prior (R-27) and, for the corrected chain, a flag that deserves a separate look; it was not pursued here.  The exact posterior under a proper prior has no mass there.
* Delphy: `Run` with the stage 3 option, fixed topology, pinned grid, mu / kappa / pi / n0 / g moves on (5 population-move pairs per global step), 1 part, **30 local moves per batch at N = 8 and 60 at N = 20**.
  Configurations: **A**, correction on with a coarse grid of 1 n0 (1,000 days; the grid error is large); **B**, no correction on the same coarse grid (the original sampler; positive control: it must fail);
  **C**, no correction on a fine grid of 0.02 n0 (the original sampler on an accurate grid; it should agree).  Three chains each (2e6 batches for A and B, 5e5 for C, every tenth batch kept, first 20% discarded);
  reference: three chains of 1.5e6 sweeps, every 15th kept.
* Gates fixed before the comparison (for the corrected configuration A): rank-normalized split R-hat <= 1.01, bulk and tail ESS >= 400 for every parameter and node age, differences of the
  posterior means below 4 standard errors, and quantile differences (2.5%, 50%, 97.5%) below 0.1 reference standard deviations.
  **Change made after seeing the data (disclosed):** the standard error of the difference of means was first computed from the arviz bulk ESS.  For N = 20, pi_A and pi_C then gave |z| of 4.3 and 4.7
  (differences of 0.015 posterior standard deviations), although the same chain's first and second half gave z = +7 and +0.25 against the reference and the uncorrected fine-grid control showed the same
  pattern (+-5): the global frequency parameters wander slowly within a chain and the ESS overstates their precision.  The gate was therefore applied with **batch-means standard errors** (each chain cut into 20 blocks, 60 blocks in all); the ESS-based
  values are reported alongside.

### 4.3 Results (all node ages, mu, n0, g, kappa and the four frequencies; N = 8: 15 parameters, N = 20: 27)

Columns: worst R-hat / minimum bulk ESS / minimum tail ESS over all parameters of the Delphy chains; worst |z| of the difference of means with ESS-based and with batch-means standard errors (and the number of parameters above 3);
largest difference of means in reference standard deviations; largest difference of a quantile in reference standard deviations.

| N | configuration | | R-hat / bulk ESS / tail ESS | worst z (ESS) | worst z (batch means) | max diff of means (ref sd) | max quantile diff (ref sd) |
|---|---|---|---|---|---|---|---|
| 8 | Delphy+correction, coarse grid (1 n0) | 1.0006 / 4132 / 9006 | 1.96 (kappa), 0 of 15 above 3 | 1.71 (kappa), 0 above 3 | 0.0204 | 0.030 |
| 8 | no correction, coarse grid (1 n0) | 1.0007 / 4725 / 8756 | 83.85 (g), 11 of 15 above 3 | 82.76 (g), 10 above 3 | 0.2922 | 0.479 |
| 8 | no correction, fine grid (0.02 n0) | 1.0020 / 1133 / 2014 | 2.07 (pi_T), 0 of 15 above 3 | 1.83 (pi_T), 0 above 3 | 0.0409 | 0.083 |
| 20 | Delphy+correction, coarse grid (1 n0) | 1.0028 / 2728 / 5323 | 4.69 (pi_C), 2 of 27 above 3 | 2.28 (pi_A), 0 above 3 | 0.0321 | 0.034 |
| 20 | no correction, coarse grid (1 n0) | 1.0028 / 3170 / 7049 | 354.31 (g), 24 of 27 above 3 | 343.88 (g), 23 above 3 | 1.3371 | 2.055 |
| 20 | no correction, fine grid (0.02 n0) | 1.0043 / 857 / 1608 | 2.62 (pi_G), 0 of 27 above 3 | 2.21 (pi_G), 0 above 3 | 0.0379 | 0.091 |

Reference chains: N = 8: 3 x 80,000 draws, worst R-hat 1.0000, minimum bulk ESS 186,807; N = 20: 3 x 80,000 draws, worst R-hat 1.0000, minimum bulk ESS 123,750.
Batch acceptance of the corrected chains: **0.963** (N = 8, 30 local moves per batch, 2e6 batches per chain) and **0.882** (N = 20, 60 moves).  Throughput (local, 1 core): 1.53e4 batches per second (N = 8), 7.44e3 (N = 20, campaign cluster).
Peak RSS 3 to 5 MB.  Raw comparisons: `results_taskL/stage4_n8_comparison.txt`, `stage4_n20_comparison.txt`; the three corrected chains' counters in the `.err` files.

**Verdict of stage 4a.**  The corrected sampler passes the gates at N = 8 and at N = 20: R-hat <= 1.003, ESS >= 2,700, no difference of means above 2.3 batch-means standard errors (the largest, 0.03 reference standard deviations),
quantiles within 0.034 reference standard deviations.  The uncorrected coarse-grid control fails decisively (n0 biased low by 8% at N = 8 and by 20% at N = 20; 10 of 15 and 23 of 27 parameters above 3 standard errors),
which shows the test can see the grid bias; the uncorrected fine-grid control agrees with the reference, which supports the reference sampler and the exactness of the target it reproduces.
Limits: one data set per size; the mean differences are resolved to about 0.02 to 0.03 posterior standard deviations; the global frequency parameters have slowly wandering chains, so their precision is limited by the three chains; the kernel is the 1-part kernel (the multi-part kernel is covered by the stationarity tests of Task K, not by this comparison).

## 5. Stage 4b: N = 100  [N, R, M]

Started only after stage 4a passed (the independent reference chains for N = 100 were started earlier, since they do not depend on Delphy).  All chains ran on the campaign cluster (single process each,
`nice`, machine load 10 to 25 from other users' jobs, so wall times carry that noise); analysis locally.

**Setup.**  Data set N = 100 (300 sites, 350 mutation events, root 1,797 days), the same proper priors as in Section 4.2.  Delphy: 1 part, **200 local moves per batch**, three chains per configuration started from the genealogy rescaled by 0.5, 1, 2;
**corrected**: pinned grid of 0.05 n0 = 50 days, 4e6 batches, every 20th kept (160,000 draws after the first 20% discarded); **original, same grid** (no correction), 0.05 n0; **original, coarse** (0.5 n0); **original, fine** (0.01 n0 = 10 days, 1.5e6 batches).
Reference: three chains of 4e5 sweeps (32,000 draws after burn-in), worst R-hat 1.0004, minimum bulk ESS 10,899.  107 parameters: 99 node ages, mu, n0, g, kappa, four frequencies.  The runs go through the `Run` API with the stage 3 option (the CLI binary was exercised separately, Section 3), the same code path the CLI uses.

**Posterior agreement** (`results_taskL/stage4_n100_comparison.txt`; gates as in 4.2):

| configuration | worst R-hat | min bulk / tail ESS | worst z (ESS) | worst z (batch means) | max diff of means (ref sd) | max quantile diff (ref sd) |
|---|---|---|---|---|---|---|
| **corrected, 0.05 n0** | 1.0023 (a192) | 1,496 / 3,787 | 3.36 (a129), 2 of 107 above 3, none above 5 | 3.14 (a129), 2 above 3 | 0.034 | 0.088 (a130) |
| original, same 0.05 n0 grid | 1.0025 | 1,435 / 5,118 | 297 (n0), 106 of 107 above 3 | 269 (n0) | 2.49 | 3.35 (n0) |
| original, coarse 0.5 n0 grid | 1.0018 | 2,944 / 6,384 | 705 (n0), 106 above 3 | 681 | 3.60 | 5.17 |
| original, fine 0.01 n0 grid | 1.0061 | 914 / 1,981 | 19.5 (n0), 103 above 3 | 20.7 (n0) | 0.35 | 0.43 (n0) |

Key parameters, posterior mean (reference in brackets): corrected: mu 5.03e-5 (5.01e-5), n0 817.9 (819.7), g 3.69e-4 (3.67e-4), root age 1,966 (1,978) days;
original, same grid: mu 6.43e-5, n0 452, g 3.24e-4, root age 1,487 (the time scale is 40% too short and n0 45% too small);
original, fine grid: mu 5.18e-5, n0 767, root age 1,909 (still 19 standard errors off).
**All pre-set gates hold for the corrected sampler**: R-hat 1.0023 <= 1.01; bulk and tail ESS above 1,400 for all 107 parameters; no |z| above 4 (largest 3.14 with batch-means errors); quantile differences 0.088 < 0.1 reference standard deviations.
Caveats: two ages have |z| a little above 3 (about 0.3 expected by chance among 107 tests, though the node ages are strongly correlated); the largest quantile difference is close to the gate and comparable to the quantile's own Monte Carlo error with a reference effective sample size of about 11,000 draws; the mean differences are resolved to about 0.03 posterior standard deviations.

**Cost** (`results_taskL/stage4_n100_ess_per_hour.txt`; three chains each, the campaign cluster wall time; harness memory):

| set | batches per chain | chain-hours (3 chains) | batches / s | peak RSS | ESS per chain-hour: mu / n0 / g / root age / kappa | minimum over all 107 |
|---|---|---|---|---|---|---|
| corrected, 0.05 n0 | 4e6 | 1.474 | 2.27e3 | 4 MB | 1,948 / 3,830 / 21,103 / 1,073 / 19,724 | 1,015 (a192) |
| original, same grid | 4e6 | 1.012 | 3.31e3 | 4 MB | 4,546 / 6,735 / 65,727 / 2,532 / 28,453 | 1,419 (a192) |
| original, coarse 0.5 n0 | 4e6 | 0.887 | 3.76e3 | 3 MB | 6,907 / 10,003 / 252,750 / 3,317 / 40,713 | 3,317 (a198) |
| original, fine 0.01 n0 | 1.5e6 | 0.574 | 2.2e3 | 4 MB | 2,918 / 5,161 / 28,606 / 1,592 / 27,531 | 1,592 (a198) |

* **True whole-batch acceptance of the corrected chains: 0.724, 0.723, 0.722** (200 local moves per batch, grid 0.05 n0).
* The corrected chain takes 1.46 times the wall time of the original sampler on the same grid (a rejected batch discards its work, and each batch pays one exact evaluation, a snapshot, and, on rejection, a rebuild of the subruns), and its ESS per hour is 0.43 (mu), 0.57 (n0), 0.42 (root age), 0.32 (g), 0.69 (kappa) and 0.72 (minimum) of the original's.
  Against the original on the finer 0.01 n0 grid, which is slower per batch but still biased, the corrected chain has 0.67 to 0.74 of its ESS per hour.  The original sampler's numbers describe a **wrong target** (Section 4 and the table above), so they bound the price of exactness from above and are not a like-for-like speed comparison.
* Not measured: multiple parts, other batch lengths (the batch here is about one sweep over the nodes; the production default is 50 sweeps, where the per-batch cost of the correction is relatively smaller, Task K), whole-process memory of the CLI, and runs without other load on the machine.

## 6. Answers

1. **Does the whole-batch correction strictly preserve the exact Kingman posterior?**  In the restricted configuration (fixed topology, same-date samples, exponential population, pinned grid, no calibrations) the derivation holds from the source (Section 2, A1 to A7): the Gaussian update is the exact conditional, the random-scan batch is reversible, the stencil depends on the topology only, the grid is pinned, the acceptance ratio needs only delta, and rollback is complete.  The validity of each individual move's proposal is inherited from earlier tasks and from numerical tests, not re-proved.  Numerically the corrected sampler reproduces an independent reference at N = 8, 20 and 100 to within about 0.03 posterior standard deviations in the means and 0.09 in the quantiles, while the original sampler on the same grid is biased by several posterior standard deviations (up to 45% in n0 at N = 100).  Not covered: free topology, automatic grid retargeting, Skygrid, heterochronous samples, calibrations (refused by the option), and the multi-part kernel in a posterior comparison (only Task K's stationarity tests).
2. **Is N = 100 consistent with the independent exact reference posterior?**  Yes under the gates fixed beforehand (Section 5), for one data set, 1 part, with the caveats listed there.
3. **Measured acceptance and ESS per hour.**  Whole-batch acceptance 0.963 (N = 8, 30 moves per batch), 0.882 (N = 20, 60), 0.723 (N = 100, 200); ESS per chain-hour at N = 100: mu 1,948, n0 3,830, root age 1,073, g 21,103, kappa 19,724, minimum over all parameters 1,015 (campaign cluster, loaded machine).
4. **Is Delphy's computational advantage kept?**  The local moves, the cells, the partition and the incremental updates are untouched, and the correction adds one O(n) evaluation per batch.  In this one-part test at N = 100 the price was 1.46 times the wall time and 0.3 to 0.7 of the ESS per hour of the original (biased) sampler on the same grid; the multi-part case and long batches were not measured here (Task K: correction 0.1% of a K = 200 batch at N = 1,000 on a 10-year grid, acceptance 0.6 to 1.0 where the grid is adequate).  Pathogen defaults and outputs are byte-identical.
5. **Conditions to enter an exact N = 1,000 posterior validation.**  Met: the mathematics for the restricted configuration, a working option with refusals and regression tests, agreement with an independent reference from N = 8 to 100.  Open before starting it: (a) the multi-part kernel compared with the reference on a posterior (N >= 20, 2 to 4 parts); (b) a fixed grid adequate for 1,000 samples chosen from the measured grid excess (at N = 100 even 0.01 n0 left a 0.35 sd bias in the original sampler; Task K measured acceptance 0.6 to 1.0 at N = 1,000 on 1-year grids); (c) proper priors for the free exponential model, since the improper prior on n0 produced a collapse in every uncorrected chain and once in a corrected one (Section 4.2), which should be understood before long runs; (d) a reference sampler that scales (its cost grows with N; about 4 ms per sweep at N = 100 here), or an alternative reference such as the uncorrected sampler on a demonstrably accurate grid; (e) your approval.  I recommend (a) and (c) first; they were not started.

## 7. Limits and things not done

One data set per size; the correction is validated for 1 part against the reference (multiple parts: Task K stationarity only); N = 100 went through the `Run` API, not the CLI binary; the batch-means standard error was adopted after seeing the data (disclosed in 4.2); the collapse near n0 = 0 in a corrected chain under the improper prior was not investigated; non-uniform grids, Skygrid, free topology, calibrations and heterochronous samples are refused or untested; no the campaign cluster job of other users or the running stage 3 was touched (the machine was shared: load 10 to 25).
