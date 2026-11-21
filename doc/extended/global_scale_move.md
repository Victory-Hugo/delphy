# Joint time-scale move: mathematical design

The move is experimental and OFF by default; no default schedule is changed.

## 1. Audit of the existing operators (no equivalent joint move)

Local moves (core/subrun.cpp, Subrun::mcmc_sub_iteration): inner_node_displace, tip_displace, branch_reform, subtree_slide (weights 7.5, 7.5,
15, 1), spr1 only with topology moves.  Each changes one node or one branch.  Global moves (core/run.cpp, Run::run_global_moves): mu_move
(Gibbs given the tree), HKY frequency and kappa moves, alpha moves, grid retargeting, pop_size_move (scale random walk on n0 given the
tree), pop_growth_rate_move (random walk on g given the tree), Skygrid moves.  Every one conditions on the current node and mutation ages;
none changes mu, n0, g and the ages together.  The only joint scale moves in the repository are inside the independent reference sampler
(tests/prototypes/ref_sampler.cpp), which shares no code with Run.  Result: there is no equivalent joint move.

## 2. The transformation

State variables (fixed topology, all tips at one sampling date T, no date uncertainty): the ages of the n_int = N - 1 internal nodes, the
ages of the M mutation events, mu, n0, g.  Ages are measured back from T: a = T - t.  For u in R and s = exp(u):

    a_v -> s a_v (internal nodes), a_m -> s a_m (mutations), mu -> mu / s, n0 -> s n0, g -> g / s.

The tips stay at T, the topology, the mutation sites, states and order, and the missation sets do not change; mutation ages keep their
order along each branch because the map is monotone.

Meaning of the parameters.  The exponential model is N(t) = n0 exp(g (t - t0)) (floored at min_pop), t0 = T.  In ages N(a) = n0 exp(-g a).
Then N'(a') = s n0 exp(-(g/s) s a) = s N(a): population size as a function of age scales with the ages, so the coalescent rate per pair at
age a' equals 1/s times its value at a; g is a rate (1/time) and transforms inversely to time.  min_pop is a fixed constant of the model
and is NOT rescaled: the target is evaluated exactly at the transformed state, so a floor that is active only makes the ratio differ from
the scale-free value, never invalidates it.  t0 must be the sampling date (refused otherwise); this keeps the map linear in (n0, g).

Jacobian.  The map is linear and diagonal in the variables: |J| = s^(n_int) s^M s^(-1) s^(+1) s^(-1) = s^(n_int + M - 1) (nodes, mutation
ages, mu, n0, g).  The proposal u ~ Uniform(-w, +w) is symmetric, the inverse of the map for u is the map for -u, so the Metropolis-Hastings
ratio of the deterministic transformation is

    log alpha = log pi(x') - log pi(x) + (n_int + M - 1) u,

where pi is the density of the full target in exactly these variables: the sequence likelihood G (including the factor mu^M of the mutation
density and exp(-mu Ttwiddle)), the EXACT Kingman density of the node ages (log P_exact), the calibration priors (functions of the calibrated
node ages), the priors on mu (Gamma, density in mu), n0 (inverse gamma, density in n0), g (truncated Laplace, density in g; proposals with g'
outside [g_min, g_max] are rejected) and the rest (alpha, nu, kappa, pi: unchanged).

Closed forms (checked numerically by scale_move_checks.cpp, min_pop = 0): G' = G s^(-M) because mu^M scales by s^(-M) and mu Ttwiddle is
invariant (Ttwiddle scales by s); log P_exact' = log P_exact - n_int u because each of the n_int coalescences contributes a factor 1/N and
the integral of the pair intensity is invariant.  Hence log alpha = (change of the priors of mu, n0, g) + (change of the calibration prior)
- u.  With the default priors (mu: uniform, n0: 1/n0) the prior part is -u + the g prior change.  The implementation does not use these
closed forms: it evaluates the target at the transformed state from scratch (Run::validate_derived_quantities) and adds the Jacobian.

Calibrations enter once, through log_calibration_prior_ at the transformed ages; the support of every prior is respected (-inf: reject).

## 3. Interaction with the whole-batch exact correction (no double counting)

The move is executed in the global step, between batches of local moves, from a state in which tree_ is assembled.  Its target is
the exact Kingman density; it never uses the cell prior or the batch variable delta, and the cell prior is recomputed from the new
genealogy as bookkeeping (log_coal_exact_correction_ is a function of the state, not an acceptance factor).  The batch correction then
treats the post-move state as its starting state: delta_before is read after the global step, so the two kernels are separate Markov
kernels leaving the exact target invariant and compose.  It requires the exact correction (the grid prior is not scale invariant),
a fixed topology, one sampling date, the exponential model, no mpox hack; otherwise std::invalid_argument.  After an accepted move the
partition is rebuilt from the new tree (repartition); after a rejected one tree_, mu and the population model are restored from the snapshot.

## 4. Implementation

core/run.h, core/run.cpp: Run::set_global_scale_move_enabled / window / proposals per global step, counters (proposed, accepted, sum of
accepted |u|), Run::global_scale_move_with(u, uniform_draw) for tests, Run::global_scale_move (draws u and the accept variable from
the run's generator) called at the end of run_global_moves only when enabled.  Disabled (the default): no code path changes, no random
number is drawn, so every output of existing workflows is unchanged.  Not exposed in the CLI.

Harness: tests/prototypes/batch_correction_real.cpp reads SCALE_WINDOW (> 0 turns the move on) and SCALE_K from the environment in the
posterior and mix modes; tests/prototypes/scale_move_checks.cpp checks the algebra, rollback and reversibility.

## 5. Frozen criteria (written before any posterior-comparison or N = 1,000 run of the move)

### 5.1 Correctness

(a) scale_move_checks.cpp, N = 8, 20, 100, scenarios none/root/clade/joint: the log ratio of Run equals the one from a freshly built Run at the
transformed state (1e-6 relative); d log G = -M u and d log P_exact = -n_int u; an accepted move equals the fresh Run; a rejected move restores
tree, mu, n0, g and the log posterior exactly; the inverse move has the opposite log ratio and restores the state; the maintained log posterior
equals a fresh recomputation after 60 global steps; unsupported configurations are refused.  All checks must pass.

(b) Independent posterior: Delphy production path with the move on (window 0.1, 3 proposals per global step, 1 partition), 6 chains per cell
(start scales 0.5, 1, 2, twice), cells (N, scenario) = (8: root, clade, joint), (20: root, joint), (100: root, clade, joint), the data, grids,
batches and seeds of Tasks M and N (reference chains of Tasks M and N, ref_sampler, 3 per cell).  Gates are those of Task N (doc 26, section 3):
H1 R-hat <= 1.01; H2 bulk and tail ESS >= 400; H3 mean agreement with the reference (Welch t of the chain means against the reference, family-wise
alpha 0.01 over the parameters of the cell, practical margin 0.05 reference sd); H4 quantile agreement judged as in Task O (conservative z of the 2.5, 50
and 97.5 per cent quantiles, the largest of three standard errors; failure if any |z| > 4.17); the move must be exercised: acceptance between 0.05 and
0.95 in every cell.  A failing cell stops the task.

(c) Regression: with the move off the pathogen workflow output is byte-identical to the Task R build for a fixed seed; the existing unit tests pass.

(a2) Added before any formal run: the three scale-carrying terms (d log G + d log P_exact + log Jacobian) are checked to add to -u to 1e-9
(`scale_free_residual`; observed 1e-12 at N = 100), so that the cancellation of large terms is monitored directly, and the prior parts of
the ratio are compared with independent formulas (mu: gamma, n0: inverse gamma, g: truncated Laplace, calibrations: truncated lognormal)
computed from the data file.  The residual diagnostic does not change any draw or decision.

### 5.2 Efficiency at N = 1,000 (same data, priors, initial states and seeds as Task R)

Arms: baseline = the 30 chains of Task R (no move), truncated to the same sampling time; treatment = the same 30 seeds and start scales with the
move on, each chain for 2 h of sampling (clock includes the cost of the move), one partition, 0.5-day grid, 50-sweep batches.
Tuning (pilot, discarded, not pooled): windows {0.02, 0.05, 0.1, 0.2} with 10 proposals per global step on 3 chains of 10 min; the window with the largest
squared jump distance of log mu per second of sampling is used; proposals per global step: the largest of {1, 5, 20, 100} whose added cost is at most 15 per cent of a batch.
Metric: pooled bulk ESS (rank-normalized, 20 per cent burn-in of the stage, arviz) per core-hour for log mu, log n0, root age, cal1997, cal1996, cal1985
at the stage of 2 h of sampling (and at 1 h), each as the ratio treatment / baseline with a chain-level bootstrap interval (resampling chains within arm, 200
resamples).  Secondary: ESS per core-hour of g and of the minimum over the 40 uncalibrated nodes; autocorrelation time of log mu in rows and in
seconds; accepted fraction and the mean accepted |u| of the move; squared jump distance per hour.
Decision: "the move substantially improves ESS per hour" if the lower 95 per cent bootstrap bound of the ratio exceeds 1.25 for every one of the six slow
quantities and the median of the six ratios is at least 2, with the ratios of g and of the minimum of the 40 nodes at least 0.8; "worth integrating" only
if in addition the pooled posteriors of the two arms agree (means within 0.1 pooled sd, conservative quantile z within 4.17 for the six slow quantities and the 40 nodes).
Anything else is reported as not demonstrated.  Convergence is not claimed from the short chains.

### 5.3 What is not claimed

Equality with a reference at N = 1,000 does not exist; agreement of the two arms at N = 1,000 is a consistency check, not a correctness proof.
Dates that differ between samples, Skygrid, free topology, several sampling dates and partitions with non-uniform rates are not supported.

## 6. Tuning record (pilot, discarded; formal configuration fixed here, before the formal chains start)

Pilot: 12 chains, 600 s of sampling each (seeds 8101 to 8112, start scales 0.5, 1, 2), 10 proposals per global step, data of Task R, one chain per
core on the campaign cluster (machine load average 35 to 65 from other users).  Burn-in: first 30 per cent of the rows.

| window | squared jump distance of log mu per second (mean of 3 chains) | acceptance | mean accepted abs log s | batches per second |
|---|---|---|---|---|
| 0.02 | 0.0186 | 0.949 | 0.0098 | 16.1 |
| 0.05 | 0.0406 | 0.873 | 0.0238 | 16.1 |
| 0.1 | 0.0428 | 0.752 | 0.0447 | 16.2 |
| 0.2 | 0.0455 | 0.552 | 0.0755 | 16.3 |

Rule of section 5.2: the largest squared jump distance per second is at window 0.2.  Cost: the three replayed chains of Task R ran 23.6 batches per
second in their first 600 s; with 10 proposals per global step the pilot ran 16.2, so one proposal costs about 1.9 ms, 4.6 per cent of a batch (42 ms);
of {1, 5, 20, 100} proposals only 1 stays within the 15 per cent limit (5 would cost about 23 per cent).  Formal configuration: window 0.2, 1 proposal per
global step, 30 chains with the labels, seeds and start scales of Task R, 2 h of sampling each.  The pilot chains are not part of any comparison.
Because the load of the shared machine differs between the Task R run and this one, the report also gives the comparison at an equal number of batches
(load independent) next to the frozen comparison at an equal sampling time.

## 7. Results: correctness (section 5.1)

(a) scale_move_checks.cpp: N = 8 (none, root, joint), N = 20 (joint, root), N = 100 (joint, clade, none): 84 to 88 checks per run, all passed, 0 failed;
largest scale-free residual (dG + dlogP_exact + jacobian + u) 4.7e-12; the independent prior and calibration changes agree with Run to 1e-9.
Not covered by a test: several partitions with the move on, Skygrid (refused by construction), dates that differ between samples (refused).

(b) Independent posterior (48 chains: 8 cells x 6, move on with window 0.1 and 3 proposals per global step, exact correction on, one partition; 3 reference chains per
cell of Tasks M and N; report tests/prototypes/results_taskS/gates_taskS.md).  All 8 cells pass every gate:

| N | scenario | scale move acceptance | H1 worst R-hat | H2 min bulk / tail ESS | H3 failing params / max abs t | H4 largest conservative abs z, count above 4.17 |
|---|---|---|---|---|---|---|
| 8 | clade | 0.816 | 1.0001 | 95370 / 268586 | 0 of 15 / 1.93 | 2.54, 0 of 45 |
| 8 | joint | 0.755 | 1.0001 | 81101 / 237992 | 0 of 15 / 1.76 | 2.76, 0 of 45 |
| 8 | root | 0.813 | 1.0001 | 94851 / 275204 | 0 of 15 / 2.25 | 2.44, 0 of 45 |
| 20 | joint | 0.753 | 1.0006 | 10407 / 25889 | 0 of 27 / 3.27 | 2.84, 0 of 81 |
| 20 | root | 0.814 | 1.0044 | 3487 / 41348 | 0 of 27 / 1.92 | 2.37, 0 of 81 |
| 100 | clade | 0.816 | 1.0053 | 1084 / 5303 | 0 of 107 / 2.42 | 2.58, 0 of 321 |
| 100 | joint | 0.756 | 1.0038 | 1348 / 5130 | 0 of 107 / 2.29 | 3.08, 0 of 321 |
| 100 | root | 0.813 | 1.0057 | 1238 / 5351 | 0 of 107 / 3.54 | 2.73, 0 of 321 |

Every cell exercised the move (acceptance 0.75 to 0.82, mean accepted |log s| 0.045).  The correctness gate is passed; the formal N = 1,000 chains were started
after this result (30 chains, window 0.2, 1 proposal per global step, 2 h of sampling each, binary with the CPU-time and load columns).
(c) Regression: with the move off, the body of the log and the trees file of a 24-sample pathogen-style run (3,000,000 steps, fixed seed) are byte-identical
to those of the Task R build (headers differ only in the commit and the invocation); the 548 unit tests of the repository pass.

## 8. Results: efficiency at N = 1,000 (section 5.2; report tests/prototypes/results_taskS/eff_taskS.md)

Configuration (fixed in section 6 before the run): window 0.2, 1 proposal per global step, the 30 labels, seeds and start scales of Task R, 7,200 s of sampling each, one pinned
core per chain on the campaign cluster.  Baseline: the 30 chains of Task R, not rerun.  Machine load: the treatment chains ran at a mean 1-minute load average of 42.6; the Task R chains did not log it
(observed 28 to 35 shortly after the additional chains started), so the two arms were not run under identical load.  Treatment CPU time was 1.002 times the wall time; CPU time of the baseline
was not recorded.  The move: 4,234,061 proposals, 2,332,083 accepted (0.551), mean accepted |log s| 0.0756; the largest scale-free residual over all proposals was 2.8e-10.

Three denominators are kept apart.  Point estimates of the pooled bulk ESS, 30 chains, 20 per cent burn-in of the stage, stage 2 h:

| quantity | ESS baseline | ESS with move | ESS per wall-hour per core, baseline -> with move (frozen comparison) | ratio per batch (equal number of batches, load independent) | ESS per CPU hour, with move |
|---|---|---|---|---|---|
| log mu | 2045 | 447326 | 34.1 -> 7455 (x 219) | x 246 | 7443 |
| log n0 | 2619 | 401651 | 43.7 -> 6694 (x 153) | x 172 | 6683 |
| root age | 1568 | 33054 | 26.1 -> 551 (x 21) | x 24 | 550 |
| cal1997 | 2395 | 68510 | 39.9 -> 1142 (x 29) | x 32 | 1140 |
| cal1996 | 2221 | 53245 | 37.0 -> 887 (x 24) | x 27 | 886 |
| cal1985 | 3916 | 174403 | 65.3 -> 2907 (x 45) | x 50 | 2902 |
| g | 669917 | 607957 | 11165 -> 10133 (x 0.91) | x 1.00 | 10116 |
| min of 40 uncalibrated nodes | 3140 | 6332 | 52.3 -> 105.5 (x 2.0) | x 2.4 | |

At 1 h the ratios are 252, 182, 14, 29, 39, 61 (the same six quantities), g 0.91, minimum of the nodes 3.4.  Autocorrelation time of log mu: 413 rows without, 1.4 rows
with the move (96 s against 0.4 s of sampling); squared jump distance of log mu per hour 15.9 against 168.  Batches per second of sampling 21.56 against 19.60: the
move costs about 9 per cent of the batch rate here (about 4.6 per cent in the pilot), which is the 0.91 of g.

Frozen decision rule: median ratio of the six slow quantities 36.6 (at 2 h; 49.9 at 1 h); every point ratio far above 1.25 (smallest 14 at 1 h and 21 at 2 h); g and the minimum of the 40
nodes at least 0.8: substantial improvement = YES at both stages.  Agreement of the pooled posteriors of the two arms: largest mean difference 0.038 pooled sd (limit 0.1), largest conservative
quantile |z| 2.80 over 138 tests (limit 4.17) at 2 h; 0.026 and 1.98 at 1 h: they agree.

Limitation of the report: the chain-level bootstrap intervals printed by taskS_efficiency.py are not reliable.  They thin the rows by 5 and resample chains with replacement; duplicated chains
and the thinning distort the ESS of an arm whose samples are almost independent, so the intervals are not centred on the point estimates (for instance log mu: point ratio 219, interval 56 to 71; root
age: 21, interval 18 to 34).  The decision above rests on the point ratios and on the size of the lower bounds, not on the intervals.  A group-wise ratio without resampling (three groups of
10 chains with the same start scale, taskS_efficiency_groups.py) was started afterwards as a supplementary check; it is not part of the frozen rule and its result is added below when it is available.

Not shown by these data: convergence (the chains are 2 h long; Task R showed R-hat of the root age at 1.013 after 4 h without the move), correctness at N = 1,000 (no independent reference), robustness to
other windows, other datasets, several partitions, other population models.  The ESS of the minimum over the 40 uncalibrated nodes improves by a factor of about 2, much less than the slow quantities.
