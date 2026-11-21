# Exact Kingman with a free topology on several parts: the threshold partition

Evidence labels: [P] proven by
reading the code and the algebra, [R] run, [N] not done.

## 1. The problem

With the exact coalescent correction (`exact_coalescent.md`), each batch of local moves is a kernel K that
is reversible with respect to the cell-prior target pi_cell, followed by one Metropolis-Hastings step on
Delta = log P_exact - log P_cell.  K is reversible only if the partition the batch uses at the start x is the
partition the reverse batch would use at the end y.  With a fixed topology this holds because the stencils depend on
the topology only.  With a free topology the default stencils (`generate_random_partition_stencil`) are drawn from a
randomized traversal that depends on subtree sizes; their probability q(z | x) cannot be evaluated, so the
correction was restricted to one part (`exact_free_topology.md`).

Note on the CLI before this branch: `--v0-threads` only sized the thread pool.  The CLI never called
`Run::set_num_parts`, so every CLI run used a single part whatever the thread count [P, `core/cmdline.cpp`].  The
default pathogen workflow is therefore untouched by anything in this document.

## 2. The scheme  [P]

Let tips(v) be the number of samples below node v.  For integers 2 <= a <= m, the **threshold partition**
C_{a,m}(tau) of a topology tau has as cut points the non-root nodes v with a <= tips(v) <= m whose parent has more
than m tips (`threshold_partition_stencil`, `core/tree_partitioning.h`).  The parts are the subtrees below the cut
points and the root part (everything else, with the cut points as frozen tips).

Before every batch, `Run::repartition` draws an auxiliary variable u independently of the state, from `bitgen_` only:

* with probability p (`single_part_probability`, default 0.02): u = "single part", no cut points, no constraint;
* otherwise an integer m uniform on [ceil(f_lo T), floor(f_hi T)] with T = tips / num_parts (at least 2), and
  a = max(2, ceil(m / r)); defaults f_lo = 0.5, f_hi = 1.0, r = 4.

The batch then uses the partition C_{a,m}(tau(x)).

**Lemma 1 (the parts below the cut points cannot change the partition).**  In the part of a cut point c, every node
has at most tips(c) <= m samples below it, and so does its parent unless the parent is c's parent, which lies in the
root part.  Hence no node of the part can satisfy "parent has more than m tips", none can become a cut point, and the
moves of the part change neither tips(c) (its tip set is fixed) nor c's parent.  [P]

**Lemma 2 (the root part is constrained to keep it).**  A topology move of the root part changes tips() only on the
two paths from the moved parent P and from the node S it was taken away from up to the root, and changes parents only
at X, S and the new sibling.  `Subrun::threshold_partition_holds_after_regraft` checks, at every node of those two
paths and at their children, that a cut point (a tip of the part standing for a clade) still has a parent with more
than m tips, and that no other node has a <= tips <= m below a parent with more than m tips.  Every other node keeps
its own and its parent's count, so its status is unchanged.  A proposal that fails the check is rejected, i.e. the
root part targets pi_cell restricted to the set A = {y : C_{a,m}(y) = C_{a,m}(x)}.  [P]  `Run::reassemble`
recomputes the partition from scratch after every batch and `CHECK`s that it is the one drawn, in every build.  [R]

**Proposition (exactness).**  The augmented target is pi(x) q(u) p(xi | x, u), where xi are the Gaussian auxiliary
variables of the very scalable coalescent prior and q(u) does not depend on x.

1. Drawing u, the partition and xi is a Gibbs step of the augmented target.
2. Each local move of a part is a Metropolis-Hastings move for J = pi_cell(x) p(xi | x, u) restricted to A: the
   restriction multiplies the target by an indicator that is 1 at x, so the existing acceptance ratio times 1_A(y) is
   the MH ratio of the restricted target.  The proposal densities are the existing ones, evaluated on the part's tree,
   which is the same tree at x and at y because the partition is (Lemmas 1, 2).
3. Within a part, every sub-iteration draws its move type afresh, so the n_p sub-iterations of part p form K_p^{n_p}
   with K_p reversible; n_p is proportional to the part size, a function of the partition only
   (`Run::run_threshold_partition_local_moves`).  Parts act on disjoint variables given xi and the frozen cut-point
   times, so the batch kernel is reversible with respect to J on A (Task L, A1 and A4).
4. The whole-batch MH ratio is pi(y) q(u) K(y -> x) / (pi(x) q(u) K(x -> y)) = exp(Delta(y) - Delta(x)): q(u) cancels
   exactly because it does not depend on the state.  This is the formula of the task description with
   q(z | y) / q(z | x) = 1.
5. A rejection restores the genealogy and calls `repartition`, which draws a fresh u: another Gibbs step.
6. Global moves change no topology and target the exact density as before.

**Irreducibility.**  With p > 0 every cycle uses, with probability p, the single-part kernel of
`exact_free_topology.md`, which is irreducible on its own; a mixture of pi-invariant kernels that contains
an irreducible one is irreducible.  How fast the partitioned kernel mixes is an empirical question (Section 5).

**Why not the probability ratio of a state-dependent q(z | x).**  A cut point needs a clade of about N / K tips, and a
tree has only O(K) such nodes, so any rule that chooses cut points from the current sizes has a q(z | x) that changes
by factors of order one under a single SPR in the root part.  Augmenting with the threshold m instead of the cut set
makes q state independent; the price is that some root-part proposals are rejected locally (they would change the
partition), which is cheaper than rejecting a whole batch.

## 3. Load balance  [R, simulation of the rule only]

On Kingman topologies (same-date tips; `balance.py` in the session scratchpad, not part of the build), with the
default f = [0.5, 1.0] and r = 4, the ideal-to-actual ratio of the largest thread's node count after greedy packing is
0.88 - 0.96 for N = 1,000 - 10,000 and K = 2 - 8, and 0.74 - 0.84 at K = 16, with about 2 - 2.3 parts per thread.
A ratio r = 8 balances better (about 0.95 at K = 16) but doubles the number of parts, and every part holds a copy of
the reference sequence (8 MB on chrY), so the default favours fewer parts.  N = 100 cannot be split usefully beyond
K = 4.

## 4. Validation

### 4.1 Unit and stationarity tests  [R]

`tests/exact_free_topology_tests.cpp`, section 5 (`Exact_threshold_partition_test`), Release build:

| Test | What it checks |
|---|---|
| `the_cut_points_are_the_maximal_clades_in_the_window` | `threshold_partition_stencil` on balanced, caterpillar and mixed shapes; every inner node in exactly one part |
| `free_topologies_on_several_parts_are_accepted_and_keep_their_partition` | accepted in either order; the root-part constraint is exercised; refusals (stencils without the mode, calibrations on a free topology in any order, invalid parameters) |
| `the_coalescent_term_stays_exact_through_parallel_batches_and_rollbacks` | 3 parts, 30 heterochronous tips, two Skygrid types: after each of 120 batches the coalescent term equals the independently computed exact density to 1e-8 and a single-part Run rebuilt from the state agrees on log G, the coalescent and other priors; rollbacks and >20 topologies seen |
| `the_number_of_threads_does_not_change_the_chain` | 4 parts on pools of 1, 2 and 5 threads give bit-identical chains |
| `eight_tips_same_date_constant_population_two_parts`, `sixteen_tips_at_several_dates_four_parts`, `twelve_tips_skygrid_with_the_curve_sampled_three_parts` | chains started from exact Kingman draws (independent forward simulation) stay on the exact target: chi-squared on root split, number of cherries and one fixed cherry; z on root age, youngest coalescence and tree length; gate p = 1e-4 |

Results of the stationarity tests (2,000 / 1,500 chains, 30 batches each):

| Case | chi2 root split | chi2 cherries | chi2 S0S1 | z root | z first | z length | batch acc. | mean parts |
|---|---|---|---|---|---|---|---|---|
| 8 tips, constant, 2 parts | 0.5 / 3 | 1.0 / 3 | 0.0 / 1 | +0.18 | -0.42 | +0.45 | 0.876 | 3.54 |
| same, cell prior (power) | 0.7 / 3 | 7.0 / 3 | 1.4 / 1 | **-9.37** | **+13.00** | **-5.06** | 1 | 3.52 |
| 16 tips, 5 dates, 4 parts | 10.7 / 7 | 7.5 / 6 | 1.0 / 1 | +0.08 | -0.36 | -0.44 | 0.829 | 5.68 |
| 12 tips, staircase Skygrid, 3 parts | 3.2 / 5 | 8.0 / 5 | 2.2 / 1 | -1.07 | +0.30 | -1.18 | 0.843 | 4.31 |
| 12 tips, log-linear Skygrid, 3 parts | 2.0 / 5 | 1.3 / 5 | 0.4 / 1 | -0.32 | +0.15 | -0.45 | 0.863 | 4.30 |

**Negative control** (temporary patch, not committed: root-part constraint and the reassembly check disabled, so the
partition can change during a batch): 8 tips cherries chi2 31.3 / 3 and root split 12.6 / 3 (FAIL), 12 tips staircase
cherries 27.3 / 5 (FAIL), log-linear root split 16.2 / 5; 16 tips 24.4 / 7 and 25.8 / 6 (below the strict gate).  The
node-age statistics do not see it; the topology statistics do.  So the constraint is necessary and the tests detect
its absence.

Added later: `parallel_batches_with_mutation_histories_keep_every_derived_quantity` (48 tips, 300 sites, random
mutation histories, 4 parts on 4 threads: after each of 60 batches a single-part Run rebuilt from the state agrees on
log G and the coalescent term; SPR moves accepted, rollbacks exercised).  Full Release suite: 799 / 799 before that
test was added; the threshold and exact free-topology tests (13) pass on the committed code.

### 4.2 Data races  [R]

ThreadSanitizer, Release flags plus `-fsanitize=thread -g -O1`, run under `setarch -R` (WSL2 otherwise aborts with
"unexpected memory mapping"): `free_topologies_...`, `the_coalescent_term_stays_exact_...`,
`the_number_of_threads_...`, `parallel_batches_with_mutation_histories_...` and `eight_tips_...` (pools of 2 to 5
threads): 0 warnings.  The parts share only immutable tables (`Global_evo_model`, cumulative site rates), each has its
own generator and coalescent part, and the pool results are joined before `reassemble`.

### 4.3 Byte identity with the option off  [R, campaign cluster]

Base `f765bd7` (Task 4 final5) against this branch, same seeds: pathogen regression 14 / 14 identical (.log and
.trees; 1 and 4 threads, both initializations, three Skygrid configurations); human mode F1 at N = 100 (8 Mb), H21,
mtDNA backbone and the cell prior with `--v0-threads 4` identical.

## 5. Real chrY, 1,043 tips, local machine  [R]

Data: the user's chrY set (66,315 variable columns plus 8,818,561 constant sites, GTR+FO, fixed rate 8.333e-10,
Skygrid 100 knots, grid pinned at 20 years) with `--v0-exact-coalescent-correction`, free topology, seed 81; one full
batch of 104,250 local moves per chunk (`--v0-log-every 104250`).  Host: WSL2, 24 logical cores, runs one at a time,
each pinned with `taskset`.  50 batches; the MCMC phase is timed from the `Step 0` line to the last `Step` line.

| Configuration | s / batch | vs single part |
|---|---|---|
| single part, 1 thread (existing path) | 0.651 | 1 |
| threshold, default parameters, 2 / 4 / 8 / 16 threads | 0.229 / 0.282 / 0.212 / 0.199 | 2.8 / 2.3 / 3.1 / 3.3 |
| threshold, 2 / 4 / 16 parts on **one** core | 0.427 / 0.412 / 0.322 | 1.5 / 1.6 / 2.0 |
| threshold, a,b,r = 0.25,0.5,8, 4 / 8 / 16 threads (22 / 41 / 81 parts) | 0.213 / 0.175 / 0.187 | 3.1 / 3.7 / 3.5 |
| threshold, 0.1,0.3,16, 4 / 8 / 16 threads (55 / 113 / 194 parts) | 0.182 / 0.175 / 0.243 | 3.6 / 3.7 / 2.7 |

Batch acceptance 99% everywhere.  Peak RSS 505 MB single part; 595 MB (4 threads, default) up to 2.9 GB (194 parts):
every part holds its own copy of the 8.9 Mb reference.

Reading:

* Much of the gain is not parallelism.  The same partitioned kernel on one core is already 1.5 - 2x faster per batch,
  because a mutation-guided SPR study is confined to its part (gprof: `Spr_study_builder::visit_cur_region` 94 M calls
  per 20 batches with one part, 16 M with four).  Whether those cheaper, more local proposals mix as well per step is
  not measured yet; speed per step is not ESS per hour.
* The parallel gain itself is 1.5x at most.  Per batch (instrumented build, not committed): serial work about 0.09 s
  (global moves 0.03, pushing the global parameters 0.03, repartition 0.02 - 0.04, root normalization 0.004), and the
  root part, which holds the deep, mutation-dense branches and costs 0.12 - 0.16 s while every other part takes
  0.01 - 0.04 s (4 threads, default parameters, 8 parts of which the root part has about 500 of 2,085 nodes).
* So beyond 2 - 4 threads nothing is gained on this data set until the root part is split further and the serial
  per-batch work shrinks.

### 5.1 ESS per hour, short chains  [R]

Same data and host; two seeds (81, 82) per configuration, all six chains at the same time on disjoint cores (14 of
24), so every chain is about 30% slower than when run alone.  About 0.24 - 0.34 h of MCMC per chain, first 20% of the
log rows discarded, Geyer ESS.  These chains are short (ESS 5 - 200) and two seeds per configuration are not enough to
claim convergence; read the numbers as a first indication only.

| ESS per wall hour (mean of two seeds) | posterior | likelihood | rootHeight | treeLength | numMuts | Skygrid precision | logPopSize median / min |
|---|---|---|---|---|---|---|---|
| single part, 1 thread (1,400 batches) | 131 | 123 | 402 | 106 | 460 | 135 | 250 / 23 |
| threshold, 2 threads, default (3,900 batches) | 146 | 143 | 288 | 136 | 139 | 386 | 693 / 42 |
| threshold, 4 threads, 0.25,0.5,8 (4,200 batches) | 149 | 146 | 342 | 111 | 92 | 251 | 859 / 52 |

Batch acceptance 99.2 - 99.5%.  Post-burn-in means: rootHeight 175,970 / 176,404 (single part), 175,700 / 175,362
(2 threads), 175,765 / 176,268 (4 threads); treeLength 9.154e6 / 9.151e6, 9.135e6 / 9.133e6, 9.141e6 / 9.143e6.

Reading: the partitioned chains run 2.8 - 3x more batches per hour, but each batch mixes the genealogy about 2 - 3x
less (ESS per batch of the posterior and tree length), so ESS per hour of the tree quantities is about the same
(0.9 - 1.3x), worse for numMuts and rootHeight in these runs, and 2 - 3x better only for the Skygrid parameters,
which gain from the larger number of global steps.  The likely cause is the SPR proposals confined to the parts.  The
small treeLength gap between configurations (about 0.2%) is not resolved by chains this short.

### 5.2 Plans A and B: more full-tree topology moves  [R]

Same data and host, two seeds (81, 82) per configuration, 2 threads each, all chains of one plan at the same time
(20 and 16 of 24 cores), about 15 minutes of MCMC per chain.  `--v0-threshold-partition-params 0.5,1.0,4,p`; plan B
adds `--v0-threshold-global-spr G` (G full-tree SPR proposals after every parallel batch, with their own whole-batch
correction).  ASDSF: mean over clades with frequency > 0.1 in either chain and 0.1 - 0.9 on average, of
|f_81 - f_82| / sqrt(2), post burn-in.

| Configuration | s / batch | ESS/h posterior | ESS/h treeLength | ESS/h rootHeight | ASDSF (uncertain) | max |
|---|---|---|---|---|---|---|
| A p = 0.02 (= B G = 0) | 0.37 - 0.40 | 146 - 158 | 129 - 139 | 181 - 196 | 0.048 | 0.707 |
| A p = 0.10 | 0.45 | 115 | 109 | 290 | 0.090 | 0.707 |
| A p = 0.25 | 0.55 | 100 | 91 | 172 | 0.095 | 0.707 |
| A p = 0.50 | 0.71 | 59 | 59 | 200 | 0.079 | 0.707 |
| A p = 1.00 | 1.02 | 28 | 23 | 266 | 0.086 | 0.707 |
| B G = 300 | 0.45 | 102 | 97 | 310 | 0.074 | 0.707 |
| B G = 1,000 | 0.58 | 78 | 71 | 865 | 0.063 | 0.707 |
| B G = 3,000 | 0.91 | 47 | 54 | 244 | 0.066 | 0.707 |

(The G = 0 chains are bit-identical to the p = 0.02 chains; the two rows differ only in the concurrent load.)
Batch acceptance 99.1 - 99.8%, also for the full-tree SPR batches.  Accepted topology changes per second of MCMC:
about 1,980 (G = 0) and 1,930 (G = 1,000).  Of the topology changes made by full-tree SPR proposals, 0.1% crossed a
boundary of the preceding partition (G = 1,000: 342 of 298,482; G = 3,000: 594 of 604,975).

Reading: neither plan raises ESS per hour of the tree quantities; both cost time and lower it roughly in proportion.
The mutation-guided SPR is local (99% of its proposals stop after one mutation), so even on the whole tree it almost
never regrafts across a partition boundary: the boundaries are not what limits topology mixing here.  Every
configuration, including the single-part one, has at least one clade with frequency near 1 in one seed and near 0 in
the other (max 0.707): the chains sit in different topological modes that none of the existing moves crosses within
15 minutes.  The ESS estimates of chains this short scatter by up to 4x between runs of the same configuration.

### 5.3 Outcome of plans A and B

Plan A needed no code: the single-part probability p is part of the scheme of Section 2 and stays at its default of
0.02.  The full-tree SPR batches of plan B (`--v0-threshold-global-spr`, commit `92bf0ec`) were removed again by
reverting that commit; their results above are kept as a negative result.  Performance and topology-mixing work on
this branch is stopped.

## 6. Status and limits

Stable: `--v0-threshold-partition` with `--v0-exact-coalescent-correction`, a free topology, the exponential,
constant or Skygrid population model and K threads.  Without the option every output is unchanged.

Not done or not supported:

* Age calibrations with a free topology on several parts (refused).
* Full SBC, long-chain convergence and ESS per hour with enough chains; the chrY numbers of Section 5 come from
  short chains and two seeds.
* Thread scaling at N = 100 / 1,000 / 4,000 on the campaign cluster (`perf.sh`) and the posterior comparison with data (`post.sh`);
  the drivers are in `tests/human_input/exact_multithread/` but were not run.
* On the real chrY set the gain in ESS per hour of the tree quantities is small (Sections 5.1 and 5.2); the
  topology mixing is limited by modes that the existing local SPR moves rarely cross, with or without parts.
