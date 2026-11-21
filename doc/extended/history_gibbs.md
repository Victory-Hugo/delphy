# Single-site mutational-history Gibbs updates

The design, the mathematics and the correctness argument of three new MCMC operators that
update Delphy's explicit mutational history one site at a time.  The experimental results are in
`40_history_gibbs_report.md`.

Delphy's architecture is unchanged: the explicit mutation-annotated tree, the sparse per-branch
mutation lists, the incremental likelihood, the mutation-guided topology proposals and the
Run/Subrun framework all stay exactly as they are, and the production likelihood engine is not
replaced.  What is added is a new way of moving the history that the existing engine already scores.

## 1. What the existing sampler does, from the source

With a fixed topology, the only move that can change the **number** of mutations in the tree is the
subtree slide, `Subrun::subtree_slide_move` (`core/subrun.cpp`), weight 1 out of 31 in
`Subrun::mcmc_sub_iteration`.  It resamples the node height and then rebuilds the mutational history
of the paths around the slid node through `Subrun::spr_move_core` and `Spr_move`.  With a free
topology the SPR move, `Subrun::spr1_move`, adds one more such opportunity in 32.  The other local
moves cannot change the event count at all: `Subrun::branch_reform_move` only randomizes the
**times** of the mutations already on a branch through `randomize_branch_mutation_times`
(`core/phylo_tree.cpp`), and `Subrun::inner_node_displace_move` and `Subrun::tip_displace_move` only
move node ages, bounded by the existing mutation times.

The history itself is proposed by `sample_mutational_history` (`core/spr_move.cpp`).  For a closed
path of duration `T` carrying `L` hot sites it draws, **for every site at once**, a Jukes-Cantor
trajectory at a single average rate `mu_JC`: endpoint-conditioned rejection sampling at the sites
that carry a delta, and a geometric skip over the sites that do not.  Its log density is assembled in
`Spr_move::propose_new_graft` as

```
log_alpha_mut = -mu_JC L T + M log(mu_JC / 3) - [ (L - d) log P_AA^JC(T) + d log P_AC^JC(T) ]
```

summed over the branch infos of the graft, and the Metropolis-Hastings ratio corrects the whole
proposal, over every site of the path, in one accept-or-reject decision.

That single decision is the problem.  The proposal is a Jukes-Cantor process at an average rate; the
target is the real model with its own exchangeabilities, frequencies, partition rates and site rates.
The importance weight is therefore a **product over sites**, and the variance of its logarithm grows
with the number of sites that can carry a hidden substitution, which is `L` times the per-site
divergence -- that is, with the total number of substitution events in the tree.  At pathogen-scale
divergence almost every site's proposed history is its parsimony history, the weight is near one, and
the move is accepted about half the time; this is the regime Delphy was built for.  At 0.72
substitutions per site the weight degenerates.  Task 3.5 measured it
(`38_substitution_model_report.md`, section 5.4, point 5): the subtree slide's acceptance is 4.24% at
20 sites, 0.02% at 100, 0.0009% at 500 and 0.0003% at 2,000, against 45.6% on the pathogen alignment
of the byte-level regression.  From 500 sites upwards the mutational history changes one to three
times in an entire run of a thousand million steps, and the reported posterior is the conditional
posterior given a frozen, nearly parsimonious history.

Two further things scale against the sampler, both recorded in the same section: the slide's proposal
width is about half the waiting time to the next mutation anywhere in the genome, so it shrinks as the
mutation density grows; and moving to a longer tree requires adding hidden events first, because the
exposure term penalizes long branches that carry no events, while no single existing move changes the
tree length and the event count together.

Three kinds of mixing have to be kept apart, and the measurements of Task 3.5 and of this task are
reported separately for each:

* **A**, the conditional mixing of the explicit history at a fixed tree and fixed parameters;
* **B**, the joint mixing of the substitution rates and the explicit history at a fixed tree;
* **C**, the joint mixing of the topology, the node times and the history.

The operators below address A directly and B through A.  Whether they are enough for C is an
experimental question, reported in `40_history_gibbs_report.md`.

## 2. What the literature does, and what is different here

BEAST's `TreeLikelihood`
(`src/beast/base/evolution/likelihood/TreeLikelihood.java`) never represents a mutational history at
all.  It computes the marginal sequence likelihood by Felsenstein pruning, integrating over every
internal state at every site, and its MCMC state contains only the tree and the parameters.  There is
therefore no history to mix, and no acceptance rate that can collapse with sequence length -- at the
price of a global likelihood recomputation whenever anything changes, which is exactly the cost
Delphy's explicit history and local incremental calculation were designed to avoid.  The comparison
is not "which likelihood is right": both target the same marginal posterior of the tree and the
parameters.  It is a comparison of state spaces, and Delphy's augmented state space needs an operator
for the augmented coordinate.

Stochastic mapping supplies that operator.  Nielsen (2002), *Mapping mutations on phylogenies*, draws
substitution histories conditional on a fixed tree, fixed parameters and the observed tip states, by
drawing the internal states from the pruning posterior and then an endpoint-conditioned path on every
branch; in that paper the histories are the **output** of an analysis, drawn once the chain has
finished.  Hobolth and Stone (2009), *Simulation from endpoint-conditioned continuous-time Markov
chains* (doi:10.1214/09-AOAS247), give the exact ways to draw that endpoint-conditioned path -- direct
forward simulation with rejection, modified rejection sampling, uniformization, and direct sampling
from the conditional distribution of the jump times -- and compare their cost.  Irvahn and Minin
(2014), *Phylogenetic stochastic mapping without matrix exponentiation*
(doi:10.1089/cmb.2014.0062), observe that uniformization needs only powers of a stochastic matrix, so
a stochastic mapping algorithm can be built without ever forming a matrix exponential or an
eigendecomposition.

What this task does is to turn that construction inside out: the same conditional draw becomes a
**Gibbs kernel inside the chain** rather than a post-hoc summary of it.  Because Delphy's augmented
posterior contains the history explicitly, and because the conditional of one site's history given
everything else is exactly the endpoint-conditioned CTMC, the draw is a Gibbs step and its acceptance
probability is one by construction.  Nothing of BEAST's or anyone else's likelihood engine is
imported; the target density is the one Delphy already computes, and the only new arithmetic is the
conditional draw.

## 3. The augmented posterior, and why one site at a time is exact

Delphy's posterior weight of a genealogy is `calc_log_root_prior` plus `calc_log_G_below_root`
(`core/phylo_tree_calc.cpp`), together with the coalescent prior, the calibration priors and the
parameter priors.  Write the effective rate matrix of site `l` as

```
R^(l) = mu_l nu_l Q^(beta(l)),
```

with `mu_l` the rate of the site's partition, `nu_l` its relative site rate and `Q^(beta)` the
partition's normalized reversible matrix.  Then `calc_branch_log_G` is, site by site,

```
log p(history of site l on branch [t_P, t_X]) =
    - sum over segments  R^(l)_escape(state) * (segment duration)
    + sum over events    log R^(l)_{from,to},
```

which is precisely the path measure of a continuous-time Markov chain with rate matrix `R^(l)`, and
`calc_log_root_prior` is `prod_l pi^(beta(l))(root state of l)`, the stationary distribution of the
same chain, with the sites missing at the root left out.  Two facts follow and are the whole basis of
what comes next.

1. **Sites factorize.**  The augmented density is a product over sites, so conditioning on the
   histories of every other site leaves the chosen site's conditional untouched.
2. **Nothing else in the posterior sees the history.**  The coalescent prior and the calibrations
   depend only on node times; the parameter priors only on the parameters.  Conditionally on the
   parameters, changing a site's history changes only its own factor of `log G`.

Therefore the full conditional of the history of one site on one branch, given the states the site
has at the two ends of that branch, is **exactly** the endpoint-conditioned CTMC of rate matrix
`R^(l)` on that branch, and drawing from it is a Gibbs step.  The same argument, with the root prior
in place of a parent branch where the node is the root, gives the conditional of a node's state, and
Felsenstein pruning for the one site gives the conditional of all its states at once.

An important consequence of the factorization: an operator that works one site at a time has **no
importance weight at all**, so there is nothing whose variance can grow with `L`.  Its cost grows
with `L` only through how many sites one has to visit to sweep the genome, which is linear, not
exponential in disguise.

## 4. The CTMC bridge (`core/ctmc_bridge.h`)

For a rate matrix `Q`, a duration `tau` and endpoint states `a` and `b`, we must draw from
`P(H | X(0) = a, X(tau) = b, Q)`, where `H` is the number of substitutions, their times and the states
passed through.  The implementation uses uniformization, which is exact and needs no matrix
exponential.

With `Omega = max_a (-Q_aa)` and `R = I + Q/Omega`, `R` is a stochastic matrix (its off-diagonal
entries are `Q_ab/Omega >= 0`, its diagonal is `1 + Q_aa/Omega >= 0`, its rows sum to one), and the
CTMC is the discrete chain `R` observed at the events of a Poisson process of rate `Omega`, some of
those events being self-jumps of `R`.  Conditioning on the endpoints,

```
P(K = k | a, b, tau) = Pois(k; Omega tau) (R^k)_ab / [exp(Q tau)]_ab.
```

The sampler

1. accumulates `w_k = Pois(k; Omega tau) (R^k)_ab` by streaming the forward row vector
   `e_a^T R^k`, which also gives `[exp(Q tau)]_ab = sum_k w_k`, so the normalization is consistent
   with the weights to the last bit and no separately computed matrix exponential can disagree with
   them;
2. draws `K` from the normalized weights;
3. draws the `K - 1` intermediate states as a discrete-time Markov bridge of `R`, using
   `P(X_i = s | X_{i-1} = r, X_K = b) ∝ R_rs (R^{K-i})_sb` with the backward column vectors
   `R^j e_b`;
4. draws the `K` jump times as the order statistics of `K` independent uniforms on the **open**
   interval, so every event is strictly inside its branch;
5. drops the self-jumps of `R`, leaving a path drawn exactly from the conditional law of the CTMC.

The transition probabilities needed by the node-state update and by the pruning are the same series
summed over the whole matrix, `exp(Q tau) = sum_k Pois(k; Omega tau) R^k`, so the whole module uses
one kind of arithmetic.

**The tail of `K`.**  The conditional distribution of `K` has mass above every finite bound, and
truncating it at a fixed index would bias the draw.  The series is instead carried until the Poisson
mass it has left to distribute can no longer change the accumulated quantity at a relative tolerance
of `1e-13`, or until that remaining mass, or the individual terms, have underflowed to zero in double
precision.  What is dropped is bounded by the remaining Poisson mass divided by the transition
probability, and that quotient is what the test holds below the tolerance -- so the bound is on the
**conditional** mass dropped, not on the unconditional Poisson mass.  A hard ceiling of `2^22`
uniformized jumps exists only so that a nonsensical rate or duration raises instead of exhausting
memory; it is far above where the terms underflow.

**Scaling.**  The effective matrix of a site is a scalar multiple of its partition's matrix, so one
sampler per partition serves every site whatever its relative rate: the duration and the rate scale
enter only through `tau = rate_scale * T`, while the jump times are uniform on the branch itself.
That is why continuous Gamma site rates cost nothing extra here.

**Degenerate cases.**  A zero duration or a zero rate admits only the empty path, which exists only
between equal endpoint states; a request for unequal states then raises, as does a request for a
transition no sequence of allowed substitutions can make (two communicating classes), a negative or
non-finite duration or rate scale, and a matrix that is not a rate matrix.

## 5. The three operators (`core/history_gibbs.h`)

### C1, the branch-history update: `gibbs_resample_branch_site_history`

Holding the tree, the parameters and the states of site `l` at both ends of the branch ending at `X`,
redraw the history of site `l` on that branch from the conditional bridge.  This is the update that
can create a hidden pair of substitutions on a branch whose endpoints agree, delete a redundant pair,
produce back substitutions and multiple hits, and move substitution times, all with acceptance
probability one.

Only the branch's own contribution to `log G` changes.  The states at both ends are unchanged, so
`lambda_i` is unchanged at `X` and everywhere below it; the intermediate states inside the branch
affect only the exposure integral, which is what the returned `delta_log_G` carries.  It is an
increment for bookkeeping, not an acceptance ratio.

This operator updates the history at **fixed endpoint states**.  It is not, and must not be called,
a complete mixing of the ancestral states; that is what C2 and C3 are for.

### C2, the node-state update: `gibbs_resample_node_site_state`

For one site `l` and one inner node `V`, hold everything outside the branches incident on `V` fixed,
and draw jointly the state of `l` at `V` and the histories of those branches.  Integrating the
histories out leaves

```
P(s_V = s) ∝ [ pi^(beta(l))(s) if V is the root else P_{parent,V}(s_parent, s) ]
            * prod over children C with the site present  P_{V,C}(s, s_C),
```

after which the incident histories are independent conditional bridges.  Both steps are exact, so the
pair is a Gibbs step on the block.

Three pieces of bookkeeping make it consistent with Delphy's derived quantities.

* `lambda_i` changes **only at `V`**, by `mu_l nu_l (q(s_new) - q(s_old))`.  At a child whose state is
  held fixed, the change in `lambda_i[V]` is cancelled exactly by the change in the first site-`l`
  mutation of the child's branch.
* A child `C` below which the site is **missing** carries no history for it, and its branch
  contributes neither a factor to the conditional nor a term to `log G`: Delphy removes the site's
  whole exposure on such a branch through `calc_delta_lambda_across_missations`.  What does have to
  follow the new state at `V` is the `from_state` of the missation, which is the state entering the
  missing region; updating it is what keeps `lambda_i` at and below `C` unchanged.  The missing state
  is never completed to the reference base, and never enters the root prior, because
  `calc_log_root_prior` removes the missing sites from the root state counts.
* At the root, the new state is recorded as a mutation above the root, at `-DBL_MAX`, which is where
  and when Delphy already keeps the difference between the reference sequence and the root sequence
  (`Run::repartition`), and `Run::normalize_root` folds it into the reference sequence at the next
  opportunity.  The root is eligible **only in the part that contains the root of the whole tree**:
  the cut point that joins one part to another has to keep the sequence the other part was built
  from, and `Subrun::calc_cur_log_G` only carries the root prior in the part that owns the root.

### C3, the whole-site refresh: `gibbs_refresh_site_history`

For one site, run Felsenstein pruning over the whole tree, draw the root state from the root prior
times the root partial likelihood, draw every internal state from the root down, and draw an exact
conditional bridge on every branch.  A tip's state is data and is never redrawn; a subtree below which
the site is missing is skipped, and the `from_state` of its missation follows its parent's new state.
The partial likelihoods are rescaled at every node, which is harmless because only their ratios are
used, and which is what keeps the pruning from underflowing on a deep tree.

This is one `O(N)` pass for the one site, with one 4x4 transition matrix per branch.  No probability
matrix of the size of the genome times the tree is ever formed.  It is the update that mixes the root
state and every internal state at once, and it is also the small-scale correctness reference for C1
and C2, because it reaches the same conditional by a completely different route.

It runs on the whole tree, so it belongs between batches of local moves, at the end of
`Run::run_global_moves`, and the tree parts are rebuilt from the changed tree afterwards.

## 6. Site selection, and why it needs no correction

All three operators draw the branch or node uniformly and the sites uniformly and independently of the
current state, so each is a random-scan Gibbs sampler with **fixed** selection weights, and the
selection contributes nothing to the acceptance ratio.  This is deliberate: a selection rule that
looked at the current number of events at a site, or at which branches currently carry mutations,
would make the kernel state-dependent and would need its own detailed-balance or
Metropolis-Hastings correction rederived, and would no longer be an unconditionally accepted Gibbs
step.  No such rule is used.

When the chosen site turns out to be missing at the chosen place, the update is the identity.  That is
not an approximation: the Gibbs kernel for an empty block **is** the identity, and a random scan over
blocks, some of which are empty in the current state, is still reversible with respect to the target.
The diagnostics report how often this happens, as the fraction of updates in which the site was
present, so that a run on data with much missingness is not silently doing nothing.

## 6b. Phase F: a node-time move with the histories integrated out

Once the history of a site can move, the binding constraint on the tree's time scale is a different
one, and it is visible in the source.  `Subrun::inner_node_displace_move` takes

```
t_min = max(parent time, latest mutation on the branch above the node)
t_max = min(children's times, earliest mutation on either child branch)
```

so the node's new time is **hard-bounded by the mutation times on the incident branches**: the node
can cross at most one mutation gap per step, whatever the proposal width.  That gap is about one over
the total mutation rate at the node, so it shrinks as the sequence gets longer or more divergent.  At
2,000 sites and 0.72 substitutions per site it is about 1.4 years, against a root-age posterior some
670 years wide; a random walk with that step needs on the order of 10^6 accepted moves per independent
sample, which is what the measured effective sample size of the root age reflects.  This also explains
why Task 3.5 saw no effect from raising `--v0-root-displace-scale-factor` by factors of 10 and 100:
the binding constraint is the **support** of the proposal, not its width.

The move that removes the constraint integrates the histories of the incident branches out, proposes
the node time against the local marginal likelihood alone, and draws the histories back from their
exact conditional at the new time.  Write

```
F(t) = prod over sites l of  sum over s  w_l(s) prod over present children c  P^{V,c}_l(s, s_c),
```

with `w_l(s)` the transition across the parent branch from the state at the parent, or the root prior
when `V` is the root, and `s_c` the state of the site at the far end of each child branch.  `F(t)` is
the augmented density of the three branches with every history and the state at `V` summed out.  Then
for a proposal `q(t -> t')` and a draw of the new history from `rho(H' | t') / F(t')`, the
Metropolis-Hastings ratio is

```
  pi(t', H') q(t' -> t) rho(H | t)/F(t)       p_coal(t') F(t') q(t' -> t)
  ------------------------------------- = ---------------------------------
  pi(t,  H ) q(t -> t') rho(H'| t')/F(t')      p_coal(t) F(t) q(t -> t')
```

because `rho(H' | t')` appears once in the target and once in the proposal and cancels, and likewise
for `rho(H | t)`.  **Neither history survives in the ratio**, which is exactly the cancellation the
move is built around: the decision is taken before the new history is drawn, and the step is limited
only by the times of the surrounding nodes.  The calibration prior joins `p_coal`, and the Jacobian of
the proposal's parametrization joins `q`.

The proposal is a plain uniform draw on `(t_parent, min children)` for an inner node, which is
symmetric on an interval the move does not change.  For the root, whose support is unbounded below,
the depth of the root below its oldest child is multiplied by `exp(u)` with `u` uniform on
`[-w, +w]`; that proposal is scale free, so it crosses a deep root's posterior in a few steps whatever
the mutation density, and its Jacobian contributes `u` to the log ratio.

On acceptance the node is moved, `gibbs_refresh_node_histories` draws the state of every site at the
node from its full conditional and then every incident history as a bridge, and the three branches'
contribution to `log G`, together with `lambda_i` at the node, is recomputed explicitly -- the old
histories cannot be scored against the new branch lengths, so an incremental update per site is not
available here.  The cost is one pass over the sites per proposal, and one matrix per group of sites
that share a partition and a relative rate, so it is cheap without site-rate heterogeneity and
proportional to the number of sites with it.

The move is `Move_kind::node_time_marginal`, weight zero by default, enabled by
`--v0-marginal-node-time-move` with `--v0-marginal-node-time-weight` and
`--v0-marginal-node-time-window`.  It needs the history updates to be of any use: on its own it still
has to put a history back, which it does correctly, but the rest of the sampler would then be unable
to move it.

## 7. Integration, defaults and compatibility

Two new local move kinds, `Move_kind::branch_history_gibbs` and `Move_kind::node_state_gibbs`, join
the random scan of `Subrun::mcmc_sub_iteration` with a shared weight of `history_gibbs_weight`, half
each.  **The default weight is zero.**  At zero the total weight of the scan, and therefore the draw
taken from the random stream, is bit for bit what it was, which is why the byte-level pathogen
regression is unaffected; the whole-site refresh likewise returns before touching the state or the
stream when its count is zero.  Nothing about the default pathogen workflow, its move schedule or its
output changes.

The command line, all of it requiring `--v0-human-mode` and all of it off by default:

| Option | Meaning |
|---|---|
| `--v0-history-gibbs` | enable C1 and C2 |
| `--v0-history-gibbs-weight W` | their shared weight in the local mix (default 15, against 7.5 for each node-time move, 15 for the branch reform and 1 each for the slide and the SPR) |
| `--v0-history-gibbs-sites N` | sites visited by one update (default 1) |
| `--v0-site-history-refreshes N` | sites refreshed by C3 in every global step (default 0, off) |
| `--v0-marginal-node-time-move` | enable the node-time move with the histories integrated out (section 6b) |
| `--v0-marginal-node-time-weight W` | its weight in the local mix (default 7.5, the same as each existing node-time move) |
| `--v0-marginal-node-time-window W` | half width of the uniform proposal of the log of the root's depth (default 0.5) |

Supported from the start, because the operators are written against `Global_evo_model` rather than
against any particular model: JC69, HKY85, TN93, GTR and every other grouping of Task 3.5, estimated
and fixed frequencies, the site partitions and relative rates of Task 3.4, the continuous Gamma site
rates, the lossless input modes and observation semantics of Task 3.2, fixed and free topology, the
strict clock, the age calibrations, and Run/Subrun with any number of threads.  The reason the list is
that long is that none of these features enters the conditional draw except through `R^(l)` and the
root prior, which the operators read from the model rather than assume.

What is **not** supported, and is recorded as such: nothing here develops FreeRate, which Task 3.5
left unimplemented; and C2 cannot resample the state at the root of a part that is not the root of the
whole tree, so with more than one part the sequence at the cut points mixes through C3 and through the
existing moves rather than through C2.

## 8. Diagnostics

A Gibbs step has no acceptance rate to report, so `History_gibbs_stats` counts what actually moved:
how many updates were proposed, in how many the chosen site was present, how often the node state
changed, how many whole-site refreshes ran, and how many substitution events were removed and
created.  `delphy` prints them at the end of a run next to the existing local-move diagnostics.  The
number that replaces an acceptance rate as the thing to watch is the rate at which node states change
and the number of substitution events moved per unit of CPU time; both are in the report.

## 9. Independent tests

`tests/ctmc_bridge_tests.cpp` checks the bridge against three references that share no code with it:
a scaled and squared long-double Taylor series for the matrix exponential, forward Gillespie
simulation with rejection on the end state for the whole conditional law (the distribution of the
number of substitutions by a chi-square between the two empirical distributions, the mean time of the
first substitution, and the share of each substitution type), and Simpson quadrature of
`E[N | a, b, tau] = (1/P_ab) sum_{i != j} Q_ij int_0^tau P_ai(s) P_jb(tau - s) ds` for the conditional
expected number of events.  It also covers the degenerate and extreme cases of section 4.

`tests/history_gibbs_tests.cpp` checks the operators on a five-node tree with two sites: that the
`delta_log_G` each update reports is the change in the whole posterior weight recomputed from scratch
to 1e-9; that the tree integrity, the missation consistency and the tip sequences survive every
update; that the other site is never touched; that `lambda_i` changes only at the resampled node and
by exactly the reported amount; that C1 reproduces the conditional bridge against an independent
rejection sampler; that C2's node conditional matches an independent matrix exponential at an inner
node and at the root with the root prior; that C3's joint posterior of the two internal states matches
independent enumeration of the sixteen pairs; that a missing site is left alone and its `from_state`
follows; that hidden pairs of substitutions appear where the endpoints agree; and that a chain of all
three updates recovers the independent conditional expectation of the number of substitutions at a
site.
