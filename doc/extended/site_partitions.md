# Site partitions: named site sets with their own relative substitution rate

This document defines the site-partition model, its parameterization and identifiability, the input
format and its validation, how the partitions interact with the observation models of Task 3.2, and what
the sampler does with them.  Evidence classes: **[P]** proof read from the source, **[N]** independent
numerical check, **[R]** check on the real kernel through the CLI or the sampler, **[M]** measurement,
**[U]** unverified.

## 1. Two different things called "partition"  [P]

| | Site partition (this task) | Run / Subrun tree partition |
|---|---|---|
| Splits | the sites of the sequence | the nodes of the tree |
| Purpose | different evolutionary rates for different sites | parallel MCMC over threads |
| Declared by | `--v0-site-partition-file` | `--v0-threads`, `--v0-local-moves-per-node-per-global-move` |
| Lives in | `Global_evo_model::partition_for_site` and `partition_evo_model` (`core/evo_model.h`) | `Partition` / `Subrun` (`core/tree_partitioning.h`, `core/subrun.h`) |
| Changes the target density | yes | no (it is a computational decomposition) |

They compose: `Run::push_global_params_to_subruns` (`core/run.cpp`) pushes the whole `Global_evo_model`,
including the per-site-partition rates, into every `Subrun`, and every local move reads a site's rate
through `evo.mu_l(site)`, so a tree part that happens to hold sites of several site partitions is handled
correctly with no extra bookkeeping.  Tests cover one and two tree parts with two site partitions.  **[R]**

## 2. The model  [P]

Let the analysed sequence have `L` sites, divided into `P` named, non-overlapping partitions.  Partition
`p` has relative rate `r_p > 0` and the substitution rate

    mu_p = mu * r_p,

while kappa, the state frequencies `pi` and the per-site relative rates `nu_l` (site rate heterogeneity,
off by default) stay shared, as do the tree, the node times and the population model.  In the code this is
one line of `Run::derive_evo`: every `partition_evo_model[p]` is the shared HKY model with
`.mu = hky_model_.mu * site_partition_rates_[p]`.  Everything downstream --- `lambda_i`, the branch
exposure, the mutation densities, the root prior, the local increments, the mutational-history proposals
--- already reads rates per site and needs no change.

The sequence factor of the target (section 1 of `observation.md`) therefore depends on the rates
through

    log G = const + sum_p [ M_p log(mu r_p) - mu r_p E_p ],         (1)

    E_p = sum_a q_a Ttwiddle^p_a = sum_{l in p} sum_a nu_l q_a T^(l)_a,

where `M_p` is the number of mutations of partition `p` in the current mutation history and `E_p` its
exposure (the state-weighted branch time of its sites).  `Ttwiddle^p_a` is maintained per partition
already (`Run::Ttwiddle_beta_a_`), and `M_p` is recomputed from the tree once per global step
(`Run::calc_cur_num_muts_per_partition`), exactly as the mpox moves recompute their own counts.  **No
local move and no incremental likelihood had to change**, which is why the per-step cost is unchanged
(section 9).

### Weights and normalization

The rates are normalized by

    sum_p L_p r_p / sum_p L_p = 1,                                   (2)

with the weight `L_p` defined as **the number of sites of partition `p` that at least one sample
observed**.  Three properties make this the right choice:

* it makes `mu` the mean substitution rate per observed site, the same quantity a single-partition run
  reports, so the clock prior, the calibrations and the reported `clock.rate` keep their meaning;
* it is a property of the data, not of the sampled state: a site observed in no sample is missing at the
  root by the missation factoring invariant, and no MCMC move changes that set, so `L_p` is fixed for the
  whole run (`observed_sites_of_tree`, `core/site_partition.cpp`);  **[P]**
* sites that no sample observed contribute nothing to the likelihood at all (they are pruned at the
  root), so counting them would make `mu` smaller than the rate of any site that was actually seen.

A partition with `L_p = 0` is refused: its relative rate would not be identified.  Sites that are
observed in only some samples count fully --- the weight is a fixed normalization constant, not an
attempt to measure information, and any sharper weighting would make `mu` depend on the sampled
missingness pattern.

### Parameterization, prior and identifiability  [P] [N]

The sampler works in the `P-1` free dimensions of the length-weighted shares

    x_p = L_p r_p / sum_j L_j,     sum_p x_p = 1,     r_p = x_p (sum_j L_j) / L_p,       (3)

so (2) holds by construction and there is no redundant direction to remove: an overall rescaling of the
rates is not a point of the parameter space.  The prior is a symmetric Dirichlet on `x` with concentration
`c` (default 1, the uniform density on the simplex), and the proposal is a delta exchange on a pair of
shares, `x_p -> x_p + d`, `x_q -> x_q - d`, with the pair drawn uniformly and `d` uniform on `(0, step)`
for a step drawn from a fixed ladder (0.3 to 0.001) --- the same construction the HKY frequency move has
used all along.  The move is a translation inside the simplex, so

* its proposal is symmetric and needs **no Hastings term**,
* it has **no Jacobian**: the map is the identity on the free coordinates, not a reparameterization,
* the acceptance ratio is `delta log G + delta log pi(x)` with `delta log G` from (1) --- only the two
  partitions involved change, because the normalization is built into the parameterization rather than
  applied after the fact.

A naive alternative --- sample every `r_p` independently and renormalize afterwards --- changes every
partition's rate at once and carries a Jacobian; it is not what this implements.

`mu` keeps its Gibbs update: by (1) its conditional is
`Gamma(M + alpha_mu, sum_p r_p E_p + beta_mu)`, which is the old formula with the exposure weighted by
the relative rates (`Run::mu_move`).  Identifiability: `mu` and `r` are confounded along
`(mu / s, s r)`, and (2) is exactly what fixes `s`; an independent likelihood implementation confirms
that the likelihood is invariant along that direction to the last digit and that the constrained maximum
likelihood estimate recovers the generating values (`tests/prototypes/check_site_partitions.py`,
checks 4 and 5).  **[N]**  With site rate heterogeneity on, `r_p` and the per-site `nu_l` are both rate
multipliers; they stay distinguishable because `E(nu_l) = 1` is enforced by the Gamma prior over all
sites, but the contrast between a partition's rate and the mean `nu` of its sites is weakly identified
when a partition is short.  The combination is allowed and the posterior is correct; it is not part of
this task's validation.

## 3. The configuration file  [P]

```
version 1

# a non-contiguous partition
partition coding
  sites 1-300,500-800
  constant_sites A=100,C=90,G=80,T=70     # only for variable-sites-only input, see section 4
end

partition noncoding
  sites 301-499
  constant_sites A=10,C=12,G=9,T=11
end
```

Sites are 1-based, inclusive, comma-separated single sites or ranges, **in the coordinates of the input
reference**, which is the same convention the site mask and the callable declarations of Task 3.2 use.
`sites` may be repeated; the intervals of a partition are merged.  `#` starts a comment.

Validation (`read_site_partition_config` and `resolve_site_partition`, `core/site_partition.cpp`), with
one test per rule:

| Rejected | Message names |
|---|---|
| missing or unsupported `version` | the supported version |
| a partition without sites, an unclosed block, `end` without a partition, a key outside a block | the line |
| a repeated partition name (up to letter case) | both lines |
| an unknown key (for example `parts`, the Run/Subrun knob) | the key |
| a malformed, empty or reversed range | the line and the range |
| a site beyond the input reference | the range and the reference length |
| two partitions claiming the same site | both partitions and the site |
| an input site that no partition claims | how many, the first few, and that masking is the way to exclude sites |
| a partition with no observed site | the partition, and that its rate is not identified |
| `constant_sites` on some but not all partitions | that a global count cannot be split |
| declared constant sites that do not match the loaded input | both lengths |
| site partitions together with `--v0-mpox-hack` | that the hack owns the partitioning |

The canonical text of a configuration round-trips (`write_site_partition_config`), which is how it is
stored in a `.dphy` checkpoint.

## 4. Restored constant sites, per partition (Task 3.2 integration)  [P] [N]

Task 3.2's completion of variable-sites-only input appends the removed constant sites after the input
reference.  With several site partitions a single global count cannot be used, because every restored site
must belong to the partition it was removed from --- otherwise its rate is wrong.  The declaration is
therefore per partition (`constant_sites` inside the block), and the layout is fixed: the blocks follow
the input sites in configuration order and, inside a block, the A sites come first, then C, G and T.
`Human_input_config::snp_only_constant_sites` carries one entry per partition and
`resolve_site_partition` assigns the appended sites in exactly that layout, so every restored site gets

* the rate of its own partition,
* the state it is declared with (as the reference base of that site),
* full callability in every sample, and
* the ordinary likelihood contribution of a constant site.

`--v0-human-snp-only-constant-sites` (the global option) and a partition file are refused together.
Verification: a data set written both as a full-length alignment and as the variable sites of each
partition plus the declared counts gives the same multiset of (partition, observed states) pairs and the
same `log G` to 1e-12 nats, for three different pairs of relative rates
(`Site_partition_observation_test.constant_sites_restored_per_partition_match_the_full_alignment`).  **[N]**

Per-sample missingness and callable declarations keep working unchanged: a site missing in some samples
belongs to its partition and contributes the pruned likelihood of section 1 of
`observation.md`; a site missing in every sample stays out of its partition's weight.  **[N]**

What is refused rather than guessed: a global constant-site count with several partitions (the
composition per partition is unknown), and the conditional likelihood of a variable site (case C3 of
`observation.md`), which remains unimplemented.

## 5. Command line

| Option | Meaning |
|---|---|
| `--v0-site-partition-file FILE` | the configuration of section 3; without it a run has one partition over every site |
| `--v0-site-partition-initial-rates r1,r2,...` | initial relative rates in configuration order, renormalized by (2); default all 1 |
| `--v0-site-partition-fix-rates` | keep the relative rates at their initial values |
| `--v0-site-partition-rate-prior-concentration c` | concentration of the Dirichlet prior on the shares; default 1 |

The options are inert without `--v0-site-partition-file`, which is inert without the rest of a normal
run.  Start-up prints one line per partition with its sites, its observed sites and its initial rate.
The `.log` file gains one `partitionRate(<name>)` column per partition **only** when a configuration is
given, so single-partition output is byte-identical to before (section 8).

## 6. What the rates reach  [P]

| Term or state | How the relative rates enter |
|---|---|
| mutation densities, branch exposure, root prior, `lambda_i` | through `evo.mu_l(site)`, which is `mu * r_p` for a site of partition `p`; no code change was needed |
| `mu` Gibbs update | exposure weighted by `r_p` (`Run::mu_move`) |
| HKY frequency and kappa moves | the branch-length term now uses `mu * r_p` per partition, and the root-prior term uses the state counts summed over all partitions (both were written for a single shared rate; the accept branch no longer overwrites the per-partition rates) |
| site rate heterogeneity (`nu_l`, `alpha`) | already used `evo.mu_l(site)`; unchanged |
| local moves, SPR, mutational-history proposals | unchanged; the proposal rate `mu_JC` is the average rate at the root (`lambda_i[root] / observed sites`), so it adapts to the mixture by construction |
| exact Kingman correction, joint time-scale move | unchanged; both recompute the target, and the scale move transforms `mu` while `r` is dimensionless |
| `log_other_priors` | gains the Dirichlet term (zero with one partition, so nothing changes for existing runs) |
| checkpoint (`.dphy`) | the canonical configuration text, the rates, the prior concentration and the move switch (`api.fbs` ids 58-61); on restore the configuration is re-resolved against the tree, which recomputes the weights |
| output | one `partitionRate(name)` column per partition |

## 7. Independent verification  [N] [R]

* **The move samples its exact conditional.**  With the genealogy, the history and every other parameter
  frozen, the conditional density of the shares is known in closed form; a chain of 240,000 rate moves,
  thinned to 6,000 draws, matches twenty equal-probability bins of that density (chi-squared 19 degrees of
  freedom, both for the flat prior and for concentration 4).
  `Site_partition_rate_move_test.samples_the_exact_conditional_of_the_rate_shares`.
* **The acceptance ratio is the full posterior difference.**  Every proposal's `delta log G` is compared
  with the difference of two independent per-site likelihood computations, and accept, reject and the
  rolled-back state are checked separately.
* **The likelihood.**  Equal rates over two partitions reproduce the single-partition likelihood to
  1e-12 nats; unequal, scattered and silent partitions match the independent per-site computation to
  1e-8 nats.
* **Independent model checks** (`tests/prototypes/check_site_partitions.py`): the rate matrix is
  normalized to 1 substitution per site per unit time; two independent matrix exponentials agree to
  3e-16; forward simulations of the mutation history reproduce the pruning probabilities of the tip
  patterns at per-partition rates; the number of events of a partition is `L_p mu r_p T` to 0.04%; the
  likelihood is invariant along `(mu / s, s r)` so (2) is what identifies `mu`; and the constrained
  maximum likelihood estimate recovers the generating `(mu, r)` within its Monte Carlo error.
* **Posterior recovery** on simulated data with known rates, several chains per data set: see the task
  report (`36_site_partition_report.md`), including the 100-replicate coverage study at N = 20 and the
  equal-rate control.

## 8. Compatibility

Nothing changes without `--v0-site-partition-file`: one partition, relative rate 1, the Dirichlet term is
zero, the rate move returns immediately, the `.log` columns and the `.dphy` fields are not written.  The
mpox hack keeps its own two context-defined partitions and refuses to be combined with site partitions.
The audit report of the human input path moves to version 3 (a `constant_sites_appended_per_partition`
array); no key was removed or changed in meaning.

## 9. Measured cost

See the task report for the numbers.  In short: the per-step cost of the sampler is unchanged, because no
local move or incremental formula was touched; the rate update costs one pass over the mutations per
global step (the same order as the existing mpox counts), and the start-up cost grows by the resolution
of the configuration.

## 10. Limitations

1. The first version shares kappa, the state frequencies and the clock across partitions; per-partition
   substitution models (TN93, GTR, per-partition kappa or frequencies) and FreeRate-style mixtures are
   Task 3.5 and are not implemented.
2. Partitions are defined on the input reference coordinates only; a partition cannot be defined on the
   restored constant-site block directly (its counts are declared inside the owning partition).
3. A partition with no observed site is refused, so a configuration must be revised when a mask or a
   callable declaration removes a whole partition.
4. With site rate heterogeneity on, a short partition's rate and the mean `nu` of its sites are weakly
   identified (section 2); the combination is allowed but not part of this task's validation.
5. The relative-rate move is a delta exchange on pairs of shares.  With many partitions of very different
   lengths the step ladder may need tuning; nothing adapts automatically.
6. A uniform Dirichlet on the shares is **not** uniform on the rates: the implied prior on the rate of
   partition `p` is uniform on `(0, sum_j L_j / L_p)` with mean `sum_j L_j / (P L_p)`, so a short
   partition starts from a prior that allows, and on average expects, a much faster rate than the rest.
   With a long partition the likelihood overwhelms this; with a 100-site partition out of 2,000 the
   posterior is visibly prior-influenced (measured in the task report).  A user who means "every rate is
   a priori about 1" should say so with a larger concentration, or keep partitions long enough for the
   data to decide.
7. Delphy's mutation-history sampler is built for data with few hidden substitutions.  The measurements in
   the task report show that at about 0.4 substitutions per site from the root to a tip the number of
   hidden events equilibrates far too slowly for the rates to be sampled; this is a property of the
   existing sampler, not of the site partitions, and it bounds the divergence at which any rate estimate
   from Delphy can be trusted.
