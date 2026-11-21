# Posterior clade tMRCA summaries: definitions and design

This document states what each reported number means; `08_posterior_summary_report.md` says
what was built, what was run and what the results do and do not establish.

Nothing here changes the posterior.  Every quantity below is a summary of trees that an MCMC run
already sampled.  Rescaling, reweighting or re-rooting a finished posterior is not done anywhere.

---

## 1. The two quantities that must not be confused

Let `T` be a set of sample IDs and let `S` be the post-burn-in posterior sample of trees.  For a tree
`g`, write `mrca_g(T)` for the most recent common ancestor of `T` in `g` and `a_g(T)` for its age.

* **Unconditional MRCA age**, reported as `mrca_age_*`:
  the posterior of `a_g(T)` over **every** tree of `S`.  It is defined in every tree, monophyletic or
  not, and it answers "how old is the common ancestor of these samples".
* **Clade age**, reported as `clade_age_*`:
  the posterior of `a_g(T)` restricted to the trees in which `T` is monophyletic, i.e. conditional on
  the clade existing.  Its sample size is `posterior_support * |S|`.
* **Posterior support**: the fraction of trees of `S` in which `T` is monophyletic.

`Mcc_tree` (`core/mcc_tree.cpp: calculate_derived_quantities`) computes the **means** of exactly these
two quantities and calls them `t_mrca` and `t`.  Neither is an interval, neither has a sample size
attached, and `delphy_mcc` writes its branch lengths from `t_mrca` while labelling each node with the
support of the *other* quantity.  That is a correct MCC tree and a misleading tMRCA report, which is
why this task adds a separate tool rather than more annotations to the MCC output.

With a fixed topology (Task 2.1) and a set that is monophyletic in the input tree, the two quantities
are identical sample for sample and the support is exactly 1.  Under a free topology they differ, and
the difference is not a rounding effect: a set that is paraphyletic in a tree has its MRCA higher up,
so the unconditional posterior is the older of the two.

Both are reported for every clade, always, with their own sample sizes.  A clade that is never
monophyletic has no clade age at all, and the table says `NA` rather than falling back to the
unconditional value.

## 2. Clade identity: exact sets, not fingerprints

`mcc_tree.cpp` gives every tip a random 64-bit `Clade_fingerprint` and XORs them up the tree, so two
different tip sets can in principle be judged equal.  With `C` distinct clades across all base trees
the chance of at least one collision is about `C^2 / 2^65`; for `C = 10^6` that is `2.7e-8`, which is
small but silent, and the fingerprints are reseeded on every run, so a collision is also not
reproducible.

`Clade_resolver` (`core/clade_summary.cpp`) does not use fingerprints:

* a clade is a **sorted vector of tip node indices**, resolved once from the sample IDs against the
  trees file's translate table;
* the MRCA is found by climbing parent links from each tip and taking the lowest common ancestor, so
  it is exact even when two node times print as equal after rounding;
* monophyly is `tips_below(mrca) == |T|`, counted in one post-order pass per tree.

`Clade_resolver_test.agrees_with_the_mcc_trees_exact_match_logic` checks that the exact resolution
reproduces `Mcc_node::posterior_support`, `t` and `t_mrca` on a real posterior sample, node for node
and base tree for base tree.  That is the statement "no collision happened here", verified, rather
than assumed.

Cost per tree: one O(N) pass for `tips_below` and depths, plus `O(|T| * depth)` per clade.  No clade
is ever resolved by a node index, and nothing is carried over from one tree to the next.

## 3. Intervals

`n` is the number of samples that enter the statistic and `mass` the credible mass (0.95 by default).

### 3.1 Equal-tailed interval (`eti_*`)

The type-7 sample quantiles at `(1 - mass)/2` and `(1 + mass)/2`, i.e. linear interpolation between
order statistics at `(n - 1) p`.  This is what R's `quantile()` and `numpy.quantile()` return.

### 3.2 Shortest contiguous credible interval (`shortest_interval_*`)

The **shortest interval spanned by `k = ceil(mass * n)` of the `n` samples**:

```
k = ceil(mass * n)
interval = argmin over i of [ x_(i+k-1) - x_(i) ]   over the sorted sample x_(1) <= ... <= x_(n)
```

This is computed exactly, by scanning all `n - k + 1` windows.

**This interval is not the same object as a highest posterior density region, and the tables never
claim it is.**  An HPD region at credible mass `m` is the set `{x : p(x) >= c}` with the largest `c`
whose set carries mass `m`.  That set is an *interval* only when the posterior is unimodal; then, and
only then, the shortest contiguous interval estimates it.  When the posterior is multimodal the HPD
region is disconnected, and the shortest contiguous interval bridges the low-density gap between the
modes: it is still a valid 95% credible interval, but it is wider than the HPD region and contains
points of lower density than points it excludes.

The summary tables therefore carry three things rather than one:

| column | meaning |
|---|---|
| `*_shortest_interval_lower`, `_upper` | the interval above, named for what it is |
| `*_interval_is_hpd` | `yes` exactly when the estimated highest density region has one component, i.e. when that interval may be called a highest posterior density interval; `no` otherwise |
| `*_hdr_intervals` | the estimated highest density region itself, as `lo..hi` parts joined by `;` -- the set the term HPD names, which need not be an interval |

When `*_interval_is_hpd` is `no`, `delphy_clades` also prints a warning on stderr naming those clades
and saying in words that the interval reported for them is not a highest posterior density interval,
and the clade's `diagnostics` column carries
`mrca_multimodal_interval_is_not_hpd(<components>_hdr_components,<bins>_bins,gap=<fraction>)`.

The interval is also never the pair of 2.5% and 97.5% quantiles, and the equal-tailed interval is
never labelled a highest-density interval: on a unit-exponential sample of 20,001 draws the shortest
interval is `[0.000025, 2.9953]` and the equal-tailed interval is `[0.0253, 3.6879]`, a 23% difference
in the upper end
(`Posterior_stats_test.shortest_interval_differs_from_the_equal_tailed_interval_for_a_skewed_sample`).

Two further properties worth stating because they are easy to get wrong:

* When `ceil(mass * n) == n` -- e.g. 11 samples at 95% -- the shortest interval is the full range of
  the sample and can be *wider* than the interpolated equal-tailed interval.  That is a property of
  having 11 samples, not a defect, and it is why every interval is printed next to its sample size.
* The interval is a property of the empirical sample.  Its endpoints are order statistics, so with a
  small sample they jump; the ESS next to them is what says how much to trust them.

### 3.3 The highest density region, and multimodality

The HDR is estimated from a histogram and its number of components is reported
(`*_hdr_components`), together with the components themselves (`*_hdr_intervals`):

1. histogram the sample into `round(sqrt(n))` bins, clamped to `[10, 100]`;
2. rank the bins by a 1-2-1 smoothed count and select them in that order until the selected bins hold
   `ceil(mass * n)` samples;
3. merge runs of selected bins separated by at most 2 unselected bins;
4. a run holding less than 5% of the selected mass is a **minor** component and does not count towards
   multimodality.

Rules 1, 3 and 4 exist because a finite sample of a perfectly unimodal density has Poisson noise of
order `sqrt(count)` per bin, and a naive bin-by-bin rule breaks its tails into a dozen pieces.  With
them, 500 / 5,600 / 20,000 normal draws give exactly one component
(`Posterior_stats_test.hdr_does_not_cry_multimodal_on_noisy_unimodal_samples`), and a 50/50 mixture of
two separated normals gives two, with the two components together less than half as wide as the
shortest contiguous interval that bridges them (`hdr_finds_two_well_separated_modes`).

`interval_gap_fraction` (reported inside the diagnostic code) is the fraction of the shortest
interval's width that falls **between** two major components.  Only interior gaps count; the ends of
the outermost components are bin edges, and the mismatch between a bin edge and the sample value at
the end of the interval is a grid artifact, not a gap in the density.  A single component therefore
gives exactly 0.

This is a diagnostic, not an estimator: the number of components depends on the bin count, so the bin
count is reported with it and can be set with `--hdr-bins`.  A multimodal posterior age is also a
common symptom of chains that have not mixed, so the warning says to compare it with
`mrca_age_rank_split_rhat` and the per-chain table before reading it as a real feature.

## 4. Effective sample size and multi-chain diagnostics

### 4.1 ESS

Geyer's initial monotone positive sequence, as Tracer and `tests/human_input/analyze_log.py` use:

```
rho_m   = autocorrelation at lag m
Gamma_m = rho_{2m} + rho_{2m+1}, truncated at the first negative term and made non-increasing
tau     = max(-1 + 2 * sum_m Gamma_m, 1)
ESS     = n / tau
```

Autocovariances are computed directly.  The lag budget defaults to `min(n - 1, 5000)`; if the positive
sequence has not terminated by then the sum is truncated, the returned ESS is an upper bound, and the
summary carries an `*_ess_lag_truncated` diagnostic.  A constant sample returns `ESS = n`; fewer than
four samples returns 0.

The ESS of several chains is the **sum** of their per-chain ESS values, which is correct because the
chains are independent.  `--out-per-chain` also writes the per-chain statistics, because a pooled ESS
hides a single stuck chain.

### 4.2 Split R-hat, rank-normalized

Every chain is split in half, all halves are trimmed to the shortest, and the usual
between-over-within ratio is formed over the `2 * num_chains` halves:

```
R-hat = sqrt( ((m - 1)/m * W + B/m) / W )
```

`rank_normalized_split_rhat` first replaces every draw by the normal score of its rank among the
pooled draws of all halves, using Blom's plotting position `(r - 3/8) / (S + 1/4)` (Vehtari et al.,
2021).  Both are reported; the rank-normalized one is the one to judge by, because a posterior with
heavy tails makes the plain statistic a function of whichever chain drew the largest outlier
(`Posterior_stats_test.rank_normalization_tames_a_heavy_tailed_target`).

With a single chain the statistic is still computed, from that chain's two halves.  It then detects a
chain that drifts but **cannot** detect a chain that settled in the wrong place, and the tool says so
in a note on every single-chain run.

### 4.3 What the thresholds do, and what they do not do

A clade whose rank-normalized split R-hat exceeds `--max-rank-split-rhat` (1.01), whose ESS is below
`--min-ess` (200), whose HDR has more than one major component, whose taxon set is not monophyletic in
every tree, or whose ESS lag budget ran out, gets a code in the `diagnostics` column; the codes are
also printed as a block on stderr, and `--fail-on-diagnostics` turns them into a non-zero exit status.

The pooled row is still written.  Suppressing it would hide the evidence; what must not happen -- and
does not -- is a pooled 95% interval presented with nothing to say that the chains disagree.

The honest limit of this: **chains that drift the same way, or that are stopped by the same internal
bound, agree with each other, and no R-hat can see that.**  Section 5.4 of the report is a measured
case of exactly that.  Multi-chain diagnostics are necessary, not sufficient.

### 4.4 What the coverage figures count

Several MCMC chains on one simulated data set are one draw of the sampling distribution, not several.
Coverage is therefore counted with the **data set** as the statistical unit, and
`summarize_clade_validation.py` reports both the per-data-set coverage and the pooled count, with the
number of data sets next to it.

## 5. Burn-in, thinning and pooling

`delphy_mcc` has always discarded `min_state + 0.3 * (max_state - min_state)` and everything below it,
with no way to change it.  That rule is now the default of a configurable interface, in both tools:

| option | meaning |
|---|---|
| `--burn-in-fraction F` | discard every state `<= min_state + floor(F * (max_state - min_state))`.  `F = 0.3` by default, and `floor(0.3 * range)` reproduces the historical integer `(range * 3) / 10` exactly |
| `--burn-in-states N` | discard every state `<= N` |
| `--burn-in-samples K` | discard the first `K` sampled trees of each chain |
| `--thin E` | keep only states that are multiples of `E` |

At most one burn-in option may be given.  The resulting counts -- trees read, trees discarded, trees
kept, first kept state -- are printed in the provenance block of every output table, per chain.

`delphy_mcc <in> <out>` with no options is byte-identical to the previous binary (report section 5.3).

Chains are given as repeated `--trees` options.  They must have the same taxa in the same order,
because clade membership is resolved against the translate table; a mismatch is an error naming both
files, not a silently reindexed pool.  Per-chain and pooled statistics are both available.

## 6. The time axis

A BEAST-format trees file carries **branch lengths only**.  `newick_to_phylo_tree` puts the root at
`t = 0` and derives every other node time from the branch lengths, so the times in a tree read back
from a file are relative to its own root.  The absolute calendar position is not in the file.

`delphy_clades` therefore reports ages in one of two explicitly named ways, and never mixes them:

* **anchored** (`--tip-dates FILE` or `--uniform-tip-date YYYY-MM-DD`, with `--age-reference-year Y`,
  default 1950): one sample with an exactly known sampling date fixes the axis.  For the anchor tip
  `A` with absolute time `tA` (days since 2020-01-01) and reference instant `t_ref` (1 January of `Y`,
  astronomical year numbering, from `astronomical_year_to_day`),

  ```
  age(node) = ( t_ref - tA - ( t_rel(node) - t_rel(A) ) ) / 365    [years]
  ```

  The reference is printed as `years_before_<Y>-01-01` and the anchor as the sample ID and its date.
  The anchor must be a sample whose date is a single instant (`t_min == t_max`); a month or a year is
  refused, because it would smear the whole axis.  The default anchor is the most recent such sample,
  ties broken by sample ID, so the choice is reproducible.  `--uniform-tip-date` is the shorthand for
  the headline human workflow, where every sample carries the same sampling date: it gives every tip
  that date instead of reading a metadata table, and is refused together with `--tip-dates`.
* **relative** (no dates given): `age(node) = (t_youngest_tip(g) - t_rel(node)) / 365` in each tree
  `g`, i.e. the height above the most recent tip, which is BEAST's `rootHeight` convention.  The
  reference is printed as `years_before_the_most_recent_tip_of_each_tree` and a note says this is not
  a calendar age.  `--age-reference-year` without dates is an error rather than an assumption.

Ages are always years, always of age (larger = older), and never calendar years: the tables carry no
mixture of BP, CE and Delphy's internal day axis.  Internal times remain days since 2020-01-01 and 1
year is 365 days, as everywhere else in the CLI.

## 7. Serialization precision at deep time

The `.trees` branch lengths were printed with the default 6 significant digits, so a 140,000-year
branch is quantized to 0.1 years, and a node age is the sum of the lengths on its root-to-tip path.
Measured end to end (`Clade_summary_test.trees_file_precision_at_human_depths`, and against the
`calAge(...)` columns of a real run, report section 5.5):

| root depth | max clade age error at 6 digits | at 9 digits | at 12 digits |
|---|---|---|---|
| 50 kyr | 0.067 yr | 3.3e-05 yr | 7.1e-08 yr |
| 100 kyr | 0.050 yr | 1.1e-04 yr | 8.2e-08 yr |
| 140 kyr | 0.444 yr | 6.0e-04 yr | 7.1e-07 yr |
| 200 kyr | 0.479 yr | 3.2e-04 yr | 2.0e-07 yr |

Half a year on a 140,000-year age is 3e-6 of the age and about 1e-5 of a typical 95% interval, so 6
digits does not affect any interval reported here.  It does affect anything that differences two node
ages on the same path, so the writer's precision is now an option:

```
--v0-out-trees-digits N     (default 6: byte-identical to every earlier version; 9 is below a day at
                             any human depth, 17 round-trips every double)
```

The default is unchanged and the pathogen regression is byte-identical (report section 5.3).  This is
output precision only; `.dphy` still stores float32 node times (risk R-02, Task 1.3), and that format
is untouched.

## 8. Memory and scale

One tree is in memory at a time.  The reader (`Beasty_trees_stream`, `core/beasty_input.cpp`) parses
the taxa block and translate table once and then yields one `Phylo_tree` per call; the burn-in cut is
chosen from a first pass that reads only the state numbers (`scan_beasty_tree_states`), so no tree is
parsed that will be discarded.  `read_beasty_trees`, which loads everything into a map, is kept with
its exact previous behaviour and is now implemented on top of the stream; `delphy_mcc` uses the stream
directly and no longer parses burn-in trees at all.

What is retained is `O(clades * post-burn-in trees)`: one double, one int and one byte per clade per
tree, about 16 bytes, independent of the number of tips.  An exact shortest credible interval needs
the whole sample of that clade, so this is not avoidable without giving up exactness -- and an
approximate quantile is not reported as a credible interval anywhere.  For a workflow that cannot hold even that, `--out-samples`
writes the raw per-tree samples and `--in-samples` summarizes them again, with a different credible
mass or a different set of clades, without re-reading a single tree.  Ages are written there with 17
significant digits, so that round trip reproduces the original summary bit for bit.

## 9. Output tables

`--out-summary` (default: standard output), one row per clade.  Every table begins with a `#` block
naming the input files, the burn-in and its per-chain counts, the age reference, the time anchor, the
credible mass and the diagnostic thresholds.  Columns:

| column | meaning |
|---|---|
| `clade_id` | the ID from the clade definition table |
| `clade_definition`, `num_taxa_specified` | `all_tips`, or how many sample IDs were named |
| `calibration_status` | `calibrated:<name>` when the taxon set is exactly a calibrated set, `not_calibrated`, or `unknown` when no calibration configuration was given |
| `posterior_support` | fraction of post-burn-in trees in which the set is monophyletic |
| `num_posterior_samples`, `num_monophyletic_samples`, `num_chains`, `samples_per_chain` | sample sizes |
| `age_units`, `age_reference`, `credible_mass` | the conventions, on every row |
| `mrca_age_mean`, `_median`, `_sd` | unconditional MRCA age (section 1) |
| `mrca_age_shortest_interval_lower`, `_upper` | shortest contiguous credible interval (section 3.2) |
| `mrca_age_interval_is_hpd` | whether that interval may be called a highest posterior density interval |
| `mrca_age_eti_lower`, `_upper` | equal-tailed interval (section 3.1) |
| `mrca_age_ess`, `_split_rhat`, `_rank_split_rhat` | diagnostics (section 4) |
| `mrca_age_hdr_components`, `_hdr_intervals` | the estimated highest density region (section 3.3) |
| `clade_age_*` | the same statistics conditional on monophyly, or `NA` |
| `tips_below_mrca_min`, `_median`, `_max` | equal to `num_taxa_specified` in every tree exactly when the support is 1 |
| `diagnostics` | `ok`, or the codes of section 4.3 |

`--out-samples` writes `clade_id, chain, state, age_years, monophyletic, tips_below_mrca` -- the raw
posterior samples, at full double precision, for an independent analysis.  `--out-per-chain` writes one row
per (clade, chain).  No column name is an internal variable name, and no age appears without its
reference instant.

## 10. Unit audit of the simulation tooling

Requested after the first draft of this task, because one of its own validation scripts had a unit
error.  `--v0-init-final-pop-size` is **in years**; `core/cmdline.cpp:1136` multiplies it by 365 and
`core/cmdline.cpp:1278` hands the result to `Exp_pop_model`, which `Run::set_pop_model`
(`core/run.h:62-65`) installs as the model of the coalescent prior itself.  With
`--v0-fix-final-pop-size` it is therefore the **actual prior for the whole run**, not an
initialization value; without it, it is the starting value of a sampled parameter and still the prior
at step 0.

| script | task | what it passes | verdict |
|---|---|---|---|
| `run_calibration.py` | 2.2 | `--v0-init-final-pop-size TAU` with `TAU = 75000` years, the simulated timescale | correct |
| `run_calibration_example.sh` | 2.2 | `50000`, documented as 50,000 years | correct |
| `run_clade_summary_cli_tests.sh` | 2.3 | `50000` years | correct |
| `run_clade_summary_validation.sh` | 2.3 | **was** `27375000` (75,000 years written in days), now `75000` | **was wrong by a factor of 365; fixed, and every number it produced before the fix is discarded** |
| `run_matrix.sh`, `run_benchmark.py` | 1.3 | does not pass it | no unit error; the runs used Delphy's default exponential model (initial size 3.0 years, both parameters sampled), which is a model mis-specification already reported in `03_performance_baseline.md` and `03_time_identifiability.md` as the free-population / 400-cell problem |
| `run_fixed_topology.py`, `run_external_tree_cli_tests.sh` | 2.1 | does not pass it | same as above; `external_tree.md` section 6 already attributes the N=1000 coverage loss to the coalescent grid and the free population model |

Other unit-bearing interfaces were checked the same way:

* `--v0-init-mutation-rate` and `--v0-mu-prior-mean` / `--v0-mu-prior-stddev` are per site **per
  year** (`core/cmdline.cpp:973`, `1004-1005`), and every script passes the simulated per-year rate.
* The simulator's `--coal-timescale-years` is years throughout (`simulate_mtdna_coalescent.py:60-78`,
  tip ages in years), so passing it straight to `--v0-init-final-pop-size` is right.
* Calibration ages are converted value by value from their declared convention
  (`calibration.md` section 3); `make_calibration_config.py` writes `years_bp` with
  `--reference-year 2020`, and `delphy_clades --age-reference-year 2020` reads the same axis.
* Internal times stay days since 2020-01-01 and internal rates per site per day; the only places
  years appear are the CLI, the BEAST log, the `.trees` branch lengths and the tables of this task.

**Consequence for earlier conclusions.**  Task 2.2's statistical results are unaffected: they passed
the timescale in years, as the flag expects.  Tasks 1.3 and 2.1 never set it, so their numbers contain
no unit error, but they were obtained with Delphy's default coalescent model rather than the simulated
one; that is already stated in their reports as the cause of the biases they measure, and it is not
re-interpreted here.  Only this task's own first campaign was invalid, and it is not reported.

## 11. Deliberately out of scope

* No change to `core/mcc_tree.cpp`.  The MCC machinery and its fingerprints are left exactly as they
  are; the new tool simply does not use them, and a test shows the two agree.
* No change to the Skygrid or coalescent code, no new MCMC operator, no change to the free-topology
  algorithms, and no joint rate/time scaling move (Task 2.2's open item).
* No re-rooting, no re-weighting and no post-hoc rescaling of sampled trees.
* `.dphy` is not read: the CLI cannot resume from one and a `.dphy` snapshot is a single state, not a
  posterior sample.  The posterior sample Delphy writes is the `.trees` file.
