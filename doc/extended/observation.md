# Observation models: full sequences, reference plus differences, callable sites and SNP ascertainment

This document states, for every input format Delphy accepts, which sites of which samples are observed, what
the likelihood therefore is, and what is refused because the input does not determine it.  It supersedes
nothing in `input_contract.md`; it extends section 5 of that document, which deferred exactly this subject.

Evidence classes: **[P]** proof read from the source, **[N]** independent numerical check, **[R]** check on the
real kernel through the CLI or the sampler, **[M]** measurement, **[U]** unverified.

## 1. What the sequence likelihood actually uses  [P]

Delphy samples an explicitly mutation-annotated tree (EMAT): the state is the timed topology plus the complete
mutation history, and the sequence factor of the target is a density over that history, not a marginal
likelihood of tip states.  With sites indexed by `l`, states by `a`, the per-partition model
(`core/evo_model.h`) giving `mu_l`, `nu_l`, `pi_l_a` and the escape and exchange rates `q_l_a`, `q_l_ab`:

    log G = sum_l [ log pi_l(s_l(root))                                     (root prior)
                    + sum_{branches X present at l} ( -mu_l nu_l q_l(state) * (segment length) )
                    + sum_{mutations m at l} log( mu_l nu_l q_l(m.from, m.to) ) ]

In the code this is `calc_log_root_prior` (`core/phylo_tree_calc.cpp:467`) plus `calc_log_G_below_root`
(`core/phylo_tree_calc.cpp:515`), assembled in `Run::calc_cur_log_G` (`core/run.cpp:617`) and kept
incrementally in `log_G_`, which `Run::log_posterior` (`core/run.h:251`) adds to the coalescent, calibration
and parameter priors.  Three properties of this form matter for everything below:

1. **The likelihood is a product over sites.**  Nothing in the formula couples two sites: the per-branch term
   is accumulated in `lambda_i[X]`, a sum over the sites present at `X` (`calc_lambda_i`,
   `core/phylo_tree_calc.cpp:420`).  A site enters only through its own state path.
2. **Sites are exchangeable inside a partition with equal relative rates.**  The only per-site quantities are
   `partition_for_site[l]` and `nu_l[l]`.  With one partition and `nu_l == 1` --- the CLI default, since
   `alpha_move_enabled_` is false (`core/run.h:440`) and `--v0-site-rate-heterogeneity` must be asked for ---
   the contribution of a site depends on its state pattern alone, not on its coordinate.  With site rate
   heterogeneity the `nu_l` are independent draws from the same Gamma prior, so sites are exchangeable a
   priori and a permutation of the sites leaves the posterior over everything else unchanged.
3. **Missing data prune, they do not fill in.**  A site in a node's `Missation_map` contributes nothing to
   `lambda` from that branch downwards (`calc_delta_lambda_across_missations`, `core/phylo_tree_calc.h`) and is
   removed from the root prior when it reaches the root (`core/phylo_tree_calc.cpp:479-489`).  Because the
   state process is a continuous-time Markov chain and the transition matrix is stochastic, pruning the
   subtree below the point where a site stops being observed *is* the marginalization over the unobserved
   states.  Verified independently: the pattern likelihood of a tree with one tip not callable equals the
   likelihood on the tree without that tip, to 5.5e-16 relative
   (`tests/prototypes/check_snp_ascertainment.py`, check 2).  **[N]**

The reference sequence is a representation, not data: `Phylo_tree::ref_sequence` is the baseline from which
mutations are deltas, and both the initial-tree builders and the external-tree importer rebase it to the root
sequence (`rereference_to_root_sequence`, `core/phylo_tree.cpp:306`, called at `core/external_tree.cpp:344`).
A site where every sample shares a state that is not the reference base therefore has the same likelihood as
the same observation written against that state; checked to 1e-12 nats in
`Human_observation_test.the_reference_base_at_an_invariant_site_does_not_change_the_likelihood`.  **[N]**

An independent re-implementation of the formula above (one site at a time, explicit states along each branch,
subtree pruning at missations, no `lambda` bookkeeping) is in `independent_log_G`
(`tests/human_observation_tests.cpp`); every likelihood claim in this document is checked against it.

## 2. The five observation modes

Write `C_i` for the set of sites that were examined ("callable") in sample `i`, and `x_il` for the state
called at site `l` of sample `i`.  The likelihood of section 1 needs, for every sample and every site of the
analysed sequence, either a state or "not observed".  The modes differ only in how much of that an input
determines.

| Mode | Input | Determines the observation? |
|---|---|---|
| A | Full-length alignment (FASTA), one column per reference site | Yes, by construction |
| B | Reference plus per-sample differences and missing regions (MAPLE) | Yes, if the producer declared every non-called site as missing |
| C | Variable-sites-only alignment ("SNP-only") | No: the removed sites are unknown |
| D | Variant list plus callable sites, same in every sample | Yes |
| E | Variant list plus callable sites per sample | Yes |

* **A** is lossless.  `N`, `?`, `-`, `.` and every IUPAC code become missing data, never reference states
  (`calculate_delta_from_reference`, `core/sequence_utils.cpp:30`, and section 5 of `input_contract.md`).
* **B** is lossless *as a representation*: the reference is full length and the missing intervals are explicit,
  so A and B carry the same information, which the tests confirm (`Human_input_test.maple_matches_fasta_representation`
  for the representation, and `Human_observation_test.the_same_observation_gives_the_same_likelihood_from_either_format`
  for the likelihood, equal to 1e-12 nats).  It is lossless *as data* only under an assumption Delphy cannot
  check: that the producer of the file wrote out every site that was not called.  A pipeline that emits only
  the variant positions and leaves the rest implicit produces a file that says "reference" where it means
  "not examined".  That assumption is on the user; it is now stated in the audit report as the observation
  mode `reference_plus_deltas`.
* **D and E** are the honest form of variant-call data and are what this task adds.  A variant list alone says
  nothing about the sites it does not mention, so Delphy refuses it without a callable declaration; with one,
  a site that is callable and not listed is a reference state, and a site that is not callable is missing.
  `Human_observation_test.variant_input_reproduces_the_equivalent_full_alignment` builds the same observation
  as an alignment and as a variant list plus per-sample callable sites and checks that the resulting tips
  (differences and missing intervals), the site classes and the likelihood agree exactly.  **[N]**
* **C** is the only mode that needs more than bookkeeping; sections 4 to 6 treat it.

### Coordinates, lengths and site classes

The analysed sequence is the input reference in its own coordinates, possibly extended by a block of restored
constant sites (section 5).  The audit report (version 2) states both lengths (`num_sites`,
`num_input_sites`) and classifies every analysed site as

* `variable`: some sample differs from the reference there;
* `invariant_nonreference`: a subset of the above where every observed sample shares one non-reference state
  (invariant data, written against a different reference base);
* `invariant_observed`: observed in at least one sample and never different from the reference;
* `never_observed`: missing or masked in every sample --- excluded from the likelihood by the root missations,
  never counted as invariant.  `Human_observation_test.never_observed_sites_do_not_enter_the_likelihood`
  adds four all-`N` columns to a data set and finds the likelihood unchanged to 1e-12 nats.  **[N]**
* `masked` and `not_callable` count the two ways a site can be withheld (see section 7).

## 3. Missing data: the assumption that is being made  [P]

Pruning is the correct marginalization only if the pattern of missing data is uninformative about the states
that were not seen, that is, if missingness is independent of the unobserved state given the tree and the
parameters (missing at random).  Coverage-driven missingness --- a region that no read covered, a sample with
a damaged extract --- satisfies this in the usual modelling sense.  Two common situations do not, and Delphy
cannot detect either:

1. A per-site filter whose outcome depends on the call, for example dropping a genotype because it is a
   singleton, has low quality *because* it is a non-reference call, or lies in a region that was masked after
   looking at the variants.  Such missingness is informative and biases the result; the only remedy is to
   apply the filter to the site in every sample (the site mask) rather than to single calls.
2. Reference-bias in mapping: positions where a non-reference allele fails to map are missing *because* of
   their state.  This is a property of the upstream pipeline, and a region mask is the only honest answer.

Both are now named in the audit report's warnings when a callable declaration is used, because the
declaration is the place where a user states which sites were examined independently of what was found there.

## 4. Ascertainment: the derivation  [P] [N]

Let the genome have `L` sites of which the producer of the input kept `m` by a rule `S` that looks at the
data.  The two rules in use are

* **R1 (polymorphism):** keep `l` if at least two distinct states occur among the samples callable at `l`;
* **R2 (difference from the reference):** keep `l` if some callable sample differs from the reference at `l`.

The data actually handed to Delphy are the kept sites.  Writing `g(d_l | theta)` for the likelihood of the
observation at site `l` (the quantity of section 1 integrated over the mutation history), the likelihood of
the *input* is

    P(kept sites | theta) = prod_{l kept} g(d_l | theta) * prod_{l removed} g(d_l | theta),

and the three cases differ in what is known about the second product:

**C1 --- the removed sites are known as counts per state.**  If the removed sites are known to be constant
with `n_a` of them in state `a` (and callable in every sample), then

    prod_{l removed} g(d_l | theta) = prod_a g(all samples show a | theta)^{n_a},

which is exactly the likelihood of `n_a` extra constant sites.  Because the likelihood is a product over
sites and sites are exchangeable within a partition (section 1, properties 1 and 2), appending `n_a` sites of
state `a` to the input reconstructs the full-length likelihood exactly, with no change to the likelihood or
to the sampler.  This identity is what `--v0-human-snp-only-constant-sites` implements; it is checked
numerically from both ends: inside Delphy
(`Human_observation_test.snp_only_completion_gives_the_full_length_likelihood`, equal to 1e-12 nats against
the full-length alignment) and against the independent pruning implementation
(`check_snp_ascertainment.py`, check 4: difference 0.0 nats between the full-length log likelihood and
"variable sites plus restored block").  **[N]**

**C2 --- only the number of removed sites is known.**  Then

    prod_{l removed} g(d_l | theta) = [ sum_a g(all samples show a | theta) ]^{L - m},

the probability of a constant pattern raised to the number of removed sites.  This is computable, but it is
not a product of per-site EMAT histories: a "constant site of unknown state" cannot be written as a reference
base plus deltas, so it cannot be represented in the EMAT state space at all.  Delphy refuses this case and
asks for the composition instead, which is available whenever `L` is (the composition of the reference over
the analysed region).

**C3 --- nothing is known about the removed sites.**  Conditioning each kept site on having passed the filter
gives the standard ascertainment-corrected likelihood

    P(kept sites | theta, kept) = prod_{l kept} g(d_l | theta) / P(S keeps l | theta),

with, for rule R1 and a set `K` of samples callable at `l`,

    P(S keeps l | theta) = 1 - sum_a P(every sample in K shows a | theta),

and for rule R2 the asymmetric variant `1 - P(every sample in K shows the reference base)`.  The normalizer is
a marginal probability over tip states: computing it needs Felsenstein pruning with transition matrices over
the whole tree, once per distinct callability set `K` and per distinct relative rate `nu_l`.

Properties of the normalizer, measured independently (`check_snp_ascertainment.py`, check 3, four tips,
HKY with kappa 2.5 and pi = (0.31, 0.19, 0.17, 0.33), mu = 3e-4 per site per day, branch lengths 400 to 1900
days):  **[N]**

| Quantity | `-log(1 - P(constant))` per kept site |
|---|---|
| mu x 0.5 | 0.663574 |
| mu (reference setting) | 0.308374 |
| mu x 2 | 0.119280 |
| mu x 5 | 0.040528 |
| branch lengths x 0.5 | 0.663574 |
| branch lengths x 2 | 0.119280 |
| callable tips {S0, S1, S2, S3} | 0.308374 |
| callable tips {S0, S1, S2} | 0.733630 |
| callable tips {S0, S3} | 0.571740 |

The correction is therefore not a constant: it changes with the rate, with every node time, with the topology
and with which samples are callable at the site.  A fixed multiplier would not be an ascertainment correction
at all, and none is used anywhere in this work.

The same script also measures what ignoring the correction does, on three variable patterns whose full-length
data have 17 constant sites:  **[N]**

| Analysis | Maximum-likelihood rate (per site per day) |
|---|---|
| Full-length data (3 variable + 17 constant sites) | 2.92e-05 |
| Variable sites only, no correction | 2.99e-04 |
| Variable sites only, conditioned (C3) | at the lower search bound (1e-08) |

The naive analysis inflates the rate by an order of magnitude, as expected, since it treats a sequence of
three sites as if every site had mutated.  The conditional likelihood is monotonically decreasing in the rate
over seven orders of magnitude (profile in the script's output: -12.619 at mu = 1e-7 to -15.812 at
mu = 1e-3), which is the second, less obvious lesson: with the number of removed sites unknown, the data
carry almost no information about the absolute rate, and the conditional posterior is dominated by the rate
prior.  Mode C3 is not merely expensive to implement; its target is weakly identified.

## 5. What is implemented

### 5.1 Lossless modes: variant lists and callable sites  (new)

`--v0-human-in-variants FILE` reads a tab-separated table with a header and the columns `sample_id`, `site`
(1-based, reference coordinates) and `base` (one of A, C, G, T).  The sample set is the metadata table, so
that samples without a single variant are present and every sample has a declared date; the reference comes
from `--v0-human-reference-fasta`.  At least one of

* `--v0-human-callable-file FILE` --- sites examined in every sample, in the format of the site mask, and
* `--v0-human-sample-callable-file FILE` --- one line per sample, `<sample_id><TAB><sites>`, where `<sites>`
  is a comma-separated list of 1-based inclusive sites or ranges, or `none`

is required.  Both declarations are also available for FASTA and MAPLE input, where they override the
characters: a site that no declaration lists is missing in that sample, whatever the alignment says.  A call
that *differs from the reference* at a site the declaration does not list is a contradiction and is refused
(a reference-matching call at such a site cannot be distinguished from "not examined" in a FASTA file, so it
is silently made missing, and the count is reported).  Deliberate exclusion of sites remains the job of
`--v0-human-mask-file`, which drops calls and counts them as `masked_overrides`.

### 5.2 SNP-only completion (C1)  (new)

`--v0-human-snp-only-constant-sites A=..,C=..,G=..,T=..` declares the sites that the variability filter
removed, by the state that every sample shared there.  They are appended as a block of constant sites after
the input reference, so that the analysed sequence has `num_input_sites + sum_a n_a` sites.  The block is
callable in every sample and carries no reference coordinates: output mutations at sites beyond
`num_input_sites` refer to it, which the audit report and a start-up warning both state.  The declaration
encodes two assumptions, and the audit says so:

1. the removed sites were callable in every sample --- the counts cannot express per-sample missingness, and
   a filter that removed sites *with* missing data loses that information irrecoverably;
2. the counts are of the state shared at the removed sites, and sites that are still in the input are not in
   the counts (a warning fires when the input itself contains sites observed as invariant).

A third assumption is enforced rather than stated: the restored block has no sequence context, so the
completion is refused together with `--v0-mpox-hack`, whose two partitions are assigned from the context of
each site.  With one partition --- the default --- and with site rate heterogeneity (where the `nu_l` are
independent draws from one prior) the exchangeability argument holds; a future per-partition input (Task 3.4)
would need the counts per partition.

The completion is exact under those assumptions; it is not an approximation of C3 and does not pretend to be.
When the assumptions do not hold, the right input is the full-length or variant-plus-callable form.

### 5.3 What is refused

| Input | Reason |
|---|---|
| Variable-sites-only alignment with no site observed as invariant and no declaration | No ascertainment handling; the message names the three ways forward |
| Variant list without a callable declaration | The observation is undetermined; filling in the reference would invent data |
| A non-reference call outside the callable declaration | Data and declaration disagree |
| `base` that is not a single A, C, G or T: `N`, `?`, `-`, IUPAC codes, multi-base alleles | Not called, mixed, or not a substitution; the message says to use the callable declaration or the mask |
| Variant at a site outside the reference, duplicate or contradictory records, unknown sample ID, reference-matching variant | Inconsistent input |
| Only the *number* of removed constant sites (C2) | The composition is needed; guessing it would change the model |
| Completion together with `--v0-mpox-hack` | The restored block has no sequence context, so it cannot be assigned to the hack's context-defined partitions |
| Conditional likelihood of variable sites (C3) | Not implemented; see section 6 |

Indels, heteroplasmy, copy-number and structural variants are not modelled anywhere in Delphy, and nothing in
this task changes that: in an alignment they are missing data (gaps and ambiguity codes, counted and warned
about by the Task 1.2 path), and in a variant list they are refused outright with a message that names the
alternative.  Data from a region that may recombine do not satisfy the single-tree assumption that the shared
timed tree encodes; Delphy cannot detect recombination, so such regions must be excluded with the mask or left
out of the callable sites, and section 9 says so in the preprocessing recipes.

## 6. Why the C3 correction is not in the sampler  [P]

Adding `- m * log(1 - P(constant | theta))` to the target is mathematically straightforward.  Putting it into
*this* sampler is not, for reasons that are properties of Delphy's architecture rather than of the statistics:

1. **The correction is a tree-wide quantity, and Delphy's moves are local.**  Every local move updates
   `log_G_` by an increment computed from the branches it touched; the normalizer changes with every node
   time, so a correct implementation recomputes a Felsenstein pruning over the whole tree per move, which is
   O(tips) per move against the current O(1).
2. **It needs a computational kernel Delphy does not have.**  There is no transition-probability matrix
   anywhere in the core: the EMAT formulation never needs one.  The correction needs `exp(Q t)` per branch.
3. **With site rate heterogeneity it is per-site.**  The normalizer depends on `nu_l`, which is a sampled
   per-site parameter when `--v0-site-rate-heterogeneity` is on, so the cost becomes O(sites x tips).
4. **With per-sample callability it is per-pattern.**  Each distinct set of callable samples needs its own
   normalizer (section 4, table), so real chrY data would need one pruning per callability class.

There is an architecturally honest route, not taken here: the whole-batch Metropolis-Hastings correction built
for the exact Kingman prior (`Run::run_corrected_local_batch`, `exact_coalescent.md`) accepts a batch
of local moves with the ratio of a global factor before and after the batch, and an ascertainment factor is
such a global factor.  It would make the target exact in the restricted configuration that correction already
requires (fixed topology, one sampling date, exponential population model, pinned grid), at one pruning per
batch plus one per global parameter move.  Doing it would require: the pruning kernel, its validation against
an independent implementation, the correction's inclusion in every global move's ratio (`mu`, the population
parameters, the joint scale move), checkpoint support, and a demonstration that the batch acceptance rate
stays usable.  That is a task of its own, and section 4 shows the resulting target is weakly identified
anyway, so the honest first step is the explicit refusal that is implemented now.

## 7. Validation rules and their limits

The validator is the Task 1.2 path (`finalize`, `core/human_input.cpp:517`) extended by this task.  Rules that
are new here:

* callable declarations: empty declaration, out-of-range or reversed ranges, a sample missing from the
  per-sample file, a sample in the file that is not in the input, a sample declared twice, a non-reference
  call outside the declaration, and a warning counting calls that failed *inside* the declared region;
* variant tables: missing or duplicate columns, empty file, field-count mismatch, non-integer or out-of-range
  site, duplicate record, contradictory records, reference-matching record, unknown sample, and the five
  classes of unsupported `base` values;
* constant-site declarations: malformed specification, a missing or repeated state, negative counts, all-zero
  counts, and the warning about invariant sites already present in the input.

**The detector of variable-sites-only input is a heuristic and stays one.**  It fires when no site at all is
observed and equal to the reference everywhere, which is what rule R2 of section 4 produces.  An input that
keeps a handful of non-polymorphic sites passes it: in the example data of
`doc/extended/examples/observation`, the R2-filtered file (`snps_only.fasta`, 126 of 300 sites) is
*not* detected, because the sites it keeps for having missing data include some that are invariant and equal
to the reference, while the strictly polymorphic file (`polymorphic_only.fasta`, 34 sites) is refused.  Both
cases are checked in `tests/cli/run_observation_cli_tests.sh`.  **[R]**  Ascertainment cannot be
detected from the data in general; what protects a user is the audit report (which now states the observation
mode, the two lengths and the four site classes) and the explicit declarations, not this check.

## 8. Command-line reference (new options)

| Option | Meaning |
|---|---|
| `--v0-human-in-variants FILE` | Variant-list input (requires `--v0-human-metadata`, `--v0-human-reference-fasta` and a callable declaration) |
| `--v0-human-callable-file FILE` | Sites examined in every sample; everything else is missing everywhere |
| `--v0-human-sample-callable-file FILE` | Sites examined in each sample, one line per sample |
| `--v0-human-snp-only-constant-sites SPEC` | Counts of the removed constant sites by state, e.g. `A=4000,C=3000,G=3100,T=2400` |

All four are inert without `--v0-human-mode`, as the other human options are.  Example data and runnable
commands: `doc/extended/examples/observation` and `tests/cli/run_observation_cli_tests.sh`.

```bash
E=doc/extended/examples/observation
# variant list plus per-sample callable sites
build/release/delphy --v0-human-mode --v0-human-in-variants $E/variants.tsv \
  --v0-human-reference-standard custom --v0-human-expected-length 300 \
  --v0-human-reference-fasta $E/reference.fasta --v0-human-metadata $E/metadata.tsv \
  --v0-human-sample-callable-file $E/sample_callable.tsv --v0-human-max-missing-fraction 0.5 \
  --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8 \
  --v0-seed 1 --v0-threads 1 --v0-steps 200 --v0-human-audit-report variants.json
# variable-sites-only alignment, completed from the declared counts
build/release/delphy --v0-human-mode --v0-in-fasta $E/polymorphic_only.fasta \
  --v0-human-reference-standard custom --v0-human-expected-length 34 --v0-human-reference-id EXREF \
  --v0-human-metadata $E/metadata.tsv \
  --v0-human-snp-only-constant-sites "$(cat $E/polymorphic_only.counts)" \
  --v0-human-max-missing-fraction 0.5 --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8 \
  --v0-seed 1 --v0-threads 1 --v0-steps 200 --v0-human-audit-report completed.json
```

## 9. Preprocessing recipes for single-tree sequence data

These are general rules; nothing in the code knows about any species, chromosome or coordinate.  A data set is
admissible when all of its sites share one timed tree, which means no recombination, one copy per sample and
unambiguous calls.

1. **Choose the coordinate system and keep it.**  Supply the full-length reference of the analysed region in
   its own coordinates (`--v0-human-reference-fasta`), declare its length, and express every site --- variants,
   masks, callable regions --- in those coordinates.  The audit report records the reference's FNV-1a checksum,
   which is what makes two runs comparable.
2. **Declare what was examined, not what was found.**  Produce the callable regions from coverage and mapping
   quality, before looking at the variants, and pass them with `--v0-human-callable-file` (one declaration for
   all samples) or `--v0-human-sample-callable-file` (per sample).  Never convert "no variant call" into a
   reference base for a site that was not examined.
3. **Exclude problem regions by site, not by call.**  Repeats, paralogous or palindromic regions, homopolymers
   and anything that may recombine go into `--v0-human-mask-file` (or out of the callable sites) for *every*
   sample.  Dropping single calls because of their state is informative missingness (section 3).
   For mitochondrial data this typically means the control-region homopolymers and the known
   alignment-ambiguous stretches; for chrY it means the heterochromatic, ampliconic and palindromic regions
   and anything outside the callable "X-degenerate" part.  Delphy ships no such list: the mask is an input,
   and a published mask with a version is part of a reproducible analysis.
4. **Resolve or drop ambiguity.**  Heteroplasmy, mixed calls, indels, copy-number and structural variants are
   not modelled.  Either resolve the call to one base, or declare the site not callable in that sample.  In a
   variant list all of these are refused with a message; in an alignment the IUPAC and gap characters become
   missing data and are counted in the audit.
5. **Keep the constant sites.**  Full-length input or a variant list plus callable sites is always preferable.
   If only a variable-sites alignment exists, recover the counts of the removed constant sites from the
   reference over the analysed, callable region and pass them with `--v0-human-snp-only-constant-sites`; check
   that the filter did not also remove sites with missing data, because the counts cannot express that.
6. **State the clock information separately.**  The observation model says nothing about time: ancient sample
   dates, calibrations or an informative rate prior remain the only sources of an absolute time scale
   (`input_contract.md` section 6, `calibration.md`).

## 10. Verification

Commands, all run on this commit (gcc 15.2.0, Linux 6.18 WSL2, 24 cores):

```bash
cmake --build --preset conan-release -j 20 && ./build/release/tests/tests
cmake --build --preset conan-debug  -j 20 && ./build/debug/tests/tests          # ASan
tests/cli/run_observation_cli_tests.sh build/release/delphy WORKDIR
tests/cli/run_pathogen_regression.sh ...                               # see the task report
python3 -I tests/prototypes/check_snp_ascertainment.py
python3 -I tests/human_input/make_observation_example.py doc/extended/examples/observation
```

New tests (`tests/human_observation_tests.cpp`, 23 cases): format equivalence (variant list against alignment,
tips, site classes and likelihood), never-observed sites, the reference-base invariance, callable declarations
and their rejections, the variant-table validator, the SNP-only completion (site patterns, likelihood,
rejections, specification parsing), and four sampler cases --- incremental against from-scratch `log_G` on
completed and on callable-declared input, each with fixed and with free topology, plus a checkpoint that
restores completed input exactly.  Results are in the task report.

## 11. Open limitations

1. Mode C3 (variable sites, nothing known about the removed ones) is refused, not implemented.  Section 6
   states what implementing it would take and section 4 shows its target is weakly identified.
2. Mode C2 (number of removed sites known, composition unknown) is refused; the composition must be supplied.
3. The SNP-only completion assumes the removed sites were callable in every sample.  When a filter removed
   sites that had missing data, that information is lost and the completion overstates the data slightly; the
   audit report says so, and the fix is better input, not a correction.
4. The detector of variable-sites-only input is a heuristic (section 7).
5. MAPLE input is lossless only if its producer declared every non-called site; Delphy cannot check this.
6. A callable declaration against a FASTA file cannot detect a reference-matching call at a non-callable site;
   such a site is made missing and counted.
7. Informative missingness (per-call filters, reference bias in mapping) is not modelled and cannot be
   detected; section 3.
8. The audit report's format version is now 2.  Keys were added (`observation_mode`, `num_input_sites`,
   `not_callable`, `invariant_nonreference`, `callable_declaration`, `constant_sites_appended`, and
   `callable` per sample); no key was removed or changed in meaning, so a reader of version 1 that ignores
   unknown keys still works.  The `.dphy` format and the public API are untouched.
