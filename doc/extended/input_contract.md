# Input contract of the opt-in sequence path (sequences and sampling-time metadata)

This document is the contract of the opt-in input path for human mtDNA alignments.  The pathogen input path
(`--v0-in-fasta` / `--v0-in-maple` without `--v0-human-mode`) is unchanged.

## 1. Scope

In scope: complete mtDNA alignments (about 16,569 sites) as FASTA, or reference-plus-differences as MAPLE; sampling
times from a separate metadata table; ancient-DNA date conventions; deterministic quality checks and an audit report;
review of float32 time precision (R-02).

Out of scope (later tasks): identifiability diagnostics and priors (1.3), topology import and fixed topology (2.1),
calibration priors (2.2), chrY callable-site models and ascertainment correction (3.2), changes to the `.dphy` format.

Task 3.2 extended this path with callable-site declarations, variant-list input and the completion of
variable-sites-only input from declared constant-site counts, and raised the audit report to version 2.  Where
this document and `observation.md` differ, the later one holds; the options and semantics described
here are unchanged.

## 2. Command line

All options are inert unless `--v0-human-mode` is given; giving any of them without it is an error.

| Option | Meaning |
|---|---|
| `--v0-human-mode` | Enable the path.  Requires `--v0-in-fasta` or `--v0-in-maple`. |
| `--v0-human-reference-standard rCRS\|RSRS\|custom` | Required.  Declares the coordinate system.  rCRS and RSRS fix the length at 16569. |
| `--v0-human-expected-length N` | Required for `custom`; must equal 16569 for rCRS / RSRS if given. |
| `--v0-human-reference-fasta FILE` | FASTA input: file with exactly one reference record. |
| `--v0-human-reference-id ID` | FASTA input: ID of the reference record inside the input FASTA (it is removed from the samples). Exactly one of the two reference options is required for FASTA; neither is allowed for MAPLE. |
| `--v0-human-metadata FILE` | Tab-separated sample metadata (section 3). |
| `--v0-human-uniform-sample-date DATE` | ISO date (`YYYY-MM-DD`, `YYYY-MM`, `YYYY`) for every sample without its own date. |
| `--v0-human-mask-file FILE` | Sites treated as missing in all samples (section 5). |
| `--v0-human-max-missing-fraction F` | Reject the input if a sample exceeds this fraction of missing sites; default 0.10; must be in (0, 1]. |
| `--v0-human-audit-report FILE` | Write the JSON audit report (section 7). |

The reference standard is a declaration checked against the length; Delphy does not embed rCRS / RSRS sequences, so the
reference file is supplied by the user and its identity is recorded as an FNV-1a 64-bit checksum in the audit report.

Example (data in `doc/extended/examples/minimal/`, 5 synthetic samples of 300 sites):

```bash
E=doc/extended/examples/minimal
build/release/delphy --v0-human-mode --v0-in-fasta $E/samples.fasta \
  --v0-human-reference-standard custom --v0-human-expected-length 300 \
  --v0-human-reference-fasta $E/reference.fasta --v0-human-metadata $E/metadata.tsv \
  --v0-human-mask-file $E/mask.txt --v0-human-audit-report audit.json \
  --v0-seed 1 --v0-threads 1 --v0-steps 2000 --v0-out-log-file ex.log --v0-out-trees-file ex.trees
```

Larger synthetic inputs: `python3 -I tests/cli/make_synthetic_mtdna.py OUTDIR --samples 1000 --sites 16569
--seed 1 --dates uniform`.  The generated data are random and have no biological meaning.

## 3. Sample identifiers and metadata schema

Sample ID: the first whitespace-delimited token after `>` in FASTA and MAPLE.  The rest of the header line is ignored.
IDs must be unique, also ignoring letter case, and may not contain control characters or `( ) , : ; [ ] ' " \`.  The
metadata is matched to sequences by exact ID.  Every sequence needs a metadata row when a metadata file is given, and
every row needs a sequence; mismatches are errors, never silent drops.  Tips are named by sample ID and keep input order.

Metadata file: UTF-8/ASCII text, one header line, tab-separated, `#` lines and blank lines ignored, `\r\n` accepted.

| Column | Required | Content |
|---|---|---|
| `sample_id` | yes | Sequence ID |
| `date_convention` | no | `iso`, `ce`, `bce`, `bp`, `cal_bp`, `c14_uncal_bp`; empty means "use the uniform sampling date" |
| `date_from` | with a convention | Older bound, or the only value, in the units of the convention |
| `date_to` | no | Younger bound; empty = point (year conventions) or the whole period (`iso`) |
| `bp_reference_year` | for `bp` | Baseline year CE; empty or 1950 for `cal_bp`; must be empty otherwise |

Other columns are ignored and listed in the audit report.  A date without a `date_convention` is an error: dates are
never interpreted implicitly.

```
sample_id	date_convention	date_from	date_to	bp_reference_year
SYN00000	cal_bp	3000	2920
SYN00001	iso	2019-03-27
SYN00002	bce	1500
SYN00003	bp	8000		1950
SYN00004	ce	1200	1300
```

## 4. Date conventions and conversion

Internal time is days since 2020-01-01 (proleptic Gregorian calendar, astronomical year numbering internally).

| Convention | `date_from` / `date_to` | Conversion |
|---|---|---|
| `iso` | `YYYY-MM-DD`, `YYYY-MM`, `YYYY`, year 0001-9999 | Same as the existing date suffix: a month or year spans the whole period; `date_to` extends the upper bound |
| `ce` | integer year >= 1 | 00:00 on 1 January of that year |
| `bce` | integer year >= 1 (1 BCE = astronomical year 0) | 00:00 on 1 January of astronomical year `1 - N` |
| `bp` | integer age >= 0 | 1 January of year `bp_reference_year - age`; the baseline year is mandatory |
| `cal_bp` | integer calibrated age >= 0 | As `bp` with baseline fixed at 1950; any other baseline is an error |
| `c14_uncal_bp` | -- | Always rejected: an uncalibrated radiocarbon age is not a calendar age and Delphy does not calibrate. The error tells the user to calibrate externally and supply `cal_bp` |

Rules: no year 0 in `ce` / `bce`; the older bound goes in `date_from` (for `bp` / `cal_bp` this is the larger number) and
a reversed range is an error; year-valued conventions have a resolution of one year, which is far below the uncertainty
of ancient dates; values must be integers with |value| <= 10,000,000.  Because 1 BCE (astronomical year 0) is a leap
year, `ce 1` minus `bce 1` is 366 days; tests cover this and 400-year calendar cycles at -40,000, -1,200 and 1,600.

The conventions describe the sampling age of a sample.  Archaeological date uncertainty is expressed as the
`date_from..date_to` interval, which Delphy treats as a uniform prior on the tip time (existing behaviour of
`tip_displace_move`).  Internal-node calibrations are a different concept and are not part of this input path (Task 2.2).

## 5. Reference, full length, invariant sites, missing data, masks

* The reference must consist of A, C, G, T only.  An ambiguous reference base is an error (the pathogen MAPLE reader
  silently turns it into `A`; this path does not).
* Every sequence must have exactly the reference length (alignment columns are reference coordinates).  The length is
  compared with the declared standard.  A trimmed alignment declared as rCRS is rejected.
* SNP-only protection: the reference is always full length, and after processing at least one site must be observed
  in some sample and equal to the reference everywhere (an observed invariant site).  If none exists the input is
  rejected as a SNP-only or variable-sites-only alignment.  (Task 3.2: the check is unchanged and is a heuristic;
  such data can now be analysed when the removed constant sites are declared as counts, and the conditional
  likelihood of a variable site is still not implemented.  See `observation.md`.)
* Site classes reported in the audit: variable, invariant-observed, never-observed (missing or masked in every sample).
  Never-observed sites are excluded from the likelihood by the existing root-missation mechanism and are not counted as
  invariant.
* `N`, `?`, `-`, `.` and IUPAC ambiguity codes become missing data, never reference states.  Gaps are not modelled as
  deletions and heteroplasmy is not modelled; both are counted per sample and reported as warnings.  `U` and any other
  character are errors, with sample, line and alignment column.
* Mask file: one entry per line, `310` or `303-315` (1-based, inclusive), `#` comments.  Masked sites become missing in
  every sample (observed differences are removed and counted as `masked_overrides`).  Out-of-range or reversed
  entries are errors.  The mask is an input; Delphy ships no region list.
* Samples whose missing fraction exceeds the limit are rejected by name and percentage; the limit can be raised
  explicitly.
* Exact duplicate processed sequences are kept and reported as groups.

MAPLE: the first record is the full-length reference; tip lines are `<base> <site>` or `<N|-|?|IUPAC> <start> [<len>]`.
In this path a mutation equal to the reference, a repeated site, a site both mutated and missing, an out-of-range site
or an unparsable line is an error naming the sample and line (the pathogen reader drops the tip with a warning).

## 6. Time signal

All dates identical across samples (same `t_min` and `t_max`) sets `no_time_signal` and prints a prominent warning:
the data carry no information about the absolute time scale.  Samples that received the uniform date are counted.

The CLI refuses to start a run in that situation unless the substitution rate is specified explicitly: a proper Gamma
prior (`--v0-mu-prior-mean` with `--v0-mu-prior-stddev`, or alpha and beta both > 0), or a fixed rate
(`--v0-fix-mutation-rate` with `--v0-init-mutation-rate`).  The pathogen default (improper uniform prior, initial
1e-3 subst/site/yr) is never silently used for such data.  When dates do vary, no threshold on the span is applied here;
judging whether the span is informative is Task 1.3.

## 7. Audit report

Always printed as a summary to stderr; JSON with `--v0-human-audit-report`.  Key order and number formatting are fixed,
so identical inputs give byte-identical reports.  Task 3.2 raised `report_version` to 2 and added keys
(`observation_mode`, `reference.num_input_sites`, `sites.not_callable`, `sites.invariant_nonreference`,
`callable_declaration`, `constant_sites_appended`, and `callable` per sample); nothing was removed or changed in
meaning.  The version 1 key set was: `report_version` (1), `input_format`, `reference`
(`standard`, `id`, `num_sites`, `checksum_fnv1a64`), `num_samples`, `sites` (`masked`, `variable`,
`invariant_observed`, `never_observed`), `time` (`dates_from_metadata`, `dates_from_uniform`, `earliest_t_min_days`,
`latest_t_max_days`, `midpoint_span_days`, `no_time_signal`), `ignored_metadata_columns`, `warnings`,
`duplicate_groups`, `samples[]` (`sample_id`, `date_source`, `date`, `t_min_days`, `t_max_days`, `differences`,
`missing`, `missing_fraction`, `masked_overrides`, `iupac_ambiguous`, `unknown`, `gap`, `duplicate_group`).  The three
character counts are -1 for MAPLE input, which does not carry them.

## 8. Error reporting

Input problems are collected and reported together (up to 25 shown), then the program exits non-zero before any
tree is built.  Examples from the example data (real output):

```
ERROR: Human input rejected: 6 problems found
  1. 1 metadata row has no sequence in the sequence file: GHOST
  2. Metadata line 2 (sample 'SYN00000'): Convention 'c14_uncal_bp' is an uncalibrated radiocarbon age, not a calendar age, and is rejected.  Calibrate it externally (for example with OxCal or IntCal) and supply the calibrated range with convention 'cal_bp', date_from = older bound, date_to = younger bound.
  3. Metadata line 3 (sample 'SYN00001'): Convention 'bp' requires bp_reference_year (the baseline year CE, for example 1950).  If the ages are calibrated radiocarbon ages, use convention 'cal_bp' instead.
  4. Metadata line 4 (sample 'SYN00002'): date_from 0 is not a valid BCE year.  BCE years start at 1 (the year before 1 CE); there is no year 0.
  5. Metadata line 5 (sample 'SYN00003'): date_from (3100) is more recent than date_to (3200).  date_from must be the older bound; for 'bp' and 'cal_bp' the older bound is the LARGER number.
  6. Metadata line 6 (sample 'SYN00004'): date_from '2019-02-30' is not a valid calendar date.

ERROR: Human input rejected: 1 problem found
  1. Reference 'SYNTH_REF' has 300 sites but the declared standard (rCRS) has 16569.  A trimmed, SNP-only or differently numbered alignment cannot be analysed as a full-length sequence; supply the full-length alignment in reference coordinates.

ERROR: Human input rejected: 1 problem found
  1. 5 samples have no sampling date: SYN00000, SYN00001, SYN00002, SYN00003, SYN00004.  Give each a date_convention / date_from in the metadata, or set a uniform sampling date explicitly.

ERROR: All samples have the same sampling date, so the data do not determine the absolute time scale, and the default mutation-rate prior and initial rate are pathogen values.  Specify the rate explicitly: ...
```

## 9. Numeric precision review (risk R-02)

Facts (code at the base commit): `Tip_desc` / `Phylo_node` `t_min`, `t_max` are `float`; `.dphy` (`api.fbs`) stores node
and mutation times and tip bounds as float32; in-memory node and mutation times are `double`.

float32 spacing: 0.25 day for |t| in [2^21, 2^22) days (about 5,700-11,500 years), 1 day in [2^23, 2^24)
(23,000-46,000 years), 2 days in [2^24, 2^25), 4 days in [2^25, 2^26).  Integer-day tip bounds are exact up to 2^24 days.

Decisions in this task (no change to `Tip_desc`, `Phylo_node`, `api.fbs` or the `.dphy` format):

1. Tip bounds with |t| > 2^25 days (about 92,000 years from 2020) are rejected with an explanatory error, which
   guarantees a rounding error of at most 1 day on any accepted bound.  The rule depends on magnitude only.
2. Conversions use `double` until the final store into `Tip_desc`; the old `read_maple` implicit narrowing is not used
   by this path.
3. Warnings are emitted when bounds are rounded by more than 0.01 day, and when the float32 resolution at the
   oldest sample exceeds 0.01 day (the slack used by `assert_phylo_tree_integrity`), because `.dphy` node and mutation
   times are then coarser than that slack.
4. Changing `.dphy` to float64 (or adding a double-precision side channel) needs a versioned migration and a test that
   old files load; it is deferred to a task that touches the format (before 2.x per the risk register).

Measurements (`Human_input_test.dphy_api_roundtrip_*`, 30 and 60 synthetic samples of 600 sites, initial tree from
`mp-plus-timing`, round trip through `phylo_tree_to_api_tree` / `api_tree_and_tree_info_to_phylo_tree`):

| Scenario | max abs t (days) | max node-time error (days) | order violations | violations beyond 1e-2 day |
|---|---|---|---|---|
| cal BP 9000-9290, ranges 10 years wide | 3,415,018 | 0.1167 | 0 | 0 |
| 60 samples all at cal BP 9000 | 3,313,066 | 0.1000 | 0 | 0 |

The error is bounded by half the float32 spacing (0.125 day) as predicted.  No ordering violation occurred in these
two trees; the risk that branches shorter than the spacing reverse order after reload remains and is not excluded by
two examples.  Tip bounds survived unchanged (they were exact days).

Date helpers: `to_iso_date`, `to_linear_year` and `parse_iso_*` were checked at astronomical years -48,000 to 1850
(verification item V-01 of the risk register); they behave correctly for negative years.

## 10. Tests

```bash
# unit and integration tests (Release and Debug/ASan builds from 01_build_baseline.md)
./build/release/tests/tests --gtest_filter='Human_*:*Human_input*'
./build/release/tests/tests
./build/debug/tests/tests
# CLI end-to-end at 5, 100 and 1000 samples, plus refusal cases
tests/cli/run_cli_scale_tests.sh build/release/delphy build/human_scale
```

Results of the run for this commit are in the Task 1.2 report.  Pathogen regression: fixed-seed `delphy` runs of the
Task 1.1 synthetic dataset (FASTA at 1 and 4 threads, MAPLE at 1 thread) were compared with a binary built from the base
commit; `.log` and `.trees` bodies were byte-identical.

## 11. Open limitations

* Rooting with identical sampling dates falls back to midpoint rooting with the pathogen fallback rate (12.17 mut/yr
  reported at start-up on the synthetic 1000-sample set, a viral value, risk R-06); initial tree ages are therefore
  not meaningful for such data.  Handled in 1.3.
* No check that the supplied reference really is rCRS or RSRS beyond length and checksum.
* No validation of the mask content against known problematic regions; no heteroplasmy model; gaps are missing data.
* The pathogen CLI aborts with an uncaught exception when no sequence has a date (unchanged legacy behaviour).
* float32 `.dphy` times: see section 9.
