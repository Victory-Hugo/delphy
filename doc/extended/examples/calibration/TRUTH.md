# Generating parameters of this synthetic example

These files are generated test data.  Nothing here describes real human mtDNA, and none of these
values is a default anywhere in Delphy.

| Quantity | Value |
|---|---|
| sites | 300 |
| samples | 6 (SAMPLE_A .. SAMPLE_F) |
| sampling date | 2020-01-01 for every sample (no time signal in the data) |
| true substitution rate | 1.6e-7 substitutions / site / year (1 year = 365 days) |
| true topology | `((((SAMPLE_A,SAMPLE_B),SAMPLE_C),(SAMPLE_D,SAMPLE_E)),SAMPLE_F)` |
| true node ages (years before 2020-01-01) | root 140,000; M 100,000; N 60,000; (A,B) 30,000; (D,E) 70,000 |
| mutations placed | 21 events on the true history |

`calibration.cal` calibrates the three nodes as follows; the ages are the user's input for this
example and are centred on the generating values on purpose, so that the example also serves as a
recovery check:

| calibration | sample set | units | prior |
|---|---|---|---|
| root | all 6 samples (the root of `tree.nwk`) | ka before 1950 | lognormal, median 140 ka, sigma 0.15, hard bounds 60-400 ka |
| M | A, B, C, D, E | ka before 1950 | normal, mean 100 ka, stddev 12 ka |
| N | A, B, C | cal BP | uniform, 45,000-75,000 |

With 300 sites and 21 mutations this example carries very little information: it exists to show the
workflow and to be cheap, not to demonstrate statistical accuracy.  The quantitative validation is
the simulated matrix in `tests/human_input/run_calibration_matrix.sh`.

## Clade summary example (Task 2.3)

`clades.tsv` names the clades that `delphy_clades` summarizes for this example: the three that
`calibration.cal` calibrates (`root`, `M`, `N`) and two that it does not (`AB` = SAMPLE_A + SAMPLE_B,
true age 30,000 yr; `DE` = SAMPLE_D + SAMPLE_E, true age 70,000 yr).  Their ages are inferred from the
genealogy and the other calibrations, which is what the two uncalibrated clades are there to show.

```bash
E=doc/extended/examples/calibration
# ... run several chains with --v0-out-trees-file, then:
build/release/delphy_clades \
  --trees chain1.trees --trees chain2.trees --trees chain3.trees --trees chain4.trees \
  --clades $E/clades.tsv --calibration-file $E/calibration.cal \
  --tip-dates $E/metadata.tsv --age-reference-year 1950 \
  --out-summary summary.tsv --out-samples samples.tsv --out-per-chain per_chain.tsv
```

With four chains of 2e7 steps (fixed coalescent timescale of 50,000 yr, 4,000 grid cells, 30% burn-in,
5,600 pooled trees) the pooled posterior was:

| clade | calibrated | true age (yr before 1950) | median | 95% HPD | ESS | rank split R-hat |
|---|---|---|---|---|---|---|
| root | yes | 140,070 | 129,012 | 100,518 -- 165,336 | 5,155 | 0.9998 |
| M | yes | 100,070 | 93,251 | 73,652 -- 114,452 | 5,061 | 1.0012 |
| N | yes | 60,070 | 52,288 | 45,000 -- 68,575 | 5,202 | 1.0001 |
| AB | no | 30,070 | 15,203 | 2,072 -- 34,780 | 5,397 | 0.9999 |
| DE | no | 70,070 | 28,138 | 7,332 -- 55,314 | 5,458 | 0.9998 |

The three calibrated ages and `AB` cover their generating values; `DE` does not.  With 300 sites and
21 mutation events the two uncalibrated nodes carry almost no information, the posterior support of
every clade is 1 because the topology is fixed, and this example exists to show the workflow, not to
demonstrate accuracy.  The quantitative check is the simulated campaign in
`tests/human_input/run_clade_summary_validation.sh`.
