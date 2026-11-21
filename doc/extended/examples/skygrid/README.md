# A Skygrid configuration for human uniparental data

Every number below is derived from a stated assumption about the analysis, not taken from a Delphy
default.  Delphy's Skygrid defaults were chosen for pathogens, and three of them are actively wrong
on a human time axis: the knot range has no usable default at all, the prior smoothness is set for a
curve that fluctuates over weeks, and the low-population barrier sits six orders of magnitude below
any plausible human coalescent timescale.  In `--v0-human-mode` Delphy now refuses to start unless
the smoothness is stated, and warns about the barrier.

Nothing here is a statement about real human mtDNA or chrY history.

## The four choices, and how to make them

Write `K` for the knot range in years (`--v0-skygrid-cutoff`), `P` for the number of parameters
(`--v0-skygrid-num-parameters`), and `dt = K / (P - 1)` for the knot spacing.

### 1. Knot range `K`

The range must cover the part of the genealogy the data can speak about: from the youngest tip back
past the expected root.  For a modern mtDNA panel whose root is expected somewhere around 150-250 ka,
a range of 400 ka leaves room above the expectation without asking the curve to describe times no
lineage reaches.  Times older than `x_0` are not unmodelled: `Skygrid_pop_model` holds
`log N(t) = gamma_0` over the whole open-ended first interval, so the choice of `x_0` decides where
the curve stops being allowed to bend, not where it stops existing.

```
--v0-skygrid-cutoff 400000
```

### 2. Number of knots `P`

`P` is a model choice, not a numerical one: more knots is a different, more flexible prior, not a
more accurate approximation.  The useful question is how many intervals contain enough coalescences
to be informed.  With `n` same-date samples and a coalescent timescale `N`, roughly `2N` years of
tree depth carry `n - 1` coalescences, most of them in the recent part, so intervals much narrower
than a few thousand years are prior-dominated for any realistic panel.  A spacing of 10 ka over a
400 ka range gives

```
--v0-skygrid-num-parameters 41          # P - 1 = 40 intervals, dt = 10,000 years
```

Report the posterior of `gamma_k` interval by interval and treat intervals whose posterior is
indistinguishable from the smoothing prior as uninformed, rather than reading them as inferred
history.  `10_coalescent_report.md` section 7 records how this behaves on simulated data.

### 3. Prior smoothness

Delphy parametrizes the Gaussian Markov random field on `log N(t)` by a "double-half time" `T`: the
timescale over which the a priori curve changes by a factor of two, one standard deviation.  The
precision it implies is

```
tau = T / (log^2(2) dt)
```

so what matters is `T / dt`, and the default `T = 0.0822` years (30 days) against `dt = 10,000` years
gives `tau = 1.7e-5` -- a prior standard deviation of 240 log units between adjacent knots, which is
no smoothing at all.  Setting `T` equal to the knot spacing gives `tau = 2.08`, a prior standard
deviation of 0.69 log units, i.e. about a factor of two per interval:

```
--v0-skygrid-prior-double-half-time 10000
```

The alternative is to infer the smoothness, which is BEAST's behaviour:

```
--v0-skygrid-infer-prior-smoothness true \
--v0-skygrid-tau-prior-alpha 0.001 --v0-skygrid-tau-prior-beta 0.001
```

Inferring it is the safer choice when the amount of signal per interval is unknown, at the cost of a
hyperparameter whose own posterior has to be checked.  Fixing it makes the smoothness an explicit,
reportable assumption.  Do one or the other and say which.

### 4. Population scale and its floor

`N(t)` in Delphy is a coalescent timescale with the dimension of time, `N_e` multiplied by a
generation time.  It is **not** a number of individuals, and turning it into one needs a generation
time, a ploidy and, for a uniparental locus, a sex-specific factor -- none of which Delphy has.
State the prior on the geometric-mean scale in years:

```
--v0-skygrid-nbar-prior-mean 75000 --v0-skygrid-nbar-prior-stddev 75000
```

and put the floor somewhere meaningful on that axis instead of leaving it at one day:

```
--v0-skygrid-low-pop-barrier-loc 100
```

or state that no floor is wanted, with `--v0-skygrid-disable-low-pop-barrier true`.

### 5. The coalescent integration grid is a separate matter

`--v0-target-coal-prior-cells` is not part of the population model; it is the numerical grid on which
the coalescent integral is evaluated, and the default of 400 cells is too coarse for a human panel of
more than a few hundred samples.  See `coalescent_grid.md` sections 1 and 4; the short
rule, for up to 3,000 samples, is at least `max(400, 3 n)` cells; it does not hold at 10,000, and in
human mode Delphy prints the measured discretization excess so the choice can be checked.

```
--v0-target-coal-prior-cells 4000 --v0-coal-prior-grid-tolerance 1.0
```

## Worked command

The absolute time scale of a Skygrid run on modern-only samples comes from the substitution-rate
prior, since there is no sampling-time signal and age calibrations require a fixed topology.  State
the rate prior explicitly and give the initial rate, or the tree starts at Delphy's pathogen default
of 1e-3 substitutions per site per year.

```bash
delphy --v0-human-mode \
  --v0-in-fasta samples.fasta \
  --v0-human-reference-standard rCRS --v0-human-reference-fasta rCRS.fasta \
  --v0-human-metadata metadata.tsv --v0-human-mask-file mask.txt \
  --v0-human-uniform-sample-date 2020-01-01 \
  --v0-mu-prior-mean 1.7e-8 --v0-mu-prior-stddev 5.1e-9 --v0-init-mutation-rate 1.7e-8 \
  --v0-pop-model skygrid --v0-skygrid-type log-linear \
  --v0-skygrid-cutoff 400000 --v0-skygrid-num-parameters 41 \
  --v0-skygrid-prior-double-half-time 10000 \
  --v0-skygrid-nbar-prior-mean 75000 --v0-skygrid-nbar-prior-stddev 75000 \
  --v0-skygrid-low-pop-barrier-loc 100 \
  --v0-target-coal-prior-cells 4000 --v0-coal-prior-grid-tolerance 1.0 \
  --v0-out-trees-digits 12 \
  --v0-seed 1 --v0-threads 8 --v0-steps 400000000 \
  --v0-log-every 400000 --v0-out-log-file run.log \
  --v0-out-trees-file run.trees --v0-tree-every 400000
```

The `skygrid.logPopSize*` columns of `run.log` carry `gamma_k` in log years, most recent knot first,
and `skygrid.precision` carries `tau`.  Summarize target clade ages from `run.trees` with
`delphy_clades` (Task 2.3).

## Checking the configuration before trusting it

1. Read the grid line Delphy prints at startup and confirm the discretization excess is small
   relative to the per-proposal distortion the report tabulates.
2. Run several chains from different seeds and compare `skygrid.logPopSize*` interval by interval,
   not only the root age: the smoothing prior can make chains agree on the summary while disagreeing
   on the curve.
3. Compare against a fixed-population run.  If the Skygrid curve is flat to within its own
   uncertainty, the extra parameters have bought nothing and the fixed-population result is the one
   to report.
4. Never read an interval older than the oldest coalescence as inferred history.

`tests/human_input/run_coalescent_models.sh` runs this configuration against simulated data with a
known constant history, which is the prior-dominance check the fourth point asks for.
