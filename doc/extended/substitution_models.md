# The unified reversible nucleotide substitution model

This document defines the substitution model Delphy samples, the six-digit encoding of its 203
exchangeability structures, the frequency modes, the priors, the exact Metropolis-Hastings
differences, and the mapping to the source.  It is the design half of Task 3.5; the measured results
are in `38_substitution_model_report.md`.

Nothing here replaces Delphy's architecture: the explicit mutation-annotated tree (EMAT), the
incremental likelihood, the mutation-guided topology proposals and the Run/Subrun framework are
untouched.  The whole model enters through `Site_evo_model`, which every likelihood path already
reads.

## 1. The model

A site evolves under a four-state, time-homogeneous, reversible continuous-time Markov chain.  Six
symmetric exchangeabilities `s_ab = s_ba > 0` and stationary frequencies `pi_a > 0` with
`sum_a pi_a = 1` give the unnormalized generator

    Qt_ab = s_ab pi_b   (a != b),        Qt_aa = -sum_{b != a} Qt_ab,

which is normalized by

    c = -sum_a pi_a Qt_aa,               Q = Qt / c,

so that

    -sum_a pi_a Q_aa = 1.                                                   (1)

(1) is the convention that makes `mu` the mean number of substitutions per site per unit time: the
expected number of events at a site over a branch of length `t` is `mu nu_l r_p t`, where `r_p` is
the relative rate of the site's partition (Task 3.4) and `nu_l` its site rate.  The product of all
three is the only thing any likelihood ever uses, so an overall factor can be moved freely between
them; (1), the partition normalization `sum_p L_p r_p / sum_p L_p = 1` and the prior mean
`E[nu_l] = 1` are what pin the three factors separately.  In particular, scaling all six
exchangeabilities by a constant leaves `Q` unchanged, which is why one of them is held at 1.

`Q` is reversible: `pi_a Q_ab = s_ab pi_a pi_b / c = pi_b Q_ba`, and `pi` is its stationary
distribution.  The root sequence is drawn site by site from `pi`, which is what
`calc_log_root_prior` computes, so the whole process on the tree is reversible and the likelihood
does not depend on the position of the root beyond the root prior itself.

### 1.1 The augmented likelihood

Delphy does not integrate the mutational history out.  Given the timed tree, the history and the
parameters, the density of the augmented state is

    log G = sum_a n_a log pi_a                                      (root states)
          - sum_p mu r_p sum_a q_a Ttwiddle^p_a                     (exposure)
          + sum_{a != b} M_ab log(mu nu_l r_p Q_ab)                 (events)

with `q_a = -Q_aa`, `n_a` the number of root sites in state `a`, `M^p_ab` the number of `a -> b`
events in partition `p`, and `Ttwiddle^p_a = sum_{l in p} nu_l T^(l)_a` the site-rate-weighted time
the sites of partition `p` spend in state `a`.  This is exactly `calc_log_root_prior` plus
`calc_branch_log_G` summed over the branches.

The consequence used throughout Task 3.5 is that **the rate matrix enters `log G` only through the
three statistics `n_a`, `M_ab` and `Ttwiddle^p_a`**, all of which Delphy already maintains.  A change
of `s` or `pi` therefore has a closed-form effect on the posterior that costs no tree traversal, and
the existing kappa and frequency moves were already written that way.  `Run::calc_delta_log_G_of_subst_model`
is that difference, shared by both moves.

## 2. The six-digit encoding and the 203 structures

The six exchangeabilities are ordered

    index  0     1     2     3     4     5
    pair   A-C   A-G   A-T   C-G   C-T   G-T

(the order IQ-TREE uses).  A model code is six digits; two exchangeabilities are constrained to be
equal exactly when their digits are equal.  Only the partition of the six indices into equality
classes matters, so `101101` and `010010` are the same structure.  The canonical form of a code is
its restricted growth string: the class of index 0 is 0, and the class of index `i` is at most one
more than the largest class among the indices before it.  The canonical codes are therefore in
bijection with the set partitions of a six-element set, of which there are the sixth Bell number,

    Bell(6) = 203.

`all_exchangeability_groupings()` enumerates them by generating the restricted growth strings;
`Exchangeability_grouping::from_code` canonicalizes any six-digit code into one of them.  A code of
the wrong length, or with a character that is not a digit, is refused with a message that names the
pair order.

**203 is the number of exchangeability structures, not the number of likelihood implementations, and
not a number of models.**  One structure corresponds to several named models because the frequency
constraint differs, and supporting a structure is not the same as having validated Bayesian
inference under it; section 7 states which claims hold for which models.

### 2.1 The relative scale

The class containing G-T (index 5) is the reference class and its exchangeability is held at 1.  A
structure with `m` classes therefore has `m - 1` free relative exchangeabilities.  This is the
convention IQ-TREE documents ("the G-T rate is always fixed at 1.0 because all rates are relative"),
and `Subst_model::validate` refuses a model whose reference class is not 1.

### 2.2 Named models

`subst_model_aliases()` carries IQ-TREE's table of named DNA models.  The name fixes both the
structure and the frequency constraint, which is why the same code appears twice:

| Name | Code | Frequencies | Free parameters |
|---|---|---|---|
| JC, JC69 | 000000 | equal | 0 |
| F81 | 000000 | estimated | 3 |
| K80, K2P | 010010 | equal | 1 |
| HKY, HKY85 | 010010 | estimated | 4 |
| TN, TN93 | 010020 | estimated | 5 |
| TNe | 010020 | equal | 2 |
| K81, K3P | 012210 | equal | 2 |
| K81u | 012210 | estimated | 5 |
| TPM2 | 010212 | equal | 2 |
| TPM2u | 010212 | estimated | 5 |
| TPM3 | 012012 | equal | 2 |
| TPM3u | 012012 | estimated | 5 |
| TIM | 012230 | estimated | 6 |
| TIMe | 012230 | equal | 3 |
| TIM2 | 010232 | estimated | 6 |
| TIM2e | 010232 | equal | 3 |
| TIM3 | 012032 | estimated | 6 |
| TIM3e | 012032 | equal | 3 |
| TVM | 012314 | estimated | 7 |
| TVMe | 012314 | equal | 4 |
| SYM | 012345 | equal | 5 |
| GTR | 012345 | estimated | 8 |

The free-parameter count is `m - 1` exchangeabilities plus 3 frequencies when they are estimated;
the table is asserted against these codes and counts in `evo_subst_model_tests.cpp`.

### 2.3 Frequency modes

* `equal` -- `pi = (1/4, 1/4, 1/4, 1/4)`, held there.  This is IQ-TREE's `+FQ`.
* `fixed` -- `pi` is a vector the user supplies and it is held there.
* `estimated` -- `pi` is sampled in the posterior under a uniform prior on the simplex.

IQ-TREE's `+FO` optimizes the frequencies by maximum likelihood.  Delphy's `estimated` mode is a
Bayesian posterior over the same parameters under an explicit prior.  The CLI accepts `+FO` as the
spelling because it is the familiar one, but the two are different statements and this document does
not claim they agree.  IQ-TREE's `+F` (empirical frequencies counted from the alignment) has no
counterpart here: a run with `estimated` frequencies starts from the frequencies of the reference
sequence and then samples, and a run that wants them held at the empirical values states them with
`--v0-subst-frequencies` together with `--v0-subst-fix-frequencies`.

## 3. Priors

### 3.1 Exchangeabilities

Every free relative exchangeability `s_k` (the rate of class `k` relative to the class of G-T) has a
log-normal prior,

    pi(s_k) = exp(-(log s_k - m_k)^2 / (2 sigma^2)) / (s_k sqrt(2 pi sigma^2)),

with

    m_k = 1   if every exchangeability in class k is a transition (A-G or C-T),
    m_k = 0   otherwise,
    sigma = 1.25.

For the structure `010010` the only free class is `{A-G, C-T}`, so this is exactly the prior Delphy
has always used for HKY85's kappa (`log kappa` with mean 1 and standard deviation 1.25, adapted from
the Gire et al. 2014 BEAST configuration).  The rule extends it in the one direction the data
support a priori -- transitions are faster than transversions -- and leaves every other class
centered on the rate of G-T.  The three numbers are reported by the CLI at start-up and can be
replaced with `--v0-subst-exchangeability-prior LOG_MEAN_TRANSITION,LOG_MEAN_OTHER,SD`.

Two consequences are stated rather than hidden.  First, the prior is proper for every structure: the
log-normal is integrable at both ends, which the exact-conditional tests rely on.  Second, nested
structures do not induce each other's priors -- the prior HKY85 puts on kappa is not the prior GTR
induces on the same subspace -- which is true of any nested pair and is why a posterior comparison
between structures is a comparison of models, not of parameterizations.

### 3.2 Frequencies

Uniform on the simplex when they are estimated, which is what Delphy has always used; no term at all
when they are equal or fixed.

### 3.3 Site rates

Unchanged from before this task and audited in section 5.

## 4. The moves

### 4.1 Exchangeabilities

One proposal per free class, in class order: `s_k' = u s_k` with `u` uniform on `[0.75, 1/0.75]`.
The proposal ratio is

    q(new -> old) / q(old -> new) = s_k / s_k',

and the acceptance probability is `min(1, exp(log alpha))` with

    log alpha = delta log G + log(pi(s_k') / pi(s_k)) + log(s_k / s_k'),

where `delta log G` is the closed-form difference of section 1.1.  With the structure `010010` this
is the historical kappa move, draw for draw: the same scale factor from the same distribution, the
same acceptance formula, and the same single proposal per sweep.  That is why a default run is
byte-identical to one from before this task.

No Jacobian appears beyond the `1/s` the log-normal density already carries, because the proposal is
written directly in `s` and the prior is a density in `s`.

### 4.2 Frequencies

Unchanged: pick two bases, add `d ~ U[0, 0.01]` to one and subtract it from the other, reject if
either leaves `(0, 1)`.  The delta exchange is a translation inside the simplex, so the proposal is
symmetric and the uniform prior cancels:

    log alpha = delta log G.

The move returns immediately unless the frequency mode is `estimated`.

### 4.3 What is shared and what is per partition

The substitution matrix -- the exchangeabilities and the frequencies -- is **shared by every site
partition**.  Each partition keeps its own relative rate `r_p` from Task 3.4, so
`evo.partition_evo_model[p].mu == mu * r_p` and the matrices are equal.  This is what makes the
aggregate statistics `M_ab` and `n_a` the correct sufficient statistics: with a per-partition matrix
they would have to be kept per partition, which is the "unlinked models" extension the two moves
have always carried a comment about.  Per-partition exchangeabilities and frequencies are therefore
**not implemented** in this task; section 7 records that boundary.

The clock rate `mu` keeps its Gibbs draw unchanged, because its conditional depends on the matrix
only through the same exposure statistic.

## 5. The continuous Gamma site rates, as implemented

Delphy's site rate heterogeneity (`--v0-site-rate-heterogeneity`, off by default) is a **continuous**
Gamma, not a discrete one:

    nu_l ~ Gamma(alpha, alpha)   in the shape-rate parameterization, independently per site,

so `E[nu_l] = 1` and `Var[nu_l] = 1/alpha`, and the prior mean of 1 is what leaves the rate scale
with `mu`.  There are no rate categories, no equal-weight classes and no category assignment.  The
known criticisms of the discrete `+G4` approximation -- that four equal-probability categories
misrepresent the tail of the Gamma -- do not apply to it, and a comparison with IQ-TREE's `+G4` is a
comparison of two different models.

Conditional on the mutational history the site rates are independent with

    nu_l | . ~ Gamma(M_l + alpha, mu_l Ttwiddle^(l) + alpha),                (2)

where `M_l` is the number of events at site `l` and `Ttwiddle^(l) = sum_a q_a T^(l)_a` is free of
`nu` (`calc_Ttwiddle_l`), and `mu_l = mu r_p` carries the partition rate.  `gibbs_sample_all_nus`
draws (2) exactly.  Integrating every `nu_l` out of (2) leaves

    log p(alpha | history) = sum_l [ alpha log alpha - log Gamma(alpha)
                                     + log Gamma(M_l + alpha)
                                     - (M_l + alpha) log(mu_l Ttwiddle^(l) + alpha) ],   (3)

which is the target of the alpha move; `calc_log_p_alpha` evaluates (3), dropping the terms of the
sites without mutations that cancel between `log Gamma(M_l + alpha)` and `log Gamma(alpha)`.  The
move is ten scaling proposals on alpha against (3) and its exponential prior with mean 1, followed
by a Gibbs sweep over every `nu_l`, which is required because the conditional of `nu` depends on
alpha.

One documented deviation: `gibbs_sample_all_nus` clamps the drawn `nu_l` below at `1e-50`, because at
a small alpha the conditional (2) of a site without mutations is so concentrated at zero that the
draw can round to exactly zero and then appear in a logarithm.  The clamp only ever touches sites
with `M_l = 0`, it is measured to be unreachable for `alpha >= 0.1`, and it affects neither the
acceptance ratio of the alpha move (which integrates `nu` out) nor any tree move (whose terms carry
`nu_l` as a factor, so a clamped site contributes nothing).  It does mean that the sampled `nu` is a
clamped Gamma rather than a Gamma, and that a clamped site adds about `(1 - alpha) * 115` nats to the
reported `log_other_priors`; the accumulated and the freshly computed value agree, so the reported
posterior is a consistent function of the clamped state.

## 6. Mapping to the source

| What | Where |
|---|---|
| exchangeability order, canonical codes, the 203 structures, aliases, parsing | `core/evo_subst_model.{h,cpp}` |
| the normalized rate matrix from an exchangeability matrix | `derive_site_evo_model_from_exchangeabilities` in the same file |
| HKY85 as the structure `010010` | `core/evo_hky.cpp`, which now only builds its exchangeability matrix and calls the shared builder |
| the model in the run, its validation, the prior settings | `Run::set_subst_model`, `Run::subst_exchangeability_prior_*` in `core/run.{h,cpp}` |
| the closed-form difference of `log G` | `Run::calc_delta_log_G_of_subst_model` |
| the moves | `Run::subst_exchangeability_move(s)`, `Run::subst_frequencies_move` |
| the prior term in the posterior | `Run::calc_cur_log_other_priors` |
| the matrix handed to every likelihood path | `Run::derive_evo`, unchanged apart from the type it builds from |
| CLI | `--v0-subst-model`, `--v0-subst-initial-exchangeabilities`, `--v0-subst-frequencies`, `--v0-subst-fix-exchangeabilities`, `--v0-subst-fix-frequencies`, `--v0-subst-exchangeability-prior` in `core/cmdline.cpp` |
| log columns | `output_subst_model_headers` / `output_subst_model_values` in `core/beasty_output.cpp` |
| checkpoint fields 62-65 | `core/api.fbs`, `core/api.cpp` |
| site rates | `Run::gibbs_sample_all_nus`, `calc_log_p_alpha`, `Run::alpha_moves`, `Run::calc_cur_log_p_alpha` |

## 7. What is claimed, and what is not

1. **Every one of the 203 structures produces a valid rate matrix** -- zero row sums, positive
   off-diagonal entries, detailed balance, `pi` stationary, mean rate 1 -- and runs inside Delphy
   with a posterior that matches two independent full recomputations.  Tested exhaustively.
2. **Bayesian inference is validated** for the exchangeabilities and the frequencies of any
   structure: the closed-form difference matches independent recomputations on accept, reject and
   rollback, and the moves reproduce the exact conditional distributions of their parameters.  The
   validation is of the sampler, structure by structure through the generator, and of posterior
   recovery for the named models listed in the report.
3. **HKY85 is unchanged.**  The matrix is bit-identical to the historical one, the move is the
   historical move, and a default run's log body is byte-identical to the base commit's.
4. **Per-partition substitution matrices are not implemented.**  One matrix is shared by every site
   partition; per-partition relative rates remain as in Task 3.4.
5. **Non-reversible models, Lie Markov models, mixtures of several matrices, heterotachy, covarion
   models, codon and amino-acid models and automatic model selection are out of scope**, as the task
   states.
6. **FreeRate (+R) is not implemented**; `38_substitution_model_report.md` section 6 carries its
   mathematical design and the reason it needs a new sampling operator, which requires separate
   approval.
7. **`+I`, discrete `+G` and `+I+R` are not delivered.**  A specification with a `+` suffix other
   than `+FQ` or `+FO` is refused rather than silently ignored.
8. **The reliable divergence range is bounded**, and the bound is not a property of this task's
   models: see `38_substitution_model_report.md` section 5.

## 8. Limitations

1. The exchangeability prior's transition rule (section 3.1) makes the prior of a structure depend on
   which of its classes are pure transitions.  It is explicit, reported at start-up and
   user-replaceable, but it is a modelling choice and not a derived quantity.
2. The frequency delta exchange has a fixed window of 0.01 and the exchangeability scale move a fixed
   factor of 0.75, both inherited.  They are not adaptive, so a structure with many free classes
   mixes more slowly than HKY85 per sweep; the report measures the effect.
3. `estimated` frequencies are sampled under a uniform prior on the simplex, which is informative
   about the frequencies of a short partition in the same way Task 3.4's Dirichlet prior is
   informative about the rate of a short partition.
4. The site rates and the exchangeabilities are sampled in separate blocks; there is no joint move,
   so a strong posterior correlation between alpha and a free exchangeability would mix slowly.  No
   such correlation was measured in the runs of the report, but none was looked for at scale.
5. A model with equal frequencies and no free exchangeability (JC69) has no substitution parameters
   at all; the run then reports no exchangeability columns and the frequency columns are constant.
