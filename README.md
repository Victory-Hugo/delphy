Delphy-Extended v0.1.0-alpha.1
==============================

Delphy-Extended is an unofficial, independently versioned extension of Delphy.  It is not a Delphy
release and is not endorsed by the Delphy authors.  It extends upstream
[broadinstitute/delphy](https://github.com/broadinstitute/delphy) at commit
`3069065f2984e37edbc5ca09b847b3ec074402ce` (Delphy 1.4.1).  `delphy --version` prints both identities;
output files record only the upstream Delphy version, exactly as before.

The aim is a species- and region-agnostic Bayesian phylogenetics tool built on Delphy's own architecture:
the explicit mutation-annotated tree (EMAT), explicit mutational histories, incremental likelihoods,
mutation-guided topology moves and the Run/Subrun partitioning are all kept.  Every extension is opt-in.
Without the new options, `delphy` writes byte-identical `.log` and `.trees` files to upstream for the
same input and seed (`tests/cli/run_pathogen_regression.sh`).

### What is added

| Area | Options / tool | Design document |
|---|---|---|
| Reference-based sequence input, sampling-date metadata (ISO, CE, BCE, BP, cal BP, uncalibrated C14 BP), masks | `--v0-human-mode`, `--v0-human-*` | [input_contract.md](doc/extended/input_contract.md) |
| Observation models: callable sites, per-sample callability, variant lists, SNP ascertainment | `--v0-human-in-variants`, `--v0-human-*-file`, `--v0-human-snp-only-constant-sites` | [observation.md](doc/extended/observation.md) |
| External rooted tree; fixed-topology MCMC | `--v0-in-tree`, `--v0-fix-topology` | [external_tree.md](doc/extended/external_tree.md) |
| Sample-set MRCA age calibrations inside the posterior | `--v0-calibration-file`, `--v0-allow-free-topology-calibration` | [calibration.md](doc/extended/calibration.md), [calibration_free_topology.md](doc/extended/calibration_free_topology.md) |
| Clade tMRCA summaries with shortest intervals | `delphy_clades` | [clade_summary.md](doc/extended/clade_summary.md) |
| Coalescent grid at deep time scales; exact Kingman correction | `--v0-coal-prior-*`, `--v0-exact-coalescent-correction` | [coalescent_grid.md](doc/extended/coalescent_grid.md), [exact_coalescent.md](doc/extended/exact_coalescent.md) |
| Site partitions with relative rates | `--v0-site-partition-*` | [site_partitions.md](doc/extended/site_partitions.md) |
| Reversible nucleotide models (JC69, K80, HKY85, TN93, GTR and every other exchangeability structure), continuous Gamma site rates | `--v0-subst-*` | [substitution_models.md](doc/extended/substitution_models.md) |
| Single-site mutational-history updates | `--v0-history-gibbs*`, `--v0-site-history-refreshes`, `--v0-marginal-node-time-*` | [history_gibbs.md](doc/extended/history_gibbs.md) |

All internal times are days since 2020-01-01 and rates are substitutions per site per day, as upstream;
the conversions to years happen at the input and output boundaries.  Delphy-Extended has no built-in
rates, ages or calibrations for any organism: every such number must come from the user's input.

### Validation status

Validated, in the configuration stated:

* Default pathogen workflows: byte-identical to upstream in 1 and 4 threads, default and `old-usher-like`
  initialization, and three Skygrid configurations.
* Fixed topology, strict clock, pinned demography, with calibrations, site masks, missing data, site
  partitions, a reversible substitution model and the exact Kingman correction: clade ages agree with an
  independent reference that shares no code with `core/`, and simulation-based calibration (SBC) of node
  ages passes at 100 and 1,000 tips.  The calibrated cells at 20 tips fail; the cause traced so far is a
  bias in the test generator, not in Delphy.
* Free topology at 5 tips: clade support and conditional clade ages agree with an independent non-MCMC
  posterior.
* Skygrid population moves with fixed smoothness, staircase and log-linear: recover their closed-form
  prior (12 tips, grid pinned).

Validation incomplete:

* A free population model together with a free clock rate does not mix (exponential growth rate ESS of
  a few units at 20 to 1,000 tips).  Formal support is limited to a pinned clock and demography.
* Skygrid with inferred smoothness: accepted at 12 tips with the grid pinned, qualified because no
  dispersed starting curve was available.  The Skygrid model as a whole is not yet scientifically accepted at deep time scales.
* Calibrations with a free topology, and the exact correction with a free topology: correct target in
  the tests run; SBC and comparison with an independent sampler are not done.

Experimental (marked so in `--help`, off by default): `--v0-exact-coalescent-correction`,
`--v0-history-gibbs`, `--v0-site-history-refreshes`, `--v0-marginal-node-time-move`,
`--v0-threshold-partition` (several tree parts with the exact correction; no ESS-per-hour gain measured
on real data) and `--v0-global-scale-move` (failed an independent-posterior check; not for use).

The full development record (per-task reports, campaign scripts and stored results) is kept in a
separate private repository and is available from the author on request.

### Known limitations

* A shared time tree assumes the sites follow one genealogy; recombining data violate the model.
* The options of the sequence-input path are named `--v0-human-*` for historical reasons; the code
  behind them is not species-specific.
* Closely spaced sampling dates do not identify an absolute clock.  Without informative ancient sample
  dates, calibrations or an external rate prior, node ages are not interpretable in years.
* The coalescent timescale parameter is not an effective population size without separate assumptions
  on inheritance, ploidy and generation time.

### Build and run

Build exactly as upstream ([INSTALL.md](INSTALL.md)); `delphy_clades` is built next to `delphy` and
`delphy_mcc`.  A minimal run on the synthetic example:

```bash
E=doc/extended/examples/minimal
build/release/delphy --v0-human-mode --v0-in-fasta $E/samples.fasta \
  --v0-human-reference-standard custom --v0-human-expected-length 300 \
  --v0-human-reference-fasta $E/reference.fasta --v0-human-metadata $E/metadata.tsv \
  --v0-human-mask-file $E/mask.txt --v0-human-audit-report audit.json \
  --v0-seed 1 --v0-threads 1 --v0-steps 2000 --v0-out-log-file ex.log --v0-out-trees-file ex.trees
```

Tests: `build/release/tests/tests` (unit tests) and the command-line suites in `tests/cli/`, each run as
`bash tests/cli/<suite>.sh build/release/delphy WORKDIR` (see `.github/workflows/ci.yml`).

The upstream Delphy README follows unchanged.

---

Delphy
======

[![CI](https://github.com/broadinstitute/delphy/actions/workflows/ci.yml/badge.svg)](https://github.com/broadinstitute/delphy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/broadinstitute/delphy/graph/badge.svg)](https://codecov.io/gh/broadinstitute/delphy)
[![GHCR](https://ghcr-badge.egpl.dev/broadinstitute/delphy/latest_tag?trim=major&label=ghcr.io)](https://github.com/broadinstitute/delphy/pkgs/container/delphy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11922813.svg)](https://doi.org/10.5281/zenodo.11922813)

Delphy is a fast, scalable, accurate and accessible tool for Bayesian phylogenetics based on Explicit Mutation-Annotated
Trees (EMATs).  EMATs are an extension of the Mutation-Annotated Trees (MATs) introduced by
[UShER](https://github.com/yatisht/usher) where nodes have explicit times, mutations are represented as explicit timed
events along branches, and missing data is explicitly represented.  EMATs open substantial simplifications and scaling
opportunities in the calculations powering Bayesian phylogenetics, at the cost of some statistical efficiency with
respect to the traditional calculations based on Felsenstein pruning.  For genomic epidemiology datasets, where the
total number of mutations on a tree is comparable to the number of samples, this is a very favorable trade-off.

These sources comprise the "core" computational engine of Delphy.  The web application that allows users to immediately
and intuitively use Delphy (currently at [https://delphy.bio](https://delphy.bio)) is developed in
collaboration with [Fathom Information Design](https://fathom.info), and its separately licensed sources are hosted
[here](https://github.com/fathominfo/delphy-web).

**NOTE**: As of 17 Dec 2025, we have relicensed Delphy under an open-source [MIT license](LICENSE), and we have withdrawn our
patent application, instead dedicating the relevant IP to the public domain.

References
----------

* [Varilly, P., Schifferli, M., Yang, K. et al. Scalable near-real-time Bayesian phylogenetics for outbreaks with Delphy. _Nature_ (2026). https://doi.org/10.1038/s41586-026-11012-6](https://doi.org/10.1038/s41586-026-11012-6).  See [here](https://github.com/broadinstitute/delphy-2026-paper-data) for data and scripts accompanying the paper.

* [Whitepaper with overview of key ideas and accuracy+speed benchmarks](delphywp.pdf)

Installation
------------
Pre-built Delphy binaries are available for Linux (x86_64 and ARM64) and macOS (ARM64) via [GitHub Releases](https://github.com/broadinstitute/delphy/releases), and as multi-arch Docker images from [GitHub Container Registry](https://github.com/broadinstitute/delphy/pkgs/container/delphy).  Delphy can also be compiled from source as a native command-line program or as a WebAssembly bundle for [`delphy-web`](https://github.com/fathominfo/delphy-web).  See [`INSTALL.md`](INSTALL.md) for all installation options, including building from source.  Delphy does not require special hardware, such as a GPU.

Delphy was developed and is primarily tested under Linux (Ubuntu 24.04 LTS and Ubuntu 26.04 LTS, x86-64).  [Delphy-web](https://github.com/fathominfo/delphy-web), deployed at [https://delphy.bio](https://delphy.bio), should work in any modern web browser; see the [Delphy-web README.md](https://github.com/fathominfo/delphy-web) for specific browsers and operating systems on which it has been tested.

Google Colab Tutorials
----------------------
Ready-to-use Google Colab tutorials for downloading and formatting sequencing data from the [NCBI Virus database](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/), as well as user-provided data, can be accessed [here](https://colab.research.google.com/github/broadinstitute/delphy/blob/main/tutorials/delphy_workflow.ipynb) (blank notebook) and [here](https://colab.research.google.com/github/broadinstitute/delphy/blob/main/tutorials/delphy_workflow_example.ipynb) (Ebola example).  These notebooks streamline the uniform formatting of sequencing data and the associated metadata for input into Delphy, and then run Delphy using a pre-compiled binary.  The output files can be visualized on the [Delphy web interface](https://delphy.bio).  

Credits and Acknowledgements
----------------------------

Delphy is developed in the [Sabeti Lab](https://www.sabetilab.org/) at the [Broad
Institute](https://www.broadinstitute.org/).

Delphy draws a lot of inspiration from:

- [UShER and matOptimize](https://github.com/yatisht/usher)
- [BEAST](https://github.com/beast-dev/beast-mcmc)
- [BEAST2](https://github.com/CompEvol/beast2)
- [MAPLE](https://github.com/NicolaDM/MAPLE)
- [IQ-TREE2](github.com/iqtree/iqtree2/)

Copyright (c) 2022-2025 Broad Institute, Inc.  See [LICENSE](LICENSE) for details.
