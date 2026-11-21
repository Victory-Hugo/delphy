# Delphy-Extended v0.1.0-alpha.1: release notes and review summary

Base: upstream Delphy `3069065f2984e37edbc5ca09b847b3ec074402ce` (1.4.1).  The extension was developed
in a private repository over many commits; this branch presents its final state as a few logical
commits on top of the upstream base.  The development record is available from the author on request.

## For the reviewer

* Scope: everything new is opt-in.  With no new option, `delphy` writes byte-identical `.log` and
  `.trees` files to upstream (14/14 configurations of `tests/cli/run_pathogen_regression.sh`: 1 and 4
  threads, default and `old-usher-like` initialization, three Skygrid configurations).
* Architecture: EMAT, explicit mutational histories, incremental likelihoods, mutation-guided SPR and
  Run/Subrun are unchanged.  New terms enter the same posterior.  Calibrations, the exact Kingman
  correction, site partitions and substitution-model parameters are part of every Metropolis-Hastings
  ratio that they affect.  They are never applied by rescaling a finished tree.
* Where to start reading: `doc/extended/README.md` (index), then `calibration.md`, `exact_coalescent.md`
  and `observation.md`.  The largest source changes are in `core/run.*`, `core/subrun.*`,
  `core/cmdline.cpp`, and the new `core/human_input.*`, `core/calibration*.*`, `core/clade_summary.*`,
  `core/history_gibbs.*`, `core/evo_subst_model.*` and `core/site_partition.*`.
* File formats: `.dphy`, the flatbuffers API (`core/api.fbs`) and BEAST-style outputs keep their upstream
  version fields.  New API fields have new ids and defaults that reproduce upstream behaviour.

## Preparation of this release from the development tree

This release cleans up the tree and adds version labels.  It changes no computation.

1. Command-line suites, their helpers and the pathogen regression moved from `tests/human_input` to
   `tests/cli`.  Campaign drivers, analysis scripts and stored results were removed (kept in the development record).
2. Design documents moved to `doc/extended` with descriptive names and the examples to
   `doc/extended/examples`.  Per-task reports and evidence were removed (kept in the development record).
3. Development task labels were removed from comments and test messages.
4. The test sources in `tests/CMakeLists.txt` are in alphabetical order again.  Against upstream the
   diff is additions only.
5. `delphy`, `delphy_mcc` and `delphy_clades` print a second version line naming Delphy-Extended and
   the upstream commit.  Upstream version constants are unchanged.
6. README section describing the extension, its validation status and its limitations.

Outside comments, the only source changes are string literals (help text, informational and error
messages that cite document paths) and the added version line.

## Verification of this release

* Release unit tests: 800/800 pass before and after (2 disabled tests, both intentional).
* CLI suites (13 suites, 344 checks), with the pre-release and the release binary on the same inputs:
  identical pass counts, no failure, and all 109 `.log`/`.trees` files byte-identical.
* Pathogen regression: 14/14 byte-identical against upstream `3069065` and 14/14 against the development tip.
* Same-seed runs (pathogen 1 thread; 1,000 synthetic sequences of 16,569 sites with 4 threads; the same
  with the exact correction): byte-identical across three repetitions.  Wall-time and RSS results are in
  the verification report.

## Known open issues

See the README section "Validation status".  In short: a free population model with a free clock does
not mix; free-topology calibration and the exact correction on a free topology lack SBC; the
threshold partition and the joint time-scale move are experimental.  The `--v0-human-*` option names
are historical, and the code behind them is not species-specific.  `--v0-human-reference-standard`
names two mtDNA coordinate systems (`rCRS`, `RSRS`, which fix the length at 16,569 sites) besides
`custom`.  The reference sequence itself is always supplied by the user.
