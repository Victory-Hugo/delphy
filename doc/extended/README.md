# Delphy-Extended design documents

These documents describe the opt-in extensions of Delphy-Extended: what each model is, how it enters
the posterior, what the input looks like and what is refused.  They were written during development
and keep its evidence labels ([P] proof from the source, [N] independent numerical check, [R] run,
[M] measurement, [U] unverified).

Numbered file names (for example `10_coalescent_report.md`) and "Task" labels inside these documents
refer to development reports and validation campaigns that are not part of the release tree.  They
are kept, with the campaign scripts and stored results, in a separate private development
repository and are available from the author on request.

| Document | Subject | Options |
|---|---|---|
| [input_contract.md](input_contract.md) | Reference-based sequence input, sampling-time metadata | `--v0-human-mode`, `--v0-human-*` |
| [observation.md](observation.md) | Full sequences, reference plus differences, callable masks, SNP ascertainment | `--v0-human-in-variants`, `--v0-human-*-file`, `--v0-human-snp-only-constant-sites` |
| [external_tree.md](external_tree.md) | External rooted tree, fixed-topology MCMC | `--v0-in-tree`, `--v0-fix-topology` |
| [calibration.md](calibration.md) | Sample-set MRCA age calibrations (fixed topology) | `--v0-calibration-file` |
| [calibration_free_topology.md](calibration_free_topology.md) | Calibrations with a free topology and monophyly constraints | `--v0-allow-free-topology-calibration` |
| [clade_summary.md](clade_summary.md) | Posterior clade tMRCA summaries | `delphy_clades` |
| [coalescent_grid.md](coalescent_grid.md) | Coalescent grid at deep time scales | `--v0-coal-prior-*` |
| [exact_coalescent.md](exact_coalescent.md) | Whole-batch exact Kingman correction | `--v0-exact-coalescent-correction` |
| [exact_skygrid.md](exact_skygrid.md) | Exact Kingman posterior with a Skygrid curve | as above, with Skygrid |
| [exact_free_topology.md](exact_free_topology.md) | Exact Kingman correction with a free topology | as above, without `--v0-fix-topology` |
| [threshold_partition.md](threshold_partition.md) | Several tree partition parts with the exact correction | `--v0-threshold-partition` |
| [site_partitions.md](site_partitions.md) | Site partitions with relative rates | `--v0-site-partition-*` |
| [substitution_models.md](substitution_models.md) | Reversible nucleotide models (JC69 to GTR), site-rate heterogeneity | `--v0-subst-*` |
| [history_gibbs.md](history_gibbs.md) | Single-site mutational-history Gibbs updates | `--v0-history-gibbs*`, `--v0-site-history-refreshes` |
| [global_scale_move.md](global_scale_move.md) | Joint time-scale move (experimental) | `--v0-global-scale-*` |

Release notes and a review summary: [RELEASE_NOTES_v0.1.0-alpha.1.md](RELEASE_NOTES_v0.1.0-alpha.1.md).

Small worked inputs are under [examples/](examples/).  The command-line suites in `tests/cli/` use them.
