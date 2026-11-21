# External rooted tree and fixed-topology MCMC

## 1. What exists and what was added

| Capability | Before | Now |
|---|---|---|
| Newick parsing | `Newick_parser` (`core/newick.cpp`); `newick_to_phylo_tree` in `core/beasty_input.cpp` only reads BEAST output with numeric tip IDs and `CHECK`-fails on anything unusual | reused unchanged except `Newick_node::has_branch_length`, which separates "no length" from "length 0" |
| Initial EMAT from a given topology | none (`build_random_tree`, `build_usher_like_tree`, `build_initial_phylo_tree` all build their own topology) | `build_phylo_tree_from_newick` (`core/external_tree.cpp`) |
| `Run::set_topology_moves_enabled` | existed (UI/WASM toggle, serialized in `.dphy` params); it dropped `subtree_slide` and `spr1` entirely | CLI flag `--v0-fix-topology`; the topology-preserving part of `subtree_slide` stays active (see section 3) |
| Move statistics | none | `Move_counts` per local move (`selected`, `accepted`, `blocked_by_fixed_topology`), printed after every CLI run |

## 2. External tree import

### CLI

```
--v0-in-tree FILE                     rooted, strictly binary Newick; tip names = sequence IDs
--v0-in-tree-branch-lengths MODE      ignore | years | days | subs-per-site   (required if the tree has lengths)
--v0-fix-topology                     no move may change the rooted topology
```

`--v0-in-tree` cannot be combined with `--v0-init*`.  `ignore` and `subs-per-site` need an explicit rate
(`--v0-init-mutation-rate`, or a rate prior via `--v0-mu-prior-*`); nothing falls back to a pathogen constant.
Human mode keeps its existing gate (same-date data without a rate source is refused).

### Call path

`cmdline.cpp: process_args` -> read FASTA/MAPLE (`maple_file.tip_descs`) -> `read_newick_tree` ->
`build_phylo_tree_from_newick` -> `Run(...)`.  Steps inside `build_phylo_tree_from_newick`:

1. Structure: every node has 0 or 2 children.  A trifurcating root (unrooted tree), polytomies and unary nodes are
   errors; no rerooting, no resolution.
2. Tip names are matched one to one with the sequence names (duplicates, unlabelled tips, tips without a sequence
   -- including sequences discarded by the reader -- and sequences missing from the tree are listed, at most 10 each).
3. Branch lengths: declared meaning required; negative or non-finite lengths rejected; zero is legal.
4. Nodes are numbered tips first (sequence order), inner nodes in post-order, root last.
5. Mutation history: Fitch parsimony per variable site on the given topology; missing data are wildcards; the root
   state prefers the reference.  Mutations on the root branch are absorbed by `rereference_to_root_sequence`.
   `fix_up_missations` establishes the missation invariants and `from_states`.  This is only a starting state; MCMC
   samples the histories.
6. Times: tips at the middle of their date window; each inner node is `max(branch length, 1e-3 day)` before its
   earliest child.  `years`/`days` use the lengths as time, `subs-per-site` divides by the rate, `ignore` uses
   parsimony mutation counts divided by (rate x observed sites).  Mutation times are spread evenly over their
   branch.  Zero-length input branches become 1e-3 day gaps (Delphy's samplers refuse coincident times).
7. `assert_phylo_tree_integrity(…, true)` and `assert_phylo_tree_matches_tip_descs(…, true)`.

## 3. Fixed-topology audit

Every operator that can change the rooted topology, by source location:

| Operator | Where | Changes topology? | With `--v0-fix-topology` |
|---|---|---|---|
| `spr1_move` | `subrun.cpp` (selected with weight 1 of 32) | yes | never selected (weight removed) |
| `subtree_slide_move`, branch where the slid parent hops past another node | `subrun.cpp` (two `spr_move_core(X, {SS, ...})` calls) | yes | proposal discarded (`blocked_by_fixed_topology`) |
| `subtree_slide_move`, no hop (`spr_move_core(X, {S, new_P_t})`) | `subrun.cpp` | no: re-times P and re-samples the hot-path mutations | kept (weight 1 of 31) |
| `wilson_balding_move` | defined, never called by `mcmc_sub_iteration` | yes | not reachable |
| `branch_reform_move` (incl. `P == root` branch) | `spr_move_core(X, {S, t_P})` regrafts at the same place | no | unchanged |
| `inner_node_displace`, `tip_displace` | node heights only | no | unchanged |
| `Run::repartition`, `reassemble`, `normalize_root` | copy subtrees back and forth, re-reference | no | unchanged |
| global moves (mu, HKY, alpha, population/skygrid) | parameters only | no | unchanged |

The restriction is a restriction of the proposal's support, not a reset: out-of-region proposals are rejected
without evaluating them and the tree is never copied back, so every kernel still satisfies detailed balance with
respect to the posterior conditioned on the fixed topology.  The default (topology enabled) draws exactly the same
random numbers as before (`total_weight` 32, same thresholds).

`topology_moves_enabled` reaches subruns through `Run::push_global_params_to_subruns` before every local batch, so
repartitioning and multi-threaded subruns always see it.  It is stored in the `.dphy` params
(`api.cpp: run_to_api_params` / `apply_api_params_to_run`).  The CLI cannot resume from `.dphy`; the UI/WASM path
restores it (tested at the params level).

Side effect for UI users: with topology moves disabled, node times and mutation histories now continue to mix
through the no-hop slide (before, only inner/tip displacement and branch re-timing ran).

## 4. Tests

C++ (`tests/external_tree_tests.cpp`, 14 tests): see section 5 for results.

* import: clade identity, tip-order independence, units, every rejection path, topology-only mode, zero-length
  branches, parsimony history consistency, zero-mutation branches, homoplasy;
* fixed topology (N=20, 100; 3, 2 data seeds; 1 and 4 threads, 3-4 partitions with forced repartitioning every
  50-200 local moves): rooted clades and parent-child relations identical after every check, node times finite and
  ordered, >= 90 % of inner node times changed, SPR never selected, topology-preserving slides accepted, hopping
  slides blocked; same-date tips; near-zero mutation density; free-topology control that does rearrange;
  params round trip and resumed run;
* float32: imported trees at 50/100/140/200 kyr, with 10 zero-length and 9 sub-spacing branches, round-tripped through
  the API serialization: no inversion, ties only where the double-precision gap is below float32 spacing, max
  time error 1.0-3.8 d.

CLI/simulation (`tests/human_input/run_fixed_topology.py`, `fixed_topology_check.py`): simulated mtDNA (16,569
sites, ancient-date mode, mask, gamma rate heterogeneity), the true rooted tree as input, topology fixed, explicit rate prior
(mean = simulation truth, sd 30 %).  Clade sets of all sampled trees are compared with the truth; node times are compared
with the true node times.

## 5. Results (all actually run)

Unit tests: Release 422/422 (408 existing + 14 new); Debug 14/14 new tests in the dedicated run (Debug 422/422
(full suite, ASan build)).  CLI: `run_external_tree_cli_tests.sh` 14 checks, 0 failures;
`run_cli_scale_tests.sh` 0 failures; `run_pathogen_regression.sh`: `.log` and `.trees` byte-identical to the
previous binary for 1 and 4 threads, default and `old-usher-like` initialization.  Startup (read + audit + import +
`--v0-steps 0`, one thread, Release): 3.6 s and 2.5 GB at N=1000, 17.3 s and 5.5 GB at N=10,000 (import only; no
inference at N=10,000 is claimed).  Summary of the CLI suite (20 runs, all exit 0, 0 NaN/Inf, every sampled tree
has exactly the true rooted clades):

| N | runs | init | node-time 95 % coverage of truth | median / min node-time ESS | steps/s |
|---|---|---|---|---|---|
| 20 | 6 + 3 | years / ignore | 0.84-1.00 (mean 0.96) | 98-257 / 13-157 | 0.7-1.2 M |
| 100 | 4 + 1 + 1 (4 threads) | years / ignore | 0.87-0.94 | 192-303 / 36-124 | 2.4-3.1 M |
| 1000, default 400-cell grid | 2 + 1 + 1 (4 threads) | years / ignore | **0.24-0.25** (years), 0.50 (ignore, ESS 7) | 33-295 / 7-81 | 5.8-9.0 M |
| 1000, 4000 cells (explicit option) | 1 | years | 0.95 | 112 / 4.4 | 4.2 M |
| 1000, 40000 cells (explicit option) | 1 | years | 0.94 | 293 / 65 | 0.95 M |

Free-topology control (N=20): 1.25 M SPR moves selected, clade set changed.

Move acceptance (N=20/100/1000): `branch_reform` 97-100 %, `inner_node_displace` 78-94 %, `subtree_slide` 49-56 / 27-34 / 4-6 %
(blocked hops: 4-12 % / 22-29 % / 44-46 % of selections), `tip_displace` 0 % (all ancient and modern dates are exact:
the move returns before a proposal, so "selected but not tried" is visible).

Mutation histories mix: 12-17 of 38 (N=20), 13-22 of 198 (N=100), 24-32 of 1998 (N=1000) branches change their mutation count between samples.

## 6. Scientific assessment and unresolved risks

* The fixed-topology sampler is correct where the model is resolved: N=20 and N=100 cover the truth at about the
  nominal rate, and N=1000 covers it (0.95) once the coalescent grid is 4000 cells.
* **Default 400-cell grid biases node times at N=1000** (coverage 0.25; population scale n0 about 7,000 yr instead of
  the simulated 75,000).  Fixing the topology does not remove the Task 1.3 grid problem; it only prevents the collapse
  of the tree.  Default not changed (task rule); this belongs to 3.1.
* N=10,000 inference is not claimed.  The importer's cost at 10,000 tips is reported in the final report only if measured.
* Memory is large even for N=20 (2.6-9.9 GB peak) because of the 1-day initial coalescent grid at deep times (N-04).
* Float32: legal zero-length or sub-spacing branches tie when stored in `.dphy`; double precision in memory keeps a
  strict order (>= 1e-3 day).  `.dphy` format untouched.
* Fitch parsimony is only an initial history; with `ignore` units the initial times come from parsimony counts and
  need burn-in (ESS is lower: see N=1000 `ignore`, ESS 7).
* A 1000-tip `ignore` start did not reach the stationary distribution within 3e8 steps (coverage 0.50, ESS 7): use
  real branch lengths or more steps.
* Only 20 simulated datasets were used; the coverage numbers are screening evidence, not calibrated coverage.
* No real data, no chrY, no WASM/UI run, no CLI checkpoint resume (the CLI has none).

## 7. Entry conditions for Task 2.2

1. Calibration targets must be defined on the tip-name clade sets that `rooted_clades` in the tests already
   computes; with a fixed topology a calibration target maps to a stable node.
2. Calibrated node times must enter `delta_log_prior` in `inner_node_displace_move`, the no-hop slide and the root
   move; the new `Move_counts` show which moves run in fixed-topology mode.
3. Joint time and rate inference needs the initial time scale from `Init_time_scale` / `External_tree_options`, not
   from the CLI rate gate.
4. The 400-cell grid issue must be known to anyone interpreting node-time posteriors (use `--v0-target-coal-prior-cells`
   explicitly until 3.1).
