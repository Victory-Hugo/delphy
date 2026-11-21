# AGENTS.md - Guide for AI Coding Agents

## CRITICAL: Read CONTRIBUTING.md First

**Before writing any code, you MUST read [CONTRIBUTING.md](CONTRIBUTING.md) in full.**

It defines the coding style for this project. Failure to follow these conventions will result in rejected code.

---

## Key Coding Conventions

These are the most important conventions from CONTRIBUTING.md that you MUST follow:

| Convention | Correct | Incorrect |
|------------|---------|-----------|
| Variable/function names | `snake_case` | `camelCase` |
| Type names (classes, structs, aliases) | `Stroustrup_case` | `snake_case` |
| Boolean negation | `not x` | `!x` |
| Natural logarithm | `log(x)` | `ln(x)` |
| Const style | `const char* str` | `char const* str` |
| Pointer/reference | `int* ptr`, `int& ref` | `int *ptr`, `int &ref` |
| Container size | `std::ssize(c)` | `c.size()` |
| Hash maps/sets | `absl::flat_hash_map` | `std::unordered_map` |
| Data containers | structs with public fields | classes with getters/setters |
| Error handling | exceptions | error codes |
| Function style (AAA) | `auto func(int x) -> int;` | `int func(int x);` |

**Include order** (each section separated by a blank line, sorted lexicographically within each section):
1. Corresponding header for this `.cpp` file, using `"..."`
2. System/standard library includes, using `<...>` (e.g., `<iostream>`, `<vector>`)
3. Third-party/library includes, using `"..."` (e.g., `"absl/log/check.h"`, `"boost/math/..."`)
4. Other Delphy project includes, using `"..."` (e.g., `"run.h"`, `"phylo_tree.h"`)

The corresponding header is listed first so that missing transitive dependencies in that
header are caught immediately.  Every header should include everything it uses directly.

**Compiler flags:** Code must compile cleanly with `-Wall -Wextra -Werror`.

---

## Repository Structure

```
delphy/
├── .github/workflows/  # GitHub Actions CI/CD
│   └── ci.yml          # Main CI workflow
├── core/           # Core library (phylogenetic algorithms, ~20K lines)
│   ├── tree.h          # Generic tree data structures
│   ├── phylo_tree.h    # Phylogenetic tree with mutations and times
│   ├── run.h           # MCMC run orchestration
│   ├── spr_move.h      # Subtree Prune and Regraft moves
│   ├── mutations.h     # Mutation and Missation types
│   ├── sequence.h      # DNA sequence types
│   └── api.fbs         # FlatBuffers schema for serialization
├── tests/          # Google Test suite (27 test files)
├── tools/          # Entry point programs
│   ├── delphy.cpp      # CLI tool
│   ├── delphy_ui.cpp   # OpenGL/SDL visualization UI
│   ├── delphy_mcc.cpp  # MCC tree computation utility
│   └── delphy_wasm.cpp # WebAssembly build
├── third-party/    # Git submodules (abseil-cpp, flatbuffers, ctpl, cppcoro, cxxopts)
├── doc/            # Documentation (dphy_file_format.md)
└── tutorials/      # Jupyter notebooks
```

---

## Build System

### Prerequisites
- CMake 3.24+
- Python 3 with venv
- Conan 2.34.0
- GCC or Clang with C++20 support

### Quick Build

```bash
# First-time setup
python3 -m venv delphy-venv
source delphy-venv/bin/activate
pip3 install 'conan==2.34.0'
conan profile detect
git submodule update --init

# Debug build
conan install . --output-folder=build/debug --build=missing --settings=build_type=Debug
cmake --preset conan-debug
cmake --build --preset conan-debug

# Release build
conan install . --output-folder=build/release --build=missing --settings=build_type=Release
cmake --preset conan-release
cmake --build --preset conan-release
```

### Running Tests

```bash
./build/debug/tests/tests              # All tests (debug)
./build/release/tests/tests            # All tests (release)
./build/debug/tests/tests --gtest_filter="Tree_test.*"  # Specific test suite
```

See [INSTALL.md](INSTALL.md) for WebAssembly builds and detailed instructions.

---

## CI/CD

GitHub Actions CI runs on every push and PR. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

### What CI Does

1. **Build & Test** - Builds on 3 platforms in parallel:
   - `ubuntu-latest` (x86_64) - static linking
   - `ubuntu-26.04-arm` (ARM64) - static linking (GitHub has no `ubuntu-latest-arm` alias; bump this label when `ubuntu-latest` moves to a new Ubuntu release)
   - `macos-latest` (ARM64) - dynamic linking, targeting macOS 14.0+

2. **Code Coverage** - Runs separately with `--coverage` flags, uploads to Codecov

3. **Docker** - Multi-arch image pushed to `ghcr.io/broadinstitute/delphy` (primary)
   - Optionally also pushed to an alternate registry (e.g. `quay.io`) when `DOCKER_ALT_REGISTRY` repo variable is set
   - Branch pushes: tagged with branch name
   - Tag pushes: tagged with version
   - Main branch: also tagged as `latest`

4. **WASM Build** - WebAssembly build using Emscripten

5. **Release Assets** - On tag push, attaches tarballs to GitHub release

6. **Cleanup** - On branch delete, removes corresponding Docker tag from GHCR (and alt registry if configured)

### Build Artifacts

- `delphy` - Main CLI tool
- `delphy_ui` - SDL2-based visualization UI
- `delphy_mcc` - MCC tree computation utility
- `beast_trees_to_dphy` - Format conversion utility

### Required Secrets & Variables

- `CODECOV_TOKEN` - Code coverage uploads
- `DOCKER_ALT_REGISTRY` (repo variable, optional) - Alternate Docker registry (e.g. `quay.io`)
- `DOCKER_USER`, `DOCKER_PASSWORD` (secrets, optional) - Credentials for the alternate registry

---

## Key Data Structures

### Node Indexing
- Nodes are referenced by `Node_index` (int), not pointers
- `k_no_node = -1` is the sentinel for null/invalid
- Branch `X` connects `parent(X)` to node `X`
- Access nodes via `tree.at(node)` or `tree.at_parent_of(node)`

### Core Types

```cpp
// From sequence.h
Seq_letter      // uint8 bitmask: A=0b0001, C=0b0010, G=0b0100, T=0b1000, N=0b1111
Real_seq_letter // Enum for unambiguous: A, C, G, T only
Site_index      // int alias for genomic position

// From mutations.h
Mutation        // {from, site, to, t} - timed state change
Seq_delta       // {site, from, to} - mutation without time
Missation       // pair<Site_index, Real_seq_letter> - missing data marker

// From tree.h / phylo_tree.h
Node_index      // int alias for node reference
Phylo_node      // Has: name, t, t_min, t_max, mutations, missations
Phylo_tree      // Tree container with ref_sequence
```

### Tree Traversals

```cpp
for (const auto& node : pre_order_traversal(tree)) { ... }
for (const auto& node : post_order_traversal(tree)) { ... }
for (const auto& node : index_order_traversal(tree)) { ... }
```

---

## Common Gotchas

1. **Use `std::ssize()` not `.size()`** - The codebase strictly avoids signed/unsigned mixing.

2. **Use `not` instead of `!`** - Boolean negation uses the keyword, not the operator.

3. **Times are fractional days** - `t` values are days since 2020-01-01.

4. **Reference sequence is the baseline** - `Phylo_tree::ref_sequence` is an arbitrary reference state, usually close to the root; mutations are deltas from it.  The mutations above the root node list the deltas from the reference state to the root state.

5. **Branch vs Node** - A branch is identified by its endpoint. Branch X connects parent(X) to X.

6. **Binary trees only** - Most MCMC code assumes exactly 2 children for inner nodes.

7. **Namespace** - All code lives in `namespace delphy { }`.

8. **Tip vs Inner** - Check with `node.is_tip()` or `node.is_inner_node()`.

9. **Header guards** - Use `#ifndef DELPHY_FILENAME_H_` style.

10. **Debug assertions** - Use `assert_tree_integrity()`, `assert_mutation_consistency()` for validation.

11. **Git staging** - Always use explicit file paths with `git add` (e.g., `git add file.cpp`).  Never use `git add -A` or `git add .` which may stage unintended files.

12. **Plain ASCII in plans** - Plans and other written documents must use only ASCII characters.  Use `->` for arrows, `---` for em dashes (parenthetical remarks), `--` for en dashes (number ranges), `>=`/`<=` for comparisons.  No Unicode arrows, check marks, em/en dashes, or other fancy characters.

---

## Git Conventions

1. **Never amend pushed commits** - Always create fresh commits. Do not use `git commit --amend` or `git rebase` on commits that have already been pushed.

---

## Dependencies

### Via Conan
- boost/1.84.0 (header-only)
- eigen/3.4.0
- ms-gsl/4.0.0

### Via Git Submodules (third-party/)
- abseil-cpp - Hash maps, random numbers, utilities
- flatbuffers - Serialization
- ctpl - Thread pool
- cppcoro - Coroutines/generators
- cxxopts - Command-line parsing

---

## Testing Patterns

Tests use Google Test:

```cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>

TEST(Tree_test, example) {
    auto tree = Tree<Binary_node>{2};
    EXPECT_EQ(tree.size(), 2);
    EXPECT_THAT(result, testing::ElementsAre(1, 2));
}
```

Test files follow the pattern `{module}_tests.cpp` in the `tests/` directory.

---

## Project Extension: Human Uniparental Phylogenetics (mtDNA / chrY)

These rules apply to work on the human uniparental extension (branches `human-uniparental/*`).
They supplement, and never override, the upstream rules above.  If a rule here conflicts with an
upstream rule, keep the upstream rule and report the conflict before continuing.

### Goal and priorities

Extend Delphy to perform statistically valid, scalable Bayesian phylogenetic inference on human
mitochondrial DNA (mtDNA) and non-recombining Y-chromosome (chrY) sequences: calibrated lineage
ages, tMRCA posteriors, clade support and population history.  Existing pathogen workflows must keep
working unchanged.

Priority order: statistical correctness > reproducibility > backward compatibility > performance >
UI polish.

### Before changing code

1. Record the commit SHA, branch, uncommitted files, toolchain versions, and build/test status.
   Work on a dedicated feature branch; never push to or merge into upstream `main` without explicit
   approval.
2. Trace the real call chain and the relevant tests before modifying any algorithm (see
   `doc/human_uniparental/01_architecture_audit.md`).
3. Check whether a requested feature already exists, exists internally without a CLI flag, or must be
   written.  Reuse existing code; do not re-implement Skygrid, MCC or tMRCA machinery without a
   demonstrated reason.
4. Only implement the assigned task.  State assumptions, dependencies and acceptance criteria up front.

### Scientific invariants

- Preserve the EMAT invariants: timed mutations, branch consistency, deltas relative to the reference
  sequence, missing-data semantics and tree integrity.
- Distinguish mutation-history density (what drives performance) from overall divergence from the
  reference; measure both on real datasets.
- Internal times are days since 2020-01-01; internal rates are substitutions / site / day.  Convert
  to and from years explicitly at external interfaces, and test conversions across long human time
  spans and calendar boundaries.
- Near-contemporaneous modern samples do not determine the absolute clock scale.  Before interpreting
  absolute node ages there must be informative ancient sample dates, an explicit calibration prior,
  or an identifiable external rate prior.  Never silently reuse viral clock defaults or invent dates.
- For ancient samples, keep sampling age, archaeological date uncertainty and internal-node
  calibrations distinct; they are not interchangeable.  Validate date conventions, reference epochs,
  bounds and ordering.
- Internal-node calibrations must enter the actual posterior density and every relevant
  Metropolis-Hastings ratio (local partition moves, root moves, topology changes, relevant global
  moves), including proposal asymmetry and Jacobians where applicable.  Rescaling tree times after a
  run is not a Bayesian calibration.
- Define calibration targets by stable taxon sets or clade semantics, never by node indices.  Specify
  and test the behaviour of an MRCA prior when the taxa are not monophyletic.
- Reconcile global and partition-local posterior computations before optimizing.  Check correctness
  after partitioning, reassembly, serialization and resume.  Adding a single global `log_prior` term
  does not prove correct sampling.
- With a fixed topology, confirm that all topology-changing moves are disabled while valid node-time
  and mutation-history moves still run.  With a free topology, all calibration mappings and
  acceptance ratios must stay valid under topology proposals.
- With the reference-plus-deltas representation, keep the full-sequence context needed for invariant
  sites.  Never treat a SNP-only alignment as a full sequence without ascertainment handling.  Track
  callable-site masks and missing data (especially for chrY); distinguish invariant sites from sites
  that were never observed.
- Do not assume mtDNA and chrY share evolutionary dynamics.  Exclude or explicitly model known
  problematic regions, heteroplasmy, structural variants, recurrent mutations and uncertain calls, as
  set by the task's data conventions.
- Keep coalescent timescale parameters distinct from interpretable head counts.  Do not label a
  timescale as an effective population size without explicit inheritance mode, ploidy, sex-specificity
  and generation-time assumptions.
- Treat numeric range limits, floating-point precision, coalescent grid resolution and initialization
  heuristics as validation targets at human time scales.  Changing any hard-coded constant requires a
  test and a justification.

### Engineering rules

1. Human-specific behaviour must be opt-in.  Default pathogen workflows and existing CLI behaviour
   must not change unless the change is explicitly approved.
2. Prefer small, reviewable changes with clear responsibilities.  Keep the CLI separate from the core
   model; get the CLI workflow right before any web UI change.
3. Comments and user-facing technical documentation are in English (plain ASCII, see Gotcha 12).  No
   speculative or verbose comments, no comments describing how the code was produced, and no comments
   about rejected alternatives.  Do not introduce new frameworks.
4. Validate inputs instead of guessing.  Reject inconsistent taxon IDs, impossible calibration bounds,
   incompatible date conventions, non-positive rates or scale parameters, and unsupported formats with
   error messages that explain how to fix the input.
5. Do not hard-code human mutation rates, haplogroup ages or archaeological dates.  All values must
   come from explicit input, documented priors, or test-only generated data.
6. Keep existing outputs compatible.  Any change to a serialization format or public API needs a
   documented migration path and a test that old outputs still load.
7. No force pushes, no amending published commits, no unrequested dependency upgrades.  Split
   different tasks into separate, reviewable commits.

### Required tests for every statistical feature

- Unit tests for log densities, invalid inputs, edge cases, time conversions, calibration-target
  membership and numerical stability.
- Checks of full-posterior differences against local / incremental differences for proposed moves,
  using an independent reference computation, covering accept, reject and rollback.
- Where supported: fixed and free topology, single-threaded and multi-partition runs, and restart.
- Synthetic datasets with known generating parameters, several seeds and several chains, with
  suitable calibration scenarios; report coverage, bias, mixing / ESS and posterior uncertainty.
  Never claim convergence from a single chain.
- Regression tests showing that existing pathogen examples and default options behave as before.
- Performance benchmarks only after statistical tests pass.  Record tips, sequence length, number of
  mutations, threads, wall time, peak RSS and baseline commit.  Scale up step by step (small -> 1,000
  -> 10,000 -> 50,000 tips) as resources allow; never report results for scales that were not run.

### Task reports

Every task report states: (1) findings, with exact source locations; (2) files changed and why, and
any changed statistical assumption; (3) commands actually run, pass/fail counts, environment and
reproducible example data, clearly separating tests not run from tests passed; (4) scientific
assessment (posterior correctness, identifiability, numerical risks, open limitations); (5) measured
performance only, with hardware and data context; (6) handover: commit SHA, artifacts, blockers and
the entry conditions for the next task.  If correctness is unresolved, stop and document it; never
hide failing tests, invent benchmarks, or claim unverified features are production-ready.
