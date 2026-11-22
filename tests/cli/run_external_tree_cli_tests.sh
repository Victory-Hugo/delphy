#!/usr/bin/env bash
# CLI checks for --v0-in-tree / --v0-in-tree-branch-lengths / --v0-fix-topology on a simulated 20-sample data set.
# Rate values below are the simulation truth, passed explicitly as user input.
# Usage: tests/cli/run_external_tree_cli_tests.sh PATH/TO/delphy WORKDIR
set -u
DELPHY=$(readlink -f "$1")
WORK=$2
HERE=$(cd "$(dirname "$0")" && pwd)
D="$WORK/data"
mkdir -p "$WORK"
[ -f "$D/truth.json" ] || python3 "$HERE/simulate_mtdna_coalescent.py" "$D" --samples 20 --seed 1 --mu-site-year 1.7e-8 \
    --coal-timescale-years 75000 --dates ancient --mask 303-315,16184-16193 > /dev/null \
    || { echo "FAIL: could not generate the test data"; exit 1; }
fail=0
base=("$DELPHY" --v0-human-mode --v0-in-fasta "$D/samples.fasta" --v0-human-reference-standard rCRS
      --v0-human-reference-fasta "$D/reference.fasta" --v0-human-mask-file "$D/mask.txt"
      --v0-human-audit-report "$WORK/audit.json" --v0-human-metadata "$D/metadata.tsv"
      --v0-mu-prior-mean 1.7e-8 --v0-mu-prior-stddev 5.1e-9 --v0-steps 20000 --v0-seed 1)

expect_error() {  # expect_error NAME PATTERN args...
  local name=$1 pat=$2; shift 2
  "${base[@]}" "$@" > /dev/null 2> "$WORK/err.txt"; local rc=$?
  if [ $rc -ne 0 ] && grep -q -- "$pat" "$WORK/err.txt"; then echo "PASS: $name"; else echo "FAIL: $name (rc=$rc)"; sed -n 1,3p "$WORK/err.txt"; fail=$((fail+1)); fi
}
expect_ok() {
  local name=$1; shift
  "${base[@]}" "$@" > /dev/null 2> "$WORK/err.txt"; local rc=$?
  if [ $rc -eq 0 ]; then echo "PASS: $name"; else echo "FAIL: $name (rc=$rc)"; sed -n 1,3p "$WORK/err.txt"; fail=$((fail+1)); fi
}

T="$D/true_tree.nwk"
expect_error "branch lengths with undeclared units are refused" "meaning was not declared" --v0-in-tree "$T"
expect_error "units without a tree are refused" "requires --v0-in-tree" --v0-in-tree-branch-lengths years
expect_error "unknown units are refused" "Unknown --v0-in-tree-branch-lengths" --v0-in-tree "$T" --v0-in-tree-branch-lengths furlongs
expect_error "--v0-in-tree with --v0-init random is refused" "ERROR" --v0-in-tree "$T" --v0-in-tree-branch-lengths years --v0-init random
expect_error "missing tree file" "Could not read tree file" --v0-in-tree "$WORK/nope.nwk" --v0-in-tree-branch-lengths years

printf '(S000000:1,S000001:1,S000002:1);' > "$WORK/tri.nwk"
expect_error "unrooted/trifurcating tree is refused" "rooted and strictly binary" --v0-in-tree "$WORK/tri.nwk" --v0-in-tree-branch-lengths years
sed 's/S000003/S000001/' "$T" > "$WORK/dup.nwk"
expect_error "duplicate tip is refused" "duplicate tip names" --v0-in-tree "$WORK/dup.nwk" --v0-in-tree-branch-lengths years
sed 's/S000003/EXTRA/' "$T" > "$WORK/extra.nwk"
expect_error "extra and missing tips are refused" "do not match the input sequences" --v0-in-tree "$WORK/extra.nwk" --v0-in-tree-branch-lengths years
sed 's/:\([0-9.]*\)/:-\1/' "$T" > "$WORK/neg.nwk"
expect_error "negative branch length is refused" "negative length" --v0-in-tree "$WORK/neg.nwk" --v0-in-tree-branch-lengths years

expect_ok "years, fixed topology" --v0-in-tree "$T" --v0-in-tree-branch-lengths years --v0-fix-topology
grep -q "topology fixed" "$WORK/err.txt" && echo "PASS: diagnostics report fixed topology" || { echo "FAIL: diagnostics"; fail=$((fail+1)); }
expect_ok "ignore lengths, rate from the prior, fixed topology" --v0-in-tree "$T" --v0-in-tree-branch-lengths ignore --v0-fix-topology
expect_ok "free topology from the same tree" --v0-in-tree "$T" --v0-in-tree-branch-lengths years
echo "failures: $fail"
exit $fail
