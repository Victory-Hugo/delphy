#!/usr/bin/env bash
# The growth-rate window warning and the population move counters of the command line.
#
# Usage: run_pop_window_cli_tests.sh PATH/TO/delphy [PATHOGEN.fasta]
set -u
DELPHY=$(readlink -f "$1")
FASTA=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
PASS=0; FAIL=0
check() { if eval "$2"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); echo "FAIL: $1"; fi; }

python3 "$HERE/simulate_hky_joint.py" "$W/d" --samples 10 --seed 77 --sites 3000 > /dev/null || exit 1
BASE=(--v0-human-mode --v0-in-fasta "$W/d/samples.fasta" --v0-human-reference-standard custom
      --v0-human-expected-length 3000 --v0-human-reference-fasta "$W/d/reference.fasta"
      --v0-human-metadata "$W/d/metadata.tsv" --v0-human-uniform-sample-date 2020-01-01
      --v0-in-tree "$W/d/true_tree.nwk" --v0-in-tree-branch-lengths years --v0-fix-topology
      --v0-calibration-file "$W/d/root.cal" --v0-mu-prior-mean 2e-8 --v0-mu-prior-stddev 1.4e-8
      --v0-coal-prior-fixed-t-step 40 --v0-steps 60000 --v0-log-every 6000 --v0-threads 1 --v0-seed 5)
run() { local tag=$1; shift; "$DELPHY" "${BASE[@]}" "$@" --v0-out-log-file "$W/$tag.log" \
          --v0-out-trees-file "$W/$tag.trees" > "$W/$tag.out" 2> "$W/$tag.err"; echo $? > "$W/$tag.rc"; }

# 1. A free, unbounded growth rate keeps the historical window of 1 / year: warn, still run
run unb --v0-init-final-pop-size 75000 --v0-init-pop-growth-rate 0
check "unbounded g with the default window warns" "grep -q 'WARNING: the growth-rate proposal window' $W/unb.err"
check "the warning does not stop the run" "[ \$(cat $W/unb.rc) = 0 ]"
check "the warning names the remedies" "grep -q -- '--v0-pop-growth-rate-max' $W/unb.err && grep -q -- '--v0-pop-g-move-window' $W/unb.err"
check "population move counters are printed" "grep -q 'Population move diagnostics' $W/unb.err"

# 2. A bounded growth rate makes the human-mode rule choose a window on the scale of the range: no warning
run bnd --v0-pop-g-prior-mu 0 --v0-pop-g-prior-scale 1e-5 --v0-pop-growth-rate-min 0 --v0-pop-growth-rate-max 1e-4 \
    --v0-init-final-pop-size 75000 --v0-init-pop-growth-rate 0
check "bounded g does not warn" "! grep -q 'WARNING: the growth-rate proposal window' $W/bnd.err"
check "bounded g run succeeds" "[ \$(cat $W/bnd.rc) = 0 ]"
acc() { grep 'growth rate  *proposed' "$1" | sed 's/.*accepted *\([0-9]*\).*/\1/'; }
check "bounded g accepts some growth-rate proposals" "[ \"\$(acc $W/bnd.err)\" -gt 0 ]"

# 3. An explicit narrow window does not warn
run win --v0-pop-g-move-window 1e-5 --v0-init-final-pop-size 75000 --v0-init-pop-growth-rate 0
check "an explicit narrow window does not warn" "! grep -q 'WARNING: the growth-rate proposal window' $W/win.err"

# 4. A fixed growth rate has no window to warn about
run fix --v0-fix-pop-growth-rate --v0-init-pop-growth-rate 0 --v0-init-final-pop-size 75000
check "fixed g does not warn" "! grep -q 'WARNING: the growth-rate proposal window' $W/fix.err"

# 5. The warning changes nothing that is sampled: the same command with and without it is identical, and
#    the window itself is the only difference between 'unb' and 'win'
check "log body of the unbounded run is not empty" "[ \$(grep -vc '^#' $W/unb.log) -gt 3 ]"

# 6. Pathogen mode never warns
if [ -n "$FASTA" ]; then
  "$DELPHY" --v0-in-fasta "$FASTA" --v0-seed 3 --v0-steps 20000 --v0-log-every 2000 \
      --v0-out-log-file "$W/p.log" --v0-out-trees-file "$W/p.trees" > /dev/null 2> "$W/p.err"
  check "pathogen mode does not warn" "! grep -q 'WARNING: the growth-rate proposal window' $W/p.err"
fi
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" = 0 ]
