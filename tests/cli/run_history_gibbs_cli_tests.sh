#!/usr/bin/env bash
# End-to-end checks of the single-site mutational-history Gibbs updates
# (doc/extended/history_gibbs.md) through the delphy CLI, on the example data of
# doc/extended/examples/observation (300 sites, 12 samples).  The prior values that appear on
# the command line are test-only placeholders.
#
# Usage: tests/cli/run_history_gibbs_cli_tests.sh PATH/TO/delphy WORKDIR
set -u
DELPHY=$(readlink -f "$1")
WORK=$2
HERE=$(cd "$(dirname "$0")" && pwd)
E=$(cd "$HERE/../../doc/extended/examples/observation" && pwd)
mkdir -p "$WORK"
fail=0

check() {  # check DESCRIPTION CONDITION-EXIT-STATUS
  if [ "$2" -eq 0 ]; then echo "PASS: $1"; else echo "FAIL: $1"; fail=$((fail + 1)); fi
}

common=(--v0-human-mode --v0-human-reference-standard custom --v0-human-expected-length 300
        --v0-human-reference-fasta "$E/reference.fasta" --v0-human-metadata "$E/metadata.tsv"
        --v0-human-max-missing-fraction 0.5 --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8
        --v0-seed 4 --v0-threads 1 --v0-steps 40000 --v0-log-every 2000 --v0-tree-every 40000
        --v0-in-fasta "$E/samples.fasta")

body() { grep -v '^# \(Produced by\|Invocation\)' "$1"; }

# 1. Off by default, and byte-identical to a run that never heard of the option.
"$DELPHY" "${common[@]}" --v0-out-log-file "$WORK/plain.log" \
    >"$WORK/plain.out" 2>"$WORK/plain.err"
check "no option at all: CLI exits 0" $?
grep -q "history Gibbs" "$WORK/plain.err"; st=$?
[ $st -ne 0 ]; check "nothing is said about the updates when they are off" $?
grep -q "branch_history_gibbs *selected *0," "$WORK/plain.err"
check "the branch-history update is never selected by default" $?
grep -q "node_state_gibbs *selected *0," "$WORK/plain.err"
check "the node-state update is never selected by default" $?

# 2. Both local updates on, with the default weight.
"$DELPHY" "${common[@]}" --v0-history-gibbs --v0-out-log-file "$WORK/gibbs.log" \
    >"$WORK/gibbs.out" 2>"$WORK/gibbs.err"
check "--v0-history-gibbs: CLI exits 0" $?
grep -q "Single-site history Gibbs updates: ON (weight 15, 1 site" "$WORK/gibbs.err"
check "start-up names the weight and the sites per update" $?
grep -q "Single-site history Gibbs diagnostics" "$WORK/gibbs.err"
check "the run reports the history-update diagnostics" $?
awk '/branch_history_gibbs/ {exit ($3+0 > 0) ? 0 : 1}' "$WORK/gibbs.err"
check "the branch-history update is selected" $?
awk '/node_state_gibbs/ {exit ($3+0 > 0) ? 0 : 1}' "$WORK/gibbs.err"
check "the node-state update is selected" $?
cmp -s <(body "$WORK/plain.log") <(body "$WORK/gibbs.log"); st=$?
[ $st -ne 0 ]; check "enabling the updates changes the chain" $?

# 3. The weight and the block size are honoured, and the whole-site refresh runs.
"$DELPHY" "${common[@]}" --v0-history-gibbs --v0-history-gibbs-weight 60 \
    --v0-history-gibbs-sites 8 --v0-site-history-refreshes 3 \
    --v0-out-log-file "$WORK/tuned.log" >"$WORK/tuned.out" 2>"$WORK/tuned.err"
check "weight, block size and refreshes together: CLI exits 0" $?
grep -q "Single-site history Gibbs updates: ON (weight 60, 8 site" "$WORK/tuned.err"
check "start-up names the tuned weight and block size" $?
grep -q "Whole-site history refresh: ON (3 site" "$WORK/tuned.err"
check "start-up names the refresh rate" $?
awk '/whole-site refreshes/ {exit ($3+0 > 0) ? 0 : 1}' "$WORK/tuned.err"
check "the whole-site refresh runs" $?
# A heavier weight must select the updates more often than the default weight did.
light=$(awk '/branch_history_gibbs/ {print $3+0}' "$WORK/gibbs.err")
heavy=$(awk '/branch_history_gibbs/ {print $3+0}' "$WORK/tuned.err")
[ "$heavy" -gt "$light" ]; check "a heavier weight selects the update more often ($light -> $heavy)" $?

# 4. The refresh alone, without the local updates.
"$DELPHY" "${common[@]}" --v0-site-history-refreshes 5 --v0-out-log-file "$WORK/refresh.log" \
    >"$WORK/refresh.out" 2>"$WORK/refresh.err"
check "--v0-site-history-refreshes alone: CLI exits 0" $?
grep -q "branch_history_gibbs *selected *0," "$WORK/refresh.err"
check "the refresh alone leaves the local mix untouched" $?
awk '/whole-site refreshes/ {exit ($3+0 > 0) ? 0 : 1}' "$WORK/refresh.err"
check "the refresh alone still refreshes" $?

# 5. The updates combine with the features they are meant to combine with.
"$DELPHY" "${common[@]}" --v0-history-gibbs --v0-subst-model GTR --v0-site-rate-heterogeneity \
    --v0-site-history-refreshes 2 --v0-out-log-file "$WORK/gtr_gamma.log" \
    >"$WORK/gtr_gamma.out" 2>"$WORK/gtr_gamma.err"
check "GTR with continuous Gamma site rates: CLI exits 0" $?
"$DELPHY" "${common[@]}" --v0-history-gibbs --v0-threads 4 --v0-out-log-file "$WORK/threads.log" \
    >"$WORK/threads.out" 2>"$WORK/threads.err"
check "four threads: CLI exits 0" $?
"$DELPHY" "${common[@]}" --v0-history-gibbs --v0-in-tree "$E/../calibration/tree.nwk" \
    --v0-out-log-file "$WORK/tree.log" >"$WORK/tree.out" 2>"$WORK/tree.err"
st=$?
# The example tree belongs to another data set, so this is only a check that the option combination
# is accepted or refused with a message, never that it crashes.
grep -qi "error\|Error" "$WORK/tree.err" || [ $st -eq 0 ]
check "a tree file is either used or refused with a message" $?

# 6. Refusals.
"$DELPHY" "${common[@]}" --v0-history-gibbs --v0-history-gibbs-weight -1 \
    --v0-out-log-file "$WORK/bad1.log" >"$WORK/bad1.out" 2>"$WORK/bad1.err"
[ $? -ne 0 ]; check "a negative weight is refused" $?
grep -q "non-negative" "$WORK/bad1.err"; check "the refusal says what is wrong with the weight" $?
"$DELPHY" "${common[@]}" --v0-history-gibbs --v0-history-gibbs-sites 0 \
    --v0-out-log-file "$WORK/bad2.log" >"$WORK/bad2.out" 2>"$WORK/bad2.err"
[ $? -ne 0 ]; check "a zero block size is refused" $?
grep -q "at least one site" "$WORK/bad2.err"; check "the refusal says what is wrong with the block size" $?
"$DELPHY" "${common[@]}" --v0-site-history-refreshes -2 \
    --v0-out-log-file "$WORK/bad3.log" >"$WORK/bad3.out" 2>"$WORK/bad3.err"
[ $? -ne 0 ]; check "a negative refresh count is refused" $?
"$DELPHY" "${common[@]}" --v0-history-gibbs-weight 20 --v0-out-log-file "$WORK/bad4.log" \
    >"$WORK/bad4.out" 2>"$WORK/bad4.err"
[ $? -ne 0 ]; check "the weight without --v0-history-gibbs is refused" $?
grep -q "need --v0-history-gibbs" "$WORK/bad4.err"; check "the refusal says which option is missing" $?
"$DELPHY" --v0-in-fasta "$E/samples.fasta" --v0-history-gibbs --v0-seed 4 --v0-steps 1000 \
    --v0-out-log-file "$WORK/bad5.log" >"$WORK/bad5.out" 2>"$WORK/bad5.err"
[ $? -ne 0 ]; check "the updates without --v0-human-mode are refused" $?
grep -q "only valid together with --v0-human-mode" "$WORK/bad5.err"
check "the refusal names the mode the option needs" $?

# 7. The locally marginalized node-time move.
"$DELPHY" "${common[@]}" --v0-history-gibbs --v0-marginal-node-time-move \
    --v0-out-log-file "$WORK/marginal.log" >"$WORK/marginal.out" 2>"$WORK/marginal.err"
check "--v0-marginal-node-time-move: CLI exits 0" $?
grep -q "Marginal node-time move: ON (weight 7.5, window 0.5" "$WORK/marginal.err"
check "start-up names the weight and the window" $?
awk '/node_time_marginal/ {exit ($3+0 > 0) ? 0 : 1}' "$WORK/marginal.err"
check "the marginal node-time move is selected" $?
awk '/node_time_marginal/ {exit ($5+0 > 0) ? 0 : 1}' "$WORK/marginal.err"
check "the marginal node-time move is accepted" $?
grep -q "node_time_marginal *selected *0," "$WORK/gibbs.err"
check "the marginal node-time move is off without its own option" $?
"$DELPHY" "${common[@]}" --v0-marginal-node-time-move --v0-marginal-node-time-weight 15 \
    --v0-marginal-node-time-window 1.5 --v0-out-log-file "$WORK/marginal2.log" \
    >"$WORK/marginal2.out" 2>"$WORK/marginal2.err"
check "the move without the history updates: CLI exits 0" $?
"$DELPHY" "${common[@]}" --v0-marginal-node-time-move --v0-marginal-node-time-weight -1 \
    --v0-out-log-file "$WORK/bad6.log" >"$WORK/bad6.out" 2>"$WORK/bad6.err"
[ $? -ne 0 ]; check "a negative marginal weight is refused" $?
"$DELPHY" "${common[@]}" --v0-marginal-node-time-move --v0-marginal-node-time-window 0 \
    --v0-out-log-file "$WORK/bad7.log" >"$WORK/bad7.out" 2>"$WORK/bad7.err"
[ $? -ne 0 ]; check "a zero marginal window is refused" $?
# A window that reaches the move's grid guard would make that guard reject asymmetrically and break
# detailed balance, so anything at or above log(1000) = 6.908 is refused and anything below is not.
"$DELPHY" "${common[@]}" --v0-marginal-node-time-move --v0-marginal-node-time-window 7 \
    --v0-out-log-file "$WORK/bad9.log" >"$WORK/bad9.out" 2>"$WORK/bad9.err"
[ $? -ne 0 ]; check "a marginal window at or above log(1000) is refused" $?
grep -q "must be smaller than" "$WORK/bad9.err"
check "the refusal names the bound" $?
"$DELPHY" "${common[@]}" --v0-marginal-node-time-move --v0-marginal-node-time-window 6.9 \
    --v0-out-log-file "$WORK/win.log" >"$WORK/win.out" 2>"$WORK/win.err"
check "the largest usable marginal window is accepted" $?
"$DELPHY" "${common[@]}" --v0-marginal-node-time-weight 3 \
    --v0-out-log-file "$WORK/bad8.log" >"$WORK/bad8.out" 2>"$WORK/bad8.err"
[ $? -ne 0 ]; check "the marginal weight without its move is refused" $?
grep -q "need --v0-marginal-node-time-move" "$WORK/bad8.err"
check "the refusal names the option that is missing" $?

echo
if [ $fail -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "$fail CHECK(S) FAILED"; fi
exit $fail
