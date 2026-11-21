#!/usr/bin/env bash
# Command-line behaviour of the exact Kingman correction under a Skygrid population model, and an
# independent reconstruction of the coalescent term the shipped binary reports.
#
# The reconstruction is the end-to-end check: with the correction on, the Skygrid column of the log
# must equal the exact Kingman density of the tree in the .trees file under the curve in the log, and
# with the correction off it must equal the cell prior on the pinned grid instead.  Both sides are
# recomputed by check_exact_skygrid_log_prior.py, which merges the events and writes out the Skygrid
# rate integral itself.
#
# Usage: run_exact_skygrid_cli_tests.sh PATH/TO/delphy WORKDIR
set -u
DELPHY=$(readlink -f "$1")
WORK=$2
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
E=$REPO/doc/extended/examples/calibration
PY=${PY:-python3}

mkdir -p "$WORK"
pass=0; fail=0

check() {  # check NAME EXPECT(ok|fail) PATTERN -- COMMAND...
  local name=$1 expect=$2 pattern=$3
  shift 4
  local out="$WORK/$(printf '%s' "$name" | tr -c 'A-Za-z0-9_' '_')"
  "$@" > "$out.stdout" 2> "$out.stderr"
  local rc=$?
  local ok=1
  [ "$expect" = ok ] && [ $rc -ne 0 ] && ok=0
  [ "$expect" = fail ] && [ $rc -eq 0 ] && ok=0
  if [ -n "$pattern" ] && ! grep -qF -- "$pattern" "$out.stderr"; then ok=0; fi
  if [ $ok -eq 1 ]; then
    pass=$((pass + 1)); echo "PASS  $name"
  else
    fail=$((fail + 1)); echo "FAIL  $name (rc=$rc, expected $expect, pattern '$pattern')"
    tail -3 "$out.stderr" | sed 's/^/        /'
  fi
}

# A deep-time Skygrid run on the example data set: 6 samples over 140 ka, 41 knots over 400 ka, the
# barrier off and the zero mode's prior off so that the GMRF prior the log folds into the Skygrid
# column is exactly the first-difference term the reconstruction subtracts.
base=(--v0-human-mode --v0-in-fasta "$E/samples.fasta"
      --v0-human-reference-standard custom --v0-human-expected-length 300
      --v0-human-reference-fasta "$E/reference.fasta" --v0-human-metadata "$E/metadata.tsv"
      --v0-human-uniform-sample-date 2020-01-01
      --v0-in-tree "$E/tree.nwk" --v0-in-tree-branch-lengths years
      --v0-fix-mutation-rate --v0-init-mutation-rate 1.6e-7
      --v0-pop-model skygrid --v0-skygrid-cutoff 400000 --v0-skygrid-num-parameters 41
      --v0-skygrid-tau 2.08 --v0-skygrid-disable-low-pop-barrier true
      --v0-skygrid-inv-nbar-prior-alpha 0 --v0-skygrid-inv-nbar-prior-beta 0
      --v0-seed 5 --v0-threads 1)
TSTEP=100    # years; 4,000 cells over the knot range, for the guard checks
# The reconstruction runs use a deliberately coarse grid: the two candidate targets are then far
# apart, so "which one does the log report" has an unambiguous answer.  On a fine grid they agree to
# within the precision of the .trees file and the question cannot be decided.
RSTEP=20000  # years; 20 cells over the knot range

check "the correction is accepted with a Skygrid curve and a pinned grid" ok "Exact coalescent correction: ON" -- \
  "$DELPHY" "${base[@]}" --v0-fix-topology --v0-coal-prior-fixed-t-step $TSTEP \
  --v0-exact-coalescent-correction true --v0-steps 20000 --v0-log-every 20000 \
  --v0-out-log-file "$WORK/a.log"

check "the correction is refused without a pinned grid" fail "grid must be pinned" -- \
  "$DELPHY" "${base[@]}" --v0-fix-topology \
  --v0-exact-coalescent-correction true --v0-steps 20000 --v0-log-every 20000 \
  --v0-out-log-file "$WORK/b.log"

check "the correction is refused with a free topology on several threads" fail "needs --v0-threads 1" -- \
  "$DELPHY" "${base[@]}" --v0-coal-prior-fixed-t-step $TSTEP --v0-threads 2 \
  --v0-exact-coalescent-correction true --v0-steps 20000 --v0-log-every 20000 \
  --v0-out-log-file "$WORK/c.log"

check "the joint time-scale move is still refused with a Skygrid curve" fail "" -- \
  "$DELPHY" "${base[@]}" --v0-fix-topology --v0-coal-prior-fixed-t-step $TSTEP \
  --v0-exact-coalescent-correction true --v0-global-scale-move true \
  --v0-steps 20000 --v0-log-every 20000 --v0-out-log-file "$WORK/d.log"

# --- the reported coalescent term, reconstructed independently
reconstruct() {  # reconstruct TAG TYPE EXACT(0|1)
  local tag=$1 type=$2 exact=$3
  local extra=()
  [ "$exact" = 1 ] && extra=(--v0-exact-coalescent-correction true)
  "$DELPHY" "${base[@]}" --v0-fix-topology --v0-coal-prior-fixed-t-step $RSTEP \
    --v0-skygrid-type "$type" "${extra[@]+"${extra[@]}"}" --v0-out-trees-digits 15 \
    --v0-steps 400000 --v0-log-every 40000 --v0-tree-every 40000 \
    --v0-out-log-file "$WORK/$tag.log" --v0-out-trees-file "$WORK/$tag.trees" \
    > /dev/null 2> "$WORK/$tag.err" || { echo "run $tag failed"; return 1; }
  "$PY" "$HERE/check_exact_skygrid_log_prior.py" "$WORK/$tag.trees" "$WORK/$tag.log" \
    --t-step-years $RSTEP > "$WORK/$tag.tsv" 2>&1 || { echo "reconstruction $tag failed"; return 1; }
}

# `want` is the column the run should match; the other one must be at least ten times further away,
# which is what makes this a two-sided check rather than a tolerance that both sides could meet.
judge() {  # judge NAME TAG want(exact|cells)
  local name=$1 tag=$2 want=$3
  local line
  line=$(grep '^# worst' "$WORK/$tag.tsv" 2>/dev/null) || line=""
  local we wc rows
  we=$(sed -n 's/.*|reported - exact| = \([^ ]*\) relative.*/\1/p' <<< "$line")
  wc=$(sed -n 's/.*|reported - cells| = \([^ ]*\) relative.*/\1/p' <<< "$line")
  rows=$(sed -n 's/.*relative; \([0-9]*\) states.*/\1/p' <<< "$line")
  local near far
  if [ "$want" = exact ]; then near=$we; far=$wc; else near=$wc; far=$we; fi
  if [ -n "$near" ] && [ -n "$far" ] && [ "${rows:-0}" -ge 5 ] \
     && awk "BEGIN{exit !($near < 1e-4)}" && awk "BEGIN{exit !($far > 10 * $near)}"; then
    pass=$((pass + 1))
    echo "PASS  $name"
    echo "        $rows states; |reported - $want| = $near relative, the other reconstruction $far"
  else
    fail=$((fail + 1))
    echo "FAIL  $name (want $want; |rep-exact|=$we |rep-cells|=$wc rows=${rows:-0})"
    tail -3 "$WORK/$tag.tsv" | sed 's/^/        /'
  fi
}

reconstruct sc_exact staircase 1   && judge "with the correction on, the staircase run reports the exact Kingman density" sc_exact exact
reconstruct sc_cell  staircase 0   && judge "with the correction off, the staircase run reports the cell prior"            sc_cell  cells
reconstruct ll_exact log-linear 1  && judge "with the correction on, the log-linear run reports the exact Kingman density" ll_exact exact
reconstruct ll_cell  log-linear 0  && judge "with the correction off, the log-linear run reports the cell prior"           ll_cell  cells

echo
echo "exact Skygrid CLI checks: $pass passed, $fail failed"
exit $((fail > 0))
