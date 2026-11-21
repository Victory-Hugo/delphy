#!/usr/bin/env bash
# Command-line behaviour of the coalescent-grid options and of the human-mode Skygrid guards.
# Each check is a single short Delphy invocation; nothing here needs a converged chain.
#
# Usage: run_coalescent_cli_tests.sh PATH/TO/delphy WORKDIR
set -u
DELPHY=$(readlink -f "$1")
WORK=$2
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
E=$REPO/doc/extended/examples/calibration

mkdir -p "$WORK"
pass=0; fail=0
LAST_OUT=""

check() {  # check NAME EXPECT(ok|fail) PATTERN -- COMMAND...
  local name=$1 expect=$2 pattern=$3
  shift 4
  LAST_OUT="$WORK/$(printf '%s' "$name" | tr -c 'A-Za-z0-9_' '_')"
  local out=$LAST_OUT
  "$@" > "$out.stdout" 2> "$out.stderr"
  local rc=$?
  local ok=1
  if [ "$expect" = ok ] && [ $rc -ne 0 ]; then ok=0; fi
  if [ "$expect" = fail ] && [ $rc -eq 0 ]; then ok=0; fi
  if [ -n "$pattern" ] && ! grep -qF -- "$pattern" "$out.stderr"; then ok=0; fi
  if [ $ok -eq 1 ]; then
    pass=$((pass + 1)); echo "PASS  $name"
  else
    fail=$((fail + 1)); echo "FAIL  $name (rc=$rc, expected $expect, pattern '$pattern')"
    tail -3 "$out.stderr" | sed 's/^/        /'
  fi
}

human=(--v0-human-mode --v0-in-fasta "$E/samples.fasta"
       --v0-human-reference-standard custom --v0-human-expected-length 300
       --v0-human-reference-fasta "$E/reference.fasta" --v0-human-metadata "$E/metadata.tsv"
       --v0-human-uniform-sample-date 2020-01-01
       --v0-in-tree "$E/tree.nwk" --v0-in-tree-branch-lengths years --v0-fix-topology
       --v0-calibration-file "$E/calibration.cal"
       --v0-fix-final-pop-size --v0-init-final-pop-size 50000
       --v0-fix-pop-growth-rate --v0-init-pop-growth-rate 0
       --v0-seed 1 --v0-threads 1 --v0-steps 20000 --v0-log-every 20000)

# --- option validation
check "init-cells rejects a negative value" fail "must be non-negative" -- \
  "$DELPHY" "${human[@]}" --v0-coal-prior-init-cells -1 --v0-out-log-file "$WORK/a.log"
check "grid-tolerance rejects a negative value" fail "must be non-negative" -- \
  "$DELPHY" "${human[@]}" --v0-coal-prior-grid-tolerance -0.5 --v0-out-log-file "$WORK/b.log"
check "max-cells must be at least the target" fail "must be at least" -- \
  "$DELPHY" "${human[@]}" --v0-target-coal-prior-cells 4000 --v0-coal-prior-max-cells 400 \
  --v0-out-log-file "$WORK/c.log"

# --- the grid is reported in human mode
check "human mode reports the grid it built" ok "Coalescent prior grid:" -- \
  "$DELPHY" "${human[@]}" --v0-target-coal-prior-cells 4000 --v0-out-log-file "$WORK/d.log"
check "human mode reports the tolerance when one is set" ok "(tolerance 0.1)" -- \
  "$DELPHY" "${human[@]}" --v0-target-coal-prior-cells 4000 --v0-coal-prior-grid-tolerance 0.1 \
  --v0-out-log-file "$WORK/e.log"
check "an unmeetable tolerance warns instead of claiming success" ok "above the requested tolerance" -- \
  "$DELPHY" "${human[@]}" --v0-target-coal-prior-cells 400 --v0-coal-prior-grid-tolerance 1e-30 \
  --v0-coal-prior-max-cells 800 --v0-out-log-file "$WORK/f.log"

# --- the legacy one-day grid is still reachable, and is what the option turns off
grid_cells() { sed -n 's/^Coalescent prior grid: \([0-9]*\) cells.*/\1/p' "$1" | tail -1; }
"$DELPHY" "${human[@]}" --v0-target-coal-prior-cells 4000 --v0-coal-prior-init-cells 0 \
  --v0-out-log-file "$WORK/legacy.log" > /dev/null 2> "$WORK/legacy.stderr"
"$DELPHY" "${human[@]}" --v0-target-coal-prior-cells 4000 \
  --v0-out-log-file "$WORK/sized.log" > /dev/null 2> "$WORK/sized.stderr"
legacy=$(grid_cells "$WORK/legacy.stderr"); sized=$(grid_cells "$WORK/sized.stderr")
if [ -n "$legacy" ] && [ -n "$sized" ] && [ "$legacy" -gt "$sized" ]; then
  pass=$((pass + 1)); echo "PASS  init-cells 0 restores the one-day grid ($legacy cells vs $sized)"
else
  fail=$((fail + 1)); echo "FAIL  init-cells 0 restores the one-day grid (legacy='$legacy' sized='$sized')"
fi

# --- human-mode Skygrid guards
skygrid=(--v0-human-mode --v0-in-fasta "$E/samples.fasta"
         --v0-human-reference-standard custom --v0-human-expected-length 300
         --v0-human-reference-fasta "$E/reference.fasta" --v0-human-metadata "$E/metadata.tsv"
         --v0-human-uniform-sample-date 2020-01-01
         --v0-mu-prior-mean 1.6e-7 --v0-mu-prior-stddev 4.8e-8 --v0-init-mutation-rate 1.6e-7
         --v0-pop-model skygrid --v0-skygrid-cutoff 400000 --v0-skygrid-num-parameters 41
         --v0-seed 1 --v0-threads 1 --v0-steps 20000 --v0-log-every 20000)

check "human skygrid refuses the pathogen smoothness default" fail "pathogen value" -- \
  "$DELPHY" "${skygrid[@]}" --v0-out-log-file "$WORK/g.log"
check "human skygrid accepts an explicit double-half time" ok "" -- \
  "$DELPHY" "${skygrid[@]}" --v0-skygrid-prior-double-half-time 10000 \
  --v0-skygrid-low-pop-barrier-loc 100 --v0-out-log-file "$WORK/h.log"
check "human skygrid accepts an explicit tau" ok "" -- \
  "$DELPHY" "${skygrid[@]}" --v0-skygrid-tau 2.08 \
  --v0-skygrid-low-pop-barrier-loc 100 --v0-out-log-file "$WORK/i.log"
check "human skygrid accepts inferring the smoothness" ok "" -- \
  "$DELPHY" "${skygrid[@]}" --v0-skygrid-infer-prior-smoothness true \
  --v0-skygrid-low-pop-barrier-loc 100 --v0-out-log-file "$WORK/j.log"
check "human skygrid warns about the one-day population floor" ok "low-population barrier is at its default" -- \
  "$DELPHY" "${skygrid[@]}" --v0-skygrid-prior-double-half-time 10000 --v0-out-log-file "$WORK/k.log"
check "disabling the barrier silences that warning" ok "" -- \
  "$DELPHY" "${skygrid[@]}" --v0-skygrid-prior-double-half-time 10000 \
  --v0-skygrid-disable-low-pop-barrier true --v0-out-log-file "$WORK/l.log"
if grep -q "low-population barrier is at its default" "$LAST_OUT.stderr"; then
  fail=$((fail + 1)); echo "FAIL  the barrier warning must not appear once the barrier is disabled"
else
  pass=$((pass + 1)); echo "PASS  the barrier warning does not appear once the barrier is disabled"
fi

# --- pathogen defaults say nothing new
cat > "$WORK/patho.fasta" <<'EOF'
>A|2021-01-01
ACGTACGTACGTACGTACGT
>B|2021-02-01
ACGTACGTACGAACGTACGT
>C|2021-03-01
ACGTACGTACGTACGTACGA
>D|2021-04-01
ACGTACGTTCGTACGTACGT
EOF
"$DELPHY" --v0-in-fasta "$WORK/patho.fasta" --v0-seed 1 --v0-threads 1 --v0-steps 20000 \
  --v0-log-every 20000 --v0-out-log-file "$WORK/patho.log" > /dev/null 2> "$WORK/patho.stderr"
if grep -qE "Coalescent prior grid|low-population barrier|discretization excess" "$WORK/patho.stderr"; then
  fail=$((fail + 1)); echo "FAIL  a default pathogen run must not gain new diagnostics"
  grep -E "Coalescent prior grid|low-population barrier|discretization excess" "$WORK/patho.stderr" | sed 's/^/        /'
else
  pass=$((pass + 1)); echo "PASS  a default pathogen run gains no new diagnostics"
fi
check "a pathogen run still accepts the new options explicitly" ok "Coalescent prior grid:" -- \
  "$DELPHY" --v0-in-fasta "$WORK/patho.fasta" --v0-seed 1 --v0-threads 1 --v0-steps 20000 \
  --v0-log-every 20000 --v0-coal-prior-grid-tolerance 0.01 --v0-out-log-file "$WORK/patho2.log"

# --- the retarget trace is opt-in, and reports the change a new cell width imposes on the log prior
"$DELPHY" "${human[@]}" --v0-target-coal-prior-cells 400 --v0-coal-prior-grid-tolerance 0.001 \
  --v0-coal-prior-trace-retargets true --v0-out-log-file "$WORK/trace_on.log" \
  > /dev/null 2> "$WORK/trace_on.stderr"
"$DELPHY" "${human[@]}" --v0-target-coal-prior-cells 400 --v0-coal-prior-grid-tolerance 0.001 \
  --v0-out-log-file "$WORK/trace_off.log" > /dev/null 2> "$WORK/trace_off.stderr"
on=$(grep -c "^RETARGET step=.* delta=" "$WORK/trace_on.stderr")
off=$(grep -c "^RETARGET" "$WORK/trace_off.stderr")
if [ "$on" -ge 1 ] && [ "$off" -eq 0 ]; then
  pass=$((pass + 1)); echo "PASS  the retarget trace is opt-in ($on lines when on, $off when off)"
else
  fail=$((fail + 1)); echo "FAIL  the retarget trace is opt-in ($on lines when on, $off when off)"
fi

# --- a pinned grid is never retargeted from the sampled state
check "fixed-t-step rejects a negative value" fail "must be non-negative" -- \
  "$DELPHY" "${human[@]}" --v0-coal-prior-fixed-t-step -1 --v0-out-log-file "$WORK/f1.log"
check "fixed-t-step cannot be combined with the tolerance" fail "cannot be combined" -- \
  "$DELPHY" "${human[@]}" --v0-coal-prior-fixed-t-step 3 --v0-coal-prior-grid-tolerance 0.1 \
  --v0-out-log-file "$WORK/f2.log"
"$DELPHY" "${human[@]}" --v0-target-coal-prior-cells 400 --v0-coal-prior-fixed-t-step 3 \
  --v0-coal-prior-trace-retargets true --v0-out-log-file "$WORK/f3.log" \
  > /dev/null 2> "$WORK/f3.stderr"
pinned=$(grep -c "^RETARGET" "$WORK/f3.stderr")
if [ "$pinned" -eq 1 ] && grep -q "^RETARGET .* -> 3 " "$WORK/f3.stderr"; then
  pass=$((pass + 1)); echo "PASS  a pinned grid is set once and never retargeted ($pinned RETARGET line)"
else
  fail=$((fail + 1)); echo "FAIL  a pinned grid is set once and never retargeted ($pinned RETARGET lines)"
fi

# --- the growth-rate proposal window
check "pop-g-move-window rejects a negative value" fail "must be non-negative" -- \
  "$DELPHY" "${human[@]}" --v0-pop-g-move-window -1 --v0-out-log-file "$WORK/w1.log"
check "pop-g-move-window is accepted" ok "" -- \
  "$DELPHY" "${human[@]}" --v0-pop-g-move-window 1e-5 --v0-out-log-file "$WORK/w2.log"

# --- local-move efficiency options (opt-in; the sampled distribution is unchanged)
check "local-moves-per-node rejects zero" fail "must be at least 1" -- \
  "$DELPHY" "${human[@]}" --v0-local-moves-per-node-per-global-move 0 --v0-out-log-file "$WORK/m1.log"
check "pop-moves-per-global-move rejects zero" fail "must be at least 1" -- \
  "$DELPHY" "${human[@]}" --v0-pop-moves-per-global-move 0 --v0-out-log-file "$WORK/m0.log"
check "root-displace-scale-factor rejects zero" fail "must be positive" -- \
  "$DELPHY" "${human[@]}" --v0-root-displace-scale-factor 0 --v0-out-log-file "$WORK/m2.log"
check "root-displace-scale-factor rejects a negative value" fail "must be positive" -- \
  "$DELPHY" "${human[@]}" --v0-root-displace-scale-factor -2 --v0-out-log-file "$WORK/m3.log"
check "the three efficiency options are accepted together" ok "" -- \
  "$DELPHY" "${human[@]}" --v0-skip-fixed-tip-moves true --v0-local-moves-per-node-per-global-move 10 \
  --v0-root-displace-scale-factor 2 --v0-out-log-file "$WORK/m4.log"
tip_line=$(grep "tip_displace" "$(printf '%s' "$LAST_OUT").stderr" | head -1)
if echo "$tip_line" | grep -q "selected  *0,"; then
  pass=$((pass + 1)); echo "PASS  with all tip dates fixed, --v0-skip-fixed-tip-moves leaves no tip move"
else
  fail=$((fail + 1)); echo "FAIL  with all tip dates fixed, --v0-skip-fixed-tip-moves leaves no tip move ($tip_line)"
fi

# --- the Skygrid smoothness precision must be the documented function of the knot spacing
# tau = T / (log^2(2) dt) with dt = cutoff / (num_parameters - 1); for T = dt that is 1/log^2(2).
"$DELPHY" "${skygrid[@]}" --v0-skygrid-prior-double-half-time 10000 \
  --v0-skygrid-low-pop-barrier-loc 100 --v0-out-log-file "$WORK/tau.log" \
  > /dev/null 2> "$WORK/tau.stderr"
got_tau=$(awk -F'\t' 'NR==1{next} /^#/{next} {for(i=1;i<=NF;i++) if(h[i]=="skygrid.precision") print $i}
                       /^state/{for(i=1;i<=NF;i++) h[i]=$i}' "$WORK/tau.log" | head -1)
want_tau=$(python3 -c "import math; print(10000.0 / (math.log(2.0)**2 * 400000.0/40.0))")
if [ -n "$got_tau" ] && python3 -c "import sys; sys.exit(0 if abs($got_tau/$want_tau - 1) < 1e-6 else 1)"; then
  pass=$((pass + 1))
  echo "PASS  the Skygrid precision follows T / (log^2(2) dt): got $got_tau, expected $want_tau"
else
  fail=$((fail + 1))
  echo "FAIL  the Skygrid precision follows T / (log^2(2) dt): got '$got_tau', expected $want_tau"
fi

# --- the coalescent log prior Delphy reports must match an independent reconstruction
"$DELPHY" "${human[@]}" --v0-fix-mutation-rate --v0-init-mutation-rate 1.6e-7 \
  --v0-target-coal-prior-cells 4000 --v0-out-trees-digits 15 \
  --v0-steps 2000000 --v0-log-every 200000 --v0-tree-every 200000 \
  --v0-out-log-file "$WORK/xcheck.log" --v0-out-trees-file "$WORK/xcheck.trees" \
  > /dev/null 2> "$WORK/xcheck.stderr"
python3 "$HERE/check_coalescent_log_prior.py" "$WORK/xcheck.trees" "$WORK/xcheck.log" \
  --pop-size-years 50000 --grid-cells 4000 --max-trees 11 > "$WORK/xcheck.tsv" 2>&1
worst=$(awk 'NR>1 {d=$6; if (d<0) d=-d; if (d>m) m=d} END {printf "%.6f", m+0}' "$WORK/xcheck.tsv")
worst_exact=$(awk 'NR>1 {d=$7; if (d<0) d=-d; if (d>m) m=d} END {printf "%.6f", m+0}' "$WORK/xcheck.tsv")
rows=$(awk 'NR>1' "$WORK/xcheck.tsv" | wc -l)
if [ "$rows" -ge 5 ] && awk "BEGIN{exit !($worst < 0.01)}" && awk "BEGIN{exit !($worst_exact < 0.01)}"; then
  pass=$((pass + 1))
  echo "PASS  the reported coalescent log prior matches an independent reconstruction"
  echo "        $rows states; worst |delphy - cells| = $worst nats, worst |cells - exact| = $worst_exact nats"
else
  fail=$((fail + 1))
  echo "FAIL  the reported coalescent log prior matches an independent reconstruction"
  echo "        rows=$rows worst|delphy-cells|=$worst worst|cells-exact|=$worst_exact"
  tail -4 "$WORK/xcheck.tsv" | sed 's/^/        /'
fi

echo
echo "coalescent CLI checks: $pass passed, $fail failed"
exit $((fail > 0))
