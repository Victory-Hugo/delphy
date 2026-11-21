#!/usr/bin/env bash
# CLI checks for --v0-exact-coalescent-correction (experimental, off by default): unsupported configurations are refused with a
# message that says what to change, the supported configuration runs, and without the flag nothing changes.
# Usage: tests/cli/run_exact_correction_cli_tests.sh PATH/TO/delphy WORKDIR
set -u
DELPHY=$(readlink -f "$1")
WORK=$2
HERE=$(cd "$(dirname "$0")" && pwd)
D="$WORK/data"
mkdir -p "$WORK"
[ -f "$D/truth.json" ] || python3 "$HERE/simulate_mtdna_coalescent.py" "$D" --samples 20 --seed 1 --mu-site-year 1.7e-8 \
    --coal-timescale-years 75000 --dates same --mask 303-315,16184-16193 > /dev/null
fail=0
T="$D/true_tree.nwk"
base=("$DELPHY" --v0-human-mode --v0-in-fasta "$D/samples.fasta" --v0-human-reference-standard rCRS
      --v0-human-reference-fasta "$D/reference.fasta" --v0-human-mask-file "$D/mask.txt"
      --v0-human-audit-report "$WORK/audit.json" --v0-human-uniform-sample-date 2020-01-01
      --v0-mu-prior-mean 1.7e-8 --v0-mu-prior-stddev 5.1e-9 --v0-steps 200000 --v0-seed 1
      --v0-in-tree "$T" --v0-in-tree-branch-lengths years)

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

cat > "$WORK/root.cal" <<'CAL'
version 1

calibration root
  taxa all
  age_units years_bp
  age_reference_year 2020
  distribution lognormal
  median 120000
  sigma 0.5
  oldest 10000000
  youngest 1000
end
CAL
# cxxopts boolean options take an implicit "true": write explicit values as --flag=true / --flag=false
expect_error "free topology with several threads is refused" "needs --v0-threads 1" --v0-exact-coalescent-correction=true --v0-coal-prior-fixed-t-step 10 --v0-threads 2
expect_ok "free topology with one thread runs" --v0-exact-coalescent-correction=true --v0-coal-prior-fixed-t-step 10 --v0-threads 1 \
    --v0-out-log-file "$WORK/free.log" --v0-out-trees-file "$WORK/free.trees"
grep -q "topology moves enabled" "$WORK/err.txt" && echo "PASS: the free-topology corrected run samples the topology" || { echo "FAIL: topology moves"; fail=$((fail+1)); }
expect_error "an unpinned grid is refused" "grid must be pinned" --v0-exact-coalescent-correction=true --v0-fix-topology
expect_ok "calibrations with several threads run" --v0-exact-coalescent-correction=true --v0-fix-topology --v0-threads 2 \
    --v0-coal-prior-fixed-t-step 10 --v0-calibration-file "$WORK/root.cal" --v0-out-log-file "$WORK/cal2.log" --v0-out-trees-file "$WORK/cal2.trees"
expect_ok "calibrations with one thread run" --v0-exact-coalescent-correction=true --v0-fix-topology --v0-threads 1 \
    --v0-coal-prior-fixed-t-step 10 --v0-calibration-file "$WORK/root.cal" --v0-out-log-file "$WORK/cal.log" --v0-out-trees-file "$WORK/cal.trees"
grep -q "logP(calibrations)" "$WORK/cal.log" && echo "PASS: the calibrated corrected run logs the calibration column" || { echo "FAIL: calibration column"; fail=$((fail+1)); }
read -r s1 s2 < <(grep '^>' "$D/samples.fasta" | sed -n '1p;5p' | sed 's/^>//; s/[ |].*//' | paste -sd' ')
{ cat "$WORK/root.cal"; printf '\ncalibration pair\n  taxa %s %s\n  monophyly not_required\n  age_units years_bp\n  age_reference_year 2020\n  distribution uniform\n  youngest 100\n  oldest 10000000\nend\n' "$s1" "$s2"; } > "$WORK/free.cal"
expect_ok "free-topology calibrations (one age-only set) with the correction run on one thread" \
    --v0-exact-coalescent-correction=true --v0-allow-free-topology-calibration --v0-threads 1 \
    --v0-coal-prior-fixed-t-step 10 --v0-calibration-file "$WORK/free.cal" --v0-out-log-file "$WORK/freecal.log" --v0-out-trees-file "$WORK/freecal.trees"
expect_error "free-topology calibrations with the correction on several threads are refused" "v0-threads 1" \
    --v0-exact-coalescent-correction=true --v0-allow-free-topology-calibration --v0-threads 2 \
    --v0-coal-prior-fixed-t-step 10 --v0-calibration-file "$WORK/free.cal"
expect_ok "the supported configuration runs" --v0-exact-coalescent-correction=true --v0-fix-topology --v0-coal-prior-fixed-t-step 10 \
    --v0-out-log-file "$WORK/on.log" --v0-out-trees-file "$WORK/on.trees"
grep -q "Exact coalescent correction: ON" "$WORK/err.txt" && echo "PASS: diagnostics report the correction" || { echo "FAIL: diagnostics"; fail=$((fail+1)); }

# Without the flag the same run is unchanged by the option's existence: --flag=false is byte-identical to the default
"${base[@]}" --v0-fix-topology --v0-coal-prior-fixed-t-step 10 --v0-out-log-file "$WORK/off1.log" --v0-out-trees-file "$WORK/off1.trees" > /dev/null 2>&1
"${base[@]}" --v0-fix-topology --v0-coal-prior-fixed-t-step 10 --v0-exact-coalescent-correction=false --v0-out-log-file "$WORK/off2.log" --v0-out-trees-file "$WORK/off2.trees" > /dev/null 2>&1
for e in log trees; do
  if cmp -s <(grep -v '^# \(Produced by\|Invocation\)' "$WORK/off1.$e") <(grep -v '^# \(Produced by\|Invocation\)' "$WORK/off2.$e"); then
    echo "PASS: explicit false equals the default (.$e)"; else echo "FAIL: explicit false differs from the default (.$e)"; fail=$((fail+1)); fi
done
"$DELPHY" --v0-in-fasta "$D/samples.fasta" --v0-exact-coalescent-correction=true > /dev/null 2> "$WORK/err2.txt"; rc=$?
if [ $rc -ne 0 ] && grep -q "only valid together with --v0-human-mode" "$WORK/err2.txt"; then echo "PASS: outside human mode the flag is refused"; else echo "FAIL: outside human mode (rc=$rc)"; fail=$((fail+1)); fi
echo "failures: $fail"
exit $fail
