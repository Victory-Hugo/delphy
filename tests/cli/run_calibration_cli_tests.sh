#!/usr/bin/env bash
# CLI checks for --v0-calibration-file on the synthetic example in doc/extended/examples/calibration
# and on a simulated 20-sample mtDNA data set.  Every age below is user input for the test.
# Usage: tests/cli/run_calibration_cli_tests.sh PATH/TO/delphy WORKDIR
set -u
DELPHY=$(readlink -f "$1")
WORK=$2
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
E="$ROOT/doc/extended/examples/calibration"
D="$WORK/data"
mkdir -p "$WORK"
[ -f "$D/truth.json" ] || python3 "$HERE/simulate_mtdna_coalescent.py" "$D" --samples 20 --seed 1 \
    --mu-site-year 1.7e-8 --coal-timescale-years 75000 --dates same --mask 303-315,16184-16193 > /dev/null \
    || { echo "FAIL: could not generate the test data"; exit 1; }
[ -f "$WORK/sim.cal" ] || python3 "$HERE/make_calibration_config.py" "$D" "$WORK/sim.cal" \
    --distribution normal --width-frac 0.1 --truth-out "$WORK/sim_clades.json" > /dev/null \
    || { echo "FAIL: could not generate the test data"; exit 1; }

fail=0
# Base command: the example data set, same-date samples, external rooted topology, NO rate information
base=("$DELPHY" --v0-human-mode --v0-in-fasta "$E/samples.fasta"
      --v0-human-reference-standard custom --v0-human-expected-length 300
      --v0-human-reference-fasta "$E/reference.fasta" --v0-human-metadata "$E/metadata.tsv"
      --v0-target-coal-prior-cells 4000 --v0-steps 20000 --v0-seed 1)

run() {  # run args... -> rc, stderr in $WORK/err.txt
  "${base[@]}" "$@" > /dev/null 2> "$WORK/err.txt"
}
expect_error() {  # expect_error NAME PATTERN args...
  local name=$1 pat=$2; shift 2
  run "$@"; local rc=$?
  if [ $rc -ne 0 ] && grep -q -- "$pat" "$WORK/err.txt"; then echo "PASS: $name"
  else echo "FAIL: $name (rc=$rc)"; sed -n '1,4p' "$WORK/err.txt"; fail=$((fail+1)); fi
}
expect_ok() {
  local name=$1; shift
  run "$@"; local rc=$?
  if [ $rc -eq 0 ]; then echo "PASS: $name"
  else echo "FAIL: $name (rc=$rc)"; sed -n '1,4p' "$WORK/err.txt"; fail=$((fail+1)); fi
}
expect_ok_with() {  # expect_ok_with NAME PATTERN args...
  local name=$1 pat=$2; shift 2
  run "$@"; local rc=$?
  if [ $rc -eq 0 ] && grep -q -- "$pat" "$WORK/err.txt"; then echo "PASS: $name"
  else echo "FAIL: $name (rc=$rc)"; sed -n '1,4p' "$WORK/err.txt"; fail=$((fail+1)); fi
}

T="$E/tree.nwk"
C="$E/calibration.cal"
tree_args=(--v0-in-tree "$T" --v0-in-tree-branch-lengths years)

# --- the headline workflow: same dates, no rate at all, calibrations only
expect_ok_with "same-date data with calibrations and no rate is accepted" "log calibration prior" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$C"
expect_ok_with "the calibration targets are reported with their initial ages" "initial age" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$C"
expect_ok_with "the sampled-MRCA caveat is printed" "constrains the MRCA of the SAMPLED lineages" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$C"

# --- the gate still refuses same-date data with neither a rate nor a calibration
expect_error "same-date data without a rate and without calibrations is refused" "age calibration" \
    "${tree_args[@]}" --v0-fix-topology

# --- a calibration file needs a fixed topology
expect_error "calibration without --v0-fix-topology is refused" "requires --v0-fix-topology" \
    "${tree_args[@]}" --v0-calibration-file "$C"
expect_error "missing calibration file" "Could not read calibration file" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$WORK/nope.cal"

# --- a calibration needs a timed starting tree when there is no rate at all
expect_error "no tree, no rate, only calibrations is refused" "cannot be timed" \
    --v0-fix-topology --v0-calibration-file "$C"
# ... but a rate prior makes the tree-less case legal again (the MP starting tree is then timed by
# the rate, and the root calibration still has to resolve)
printf '%s\n' "version 1" "calibration root" "  taxa all" "  age_units ka" "  age_reference_year 1950" \
    "  distribution lognormal" "  median 140" "  sigma 1.5" "end" > "$WORK/root_only.cal"
expect_ok "no tree but an explicit rate prior is accepted" \
    --v0-fix-topology --v0-calibration-file "$WORK/root_only.cal" \
    --v0-mu-prior-mean 1.6e-7 --v0-mu-prior-stddev 4e-8

# --- malformed configurations
mkcal() { printf '%s\n' "$@" > "$WORK/tmp.cal"; }
mkcal "version 2" "calibration root" "  taxa all" "end"
expect_error "wrong version is refused" "must start with 'version 1'" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$WORK/tmp.cal"
mkcal "version 1" "calibration root" "  taxa all" "  age_units ka" "  age_reference_year 1950" \
      "  distribution normal" "  mean 140" "  stddev 0" "end"
expect_error "zero stddev is refused" "stddev must be > 0" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$WORK/tmp.cal"
mkcal "version 1" "calibration root" "  taxa all" "  age_units ka" "  age_reference_year 1950" \
      "  distribution normal" "  mean 140" "  stddev 10" "  colour blue" "end"
expect_error "unknown key is refused" "unknown key" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$WORK/tmp.cal"
mkcal "version 1" "calibration ghost" "  taxa SAMPLE_A NO_SUCH_SAMPLE" "  age_units ka" \
      "  age_reference_year 1950" "  distribution normal" "  mean 140" "  stddev 10" "end"
expect_error "unknown sample ID is refused" "not tips of the tree" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$WORK/tmp.cal"
mkcal "version 1" "calibration para" "  taxa SAMPLE_A SAMPLE_D" "  age_units ka" \
      "  age_reference_year 1950" "  distribution normal" "  mean 140" "  stddev 10" "end"
expect_error "paraphyletic sample set is refused" "not monophyletic" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$WORK/tmp.cal"
mkcal "version 1" \
      "calibration root" "  taxa all" "  age_units ka" "  age_reference_year 1950" \
      "  distribution uniform" "  oldest 20" "  youngest 10" "end" \
      "calibration M" "  taxa SAMPLE_A SAMPLE_B SAMPLE_C SAMPLE_D SAMPLE_E" "  age_units ka" \
      "  age_reference_year 1950" "  distribution uniform" "  oldest 300" "  youngest 100" "end"
expect_error "mutually unsatisfiable nested windows are refused" "cannot both be satisfied" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$WORK/tmp.cal"
mkcal "version 1" "calibration root" "  taxa all" "  age_units ka" "  age_reference_year 1950" \
      "  distribution uniform" "  oldest 1000" "  youngest 900" "end"
expect_error "starting tree outside the hard bounds is refused" "outside the hard bounds" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$WORK/tmp.cal"
mkcal "version 1" "calibration a" "  taxa all" "  age_units ka" "  age_reference_year 1950" \
      "  distribution normal" "  mean 140" "  stddev 10" "end" \
      "calibration b" "  taxa SAMPLE_A SAMPLE_B SAMPLE_C SAMPLE_D SAMPLE_E SAMPLE_F" "  age_units ka" \
      "  age_reference_year 1950" "  distribution normal" "  mean 140" "  stddev 10" "end"
expect_error "two calibrations on one node are refused" "both resolve to the same node" \
    "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$WORK/tmp.cal"

# --- the log file carries the calibration columns
run "${tree_args[@]}" --v0-fix-topology --v0-calibration-file "$C" \
    --v0-out-log-file "$WORK/cols.log" --v0-log-every 10000
if head -5 "$WORK/cols.log" | grep -q "logP(calibrations)" && \
   head -5 "$WORK/cols.log" | grep -q "calAge(root)" && \
   head -5 "$WORK/cols.log" | grep -q "calAge(M)" && \
   head -5 "$WORK/cols.log" | grep -q "calAge(N)"; then
  echo "PASS: the log file has the calibration columns"
else
  echo "FAIL: the log file has the calibration columns"; fail=$((fail+1))
fi
# ... and an uncalibrated run does not
"${base[@]}" "${tree_args[@]}" --v0-fix-topology --v0-mu-prior-mean 1.6e-7 --v0-mu-prior-stddev 4e-8 \
    --v0-out-log-file "$WORK/nocols.log" --v0-log-every 10000 > /dev/null 2>&1
if head -5 "$WORK/nocols.log" | grep -q "calAge("; then
  echo "FAIL: an uncalibrated run must not log calibration columns"; fail=$((fail+1))
else
  echo "PASS: an uncalibrated run logs no calibration columns"
fi

# --- a simulated 20-sample mtDNA data set with root/M/N calibrations from the simulation truth
sim=("$DELPHY" --v0-human-mode --v0-in-fasta "$D/samples.fasta"
     --v0-human-reference-standard rCRS --v0-human-reference-fasta "$D/reference.fasta"
     --v0-human-mask-file "$D/mask.txt" --v0-human-metadata "$D/metadata.tsv"
     --v0-human-uniform-sample-date 2020-01-01
     --v0-in-tree "$D/true_tree.nwk" --v0-in-tree-branch-lengths years --v0-fix-topology
     --v0-calibration-file "$WORK/sim.cal" --v0-target-coal-prior-cells 4000
     --v0-steps 50000 --v0-seed 1)
if "${sim[@]}" > /dev/null 2> "$WORK/sim_err.txt"; then
  echo "PASS: simulated 20-sample run with root/M/N calibrations and no rate"
else
  echo "FAIL: simulated 20-sample run (rc=$?)"; sed -n '1,6p' "$WORK/sim_err.txt"; fail=$((fail+1))
fi

echo "failures: $fail"
exit $fail
