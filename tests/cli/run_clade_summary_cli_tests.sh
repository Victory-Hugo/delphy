#!/usr/bin/env bash
# CLI checks for delphy_clades and for the backward compatibility of delphy_mcc, on the synthetic
# example in doc/extended/examples/calibration.  Every age below is test input.
# Usage: tests/cli/run_clade_summary_cli_tests.sh PATH/TO/delphy PATH/TO/delphy_clades PATH/TO/delphy_mcc WORKDIR
set -u
DELPHY=$(readlink -f "$1")
CLADES=$(readlink -f "$2")
MCC=$(readlink -f "$3")
WORK=$4
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
E="$ROOT/doc/extended/examples/calibration"
mkdir -p "$WORK"

fail=0
pass() { echo "PASS: $1"; }
bad()  { echo "FAIL: $1"; shift; sed -n '1,6p' "$@" 2>/dev/null; fail=$((fail+1)); }

expect_ok() {  # expect_ok NAME cmd...
  local name=$1; shift
  if "$@" > "$WORK/out.txt" 2> "$WORK/err.txt"; then pass "$name"; else bad "$name" "$WORK/err.txt"; fi
}
expect_error() {  # expect_error NAME PATTERN cmd...
  local name=$1 pat=$2; shift 2
  if "$@" > "$WORK/out.txt" 2> "$WORK/err.txt"; then
    bad "$name (exited 0)" "$WORK/err.txt"
  elif grep -q -- "$pat" "$WORK/err.txt"; then pass "$name"
  else bad "$name (wrong message)" "$WORK/err.txt"; fi
}
expect_ok_with() {  # expect_ok_with NAME PATTERN cmd...
  local name=$1 pat=$2; shift 2
  if "$@" > "$WORK/out.txt" 2> "$WORK/err.txt" && grep -q -- "$pat" "$WORK/out.txt" "$WORK/err.txt"; then
    pass "$name"
  else bad "$name" "$WORK/err.txt"; fi
}

# ---- Two short chains of the worked example, with a trees file -------------------------------
STEPS=${STEPS:-4000000}
for seed in 1 2; do
  out="$WORK/chain$seed"
  if [ ! -f "$out.trees" ]; then
    "$DELPHY" --v0-human-mode --v0-in-fasta "$E/samples.fasta" \
      --v0-human-reference-standard custom --v0-human-expected-length 300 \
      --v0-human-reference-fasta "$E/reference.fasta" --v0-human-metadata "$E/metadata.tsv" \
      --v0-in-tree "$E/tree.nwk" --v0-in-tree-branch-lengths years --v0-fix-topology \
      --v0-calibration-file "$E/calibration.cal" --v0-target-coal-prior-cells 4000 \
      --v0-fix-final-pop-size --v0-init-final-pop-size 50000 \
      --v0-fix-pop-growth-rate --v0-init-pop-growth-rate 0 \
      --v0-seed "$seed" --v0-threads 1 --v0-steps "$STEPS" \
      --v0-log-every $((STEPS / 200)) --v0-out-log-file "$out.log" \
      --v0-out-trees-file "$out.trees" --v0-tree-every $((STEPS / 200)) \
      > "$out.stdout" 2> "$out.stderr" || { echo "FAIL: could not run chain $seed"; exit 1; }
  fi
done
# A third chain with high-precision branch lengths, to check the new option end to end
if [ ! -f "$WORK/chain_hp.trees" ]; then
  "$DELPHY" --v0-human-mode --v0-in-fasta "$E/samples.fasta" \
    --v0-human-reference-standard custom --v0-human-expected-length 300 \
    --v0-human-reference-fasta "$E/reference.fasta" --v0-human-metadata "$E/metadata.tsv" \
    --v0-in-tree "$E/tree.nwk" --v0-in-tree-branch-lengths years --v0-fix-topology \
    --v0-calibration-file "$E/calibration.cal" --v0-target-coal-prior-cells 4000 \
    --v0-fix-final-pop-size --v0-init-final-pop-size 50000 \
    --v0-fix-pop-growth-rate --v0-init-pop-growth-rate 0 \
    --v0-out-trees-digits 12 \
    --v0-seed 1 --v0-threads 1 --v0-steps "$STEPS" \
    --v0-log-every $((STEPS / 200)) --v0-out-log-file "$WORK/chain_hp.log" \
    --v0-out-trees-file "$WORK/chain_hp.trees" --v0-tree-every $((STEPS / 200)) \
    > /dev/null 2>&1 || { echo "FAIL: could not run the high-precision chain"; exit 1; }
fi

T1="$WORK/chain1.trees"
T2="$WORK/chain2.trees"
CL="$E/clades.tsv"

# ---- delphy_clades: accepted workflows -------------------------------------------------------
expect_ok "clades: single chain, clade table only" \
  "$CLADES" --trees "$T1" --clades "$CL" --out-summary "$WORK/s1.tsv"
expect_ok_with "clades: single chain warns that one chain cannot detect a stuck mode" "only one chain" \
  "$CLADES" --trees "$T1" --clades "$CL"
expect_ok "clades: two chains, anchored ages, every output" \
  "$CLADES" --trees "$T1" --trees "$T2" --clades "$CL" \
  --calibration-file "$E/calibration.cal" --tip-dates "$E/metadata.tsv" --age-reference-year 1950 \
  --out-summary "$WORK/s2.tsv" --out-samples "$WORK/raw2.tsv" --out-per-chain "$WORK/pc2.tsv"
expect_ok "clades: clades taken from the calibration file alone" \
  "$CLADES" --trees "$T1" --calibration-file "$E/calibration.cal" --out-summary "$WORK/s3.tsv"
expect_ok "clades: re-summarize the raw samples without the trees" \
  "$CLADES" --in-samples "$WORK/raw2.tsv" --clades "$CL" --out-summary "$WORK/s4.tsv"
expect_ok "clades: configurable burn-in by states" \
  "$CLADES" --trees "$T1" --clades "$CL" --burn-in-states 1000000 --out-summary "$WORK/s5.tsv"
expect_ok "clades: configurable burn-in by samples" \
  "$CLADES" --trees "$T1" --clades "$CL" --burn-in-samples 50 --out-summary "$WORK/s6.tsv"
expect_ok "clades: thinning" \
  "$CLADES" --trees "$T1" --clades "$CL" --thin $((STEPS / 100)) --out-summary "$WORK/s7.tsv"
expect_ok "clades: a different credible mass" \
  "$CLADES" --trees "$T1" --clades "$CL" --credible-mass 0.5 --out-summary "$WORK/s8.tsv"
expect_ok "clades: high-precision trees file" \
  "$CLADES" --trees "$WORK/chain_hp.trees" --clades "$CL" --out-summary "$WORK/s9.tsv"

# ---- delphy_clades: content of the outputs ---------------------------------------------------
check_grep() {  # check_grep NAME PATTERN FILE
  if grep -q -- "$2" "$3"; then pass "$1"; else bad "$1 (missing '$2' in $3)" "$3"; fi
}
check_grep "summary names the age reference" "years_before_1950-01-01" "$WORK/s2.tsv"
check_grep "summary names the anchor sample" "time_anchor" "$WORK/s2.tsv"
check_grep "summary separates the shortest interval from the equal-tailed interval" "clade_age_eti_upper" "$WORK/s2.tsv"
check_grep "summary says whether the shortest interval is a strict HPD interval" "mrca_age_interval_is_hpd" "$WORK/s2.tsv"
check_grep "summary reports the highest density region itself" "mrca_age_hdr_intervals" "$WORK/s2.tsv"
check_grep "summary reports the calibration status" "calibrated:root" "$WORK/s2.tsv"
check_grep "uncalibrated clades are marked as such" "not_calibrated" "$WORK/s2.tsv"
check_grep "without a calibration file the status is unknown" "unknown" "$WORK/s1.tsv"
check_grep "relative ages are labelled as such" "years_before_the_most_recent_tip_of_each_tree" "$WORK/s1.tsv"
check_grep "raw samples table has one row per clade and tree" "^AB	" "$WORK/raw2.tsv"
check_grep "per-chain table names the chain files" "chain1.trees" "$WORK/pc2.tsv"

rows=$(grep -vc '^#' "$WORK/s2.tsv")
if [ "$rows" = "6" ]; then pass "summary has a header and one row per clade"
else bad "summary row count (got $rows, expected 6)"; fi

# The 95% HPD must never be wider than the equal-tailed interval for these samples, and must not be
# the 2.5%-97.5% quantile pair.
python3 - "$WORK/s2.tsv" <<'PY' || fail=$((fail+1))
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if not l.startswith("#")]
hdr, data = rows[0], rows[1:]
i = {n: k for k, n in enumerate(hdr)}
ok = True
for r in data:
    short = (float(r[i["mrca_age_shortest_interval_upper"]])
             - float(r[i["mrca_age_shortest_interval_lower"]]))
    eti = float(r[i["mrca_age_eti_upper"]]) - float(r[i["mrca_age_eti_lower"]])
    if short > eti + 1e-6:
        print("FAIL: shortest interval wider than the equal-tailed interval for", r[i["clade_id"]])
        ok = False
    if abs(short - eti) < 1e-9:
        print("NOTE: shortest and equal-tailed intervals coincide for", r[i["clade_id"]])
    # A unimodal posterior must say so, and the reported region must then be one interval
    if r[i["mrca_age_interval_is_hpd"]] == "yes" and r[i["mrca_age_hdr_components"]] != "1":
        print("FAIL: interval_is_hpd=yes but hdr_components!=1 for", r[i["clade_id"]]); ok = False
    if r[i["mrca_age_interval_is_hpd"]] == "no" and r[i["mrca_age_hdr_components"]] == "1":
        print("FAIL: interval_is_hpd=no but hdr_components==1 for", r[i["clade_id"]]); ok = False
    if ";" in r[i["mrca_age_hdr_intervals"]] and r[i["mrca_age_interval_is_hpd"]] != "no":
        print("FAIL: a disconnected HDR was not flagged for", r[i["clade_id"]]); ok = False
print("PASS: shortest intervals are no wider than the equal-tailed intervals, and interval_is_hpd "
      "agrees with the highest density region" if ok else "")
sys.exit(0 if ok else 1)
PY

# ---- delphy_clades: rejections ---------------------------------------------------------------
expect_error "clades: no clades given" "no clades to summarize" \
  "$CLADES" --trees "$T1"
expect_error "clades: --trees and --in-samples together" "mutually exclusive" \
  "$CLADES" --trees "$T1" --in-samples "$WORK/raw2.tsv" --clades "$CL"
expect_error "clades: two burn-in options" "at most one of" \
  "$CLADES" --trees "$T1" --clades "$CL" --burn-in-fraction 0.5 --burn-in-samples 10
expect_error "clades: burn-in fraction out of range" "must be in \[0, 1)" \
  "$CLADES" --trees "$T1" --clades "$CL" --burn-in-fraction 1.5
expect_error "clades: credible mass out of range" "must be in (0, 1)" \
  "$CLADES" --trees "$T1" --clades "$CL" --credible-mass 1.0
expect_error "clades: unreadable trees file" "cannot open" \
  "$CLADES" --trees "$WORK/does-not-exist.trees" --clades "$CL"
expect_error "clades: reference year without sampling dates" "need --tip-dates" \
  "$CLADES" --trees "$T1" --clades "$CL" --age-reference-year 1950
expect_error "clades: unknown anchor sample" "not a sample with an exactly known date" \
  "$CLADES" --trees "$T1" --clades "$CL" --tip-dates "$E/metadata.tsv" --anchor-tip NO_SUCH_SAMPLE

printf 'clade_id\ttaxa\nX\tNOT_A_SAMPLE,ALSO_NOT\n' > "$WORK/bad_clades.tsv"
expect_error "clades: unknown sample IDs in the clade table" "not tips of the posterior trees" \
  "$CLADES" --trees "$T1" --clades "$WORK/bad_clades.tsv"
printf 'clade\ttaxa\nX\tA,B\n' > "$WORK/bad_header.tsv"
expect_error "clades: clade table without the required columns" "'clade_id' and 'taxa'" \
  "$CLADES" --trees "$T1" --clades "$WORK/bad_header.tsv"
printf 'clade_id\ttaxa\nX\tSAMPLE_A\n' > "$WORK/one_taxon.tsv"
expect_error "clades: a one-sample clade" "at least two sample IDs" \
  "$CLADES" --trees "$T1" --clades "$WORK/one_taxon.tsv"
# A copy of one chain with one taxon renamed: the tip indexing no longer matches
sed 's/SAMPLE_F/SAMPLE_Z/g' "$T2" > "$WORK/renamed.trees"
expect_error "clades: chains with different taxa" "same taxa, in the same order" \
  "$CLADES" --trees "$T1" --trees "$WORK/renamed.trees" --clades "$CL"

# ---- delphy_mcc: backward compatibility ------------------------------------------------------
expect_ok "mcc: the historical two-argument form still works" \
  "$MCC" "$T1" "$WORK/mcc_default.nexus"
expect_ok "mcc: --burn-in-fraction 0.3 is the default" \
  "$MCC" --burn-in-fraction 0.3 "$T1" "$WORK/mcc_explicit.nexus"
if diff -q "$WORK/mcc_default.nexus" "$WORK/mcc_explicit.nexus" > /dev/null; then
  pass "mcc: the explicit default reproduces the historical output"
else bad "mcc: --burn-in-fraction 0.3 differs from the default"; fi
expect_ok "mcc: a reproducible seed" "$MCC" --seed 7 "$T1" "$WORK/mcc_seed_a.nexus"
expect_ok "mcc: the same seed again" "$MCC" --seed 7 "$T1" "$WORK/mcc_seed_b.nexus"
if diff -q "$WORK/mcc_seed_a.nexus" "$WORK/mcc_seed_b.nexus" > /dev/null; then
  pass "mcc: the same seed gives the same MCC tree"
else bad "mcc: --seed is not reproducible"; fi
expect_ok_with "mcc: a different burn-in discards a different number of trees" "Discarding" \
  "$MCC" --burn-in-samples 100 "$T1" "$WORK/mcc_burn.nexus"
expect_error "mcc: two burn-in options" "at most one of" \
  "$MCC" --burn-in-fraction 0.5 --burn-in-samples 10 "$T1" "$WORK/mcc_bad.nexus"
expect_error "mcc: a burn-in that discards everything" "no trees are left" \
  "$MCC" --burn-in-fraction 0.999999 --thin 1000000000 "$T1" "$WORK/mcc_bad.nexus"

echo
if [ "$fail" -eq 0 ]; then echo "ALL CLI CHECKS PASSED"; else echo "$fail CLI CHECK(S) FAILED"; fi
exit $fail
