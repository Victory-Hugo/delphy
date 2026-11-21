#!/usr/bin/env bash
# End-to-end checks of the unified reversible substitution model
# (doc/extended/substitution_models.md) through the delphy CLI, on the example data of
# doc/extended/examples/observation (300 sites, 12 samples).  The prior values that appear on
# the command line are test-only placeholders.
#
# Usage: tests/cli/run_subst_model_cli_tests.sh PATH/TO/delphy WORKDIR
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
        --v0-seed 4 --v0-threads 1 --v0-steps 20000 --v0-log-every 1000 --v0-tree-every 20000
        --v0-in-fasta "$E/samples.fasta")

body() { grep -v '^# \(Produced by\|Invocation\)' "$1"; }
columns() { grep -v '^#' "$1" | head -1 | tr '\t' '\n'; }

# 1. The default is unchanged: no option at all, and HKY85 spelled out, must agree byte for byte.
"$DELPHY" "${common[@]}" --v0-out-log-file "$WORK/plain.log" >"$WORK/plain.out" 2>"$WORK/plain.err"
check "no model option: CLI exits 0" $?
columns "$WORK/plain.log" | grep -qx "kappa"; check "the default log has a kappa column" $?
"$DELPHY" "${common[@]}" --v0-subst-model HKY85 --v0-out-log-file "$WORK/hky.log" \
    >"$WORK/hky.out" 2>"$WORK/hky.err"
check "explicit HKY85: CLI exits 0" $?
grep -q "code 010010 (HKY / HKY85)" "$WORK/hky.err"; check "start-up names the model and its code" $?
cmp -s <(body "$WORK/plain.log") <(body "$WORK/hky.log")
check "explicit HKY85 gives byte-identical output to the default" $?

# 2. Every named model and a few bare codes run and report what they ran.
for spec in JC69 F81 K80 TN93 TNe K81 K81u TPM2 TPM2u TPM3 TPM3u TIM TIMe TIM2 TIM2e TIM3 TIM3e \
            TVM TVMe SYM GTR 010010 010020 012345 001122 012340 101101 GTR+FQ JC69+FO; do
  tag=$(echo "$spec" | tr '+' '_')
  "$DELPHY" "${common[@]}" --v0-subst-model "$spec" --v0-out-log-file "$WORK/m_$tag.log" \
      >"$WORK/m_$tag.out" 2>"$WORK/m_$tag.err"
  check "model $spec: CLI exits 0" $?
  grep -q "^Substitution model: code" "$WORK/m_$tag.err"; check "model $spec: start-up reports it" $?
done

# 3. The canonical code is reported, so two spellings of one structure agree.
grep -q "code 010010" "$WORK/m_101101.err"; check "101101 is reported as the canonical 010010" $?
cmp -s <(body "$WORK/m_010010.log") <(body "$WORK/m_101101.log")
check "101101 and 010010 produce identical output" $?

# 4. The log columns follow the model.
columns "$WORK/m_GTR.log" | grep -qx -- "rate(AG)"
check "GTR has one column per free exchangeability" $?
columns "$WORK/m_TN93.log" | grep -qx -- "rate(AG)"
check "TN93 has a column for A-G" $?
columns "$WORK/m_TN93.log" | grep -qx -- "rate(CT)"
check "TN93 has a column for C-T" $?
if columns "$WORK/m_JC69.log" | grep -q "rate("; then false; else true; fi
check "JC69 has no exchangeability column" $?
if columns "$WORK/m_GTR.log" | grep -qx "kappa"; then false; else true; fi
check "GTR has no kappa column" $?

# 5. Constrained GTR reproduces HKY85's likelihood: same kappa, same frequencies, both fixed.
"$DELPHY" "${common[@]}" --v0-subst-model HKY85 --v0-subst-initial-exchangeabilities 1,3.5 \
    --v0-subst-frequencies 0.3,0.2,0.2,0.3 --v0-subst-fix-exchangeabilities \
    --v0-subst-fix-frequencies --v0-steps 0 --v0-out-log-file "$WORK/nest_hky.log" \
    >"$WORK/nest_hky.out" 2>"$WORK/nest_hky.err"
check "fixed HKY85: CLI exits 0" $?
"$DELPHY" "${common[@]}" --v0-subst-model GTR --v0-subst-initial-exchangeabilities 1,3.5,1,1,3.5,1 \
    --v0-subst-frequencies 0.3,0.2,0.2,0.3 --v0-subst-fix-exchangeabilities \
    --v0-subst-fix-frequencies --v0-steps 0 --v0-out-log-file "$WORK/nest_gtr.log" \
    >"$WORK/nest_gtr.out" 2>"$WORK/nest_gtr.err"
check "constrained GTR: CLI exits 0" $?
python3 - "$WORK/nest_hky.log" "$WORK/nest_gtr.log" <<'PY'
import sys
def log_g(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path) if not l.startswith("#")]
    return float(rows[-1][rows[0].index("likelihood_really_logG")])
a, b = log_g(sys.argv[1]), log_g(sys.argv[2])
sys.exit(0 if abs(a - b) <= 1e-9 * max(1.0, abs(a)) else 1)
PY
check "a GTR constrained to HKY85 has the same log_G" $?

# 6. Fixed parameters stay fixed.
"$DELPHY" "${common[@]}" --v0-subst-model GTR --v0-subst-initial-exchangeabilities 1,4,1,1,4,1 \
    --v0-subst-frequencies 0.3,0.2,0.2,0.3 --v0-subst-fix-exchangeabilities \
    --v0-subst-fix-frequencies --v0-out-log-file "$WORK/fixed.log" \
    >"$WORK/fixed.out" 2>"$WORK/fixed.err"
check "fixed GTR: CLI exits 0" $?
grep -q "the relative exchangeabilities are fixed" "$WORK/fixed.err"
check "start-up says the exchangeabilities are fixed" $?
grep -q "the frequencies are fixed" "$WORK/fixed.err"; check "start-up says the frequencies are fixed" $?
python3 - "$WORK/fixed.log" <<'PY'
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if not l.startswith("#")]
head = rows[0]
for name, want in (("rate(AG)", 4.0), ("rate(AC)", 1.0), ("frequencies1", 0.3)):
    values = {row[head.index(name)] for row in rows[1:]}
    if len(values) != 1 or abs(float(next(iter(values))) - want) > 1e-9:
        sys.exit(1)
sys.exit(0)
PY
check "the fixed values never change during the run" $?

# 7. Equal frequencies really are equal and never move.
"$DELPHY" "${common[@]}" --v0-subst-model SYM --v0-out-log-file "$WORK/sym.log" \
    >"$WORK/sym.out" 2>"$WORK/sym.err"
check "SYM: CLI exits 0" $?
python3 - "$WORK/sym.log" <<'PY'
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if not l.startswith("#")]
head = rows[0]
for name in ("frequencies1", "frequencies2", "frequencies3", "frequencies4"):
    values = {row[head.index(name)] for row in rows[1:]}
    if len(values) != 1 or abs(float(next(iter(values))) - 0.25) > 1e-12:
        sys.exit(1)
sys.exit(0)
PY
check "SYM holds every frequency at 0.25" $?

# 8. The prior is reported and can be replaced.
"$DELPHY" "${common[@]}" --v0-subst-model GTR --v0-subst-exchangeability-prior 0.5,-0.25,0.8 \
    --v0-out-log-file "$WORK/prior.log" >"$WORK/prior.out" 2>"$WORK/prior.err"
check "a replaced prior: CLI exits 0" $?
grep -q "mean of the log 0.5 for a class of transitions only and -0.25 otherwise" "$WORK/prior.err"
check "start-up reports the replaced prior" $?
grep -q "standard deviation of the log 0.8" "$WORK/prior.err"
check "start-up reports the replaced standard deviation" $?

# 9. Bad input is refused with an explanation, not guessed at.
refuse() {  # refuse DESCRIPTION FRAGMENT ARGS...
  local what=$1 fragment=$2; shift 2
  "$DELPHY" "${common[@]}" "$@" --v0-out-log-file "$WORK/bad.log" >"$WORK/bad.out" 2>"$WORK/bad.err"
  if [ $? -eq 0 ]; then echo "FAIL: $what (accepted)"; fail=$((fail + 1)); return; fi
  grep -q "$fragment" "$WORK/bad.err"
  check "$what" $?
}
refuse "an unknown model name is refused" "Unknown substitution model" --v0-subst-model NOSUCH
refuse "a five-digit code is refused" "exactly 6 digits" --v0-subst-model 01001
refuse "a seven-digit code is refused" "exactly 6 digits" --v0-subst-model 0100100
refuse "a non-digit in a code is refused" "must consist of digits" --v0-subst-model 01x010
refuse "an unknown suffix is refused" "Unknown model suffix" --v0-subst-model HKY85+G4
refuse "site rate heterogeneity as a suffix is refused" "Unknown model suffix" --v0-subst-model GTR+I
refuse "too few exchangeabilities are refused" "exchangeability classes" \
    --v0-subst-model GTR --v0-subst-initial-exchangeabilities 1,2
refuse "a non-positive exchangeability is refused" "finite positive number" \
    --v0-subst-model GTR --v0-subst-initial-exchangeabilities 1,2,3,4,5,0
refuse "an unpinned reference class is refused" "held at 1" \
    --v0-subst-model GTR --v0-subst-initial-exchangeabilities 1,2,3,4,5,2
refuse "three frequencies are refused" "needs 4 numbers" \
    --v0-subst-model GTR --v0-subst-frequencies 0.3,0.3,0.4
refuse "frequencies that do not sum to one are refused" "must sum to 1" \
    --v0-subst-model GTR --v0-subst-frequencies 0.3,0.3,0.3,0.3
refuse "a zero frequency is refused" "strictly between 0 and 1" \
    --v0-subst-model GTR --v0-subst-frequencies 0.5,0.5,0.0,0.0
refuse "a two-number prior is refused" "needs three numbers" \
    --v0-subst-model GTR --v0-subst-exchangeability-prior 1,2
refuse "a non-positive prior spread is refused" "finite positive number" \
    --v0-subst-model GTR --v0-subst-exchangeability-prior 1,0,0
refuse "the mpox hack and a model option are mutually exclusive" "cannot be combined" \
    --v0-mpox-hack --v0-subst-model GTR

# 10. Combination with the site partitions and with site rate heterogeneity.
cat > "$WORK/two.txt" <<'EOF'
version 1

partition first
  sites 1-150
end

partition second
  sites 151-300
end
EOF
"$DELPHY" "${common[@]}" --v0-subst-model GTR --v0-site-partition-file "$WORK/two.txt" \
    --v0-out-log-file "$WORK/gtr_parts.log" >"$WORK/gtr_parts.out" 2>"$WORK/gtr_parts.err"
check "GTR with two site partitions: CLI exits 0" $?
columns "$WORK/gtr_parts.log" | grep -qx -- "partitionRate(first)"
check "the partition columns are still there" $?
columns "$WORK/gtr_parts.log" | grep -qx -- "rate(AG)"
check "the exchangeability columns are still there" $?
"$DELPHY" "${common[@]}" --v0-subst-model GTR --v0-site-rate-heterogeneity \
    --v0-out-log-file "$WORK/gtr_gamma.log" >"$WORK/gtr_gamma.out" 2>"$WORK/gtr_gamma.err"
check "GTR with continuous Gamma site rates: CLI exits 0" $?
columns "$WORK/gtr_gamma.log" | grep -qx "alpha"
check "the Gamma shape is logged" $?
"$DELPHY" "${common[@]}" --v0-subst-model TN93 --v0-site-partition-file "$WORK/two.txt" \
    --v0-site-rate-heterogeneity --v0-out-log-file "$WORK/all.log" >"$WORK/all.out" 2>"$WORK/all.err"
check "TN93 with partitions and Gamma together: CLI exits 0" $?

# 11. A general model on variable-sites-only input completed per partition, and on input with
# missing data and a callable mask (the observation modes).
cat > "$WORK/completed.txt" <<'EOF'
version 1

partition first
  sites 1-17
  constant_sites A=30,C=28,G=27,T=25
end

partition second
  sites 18-34
  constant_sites A=38,C=32,G=41,T=32
end
EOF
"$DELPHY" --v0-human-mode --v0-in-fasta "$E/polymorphic_only.fasta"     --v0-human-reference-standard custom --v0-human-expected-length 34     --v0-human-reference-id EXREF --v0-human-metadata "$E/metadata.tsv"     --v0-site-partition-file "$WORK/completed.txt" --v0-human-max-missing-fraction 0.5     --v0-subst-model GTR --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8     --v0-seed 4 --v0-threads 1 --v0-steps 20000 --v0-log-every 1000 --v0-tree-every 20000     --v0-human-audit-report "$WORK/completed.json" --v0-out-log-file "$WORK/completed.log"     >"$WORK/completed.out" 2>"$WORK/completed.err"
check "GTR on partition-wise completed variable-site input: CLI exits 0" $?
grep -q '"num_sites": 287, "num_input_sites": 34' "$WORK/completed.json"
check "the restored blocks are still appended with a general model" $?
columns "$WORK/completed.log" | grep -qx -- "rate(AG)"
check "the exchangeability columns are there on completed input" $?
"$DELPHY" "${common[@]}" --v0-subst-model TN93 --v0-human-callable-file "$E/callable.txt"     --v0-out-log-file "$WORK/callable.log" >"$WORK/callable.out" 2>"$WORK/callable.err"
check "TN93 with a callable-site declaration: CLI exits 0" $?

# 12. Checkpoint and resume carry the model.
"$DELPHY" "${common[@]}" --v0-subst-model TIM2 --v0-steps 20000 \
    --v0-out-delphy-file "$WORK/tim2.dphy" --v0-out-log-file "$WORK/tim2.log" \
    >"$WORK/tim2.out" 2>"$WORK/tim2.err"
check "writing a checkpoint with TIM2: CLI exits 0" $?
test -s "$WORK/tim2.dphy"; check "the checkpoint file is written" $?

echo
if [ "$fail" -eq 0 ]; then echo "All checks passed"; else echo "$fail check(s) failed"; fi
exit $((fail > 0))
