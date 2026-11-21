#!/usr/bin/env bash
# End-to-end checks of the human input path through the delphy CLI, with 5, 100 and 1000 synthetic
# 16,569-site samples that all share one sampling date.  The rate prior values are test-only placeholders.
#
# Usage: tests/cli/run_cli_scale_tests.sh PATH/TO/delphy WORKDIR
set -u
DELPHY=$(readlink -f "$1")
WORK=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK"
fail=0

check() {  # check DESCRIPTION CONDITION-EXIT-STATUS
  if [ "$2" -eq 0 ]; then echo "PASS: $1"; else echo "FAIL: $1"; fail=$((fail + 1)); fi
}

for n in 5 100 1000; do
  d="$WORK/n$n"
  python3 -I "$HERE/make_synthetic_mtdna.py" "$d" --samples "$n" --sites 16569 --seed "$n" --dates uniform
  start=$(date +%s.%N)
  "$DELPHY" --v0-human-mode --v0-in-fasta "$d/samples.fasta" \
      --v0-human-reference-standard rCRS --v0-human-reference-fasta "$d/reference.fasta" \
      --v0-human-uniform-sample-date 2015-06-01 --v0-human-audit-report "$d/audit.json" \
      --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8 \
      --v0-seed 1 --v0-threads 1 --v0-steps 100 --v0-log-every 100 --v0-tree-every 100 \
      >"$d/stdout.txt" 2>"$d/stderr.txt"
  status=$?
  end=$(date +%s.%N)
  check "n=$n: CLI exits 0" $status
  grep -q "Built initial tree with $n tips" "$d/stderr.txt"; check "n=$n: all $n samples reach the initial tree" $?
  grep -q "\"num_samples\": $n," "$d/audit.json"; check "n=$n: audit lists $n samples" $?
  grep -q '"no_time_signal": true' "$d/audit.json"; check "n=$n: no-time-signal marker recorded" $?
  grep -q "NO TIME SIGNAL" "$d/stderr.txt"; check "n=$n: no-time-signal warning printed" $?
  printf 'n=%s wall=%.1fs\n' "$n" "$(echo "$end - $start" | bc)"
done

# Uniform dates without an explicit rate prior must be refused
d="$WORK/n5"
"$DELPHY" --v0-human-mode --v0-in-fasta "$d/samples.fasta" --v0-human-reference-standard rCRS \
    --v0-human-reference-fasta "$d/reference.fasta" --v0-human-uniform-sample-date 2015-06-01 \
    --v0-seed 1 --v0-threads 1 --v0-steps 10 >/dev/null 2>"$d/refused.txt"
[ $? -ne 0 ] && grep -q "do not determine the absolute time" "$d/refused.txt"; check "uniform dates without rate prior are refused" $?

# Human flags without --v0-human-mode are refused
"$DELPHY" --v0-in-fasta "$d/samples.fasta" --v0-human-metadata "$d/metadata.tsv" --v0-steps 10 >/dev/null 2>"$d/nomode.txt"
[ $? -ne 0 ] && grep -q "only valid together with --v0-human-mode" "$d/nomode.txt"; check "human flags need --v0-human-mode" $?

# Same data through the legacy path: undated IDs are dropped as before (documented legacy behaviour)
"$DELPHY" --v0-in-fasta "$d/samples.fasta" --v0-steps 10 >/dev/null 2>"$d/legacy.txt"
[ $? -ne 0 ] && grep -q "No valid date\|could not be determined" "$d/legacy.txt"; check "legacy path still rejects undated sequence IDs" $?

# Truncated alignment declared as rCRS is refused
head -c 5000 "$d/samples.fasta" | sed '$d' >"$d/truncated.fasta"
"$DELPHY" --v0-human-mode --v0-in-fasta "$d/samples.fasta" --v0-human-reference-standard custom \
    --v0-human-expected-length 16569 --v0-human-reference-fasta "$d/reference.fasta" \
    --v0-human-uniform-sample-date 2015 --v0-human-max-missing-fraction 0 >/dev/null 2>"$d/badfrac.txt"
[ $? -ne 0 ] && grep -q "max missing fraction" "$d/badfrac.txt"; check "invalid max-missing-fraction refused" $?

echo "failures: $fail"
exit $fail
