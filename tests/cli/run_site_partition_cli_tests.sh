#!/usr/bin/env bash
# End-to-end checks of the site-partition model (doc/extended/site_partitions.md) through the
# delphy CLI, on the example data of doc/extended/examples/observation (300 sites, 12 samples).
# The rate prior values are test-only placeholders.
#
# Usage: tests/cli/run_site_partition_cli_tests.sh PATH/TO/delphy WORKDIR
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
        --v0-seed 4 --v0-threads 1 --v0-steps 20000 --v0-log-every 1000 --v0-tree-every 20000)

cat > "$WORK/two.txt" <<'EOF'
version 1

partition first
  sites 1-150
end

partition second
  sites 151-300
end
EOF
cat > "$WORK/scattered.txt" <<'EOF'
version 1

# two interleaved blocks per partition
partition odd_blocks
  sites 1-75,151-225
end

partition even_blocks
  sites 76-150,226-300
end
EOF
cat > "$WORK/one.txt" <<'EOF'
version 1

partition all
  sites 1-300
end
EOF

# 1. Two contiguous partitions
"$DELPHY" "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-site-partition-file "$WORK/two.txt" \
    --v0-out-log-file "$WORK/two.log" >"$WORK/two.out" 2>"$WORK/two.err"
check "two partitions: CLI exits 0" $?
grep -q "Site partitions (2)" "$WORK/two.err"; check "start-up reports the partitions" $?
grep -q "partitionRate(first)" "$WORK/two.log"; check "the log has a column per partition" $?
python3 - "$WORK/two.log" <<'PY'
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if not l.startswith("#")]
head = rows[0]
i, j = head.index("partitionRate(first)"), head.index("partitionRate(second)")
last = rows[-1]
r1, r2 = float(last[i]), float(last[j])
# 150 observed sites in "first", 130 in "second" (sites 281-300 are never observed)
mean = (150 * r1 + 130 * r2) / 280
sys.exit(0 if abs(mean - 1.0) < 1e-4 else 1)
PY
check "the weighted mean of the sampled rates is 1" $?

# 2. Non-contiguous partitions
"$DELPHY" "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-site-partition-file "$WORK/scattered.txt" \
    --v0-out-log-file "$WORK/scattered.log" >"$WORK/scattered.out" 2>"$WORK/scattered.err"
check "scattered partitions: CLI exits 0" $?
grep -q "partitionRate(odd_blocks)" "$WORK/scattered.log"; check "scattered partitions are named in the log" $?

# 3. One explicit partition must reproduce a run without a configuration, byte for byte
"$DELPHY" "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-out-log-file "$WORK/plain.log" \
    >"$WORK/plain.out" 2>"$WORK/plain.err"
check "no partition file: CLI exits 0" $?
"$DELPHY" "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-site-partition-file "$WORK/one.txt" \
    --v0-out-log-file "$WORK/one.log" >"$WORK/one.out" 2>"$WORK/one.err"
check "one explicit partition: CLI exits 0" $?
for f in plain one; do grep -v '^# \(Produced by\|Invocation\)' "$WORK/$f.log" > "$WORK/$f.body"; done
cmp -s "$WORK/plain.body" "$WORK/one.body"
check "one explicit partition gives byte-identical output to no configuration" $?

# 4. Fixed rates stay fixed
"$DELPHY" "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-site-partition-file "$WORK/two.txt" \
    --v0-site-partition-initial-rates 0.5,2.0 --v0-site-partition-fix-rates \
    --v0-out-log-file "$WORK/fixed.log" >"$WORK/fixed.out" 2>"$WORK/fixed.err"
check "fixed rates: CLI exits 0" $?
grep -q "the relative rates are fixed at these values" "$WORK/fixed.err"; check "start-up says the rates are fixed" $?
python3 - "$WORK/fixed.log" <<'PY'
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if not l.startswith("#")]
head = rows[0]
i = head.index("partitionRate(first)")
values = {row[i] for row in rows[1:]}
# 0.5 and 2.0 renormalized by the weighted mean (150 * 0.5 + 130 * 2.0) / 280
expected = 0.5 / ((150 * 0.5 + 130 * 2.0) / 280)
sys.exit(0 if len(values) == 1 and abs(float(values.pop()) - expected) < 1e-4 else 1)
PY
check "the fixed rate is the renormalized value and never changes" $?

# 5. Variable-sites-only input completed per partition
cat > "$WORK/completed.txt" <<EOF
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
"$DELPHY" --v0-human-mode --v0-in-fasta "$E/polymorphic_only.fasta" --v0-human-reference-standard custom \
    --v0-human-expected-length 34 --v0-human-reference-id EXREF --v0-human-metadata "$E/metadata.tsv" \
    --v0-site-partition-file "$WORK/completed.txt" --v0-human-max-missing-fraction 0.5 \
    --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8 --v0-seed 4 --v0-threads 1 --v0-steps 20000 \
    --v0-log-every 1000 --v0-tree-every 20000 --v0-human-audit-report "$WORK/completed.json" \
    --v0-out-log-file "$WORK/completed.log" >"$WORK/completed.out" 2>"$WORK/completed.err"
check "per-partition constant sites: CLI exits 0" $?
grep -q '"num_sites": 287, "num_input_sites": 34' "$WORK/completed.json"
check "the restored blocks are appended to the analysed sequence" $?
grep -q '"constant_sites_appended_per_partition": \[{"A": 30' "$WORK/completed.json"
check "the audit records the restored sites per partition" $?

# 6. Rejections
reject() {  # reject DESCRIPTION EXPECTED-MESSAGE ARGS...
  local what=$1 msg=$2; shift 2
  "$DELPHY" "$@" >/dev/null 2>"$WORK/reject.err"
  local status=$?
  [ $status -ne 0 ] && grep -q "$msg" "$WORK/reject.err"
  check "$what" $?
}
printf 'version 1\n\npartition a\n  sites 1-200\nend\n\npartition b\n  sites 100-300\nend\n' > "$WORK/overlap.txt"
printf 'version 1\n\npartition a\n  sites 1-200\nend\n' > "$WORK/gap.txt"
printf 'version 1\n\npartition a\n  sites 1-400\nend\n' > "$WORK/range.txt"
printf 'version 1\n\npartition a\n  parts 4\n  sites 1-300\nend\n' > "$WORK/parts.txt"
reject "overlapping partitions are refused" "must not overlap" \
    "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-site-partition-file "$WORK/overlap.txt"
reject "unassigned sites are refused" "belong to no partition" \
    "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-site-partition-file "$WORK/gap.txt"
reject "sites beyond the reference are refused" "the input reference has 300 sites" \
    "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-site-partition-file "$WORK/range.txt"
reject "the Run/Subrun knob is not a partition key" "unknown key 'parts'" \
    "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-site-partition-file "$WORK/parts.txt"
reject "partition options without a configuration are refused" "only valid together with" \
    "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-site-partition-fix-rates
reject "a wrong number of initial rates is refused" "rates for 2 partitions" \
    "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-site-partition-file "$WORK/two.txt" \
    --v0-site-partition-initial-rates 1.0,2.0,3.0
reject "the mpox hack and site partitions are refused together" "cannot be combined with" \
    "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-site-partition-file "$WORK/two.txt" --v0-mpox-hack
reject "a global constant-site count with partitions is refused" "declared inside each partition" \
    --v0-human-mode --v0-in-fasta "$E/polymorphic_only.fasta" --v0-human-reference-standard custom \
    --v0-human-expected-length 34 --v0-human-reference-id EXREF --v0-human-metadata "$E/metadata.tsv" \
    --v0-site-partition-file "$WORK/two.txt" --v0-human-snp-only-constant-sites "A=1,C=1,G=1,T=1" \
    --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8 --v0-seed 4 --v0-steps 10
# A partition whose sites are all masked has no observed site and cannot have a rate
printf '151-300\n' > "$WORK/mask_second.txt"
reject "a partition without an observed site is refused" "not one of them is observed in any sample" \
    --v0-human-mode --v0-human-reference-standard custom --v0-human-expected-length 300 \
    --v0-human-reference-fasta "$E/reference.fasta" --v0-human-metadata "$E/metadata.tsv" \
    --v0-human-max-missing-fraction 1.0 --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8 \
    --v0-seed 4 --v0-threads 1 --v0-steps 10 \
    --v0-in-fasta "$E/samples.fasta" --v0-site-partition-file "$WORK/two.txt" \
    --v0-human-mask-file "$WORK/mask_second.txt"

echo
if [ "$fail" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "$fail CHECK(S) FAILED"; fi
exit $((fail > 0))
