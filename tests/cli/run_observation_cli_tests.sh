#!/usr/bin/env bash
# End-to-end checks of the observation modes of doc/extended/observation.md through the
# delphy CLI, on the example data of doc/extended/examples/observation (300 sites, 12 samples).
# The rate prior values are test-only placeholders.
#
# Usage: tests/cli/run_observation_cli_tests.sh PATH/TO/delphy WORKDIR
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
        --v0-seed 1 --v0-threads 1 --v0-steps 200 --v0-log-every 200 --v0-tree-every 200)

# 1. Full-length alignment with the callable sites declared for every sample
"$DELPHY" "${common[@]}" --v0-in-fasta "$E/samples.fasta" \
    --v0-human-sample-callable-file "$E/sample_callable.tsv" \
    --v0-human-audit-report "$WORK/fasta.json" >"$WORK/fasta.out" 2>"$WORK/fasta.err"
check "full alignment with per-sample callable sites: CLI exits 0" $?
grep -q '"observation_mode": "full_alignment"' "$WORK/fasta.json"; check "audit names the observation mode" $?

# 2. The same observation as a variant list: the site classes must come out identical
"$DELPHY" "${common[@]}" --v0-human-in-variants "$E/variants.tsv" \
    --v0-human-sample-callable-file "$E/sample_callable.tsv" \
    --v0-human-audit-report "$WORK/variants.json" >"$WORK/variants.out" 2>"$WORK/variants.err"
check "variant list with per-sample callable sites: CLI exits 0" $?
grep -q '"observation_mode": "variants_plus_callable"' "$WORK/variants.json"; check "audit names the variant mode" $?
for key in '"sites"' '"num_samples"'; do
  a=$(grep -o "$key: .*" "$WORK/fasta.json")
  b=$(grep -o "$key: .*" "$WORK/variants.json")
  [ "$a" = "$b" ]; check "variant input gives the same $key as the alignment" $?
done

# 3. Variable-sites-only input, completed from the declared counts of the removed constant sites
counts=$(cat "$E/snps_only.counts")
"$DELPHY" --v0-human-mode --v0-in-fasta "$E/snps_only.fasta" --v0-human-reference-standard custom \
    --v0-human-expected-length 126 --v0-human-reference-id EXREF --v0-human-metadata "$E/metadata.tsv" \
    --v0-human-snp-only-constant-sites "$counts" --v0-human-max-missing-fraction 0.5 \
    --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8 --v0-seed 1 --v0-threads 1 --v0-steps 200 \
    --v0-log-every 200 --v0-tree-every 200 --v0-human-audit-report "$WORK/snps.json" \
    >"$WORK/snps.out" 2>"$WORK/snps.err"
check "variable-sites-only input with declared constant sites: CLI exits 0" $?
grep -q '"num_sites": 300, "num_input_sites": 126' "$WORK/snps.json"; check "completion restores 300 sites" $?
grep -q '"observation_mode": "full_alignment+completed_constant_sites"' "$WORK/snps.json"
check "audit names the completion" $?
grep -q 'SNP-only completion' "$WORK/snps.err"; check "completion is reported as a warning" $?
a=$(grep -o '"invariant_observed": [0-9]*' "$WORK/fasta.json")
b=$(grep -o '"invariant_observed": [0-9]*' "$WORK/snps.json")
[ "$a" = "$b" ]; check "completed input has the same number of invariant sites as the alignment" $?

# 4. Input filtered down to the polymorphic sites: refused without the declaration, accepted with it.
#    The looser filter of snps_only.fasta keeps sites that have missing data, some of which are invariant
#    and equal to the reference, so the heuristic does not fire on it; that limit is documented in
#    doc/extended/observation.md, section 7.
"$DELPHY" --v0-human-mode --v0-in-fasta "$E/polymorphic_only.fasta" --v0-human-reference-standard custom \
    --v0-human-expected-length 34 --v0-human-reference-id EXREF --v0-human-metadata "$E/metadata.tsv" \
    --v0-human-max-missing-fraction 0.5 --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8 \
    --v0-seed 1 --v0-threads 1 --v0-steps 10 >/dev/null 2>"$WORK/poly_refused.err"
status=$?
[ $status -ne 0 ] && grep -q "No site is observed as invariant" "$WORK/poly_refused.err"
check "polymorphic-sites-only input without a declaration is refused" $?

"$DELPHY" --v0-human-mode --v0-in-fasta "$E/polymorphic_only.fasta" --v0-human-reference-standard custom \
    --v0-human-expected-length 34 --v0-human-reference-id EXREF --v0-human-metadata "$E/metadata.tsv" \
    --v0-human-snp-only-constant-sites "$(cat "$E/polymorphic_only.counts")" \
    --v0-human-max-missing-fraction 0.5 --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8 \
    --v0-seed 1 --v0-threads 1 --v0-steps 200 --v0-log-every 200 --v0-tree-every 200 \
    --v0-human-audit-report "$WORK/poly.json" >"$WORK/poly.out" 2>"$WORK/poly.err"
check "polymorphic-sites-only input with declared constant sites: CLI exits 0" $?
grep -q '"num_sites": 280, "num_input_sites": 34' "$WORK/poly.json"
check "completion restores the 246 removed constant sites" $?

"$DELPHY" --v0-human-mode --v0-in-fasta "$E/snps_only.fasta" --v0-human-reference-standard custom \
    --v0-human-expected-length 126 --v0-human-reference-id EXREF --v0-human-metadata "$E/metadata.tsv" \
    --v0-human-max-missing-fraction 0.5 --v0-mu-prior-mean 1e-7 --v0-mu-prior-stddev 5e-8 \
    --v0-seed 1 --v0-threads 1 --v0-steps 200 --v0-log-every 200 --v0-tree-every 200 \
    --v0-human-audit-report "$WORK/undetected.json" >"$WORK/undetected.out" 2>"$WORK/undetected.err"
check "the heuristic does not detect the looser filter (documented limit): CLI exits 0" $?

# 5. A variant list without a callable declaration is refused
"$DELPHY" "${common[@]}" --v0-human-in-variants "$E/variants.tsv" >/dev/null 2>"$WORK/nocallable.err"
status=$?
[ $status -ne 0 ] && grep -q "requires a callable declaration" "$WORK/nocallable.err"
check "variant input without a callable declaration is refused" $?

# 6. A call outside the declared callable sites is refused; the mask excludes sites instead
awk 'NR==1 {print; next} {print "EX00\t290\tA"; exit}' "$E/variants.tsv" > "$WORK/bad_variants.tsv"
"$DELPHY" "${common[@]}" --v0-human-in-variants "$WORK/bad_variants.tsv" \
    --v0-human-callable-file "$E/callable.txt" >/dev/null 2>"$WORK/contradiction.err"
status=$?
[ $status -ne 0 ] && grep -q "callable declaration does not list" "$WORK/contradiction.err"
check "a call outside the callable declaration is refused" $?

# 7. An unsupported call (IUPAC code) is refused with the reason
printf 'sample_id\tsite\tbase\nEX00\t10\tR\n' > "$WORK/iupac.tsv"
"$DELPHY" "${common[@]}" --v0-human-in-variants "$WORK/iupac.tsv" \
    --v0-human-callable-file "$E/callable.txt" >/dev/null 2>"$WORK/iupac.err"
status=$?
[ $status -ne 0 ] && grep -q "IUPAC ambiguity code" "$WORK/iupac.err"
check "an IUPAC call in a variant list is refused" $?

# 8. The mask stays available next to the callable declaration
"$DELPHY" "${common[@]}" --v0-in-fasta "$E/samples.fasta" --v0-human-callable-file "$E/callable.txt" \
    --v0-human-mask-file "$E/mask.txt" --v0-human-audit-report "$WORK/masked.json" \
    >"$WORK/masked.out" 2>"$WORK/masked.err"
check "mask and callable declaration together: CLI exits 0" $?
grep -q '"masked": 10, "not_callable": 20' "$WORK/masked.json"; check "audit counts masked and non-callable sites" $?

echo
if [ "$fail" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "$fail CHECK(S) FAILED"; fi
exit $((fail > 0))
