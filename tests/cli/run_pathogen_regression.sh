#!/usr/bin/env bash
# Pathogen regression: with identical inputs and seeds, two Delphy binaries must write byte-identical .log and
# .trees files (default options; 1 and 4 threads; default and old-usher-like initialization), and the
# same for a pathogen-scale Skygrid population model in three configurations.
# Usage: run_pathogen_regression.sh BASE_BINARY NEW_BINARY INPUT.fasta WORKDIR
set -u
BASE=$(readlink -f "$1"); NEW=$(readlink -f "$2"); FASTA=$(readlink -f "$3"); WORK=$4
mkdir -p "$WORK"
fail=0
for threads in 1 4; do
  for init in "" "old-usher-like"; do
    for which in base new; do
      bin=$BASE; [ "$which" = new ] && bin=$NEW
      extra=(); [ -n "$init" ] && extra=(--v0-init "$init")
      out="$WORK/$which.t$threads.${init:-default}"
      nice -n 10 "$bin" --v0-in-fasta "$FASTA" --v0-seed 3 --v0-threads $threads --v0-steps 300000 \
          --v0-log-every 3000 --v0-tree-every 30000 "${extra[@]}" \
          --v0-out-log-file "$out.log" --v0-out-trees-file "$out.trees" > /dev/null 2> "$out.err"
      # Drop the header lines that name the binary, the invocation and the output paths
      grep -v '^# \(Produced by\|Invocation\)' "$out.log" > "$out.log.body"
      grep -v '^# \(Produced by\|Invocation\)' "$out.trees" > "$out.trees.body"
    done
    for ext in log trees; do
      if cmp -s "$WORK/base.t$threads.${init:-default}.$ext.body" "$WORK/new.t$threads.${init:-default}.$ext.body"; then
        echo "IDENTICAL threads=$threads init=${init:-default} .$ext"
      else
        echo "DIFFERENT threads=$threads init=${init:-default} .$ext"; fail=1
      fi
    done
  done
done
# Skygrid at pathogen scale: a two-year knot range, with the staircase and log-linear curves, the
# inferred and the fixed smoothness, and the low-population barrier on and off.
i=0
for cfg in \
    "--v0-pop-model skygrid --v0-skygrid-cutoff 2 --v0-skygrid-num-parameters 20" \
    "--v0-pop-model skygrid --v0-skygrid-cutoff 2 --v0-skygrid-num-parameters 20 --v0-skygrid-type log-linear --v0-skygrid-infer-prior-smoothness true" \
    "--v0-pop-model skygrid --v0-skygrid-cutoff 2 --v0-skygrid-num-parameters 20 --v0-skygrid-disable-low-pop-barrier true"
do
  i=$((i + 1))
  for which in base new; do
    bin=$BASE; [ "$which" = new ] && bin=$NEW
    out="$WORK/$which.skygrid$i"
    nice -n 10 "$bin" --v0-in-fasta "$FASTA" --v0-seed 5 --v0-threads 2 --v0-steps 200000 \
        --v0-log-every 2000 --v0-tree-every 20000 $cfg \
        --v0-out-log-file "$out.log" --v0-out-trees-file "$out.trees" > /dev/null 2> "$out.err"
    grep -v '^# \(Produced by\|Invocation\)' "$out.log" > "$out.log.body"
    grep -v '^# \(Produced by\|Invocation\)' "$out.trees" > "$out.trees.body"
  done
  for ext in log trees; do
    if cmp -s "$WORK/base.skygrid$i.$ext.body" "$WORK/new.skygrid$i.$ext.body"; then
      echo "IDENTICAL skygrid=$i .$ext"
    else
      echo "DIFFERENT skygrid=$i .$ext"; fail=1
    fi
  done
done

exit $fail
