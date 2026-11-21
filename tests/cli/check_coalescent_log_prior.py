#!/usr/bin/env python3
"""Independent check of the coalescent log prior Delphy reports.

Reads a BEAST-style `.trees` file written by Delphy and the `.log` file from the same run, and for
every sampled state recomputes the coalescent log prior two ways:

  exact   the Kingman density of the genealogy,
            log P = - sum_intervals C(k,2) int dt / N(t)  -  sum_coalescences log N(t_i),
          evaluated interval by interval between successive events;
  cells   the same quantity the way Delphy's samplers evaluate it, on a grid of fixed-width cells in
          which k(t) is replaced by its time average and 1/N(t) by the reciprocal of the time average
          of N(t).

Both are compared with the `coalescent` column of the log.  Only a constant coalescent timescale is
supported, which is the configuration used for the validation campaigns; a run with a free or
Skygrid population model writes the population parameters to the log and would need them read back.

A `.trees` file carries branch lengths only, so the tips are assumed to share a sampling date, which
is what `--v0-human-uniform-sample-date` produces.  Ages are then distances below that date.

Units: Delphy writes `.trees` branch lengths in years (`core/beasty_output.cpp`, "years!") even
though its internal time is days, and writes the log's `coalescent` column as the internal log prior
plus `(n-1) log 365`, which is the same density evaluated with the coalescent timescale in years.
Both sides of this comparison are therefore in years throughout, and no conversion is applied.

Usage:
  check_coalescent_log_prior.py RUN.trees RUN.log --pop-size-years 75000 [--max-trees 20]
"""
import argparse
import math
import re
import sys


def read_log(path):
    header, rows = None, []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            while fields and fields[-1] == "":
                fields.pop()
            if header is None:
                header = fields
                continue
            if len(fields) == len(header):
                rows.append(fields)
    return {n: i for i, n in enumerate(header or [])}, rows


# Delphy annotates a branch between the colon and the length, as `name:[&mutations={...}]30000`,
# so the annotations are removed before tokenizing rather than skipped during the parse.
ANNOTATION = re.compile(r"\[[^\]]*\]")
TOKEN = re.compile(r"[(),;]|:[-+0-9.eE]+|[^(),;:]+")


def node_ages(newick):
    """Ages, in the units of the branch lengths, of the internal nodes and of the tips, measured
    below the youngest tip.  Returns (coalescence_ages, tip_ages), both as positive depths."""
    tokens = TOKEN.findall(ANNOTATION.sub("", newick))
    pos = 0

    def parse():
        nonlocal pos
        if tokens[pos] == "(":
            pos += 1
            child_heights = []
            child_internals = []
            while True:
                h, internals = parse()
                child_heights.append(h)
                child_internals.extend(internals)
                if tokens[pos] == ",":
                    pos += 1
                    continue
                assert tokens[pos] == ")", tokens[pos]
                pos += 1
                break
            while pos < len(tokens) and tokens[pos] not in (",", ")", ";") and not tokens[pos].startswith(":"):
                pos += 1  # internal node label
            height = max(child_heights)
            branch = 0.0
            if pos < len(tokens) and tokens[pos].startswith(":"):
                branch = float(tokens[pos][1:])
                pos += 1
            return height + branch, child_internals + [height]
        # a tip
        pos += 1  # its name
        branch = 0.0
        if pos < len(tokens) and tokens[pos].startswith(":"):
            branch = float(tokens[pos][1:])
            pos += 1
        return branch, []

    root_height, internals = parse()
    # Heights above the tips; convert to depths below the youngest tip, which is at height 0
    return [h for h in internals], root_height


def exact_log_prior(coal_heights, num_tips, pop):
    """Constant coalescent timescale `pop`; every tip at height 0."""
    events = sorted([(h, +1) for h in coal_heights] + [(0.0, -1)] * num_tips,
                    key=lambda p: (-p[0], -p[1]))
    # Walk forwards in time, i.e. from the largest height down to 0
    out = 0.0
    k = 1
    prev = None
    for h, kind in events:
        if prev is not None and k >= 2:
            out -= (k * (k - 1)) // 2 * (prev - h) / pop
        prev = h
        k += kind
        if kind > 0:
            out -= math.log(pop)
    return out


def cell_log_prior(coal_heights, num_tips, pop, t_step):
    """Delphy's grid: cell c covers heights [c t_step, (c+1) t_step) below the tips."""
    out = -len(coal_heights) * math.log(pop)
    if not coal_heights:
        return out
    num_cells = int(math.floor(max(coal_heights) / t_step)) + 1
    events = sorted([(h, +1) for h in coal_heights], key=lambda p: -p[0])
    # k(h) = 1 + number of coalescences at heights above h, counting tips at height 0
    # Walk cells from the oldest to the youngest, accumulating the time average of k
    idx = 0
    k = 1
    for c in range(num_cells - 1, -1, -1):
        lo, hi = c * t_step, (c + 1) * t_step
        m1, prev = 0.0, hi
        while idx < len(events) and events[idx][0] > lo:
            h = min(hi, events[idx][0])
            m1 += k * (prev - h)
            prev = h
            k += events[idx][1]
            idx += 1
        m1 += k * (prev - lo)
        kbar = m1 / t_step
        out -= 0.5 * t_step * (kbar * kbar - kbar) / pop
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trees")
    ap.add_argument("log")
    ap.add_argument("--pop-size-years", type=float, required=True)
    ap.add_argument("--max-trees", type=int, default=20)
    ap.add_argument("--grid-cells", type=int, default=4000,
                    help="Cells the run used, i.e. --v0-target-coal-prior-cells, so that the cell "
                         "reconstruction uses the same resolution Delphy did.")
    ap.add_argument("--t-step-years", type=float, default=0.0,
                    help="Pinned cell width (--v0-coal-prior-fixed-t-step); overrides --grid-cells.")
    args = ap.parse_args()

    pop = args.pop_size_years  # see the note on units at the top of this file
    header, rows = read_log(args.log)
    col = None
    for name in ("coalescent", "coalescent.all", "Coalescent"):
        if name in header:
            col = header[name]
            break
    if col is None:
        sys.exit("no coalescent column in %s" % args.log)
    state_col = header.get("state", header.get("Sample", 0))
    by_state = {int(float(r[state_col])): float(r[col]) for r in rows}

    print("%10s %7s %16s %16s %16s %12s %12s"
          % ("state", "tips", "delphy", "cells", "exact", "delphy-cells", "cells-exact"))
    shown = 0
    with open(args.trees) as f:
        for line in f:
            m = re.match(r"\s*tree STATE_(\d+)\s*=\s*(.*;)", line)
            if not m:
                continue
            state = int(m.group(1))
            if state not in by_state:
                continue
            internals, root_height = node_ages(m.group(2))
            num_tips = len(internals) + 1
            t_step = args.t_step_years if args.t_step_years > 0 else root_height / args.grid_cells
            ex = exact_log_prior(internals, num_tips, pop)
            ce = cell_log_prior(internals, num_tips, pop, t_step)
            print("%10d %7d %16.4f %16.4f %16.4f %12.4f %12.4f"
                  % (state, num_tips, by_state[state], ce, ex,
                     by_state[state] - ce, ce - ex))
            shown += 1
            if shown >= args.max_trees:
                break


if __name__ == "__main__":
    main()
