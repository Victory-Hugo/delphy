#!/usr/bin/env python3
"""Independent check of the coalescent log prior a Skygrid run reports (test tooling).

`check_coalescent_log_prior.py` does this for a constant coalescent timescale and says so; this script
does it for a Skygrid curve, which is what the exact Kingman correction makes
worth checking: with the correction on, the coalescent term of the target is the exact Kingman density
of the sampled genealogy, and the sampled curve is in the log, so the whole claim is checkable from the
files the shipped binary writes.

For every sampled state present in both files it recomputes, from scratch,

  exact   log P = - sum_j C(k_j, 2) int_{a_j}^{b_j} dt / N(t)  -  sum_i log N(t_i),
          with the intervals delimited by the genealogy's own events;
  cells   the same on a grid of fixed-width cells, with k(t) replaced by its time average over each
          cell and 1 / N(t) by the reciprocal of the time average of N(t), which is what
          `Scalable_coalescent_prior` evaluates.

Neither reads `core/exact_coalescent.cpp`: the events are merged here, the lineage counts are tracked
here and the Skygrid rate integral is written out here from the model definition in `core/pop_model.h`.

The log's Skygrid column carries the coalescent term plus the GMRF prior
(`core/beasty_output.cpp`), so the run must be made with the low-population barrier off and with
`--v0-skygrid-inv-nbar-prior-alpha 0 --v0-skygrid-inv-nbar-prior-beta 0`; the GMRF prior is then
`sum_{k=1..M} [log(tau / 2 pi) / 2 - tau delta_k^2 / 2]`, which this script subtracts.  Units are years
throughout, as in the sister script.

Knot layout: the log writes `skygrid.logPopSize{j}` = gamma_{M-j+1} in years, so knot j - 1 of the age
axis (age 0 is the most recent knot) carries `logPopSize{j}`, and the knot ages are equally spaced over
`skygrid.cutOff`.

Usage:
  check_exact_skygrid_log_prior.py RUN.trees RUN.log [--max-trees 20] [--t-step-years W]
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


ANNOTATION = re.compile(r"\[[^\]]*\]")
TOKEN = re.compile(r"[(),;]|:[-+0-9.eE]+|[^(),;:]+")


def node_ages(newick):
    """Heights of the internal nodes above the youngest tip, and the root height."""
    tokens = TOKEN.findall(ANNOTATION.sub("", newick))
    pos = 0

    def parse():
        nonlocal pos
        if tokens[pos] == "(":
            pos += 1
            heights, internals = [], []
            while True:
                h, inner = parse()
                heights.append(h)
                internals.extend(inner)
                if tokens[pos] == ",":
                    pos += 1
                    continue
                assert tokens[pos] == ")", tokens[pos]
                pos += 1
                break
            while pos < len(tokens) and tokens[pos] not in (",", ")", ";") \
                    and not tokens[pos].startswith(":"):
                pos += 1
            height = max(heights)
            branch = 0.0
            if pos < len(tokens) and tokens[pos].startswith(":"):
                branch = float(tokens[pos][1:])
                pos += 1
            return height + branch, internals + [height]
        pos += 1
        branch = 0.0
        if pos < len(tokens) and tokens[pos].startswith(":"):
            branch = float(tokens[pos][1:])
            pos += 1
        return branch, []

    root_height, internals = parse()
    return internals, root_height


class Curve:
    """The Skygrid curve in the age coordinate: knot j at age j K / M carries value v[j]."""

    def __init__(self, values, cutoff, log_linear):
        self.v = list(values)
        self.m = len(values) - 1
        self.a = [j * cutoff / self.m for j in range(self.m + 1)]
        self.log_linear = bool(log_linear)

    def log_n(self, age):
        if age >= self.a[-1]:
            return self.v[-1]
        j = 0
        while j + 1 <= self.m and self.a[j + 1] <= age:
            j += 1
        if not self.log_linear:
            return self.v[j]
        c = (age - self.a[j]) / (self.a[j + 1] - self.a[j])
        return (1 - c) * self.v[j] + c * self.v[j + 1]

    def n(self, age):
        return math.exp(self.log_n(age))

    def _piece(self, lo, hi):
        """int_lo^hi dt / N(t) with [lo, hi] inside one knot interval, or past the last knot."""
        if hi <= lo:
            return 0.0
        if lo >= self.a[-1]:
            return (hi - lo) * math.exp(-self.v[-1])
        j = 0
        while j + 1 <= self.m and self.a[j + 1] <= lo:
            j += 1
        if not self.log_linear:
            return (hi - lo) * math.exp(-self.v[j])
        width = self.a[j + 1] - self.a[j]
        g_lo = (1 - (lo - self.a[j]) / width) * self.v[j] + ((lo - self.a[j]) / width) * self.v[j + 1]
        g_hi = (1 - (hi - self.a[j]) / width) * self.v[j] + ((hi - self.a[j]) / width) * self.v[j + 1]
        d = g_hi - g_lo
        if d == 0.0:
            return (hi - lo) * math.exp(-g_lo)
        return (hi - lo) * math.exp(-g_lo) * (-math.expm1(-d)) / d

    def intensity(self, lo, hi):
        """int_lo^hi dt / N(t), cut at every knot inside the range."""
        if hi <= lo:
            return 0.0
        cuts = [lo] + [a for a in self.a if lo < a < hi] + [hi]
        return sum(self._piece(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1))

    def pop_average(self, lo, hi):
        """(1 / (hi - lo)) int_lo^hi dt N(t), as `Scalable_coalescent_prior` uses."""
        if hi <= lo:
            return math.exp(self.v[-1])
        cuts = [lo] + [a for a in self.a if lo < a < hi] + [hi]
        total = 0.0
        for i in range(len(cuts) - 1):
            p, q = cuts[i], cuts[i + 1]
            if p >= self.a[-1]:
                total += (q - p) * math.exp(self.v[-1])
                continue
            j = 0
            while j + 1 <= self.m and self.a[j + 1] <= p:
                j += 1
            if not self.log_linear:
                total += (q - p) * math.exp(self.v[j])
            else:
                width = self.a[j + 1] - self.a[j]
                g_p = (1 - (p - self.a[j]) / width) * self.v[j] + ((p - self.a[j]) / width) * self.v[j + 1]
                g_q = (1 - (q - self.a[j]) / width) * self.v[j] + ((q - self.a[j]) / width) * self.v[j + 1]
                d = g_q - g_p
                total += (q - p) * math.exp(g_p) * (math.expm1(d) / d if d != 0.0 else 1.0)
        return total / (hi - lo)


def exact_log_prior(coal_heights, num_tips, curve):
    """Ages are heights above the youngest tip; every tip is at height 0."""
    events = sorted([(h, +1) for h in coal_heights] + [(0.0, -1)] * num_tips,
                    key=lambda p: (-p[0], -p[1]))
    out = 0.0
    k = 1
    prev = None
    for h, kind in events:
        if prev is not None and k >= 2:
            out -= (k * (k - 1)) // 2 * curve.intensity(h, prev)
        prev = h
        k += kind
        if kind > 0:
            out -= curve.log_n(h)
    return out


def cell_log_prior(coal_heights, num_tips, curve, t_step):
    out = -sum(curve.log_n(h) for h in coal_heights)
    if not coal_heights:
        return out
    num_cells = int(math.floor(max(coal_heights) / t_step)) + 1
    events = sorted([(h, +1) for h in coal_heights], key=lambda p: -p[0])
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
        out -= 0.5 * t_step * (kbar * kbar - kbar) / curve.pop_average(lo, hi)
    return out


def gmrf_log_prior(values, tau):
    """sum_{k=1..M} [log(tau / 2 pi) / 2 - tau delta_k^2 / 2]; the differences are unit free."""
    out = 0.0
    for j in range(len(values) - 1):
        d = values[j + 1] - values[j]
        out += 0.5 * math.log(tau / (2.0 * math.pi)) - 0.5 * tau * d * d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trees")
    ap.add_argument("log")
    ap.add_argument("--max-trees", type=int, default=20)
    ap.add_argument("--t-step-years", type=float, default=0.0,
                    help="Pinned cell width (--v0-coal-prior-fixed-t-step), for the cell column.")
    args = ap.parse_args()

    header, rows = read_log(args.log)
    if "skygrid" not in header:
        sys.exit("no skygrid column in %s; is this a Skygrid run?" % args.log)
    knots = sorted((h for h in header if h.startswith("skygrid.logPopSize")),
                   key=lambda h: int(h[len("skygrid.logPopSize"):]))
    if not knots:
        sys.exit("no skygrid.logPopSize columns in %s" % args.log)
    state_col = header.get("state", header.get("Sample", 0))
    by_state = {}
    for r in rows:
        st = int(float(r[state_col]))
        by_state[st] = dict(
            reported=float(r[header["skygrid"]]),
            values=[float(r[header[k]]) for k in knots],   # knot j of the age axis is logPopSize{j+1}
            tau=float(r[header["skygrid.precision"]]),
            cutoff=float(r[header["skygrid.cutOff"]]),
            log_linear=float(r[header["skygrid.isloglinear"]]) > 0.5)

    print("%10s %7s %16s %16s %16s %13s %13s"
          % ("state", "tips", "reported", "exact", "cells", "rep-exact", "rep-cells"))
    shown = 0
    worst_exact = 0.0
    worst_cells = 0.0
    with open(args.trees) as f:
        for line in f:
            m = re.match(r"\s*tree STATE_(\d+)\s*=\s*(.*;)", line)
            if not m:
                continue
            state = int(m.group(1))
            if state not in by_state:
                continue
            e = by_state[state]
            internals, root_height = node_ages(m.group(2))
            num_tips = len(internals) + 1
            curve = Curve(e["values"], e["cutoff"], e["log_linear"])
            n_inner = len(internals)
            # The column is (coalescent term in years) + (GMRF prior)
            reported = e["reported"] - gmrf_log_prior(e["values"], e["tau"])
            ex = exact_log_prior(internals, num_tips, curve)
            t_step = args.t_step_years if args.t_step_years > 0 else root_height / 4000.0
            ce = cell_log_prior(internals, num_tips, curve, t_step)
            # Scaled by the magnitude of the quantity: the first logged state can start from a
            # curve wildly out of scale with the tree, where the density is 1e5 nats and a relative
            # 1e-7 is 0.1 nats of pure cancellation.
            scale = max(1.0, abs(reported))
            worst_exact = max(worst_exact, abs(reported - ex) / scale)
            worst_cells = max(worst_cells, abs(reported - ce) / scale)
            print("%10d %7d %16.4f %16.4f %16.4f %13.5f %13.5f"
                  % (state, num_tips, reported, ex, ce, reported - ex, reported - ce))
            shown += 1
            if shown >= args.max_trees:
                break
    if shown == 0:
        sys.exit("no state appears in both files")
    print("# worst |reported - exact| = %.6g relative; worst |reported - cells| = %.6g relative; %d states"
          % (worst_exact, worst_cells, shown))


if __name__ == "__main__":
    main()
