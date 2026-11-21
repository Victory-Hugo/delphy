#!/usr/bin/env python3
"""Build a Delphy calibration configuration from a simulated data set (test tooling only).

Reads DATA/true_tree.nwk and DATA/tips.tsv (written by simulate_mtdna_coalescent.py), picks three
nested internal nodes -- the root, the larger child subtree of the root, and the larger child subtree
of that node -- and writes an age calibration for each.  The names "root", "M" and "N" are labels for
these simulated clades only; nothing here is a statement about real human mtDNA haplogroups, and the
ages come from the simulation truth, never from a built-in table.

WARNING -- coverage validation.  By default every prior is centred on the TRUE age of its clade.  The
inference model then depends on the truth, and the data were not generated from the model being
fitted, so the coverage of any node age in such a run is not a simulation-based calibration result.
It is fit for exercising the calibration machinery, and for comparing two analyses that share the
design, and for nothing else.  For a valid coverage experiment use --measurement-seed, below.

Usage:
  make_calibration_config.py DATA OUT.cal [--which root,M,N] [--distribution normal|lognormal|uniform]
      [--width-frac 0.1] [--scale 1.0] [--units years_bp|ka|cal_bp] [--reference-year 2020]
      [--conflict] [--truth-out truth_clades.json]
      [--measurement-seed S --reference-pop-years N]

--width-frac   normal: stddev / age; lognormal: sigma (dimensionless); uniform: half width / age
--scale        multiplies the centre of every prior, to test a deliberately wrong calibration
--conflict     writes mutually unsatisfiable uniform windows (the root younger than its descendant)
--measurement-seed S
               treats each calibration as a noisy MEASUREMENT of its clade's age, y = a + sigma eps,
               eps ~ N(0,1) drawn from seed S, and writes the normal prior N(a; y, sigma).  That prior
               is exactly the measurement likelihood L(y | a), so the analysis is a correct Bayesian
               update and simulation-based calibration applies.  sigma depends only on the
               coalescent timescale and the number of tips below the clade,
                   sigma = width_frac * 2 N (1 - 1/k),
               which is the expected age of the MRCA of k tips under a constant-size coalescent, and
               never on the realized age.  Requires --reference-pop-years N (the timescale in years,
               an input of the experiment and not read from the simulation truth file).
--reference-pop-years N
               the coalescent timescale in years used for sigma above
Prints one JSON object with the true ages of the calibrated clades.
"""
import argparse
import json
import os
import sys


def days_from_civil(y, m, d):
    """Days since 1970-01-01 in the proleptic Gregorian calendar, astronomical year numbering."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def parse_newick(text):
    """Returns (parent, length, name) with tips and inner nodes in discovery order, root first."""
    text = text.strip()
    if text.endswith(";"):
        text = text[:-1]
    parent, length, name = [], [], []
    stack = []
    i = 0
    cur = None

    def new_node(p):
        parent.append(p)
        length.append(0.0)
        name.append("")
        return len(parent) - 1

    while i < len(text):
        c = text[i]
        if c == "(":
            cur = new_node(stack[-1] if stack else -1)
            stack.append(cur)
            i += 1
        elif c == ",":
            i += 1
        elif c == ")":
            cur = stack.pop()
            i += 1
        elif c == ":":
            j = i + 1
            while j < len(text) and (text[j].isdigit() or text[j] in ".eE+-"):
                j += 1
            length[cur] = float(text[i + 1:j])
            i = j
        else:
            j = i
            while j < len(text) and text[j] not in "(),:;":
                j += 1
            cur = new_node(stack[-1] if stack else -1)
            name[cur] = text[i:j]
            i = j
    return parent, length, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--which", default="root,M,N")
    ap.add_argument("--distribution", default="normal", choices=["normal", "lognormal", "uniform"])
    ap.add_argument("--width-frac", type=float, default=0.1)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--units", default="years_bp", choices=["years_bp", "ka", "cal_bp"])
    ap.add_argument("--reference-year", type=int, default=2020)
    ap.add_argument("--conflict", action="store_true")
    ap.add_argument("--truth-out", default="")
    ap.add_argument("--measurement-seed", type=int, default=None)
    ap.add_argument("--reference-pop-years", type=float, default=None)
    args = ap.parse_args()

    if args.measurement_seed is not None:
        if args.reference_pop_years is None or not args.reference_pop_years > 0:
            sys.exit("--measurement-seed needs --reference-pop-years > 0")
        if args.distribution != "normal" or args.conflict or args.scale != 1.0:
            sys.exit("--measurement-seed writes normal priors and cannot be combined with "
                     "--distribution other than normal, --conflict or --scale")
        if args.units != "years_bp":
            sys.exit("--measurement-seed is implemented for --units years_bp only")
    else:
        print("WARNING: calibration priors are centred on the true clade ages; coverage obtained with "
              "them is not a valid calibration check (use --measurement-seed)", file=sys.stderr)

    with open(os.path.join(args.data, "true_tree.nwk")) as f:
        parent, length, name = parse_newick(f.read())
    tip_age = {}
    with open(os.path.join(args.data, "tips.tsv")) as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            tip_age[p[0]] = float(p[1])

    n = len(parent)
    kids = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p >= 0:
            kids[p].append(i)
    root = next(i for i, p in enumerate(parent) if p < 0)

    depth = [0.0] * n
    for i in range(n):
        if parent[i] >= 0:
            depth[i] = depth[parent[i]] + length[i]
    # node age (years before 2020) = tip age + (tip depth - node depth) for any tip below it
    below = [None] * n
    for i in range(n - 1, -1, -1):
        below[i] = set([name[i]]) if not kids[i] else set().union(*[below[c] for c in kids[i]])
    age = [0.0] * n
    for i in range(n):
        some_tip = sorted(below[i])[0]
        tip_node = next(j for j in range(n) if name[j] == some_tip)
        age[i] = tip_age[some_tip] + (depth[tip_node] - depth[i])

    def bigger_child(v):
        if not kids[v]:
            return None
        return max(kids[v], key=lambda c: (len(below[c]), -c))

    targets = {"root": root}
    m = bigger_child(root)
    if m is not None and len(below[m]) >= 2:
        targets["M"] = m
        nn = bigger_child(m)
        if nn is not None and len(below[nn]) >= 2:
            targets["N"] = nn

    wanted = [w for w in args.which.split(",") if w]
    missing = [w for w in wanted if w not in targets]
    if missing:
        sys.exit("cannot place calibration(s) %s: the simulated tree has no suitable clade" % ",".join(missing))

    scale_to_units = {"years_bp": 1.0, "cal_bp": 1.0, "ka": 1e-3}[args.units]
    # The true ages above are years before 2020-01-01.  A calibration measures ages back from
    # 1 January of its own reference year (1950 for cal_bp), so convert through exact day counts;
    # 70 calendar years are 25,567 days (17 leap days), i.e. 70.0466 years of 365 days.
    ref_year = 1950 if args.units == "cal_bp" else args.reference_year
    shift = (days_from_civil(2020, 1, 1) - days_from_civil(ref_year, 1, 1)) / 365.0

    rng = None
    if args.measurement_seed is not None:
        import random
        rng = random.Random(args.measurement_seed)

    lines = ["# Generated by make_calibration_config.py from simulated truth -- test input only.",
             "# The names root/M/N label clades of this simulated tree, not real haplogroups.",
             "version 1"]
    truth = {}
    measurement = {}
    for w in wanted:
        node = targets[w]
        true_age_2020 = age[node]
        # Age on this calibration's own reference instant
        centre = (true_age_2020 - shift) * args.scale  # age before the calibration's own reference
        taxa = sorted(below[node])
        lines.append("")
        lines.append("calibration %s" % w)
        if node == root:
            lines.append("  taxa all")
        else:
            for i in range(0, len(taxa), 8):
                lines.append("  taxa " + " ".join(taxa[i:i + 8]))
        lines.append("  age_units %s" % args.units)
        if args.units in ("years_bp", "ka"):
            lines.append("  age_reference_year %d" % args.reference_year)
        if args.conflict:
            # The root window is entirely younger than the descendant window: unsatisfiable by design
            lo, hi = (0.05, 0.10) if w == "root" else (0.50, 2.00)
            lines.append("  distribution uniform")
            lines.append("  oldest %.10g" % (centre * hi * scale_to_units))
            lines.append("  youngest %.10g" % (centre * lo * scale_to_units))
        elif rng is not None:
            k = len(taxa)
            sigma = args.width_frac * 2.0 * args.reference_pop_years * (1.0 - 1.0 / k)
            y = centre + sigma * rng.gauss(0.0, 1.0)
            lines.append("  distribution normal")
            lines.append("  mean %.10g" % (y * scale_to_units))
            lines.append("  stddev %.10g" % (sigma * scale_to_units))
            measurement[w] = {"measured_age": y, "sigma": sigma}
        elif args.distribution == "normal":
            lines.append("  distribution normal")
            lines.append("  mean %.10g" % (centre * scale_to_units))
            lines.append("  stddev %.10g" % (centre * args.width_frac * scale_to_units))
        elif args.distribution == "lognormal":
            lines.append("  distribution lognormal")
            lines.append("  median %.10g" % (centre * scale_to_units))
            lines.append("  sigma %.10g" % args.width_frac)
        else:
            lines.append("  distribution uniform")
            lines.append("  oldest %.10g" % (centre * (1.0 + args.width_frac) * scale_to_units))
            lines.append("  youngest %.10g" % (centre * (1.0 - args.width_frac) * scale_to_units))
        lines.append("end")
        truth[w] = {"true_age_years_before_2020": true_age_2020,
                    "prior_centre_on_reference": centre,
                    "num_taxa": len(taxa),
                    "clade": taxa if len(taxa) <= 50 else taxa[:50]}
        if w in measurement:
            truth[w].update(measurement[w])

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    if args.truth_out:
        with open(args.truth_out, "w") as f:
            json.dump(truth, f, indent=1, sort_keys=True)
            f.write("\n")
    print(json.dumps({k: {"true_age_years_before_2020": v["true_age_years_before_2020"],
                          "num_taxa": v["num_taxa"]} for k, v in truth.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
