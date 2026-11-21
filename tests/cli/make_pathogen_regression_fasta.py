#!/usr/bin/env python3
"""The synthetic pathogen alignment used by the byte-level regression (test tooling only).

The byte-level regression of `tests/cli/run_pathogen_regression.sh` compares two Delphy
binaries on the default pathogen path, so the alignment only has to be a fixed, pathogen-scale input
that every binary reads the same way.  It is generated here from a fixed seed with the standard
library alone, so the file is reproducible on any machine without numpy.

Sixty sequences of 3,000 sites, sampled over one year, on a random coalescent genealogy of about two
years' depth at 1e-3 substitutions per site per year: about 0.002 substitutions per site from the
root to a tip, which is the low-divergence regime Delphy was built for and in which its all-sites
mutational-history proposal is accepted about half the time.

Usage: make_pathogen_regression_fasta.py OUT.fasta [--tips 60] [--sites 3000] [--seed 20261114]
"""
import argparse
import random

BASES = "ACGT"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--tips", type=int, default=60)
    ap.add_argument("--sites", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--mu", type=float, default=1e-3, help="substitutions per site per year")
    ap.add_argument("--root-age", type=float, default=2.0, help="years before the last sample")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Sampling times in years before the last sample, over one year.
    tip_time = [rng.uniform(0.0, 1.0) for _ in range(args.tips)]

    # A random genealogy: repeatedly join the two youngest lineages at a uniformly drawn older time.
    # The shape does not matter for a regression; only that it is a fixed, valid timed tree.
    nodes = []  # (age_before_last_sample, sequence_index or None, children)
    lineages = []
    for i, t in enumerate(tip_time):
        nodes.append({"age": t, "tip": i, "children": []})
        lineages.append(len(nodes) - 1)
    while len(lineages) > 1:
        lineages.sort(key=lambda n: nodes[n]["age"])
        a, b = lineages.pop(0), lineages.pop(0)
        age = max(nodes[a]["age"], nodes[b]["age"]) + rng.uniform(0.02, 0.35)
        age = min(age, args.root_age)
        nodes.append({"age": age, "tip": None, "children": [a, b]})
        lineages.append(len(nodes) - 1)
    root = lineages[0]
    nodes[root]["age"] = args.root_age

    # Evolve a Jukes-Cantor sequence down the tree.
    root_seq = [rng.choice(BASES) for _ in range(args.sites)]
    tip_seq = [None] * args.tips
    stack = [(root, root_seq)]
    while stack:
        node, seq = stack.pop()
        if nodes[node]["tip"] is not None:
            tip_seq[nodes[node]["tip"]] = "".join(seq)
            continue
        for child in nodes[node]["children"]:
            branch = nodes[node]["age"] - nodes[child]["age"]
            child_seq = list(seq)
            expected = args.mu * branch * args.sites
            for _ in range(_poisson(rng, expected)):
                l = rng.randrange(args.sites)
                old = child_seq[l]
                child_seq[l] = rng.choice([b for b in BASES if b != old])
            stack.append((child, child_seq))

    # Dated IDs, with the last sample on 2022-01-01.
    with open(args.out, "w") as out:
        for i in range(args.tips):
            day = int(round(tip_time[i] * 365.0))
            out.write(">sample_%03d|%s\n" % (i, _iso_date_before(2022, 1, 1, day)))
            seq = tip_seq[i]
            for start in range(0, len(seq), 60):
                out.write(seq[start:start + 60] + "\n")


def _poisson(rng, mean):
    """Knuth's method; `mean` here is always small enough for it."""
    import math
    limit = math.exp(-mean)
    k, p = 0, rng.random()
    while p > limit:
        k += 1
        p *= rng.random()
    return k


def _iso_date_before(year, month, day, days_before):
    import datetime
    d = datetime.date(year, month, day) - datetime.timedelta(days=days_before)
    return d.isoformat()


if __name__ == "__main__":
    main()
