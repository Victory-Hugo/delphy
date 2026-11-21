#!/usr/bin/env python3
"""Test-only simulator of non-recombining mtDNA-like alignments with a known true tree and rate.

A heterochronous coalescent tree (constant or exponentially growing population) is simulated
backwards in time, mutations are placed on branches (Poisson in branch length, gamma site-rate
heterogeneity, optional hotspot class, transition bias), and the leaf sequences are written together
with the truth needed to audit Delphy: Newick tree, per-branch event counts, parameters, seed.

Units: 1 year = 365 days (the convention of Delphy's command line).  All ages are years before
2020-01-01.  The substitution rate and the coalescent timescale are explicit inputs; nothing here
is a statement about real human mtDNA, and the output must never be used as a default in Delphy.

Usage:
  simulate_mtdna_coalescent.py OUTDIR --samples N --seed S --mu-site-year MU --coal-timescale-years TAU
      [--dates same|modern_spread|ancient] [--ancient-frac F] [--ancient-max-age-years Y]
      [--growth-rate G] [--sites 16569] [--site-alpha A] [--hotspot-frac H --hotspot-mult M]
      [--kappa K] [--root-ref-diffs D] [--missing-sample-frac P] [--mask 303-315,16184-16193]

Writes into OUTDIR: reference.fasta, samples.fasta, metadata.tsv, mask.txt, true_tree.nwk,
branches.tsv, tips.tsv, truth.json.  Identical arguments give byte-identical output.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

BASES = np.frombuffer(b"ACGT", dtype=np.uint8)
TRANSITION = {0: 2, 1: 3, 2: 0, 3: 1}  # A<->G, C<->T
YEAR_DAYS = 365.0


def days_from_civil(y, m, d):
    """Days since 1970-01-01 in the proleptic Gregorian calendar, astronomical year numbering."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


DAY_2020 = days_from_civil(2020, 1, 1)


def parse_mask(spec):
    out = []
    if not spec:
        return out
    for part in spec.split(","):
        a, _, b = part.partition("-")
        lo = int(a)
        hi = int(b) if b else lo
        out.append((lo, hi))
    return out


def simulate_tree(rng, ages, tau, growth):
    """Backwards heterochronous coalescent.  ages: tip ages in years (before 2020).
    Returns parent (list), node_age (list); tips are 0..n-1, internal nodes n..2n-2."""
    n = len(ages)
    order = sorted(range(n), key=lambda i: ages[i])
    parent = [-1] * (2 * n - 1)
    node_age = [0.0] * (2 * n - 1)
    for i in range(n):
        node_age[i] = float(ages[i])
    active = []
    next_internal = n
    t = 0.0
    pos = 0

    def big_lambda(x):
        return x / tau if growth == 0.0 else math.expm1(growth * x) / (growth * tau)

    def inv_lambda(L):
        return L * tau if growth == 0.0 else math.log1p(growth * tau * L) / growth

    while pos < n or len(active) > 1:
        k = len(active)
        next_sample = node_age[order[pos]] if pos < n else math.inf
        if k < 2:
            t = max(t, next_sample)
            while pos < n and node_age[order[pos]] <= t:
                active.append(order[pos])
                pos += 1
            continue
        e = rng.exponential()
        t_c = inv_lambda(big_lambda(t) + e / (k * (k - 1) / 2.0))
        if t_c < next_sample:
            t = t_c
            i = int(rng.integers(k))
            active[i], active[-1] = active[-1], active[i]
            a = active.pop()
            j = int(rng.integers(k - 1))
            active[j], active[-1] = active[-1], active[j]
            b = active.pop()
            parent[a] = next_internal
            parent[b] = next_internal
            node_age[next_internal] = t
            active.append(next_internal)
            next_internal += 1
        else:
            t = next_sample
            while pos < n and node_age[order[pos]] <= t:
                active.append(order[pos])
                pos += 1
    return parent, node_age


def to_newick(children, node_age, n, names):
    sys.setrecursionlimit(1000000)
    root = 2 * n - 2
    out = []
    stack = [(root, 0)]
    # iterative post-order serialisation
    parts = {}
    order = []
    st = [root]
    while st:
        v = st.pop()
        order.append(v)
        st.extend(children.get(v, []))
    for v in reversed(order):
        if v < n:
            parts[v] = names[v]
        else:
            parts[v] = "(" + ",".join(
                "%s:%.6f" % (parts.pop(c), node_age[v] - node_age[c]) for c in children[v]) + ")"
    return parts[root] + ";"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sites", type=int, default=16569)
    ap.add_argument("--mu-site-year", type=float, required=True,
                    help="TRUE substitution rate, substitutions per site per year (365 days)")
    ap.add_argument("--coal-timescale-years", type=float, required=True,
                    help="Pairwise coalescent timescale at the present, in years (not a head count)")
    ap.add_argument("--growth-rate", type=float, default=0.0,
                    help="Forward exponential growth rate per year (0 = constant size)")
    ap.add_argument("--dates", choices=["same", "modern_spread", "ancient"], default="same")
    ap.add_argument("--modern-spread-years", type=float, default=30.0)
    ap.add_argument("--ancient-frac", type=float, default=0.05)
    ap.add_argument("--ancient-min-age-years", type=float, default=500.0)
    ap.add_argument("--ancient-max-age-years", type=float, default=40000.0)
    ap.add_argument("--site-alpha", type=float, default=0.3, help="Gamma shape of site rates")
    ap.add_argument("--hotspot-frac", type=float, default=0.01)
    ap.add_argument("--hotspot-mult", type=float, default=20.0)
    ap.add_argument("--kappa", type=float, default=10.0, help="Transition/transversion bias")
    ap.add_argument("--root-ref-diffs", type=int, default=20,
                    help="Number of sites where the root differs from the reference")
    ap.add_argument("--missing-sample-frac", type=float, default=0.05,
                    help="Fraction of samples carrying runs of missing data")
    ap.add_argument("--heavy-missing-samples", type=int, default=1,
                    help="Samples with about 8 percent missing data")
    ap.add_argument("--mask", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, L = args.samples, args.sites
    if n < 2:
        sys.exit("need at least 2 samples")

    # --- sampling ages (years before 2020-01-01) and metadata rows
    meta = []
    ages = []
    n_anc = int(round(args.ancient_frac * n)) if args.dates == "ancient" else 0
    for i in range(n):
        sid = "S%06d" % i
        if args.dates == "same":
            ages.append(0.0)
            meta.append((sid, "", "", "", ""))
        elif args.dates == "modern_spread" or i >= n_anc:
            days = int(rng.integers(0, int(args.modern_spread_years * YEAR_DAYS) + 1))
            d = DAY_2020 - days
            # convert back to civil date for the metadata
            z = d + 719468
            era = (z if z >= 0 else z - 146096) // 146097
            doe = z - era * 146097
            yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
            y = yoe + era * 400
            doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
            mp = (5 * doy + 2) // 153
            dd = doy - (153 * mp + 2) // 5 + 1
            mm = mp + (3 if mp < 10 else -9)
            y += mm <= 2
            ages.append((DAY_2020 - days_from_civil(y, mm, dd)) / YEAR_DAYS)
            meta.append((sid, "iso", "%04d-%02d-%02d" % (y, mm, dd), "", ""))
        else:
            cal_bp = int(rng.integers(int(args.ancient_min_age_years), int(args.ancient_max_age_years) + 1))
            # cal_bp a -> 1 January of year 1950 - a
            days_before = DAY_2020 - days_from_civil(1950 - cal_bp, 1, 1)
            ages.append(days_before / YEAR_DAYS)
            meta.append((sid, "cal_bp", str(cal_bp), "", ""))
    names = ["S%06d" % i for i in range(n)]

    # --- true tree
    parent, node_age = simulate_tree(rng, ages, args.coal_timescale_years, args.growth_rate)
    children = {}
    for v, p in enumerate(parent):
        if p >= 0:
            children.setdefault(p, []).append(v)
    root = 2 * n - 2
    assert parent[root] == -1

    # --- site model
    rates = rng.gamma(args.site_alpha, 1.0 / args.site_alpha, size=L)
    n_hot = int(round(args.hotspot_frac * L))
    if n_hot:
        hot = rng.choice(L, size=n_hot, replace=False)
        rates[hot] *= args.hotspot_mult
    rates /= rates.mean()
    cum = np.cumsum(rates)
    cum /= cum[-1]
    ref_idx = rng.choice(4, size=L, p=[0.31, 0.31, 0.13, 0.25]).astype(np.uint8)
    root_seq = ref_idx.copy()
    root_sites = rng.choice(L, size=args.root_ref_diffs, replace=False)
    for s in root_sites:
        b = root_seq[s]
        root_seq[s] = TRANSITION[int(b)] if rng.random() < args.kappa / (args.kappa + 2) else \
            rng.choice([x for x in range(4) if x not in (int(b), TRANSITION[int(b)])])
    p_ts = args.kappa / (args.kappa + 2.0)

    def mutate(seq, k):
        sites = np.searchsorted(cum, rng.random(k))
        for s in sites:
            b = int(seq[s])
            if rng.random() < p_ts:
                seq[s] = TRANSITION[b]
            else:
                seq[s] = rng.choice([x for x in range(4) if x not in (b, TRANSITION[b])])
        return sites

    mu_year_total = args.mu_site_year * L
    seqs = np.zeros((n, L), dtype=np.uint8)
    branches = []  # node, length_years, events, net_diffs
    hit_count = np.zeros(L, dtype=np.int32)
    st = [(root, root_seq)]
    while st:
        v, seq = st.pop()
        if v < n:
            seqs[v] = seq
        for c in children.get(v, []):
            length = node_age[v] - node_age[c]
            k = int(rng.poisson(mu_year_total * length))
            cseq = seq.copy()
            if k:
                hit = mutate(cseq, k)
                np.add.at(hit_count, hit, 1)
            branches.append((c, length, k, int(np.count_nonzero(cseq != seq))))
            st.append((c, cseq))

    # --- missing data (runs of N) and mask
    miss = np.zeros((n, L), dtype=bool)
    n_miss = int(round(args.missing_sample_frac * n))
    chosen = rng.choice(n, size=min(n, n_miss + args.heavy_missing_samples), replace=False)
    for j, i in enumerate(chosen):
        frac = 0.08 if j >= n_miss else float(rng.uniform(0.001, 0.03))
        total = int(frac * L)
        runs = int(rng.integers(1, 6))
        for r in range(runs):
            ln = max(1, total // runs)
            st0 = int(rng.integers(0, L - ln))
            miss[i, st0:st0 + ln] = True
    mask = parse_mask(args.mask)
    masked_sites = np.zeros(L, dtype=bool)
    for lo, hi in mask:
        masked_sites[lo - 1:hi] = True

    os.makedirs(args.outdir, exist_ok=True)
    out = lambda f: os.path.join(args.outdir, f)
    ref_bytes = BASES[ref_idx].tobytes().decode()
    with open(out("reference.fasta"), "w") as f:
        f.write(">SIM_REF\n" + ref_bytes + "\n")
    with open(out("samples.fasta"), "wb") as f:
        for i in range(n):
            row = BASES[seqs[i]].copy()
            row[miss[i]] = ord("N")
            f.write((">%s\n" % names[i]).encode() + row.tobytes() + b"\n")
    with open(out("metadata.tsv"), "w") as f:
        f.write("sample_id\tdate_convention\tdate_from\tdate_to\tbp_reference_year\n")
        for row in meta:
            f.write("\t".join(row) + "\n")
    with open(out("mask.txt"), "w") as f:
        for lo, hi in mask:
            f.write("%d-%d\n" % (lo, hi))
    with open(out("true_tree.nwk"), "w") as f:
        f.write(to_newick(children, node_age, n, names) + "\n")
    with open(out("branches.tsv"), "w") as f:
        f.write("node\tis_tip\tlength_years\tmutation_events\tnet_differences\n")
        for c, ln, k, d in branches:
            f.write("%d\t%d\t%.6f\t%d\t%d\n" % (c, int(c < n), ln, k, d))

    ev = np.array([b[2] for b in branches])
    root_depth = node_age[root]
    # truth for root-to-tip event counts
    r2t = np.zeros(2 * n - 1, dtype=np.int64)
    for c, ln, k, d in sorted(branches, key=lambda b: -node_age[b[0]]):
        r2t[c] = r2t[parent[c]] + k
    with open(out("tips.tsv"), "w") as f:
        f.write("sample_id\tage_years\ttrue_root_to_tip_events\ttrue_root_to_tip_years\n")
        for i in range(n):
            f.write("%s\t%.6f\t%d\t%.6f\n" % (names[i], ages[i], r2t[i], root_depth - ages[i]))
    tip_ages = np.array(ages)
    unique_seqs = len({seqs[i].tobytes() for i in range(n)})
    truth = {
        "generator": "simulate_mtdna_coalescent.py",
        "args": vars(args),
        "seed": args.seed,
        "year_days": YEAR_DAYS,
        "num_samples": n,
        "num_sites": L,
        "masked_sites": int(masked_sites.sum()),
        "effective_sites": int(L - masked_sites.sum()),
        "true_mu_site_year": args.mu_site_year,
        "true_mu_site_day": args.mu_site_year / YEAR_DAYS,
        "true_mutations_per_year_effective": args.mu_site_year * (L - int(masked_sites.sum())),
        "true_tmrca_years": root_depth,
        "tip_age_min_years": float(tip_ages.min()),
        "tip_age_max_years": float(tip_ages.max()),
        "sampling_span_years": float(tip_ages.max() - tip_ages.min()),
        "num_branches": len(branches),
        "total_events": int(ev.sum()),
        "frac_branches_zero_events": float((ev == 0).mean()),
        "root_to_tip_events_mean": float(r2t[:n].mean()),
        "root_to_tip_events_max": int(r2t[:n].max()),
        "homoplastic_sites_hit_2plus": int((hit_count >= 2).sum()),
        "unique_sequences_before_missing": unique_seqs,
        "missing_fraction_total": float(miss.mean()),
        "root_reference_differences": int(args.root_ref_diffs),
    }
    with open(out("truth.json"), "w") as f:
        json.dump(truth, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: truth[k] for k in (
        "num_samples", "true_tmrca_years", "sampling_span_years", "total_events",
        "frac_branches_zero_events", "root_to_tip_events_mean", "unique_sequences_before_missing")}))


if __name__ == "__main__":
    main()
