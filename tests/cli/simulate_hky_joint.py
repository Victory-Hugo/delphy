#!/usr/bin/env python3
"""Joint generator for Delphy's calibrated strict-clock HKY model (test tooling only).

The data are drawn from the same joint distribution that Delphy samples from, with the substitution
model written exactly as `Hky_model::derive_site_evo_model` (core/evo_hky.cpp) builds it:

    r_ab = kappa for the transitions A<->G and C<->T, 1 otherwise,
    R    = pi^T r pi                        (sum over ALL ordered pairs, including a == b, where r_aa = 0),
    q_ab = r_ab pi_b / R   (a != b),   q_aa = -sum_b q_ab,
    rate matrix of one site per unit time = mu * q   (nu_l = 1: no site-rate heterogeneity).

Under q the mean substitution rate sum_a pi_a q_a is 1, pi is the stationary and root distribution
(sites are i.i.d. draws from pi at the root, as in `calc_log_root_prior`), and the process is
reversible.  A branch of `t` years carries the transition matrix expm(mu t q) with mu in
substitutions per site per YEAR; Delphy's internal rate is per site per DAY (mu / 365).

Joint distribution.  Parameters and genealogy are drawn from

    p(pi) p(kappa) p(mu) p(n0, g) * Kingman(tree | n0, g) * cal(root age) / Z

where p(pi) is uniform on the simplex, log kappa ~ N(1, 1.25^2), mu ~ Gamma(shape, rate) and
n0 ~ InverseGamma(shape, scale) with g either fixed or Laplace-distributed on [g_min, g_max], exactly
the priors Delphy's `calc_cur_log_other_priors` uses.  The root-age calibration is a factor of
Delphy's prior on the genealogy, so the draw is made by rejection from the coalescent proposal: the
pair (n0, g, tree) is accepted with probability cal(root age) / max cal.  Drawing n0 first and then
reweighting only the tree would give n0 the wrong marginal, because Z depends on n0.

Any parameter can instead be fixed on the command line; such a run is a conditional test and the
truth file says which parameters were fixed.  Independent random-number streams are used for the
parameters, the genealogy, the root sequence, the mutations and the reference encoding.

Output in OUTDIR: samples.fasta, reference.fasta, metadata.tsv, mask.txt, true_tree.nwk (branch
lengths in years), tips.tsv, root.cal (the calibration Delphy is given), truth.json.

Usage: simulate_hky_joint.py OUTDIR --samples N --seed S [--sites 16569]
         [--pi A C G T | --draw-pi] [--kappa K | --draw-kappa] [--mu M | --draw-mu SHAPE RATE]
         [--n0 YEARS | --draw-n0 SHAPE SCALE] [--growth G | --draw-growth MU SCALE GMIN GMAX]
         [--pop-floor 100] [--root-cal MEDIAN SIGMA YOUNGEST OLDEST] [--root-ref-diffs 20]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

BASES = "ACGT"
TRANSITION = {(0, 2), (2, 0), (1, 3), (3, 1)}  # A<->G, C<->T


PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # AC, AG, AT, CG, CT, GT


def canonical_code(code):
    """The canonical six-digit code of `code`: its restricted growth string."""
    if len(code) != 6 or not all(c.isdigit() for c in code):
        sys.exit("--subst-model takes a six-digit code over the pairs AC, AG, AT, CG, CT, GT")
    seen, out, nxt = {}, [], 0
    for c in code:
        if c not in seen:
            seen[c] = nxt
            nxt += 1
        out.append(seen[c])
    return out, nxt


def reversible_rate_matrix(pi, exchangeabilities):
    """Normalized reversible rate matrix from pi and the six exchangeabilities in the fixed order.

    Qt_ij = s_ij pi_j for i != j, the diagonal makes the rows sum to zero, and the matrix is divided by
    c = -sum_i pi_i Qt_ii so that the mean substitution rate sum_i pi_i (-Q_ii) is one.  With the HKY
    exchangeabilities (kappa for A<->G and C<->T, 1 otherwise) this is `hky_rate_matrix`.
    """
    pi = np.asarray(pi, dtype=float)
    r = np.zeros((4, 4))
    for value, (a, b) in zip(exchangeabilities, PAIRS):
        r[a, b] = r[b, a] = float(value)
    q = r * pi[None, :]
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))
    return q / -float(pi @ np.diag(q))


def hky_rate_matrix(pi, kappa):
    """Rate matrix of Delphy's HKY model, normalized to mean substitution rate one."""
    pi = np.asarray(pi, dtype=float)
    r = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            if a != b:
                r[a, b] = kappa if (a, b) in TRANSITION else 1.0
    big_r = pi @ r @ pi
    q = r * pi[None, :] / big_r
    np.fill_diagonal(q, -q.sum(axis=1))
    return q


def transition_matrix(q, t):
    """expm(q t) for the reversible rate matrix q by symmetrization and an eigendecomposition."""
    pi = stationary(q)
    s = np.sqrt(pi)
    sym = (s[:, None] * q) / s[None, :]
    sym = 0.5 * (sym + sym.T)
    w, v = np.linalg.eigh(sym)
    p = (v * np.exp(w * t)) @ v.T
    return (p / s[:, None]) * s[None, :]


def stationary(q):
    """Stationary distribution of the rate matrix: left null vector."""
    w, v = np.linalg.eig(q.T)
    i = np.argmin(np.abs(w))
    p = np.real(v[:, i])
    return p / p.sum()


class Population:
    """N(a) = max(floor, n0 exp(-g a)) at age a before the present; the model of Exp_pop_model."""

    def __init__(self, n0, g, floor):
        if g < 0:
            raise SystemExit("negative growth rates are not supported by this generator")
        self.n0, self.g, self.floor = n0, g, floor
        self.a_floor = math.log(n0 / floor) / g if (g > 0 and n0 > floor) else math.inf

    def big_lambda(self, a):
        """integral_0^a dx / N(x)."""
        n0, g, fl = self.n0, self.g, self.floor
        if g == 0:
            return a / max(n0, fl)
        if n0 <= fl:
            return a / fl
        if a <= self.a_floor:
            return math.expm1(g * a) / (g * n0)
        return math.expm1(g * self.a_floor) / (g * n0) + (a - self.a_floor) / fl

    def inv_lambda(self, lam):
        n0, g, fl = self.n0, self.g, self.floor
        if g == 0:
            return lam * max(n0, fl)
        if n0 <= fl:
            return lam * fl
        lam_f = math.expm1(g * self.a_floor) / (g * n0)
        if lam <= lam_f:
            return math.log1p(lam * g * n0) / g
        return self.a_floor + (lam - lam_f) * fl


def draw_kingman_heights(rng, n, pop):
    """Coalescence ages (increasing) of an n-tip same-date Kingman tree, and the lineage pairs."""
    ages = []
    a = 0.0
    lam = 0.0
    for k in range(n, 1, -1):
        lam += rng.exponential(1.0) / (k * (k - 1) / 2.0)
        a = pop.inv_lambda(lam)
        ages.append(a)
    return ages


def build_tree(rng, n, ages):
    """Random pairing of lineages at each coalescence.  Returns children, node_age; tips are 0..n-1."""
    active = list(range(n))
    children = {}
    node_age = {i: 0.0 for i in range(n)}
    nxt = n
    for a in ages:
        i, j = rng.choice(len(active), size=2, replace=False)
        x, y = active[i], active[j]
        active = [z for z in active if z not in (x, y)]
        children[nxt] = (x, y)
        node_age[nxt] = a
        active.append(nxt)
        nxt += 1
    return children, node_age, nxt - 1


def to_newick(children, node_age, node, names):
    if node not in children:
        return names[node]
    parts = []
    for c in children[node]:
        parts.append("%s:%.9f" % (to_newick(children, node_age, c, names), node_age[node] - node_age[c]))
    return "(" + ",".join(parts) + ")"


def lognormal_cal_log_density(x, median, sigma, youngest, oldest):
    if x < youngest or x > oldest:
        return -math.inf
    return -((math.log(x) - math.log(median)) ** 2) / (2 * sigma * sigma) - math.log(x * sigma * math.sqrt(2 * math.pi))


def cal_log_max(median, sigma, youngest, oldest):
    mode = median * math.exp(-sigma * sigma)
    x = min(max(mode, youngest), oldest)
    return lognormal_cal_log_density(x, median, sigma, youngest, oldest)


def simulate_sequences(rng, children, node_age, root, n, q, mu_year, sites, pi, site_rate=None):
    """Tip and root sequences (arrays of 0..3).  Branch of t years uses expm(mu r t q) per site.

    `site_rate` is the relative rate of every site (None: all 1).  Sites with the same relative rate
    share one transition matrix per branch, so the draw is made group by group; with a single group the
    random numbers are drawn exactly as they were before this argument existed."""
    groups = [(1.0, np.arange(sites))] if site_rate is None else [
        (float(r), np.flatnonzero(site_rate == r)) for r in np.unique(site_rate)]
    states = {root: rng.choice(4, size=sites, p=pi)}
    root_seq = states[root].copy()
    stack = [root]
    while stack:
        p = stack.pop()
        for c in children.get(p, ()):
            t = node_age[p] - node_age[c]
            child_states = np.empty(sites, dtype=np.int64)
            for rate, idx in groups:
                pm = transition_matrix(q, mu_year * rate * t)
                pm = np.clip(pm, 0.0, None)
                cum = np.cumsum(pm / pm.sum(axis=1, keepdims=True), axis=1)
                u = rng.random(len(idx))
                parent = states[p][idx]
                child_states[idx] = np.minimum((u[:, None] > cum[parent]).sum(axis=1), 3)
            states[c] = child_states
            stack.append(c)
    return [states[i] for i in range(n)], root_seq


def resolve_partitions(args):
    """Per-site relative rates and the truth record, or (None, None) without partitions."""
    if args.partition_sites is None and args.partition_rates is None:
        return None, None
    if args.partition_sites is None or args.partition_rates is None:
        sys.exit("--partition-sites and --partition-rates must be given together")
    if len(args.partition_sites) != len(args.partition_rates):
        sys.exit("--partition-sites and --partition-rates must have the same number of entries")
    names = args.partition_names or ["p%d" % (i + 1) for i in range(len(args.partition_rates))]
    if len(names) != len(args.partition_rates):
        sys.exit("--partition-names must have one name per partition")
    owner = np.full(args.sites, -1, dtype=np.int64)
    for p, spec in enumerate(args.partition_sites):
        for token in spec.split(","):
            token = token.strip()
            if not token:
                continue
            bounds = token.split("-")
            first = int(bounds[0])
            last = int(bounds[-1]) if len(bounds) == 2 else first
            if first < 1 or last < first or last > args.sites:
                sys.exit("partition %s: range %s is outside 1-%d" % (names[p], token, args.sites))
            if (owner[first - 1:last] >= 0).any():
                sys.exit("partition %s: range %s overlaps another partition" % (names[p], token))
            owner[first - 1:last] = p
    if (owner < 0).any():
        sys.exit("%d sites belong to no partition" % int((owner < 0).sum()))
    sizes = np.array([(owner == p).sum() for p in range(len(names))], dtype=float)
    if (sizes == 0).any():
        sys.exit("every partition must contain at least one site")
    raw = np.array(args.partition_rates, dtype=float)
    if (raw <= 0).any():
        sys.exit("partition rates must be positive")
    rates = raw / (float((sizes * raw).sum()) / float(sizes.sum()))
    site_rate = rates[owner]
    truth = dict(names=names, sites=list(args.partition_sites), num_sites=[int(x) for x in sizes],
                 rates_given=[float(x) for x in raw], rates=[float(x) for x in rates])
    return site_rate, truth


def days_from_civil(y, m, d):
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def draw_parameters(args, rng):
    truth = {"fixed": []}
    if args.pi is not None:
        pi = np.array(args.pi, dtype=float)
        pi /= pi.sum()
        truth["fixed"].append("pi")
    else:
        pi = rng.dirichlet([1, 1, 1, 1])
    kappa = args.kappa if args.kappa is not None else float(np.exp(rng.normal(1.0, 1.25)))
    if args.kappa is not None:
        truth["fixed"].append("kappa")
    if args.mu is not None:
        mu = args.mu
        truth["fixed"].append("mu")
    else:
        mu = float(rng.gamma(args.draw_mu[0], 1.0 / args.draw_mu[1]))
    truth.update(pi=[float(x) for x in pi], kappa=float(kappa), mu_site_year=float(mu))
    return truth


def draw_population(args, rng):
    if args.n0 is not None:
        n0, fixed_n0 = args.n0, True
    else:
        n0, fixed_n0 = float(args.draw_n0[1] / rng.gamma(args.draw_n0[0], 1.0)), False
    if args.growth is not None:
        g, fixed_g = args.growth, True
    else:
        mu_g, scale, gmin, gmax = args.draw_growth
        while True:
            g = mu_g + rng.laplace(0.0, scale)
            if gmin <= g <= gmax:
                break
        fixed_g = False
    return n0, g, fixed_n0, fixed_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sites", type=int, default=16569)
    ap.add_argument("--pi", type=float, nargs=4)
    ap.add_argument("--kappa", type=float)
    ap.add_argument("--subst-model", metavar="CODE",
                    help="six-digit exchangeability code over the pairs AC, AG, AT, CG, CT, GT "
                         "(equal digits constrain the rates to be equal).  Needs --exchangeabilities; "
                         "without it the generator uses HKY85 with --kappa, exactly as before.")
    ap.add_argument("--exchangeabilities", type=float, nargs="+", default=None,
                    help="one relative exchangeability per class of --subst-model, in class order, "
                         "with the class of G-T at 1")
    ap.add_argument("--site-rate-alpha", type=float, default=None,
                    help="draw a relative rate per site from Gamma(alpha, alpha) (mean one), which is "
                         "Delphy's continuous Gamma site rate model; the rates multiply the partition "
                         "rates")
    ap.add_argument("--mu", type=float, help="substitutions per site per year")
    ap.add_argument("--draw-mu", type=float, nargs=2, metavar=("SHAPE", "RATE"), default=[2.0, 1e8])
    ap.add_argument("--n0", type=float, help="coalescent timescale in years")
    ap.add_argument("--draw-n0", type=float, nargs=2, metavar=("SHAPE", "SCALE"), default=[3.0, 150000.0])
    ap.add_argument("--growth", type=float, help="growth rate per year; 0 is a constant population")
    ap.add_argument("--draw-growth", type=float, nargs=4, metavar=("MU", "SCALE", "GMIN", "GMAX"),
                    default=[0.0, 1e-5, 0.0, 1e-4])
    ap.add_argument("--pop-floor", type=float, default=100.0)
    ap.add_argument("--root-cal", type=float, nargs=4, metavar=("MEDIAN", "SIGMA", "YOUNGEST", "OLDEST"),
                    default=[140000.0, 0.25, 30000.0, 400000.0])
    ap.add_argument("--no-root-cal", action="store_true")
    ap.add_argument("--root-ref-diffs", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=100000)
    ap.add_argument("--partition-sites", action="append", default=None, metavar="SITES",
                    help="one per site partition, in the configuration order Delphy will read: a "
                         "comma-separated list of 1-based inclusive sites or ranges ('1-100,201-300'). "
                         "Default: one partition covering every site.")
    ap.add_argument("--partition-rates", type=float, nargs="+", default=None,
                    help="relative substitution rate of every partition; they are renormalized so that "
                         "their mean, weighted by the number of sites, is 1 (Delphy's convention)")
    ap.add_argument("--partition-names", nargs="+", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda f: os.path.join(args.outdir, f)

    ss = np.random.SeedSequence(args.seed)
    # The seventh stream is spawned unconditionally: SeedSequence.spawn is sequential, so the first
    # six streams are the ones every earlier run used and a data set without site rate heterogeneity
    # is unchanged.
    (s_par, s_tree, s_root, s_mut, s_ref, s_acc, s_nu) = [
        np.random.default_rng(s) for s in ss.spawn(7)]

    truth = draw_parameters(args, s_par)
    pi = np.array(truth["pi"])
    if args.subst_model is not None or args.exchangeabilities is not None:
        if args.subst_model is None or args.exchangeabilities is None:
            sys.exit("--subst-model and --exchangeabilities must be given together")
        class_of, num_classes = canonical_code(args.subst_model)
        if len(args.exchangeabilities) != num_classes:
            sys.exit("--exchangeabilities needs %d values for the code %s" %
                     (num_classes, "".join(str(k) for k in class_of)))
        reference = class_of[5]
        if args.exchangeabilities[reference] != 1.0:
            sys.exit("the class of G-T carries the relative scale and must be 1, but class %d is %g"
                     % (reference, args.exchangeabilities[reference]))
        if any(not (v > 0) for v in args.exchangeabilities):
            sys.exit("every exchangeability must be positive")
        s_vector = [args.exchangeabilities[class_of[i]] for i in range(6)]
        q = reversible_rate_matrix(pi, s_vector)
        truth["subst_model_code"] = "".join(str(k) for k in class_of)
        truth["exchangeability_classes"] = list(args.exchangeabilities)
        truth["exchangeabilities"] = s_vector
        truth["fixed"].append("subst_model")
    else:
        q = hky_rate_matrix(pi, truth["kappa"])

    n = args.samples
    med, sig, young, old = args.root_cal
    log_max = cal_log_max(med, sig, young, old)
    attempts = 0
    while True:
        attempts += 1
        if attempts > args.max_attempts:
            sys.exit("joint rejection sampler exceeded --max-attempts; the calibration and the "
                     "population prior put almost no joint mass together")
        n0, g, fixed_n0, fixed_g = draw_population(args, s_par)
        pop = Population(n0, g, args.pop_floor)
        ages = draw_kingman_heights(s_tree, n, pop)
        if args.no_root_cal:
            break
        lc = lognormal_cal_log_density(ages[-1], med, sig, young, old)
        if math.log(s_acc.random()) < lc - log_max:
            break
    if fixed_n0:
        truth["fixed"].append("n0")
    if fixed_g:
        truth["fixed"].append("growth")
    children, node_age, root = build_tree(s_tree, n, ages)
    names = ["S%06d" % i for i in range(n)]

    site_rate, partition_truth = resolve_partitions(args)
    if args.site_rate_alpha is not None:
        if not (args.site_rate_alpha > 0):
            sys.exit("--site-rate-alpha must be positive")
        alpha = args.site_rate_alpha
        # nu_l ~ Gamma(alpha, alpha) in the shape-rate parameterization: mean one, variance 1/alpha.
        nu = s_nu.gamma(shape=alpha, scale=1.0 / alpha, size=args.sites)
        site_rate = nu if site_rate is None else site_rate * nu
        truth["site_rate_alpha"] = alpha
        truth["site_rate_mean"] = float(np.mean(nu))
        truth["fixed"].append("site_rate_alpha")
    tips, root_seq = simulate_sequences(s_mut, children, node_age, root, n, q, truth["mu_site_year"],
                                        args.sites, pi, site_rate)
    if partition_truth is not None:
        truth["site_partitions"] = partition_truth
        with open(out("partitions.txt"), "w") as f:
            f.write("version 1\n")
            for name, sites_text in zip(partition_truth["names"], partition_truth["sites"]):
                f.write("\npartition %s\n  sites %s\nend\n" % (name, sites_text))
    ref = root_seq.copy()
    if args.root_ref_diffs > 0:
        idx = s_ref.choice(args.sites, size=args.root_ref_diffs, replace=False)
        ref[idx] = (ref[idx] + s_ref.integers(1, 4, size=args.root_ref_diffs)) % 4

    table = np.frombuffer(BASES.encode(), dtype=np.uint8)
    with open(out("reference.fasta"), "wb") as f:
        f.write(b">SIM_REF\n" + table[ref].tobytes() + b"\n")
    with open(out("samples.fasta"), "wb") as f:
        for i in range(n):
            f.write((">%s\n" % names[i]).encode() + table[tips[i]].tobytes() + b"\n")
    with open(out("metadata.tsv"), "w") as f:
        f.write("sample_id\tdate_convention\tdate_from\tdate_to\tbp_reference_year\n")
        for nm in names:
            f.write("%s\tiso\t2020-01-01\t\t\n" % nm)
    open(out("mask.txt"), "w").close()
    with open(out("true_tree.nwk"), "w") as f:
        f.write(to_newick(children, node_age, root, names) + ";\n")
    with open(out("tips.tsv"), "w") as f:
        f.write("sample_id\tage_years\n")
        for nm in names:
            f.write("%s\t0.0\n" % nm)
    if not args.no_root_cal:
        with open(out("root.cal"), "w") as f:
            f.write("version 1\n\ncalibration root\n  taxa all\n  age_units years_bp\n  age_reference_year 2020\n"
                    "  distribution lognormal\n  median %.17g\n  sigma %.17g\n  oldest %.17g\n  youngest %.17g\nend\n"
                    % (med, sig, old, young))
    truth.update(
        num_samples=n, num_sites=args.sites, n0_years=float(n0), growth_per_year=float(g),
        pop_floor_years=args.pop_floor, root_age_years=float(ages[-1]),
        coalescence_ages_years=[float(a) for a in ages], joint_rejection_attempts=attempts,
        root_calibration=None if args.no_root_cal else dict(median=med, sigma=sig, youngest=young, oldest=old),
        seed=args.seed, root_ref_diffs=args.root_ref_diffs,
        node_ages_years={str(k): float(v) for k, v in node_age.items() if k >= n},
        children={str(k): list(v) for k, v in children.items()})
    with open(out("truth.json"), "w") as f:
        json.dump(truth, f, indent=1, sort_keys=True)
    print(json.dumps({k: truth[k] for k in ("kappa", "mu_site_year", "n0_years", "growth_per_year",
                                           "root_age_years", "joint_rejection_attempts", "fixed")}))


if __name__ == "__main__":
    main()
