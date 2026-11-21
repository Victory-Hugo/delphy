#!/usr/bin/env python3
"""Test-only generator of synthetic mtDNA-like alignments for the human input path.

The data are random sequences with tree structure (each sample derives from an earlier one by
1-4 substitutions).  They are not human data and carry no biological meaning.

Usage: make_synthetic_mtdna.py OUTDIR [--samples N] [--sites L] [--seed S] [--dates MODE]

MODE: uniform  - metadata leaves dates empty (use --v0-human-uniform-sample-date)
      modern   - ISO dates spread over 2010-2020
      ancient  - mixed: cal_bp ranges for older samples, ISO dates for the rest

Writes reference.fasta, samples.fasta and metadata.tsv into OUTDIR.
"""
import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--sites", type=int, default=16569)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dates", choices=["uniform", "modern", "ancient"], default="uniform")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bases = "ACGT"
    ref = [rng.choice(bases) for _ in range(args.sites)]
    seqs = []
    for i in range(args.samples):
        s = list(ref) if i == 0 else list(seqs[rng.randrange(i)])
        for _ in range(rng.randint(1, 4)):
            site = rng.randrange(args.sites)
            s[site] = rng.choice([b for b in bases if b != s[site]])
        seqs.append(s)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "reference.fasta"), "w") as f:
        f.write(">SYNTH_REF synthetic reference\n" + "".join(ref) + "\n")
    with open(os.path.join(args.outdir, "samples.fasta"), "w") as f:
        for i, s in enumerate(seqs):
            f.write(">SYN%05d synthetic sample\n%s\n" % (i, "".join(s)))
    with open(os.path.join(args.outdir, "metadata.tsv"), "w") as f:
        f.write("sample_id\tdate_convention\tdate_from\tdate_to\tbp_reference_year\n")
        for i in range(args.samples):
            sid = "SYN%05d" % i
            if args.dates == "uniform":
                f.write("%s\t\t\t\t\n" % sid)
            elif args.dates == "modern":
                f.write("%s\tiso\t%d-%02d-%02d\t\t\n" % (sid, 2010 + rng.randrange(11), 1 + rng.randrange(12), 1 + rng.randrange(28)))
            else:
                if i % 2 == 0:
                    older = 3000 + 100 * i
                    f.write("%s\tcal_bp\t%d\t%d\t\n" % (sid, older, older - 80))
                else:
                    f.write("%s\tiso\t%d-%02d-%02d\t\t\n" % (sid, 2010 + rng.randrange(11), 1 + rng.randrange(12), 1 + rng.randrange(28)))


if __name__ == "__main__":
    main()
