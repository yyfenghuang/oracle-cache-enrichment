"""Paired analysis of Direct vs Oracle, and outcome classification.

Usage:
    python analysis.py results/run_2026-07-18.json

Exit 0 whichever outcome is found. A run that yields outcome B is a result,
not a failure, and the exit code says so. Non-zero exit is reserved for the
analysis itself failing: bad input, missing conditions, nothing to compare.
"""

import argparse
import json
import sys

import numpy as np

from extractor import EXTRACTORS

CI_LEVEL = 0.95
N_RESAMPLES = 10000


def score(rows, condition):
    """Return {sample_id: bool} for one condition, using that condition's parser."""
    extract = EXTRACTORS[condition]
    out = {}
    for row in rows:
        if row["condition"] != condition:
            continue
        predicted = extract(row["raw_output"])
        out[row["sample_id"]] = predicted == row["gold"]
    return out


def bootstrap_ci(a, b, n_resamples=N_RESAMPLES, level=CI_LEVEL, seed=0):
    """Bootstrap CI over the paired difference in accuracy, b minus a.

    Resamples samples, not conditions. Both conditions saw the same questions,
    so the pairing is preserved by resampling sample indices and reading both
    conditions at each drawn index. Resampling the two conditions independently
    would throw away the pairing and inflate the interval.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)

    idx = rng.integers(0, n, size=(n_resamples, n))
    deltas = b[idx].mean(axis=1) - a[idx].mean(axis=1)

    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(deltas, [alpha, 1.0 - alpha])
    return float(lo), float(hi), deltas


def mcnemar(a, b):
    """McNemar test over paired correct/incorrect.

    n01: a wrong, b right. n10: a right, b wrong. Samples both got right or
    both got wrong carry no information about the difference and drop out.

    Uses the exact binomial test rather than the chi-square approximation.
    The discordant count here is small enough that the approximation is not
    reliable, and the exact test costs nothing at this scale.
    """
    from scipy.stats import binomtest

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n_discordant = n01 + n10

    if n_discordant == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0}

    p = binomtest(n01, n_discordant, 0.5).pvalue
    return {"n01": n01, "n10": n10, "p_value": float(p)}


def classify_outcome(ci_lo, ci_hi):
    """Return 'A', 'B', or 'C' per the criteria in README.md.

    The interval decides, not the point estimate. An interval that spans zero
    means the sign of the difference is not established, whatever the point
    estimate happens to be. That is outcome C, and it is not an endpoint.
    """
    if ci_lo > 0:
        return "A"
    if ci_hi < 0:
        return "B"
    return "C"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_file", type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.run_file) as f:
        run = json.load(f)
    rows = run["rows"]

    direct = score(rows, "direct")
    oracle = score(rows, "oracle")

    shared = sorted(set(direct) & set(oracle))
    if not shared:
        print("no samples shared between direct and oracle", file=sys.stderr)
        return 1

    a = [direct[i] for i in shared]
    b = [oracle[i] for i in shared]

    acc_direct = float(np.mean(a))
    acc_oracle = float(np.mean(b))
    delta = acc_oracle - acc_direct

    ci_lo, ci_hi, _ = bootstrap_ci(a, b, seed=args.seed)
    mc = mcnemar(a, b)
    outcome = classify_outcome(ci_lo, ci_hi)

    unparsed = {
        cond: sum(
            1
            for r in rows
            if r["condition"] == cond and EXTRACTORS[cond](r["raw_output"]) is None
        )
        for cond in ("direct", "few_shot", "oracle")
    }

    print(f"n paired          {len(shared)}")
    print(f"accuracy direct   {acc_direct:.3f}")
    print(f"accuracy oracle   {acc_oracle:.3f}")
    print(f"delta             {delta:+.3f}")
    print(f"bootstrap {int(CI_LEVEL * 100)}% CI  [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"mcnemar n01/n10   {mc['n01']}/{mc['n10']}")
    print(f"mcnemar p         {mc['p_value']:.4f}")
    print(f"unparsed          {unparsed}")
    print(f"outcome           {outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())