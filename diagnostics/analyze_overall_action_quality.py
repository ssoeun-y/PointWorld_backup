#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

Q = "diagnostics/results/natural_action_quality_per_sample.csv"
M = "diagnostics/results/natural_action_structural_baseline.csv"

q = pd.read_csv(Q)
m = pd.read_csv(M)

df = q.merge(m, on="clip", how="inner", suffixes=("", "_model"))
assert len(df) == 5288, len(df)

TEMP = [
    "speed_cv",
    "accel_p95",
    "jerk_p95",
    "reversal_fraction",
]

SPATIAL = [
    "path_inefficiency",
    "turn_p95_deg",
]

METRICS = [
    "l2_moved_mm",
    "EPE3D_mm",
    "response_abs_error_mm",
]


def task_pct(col):
    return (
        df.groupby("task")[col]
        .rank(method="average", pct=True)
    )


# -------------------------------------------------------
# 1. Each primitive metric -> within-task percentile
# -------------------------------------------------------
for c in TEMP + SPATIAL:
    df[c + "_quality_pct"] = task_pct(c)


# -------------------------------------------------------
# 2. Independent sub-scores
# Higher = more irregular / lower quality
# -------------------------------------------------------
df["temporal_irregularity"] = df[
    [c + "_quality_pct" for c in TEMP]
].mean(axis=1)

df["spatial_irregularity"] = df[
    [c + "_quality_pct" for c in SPATIAL]
].mean(axis=1)


# Normalize each subscore within task again before combining,
# so temporal does NOT dominate simply because it has more metrics.
df["temporal_score"] = (
    df.groupby("task")["temporal_irregularity"]
    .rank(method="average", pct=True)
)

df["spatial_score"] = (
    df.groupby("task")["spatial_irregularity"]
    .rank(method="average", pct=True)
)

# Equal contribution from the two independent quality axes.
df["overall_action_irregularity"] = (
    0.5 * df["temporal_score"]
    + 0.5 * df["spatial_score"]
)

df["overall_score"] = (
    df.groupby("task")["overall_action_irregularity"]
    .rank(method="average", pct=True)
)


SCORES = {
    "TEMPORAL": "temporal_score",
    "SPATIAL_ACTION_SHAPE": "spatial_score",
    "OVERALL_ACTION": "overall_score",
}


def analyze(name, score_col):

    print("\n" + "=" * 95)
    print(name)
    print("=" * 95)

    score = df[score_col]

    hq = df[score <= 0.25]
    lq = df[score >= 0.75]

    print("HQ N =", len(hq), "| LQ N =", len(lq))

    print("\nHQ vs LQ")

    for metric in METRICS:
        h = hq[metric].dropna()
        l = lq[metric].dropna()

        stat, p = mannwhitneyu(
            l, h,
            alternative="two-sided"
        )

        hm = h.median()
        lm = l.median()

        print(
            f"{metric:24s} "
            f"HQ={hm:.6f} | "
            f"LQ={lm:.6f} | "
            f"delta={lm-hm:+.6f} | "
            f"ratio={lm/hm:.3f}x | "
            f"p={p:.3e}"
        )

    print("\nDOSE RESPONSE (Q1 -> Q5)")

    quintile = np.ceil(score * 5).clip(1, 5).astype(int)

    for metric in METRICS:
        vals = []

        for k in range(1, 6):
            x = df.loc[quintile == k, metric].dropna()
            vals.append(x.median())

        print(
            f"{metric:24s}: "
            + " -> ".join(f"{x:.4f}" for x in vals)
        )

    print("\nSPEARMAN")

    for metric in METRICS:
        valid = score.notna() & df[metric].notna()

        rho, p = spearmanr(
            score[valid],
            df.loc[valid, metric]
        )

        print(
            f"{metric:24s} "
            f"rho={rho:+.4f} | p={p:.3e}"
        )

    print("\nTASKWISE moved 3D")

    for task in sorted(df.task.unique()):
        d = df[df.task == task]
        s = d[score_col]

        h = d[s <= 0.25]["l2_moved_mm"].dropna()
        l = d[s >= 0.75]["l2_moved_mm"].dropna()

        print(
            f"{task}: "
            f"HQ={h.median():.4f} | "
            f"LQ={l.median():.4f} | "
            f"ratio={l.median()/h.median():.3f}x"
        )


for name, col in SCORES.items():
    analyze(name, col)


print("\n" + "#" * 95)
print("DIRECT COMPARISON")
print("#" * 95)

for metric in METRICS:

    print("\n", metric)

    for name, col in SCORES.items():

        score = df[col]

        h = df.loc[score <= 0.25, metric].dropna()
        l = df.loc[score >= 0.75, metric].dropna()

        valid = score.notna() & df[metric].notna()

        rho, _ = spearmanr(
            score[valid],
            df.loc[valid, metric]
        )

        print(
            f"{name:22s} "
            f"HQ→LQ={l.median()/h.median():.3f}x | "
            f"rho={rho:+.4f}"
        )


OUT = (
    "diagnostics/results/"
    "natural_action_overall_quality_analysis.csv"
)

df.to_csv(OUT, index=False)

print("\nsaved =", OUT)
print("SUCCESS = True")
