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

ALL = TEMP + SPATIAL

METRICS = [
    "l2_moved_mm",
    "EPE3D_mm",
    "response_abs_error_mm",
]


def pct_rank(col):
    return (
        df.groupby("task")[col]
        .rank(method="average", pct=True)
    )


# primitive metrics -> within-task percentile
for c in ALL:
    df[c + "_pct"] = pct_rank(c)


# temporal/spatial subscore
df["temporal_raw"] = df[
    [c + "_pct" for c in TEMP]
].mean(axis=1)

df["spatial_raw"] = df[
    [c + "_pct" for c in SPATIAL]
].mean(axis=1)

# normalize each axis again within task
df["temporal_score"] = (
    df.groupby("task")["temporal_raw"]
    .rank(method="average", pct=True)
)

df["spatial_score"] = (
    df.groupby("task")["spatial_raw"]
    .rank(method="average", pct=True)
)


# ---------------------------------------------------------
# Different reasonable definitions
# ---------------------------------------------------------
DEFINITIONS = {
    "TEMP30_SPATIAL70":
        0.30 * df["temporal_score"]
        + 0.70 * df["spatial_score"],

    "TEMP50_SPATIAL50":
        0.50 * df["temporal_score"]
        + 0.50 * df["spatial_score"],

    "TEMP70_SPATIAL30":
        0.70 * df["temporal_score"]
        + 0.30 * df["spatial_score"],

    # each primitive metric gets exactly equal weight
    "EQUAL_ALL_6":
        df[[c + "_pct" for c in ALL]].mean(axis=1),
}


rows = []

print("=" * 105)
print("ACTION QUALITY WEIGHT SENSITIVITY")
print("=" * 105)

for name, raw_score in DEFINITIONS.items():

    # final task-wise percentile so HQ/LQ sizes are comparable
    tmp = pd.DataFrame({
        "task": df["task"],
        "raw": raw_score,
    })

    score = (
        tmp.groupby("task")["raw"]
        .rank(method="average", pct=True)
    )

    df[name + "_score"] = score

    hq = df[score <= 0.25]
    lq = df[score >= 0.75]

    print("\n" + "#" * 105)
    print(name)
    print("#" * 105)

    for metric in METRICS:

        h = hq[metric].dropna()
        l = lq[metric].dropna()

        hm = h.median()
        lm = l.median()

        _, p = mannwhitneyu(
            l,
            h,
            alternative="two-sided",
        )

        valid = score.notna() & df[metric].notna()

        rho, sp = spearmanr(
            score[valid],
            df.loc[valid, metric],
        )

        ratio = lm / hm

        print(
            f"{metric:24s} "
            f"HQ={hm:.4f} | "
            f"LQ={lm:.4f} | "
            f"ratio={ratio:.3f}x | "
            f"rho={rho:+.4f} | "
            f"MWU p={p:.3e}"
        )

        rows.append({
            "definition": name,
            "metric": metric,
            "hq_n": len(h),
            "lq_n": len(l),
            "hq_median": hm,
            "lq_median": lm,
            "ratio_lq_over_hq": ratio,
            "spearman_rho": rho,
            "spearman_p": sp,
            "mannwhitney_p": p,
        })

    # Taskwise moved-3D direction check
    print("\nTASKWISE moved 3D")

    same_direction = 0

    for task in sorted(df.task.unique()):

        d = df[df.task == task]
        s = score.loc[d.index]

        h = d[s <= 0.25]["l2_moved_mm"].dropna()
        l = d[s >= 0.75]["l2_moved_mm"].dropna()

        ratio = l.median() / h.median()

        if ratio > 1:
            same_direction += 1

        print(
            f"{task}: "
            f"HQ={h.median():.4f} | "
            f"LQ={l.median():.4f} | "
            f"ratio={ratio:.3f}x"
        )

    print(
        f"same-direction tasks = "
        f"{same_direction}/{df.task.nunique()}"
    )


res = pd.DataFrame(rows)

OUT = (
    "diagnostics/results/"
    "action_quality_weight_sensitivity.csv"
)

res.to_csv(OUT, index=False)

print("\n" + "=" * 105)
print("SUMMARY — moved 3D")
print("=" * 105)

print(
    res[res.metric == "l2_moved_mm"][
        [
            "definition",
            "hq_median",
            "lq_median",
            "ratio_lq_over_hq",
            "spearman_rho",
            "mannwhitney_p",
        ]
    ].to_string(index=False)
)

print("\nsaved =", OUT)
print("SUCCESS = True")
