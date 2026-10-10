#!/usr/bin/env python3

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

SEED = 20260902

ROOT = Path(__file__).resolve().parents[1]

METRICS_CSV = (
    ROOT /
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)

CORR_CSV = (
    ROOT /
    "diagnostics/results/"
    "goal_preserved_global_boundary_taskcontrolled_pointcloud_structure/"
    "global_fixed_correct_5task__pointcloud_structure.csv"
)

OUTDIR = (
    ROOT /
    "diagnostics/results/"
    "boundary_identifiability_context_C/"
    "C5_context_stratified_correction"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Frozen 70/30 split
# ============================================================

def stable_number(text):
    h = hashlib.sha256(
        f"{SEED}|{text}".encode()
    ).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def exact_task_split(df):
    result = pd.Series(
        index=df.index,
        dtype="object",
    )

    for task, g in df.groupby("task"):
        idx = sorted(
            g.index,
            key=lambda i:
                stable_number(
                    str(df.at[i, "clip"])
                )
        )

        n_cal = round(len(idx) * 0.70)

        result.loc[idx[:n_cal]] = "calibration"
        result.loc[idx[n_cal:]] = "validation"

    return result


# ============================================================
# Statistics
# ============================================================

def paired_summary(g, original, corrected):
    a = g[original].to_numpy(float)
    b = g[corrected].to_numpy(float)

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (a > 0)
    )

    a = a[valid]
    b = b[valid]

    if len(a) == 0:
        return {
            "n": 0,
            "original_median": np.nan,
            "corrected_median": np.nan,
            "median_paired_ratio": np.nan,
            "improvement_pct": np.nan,
            "corrected_lower_pct": np.nan,
            "wilcoxon_p": np.nan,
        }

    ratios = b / a
    median_ratio = float(np.median(ratios))

    if len(a) >= 2 and np.any(a != b):
        try:
            p = float(
                wilcoxon(
                    b,
                    a,
                    alternative="two-sided",
                    zero_method="wilcox",
                ).pvalue
            )
        except ValueError:
            p = np.nan
    else:
        p = np.nan

    return {
        "n": int(len(a)),
        "original_median": float(np.median(a)),
        "corrected_median": float(np.median(b)),
        "median_paired_ratio": median_ratio,

        # Correct definition:
        # median of per-sample corrected/original ratios
        "improvement_pct":
            float((1.0 - median_ratio) * 100.0),

        "corrected_lower_pct":
            float(np.mean(b < a) * 100.0),

        "wilcoxon_p": p,
    }


# ============================================================
# Load
# ============================================================

m = pd.read_csv(METRICS_CSV)
c = pd.read_csv(CORR_CSV)

assert len(m) == 7987
assert len(c) == 687

m["split"] = exact_task_split(m)

print("metric population =", len(m))
print("calibration =", (m["split"] == "calibration").sum())
print("validation  =", (m["split"] == "validation").sum())
print("correction scored =", len(c))


# ============================================================
# Dominant-hand path length
# ============================================================

hand = (
    m["dominant_hand"]
    .astype(str)
    .str.lower()
)

m["dominant_path_length_m"] = np.where(
    hand.str.contains("right"),
    m["right_path_length_m"],
    np.where(
        hand.str.contains("left"),
        m["left_path_length_m"],
        np.nan,
    )
)

# Safety fallback only if dominant_hand encoding is unexpected.
missing = m["dominant_path_length_m"].isna()

if missing.any():
    print(
        "WARNING: unresolved dominant hand rows =",
        int(missing.sum())
    )
    print(
        "dominant_hand values =",
        sorted(m.loc[missing, "dominant_hand"].astype(str).unique())
    )


# ============================================================
# IMPORTANT:
# Context bin thresholds come ONLY from calibration data.
#
# Path length = action spatial execution magnitude/context.
# moved-L2 = original prediction difficulty.
#
# We calculate thresholds WITHIN TASK to avoid one task's
# scale/distribution dominating another.
# ============================================================

cal = m[m["split"] == "calibration"].copy()

threshold_rows = []

for task, g in cal.groupby("task"):

    for variable in [
        "dominant_path_length_m",
        "l2_moved_mm",
    ]:
        x = g[variable].to_numpy(float)
        x = x[np.isfinite(x)]

        q25, q50, q75 = np.quantile(
            x,
            [0.25, 0.50, 0.75]
        )

        threshold_rows.append({
            "task": task,
            "variable": variable,
            "q25": float(q25),
            "q50": float(q50),
            "q75": float(q75),
            "calibration_n": int(len(x)),
        })

thr = pd.DataFrame(threshold_rows)

thr.to_csv(
    OUTDIR / "C5_calibration_context_thresholds.csv",
    index=False
)


def assign_q(value, q25, q50, q75):
    if not np.isfinite(value):
        return np.nan
    if value <= q25:
        return "Q1"
    if value <= q50:
        return "Q2"
    if value <= q75:
        return "Q3"
    return "Q4"


# ============================================================
# Join correction result to original action/context
# ============================================================

keep = [
    "clip",
    "task",
    "split",
    "dominant_path_length_m",
    "l2_moved_mm",
    "rms_acceleration",
    "mean_squared_jerk",
]

eval_df = c.merge(
    m[keep],
    on=["clip", "task"],
    how="left",
    validate="one_to_one",
)

assert len(eval_df) == 687

missing_join = eval_df["split"].isna().sum()
assert missing_join == 0

# These should all belong to frozen validation.
print(
    "correction rows by split =",
    eval_df["split"].value_counts(dropna=False).to_dict()
)


# ============================================================
# Assign calibration-fixed context quartiles
# ============================================================

for variable, new_col in [
    ("dominant_path_length_m", "path_q"),
    ("l2_moved_mm", "error_q"),
]:
    eval_df[new_col] = np.nan

    for task, gidx in eval_df.groupby("task").groups.items():

        row = thr[
            (thr["task"] == task)
            & (thr["variable"] == variable)
        ].iloc[0]

        for i in gidx:
            eval_df.at[i, new_col] = assign_q(
                float(eval_df.at[i, variable]),
                row["q25"],
                row["q50"],
                row["q75"],
            )


eval_df.to_csv(
    OUTDIR / "C5_joined_scored687.csv",
    index=False
)


# ============================================================
# Metrics to evaluate
# ============================================================

PCL_METRICS = {
    "rigidity_mean": (
        "original_rigidity_mean_mm",
        "corrected_rigidity_mean_mm",
    ),
    "rigidity_p95": (
        "original_rigidity_p95_mm",
        "corrected_rigidity_p95_mm",
    ),
    "final_rigidity_mean": (
        "original_final_rigidity_mean_mm",
        "corrected_final_rigidity_mean_mm",
    ),
    "jitter_mean": (
        "original_jitter_mean_mm",
        "corrected_jitter_mean_mm",
    ),
    "jitter_p95": (
        "original_jitter_p95_mm",
        "corrected_jitter_p95_mm",
    ),
}


# ============================================================
# C5-A Overall + per task
# ============================================================

rows = []

for metric_name, (orig, corr) in PCL_METRICS.items():

    rows.append({
        "stratification": "overall",
        "group": "ALL",
        "metric": metric_name,
        **paired_summary(
            eval_df,
            orig,
            corr,
        )
    })

    for task, g in eval_df.groupby("task"):
        rows.append({
            "stratification": "task",
            "group": str(task),
            "metric": metric_name,
            **paired_summary(
                g,
                orig,
                corr,
            )
        })

summary_overall_task = pd.DataFrame(rows)

summary_overall_task.to_csv(
    OUTDIR / "C5A_overall_task_efficacy.csv",
    index=False
)


# ============================================================
# C5-B Spatial difficulty:
# calibration-defined path-length quartiles
# ============================================================

rows = []

for q in ["Q1", "Q2", "Q3", "Q4"]:

    g = eval_df[
        eval_df["path_q"] == q
    ]

    for metric_name, (orig, corr) in PCL_METRICS.items():
        rows.append({
            "path_q": q,
            "metric": metric_name,
            **paired_summary(
                g,
                orig,
                corr,
            )
        })

path_df = pd.DataFrame(rows)

path_df.to_csv(
    OUTDIR / "C5B_path_context_efficacy.csv",
    index=False
)


# ============================================================
# C5-C Original prediction-difficulty strata:
# thresholds fixed from calibration moved-L2.
#
# This tests whether correction benefit is confined to
# originally easy cases or survives in difficult predictions.
# ============================================================

rows = []

for q in ["Q1", "Q2", "Q3", "Q4"]:

    g = eval_df[
        eval_df["error_q"] == q
    ]

    for metric_name, (orig, corr) in PCL_METRICS.items():
        rows.append({
            "error_q": q,
            "metric": metric_name,
            **paired_summary(
                g,
                orig,
                corr,
            )
        })

error_df = pd.DataFrame(rows)

error_df.to_csv(
    OUTDIR / "C5C_prediction_difficulty_efficacy.csv",
    index=False
)


# ============================================================
# C5-D Continuous context vs correction benefit
#
# No threshold tuning.
# Benefit > 0 means corrected structural metric is lower.
# ============================================================

continuous_rows = []

for metric_name, (orig, corr) in PCL_METRICS.items():

    a = eval_df[orig].to_numpy(float)
    b = eval_df[corr].to_numpy(float)

    benefit = np.where(
        np.isfinite(a)
        & np.isfinite(b)
        & (a > 0),
        1.0 - b / a,
        np.nan,
    )

    tmp = eval_df[
        [
            "dominant_path_length_m",
            "l2_moved_mm",
            "task",
        ]
    ].copy()

    tmp["benefit"] = benefit

    for context in [
        "dominant_path_length_m",
        "l2_moved_mm",
    ]:

        z = tmp[
            [context, "benefit", "task"]
        ].dropna().copy()

        rho_raw, p_raw = spearmanr(
            z[context],
            z["benefit"],
        )

        # Within-task rank association avoids task composition
        # driving the result.
        z["context_rank"] = (
            z.groupby("task")[context]
            .rank(pct=True)
        )

        z["benefit_rank"] = (
            z.groupby("task")["benefit"]
            .rank(pct=True)
        )

        rho_task, p_task = spearmanr(
            z["context_rank"],
            z["benefit_rank"],
        )

        continuous_rows.append({
            "metric": metric_name,
            "context": context,
            "n": len(z),
            "raw_rho": float(rho_raw),
            "raw_p": float(p_raw),
            "within_task_rho": float(rho_task),
            "within_task_p": float(p_task),
        })

continuous_df = pd.DataFrame(
    continuous_rows
)

continuous_df.to_csv(
    OUTDIR / "C5D_continuous_context_benefit.csv",
    index=False
)


# ============================================================
# Print only paper-relevant jitter results
# ============================================================

print()
print("=" * 100)
print("C5-A — JITTER MEAN: OVERALL / TASK")
print("=" * 100)

print(
    summary_overall_task[
        summary_overall_task["metric"]
        == "jitter_mean"
    ][
        [
            "stratification",
            "group",
            "n",
            "improvement_pct",
            "corrected_lower_pct",
            "wilcoxon_p",
        ]
    ].to_string(index=False)
)


print()
print("=" * 100)
print("C5-B — JITTER MEAN BY CALIBRATION-FIXED PATH-LENGTH QUARTILE")
print("=" * 100)

print(
    path_df[
        path_df["metric"]
        == "jitter_mean"
    ][
        [
            "path_q",
            "n",
            "improvement_pct",
            "corrected_lower_pct",
            "wilcoxon_p",
        ]
    ].to_string(index=False)
)


print()
print("=" * 100)
print("C5-C — JITTER MEAN BY CALIBRATION-FIXED ORIGINAL ERROR QUARTILE")
print("=" * 100)

print(
    error_df[
        error_df["metric"]
        == "jitter_mean"
    ][
        [
            "error_q",
            "n",
            "improvement_pct",
            "corrected_lower_pct",
            "wilcoxon_p",
        ]
    ].to_string(index=False)
)


print()
print("=" * 100)
print("C5-D — CONTINUOUS CONTEXT vs BENEFIT")
print("=" * 100)

print(
    continuous_df[
        continuous_df["metric"]
        == "jitter_mean"
    ].to_string(index=False)
)


print()
print("=" * 100)
print("OUTPUT DIR =", OUTDIR)
print("SUCCESS = True")
print("=" * 100)
