#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

PC_ROOT = (
    ROOT / "diagnostics/results/"
    "goal_preserved_pointcloud_structure"
)

OUTDIR = (
    ROOT / "diagnostics/results/"
    "goal_preserved_pointcloud_dose_response"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Point-cloud paired results
# ------------------------------------------------------------
files = [
    PC_ROOT / "task0000__pointcloud_structure.csv",
    PC_ROOT / "task0001__pointcloud_structure.csv",
    PC_ROOT / "task0006__pointcloud_structure.csv",
]

pc = pd.concat(
    [pd.read_csv(p) for p in files],
    ignore_index=True,
)

pc = pc[pc["status"] == "OK"].copy()

# ------------------------------------------------------------
# Original established Acc / MSJ
# ------------------------------------------------------------
est = pd.read_csv(
    ROOT / "diagnostics/results/"
    "established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)

est = est[
    [
        "clip",
        "task",
        "rms_acceleration",
        "mean_squared_jerk",
    ]
].copy()

df = pc.merge(
    est,
    on=["clip", "task"],
    how="left",
    validate="one_to_one",
)

if df["rms_acceleration"].isna().any():
    raise RuntimeError("Missing established metrics after merge")

# ------------------------------------------------------------
# Validated taskwise boundaries used by correction v2
# ------------------------------------------------------------
ACC_B = {
    "task-0000": 0.00681915641438833,
    "task-0001": 0.007750296718789663,
    "task-0006": 0.00973419551589932,
}

MSJ_B = {
    "task-0000": 3.262687321201812e-05,
    "task-0001": 4.473807247045902e-05,
    "task-0006": 9.884830273773934e-05,
}

df["acc_boundary"] = df["task"].map(ACC_B)
df["msj_boundary"] = df["task"].map(MSJ_B)

# Dimensionless violation severity.
# Example: 0.50 = 50% above boundary.
df["acc_excess"] = (
    df["rms_acceleration"] / df["acc_boundary"] - 1.0
)

df["msj_excess"] = (
    df["mean_squared_jerk"] / df["msj_boundary"] - 1.0
)

# ------------------------------------------------------------
# Point-cloud improvement
#
# Positive = corrected is better.
# 0.05 = 5% reduction of structural error.
# ------------------------------------------------------------
for metric in [
    "rigidity_mean_mm",
    "rigidity_p95_mm",
    "jitter_mean_mm",
    "jitter_p95_mm",
]:
    orig = df[f"original_{metric}"].astype(float)
    corr = df[f"corrected_{metric}"].astype(float)

    df[f"{metric}_improvement_frac"] = (
        1.0 - corr / np.maximum(orig, 1e-12)
    )

# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------
primary_metrics = [
    "rigidity_mean_mm",
    "jitter_mean_mm",
]

lines = []

def emit(x=""):
    print(x)
    lines.append(str(x))


emit("=" * 92)
emit("BOUNDARY VIOLATION → POINT-CLOUD IMPROVEMENT DOSE RESPONSE")
emit("=" * 92)
emit(f"Structurally scored total N = {len(df)}")
emit()

for axis, excess_col in [
    ("ACCELERATION", "acc_excess"),
    ("MSJ", "msj_excess"),
]:
    # Only analyze samples that actually violated this boundary.
    g = df[
        np.isfinite(df[excess_col])
        & (df[excess_col] > 0)
    ].copy()

    emit("=" * 92)
    emit(axis)
    emit("=" * 92)
    emit(f"Boundary-violating N = {len(g)}")
    emit(
        "Median exceedance = "
        f"{100*np.median(g[excess_col]):.2f}% above boundary"
    )
    emit()

    for metric in primary_metrics:
        ycol = f"{metric}_improvement_frac"

        z = g[
            np.isfinite(g[excess_col])
            & np.isfinite(g[ycol])
        ].copy()

        rho, p = spearmanr(
            z[excess_col],
            z[ycol],
        )

        emit(metric)
        emit(
            f"  Spearman excess vs improvement: "
            f"rho={rho:+.4f}, p={p:.6g}, N={len(z)}"
        )
        emit(
            f"  Median improvement: "
            f"{100*np.median(z[ycol]):+.3f}%"
        )
        emit(
            f"  Improved samples: "
            f"{100*np.mean(z[ycol] > 0):.2f}%"
        )

        # ----------------------------------------------------
        # Dose-response quartiles by normalized violation
        # ----------------------------------------------------
        try:
            z["severity_bin"] = pd.qcut(
                z[excess_col],
                q=4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop",
            )

            emit("  Dose-response:")

            for q, qg in z.groupby(
                "severity_bin",
                observed=True,
            ):
                emit(
                    f"    {q}: "
                    f"N={len(qg):3d} | "
                    f"excess median="
                    f"{100*np.median(qg[excess_col]):7.2f}% | "
                    f"improvement median="
                    f"{100*np.median(qg[ycol]):+7.3f}% | "
                    f"improved="
                    f"{100*np.mean(qg[ycol] > 0):5.1f}%"
                )
        except ValueError:
            emit("  Dose-response quartiles unavailable")

        emit()

    # --------------------------------------------------------
    # Taskwise check
    # --------------------------------------------------------
    emit("TASKWISE")

    for task, tg in g.groupby("task"):
        emit(f"  {task}: N={len(tg)}")

        for metric in primary_metrics:
            ycol = f"{metric}_improvement_frac"

            z = tg[
                np.isfinite(tg[excess_col])
                & np.isfinite(tg[ycol])
            ]

            if len(z) < 5:
                emit(f"    {metric}: too few samples")
                continue

            rho, p = spearmanr(
                z[excess_col],
                z[ycol],
            )

            emit(
                f"    {metric}: "
                f"rho={rho:+.4f}, "
                f"p={p:.6g}, "
                f"median improvement="
                f"{100*np.median(z[ycol]):+.3f}%"
            )

    emit()

# ------------------------------------------------------------
# Save per-sample table + text summary
# ------------------------------------------------------------
df.to_csv(
    OUTDIR / "pointcloud_dose_response_per_sample.csv",
    index=False,
)

with open(
    OUTDIR / "pointcloud_dose_response_summary.txt",
    "w",
) as f:
    f.write("\n".join(lines) + "\n")

print("=" * 92)
print("saved =", OUTDIR)
print("SUCCESS = True")
