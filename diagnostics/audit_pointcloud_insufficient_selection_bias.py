#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]

PC_ROOT = (
    ROOT / "diagnostics/results/"
    "goal_preserved_pointcloud_structure"
)

OUTDIR = (
    ROOT / "diagnostics/results/"
    "goal_preserved_pointcloud_structure/"
    "insufficient_graph_audit"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

FILES = [
    PC_ROOT / "task0000__pointcloud_structure.csv",
    PC_ROOT / "task0001__pointcloud_structure.csv",
    PC_ROOT / "task0006__pointcloud_structure.csv",
]

df = pd.concat(
    [pd.read_csv(p) for p in FILES],
    ignore_index=True,
)

est = pd.read_csv(
    ROOT / "diagnostics/results/"
    "established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)[[
    "clip",
    "task",
    "rms_acceleration",
    "mean_squared_jerk",
]]

df = df.merge(
    est,
    on=["clip", "task"],
    how="left",
    validate="one_to_one",
)

if len(df) != 477:
    raise RuntimeError(f"Expected 477 rows, got {len(df)}")

if df[
    ["rms_acceleration", "mean_squared_jerk"]
].isna().any().any():
    raise RuntimeError("Missing established metrics")

df["excluded"] = (
    df["status"] == "INSUFFICIENT_GRAPH"
)

ok = df[~df["excluded"]].copy()
bad = df[df["excluded"]].copy()


def cliffs_delta(x, y):
    """
    Positive delta = first group tends to be larger.
    Here x = excluded, y = included.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    u = mannwhitneyu(
        x, y,
        alternative="two-sided",
        method="auto",
    ).statistic

    return 2.0 * u / (len(x) * len(y)) - 1.0


print("=" * 100)
print("POINT-CLOUD STRUCTURAL COVERAGE SELECTION-BIAS AUDIT")
print("=" * 100)

print()
print("COVERAGE")
print(f"Included     = {len(ok)} / {len(df)} ({100*len(ok)/len(df):.2f}%)")
print(f"Insufficient = {len(bad)} / {len(df)} ({100*len(bad)/len(df):.2f}%)")

print()
print("BY TASK")
print(pd.crosstab(df["task"], df["status"]))

summary = []

for metric in [
    "rms_acceleration",
    "mean_squared_jerk",
]:
    a = bad[metric].to_numpy(float)
    b = ok[metric].to_numpy(float)

    u = mannwhitneyu(
        a,
        b,
        alternative="two-sided",
        method="auto",
    )

    delta = cliffs_delta(a, b)

    print()
    print("=" * 100)
    print(metric)
    print("=" * 100)
    print(f"Included median        = {np.median(b):.10g}")
    print(f"Insufficient median    = {np.median(a):.10g}")
    print(f"Median ratio bad/ok    = {np.median(a)/np.median(b):.6f}")
    print(f"Mann-Whitney p         = {u.pvalue:.6g}")
    print(f"Cliff delta bad-vs-ok  = {delta:+.4f}")

    summary.append({
        "analysis": "global",
        "metric": metric,
        "included_n": len(b),
        "excluded_n": len(a),
        "included_median": np.median(b),
        "excluded_median": np.median(a),
        "ratio": np.median(a)/np.median(b),
        "p": u.pvalue,
        "cliff_delta": delta,
    })

    print()
    print("TASKWISE")

    for task, g in df.groupby("task"):
        gbad = g[g["excluded"]][metric].to_numpy(float)
        gok = g[~g["excluded"]][metric].to_numpy(float)

        if len(gbad) == 0 or len(gok) == 0:
            continue

        uu = mannwhitneyu(
            gbad,
            gok,
            alternative="two-sided",
            method="auto",
        )

        dd = cliffs_delta(gbad, gok)

        print(
            f"  {task}: "
            f"Nbad={len(gbad):2d}, Nok={len(gok):3d} | "
            f"median {np.median(gok):.6g} -> {np.median(gbad):.6g} | "
            f"ratio={np.median(gbad)/np.median(gok):.3f} | "
            f"p={uu.pvalue:.4g} | "
            f"delta={dd:+.3f}"
        )

# ============================================================
# Task-controlled audit
# Convert each metric to percentile rank WITHIN task.
# Then test whether excluded samples systematically occupy
# higher ranks after removing between-task scale differences.
# ============================================================

for metric in [
    "rms_acceleration",
    "mean_squared_jerk",
]:
    rank_col = metric + "__within_task_rank"

    df[rank_col] = (
        df.groupby("task")[metric]
        .rank(method="average", pct=True)
    )

    a = df.loc[df["excluded"], rank_col].to_numpy(float)
    b = df.loc[~df["excluded"], rank_col].to_numpy(float)

    u = mannwhitneyu(
        a,
        b,
        alternative="two-sided",
        method="auto",
    )

    delta = cliffs_delta(a, b)

    print()
    print("=" * 100)
    print(metric, "— WITHIN-TASK RANK CONTROL")
    print("=" * 100)
    print(f"Included median rank       = {np.median(b):.4f}")
    print(f"Insufficient median rank   = {np.median(a):.4f}")
    print(f"Mann-Whitney p             = {u.pvalue:.6g}")
    print(f"Cliff delta bad-vs-ok      = {delta:+.4f}")

    summary.append({
        "analysis": "within_task_rank",
        "metric": metric,
        "included_n": len(b),
        "excluded_n": len(a),
        "included_median": np.median(b),
        "excluded_median": np.median(a),
        "ratio": np.nan,
        "p": u.pvalue,
        "cliff_delta": delta,
    })

pd.DataFrame(summary).to_csv(
    OUTDIR / "selection_bias_summary.csv",
    index=False,
)

df.to_csv(
    OUTDIR / "selection_bias_per_sample.csv",
    index=False,
)

print()
print("=" * 100)
print("saved =", OUTDIR)
print("SUCCESS = True")
