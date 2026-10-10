#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

ROOT = Path("diagnostics/results/acc_only_pa_intervention_5task_current10457")
OUT = ROOT / "analysis_5task"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["0000","0001","0006","0008","0010"]
DOSES = [5,10,15,20]

# ------------------------------------------------------------
# Load successful paired clip rows
# ------------------------------------------------------------
dfs = []
status_dfs = []

for t in TASKS:
    p = ROOT / f"task-{t}" / "per_clip_intervention.csv"
    s = ROOT / f"task-{t}" / "clip_status.csv"

    d = pd.read_csv(p)
    d["task"] = f"task-{t}"
    dfs.append(d)

    st = pd.read_csv(s)
    st["task"] = f"task-{t}"
    status_dfs.append(st)

df = pd.concat(dfs, ignore_index=True)
status = pd.concat(status_dfs, ignore_index=True)

# Average the two identical baseline inputs to reduce stochastic baseline noise.
df["baseline_ref_pa_mm"] = (
    df["base_pa_mean_mm"] +
    df["base_repeat_pa_mean_mm"]
) / 2.0

# ------------------------------------------------------------
# Matching / usable counts
# ------------------------------------------------------------
print("="*100)
print("POPULATION / MATCHING")
print("="*100)

for task, g in status.groupby("task"):
    total = len(g)
    matched = int(g["all_doses_matched"].sum())
    inferred = int(g["inferred"].sum())
    usable = int((df["task"] == task).sum())

    print(
        f"{task}: total={total}, "
        f"all-dose matched={matched} ({matched/total*100:.2f}%), "
        f"inferred={inferred}, usable PA clips={usable}"
    )

print(
    f"ALL: total={len(status)}, "
    f"all-dose matched={int(status['all_doses_matched'].sum())} "
    f"({status['all_doses_matched'].mean()*100:.2f}%), "
    f"usable PA clips={len(df)}"
)

# ------------------------------------------------------------
# Actual achieved Acc dose audit
# ------------------------------------------------------------
print("\n" + "="*100)
print("ACHIEVED RMS ACC DOSE")
print("="*100)

for dose in DOSES:
    c = f"acc{dose:02d}_actual_acc_si_m_s2"

    achieved = (
        df[c] / df["base_acc_si_m_s2"] - 1.0
    ) * 100.0

    print(
        f"+{dose}% requested: "
        f"median achieved={np.median(achieved):.3f}% "
        f"[p5={np.percentile(achieved,5):.3f}, "
        f"p95={np.percentile(achieved,95):.3f}]"
    )

# ------------------------------------------------------------
# Primary paired dose response
# ------------------------------------------------------------
rows = []

print("\n" + "="*100)
print("POOLED PA-MPJPE DOSE RESPONSE")
print("="*100)

base = df["baseline_ref_pa_mm"].to_numpy()

print(
    f"Baseline PA-MPJPE: "
    f"median={np.median(base):.6f} mm, "
    f"mean={np.mean(base):.6f} mm"
)

for dose in DOSES:
    y = df[f"acc{dose:02d}_pa_mean_mm"].to_numpy()
    delta = y - base
    ratio = y / base

    try:
        w = wilcoxon(y, base, alternative="two-sided")
        p = float(w.pvalue)
    except Exception:
        p = np.nan

    row = {
        "dose_percent": dose,
        "N": len(df),
        "baseline_median_mm": np.median(base),
        "dose_median_mm": np.median(y),
        "median_delta_mm": np.median(delta),
        "mean_delta_mm": np.mean(delta),
        "median_ratio": np.median(ratio),
        "positive_fraction": np.mean(delta > 0),
        "wilcoxon_p": p,
    }
    rows.append(row)

    print(
        f"+{dose:02d}%: "
        f"PA median={np.median(y):.6f} mm | "
        f"Δmedian={np.median(delta):+.6f} mm | "
        f"ratio={np.median(ratio):.4f}x | "
        f"PA increased in {np.mean(delta>0)*100:.2f}% clips | "
        f"Wilcoxon p={p:.3e}"
    )

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "pooled_dose_response.csv", index=False)

# ------------------------------------------------------------
# Stochastic baseline floor
# ------------------------------------------------------------
noise = np.abs(
    df["base_repeat_pa_mean_mm"] -
    df["base_pa_mean_mm"]
).to_numpy()

print("\n" + "="*100)
print("MODEL STOCHASTIC BASELINE FLOOR")
print("="*100)

print(
    f"Identical-input baseline vs repeat absolute difference: "
    f"median={np.median(noise):.6f} mm, "
    f"p95={np.percentile(noise,95):.6f} mm"
)

for dose in DOSES:
    y = df[f"acc{dose:02d}_pa_mean_mm"].to_numpy()
    delta = y - base

    print(
        f"+{dose:02d}%: "
        f"|intervention Δ| median={np.median(np.abs(delta)):.6f} mm, "
        f"|Δ| > baseline-repeat noise in "
        f"{np.mean(np.abs(delta) > noise)*100:.2f}% clips"
    )

# ------------------------------------------------------------
# Per-clip monotonic dose response
# ------------------------------------------------------------
Y = np.column_stack([
    df["baseline_ref_pa_mm"].to_numpy(),
    df["acc05_pa_mean_mm"].to_numpy(),
    df["acc10_pa_mean_mm"].to_numpy(),
    df["acc15_pa_mean_mm"].to_numpy(),
    df["acc20_pa_mean_mm"].to_numpy(),
])

x = np.array([0,5,10,15,20], dtype=float)

nondecreasing = np.all(
    np.diff(Y, axis=1) >= 0,
    axis=1,
)

slopes = []
rhos = []

for yy in Y:
    slopes.append(np.polyfit(x, yy, 1)[0])
    rhos.append(spearmanr(x, yy).statistic)

slopes = np.asarray(slopes)
rhos = np.asarray(rhos)

print("\n" + "="*100)
print("WITHIN-CLIP DOSE TREND")
print("="*100)

print(
    f"Strict non-decreasing PA across 0/5/10/15/20%: "
    f"{nondecreasing.mean()*100:.2f}%"
)
print(
    f"Positive fitted PA-vs-dose slope: "
    f"{np.mean(slopes > 0)*100:.2f}%"
)
print(
    f"Median slope: {np.median(slopes):+.6f} mm per 1% Acc"
)
print(
    f"Median within-clip Spearman(dose, PA): "
    f"{np.nanmedian(rhos):+.4f}"
)

# ------------------------------------------------------------
# Task-wise
# ------------------------------------------------------------
task_rows = []

print("\n" + "="*100)
print("TASK-WISE")
print("="*100)

for task, g in df.groupby("task"):
    b = g["baseline_ref_pa_mm"].to_numpy()

    print(f"\n{task} N={len(g)}")

    for dose in DOSES:
        y = g[f"acc{dose:02d}_pa_mean_mm"].to_numpy()
        delta = y - b

        task_rows.append({
            "task": task,
            "dose_percent": dose,
            "N": len(g),
            "baseline_median_mm": np.median(b),
            "dose_median_mm": np.median(y),
            "median_delta_mm": np.median(delta),
            "positive_fraction": np.mean(delta > 0),
        })

        print(
            f"  +{dose:02d}%: "
            f"{np.median(b):.4f} -> {np.median(y):.4f} mm, "
            f"Δ={np.median(delta):+.4f} mm, "
            f"increased={np.mean(delta>0)*100:.1f}%"
        )

pd.DataFrame(task_rows).to_csv(
    OUT / "taskwise_dose_response.csv",
    index=False,
)

df.to_csv(
    OUT / "merged_per_clip_intervention.csv",
    index=False,
)

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("N =", len(df))
print("SAVED =", OUT)
print("="*100)
