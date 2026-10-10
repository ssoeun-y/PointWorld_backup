from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]

IN = (
    ROOT
    / "diagnostics/results/acc_intervention_dose_response_onepass/"
      "task0006_0008_0010__acc_intervention_dose_response_ONEPASS.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/acc_intervention_dose_response_onepass"
)

d = pd.read_csv(IN)

DOSES = [5, 10, 15, 20]

# ------------------------------------------------------------
# Fair population = all 4 intervention magnitudes matched
# ------------------------------------------------------------
mask = np.ones(len(d), dtype=bool)

for dose in DOSES:
    mask &= d[f"acc_{dose:02d}_matched"].astype(bool)

q = d[mask].copy()

print("=" * 100)
print("ACC INTERVENTION DOSE-RESPONSE")
print("=" * 100)

print("TOTAL =", len(d))
print("ALL-4-DOSE MATCHED =", len(q))

print("\nBY TASK:")
print(q["task"].value_counts().sort_index())

# ------------------------------------------------------------
# Global dose response
# ------------------------------------------------------------
rows = []

for dose in DOSES:
    col = f"acc_{dose:02d}_prediction_shift_mean_mm"

    rows.append({
        "dose_pct_of_boundary": dose,
        "n": int(q[col].notna().sum()),
        "median_shift_mm": float(q[col].median()),
        "mean_shift_mm": float(q[col].mean()),
        "p95_shift_mm": float(q[col].quantile(.95)),
    })

summary = pd.DataFrame(rows)

base = summary.loc[
    summary["dose_pct_of_boundary"] == 5,
    "median_shift_mm",
].iloc[0]

summary["median_ratio_vs_5pct"] = (
    summary["median_shift_mm"] / base
)

print("\nGLOBAL:")
print(summary.to_string(index=False))

# ------------------------------------------------------------
# Within-sample monotonicity
# ------------------------------------------------------------
vals = q[
    [
        f"acc_{dose:02d}_prediction_shift_mean_mm"
        for dose in DOSES
    ]
].to_numpy(float)

strict_monotonic = np.all(
    np.diff(vals, axis=1) > 0,
    axis=1,
)

nondecreasing = np.all(
    np.diff(vals, axis=1) >= 0,
    axis=1,
)

print("\nWITHIN-SAMPLE MONOTONICITY")
print(
    "strict increasing =",
    f"{strict_monotonic.mean()*100:.2f}%"
)
print(
    "non-decreasing =",
    f"{nondecreasing.mean()*100:.2f}%"
)

# ------------------------------------------------------------
# Pooled dose ↔ prediction shift association
# ------------------------------------------------------------
long_rows = []

for dose in DOSES:
    col = f"acc_{dose:02d}_prediction_shift_mean_mm"

    for v in q[col].to_numpy(float):
        long_rows.append((dose, v))

long = pd.DataFrame(
    long_rows,
    columns=["dose", "shift_mm"],
)

rho, p = spearmanr(
    long["dose"],
    long["shift_mm"],
)

print("\nPOOLED DOSE RESPONSE")
print("Spearman rho =", rho)
print("p =", p)

# ------------------------------------------------------------
# Paired adjacent-dose tests
# ------------------------------------------------------------
pair_rows = []

for lo, hi in zip(DOSES[:-1], DOSES[1:]):

    a = q[
        f"acc_{lo:02d}_prediction_shift_mean_mm"
    ].to_numpy(float)

    b = q[
        f"acc_{hi:02d}_prediction_shift_mean_mm"
    ].to_numpy(float)

    stat, pv = wilcoxon(
        b,
        a,
        alternative="greater",
    )

    pair_rows.append({
        "comparison": f"{hi}% > {lo}%",
        "median_lo_mm": float(np.median(a)),
        "median_hi_mm": float(np.median(b)),
        "median_delta_mm":
            float(np.median(b - a)),
        "fraction_hi_gt_lo":
            float(np.mean(b > a)),
        "wilcoxon_p": float(pv),
    })

pairs = pd.DataFrame(pair_rows)

print("\nPAIRED ADJACENT DOSES:")
print(pairs.to_string(index=False))

# ------------------------------------------------------------
# Task-wise
# ------------------------------------------------------------
task_rows = []

for task, g in q.groupby("task"):

    for dose in DOSES:
        col = (
            f"acc_{dose:02d}_prediction_shift_mean_mm"
        )

        task_rows.append({
            "task": task,
            "dose_pct_of_boundary": dose,
            "n": len(g),
            "median_shift_mm":
                float(g[col].median()),
        })

taskdf = pd.DataFrame(task_rows)

print("\nTASK-WISE:")
print(taskdf.to_string(index=False))

summary.to_csv(
    OUTDIR / "dose_response_summary.csv",
    index=False,
)

pairs.to_csv(
    OUTDIR / "dose_response_paired_tests.csv",
    index=False,
)

taskdf.to_csv(
    OUTDIR / "dose_response_by_task.csv",
    index=False,
)

print("\nSUCCESS = True")
