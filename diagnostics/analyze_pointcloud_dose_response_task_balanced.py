#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

IN = (
    ROOT / "diagnostics/results/"
    "goal_preserved_pointcloud_dose_response/"
    "pointcloud_dose_response_per_sample.csv"
)

OUTDIR = (
    ROOT / "diagnostics/results/"
    "goal_preserved_pointcloud_dose_response/"
    "task_balanced"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 20260906
N_PERM = 10000

df = pd.read_csv(IN)

TASKS = ["task-0000", "task-0001", "task-0006"]

df = df[df["task"].isin(TASKS)].copy()

PRIMARY = [
    "rigidity_mean_mm",
    "jitter_mean_mm",
]


def weighted_corr(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
    )

    x = x[good]
    y = y[good]
    w = w[good]

    w = w / w.sum()

    mx = np.sum(w * x)
    my = np.sum(w * y)

    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)

    if vx <= 0 or vy <= 0:
        return np.nan

    cov = np.sum(
        w * (x - mx) * (y - my)
    )

    return float(
        cov / np.sqrt(vx * vy)
    )


def within_task_rank(s):
    return s.rank(
        method="average",
        pct=True,
    )


def balanced_analysis(
    data,
    excess_col,
    improvement_col,
):
    g = data[
        np.isfinite(data[excess_col])
        & np.isfinite(data[improvement_col])
        & (data[excess_col] > 0)
    ].copy()

    # ------------------------------------------
    # Rank independently INSIDE each task.
    # This removes between-task scale differences.
    # ------------------------------------------
    g["x_rank"] = (
        g.groupby("task")[excess_col]
        .transform(within_task_rank)
    )

    g["y_rank"] = (
        g.groupby("task")[improvement_col]
        .transform(within_task_rank)
    )

    # ------------------------------------------
    # Equal task weighting:
    # every task contributes total weight = 1.
    # ------------------------------------------
    counts = g.groupby("task").size()

    g["weight"] = g["task"].map(
        {t: 1.0 / n for t, n in counts.items()}
    )

    rho_bal = weighted_corr(
        g["x_rank"],
        g["y_rank"],
        g["weight"],
    )

    # ------------------------------------------
    # Stratified permutation:
    # shuffle improvement ranks only WITHIN task.
    # Preserves task sizes/distributions.
    # ------------------------------------------
    rng = np.random.default_rng(SEED)

    null = np.empty(
        N_PERM,
        dtype=np.float64,
    )

    task_indices = {
        t: np.where(
            g["task"].to_numpy() == t
        )[0]
        for t in g["task"].unique()
    }

    x = g["x_rank"].to_numpy(float)
    y = g["y_rank"].to_numpy(float)
    w = g["weight"].to_numpy(float)

    for b in range(N_PERM):
        yp = y.copy()

        for t, idx in task_indices.items():
            yp[idx] = rng.permutation(
                yp[idx]
            )

        null[b] = weighted_corr(
            x,
            yp,
            w,
        )

    p_perm = (
        1.0
        + np.sum(
            np.abs(null) >= abs(rho_bal)
        )
    ) / (N_PERM + 1.0)

    # ------------------------------------------
    # Individual task results
    # ------------------------------------------
    task_rows = []

    for task, tg in g.groupby("task"):
        rho, p = spearmanr(
            tg[excess_col],
            tg[improvement_col],
        )

        task_rows.append({
            "task": task,
            "N": len(tg),
            "rho": rho,
            "p": p,
            "median_excess_pct":
                100.0 * np.median(
                    tg[excess_col]
                ),
            "median_improvement_pct":
                100.0 * np.median(
                    tg[improvement_col]
                ),
            "improved_fraction":
                np.mean(
                    tg[improvement_col] > 0
                ),
        })

    return (
        g,
        rho_bal,
        p_perm,
        pd.DataFrame(task_rows),
    )


summary_rows = []

print("=" * 100)
print("TASK-BALANCED WITHIN-TASK DOSE RESPONSE")
print("=" * 100)

for axis, excess_col in [
    ("ACCELERATION", "acc_excess"),
    ("MSJ", "msj_excess"),
]:

    print()
    print("=" * 100)
    print(axis)
    print("=" * 100)

    for metric in PRIMARY:

        improvement_col = (
            f"{metric}_improvement_frac"
        )

        g, rho, p, task_df = balanced_analysis(
            df,
            excess_col,
            improvement_col,
        )

        print()
        print(metric)
        print(
            f"  Task-balanced within-task rank rho "
            f"= {rho:+.4f}"
        )
        print(
            f"  Stratified permutation p "
            f"= {p:.6g}"
        )
        print(
            f"  N = {len(g)} "
            f"across {g['task'].nunique()} tasks"
        )

        print("  TASKWISE:")

        for _, r in task_df.iterrows():
            print(
                f"    {r['task']} | "
                f"N={int(r['N']):3d} | "
                f"rho={r['rho']:+.4f} | "
                f"p={r['p']:.6g} | "
                f"median improve="
                f"{r['median_improvement_pct']:+.3f}%"
            )

        summary_rows.append({
            "axis": axis,
            "metric": metric,
            "N": len(g),
            "task_balanced_rho": rho,
            "permutation_p": p,
        })

        task_df.to_csv(
            OUTDIR /
            f"{axis.lower()}__{metric}__taskwise.csv",
            index=False,
        )

pd.DataFrame(
    summary_rows
).to_csv(
    OUTDIR / "task_balanced_summary.csv",
    index=False,
)

print()
print("=" * 100)
print("saved =", OUTDIR)
print("SUCCESS = True")
