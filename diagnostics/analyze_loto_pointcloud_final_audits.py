#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]

PC_ROOT = (
    ROOT / "diagnostics/results/"
    "goal_preserved_loto_pointcloud_structure"
)

TARGET_CSV = (
    ROOT / "diagnostics/results/"
    "action_working_range_loto_5task/"
    "loto_correction_targets_all.csv"
)

OUTDIR = (
    ROOT / "diagnostics/results/"
    "goal_preserved_loto_pointcloud_structure/"
    "final_audits"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

TASK_FILES = [
    "task0000",
    "task0001",
    "task0006",
    "task0008",
    "task0010",
]

pc = pd.concat(
    [
        pd.read_csv(
            PC_ROOT / f"{t}__pointcloud_structure.csv"
        )
        for t in TASK_FILES
    ],
    ignore_index=True,
)

assert len(pc) == 2846, f"Expected 2846, got {len(pc)}"

targets = pd.read_csv(TARGET_CSV)

keep = [
    "clip",
    "heldout_task",
    "rms_acceleration",
    "mean_squared_jerk",
    "loto_acc_boundary",
    "loto_msj_boundary",
]

targets = targets[keep].copy()

targets = targets.rename(
    columns={"heldout_task": "task"}
)

df = pc.merge(
    targets,
    on=["clip", "task"],
    how="left",
    validate="one_to_one",
)

if df[
    [
        "rms_acceleration",
        "mean_squared_jerk",
        "loto_acc_boundary",
        "loto_msj_boundary",
    ]
].isna().any().any():
    raise RuntimeError("Missing LOTO metric/boundary after merge")

# ============================================================
# Helpers
# ============================================================

def cliff_delta(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    u = mannwhitneyu(
        x,
        y,
        alternative="two-sided",
    ).statistic

    return (
        2.0 * u / (len(x) * len(y))
        - 1.0
    )


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

    x, y, w = x[good], y[good], w[good]

    w = w / w.sum()

    mx = np.sum(w * x)
    my = np.sum(w * y)

    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)

    if vx <= 0 or vy <= 0:
        return np.nan

    return float(
        np.sum(w * (x-mx) * (y-my))
        / np.sqrt(vx * vy)
    )


# ============================================================
# 1. INSUFFICIENT-GRAPH AUDIT
# ============================================================

print("=" * 105)
print("1. LOTO STRUCTURAL COVERAGE / SELECTION-BIAS AUDIT")
print("=" * 105)

ok = df[df["status"] == "OK"].copy()
bad = df[
    df["status"] == "INSUFFICIENT_GRAPH"
].copy()

print(
    f"OK = {len(ok)} / {len(df)} "
    f"({100*len(ok)/len(df):.2f}%)"
)

print(
    f"INSUFFICIENT = {len(bad)} / {len(df)} "
    f"({100*len(bad)/len(df):.2f}%)"
)

print()
print("BY TASK")
print(pd.crosstab(df["task"], df["status"]))

print()
print("SUPPORT POINTS IN EXCLUDED CASES")
print(
    bad.groupby("task")["num_support_points"]
    .agg(["count", "min", "median", "max"])
)

print()
print(
    "Excluded with <= 8 support points =",
    int((bad["num_support_points"] <= 8).sum()),
    "/",
    len(bad),
)

for metric in [
    "rms_acceleration",
    "mean_squared_jerk",
]:

    a = bad[metric].to_numpy(float)
    b = ok[metric].to_numpy(float)

    mw = mannwhitneyu(
        a,
        b,
        alternative="two-sided",
    )

    d = cliff_delta(a, b)

    print()
    print(metric)
    print(
        "  OK median          =",
        np.median(b),
    )
    print(
        "  EXCLUDED median    =",
        np.median(a),
    )
    print(
        "  excluded/OK ratio  =",
        np.median(a) / np.median(b),
    )
    print(
        "  Mann-Whitney p     =",
        mw.pvalue,
    )
    print(
        "  Cliff delta        =",
        f"{d:+.4f}",
    )

    # Within-task rank control.
    rank_col = metric + "__rank"

    df[rank_col] = (
        df.groupby("task")[metric]
        .rank(method="average", pct=True)
    )

    ar = df.loc[
        df["status"] == "INSUFFICIENT_GRAPH",
        rank_col,
    ].to_numpy(float)

    br = df.loc[
        df["status"] == "OK",
        rank_col,
    ].to_numpy(float)

    mwr = mannwhitneyu(
        ar,
        br,
        alternative="two-sided",
    )

    dr = cliff_delta(ar, br)

    print(
        "  within-task rank p =",
        mwr.pvalue,
    )
    print(
        "  within-task delta  =",
        f"{dr:+.4f}",
    )


# ============================================================
# 2. LOTO DOSE RESPONSE
# ============================================================

print()
print("=" * 105)
print("2. 5-TASK LOTO BOUNDARY EXCESS → POINT-CLOUD IMPROVEMENT")
print("=" * 105)

g = ok.copy()

g["acc_excess"] = (
    g["rms_acceleration"]
    / g["loto_acc_boundary"]
    - 1.0
)

g["msj_excess"] = (
    g["mean_squared_jerk"]
    / g["loto_msj_boundary"]
    - 1.0
)

for m in [
    "rigidity_mean_mm",
    "jitter_mean_mm",
    "jitter_p95_mm",
]:
    g[f"{m}_improvement"] = (
        1.0
        -
        g[f"corrected_{m}"]
        /
        np.maximum(
            g[f"original_{m}"],
            1e-12,
        )
    )


rng = np.random.default_rng(20260907)
N_PERM = 10000

summary_rows = []

for axis, excess_col in [
    ("ACCELERATION", "acc_excess"),
    ("MSJ", "msj_excess"),
]:

    zaxis = g[
        np.isfinite(g[excess_col])
        & (g[excess_col] > 0)
    ].copy()

    print()
    print("=" * 105)
    print(axis)
    print("=" * 105)
    print("violating N =", len(zaxis))
    print(
        "median excess =",
        f"{100*np.median(zaxis[excess_col]):.2f}%",
    )

    for metric in [
        "rigidity_mean_mm",
        "jitter_mean_mm",
        "jitter_p95_mm",
    ]:

        ycol = f"{metric}_improvement"

        z = zaxis[
            np.isfinite(zaxis[ycol])
        ].copy()

        rho, p = spearmanr(
            z[excess_col],
            z[ycol],
        )

        print()
        print(metric)
        print(
            f"  GLOBAL Spearman rho={rho:+.4f}, "
            f"p={p:.6g}, N={len(z)}"
        )
        print(
            f"  median improvement="
            f"{100*np.median(z[ycol]):+.3f}%"
        )

        # ----------------------------------------------
        # Quartile dose response
        # ----------------------------------------------
        z["severity_bin"] = pd.qcut(
            z[excess_col],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )

        print("  DOSE RESPONSE")

        for q, qg in z.groupby(
            "severity_bin",
            observed=True,
        ):
            print(
                f"    {q}: "
                f"N={len(qg):4d} | "
                f"excess="
                f"{100*np.median(qg[excess_col]):7.2f}% | "
                f"improve="
                f"{100*np.median(qg[ycol]):+7.3f}% | "
                f"better="
                f"{100*np.mean(qg[ycol] > 0):5.1f}%"
            )

        # ----------------------------------------------
        # Taskwise
        # ----------------------------------------------
        print("  TASKWISE")

        for task, tg in z.groupby("task"):

            rr, pp = spearmanr(
                tg[excess_col],
                tg[ycol],
            )

            print(
                f"    {task}: "
                f"N={len(tg):4d} | "
                f"rho={rr:+.4f} | "
                f"p={pp:.4g} | "
                f"median improve="
                f"{100*np.median(tg[ycol]):+.2f}%"
            )

        # ----------------------------------------------
        # Task-balanced within-task rank analysis
        # ----------------------------------------------
        z["x_rank"] = (
            z.groupby("task")[excess_col]
            .rank(method="average", pct=True)
        )

        z["y_rank"] = (
            z.groupby("task")[ycol]
            .rank(method="average", pct=True)
        )

        counts = z.groupby("task").size()

        z["weight"] = z["task"].map(
            {
                task: 1.0 / n
                for task, n in counts.items()
            }
        )

        rho_bal = weighted_corr(
            z["x_rank"],
            z["y_rank"],
            z["weight"],
        )

        x = z["x_rank"].to_numpy(float)
        y = z["y_rank"].to_numpy(float)
        w = z["weight"].to_numpy(float)
        tasks = z["task"].to_numpy()

        indices = {
            task: np.where(tasks == task)[0]
            for task in np.unique(tasks)
        }

        null = np.empty(N_PERM)

        for b in range(N_PERM):

            yp = y.copy()

            for task, idx in indices.items():
                yp[idx] = rng.permutation(
                    yp[idx]
                )

            null[b] = weighted_corr(
                x,
                yp,
                w,
            )

        p_perm = (
            1
            + np.sum(
                np.abs(null)
                >= abs(rho_bal)
            )
        ) / (N_PERM + 1)

        print(
            f"  TASK-BALANCED rho={rho_bal:+.4f}, "
            f"permutation p={p_perm:.6g}"
        )

        summary_rows.append({
            "axis": axis,
            "metric": metric,
            "N": len(z),
            "global_rho": rho,
            "global_p": p,
            "task_balanced_rho": rho_bal,
            "task_balanced_perm_p": p_perm,
        })


pd.DataFrame(summary_rows).to_csv(
    OUTDIR / "loto_dose_response_summary.csv",
    index=False,
)

df.to_csv(
    OUTDIR / "loto_structural_coverage_audit.csv",
    index=False,
)

print()
print("=" * 105)
print("saved =", OUTDIR)
print("SUCCESS = True")
