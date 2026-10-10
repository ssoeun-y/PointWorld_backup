from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(
    "diagnostics/results/object_pa_mpjpe_acc_5task_current10457"
)

TASKS = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]

OUT = ROOT / "analysis_5task"
OUT.mkdir(parents=True, exist_ok=True)


def find_col(df, candidates, contains=None):
    for c in candidates:
        if c in df.columns:
            return c

    if contains:
        for c in df.columns:
            lc = c.lower()
            if all(x.lower() in lc for x in contains):
                return c

    raise RuntimeError(
        f"Could not find column.\n"
        f"Candidates={candidates}\n"
        f"Columns={list(df.columns)}"
    )


def cliffs_delta(x, y):
    """
    delta > 0 means values in x tend to be larger than y.
    Here x = HIGH Acc, y = LOW Acc.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    gt = 0
    lt = 0

    for v in x:
        gt += np.sum(v > y)
        lt += np.sum(v < y)

    return (gt - lt) / (len(x) * len(y))


frames = []

for task in TASKS:
    f = ROOT / task / "per_clip_metrics.csv"

    if not f.exists():
        raise FileNotFoundError(f)

    df = pd.read_csv(f)

    if "task" not in df.columns:
        df["task"] = task

    frames.append(df)

    print(
        task,
        "per_clip rows =",
        len(df),
    )

merged = pd.concat(
    frames,
    ignore_index=True,
)

print("\nMERGED COLUMNS:")
for c in merged.columns:
    print(" ", c)

clip_col = find_col(
    merged,
    ["clip", "clip_key", "__key__"],
    contains=["clip"],
)

acc_col = find_col(
    merged,
    [
        "rms_acceleration_si_m_s2",
        "rms_acc_si_m_s2",
        "rms_acceleration",
    ],
    contains=["rms", "acc"],
)

pa_col = find_col(
    merged,
    [
        "pa_mpjpe_mm",
        "mean_pa_mpjpe_mm",
        "clip_pa_mpjpe_mm",
        "pa_mpjpe",
    ],
    contains=["pa", "mpjpe"],
)

# Must exclude PA column while finding raw MPJPE.
mpjpe_candidates = [
    "mpjpe_mm",
    "mean_mpjpe_mm",
    "clip_mpjpe_mm",
    "mpjpe",
]

mpjpe_col = None

for c in mpjpe_candidates:
    if c in merged.columns:
        mpjpe_col = c
        break

if mpjpe_col is None:
    for c in merged.columns:
        lc = c.lower()
        if "mpjpe" in lc and "pa" not in lc:
            mpjpe_col = c
            break

if mpjpe_col is None:
    raise RuntimeError(
        f"Could not find MPJPE column. "
        f"Columns={list(merged.columns)}"
    )


print("\nSELECTED COLUMNS")
print("clip  =", clip_col)
print("Acc   =", acc_col)
print("MPJPE =", mpjpe_col)
print("PA    =", pa_col)


for c in [acc_col, mpjpe_col, pa_col]:
    merged[c] = pd.to_numeric(
        merged[c],
        errors="coerce",
    )

clean = merged.dropna(
    subset=[
        "task",
        clip_col,
        acc_col,
        mpjpe_col,
        pa_col,
    ]
).copy()

if clean[clip_col].duplicated().any():
    dups = clean.loc[
        clean[clip_col].duplicated(),
        clip_col,
    ].head(20).tolist()

    raise RuntimeError(
        f"Duplicate clip rows found: {dups}"
    )


merged.to_csv(
    OUT / "merged_per_clip_metrics_5task.csv",
    index=False,
)

clean.to_csv(
    OUT / "merged_per_clip_metrics_5task_clean.csv",
    index=False,
)


# ============================================================
# 1. BASIC COUNTS
# ============================================================

counts = (
    clean.groupby("task")
    .size()
    .rename("eligible_clips")
    .reset_index()
)

counts.loc[len(counts)] = [
    "ALL",
    len(clean),
]

counts.to_csv(
    OUT / "eligible_clip_counts.csv",
    index=False,
)


# ============================================================
# 2. SPEARMAN CORRELATION
#    Acc vs PA-MPJPE
#    Acc vs MPJPE
# ============================================================

corr_rows = []

for task in TASKS + ["ALL"]:

    if task == "ALL":
        g = clean
    else:
        g = clean[
            clean["task"] == task
        ]

    for metric_name, metric_col in [
        ("PA-MPJPE", pa_col),
        ("MPJPE", mpjpe_col),
    ]:

        rho, p = spearmanr(
            g[acc_col],
            g[metric_col],
        )

        corr_rows.append({
            "task": task,
            "metric": metric_name,
            "N": len(g),
            "spearman_rho": rho,
            "p_value": p,
            "median_acc": g[acc_col].median(),
            "median_metric_mm": g[metric_col].median(),
        })

corr_df = pd.DataFrame(corr_rows)

corr_df.to_csv(
    OUT / "spearman_acc_vs_shape.csv",
    index=False,
)


# ============================================================
# 3. WITHIN-TASK RANK CONTROL
#
# Removes simple between-task scale differences.
# Rank Acc and error separately WITHIN each task,
# then pool ranks.
# ============================================================

ranked = clean.copy()

ranked["acc_rank_within_task"] = (
    ranked.groupby("task")[acc_col]
    .rank(pct=True, method="average")
)

ranked["pa_rank_within_task"] = (
    ranked.groupby("task")[pa_col]
    .rank(pct=True, method="average")
)

ranked["mpjpe_rank_within_task"] = (
    ranked.groupby("task")[mpjpe_col]
    .rank(pct=True, method="average")
)

rho_pa_rank, p_pa_rank = spearmanr(
    ranked["acc_rank_within_task"],
    ranked["pa_rank_within_task"],
)

rho_mp_rank, p_mp_rank = spearmanr(
    ranked["acc_rank_within_task"],
    ranked["mpjpe_rank_within_task"],
)

rank_control = pd.DataFrame([
    {
        "metric": "PA-MPJPE",
        "N": len(ranked),
        "within_task_rank_spearman_rho": rho_pa_rank,
        "p_value": p_pa_rank,
    },
    {
        "metric": "MPJPE",
        "N": len(ranked),
        "within_task_rank_spearman_rho": rho_mp_rank,
        "p_value": p_mp_rank,
    },
])

rank_control.to_csv(
    OUT / "within_task_rank_control.csv",
    index=False,
)


# ============================================================
# 4. ACC LOW Q1 vs HIGH Q4
#
# Quartiles are defined separately INSIDE EACH TASK.
# This prevents a task with naturally larger Acc
# from dominating the comparison.
# ============================================================

quartile_rows = []

pooled_low = {
    "PA-MPJPE": [],
    "MPJPE": [],
}

pooled_high = {
    "PA-MPJPE": [],
    "MPJPE": [],
}

for task in TASKS:

    g = clean[
        clean["task"] == task
    ].copy()

    q25 = g[acc_col].quantile(0.25)
    q75 = g[acc_col].quantile(0.75)

    low = g[
        g[acc_col] <= q25
    ]

    high = g[
        g[acc_col] >= q75
    ]

    for metric_name, metric_col in [
        ("PA-MPJPE", pa_col),
        ("MPJPE", mpjpe_col),
    ]:

        lo = low[metric_col].to_numpy(float)
        hi = high[metric_col].to_numpy(float)

        med_lo = np.median(lo)
        med_hi = np.median(hi)

        ratio = (
            med_hi / med_lo
            if med_lo != 0
            else np.nan
        )

        u, p = mannwhitneyu(
            hi,
            lo,
            alternative="two-sided",
        )

        delta = cliffs_delta(
            hi,
            lo,
        )

        quartile_rows.append({
            "task": task,
            "metric": metric_name,
            "N_low_Q1": len(lo),
            "N_high_Q4": len(hi),
            "acc_q25": q25,
            "acc_q75": q75,
            "low_median_mm": med_lo,
            "high_median_mm": med_hi,
            "high_over_low_ratio": ratio,
            "percent_change":
                (ratio - 1.0) * 100.0,
            "mannwhitney_p": p,
            "cliffs_delta_high_vs_low": delta,
        })

        pooled_low[metric_name].append(lo)
        pooled_high[metric_name].append(hi)


# Pooled Q1/Q4 after task-wise quartile assignment.
for metric_name in [
    "PA-MPJPE",
    "MPJPE",
]:

    lo = np.concatenate(
        pooled_low[metric_name]
    )

    hi = np.concatenate(
        pooled_high[metric_name]
    )

    med_lo = np.median(lo)
    med_hi = np.median(hi)

    ratio = (
        med_hi / med_lo
        if med_lo != 0
        else np.nan
    )

    u, p = mannwhitneyu(
        hi,
        lo,
        alternative="two-sided",
    )

    delta = cliffs_delta(
        hi,
        lo,
    )

    quartile_rows.append({
        "task": "ALL_taskwise_quartiles",
        "metric": metric_name,
        "N_low_Q1": len(lo),
        "N_high_Q4": len(hi),
        "acc_q25": np.nan,
        "acc_q75": np.nan,
        "low_median_mm": med_lo,
        "high_median_mm": med_hi,
        "high_over_low_ratio": ratio,
        "percent_change":
            (ratio - 1.0) * 100.0,
        "mannwhitney_p": p,
        "cliffs_delta_high_vs_low": delta,
    })


quartile_df = pd.DataFrame(
    quartile_rows
)

quartile_df.to_csv(
    OUT / "acc_q1_vs_q4_shape_metrics.csv",
    index=False,
)


# ============================================================
# PRINT CORE RESULT
# ============================================================

print("\n" + "=" * 78)
print("5-TASK ACC vs RIGID-OBJECT SHAPE ANALYSIS")
print("=" * 78)

print("\n[ELIGIBLE CLIPS]")
print(
    counts.to_string(index=False)
)

print("\n[SPEARMAN: RMS Acc vs PA-MPJPE / MPJPE]")
print(
    corr_df[
        [
            "task",
            "metric",
            "N",
            "spearman_rho",
            "p_value",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print("\n[WITHIN-TASK RANK CONTROL]")
print(
    rank_control.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print("\n[ACC Q1 vs Q4]")
show = quartile_df[
    [
        "task",
        "metric",
        "N_low_Q1",
        "N_high_Q4",
        "low_median_mm",
        "high_median_mm",
        "high_over_low_ratio",
        "percent_change",
        "mannwhitney_p",
        "cliffs_delta_high_vs_low",
    ]
]

print(
    show.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print("\nSAVED:")
for f in sorted(OUT.glob("*.csv")):
    print(" ", f)

print("=" * 78)
