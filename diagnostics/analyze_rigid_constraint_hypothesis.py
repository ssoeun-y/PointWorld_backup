#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]

B = 0.24580418

REPAIR_CSV = (
    ROOT
    / "diagnostics/results/rigid_projection_repair_current10457"
    / "combined_5task_repair_per_clip.csv"
)

ACC_CSV = (
    ROOT
    / "diagnostics/results/current_behavior_established_acc_10457"
    / "per_sample_current_established_acc_10457.csv"
)

PA_ROOT = (
    ROOT
    / "diagnostics/results/object_pa_mpjpe_acc_5task_current10457"
)

OUT = (
    ROOT
    / "diagnostics/results/rigid_constraint_hypothesis"
)
OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]

EXPECTED = {
    "task-0000": 383,
    "task-0001": 2307,
    "task-0006": 1823,
    "task-0008": 1824,
    "task-0010": 1610,
}


def choose_column(
    df,
    exact_candidates,
    required_tokens,
    forbidden_tokens=(),
):
    for c in exact_candidates:
        if c in df.columns:
            return c

    matches = []

    for c in df.columns:
        x = c.lower()

        if not all(
            t.lower() in x
            for t in required_tokens
        ):
            continue

        if any(
            t.lower() in x
            for t in forbidden_tokens
        ):
            continue

        matches.append(c)

    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely identify column.\n"
            f"required={required_tokens}\n"
            f"forbidden={forbidden_tokens}\n"
            f"matches={matches}\n"
            f"all columns={list(df.columns)}"
        )

    return matches[0]


def safe_spearman(x, y):
    m = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())

    r, p = spearmanr(
        x[m],
        y[m],
    )

    return float(r), float(p), int(m.sum())


def within_task_rank_rho(
    df,
    xcol,
    ycol,
):
    q = df[
        ["task", xcol, ycol]
    ].dropna().copy()

    q["xr"] = (
        q.groupby("task")[xcol]
        .rank(
            method="average",
            pct=True,
        )
    )

    q["yr"] = (
        q.groupby("task")[ycol]
        .rank(
            method="average",
            pct=True,
        )
    )

    return safe_spearman(
        q["xr"].to_numpy(float),
        q["yr"].to_numpy(float),
    )


def summarize_group(
    name,
    q,
):
    return {
        "group": name,
        "n": len(q),

        "acc_median":
            float(
                q["rms_acceleration_si_m_s2"]
                .median()
            ),

        "raw_pa_median_mm":
            float(
                q["pa_mpjpe_objectmean_mm"]
                .median()
            ),

        "raw_mpjpe_median_mm":
            float(
                q["raw_constraint_mpjpe_mm"]
                .median()
            ),

        "constraint_mpjpe_median_mm":
            float(
                q["constraint_mpjpe_mm"]
                .median()
            ),

        "median_gain_mm":
            float(
                q["constraint_gain_mm"]
                .median()
            ),

        "median_gain_pct":
            float(
                q["constraint_gain_pct"]
                .median()
            ),

        "improved_pct":
            float(
                100.0
                * (
                    q["constraint_gain_mm"] > 0
                ).mean()
            ),
    }


# ============================================================
# Load rigid-constraint intervention
# ============================================================

repair = pd.read_csv(
    REPAIR_CSV
)

print("=" * 100)
print("REPAIR CSV COLUMNS")
print("=" * 100)
print(list(repair.columns))

raw_col = "raw_mp"
constraint_col = "repaired_mp"

if raw_col not in repair.columns:
    raise RuntimeError(f"Missing {raw_col}")

if constraint_col not in repair.columns:
    raise RuntimeError(f"Missing {constraint_col}")

print()
print("SELECTED RAW MPJPE COLUMN =", raw_col)
print(
    "SELECTED CONSTRAINT MPJPE COLUMN =",
    constraint_col,
)

repair = repair[
    [
        "clip",
        "task",
        raw_col,
        constraint_col,
    ]
].copy()

repair = repair.rename(
    columns={
        raw_col:
            "raw_constraint_mpjpe_mm",

        constraint_col:
            "constraint_mpjpe_mm",
    }
)

if repair["clip"].duplicated().any():
    raise RuntimeError(
        "Repair CSV has duplicate clips"
    )


# ============================================================
# Load established Action RMS Acc
# ============================================================

acc = pd.read_csv(
    ACC_CSV
)

if "rms_acceleration_si_m_s2" not in acc.columns:
    raise RuntimeError(
        "Established SI RMS Acc column missing"
    )

acc = acc[
    [
        "clip",
        "task",
        "rms_acceleration_si_m_s2",
    ]
].copy()

if acc["clip"].duplicated().any():
    raise RuntimeError(
        "ACC CSV has duplicate clips"
    )


# ============================================================
# Load current main PA-MPJPE / MPJPE population
# ============================================================

pa_parts = []

for task in TASKS:

    p = (
        PA_ROOT
        / task
        / "per_clip_metrics.csv"
    )

    if not p.exists():
        raise RuntimeError(
            f"Missing PA file: {p}"
        )

    x = pd.read_csv(p)

    needed = [
        "clip",
        "task",
        "mpjpe_objectmean_mm",
        "pa_mpjpe_objectmean_mm",
    ]

    miss = [
        c for c in needed
        if c not in x.columns
    ]

    if miss:
        raise RuntimeError(
            f"{task}: missing columns {miss}"
        )

    x = x[needed].copy()

    if len(x) != EXPECTED[task]:
        raise RuntimeError(
            f"{task}: PA population "
            f"{len(x)} != {EXPECTED[task]}"
        )

    pa_parts.append(x)

pa = pd.concat(
    pa_parts,
    ignore_index=True,
)

if pa["clip"].duplicated().any():
    raise RuntimeError(
        "PA population has duplicate clips"
    )


# ============================================================
# Merge exact current 7,947 population
# ============================================================

d = (
    pa
    .merge(
        repair,
        on=[
            "clip",
            "task",
        ],
        how="inner",
        validate="one_to_one",
    )
    .merge(
        acc,
        on=[
            "clip",
            "task",
        ],
        how="inner",
        validate="one_to_one",
    )
)

print()
print("=" * 100)
print("POPULATION AUDIT")
print("=" * 100)
print("PA N      =", len(pa))
print("REPAIR N  =", len(repair))
print("MERGED N  =", len(d))

print()
print(
    d["task"]
    .value_counts()
    .sort_index()
)

if len(d) != 7947:
    raise RuntimeError(
        f"Expected exact N=7947, got {len(d)}"
    )


# ============================================================
# Constraint benefit
# Positive = rigid constraint improved real GT MPJPE
# ============================================================

d["constraint_gain_mm"] = (
    d["raw_constraint_mpjpe_mm"]
    - d["constraint_mpjpe_mm"]
)

d["constraint_gain_pct"] = (
    100.0
    * d["constraint_gain_mm"]
    / d["raw_constraint_mpjpe_mm"]
)

d["above_B"] = (
    d["rms_acceleration_si_m_s2"]
    > B
)


# ============================================================
# Test 1:
# Does the rigid constraint improve actual GT MPJPE?
# ============================================================

w = wilcoxon(
    d["raw_constraint_mpjpe_mm"],
    d["constraint_mpjpe_mm"],
    alternative="greater",
)

print()
print("=" * 100)
print("TEST 1 — DOES RIGID CONSTRAINT RECOVER GT MPJPE?")
print("=" * 100)

print(
    "Raw MPJPE median       =",
    float(
        d[
            "raw_constraint_mpjpe_mm"
        ].median()
    ),
    "mm",
)

print(
    "Constraint MPJPE median=",
    float(
        d[
            "constraint_mpjpe_mm"
        ].median()
    ),
    "mm",
)

print(
    "Aggregate median change=",
    100.0
    * (
        d["constraint_mpjpe_mm"].median()
        / d["raw_constraint_mpjpe_mm"].median()
        - 1.0
    ),
    "%",
)

print(
    "Paired median gain     =",
    float(
        d[
            "constraint_gain_mm"
        ].median()
    ),
    "mm",
)

print(
    "Paired median gain %   =",
    float(
        d[
            "constraint_gain_pct"
        ].median()
    ),
    "%",
)

print(
    "Improved clips         =",
    100.0
    * (
        d["constraint_gain_mm"] > 0
    ).mean(),
    "%",
)

print(
    "Wilcoxon p             =",
    float(w.pvalue),
)


# ============================================================
# Test 2:
# If raw shape distortion is the damaging component,
# higher PA-MPJPE should predict larger MPJPE recovery.
# ============================================================

r_pa_gain = safe_spearman(
    d[
        "pa_mpjpe_objectmean_mm"
    ].to_numpy(float),

    d[
        "constraint_gain_mm"
    ].to_numpy(float),
)

r_pa_gain_task = (
    within_task_rank_rho(
        d,
        "pa_mpjpe_objectmean_mm",
        "constraint_gain_mm",
    )
)

print()
print("=" * 100)
print(
    "TEST 2 — DOES RAW PA-MPJPE PREDICT "
    "CONSTRAINT BENEFIT?"
)
print("=" * 100)

print(
    "Pooled Spearman rho =",
    r_pa_gain[0],
    "| p =",
    r_pa_gain[1],
    "| N =",
    r_pa_gain[2],
)

print(
    "Within-task rank rho=",
    r_pa_gain_task[0],
    "| p =",
    r_pa_gain_task[1],
    "| N =",
    r_pa_gain_task[2],
)


# ============================================================
# Test 3:
# Does Action RMS Acc predict larger constraint benefit?
# ============================================================

r_acc_gain = safe_spearman(
    d[
        "rms_acceleration_si_m_s2"
    ].to_numpy(float),

    d[
        "constraint_gain_mm"
    ].to_numpy(float),
)

r_acc_gain_task = (
    within_task_rank_rho(
        d,
        "rms_acceleration_si_m_s2",
        "constraint_gain_mm",
    )
)

print()
print("=" * 100)
print(
    "TEST 3 — DOES HIGH RMS ACC REQUIRE "
    "MORE RIGID CONSTRAINT?"
)
print("=" * 100)

print(
    "Pooled Spearman rho =",
    r_acc_gain[0],
    "| p =",
    r_acc_gain[1],
)

print(
    "Within-task rank rho=",
    r_acc_gain_task[0],
    "| p =",
    r_acc_gain_task[1],
)


# ============================================================
# Boundary stratification
# ============================================================

groups = []

groups.append(
    summarize_group(
        "<=B",
        d[
            d[
                "rms_acceleration_si_m_s2"
            ] <= B
        ],
    )
)

groups.append(
    summarize_group(
        ">B",
        d[
            d[
                "rms_acceleration_si_m_s2"
            ] > B
        ],
    )
)

groups.append(
    summarize_group(
        "1.0–1.5B",
        d[
            (
                d[
                    "rms_acceleration_si_m_s2"
                ] > B
            )
            &
            (
                d[
                    "rms_acceleration_si_m_s2"
                ] < 1.5 * B
            )
        ],
    )
)

groups.append(
    summarize_group(
        "1.5–2.0B",
        d[
            (
                d[
                    "rms_acceleration_si_m_s2"
                ] >= 1.5 * B
            )
            &
            (
                d[
                    "rms_acceleration_si_m_s2"
                ] < 2.0 * B
            )
        ],
    )
)

groups.append(
    summarize_group(
        ">=2B",
        d[
            d[
                "rms_acceleration_si_m_s2"
            ] >= 2.0 * B
        ],
    )
)

gdf = pd.DataFrame(groups)

print()
print("=" * 100)
print("BOUNDARY STRATIFICATION")
print("=" * 100)
print(
    gdf.to_string(
        index=False
    )
)


# ============================================================
# Taskwise low/top quartile
# Same convention as current PA analysis:
# within each task.
# ============================================================

quartile_rows = []

for task, q in d.groupby("task"):

    q = q.copy()

    lo = float(
        q[
            "rms_acceleration_si_m_s2"
        ].quantile(0.25)
    )

    hi = float(
        q[
            "rms_acceleration_si_m_s2"
        ].quantile(0.75)
    )

    low = q[
        q[
            "rms_acceleration_si_m_s2"
        ] <= lo
    ]

    high = q[
        q[
            "rms_acceleration_si_m_s2"
        ] >= hi
    ]

    for label, z in [
        ("LOW_Q25", low),
        ("HIGH_Q25", high),
    ]:

        row = summarize_group(
            label,
            z,
        )

        row["task"] = task
        quartile_rows.append(row)

quart = pd.DataFrame(
    quartile_rows
)

print()
print("=" * 100)
print("TASKWISE ACC QUARTILES")
print("=" * 100)
print(
    quart.to_string(
        index=False
    )
)


# ============================================================
# Direct >B vs <=B statistical test on benefit
# ============================================================

lo = d.loc[
    ~d["above_B"],
    "constraint_gain_mm",
].to_numpy(float)

hi = d.loc[
    d["above_B"],
    "constraint_gain_mm",
].to_numpy(float)

u = mannwhitneyu(
    hi,
    lo,
    alternative="greater",
)

print()
print("=" * 100)
print("HIGH-DYNAMICS BENEFIT TEST")
print("=" * 100)

print(
    "<=B median gain =",
    float(np.median(lo)),
    "mm",
)

print(
    ">B median gain  =",
    float(np.median(hi)),
    "mm",
)

print(
    ">B / <=B gain ratio =",
    float(
        np.median(hi)
        / np.median(lo)
    ),
)

print(
    "Mann-Whitney p =",
    float(u.pvalue),
)


# ============================================================
# Save
# ============================================================

d.to_csv(
    OUT
    / "rigid_constraint_merged_7947.csv",
    index=False,
)

gdf.to_csv(
    OUT
    / "boundary_stratification.csv",
    index=False,
)

quart.to_csv(
    OUT
    / "taskwise_acc_quartiles.csv",
    index=False,
)

summary = pd.DataFrame(
    [
        {
            "n": len(d),

            "raw_mpjpe_median_mm":
                d[
                    "raw_constraint_mpjpe_mm"
                ].median(),

            "constraint_mpjpe_median_mm":
                d[
                    "constraint_mpjpe_mm"
                ].median(),

            "paired_gain_median_mm":
                d[
                    "constraint_gain_mm"
                ].median(),

            "paired_gain_median_pct":
                d[
                    "constraint_gain_pct"
                ].median(),

            "improved_pct":
                100.0
                * (
                    d[
                        "constraint_gain_mm"
                    ] > 0
                ).mean(),

            "pa_vs_gain_rho":
                r_pa_gain[0],

            "pa_vs_gain_within_task_rho":
                r_pa_gain_task[0],

            "acc_vs_gain_rho":
                r_acc_gain[0],

            "acc_vs_gain_within_task_rho":
                r_acc_gain_task[0],

            "wilcoxon_p":
                w.pvalue,

            "high_vs_low_gain_p":
                u.pvalue,
        }
    ]
)

summary.to_csv(
    OUT
    / "summary.csv",
    index=False,
)

print()
print("=" * 100)
print("SUCCESS")
print("=" * 100)
print(
    "OUTPUT =",
    OUT,
)
