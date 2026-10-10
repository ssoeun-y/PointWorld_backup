from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import (
    spearmanr,
    pearsonr,
    rankdata,
)

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

OUT = ROOT / "analysis_5task" / "controls"
OUT.mkdir(parents=True, exist_ok=True)


def find_col(df, exact=None, contains=None, exclude=None):
    exact = exact or []
    contains = contains or []
    exclude = exclude or []

    for c in exact:
        if c in df.columns:
            return c

    for c in df.columns:
        lc = c.lower()

        if all(x.lower() in lc for x in contains):
            if not any(x.lower() in lc for x in exclude):
                return c

    raise RuntimeError(
        f"Column not found.\n"
        f"exact={exact}\n"
        f"contains={contains}\n"
        f"exclude={exclude}\n"
        f"columns={list(df.columns)}"
    )


def residualize(y, X):
    beta, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )
    return y - X @ beta


# ============================================================
# LOAD PER-OBJECT RESULTS
# ============================================================

parts = []

for task in TASKS:
    f = ROOT / task / "per_object_metrics.csv"

    if not f.exists():
        raise FileNotFoundError(f)

    d = pd.read_csv(f)

    if "task" not in d.columns:
        d["task"] = task

    parts.append(d)

obj = pd.concat(
    parts,
    ignore_index=True,
)

print("\nPER-OBJECT COLUMNS")
for c in obj.columns:
    print(" ", c)


acc_col = find_col(
    obj,
    exact=[
        "rms_acceleration_si_m_s2",
        "rms_acc_si_m_s2",
        "rms_acceleration",
    ],
    contains=["rms", "acc"],
)

pa_col = find_col(
    obj,
    exact=[
        "pa_mpjpe_mm",
        "mean_pa_mpjpe_mm",
        "object_pa_mpjpe_mm",
        "pa_mpjpe",
    ],
    contains=["pa", "mpjpe"],
)

point_col = find_col(
    obj,
    exact=[
        "final_object_point_count",
        "object_point_count",
        "point_count",
        "n_points",
    ],
    contains=["point", "count"],
)

object_col = find_col(
    obj,
    exact=[
        "object_short_name",
        "object_full_name",
        "object_name",
    ],
    contains=["object", "name"],
)

clip_col = find_col(
    obj,
    exact=[
        "clip",
        "clip_key",
        "__key__",
    ],
    contains=["clip"],
)

for c in [
    acc_col,
    pa_col,
    point_col,
]:
    obj[c] = pd.to_numeric(
        obj[c],
        errors="coerce",
    )

obj = obj.dropna(
    subset=[
        "task",
        clip_col,
        object_col,
        acc_col,
        pa_col,
        point_col,
    ]
).copy()

obj = obj[
    obj[point_col] > 0
].copy()

print("\nSELECTED COLUMNS")
print("Acc        =", acc_col)
print("PA-MPJPE   =", pa_col)
print("Point count=", point_col)
print("Object     =", object_col)
print("Clip       =", clip_col)
print("Object rows=", len(obj))


# ============================================================
# CONTROL 1:
# IS POINT COUNT EXPLAINING THE RESULT?
# ============================================================

rho_points_pa, p_points_pa = spearmanr(
    obj[point_col],
    obj[pa_col],
)

rho_acc_points, p_acc_points = spearmanr(
    obj[acc_col],
    obj[point_col],
)

# Partial Spearman-style control:
# rank transform Acc and PA,
# then remove task + log(point count).
rx = rankdata(
    obj[acc_col].to_numpy(float)
)

ry = rankdata(
    obj[pa_col].to_numpy(float)
)

rz = rankdata(
    np.log1p(
        obj[point_col].to_numpy(float)
    )
)

task_dummy = pd.get_dummies(
    obj["task"],
    drop_first=True,
    dtype=float,
).to_numpy()

X = np.column_stack([
    np.ones(len(obj)),
    rz,
    task_dummy,
])

rx_res = residualize(rx, X)
ry_res = residualize(ry, X)

partial_r, partial_p = pearsonr(
    rx_res,
    ry_res,
)

point_control = pd.DataFrame([
    {
        "analysis":
            "point_count_vs_PA-MPJPE",

        "N":
            len(obj),

        "rho_or_r":
            rho_points_pa,

        "p_value":
            p_points_pa,
    },
    {
        "analysis":
            "Acc_vs_point_count",

        "N":
            len(obj),

        "rho_or_r":
            rho_acc_points,

        "p_value":
            p_acc_points,
    },
    {
        "analysis":
            "Acc_vs_PA_control_task_and_point_count",

        "N":
            len(obj),

        "rho_or_r":
            partial_r,

        "p_value":
            partial_p,
    },
])

point_control.to_csv(
    OUT / "point_count_control.csv",
    index=False,
)


# ============================================================
# CONTROL 2:
# SAME OBJECT IDENTITY CONTROL
#
# Compare Acc vs PA-MPJPE WITHIN the same task + object name.
# Only use object groups appearing >= 10 times.
# ============================================================

obj["object_group"] = (
    obj["task"].astype(str)
    + "||"
    + obj[object_col].astype(str)
)

group_sizes = (
    obj.groupby("object_group")
    .size()
)

valid_groups = group_sizes[
    group_sizes >= 10
].index

same_obj = obj[
    obj["object_group"].isin(
        valid_groups
    )
].copy()

same_obj["acc_rank_within_object"] = (
    same_obj
    .groupby("object_group")[acc_col]
    .rank(
        pct=True,
        method="average",
    )
)

same_obj["pa_rank_within_object"] = (
    same_obj
    .groupby("object_group")[pa_col]
    .rank(
        pct=True,
        method="average",
    )
)

same_obj["point_rank_within_object"] = (
    same_obj
    .groupby("object_group")[point_col]
    .rank(
        pct=True,
        method="average",
    )
)

rho_same_obj, p_same_obj = spearmanr(
    same_obj["acc_rank_within_object"],
    same_obj["pa_rank_within_object"],
)

# additionally control point count INSIDE same object groups
a = same_obj[
    "acc_rank_within_object"
].to_numpy(float)

y = same_obj[
    "pa_rank_within_object"
].to_numpy(float)

z = same_obj[
    "point_rank_within_object"
].to_numpy(float)

X2 = np.column_stack([
    np.ones(len(same_obj)),
    z,
])

a_res = residualize(a, X2)
y_res = residualize(y, X2)

same_obj_point_r, same_obj_point_p = pearsonr(
    a_res,
    y_res,
)

object_control_summary = pd.DataFrame([
    {
        "analysis":
            "within_same_task_and_object",

        "N":
            len(same_obj),

        "object_groups":
            len(valid_groups),

        "rho_or_r":
            rho_same_obj,

        "p_value":
            p_same_obj,
    },
    {
        "analysis":
            "within_same_object_plus_point_count_control",

        "N":
            len(same_obj),

        "object_groups":
            len(valid_groups),

        "rho_or_r":
            same_obj_point_r,

        "p_value":
            same_obj_point_p,
    },
])

object_control_summary.to_csv(
    OUT / "same_object_control.csv",
    index=False,
)


# ============================================================
# OBJECT-BY-OBJECT CORRELATIONS
# Is one specific object driving everything?
# Use groups with >= 20 samples.
# ============================================================

object_rows = []

for (task, name), g in obj.groupby(
    ["task", object_col]
):
    if len(g) < 20:
        continue

    rho, p = spearmanr(
        g[acc_col],
        g[pa_col],
    )

    object_rows.append({
        "task": task,
        "object": name,
        "N": len(g),
        "spearman_rho": rho,
        "p_value": p,
        "median_points":
            g[point_col].median(),
        "median_pa_mpjpe_mm":
            g[pa_col].median(),
    })

object_corr = pd.DataFrame(
    object_rows
)

if len(object_corr):
    object_corr = object_corr.sort_values(
        ["task", "N"],
        ascending=[True, False],
    )

object_corr.to_csv(
    OUT / "objectwise_acc_pa_correlations.csv",
    index=False,
)

if len(object_corr):
    positive_fraction = float(
        (object_corr["spearman_rho"] > 0)
        .mean()
    )

    median_object_rho = float(
        object_corr["spearman_rho"]
        .median()
    )
else:
    positive_fraction = np.nan
    median_object_rho = np.nan


# ============================================================
# CONTROL 3:
# LEAVE-ONE-TASK-OUT
#
# If one task drives the whole result,
# correlation should collapse when that task is removed.
# ============================================================

clip_file = (
    ROOT
    / "analysis_5task"
    / "merged_per_clip_metrics_5task_clean.csv"
)

if not clip_file.exists():
    raise FileNotFoundError(
        clip_file
    )

clip = pd.read_csv(
    clip_file
)

clip_acc = find_col(
    clip,
    exact=[
        "rms_acceleration_si_m_s2",
        "rms_acc_si_m_s2",
        "rms_acceleration",
    ],
    contains=["rms", "acc"],
)

clip_pa = find_col(
    clip,
    exact=[
        "pa_mpjpe_mm",
        "mean_pa_mpjpe_mm",
        "clip_pa_mpjpe_mm",
        "pa_mpjpe",
    ],
    contains=["pa", "mpjpe"],
)

loto_rows = []

for omitted in TASKS:
    g = clip[
        clip["task"] != omitted
    ].copy()

    rho, p = spearmanr(
        g[clip_acc],
        g[clip_pa],
    )

    loto_rows.append({
        "omitted_task": omitted,
        "N": len(g),
        "spearman_rho": rho,
        "p_value": p,
    })

loto = pd.DataFrame(
    loto_rows
)

loto.to_csv(
    OUT / "leave_one_task_out.csv",
    index=False,
)


# ============================================================
# PRINT FINAL CONTROL REPORT
# ============================================================

print("\n" + "=" * 78)
print("CONTROL ANALYSIS: RMS Acc vs PA-MPJPE")
print("=" * 78)

print("\n[1. POINT COUNT CONTROL]")
print(
    point_control.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print("\n[2. SAME OBJECT CONTROL]")
print(
    object_control_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print("\n[OBJECT-WISE SUMMARY, groups N>=20]")
print(
    "OBJECT GROUPS =",
    len(object_corr),
)
print(
    "FRACTION WITH rho > 0 =",
    f"{positive_fraction:.3f}"
    if np.isfinite(positive_fraction)
    else "NA",
)
print(
    "MEDIAN OBJECT rho =",
    f"{median_object_rho:.4f}"
    if np.isfinite(median_object_rho)
    else "NA",
)

print("\n[3. LEAVE-ONE-TASK-OUT]")
print(
    loto.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print("\nINTERPRETATION GUIDE")
print(
    "- Acc vs PA after task + point-count control "
    "should remain positive."
)
print(
    "- Within the same task/object, Acc vs PA "
    "should remain positive."
)
print(
    "- Leave-one-task-out rho should stay positive "
    "for every omitted task."
)
print(
    "- Object-wise rho should be positive for "
    "many objects, not just one."
)

print("\nSAVED:")
for f in sorted(OUT.glob("*.csv")):
    print(" ", f)

print("=" * 78)
