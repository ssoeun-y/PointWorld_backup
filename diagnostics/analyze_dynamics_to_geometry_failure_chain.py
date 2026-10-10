#!/usr/bin/env python3

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, rankdata

ROOT = Path(__file__).resolve().parents[1]
RDIR = ROOT / "diagnostics/results"

SEED_SPLIT = 20260902
ACC_BOUNDARY = 0.009832167363499603
MSJ_BOUNDARY = 8.152857428804615e-05

GT_FILES = [
    RDIR / "dynamics_to_geometry_failure_analysis/task0000__gt_contact_structure.csv",
    RDIR / "dynamics_to_geometry_failure_analysis/task0001__gt_contact_structure.csv",
    RDIR / "dynamics_to_geometry_failure_analysis/task0006_0008_0010__gt_contact_structure.csv",
]

PCL_FILE = (
    RDIR
    / "goal_preserved_global_boundary_taskcontrolled_pointcloud_structure"
    / "global_fixed_correct_5task__pointcloud_structure.csv"
)

OUTDIR = RDIR / "dynamics_to_geometry_failure_analysis/final_chain"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXACT original 70/30 split
# ============================================================

def stable_number(text):
    h = hashlib.sha256(
        f"{SEED_SPLIT}|{text}".encode()
    ).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def exact_split(df):
    out = pd.Series(index=df.index, dtype="object")

    for task, g in df.groupby("task"):
        idx = sorted(
            g.index,
            key=lambda i: stable_number(str(df.at[i, "clip"]))
        )

        ncal = round(len(idx) * 0.70)

        out.loc[idx[:ncal]] = "calibration"
        out.loc[idx[ncal:]] = "validation"

    return out


# ============================================================
# Partial Spearman:
# rank-transform X/Y/controls, regress out controls + task
# ============================================================

def partial_spearman(df, x, y, controls):
    cols = [x, y, "task"] + controls
    d = df[cols].copy()

    for c in [x, y] + controls:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna()

    if len(d) < 20:
        return np.nan, np.nan, len(d)

    rx = rankdata(d[x].to_numpy())
    ry = rankdata(d[y].to_numpy())

    mats = [np.ones(len(d))]

    # task fixed effects
    task_dm = pd.get_dummies(
        d["task"].astype(str),
        drop_first=True,
        dtype=float
    )

    if task_dm.shape[1]:
        mats.append(task_dm.to_numpy())

    for c in controls:
        mats.append(
            rankdata(d[c].to_numpy())[:, None]
        )

    Z = np.column_stack(mats)

    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)

    ex = rx - Z @ bx
    ey = ry - Z @ by

    rho, p = spearmanr(ex, ey)

    return float(rho), float(p), len(d)


def simple_spearman(df, x, y):
    d = df[[x, y]].apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    if len(d) < 10:
        return np.nan, np.nan, len(d)

    r, p = spearmanr(d[x], d[y])

    return float(r), float(p), len(d)


# ============================================================
# Load
# ============================================================

gt = pd.concat(
    [pd.read_csv(f) for f in GT_FILES],
    ignore_index=True
)

gt["task"] = gt["task"].astype(str)
gt["clip"] = gt["clip"].astype(str)

if len(gt) != 7987:
    raise RuntimeError(f"Expected 7987 GT rows, got {len(gt)}")

gt["split"] = exact_split(gt)

val = gt[gt["split"] == "validation"].copy()
cal = gt[gt["split"] == "calibration"].copy()

print("=" * 110)
print("DYNAMICS -> GEOMETRY FAILURE CHAIN")
print("=" * 110)
print("ALL N =", len(gt))
print("CALIBRATION N =", len(cal))
print("HELD-OUT VALIDATION N =", len(val))
print("Expected validation ≈ 2397")
print()


# ============================================================
# B5 — Higher-order / temporal scene complexity
#
# Question:
# Does high Action dynamics correspond to genuinely more
# temporally complex GT scene evolution?
# ============================================================

print("=" * 110)
print("B5 — HIGHER-ORDER / TEMPORAL SCENE COMPLEXITY")
print("MAIN ANALYSIS: HELD-OUT VALIDATION ONLY")
print("=" * 110)

pairs = [
    (
        "action_rms_acceleration",
        "scene_acc_rms",
        "Action RMS Acc -> GT Scene RMS Acc"
    ),
    (
        "action_rms_acceleration",
        "scene_msj",
        "Action RMS Acc -> GT Scene MSJ"
    ),
    (
        "action_msj",
        "scene_acc_rms",
        "Action MSJ -> GT Scene RMS Acc"
    ),
    (
        "action_msj",
        "scene_msj",
        "Action MSJ -> GT Scene MSJ"
    ),
]

b5_rows = []

for x, y, label in pairs:
    r, p, n = simple_spearman(
        val, x, y
    )

    # Remove overall scene-motion magnitude + task effects.
    pr, pp, pn = partial_spearman(
        val,
        x,
        y,
        controls=[
            "scene_step_mean_m",
            "scene_step_p95_m",
        ]
    )

    print()
    print(label)
    print(f"  raw Spearman              rho={r:+.4f}, p={p:.3e}, N={n}")
    print(f"  motion+task partial rank  rho={pr:+.4f}, p={pp:.3e}, N={pn}")

    b5_rows.append({
        "analysis": label,
        "raw_rho": r,
        "raw_p": p,
        "partial_rho": pr,
        "partial_p": pp,
        "N": pn,
    })


# ------------------------------------------------------------
# Boundary severity -> scene complexity dose response
# ------------------------------------------------------------

for axis, col, tau in [
    ("ACC", "action_rms_acceleration", ACC_BOUNDARY),
    ("MSJ", "action_msj", MSJ_BOUNDARY),
]:
    d = val.copy()

    d["severity"] = (
        pd.to_numeric(d[col], errors="coerce") / tau - 1.0
    )

    d = d[d["severity"] > 0].copy()

    d["Q"] = pd.qcut(
        d["severity"].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"]
    )

    print()
    print("-" * 110)
    print(f"{axis} boundary-excess dose response")
    print("-" * 110)

    for y in ["scene_acc_rms", "scene_msj"]:
        print(y)

        med = {}

        for q in ["Q1", "Q2", "Q3", "Q4"]:
            x = pd.to_numeric(
                d.loc[d["Q"] == q, y],
                errors="coerce"
            ).dropna()

            med[q] = float(x.median())

            print(
                f"  {q}: N={len(x):4d}, median={med[q]:.8g}"
            )

        r, p, _ = simple_spearman(
            d, "severity", y
        )

        print(
            f"  severity Spearman rho={r:+.4f}, p={p:.3e}"
        )


# ============================================================
# B6 — Contact / interaction proximity
#
# Binary collision is ~always true, so primary analysis uses
# continuous object proximity.
# Smaller distance = stronger/closer interaction.
# ============================================================

print()
print("=" * 110)
print("B6 — CONTACT / INTERACTION PROXIMITY")
print("=" * 110)

dist = pd.to_numeric(
    val["min_object_distance_m"],
    errors="coerce"
)

print(
    "distance metadata coverage =",
    f"{100*dist.notna().mean():.2f}%"
)

print(
    "binary contact prevalence =",
    f"{100*pd.to_numeric(val['contact_any'], errors='coerce').mean():.2f}%"
)

vd = val[dist.notna()].copy()
vd["min_object_distance_m"] = pd.to_numeric(
    vd["min_object_distance_m"],
    errors="coerce"
)

# Within-task proximity rank:
# 1.0 = closest interaction, 0 = farthest.
vd["proximity_rank"] = (
    1.0
    - vd.groupby("task")["min_object_distance_m"]
        .rank(pct=True)
)

# Divide proximity into near/mid/far by within-task rank.
vd["proximity_group"] = pd.cut(
    vd["proximity_rank"],
    bins=[-1e-9, 1/3, 2/3, 1.000001],
    labels=["Far", "Mid", "Near"],
    include_lowest=True,
)

b6_rows = []

for dyn in [
    "action_rms_acceleration",
    "action_msj",
]:
    print()
    print("-" * 110)
    print(dyn)
    print("-" * 110)

    for outcome in [
        "scene_acc_rms",
        "scene_msj",
        "official_moved_l2",
    ]:
        if outcome not in vd.columns:
            continue

        print(f"\nOutcome: {outcome}")

        for grp in ["Far", "Mid", "Near"]:
            g = vd[
                vd["proximity_group"] == grp
            ]

            r, p, n = simple_spearman(
                g,
                dyn,
                outcome,
            )

            print(
                f"  {grp:4s}: rho={r:+.4f}, p={p:.3e}, N={n}"
            )

            b6_rows.append({
                "dynamics": dyn,
                "outcome": outcome,
                "proximity": grp,
                "rho": r,
                "p": p,
                "N": n,
            })


# ============================================================
# B7 — Point-wise structural drift
#
# Use a calibration-fixed GT rigidity threshold.
# We ask:
# among scenes whose GT geometry itself is relatively stable,
# does high Action dynamics correspond to worse ORIGINAL
# PointWorld structural quality?
#
# IMPORTANT:
# This is a screening analysis.
# Exact GT-vs-pred comparison on the identical graph will be
# audited separately if this signal is positive.
# ============================================================

print()
print("=" * 110)
print("B7 — POINT-WISE STRUCTURAL DRIFT")
print("=" * 110)

GT_RIGID_THRESHOLD = float(
    pd.to_numeric(
        cal["gt_local_isometry_mean_mm"],
        errors="coerce"
    ).dropna().median()
)

print(
    "Calibration-fixed GT-rigid threshold "
    f"(median GT local distortion) = {GT_RIGID_THRESHOLD:.6f} mm"
)

pcl = pd.read_csv(PCL_FILE)
pcl["task"] = pcl["task"].astype(str)
pcl["clip"] = pcl["clip"].astype(str)

merged = pcl.merge(
    val[
        [
            "clip",
            "task",
            "action_rms_acceleration",
            "action_msj",
            "gt_local_isometry_mean_mm",
            "gt_temporal_jitter_mean_mm",
        ]
    ],
    on=["clip", "task"],
    how="left",
    validate="one_to_one",
)

if merged["action_rms_acceleration"].isna().any():
    # Some files may already use fully canonical clip names with
    # task duplicated differently. Fall back by clip only.
    pcl2 = pcl.drop(columns=["task"]).copy()

    tmp = val[
        [
            "clip",
            "task",
            "action_rms_acceleration",
            "action_msj",
            "gt_local_isometry_mean_mm",
            "gt_temporal_jitter_mean_mm",
        ]
    ].drop_duplicates("clip")

    merged = pcl2.merge(
        tmp,
        on="clip",
        how="left",
        validate="one_to_one",
    )

rigid = merged[
    pd.to_numeric(
        merged["gt_local_isometry_mean_mm"],
        errors="coerce"
    )
    <= GT_RIGID_THRESHOLD
].copy()

print("PCL scored N =", len(merged))
print("GT-rigid subset N =", len(rigid))

b7_rows = []

for axis, dyn, tau in [
    ("ACC", "action_rms_acceleration", ACC_BOUNDARY),
    ("MSJ", "action_msj", MSJ_BOUNDARY),
]:
    rigid["severity"] = (
        pd.to_numeric(rigid[dyn], errors="coerce") / tau - 1.0
    )

    outside = rigid[
        rigid["severity"] > 0
    ].copy()

    if len(outside) < 20:
        continue

    outside["Q"] = pd.qcut(
        outside["severity"].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"]
    )

    print()
    print("-" * 110)
    print(f"{axis} — GT-rigid subset")
    print("-" * 110)

    for outcome in [
        "original_rigidity_mean_mm",
        "original_rigidity_p95_mm",
        "original_jitter_mean_mm",
        "original_jitter_p95_mm",
    ]:
        r, p, n = simple_spearman(
            outside,
            "severity",
            outcome,
        )

        q1 = pd.to_numeric(
            outside.loc[
                outside["Q"] == "Q1",
                outcome
            ],
            errors="coerce"
        ).dropna()

        q4 = pd.to_numeric(
            outside.loc[
                outside["Q"] == "Q4",
                outcome
            ],
            errors="coerce"
        ).dropna()

        mw = mannwhitneyu(
            q4,
            q1,
            alternative="two-sided"
        )

        ratio = (
            float(q4.median() / q1.median())
            if len(q1) and q1.median() != 0
            else np.nan
        )

        print(
            f"{outcome}\n"
            f"  Q1 median={q1.median():.6f} mm\n"
            f"  Q4 median={q4.median():.6f} mm\n"
            f"  Q4/Q1={ratio:.3f}x\n"
            f"  severity rho={r:+.4f}, p={p:.3e}\n"
            f"  Q4 vs Q1 p={mw.pvalue:.3e}"
        )

        b7_rows.append({
            "axis": axis,
            "outcome": outcome,
            "N": n,
            "Q1_median": q1.median(),
            "Q4_median": q4.median(),
            "Q4_Q1_ratio": ratio,
            "rho": r,
            "spearman_p": p,
            "Q4_Q1_p": mw.pvalue,
        })


# ============================================================
# SAVE
# ============================================================

pd.DataFrame(b5_rows).to_csv(
    OUTDIR / "B5_temporal_complexity.csv",
    index=False
)

pd.DataFrame(b6_rows).to_csv(
    OUTDIR / "B6_contact_proximity.csv",
    index=False
)

pd.DataFrame(b7_rows).to_csv(
    OUTDIR / "B7_structural_drift_screen.csv",
    index=False
)

val.to_csv(
    OUTDIR / "heldout_failure_chain_samples.csv",
    index=False
)

print()
print("=" * 110)
print("SAVED =", OUTDIR)
print("SUCCESS = True")
print("=" * 110)
