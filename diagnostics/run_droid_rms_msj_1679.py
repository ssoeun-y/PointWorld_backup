from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr, mannwhitneyu

from dataset_components.transforms import compute_flow_derivatives
from diagnostics.visibility_consistency import (
    build_official_dataloader,
    normalize_key,
)

# ============================================================
# Fixed paper subset / paths
# ============================================================

BASE = Path(
    "diagnostics/results/droid_temporal_quality_final_1679.csv"
)

DROID_WDS = Path(
    "/home/soenchun/cse/PointWorld_workspace/datasets/PointWorld-DROID-subset_restored/droid/wds"
)

OUTDIR = Path(
    "diagnostics/results/droid_established_dynamics_1679"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "droid_rms_msj_filtered_l2_1679.csv"


# ============================================================
# Helpers
# ============================================================

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def unbatch_xyz(x):
    x = to_numpy(x)

    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]

    assert x.ndim == 3 and x.shape[-1] == 3, x.shape
    return x.astype(np.float64)


def unbatch_mask(x, shape):
    if x is None:
        return np.ones(shape, dtype=bool)

    x = to_numpy(x)

    if x.ndim >= 3 and x.shape[0] == 1:
        x = x[0]

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    assert x.shape == shape, (x.shape, shape)
    return x.astype(bool)


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    more = 0
    less = 0

    for i in range(0, len(x), 500):
        xx = x[i:i+500, None]
        more += np.sum(xx > y[None, :])
        less += np.sum(xx < y[None, :])

    return (more - less) / (len(x) * len(y))


# ============================================================
# Load exact released-confidence 1,679 clip population
# ============================================================

base = pd.read_csv(BASE)

required = [
    "clip",
    "episode",
    "site",
    "filtered_l2_moved_mm",
]

for c in required:
    assert c in base.columns, (c, base.columns.tolist())

assert len(base) == 1679, len(base)
assert base["clip"].nunique() == 1679

target = set(base["clip"].astype(str))

print("TARGET =", len(target))


# ============================================================
# Official deterministic PointWorld DROID dataloader
# ============================================================

cli = SimpleNamespace(
    data_dir=DROID_WDS,
    seed=42,
    num_workers=0,
    split="test",
)

_, dataloader, _ = build_official_dataloader(cli)


# ============================================================
# Compute CURRENT established Action metrics
#
# RMS Acceleration:
#   official compute_flow_derivatives(robot_flows) acceleration
#   sqrt(mean(||a||^2))
#
# MSJ:
#   temporal derivative of official acceleration using same
#   midpoint derivative operator
#   mean(||jerk||^2)
# ============================================================

rows = []
seen = set()

for i, batch in enumerate(dataloader, 1):

    key = normalize_key(batch["__key__"])

    if key not in target:
        continue

    if key in seen:
        continue

    seen.add(key)

    robot = unbatch_xyz(batch["robot_flows"])

    robot_exists = unbatch_mask(
        batch.get("robot_exists"),
        robot.shape[:2],
    )

    # Official PointWorld derivative
    _, acceleration = compute_flow_derivatives(robot)

    # Use same midpoint derivative convention for jerk:
    # derivative of acceleration.
    jerk, _ = compute_flow_derivatives(acceleration)

    accel_sq = np.sum(acceleration ** 2, axis=-1)
    jerk_sq = np.sum(jerk ** 2, axis=-1)

    valid_a = robot_exists
    valid_j = robot_exists

    rms_acceleration = float(
        np.sqrt(np.mean(accel_sq[valid_a]))
    )

    mean_squared_jerk = float(
        np.mean(jerk_sq[valid_j])
    )

    rows.append({
        "clip": key,
        "rms_acceleration": rms_acceleration,
        "mean_squared_jerk": mean_squared_jerk,
    })

    if len(rows) % 100 == 0:
        print(
            f"[{len(rows)}/1679] "
            f"{key} "
            f"RMS={rms_acceleration:.8f} "
            f"MSJ={mean_squared_jerk:.8e}"
        )

    if len(rows) == len(target):
        break


metrics = pd.DataFrame(rows)

print("\nFOUND =", len(metrics))
print("UNIQUE =", metrics["clip"].nunique())

missing = sorted(target - set(metrics["clip"]))

if missing:
    print("MISSING N =", len(missing))
    print("FIRST MISSING =", missing[:10])

assert len(metrics) == 1679
assert metrics["clip"].nunique() == 1679


# ============================================================
# Merge exact official filtered error
# ============================================================

d = base.merge(
    metrics,
    on="clip",
    validate="one_to_one",
)

assert len(d) == 1679

d.to_csv(OUT, index=False)

print("\nSAVED =", OUT)


# ============================================================
# A. OUTPUT-FIRST
#
# First define good / bad Prediction from filtered_l2_moved.
# Then inspect what Action input entered those groups.
# ============================================================

d["error_group"] = pd.qcut(
    d["filtered_l2_moved_mm"].rank(method="first"),
    4,
    labels=["Q1_low_error", "Q2", "Q3", "Q4_high_error"],
)

low_error = d[d["error_group"] == "Q1_low_error"]
high_error = d[d["error_group"] == "Q4_high_error"]

print("\n" + "=" * 90)
print("A. OUTPUT-FIRST: LOW ERROR vs HIGH ERROR -> WHAT ACTION ENTERED?")
print("=" * 90)

print(
    "LOW ERROR N =", len(low_error),
    "| median error =", low_error["filtered_l2_moved_mm"].median()
)
print(
    "HIGH ERROR N =", len(high_error),
    "| median error =", high_error["filtered_l2_moved_mm"].median()
)

for metric in [
    "rms_acceleration",
    "mean_squared_jerk",
]:
    lo = low_error[metric]
    hi = high_error[metric]

    u, p = mannwhitneyu(
        hi,
        lo,
        alternative="two-sided",
    )

    print("\n", metric)
    print("Low-error median  =", lo.median())
    print("High-error median =", hi.median())
    print("High / Low        =", hi.median() / lo.median())
    print("Mann-Whitney p    =", p)
    print("Cliff delta       =", cliffs_delta(hi, lo))


# ============================================================
# B. INPUT-FIRST
#
# First define Low/High Action Dynamics.
# Then inspect filtered prediction error.
# ============================================================

print("\n" + "=" * 90)
print("B. INPUT-FIRST: LOW/HIGH ACTION DYNAMICS -> PREDICTION ERROR")
print("=" * 90)

for metric in [
    "rms_acceleration",
    "mean_squared_jerk",
]:

    group_col = metric + "_group"

    d[group_col] = pd.qcut(
        d[metric].rank(method="first"),
        4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
    )

    rho, p_rho = spearmanr(
        d[metric],
        d["filtered_l2_moved_mm"],
    )

    print("\n", metric)
    print("Spearman rho =", rho)
    print("Spearman p   =", p_rho)

    print("\nDose response:")
    for q in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        g = d[d[group_col] == q]
        print(
            q,
            "N =", len(g),
            "median error =",
            g["filtered_l2_moved_mm"].median(),
        )

    lo = d[d[group_col] == "Q1_low"][
        "filtered_l2_moved_mm"
    ]

    hi = d[d[group_col] == "Q4_high"][
        "filtered_l2_moved_mm"
    ]

    u, p = mannwhitneyu(
        hi,
        lo,
        alternative="two-sided",
    )

    print("\nQ4 HIGH vs Q1 LOW")
    print("Low dynamics error  =", lo.median())
    print("High dynamics error =", hi.median())
    print("High / Low ratio    =", hi.median() / lo.median())
    print("Delta mm            =", hi.median() - lo.median())
    print("Mann-Whitney p      =", p)
    print("Cliff delta         =", cliffs_delta(hi, lo))


# ============================================================
# C. DROID IMPORTANT CONTROL:
#    within-site rank analysis
#
# DROID varies by collection site. Repeat the same relationship
# after removing between-site ranking differences.
# ============================================================

print("\n" + "=" * 90)
print("C. WITHIN-SITE RANK CONTROL")
print("=" * 90)

for metric in [
    "rms_acceleration",
    "mean_squared_jerk",
]:

    xrank = (
        d.groupby("site")[metric]
        .rank(pct=True, method="average")
    )

    yrank = (
        d.groupby("site")["filtered_l2_moved_mm"]
        .rank(pct=True, method="average")
    )

    valid = xrank.notna() & yrank.notna()

    rho, p = spearmanr(
        xrank[valid],
        yrank[valid],
    )

    print(
        metric,
        "| within-site rho =", rho,
        "| p =", p,
        "| N =", int(valid.sum()),
    )


# Save group assignments as well
d.to_csv(
    OUTDIR / "droid_rms_msj_full_analysis_1679.csv",
    index=False,
)

print("\nSUCCESS = True")
