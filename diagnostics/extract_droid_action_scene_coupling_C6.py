#!/usr/bin/env python3

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from diagnostics.visibility_consistency import (
    build_official_dataloader,
    normalize_key,
)

ROOT = Path(__file__).resolve().parents[1]

BASE_CSV = (
    ROOT /
    "diagnostics/results/droid_established_dynamics_1679/"
    "timestamp_aware_si/droid_eef_timestamp_aware_si_1679.csv"
)

DROID_WDS = Path(
    "/home/soenchun/cse/PointWorld_workspace/datasets/"
    "PointWorld-DROID-subset_restored/droid/wds"
)

TIMESTAMP_CACHE = Path(
    "/tmp/droid_validation503_timestamps.jsonl"
)

OUTDIR = (
    ROOT /
    "diagnostics/results/"
    "boundary_identifiability_context_C/"
    "C6_droid_action_scene_coupling"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PER_CLIP = OUTDIR / "C6_droid_action_scene_dynamics_1679.csv"
OUT_ASSOC = OUTDIR / "C6_droid_action_scene_association.csv"


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
    return x.astype(np.float64, copy=False)


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


def nonuniform_derivative(x, t):
    """
    Same timestamp-aware derivative convention already used
    for the aligned DROID EEF metrics:

      first:   (x1-x0)/(t1-t0)
      middle:  (x[t+1]-x[t-1])/(time[t+1]-time[t-1])
      last:    backward difference
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    assert x.shape[0] == len(t)
    assert len(t) >= 2

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Non-increasing timestamp sequence")

    out = np.empty_like(x, dtype=np.float64)

    out[0] = (
        x[1] - x[0]
    ) / (
        t[1] - t[0]
    )

    out[-1] = (
        x[-1] - x[-2]
    ) / (
        t[-1] - t[-2]
    )

    if len(t) > 2:
        denom = (
            t[2:] - t[:-2]
        )

        shape = (
            len(denom),
        ) + (1,) * (x.ndim - 1)

        out[1:-1] = (
            x[2:] - x[:-2]
        ) / denom.reshape(shape)

    return out


def nan_rms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(x ** 2)))


def nan_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def nan_p95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, 0.95))


# ============================================================
# Timestamp mapping
# ============================================================

timestamp_by_episode = {}

with open(TIMESTAMP_CACHE, "r") as f:
    for line in f:
        z = json.loads(line)
        timestamp_by_episode[
            str(z["episode"])
        ] = np.asarray(
            z["canonical_timestamps"],
            dtype=np.float64
        ) / 1000.0

print("timestamp episodes =", len(timestamp_by_episode))
assert len(timestamp_by_episode) == 1067


def parse_clip_slice(clip):
    """
    Example:
      TRI+...+..._flows-200:211

    -> episode:
      TRI+...+..._flows
    -> start=200, end=211
    """
    m = re.match(
        r"^(.*_flows)-(\d+):(\d+)$",
        str(clip)
    )

    if m is None:
        raise ValueError(
            f"Could not parse clip slice: {clip}"
        )

    return (
        m.group(1),
        int(m.group(2)),
        int(m.group(3)),
    )


def timestamps_for_clip(clip, T):
    episode, start, end = parse_clip_slice(clip)

    if episode not in timestamp_by_episode:
        raise KeyError(
            f"Timestamp episode missing: {episode}"
        )

    full = timestamp_by_episode[episode]

    t = full[start:end]

    if len(t) != T:
        raise RuntimeError(
            f"Timestamp length mismatch for {clip}: "
            f"slice={start}:{end}, ts={len(t)}, T={T}"
        )

    # numerically stable local time origin
    return t - t[0]


# ============================================================
# Scene temporal metrics
# ============================================================

def scene_temporal_metrics(gt, valid, moved, t):
    """
    DROID equivalent of BEHAVIOR scene_temporal_metrics,
    except derivatives use true DROID canonical timestamps.

    Metrics only use:
       (~scene_context_mask)
       & scene_exists
       & scene_supervised_mask
       & scene_moved_mask
    """

    velocity = nonuniform_derivative(gt, t)
    acceleration = nonuniform_derivative(velocity, t)
    jerk = nonuniform_derivative(acceleration, t)

    vm = valid & moved

    acc_mag = np.linalg.norm(
        acceleration,
        axis=-1
    )

    jerk_mag = np.linalg.norm(
        jerk,
        axis=-1
    )

    acc_vals = acc_mag[vm]
    jerk_vals = jerk_mag[vm]

    # Spatial scene-motion magnitude control.
    # This remains displacement in meters rather than /dt,
    # matching the BEHAVIOR control concept.
    step = gt[1:] - gt[:-1]

    step_mag = np.linalg.norm(
        step,
        axis=-1
    )

    transition_valid = (
        valid[1:]
        & valid[:-1]
    )

    transition_moved = (
        moved[1:]
        | moved[:-1]
    )

    tm = (
        transition_valid
        & transition_moved
    )

    step_vals = step_mag[tm]

    return {
        "scene_acc_rms_si_m_s2":
            nan_rms(acc_vals),

        "scene_acc_mean_si_m_s2":
            nan_mean(acc_vals),

        "scene_acc_p95_si_m_s2":
            nan_p95(acc_vals),

        "scene_msj_si_m2_s6":
            (
                float(np.mean(jerk_vals ** 2))
                if len(jerk_vals)
                else np.nan
            ),

        "scene_jerk_rms_si_m_s3":
            nan_rms(jerk_vals),

        "scene_jerk_p95_si_m_s3":
            nan_p95(jerk_vals),

        "scene_step_mean_m":
            nan_mean(step_vals),

        "scene_step_p95_m":
            nan_p95(step_vals),

        "num_valid_moved_pointframes":
            int(np.sum(vm)),
    }


def eef_path_length(batch):
    """
    Independent spatial Action magnitude control.
    DROID has right_gripper_pose.
    """
    pose = to_numpy(
        batch["right_gripper_pose"]
    )

    if pose.ndim == 3 and pose.shape[0] == 1:
        pose = pose[0]

    xyz = pose[:, :3].astype(
        np.float64,
        copy=False
    )

    if len(xyz) < 2:
        return np.nan

    return float(
        np.linalg.norm(
            xyz[1:] - xyz[:-1],
            axis=-1
        ).sum()
    )


# ============================================================
# Load fixed 1,679 population + already-established EEF metrics
# ============================================================

base = pd.read_csv(BASE_CSV)

required = [
    "clip",
    "site",
    "rms_acceleration_si_m_s2",
    "mean_squared_jerk_si_m2_s6",
    "filtered_l2_moved_mm",
]

for c in required:
    assert c in base.columns, c

assert len(base) == 1679
assert base["clip"].nunique() == 1679

target = set(
    base["clip"].astype(str)
)

print("TARGET =", len(target))


# ============================================================
# Official deterministic PointWorld DROID loader
# ============================================================

cli = SimpleNamespace(
    data_dir=DROID_WDS,
    seed=42,
    num_workers=0,
    split="test",
)

_, dataloader, _ = build_official_dataloader(cli)


# ============================================================
# Single WDS scan
# ============================================================

rows = []
seen = set()

for bi, batch in enumerate(dataloader, 1):

    key = normalize_key(
        batch["__key__"]
    )

    if key not in target:
        continue

    if key in seen:
        continue

    seen.add(key)

    gt = unbatch_xyz(
        batch["gt_scene_flows"]
    )

    T, N, _ = gt.shape

    exists = unbatch_mask(
        batch.get("scene_exists"),
        (T, N)
    )

    context = unbatch_mask(
        batch.get("scene_context_mask"),
        (T, N)
    )

    supervised = unbatch_mask(
        batch.get("scene_supervised_mask"),
        (T, N)
    )

    moved = unbatch_mask(
        batch.get("scene_moved_mask"),
        (T, N)
    )

    # Same official validity contract as BEHAVIOR.
    valid = (
        (~context)
        & exists
        & supervised
    )

    t = timestamps_for_clip(
        key,
        T
    )

    temporal = scene_temporal_metrics(
        gt,
        valid,
        moved,
        t
    )

    row = {
        "clip": key,
        "eef_path_length_m":
            eef_path_length(batch),

        "dt_min_s":
            float(np.min(np.diff(t))),

        "dt_median_s":
            float(np.median(np.diff(t))),

        "dt_mean_s":
            float(np.mean(np.diff(t))),

        "dt_max_s":
            float(np.max(np.diff(t))),

        **temporal,
    }

    rows.append(row)

    if len(rows) % 100 == 0:
        print(
            f"[{len(rows)}/1679] "
            f"{key} | "
            f"sceneAcc={row['scene_acc_rms_si_m_s2']:.6g} | "
            f"sceneMSJ={row['scene_msj_si_m2_s6']:.6g}"
        )

    if len(rows) == len(target):
        break


scene = pd.DataFrame(rows)

print()
print("FOUND =", len(scene))
print("UNIQUE =", scene["clip"].nunique())

missing = sorted(
    target - set(scene["clip"])
)

if missing:
    print("MISSING =", len(missing))
    print("FIRST =", missing[:10])

assert len(scene) == 1679
assert scene["clip"].nunique() == 1679


# ============================================================
# Merge action + error + context
# ============================================================

d = base.merge(
    scene,
    on="clip",
    how="inner",
    validate="one_to_one"
)

assert len(d) == 1679

d.to_csv(
    OUT_PER_CLIP,
    index=False
)


# ============================================================
# Association helpers
# ============================================================

def residualize(y, X):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    X = np.column_stack([
        np.ones(len(y)),
        X
    ])

    beta, *_ = np.linalg.lstsq(
        X, y,
        rcond=None
    )

    return y - X @ beta


def rank01(x):
    return (
        pd.Series(x)
        .rank(method="average")
        .to_numpy(float)
    )


def association_rows(
    df,
    action_col,
    scene_col,
    label,
):
    use_cols = [
        action_col,
        scene_col,
        "site",
        "eef_path_length_m",
        "scene_step_mean_m",
    ]

    z = df[use_cols].dropna().copy()

    # --------------------------------------------------------
    # 1. Raw
    # --------------------------------------------------------
    rho_raw, p_raw = spearmanr(
        z[action_col],
        z[scene_col]
    )

    # --------------------------------------------------------
    # 2. Within-site rank association
    # --------------------------------------------------------
    z["a_site_rank"] = (
        z.groupby("site")[action_col]
        .rank(pct=True)
    )

    z["s_site_rank"] = (
        z.groupby("site")[scene_col]
        .rank(pct=True)
    )

    rho_site, p_site = spearmanr(
        z["a_site_rank"],
        z["s_site_rank"]
    )

    # --------------------------------------------------------
    # 3. Site + Action spatial-motion control
    #
    # Rank variables first, then residualize both Action
    # dynamics and Scene dynamics on:
    #   site dummies + EEF path length.
    #
    # This asks whether higher-order Action dynamics couples
    # to higher-order Scene dynamics beyond simple path length.
    # --------------------------------------------------------
    site_dm = pd.get_dummies(
        z["site"].astype(str),
        drop_first=True,
        dtype=float
    ).to_numpy(float)

    a = rank01(
        z[action_col].to_numpy(float)
    )

    s = rank01(
        z[scene_col].to_numpy(float)
    )

    path = rank01(
        z["eef_path_length_m"].to_numpy(float)
    )[:, None]

    X_motion = np.column_stack([
        site_dm,
        path
    ])

    ar = residualize(
        a,
        X_motion
    )

    sr = residualize(
        s,
        X_motion
    )

    rho_motion, p_motion = spearmanr(
        ar,
        sr
    )

    # --------------------------------------------------------
    # 4. Stronger descriptive control:
    # site + EEF path + actual scene displacement magnitude.
    #
    # Important: scene_step is downstream physical response,
    # so this is NOT the primary causal statistic.
    # It is only a robustness check asking whether coupling
    # concerns higher-order dynamics beyond displacement size.
    # --------------------------------------------------------
    scene_step = rank01(
        z["scene_step_mean_m"].to_numpy(float)
    )[:, None]

    X_strict = np.column_stack([
        site_dm,
        path,
        scene_step
    ])

    ar2 = residualize(
        a,
        X_strict
    )

    sr2 = residualize(
        s,
        X_strict
    )

    rho_strict, p_strict = spearmanr(
        ar2,
        sr2
    )

    return {
        "pair": label,
        "n": len(z),

        "raw_rho":
            float(rho_raw),
        "raw_p":
            float(p_raw),

        "within_site_rho":
            float(rho_site),
        "within_site_p":
            float(p_site),

        "site_path_control_rho":
            float(rho_motion),
        "site_path_control_p":
            float(p_motion),

        "site_path_scene_motion_control_rho":
            float(rho_strict),
        "site_path_scene_motion_control_p":
            float(p_strict),
    }


assoc = []

assoc.append(
    association_rows(
        d,
        "rms_acceleration_si_m_s2",
        "scene_acc_rms_si_m_s2",
        "Action Acc -> Scene Acc"
    )
)

assoc.append(
    association_rows(
        d,
        "mean_squared_jerk_si_m2_s6",
        "scene_msj_si_m2_s6",
        "Action MSJ -> Scene MSJ"
    )
)

assoc_df = pd.DataFrame(
    assoc
)

assoc_df.to_csv(
    OUT_ASSOC,
    index=False
)


# ============================================================
# Site-specific coupling
# ============================================================

site_rows = []

for site, g in d.groupby(
    "site",
    sort=True
):

    for (
        action_col,
        scene_col,
        label
    ) in [
        (
            "rms_acceleration_si_m_s2",
            "scene_acc_rms_si_m_s2",
            "acc"
        ),
        (
            "mean_squared_jerk_si_m2_s6",
            "scene_msj_si_m2_s6",
            "msj"
        ),
    ]:

        z = g[
            [action_col, scene_col]
        ].dropna()

        if len(z) < 5:
            rho = np.nan
            p = np.nan
        else:
            rho, p = spearmanr(
                z[action_col],
                z[scene_col]
            )

        site_rows.append({
            "site": site,
            "metric": label,
            "n": len(z),
            "rho": rho,
            "p": p,
        })


site_df = pd.DataFrame(
    site_rows
)

site_df.to_csv(
    OUTDIR / "C6_site_specific_coupling.csv",
    index=False
)


# ============================================================
# Print
# ============================================================

print()
print("=" * 100)
print("C6 — DROID ACTION -> GT SCENE DYNAMICS")
print("=" * 100)

print(
    assoc_df.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("C6 — SITE-SPECIFIC COUPLING")
print("=" * 100)

print(
    site_df.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("DT SANITY")
print("=" * 100)

print(
    d[
        [
            "dt_min_s",
            "dt_median_s",
            "dt_mean_s",
            "dt_max_s",
        ]
    ].describe().to_string()
)

print()
print("OUTPUT =", OUT_PER_CLIP)
print("ASSOC  =", OUT_ASSOC)
print("SUCCESS = True")
