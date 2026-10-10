from pathlib import Path
from io import BytesIO
import pickle
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu


# ============================================================
# Paths
# ============================================================

BASE_CSV = Path(
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)

ROOTS = [
    Path("/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000"),
    Path("/home/soenchun/cse/PointWorld_workspace/PointWorld/diagnostics/data/"
         "pointworld_behavior_task0001_restored/behavior/wds"),
    Path("/home/soenchun/cse/PointWorld_workspace/PointWorld/diagnostics/data/"
         "pointworld_behavior_multitask_restored/behavior/wds"),
]

OUTDIR = Path(
    "diagnostics/results/raw_action_good_bad_7987"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fields from ORIGINAL Action source
# ============================================================

FIELDS = [
    "joint_positions",
    "base_pose",
    "right_gripper_pose",
    "left_gripper_pose",
    "right_gripper_open",
    "left_gripper_open",

    # DROID-style aliases just in case
    "gripper_pose",
    "gripper_open",
    "gripper_positions",
]


# ============================================================
# Utilities
# ============================================================

def decode_member(raw, suffix):
    if suffix.endswith(".npy"):
        return np.load(BytesIO(raw), allow_pickle=True)

    if suffix.endswith(".pyd"):
        return pickle.loads(raw)

    return None


def member_key_and_field(name):
    name = Path(name).name

    for field in FIELDS:
        for ext in [".npy", ".pyd"]:
            suffix = "." + field + ext
            if name.endswith(suffix):
                key = name[:-len(suffix)]
                return key, field, field + ext

    return None, None, None


def xyz_from_pose(pose):
    if pose is None:
        return None

    a = np.asarray(pose)

    # [T, 7] = xyz + quaternion
    if a.ndim == 2 and a.shape[1] >= 3:
        return a[:, :3].astype(np.float64)

    # [T, 4, 4]
    if a.ndim == 3 and a.shape[-2:] == (4, 4):
        return a[:, :3, 3].astype(np.float64)

    return None


def quat_from_pose(pose):
    if pose is None:
        return None

    a = np.asarray(pose)

    if a.ndim == 2 and a.shape[1] >= 7:
        q = a[:, 3:7].astype(np.float64)
        n = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.maximum(n, 1e-12)
        return q

    return None


def path_length(xyz):
    if xyz is None or len(xyz) < 2:
        return np.nan

    return float(
        np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    )


def net_displacement(xyz):
    if xyz is None or len(xyz) < 2:
        return np.nan

    return float(np.linalg.norm(xyz[-1] - xyz[0]))


def derivative_features(xyz):
    """
    Exploratory temporal profile from raw end-effector xyz.
    Frame-based units. This does NOT replace official robot-flow RMS/MSJ.
    """

    out = {
        "ee_speed_mean": np.nan,
        "ee_speed_p95": np.nan,
        "ee_speed_cv": np.nan,
        "ee_accel_rms": np.nan,
        "ee_accel_p95": np.nan,
        "ee_jerk_ms": np.nan,
        "ee_jerk_p95": np.nan,
    }

    if xyz is None or len(xyz) < 4:
        return out

    v = np.diff(xyz, axis=0)
    a = np.diff(v, axis=0)
    j = np.diff(a, axis=0)

    speed = np.linalg.norm(v, axis=1)
    accel = np.linalg.norm(a, axis=1)
    jerk = np.linalg.norm(j, axis=1)

    out["ee_speed_mean"] = float(np.mean(speed))
    out["ee_speed_p95"] = float(np.percentile(speed, 95))

    if np.mean(speed) > 1e-12:
        out["ee_speed_cv"] = float(
            np.std(speed) / np.mean(speed)
        )

    out["ee_accel_rms"] = float(
        np.sqrt(np.mean(accel ** 2))
    )

    out["ee_accel_p95"] = float(
        np.percentile(accel, 95)
    )

    out["ee_jerk_ms"] = float(
        np.mean(jerk ** 2)
    )

    out["ee_jerk_p95"] = float(
        np.percentile(jerk, 95)
    )

    return out


def rotation_features(pose):
    q = quat_from_pose(pose)

    if q is None or len(q) < 2:
        return {
            "ee_rotation_total_rad": np.nan,
            "ee_rotation_step_p95_rad": np.nan,
        }

    dots = np.sum(q[:-1] * q[1:], axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)

    ang = 2.0 * np.arccos(dots)

    return {
        "ee_rotation_total_rad": float(ang.sum()),
        "ee_rotation_step_p95_rad":
            float(np.percentile(ang, 95)),
    }


def gripper_features(g):
    if g is None:
        return {
            "gripper_range": np.nan,
            "gripper_total_variation": np.nan,
            "gripper_max_step": np.nan,
        }

    x = np.asarray(g, dtype=np.float64).reshape(-1)

    if len(x) < 2:
        return {
            "gripper_range": np.nan,
            "gripper_total_variation": np.nan,
            "gripper_max_step": np.nan,
        }

    dx = np.abs(np.diff(x))

    return {
        "gripper_range": float(x.max() - x.min()),
        "gripper_total_variation": float(dx.sum()),
        "gripper_max_step": float(dx.max()),
    }


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    more = 0
    less = 0

    for i in range(0, len(x), 500):
        xx = x[i:i+500, None]
        more += np.sum(xx > y[None, :])
        less += np.sum(xx < y[None, :])

    return float(
        (more - less) / (len(x) * len(y))
    )


# ============================================================
# 1. Prediction GOOD / BAD groups are defined ONLY by OUTPUT.
# ============================================================

base = pd.read_csv(BASE_CSV)

assert len(base) == 7987
assert base["clip"].nunique() == 7987

base["prediction_group"] = (
    base.groupby("task")["l2_moved_mm"]
    .transform(
        lambda x: pd.qcut(
            x.rank(method="first"),
            4,
            labels=[
                "Q1_low_error",
                "Q2",
                "Q3",
                "Q4_high_error"
            ]
        )
    )
)

targets = set(base["clip"].astype(str))

print("Total target clips:", len(targets))


# ============================================================
# 2. Read ORIGINAL raw Action source from WDS
# ============================================================

raw_by_clip = {}

tar_files = []

for root in ROOTS:
    tar_files.extend(sorted((root / "test").glob("*.tar")))
    tar_files.extend(sorted(root.glob("test-*.tar")))

tar_files = sorted(set(tar_files))

print("Found tar files:", len(tar_files))

for ti, tar_path in enumerate(tar_files, 1):

    with tarfile.open(tar_path, "r") as tar:

        for member in tar:

            if not member.isfile():
                continue

            clip, field, suffix = member_key_and_field(member.name)

            if clip is None or clip not in targets:
                continue

            f = tar.extractfile(member)

            if f is None:
                continue

            try:
                arr = decode_member(f.read(), suffix)
            except Exception:
                continue

            raw_by_clip.setdefault(clip, {})[field] = arr

    if ti % 10 == 0 or ti == len(tar_files):
        print(
            f"[{ti}/{len(tar_files)}] "
            f"clips found={len(raw_by_clip)}"
        )


# ============================================================
# 3. Convert each RAW Action Sequence into broad descriptors
# ============================================================

rows = []

for _, r in base.iterrows():

    clip = str(r["clip"])
    raw = raw_by_clip.get(clip, {})

    right_pose = raw.get(
        "right_gripper_pose",
        raw.get("gripper_pose")
    )

    left_pose = raw.get("left_gripper_pose")

    right_xyz = xyz_from_pose(right_pose)
    left_xyz = xyz_from_pose(left_pose)

    # Dominant arm = larger spatial path.
    right_len = path_length(right_xyz)
    left_len = path_length(left_xyz)

    if np.isfinite(right_len) and np.isfinite(left_len):
        if right_len >= left_len:
            dom_pose = right_pose
            dom_xyz = right_xyz
            dom_side = "right"
        else:
            dom_pose = left_pose
            dom_xyz = left_xyz
            dom_side = "left"

    elif np.isfinite(right_len):
        dom_pose = right_pose
        dom_xyz = right_xyz
        dom_side = "right"

    elif np.isfinite(left_len):
        dom_pose = left_pose
        dom_xyz = left_xyz
        dom_side = "left"

    else:
        dom_pose = None
        dom_xyz = None
        dom_side = "none"

    feat = {
        "clip": clip,
        "task": r["task"],
        "l2_moved_mm": r["l2_moved_mm"],
        "prediction_group": r["prediction_group"],

        "dominant_arm": dom_side,
        "ee_path_length": path_length(dom_xyz),
        "ee_net_displacement": net_displacement(dom_xyz),
    }

    if (
        np.isfinite(feat["ee_path_length"])
        and np.isfinite(feat["ee_net_displacement"])
    ):
        feat["ee_path_inefficiency"] = (
            feat["ee_path_length"]
            / max(feat["ee_net_displacement"], 1e-8)
        )
    else:
        feat["ee_path_inefficiency"] = np.nan

    feat.update(derivative_features(dom_xyz))
    feat.update(rotation_features(dom_pose))

    # gripper state of dominant side
    if dom_side == "left":
        grip = raw.get("left_gripper_open")
    else:
        grip = raw.get(
            "right_gripper_open",
            raw.get(
                "gripper_open",
                raw.get("gripper_positions")
            )
        )

    feat.update(gripper_features(grip))

    # Joint trajectory
    jp = raw.get("joint_positions")

    if jp is not None:
        jp = np.asarray(jp, dtype=np.float64)

        if jp.ndim == 2 and len(jp) >= 2:
            dj = np.diff(jp, axis=0)

            feat["joint_path_length"] = float(
                np.linalg.norm(dj, axis=1).sum()
            )
            feat["joint_net_displacement"] = float(
                np.linalg.norm(jp[-1] - jp[0])
            )
            feat["joint_step_p95"] = float(
                np.percentile(
                    np.linalg.norm(dj, axis=1),
                    95
                )
            )
        else:
            feat["joint_path_length"] = np.nan
            feat["joint_net_displacement"] = np.nan
            feat["joint_step_p95"] = np.nan
    else:
        feat["joint_path_length"] = np.nan
        feat["joint_net_displacement"] = np.nan
        feat["joint_step_p95"] = np.nan

    # Base motion
    bp = raw.get("base_pose")
    bxyz = xyz_from_pose(bp)

    feat["base_path_length"] = path_length(bxyz)
    feat["base_net_displacement"] = net_displacement(bxyz)

    rows.append(feat)


profiles = pd.DataFrame(rows)

profiles.to_csv(
    OUTDIR / "raw_action_profiles_7987.csv",
    index=False
)

print("\nRaw profiles saved.")
print("Rows =", len(profiles))


# ============================================================
# 4. GOOD vs BAD prediction:
#    Which RAW INPUT characteristics differ?
# ============================================================

ignore = {
    "clip",
    "task",
    "l2_moved_mm",
    "prediction_group",
    "dominant_arm",
}

features = [
    c for c in profiles.columns
    if c not in ignore
    and pd.api.types.is_numeric_dtype(profiles[c])
]

low = profiles[
    profiles["prediction_group"] == "Q1_low_error"
]

high = profiles[
    profiles["prediction_group"] == "Q4_high_error"
]

results = []

for col in features:

    a = low[col].dropna()
    b = high[col].dropna()

    if len(a) < 20 or len(b) < 20:
        continue

    u, p = mannwhitneyu(
        b,
        a,
        alternative="two-sided"
    )

    delta = cliffs_delta(b.values, a.values)

    valid = profiles[
        [col, "l2_moved_mm"]
    ].dropna()

    rho, rho_p = spearmanr(
        valid[col],
        valid["l2_moved_mm"]
    )

    task_signs = []

    for task, g in profiles.groupby("task"):

        gl = g[
            g["prediction_group"] == "Q1_low_error"
        ][col].dropna()

        gh = g[
            g["prediction_group"] == "Q4_high_error"
        ][col].dropna()

        if len(gl) < 5 or len(gh) < 5:
            continue

        task_signs.append(
            np.sign(gh.median() - gl.median())
        )

    positive_tasks = int(
        np.sum(np.asarray(task_signs) > 0)
    )
    negative_tasks = int(
        np.sum(np.asarray(task_signs) < 0)
    )

    med_low = float(a.median())
    med_high = float(b.median())

    results.append({
        "feature": col,
        "N_low_error": len(a),
        "N_high_error": len(b),
        "low_error_median": med_low,
        "high_error_median": med_high,
        "high_minus_low": med_high - med_low,
        "high_over_low":
            med_high / med_low
            if abs(med_low) > 1e-12
            else np.nan,
        "cliffs_delta": delta,
        "mannwhitney_p": p,
        "spearman_vs_error": rho,
        "spearman_p": rho_p,
        "positive_task_count": positive_tasks,
        "negative_task_count": negative_tasks,
    })


res = pd.DataFrame(results)

res["abs_cliffs_delta"] = (
    res["cliffs_delta"].abs()
)

res = res.sort_values(
    "abs_cliffs_delta",
    ascending=False
)

res.to_csv(
    OUTDIR / "good_vs_bad_raw_action_features.csv",
    index=False
)


# ============================================================
# 5. Representative raw samples
# ============================================================

rep = []

for task, g in profiles.groupby("task"):

    good = g.nsmallest(5, "l2_moved_mm")
    bad = g.nlargest(5, "l2_moved_mm")

    good = good.assign(
        representative_type="best_prediction"
    )

    bad = bad.assign(
        representative_type="worst_prediction"
    )

    rep.append(good)
    rep.append(bad)

rep = pd.concat(rep, ignore_index=True)

rep.to_csv(
    OUTDIR / "representative_good_bad_clips.csv",
    index=False
)


# ============================================================
# Print main result
# ============================================================

print("\n" + "=" * 90)
print("GOOD vs BAD PREDICTION: RAW ACTION INPUT CHARACTERISTICS")
print("=" * 90)

show_cols = [
    "feature",
    "low_error_median",
    "high_error_median",
    "high_over_low",
    "cliffs_delta",
    "spearman_vs_error",
    "positive_task_count",
    "negative_task_count",
    "mannwhitney_p",
]

print(
    res[show_cols]
    .head(20)
    .to_string(index=False)
)

print("\nSaved:")
print(
    OUTDIR / "raw_action_profiles_7987.csv"
)
print(
    OUTDIR / "good_vs_bad_raw_action_features.csv"
)
print(
    OUTDIR / "representative_good_bad_clips.csv"
)
