from pathlib import Path
import re
import numpy as np
import pandas as pd
import webdataset as wds

from dataset_components.decoders import decode_data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except ImportError:
    from sklearn.model_selection import GroupKFold
    HAS_SGKF = False


ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "fixed_camera_pair_task0006"
)

PAIR_CSV = ROOT / "task0006_three_pair_merged_analysis.csv"

WDS_ROOT = Path(
    "/home/soenchun/cse/PointWorld_workspace/"
    "datasets/behavior/wds_task0006/test"
)

OUT_FEATURES = ROOT / "task0006_action_descriptors_best_pair.csv"

DT = 0.2


# ============================================================
# Load valid best-pair labels
# ============================================================
pair_df = pd.read_csv(PAIR_CSV)

err_cols = [
    "err_HL_mm",
    "err_HR_mm",
    "err_LR_mm",
]

valid = pair_df.dropna(subset=err_cols).copy()

valid["best_pair"] = (
    valid[err_cols]
    .idxmin(axis=1)
    .str.replace("err_", "", regex=False)
    .str.replace("_mm", "", regex=False)
)

label_map = dict(
    zip(valid["clip"], valid["best_pair"])
)

print("=" * 100)
print("ACTION -> BEST VIEW PAIR DIAGNOSTIC")
print("=" * 100)
print("VALID LABELS =", len(label_map))


# ============================================================
# Helpers
# ============================================================
def rms(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x ** 2)))


def safe_p95(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, 95))


def episode_from_clip(clip):
    m = re.match(r"^(task-\d+_episode_\d+)", str(clip))
    if m:
        return m.group(1)
    return str(clip)


def quaternion_step_angles(q):
    q = np.asarray(q, dtype=np.float64)

    if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] != 4:
        return np.zeros(0, dtype=np.float64)

    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.clip(n, 1e-12, None)

    dots = np.sum(q[:-1] * q[1:], axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)

    return 2.0 * np.arccos(dots)


def trajectory_features(prefix, pose):
    pose = np.asarray(pose, dtype=np.float64)

    xyz = pose[:, :3]

    dp = np.diff(xyz, axis=0)
    step = np.linalg.norm(dp, axis=1)

    vel = dp / DT

    if len(xyz) >= 3:
        acc = np.diff(xyz, n=2, axis=0) / (DT ** 2)
        acc_mag = np.linalg.norm(acc, axis=1)
    else:
        acc_mag = np.zeros(0)

    if len(xyz) >= 4:
        jerk = np.diff(xyz, n=3, axis=0) / (DT ** 3)
        jerk_mag = np.linalg.norm(jerk, axis=1)
        msj = float(np.mean(np.sum(jerk ** 2, axis=1)))
    else:
        jerk_mag = np.zeros(0)
        msj = 0.0

    speed = np.linalg.norm(vel, axis=1)

    # Turn angle between consecutive displacement vectors
    turns = []
    for a, b in zip(dp[:-1], dp[1:]):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na > 1e-12 and nb > 1e-12:
            c = np.dot(a, b) / (na * nb)
            turns.append(np.arccos(np.clip(c, -1.0, 1.0)))

    turns = np.asarray(turns, dtype=np.float64)

    # Reversal: consecutive displacement directions > 90 degrees
    reversal_fraction = (
        float(np.mean(turns > (np.pi / 2)))
        if turns.size
        else 0.0
    )

    rot_steps = (
        quaternion_step_angles(pose[:, 3:7])
        if pose.shape[1] >= 7
        else np.zeros(0)
    )

    out = {
        f"{prefix}_path_length": float(step.sum()),
        f"{prefix}_net_displacement": float(
            np.linalg.norm(xyz[-1] - xyz[0])
        ),
        f"{prefix}_step_mean": float(step.mean()) if step.size else 0.0,
        f"{prefix}_step_max": float(step.max()) if step.size else 0.0,

        f"{prefix}_speed_mean": float(speed.mean()) if speed.size else 0.0,
        f"{prefix}_speed_std": float(speed.std()) if speed.size else 0.0,
        f"{prefix}_speed_max": float(speed.max()) if speed.size else 0.0,

        f"{prefix}_acc_rms": rms(acc_mag),
        f"{prefix}_acc_p95": safe_p95(acc_mag),

        f"{prefix}_jerk_rms": rms(jerk_mag),
        f"{prefix}_jerk_p95": safe_p95(jerk_mag),
        f"{prefix}_msj": msj,

        f"{prefix}_turn_mean_rad": (
            float(turns.mean()) if turns.size else 0.0
        ),
        f"{prefix}_turn_p95_rad": safe_p95(turns),
        f"{prefix}_reversal_fraction": reversal_fraction,

        f"{prefix}_rot_step_mean_rad": (
            float(rot_steps.mean()) if rot_steps.size else 0.0
        ),
        f"{prefix}_rot_step_p95_rad": safe_p95(rot_steps),
    }

    return out


def gripper_features(prefix, x):
    x = np.asarray(x).reshape(-1).astype(np.float64)

    if len(x) < 2:
        changes = 0
    else:
        changes = int(np.sum(x[1:] != x[:-1]))

    return {
        f"{prefix}_gripper_change_count": changes,
        f"{prefix}_gripper_change": int(changes > 0),
        f"{prefix}_gripper_open_fraction": float(x.mean()),
    }


def joint_features(joints):
    q = np.asarray(joints, dtype=np.float64)

    dq = np.diff(q, axis=0)

    if q.shape[0] >= 3:
        ddq = np.diff(q, n=2, axis=0)
    else:
        ddq = np.zeros((0, q.shape[1]))

    return {
        "joint_delta_rms": rms(dq),
        "joint_delta_max": (
            float(np.abs(dq).max()) if dq.size else 0.0
        ),
        "joint_acc_rms": rms(ddq),
        "joint_total_motion": (
            float(np.linalg.norm(dq, axis=1).sum())
            if dq.size else 0.0
        ),
        "joint_net_motion": float(
            np.linalg.norm(q[-1] - q[0])
        ),
    }


# ============================================================
# One-pass WDS extraction
# ============================================================
shards = sorted(WDS_ROOT.glob("test-*.tar"))
assert shards, WDS_ROOT

ds = (
    wds.WebDataset(
        [str(x) for x in shards],
        shardshuffle=False,
    )
    .map(lambda x: decode_data(x, domain="behavior"))
)

rows = []

for i, sample in enumerate(ds, start=1):

    clip = str(sample.get("__key__", ""))

    # Only 1807 complete-case labels are needed.
    if clip not in label_map:
        continue

    row = {
        "clip": clip,
        "episode": episode_from_clip(clip),
        "best_pair": label_map[clip],
    }

    row.update(
        trajectory_features(
            "left",
            sample["left_gripper_pose"],
        )
    )

    row.update(
        trajectory_features(
            "right",
            sample["right_gripper_pose"],
        )
    )

    row.update(
        gripper_features(
            "left",
            sample["left_gripper_open"],
        )
    )

    row.update(
        gripper_features(
            "right",
            sample["right_gripper_open"],
        )
    )

    row.update(
        joint_features(
            sample["joint_positions"]
        )
    )

    rows.append(row)

df = pd.DataFrame(rows)

assert len(df) == len(label_map), (
    f"Expected {len(label_map)} clips, got {len(df)}"
)

df.to_csv(OUT_FEATURES, index=False)

print("\nACTION FEATURE CSV =", OUT_FEATURES)
print("ROWS =", len(df))
print("EPISODES =", df["episode"].nunique())
print("FEATURES =", len(df.columns) - 3)


# ============================================================
# Grouped CV: Action-only -> best pair
# ============================================================
feature_cols = [
    c for c in df.columns
    if c not in {
        "clip",
        "episode",
        "best_pair",
    }
]

X = (
    df[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

y = df["best_pair"]
groups = df["episode"]

majority_class = y.value_counts().idxmax()
majority_acc = (y == majority_class).mean()

print("\n" + "=" * 100)
print("BASELINE")
print("=" * 100)
print("Majority class =", majority_class)
print(f"Majority accuracy = {majority_acc*100:.2f}%")
print("Random 3-way balanced chance = 33.33%")


if HAS_SGKF:
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    splits = splitter.split(X, y, groups)
    print("CV = StratifiedGroupKFold(5), grouped by episode")
else:
    splitter = GroupKFold(n_splits=5)
    splits = splitter.split(X, y, groups)
    print("CV = GroupKFold(5), grouped by episode")


models = {
    "LogisticRegression": Pipeline([
        ("scale", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]),

    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    ),
}


results = []

# Materialize folds so same folds are used for every model
if HAS_SGKF:
    folds = list(
        StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        ).split(X, y, groups)
    )
else:
    folds = list(
        GroupKFold(n_splits=5).split(X, y, groups)
    )


for model_name, model in models.items():

    fold_metrics = []

    print("\n" + "=" * 100)
    print(model_name)
    print("=" * 100)

    for fold_idx, (tr, te) in enumerate(folds, start=1):

        Xtr = X.iloc[tr]
        Xte = X.iloc[te]
        ytr = y.iloc[tr]
        yte = y.iloc[te]

        # Confirm no episode leakage
        train_groups = set(groups.iloc[tr])
        test_groups = set(groups.iloc[te])

        assert train_groups.isdisjoint(test_groups)

        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        acc = accuracy_score(yte, pred)
        bacc = balanced_accuracy_score(yte, pred)
        macro_f1 = f1_score(
            yte,
            pred,
            average="macro",
        )

        fold_metrics.append(
            (acc, bacc, macro_f1)
        )

        print(
            f"Fold {fold_idx}: "
            f"N={len(te)} | "
            f"acc={acc*100:.2f}% | "
            f"balanced_acc={bacc*100:.2f}% | "
            f"macroF1={macro_f1:.4f}"
        )

    arr = np.asarray(fold_metrics)

    print(
        f"\n{model_name} MEAN: "
        f"accuracy={arr[:,0].mean()*100:.2f}% "
        f"± {arr[:,0].std(ddof=1)*100:.2f} | "
        f"balanced_acc={arr[:,1].mean()*100:.2f}% "
        f"± {arr[:,1].std(ddof=1)*100:.2f} | "
        f"macroF1={arr[:,2].mean():.4f}"
    )

    results.append({
        "model": model_name,
        "accuracy_mean": arr[:,0].mean(),
        "accuracy_std": arr[:,0].std(ddof=1),
        "balanced_accuracy_mean": arr[:,1].mean(),
        "balanced_accuracy_std": arr[:,1].std(ddof=1),
        "macro_f1_mean": arr[:,2].mean(),
    })


res = pd.DataFrame(results)

OUT_RESULTS = ROOT / "task0006_action_only_best_pair_groupcv.csv"
res.to_csv(OUT_RESULTS, index=False)

print("\n" + "=" * 100)
print("FINAL")
print("=" * 100)
print(res.to_string(index=False))
print("\nRESULT CSV =", OUT_RESULTS)
print("SUCCESS = True")
