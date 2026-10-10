from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dl
from dataset_components.cameras import sample_cameras as official_sample_cameras


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "fixed_camera_pair_task0006"
)

PAIR_CSV = ROOT / "task0006_three_pair_merged_analysis.csv"

FEATURE_CSV = (
    ROOT /
    "task0006_head_scene_action_query_features.csv"
)

RESULT_CSV = (
    ROOT /
    "task0006_scene_vs_action_vs_scene_action_query.csv"
)

DT = 0.2


# ============================================================
# Labels:
# Current view = HEAD
#
# HL = query LEFT
# HR = query RIGHT
#
# label L if HEAD+LEFT has lower official moved-L2
# label R if HEAD+RIGHT has lower official moved-L2
# ============================================================

pair_df = pd.read_csv(PAIR_CSV)

pair_df = pair_df.dropna(
    subset=["err_HL_mm", "err_HR_mm"]
).copy()

pair_df = pair_df[
    pair_df["err_HL_mm"] != pair_df["err_HR_mm"]
].copy()

pair_df["query_label"] = np.where(
    pair_df["err_HL_mm"] < pair_df["err_HR_mm"],
    "L",
    "R",
)

target_clips = set(pair_df["clip"])


def episode_from_clip(clip):
    m = re.match(
        r"^(task-\d+_episode_\d+)",
        str(clip),
    )
    return m.group(1) if m else str(clip)


pair_df["episode"] = pair_df["clip"].map(
    episode_from_clip
)


# ============================================================
# Action descriptors
# ============================================================

def rms(x):
    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:
        return 0.0

    return float(
        np.sqrt(np.mean(x ** 2))
    )


def p95(x):
    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:
        return 0.0

    return float(
        np.percentile(x, 95)
    )


def quaternion_step_angles(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    if len(q) < 2:
        return np.zeros(0)

    norm = np.linalg.norm(
        q,
        axis=1,
        keepdims=True,
    )

    q = q / np.clip(
        norm,
        1e-12,
        None,
    )

    dots = np.sum(
        q[:-1] * q[1:],
        axis=1,
    )

    dots = np.clip(
        np.abs(dots),
        0,
        1,
    )

    return (
        2.0 *
        np.arccos(dots)
    )


def trajectory_features(prefix, pose):

    pose = np.asarray(
        pose,
        dtype=np.float64,
    )

    xyz = pose[:, :3]

    dp = np.diff(
        xyz,
        axis=0,
    )

    step = np.linalg.norm(
        dp,
        axis=1,
    )

    vel = dp / DT

    speed = np.linalg.norm(
        vel,
        axis=1,
    )

    if len(xyz) >= 3:

        acc = (
            np.diff(
                xyz,
                n=2,
                axis=0,
            )
            / (DT ** 2)
        )

        acc_mag = np.linalg.norm(
            acc,
            axis=1,
        )

    else:
        acc_mag = np.zeros(0)

    if len(xyz) >= 4:

        jerk = (
            np.diff(
                xyz,
                n=3,
                axis=0,
            )
            / (DT ** 3)
        )

        jerk_mag = np.linalg.norm(
            jerk,
            axis=1,
        )

        msj = float(
            np.mean(
                np.sum(
                    jerk ** 2,
                    axis=1,
                )
            )
        )

    else:
        jerk_mag = np.zeros(0)
        msj = 0.0

    turns = []

    for a, b in zip(
        dp[:-1],
        dp[1:],
    ):

        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if (
            na > 1e-12
            and nb > 1e-12
        ):
            c = (
                np.dot(a, b)
                /
                (na * nb)
            )

            turns.append(
                np.arccos(
                    np.clip(
                        c,
                        -1,
                        1,
                    )
                )
            )

    turns = np.asarray(
        turns,
        dtype=np.float64,
    )

    rot = quaternion_step_angles(
        pose[:, 3:7]
    )

    return {

        f"action_{prefix}_path_length":
            float(step.sum()),

        f"action_{prefix}_net_disp":
            float(
                np.linalg.norm(
                    xyz[-1] - xyz[0]
                )
            ),

        f"action_{prefix}_speed_mean":
            float(speed.mean())
            if speed.size else 0.0,

        f"action_{prefix}_speed_std":
            float(speed.std())
            if speed.size else 0.0,

        f"action_{prefix}_speed_max":
            float(speed.max())
            if speed.size else 0.0,

        f"action_{prefix}_acc_rms":
            rms(acc_mag),

        f"action_{prefix}_acc_p95":
            p95(acc_mag),

        f"action_{prefix}_jerk_rms":
            rms(jerk_mag),

        f"action_{prefix}_jerk_p95":
            p95(jerk_mag),

        f"action_{prefix}_msj":
            msj,

        f"action_{prefix}_turn_mean":
            float(turns.mean())
            if turns.size else 0.0,

        f"action_{prefix}_turn_p95":
            p95(turns),

        f"action_{prefix}_reversal_fraction":
            float(
                np.mean(
                    turns > np.pi / 2
                )
            )
            if turns.size else 0.0,

        f"action_{prefix}_rot_mean":
            float(rot.mean())
            if rot.size else 0.0,

        f"action_{prefix}_rot_p95":
            p95(rot),
    }


def gripper_features(prefix, x):

    x = np.asarray(x).reshape(-1)

    changes = (
        int(
            np.sum(
                x[1:] != x[:-1]
            )
        )
        if len(x) > 1
        else 0
    )

    return {

        f"action_{prefix}_gripper_change_count":
            changes,

        f"action_{prefix}_gripper_change":
            int(changes > 0),

        f"action_{prefix}_gripper_open_fraction":
            float(
                x.astype(
                    np.float64
                ).mean()
            ),
    }


def joint_features(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    dq = np.diff(
        q,
        axis=0,
    )

    if len(q) >= 3:
        ddq = np.diff(
            q,
            n=2,
            axis=0,
        )
    else:
        ddq = np.zeros(
            (0, q.shape[1])
        )

    return {

        "action_joint_delta_rms":
            rms(dq),

        "action_joint_delta_max":
            float(
                np.abs(dq).max()
            )
            if dq.size else 0.0,

        "action_joint_acc_rms":
            rms(ddq),

        "action_joint_total_motion":
            float(
                np.linalg.norm(
                    dq,
                    axis=1,
                ).sum()
            )
            if dq.size else 0.0,

        "action_joint_net_motion":
            float(
                np.linalg.norm(
                    q[-1] - q[0]
                )
            ),
    }


# ============================================================
# Force HEAD as CURRENT observation.
#
# Important:
# This is NOT a PointWorld performance evaluation condition.
# It is only used to extract the selector's current-scene
# representation.
#
# Final performance labels/errors remain the official
# 2-camera HL vs HR PointWorld runs.
# ============================================================

def fixed_head_selector(
    sample,
    min_num_cameras=1,
    max_num_cameras=None,
    deterministic=False,
    seed=None,
):

    prefixes = sorted({
        k.split("_scene_")[0]
        for k in sample.keys()
        if "__" not in k
        and "_scene_flows" in k
        and k != "scene_flows"
    })

    wanted = "camera_head"

    if wanted not in prefixes:
        raise RuntimeError(
            f"{sample.get('__key__')} "
            f"has no camera_head; "
            f"available={prefixes}"
        )

    image_suffixes = (
        "_initial_rgb",
        "_initial_depth",
        "_intrinsic",
        "_extrinsic",
    )

    for prefix in prefixes:

        if prefix == wanted:
            continue

        for key in list(
            sample.keys()
        ):

            if key.startswith(
                prefix + "_scene_"
            ):
                sample.pop(
                    key,
                    None,
                )
                continue

            for suffix in image_suffixes:

                if key == (
                    prefix + suffix
                ):
                    sample.pop(
                        key,
                        None,
                    )

    return official_sample_cameras(
        sample,
        min_num_cameras=1,
        max_num_cameras=1,
        deterministic=True,
        seed=seed,
    )


# ============================================================
# Feature extraction.
# Cached after first run.
# ============================================================

if not FEATURE_CSV.exists():

    print("=" * 100)
    print("EXTRACTING HEAD-ONLY POINTWORLD SCENE FEATURES")
    print("=" * 100)

    args = parse_args()

    # IMPORTANT:
    # Tester/model is initialized with the official checkpoint contract.
    # We do NOT override eval camera count on CLI.
    tester = Tester(args)

    model = tester.model
    model.eval()

    # Only after official checkpoint initialization,
    # use one camera for selector feature extraction.
    tester.args.eval_min_num_cameras = 1
    tester.args.eval_max_num_cameras = 1

    dl.sample_cameras = fixed_head_selector

    loader, info = tester._build_eval_loader(
        "test"
    )

    rows = []

    with torch.no_grad():

        for idx, batch in enumerate(
            loader,
            start=1,
        ):

            clips = [
                str(x)
                for x in batch["__key__"]
            ]

            clip = clips[0]

            # Still need loader preprocessing,
            # but no feature needs to be saved
            # for invalid HL/HR labels.
            if clip not in target_clips:
                continue

            batch = {
                k: (
                    v.to(
                        tester.device,
                        non_blocking=True,
                    )
                    if isinstance(
                        v,
                        torch.Tensor,
                    )
                    else v
                )
                for k, v in batch.items()
            }

            # Official frozen PointWorld scene encoder.
            encoded = model.encode_scene_features(
                batch
            )

            exists = (
                batch["scene_exists"][:, 0]
                .bool()
            )

            f = encoded[0][
                exists[0]
            ]

            if f.shape[0] == 0:
                raise RuntimeError(
                    f"No scene points: {clip}"
                )

            # Strong but simple frozen representation:
            # mean + max pooling of PointWorld
            # per-point scene embeddings.
            mean_feat = (
                f.mean(dim=0)
                .float()
                .cpu()
                .numpy()
            )

            max_feat = (
                f.max(dim=0)
                .values
                .float()
                .cpu()
                .numpy()
            )

            coords = (
                batch["scene_flows"][
                    0,
                    0,
                ][exists[0]]
                .float()
            )

            row = {
                "clip": clip,
                "episode":
                    episode_from_clip(
                        clip
                    ),
            }

            # PointWorld scene representation
            for j, value in enumerate(
                mean_feat
            ):
                row[
                    f"scene_pw_mean_{j}"
                ] = float(value)

            for j, value in enumerate(
                max_feat
            ):
                row[
                    f"scene_pw_max_{j}"
                ] = float(value)

            # Current t0 geometry summary
            coord_mean = (
                coords.mean(dim=0)
                .cpu()
                .numpy()
            )

            coord_std = (
                coords.std(dim=0)
                .cpu()
                .numpy()
            )

            for j, axis in enumerate(
                ["x", "y", "z"]
            ):
                row[
                    f"scene_coord_mean_{axis}"
                ] = float(
                    coord_mean[j]
                )

                row[
                    f"scene_coord_std_{axis}"
                ] = float(
                    coord_std[j]
                )

            row["scene_point_count"] = int(
                f.shape[0]
            )

            # ------------------------------------------------
            # Candidate Action descriptors
            # ------------------------------------------------

            left_pose = (
                batch[
                    "left_gripper_pose"
                ][0]
                .detach()
                .cpu()
                .numpy()
            )

            right_pose = (
                batch[
                    "right_gripper_pose"
                ][0]
                .detach()
                .cpu()
                .numpy()
            )

            left_open = (
                batch[
                    "left_gripper_open"
                ][0]
                .detach()
                .cpu()
                .numpy()
            )

            right_open = (
                batch[
                    "right_gripper_open"
                ][0]
                .detach()
                .cpu()
                .numpy()
            )

            joints = (
                batch[
                    "joint_positions"
                ][0]
                .detach()
                .cpu()
                .numpy()
            )

            row.update(
                trajectory_features(
                    "left",
                    left_pose,
                )
            )

            row.update(
                trajectory_features(
                    "right",
                    right_pose,
                )
            )

            row.update(
                gripper_features(
                    "left",
                    left_open,
                )
            )

            row.update(
                gripper_features(
                    "right",
                    right_open,
                )
            )

            row.update(
                joint_features(
                    joints
                )
            )

            rows.append(row)

            if len(rows) % 100 == 0:
                print(
                    f"saved features: "
                    f"{len(rows)}"
                )

    feat_df = pd.DataFrame(rows)

    print(
        "EXTRACTED =",
        len(feat_df),
    )

    feat_df.to_csv(
        FEATURE_CSV,
        index=False,
    )

    print(
        "FEATURE CSV =",
        FEATURE_CSV,
    )

else:

    print(
        "Reusing cached feature CSV:",
        FEATURE_CSV,
    )

    feat_df = pd.read_csv(
        FEATURE_CSV
    )


# ============================================================
# Merge official HL/HR errors
# ============================================================

df = pair_df.merge(
    feat_df,
    on=[
        "clip",
        "episode",
    ],
    how="inner",
    validate="one_to_one",
)

print("\n" + "=" * 100)
print("SCENE vs ACTION vs SCENE+ACTION")
print("=" * 100)

print("N =", len(df))
print(
    "EPISODES =",
    df["episode"].nunique(),
)

print("\nQUERY LABELS:")

print(
    df["query_label"]
    .value_counts()
    .to_string()
)

print(
    "\nMajority accuracy =",
    f"{df['query_label'].value_counts(normalize=True).max()*100:.2f}%"
)


# ============================================================
# Feature sets
# ============================================================

scene_cols = [
    c for c in df.columns
    if c.startswith("scene_")
]

action_cols = [
    c for c in df.columns
    if c.startswith("action_")
]

feature_sets = {

    "Scene-only":
        scene_cols,

    "Action-only":
        action_cols,

    "Scene+Action":
        scene_cols + action_cols,
}

print(
    "\nScene features =",
    len(scene_cols),
)

print(
    "Action features =",
    len(action_cols),
)


# ============================================================
# Episode-grouped CV.
# Same folds for every model / feature set.
# ============================================================

y = df["query_label"]
groups = df["episode"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

dummy_X = np.zeros(
    (len(df), 1)
)

folds = list(
    splitter.split(
        dummy_X,
        y,
        groups,
    )
)


models = {

    "LogisticRegression":

        lambda: Pipeline([
            (
                "scale",
                StandardScaler(),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    C=1.0,
                    random_state=42,
                ),
            ),
        ]),

    "RandomForest":

        lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
}


results = []


# ============================================================
# Analytical random-query baseline.
# Expected error of 50/50 LEFT/RIGHT.
# ============================================================

random_expected = (
    0.5 * df["err_HL_mm"]
    +
    0.5 * df["err_HR_mm"]
).mean()

oracle_mean = (
    df[
        [
            "err_HL_mm",
            "err_HR_mm",
        ]
    ]
    .min(axis=1)
    .mean()
)

print("\n" + "=" * 100)
print("OFFICIAL POINTWORLD MOVED-L2 BASELINES")
print("=" * 100)

print(
    "Random LEFT/RIGHT expected mean =",
    f"{random_expected:.4f} mm",
)

print(
    "Oracle LEFT/RIGHT mean =",
    f"{oracle_mean:.4f} mm",
)


# ============================================================
# Models
# ============================================================

for model_name, model_factory in models.items():

    for feature_name, cols in feature_sets.items():

        X = (
            df[cols]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        all_true = []
        all_pred = []

        selected_errors = []
        fixed_errors = []
        oracle_errors = []

        fold_acc = []

        print(
            "\n" +
            "=" * 100
        )

        print(
            model_name,
            "|",
            feature_name,
        )

        print(
            "=" * 100
        )

        for fold_id, (
            train_idx,
            test_idx,
        ) in enumerate(
            folds,
            start=1,
        ):

            train_groups = set(
                groups.iloc[
                    train_idx
                ]
            )

            test_groups = set(
                groups.iloc[
                    test_idx
                ]
            )

            assert (
                train_groups
                .isdisjoint(
                    test_groups
                )
            )

            model = model_factory()

            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
            )

            pred = model.predict(
                X.iloc[test_idx]
            )

            y_test = y.iloc[
                test_idx
            ].to_numpy()

            fold_df = df.iloc[
                test_idx
            ]

            # ---------------------------------------------
            # Actual PointWorld moved-L2 achieved by selector
            # ---------------------------------------------

            selected = np.where(
                pred == "L",
                fold_df[
                    "err_HL_mm"
                ].to_numpy(),
                fold_df[
                    "err_HR_mm"
                ].to_numpy(),
            )

            # Fixed-view baseline is selected only
            # from TRAIN fold to avoid test leakage.
            train_df = df.iloc[
                train_idx
            ]

            train_HL_mean = (
                train_df[
                    "err_HL_mm"
                ].mean()
            )

            train_HR_mean = (
                train_df[
                    "err_HR_mm"
                ].mean()
            )

            fixed_pair = (
                "L"
                if (
                    train_HL_mean
                    <
                    train_HR_mean
                )
                else "R"
            )

            if fixed_pair == "L":
                fixed = fold_df[
                    "err_HL_mm"
                ].to_numpy()
            else:
                fixed = fold_df[
                    "err_HR_mm"
                ].to_numpy()

            oracle = np.minimum(
                fold_df[
                    "err_HL_mm"
                ].to_numpy(),
                fold_df[
                    "err_HR_mm"
                ].to_numpy(),
            )

            acc = accuracy_score(
                y_test,
                pred,
            )

            fold_acc.append(acc)

            all_true.extend(
                y_test
            )

            all_pred.extend(
                pred
            )

            selected_errors.extend(
                selected
            )

            fixed_errors.extend(
                fixed
            )

            oracle_errors.extend(
                oracle
            )

            print(
                f"Fold {fold_id}: "
                f"N={len(test_idx)} | "
                f"acc={acc*100:.2f}% | "
                f"fixed={fixed_pair} | "
                f"selected moved-L2="
                f"{np.mean(selected):.4f} mm | "
                f"fixed moved-L2="
                f"{np.mean(fixed):.4f} mm"
            )

        all_true = np.asarray(
            all_true
        )

        all_pred = np.asarray(
            all_pred
        )

        selected_errors = np.asarray(
            selected_errors
        )

        fixed_errors = np.asarray(
            fixed_errors
        )

        oracle_errors = np.asarray(
            oracle_errors
        )

        accuracy = accuracy_score(
            all_true,
            all_pred,
        )

        balanced_acc = (
            balanced_accuracy_score(
                all_true,
                all_pred,
            )
        )

        macro_f1 = f1_score(
            all_true,
            all_pred,
            average="macro",
        )

        selected_mean = float(
            selected_errors.mean()
        )

        selected_median = float(
            np.median(
                selected_errors
            )
        )

        fixed_mean = float(
            fixed_errors.mean()
        )

        oracle_cv_mean = float(
            oracle_errors.mean()
        )

        gain_vs_fixed_pct = (
            100.0 *
            (
                fixed_mean
                -
                selected_mean
            )
            /
            fixed_mean
        )

        oracle_gap_pct = (
            100.0 *
            (
                selected_mean
                -
                oracle_cv_mean
            )
            /
            oracle_cv_mean
        )

        print("\nFINAL:")

        print(
            f"accuracy="
            f"{accuracy*100:.2f}%"
        )

        print(
            f"balanced_acc="
            f"{balanced_acc*100:.2f}%"
        )

        print(
            f"macroF1="
            f"{macro_f1:.4f}"
        )

        print(
            "Official moved-L2 selected =",
            f"{selected_mean:.4f} mm",
        )

        print(
            "Official moved-L2 fixed =",
            f"{fixed_mean:.4f} mm",
        )

        print(
            "Official moved-L2 oracle =",
            f"{oracle_cv_mean:.4f} mm",
        )

        print(
            "Improvement vs fixed =",
            f"{gain_vs_fixed_pct:.2f}%"
        )

        print(
            "Gap to oracle =",
            f"{oracle_gap_pct:.2f}%"
        )

        results.append({

            "model":
                model_name,

            "features":
                feature_name,

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_acc,

            "macro_f1":
                macro_f1,

            "moved_l2_selected_mean_mm":
                selected_mean,

            "moved_l2_selected_median_mm":
                selected_median,

            "moved_l2_fixed_mean_mm":
                fixed_mean,

            "moved_l2_oracle_mean_mm":
                oracle_cv_mean,

            "improvement_vs_fixed_pct":
                gain_vs_fixed_pct,

            "oracle_gap_pct":
                oracle_gap_pct,

            "fold_accuracy_std":
                float(
                    np.std(
                        fold_acc,
                        ddof=1,
                    )
                ),
        })


# ============================================================
# Save
# ============================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    RESULT_CSV,
    index=False,
)

print("\n" + "=" * 100)
print("FINAL COMPARISON")
print("=" * 100)

print(
    result_df.to_string(
        index=False
    )
)

print(
    "\nRESULT CSV =",
    RESULT_CSV
)

print(
    "\nSUCCESS = True"
)
