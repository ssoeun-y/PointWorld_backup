import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dl
from dataset_components.cameras import sample_cameras as official_sample_cameras


TASK = os.environ["PW_TASK"]

ROOT = Path(
    f"diagnostics/results/action_conditioned_active_world_model/"
    f"fixed_camera_pair_task{TASK}"
)

OUT = ROOT / f"task{TASK}_head_scene_action_query_features.csv"

HL = ROOT / f"task{TASK}_pair_HL__per_sample.csv"
HR = ROOT / f"task{TASK}_pair_HR__per_sample.csv"

DT = 0.2


def load_metric(path, pair):
    df = pd.read_csv(path)

    metric = (
        "behavior/l2_moved/mean"
        if "behavior/l2_moved/mean" in df.columns
        else "l2_moved/mean"
    )

    return (
        df[["clip", metric]]
        .rename(columns={metric: f"err_{pair}"})
    )


labels = (
    load_metric(HL, "HL")
    .merge(
        load_metric(HR, "HR"),
        on="clip",
        validate="one_to_one",
    )
    .dropna(
        subset=["err_HL", "err_HR"]
    )
)

target_clips = set(labels["clip"])


def episode_from_clip(clip):
    m = re.match(
        r"^(task-\d+_episode_\d+)",
        str(clip),
    )
    return m.group(1) if m else str(clip)


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return (
        float(np.sqrt(np.mean(x ** 2)))
        if x.size
        else 0.0
    )


def p95(x):
    x = np.asarray(x, dtype=np.float64)
    return (
        float(np.percentile(x, 95))
        if x.size
        else 0.0
    )


def quaternion_step_angles(q):
    q = np.asarray(q, dtype=np.float64)

    if len(q) < 2:
        return np.zeros(0)

    q = q / np.clip(
        np.linalg.norm(
            q,
            axis=1,
            keepdims=True,
        ),
        1e-12,
        None,
    )

    d = np.sum(
        q[:-1] * q[1:],
        axis=1,
    )

    return 2 * np.arccos(
        np.clip(
            np.abs(d),
            0,
            1,
        )
    )


def trajectory_features(prefix, pose):
    pose = np.asarray(
        pose,
        dtype=np.float64,
    )

    xyz = pose[:, :3]
    dp = np.diff(xyz, axis=0)
    step = np.linalg.norm(dp, axis=1)
    speed = np.linalg.norm(dp / DT, axis=1)

    if len(xyz) >= 3:
        acc = np.diff(
            xyz,
            n=2,
            axis=0,
        ) / (DT ** 2)

        acc_mag = np.linalg.norm(
            acc,
            axis=1,
        )
    else:
        acc_mag = np.zeros(0)

    if len(xyz) >= 4:
        jerk = np.diff(
            xyz,
            n=3,
            axis=0,
        ) / (DT ** 3)

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

        if na > 1e-12 and nb > 1e-12:
            c = np.dot(a, b) / (na * nb)
            turns.append(
                np.arccos(
                    np.clip(c, -1, 1)
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

    dq = np.diff(q, axis=0)

    ddq = (
        np.diff(
            q,
            n=2,
            axis=0,
        )
        if len(q) >= 3
        else np.zeros(
            (0, q.shape[1])
        )
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


def fixed_head_selector(
    sample,
    min_num_cameras=1,
    max_num_cameras=None,
    deterministic=False,
    seed=None,
):
    prefixes = sorted({
        k.rsplit(
            "_initial_rgb",
            1,
        )[0]
        for k in sample
        if k.endswith(
            "_initial_rgb"
        )
    })

    wanted = "camera_head"

    if wanted not in prefixes:
        raise RuntimeError(
            f"{sample.get('__key__')} "
            f"missing camera_head"
        )

    for prefix in prefixes:
        if prefix == wanted:
            continue

        for key in list(
            sample.keys()
        ):
            if key.startswith(
                prefix + "_"
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


if OUT.exists():
    print(
        "CACHE EXISTS:",
        OUT
    )
    raise SystemExit(0)


args = parse_args()
tester = Tester(args)

tester.model.eval()

tester.args.eval_min_num_cameras = 1
tester.args.eval_max_num_cameras = 1

dl.sample_cameras = fixed_head_selector

loader, _ = tester._build_eval_loader(
    "test"
)

rows = []

with torch.no_grad():

    for batch in loader:

        clip = str(
            batch["__key__"][0]
        )

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

        encoded = (
            tester.model
            .encode_scene_features(
                batch
            )
        )

        exists = (
            batch[
                "scene_exists"
            ][:, 0]
            .bool()
        )

        f = encoded[0][
            exists[0]
        ]

        row = {
            "clip": clip,
            "episode":
                episode_from_clip(
                    clip
                ),
            "task": TASK,
        }

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

        for j, x in enumerate(
            mean_feat
        ):
            row[
                f"scene_pw_mean_{j}"
            ] = float(x)

        for j, x in enumerate(
            max_feat
        ):
            row[
                f"scene_pw_max_{j}"
            ] = float(x)

        coords = (
            batch[
                "scene_flows"
            ][0, 0][
                exists[0]
            ]
            .float()
        )

        cm = (
            coords.mean(dim=0)
            .cpu()
            .numpy()
        )

        cs = (
            coords.std(dim=0)
            .cpu()
            .numpy()
        )

        for j, axis in enumerate(
            ["x", "y", "z"]
        ):
            row[
                f"scene_coord_mean_{axis}"
            ] = float(cm[j])

            row[
                f"scene_coord_std_{axis}"
            ] = float(cs[j])

        row[
            "scene_point_count"
        ] = int(
            f.shape[0]
        )

        row.update(
            trajectory_features(
                "left",
                batch[
                    "left_gripper_pose"
                ][0]
                .cpu()
                .numpy(),
            )
        )

        row.update(
            trajectory_features(
                "right",
                batch[
                    "right_gripper_pose"
                ][0]
                .cpu()
                .numpy(),
            )
        )

        row.update(
            gripper_features(
                "left",
                batch[
                    "left_gripper_open"
                ][0]
                .cpu()
                .numpy(),
            )
        )

        row.update(
            gripper_features(
                "right",
                batch[
                    "right_gripper_open"
                ][0]
                .cpu()
                .numpy(),
            )
        )

        row.update(
            joint_features(
                batch[
                    "joint_positions"
                ][0]
                .cpu()
                .numpy()
            )
        )

        rows.append(row)

        if len(rows) % 200 == 0:
            print(
                TASK,
                "features:",
                len(rows),
            )


df = pd.DataFrame(rows)

df.to_csv(
    OUT,
    index=False,
)

print(
    "TASK =", TASK,
    "| rows =", len(df),
    "| output =", OUT
)

print("SUCCESS = True")
