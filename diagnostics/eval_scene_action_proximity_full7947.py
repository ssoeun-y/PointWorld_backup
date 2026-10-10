#!/usr/bin/env python3

import os

# Required BEFORE importing validated helper.
os.environ.setdefault(
    "PW_TARGET_TASKS",
    "task-0000,task-0001,task-0006,task-0008,task-0010",
)
os.environ.setdefault("PW_TAG", "proximity_full7947")
os.environ.setdefault("PW_CONDITION", "original")

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import webdataset as wds

from arguments import parse_args

import dataset_components.dataloader as dlmod
from dataset_components.decoders import decode_data
from dataset_components.cameras import sample_cameras
from dataset_components.robot import canonicalize_gripper_keys_and_flags
from dataset_components.pipeline import apply_release_pipeline_to_sample
from dataset_components.constants import RELEASE_GRIPPER_ONLY

from robot_sampler import RobotSampler as TorchRobotSampler
from utils import resolve_robot_urdf

from diagnostics import eval_prediction_shift_paired as old
from dataset_components.transforms import compute_robot_distances


ROOT = Path(__file__).resolve().parents[1]

REPAIR_CSV = (
    ROOT
    / "diagnostics/results/rigid_projection_repair_current10457"
    / "combined_5task_repair_per_clip.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/scene_action_proximity_full7947"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_COMBINED = (
    OUTDIR
    / "scene_action_proximity_full7947.csv"
)

EXPECTED = {
    "task-0000": 383,
    "task-0001": 2307,
    "task-0006": 1823,
    "task-0008": 1824,
    "task-0010": 1610,
}

# Exact previously validated/restored BEHAVIOR WDS roots.
DATA_GROUPS = [
    {
        "name": "task0000",
        "tasks": ["task-0000"],
        "path":
            "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
    },
    {
        "name": "task0001",
        "tasks": ["task-0001"],
        "path":
            "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
            "diagnostics/data/pointworld_behavior_task0001_restored/behavior/wds",
    },
    {
        "name": "task0006_0008_0010",
        "tasks": [
            "task-0006",
            "task-0008",
            "task-0010",
        ],
        "path":
            "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
            "diagnostics/data/pointworld_behavior_multitask_restored/behavior/wds",
    },
]


def release_process(sample, args):

    sample = sample_cameras(
        sample,
        min_num_cameras=args.eval_min_num_cameras,
        max_num_cameras=args.eval_max_num_cameras,
        deterministic=True,
        seed=args.seed,
    )

    sample = canonicalize_gripper_keys_and_flags(
        sample
    )

    sample = apply_release_pipeline_to_sample(
        sample,
        domain="behavior",
        mode="test",
        args=args,
        has_bimanual_robot=True,
        rank=0,
        include_scene_data=False,
        skip_scene_sampling=False,
    )

    return sample


def apply_original_identity_phase(out):

    T = len(out["robot_flows"])

    phase = np.linspace(
        0.0,
        1.0,
        T,
    )

    original = out["robot_flows"].copy()

    out["robot_flows"] = old.resample(
        out["robot_flows"],
        phase,
    )

    if not np.allclose(
        out["robot_flows"],
        original,
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError(
            f"{out.get('__key__')}: "
            "identity phase changed robot_flows"
        )

    if "robot_normals" in out:

        out["robot_normals"] = old.resample(
            out["robot_normals"],
            phase,
            normalize=True,
        )

    for name in [
        "right_gripper_open",
        "left_gripper_open",
        "gripper_positions",
    ]:

        if name not in out:
            continue

        a = np.asarray(out[name])

        if (
            a.ndim >= 1
            and a.shape[0] == T
        ):
            out[name] = old.resample(
                a,
                phase,
            )

    return out


def main():

    repair = pd.read_csv(REPAIR_CSV)

    required = {"clip", "task"}

    if not required.issubset(repair.columns):
        raise RuntimeError(
            f"repair CSV missing columns: "
            f"{required - set(repair.columns)}"
        )

    if repair["clip"].duplicated().any():
        raise RuntimeError(
            "repair CSV contains duplicate clips"
        )

    selected = [
        x.strip()
        for x in os.environ.get(
            "PW_FULL_PROX_TASKS",
            "task-0000,task-0001,"
            "task-0006,task-0008,task-0010",
        ).split(",")
        if x.strip()
    ]

    unknown = [
        x for x in selected
        if x not in EXPECTED
    ]

    if unknown:
        raise RuntimeError(
            f"unknown tasks: {unknown}"
        )

    targets_by_task = {}
    task_by_key = {}

    for task in selected:

        q = repair[
            repair["task"].astype(str) == task
        ].copy()

        keys = set(
            q["clip"].astype(str)
        )

        if len(keys) != EXPECTED[task]:
            raise RuntimeError(
                f"{task}: "
                f"N={len(keys)} "
                f"expected={EXPECTED[task]}"
            )

        targets_by_task[task] = keys

        for key in keys:

            if key in task_by_key:
                raise RuntimeError(
                    f"duplicate key across tasks: {key}"
                )

            task_by_key[key] = task

    all_targets = set(task_by_key)

    if len(all_targets) != sum(
        EXPECTED[t]
        for t in selected
    ):
        raise RuntimeError(
            "total target count mismatch"
        )

    # No Tester().
    # No model loading.
    # Explicitly freeze the established eval contract.
    args = parse_args()

    args.seed = 42
    args.eval_min_num_cameras = 2
    args.eval_max_num_cameras = 2

    print("=" * 100)
    print("FULL CURRENT PROXIMITY EXTRACTION")
    print("=" * 100)
    print("PointWorld forward = 0")
    print("GPU required = False")
    print("seed = 42")
    print("cameras = 2")
    print("targets =", len(all_targets))

    for task in selected:
        print(
            task,
            "=",
            len(targets_by_task[task]),
        )

    urdf_path = resolve_robot_urdf(
        "behavior"
    )

    robot_sampler = TorchRobotSampler(
        urdf_path=urdf_path,
        gripper_only=RELEASE_GRIPPER_ONLY,
        device="cpu",
    )

    force_single_arm = False

    rows = []
    found = set()

    for group in DATA_GROUPS:

        active_tasks = [
            t
            for t in group["tasks"]
            if t in selected
        ]

        if not active_tasks:
            continue

        group_targets = set()

        for task in active_tasks:
            group_targets |= targets_by_task[task]

        data_dir = group["path"]

        print()
        print("=" * 100)
        print("DATA GROUP =", group["name"])
        print("PATH =", data_dir)
        print("TARGETS =", len(group_targets))
        print("=" * 100)

        shard_paths = dlmod.gather_shard_paths(
            data_dir,
            "test",
            rank=0,
        )

        if not shard_paths:
            raise RuntimeError(
                f"{group['name']}: no test shards "
                f"at {data_dir}"
            )

        random.Random(
            args.seed
        ).shuffle(shard_paths)

        dataset = wds.WebDataset(
            shard_paths,
            shardshuffle=False,
            detshuffle=True,
            resampled=False,
            nodesplitter=None,
            seed=args.seed,
            handler=wds.warn_and_continue,
        )

        found_group = set()
        scanned = 0

        with torch.inference_mode():

            for raw in dataset:

                scanned += 1

                raw_key = str(
                    raw.get(
                        "__key__",
                        "",
                    )
                )

                # Early filtering before decode/build.
                if raw_key not in group_targets:
                    continue

                if raw_key in found:
                    raise RuntimeError(
                        f"duplicate target: {raw_key}"
                    )

                decoded = decode_data(
                    raw,
                    domain="behavior",
                )

                key = str(
                    decoded.get(
                        "__key__",
                        raw_key,
                    )
                )

                if key != raw_key:
                    raise RuntimeError(
                        f"{raw_key}: "
                        f"decoded key mismatch={key}"
                    )

                built = old.OFFICIAL_BUILD(
                    decoded,
                    domain="behavior",
                    robot_sampler=robot_sampler,
                    max_robot_points=args.max_robot_points,
                    deterministic=True,
                    seed=args.seed,
                    force_single_arm=force_single_arm,
                )

                built_key = str(
                    built.get(
                        "__key__",
                        "",
                    )
                )

                if built_key != key:
                    raise RuntimeError(
                        f"{key}: "
                        f"built key mismatch={built_key}"
                    )

                # CONDITION=original:
                # exact identity temporal phase.
                built = apply_original_identity_phase(
                    copy.deepcopy(built)
                )

                sample = release_process(
                    built,
                    args,
                )

                scene0 = (
                    sample["scene_flows"][0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                robot_seq = (
                    sample["robot_flows"]
                    .detach()
                    .cpu()
                    .numpy()
                )

                all_d = []

                for t in range(
                    robot_seq.shape[0]
                ):

                    dt = np.asarray(
                        compute_robot_distances(
                            scene0,
                            robot_seq[t],
                        ),
                        dtype=np.float64,
                    )

                    dt = dt[
                        np.isfinite(dt)
                    ]

                    if dt.size:
                        all_d.append(dt)

                if not all_d:
                    raise RuntimeError(
                        f"{key}: "
                        "no finite dist2robot"
                    )

                all_d = np.concatenate(
                    all_d,
                    axis=0,
                )

                task = task_by_key[key]

                rows.append(
                    {
                        "clip": key,
                        "task": task,
                        "scene_action_dist_p05_native":
                            float(
                                np.quantile(
                                    all_d,
                                    0.05,
                                )
                            ),
                        "scene_action_dist_p10_native":
                            float(
                                np.quantile(
                                    all_d,
                                    0.10,
                                )
                            ),
                        "scene_action_dist_p50_native":
                            float(
                                np.quantile(
                                    all_d,
                                    0.50,
                                )
                            ),
                        "scene_action_dist_min_native":
                            float(
                                np.min(all_d)
                            ),
                    }
                )

                found.add(key)
                found_group.add(key)

                n = len(found_group)

                if (
                    n <= 5
                    or n % 100 == 0
                    or n == len(group_targets)
                ):
                    print(
                        f"[{n}/{len(group_targets)}] "
                        f"{key} | {task}"
                    )

                if (
                    len(found_group)
                    == len(group_targets)
                ):
                    break

        missing_group = (
            group_targets
            - found_group
        )

        if missing_group:
            raise RuntimeError(
                f"{group['name']}: "
                f"missing={len(missing_group)} "
                f"first={sorted(missing_group)[:10]}"
            )

        print(
            "GROUP COMPLETE | found=",
            len(found_group),
            "| scanned=",
            scanned,
        )

    missing = all_targets - found

    if missing:
        raise RuntimeError(
            f"FINAL missing={len(missing)} "
            f"first={sorted(missing)[:10]}"
        )

    d = pd.DataFrame(rows)

    if d["clip"].duplicated().any():
        raise RuntimeError(
            "duplicate output clips"
        )

    if len(d) != len(all_targets):
        raise RuntimeError(
            f"output N={len(d)} "
            f"expected={len(all_targets)}"
        )

    got_counts = (
        d["task"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_selected = {
        t: EXPECTED[t]
        for t in selected
    }

    if got_counts != expected_selected:
        raise RuntimeError(
            f"task counts mismatch: "
            f"{got_counts} != "
            f"{expected_selected}"
        )

    d = d.sort_values(
        ["task", "clip"]
    ).reset_index(drop=True)

    d.to_csv(
        OUT_COMBINED,
        index=False,
    )

    for task in selected:

        q = d[
            d["task"] == task
        ].copy()

        q.to_csv(
            OUTDIR
            / f"{task}__proximity.csv",
            index=False,
        )

    print()
    print("=" * 100)
    print("SUCCESS")
    print("=" * 100)
    print("N =", len(d))
    print("TASK COUNTS =", got_counts)

    for task in selected:

        q = d[
            d["task"] == task
        ]

        print(
            task,
            "| N=",
            len(q),
            "| median p05=",
            q[
                "scene_action_dist_p05_native"
            ].median(),
        )

    print(
        "OUTPUT =",
        OUT_COMBINED,
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
