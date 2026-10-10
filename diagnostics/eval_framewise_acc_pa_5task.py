#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import diagnostics.eval_acc_only_pa_intervention_5task as old


ROOT = Path(__file__).resolve().parents[1]

OUT_ROOT = (
    ROOT
    / "diagnostics/results/"
      "framewise_acc_pa_5task_current10457"
)

TASKS = {
    "task-0000": old.TASK_DIRS["task-0000"],
    "task-0001": old.TASK_DIRS["task-0001"],
    "task-0006": old.TASK_DIRS["task-0006"],
    "task-0008": old.TASK_DIRS["task-0008"],
    "task-0010": old.TASK_DIRS["task-0010"],
}

SEED = 42
MIN_POINTS = 8

# Established BEHAVIOR time base:
# 6 native pose samples -> 0.2 s action interval.
RATIO = 6
DT = 0.2


def cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        required=True,
        choices=list(TASKS),
    )
    return p.parse_args()


def arr(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def get_xyz(decoded, key):
    if key not in decoded:
        raise KeyError(
            f"Missing {key}; keys={list(decoded.keys())[:50]}"
        )

    x = arr(decoded[key]).astype(np.float64)

    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]

    if x.ndim != 2 or x.shape[1] < 3:
        raise RuntimeError(
            f"{key}: unexpected shape {x.shape}"
        )

    return x[:, :3]


def choose_dominant_xyz(right, left):
    def travel(x):
        if len(x) < 2:
            return 0.0
        return float(
            np.linalg.norm(
                np.diff(x, axis=0),
                axis=1,
            ).sum()
        )

    return right if travel(right) >= travel(left) else left


def local_acc_series(xyz):
    """
    Frame-wise translational acceleration.

    IMPORTANT:
    official BEHAVIOR dataloader already returns the
    11-step model/action sequence.

    Therefore DO NOT apply [::6] again.

    Established model time step:
        dt = 0.2 s
    """

    x = np.asarray(
        xyz,
        dtype=np.float64,
    )

    if (
        x.ndim != 2
        or x.shape[1] < 3
        or len(x) < 3
    ):
        return np.array(
            [],
            dtype=np.float64,
        )

    x = x[:, :3]

    a = (
        x[2:]
        - 2.0 * x[1:-1]
        + x[:-2]
    ) / (DT ** 2)

    mag = np.linalg.norm(
        a,
        axis=1,
    )

    out = np.full(
        len(x),
        np.nan,
        dtype=np.float64,
    )

    out[1:-1] = mag

    # Edge frames: nearest valid acceleration.
    out[0] = out[1]
    out[-1] = out[-2]

    return out


def resample_scalar_to_T(x, T):
    """
    Align local acceleration sequence to model future-frame
    normalized temporal coordinates.
    """

    x = np.asarray(x, dtype=np.float64)

    if len(x) < 2:
        return np.full(T, np.nan)

    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, T)

    return np.interp(dst, src, x)


def main():

    task = cli().task

    out_dir = OUT_ROOT / task
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args = old.parse_args(
        skip_command_line=True
    )

    args.seed = SEED
    args.model_path = old.MODEL_PATH
    args.norm_stats_path = old.NORM_STATS
    args.domains = ["behavior"]
    args.data_dirs = [TASKS[task]]
    args.batch_size = 1
    args.num_workers = 0
    args.eval_num_workers = 0
    args.eval_num_batches = -1
    args.device = "cuda"
    args.eval_skip_viz = True

    tester = old.Tester(args)
    tester.model.eval()

    domain = "behavior"
    device = tester.device

    urdf_path = old.resolve_robot_urdf(
        domain
    )

    robot_sampler = old.TorchRobotSampler(
        urdf_path=urdf_path,
        gripper_only=old.RELEASE_GRIPPER_ONLY,
        device="cpu",
    )

    force_single_arm = False

    if getattr(
        args,
        "_eval_override_active",
        False,
    ):
        force_single_arm = (
            domain
            in getattr(
                args,
                "eval_force_single_arm_domains",
                set(),
            )
        )

    old.omap.pmod.grid_sample_transform = (
        old.omap.grid_sample_with_object_ids
    )

    old.omap.pmod.gather = (
        old.omap.gather_with_object_ids
    )

    old.reset_seed(SEED)

    shard_paths = old.dlmod.gather_shard_paths(
        TASKS[task],
        "test",
        rank=tester.rank,
    )

    random.Random(
        SEED + tester.rank
    ).shuffle(shard_paths)

    dataset = old.wds.WebDataset(
        shard_paths,
        shardshuffle=False,
        detshuffle=True,
        resampled=False,
        nodesplitter=(
            old.wds.split_by_node
            if tester.world_size > 1
            else None
        ),
        seed=SEED + tester.rank,
        handler=old.wds.warn_and_continue,
    )

    rows = []
    nclip = 0

    try:
        with torch.inference_mode():

            for raw in dataset:

                decoded = old.decode_data(
                    raw,
                    domain=domain,
                )

                key = str(
                    decoded.get(
                        "__key__",
                        raw.get("__key__", ""),
                    )
                )

                # ----------------------------------------
                # Local Action acceleration
                # ----------------------------------------

                right = get_xyz(
                    decoded,
                    "right_gripper_pose",
                )

                left = get_xyz(
                    decoded,
                    "left_gripper_pose",
                )

                eef = choose_dominant_xyz(
                    right,
                    left,
                )

                local_acc = local_acc_series(
                    eef
                )

                if len(local_acc) < 2:
                    continue

                # ----------------------------------------
                # Official PointWorld sample
                # ----------------------------------------

                built = (
                    old.omap.build_flow_sample_with_object_ids(
                        decoded,
                        domain=domain,
                        robot_sampler=robot_sampler,
                        max_robot_points=args.max_robot_points,
                        deterministic=True,
                        seed=SEED + tester.rank,
                        force_single_arm=force_single_arm,
                    )
                )

                sample = old.release_process(
                    built,
                    args,
                    domain,
                    tester.rank,
                )

                batch_cpu = old.custom_collate_fn(
                    [sample],
                    args=args,
                )

                batch_gpu = old.to_device(
                    batch_cpu,
                    device,
                )

                out = tester.model(
                    batch_gpu,
                    training=False,
                )

                pred = (
                    out["scene_flows"][0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                gt = (
                    old.arr(
                        batch_cpu[
                            "gt_scene_flows"
                        ]
                    )[0]
                )

                object_ids = (
                    old.mask_array(
                        batch_cpu[
                            "scene_object_id"
                        ]
                    )[0]
                )

                context = (
                    old.mask_array(
                        batch_cpu[
                            "scene_context_mask"
                        ]
                    )[0]
                    .astype(bool)
                )

                exists = (
                    old.mask_array(
                        batch_cpu[
                            "scene_exists"
                        ]
                    )[0]
                    .astype(bool)
                )

                supervised = (
                    old.mask_array(
                        batch_cpu[
                            "scene_supervised_mask"
                        ]
                    )[0]
                    .astype(bool)
                )

                moved = (
                    old.mask_array(
                        batch_cpu[
                            "scene_moved_mask"
                        ]
                    )[0]
                    .astype(bool)
                )

                valid = (
                    (~context)
                    & exists
                    & supervised
                )

                T = gt.shape[0]

                acc_T = resample_scalar_to_T(
                    local_acc,
                    T,
                )

                table = (
                    old.omap.OBJECT_TABLE[key]
                )

                clip_had = False

                for oid, mesh_name in table.items():

                    oid = int(oid)

                    omask = (
                        object_ids == oid
                    )

                    moved_support = int(
                        (
                            omask
                            & moved
                            & valid
                        ).sum()
                    )

                    if moved_support <= 0:
                        continue

                    for t in range(1, T):

                        m = (
                            omask[t]
                            & valid[t]
                        )

                        if int(m.sum()) < MIN_POINTS:
                            continue

                        gt_t = gt[t, m]

                        centered = (
                            gt_t
                            - gt_t.mean(
                                axis=0,
                                keepdims=True,
                            )
                        )

                        if (
                            np.linalg.matrix_rank(
                                centered,
                                tol=1e-8,
                            )
                            < 2
                        ):
                            continue

                        try:
                            pa, _ = (
                                old.p_mpjpe_single(
                                    pred[t, m],
                                    gt_t,
                                )
                            )
                        except Exception:
                            continue

                        if not np.isfinite(
                            acc_T[t]
                        ):
                            continue

                        rows.append({
                            "task": task,
                            "clip": key,
                            "object":
                                old.short_name(
                                    mesh_name
                                ),
                            "frame": t,
                            "local_acc_mps2":
                                float(
                                    acc_T[t]
                                ),
                            "pa_mpjpe_mm":
                                float(pa)
                                * 1000.0,
                        })

                        clip_had = True

                if clip_had:
                    nclip += 1

                if (
                    nclip % 100 == 0
                    and nclip > 0
                ):
                    print(
                        f"[PROGRESS] "
                        f"{task} clips={nclip} "
                        f"rows={len(rows)}"
                    )

    finally:

        old.omap.pmod.grid_sample_transform = (
            old.omap.ORIG_GRID_SAMPLE
        )

        old.omap.pmod.gather = (
            old.omap.ORIG_GATHER
        )

    if nclip == 0 or len(rows) == 0:
        raise RuntimeError(
            f"{task}: ZERO eligible clips / rows. "
            "Do not mark this task complete."
        )

    out_csv = (
        out_dir
        / "framewise_acc_pa.csv"
    )

    pd.DataFrame(rows).to_csv(
        out_csv,
        index=False,
    )

    print("=" * 80)
    print("TASK =", task)
    print("ELIGIBLE CLIPS =", nclip)
    print("FRAME ROWS =", len(rows))
    print("SAVED =", out_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()
