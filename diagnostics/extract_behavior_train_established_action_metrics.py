#!/usr/bin/env python3

from copy import deepcopy
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from arguments import parse_args
from dataset_components.dataloader import build_dataloader

# IMPORTANT:
# Reuse the EXACT established implementation.
from diagnostics.run_established_action_metrics_5task import (
    task_from_clip,
    decode_key,
    to_numpy,
    established_metrics_for_clip,
)


# ============================================================
# Fixed BEHAVIOR 5-task TRAIN population
# ============================================================

ROOTS = [
    (
        "task0000",
        "/home/soenchun/cse/PointWorld_workspace/"
        "datasets/behavior/wds_task0000",
        {"task-0000"},
        5202,
    ),
    (
        "task0001",
        "/home/soenchun/cse/PointWorld_workspace/"
        "PointWorld/diagnostics/data/"
        "pointworld_behavior_task0001_restored/behavior/wds",
        {"task-0001"},
        22414,
    ),
    (
        "task0006_0008_0010",
        "/home/soenchun/cse/PointWorld_workspace/"
        "PointWorld/diagnostics/data/"
        "pointworld_behavior_multitask_restored/behavior/wds",
        {"task-0006", "task-0008", "task-0010"},
        64659,
    ),
]

EXPECTED_TOTAL = 92275

OUTDIR = Path(
    "diagnostics/results/"
    "behavior_train_established_action_metrics_5task"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

FINAL_OUT = OUTDIR / "per_sample_train_established_metrics_92275.csv"


def extract_root(
    tag,
    root,
    expected_tasks,
    expected_n,
    args0,
):

    out = OUTDIR / f"{tag}__train_metrics.csv"

    print()
    print("=" * 110)
    print("ROOT =", root)
    print("EXPECTED N =", expected_n)
    print("EXPECTED TASKS =", sorted(expected_tasks))
    print("=" * 110)

    # --------------------------------------------------------
    # Safe resume:
    # reuse only if the completed partial file is internally valid.
    # --------------------------------------------------------

    if out.exists():
        old = pd.read_csv(out)

        tasks = set(
            old["task"]
            .dropna()
            .astype(str)
            .unique()
        )

        if (
            len(old) == expected_n
            and old["clip"].is_unique
            and tasks == expected_tasks
        ):
            print("VALID COMPLETE PARTIAL EXISTS — REUSING")
            print("FILE =", out)
            return old

        print("PARTIAL EXISTS BUT FAILED AUDIT — RECOMPUTING")
        print("existing N =", len(old))
        print("existing tasks =", sorted(tasks))

    args = deepcopy(args0)

    args.domains = ["behavior"]
    args.data_dirs = [root]

    # IMPORTANT:
    # Use evaluation preprocessing, but explicitly read TRAIN split.
    # This avoids training-time resampling/augmentation while preserving
    # the same official sample construction used for established metrics.
    loader, info = build_dataloader(
        args=args,
        mode="test",
        rank=0,
        world_size=1,
        override_splits="train",
        force_resampled_eval=False,
    )

    if loader is None:
        raise RuntimeError(
            f"Loader is None for {root}"
        )

    total = int(info["total_samples"])

    if total != expected_n:
        raise RuntimeError(
            f"Train population mismatch for {root}: "
            f"{total} != {expected_n}"
        )

    rows = []
    seen = set()
    tasks_seen = set()

    for batch in tqdm(
        loader,
        total=info["batches_per_rank"],
        desc=f"B1 {tag}",
    ):
        B = len(batch["__key__"])

        for b in range(B):

            clip = decode_key(
                batch["__key__"][b]
            )

            if clip in seen:
                raise RuntimeError(
                    f"Duplicate clip in {tag}: {clip}"
                )

            task = task_from_clip(clip)

            if task is None:
                raise RuntimeError(
                    f"Could not recover task from clip: {clip}"
                )

            tasks_seen.add(task)

            right_pose = to_numpy(
                batch["right_gripper_pose"][b]
            )

            left_pose = to_numpy(
                batch["left_gripper_pose"][b]
            )

            metrics = established_metrics_for_clip(
                right_pose,
                left_pose,
            )

            rows.append({
                "clip": clip,
                "task": task,
                **metrics,
            })

            seen.add(clip)

    df = pd.DataFrame(rows)

    if len(df) != expected_n:
        raise RuntimeError(
            f"{tag}: extracted {len(df)} "
            f"!= expected {expected_n}"
        )

    if not df["clip"].is_unique:
        raise RuntimeError(
            f"{tag}: duplicate clip keys"
        )

    if tasks_seen != expected_tasks:
        raise RuntimeError(
            f"{tag}: task mismatch "
            f"expected={sorted(expected_tasks)}, "
            f"seen={sorted(tasks_seen)}"
        )

    required = [
        "rms_acceleration",
        "mean_squared_jerk",
    ]

    for c in required:
        if c not in df.columns:
            raise RuntimeError(
                f"Missing established metric column: {c}"
            )

        if df[c].isna().any():
            raise RuntimeError(
                f"NaN detected in {c}"
            )

    df.to_csv(out, index=False)

    print()
    print("ROOT AUDIT PASSED")
    print("N =", len(df))
    print(df.groupby("task").size())
    print("SAVED =", out)

    return df


def main():

    args0 = parse_args(
        skip_command_line=True
    )

    args0.seed = 42
    args0.batch_size = 1

    # This job is I/O + CPU preprocessing bound.
    # Moderate worker count to improve throughput safely.
    args0.num_workers = 8
    args0.eval_num_workers = 8

    args0.deterministic_data = True

    all_df = []

    for (
        tag,
        root,
        expected_tasks,
        expected_n,
    ) in ROOTS:

        part = extract_root(
            tag=tag,
            root=root,
            expected_tasks=expected_tasks,
            expected_n=expected_n,
            args0=args0,
        )

        all_df.append(part)

    df = pd.concat(
        all_df,
        ignore_index=True,
    )

    print()
    print("=" * 110)
    print("FINAL B1 AUDIT")
    print("=" * 110)

    print("TOTAL =", len(df))
    print()
    print(df.groupby("task").size())

    if len(df) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Final total {len(df)} "
            f"!= {EXPECTED_TOTAL}"
        )

    if not df["clip"].is_unique:
        dup = df[df["clip"].duplicated(
            keep=False
        )]

        raise RuntimeError(
            "Cross-root duplicate clip keys:\n"
            + dup[
                ["task", "clip"]
            ].head(30).to_string(index=False)
        )

    expected_tasks = {
        "task-0000",
        "task-0001",
        "task-0006",
        "task-0008",
        "task-0010",
    }

    actual_tasks = set(
        df["task"].unique()
    )

    if actual_tasks != expected_tasks:
        raise RuntimeError(
            f"Final task mismatch: {actual_tasks}"
        )

    df.to_csv(
        FINAL_OUT,
        index=False,
    )

    print()
    print("RMS ACC QUANTILES")
    print(
        df["rms_acceleration"].quantile(
            [0.50, 0.75, 0.90, 0.95, 0.975, 0.99]
        )
    )

    print()
    print("MSJ QUANTILES")
    print(
        df["mean_squared_jerk"].quantile(
            [0.50, 0.75, 0.90, 0.95, 0.975, 0.99]
        )
    )

    print()
    print("FINAL SAVED =", FINAL_OUT)
    print("SUCCESS = True")


if __name__ == "__main__":
    main()
