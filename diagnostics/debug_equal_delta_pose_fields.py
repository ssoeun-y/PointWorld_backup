#!/usr/bin/env python3

import os

os.environ.setdefault("PW_TARGET_TASKS", "task-0000")

from arguments import parse_args
from evaluation.tester import Tester
import dataset_components.dataloader as dlmod

from diagnostics import eval_prediction_shift_paired as old


TARGETS = {
    "task-0000_episode_00000180-245:256",
    "task-0000_episode_00000180-300:311",
}


ORIG = old.OFFICIAL_BUILD


def debug_builder(
    sample,
    domain,
    robot_sampler,
    max_robot_points,
    deterministic=False,
    seed=None,
    force_single_arm=False,
):
    raw_key = str(sample.get("__key__", ""))

    if raw_key in TARGETS:
        print(
            "\nRAW",
            raw_key,
            "right=", "right_gripper_pose" in sample,
            "left=", "left_gripper_pose" in sample,
        )

    out = ORIG(
        sample,
        domain=domain,
        robot_sampler=robot_sampler,
        max_robot_points=max_robot_points,
        deterministic=deterministic,
        seed=seed,
        force_single_arm=force_single_arm,
    )

    key = str(out.get("__key__", raw_key))

    if key in TARGETS:
        print(
            "AFTER_OFFICIAL_BUILD",
            key,
            "right=", "right_gripper_pose" in out,
            "left=", "left_gripper_pose" in out,
        )

    return out


def main():
    args = parse_args()
    tester = Tester(args)

    dlmod.build_flow_sample = debug_builder

    loader, _ = dlmod.build_dataloader(
        tester.args,
        mode="test",
        rank=tester.rank,
        world_size=tester.world_size,
        override_splits="test",
    )

    for _ in loader:
        pass


if __name__ == "__main__":
    main()
