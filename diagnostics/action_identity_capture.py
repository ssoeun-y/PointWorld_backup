#!/usr/bin/env python3

import json
import os
import runpy
from pathlib import Path

import numpy as np
import torch

import dataset_components.robot as robot_module
from evaluation.tester import Tester


MODE = os.environ.get("ACTION_IDENTITY_MODE")
RUN_ID = os.environ.get("ACTION_IDENTITY_RUN")

if MODE not in {"baseline", "identity"}:
    raise RuntimeError(
        "ACTION_IDENTITY_MODE must be baseline or identity"
    )

if not RUN_ID:
    raise RuntimeError("ACTION_IDENTITY_RUN is required")


HOOK_STATE = {
    "call_count": 0,
    "joint_copy_exact": True,
    "gripper_copy_exact": True,
}


if MODE == "identity":
    original_get_robot_flows_droid = (
        robot_module._get_robot_flows_droid
    )

    def identity_get_robot_flows_droid(
        sample,
        robot_sampler,
        max_robot_points,
        seed=None,
    ):
        """
        Identity diagnostic hook:
        copy joint/gripper trajectories without changing any value,
        then call the official function unchanged.
        """
        HOOK_STATE["call_count"] += 1

        if "joint_positions" not in sample:
            raise KeyError("joint_positions missing before robot flow generation")

        if "gripper_positions" not in sample:
            raise KeyError(
                "gripper_positions missing before robot flow generation"
            )

        original_joint = np.asarray(sample["joint_positions"])
        original_gripper = np.asarray(sample["gripper_positions"])

        hooked_sample = dict(sample)
        hooked_sample["joint_positions"] = original_joint.copy()
        hooked_sample["gripper_positions"] = original_gripper.copy()

        HOOK_STATE["joint_copy_exact"] &= np.array_equal(
            original_joint,
            hooked_sample["joint_positions"],
        )
        HOOK_STATE["gripper_copy_exact"] &= np.array_equal(
            original_gripper,
            hooked_sample["gripper_positions"],
        )

        # Call the original official implementation.
        return original_get_robot_flows_droid(
            hooked_sample,
            robot_sampler,
            max_robot_points,
            seed=seed,
        )

    robot_module._get_robot_flows_droid = (
        identity_get_robot_flows_droid
    )


@torch.no_grad()
def capture_first_batch(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):
    self.model.eval()

    batch = next(iter(dl))

    device_batch = {
        key: value.to(self.device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }

    outputs = self.model(device_batch, training=False)

    if MODE == "identity":
        if HOOK_STATE["call_count"] < 1:
            raise RuntimeError(
                "Identity hook was never called. "
                "The diagnostic interception point is invalid."
            )

        if not HOOK_STATE["joint_copy_exact"]:
            raise RuntimeError("Identity joint copy changed values")

        if not HOOK_STATE["gripper_copy_exact"]:
            raise RuntimeError("Identity gripper copy changed values")

    batch_keys = [
        "joint_positions",
        "gripper_positions",
        "robot_flows",
        "robot_features",
        "robot_exists",
        "scene_flows",
        "scene_features",
        "scene_exists",
        "scene_context_mask",
        "cam0_initial_rgb",
        "cam0_initial_depth",
        "cam0_intrinsic",
        "cam0_extrinsic",
    ]

    missing_batch_keys = [
        key for key in batch_keys
        if key not in device_batch
    ]

    if missing_batch_keys:
        raise RuntimeError(
            f"Missing required batch keys: {missing_batch_keys}"
        )

    output_keys = [
        "pred",
        "scene_flows",
        "scene_relative",
        "scene_relative_norm",
        "confidence",
        "log_var",
    ]

    payload = {
        "mode": MODE,
        "run_id": RUN_ID,
        "sample_key": list(device_batch["__key__"]),
        "hook_state": dict(HOOK_STATE),
        "batch": {
            key: device_batch[key].detach().cpu()
            for key in batch_keys
        },
        "outputs": {
            key: outputs[key].detach().cpu()
            for key in output_keys
            if key in outputs
            and isinstance(outputs[key], torch.Tensor)
        },
    }

    output_dir = (
        Path("diagnostics/results")
        / f"action_identity_{RUN_ID}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{MODE}.pt"

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}"
        )

    torch.save(payload, output_path)

    metadata = {
        "mode": MODE,
        "run_id": RUN_ID,
        "sample_key": payload["sample_key"],
        "hook_state": payload["hook_state"],
        "batch_shapes": {
            key: list(value.shape)
            for key, value in payload["batch"].items()
        },
        "output_shapes": {
            key: list(value.shape)
            for key, value in payload["outputs"].items()
        },
    }

    with open(
        output_dir / f"{MODE}.json",
        "w",
    ) as file:
        json.dump(metadata, file, indent=2)

    print()
    print("=" * 72)
    print(f"Action identity capture completed: {MODE}")
    print(f"Output: {output_path}")
    print(f"Sample: {payload['sample_key']}")
    print(f"Hook calls: {HOOK_STATE['call_count']}")
    print("=" * 72)

    return {
        f"diagnostic/{split}/captured_batches": 1,
    }


Tester._accumulate_metrics = capture_first_batch
runpy.run_path("eval.py", run_name="__main__")
