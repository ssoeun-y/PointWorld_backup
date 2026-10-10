#!/usr/bin/env python3

import json
import os
import runpy
from pathlib import Path

import numpy as np
import torch

import dataset_components.robot as robot_module
from evaluation.tester import Tester


RUN_ID = os.environ.get("ACTION_PERTURB_RUN")
JOINT_INDEX = int(os.environ.get("ACTION_PERTURB_JOINT", "3"))
AMPLITUDE = float(os.environ.get("ACTION_PERTURB_AMPLITUDE", "0.15"))

if not RUN_ID:
    raise RuntimeError("ACTION_PERTURB_RUN is required")

if JOINT_INDEX < 0 or JOINT_INDEX >= 7:
    raise ValueError(f"JOINT_INDEX must be 0..6, got {JOINT_INDEX}")


original_get_robot_flows_droid = robot_module._get_robot_flows_droid

HOOK_STATE = {
    "call_count": 0,
    "joint_index_zero_based": JOINT_INDEX,
    "joint_name": f"panda_joint{JOINT_INDEX + 1}",
    "requested_amplitude_rad": AMPLITUDE,
    "t0_exact": True,
    "final_exact": True,
    "within_joint_limit": True,
    "actual_max_abs_delta_rad": None,
    "joint_lower_limit": None,
    "joint_upper_limit": None,
}

CAPTURE_STATE = {}


def perturb_get_robot_flows_droid(
    sample,
    robot_sampler,
    max_robot_points,
    seed=None,
):
    HOOK_STATE["call_count"] += 1

    if "joint_positions" not in sample:
        raise KeyError("joint_positions missing")

    if "gripper_positions" not in sample:
        raise KeyError("gripper_positions missing")

    original_joint = np.asarray(
        sample["joint_positions"],
        dtype=np.float32,
    ).copy()

    original_gripper = np.asarray(
        sample["gripper_positions"],
        dtype=np.float32,
    ).copy()

    if original_joint.ndim != 2 or original_joint.shape[1] != 7:
        raise ValueError(
            f"Expected joint_positions (T,7), got {original_joint.shape}"
        )

    num_steps = original_joint.shape[0]

    # t=0과 마지막 timestep에서 0이고 중간에서 최대가 되는 smooth bump.
    phase = np.linspace(
        0.0,
        np.pi,
        num_steps,
        dtype=np.float32,
    )
    delta = AMPLITUDE * np.sin(phase)

    # 부동소수점 sin(pi) 오차를 제거해 양 끝 timestep을 정확히 보존한다.
    delta[0] = 0.0
    delta[-1] = 0.0

    perturbed_joint = original_joint.copy()
    perturbed_joint[:, JOINT_INDEX] += delta

    joint_name = f"panda_joint{JOINT_INDEX + 1}"

    if joint_name not in robot_sampler.joint_limits:
        raise KeyError(f"Joint limit unavailable: {joint_name}")

    lower, upper = robot_sampler.joint_limits[joint_name]
    lower = float(lower)
    upper = float(upper)

    HOOK_STATE["joint_lower_limit"] = lower
    HOOK_STATE["joint_upper_limit"] = upper

    perturbed_joint[:, JOINT_INDEX] = np.clip(
        perturbed_joint[:, JOINT_INDEX],
        lower,
        upper,
    )

    actual_delta = (
        perturbed_joint[:, JOINT_INDEX]
        - original_joint[:, JOINT_INDEX]
    )

    HOOK_STATE["actual_max_abs_delta_rad"] = float(
        np.max(np.abs(actual_delta))
    )
    HOOK_STATE["t0_exact"] = bool(
        np.array_equal(original_joint[0], perturbed_joint[0])
    )
    HOOK_STATE["final_exact"] = bool(
        np.array_equal(original_joint[-1], perturbed_joint[-1])
    )
    HOOK_STATE["within_joint_limit"] = bool(
        np.all(perturbed_joint[:, JOINT_INDEX] >= lower)
        and np.all(perturbed_joint[:, JOINT_INDEX] <= upper)
    )

    if not HOOK_STATE["t0_exact"]:
        raise RuntimeError("Perturbation changed t=0")

    if not HOOK_STATE["final_exact"]:
        raise RuntimeError("Perturbation changed final timestep")

    if not HOOK_STATE["within_joint_limit"]:
        raise RuntimeError("Perturbation violates joint limit")

    CAPTURE_STATE["original_joint_positions"] = original_joint
    CAPTURE_STATE["perturbed_joint_positions"] = perturbed_joint
    CAPTURE_STATE["joint_delta"] = perturbed_joint - original_joint

    hooked_sample = dict(sample)
    hooked_sample["joint_positions"] = perturbed_joint
    hooked_sample["gripper_positions"] = original_gripper

    # 공식 robot-flow 생성 함수를 그대로 호출한다.
    return original_get_robot_flows_droid(
        hooked_sample,
        robot_sampler,
        max_robot_points,
        seed=seed,
    )


robot_module._get_robot_flows_droid = perturb_get_robot_flows_droid


@torch.no_grad()
def capture_first_batch(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):
    self.model.eval()

    batch = next(iter(dl))

    if HOOK_STATE["call_count"] != 1:
        raise RuntimeError(
            f"Expected exactly one hook call, got {HOOK_STATE['call_count']}"
        )

    device_batch = {
        key: value.to(self.device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }

    outputs = self.model(device_batch, training=False)

    batch_keys = [
        "joint_positions",
        "gripper_positions",
        "robot_flows",
        "robot_exists",
        "scene_flows",
        "scene_exists",
        "scene_context_mask",
        "cam0_initial_rgb",
        "cam0_initial_depth",
        "cam0_intrinsic",
        "cam0_extrinsic",
    ]

    missing = [
        key for key in batch_keys
        if key not in device_batch
    ]
    if missing:
        raise RuntimeError(f"Missing batch keys: {missing}")

    output_dir = (
        Path("diagnostics/results")
        / f"action_perturb_{RUN_ID}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "perturb.pt"
    metadata_path = output_dir / "perturb.json"

    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite results in {output_dir}"
        )

    payload = {
        "run_id": RUN_ID,
        "sample_key": list(device_batch["__key__"]),
        "hook_state": dict(HOOK_STATE),
        "original_joint_positions": torch.from_numpy(
            CAPTURE_STATE["original_joint_positions"]
        ),
        "perturbed_joint_positions": torch.from_numpy(
            CAPTURE_STATE["perturbed_joint_positions"]
        ),
        "joint_delta": torch.from_numpy(
            CAPTURE_STATE["joint_delta"]
        ),
        "batch": {
            key: device_batch[key].detach().cpu()
            for key in batch_keys
        },
        "outputs": {
            key: value.detach().cpu()
            for key, value in outputs.items()
            if isinstance(value, torch.Tensor)
            and key in {
                "pred",
                "scene_flows",
                "scene_relative",
                "scene_relative_norm",
                "confidence",
                "log_var",
            }
        },
    }

    torch.save(payload, output_path)

    metadata = {
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

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)

    print()
    print("=" * 72)
    print("Action perturbation capture completed")
    print(f"Output: {output_path}")
    print(f"Sample: {payload['sample_key']}")
    print(json.dumps(HOOK_STATE, indent=2))
    print("=" * 72)

    return {
        f"diagnostic/{split}/captured_batches": 1,
    }


Tester._accumulate_metrics = capture_first_batch
runpy.run_path("eval.py", run_name="__main__")
