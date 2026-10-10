#!/usr/bin/env python3

from pathlib import Path
import os
import torch

ORIGINAL = Path(os.environ["PW_ORIGINAL_ROOT"])
CORRECTED = Path(os.environ["PW_CORRECTED_ROOT"])

PAIR = "camera_head__camera_left"

VARIANTS = [
    "speed_osc",
    "accel_spike",
    "reversal",
]

BASE = torch.load(
    ORIGINAL / f"{PAIR}__baseline.pt",
    map_location="cpu",
    weights_only=False,
)

pred_base = BASE["first_prediction"].float()
robot_base = BASE["robot_flows"].float()

obj = BASE["scene_object_id"]

if obj.ndim == 4 and obj.shape[-1] == 1:
    obj = obj.squeeze(-1)

exists = BASE["scene_exists"]

if exists.ndim == 4 and exists.shape[-1] == 1:
    exists = exists.squeeze(-1)

mask = (
    (obj == 2)
    & exists.bool()
)


def mean_acc(x):
    v = x[:,1:] - x[:,:-1]
    a = v[:,1:] - v[:,:-1]

    return float(
        torch.linalg.vector_norm(
            a,
            dim=-1,
        ).mean()
    )


def mean_jerk(x):
    v = x[:,1:] - x[:,:-1]
    a = v[:,1:] - v[:,:-1]
    j = a[:,1:] - a[:,:-1]

    return float(
        torch.linalg.vector_norm(
            j,
            dim=-1,
        ).mean()
    )


print("=" * 92)
print("ACTION QUALITY CORRECTION -> BASELINE RECOVERY")
print("=" * 92)

for variant in VARIANTS:

    pert = torch.load(
        ORIGINAL / f"{PAIR}__{variant}.pt",
        map_location="cpu",
        weights_only=False,
    )

    corr = torch.load(
        CORRECTED /
        f"{PAIR}__{variant}__quality_fix.pt",
        map_location="cpu",
        weights_only=False,
    )

    # ------------------------------------------
    # Critical controls
    # ------------------------------------------
    for k in [
        "scene_flows",
        "scene_object_id",
        "scene_exists",
        "cam0_initial_rgb",
        "cam1_initial_rgb",
        "cam0_initial_depth",
        "cam1_initial_depth",
    ]:
        if not torch.equal(BASE[k], corr[k]):
            raise RuntimeError(
                f"{variant}: control changed: {k}"
            )

    robot_p = pert["robot_flows"].float()
    robot_c = corr["robot_flows"].float()

    if not torch.equal(
        robot_base[:,0],
        robot_c[:,0],
    ):
        raise RuntimeError(
            f"{variant}: corrected t0 changed"
        )

    if not torch.equal(
        robot_base[:,-1],
        robot_c[:,-1],
    ):
        raise RuntimeError(
            f"{variant}: corrected endpoint changed"
        )

    pred_p = pert["first_prediction"].float()
    pred_c = corr["first_prediction"].float()

    # ------------------------------------------
    # Distance from BASELINE prediction
    # ------------------------------------------
    delta_p = torch.linalg.vector_norm(
        pred_p - pred_base,
        dim=-1,
    )

    delta_c = torch.linalg.vector_norm(
        pred_c - pred_base,
        dim=-1,
    )

    obj_p = float(
        delta_p[mask].mean()
    )

    obj_c = float(
        delta_c[mask].mean()
    )

    recovery = (
        1.0 - obj_c / obj_p
        if obj_p > 1e-12
        else float("nan")
    )

    # How much action itself moved back toward baseline.
    action_p = float(
        torch.linalg.vector_norm(
            robot_p - robot_base,
            dim=-1,
        ).mean()
    )

    action_c = float(
        torch.linalg.vector_norm(
            robot_c - robot_base,
            dim=-1,
        ).mean()
    )

    print(f"\n{variant}")
    print(
        "  action distance to baseline:"
        f" {action_p*1000:.3f}"
        f" -> {action_c*1000:.3f} mm"
    )

    print(
        "  mean acceleration:"
        f" {mean_acc(robot_p)*1000:.3f}"
        f" -> {mean_acc(robot_c)*1000:.3f}"
        " mm/step²"
    )

    print(
        "  mean jerk:"
        f" {mean_jerk(robot_p)*1000:.3f}"
        f" -> {mean_jerk(robot_c)*1000:.3f}"
        " mm/step³"
    )

    print(
        "  object prediction distance to baseline:"
        f" {obj_p*1000:.3f}"
        f" -> {obj_c*1000:.3f} mm"
    )

    print(
        "  prediction recovery ="
        f" {100*recovery:+.2f}%"
    )

print("=" * 92)
