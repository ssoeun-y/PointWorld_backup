#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np
import torch


BASELINE_PATH = Path(
    "diagnostics/results/"
    "action_identity_camera_check_20260805/identity.pt"
)
PERTURB_PATH = Path(
    "diagnostics/results/"
    "action_perturb_joint4_minus015_20260805/perturb.pt"
)
OUTPUT_DIR = Path(
    "diagnostics/results/"
    "visibility_group_joint4_minus015_20260805"
)

DEPTH_TOLERANCE_M = 0.03
SPLAT_RADIUS_PX = 3


def load_pt(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def transform_points(points, transform):
    ones = np.ones(
        (points.shape[0], 1),
        dtype=points.dtype,
    )
    points_h = np.concatenate(
        [points, ones],
        axis=1,
    )
    camera_h = (transform @ points_h.T).T
    return camera_h[:, :3]


def project_points(
    points_camera,
    intrinsic,
    width,
    height,
):
    z = points_camera[:, 2]

    positive = (
        np.isfinite(z)
        & (z > 1e-6)
    )

    u = np.full(
        z.shape,
        -1.0,
        dtype=np.float32,
    )
    v = np.full(
        z.shape,
        -1.0,
        dtype=np.float32,
    )

    projected = (
        intrinsic @ points_camera[positive].T
    ).T

    u[positive] = (
        projected[:, 0] / projected[:, 2]
    )
    v[positive] = (
        projected[:, 1] / projected[:, 2]
    )

    valid = (
        positive
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    return u, v, z, valid


def build_robot_zbuffer(
    robot_points,
    robot_exists,
    intrinsic,
    extrinsic,
    width,
    height,
):
    valid_robot = (
        robot_exists.astype(bool)
        & np.isfinite(robot_points).all(axis=1)
    )

    robot = robot_points[valid_robot]
    robot_camera = transform_points(
        robot,
        extrinsic,
    )

    u, v, z, projected = project_points(
        robot_camera,
        intrinsic,
        width,
        height,
    )

    zbuffer = np.full(
        (height, width),
        np.inf,
        dtype=np.float32,
    )

    for index in np.flatnonzero(projected):
        px = int(round(float(u[index])))
        py = int(round(float(v[index])))

        x0 = max(
            0,
            px - SPLAT_RADIUS_PX,
        )
        x1 = min(
            width,
            px + SPLAT_RADIUS_PX + 1,
        )
        y0 = max(
            0,
            py - SPLAT_RADIUS_PX,
        )
        y1 = min(
            height,
            py + SPLAT_RADIUS_PX + 1,
        )

        patch = zbuffer[y0:y1, x0:x1]
        np.minimum(
            patch,
            float(z[index]),
            out=patch,
        )

    return zbuffer


def compute_occlusion_mask(
    scene_points,
    scene_exists,
    robot_points,
    robot_exists,
    intrinsic,
    extrinsic,
    width,
    height,
):
    """
    Returns full point-wise masks with shape (N,).
    """
    num_points = scene_points.shape[0]

    valid_scene = (
        scene_exists.astype(bool)
        & np.isfinite(scene_points).all(axis=1)
    )

    scene_camera = transform_points(
        scene_points,
        extrinsic,
    )

    u, v, z, projected = project_points(
        scene_camera,
        intrinsic,
        width,
        height,
    )

    valid_projected = (
        valid_scene
        & projected
    )

    zbuffer = build_robot_zbuffer(
        robot_points,
        robot_exists,
        intrinsic,
        extrinsic,
        width,
        height,
    )

    occluded = np.zeros(
        num_points,
        dtype=bool,
    )

    indices = np.flatnonzero(valid_projected)

    if indices.size:
        px = np.clip(
            np.rint(u[indices]).astype(np.int64),
            0,
            width - 1,
        )
        py = np.clip(
            np.rint(v[indices]).astype(np.int64),
            0,
            height - 1,
        )

        robot_depth = zbuffer[py, px]
        scene_depth = z[indices]

        occluded[indices] = (
            np.isfinite(robot_depth)
            & (
                robot_depth
                < scene_depth - DEPTH_TOLERANCE_M
            )
        )

    return occluded, valid_projected


def safe_stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(
            np.percentile(values, 90)
        ),
        "max": float(np.max(values)),
    }


baseline = load_pt(BASELINE_PATH)
perturb = load_pt(PERTURB_PATH)

if baseline["sample_key"] != perturb["sample_key"]:
    raise RuntimeError(
        "Baseline and perturb sample keys differ"
    )

base_batch = baseline["batch"]
pert_batch = perturb["batch"]

for key in [
    "scene_flows",
    "scene_exists",
    "cam0_intrinsic",
    "cam0_extrinsic",
    "cam0_initial_depth",
]:
    a = to_numpy(base_batch[key])
    b = to_numpy(pert_batch[key])

    if not np.array_equal(a, b):
        raise RuntimeError(
            f"Baseline/perturb mismatch: {key}"
        )

scene_flows = to_numpy(
    base_batch["scene_flows"]
)[0]

scene_exists = to_numpy(
    base_batch["scene_exists"]
)[0]

base_robot = to_numpy(
    base_batch["robot_flows"]
)[0]

pert_robot = to_numpy(
    pert_batch["robot_flows"]
)[0]

base_robot_exists = to_numpy(
    base_batch["robot_exists"]
)[0]

pert_robot_exists = to_numpy(
    pert_batch["robot_exists"]
)[0]

intrinsic = to_numpy(
    base_batch["cam0_intrinsic"]
)[0]

extrinsic = to_numpy(
    base_batch["cam0_extrinsic"]
)[0]

depth = to_numpy(
    base_batch["cam0_initial_depth"]
)[0]

base_prediction = to_numpy(
    baseline["outputs"]["scene_flows"]
)[0]

pert_prediction = to_numpy(
    perturb["outputs"]["scene_flows"]
)[0]

if base_prediction.shape != pert_prediction.shape:
    raise RuntimeError(
        "Prediction shapes differ"
    )

num_steps, num_points, _ = scene_flows.shape
height, width = depth.shape

# Point-wise prediction sensitivity.
prediction_change = np.linalg.norm(
    pert_prediction - base_prediction,
    axis=-1,
)

# Velocity sensitivity.
base_velocity = np.diff(
    base_prediction,
    axis=0,
)
pert_velocity = np.diff(
    pert_prediction,
    axis=0,
)

velocity_change = np.linalg.norm(
    pert_velocity - base_velocity,
    axis=-1,
)

# Acceleration sensitivity.
base_acceleration = np.diff(
    base_velocity,
    axis=0,
)
pert_acceleration = np.diff(
    pert_velocity,
    axis=0,
)

acceleration_change = np.linalg.norm(
    pert_acceleration - base_acceleration,
    axis=-1,
)

rows = []

group_values = {
    "newly_occluded": {
        "prediction": [],
        "velocity": [],
        "acceleration": [],
    },
    "persistent_visible": {
        "prediction": [],
        "velocity": [],
        "acceleration": [],
    },
    "newly_visible": {
        "prediction": [],
        "velocity": [],
        "acceleration": [],
    },
    "persistent_occluded": {
        "prediction": [],
        "velocity": [],
        "acceleration": [],
    },
}

unique_point_ids = {
    key: set()
    for key in group_values
}

for timestep in range(num_steps):
    base_occluded, base_valid = (
        compute_occlusion_mask(
            scene_flows[timestep],
            scene_exists[timestep],
            base_robot[timestep],
            base_robot_exists[timestep],
            intrinsic,
            extrinsic,
            width,
            height,
        )
    )

    pert_occluded, pert_valid = (
        compute_occlusion_mask(
            scene_flows[timestep],
            scene_exists[timestep],
            pert_robot[timestep],
            pert_robot_exists[timestep],
            intrinsic,
            extrinsic,
            width,
            height,
        )
    )

    common_valid = (
        base_valid
        & pert_valid
    )

    groups = {
        "newly_occluded": (
            common_valid
            & (~base_occluded)
            & pert_occluded
        ),
        "persistent_visible": (
            common_valid
            & (~base_occluded)
            & (~pert_occluded)
        ),
        "newly_visible": (
            common_valid
            & base_occluded
            & (~pert_occluded)
        ),
        "persistent_occluded": (
            common_valid
            & base_occluded
            & pert_occluded
        ),
    }

    row = {
        "timestep": timestep,
        "common_valid_count": int(
            common_valid.sum()
        ),
    }

    for group_name, mask in groups.items():
        ids = np.flatnonzero(mask)
        unique_point_ids[group_name].update(
            ids.tolist()
        )

        pred_values = prediction_change[
            timestep,
            mask,
        ]

        group_values[group_name][
            "prediction"
        ].extend(
            pred_values.tolist()
        )

        pred_stats = safe_stats(pred_values)

        row[
            f"{group_name}_count"
        ] = int(mask.sum())

        row[
            f"{group_name}_prediction_change_mean_m"
        ] = pred_stats["mean"]

        # velocity[t-1] corresponds to motion
        # from timestep-1 to timestep.
        if timestep >= 1:
            vel_values = velocity_change[
                timestep - 1,
                mask,
            ]

            group_values[group_name][
                "velocity"
            ].extend(
                vel_values.tolist()
            )

            vel_stats = safe_stats(vel_values)
            row[
                f"{group_name}_velocity_change_mean_m"
            ] = vel_stats["mean"]
        else:
            row[
                f"{group_name}_velocity_change_mean_m"
            ] = None

        # acceleration[t-2] corresponds to
        # second derivative ending at timestep.
        if timestep >= 2:
            acc_values = acceleration_change[
                timestep - 2,
                mask,
            ]

            group_values[group_name][
                "acceleration"
            ].extend(
                acc_values.tolist()
            )

            acc_stats = safe_stats(acc_values)
            row[
                f"{group_name}_acceleration_change_mean_m"
            ] = acc_stats["mean"]
        else:
            row[
                f"{group_name}_acceleration_change_mean_m"
            ] = None

    rows.append(row)

summary_groups = {}

for group_name, metrics in group_values.items():
    summary_groups[group_name] = {
        "unique_point_count": len(
            unique_point_ids[group_name]
        ),
        "point_timestep_count": len(
            metrics["prediction"]
        ),
        "prediction_change_m": safe_stats(
            metrics["prediction"]
        ),
        "velocity_change_m": safe_stats(
            metrics["velocity"]
        ),
        "acceleration_change_m": safe_stats(
            metrics["acceleration"]
        ),
    }

new_occ_mean = summary_groups[
    "newly_occluded"
]["prediction_change_m"]["mean"]

visible_mean = summary_groups[
    "persistent_visible"
]["prediction_change_m"]["mean"]

if (
    new_occ_mean is not None
    and visible_mean is not None
    and visible_mean > 0
):
    sensitivity_ratio = (
        new_occ_mean / visible_mean
    )
else:
    sensitivity_ratio = None

summary = {
    "sample_key": baseline["sample_key"],
    "action": "joint4_minus015",
    "num_timesteps": num_steps,
    "num_scene_points": num_points,
    "depth_tolerance_m": DEPTH_TOLERANCE_M,
    "splat_radius_px": SPLAT_RADIUS_PX,
    "t0_robot_exact": bool(
        np.array_equal(
            base_robot[0],
            pert_robot[0],
        )
    ),
    "groups": summary_groups,
    "newly_occluded_vs_persistent_visible_prediction_change_ratio": (
        sensitivity_ratio
    ),
    "interpretation_warning": (
        "These metrics measure model-output sensitivity "
        "between original and counterfactual actions. "
        "They are not prediction error because no "
        "counterfactual ground truth is available."
    ),
}

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

csv_path = (
    OUTPUT_DIR
    / "visibility_group_by_timestep.csv"
)

json_path = (
    OUTPUT_DIR
    / "summary.json"
)

with open(
    csv_path,
    "w",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

with open(
    json_path,
    "w",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
    )

print("=" * 80)
print("Visibility-group sensitivity analysis completed")
print(json.dumps(summary, indent=2))
print("CSV:", csv_path)
print("JSON:", json_path)
print("=" * 80)
