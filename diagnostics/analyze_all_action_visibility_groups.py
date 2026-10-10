#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch


BASELINE_PATH = Path(
    "diagnostics/results/"
    "action_identity_camera_check_20260805/identity.pt"
)
RESULT_ROOT = Path("diagnostics/results")
OUTPUT_DIR = RESULT_ROOT / "visibility_groups_all_actions_20260805"

DEPTH_TOLERANCE_M = 0.03
SPLAT_RADIUS_PX = 3


def load_pt(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def npy(value):
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
    return (transform @ points_h.T).T[:, :3]


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

        x0 = max(0, px - SPLAT_RADIUS_PX)
        x1 = min(width, px + SPLAT_RADIUS_PX + 1)
        y0 = max(0, py - SPLAT_RADIUS_PX)
        y1 = min(height, py + SPLAT_RADIUS_PX + 1)

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


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return float(values.mean())


def safe_ratio(numerator, denominator):
    if (
        numerator is None
        or denominator is None
        or denominator <= 0
    ):
        return None

    return float(numerator / denominator)


def pearson_correlation(x, y):
    pairs = [
        (a, b)
        for a, b in zip(x, y)
        if a is not None
        and b is not None
        and np.isfinite(a)
        and np.isfinite(b)
    ]

    if len(pairs) < 3:
        return None

    x_array = np.asarray(
        [pair[0] for pair in pairs],
        dtype=np.float64,
    )
    y_array = np.asarray(
        [pair[1] for pair in pairs],
        dtype=np.float64,
    )

    if (
        np.std(x_array) == 0
        or np.std(y_array) == 0
    ):
        return None

    return float(
        np.corrcoef(x_array, y_array)[0, 1]
    )


baseline = load_pt(BASELINE_PATH)
base_batch = baseline["batch"]

base_sample_key = baseline["sample_key"]

scene_flows = npy(
    base_batch["scene_flows"]
)[0]

scene_exists = npy(
    base_batch["scene_exists"]
)[0]

base_robot = npy(
    base_batch["robot_flows"]
)[0]

base_robot_exists = npy(
    base_batch["robot_exists"]
)[0]

intrinsic = npy(
    base_batch["cam0_intrinsic"]
)[0]

extrinsic = npy(
    base_batch["cam0_extrinsic"]
)[0]

depth = npy(
    base_batch["cam0_initial_depth"]
)[0]

base_prediction = npy(
    baseline["outputs"]["scene_flows"]
)[0]

num_steps, num_points, _ = scene_flows.shape
height, width = depth.shape

base_occlusion_masks = []
base_valid_masks = []

for timestep in range(num_steps):
    occluded, valid = compute_occlusion_mask(
        scene_flows[timestep],
        scene_exists[timestep],
        base_robot[timestep],
        base_robot_exists[timestep],
        intrinsic,
        extrinsic,
        width,
        height,
    )

    base_occlusion_masks.append(occluded)
    base_valid_masks.append(valid)

base_occlusion_masks = np.stack(
    base_occlusion_masks,
    axis=0,
)
base_valid_masks = np.stack(
    base_valid_masks,
    axis=0,
)

baseline_mean_occlusion = float(
    np.mean(
        base_occlusion_masks[
            base_valid_masks
        ]
    )
)

perturb_paths = sorted(
    RESULT_ROOT.glob(
        "action_perturb_joint*_20260805/perturb.pt"
    )
)

if len(perturb_paths) != 14:
    raise RuntimeError(
        f"Expected 14 actions, found {len(perturb_paths)}"
    )

rows = []

for path in perturb_paths:
    payload = load_pt(path)

    if payload["sample_key"] != base_sample_key:
        raise RuntimeError(
            f"Sample mismatch: {path}"
        )

    match = re.search(
        r"joint(\d+)_(plus|minus)015",
        path.parent.name,
    )

    if not match:
        raise RuntimeError(
            f"Cannot parse action: {path}"
        )

    action = (
        f"joint{match.group(1)}_"
        f"{match.group(2)}015"
    )

    batch = payload["batch"]

    robot = npy(
        batch["robot_flows"]
    )[0]

    robot_exists = npy(
        batch["robot_exists"]
    )[0]

    prediction = npy(
        payload["outputs"]["scene_flows"]
    )[0]

    prediction_change = np.linalg.norm(
        prediction - base_prediction,
        axis=-1,
    )

    base_velocity = np.diff(
        base_prediction,
        axis=0,
    )
    pert_velocity = np.diff(
        prediction,
        axis=0,
    )

    velocity_change = np.linalg.norm(
        pert_velocity - base_velocity,
        axis=-1,
    )

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

    perturb_occlusion_masks = []
    perturb_valid_masks = []

    for timestep in range(num_steps):
        occluded, valid = compute_occlusion_mask(
            scene_flows[timestep],
            scene_exists[timestep],
            robot[timestep],
            robot_exists[timestep],
            intrinsic,
            extrinsic,
            width,
            height,
        )

        perturb_occlusion_masks.append(occluded)
        perturb_valid_masks.append(valid)

    perturb_occlusion_masks = np.stack(
        perturb_occlusion_masks,
        axis=0,
    )
    perturb_valid_masks = np.stack(
        perturb_valid_masks,
        axis=0,
    )

    for timestep in range(num_steps):
        common_valid = (
            base_valid_masks[timestep]
            & perturb_valid_masks[timestep]
        )

        base_occ = base_occlusion_masks[timestep]
        pert_occ = perturb_occlusion_masks[timestep]

        groups = {
            "newly_occluded": (
                common_valid
                & (~base_occ)
                & pert_occ
            ),
            "persistent_visible": (
                common_valid
                & (~base_occ)
                & (~pert_occ)
            ),
            "newly_visible": (
                common_valid
                & base_occ
                & (~pert_occ)
            ),
            "persistent_occluded": (
                common_valid
                & base_occ
                & pert_occ
            ),
        }

        for group_name, mask in groups.items():
            group_values[
                group_name
            ]["prediction"].extend(
                prediction_change[
                    timestep,
                    mask,
                ].tolist()
            )

            if timestep >= 1:
                group_values[
                    group_name
                ]["velocity"].extend(
                    velocity_change[
                        timestep - 1,
                        mask,
                    ].tolist()
                )

            if timestep >= 2:
                group_values[
                    group_name
                ]["acceleration"].extend(
                    acceleration_change[
                        timestep - 2,
                        mask,
                    ].tolist()
                )

    perturb_mean_occlusion = float(
        np.mean(
            perturb_occlusion_masks[
                perturb_valid_masks
            ]
        )
    )

    new_occ_pred = safe_mean(
        group_values[
            "newly_occluded"
        ]["prediction"]
    )
    visible_pred = safe_mean(
        group_values[
            "persistent_visible"
        ]["prediction"]
    )

    new_occ_vel = safe_mean(
        group_values[
            "newly_occluded"
        ]["velocity"]
    )
    visible_vel = safe_mean(
        group_values[
            "persistent_visible"
        ]["velocity"]
    )

    new_occ_acc = safe_mean(
        group_values[
            "newly_occluded"
        ]["acceleration"]
    )
    visible_acc = safe_mean(
        group_values[
            "persistent_visible"
        ]["acceleration"]
    )

    row = {
        "action": action,
        "baseline_mean_occlusion": (
            baseline_mean_occlusion
        ),
        "perturb_mean_occlusion": (
            perturb_mean_occlusion
        ),
        "mean_occlusion_delta": (
            perturb_mean_occlusion
            - baseline_mean_occlusion
        ),
        "newly_occluded_point_timestep_count": len(
            group_values[
                "newly_occluded"
            ]["prediction"]
        ),
        "persistent_visible_point_timestep_count": len(
            group_values[
                "persistent_visible"
            ]["prediction"]
        ),
        "newly_occluded_prediction_change_m": (
            new_occ_pred
        ),
        "persistent_visible_prediction_change_m": (
            visible_pred
        ),
        "prediction_sensitivity_ratio": safe_ratio(
            new_occ_pred,
            visible_pred,
        ),
        "newly_occluded_velocity_change_m": (
            new_occ_vel
        ),
        "persistent_visible_velocity_change_m": (
            visible_vel
        ),
        "velocity_sensitivity_ratio": safe_ratio(
            new_occ_vel,
            visible_vel,
        ),
        "newly_occluded_acceleration_change_m": (
            new_occ_acc
        ),
        "persistent_visible_acceleration_change_m": (
            visible_acc
        ),
        "acceleration_sensitivity_ratio": safe_ratio(
            new_occ_acc,
            visible_acc,
        ),
    }

    rows.append(row)

rows.sort(
    key=lambda row: row["mean_occlusion_delta"],
    reverse=True,
)

occlusion_deltas = [
    row["mean_occlusion_delta"]
    for row in rows
]

prediction_ratios = [
    row["prediction_sensitivity_ratio"]
    for row in rows
]

velocity_ratios = [
    row["velocity_sensitivity_ratio"]
    for row in rows
]

acceleration_ratios = [
    row["acceleration_sensitivity_ratio"]
    for row in rows
]

summary = {
    "sample_key": base_sample_key,
    "num_actions": len(rows),
    "num_timesteps": num_steps,
    "num_scene_points": num_points,
    "depth_tolerance_m": DEPTH_TOLERANCE_M,
    "splat_radius_px": SPLAT_RADIUS_PX,
    "baseline_mean_occlusion": (
        baseline_mean_occlusion
    ),
    "correlations": {
        "occlusion_delta_vs_prediction_sensitivity_ratio": (
            pearson_correlation(
                occlusion_deltas,
                prediction_ratios,
            )
        ),
        "occlusion_delta_vs_velocity_sensitivity_ratio": (
            pearson_correlation(
                occlusion_deltas,
                velocity_ratios,
            )
        ),
        "occlusion_delta_vs_acceleration_sensitivity_ratio": (
            pearson_correlation(
                occlusion_deltas,
                acceleration_ratios,
            )
        ),
    },
    "actions_with_prediction_ratio_above_1": int(
        sum(
            row["prediction_sensitivity_ratio"] is not None
            and row["prediction_sensitivity_ratio"] > 1.0
            for row in rows
        )
    ),
    "actions_with_velocity_ratio_above_1": int(
        sum(
            row["velocity_sensitivity_ratio"] is not None
            and row["velocity_sensitivity_ratio"] > 1.0
            for row in rows
        )
    ),
    "actions_with_acceleration_ratio_above_1": int(
        sum(
            row["acceleration_sensitivity_ratio"] is not None
            and row["acceleration_sensitivity_ratio"] > 1.0
            for row in rows
        )
    ),
    "highest_occlusion_increase_action": rows[0],
    "highest_velocity_ratio_action": max(
        rows,
        key=lambda row: (
            row["velocity_sensitivity_ratio"]
            if row["velocity_sensitivity_ratio"] is not None
            else -np.inf
        ),
    ),
    "highest_acceleration_ratio_action": max(
        rows,
        key=lambda row: (
            row["acceleration_sensitivity_ratio"]
            if row["acceleration_sensitivity_ratio"] is not None
            else -np.inf
        ),
    ),
    "interpretation_warning": (
        "These are counterfactual model-output "
        "sensitivity metrics, not prediction error. "
        "Counterfactual ground truth is unavailable."
    ),
}

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

csv_path = OUTPUT_DIR / "all_action_visibility_groups.csv"
json_path = OUTPUT_DIR / "summary.json"

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

print("=" * 90)
print("ALL-ACTION VISIBILITY GROUP ANALYSIS")
print("=" * 90)

for row in rows:
    print(
        f"{row['action']:<18} "
        f"occ={row['mean_occlusion_delta']:+.6f} "
        f"pred_ratio={row['prediction_sensitivity_ratio']} "
        f"vel_ratio={row['velocity_sensitivity_ratio']} "
        f"acc_ratio={row['acceleration_sensitivity_ratio']}"
    )

print()
print(json.dumps(summary, indent=2))
print("CSV:", csv_path)
print("JSON:", json_path)
