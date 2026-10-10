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
OUTPUT_DIR = RESULT_ROOT / "action_perturb_grid_analysis_20260805"

DEPTH_TOLERANCE_M = 0.03
SPLAT_RADII = [2, 3, 4]


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def npy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def transform_points(points, transform):
    points_h = np.concatenate(
        [
            points,
            np.ones((len(points), 1), dtype=points.dtype),
        ],
        axis=1,
    )
    return (transform @ points_h.T).T[:, :3]


def project(points_camera, intrinsic, width, height):
    z = points_camera[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-6)

    u = np.full(len(z), -1.0, dtype=np.float32)
    v = np.full(len(z), -1.0, dtype=np.float32)

    projected = (intrinsic @ points_camera[valid_z].T).T
    u[valid_z] = projected[:, 0] / projected[:, 2]
    v[valid_z] = projected[:, 1] / projected[:, 2]

    valid = (
        valid_z
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    return u, v, z, valid


def robot_zbuffer(
    robot_points,
    intrinsic,
    extrinsic,
    width,
    height,
    radius,
):
    robot_camera = transform_points(robot_points, extrinsic)
    u, v, z, valid = project(
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

    for index in np.flatnonzero(valid):
        px = int(round(float(u[index])))
        py = int(round(float(v[index])))

        x0 = max(0, px - radius)
        x1 = min(width, px + radius + 1)
        y0 = max(0, py - radius)
        y1 = min(height, py + radius + 1)

        patch = zbuffer[y0:y1, x0:x1]
        np.minimum(patch, float(z[index]), out=patch)

    return zbuffer


def occlusion_ratio(
    scene_points,
    scene_exists,
    robot_points,
    robot_exists,
    intrinsic,
    extrinsic,
    width,
    height,
    radius,
):
    scene_mask = (
        scene_exists.astype(bool)
        & np.isfinite(scene_points).all(axis=1)
    )
    scene = scene_points[scene_mask]

    scene_camera = transform_points(scene, extrinsic)
    u, v, z, valid = project(
        scene_camera,
        intrinsic,
        width,
        height,
    )

    robot_mask = (
        robot_exists.astype(bool)
        & np.isfinite(robot_points).all(axis=1)
    )
    robot = robot_points[robot_mask]

    zbuffer = robot_zbuffer(
        robot,
        intrinsic,
        extrinsic,
        width,
        height,
        radius,
    )

    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return np.nan

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

    occluded = (
        np.isfinite(robot_depth)
        & (robot_depth < scene_depth - DEPTH_TOLERANCE_M)
    )

    return float(np.mean(occluded))


baseline = load_pt(BASELINE_PATH)
base_batch = baseline["batch"]

base_sample_key = baseline["sample_key"]
base_robot = npy(base_batch["robot_flows"])[0]
base_robot_exists = npy(base_batch["robot_exists"])[0]
scene_flows = npy(base_batch["scene_flows"])[0]
scene_exists = npy(base_batch["scene_exists"])[0]
intrinsic = npy(base_batch["cam0_intrinsic"])[0]
extrinsic = npy(base_batch["cam0_extrinsic"])[0]
depth = npy(base_batch["cam0_initial_depth"])[0]

base_pred = npy(baseline["outputs"]["scene_flows"])[0]

height, width = depth.shape
num_steps = scene_flows.shape[0]

baseline_occlusion = {}

for radius in SPLAT_RADII:
    ratios = []
    for timestep in range(num_steps):
        ratios.append(
            occlusion_ratio(
                scene_flows[timestep],
                scene_exists[timestep],
                base_robot[timestep],
                base_robot_exists[timestep],
                intrinsic,
                extrinsic,
                width,
                height,
                radius,
            )
        )
    baseline_occlusion[radius] = np.asarray(ratios)

perturb_paths = sorted(
    RESULT_ROOT.glob(
        "action_perturb_joint*_20260805/perturb.pt"
    )
)

if len(perturb_paths) != 14:
    raise RuntimeError(
        f"Expected 14 perturbations, found {len(perturb_paths)}"
    )

rows = []

for path in perturb_paths:
    payload = load_pt(path)

    if payload["sample_key"] != base_sample_key:
        raise RuntimeError(
            f"Sample mismatch in {path}"
        )

    match = re.search(
        r"joint(\d+)_(plus|minus)015",
        path.parent.name,
    )
    if not match:
        raise RuntimeError(f"Cannot parse action name: {path}")

    joint_number = int(match.group(1))
    direction = match.group(2)

    batch = payload["batch"]
    robot = npy(batch["robot_flows"])[0]
    robot_exists = npy(batch["robot_exists"])[0]
    pred = npy(payload["outputs"]["scene_flows"])[0]

    if not np.array_equal(
        npy(batch["scene_flows"]),
        npy(base_batch["scene_flows"]),
    ):
        raise RuntimeError(
            f"Scene flow mismatch: {path}"
        )

    robot_delta = np.linalg.norm(
        robot - base_robot,
        axis=-1,
    )

    pred_delta = np.linalg.norm(
        pred - base_pred,
        axis=-1,
    )

    row = {
        "action": f"joint{joint_number}_{direction}015",
        "joint_number": joint_number,
        "direction": direction,
        "mean_robot_point_change_m": float(
            robot_delta.mean()
        ),
        "max_robot_point_change_m": float(
            robot_delta.max()
        ),
        "mean_prediction_change_m": float(
            pred_delta.mean()
        ),
        "max_prediction_change_m": float(
            pred_delta.max()
        ),
    }

    radius_signs = []

    for radius in SPLAT_RADII:
        ratios = []

        for timestep in range(num_steps):
            ratios.append(
                occlusion_ratio(
                    scene_flows[timestep],
                    scene_exists[timestep],
                    robot[timestep],
                    robot_exists[timestep],
                    intrinsic,
                    extrinsic,
                    width,
                    height,
                    radius,
                )
            )

        ratios = np.asarray(ratios)
        delta = ratios - baseline_occlusion[radius]

        row[f"mean_occlusion_ratio_r{radius}"] = float(
            np.nanmean(ratios)
        )
        row[f"mean_delta_occlusion_r{radius}"] = float(
            np.nanmean(delta)
        )
        row[f"max_delta_occlusion_r{radius}"] = float(
            np.nanmax(delta)
        )
        row[f"min_delta_occlusion_r{radius}"] = float(
            np.nanmin(delta)
        )

        radius_signs.append(
            np.sign(np.nanmean(delta))
        )

    row["direction_stable_across_radii"] = bool(
        radius_signs[0]
        == radius_signs[1]
        == radius_signs[2]
    )

    rows.append(row)

rows.sort(
    key=lambda item: item["mean_delta_occlusion_r3"],
    reverse=True,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

csv_path = OUTPUT_DIR / "action_ranking.csv"
json_path = OUTPUT_DIR / "summary.json"

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "sample_key": base_sample_key,
    "num_actions": len(rows),
    "num_timesteps": num_steps,
    "depth_tolerance_m": DEPTH_TOLERANCE_M,
    "splat_radii_px": SPLAT_RADII,
    "baseline_mean_occlusion": {
        str(radius): float(
            np.nanmean(baseline_occlusion[radius])
        )
        for radius in SPLAT_RADII
    },
    "highest_occlusion_increase_action": rows[0],
    "highest_occlusion_decrease_action": rows[-1],
    "largest_prediction_change_action": max(
        rows,
        key=lambda item: item["mean_prediction_change_m"],
    ),
    "all_direction_stable_count": int(
        sum(
            row["direction_stable_across_radii"]
            for row in rows
        )
    ),
    "interpretation_warning": (
        "Prediction change is model sensitivity only. "
        "It is not prediction error because no counterfactual GT exists."
    ),
}

with open(json_path, "w") as file:
    json.dump(summary, file, indent=2)

print("=" * 80)
print("RANKED BY MEAN OCCLUSION DELTA, RADIUS=3 PX")
print("=" * 80)

for rank, row in enumerate(rows, start=1):
    print(
        f"{rank:02d}. {row['action']:<18} "
        f"occ_delta={row['mean_delta_occlusion_r3']:+.6f} "
        f"pred_change={row['mean_prediction_change_m']:.6f} m "
        f"stable={row['direction_stable_across_radii']}"
    )

print()
print("CSV:", csv_path)
print("JSON:", json_path)
