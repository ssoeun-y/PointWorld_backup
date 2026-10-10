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
    "action_perturb_joint4_plus015_20260805/perturb.pt"
)
OUTPUT_DIR = Path(
    "diagnostics/results/action_occlusion_joint4_plus015_20260805"
)

DEPTH_TOLERANCE_M = 0.03
SPLAT_RADIUS_PX = 3


def load_pt(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def transform_points(points, transform):
    """
    points: (N,3)
    transform: (4,4), world -> camera
    """
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.concatenate([points, ones], axis=1)
    camera_h = (transform @ points_h.T).T
    return camera_h[:, :3]


def project_points(points_camera, intrinsic, width, height):
    z = points_camera[:, 2]

    valid_z = np.isfinite(z) & (z > 1e-6)

    u = np.full(z.shape, -1.0, dtype=np.float32)
    v = np.full(z.shape, -1.0, dtype=np.float32)

    xyz = points_camera[valid_z]
    projected = (intrinsic @ xyz.T).T

    u[valid_z] = projected[:, 0] / projected[:, 2]
    v[valid_z] = projected[:, 1] / projected[:, 2]

    pixel_valid = (
        valid_z
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    return u, v, z, pixel_valid


def build_robot_zbuffer(
    robot_points,
    intrinsic,
    extrinsic,
    width,
    height,
    radius,
):
    robot_camera = transform_points(robot_points, extrinsic)

    u, v, z, valid = project_points(
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

    valid_indices = np.flatnonzero(valid)

    for index in valid_indices:
        px = int(round(float(u[index])))
        py = int(round(float(v[index])))
        depth = float(z[index])

        x0 = max(0, px - radius)
        x1 = min(width, px + radius + 1)
        y0 = max(0, py - radius)
        y1 = min(height, py + radius + 1)

        patch = zbuffer[y0:y1, x0:x1]
        np.minimum(patch, depth, out=patch)

    return zbuffer, int(valid.sum())


def compute_occlusion(
    scene_points,
    scene_exists,
    robot_points,
    robot_exists,
    intrinsic,
    extrinsic,
    width,
    height,
):
    valid_scene_input = (
        scene_exists.astype(bool)
        & np.isfinite(scene_points).all(axis=1)
    )

    scene = scene_points[valid_scene_input]
    scene_camera = transform_points(scene, extrinsic)

    u, v, z, projected = project_points(
        scene_camera,
        intrinsic,
        width,
        height,
    )

    robot = robot_points[
        robot_exists.astype(bool)
        & np.isfinite(robot_points).all(axis=1)
    ]

    robot_zbuffer, projected_robot_count = build_robot_zbuffer(
        robot,
        intrinsic,
        extrinsic,
        width,
        height,
        SPLAT_RADIUS_PX,
    )

    indices = np.flatnonzero(projected)

    if indices.size == 0:
        return {
            "valid_scene_count": 0,
            "occluded_count": 0,
            "occlusion_ratio": float("nan"),
            "projected_robot_count": projected_robot_count,
        }

    px = np.rint(u[indices]).astype(np.int64)
    py = np.rint(v[indices]).astype(np.int64)

    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)

    robot_depth = robot_zbuffer[py, px]
    scene_depth = z[indices]

    occluded = (
        np.isfinite(robot_depth)
        & (robot_depth < scene_depth - DEPTH_TOLERANCE_M)
    )

    return {
        "valid_scene_count": int(indices.size),
        "occluded_count": int(occluded.sum()),
        "occlusion_ratio": float(occluded.mean()),
        "projected_robot_count": projected_robot_count,
    }


baseline = load_pt(BASELINE_PATH)
perturb = load_pt(PERTURB_PATH)

baseline_key = baseline["sample_key"]
perturb_key = perturb["sample_key"]

if baseline_key != perturb_key:
    raise RuntimeError(
        f"Sample mismatch: {baseline_key} vs {perturb_key}"
    )

base_batch = baseline["batch"]
pert_batch = perturb["batch"]

for key in [
    "scene_flows",
    "scene_exists",
    "cam0_initial_depth",
    "cam0_intrinsic",
    "cam0_extrinsic",
]:
    a = to_numpy(base_batch[key])
    b = to_numpy(pert_batch[key])

    if not np.array_equal(a, b):
        max_error = float(np.max(np.abs(a.astype(float) - b.astype(float))))
        raise RuntimeError(
            f"Baseline/perturb mismatch for {key}; max error={max_error}"
        )

base_robot = to_numpy(base_batch["robot_flows"])[0]
pert_robot = to_numpy(pert_batch["robot_flows"])[0]

base_robot_exists = to_numpy(base_batch["robot_exists"])[0]
pert_robot_exists = to_numpy(pert_batch["robot_exists"])[0]

scene_flows = to_numpy(base_batch["scene_flows"])[0]
scene_exists = to_numpy(base_batch["scene_exists"])[0]

depth_image = to_numpy(base_batch["cam0_initial_depth"])[0]
intrinsic = to_numpy(base_batch["cam0_intrinsic"])[0]
extrinsic = to_numpy(base_batch["cam0_extrinsic"])[0]

height, width = depth_image.shape
num_steps = scene_flows.shape[0]

robot_l2 = np.linalg.norm(
    pert_robot - base_robot,
    axis=-1,
)

if not np.allclose(base_robot[0], pert_robot[0], atol=1e-7):
    raise RuntimeError("Robot flows differ at t=0")

# 좌표계 검산:
# t=0 scene point를 cam0에 투영했을 때 z가 initial depth와 대략 맞는지 검사.
scene_t0_camera = transform_points(scene_flows[0], extrinsic)

u0, v0, z0, valid0 = project_points(
    scene_t0_camera,
    intrinsic,
    width,
    height,
)

valid_indices = np.flatnonzero(
    valid0
    & scene_exists[0].astype(bool)
)

px0 = np.rint(u0[valid_indices]).astype(np.int64)
py0 = np.rint(v0[valid_indices]).astype(np.int64)

px0 = np.clip(px0, 0, width - 1)
py0 = np.clip(py0, 0, height - 1)

observed_depth = depth_image[py0, px0]
predicted_depth = z0[valid_indices]

depth_valid = (
    np.isfinite(observed_depth)
    & (observed_depth > 0)
    & np.isfinite(predicted_depth)
)

depth_error = np.abs(
    observed_depth[depth_valid]
    - predicted_depth[depth_valid]
)

coordinate_check = {
    "projected_scene_count": int(valid_indices.size),
    "depth_comparable_count": int(depth_valid.sum()),
    "depth_error_median_m": (
        float(np.median(depth_error))
        if depth_error.size else None
    ),
    "depth_error_mean_m": (
        float(np.mean(depth_error))
        if depth_error.size else None
    ),
    "depth_error_p90_m": (
        float(np.percentile(depth_error, 90))
        if depth_error.size else None
    ),
    "within_3cm_ratio": (
        float(np.mean(depth_error <= 0.03))
        if depth_error.size else None
    ),
    "positive_camera_z_ratio": float(
        np.mean(scene_t0_camera[:, 2] > 0)
    ),
}

rows = []

for timestep in range(num_steps):
    base_result = compute_occlusion(
        scene_points=scene_flows[timestep],
        scene_exists=scene_exists[timestep],
        robot_points=base_robot[timestep],
        robot_exists=base_robot_exists[timestep],
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        width=width,
        height=height,
    )

    perturb_result = compute_occlusion(
        scene_points=scene_flows[timestep],
        scene_exists=scene_exists[timestep],
        robot_points=pert_robot[timestep],
        robot_exists=pert_robot_exists[timestep],
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        width=width,
        height=height,
    )

    rows.append({
        "timestep": timestep,
        "baseline_occluded_count": base_result["occluded_count"],
        "baseline_valid_scene_count": base_result["valid_scene_count"],
        "baseline_occlusion_ratio": base_result["occlusion_ratio"],
        "perturb_occluded_count": perturb_result["occluded_count"],
        "perturb_valid_scene_count": perturb_result["valid_scene_count"],
        "perturb_occlusion_ratio": perturb_result["occlusion_ratio"],
        "delta_occlusion_ratio": (
            perturb_result["occlusion_ratio"]
            - base_result["occlusion_ratio"]
        ),
        "mean_robot_point_l2_m": float(
            robot_l2[timestep].mean()
        ),
        "max_robot_point_l2_m": float(
            robot_l2[timestep].max()
        ),
        "baseline_projected_robot_count": (
            base_result["projected_robot_count"]
        ),
        "perturb_projected_robot_count": (
            perturb_result["projected_robot_count"]
        ),
    })

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

csv_path = OUTPUT_DIR / "occlusion_by_timestep.csv"
json_path = OUTPUT_DIR / "summary.json"

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

base_ratios = np.asarray(
    [row["baseline_occlusion_ratio"] for row in rows],
    dtype=np.float64,
)
pert_ratios = np.asarray(
    [row["perturb_occlusion_ratio"] for row in rows],
    dtype=np.float64,
)
delta_ratios = pert_ratios - base_ratios

summary = {
    "sample_key": baseline_key,
    "baseline_path": str(BASELINE_PATH),
    "perturb_path": str(PERTURB_PATH),
    "num_timesteps": num_steps,
    "image_height": height,
    "image_width": width,
    "depth_tolerance_m": DEPTH_TOLERANCE_M,
    "robot_splat_radius_px": SPLAT_RADIUS_PX,
    "coordinate_check": coordinate_check,
    "t0_robot_exact": bool(
        np.array_equal(base_robot[0], pert_robot[0])
    ),
    "baseline_mean_occlusion_ratio": float(
        np.nanmean(base_ratios)
    ),
    "perturb_mean_occlusion_ratio": float(
        np.nanmean(pert_ratios)
    ),
    "mean_delta_occlusion_ratio": float(
        np.nanmean(delta_ratios)
    ),
    "max_delta_occlusion_ratio": float(
        np.nanmax(delta_ratios)
    ),
    "min_delta_occlusion_ratio": float(
        np.nanmin(delta_ratios)
    ),
    "max_robot_point_l2_m": float(robot_l2.max()),
}

with open(json_path, "w") as file:
    json.dump(summary, file, indent=2)

print("=" * 72)
print("Action occlusion comparison completed")
print(json.dumps(summary, indent=2))
print(f"CSV: {csv_path}")
print(f"JSON: {json_path}")
print("=" * 72)
