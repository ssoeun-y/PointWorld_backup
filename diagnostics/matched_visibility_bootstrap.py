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
OUTPUT_DIR = RESULT_ROOT / "visibility_matched_bootstrap_20260805"

DEPTH_TOLERANCE_M = 0.03
SPLAT_RADIUS_PX = 3
NUM_BOOTSTRAP = 5000
RANDOM_SEED = 42

# Matching features:
# camera-frame xyz와 baseline motion magnitude를 표준화해서 최근접 매칭.
MAX_MATCH_DISTANCE = 2.5


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

    positive = np.isfinite(z) & (z > 1e-6)

    u = np.full(z.shape, -1.0, dtype=np.float32)
    v = np.full(z.shape, -1.0, dtype=np.float32)

    projected = (
        intrinsic @ points_camera[positive].T
    ).T

    u[positive] = projected[:, 0] / projected[:, 2]
    v[positive] = projected[:, 1] / projected[:, 2]

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
    mask = (
        robot_exists.astype(bool)
        & np.isfinite(robot_points).all(axis=1)
    )

    robot_camera = transform_points(
        robot_points[mask],
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


def occlusion_mask(
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

    valid = valid_scene & projected

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

    indices = np.flatnonzero(valid)

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

    return occluded, valid, scene_camera


def bootstrap_paired_difference(
    differences,
    rng,
):
    differences = np.asarray(
        differences,
        dtype=np.float64,
    )
    differences = differences[
        np.isfinite(differences)
    ]

    if differences.size < 2:
        return {
            "count": int(differences.size),
            "mean_difference": None,
            "median_difference": None,
            "ci95_low": None,
            "ci95_high": None,
            "probability_mean_above_zero": None,
        }

    n = differences.size

    bootstrap_means = np.empty(
        NUM_BOOTSTRAP,
        dtype=np.float64,
    )

    for index in range(NUM_BOOTSTRAP):
        sample_indices = rng.integers(
            0,
            n,
            size=n,
        )
        bootstrap_means[index] = np.mean(
            differences[sample_indices]
        )

    return {
        "count": int(n),
        "mean_difference": float(
            np.mean(differences)
        ),
        "median_difference": float(
            np.median(differences)
        ),
        "ci95_low": float(
            np.percentile(bootstrap_means, 2.5)
        ),
        "ci95_high": float(
            np.percentile(bootstrap_means, 97.5)
        ),
        "probability_mean_above_zero": float(
            np.mean(bootstrap_means > 0)
        ),
    }


baseline = load_pt(BASELINE_PATH)
base_batch = baseline["batch"]

sample_key = baseline["sample_key"]

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

height, width = depth.shape
num_steps, num_points, _ = scene_flows.shape

base_velocity = np.diff(
    base_prediction,
    axis=0,
)
base_acceleration = np.diff(
    base_velocity,
    axis=0,
)

# Baseline occlusion과 camera coordinates를 미리 계산.
base_occ = []
base_valid = []
scene_camera_all = []

for timestep in range(num_steps):
    occ, valid, scene_camera = occlusion_mask(
        scene_flows[timestep],
        scene_exists[timestep],
        base_robot[timestep],
        base_robot_exists[timestep],
        intrinsic,
        extrinsic,
        width,
        height,
    )

    base_occ.append(occ)
    base_valid.append(valid)
    scene_camera_all.append(scene_camera)

base_occ = np.stack(base_occ)
base_valid = np.stack(base_valid)
scene_camera_all = np.stack(scene_camera_all)

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
action_details = {}

for action_index, path in enumerate(perturb_paths):
    payload = load_pt(path)

    if payload["sample_key"] != sample_key:
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

    pert_velocity = np.diff(
        prediction,
        axis=0,
    )
    pert_acceleration = np.diff(
        pert_velocity,
        axis=0,
    )

    velocity_change = np.linalg.norm(
        pert_velocity - base_velocity,
        axis=-1,
    )
    acceleration_change = np.linalg.norm(
        pert_acceleration - base_acceleration,
        axis=-1,
    )

    pert_occ = []
    pert_valid = []

    for timestep in range(num_steps):
        occ, valid, _ = occlusion_mask(
            scene_flows[timestep],
            scene_exists[timestep],
            robot[timestep],
            robot_exists[timestep],
            intrinsic,
            extrinsic,
            width,
            height,
        )

        pert_occ.append(occ)
        pert_valid.append(valid)

    pert_occ = np.stack(pert_occ)
    pert_valid = np.stack(pert_valid)

    velocity_differences = []
    acceleration_differences = []
    matching_distances = []

    candidate_count = 0
    matched_count = 0

    for timestep in range(1, num_steps):
        common_valid = (
            base_valid[timestep]
            & pert_valid[timestep]
        )

        newly_occluded = (
            common_valid
            & (~base_occ[timestep])
            & pert_occ[timestep]
        )

        persistent_visible = (
            common_valid
            & (~base_occ[timestep])
            & (~pert_occ[timestep])
        )

        treated_ids = np.flatnonzero(
            newly_occluded
        )
        control_ids = np.flatnonzero(
            persistent_visible
        )

        candidate_count += len(treated_ids)

        if (
            len(treated_ids) == 0
            or len(control_ids) == 0
        ):
            continue

        # Baseline motion magnitude를 matching feature로 사용.
        baseline_motion = np.linalg.norm(
            base_velocity[timestep - 1],
            axis=-1,
        )

        treated_features = np.concatenate(
            [
                scene_camera_all[
                    timestep,
                    treated_ids,
                ],
                baseline_motion[
                    treated_ids,
                    None,
                ],
            ],
            axis=1,
        )

        control_features = np.concatenate(
            [
                scene_camera_all[
                    timestep,
                    control_ids,
                ],
                baseline_motion[
                    control_ids,
                    None,
                ],
            ],
            axis=1,
        )

        combined = np.concatenate(
            [
                treated_features,
                control_features,
            ],
            axis=0,
        )

        feature_mean = np.mean(
            combined,
            axis=0,
        )
        feature_std = np.std(
            combined,
            axis=0,
        )
        feature_std[
            feature_std < 1e-8
        ] = 1.0

        treated_z = (
            treated_features - feature_mean
        ) / feature_std
        control_z = (
            control_features - feature_mean
        ) / feature_std

        # Control 재사용을 허용하는 nearest-neighbor matching.
        for local_index, treated_id in enumerate(
            treated_ids
        ):
            distances = np.linalg.norm(
                control_z - treated_z[local_index],
                axis=1,
            )

            nearest_position = int(
                np.argmin(distances)
            )
            nearest_distance = float(
                distances[nearest_position]
            )

            if nearest_distance > MAX_MATCH_DISTANCE:
                continue

            control_id = int(
                control_ids[nearest_position]
            )

            velocity_differences.append(
                float(
                    velocity_change[
                        timestep - 1,
                        treated_id,
                    ]
                    - velocity_change[
                        timestep - 1,
                        control_id,
                    ]
                )
            )

            if timestep >= 2:
                acceleration_differences.append(
                    float(
                        acceleration_change[
                            timestep - 2,
                            treated_id,
                        ]
                        - acceleration_change[
                            timestep - 2,
                            control_id,
                        ]
                    )
                )

            matching_distances.append(
                nearest_distance
            )
            matched_count += 1

    rng = np.random.default_rng(
        RANDOM_SEED + action_index
    )

    velocity_result = bootstrap_paired_difference(
        velocity_differences,
        rng,
    )
    acceleration_result = bootstrap_paired_difference(
        acceleration_differences,
        rng,
    )

    row = {
        "action": action,
        "candidate_newly_occluded_count": (
            candidate_count
        ),
        "matched_count": matched_count,
        "match_rate": (
            float(matched_count / candidate_count)
            if candidate_count > 0
            else None
        ),
        "mean_matching_distance": (
            float(np.mean(matching_distances))
            if matching_distances
            else None
        ),
        "velocity_mean_paired_difference_m": (
            velocity_result["mean_difference"]
        ),
        "velocity_ci95_low_m": (
            velocity_result["ci95_low"]
        ),
        "velocity_ci95_high_m": (
            velocity_result["ci95_high"]
        ),
        "velocity_probability_above_zero": (
            velocity_result[
                "probability_mean_above_zero"
            ]
        ),
        "acceleration_mean_paired_difference_m": (
            acceleration_result["mean_difference"]
        ),
        "acceleration_ci95_low_m": (
            acceleration_result["ci95_low"]
        ),
        "acceleration_ci95_high_m": (
            acceleration_result["ci95_high"]
        ),
        "acceleration_probability_above_zero": (
            acceleration_result[
                "probability_mean_above_zero"
            ]
        ),
    }

    rows.append(row)

    action_details[action] = {
        "velocity": velocity_result,
        "acceleration": acceleration_result,
    }

rows.sort(
    key=lambda row: (
        row["velocity_mean_paired_difference_m"]
        if row["velocity_mean_paired_difference_m"]
        is not None
        else -np.inf
    ),
    reverse=True,
)

velocity_significant_positive = [
    row["action"]
    for row in rows
    if (
        row["velocity_ci95_low_m"] is not None
        and row["velocity_ci95_low_m"] > 0
    )
]

acceleration_significant_positive = [
    row["action"]
    for row in rows
    if (
        row["acceleration_ci95_low_m"] is not None
        and row["acceleration_ci95_low_m"] > 0
    )
]

summary = {
    "sample_key": sample_key,
    "num_actions": len(rows),
    "matching_features": [
        "camera_x",
        "camera_y",
        "camera_z",
        "baseline_velocity_magnitude",
    ],
    "max_standardized_match_distance": (
        MAX_MATCH_DISTANCE
    ),
    "num_bootstrap": NUM_BOOTSTRAP,
    "velocity_significant_positive_actions": (
        velocity_significant_positive
    ),
    "acceleration_significant_positive_actions": (
        acceleration_significant_positive
    ),
    "velocity_significant_positive_count": len(
        velocity_significant_positive
    ),
    "acceleration_significant_positive_count": len(
        acceleration_significant_positive
    ),
    "rows": rows,
    "interpretation": (
        "A positive paired difference means newly occluded "
        "points changed more than spatially and motion-matched "
        "persistent-visible controls. A 95% CI entirely above "
        "zero is evidence of higher local temporal sensitivity, "
        "not prediction error."
    ),
}

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

csv_path = OUTPUT_DIR / "matched_bootstrap.csv"
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
print("MATCHED VISIBILITY BOOTSTRAP")
print("=" * 90)

for row in rows:
    print(
        f"{row['action']:<18} "
        f"matched={row['matched_count']:<5} "
        f"vel_diff={row['velocity_mean_paired_difference_m']} "
        f"vel_CI=[{row['velocity_ci95_low_m']}, "
        f"{row['velocity_ci95_high_m']}] "
        f"acc_diff={row['acceleration_mean_paired_difference_m']} "
        f"acc_CI=[{row['acceleration_ci95_low_m']}, "
        f"{row['acceleration_ci95_high_m']}]"
    )

print()
print(json.dumps(summary, indent=2))
print("CSV:", csv_path)
print("JSON:", json_path)
