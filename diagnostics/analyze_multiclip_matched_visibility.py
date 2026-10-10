#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np
import torch


MANIFEST_PATH = Path(
    "diagnostics/data/visibility_single_sample_wds_20260805/manifest.json"
)
IDENTITY_ROOT = Path(
    "diagnostics/results/visibility_multiclip_identity_20260805"
)
PERTURB_ROOT = Path(
    "diagnostics/results/visibility_multiclip_perturbations_20260805"
)
OUTPUT_DIR = Path(
    "diagnostics/results/visibility_multiclip_matched_analysis_20260805"
)

DEPTH_TOLERANCE_M = 0.03
SPLAT_RADIUS_PX = 3
MAX_MATCH_DISTANCE = 2.5
NUM_BOOTSTRAP = 10000
RANDOM_SEED = 42


def load_pt(path):
    if not path.exists():
        raise FileNotFoundError(path)

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


def project_points(points_camera, intrinsic, width, height):
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
        scene_points.shape[0],
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


def mean_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return float(values.mean())


def percentile_bootstrap(values, rng):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "probability_above_zero": None,
        }

    means = np.empty(
        NUM_BOOTSTRAP,
        dtype=np.float64,
    )

    for index in range(NUM_BOOTSTRAP):
        sample = rng.choice(
            values,
            size=values.size,
            replace=True,
        )
        means[index] = sample.mean()

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "probability_above_zero": float(
            np.mean(means > 0)
        ),
    }


manifest = json.loads(
    MANIFEST_PATH.read_text()
)

actions = [
    f"joint{joint}_{sign}015"
    for joint in range(1, 8)
    for sign in ("plus", "minus")
]

all_rows = []
clip_rows = []

for clip_item in manifest:
    clip_id = clip_item["clip_id"]
    expected_key = clip_item["sample_key"]

    identity = load_pt(
        IDENTITY_ROOT
        / clip_id
        / "identity.pt"
    )

    if list(identity["sample_key"]) != [expected_key]:
        raise RuntimeError(
            f"{clip_id}: identity sample mismatch"
        )

    batch = identity["batch"]

    scene_flows = npy(
        batch["scene_flows"]
    )[0]
    scene_exists = npy(
        batch["scene_exists"]
    )[0]
    base_robot = npy(
        batch["robot_flows"]
    )[0]
    base_robot_exists = npy(
        batch["robot_exists"]
    )[0]
    intrinsic = npy(
        batch["cam0_intrinsic"]
    )[0]
    extrinsic = npy(
        batch["cam0_extrinsic"]
    )[0]
    depth = npy(
        batch["cam0_initial_depth"]
    )[0]
    base_prediction = npy(
        identity["outputs"]["scene_flows"]
    )[0]

    height, width = depth.shape
    num_steps = scene_flows.shape[0]

    base_velocity = np.diff(
        base_prediction,
        axis=0,
    )
    base_acceleration = np.diff(
        base_velocity,
        axis=0,
    )

    base_occ = []
    base_valid = []
    scene_camera_all = []

    for timestep in range(num_steps):
        occ, valid, scene_camera = compute_occlusion(
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

    clip_velocity_differences = []
    clip_acceleration_differences = []
    clip_action_velocity_means = []
    clip_action_acceleration_means = []

    for action in actions:
        perturb = load_pt(
            PERTURB_ROOT
            / clip_id
            / action
            / "perturb.pt"
        )

        if list(perturb["sample_key"]) != [expected_key]:
            raise RuntimeError(
                f"{clip_id}/{action}: sample mismatch"
            )

        robot = npy(
            perturb["batch"]["robot_flows"]
        )[0]
        robot_exists = npy(
            perturb["batch"]["robot_exists"]
        )[0]
        prediction = npy(
            perturb["outputs"]["scene_flows"]
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
            occ, valid, _ = compute_occlusion(
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

            feature_mean = combined.mean(axis=0)
            feature_std = combined.std(axis=0)
            feature_std[feature_std < 1e-8] = 1.0

            treated_z = (
                treated_features - feature_mean
            ) / feature_std
            control_z = (
                control_features - feature_mean
            ) / feature_std

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

        velocity_mean = mean_or_none(
            velocity_differences
        )
        acceleration_mean = mean_or_none(
            acceleration_differences
        )

        if velocity_mean is not None:
            clip_action_velocity_means.append(
                velocity_mean
            )

        if acceleration_mean is not None:
            clip_action_acceleration_means.append(
                acceleration_mean
            )

        clip_velocity_differences.extend(
            velocity_differences
        )
        clip_acceleration_differences.extend(
            acceleration_differences
        )

        common_valid_all = (
            base_valid
            & pert_valid
        )

        baseline_occ_ratio = float(
            base_occ[common_valid_all].mean()
        )
        perturb_occ_ratio = float(
            pert_occ[common_valid_all].mean()
        )

        all_rows.append(
            {
                "clip_id": clip_id,
                "sample_key": expected_key,
                "action": action,
                "candidate_newly_occluded_count": (
                    candidate_count
                ),
                "matched_count": matched_count,
                "match_rate": (
                    matched_count / candidate_count
                    if candidate_count > 0
                    else None
                ),
                "mean_matching_distance": (
                    mean_or_none(matching_distances)
                ),
                "baseline_occlusion_ratio": (
                    baseline_occ_ratio
                ),
                "perturb_occlusion_ratio": (
                    perturb_occ_ratio
                ),
                "occlusion_delta": (
                    perturb_occ_ratio
                    - baseline_occ_ratio
                ),
                "velocity_mean_paired_difference_m": (
                    velocity_mean
                ),
                "acceleration_mean_paired_difference_m": (
                    acceleration_mean
                ),
            }
        )

    clip_rows.append(
        {
            "clip_id": clip_id,
            "sample_key": expected_key,
            "num_actions": len(actions),
            "velocity_positive_action_count": int(
                sum(
                    value > 0
                    for value
                    in clip_action_velocity_means
                )
            ),
            "acceleration_positive_action_count": int(
                sum(
                    value > 0
                    for value
                    in clip_action_acceleration_means
                )
            ),
            "mean_action_velocity_difference_m": (
                mean_or_none(
                    clip_action_velocity_means
                )
            ),
            "mean_action_acceleration_difference_m": (
                mean_or_none(
                    clip_action_acceleration_means
                )
            ),
            "pooled_pair_velocity_difference_m": (
                mean_or_none(
                    clip_velocity_differences
                )
            ),
            "pooled_pair_acceleration_difference_m": (
                mean_or_none(
                    clip_acceleration_differences
                )
            ),
            "total_matched_velocity_pairs": len(
                clip_velocity_differences
            ),
            "total_matched_acceleration_pairs": len(
                clip_acceleration_differences
            ),
        }
    )

velocity_clip_means = [
    row["mean_action_velocity_difference_m"]
    for row in clip_rows
]

acceleration_clip_means = [
    row["mean_action_acceleration_difference_m"]
    for row in clip_rows
]

rng = np.random.default_rng(RANDOM_SEED)

velocity_bootstrap = percentile_bootstrap(
    velocity_clip_means,
    rng,
)

acceleration_bootstrap = percentile_bootstrap(
    acceleration_clip_means,
    rng,
)

summary = {
    "num_clips": len(clip_rows),
    "num_actions_per_clip": len(actions),
    "num_total_runs": len(all_rows),
    "depth_tolerance_m": DEPTH_TOLERANCE_M,
    "splat_radius_px": SPLAT_RADIUS_PX,
    "max_standardized_match_distance": (
        MAX_MATCH_DISTANCE
    ),
    "num_bootstrap": NUM_BOOTSTRAP,
    "velocity_clip_level_bootstrap": (
        velocity_bootstrap
    ),
    "acceleration_clip_level_bootstrap": (
        acceleration_bootstrap
    ),
    "clips_with_positive_mean_velocity_difference": int(
        sum(
            value is not None and value > 0
            for value in velocity_clip_means
        )
    ),
    "clips_with_positive_mean_acceleration_difference": int(
        sum(
            value is not None and value > 0
            for value in acceleration_clip_means
        )
    ),
    "total_action_rows_with_positive_velocity_difference": int(
        sum(
            row[
                "velocity_mean_paired_difference_m"
            ] is not None
            and row[
                "velocity_mean_paired_difference_m"
            ] > 0
            for row in all_rows
        )
    ),
    "total_action_rows_with_positive_acceleration_difference": int(
        sum(
            row[
                "acceleration_mean_paired_difference_m"
            ] is not None
            and row[
                "acceleration_mean_paired_difference_m"
            ] > 0
            for row in all_rows
        )
    ),
    "clip_summaries": clip_rows,
    "interpretation_warning": (
        "These results measure counterfactual output "
        "sensitivity, not prediction error. The five clips "
        "come from one DROID session, so this establishes "
        "within-session repeatability only."
    ),
}

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

with open(
    OUTPUT_DIR / "action_level_results.csv",
    "w",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(all_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(all_rows)

with open(
    OUTPUT_DIR / "clip_level_results.csv",
    "w",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(clip_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(clip_rows)

with open(
    OUTPUT_DIR / "summary.json",
    "w",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
    )

print("=" * 90)
print("MULTICLIP MATCHED VISIBILITY ANALYSIS COMPLETE")
print("=" * 90)

for row in clip_rows:
    print(
        row["clip_id"],
        "vel_mean =",
        row["mean_action_velocity_difference_m"],
        "acc_mean =",
        row["mean_action_acceleration_difference_m"],
        "vel_positive_actions =",
        row["velocity_positive_action_count"],
        "/14",
        "acc_positive_actions =",
        row["acceleration_positive_action_count"],
        "/14",
    )

print()
print(json.dumps(summary, indent=2))
print("output =", OUTPUT_DIR)
