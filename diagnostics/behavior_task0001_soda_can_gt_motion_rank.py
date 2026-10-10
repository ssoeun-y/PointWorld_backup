from pathlib import Path
import re
import math

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(
    "diagnostics/data/"
    "pointworld_behavior_task0001_restored/"
    "behavior/flows/task-0001"
)

OUT = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_gt_motion_rank"
)

OUT.mkdir(parents=True, exist_ok=True)

FILES = sorted(ROOT.glob("episode_*.hdf5"))

assert len(FILES) == 200, len(FILES)

CAN_RE = re.compile(
    r"World__scene_0__can_of_soda_(\d+)__base_link__visuals"
)


rows = []

quat_norm_min = np.inf
quat_norm_max = -np.inf
n_pose_checked = 0


for fi, path in enumerate(FILES):

    if fi % 20 == 0:
        print(
            f"processing {fi:3d}/{len(FILES)}  {path.name}"
        )

    with h5py.File(path, "r") as f:

        clip_keys = sorted(
            k for k in f.keys()
            if ":" in k
            and isinstance(f[k], h5py.Group)
        )

        for clip_key in clip_keys:

            clip = f[clip_key]

            if "camera_head" not in clip:
                continue

            cam = clip["camera_head"]

            if "scene_mesh_trajectories" not in cam:
                continue

            traj_group = cam["scene_mesh_trajectories"]

            point_group = (
                cam["local_scene_points"]
                if "local_scene_points" in cam
                else None
            )

            for mesh_name in traj_group.keys():

                m = CAN_RE.fullmatch(mesh_name)

                if m is None:
                    continue

                can_id = int(m.group(1))

                pose = np.asarray(
                    traj_group[mesh_name][:],
                    dtype=np.float64,
                )

                assert pose.shape == (11, 7), (
                    path,
                    clip_key,
                    mesh_name,
                    pose.shape,
                )

                assert np.isfinite(pose).all()

                # --------------------------------------------------
                # Schema sanity:
                # observed format = 3 position values + 4D unit quat
                # We do not proceed if the last four values fail
                # the unit-quaternion sanity check.
                # --------------------------------------------------

                q = pose[:, 3:7]

                qnorm = np.linalg.norm(q, axis=1)

                quat_norm_min = min(
                    quat_norm_min,
                    float(qnorm.min()),
                )

                quat_norm_max = max(
                    quat_norm_max,
                    float(qnorm.max()),
                )

                n_pose_checked += 1

                if not np.allclose(
                    qnorm,
                    1.0,
                    atol=2e-3,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        (
                            "Quaternion sanity failed",
                            path,
                            clip_key,
                            mesh_name,
                            qnorm,
                        )
                    )

                # --------------------------------------------------
                # Translation: compare only WITHIN this 11-frame clip
                # --------------------------------------------------

                xyz = pose[:, :3]

                step = np.linalg.norm(
                    xyz[1:] - xyz[:-1],
                    axis=1,
                )

                endpoint = np.linalg.norm(
                    xyz[-1] - xyz[0]
                )

                path_length = step.sum()

                max_step = step.max()

                # --------------------------------------------------
                # Rotation change using quaternion dot product.
                #
                # q and -q represent the same orientation, hence abs.
                # Quaternion component order does not affect this
                # relative-angle calculation as long as all rows use
                # the same ordering.
                # --------------------------------------------------

                q0 = q[0] / np.linalg.norm(q[0])
                q1 = q[-1] / np.linalg.norm(q[-1])

                dot = float(
                    np.clip(
                        abs(np.dot(q0, q1)),
                        0.0,
                        1.0,
                    )
                )

                rotation_deg = math.degrees(
                    2.0 * math.acos(dot)
                )

                # max rotation from t0
                rot_from_t0 = []

                for qt in q:
                    qt = qt / np.linalg.norm(qt)

                    d = float(
                        np.clip(
                            abs(np.dot(q0, qt)),
                            0.0,
                            1.0,
                        )
                    )

                    rot_from_t0.append(
                        math.degrees(
                            2.0 * math.acos(d)
                        )
                    )

                max_rotation_deg = max(rot_from_t0)

                # Point count only; no filtering yet
                n_local_points = np.nan

                if (
                    point_group is not None
                    and mesh_name in point_group
                ):
                    pts = point_group[mesh_name]
                    n_local_points = int(
                        pts.shape[0]
                    )

                start_s, end_s = clip_key.split(":")

                rows.append({
                    "episode": path.stem,
                    "episode_file": path.name,
                    "window": clip_key,
                    "window_start": int(start_s),
                    "window_end": int(end_s),
                    "can_id": can_id,
                    "mesh_name": mesh_name,
                    "n_local_points_head": n_local_points,

                    "translation_t0_t10_mm":
                        endpoint * 1000.0,

                    "translation_path_mm":
                        path_length * 1000.0,

                    "max_step_translation_mm":
                        max_step * 1000.0,

                    "rotation_t0_t10_deg":
                        rotation_deg,

                    "max_rotation_from_t0_deg":
                        max_rotation_deg,

                    "x0": xyz[0, 0],
                    "y0": xyz[0, 1],
                    "z0": xyz[0, 2],

                    "x10": xyz[-1, 0],
                    "y10": xyz[-1, 1],
                    "z10": xyz[-1, 2],
                })


df = pd.DataFrame(rows)

assert len(df) > 0


# ------------------------------------------------------------
# Save complete inventory
# ------------------------------------------------------------

csv_all = OUT / "soda_can_gt_motion_all_windows.csv"

df.to_csv(
    csv_all,
    index=False,
)


# ------------------------------------------------------------
# Primary ranking:
#
# translation_path_mm is used because it measures actual motion
# across the whole 11-frame window and does not disappear if an
# object partially returns toward its starting position.
#
# This is GT-only selection; prediction outcome is not involved.
# ------------------------------------------------------------

ranked = df.sort_values(
    [
        "translation_path_mm",
        "max_rotation_from_t0_deg",
    ],
    ascending=False,
).reset_index(drop=True)


csv_ranked = OUT / "soda_can_gt_motion_ranked.csv"

ranked.to_csv(
    csv_ranked,
    index=False,
)


# ------------------------------------------------------------
# Episode-disjoint ranking:
# one most-moving can/window per episode
# ------------------------------------------------------------

episode_disjoint = (
    ranked
    .drop_duplicates(
        subset=["episode"],
        keep="first",
    )
    .reset_index(drop=True)
)


csv_disjoint = (
    OUT /
    "soda_can_gt_motion_episode_disjoint.csv"
)

episode_disjoint.to_csv(
    csv_disjoint,
    index=False,
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 100)
print("SODA-CAN GT MOTION SUMMARY")
print("=" * 100)

print("n_episode_files =", len(FILES))
print("n_can_windows   =", len(df))
print(
    "episodes_with_can_windows =",
    df["episode"].nunique(),
)

print(
    "can_ids =",
    sorted(
        df["can_id"].unique().tolist()
    ),
)

print()
print("POSE FORMAT SANITY")
print(
    "n_pose_checked =",
    n_pose_checked,
)
print(
    "quat_norm_min =",
    quat_norm_min,
)
print(
    "quat_norm_max =",
    quat_norm_max,
)

print()
print("GT translation_path_mm distribution")
print(
    ranked["translation_path_mm"]
    .describe(
        percentiles=[
            0.5,
            0.75,
            0.9,
            0.95,
            0.99,
        ]
    )
)

print()
print("=" * 100)
print("TOP 20 GT-MOVING WINDOWS")
print("=" * 100)

cols = [
    "episode",
    "window",
    "can_id",
    "n_local_points_head",
    "translation_path_mm",
    "translation_t0_t10_mm",
    "max_step_translation_mm",
    "max_rotation_from_t0_deg",
]

print(
    ranked[cols]
    .head(20)
    .to_string(index=False)
)


print()
print("=" * 100)
print("TOP 15 EPISODE-DISJOINT GT-MOVING WINDOWS")
print("=" * 100)

print(
    episode_disjoint[cols]
    .head(15)
    .to_string(index=False)
)


# ------------------------------------------------------------
# Real-data visualization
# No generated/fabricated image.
# ------------------------------------------------------------

top_plot = episode_disjoint.head(15).copy()

labels = [
    f"{r.episode.replace('episode_', '')}\n"
    f"{r.window}\ncan{r.can_id}"
    for r in top_plot.itertuples()
]

plt.figure(figsize=(14, 6))

plt.bar(
    np.arange(len(top_plot)),
    top_plot["translation_path_mm"].to_numpy(),
)

plt.xticks(
    np.arange(len(top_plot)),
    labels,
    rotation=60,
    ha="right",
)

plt.ylabel(
    "GT soda-can translation path within 11-frame window (mm)"
)

plt.title(
    "PointWorld BEHAVIOR task-0001: "
    "episode-disjoint soda-can GT motion"
)

plt.tight_layout()

plot_path = (
    OUT /
    "soda_can_gt_motion_episode_disjoint_top15.png"
)

plt.savefig(
    plot_path,
    dpi=180,
)

plt.close()


print()
print("=" * 100)
print("OUTPUTS")
print("=" * 100)

print("all      =", csv_all)
print("ranked   =", csv_ranked)
print("disjoint =", csv_disjoint)
print("plot     =", plot_path)

print()
print("NO INFERENCE")
print("NO MBNSF")
print("GT-ONLY SELECTION")
print("POINTWORLD OFFICIAL RESTORED HDF5 ONLY")
print("SUCCESS = True")
print("=" * 100)
