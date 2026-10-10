from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

from transform_utils import pose2mat


# ============================================================
# Fixed GT-only selected sample
# Selected BEFORE looking at model prediction.
# ============================================================

EPISODE = "episode_00012190"
WINDOW = "460:471"
CAN_ID = 114
CAMERA = "camera_head"

ROOT = Path(
    "diagnostics/data/"
    "pointworld_behavior_task0001_restored/"
    "behavior/flows/task-0001"
)

OUT = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_gt_visual_top1"
)

OUT.mkdir(parents=True, exist_ok=True)

H5 = ROOT / f"{EPISODE}.hdf5"

MESH = (
    f"World__scene_0__can_of_soda_{CAN_ID}"
    "__base_link__visuals"
)


# ============================================================
# Read exact PointWorld annotations
# ============================================================

with h5py.File(H5, "r") as f:

    assert WINDOW in f, WINDOW

    cam = f[WINDOW][CAMERA]

    assert MESH in cam["local_scene_points"]
    assert MESH in cam["scene_mesh_trajectories"]

    local = np.asarray(
        cam["local_scene_points"][MESH][:],
        dtype=np.float64,
    )

    pose = np.asarray(
        cam["scene_mesh_trajectories"][MESH][:],
        dtype=np.float64,
    )


assert local.ndim == 2
assert local.shape[1] == 3

assert pose.shape == (11, 7)
assert np.isfinite(local).all()
assert np.isfinite(pose).all()


print("=" * 90)
print("POINTWORLD SODA-CAN GT VISUALIZATION")
print("=" * 90)

print("episode =", EPISODE)
print("window  =", WINDOW)
print("can_id  =", CAN_ID)
print("camera  =", CAMERA)
print("n_points =", len(local))

print()
print("pose shape =", pose.shape)

qnorm = np.linalg.norm(
    pose[:, 3:7],
    axis=1,
)

print(
    "quat norm range =",
    float(qnorm.min()),
    float(qnorm.max()),
)

assert np.allclose(
    qnorm,
    1.0,
    atol=2e-3,
    rtol=0.0,
)


# ============================================================
# Official PointWorld transform path:
#
# pose = [x, y, z, qx, qy, qz, qw]
# pose2mat((position, quaternion))
# ============================================================

world = []

for t in range(11):

    pos = pose[t, :3]
    quat = pose[t, 3:7]

    T = pose2mat(
        (
            pos.astype(np.float64),
            quat.astype(np.float64),
        )
    )

    assert T.shape == (4, 4)

    pts_world = (
        local @ T[:3, :3].T
        + T[:3, 3][None, :]
    )

    world.append(pts_world)


world = np.stack(world, axis=0)

assert world.shape == (
    11,
    len(local),
    3,
)


# ============================================================
# Reconstruction audit
# Rigid pose transform must preserve all pairwise distances.
# ============================================================

def pair_distances(x):
    delta = (
        x[:, None, :]
        - x[None, :, :]
    )

    d = np.linalg.norm(
        delta,
        axis=-1,
    )

    iu = np.triu_indices(
        len(x),
        k=1,
    )

    return d[iu]


d0 = pair_distances(world[0])
d10 = pair_distances(world[10])

pair_change_mm = (
    np.abs(d10 - d0)
    * 1000.0
)

print()
print("===== RIGID RECONSTRUCTION AUDIT =====")
print(
    "pair_change median_mm =",
    float(np.median(pair_change_mm)),
)
print(
    "pair_change p95_mm    =",
    float(np.quantile(pair_change_mm, 0.95)),
)
print(
    "pair_change max_mm    =",
    float(pair_change_mm.max()),
)


# This is an exact rigid-coordinate reconstruction.
# Numerical residual should be essentially zero.
assert pair_change_mm.max() < 1e-8, (
    "Rigid transform reconstruction unexpectedly "
    "changed pair distances",
    pair_change_mm.max(),
)


# ============================================================
# Motion audit
# ============================================================

centroid = world.mean(axis=1)

centroid_steps = np.linalg.norm(
    centroid[1:] - centroid[:-1],
    axis=1,
)

centroid_path_mm = (
    centroid_steps.sum()
    * 1000.0
)

centroid_endpoint_mm = (
    np.linalg.norm(
        centroid[-1] - centroid[0]
    )
    * 1000.0
)


print()
print("===== GT MOTION =====")

print(
    "centroid_path_mm =",
    centroid_path_mm,
)

print(
    "centroid_t0_t10_mm =",
    centroid_endpoint_mm,
)

print(
    "pose_translation_path_mm =",
    np.linalg.norm(
        pose[1:, :3] - pose[:-1, :3],
        axis=1,
    ).sum() * 1000.0,
)


# centroid translation should agree with pose translation
# because the object is transformed rigidly.
pose_path = (
    np.linalg.norm(
        pose[1:, :3] - pose[:-1, :3],
        axis=1,
    ).sum()
    * 1000.0
)

print(
    "centroid_vs_pose_path_diff_mm =",
    abs(
        centroid_path_mm
        - pose_path
    ),
)


# ============================================================
# Visualization 1:
# world-coordinate trajectory
# ============================================================

fig = plt.figure(
    figsize=(14, 10)
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

show_t = [
    0,
    2,
    4,
    6,
    8,
    10,
]

for t in show_t:

    pts = world[t]

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=12,
        alpha=0.60,
        label=f"t{t}",
    )


ax.plot(
    centroid[:, 0],
    centroid[:, 1],
    centroid[:, 2],
    marker="o",
    linewidth=2,
    label="centroid trajectory",
)

ax.scatter(
    centroid[0, 0],
    centroid[0, 1],
    centroid[0, 2],
    s=100,
    marker="o",
)

ax.scatter(
    centroid[-1, 0],
    centroid[-1, 1],
    centroid[-1, 2],
    s=120,
    marker="X",
)


# Equal-ish axis scaling
mins = world.reshape(-1, 3).min(axis=0)
maxs = world.reshape(-1, 3).max(axis=0)

center = (
    mins + maxs
) / 2.0

radius = (
    max(maxs - mins)
    / 2.0
)

if radius <= 0:
    radius = 0.1

ax.set_xlim(
    center[0] - radius,
    center[0] + radius,
)

ax.set_ylim(
    center[1] - radius,
    center[1] + radius,
)

ax.set_zlim(
    center[2] - radius,
    center[2] + radius,
)

ax.set_xlabel("World X (m)")
ax.set_ylabel("World Y (m)")
ax.set_zlabel("World Z (m)")

ax.set_title(
    "PointWorld BEHAVIOR task-0001\n"
    "GT soda-can motion: "
    f"{EPISODE}, {WINDOW}, can_{CAN_ID}"
)

ax.legend()

plt.tight_layout()

WORLD_PNG = (
    OUT /
    "top1_soda_can_gt_world_trajectory.png"
)

plt.savefig(
    WORLD_PNG,
    dpi=200,
)

plt.close()


# ============================================================
# Visualization 2:
# centroid-centered shapes
#
# Removes translation only.
# Rotation remains visible.
# If rigid reconstruction is correct, shape must remain intact.
# ============================================================

fig = plt.figure(
    figsize=(14, 10)
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

for t in show_t:

    pts = (
        world[t]
        - centroid[t][None, :]
    )

    ax.scatter(
        pts[:, 0] * 1000.0,
        pts[:, 1] * 1000.0,
        pts[:, 2] * 1000.0,
        s=15,
        alpha=0.65,
        label=f"t{t}",
    )


all_centered = (
    world
    - centroid[:, None, :]
) * 1000.0

mins = all_centered.reshape(
    -1,
    3,
).min(axis=0)

maxs = all_centered.reshape(
    -1,
    3,
).max(axis=0)

center2 = (
    mins + maxs
) / 2.0

radius2 = (
    max(maxs - mins)
    / 2.0
)

ax.set_xlim(
    center2[0] - radius2,
    center2[0] + radius2,
)

ax.set_ylim(
    center2[1] - radius2,
    center2[1] + radius2,
)

ax.set_zlim(
    center2[2] - radius2,
    center2[2] + radius2,
)

ax.set_xlabel("Centered X (mm)")
ax.set_ylabel("Centered Y (mm)")
ax.set_zlabel("Centered Z (mm)")

ax.set_title(
    "GT soda-can geometry after removing translation\n"
    "(rotation preserved; rigid shape must stay intact)"
)

ax.legend()

plt.tight_layout()

CENTERED_PNG = (
    OUT /
    "top1_soda_can_gt_centroid_centered.png"
)

plt.savefig(
    CENTERED_PNG,
    dpi=200,
)

plt.close()


# ============================================================
# Save exact arrays for downstream comparison
# ============================================================

NPZ = OUT / "top1_soda_can_gt_geometry.npz"

np.savez_compressed(
    NPZ,
    episode=np.array(EPISODE),
    window=np.array(WINDOW),
    can_id=np.array(CAN_ID),
    mesh_name=np.array(MESH),
    local_points=local,
    mesh_pose=pose,
    world_points=world,
    centroid=centroid,
)


print()
print("=" * 90)
print("OUTPUTS")
print("=" * 90)

print("world plot    =", WORLD_PNG)
print("centered plot =", CENTERED_PNG)
print("npz           =", NPZ)

print()
print("NO MODEL INFERENCE")
print("NO MBNSF")
print("OFFICIAL POINTWORLD TRANSFORM_UTILS USED")
print("GT-ONLY OUTCOME-BLIND SAMPLE SELECTION")
print("SUCCESS = True")
print("=" * 90)
