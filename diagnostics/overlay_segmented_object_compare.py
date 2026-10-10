from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

def find_first(npz, candidates):
    for k in candidates:
        if k in npz:
            return npz[k], k
    raise KeyError(f"Cannot find any of keys: {candidates}")

def get_frame(arr, frame_idx):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if frame_idx < 0:
            frame_idx = arr.shape[0] + frame_idx
        return arr[frame_idx]
    elif arr.ndim == 2:
        return arr
    else:
        raise RuntimeError(f"Unexpected array shape: {arr.shape}")

def set_axes_equal(ax, pts_a, pts_b):
    pts = np.concatenate([pts_a, pts_b], axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    if radius <= 0:
        radius = 1e-3
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

p = argparse.ArgumentParser()
p.add_argument("--orig", type=str, required=True)
p.add_argument("--corr", type=str, required=True)
p.add_argument("--frame", type=int, default=-1)
p.add_argument("--object_id", type=int, default=None)
p.add_argument("--out", type=str, required=True)
args = p.parse_args()

orig_npz = np.load(args.orig, allow_pickle=True)
corr_npz = np.load(args.corr, allow_pickle=True)

xyz_candidates = [
    "pred_world_points",
    "pred_points",
    "scene_xyz",
    "scene_points",
    "xyz",
]
id_candidates = [
    "scene_object_id",
    "object_id",
    "pred_object_id",
]

orig_xyz_all, orig_xyz_key = find_first(orig_npz, xyz_candidates)
corr_xyz_all, corr_xyz_key = find_first(corr_npz, xyz_candidates)
orig_id_all, orig_id_key = find_first(orig_npz, id_candidates)
corr_id_all, corr_id_key = find_first(corr_npz, id_candidates)

orig_xyz = get_frame(orig_xyz_all, args.frame)
corr_xyz = get_frame(corr_xyz_all, args.frame)
orig_ids = get_frame(orig_id_all, args.frame)
corr_ids = get_frame(corr_id_all, args.frame)

orig_ids = np.asarray(orig_ids).reshape(-1)
corr_ids = np.asarray(corr_ids).reshape(-1)

common_ids = sorted(set(orig_ids.tolist()) & set(corr_ids.tolist()))
common_ids = [x for x in common_ids if x >= 0]

if len(common_ids) == 0:
    raise RuntimeError("No common object IDs found between original and corrected npz.")

if args.object_id is None:
    counts = []
    for oid in common_ids:
        c = int((orig_ids == oid).sum() + (corr_ids == oid).sum())
        counts.append((c, oid))
    counts.sort(reverse=True)
    object_id = counts[0][1]
else:
    object_id = args.object_id
    if object_id not in common_ids:
        raise RuntimeError(
            f"Requested object_id={object_id} not found in common IDs: {common_ids}"
        )

orig_mask = (orig_ids == object_id)
corr_mask = (corr_ids == object_id)

orig_xyz = orig_xyz[orig_mask]
corr_xyz = corr_xyz[corr_mask]

if len(orig_xyz) == 0 or len(corr_xyz) == 0:
    raise RuntimeError("Selected object has zero points in original or corrected result.")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    orig_xyz[:, 0], orig_xyz[:, 1], orig_xyz[:, 2],
    s=8, alpha=0.75, c="tab:blue", label="Original"
)
ax.scatter(
    corr_xyz[:, 0], corr_xyz[:, 1], corr_xyz[:, 2],
    s=8, alpha=0.75, c="tab:red", label="Corrected"
)

set_axes_equal(ax, orig_xyz, corr_xyz)

title = (
    f"Segmented object overlay | object_id={object_id} | frame={args.frame}\n"
    f"Blue=Original, Red=Corrected"
)
ax.set_title(title, pad=20)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(args.out, dpi=200)
print("SAVE:", args.out)
print("orig xyz key:", orig_xyz_key)
print("corr xyz key:", corr_xyz_key)
print("orig id key :", orig_id_key)
print("corr id key :", corr_id_key)
print("object_id   :", object_id)
print("orig points :", len(orig_xyz))
print("corr points :", len(corr_xyz))
