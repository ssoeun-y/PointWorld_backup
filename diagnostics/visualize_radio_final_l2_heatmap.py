#!/usr/bin/env python3

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NPZ = Path(
    "diagnostics/results/"
    "behavior_radio_debris_visual_smoke/"
    "task-0000_episode_00000010-175_186_radio_89_point_errors.npz"
)

OUT = Path(
    "diagnostics/results/"
    "behavior_radio_debris_visual_smoke/"
    "radio_89_t10_final_l2_heatmap.png"
)


def equal_3d(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)

    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0

    if radius <= 0:
        radius = 1e-3

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


d = np.load(NPZ)

gt = d["gt_xyz"]
pred = d["pred_xyz"]
l2 = d["l2"]
valid = d["valid"].astype(bool)

assert gt.shape == (11, 206, 3)
assert pred.shape == (11, 206, 3)
assert l2.shape == (11, 206)

t = 10

assert valid[t].all()

gt_t = gt[t]
pred_t = pred[t]
err_t = l2[t]

all_xyz = np.concatenate(
    [gt_t, pred_t],
    axis=0,
)

fig = plt.figure(figsize=(14, 6))

# ---------------------------------------------------------
# Prediction colored by final L2
# ---------------------------------------------------------
ax1 = fig.add_subplot(121, projection="3d")

sc1 = ax1.scatter(
    pred_t[:, 0],
    pred_t[:, 1],
    pred_t[:, 2],
    c=err_t * 1000.0,
    cmap="turbo",
    s=28,
)

ax1.set_title(
    "PointWorld prediction t=10\n"
    "Color = final L2 error"
)

equal_3d(ax1, all_xyz)

cb1 = fig.colorbar(
    sc1,
    ax=ax1,
    shrink=0.72,
    pad=0.08,
)

cb1.set_label("L2 error [mm]")


# ---------------------------------------------------------
# GT + colored prediction overlay
# ---------------------------------------------------------
ax2 = fig.add_subplot(122, projection="3d")

ax2.scatter(
    gt_t[:, 0],
    gt_t[:, 1],
    gt_t[:, 2],
    s=12,
    c="gray",
    alpha=0.30,
    label="GT",
)

sc2 = ax2.scatter(
    pred_t[:, 0],
    pred_t[:, 1],
    pred_t[:, 2],
    c=err_t * 1000.0,
    cmap="turbo",
    s=28,
    label="Prediction",
)

ax2.set_title(
    "GT vs Prediction t=10\n"
    "Prediction colored by error"
)

equal_3d(ax2, all_xyz)

cb2 = fig.colorbar(
    sc2,
    ax=ax2,
    shrink=0.72,
    pad=0.08,
)

cb2.set_label("L2 error [mm]")

for ax in (ax1, ax2):
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

plt.tight_layout()

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    OUT,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

print("===== FINAL L2 HEATMAP =====")
print("point_count =", len(err_t))
print("final_l2_mean_mm =", float(err_t.mean() * 1000))
print("final_l2_median_mm =", float(np.median(err_t) * 1000))
print("final_l2_max_mm =", float(err_t.max() * 1000))
print("max_error_local_index =", int(np.argmax(err_t)))

print(
    "max_error_scene_point_index =",
    int(
        d["scene_point_index"][
            np.argmax(err_t)
        ]
    ),
)

print("PNG =", OUT)
print("SUCCESS_HEATMAP =", OUT.is_file())
