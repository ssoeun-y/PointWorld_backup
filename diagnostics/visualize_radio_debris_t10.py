#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


NPZ = Path(
    "diagnostics/results/"
    "behavior_radio_debris_visual_smoke/"
    "task-0000_episode_00000010-175_186_radio_89_point_errors.npz"
)

OUT = Path(
    "diagnostics/results/"
    "behavior_radio_debris_visual_smoke/"
    "radio_89_t10_final_top5.png"
)


def set_equal_3d(ax, xyz):
    xyz = np.asarray(xyz)

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)

    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0

    if radius == 0:
        radius = 1e-3

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

    ax.set_box_aspect((1, 1, 1))


d = np.load(NPZ)

gt = d["gt_xyz"]
pred = d["pred_xyz"]
l2 = d["l2"]

top5 = d["top_final_5pct"].astype(bool)
valid = d["valid"].astype(bool)

assert gt.shape == (11, 206, 3)
assert pred.shape == (11, 206, 3)
assert int(top5.sum()) == 11
assert valid[10].all()

t = 10

gt_t = gt[t]
pred_t = pred[t]
l2_t = l2[t]

all_xyz = np.concatenate(
    [gt_t, pred_t],
    axis=0,
)

fig = plt.figure(
    figsize=(18, 6)
)

# ---------------------------------------------------------
# 1. GT
# ---------------------------------------------------------
ax1 = fig.add_subplot(
    131,
    projection="3d",
)

ax1.scatter(
    gt_t[:, 0],
    gt_t[:, 1],
    gt_t[:, 2],
    s=12,
    c="lightgray",
)

ax1.scatter(
    gt_t[top5, 0],
    gt_t[top5, 1],
    gt_t[top5, 2],
    s=50,
    c="red",
)

ax1.set_title(
    "GT t=10\nTop 5% final-error point IDs highlighted"
)

set_equal_3d(
    ax1,
    all_xyz,
)

# ---------------------------------------------------------
# 2. Prediction
# ---------------------------------------------------------
ax2 = fig.add_subplot(
    132,
    projection="3d",
)

ax2.scatter(
    pred_t[:, 0],
    pred_t[:, 1],
    pred_t[:, 2],
    s=12,
    c="lightgray",
)

ax2.scatter(
    pred_t[top5, 0],
    pred_t[top5, 1],
    pred_t[top5, 2],
    s=50,
    c="red",
)

ax2.set_title(
    "PointWorld prediction t=10\nTop 5% final L2"
)

set_equal_3d(
    ax2,
    all_xyz,
)

# ---------------------------------------------------------
# 3. GT ↔ Pred overlay
# ---------------------------------------------------------
ax3 = fig.add_subplot(
    133,
    projection="3d",
)

ax3.scatter(
    gt_t[:, 0],
    gt_t[:, 1],
    gt_t[:, 2],
    s=10,
    c="gray",
    alpha=0.45,
    label="GT",
)

ax3.scatter(
    pred_t[:, 0],
    pred_t[:, 1],
    pred_t[:, 2],
    s=10,
    c="blue",
    alpha=0.45,
    label="Pred",
)

ax3.scatter(
    pred_t[top5, 0],
    pred_t[top5, 1],
    pred_t[top5, 2],
    s=55,
    c="red",
    label="Top 5% final error",
)

# Draw GT -> Pred correspondence only for top-error points.
for i in np.flatnonzero(top5):
    ax3.plot(
        [gt_t[i, 0], pred_t[i, 0]],
        [gt_t[i, 1], pred_t[i, 1]],
        [gt_t[i, 2], pred_t[i, 2]],
        c="red",
        linewidth=1.0,
        alpha=0.8,
    )

ax3.set_title(
    "GT vs Prediction t=10"
)

ax3.legend()

set_equal_3d(
    ax3,
    all_xyz,
)

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

fig.suptitle(
    "radio_89 — Final-timestep point-level error",
    fontsize=14,
)

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

top_idx = np.flatnonzero(top5)

print("===== T10 FINAL TOP-5% VISUAL =====")
print("top5_count =", len(top_idx))
print("top5_scene_point_indices =", d["scene_point_index"][top_idx].tolist())

print(
    "top5_final_l2_m =",
    np.round(
        l2_t[top5],
        6,
    ).tolist(),
)

print(
    "top5_final_l2_mean_m =",
    float(
        l2_t[top5].mean()
    ),
)

print(
    "all_final_l2_mean_m =",
    float(
        l2_t.mean()
    ),
)

print("PNG =", OUT)
print("SUCCESS_VIS =", OUT.is_file())
