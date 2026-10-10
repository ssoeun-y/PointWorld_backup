import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

NPZ = (
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237/"
    "task-0000_episode_00002530-300_311.npz"
)

OUTDIR = (
    "diagnostics/results/"
    "behavior_radio_top1_visual_02530"
)

TARGET_ID = 2

os.makedirs(OUTDIR, exist_ok=True)

d = np.load(NPZ, allow_pickle=True)

gt = d["gt_scene_flows"]
pred = d["pred_scene_flows"]
obj = d["scene_object_id"]
valid = d["official_valid_mask"]
point_index = d["point_index"]

assert d["official_l2_exact"].item()
assert d["temporal_object_id_exact"].item()
assert gt.shape == pred.shape
assert gt.shape[0] == 11

# 동일 radio point set 고정
radio = (obj[0] == TARGET_ID) & valid[-1]

gt_r = gt[:, radio]
pred_r = pred[:, radio]
idx_r = point_index[radio]

assert radio.sum() == 425
assert gt_r.shape == (11, 425, 3)
assert pred_r.shape == (11, 425, 3)

print("=" * 80)
print("SAMPLE =", d["sample_key"].item())
print("RADIO_POINTS =", radio.sum())
print("GT_SHAPE =", gt_r.shape)
print("PRED_SHAPE =", pred_r.shape)
print("OFFICIAL_L2_EXACT =", d["official_l2_exact"].item())
print("TEMPORAL_OBJECT_ID_EXACT =", d["temporal_object_id_exact"].item())

# GT + Pred 전체 trajectory 공통 axis
all_xyz = np.concatenate(
    [gt_r.reshape(-1, 3), pred_r.reshape(-1, 3)],
    axis=0
)

mins = all_xyz.min(axis=0)
maxs = all_xyz.max(axis=0)

center = (mins + maxs) / 2
span = (maxs - mins).max()
half = span * 0.58

xlim = (center[0] - half, center[0] + half)
ylim = (center[1] - half, center[1] + half)
zlim = (center[2] - half, center[2] + half)

colors = np.arange(len(idx_r))

def setup_axis(ax, title):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=45)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

# ============================================================
# GIF
# ============================================================
fig = plt.figure(figsize=(12, 6))

ax_gt = fig.add_subplot(121, projection="3d")
ax_pr = fig.add_subplot(122, projection="3d")

setup_axis(ax_gt, "GT")
setup_axis(ax_pr, "Prediction")

sc_gt = ax_gt.scatter(
    gt_r[0,:,0],
    gt_r[0,:,1],
    gt_r[0,:,2],
    c=colors,
    cmap="viridis",
    s=14,
)

sc_pr = ax_pr.scatter(
    pred_r[0,:,0],
    pred_r[0,:,1],
    pred_r[0,:,2],
    c=colors,
    cmap="viridis",
    s=14,
)

title = fig.suptitle(
    f"{d['sample_key'].item()} | radio_89 | t=0",
    fontsize=14,
)

def update(t):
    a = gt_r[t]
    b = pred_r[t]

    sc_gt._offsets3d = (
        a[:,0], a[:,1], a[:,2]
    )

    sc_pr._offsets3d = (
        b[:,0], b[:,1], b[:,2]
    )

    title.set_text(
        f"{d['sample_key'].item()} | radio_89 | t={t}"
    )

    return sc_gt, sc_pr, title

ani = FuncAnimation(
    fig,
    update,
    frames=11,
    interval=700,
    blit=False,
)

gif_path = os.path.join(
    OUTDIR,
    "radio_02530_gt_vs_pred_t0_t10.gif",
)

ani.save(
    gif_path,
    writer=PillowWriter(fps=2),
)

plt.close(fig)

# ============================================================
# t0 / t3 / t6 / t10 frame strip
# ============================================================
frames = [0, 3, 6, 10]

fig = plt.figure(figsize=(16, 8))

for col, t in enumerate(frames):

    ax = fig.add_subplot(
        2, 4, col + 1,
        projection="3d"
    )
    setup_axis(ax, f"GT t={t}")
    ax.scatter(
        gt_r[t,:,0],
        gt_r[t,:,1],
        gt_r[t,:,2],
        c=colors,
        cmap="viridis",
        s=10,
    )

    ax = fig.add_subplot(
        2, 4, 5 + col,
        projection="3d"
    )
    setup_axis(ax, f"Prediction t={t}")
    ax.scatter(
        pred_r[t,:,0],
        pred_r[t,:,1],
        pred_r[t,:,2],
        c=colors,
        cmap="viridis",
        s=10,
    )

fig.suptitle(
    f"{d['sample_key'].item()} | radio_89 | GT vs Prediction",
    fontsize=15,
)

plt.tight_layout()

strip_path = os.path.join(
    OUTDIR,
    "radio_02530_gt_vs_pred_frame_strip.png"
)

plt.savefig(
    strip_path,
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)

print("\nGIF =", gif_path)
print("FRAME_STRIP =", strip_path)
print("GIF_EXISTS =", os.path.exists(gif_path))
print("STRIP_EXISTS =", os.path.exists(strip_path))
print("TOP1_RADIO_VIS_SUCCESS = True")
