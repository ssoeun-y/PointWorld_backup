import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

NPZ = (
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237/"
    "task-0000_episode_00000470-375_386.npz"
)

OUTDIR = (
    "diagnostics/results/"
    "behavior_radio_gt_pred_visual_00470"
)

TARGET_ID = 2

os.makedirs(OUTDIR, exist_ok=True)

d = np.load(NPZ, allow_pickle=True)

sample_key = d["sample_key"].item()
gt = d["gt_scene_flows"]
pred = d["pred_scene_flows"]
obj = d["scene_object_id"]
valid = d["official_valid_mask"]
point_index = d["point_index"]

assert gt.shape == pred.shape
assert gt.shape[0] == 11
assert d["official_l2_exact"].item()
assert d["temporal_object_id_exact"].item()

# ------------------------------------------------------------
# 동일 radio point set을 전체 timestep에서 유지
# 기존 geometry 분석과 동일하게 final-official-valid point 사용
# ------------------------------------------------------------
radio = (obj[0] == TARGET_ID) & valid[-1]

gt_r = gt[:, radio]
pred_r = pred[:, radio]
idx_r = point_index[radio]

print("=" * 80)
print("SAMPLE =", sample_key)
print("TARGET_ID =", TARGET_ID)
print("RADIO_POINTS =", radio.sum())
print("GT_SHAPE =", gt_r.shape)
print("PRED_SHAPE =", pred_r.shape)
print("OFFICIAL_L2_EXACT =", d["official_l2_exact"].item())
print("TEMPORAL_OBJECT_ID_EXACT =", d["temporal_object_id_exact"].item())

assert radio.sum() == 393
assert gt_r.shape == (11, 393, 3)
assert pred_r.shape == (11, 393, 3)

# ------------------------------------------------------------
# GT + Prediction 전체 trajectory를 함께 포함하는 공통 axis
# 두 패널에 완전히 동일한 scale 적용
# ------------------------------------------------------------
all_xyz = np.concatenate(
    [gt_r.reshape(-1, 3), pred_r.reshape(-1, 3)],
    axis=0,
)

mins = all_xyz.min(axis=0)
maxs = all_xyz.max(axis=0)

center = (mins + maxs) / 2.0
span = (maxs - mins).max()

# 약간의 여백
half = span * 0.58

xlim = (center[0] - half, center[0] + half)
ylim = (center[1] - half, center[1] + half)
zlim = (center[2] - half, center[2] + half)

print("\nCOMMON AXIS")
print("xlim =", xlim)
print("ylim =", ylim)
print("zlim =", zlim)

# 같은 point identity는 GT/Pred 모두 같은 색
colors = np.arange(len(idx_r))


def setup_axis(ax, title):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    ax.set_box_aspect((1, 1, 1))

    # 두 패널 같은 카메라
    ax.view_init(elev=20, azim=45)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


# ============================================================
# 1. Animation
# ============================================================

fig = plt.figure(figsize=(12, 6))

ax_gt = fig.add_subplot(121, projection="3d")
ax_pr = fig.add_subplot(122, projection="3d")

setup_axis(ax_gt, "GT")
setup_axis(ax_pr, "Prediction")

sc_gt = ax_gt.scatter(
    gt_r[0, :, 0],
    gt_r[0, :, 1],
    gt_r[0, :, 2],
    c=colors,
    cmap="viridis",
    s=14,
)

sc_pr = ax_pr.scatter(
    pred_r[0, :, 0],
    pred_r[0, :, 1],
    pred_r[0, :, 2],
    c=colors,
    cmap="viridis",
    s=14,
)

title = fig.suptitle(
    f"{sample_key} | radio_89 | t=0",
    fontsize=14,
)


def update(t):
    xyz_gt = gt_r[t]
    xyz_pr = pred_r[t]

    sc_gt._offsets3d = (
        xyz_gt[:, 0],
        xyz_gt[:, 1],
        xyz_gt[:, 2],
    )

    sc_pr._offsets3d = (
        xyz_pr[:, 0],
        xyz_pr[:, 1],
        xyz_pr[:, 2],
    )

    title.set_text(
        f"{sample_key} | radio_89 | t={t}"
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
    "radio_00470_gt_vs_pred_t0_t10.gif",
)

ani.save(
    gif_path,
    writer=PillowWriter(fps=2),
)

plt.close(fig)


# ============================================================
# 2. 교수님/보고용 representative frame strip
#    t0 / t3 / t6 / t10
# ============================================================

frames = [0, 3, 6, 10]

fig = plt.figure(figsize=(16, 8))

for col, t in enumerate(frames):

    ax = fig.add_subplot(
        2,
        4,
        col + 1,
        projection="3d",
    )

    setup_axis(ax, f"GT t={t}")

    ax.scatter(
        gt_r[t, :, 0],
        gt_r[t, :, 1],
        gt_r[t, :, 2],
        c=colors,
        cmap="viridis",
        s=10,
    )

    ax = fig.add_subplot(
        2,
        4,
        4 + col + 1,
        projection="3d",
    )

    setup_axis(ax, f"Prediction t={t}")

    ax.scatter(
        pred_r[t, :, 0],
        pred_r[t, :, 1],
        pred_r[t, :, 2],
        c=colors,
        cmap="viridis",
        s=10,
    )


fig.suptitle(
    f"{sample_key} | radio_89 | GT vs Prediction",
    fontsize=15,
)

plt.tight_layout()

strip_path = os.path.join(
    OUTDIR,
    "radio_00470_gt_vs_pred_frame_strip.png",
)

plt.savefig(
    strip_path,
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)

print("\n" + "=" * 80)
print("GIF =", gif_path)
print("FRAME_STRIP =", strip_path)
print("GIF_EXISTS =", os.path.exists(gif_path))
print("STRIP_EXISTS =", os.path.exists(strip_path))
print("RADIO_GT_PRED_VISUAL_SUCCESS = True")
