import os
import numpy as np
import matplotlib.pyplot as plt

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
T = 10

os.makedirs(OUTDIR, exist_ok=True)

d = np.load(NPZ, allow_pickle=True)

gt = d["gt_scene_flows"]
pred = d["pred_scene_flows"]
obj = d["scene_object_id"]
valid = d["official_valid_mask"]

assert d["official_l2_exact"].item()
assert d["temporal_object_id_exact"].item()

# 기존과 동일한 radio point set
radio = (obj[0] == TARGET_ID) & valid[-1]

gt_t = gt[T, radio]
pred_t = pred[T, radio]

assert gt_t.shape == pred_t.shape
assert gt_t.shape[0] == 393

# point-wise GT ↔ Pred 거리
err = np.linalg.norm(pred_t - gt_t, axis=1)

print("=" * 80)
print("SAMPLE =", d["sample_key"].item())
print("TIMESTEP =", T)
print("RADIO_POINTS =", len(gt_t))
print("POINT_ERROR_MEAN_MM =", err.mean() * 1000)
print("POINT_ERROR_MEDIAN_MM =", np.median(err) * 1000)
print("POINT_ERROR_P95_MM =", np.quantile(err, 0.95) * 1000)
print("POINT_ERROR_MAX_MM =", err.max() * 1000)

# 공통 axis
xyz = np.concatenate([gt_t, pred_t], axis=0)
mins = xyz.min(axis=0)
maxs = xyz.max(axis=0)

center = (mins + maxs) / 2
span = (maxs - mins).max()
half = span * 0.58

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection="3d")

# 동일 point GT ↔ Pred 연결
for i in range(len(gt_t)):
    ax.plot(
        [gt_t[i, 0], pred_t[i, 0]],
        [gt_t[i, 1], pred_t[i, 1]],
        [gt_t[i, 2], pred_t[i, 2]],
        color="gray",
        alpha=0.28,
        linewidth=0.55,
    )

ax.scatter(
    gt_t[:, 0],
    gt_t[:, 1],
    gt_t[:, 2],
    c="blue",
    s=18,
    alpha=0.75,
    label="GT",
)

ax.scatter(
    pred_t[:, 0],
    pred_t[:, 1],
    pred_t[:, 2],
    c="red",
    s=18,
    alpha=0.75,
    label="Prediction",
)

ax.set_xlim(center[0] - half, center[0] + half)
ax.set_ylim(center[1] - half, center[1] + half)
ax.set_zlim(center[2] - half, center[2] + half)

ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=20, azim=45)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title(
    "radio_89 | t=10 | GT vs Prediction\n"
    "Blue=GT, Red=Prediction, Gray=same-point displacement"
)

ax.legend()

out = os.path.join(
    OUTDIR,
    "radio_00470_t10_gt_pred_overlay.png"
)

plt.tight_layout()
plt.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)

print("\nOUTPUT =", out)
print("OUTPUT_EXISTS =", os.path.exists(out))
print("RADIO_OVERLAY_SUCCESS = True")
