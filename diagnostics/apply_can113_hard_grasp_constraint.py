from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(
    "diagnostics/results/"
    "acc_only_boundary_gripper_retention_current10457/"
    "candidate_vis"
)

z = np.load(
    ROOT / "task0001_can113_candidate.npz",
    allow_pickle=True,
)

# ---------------------------------------------------------
# Existing predictions
# ---------------------------------------------------------
base_obj = [
    np.asarray(x, np.float32)
    for x in z["base_obj_frames"]
]

corr_obj = [
    np.asarray(x, np.float32)
    for x in z["corr_obj_frames"]
]

base_grip = [
    np.asarray(x, np.float32)
    for x in z["base_grip_frames"]
]

corr_grip = [
    np.asarray(x, np.float32)
    for x in z["corr_grip_frames"]
]

corr_scene = [
    np.asarray(x, np.float32)
    for x in z["corr_scene_frames"]
]

used = np.asarray(
    z["used_frames"],
    dtype=np.int32,
)

R_pose = np.asarray(
    z["right_gripper_pose_orig"],
    dtype=np.float64,
)

L_pose = np.asarray(
    z["left_gripper_pose_orig"],
    dtype=np.float64,
)

phase = np.asarray(
    z["phase_frames"],
    dtype=np.float64,
).reshape(-1)

T = len(L_pose)

# ---------------------------------------------------------
# Corrected left/right gripper pose:
# use exactly the same temporal phase as our retiming.
# ---------------------------------------------------------
if np.nanmax(phase) <= 1.000001:
    xnew = phase * float(T - 1)
else:
    xnew = phase.copy()

xnew = np.clip(xnew, 0.0, T - 1.0)
xold = np.arange(T, dtype=np.float64)


def retime_pose7(pose):
    out = np.zeros_like(pose)

    for d in range(3):
        out[:, d] = np.interp(
            xnew,
            xold,
            pose[:, d],
        )

    rot = Rotation.from_quat(
        pose[:, 3:7]
    )

    out[:, 3:7] = Slerp(
        xold,
        rot,
    )(xnew).as_quat()

    return out


R_corr_pose = retime_pose7(R_pose)
L_corr_pose = retime_pose7(L_pose)


# ---------------------------------------------------------
# Exact right / left gripper surface split
# ---------------------------------------------------------
def split_hands(G, right_center, left_center):
    dr = np.linalg.norm(
        G - right_center[None, :],
        axis=1,
    )
    dl = np.linalg.norm(
        G - left_center[None, :],
        axis=1,
    )

    RH = G[dr <= dl]
    LH = G[dl < dr]

    if len(RH) == 0 or len(LH) == 0:
        raise RuntimeError(
            f"bad hand split R={len(RH)} L={len(LH)}"
        )

    return RH, LH


def min_distance_mm(A, B):
    return float(
        np.min(
            cKDTree(B).query(A, k=1)[0]
        ) * 1000.0
    )


# ---------------------------------------------------------
# Determine the grasp anchor from ORIGINAL rollout.
#
# We use the frame where the can is closest to the LEFT
# gripper. This is only used to establish the rigid
# hand-object transform; after this point there is NO
# free object drift.
# ---------------------------------------------------------
left_dists = []

for j, tt in enumerate(used):

    _, LH = split_hands(
        base_grip[j],
        R_pose[tt, :3],
        L_pose[tt, :3],
    )

    d = min_distance_mm(
        base_obj[j],
        LH,
    )

    left_dists.append(d)


anchor_j = int(np.argmin(left_dists))
anchor_t = int(used[anchor_j])

print("=" * 80)
print("HARD GRASP CONSTRAINT")
print("=" * 80)
print(
    "grasp anchor:",
    f"frame={anchor_t + 1}",
    f"original left distance={left_dists[anchor_j]:.3f} mm",
)


# ---------------------------------------------------------
# Preserve object pose in LEFT-GRIPPER coordinates.
#
# Relative transform is taken from ORIGINAL grasp state.
#
# p_rel = R_hand^-1 (p_world - t_hand)
#
# Corrected:
# p_world = R_hand_corr p_rel + t_hand_corr
# ---------------------------------------------------------
P_anchor = base_obj[anchor_j]

G0 = L_pose[anchor_t]

R0 = Rotation.from_quat(
    G0[3:7]
).as_matrix()

t0 = G0[:3]

# Object points represented in the left-hand frame.
P_rel = (
    R0.T
    @ (P_anchor - t0[None, :]).T
).T


hard_obj = []

for j, tt in enumerate(used):

    # Before grasp: keep PointWorld corrected prediction.
    if j < anchor_j:
        hard_obj.append(
            corr_obj[j].copy()
        )
        continue

    Gt = L_corr_pose[tt]

    Rt = Rotation.from_quat(
        Gt[3:7]
    ).as_matrix()

    tt_xyz = Gt[:3]

    P_fixed = (
        Rt @ P_rel.T
    ).T + tt_xyz[None, :]

    hard_obj.append(
        P_fixed.astype(np.float32)
    )


# ---------------------------------------------------------
# Quantitative comparison
# ---------------------------------------------------------
print("\nFRAMEWISE LEFT-HAND DISTANCE")
print("-" * 80)

rows = []

for j, tt in enumerate(used):

    _, L_orig = split_hands(
        base_grip[j],
        R_pose[tt, :3],
        L_pose[tt, :3],
    )

    _, L_corr = split_hands(
        corr_grip[j],
        R_corr_pose[tt, :3],
        L_corr_pose[tt, :3],
    )

    d_orig = min_distance_mm(
        base_obj[j],
        L_orig,
    )

    d_corr = min_distance_mm(
        corr_obj[j],
        L_corr,
    )

    d_hard = min_distance_mm(
        hard_obj[j],
        L_corr,
    )

    rows.append(
        (
            int(tt + 1),
            d_orig,
            d_corr,
            d_hard,
        )
    )

    print(
        f"frame={tt+1:2d} | "
        f"Original={d_orig:7.3f} mm | "
        f"Corrected={d_corr:7.3f} mm | "
        f"Hard={d_hard:7.3f} mm"
    )


f, do, dc, dh = rows[-1]

print("\n" + "=" * 80)
print("FINAL FRAME")
print("=" * 80)

print(
    f"Original left-can  = {do:.3f} mm"
)
print(
    f"Corrected left-can = {dc:.3f} mm"
)
print(
    f"Hard left-can      = {dh:.3f} mm"
)

print(
    f"Corrected error vs Original = "
    f"{dc-do:+.3f} mm"
)

print(
    f"Hard error vs Original      = "
    f"{dh-do:+.3f} mm"
)


# ---------------------------------------------------------
# Support / floating check
# ---------------------------------------------------------
def support_gap_mm(O, S):

    xy_min = O[:, :2].min(axis=0) - 0.025
    xy_max = O[:, :2].max(axis=0) + 0.025

    bottom = float(
        O[:, 2].min()
    )

    m = (
        (S[:, 0] >= xy_min[0])
        & (S[:, 0] <= xy_max[0])
        & (S[:, 1] >= xy_min[1])
        & (S[:, 1] <= xy_max[1])
        & (S[:, 2] <= bottom + 0.01)
    )

    if not np.any(m):
        return np.nan

    top = float(
        S[m, 2].max()
    )

    return (
        bottom - top
    ) * 1000.0


corr_gap = support_gap_mm(
    corr_obj[-1],
    corr_scene[-1],
)

hard_gap = support_gap_mm(
    hard_obj[-1],
    corr_scene[-1],
)

print(
    f"\nCorrected support gap = "
    f"{corr_gap:.3f} mm"
)

print(
    f"Hard support gap      = "
    f"{hard_gap:.3f} mm"
)


# ---------------------------------------------------------
# Save hard-constrained result
# ---------------------------------------------------------
np.savez(
    ROOT / "task0001_can113_hard_grasp.npz",
    used_frames=used,
    hard_obj_frames=np.asarray(
        hard_obj,
        dtype=object,
    ),
    grasp_anchor_frame=np.asarray(
        anchor_t,
    ),
)

# ---------------------------------------------------------
# Distance curve
# ---------------------------------------------------------
fr = [
    x[0] for x in rows
]

orig_d = [
    x[1] for x in rows
]

corr_d = [
    x[2] for x in rows
]

hard_d = [
    x[3] for x in rows
]

plt.figure(figsize=(9, 5))

plt.plot(
    fr,
    orig_d,
    marker="o",
    label="Original",
)

plt.plot(
    fr,
    corr_d,
    marker="o",
    label="Corrected",
)

plt.plot(
    fr,
    hard_d,
    marker="o",
    label="Hard grasp constraint",
)

plt.axvline(
    anchor_t + 1,
    linestyle="--",
    alpha=0.5,
    label="grasp anchor",
)

plt.xlabel("frame")
plt.ylabel("left gripper - can min distance [mm]")
plt.title("Hand-object coupling")

plt.legend()
plt.tight_layout()

curve_out = (
    ROOT
    / "task0001_can113_hard_grasp_distance.png"
)

plt.savefig(
    curve_out,
    dpi=180,
)
plt.close()


# ---------------------------------------------------------
# GIF: Original / Corrected / Hard
# ---------------------------------------------------------
all_pts = []

for arrs in [
    base_obj,
    corr_obj,
    hard_obj,
    base_grip,
    corr_grip,
]:

    for a in arrs:
        all_pts.append(a)

P = np.concatenate(
    all_pts,
    axis=0,
)

mn = P.min(axis=0)
mx = P.max(axis=0)

ctr = (
    mn + mx
) / 2.0

rad = max(
    float(
        np.max(mx - mn)
        * 0.58
    ),
    0.09,
)

fig = plt.figure(
    figsize=(15, 5)
)

ax1 = fig.add_subplot(
    131,
    projection="3d",
)

ax2 = fig.add_subplot(
    132,
    projection="3d",
)

ax3 = fig.add_subplot(
    133,
    projection="3d",
)


def setup(ax, title):

    ax.set_title(title)

    ax.set_xlim(
        ctr[0] - rad,
        ctr[0] + rad,
    )

    ax.set_ylim(
        ctr[1] - rad,
        ctr[1] + rad,
    )

    ax.set_zlim(
        ctr[2] - rad,
        ctr[2] + rad,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(
        elev=20,
        azim=-60,
    )


def draw(ax, G, O, Rc, Lc, title):

    _, LH = split_hands(
        G,
        Rc,
        Lc,
    )

    ax.scatter(
        G[:, 0],
        G[:, 1],
        G[:, 2],
        s=5,
        alpha=0.3,
        label="grippers",
    )

    ax.scatter(
        LH[:, 0],
        LH[:, 1],
        LH[:, 2],
        s=8,
        alpha=0.8,
        label="LEFT",
    )

    ax.scatter(
        O[:, 0],
        O[:, 1],
        O[:, 2],
        s=14,
        alpha=0.95,
        label="can",
    )

    ax.set_title(title)
    ax.legend(fontsize=7)


def update(j):

    ax1.cla()
    ax2.cla()
    ax3.cla()

    tt = int(
        used[j]
    )

    setup(
        ax1,
        f"Original | frame {tt+1}",
    )

    setup(
        ax2,
        f"Corrected | frame {tt+1}",
    )

    setup(
        ax3,
        f"Hard constraint | frame {tt+1}",
    )

    draw(
        ax1,
        base_grip[j],
        base_obj[j],
        R_pose[tt, :3],
        L_pose[tt, :3],
        "Original",
    )

    draw(
        ax2,
        corr_grip[j],
        corr_obj[j],
        R_corr_pose[tt, :3],
        L_corr_pose[tt, :3],
        "Corrected",
    )

    draw(
        ax3,
        corr_grip[j],
        hard_obj[j],
        R_corr_pose[tt, :3],
        L_corr_pose[tt, :3],
        "Hard grasp",
    )


ani = FuncAnimation(
    fig,
    update,
    frames=len(used),
    interval=850,
    repeat=True,
)

gif_out = (
    ROOT
    / "task0001_can113_hard_grasp.gif"
)

ani.save(
    gif_out,
    writer=PillowWriter(fps=1),
)

plt.close()

print("\nSAVED:")
print(curve_out)
print(gif_out)
