from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import cKDTree

ROOT = Path(
    "diagnostics/results/"
    "acc_only_boundary_gripper_retention_current10457/"
    "candidate_vis"
)

dump = ROOT / "task0001_can113_candidate.npz"
z = np.load(dump, allow_pickle=True)

bo = [np.asarray(x, np.float32) for x in z["base_obj_frames"]]
co = [np.asarray(x, np.float32) for x in z["corr_obj_frames"]]

bg = [np.asarray(x, np.float32) for x in z["base_grip_frames"]]
cg = [np.asarray(x, np.float32) for x in z["corr_grip_frames"]]

bs = [np.asarray(x, np.float32) for x in z["base_scene_frames"]]
cs = [np.asarray(x, np.float32) for x in z["corr_scene_frames"]]

used = np.asarray(z["used_frames"]).astype(int)

R0 = np.asarray(z["right_gripper_pose_orig"], np.float32)[:, :3]
L0 = np.asarray(z["left_gripper_pose_orig"], np.float32)[:, :3]
phase = np.asarray(z["phase_frames"], np.float64).reshape(-1)

T = len(R0)

if len(L0) != T or len(phase) != T:
    raise RuntimeError(
        f"shape mismatch R={R0.shape} L={L0.shape} phase={phase.shape}"
    )

# ---------------------------------------------------------
# Apply exactly the same temporal phase to gripper XYZ.
# phase may be saved either as [0..T-1] or normalized [0..1].
# ---------------------------------------------------------
idx = phase.copy()

if np.nanmax(idx) <= 1.000001:
    idx = idx * (T - 1)

idx = np.clip(idx, 0.0, T - 1.0)

def interp_xyz(xyz):
    grid = np.arange(T, dtype=np.float64)
    out = np.empty((T, 3), dtype=np.float32)

    for d in range(3):
        out[:, d] = np.interp(
            idx,
            grid,
            xyz[:, d].astype(np.float64),
        )

    return out

Rcorr = interp_xyz(R0)
Lcorr = interp_xyz(L0)

# endpoint must remain identical
print(
    "[ENDPOINT CHECK] right =",
    np.linalg.norm(Rcorr[-1] - R0[-1]) * 1000.0,
    "mm"
)
print(
    "[ENDPOINT CHECK] left  =",
    np.linalg.norm(Lcorr[-1] - L0[-1]) * 1000.0,
    "mm"
)

# ---------------------------------------------------------
# Split merged gripper-only robot surface using the ACTUAL
# right/left gripper centers, not x-coordinate heuristics.
# ---------------------------------------------------------
def split_exact(G, right_center, left_center):
    dr = np.linalg.norm(
        G - right_center[None, :],
        axis=1,
    )
    dl = np.linalg.norm(
        G - left_center[None, :],
        axis=1,
    )

    right = G[dr <= dl]
    left = G[dl < dr]

    if len(right) == 0 or len(left) == 0:
        raise RuntimeError(
            f"bad hand split right={len(right)} left={len(left)}"
        )

    return right, left


def object_surface_distance(O, H):
    d = cKDTree(H).query(O, k=1)[0] * 1000.0

    return {
        "min": float(np.min(d)),
        "q10": float(np.percentile(d, 10)),
        "median": float(np.median(d)),
    }


def environment_metrics(O, S):
    # 3D nearest surrounding-scene distance.
    env_min = float(
        np.min(cKDTree(S).query(O, k=1)[0]) * 1000.0
    )

    # Vertical support gap directly underneath object's XY footprint.
    xy_min = O[:, :2].min(axis=0) - 0.025
    xy_max = O[:, :2].max(axis=0) + 0.025

    bottom = float(O[:, 2].min())

    under = (
        (S[:, 0] >= xy_min[0])
        & (S[:, 0] <= xy_max[0])
        & (S[:, 1] >= xy_min[1])
        & (S[:, 1] <= xy_max[1])
        & (S[:, 2] <= bottom + 0.01)
    )

    if np.any(under):
        support_top = float(np.max(S[under, 2]))
        support_gap = (bottom - support_top) * 1000.0
    else:
        support_gap = np.nan

    return env_min, support_gap


rows = []
vis_data = []

for j, tt in enumerate(used):

    for tag, O, G, S, Rc, Lc in [
        (
            "Original",
            bo[j], bg[j], bs[j],
            R0[tt], L0[tt],
        ),
        (
            "Corrected",
            co[j], cg[j], cs[j],
            Rcorr[tt], Lcorr[tt],
        ),
    ]:

        RH, LH = split_exact(G, Rc, Lc)

        rd = object_surface_distance(O, RH)
        ld = object_surface_distance(O, LH)

        env_min, support_gap = environment_metrics(O, S)

        object_centroid = O.mean(axis=0)

        rcent = float(
            np.linalg.norm(object_centroid - Rc) * 1000.0
        )
        lcent = float(
            np.linalg.norm(object_centroid - Lc) * 1000.0
        )

        rows.append({
            "frame": int(tt + 1),
            "tag": tag,

            "right_min_mm": rd["min"],
            "right_q10_mm": rd["q10"],
            "right_median_mm": rd["median"],

            "left_min_mm": ld["min"],
            "left_q10_mm": ld["q10"],
            "left_median_mm": ld["median"],

            "right_centroid_mm": rcent,
            "left_centroid_mm": lcent,

            "scene_min_mm": env_min,
            "support_gap_mm": support_gap,
        })

        vis_data.append(
            (int(tt + 1), tag, O, RH, LH, S)
        )


# ---------------------------------------------------------
# CSV
# ---------------------------------------------------------
csv_out = ROOT / "task0001_can113_exact_hand_metrics.csv"

with open(csv_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nEXACT LEFT / RIGHT HAND METRICS")
print("=" * 115)

for r in rows:
    print(
        f"frame={r['frame']:2d} "
        f"{r['tag']:9s} | "
        f"Rmin={r['right_min_mm']:7.2f} "
        f"Lmin={r['left_min_mm']:7.2f} | "
        f"Rq10={r['right_q10_mm']:7.2f} "
        f"Lq10={r['left_q10_mm']:7.2f} | "
        f"scene={r['scene_min_mm']:7.2f} "
        f"support_gap={r['support_gap_mm']:7.2f}"
    )


# ---------------------------------------------------------
# FINAL FRAME result
# ---------------------------------------------------------
final_frame = max(r["frame"] for r in rows)

orig = [
    r for r in rows
    if r["frame"] == final_frame and r["tag"] == "Original"
][0]

corr = [
    r for r in rows
    if r["frame"] == final_frame and r["tag"] == "Corrected"
][0]

print("\n" + "=" * 80)
print("FINAL FRAME =", final_frame)
print("=" * 80)

print(
    "ORIGINAL  "
    f"Rmin={orig['right_min_mm']:.3f} mm | "
    f"Lmin={orig['left_min_mm']:.3f} mm | "
    f"scene_min={orig['scene_min_mm']:.3f} mm | "
    f"support_gap={orig['support_gap_mm']:.3f} mm"
)

print(
    "CORRECTED "
    f"Rmin={corr['right_min_mm']:.3f} mm | "
    f"Lmin={corr['left_min_mm']:.3f} mm | "
    f"scene_min={corr['scene_min_mm']:.3f} mm | "
    f"support_gap={corr['support_gap_mm']:.3f} mm"
)

print(
    "DELTA     "
    f"Rmin={corr['right_min_mm']-orig['right_min_mm']:+.3f} mm | "
    f"Lmin={corr['left_min_mm']-orig['left_min_mm']:+.3f} mm"
)


# ---------------------------------------------------------
# Curves
# ---------------------------------------------------------
frames = sorted(set(r["frame"] for r in rows))

def vals(tag, key):
    m = {
        r["frame"]: r[key]
        for r in rows
        if r["tag"] == tag
    }
    return [m[f] for f in frames]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    frames, vals("Original", "right_min_mm"),
    label="Original right"
)
ax.plot(
    frames, vals("Original", "left_min_mm"),
    label="Original left"
)
ax.plot(
    frames, vals("Corrected", "right_min_mm"),
    label="Corrected right"
)
ax.plot(
    frames, vals("Corrected", "left_min_mm"),
    label="Corrected left"
)

ax.set_xlabel("frame")
ax.set_ylabel("object-to-gripper surface min distance [mm]")
ax.set_title("Exact left/right gripper distance")
ax.legend()
ax.grid(alpha=0.25)

curve_out = ROOT / "task0001_can113_exact_hand_distance.png"
plt.tight_layout()
plt.savefig(curve_out, dpi=180)
plt.close()


# ---------------------------------------------------------
# Shared zoom bounds for Original / Corrected.
# ---------------------------------------------------------
all_focus = []

for O in bo + co + bg + cg:
    all_focus.append(O)

P = np.concatenate(all_focus, axis=0)
mn = P.min(axis=0)
mx = P.max(axis=0)

ctr = (mn + mx) / 2.0
rad = max(float(np.max(mx - mn) * 0.58), 0.09)


def setup(ax, title):
    ax.set_title(title)

    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(elev=20, azim=-60)


# ---------------------------------------------------------
# Final-frame PNG with exact RIGHT / LEFT hands.
# ---------------------------------------------------------
j = len(used) - 1
tt = used[j]

RO, LO = split_exact(bg[j], R0[tt], L0[tt])
RC, LC = split_exact(cg[j], Rcorr[tt], Lcorr[tt])

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

setup(ax1, f"Original | frame {tt+1}")
setup(ax2, f"Corrected | frame {tt+1}")

# local scene only, so support surface remains visible
def local_scene(S, O):
    c = O.mean(axis=0)
    dxy = np.linalg.norm(
        S[:, :2] - c[None, :2],
        axis=1,
    )
    return S[dxy < 0.30]

S1 = local_scene(bs[j], bo[j])
S2 = local_scene(cs[j], co[j])

ax1.scatter(
    S1[:,0], S1[:,1], S1[:,2],
    s=2, alpha=0.12, label="scene"
)
ax1.scatter(
    RO[:,0], RO[:,1], RO[:,2],
    s=7, alpha=0.65, label="RIGHT gripper"
)
ax1.scatter(
    LO[:,0], LO[:,1], LO[:,2],
    s=7, alpha=0.65, label="LEFT gripper"
)
ax1.scatter(
    bo[j][:,0], bo[j][:,1], bo[j][:,2],
    s=14, alpha=0.95, label="can"
)

ax2.scatter(
    S2[:,0], S2[:,1], S2[:,2],
    s=2, alpha=0.12, label="scene"
)
ax2.scatter(
    RC[:,0], RC[:,1], RC[:,2],
    s=7, alpha=0.65, label="RIGHT gripper"
)
ax2.scatter(
    LC[:,0], LC[:,1], LC[:,2],
    s=7, alpha=0.65, label="LEFT gripper"
)
ax2.scatter(
    co[j][:,0], co[j][:,1], co[j][:,2],
    s=14, alpha=0.95, label="can"
)

ax1.legend(fontsize=8)
ax2.legend(fontsize=8)

png_out = ROOT / "task0001_can113_exact_hands_final.png"
plt.tight_layout()
plt.savefig(png_out, dpi=200, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------
# GIF with exact RIGHT / LEFT hands.
# ---------------------------------------------------------
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

def update(k):
    ax1.cla()
    ax2.cla()

    tt = used[k]

    setup(ax1, f"Original | frame {tt+1}")
    setup(ax2, f"Corrected | frame {tt+1}")

    RO, LO = split_exact(
        bg[k],
        R0[tt],
        L0[tt],
    )

    RC, LC = split_exact(
        cg[k],
        Rcorr[tt],
        Lcorr[tt],
    )

    S1 = local_scene(bs[k], bo[k])
    S2 = local_scene(cs[k], co[k])

    ax1.scatter(
        S1[:,0], S1[:,1], S1[:,2],
        s=2, alpha=0.10, label="scene"
    )
    ax1.scatter(
        RO[:,0], RO[:,1], RO[:,2],
        s=7, alpha=0.65, label="RIGHT"
    )
    ax1.scatter(
        LO[:,0], LO[:,1], LO[:,2],
        s=7, alpha=0.65, label="LEFT"
    )
    ax1.scatter(
        bo[k][:,0], bo[k][:,1], bo[k][:,2],
        s=14, alpha=0.95, label="can"
    )

    ax2.scatter(
        S2[:,0], S2[:,1], S2[:,2],
        s=2, alpha=0.10, label="scene"
    )
    ax2.scatter(
        RC[:,0], RC[:,1], RC[:,2],
        s=7, alpha=0.65, label="RIGHT"
    )
    ax2.scatter(
        LC[:,0], LC[:,1], LC[:,2],
        s=7, alpha=0.65, label="LEFT"
    )
    ax2.scatter(
        co[k][:,0], co[k][:,1], co[k][:,2],
        s=14, alpha=0.95, label="can"
    )

    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)

ani = FuncAnimation(
    fig,
    update,
    frames=len(used),
    interval=850,
    repeat=True,
)

gif_out = ROOT / "task0001_can113_exact_hands.gif"

ani.save(
    gif_out,
    writer=PillowWriter(fps=1),
)

plt.close()

print("\nSAVED:")
print(csv_out)
print(curve_out)
print(png_out)
print(gif_out)
