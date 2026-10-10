from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(
    "diagnostics/results/"
    "acc_only_boundary_gripper_retention_current10457/"
    "candidate_vis"
)

z = np.load(
    ROOT / "task0001_can113_candidate.npz",
    allow_pickle=True,
)

bo = [np.asarray(x, np.float32) for x in z["base_obj_frames"]]
ro = [np.asarray(x, np.float32) for x in z["repeat_obj_frames"]]
co = [np.asarray(x, np.float32) for x in z["corr_obj_frames"]]

bg = [np.asarray(x, np.float32) for x in z["base_grip_frames"]]
rg = [np.asarray(x, np.float32) for x in z["repeat_grip_frames"]]
cg = [np.asarray(x, np.float32) for x in z["corr_grip_frames"]]

bs = [np.asarray(x, np.float32) for x in z["base_scene_frames"]]
rs = [np.asarray(x, np.float32) for x in z["repeat_scene_frames"]]
cs = [np.asarray(x, np.float32) for x in z["corr_scene_frames"]]

used = np.asarray(z["used_frames"]).astype(int)

R0 = np.asarray(z["right_gripper_pose_orig"], np.float32)[:, :3]
L0 = np.asarray(z["left_gripper_pose_orig"], np.float32)[:, :3]
phase = np.asarray(z["phase_frames"], np.float64)

T = len(R0)
grid = np.arange(T, dtype=np.float64)

idx = phase.copy()
if np.max(idx) <= 1.000001:
    idx *= (T - 1)

def interp_xyz(xyz):
    out = np.empty_like(xyz)
    for d in range(3):
        out[:, d] = np.interp(idx, grid, xyz[:, d])
    return out

Rc = interp_xyz(R0)
Lc = interp_xyz(L0)

def split_hands(G, rcenter, lcenter):
    dr = np.linalg.norm(G-rcenter[None,:], axis=1)
    dl = np.linalg.norm(G-lcenter[None,:], axis=1)
    return G[dr <= dl], G[dl < dr]

def mindist(A, B):
    return float(np.min(cKDTree(B).query(A, k=1)[0]) * 1000.)

def support_gap(O, S):
    mn = O[:, :2].min(0) - 0.025
    mx = O[:, :2].max(0) + 0.025
    bottom = float(O[:,2].min())

    m = (
        (S[:,0] >= mn[0]) & (S[:,0] <= mx[0]) &
        (S[:,1] >= mn[1]) & (S[:,1] <= mx[1]) &
        (S[:,2] <= bottom + 0.01)
    )

    if not np.any(m):
        return np.nan

    return float((bottom - S[m,2].max()) * 1000.)

i = len(used) - 1
tt = used[i]

BR, BL = split_hands(bg[i], R0[tt], L0[tt])
RR, RL = split_hands(rg[i], R0[tt], L0[tt])
CR, CL = split_hands(cg[i], Rc[tt], Lc[tt])

vals = {}

for name, O, RH, LH, S in [
    ("ORIGINAL", bo[i], BR, BL, bs[i]),
    ("ORIG_REPEAT", ro[i], RR, RL, rs[i]),
    ("CORRECTED", co[i], CR, CL, cs[i]),
]:
    vals[name] = {
        "R": mindist(O, RH),
        "L": mindist(O, LH),
        "support": support_gap(O, S),
    }

print("\nPOINTWORLD ISOLATION — FINAL FRAME")
print("="*78)

for name, v in vals.items():
    print(
        f"{name:12s} "
        f"Rmin={v['R']:8.3f} mm | "
        f"Lmin={v['L']:8.3f} mm | "
        f"support_gap={v['support']:8.3f} mm"
    )

o = vals["ORIGINAL"]
r = vals["ORIG_REPEAT"]
c = vals["CORRECTED"]

print("\nLEFT-HAND CHANGE")
print(
    f"same-input stochastic repeat: "
    f"{r['L']-o['L']:+.3f} mm"
)
print(
    f"kinematic-corrected action:   "
    f"{c['L']-o['L']:+.3f} mm"
)

print("\nSUPPORT-GAP CHANGE")
print(
    f"same-input stochastic repeat: "
    f"{r['support']-o['support']:+.3f} mm"
)
print(
    f"kinematic-corrected action:   "
    f"{c['support']-o['support']:+.3f} mm"
)

repeat_noise = abs(r["L"] - o["L"])
corr_shift = abs(c["L"] - o["L"])

print("\nRATIO")
if repeat_noise > 1e-9:
    print(
        "corrected hand-object shift / same-input repeat =",
        corr_shift / repeat_noise,
    )
else:
    print("repeat noise ~0; corrected shift =", corr_shift, "mm")
