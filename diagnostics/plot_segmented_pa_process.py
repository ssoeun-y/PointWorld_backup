#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def find_first(npz, candidates):
    for k in candidates:
        if k in npz:
            return npz[k]
    raise KeyError(f"Cannot find keys among: {candidates}")

def to_frame_xyz(arr, frame_idx):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 3:      # [T,N,3]
        if frame_idx < 0:
            frame_idx = arr.shape[0] + frame_idx
        return arr[frame_idx]
    raise ValueError(f"Unexpected xyz shape: {arr.shape}")

def to_frame_ids(arr, frame_idx):
    arr = np.asarray(arr)
    if arr.ndim == 2:                             # [T,N]
        if frame_idx < 0:
            frame_idx = arr.shape[0] + frame_idx
        return arr[frame_idx]
    raise ValueError(f"Unexpected id shape: {arr.shape}")

def center_of(x):
    return np.mean(x, axis=0, keepdims=True)

def centered(x):
    c = center_of(x)
    return x - c, c

def rms_size(x):
    return np.sqrt(np.mean(np.sum(x * x, axis=1)))

def kabsch_rotation(src, tgt):
    # src, tgt: [N,3], centered and scale-normalized
    H = src.T @ tgt
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

def pa_steps(pred, ref):
    # pred/ref: [N,3]
    out = {}

    # step 1 raw
    out["raw_pred"] = pred.copy()
    out["raw_ref"] = ref.copy()

    # step 2 translation align
    pred_c, pred_ctr = centered(pred)
    ref_c, ref_ctr = centered(ref)
    out["trans_pred"] = pred_c
    out["trans_ref"] = ref_c

    # step 3 scale align
    sp = rms_size(pred_c)
    sr = rms_size(ref_c)
    if sp < 1e-12 or sr < 1e-12:
        raise RuntimeError("Degenerate object size for PA alignment.")
    pred_s = pred_c / sp
    ref_s = ref_c / sr
    out["scale_pred"] = pred_s
    out["scale_ref"] = ref_s

    # step 4 rotation align
    R = kabsch_rotation(pred_s, ref_s)
    pred_r = pred_s @ R.T
    out["rot_pred"] = pred_r
    out["rot_ref"] = ref_s

    # residual
    residual = np.mean(np.linalg.norm(pred_r - ref_s, axis=1)) * 1000.0
    out["residual_mm"] = residual
    return out

def set_equal_axes(ax, xyz_list):
    pts = np.concatenate(xyz_list, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ctr = (mins + maxs) / 2.0
    rad = np.max(maxs - mins) / 2.0
    if rad < 1e-6:
        rad = 1e-3
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)

def draw_overlay(ax, ref, orig, corr, title):
    ax.scatter(ref[:,0],  ref[:,1],  ref[:,2],  s=8, alpha=0.25, label="Initial ref", color="gray")
    ax.scatter(orig[:,0], orig[:,1], orig[:,2], s=12, alpha=0.85, label="Original", color="blue")
    ax.scatter(corr[:,0], corr[:,1], corr[:,2], s=12, alpha=0.65, label="Corrected", color="red")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_equal_axes(ax, [ref, orig, corr])

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig", type=str, required=True)
    p.add_argument("--corr", type=str, required=True)
    p.add_argument("--frame", type=int, default=-1)
    p.add_argument("--object_id", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    orig_npz = np.load(args.orig, allow_pickle=True)
    corr_npz = np.load(args.corr, allow_pickle=True)

    xyz_keys = ["scene_xyz", "scene_flows", "pred_points", "xyz"]
    id_keys  = ["scene_object_id", "object_id"]

    orig_xyz_all = to_frame_xyz(find_first(orig_npz, xyz_keys), args.frame)
    corr_xyz_all = to_frame_xyz(find_first(corr_npz, xyz_keys), args.frame)

    orig_ids = to_frame_ids(find_first(orig_npz, id_keys), args.frame)
    corr_ids = to_frame_ids(find_first(corr_npz, id_keys), args.frame)

    # initial reference always from t=0 of original npz
    init_xyz_all = to_frame_xyz(find_first(orig_npz, xyz_keys), 0)
    init_ids     = to_frame_ids(find_first(orig_npz, id_keys), 0)

    oid = args.object_id

    init_obj = init_xyz_all[init_ids == oid]
    orig_obj = orig_xyz_all[orig_ids == oid]
    corr_obj = corr_xyz_all[corr_ids == oid]

    if len(init_obj) == 0 or len(orig_obj) == 0 or len(corr_obj) == 0:
        raise RuntimeError(f"Object {oid} missing in one of the required frames.")

    # correspondence-safe min length
    n = min(len(init_obj), len(orig_obj), len(corr_obj))
    init_obj = init_obj[:n]
    orig_obj = orig_obj[:n]
    corr_obj = corr_obj[:n]

    orig_pa = pa_steps(orig_obj, init_obj)
    corr_pa = pa_steps(corr_obj, init_obj)

    # figure
    fig = plt.figure(figsize=(22, 11))
    fig.suptitle(
        f"Segmented rigid-object PA visualization\n"
        f"object_id={oid}, frame={args.frame if args.frame >= 0 else 'last'}",
        fontsize=18, y=0.98
    )

    # 1) scene highlight
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.scatter(orig_xyz_all[:,0], orig_xyz_all[:,1], orig_xyz_all[:,2], s=3, alpha=0.05, color="gray")
    ax1.scatter(orig_obj[:,0], orig_obj[:,1], orig_obj[:,2], s=10, alpha=0.95, color="blue", label="Target object")
    ax1.set_title("1. Full scene with target object highlighted", fontsize=11)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    set_equal_axes(ax1, [orig_xyz_all])

    # 2) segmented object raw overlay
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    draw_overlay(ax2, init_obj, orig_obj, corr_obj, "2. Raw segmented object")

    # 3) translation aligned
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    draw_overlay(ax3,
                 orig_pa["trans_ref"],
                 orig_pa["trans_pred"],
                 corr_pa["trans_pred"],
                 "3. Translation aligned")

    # 4) scale aligned
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    draw_overlay(ax4,
                 orig_pa["scale_ref"],
                 orig_pa["scale_pred"],
                 corr_pa["scale_pred"],
                 "4. Scale aligned")

    # 5) final PA aligned
    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    draw_overlay(ax5,
                 orig_pa["rot_ref"],
                 orig_pa["rot_pred"],
                 corr_pa["rot_pred"],
                 "5. Rotation aligned (final PA comparison)")

    # 6) final clean blue/red only
    ax6 = fig.add_subplot(2, 3, 6, projection="3d")
    ax6.scatter(orig_pa["rot_pred"][:,0], orig_pa["rot_pred"][:,1], orig_pa["rot_pred"][:,2],
                s=14, alpha=0.85, color="blue", label="Original")
    ax6.scatter(corr_pa["rot_pred"][:,0], corr_pa["rot_pred"][:,1], corr_pa["rot_pred"][:,2],
                s=14, alpha=0.65, color="red", label="Corrected")
    ax6.set_title("6. Final blue/red overlay only", fontsize=11)
    ax6.set_xlabel("X"); ax6.set_ylabel("Y"); ax6.set_zlabel("Z")
    set_equal_axes(ax6, [orig_pa["rot_pred"], corr_pa["rot_pred"]])

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.view_init(elev=18, azim=-60)

    handles, labels = ax6.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.93), fontsize=11)

    fig.text(
        0.66, 0.06,
        f"Original PA residual:  {orig_pa['residual_mm']:.3f} mm\n"
        f"Corrected PA residual: {corr_pa['residual_mm']:.3f} mm",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    plt.savefig(args.out, dpi=220)
    print("SAVED =", args.out)
    print("Original residual =", orig_pa["residual_mm"], "mm")
    print("Corrected residual =", corr_pa["residual_mm"], "mm")

if __name__ == "__main__":
    main()
