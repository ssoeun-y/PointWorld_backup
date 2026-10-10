#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def load_npz_dict(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}

def find_xyz_key(d):
    preferred = [
        "scene_xyz",
        "pred_points",
        "scene_points",
        "points",
        "xyz",
    ]
    for k in preferred:
        if k in d and np.asarray(d[k]).ndim == 3 and np.asarray(d[k]).shape[-1] == 3:
            return k
    for k, v in d.items():
        a = np.asarray(v)
        if a.ndim == 3 and a.shape[-1] == 3:
            return k
    raise KeyError(f"XYZ key not found. keys={list(d.keys())}")

def find_id_key(d):
    preferred = [
        "scene_object_id",
        "object_id",
        "scene_ids",
        "ids",
    ]
    for k in preferred:
        if k in d and np.asarray(d[k]).ndim == 2:
            return k
    for k, v in d.items():
        a = np.asarray(v)
        if a.ndim == 2 and np.issubdtype(a.dtype, np.integer):
            return k
    raise KeyError(f"Object-ID key not found. keys={list(d.keys())}")

def frame_index(T, idx):
    return T + idx if idx < 0 else idx

def similarity_align(X, Y):
    """
    Align Y -> X with similarity transform.
    X, Y: [N,3], assumed pointwise correspondence.
    Returns:
      resid_m
      stages dict:
        raw_ref, raw_y
        ctr_ref, ctr_y
        scl_ref, scl_y
        rot_ref, rot_y
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    muX = X.mean(0)
    muY = Y.mean(0)

    Xc = X - muX
    Yc = Y - muY

    nX = np.linalg.norm(Xc)
    nY = np.linalg.norm(Yc)
    if nX < 1e-12 or nY < 1e-12:
        raise ValueError("degenerate point set")

    Xn = Xc / nX
    Yn = Yc / nY

    H = Yn.T @ Xn
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    scale = nX / nY

    Y_aligned = (Yc @ R) * scale + muX
    resid = np.mean(np.linalg.norm(Y_aligned - X, axis=1))

    stages = {
        "raw_ref": X,
        "raw_y": Y,
        "ctr_ref": Xc,
        "ctr_y": Yc,
        "scl_ref": Xn,
        "scl_y": Yn * 1.0,  # before rotation
        "rot_ref": Xn,
        "rot_y": Yn @ R,
    }
    return resid, stages

def common_npz_files(orig_root, corr_root):
    a = {p.name: p for p in Path(orig_root).rglob("*.npz")}
    b = {p.name: p for p in Path(corr_root).rglob("*.npz")}
    common = sorted(set(a.keys()) & set(b.keys()))
    return [(a[n], b[n]) for n in common]

def set_equal_3d_axes(ax, pts_list, pad=0.05):
    allp = np.concatenate(pts_list, axis=0)
    mn = allp.min(0)
    mx = allp.max(0)
    center = (mn + mx) / 2.0
    radius = max((mx - mn).max() / 2.0, 1e-6) * (1.0 + pad)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

def pick_best_case(orig_root, corr_root, frame_idx=-1, min_points=80):
    best = None
    pairs = common_npz_files(orig_root, corr_root)
    if not pairs:
        raise RuntimeError("No common NPZ files found between orig_root and corr_root.")

    for orig_p, corr_p in pairs:
        try:
            do = load_npz_dict(orig_p)
            dc = load_npz_dict(corr_p)

            okx = find_xyz_key(do)
            oki = find_id_key(do)
            ckx = find_xyz_key(dc)
            cki = find_id_key(dc)

            ox = np.asarray(do[okx])
            oi = np.asarray(do[oki])
            cx = np.asarray(dc[ckx])
            ci = np.asarray(dc[cki])

            if ox.ndim != 3 or cx.ndim != 3:
                continue
            if oi.ndim != 2 or ci.ndim != 2:
                continue
            if ox.shape[0] != cx.shape[0]:
                continue

            T = ox.shape[0]
            f = frame_index(T, frame_idx)
            if f < 0 or f >= T:
                continue

            ref_ids = oi[0]
            obj_ids = np.intersect1d(np.unique(ref_ids), np.unique(oi[f]))
            obj_ids = np.intersect1d(obj_ids, np.unique(ci[f]))

            for obj_id in obj_ids:
                # 필요하면 background성 0 제외
                if int(obj_id) == 0:
                    continue

                ref_mask = (oi[0] == obj_id)
                o_mask = (oi[f] == obj_id)
                c_mask = (ci[f] == obj_id)

                n_ref = int(ref_mask.sum())
                n_o = int(o_mask.sum())
                n_c = int(c_mask.sum())

                if min(n_ref, n_o, n_c) < min_points:
                    continue
                if not (n_ref == n_o == n_c):
                    continue

                X = ox[0][ref_mask]
                Yo = ox[f][o_mask]
                Yc = cx[f][c_mask]

                try:
                    ro, so = similarity_align(X, Yo)
                    rc, sc = similarity_align(X, Yc)
                except Exception:
                    continue

                gain = ro - rc  # positive => corrected better
                item = {
                    "orig_path": orig_p,
                    "corr_path": corr_p,
                    "object_id": int(obj_id),
                    "frame": int(f),
                    "n_points": int(len(X)),
                    "orig_resid_mm": float(ro * 1000.0),
                    "corr_resid_mm": float(rc * 1000.0),
                    "gain_mm": float(gain * 1000.0),
                    "stages_orig": so,
                    "stages_corr": sc,
                }

                if best is None or item["gain_mm"] > best["gain_mm"]:
                    best = item

        except Exception:
            continue

    if best is None:
        raise RuntimeError("No valid case found. Try lowering --min_points or checking NPZ structure.")
    return best

def plot_case(best, out_path):
    so = best["stages_orig"]
    sc = best["stages_corr"]

    fig = plt.figure(figsize=(20, 5))
    titles = [
        "1. Raw prediction",
        "2. Translation removed\n(same center)",
        "3. Scale normalized\n(same overall size)",
        "4. Rotation aligned\n(final PA shape comparison)",
    ]

    panels = [
        (so["raw_ref"], so["raw_y"], sc["raw_y"]),
        (so["ctr_ref"], so["ctr_y"], sc["ctr_y"]),
        (so["scl_ref"], so["scl_y"], sc["scl_y"]),
        (so["rot_ref"], so["rot_y"], sc["rot_y"]),
    ]

    colors = {
        "ref": "#9e9e9e",
        "orig": "blue",
        "corr": "red",
    }

    for i, (R, O, C) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 4, i, projection="3d")

        ax.scatter(R[:,0], R[:,1], R[:,2], s=8, alpha=0.35, c=colors["ref"], label="Initial rigid reference")
        ax.scatter(O[:,0], O[:,1], O[:,2], s=10, alpha=0.75, c=colors["orig"], label="Original Action")
        ax.scatter(C[:,0], C[:,1], C[:,2], s=10, alpha=0.55, c=colors["corr"], label="Corrected Action")

        ax.set_title(titles[i-1], fontsize=13, pad=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_equal_3d_axes(ax, [R, O, C])

    fig.suptitle(
        f"{best['orig_path'].stem}\n"
        f"Segmented rigid-object alignment process | frame={best['frame']}, object_id={best['object_id']}\n"
        f"Original residual: {best['orig_resid_mm']:.3f} mm   |   "
        f"Corrected residual: {best['corr_resid_mm']:.3f} mm   |   "
        f"Gain: {best['gain_mm']:.3f} mm",
        fontsize=16,
        y=0.98,
    )

    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.90), fontsize=11)

    plt.tight_layout(rect=[0, 0, 0.96, 0.90])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    print("SAVE:", out_path)
    print("BEST CASE:")
    print("  orig npz :", best["orig_path"])
    print("  corr npz :", best["corr_path"])
    print("  object_id:", best["object_id"])
    print("  frame    :", best["frame"])
    print("  n_points :", best["n_points"])
    print("  orig mm  :", best["orig_resid_mm"])
    print("  corr mm  :", best["corr_resid_mm"])
    print("  gain mm  :", best["gain_mm"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_root", type=str, required=True)
    ap.add_argument("--corr_root", type=str, required=True)
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--min_points", type=int, default=80)
    ap.add_argument(
        "--out",
        type=str,
        default="diagnostics/results/best_pa_alignment_case.png",
    )
    args = ap.parse_args()

    best = pick_best_case(
        args.orig_root,
        args.corr_root,
        frame_idx=args.frame,
        min_points=args.min_points,
    )
    plot_case(best, args.out)

if __name__ == "__main__":
    main()
