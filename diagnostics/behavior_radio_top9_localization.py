import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANK_CSV = "diagnostics/results/behavior_radio_rigidity_rank.csv"
NPZ_ROOT = (
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

OUTDIR = (
    "diagnostics/results/"
    "behavior_radio_top9_localization"
)
os.makedirs(OUTDIR, exist_ok=True)

rank = pd.read_csv(RANK_CSV).head(9).copy()

def rigid_align(src, dst):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    sc = src.mean(axis=0)
    dc = dst.mean(axis=0)

    X = src - sc
    Y = dst - dc

    U, S, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = dc - sc @ R.T
    return src @ R.T + t


samples = []

for rank_idx, row in rank.iterrows():
    key = row["sample_key"]
    object_id = int(row["object_id"])

    fname = key.replace(":", "_") + ".npz"
    path = os.path.join(NPZ_ROOT, fname)

    assert os.path.exists(path), path

    d = np.load(path, allow_pickle=True)

    assert d["official_l2_exact"].item()
    assert d["temporal_object_id_exact"].item()

    gt = d["gt_scene_flows"].astype(np.float64)
    pred = d["pred_scene_flows"].astype(np.float64)
    obj = d["scene_object_id"]
    valid = d["official_valid_mask"]

    mask = (obj[0] == object_id) & valid[-1]

    x0 = gt[0, mask]
    xgt = gt[-1, mask]
    xpred = pred[-1, mask]

    assert len(x0) == int(row["n_points"])

    # Remove only global rigid pose difference at t10
    xpred_aligned = rigid_align(xpred, xgt)

    residual_mm = (
        np.linalg.norm(xpred_aligned - xgt, axis=1) * 1000.0
    )

    # Display coordinates:
    # center t0 only. No deformation/NN rematching.
    display = x0 - x0.mean(axis=0)

    samples.append({
        "rank": rank_idx + 1,
        "key": key,
        "xyz": display,
        "residual_mm": residual_mm,
        "mean_mm": residual_mm.mean(),
        "p95_mm": np.quantile(residual_mm, 0.95),
        "max_mm": residual_mm.max(),
        "n": len(x0),
    })


# One common color scale for all 9 samples.
# Use global P95 only for DISPLAY saturation;
# raw residual values themselves are untouched.
all_res = np.concatenate([s["residual_mm"] for s in samples])
vmax = np.quantile(all_res, 0.95)

fig = plt.figure(figsize=(15, 14))

for i, s in enumerate(samples):
    ax = fig.add_subplot(3, 3, i + 1, projection="3d")

    sc = ax.scatter(
        s["xyz"][:, 0],
        s["xyz"][:, 1],
        s["xyz"][:, 2],
        c=s["residual_mm"],
        cmap="inferno",
        vmin=0,
        vmax=vmax,
        s=16,
    )

    ax.set_title(
        f"Rank {s['rank']} | n={s['n']}\n"
        f"mean={s['mean_mm']:.1f}, "
        f"P95={s['p95_mm']:.1f} mm",
        fontsize=10
    )

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=45)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

fig.suptitle(
    "Top-9 radio rigidity failures\n"
    "Color = final non-rigid residual mapped onto original GT t0 points",
    fontsize=15
)

cbar = fig.colorbar(
    sc,
    ax=fig.axes,
    shrink=0.55,
    pad=0.03
)
cbar.set_label("Non-rigid residual (mm)")

out = os.path.join(
    OUTDIR,
    "radio_top9_deformation_localization_t0.png"
)

plt.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)

print("=" * 100)
print("TOP9 LOCALIZATION")

for s in samples:
    print(
        f"rank={s['rank']:2d} "
        f"n={s['n']:4d} "
        f"mean={s['mean_mm']:.3f}mm "
        f"p95={s['p95_mm']:.3f}mm "
        f"max={s['max_mm']:.3f}mm "
        f"key={s['key']}"
    )

print("\nDISPLAY_VMAX_GLOBAL_P95_MM =", vmax)
print("OUTPUT =", out)
print("OUTPUT_EXISTS =", os.path.exists(out))
print("TOP9_LOCALIZATION_SUCCESS = True")
