from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from arguments import parse_args
from evaluation.tester import Tester
import dataset_components.dataloader as dlmod

import diagnostics.behavior_object_mapping_smoke_task0006 as omap
from diagnostics.behavior_object_mpjpe_pa_smoke_task0006 import (
    MODEL_PATH,
    NORM_STATS,
)

# ============================================================
# CONFIG
# ============================================================

BOUNDARY = 0.24580418
TARGET_OBJECT = "radio_89"

TASKS = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]

TASK_DIRS = {
    "task-0000": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
    "task-0001": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0001",
    "task-0006": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006",
    "task-0008": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0008",
    "task-0010": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0010",
}

PA_ROOT = Path(
    "diagnostics/results/object_pa_mpjpe_acc_5task_current10457"
)

OUT = Path(
    "diagnostics/results/radio_breakage_visualization"
)
OUT.mkdir(parents=True, exist_ok=True)

# residual arrows are enlarged ONLY for visualization
RESIDUAL_MAGNIFICATION = 4.0

# ============================================================
# HELPERS
# ============================================================

def reset_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_key(x):
    if isinstance(x, (list, tuple)):
        x = x[0]

    if torch.is_tensor(x) and x.numel() == 1:
        x = x.item()

    if isinstance(x, bytes):
        x = x.decode()

    return str(x)


def to_device(batch, device):
    return {
        k: (
            v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


def mask3(batch, name):
    x = batch[name][0]

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    x = np.asarray(x).astype(bool)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    return x


# ============================================================
# SIMILARITY PROCRUSTES
# ============================================================

def fit_similarity(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    mx = X.mean(axis=0, keepdims=True)
    my = Y.mean(axis=0, keepdims=True)

    X0 = X - mx
    Y0 = Y - my

    nx = np.linalg.norm(X0)
    ny = np.linalg.norm(Y0)

    if nx < 1e-12 or ny < 1e-12:
        raise RuntimeError("Degenerate Procrustes geometry.")

    Xn = X0 / nx
    Yn = Y0 / ny

    H = Xn.T @ Yn
    U, S, Vt = np.linalg.svd(H)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        S[-1] *= -1
        R = U @ Vt

    scale = (S.sum() * ny) / nx
    trans = my - scale * (mx @ R)

    return float(scale), R, trans


def pa_align(X, Y):
    s, R, t = fit_similarity(X, Y)
    Xa = s * X @ R + t

    residual_m = np.linalg.norm(
        Xa - Y,
        axis=1,
    )

    return (
        Xa,
        float(residual_m.mean() * 1000.0),
        residual_m * 1000.0,
    )


# ============================================================
# 3D DISPLAY
# ============================================================

def equal3d(ax, arrays, pad=1.08):
    P = np.concatenate(arrays, axis=0)

    lo = np.nanmin(P, axis=0)
    hi = np.nanmax(P, axis=0)

    ctr = 0.5 * (lo + hi)
    r = 0.5 * np.max(hi - lo)
    r = max(r, 1e-8) * pad

    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)

    ax.set_box_aspect((1, 1, 1))


def depth_gray(P):
    z = P[:, 2]
    d = z.max() - z.min()

    if d < 1e-12:
        q = np.ones(len(P)) * 0.5
    else:
        q = (z - z.min()) / d

    rgba = np.stack([
        0.12 + 0.55*q,
        0.12 + 0.55*q,
        0.12 + 0.55*q,
        0.45 + 0.50*q,
    ], axis=1)

    size = 7 + 10*q
    return rgba, size


# ============================================================
# 1. FIND A VISIBLY BROKEN BUT NON-EXTREME RADIO
# ============================================================

parts = []

for task in TASKS:
    f = PA_ROOT / task / "per_frame_metrics.csv"

    if not f.exists():
        continue

    d = pd.read_csv(f)

    if "object_short_name" not in d.columns:
        continue

    d = d[
        d["object_short_name"].astype(str)
        == TARGET_OBJECT
    ].copy()

    if len(d):
        parts.append(d)

if not parts:
    raise RuntimeError(
        f"No {TARGET_OBJECT} rows found."
    )

radio = pd.concat(
    parts,
    ignore_index=True,
)

# one row already corresponds to object/frame in this evaluator
tracks = []

for (
    task,
    clip,
    oid,
), g in radio.groupby(
    ["task", "clip", "object_id"]
):

    g = g.sort_values("frame")

    if g["frame"].nunique() != 10:
        continue

    acc = float(
        g["rms_acceleration_si_m_s2"].median()
    )

    if acc <= BOUNDARY:
        continue

    peak_idx = g["pa_mpjpe_mm"].idxmax()
    peak = g.loc[peak_idx]

    tracks.append({
        "task": task,
        "clip": clip,
        "object_id": int(oid),
        "acc": acc,
        "points_median": float(
            g["valid_point_count"].median()
        ),
        "mean_pa": float(
            g["pa_mpjpe_mm"].mean()
        ),
        "peak_pa": float(
            peak["pa_mpjpe_mm"]
        ),
        "peak_frame": int(
            peak["frame"]
        ),
    })

T = pd.DataFrame(tracks)

if len(T) == 0:
    raise RuntimeError(
        "No high-dynamics complete radio tracks."
    )

# Prefer enough visible points
if (T["points_median"] >= 80).any():
    T = T[
        T["points_median"] >= 80
    ].copy()

# ------------------------------------------------------------
# Choose near 75th percentile peak PA:
# clearly visible failure, but not maximum cherry-pick
# ------------------------------------------------------------
target_peak = float(
    T["peak_pa"].quantile(0.75)
)

T["distance_to_q75"] = (
    T["peak_pa"] - target_peak
).abs()

T = T.sort_values(
    [
        "distance_to_q75",
        "points_median",
    ],
    ascending=[
        True,
        False,
    ],
)

choice = T.iloc[0]

TASK = str(choice["task"])
TARGET = str(choice["clip"])
OID = int(choice["object_id"])
FAIL_T = int(choice["peak_frame"])
ESTABLISHED_PA = float(choice["peak_pa"])
ACC = float(choice["acc"])

T.to_csv(
    OUT / "radio_highdyn_candidates.csv",
    index=False,
)

pd.DataFrame([choice]).to_csv(
    OUT / "selected_radio_case.csv",
    index=False,
)

print("=" * 90)
print("SELECTED RADIO BREAKAGE CASE")
print("=" * 90)
print("N high-dynamics radio tracks =", len(T))
print("75th-percentile peak PA      =", target_peak)
print("TASK                         =", TASK)
print("CLIP                         =", TARGET)
print("OBJECT                       =", TARGET_OBJECT)
print("OBJECT ID                    =", OID)
print("FAILURE FRAME                =", FAIL_T)
print("RMS ACC                      =", ACC)
print("ESTABLISHED PEAK PA          =", ESTABLISHED_PA)
print("MEDIAN POINTS                =", choice["points_median"])
print("=" * 90)


# ============================================================
# 2. OFFICIAL OBJECT-ID LOADER
# ============================================================

OBJECT_BUILD = (
    omap.build_flow_sample_with_object_ids
)

omap.pmod.grid_sample_transform = (
    omap.grid_sample_with_object_ids
)

omap.pmod.gather = (
    omap.gather_with_object_ids
)

dlmod.build_flow_sample = OBJECT_BUILD


# ============================================================
# 3. POINTWORLD ORIGINAL ACTION INFERENCE
# ============================================================

pw = parse_args(
    skip_command_line=True
)

pw.model_path = MODEL_PATH
pw.norm_stats_path = NORM_STATS

pw.domains = ["behavior"]
pw.data_dirs = [TASK_DIRS[TASK]]

pw.batch_size = 1
pw.num_workers = 0
pw.eval_num_workers = 0
pw.eval_num_batches = -1
pw.device = "cuda"
pw.eval_skip_viz = True

reset_seed(42)

tester = Tester(pw)
tester.model.eval()

reset_seed(42)

loader, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

found = False

for batch in loader:

    key = normalize_key(
        batch["__key__"]
    )

    if key != TARGET:
        continue

    found = True
    print("FOUND TARGET =", key)

    gb = to_device(
        batch,
        tester.device,
    )

    reset_seed(12345)

    with torch.no_grad():
        pred = (
            tester.model(
                gb,
                training=False,
            )["scene_flows"][0]
            .detach()
            .cpu()
            .numpy()
        )

    object_id = (
        batch["scene_object_id"][0]
        .detach()
        .cpu()
        .numpy()
    )

    context = mask3(
        batch,
        "scene_context_mask",
    )

    exists = mask3(
        batch,
        "scene_exists",
    )

    supervised = mask3(
        batch,
        "scene_supervised_mask",
    )

    moved = mask3(
        batch,
        "scene_moved_mask",
    )

    valid = (
        (~context)
        & exists
        & supervised
    )

    gt = (
        batch["gt_scene_flows"][0]
        .detach()
        .cpu()
        .numpy()
    )

    # ========================================================
    # 4. FULL TRUE RADIO SHAPE
    #
    # use every persistent radio point at t=0
    # not just PA evaluation subset
    # ========================================================

    full_mask = (
        (object_id[0] == OID)
        & exists[0]
    )

    full_idx = np.where(
        full_mask
    )[0]

    full_radio = gt[
        0,
        full_idx,
    ]

    print("FULL RADIO POINTS =", len(full_idx))

    # ========================================================
    # 5. FAILURE FRAME — PA ELIGIBLE POINTS
    # ========================================================

    m = (
        (object_id[FAIL_T] == OID)
        & valid[FAIL_T]
        & moved[FAIL_T]
        & (object_id[0] == OID)
        & exists[0]
    )

    idx = np.where(m)[0]

    if len(idx) < 8:
        raise RuntimeError(
            f"Only {len(idx)} points at failure frame."
        )

    pred_raw = pred[
        FAIL_T,
        idx,
    ]

    gt_fail = gt[
        FAIL_T,
        idx,
    ]

    pred_aligned, fresh_pa, residual_mm = (
        pa_align(
            pred_raw,
            gt_fail,
        )
    )

    print("PA POINTS                =", len(idx))
    print("FRESH PA                 =", fresh_pa)
    print("ESTABLISHED PA           =", ESTABLISHED_PA)
    print("ABS DIFF                  =", abs(fresh_pa - ESTABLISHED_PA))

    p50 = float(
        np.percentile(
            residual_mm,
            50,
        )
    )

    p90 = float(
        np.percentile(
            residual_mm,
            90,
        )
    )

    p95 = float(
        np.percentile(
            residual_mm,
            95,
        )
    )

    max_res = float(
        residual_mm.max()
    )

    # ========================================================
    # 6. COMMON DISPLAY ORIENTATION
    #
    # Rigid rotation only; shape itself is unchanged.
    # ========================================================

    center = full_radio.mean(
        axis=0,
        keepdims=True,
    )

    X = full_radio - center

    _, _, vt = np.linalg.svd(
        X,
        full_matrices=False,
    )

    B = vt.T

    if np.linalg.det(B) < 0:
        B[:, 2] *= -1

    def vis(P, origin=None):
        if origin is None:
            origin = center
        return (
            np.asarray(P)
            - origin
        ) @ B

    full_vis = vis(
        full_radio
    )

    fail_center = gt_fail.mean(
        axis=0,
        keepdims=True,
    )

    gt_vis = (
        gt_fail - fail_center
    ) @ B

    pred_vis = (
        pred_aligned - fail_center
    ) @ B

    residual_vec_vis = (
        pred_vis - gt_vis
    )

    # ========================================================
    # 7. CONTACT SHEET OF RADIO VIEW ANGLES
    #
    # Useful if the default camera does not show radio shape well.
    # ========================================================

    fig = plt.figure(
        figsize=(12, 7)
    )

    azims = [
        -90, -60, -30, 0,
        30, 60, 90, 120,
        150, 180, 210, 240,
    ]

    rgba, sizes = depth_gray(
        full_vis
    )

    for i, az in enumerate(
        azims
    ):
        ax = fig.add_subplot(
            3,
            4,
            i + 1,
            projection="3d",
        )

        ax.scatter(
            full_vis[:, 0],
            full_vis[:, 1],
            full_vis[:, 2],
            s=sizes,
            c=rgba,
            linewidths=0,
            depthshade=False,
        )

        equal3d(
            ax,
            [full_vis],
        )

        ax.view_init(
            elev=18,
            azim=az,
        )

        ax.set_title(
            f"azim={az}°",
            fontsize=9,
        )

        ax.set_axis_off()

    plt.tight_layout()

    fig.savefig(
        OUT / "radio_view_angle_sheet.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Default view.
    # Change only this number after looking at view_angle_sheet.
    VIEW = (18, -60)

    # ========================================================
    # 8. FINAL 4-PANEL BREAKAGE FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=(14.4, 3.7)
    )

    # --------------------------------------------------------
    # Panel 1 — TRUE full radio
    # --------------------------------------------------------

    ax1 = fig.add_subplot(
        1,
        4,
        1,
        projection="3d",
    )

    rgba, sizes = depth_gray(
        full_vis
    )

    ax1.scatter(
        full_vis[:, 0],
        full_vis[:, 1],
        full_vis[:, 2],
        s=sizes,
        c=rgba,
        linewidths=0,
        depthshade=False,
    )

    equal3d(
        ax1,
        [full_vis],
    )

    ax1.view_init(
        elev=VIEW[0],
        azim=VIEW[1],
    )

    ax1.set_axis_off()

    ax1.set_title(
        "(a) Full segmented radio\nGT reference",
        fontsize=11,
        weight="bold",
    )

    ax1.text2D(
        0.5,
        0.02,
        f"{len(full_idx)} object points",
        transform=ax1.transAxes,
        ha="center",
        fontsize=8.5,
        color="0.35",
    )

    # --------------------------------------------------------
    # Panel 2 — actual PA-aligned overlap
    # --------------------------------------------------------

    ax2 = fig.add_subplot(
        1,
        4,
        2,
        projection="3d",
    )

    ax2.scatter(
        gt_vis[:, 0],
        gt_vis[:, 1],
        gt_vis[:, 2],
        s=15,
        c="0.72",
        alpha=0.52,
        linewidths=0,
        depthshade=False,
        label="GT",
    )

    ax2.scatter(
        pred_vis[:, 0],
        pred_vis[:, 1],
        pred_vis[:, 2],
        s=13,
        c="#D95F3A",
        alpha=0.82,
        linewidths=0,
        depthshade=False,
        label="Prediction",
    )

    equal3d(
        ax2,
        [gt_vis, pred_vis],
    )

    ax2.view_init(
        elev=VIEW[0],
        azim=VIEW[1],
    )

    ax2.set_axis_off()

    ax2.set_title(
        "(b) Rigid-shape mismatch\n"
        "after PA alignment",
        fontsize=11,
        weight="bold",
    )

    ax2.legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
    )

    ax2.text2D(
        0.5,
        0.02,
        f"PA-MPJPE = {fresh_pa:.2f} mm",
        transform=ax2.transAxes,
        ha="center",
        fontsize=9,
        weight="bold",
    )

    # --------------------------------------------------------
    # Panel 3 — point-wise residual heatmap
    # --------------------------------------------------------

    ax3 = fig.add_subplot(
        1,
        4,
        3,
        projection="3d",
    )

    vmax = max(
        float(
            np.percentile(
                residual_mm,
                95,
            )
        ),
        1e-6,
    )

    norm = colors.Normalize(
        vmin=0.0,
        vmax=vmax,
    )

    sc = ax3.scatter(
        gt_vis[:, 0],
        gt_vis[:, 1],
        gt_vis[:, 2],
        s=20,
        c=residual_mm,
        cmap="turbo",
        norm=norm,
        linewidths=0,
        depthshade=False,
    )

    equal3d(
        ax3,
        [gt_vis],
    )

    ax3.view_init(
        elev=VIEW[0],
        azim=VIEW[1],
    )

    ax3.set_axis_off()

    ax3.set_title(
        "(c) Where does the\nshape break?",
        fontsize=11,
        weight="bold",
    )

    cb = fig.colorbar(
        sc,
        ax=ax3,
        fraction=0.046,
        pad=0.01,
        shrink=0.68,
    )

    cb.set_label(
        "Point residual (mm)",
        fontsize=8,
    )

    ax3.text2D(
        0.5,
        0.01,
        f"median={p50:.2f}  |  p95={p95:.2f} mm",
        transform=ax3.transAxes,
        ha="center",
        fontsize=8.5,
    )

    # --------------------------------------------------------
    # Panel 4 — magnified displacement directions
    # --------------------------------------------------------

    ax4 = fig.add_subplot(
        1,
        4,
        4,
        projection="3d",
    )

    ax4.scatter(
        gt_vis[:, 0],
        gt_vis[:, 1],
        gt_vis[:, 2],
        s=8,
        c="0.75",
        alpha=0.38,
        linewidths=0,
        depthshade=False,
    )

    # show a subset of arrows to avoid clutter
    n = len(gt_vis)
    max_arrows = 60

    if n <= max_arrows:
        sel = np.arange(n)
    else:
        # Prefer points with larger residual while still spreading samples
        order = np.argsort(
            residual_mm
        )[::-1]

        sel = order[
            np.linspace(
                0,
                n - 1,
                max_arrows,
            ).astype(int)
        ]

    V = (
        residual_vec_vis[sel]
        * RESIDUAL_MAGNIFICATION
    )

    ax4.quiver(
        gt_vis[sel, 0],
        gt_vis[sel, 1],
        gt_vis[sel, 2],
        V[:, 0],
        V[:, 1],
        V[:, 2],
        color="#D95F3A",
        linewidth=0.8,
        arrow_length_ratio=0.20,
        normalize=False,
    )

    equal3d(
        ax4,
        [
            gt_vis,
            gt_vis
            + residual_vec_vis
            * RESIDUAL_MAGNIFICATION,
        ],
    )

    ax4.view_init(
        elev=VIEW[0],
        azim=VIEW[1],
    )

    ax4.set_axis_off()

    ax4.set_title(
        "(d) Shape-distortion vectors",
        fontsize=11,
        weight="bold",
    )

    ax4.text2D(
        0.5,
        0.02,
        (
            f"vectors ×{RESIDUAL_MAGNIFICATION:.0f} "
            "for visibility"
        ),
        transform=ax4.transAxes,
        ha="center",
        fontsize=8.5,
        color="#D95F3A",
    )

    # ========================================================
    # FIGURE TITLE
    # ========================================================

    fig.suptitle(
        "Rigid-object shape distortion under a high-dynamics action",
        fontsize=14,
        weight="bold",
        y=0.995,
    )

    plt.subplots_adjust(
        left=0.01,
        right=0.985,
        bottom=0.08,
        top=0.86,
        wspace=0.04,
    )

    final_png = (
        OUT
        / "radio_breakage_final.png"
    )

    final_pdf = (
        OUT
        / "radio_breakage_final.pdf"
    )

    fig.savefig(
        final_png,
        dpi=500,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        final_pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    # ========================================================
    # 9. SAVE NUMERICAL AUDIT
    # ========================================================

    audit = (
        f"TASK={TASK}\n"
        f"CLIP={TARGET}\n"
        f"OBJECT={TARGET_OBJECT}\n"
        f"OBJECT_ID={OID}\n"
        f"FRAME={FAIL_T}\n"
        f"RMS_ACC={ACC:.9f}\n"
        f"FULL_OBJECT_POINTS={len(full_idx)}\n"
        f"PA_POINTS={len(idx)}\n"
        f"FRESH_PA_MPJPE={fresh_pa:.9f}\n"
        f"ESTABLISHED_PA_MPJPE={ESTABLISHED_PA:.9f}\n"
        f"PA_DIFF={fresh_pa - ESTABLISHED_PA:+.9f}\n"
        f"POINT_RESIDUAL_MEDIAN={p50:.9f}\n"
        f"POINT_RESIDUAL_P90={p90:.9f}\n"
        f"POINT_RESIDUAL_P95={p95:.9f}\n"
        f"POINT_RESIDUAL_MAX={max_res:.9f}\n"
        f"ARROW_MAGNIFICATION={RESIDUAL_MAGNIFICATION}\n"
    )

    (
        OUT
        / "AUDIT.txt"
    ).write_text(
        audit
    )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)
    print(audit)
    print("FINAL PNG =", final_png)
    print("FINAL PDF =", final_pdf)
    print(
        "VIEW SHEET =",
        OUT / "radio_view_angle_sheet.png"
    )
    print("=" * 90)

    break


if not found:
    raise RuntimeError(
        f"Target clip not found: {TARGET}"
    )
