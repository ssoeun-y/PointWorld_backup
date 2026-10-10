from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arguments import parse_args
from evaluation.tester import Tester
import dataset_components.dataloader as dlmod

import diagnostics.behavior_object_mapping_smoke_task0006 as omap
from diagnostics.behavior_object_mpjpe_pa_smoke_task0006 import (
    MODEL_PATH,
    NORM_STATS,
)

# ============================================================
# FIXED PATHS
# ============================================================

TASK_DIRS = {
    "task-0000": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
    "task-0001": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0001",
    "task-0006": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006",
    "task-0008": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0008",
    "task-0010": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0010",
}

CHOICE_CSV = Path(
    "diagnostics/results/problem_failure_selection/"
    "representative_highdyn_choice.csv"
)

PA_ROOT = Path(
    "diagnostics/results/object_pa_mpjpe_acc_5task_current10457"
)

OUT = Path(
    "diagnostics/results/problem_failure_qualitative_final"
)
OUT.mkdir(parents=True, exist_ok=True)

FRAMES = [1, 5, 10]

C_PRED = "#D95F3A"
C_GT = "#A8A8A8"
C_TARGET = "#E89B2D"

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

    if torch.is_tensor(x):
        if x.numel() == 1:
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
# Similarity Procrustes
#
# Find s,R,t:
#       s * X @ R + t ~= Y
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
        raise RuntimeError("Degenerate geometry in Procrustes.")

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

    t = my - scale * (mx @ R)

    return float(scale), R, t


def apply_similarity(X, s, R, t):
    return s * np.asarray(X) @ R + t


def pa_align(X, Y):
    s, R, t = fit_similarity(X, Y)

    aligned = apply_similarity(
        X,
        s,
        R,
        t,
    )

    pa_mm = (
        np.linalg.norm(
            aligned - Y,
            axis=1,
        ).mean()
        * 1000.0
    )

    return aligned, float(pa_mm)


# ============================================================
# PLOTTING HELPERS
# ============================================================

def equal_3d(ax, arrays, pad=1.10):
    P = np.concatenate(
        [np.asarray(a) for a in arrays],
        axis=0,
    )

    lo = P.min(axis=0)
    hi = P.max(axis=0)

    center = 0.5 * (lo + hi)

    radius = 0.5 * np.max(hi - lo)
    radius = max(radius, 1e-8) * pad

    ax.set_xlim(
        center[0] - radius,
        center[0] + radius,
    )
    ax.set_ylim(
        center[1] - radius,
        center[1] + radius,
    )
    ax.set_zlim(
        center[2] - radius,
        center[2] + radius,
    )

    ax.set_box_aspect((1, 1, 1))


def depth_values(P):
    z = np.asarray(P)[:, 2]

    span = z.max() - z.min()

    if span < 1e-12:
        return np.full(len(z), 0.5)

    return (z - z.min()) / span


def draw_shape(
    ax,
    gt,
    pred,
    view,
    title=None,
):
    gt = np.asarray(gt)
    pred = np.asarray(pred)

    zg = depth_values(gt)
    zp = depth_values(pred)

    gt_rgba = np.stack([
        0.58 + 0.25 * zg,
        0.58 + 0.25 * zg,
        0.58 + 0.25 * zg,
        0.18 + 0.42 * zg,
    ], axis=1)

    pred_rgba = np.stack([
        0.74 + 0.20 * zp,
        0.20 + 0.16 * zp,
        0.10 + 0.10 * zp,
        0.45 + 0.50 * zp,
    ], axis=1)

    # GT reference
    ax.scatter(
        gt[:, 0],
        gt[:, 1],
        gt[:, 2],
        s=7 + 8 * zg,
        c=gt_rgba,
        linewidths=0,
        depthshade=False,
        label="GT",
    )

    # predicted rigid object
    ax.scatter(
        pred[:, 0],
        pred[:, 1],
        pred[:, 2],
        s=10 + 14 * zp,
        c=pred_rgba,
        linewidths=0,
        depthshade=False,
        label="Prediction",
    )

    equal_3d(
        ax,
        [gt, pred],
        pad=1.08,
    )

    ax.view_init(
        elev=view[0],
        azim=view[1],
    )

    if title:
        ax.set_title(
            title,
            fontsize=12,
            pad=7,
        )

    ax.set_axis_off()


def save_single_panel(
    path,
    gt,
    pred,
    view,
    frame,
    pa,
    established_pa,
    highlight=False,
):
    fig = plt.figure(
        figsize=(3.1, 3.0)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    draw_shape(
        ax,
        gt,
        pred,
        view,
    )

    ax.text2D(
        0.5,
        0.98,
        f"t = {frame}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        weight="bold",
    )

    ax.text2D(
        0.5,
        0.02,
        f"PA-MPJPE {established_pa:.2f} mm",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    if highlight:
        # simple red rectangle in figure coordinates
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (0.015, 0.015),
            0.97,
            0.97,
            transform=fig.transFigure,
            fill=False,
            linewidth=3.2,
            edgecolor=C_PRED,
        )
        fig.add_artist(rect)

    fig.savefig(
        path,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


# ============================================================
# 1. READ REPRESENTATIVE CASE
# ============================================================

if not CHOICE_CSV.exists():
    raise RuntimeError(
        f"Representative choice missing:\n{CHOICE_CSV}\n"
        "Run select_representative_highdyn_failure.py first."
    )

choice = pd.read_csv(
    CHOICE_CSV
).iloc[0]

TASK = str(choice["task"])
TARGET = str(choice["clip"])
OID = int(choice["object_id"])
OBJECT_NAME = str(choice["object_short_name"])

EST_PA = {
    1: float(choice["pa_t1"]),
    5: float(choice["pa_t5"]),
    10: float(choice["pa_t10"]),
}

ACC = float(choice["acc_median"])
POINTS_MED = float(choice["points_median"])
SLOPE = float(choice["pa_slope"])

print("=" * 88)
print("QUALITATIVE PROBLEM FIGURE")
print("=" * 88)
print("TASK        =", TASK)
print("CLIP        =", TARGET)
print("OBJECT      =", OBJECT_NAME)
print("OBJECT ID   =", OID)
print("RMS ACC     =", ACC)
print("POINTS med  =", POINTS_MED)
print("PA slope    =", SLOPE)
print(
    "Established PA:",
    f"t1={EST_PA[1]:.6f}",
    f"t5={EST_PA[5]:.6f}",
    f"t10={EST_PA[10]:.6f}",
)
print("=" * 88)

# ============================================================
# 2. OFFICIAL OBJECT-ID WRAPPER
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
# 3. POINTWORLD — ORIGINAL ACTION ONLY
# ============================================================

pw = parse_args(
    skip_command_line=True
)

pw.model_path = MODEL_PATH
pw.norm_stats_path = NORM_STATS

pw.domains = [
    "behavior"
]

pw.data_dirs = [
    TASK_DIRS[TASK]
]

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

# ============================================================
# 4. FIND TARGET CLIP — ONE PASS
# ============================================================

found = False

for b0 in loader:

    key = normalize_key(
        b0["__key__"]
    )

    if key != TARGET:
        continue

    found = True

    print("FOUND =", key)

    g0 = to_device(
        b0,
        tester.device,
    )

    # fixed seed for reproducible visualization
    reset_seed(12345)

    with torch.no_grad():
        pred = (
            tester.model(
                g0,
                training=False,
            )["scene_flows"][0]
            .detach()
            .cpu()
            .numpy()
        )

    object_id = (
        b0["scene_object_id"][0]
        .detach()
        .cpu()
        .numpy()
    )

    context = mask3(
        b0,
        "scene_context_mask",
    )

    exists = mask3(
        b0,
        "scene_exists",
    )

    supervised = mask3(
        b0,
        "scene_supervised_mask",
    )

    moved = mask3(
        b0,
        "scene_moved_mask",
    )

    valid = (
        (~context)
        & exists
        & supervised
    )

    full_gt = (
        b0["gt_scene_flows"][0]
        .detach()
        .cpu()
        .numpy()
    )

    # ========================================================
    # 5. EXTRACT FRAMES + EXACT PA ALIGNMENT
    # ========================================================

    frame_data = {}

    for t in FRAMES:

        m = (
            (object_id[t] == OID)
            & valid[t]
            & moved[t]
            & (object_id[0] == OID)
            & exists[0]
        )

        idx = np.where(m)[0]

        if len(idx) < 8:
            raise RuntimeError(
                f"Object {OID} has only {len(idx)} eligible points at t={t}"
            )

        pred_t = pred[t, idx]
        gt_t = full_gt[t, idx]

        # same protocol as main PA interpretation:
        # prediction(t) -> GT(t)
        pred_pa, pa_fresh = pa_align(
            pred_t,
            gt_t,
        )

        # ----------------------------------------------------
        # Canonical pose only for visualization:
        # GT(t) -> GT(0), same transform applied to pred_pa.
        #
        # This does NOT change the PA-MPJPE calculation above.
        # It only makes t=1/5/10 directly visually comparable.
        # ----------------------------------------------------
        gt0_same_idx = full_gt[0, idx]

        s_c, R_c, tr_c = fit_similarity(
            gt_t,
            gt0_same_idx,
        )

        gt_canon = apply_similarity(
            gt_t,
            s_c,
            R_c,
            tr_c,
        )

        pred_canon = apply_similarity(
            pred_pa,
            s_c,
            R_c,
            tr_c,
        )

        frame_data[t] = {
            "idx": idx,
            "gt": gt_canon,
            "pred": pred_canon,
            "pa_fresh": pa_fresh,
        }

        print(
            f"t={t:2d} | points={len(idx):4d} | "
            f"fresh PA={pa_fresh:.6f} mm | "
            f"established PA={EST_PA[t]:.6f} mm | "
            f"diff={pa_fresh - EST_PA[t]:+.6f} mm"
        )

    # ========================================================
    # 6. ONE COMMON VIEW FOR ALL THREE PANELS
    # ========================================================

    # use canonical GT from start frame
    ref = frame_data[1]["gt"]
    ref0 = (
        ref
        - ref.mean(
            axis=0,
            keepdims=True,
        )
    )

    _, _, vt = np.linalg.svd(
        ref0,
        full_matrices=False,
    )

    basis = vt.T

    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1

    for t in FRAMES:

        gt = frame_data[t]["gt"]
        pr = frame_data[t]["pred"]

        center = gt.mean(
            axis=0,
            keepdims=True,
        )

        frame_data[t]["gt_vis"] = (
            (gt - center)
            @ basis
        )

        frame_data[t]["pred_vis"] = (
            (pr - center)
            @ basis
        )

    VIEW = (18, -62)

    # ========================================================
    # 7. CONTEXT PANEL
    #    local GT scene with selected trash_can highlighted
    # ========================================================

    tctx = 1

    scene_mask = (
        exists[tctx]
        & (~context[tctx])
    )

    scene_idx = np.where(
        scene_mask
    )[0]

    target_idx = np.where(
        (object_id[tctx] == OID)
        & exists[tctx]
    )[0]

    if len(target_idx) == 0:
        raise RuntimeError(
            "Target object absent in context frame."
        )

    scene_xyz = full_gt[
        tctx,
        scene_idx,
    ]

    target_xyz = full_gt[
        tctx,
        target_idx,
    ]

    target_center = target_xyz.mean(
        axis=0
    )

    dist = np.linalg.norm(
        scene_xyz - target_center,
        axis=1,
    )

    # nearest local scene points only
    K = min(
        2500,
        len(scene_xyz),
    )

    nearest = np.argsort(
        dist
    )[:K]

    local_scene = scene_xyz[
        nearest
    ]

    context_png = (
        OUT
        / "context.png"
    )

    fig = plt.figure(
        figsize=(3.2, 3.0)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.scatter(
        local_scene[:, 0],
        local_scene[:, 1],
        local_scene[:, 2],
        s=2.0,
        c="0.82",
        alpha=0.20,
        linewidths=0,
        depthshade=True,
    )

    ax.scatter(
        target_xyz[:, 0],
        target_xyz[:, 1],
        target_xyz[:, 2],
        s=13,
        c=C_TARGET,
        alpha=0.95,
        linewidths=0,
        depthshade=True,
    )

    equal_3d(
        ax,
        [local_scene, target_xyz],
        pad=1.03,
    )

    ax.view_init(
        elev=18,
        azim=-58,
    )

    ax.set_axis_off()

    ax.set_title(
        f"Target: {OBJECT_NAME}",
        fontsize=11,
        weight="bold",
        pad=4,
    )

    fig.savefig(
        context_png,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    # ========================================================
    # 8. SAVE EACH FRAME INDIVIDUALLY
    # ========================================================

    for t in FRAMES:

        p = (
            OUT
            / f"t{t}_prediction.png"
        )

        save_single_panel(
            p,
            frame_data[t]["gt_vis"],
            frame_data[t]["pred_vis"],
            VIEW,
            t,
            frame_data[t]["pa_fresh"],
            EST_PA[t],
            highlight=(t == 10),
        )

    # ========================================================
    # 9. FINAL INTRO / ABSTRACT FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=(12.8, 4.35)
    )

    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[
            1.12,
            1.00,
            0.86,
            1.06,
        ],
        left=0.035,
        right=0.985,
        top=0.77,
        bottom=0.19,
        wspace=0.035,
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    ax0 = fig.add_subplot(
        gs[0, 0],
        projection="3d",
    )

    ax0.scatter(
        local_scene[:, 0],
        local_scene[:, 1],
        local_scene[:, 2],
        s=1.8,
        c="0.82",
        alpha=0.16,
        linewidths=0,
        depthshade=True,
    )

    ax0.scatter(
        target_xyz[:, 0],
        target_xyz[:, 1],
        target_xyz[:, 2],
        s=12,
        c=C_TARGET,
        alpha=0.95,
        linewidths=0,
        depthshade=True,
    )

    equal_3d(
        ax0,
        [local_scene, target_xyz],
        pad=1.03,
    )

    ax0.view_init(
        elev=18,
        azim=-58,
    )

    ax0.set_axis_off()

    ax0.set_title(
        f"Target object\n{OBJECT_NAME}",
        fontsize=10.5,
        pad=3,
    )

    # --------------------------------------------------------
    # t1 / t5 / t10
    # --------------------------------------------------------

    axes = []

    labels = {
        1: "t = start",
        5: "t = mid",
        10: "t = end",
    }

    for col, t in enumerate(
        FRAMES,
        start=1,
    ):

        ax = fig.add_subplot(
            gs[0, col],
            projection="3d",
        )

        draw_shape(
            ax,
            frame_data[t]["gt_vis"],
            frame_data[t]["pred_vis"],
            VIEW,
        )

        ax.set_title(
            labels[t],
            fontsize=11.5,
            weight="bold"
            if t in [1, 10]
            else "normal",
            color=(
                C_PRED
                if t == 10
                else "black"
            ),
            pad=5,
        )

        ax.text2D(
            0.5,
            0.01,
            f"{EST_PA[t]:.2f} mm",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=(
                C_PRED
                if t == 10
                else "0.25"
            ),
            weight=(
                "bold"
                if t == 10
                else "normal"
            ),
        )

        axes.append(ax)

    # red failure box only around final frame
    from matplotlib.patches import Rectangle

    pos = axes[-1].get_position()

    rect = Rectangle(
        (
            pos.x0 - 0.006,
            pos.y0 - 0.008,
        ),
        pos.width + 0.012,
        pos.height + 0.016,
        transform=fig.transFigure,
        fill=False,
        linewidth=3.0,
        edgecolor=C_PRED,
        clip_on=False,
    )

    fig.add_artist(rect)

    axes[-1].text2D(
        0.97,
        0.94,
        "✕",
        transform=axes[-1].transAxes,
        ha="right",
        va="top",
        fontsize=18,
        color=C_PRED,
        weight="bold",
    )

    # ========================================================
    # HEADLINE
    # ========================================================

    fig.text(
        0.51,
        0.925,
        "Point clouds lose rigid-shape consistency under high-dynamics actions",
        ha="center",
        va="center",
        fontsize=16.5,
        weight="bold",
    )

    fig.text(
        0.51,
        0.865,
        (
            "The same rigid object becomes increasingly distorted "
            "as the PointWorld rollout progresses."
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        color="0.28",
    )

    # ========================================================
    # SMALL BOTTOM NOTE — no table
    # ========================================================

    fig.text(
        0.52,
        0.075,
        (
            f"PA-MPJPE: {EST_PA[1]:.2f} → {EST_PA[10]:.2f} mm"
            f"   |   RMS Acc: {ACC:.3f} m/s²"
        ),
        ha="center",
        va="center",
        fontsize=10,
    )

    final_png = (
        OUT
        / "problem_failure_qualitative_final.png"
    )

    final_pdf = (
        OUT
        / "problem_failure_qualitative_final.pdf"
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
    # 10. CAPTION + AUDIT
    # ========================================================

    caption = (
        "A rigid object gradually loses shape consistency during "
        "PointWorld rollout under a high-dynamics action. "
        "The example is a representative high-dynamics rigid-object "
        "track selected near the median PA-MPJPE degradation after "
        "visibility and completeness filtering; aggregate statistics "
        "across 7,947 clips are reported separately. "
        f"PA-MPJPE increases from {EST_PA[1]:.2f} mm at t=1 "
        f"to {EST_PA[10]:.2f} mm at t=10."
    )

    (
        OUT
        / "CAPTION.txt"
    ).write_text(
        caption
    )

    audit_lines = [
        f"TASK={TASK}",
        f"CLIP={TARGET}",
        f"OBJECT={OBJECT_NAME}",
        f"OBJECT_ID={OID}",
        f"RMS_ACC={ACC}",
        "",
    ]

    for t in FRAMES:
        audit_lines.append(
            f"t={t}: "
            f"points={len(frame_data[t]['idx'])}, "
            f"fresh_PA={frame_data[t]['pa_fresh']:.9f}, "
            f"established_PA={EST_PA[t]:.9f}, "
            f"diff={frame_data[t]['pa_fresh'] - EST_PA[t]:+.9f}"
        )

    (
        OUT
        / "AUDIT.txt"
    ).write_text(
        "\n".join(
            audit_lines
        )
    )

    print()
    print("=" * 88)
    print("DONE")
    print("=" * 88)
    print("FINAL PNG :", final_png)
    print("FINAL PDF :", final_pdf)
    print("CONTEXT   :", context_png)

    for t in FRAMES:
        print(
            f"t={t} PANEL :",
            OUT / f"t{t}_prediction.png"
        )

    print("CAPTION   :", OUT / "CAPTION.txt")
    print("AUDIT     :", OUT / "AUDIT.txt")
    print()
    print("CAPTION:")
    print(caption)
    print("=" * 88)

    break


if not found:
    raise RuntimeError(
        f"Target clip not found: {TARGET}"
    )
