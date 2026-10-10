#!/usr/bin/env python3

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pmod

from diagnostics.behavior_object_mapping_smoke_task0006 import (
    build_flow_sample_with_object_ids,
    grid_sample_with_object_ids,
    gather_with_object_ids,
    ORIG_BUILD_FLOW_SAMPLE,
    ORIG_GRID_SAMPLE,
    ORIG_GATHER,
    OBJECT_TABLE,
    normalize_key,
)


DATA_DIR = (
    "/home/soenchun/cse/PointWorld_workspace/"
    "datasets/behavior/wds_task0006"
)

MODEL_PATH = (
    "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
    "pretrained_checkpoints/large-droid+behavior/model-best.pt"
)

NORM_STATS = (
    "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
    "stats/droid_behavior"
)

TARGET_TOKEN = "easter_egg_179"

# Final future frame of this 11-frame clip.
VIS_FRAME = 10

OUT_DIR = (
    "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
    "diagnostics/results/object_pa_visualization"
)


def bool_mask(x):
    x = x.bool()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x


def procrustes_video_pose3d(predicted, target):
    """
    Exact VideoPose3D P-MPJPE / PA-MPJPE algebra.

    predicted : (N,3)
    target    : (N,3)

    Returns:
        aligned
        scale
        R
        translation
        pa_error
        mu_pred
        mu_gt
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    muX = np.mean(target, axis=0, keepdims=True)
    muY = np.mean(predicted, axis=0, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2))
    normY = np.sqrt(np.sum(Y0 ** 2))

    if normX <= 0 or normY <= 0:
        raise RuntimeError("Degenerate geometry.")

    X0 /= normX
    Y0 /= normY

    H = X0.T @ Y0
    U, s, Vt = np.linalg.svd(H)

    V = Vt.T
    R = V @ U.T

    sign_detR = np.sign(np.linalg.det(R))
    if sign_detR == 0:
        sign_detR = 1.0

    V[:, -1] *= sign_detR
    s[-1] *= sign_detR

    R = V @ U.T

    tr = np.sum(s)

    scale = tr * normX / normY
    translation = muX - scale * (muY @ R)

    aligned = scale * (predicted @ R) + translation

    pa_error = np.mean(
        np.linalg.norm(aligned - target, axis=-1)
    )

    return (
        aligned,
        float(scale),
        R,
        translation,
        float(pa_error),
        muY,
        muX,
    )


def set_equal_limits(axes, arrays):
    all_points = np.concatenate(arrays, axis=0)

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))

    if span <= 0:
        span = 1.0

    half = span * 0.60

    for ax in axes:
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))


def make_gt_knn_edges(gt, k=2):
    """
    Visualization only.
    For this diagnostic sample (10 points), use the complete graph
    so the whole rigid-object shape is visually obvious.
    """
    n = len(gt)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return edges


def draw_pair(ax, gt, pred, title):
    # Complete graph on GT indices; apply same connectivity to GT and Pred.
    edges = make_gt_knn_edges(gt, k=2)

    # GT structure: red
    for i, j in edges:
        ax.plot(
            [gt[i, 0], gt[j, 0]],
            [gt[i, 1], gt[j, 1]],
            [gt[i, 2], gt[j, 2]],
            c="red",
            linewidth=1.1,
            alpha=0.28,
        )

    # Prediction structure: blue
    for i, j in edges:
        ax.plot(
            [pred[i, 0], pred[j, 0]],
            [pred[i, 1], pred[j, 1]],
            [pred[i, 2], pred[j, 2]],
            c="blue",
            linewidth=1.1,
            alpha=0.28,
        )

    # Point-to-point correspondence: dark gray dashed
    for i in range(gt.shape[0]):
        ax.plot(
            [gt[i, 0], pred[i, 0]],
            [gt[i, 1], pred[i, 1]],
            [gt[i, 2], pred[i, 2]],
            c="gray",
            alpha=0.45,
            linewidth=0.9,
            linestyle="--",
        )

    ax.scatter(
        gt[:, 0],
        gt[:, 1],
        gt[:, 2],
        c="red",
        s=48,
        label="GT",
    )

    ax.scatter(
        pred[:, 0],
        pred[:, 1],
        pred[:, 2],
        c="blue",
        s=48,
        label="PointWorld Pred",
    )

    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")


def main():

    # --------------------------------------------------------
    # Runtime-only object-ID tracking.
    # Official scene tensors themselves remain unchanged.
    # --------------------------------------------------------
    dlmod.build_flow_sample = build_flow_sample_with_object_ids
    pmod.grid_sample_transform = grid_sample_with_object_ids
    pmod.gather = gather_with_object_ids

    try:
        args = parse_args(skip_command_line=True)

        args.model_path = MODEL_PATH
        args.domains = ["behavior"]
        args.data_dirs = [DATA_DIR]
        args.norm_stats_path = NORM_STATS

        args.batch_size = 1
        args.num_workers = 0
        args.eval_num_workers = 0
        args.eval_num_batches = 1

        args.device = "cuda"
        args.eval_skip_viz = True

        tester = Tester(args)

        loader, _ = tester._build_eval_loader(
            "test",
            enable_mask=False,
        )

        batch = next(iter(loader))

        batch = {
            k: (
                v.to(tester.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

        tester.model.eval()

        with torch.no_grad():
            outputs = tester.model(
                batch,
                training=False,
            )

    finally:
        dlmod.build_flow_sample = ORIG_BUILD_FLOW_SAMPLE
        pmod.grid_sample_transform = ORIG_GRID_SAMPLE
        pmod.gather = ORIG_GATHER

    # --------------------------------------------------------
    # Identify exact rigid object points.
    # --------------------------------------------------------
    key = normalize_key(batch["__key__"])

    labels = batch["scene_object_id"]

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    labels_np = labels_np[0]  # (T,N)

    table = OBJECT_TABLE[key]

    target_ids = [
        oid
        for oid, mesh_name in table.items()
        if TARGET_TOKEN in mesh_name
    ]

    if not target_ids:
        raise RuntimeError(
            f"{TARGET_TOKEN} not found in sample {key}"
        )

    object_mask_np = np.isin(
        labels_np,
        np.asarray(target_ids, dtype=np.int32),
    )

    object_mask = torch.from_numpy(
        object_mask_np
    ).to(tester.device)

    # --------------------------------------------------------
    # Official PointWorld valid mask.
    # --------------------------------------------------------
    context = bool_mask(
        batch["scene_context_mask"]
    )

    exists = bool_mask(
        batch["scene_exists"]
    )

    supervised = bool_mask(
        batch["scene_supervised_mask"]
    )

    official_valid = (
        (~context)
        & exists
        & supervised
    )

    valid = (
        official_valid[0]
        & object_mask
    )

    # --------------------------------------------------------
    # Pred(t) vs same-action same-time GT(t).
    # --------------------------------------------------------
    pred = (
        outputs["scene_flows"][0]
        .detach()
        .cpu()
        .numpy()
    )

    gt = (
        batch["gt_scene_flows"][0]
        .detach()
        .cpu()
        .numpy()
    )

    valid_np = (
        valid.detach()
        .cpu()
        .numpy()
    )

    t = VIS_FRAME

    m = valid_np[t]

    if int(m.sum()) < 8:
        raise RuntimeError(
            f"Need >=8 valid object points, got {int(m.sum())}"
        )

    pred_t = pred[t, m]
    gt_t = gt[t, m]

    # --------------------------------------------------------
    # MPJPE before alignment.
    # --------------------------------------------------------
    mpjpe = np.mean(
        np.linalg.norm(
            pred_t - gt_t,
            axis=-1,
        )
    )

    (
        aligned,
        scale,
        R,
        translation,
        pa,
        mu_pred,
        mu_gt,
    ) = procrustes_video_pose3d(
        pred_t,
        gt_t,
    )

    # --------------------------------------------------------
    # Visualization decomposition.
    #
    # These three visual steps compose to the exact final
    # VideoPose3D Procrustes transform:
    #
    # 1) scale prediction around its own centroid
    # 2) shift centroid to GT centroid
    # 3) rotate around GT centroid
    # --------------------------------------------------------

    # Step 1: scale around prediction centroid.
    pred_scale = (
        scale * (pred_t - mu_pred)
        + mu_pred
    )

    # Step 2: move scaled prediction centroid to GT centroid.
    pred_shift = (
        pred_scale
        - mu_pred
        + mu_gt
    )

    # Step 3: rotate around GT centroid.
    pred_rotate = (
        (pred_shift - mu_gt) @ R
        + mu_gt
    )

    # Exact consistency check.
    max_diff = np.max(
        np.abs(pred_rotate - aligned)
    )

    if max_diff > 1e-8:
        raise RuntimeError(
            "Visualization decomposition does not match "
            f"official Procrustes transform: {max_diff}"
        )

    # --------------------------------------------------------
    # Terminal report.
    # --------------------------------------------------------
    print()
    print("========== PROCRUSTES VISUALIZATION ==========")
    print("TASK   = task-0006")
    print("SAMPLE =", key)

    for oid in target_ids:
        print("OBJECT =", table[oid])

    print("FRAME  =", t)
    print("POINTS =", int(m.sum()))
    print("GT     = same-action same-time rigid-object GT")
    print("PRED   = PointWorld predicted rigid-object points")
    print("ALIGN  = scale + proper rotation + translation")
    print("DOF    = 7")
    print()
    print(f"MPJPE      = {mpjpe * 1000.0:.6f} mm")
    print(f"PA-MPJPE   = {pa * 1000.0:.6f} mm")
    print(f"fitted s   = {scale:.6f}")
    print(f"decomp err = {max_diff:.3e}")
    print("==============================================")

    # --------------------------------------------------------
    # Figure.
    # --------------------------------------------------------
    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(18, 5)
    )

    axes = [
        fig.add_subplot(
            1, 4, i + 1,
            projection="3d"
        )
        for i in range(4)
    ]

    draw_pair(
        axes[0],
        gt_t,
        pred_t,
        "(1) Before alignment",
    )

    draw_pair(
        axes[1],
        gt_t,
        pred_scale,
        f"(2) Scale  s={scale:.3f}",
    )

    draw_pair(
        axes[2],
        gt_t,
        pred_shift,
        "(3) Translation",
    )

    draw_pair(
        axes[3],
        gt_t,
        pred_rotate,
        (
            "(4) Rotation / Final\n"
            f"PA-MPJPE={pa * 1000.0:.2f} mm"
        ),
    )

    set_equal_limits(
        axes,
        [
            gt_t,
            pred_t,
            pred_scale,
            pred_shift,
            pred_rotate,
        ],
    )

    axes[0].legend(
        loc="best"
    )

    fig.suptitle(
        (
            "PointWorld rigid-object Procrustes alignment\n"
            f"task-0006 | {key} | {TARGET_TOKEN} | frame {t}"
        ),
        fontsize=13,
    )

    fig.tight_layout()

    safe_key = key.replace(
        ":", "-"
    )

    out_path = os.path.join(
        OUT_DIR,
        (
            f"{safe_key}__{TARGET_TOKEN}"
            f"__frame{t:02d}__procrustes.png"
        ),
    )

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("SAVED =", out_path)


if __name__ == "__main__":
    main()
