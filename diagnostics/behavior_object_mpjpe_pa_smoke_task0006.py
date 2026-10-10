#!/usr/bin/env python3

import numpy as np
import torch

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

DATA_DIR = "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006"
MODEL_PATH = "/home/soenchun/cse/PointWorld_workspace/PointWorld/pretrained_checkpoints/large-droid+behavior/model-best.pt"
NORM_STATS = "/home/soenchun/cse/PointWorld_workspace/PointWorld/stats/droid_behavior"

TARGET_TOKEN = "easter_egg_179"


# ------------------------------------------------------------
# Official VideoPose3D P-MPJPE algebra:
# scale + proper rotation + translation, then mean Euclidean L2.
# Here joints are replaced by corresponding rigid-object 3D points.
# ------------------------------------------------------------
def p_mpjpe_single(predicted, target):
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    muX = np.mean(target, axis=0, keepdims=True)
    muY = np.mean(predicted, axis=0, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2))
    normY = np.sqrt(np.sum(Y0 ** 2))

    if normX <= 0 or normY <= 0:
        raise RuntimeError("Degenerate point geometry for PA-MPJPE")

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

    error = np.mean(
        np.linalg.norm(aligned - target, axis=-1)
    )

    return float(error), float(scale)


def bool_mask(x):
    x = x.bool()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x


def main():

    # Runtime-only hooks: preserve official scene tensors exactly,
    # while carrying a parallel rigid-mesh ID array.
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

    context = bool_mask(batch["scene_context_mask"])
    exists = bool_mask(batch["scene_exists"])
    supervised = bool_mask(batch["scene_supervised_mask"])

    official_valid = (
        (~context)
        & exists
        & supervised
    )

    valid = (
        official_valid[0]
        & object_mask
    )

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

    valid_np = valid.detach().cpu().numpy()

    print()
    print("========== OBJECT MPJPE / PA-MPJPE SMOKE ==========")
    print("TASK   = task-0006")
    print("SAMPLE =", key)

    for oid in target_ids:
        print("OBJECT =", table[oid])

    print("FINAL OBJECT POINTS =", int(object_mask_np[0].sum()))
    print("METRIC PAIR = PointWorld Pred(t) vs same-action same-time GT(t)")
    print("PA ALIGNMENT = 3D scale + proper rotation + translation (7 DoF)")
    print()

    frame_mpjpe = []
    frame_pa = []

    for t in range(gt.shape[0]):
        m = valid_np[t]
        n = int(m.sum())

        if n < 3:
            print(
                f"t={t:02d} valid_object_points={n} SKIP"
            )
            continue

        pred_t = pred[t, m]
        gt_t = gt[t, m]

        mpjpe_m = np.mean(
            np.linalg.norm(
                pred_t - gt_t,
                axis=-1,
            )
        )

        pa_m, fitted_scale = p_mpjpe_single(
            pred_t,
            gt_t,
        )

        mpjpe_mm = mpjpe_m * 1000.0
        pa_mm = pa_m * 1000.0

        frame_mpjpe.append(mpjpe_mm)
        frame_pa.append(pa_mm)

        print(
            f"t={t:02d} "
            f"points={n:3d} "
            f"MPJPE={mpjpe_mm:.6f} mm "
            f"PA-MPJPE={pa_mm:.6f} mm "
            f"fitted_scale={fitted_scale:.6f}"
        )

    if not frame_pa:
        raise RuntimeError(
            "No eligible future object frame for PA-MPJPE."
        )

    print()
    print("---------- FUTURE-FRAME SUMMARY ----------")
    print(
        "MPJPE mean =",
        f"{np.mean(frame_mpjpe):.6f} mm",
    )
    print(
        "PA-MPJPE mean =",
        f"{np.mean(frame_pa):.6f} mm",
    )
    print(
        "PA-MPJPE median =",
        f"{np.median(frame_pa):.6f} mm",
    )
    print("====================================================")


if __name__ == "__main__":
    main()
