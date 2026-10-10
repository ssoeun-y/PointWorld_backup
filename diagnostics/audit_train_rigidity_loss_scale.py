#!/usr/bin/env python3

import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["PW_TARGET_TASKS"] = "task-0000,task-0001,task-0006,task-0008,task-0010"
os.environ["PW_TAG"] = "rigidity_loss_scale"
os.environ["PW_CONDITION"] = "original"

import numpy as np
import copy
import torch
import torch.nn.functional as F

import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pmod

from arguments import parse_args

from diagnostics.behavior_object_mapping_smoke_task0006 import (
    ORIG_BUILD_FLOW_SAMPLE,
    ORIG_GRID_SAMPLE,
    ORIG_GATHER,
    build_flow_sample_with_object_ids,
    grid_sample_with_object_ids,
    gather_with_object_ids,
)

MIN_POINTS = 8
MAX_POINTS = 64


def rigid_pairwise_loss(outputs, batch):
    """
    GT-future-free rigidity constraint.

    Initial observed shape:
        batch["scene_flows"][:, 0]

    Prediction:
        outputs["scene_flows"]

    For each moved object, preserve internal pairwise distances.
    Translation / rotation are free.
    """

    pred = outputs["scene_flows"].float()
    x0 = batch["scene_flows"][:, 0].float()

    labels = batch["scene_object_id"].long()

    context = batch["scene_context_mask"].squeeze(-1).bool()
    exists = batch["scene_exists"].squeeze(-1).bool()
    supervised = batch["scene_supervised_mask"].squeeze(-1).bool()
    moved = batch["scene_moved_mask"].squeeze(-1).bool()

    valid = (~context) & exists & supervised

    B, T, N, _ = pred.shape

    T_use = min(
        T,
        labels.shape[1],
        valid.shape[1],
        moved.shape[1],
    )

    losses = []
    objects_used = 0
    frames_used = 0

    for b in range(B):

        object_ids = torch.unique(labels[b, 0])

        for oid in object_ids:

            object_mask = labels[b, :T_use] == oid

            moved_support = (
                object_mask
                & moved[b, :T_use]
                & valid[b, :T_use]
            ).sum()

            # Same rule as current PA/MPJPE evaluator:
            # only objects that actually move.
            if int(moved_support.item()) <= 0:
                continue

            object_had_frame = False

            for t in range(T_use):

                m = (
                    object_mask[t]
                    & valid[b, t]
                    & (labels[b, 0] == oid)
                    & exists[b, 0]
                )

                idx = torch.where(m)[0]

                if idx.numel() < MIN_POINTS:
                    continue

                # Deterministic cap to avoid O(N^2) explosion.
                if idx.numel() > MAX_POINTS:
                    pos = torch.linspace(
                        0,
                        idx.numel() - 1,
                        MAX_POINTS,
                        device=idx.device,
                    ).round().long()

                    idx = idx[pos]

                ref = x0[b, idx]
                prd = pred[b, t, idx]

                # Need at least non-collinear geometry.
                centered = ref - ref.mean(dim=0, keepdim=True)

                if int(torch.linalg.matrix_rank(centered).item()) < 2:
                    continue

                # Pairwise distances are invariant to translation + rotation.
                d0 = torch.pdist(ref, p=2).detach()
                dp = torch.pdist(prd, p=2)

                # Distance-preservation error in meters.
                loss = F.smooth_l1_loss(
                    dp,
                    d0,
                    beta=0.001,
                    reduction="mean",
                )

                losses.append(loss)
                frames_used += 1
                object_had_frame = True

            if object_had_frame:
                objects_used += 1

    if not losses:
        return pred.sum() * 0.0, 0, 0

    return (
        torch.stack(losses).mean(),
        objects_used,
        frames_used,
    )


def main():

    args = parse_args()

    # ------------------------------------------------------------
    # IMPORTANT:
    # Dataset mixing and model normalization domains are different.
    #
    # Loader:
    #   five BEHAVIOR task WDS roots, all carrying domain="behavior"
    #
    # Model:
    #   must retain the checkpoint's original DROID+BEHAVIOR
    #   normalization contract.
    # ------------------------------------------------------------

    loader_args = copy.deepcopy(args)

    loader_args.data_dirs = [
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0001",
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006",
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0008",
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0010",
    ]

    loader_args.domains = [
        "behavior",
        "behavior",
        "behavior",
        "behavior",
        "behavior",
    ]

    loader_args.norm_stats_path = "stats/droid_behavior"
    loader_args.batch_size = 1
    loader_args.num_workers = 0
    loader_args.deterministic_train = True

    # Parallel object-ID propagation only.
    dlmod.build_flow_sample = build_flow_sample_with_object_ids
    pmod.grid_sample_transform = grid_sample_with_object_ids
    pmod.gather = gather_with_object_ids

    loader, info = dlmod.build_dataloader(
        args=loader_args,
        mode="train",
        rank=0,
        world_size=1,
        override_splits="train",
    )

    first = next(iter(loader))

    data_info = {
        "robot_features_dim":
            first["robot_features"].shape[-1],
        "scene_features_dim":
            first["scene_features"].shape[-1],
        "data_keys":
            list(first.keys()),
    }

    # The pretrained checkpoint was trained with two normalization
    # domains. Keep its canonical data contract for model creation.
    ckpt = torch.load(
        args.model_path,
        map_location="cpu",
        weights_only=False,
    )

    if "data_contract" not in ckpt:
        raise RuntimeError(
            "checkpoint missing canonical data_contract"
        )

    ckpt_domains = list(
        ckpt["data_contract"]["domains"]
    )

    print("CHECKPOINT DOMAINS =", ckpt_domains)

    if "behavior" not in ckpt_domains:
        raise RuntimeError(
            f"behavior missing from checkpoint domains: {ckpt_domains}"
        )

    args.domains = ckpt_domains
    args.norm_stats_path = "stats/droid_behavior"

    # Import only after loader patch.
    from training.trainer import Trainer

    trainer = Trainer(
        args,
        inference_only=True,
        data_info_dict=data_info,
    )

    model = trainer.model
    model.eval()

    device = trainer.device

    base_vals = []
    rigid_vals = []
    ratios = []

    object_count = 0
    frame_count = 0

    MAX_BATCHES = 32

    print("=" * 100)
    print("RIGIDITY LOSS SCALE AUDIT")
    print("=" * 100)
    print("checkpoint =", args.model_path)
    print("max batches =", MAX_BATCHES)
    print("MIN_POINTS =", MIN_POINTS)
    print("MAX_POINTS =", MAX_POINTS)

    with torch.no_grad():

        for i, batch in enumerate(loader):

            if i >= MAX_BATCHES:
                break

            batch = {
                k: (
                    v.to(device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=trainer.amp_dtype,
            ):
                outputs = model(
                    batch,
                    training=False,
                )

                base_loss, _ = model.loss_fn(
                    outputs,
                    batch,
                    training=False,
                )

                rigid_loss, no, nf = (
                    rigid_pairwise_loss(
                        outputs,
                        batch,
                    )
                )

            b = float(base_loss.detach().cpu())
            r = float(rigid_loss.detach().cpu())

            base_vals.append(b)
            rigid_vals.append(r)

            if b > 0:
                ratios.append(r / b)

            object_count += no
            frame_count += nf

            print(
                f"[{i+1:02d}/{MAX_BATCHES}] "
                f"base={b:.8f} "
                f"rigid={r:.8f} "
                f"ratio={r/b if b>0 else float('nan'):.8f} "
                f"objects={no} frames={nf}"
            )

    base = np.asarray(base_vals)
    rigid = np.asarray(rigid_vals)
    ratio = np.asarray(ratios)

    print()
    print("=" * 100)
    print("RESULT")
    print("=" * 100)

    print(
        "base loss median =",
        float(np.median(base)),
    )

    print(
        "rigid loss median =",
        float(np.median(rigid)),
    )

    print(
        "rigid/base median ratio =",
        float(np.median(ratio)),
    )

    print(
        "rigid/base p25/p75 =",
        float(np.quantile(ratio, 0.25)),
        float(np.quantile(ratio, 0.75)),
    )

    print(
        "objects used =",
        object_count,
    )

    print(
        "object-frames used =",
        frame_count,
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
