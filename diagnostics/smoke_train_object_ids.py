#!/usr/bin/env python3

import os
import torch

os.environ["PW_TARGET_TASKS"] = "task-0006"
os.environ["PW_TAG"] = "train_object_id_smoke"
os.environ["PW_CONDITION"] = "original"

import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pmod

from diagnostics.behavior_object_mapping_smoke_task0006 import (
    ORIG_BUILD_FLOW_SAMPLE,
    ORIG_GRID_SAMPLE,
    ORIG_GATHER,
    build_flow_sample_with_object_ids,
    grid_sample_with_object_ids,
    gather_with_object_ids,
)

from arguments import parse_args


def main():

    args = parse_args()

    # Exact smoke configuration.
    args.data_dirs = [
        "/home/soenchun/cse/PointWorld_workspace/"
        "datasets/behavior/wds_task0006"
    ]
    args.domains = ["behavior"]

    # Keep monkey-patched loader in this process.
    args.deterministic_train = True
    args.num_workers = 0
    args.batch_size = 1

    dlmod.build_flow_sample = build_flow_sample_with_object_ids
    pmod.grid_sample_transform = grid_sample_with_object_ids
    pmod.gather = gather_with_object_ids

    try:
        loader, info = dlmod.build_dataloader(
            args=args,
            mode="train",
            rank=0,
            world_size=1,
            override_splits="train",
        )

        if loader is None:
            raise RuntimeError("train loader is None")

        batch = next(iter(loader))

    finally:
        dlmod.build_flow_sample = ORIG_BUILD_FLOW_SAMPLE
        pmod.grid_sample_transform = ORIG_GRID_SAMPLE
        pmod.gather = ORIG_GATHER

    print("INFO =", info)
    print("BATCH KEYS =", sorted(batch.keys()))

    if "scene_object_id" not in batch:
        raise RuntimeError(
            "scene_object_id missing from TRAIN batch"
        )

    labels = batch["scene_object_id"]
    scene = batch["scene_flows"]

    print("scene_flows shape     =", tuple(scene.shape))
    print("scene_object_id shape =", tuple(labels.shape))

    # Expected collated form:
    # scene_flows:     (B,T,N,3)
    # scene_object_id: (B,T,N)
    if labels.shape != scene.shape[:3]:
        raise RuntimeError(
            f"shape mismatch: labels={labels.shape}, "
            f"scene={scene.shape}"
        )

    temporal_exact = torch.equal(
        labels[:, 0],
        labels[:, -1],
    )

    print(
        "temporal object-ID exact =",
        temporal_exact
    )

    print(
        "unique object IDs =",
        torch.unique(labels).numel()
    )

    print(
        "first IDs =",
        torch.unique(labels)[:20].cpu().tolist()
    )

    if not temporal_exact:
        raise RuntimeError(
            "object identity changed across timesteps"
        )

    print("TRAIN OBJECT-ID SMOKE PASS = True")


if __name__ == "__main__":
    main()
