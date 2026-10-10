#!/usr/bin/env python3

import random
import numpy as np
import torch

import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pmod

from arguments import parse_args

from diagnostics.behavior_object_mapping_smoke_task0006 import (
    ORIG_BUILD_FLOW_SAMPLE,
    ORIG_GRID_SAMPLE,
    ORIG_GATHER,
    build_flow_sample_with_object_ids,
    gather_with_object_ids,
)

from diagnostics.train_object_id_exact import (
    grid_sample_with_object_ids_exact,
)


SEED = 24680

DATA_DIR = (
    "/home/soenchun/cse/PointWorld_workspace/"
    "datasets/behavior/wds_task0006"
)


def reset_rng():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_args():
    args = parse_args(skip_command_line=True)

    args.seed = SEED
    args.data_dirs = [DATA_DIR]
    args.domains = ["behavior"]

    args.norm_stats_path = "stats/droid_behavior"

    args.batch_size = 1
    args.num_workers = 0

    # Official deterministic-train branch.
    args.deterministic_train = True
    args.deterministic_data = True

    return args


def build_one(args):
    loader, info = dlmod.build_dataloader(
        args=args,
        mode="train",
        rank=0,
        world_size=1,
        override_splits="train",
    )

    if loader is None:
        raise RuntimeError("loader is None")

    return next(iter(loader)), info


def assert_tensor_equal(name, a, b):
    if a.shape != b.shape:
        raise RuntimeError(
            f"{name}: shape mismatch "
            f"{tuple(a.shape)} != {tuple(b.shape)}"
        )

    if not torch.equal(a.cpu(), b.cpu()):
        diff = (
            a.float().cpu()
            - b.float().cpu()
        ).abs().max().item()

        raise RuntimeError(
            f"{name}: NOT identical, max_abs={diff}"
        )


def main():

    args = make_args()

    # =========================================================
    # 1. Completely official baseline loader
    # =========================================================
    dlmod.build_flow_sample = ORIG_BUILD_FLOW_SAMPLE
    pmod.grid_sample_transform = ORIG_GRID_SAMPLE
    pmod.gather = ORIG_GATHER

    reset_rng()

    baseline, info0 = build_one(args)

    # =========================================================
    # 2. Same loader + parallel object IDs
    # =========================================================
    dlmod.build_flow_sample = build_flow_sample_with_object_ids
    pmod.grid_sample_transform = grid_sample_with_object_ids_exact
    pmod.gather = gather_with_object_ids

    reset_rng()

    mapped, info1 = build_one(args)

    # Restore globals immediately.
    dlmod.build_flow_sample = ORIG_BUILD_FLOW_SAMPLE
    pmod.grid_sample_transform = ORIG_GRID_SAMPLE
    pmod.gather = ORIG_GATHER

    print("BASE INFO =", info0)
    print("MAP  INFO =", info1)

    if baseline["__key__"] != mapped["__key__"]:
        raise RuntimeError(
            f"sample key mismatch: "
            f"{baseline['__key__']} vs {mapped['__key__']}"
        )

    if baseline["__domain__"] != mapped["__domain__"]:
        raise RuntimeError("domain mismatch")

    # Every official tensor must remain EXACTLY identical.
    checked = 0

    for key, value in baseline.items():

        if isinstance(value, torch.Tensor):

            if key not in mapped:
                raise RuntimeError(
                    f"mapped batch missing official key {key}"
                )

            assert_tensor_equal(
                key,
                value,
                mapped[key],
            )

            checked += 1

    if "scene_object_id" not in mapped:
        raise RuntimeError(
            "scene_object_id missing"
        )

    labels = mapped["scene_object_id"]
    scene = mapped["scene_flows"]

    if labels.shape != scene.shape[:3]:
        raise RuntimeError(
            f"object-id shape mismatch: "
            f"{labels.shape} vs {scene.shape}"
        )

    if not torch.equal(
        labels[:, 0].cpu(),
        labels[:, -1].cpu(),
    ):
        raise RuntimeError(
            "object identity changed across time"
        )

    print()
    print("sample =", mapped["__key__"])
    print("official tensors checked =", checked)
    print("scene_flows =", tuple(scene.shape))
    print("scene_object_id =", tuple(labels.shape))
    print(
        "unique object IDs =",
        torch.unique(labels).cpu().tolist(),
    )

    print()
    print(
        "OFFICIAL TRAIN OUTPUT BITWISE IDENTICAL = True"
    )
    print(
        "TRAIN OBJECT-ID EXACT SMOKE PASS = True"
    )


if __name__ == "__main__":
    main()
