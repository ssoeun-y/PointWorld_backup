#!/usr/bin/env python3

import copy
import numpy as np
import torch
import torch.nn.functional as F

import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pmod

from arguments import parse_args
from pointworld.base import BaseModel
from pointworld.checkpoint_contract import (
    read_checkpoint_contract,
    apply_model_contract_to_args,
)

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


CKPT = (
    "pretrained_checkpoints/"
    "large-droid+behavior/model-best.pt"
)

NORM_STATS = "stats/droid_behavior"

DATA_DIRS = [
    "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
    "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0001",
    "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006",
    "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0008",
    "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0010",
]

SEED = 24680
MAX_BATCHES = 50
MIN_POINTS = 8
MAX_POINTS = 64


def rigidity_loss(outputs, batch):
    """
    Differentiable rigid-shape consistency loss.

    Reference:
        observed initial object geometry only.

    Target:
        predicted future geometry.

    No future GT coordinates are used as the target.
    scene_moved_mask is used only during TRAINING to select
    manipulated/moving object meshes, matching the established
    rigid-object evaluation population logic.
    """

    pred = outputs["scene_flows"].float()
    x0 = batch["scene_flows"][:, 0].float()

    labels = batch["scene_object_id"].long()

    context = (
        batch["scene_context_mask"]
        .squeeze(-1)
        .bool()
    )

    exists = (
        batch["scene_exists"]
        .squeeze(-1)
        .bool()
    )

    supervised = (
        batch["scene_supervised_mask"]
        .squeeze(-1)
        .bool()
    )

    moved = (
        batch["scene_moved_mask"]
        .squeeze(-1)
        .bool()
    )

    valid_future = (
        (~context)
        & exists
        & supervised
    )

    B, T, N, _ = pred.shape

    if labels.shape != (B, T, N):
        raise RuntimeError(
            f"label shape {labels.shape} "
            f"!= expected {(B,T,N)}"
        )

    losses = []
    abs_mm = []

    objects_used = 0
    frames_used = 0

    for b in range(B):

        ids = torch.unique(labels[b, 0])

        for oid in ids:

            object_all_t = (
                labels[b] == oid
            )

            # Exact established selection logic:
            # moved mask only decides whether this object moved.
            moved_support = (
                object_all_t
                & moved[b]
                & valid_future[b]
            ).sum()

            if int(moved_support.item()) <= 0:
                continue

            used_this_object = False

            for t in range(1, T):

                # Same physical object points available
                # initially and valid at this future frame.
                m = (
                    (labels[b, 0] == oid)
                    & (labels[b, t] == oid)
                    & exists[b, 0]
                    & valid_future[b, t]
                )

                idx = torch.where(m)[0]

                if idx.numel() < MIN_POINTS:
                    continue

                # Deterministic point cap for O(N^2) pdist.
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

                centered = (
                    ref
                    - ref.mean(
                        dim=0,
                        keepdim=True,
                    )
                )

                if int(
                    torch.linalg.matrix_rank(
                        centered
                    ).item()
                ) < 2:
                    continue

                d0 = torch.pdist(
                    ref,
                    p=2,
                ).detach()

                dp = torch.pdist(
                    prd,
                    p=2,
                )

                if d0.numel() == 0:
                    continue

                # Physical rigid-body prior:
                # internal distances should stay constant.
                #
                # beta = 1 mm, coordinates are meters.
                l = F.smooth_l1_loss(
                    dp,
                    d0,
                    beta=0.001,
                    reduction="mean",
                )

                losses.append(l)

                abs_mm.append(
                    (
                        (dp - d0)
                        .abs()
                        .mean()
                        * 1000.0
                    )
                )

                frames_used += 1
                used_this_object = True

            if used_this_object:
                objects_used += 1

    if not losses:
        zero = pred.sum() * 0.0
        return zero, zero, 0, 0

    return (
        torch.stack(losses).mean(),
        torch.stack(abs_mm).mean(),
        objects_used,
        frames_used,
    )


def build_loader():
    args = parse_args(
        skip_command_line=True
    )

    args.seed = SEED
    args.data_dirs = list(DATA_DIRS)

    # Correct for the LOADER:
    # five BEHAVIOR datasets.
    args.domains = [
        "behavior",
        "behavior",
        "behavior",
        "behavior",
        "behavior",
    ]

    args.batch_size = 1
    args.num_workers = 0

    args.deterministic_train = True
    args.deterministic_data = True

    args.norm_stats_path = NORM_STATS

    dlmod.build_flow_sample = (
        build_flow_sample_with_object_ids
    )

    pmod.grid_sample_transform = (
        grid_sample_with_object_ids_exact
    )

    pmod.gather = (
        gather_with_object_ids
    )

    try:
        loader, info = (
            dlmod.build_dataloader(
                args=args,
                mode="train",
                rank=0,
                world_size=1,
                override_splits="train",
            )
        )

    except Exception:
        dlmod.build_flow_sample = (
            ORIG_BUILD_FLOW_SAMPLE
        )
        pmod.grid_sample_transform = (
            ORIG_GRID_SAMPLE
        )
        pmod.gather = ORIG_GATHER
        raise

    return loader, info


def build_model(data_info):
    ckpt = torch.load(
        CKPT,
        map_location="cpu",
        weights_only=False,
    )

    model_contract, data_contract = (
        read_checkpoint_contract(
            ckpt,
            context=CKPT,
        )
    )

    args = parse_args(
        skip_command_line=True
    )

    args.device = "cuda"
    args.distributed = False
    args.norm_stats_path = NORM_STATS
    args.model_path = CKPT

    # Model architecture:
    # exactly canonical checkpoint contract.
    apply_model_contract_to_args(
        args,
        model_contract,
        context=CKPT,
        explicit_cli_dests=set(),
    )

    # Model normalization:
    # exactly canonical checkpoint data contract.
    args.domains = list(
        data_contract["domains"]
    )

    print(
        "MODEL DOMAINS =",
        args.domains,
    )

    model = BaseModel(
        args,
        data_info,
        rank=0,
        cpu_pg=None,
    )

    state = ckpt["model"]

    expected = set(
        model.state_dict().keys()
    )

    actual = set(
        state.keys()
    )

    if expected != actual:
        missing = sorted(
            expected - actual
        )
        extra = sorted(
            actual - expected
        )

        raise RuntimeError(
            "checkpoint/model key mismatch\n"
            f"missing={missing[:10]}\n"
            f"extra={extra[:10]}"
        )

    # Fail BEFORE load_state_dict if any shape differs.
    for k, v in model.state_dict().items():
        if tuple(v.shape) != tuple(
            state[k].shape
        ):
            raise RuntimeError(
                f"shape mismatch {k}: "
                f"model={tuple(v.shape)} "
                f"ckpt={tuple(state[k].shape)}"
            )

    model.load_state_dict(
        state,
        strict=True,
    )

    model = model.cuda()

    print(
        "CHECKPOINT LOAD EXACT = True"
    )

    print(
        "MODEL norm buffer =",
        tuple(
            model
            .norm_stats_per_step_mean
            .shape
        ),
    )

    return model


def main():

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    loader, info = build_loader()

    iterator = iter(loader)

    first = next(iterator)

    data_info = {
        "robot_features_dim":
            int(
                first[
                    "robot_features"
                ].shape[-1]
            ),

        "scene_features_dim":
            int(
                first[
                    "scene_features"
                ].shape[-1]
            ),

        "data_keys":
            list(first.keys()),
    }

    print(
        "DATA INFO =",
        data_info[
            "robot_features_dim"
        ],
        data_info[
            "scene_features_dim"
        ],
    )

    model = build_model(
        data_info
    )

    # Audit actual TRAIN-mode scale.
    model.train()

    if (
        torch.cuda.is_available()
        and
        torch.cuda.is_bf16_supported()
    ):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    batches = [first]

    for _ in range(
        MAX_BATCHES - 1
    ):
        batches.append(
            next(iterator)
        )

    base_values = []
    rigid_values = []
    abs_mm_values = []
    ratios = []

    total_objects = 0
    total_frames = 0

    print()
    print("=" * 80)
    print("LOSS SCALE AUDIT")
    print("=" * 80)

    with torch.no_grad():

        for i, batch in enumerate(
            batches,
            start=1,
        ):

            batch = {
                k: (
                    v.cuda(
                        non_blocking=True
                    )
                    if isinstance(
                        v,
                        torch.Tensor,
                    )
                    else v
                )
                for k, v
                in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
            ):

                outputs = model(
                    batch,
                    training=True,
                )

                base_loss, _ = (
                    model.loss_fn(
                        outputs,
                        batch,
                        training=True,
                    )
                )

                (
                    rigid,
                    abs_mm,
                    nobj,
                    nframe,
                ) = rigidity_loss(
                    outputs,
                    batch,
                )

            b = float(
                base_loss
                .detach()
                .float()
                .cpu()
            )

            r = float(
                rigid
                .detach()
                .float()
                .cpu()
            )

            a = float(
                abs_mm
                .detach()
                .float()
                .cpu()
            )

            if not np.isfinite(b):
                raise RuntimeError(
                    f"non-finite base loss "
                    f"at batch {i}"
                )

            if not np.isfinite(r):
                raise RuntimeError(
                    f"non-finite rigid loss "
                    f"at batch {i}"
                )

            base_values.append(b)
            rigid_values.append(r)
            abs_mm_values.append(a)

            if b > 0:
                ratios.append(r / b)

            total_objects += nobj
            total_frames += nframe

            print(
                f"[{i:02d}/{MAX_BATCHES}] "
                f"domain={batch['__domain__']} "
                f"base={b:.8e} "
                f"rigid={r:.8e} "
                f"pairdist_abs={a:.4f}mm "
                f"ratio={r/b if b>0 else float('nan'):.8e} "
                f"objects={nobj} "
                f"frames={nframe}"
            )

    base = np.asarray(
        base_values,
        dtype=np.float64,
    )

    rigid = np.asarray(
        rigid_values,
        dtype=np.float64,
    )

    abs_mm = np.asarray(
        abs_mm_values,
        dtype=np.float64,
    )

    ratio = np.asarray(
        ratios,
        dtype=np.float64,
    )

    print()
    print("=" * 80)
    print("FINAL")
    print("=" * 80)

    print(
        "base median =",
        float(np.median(base)),
    )

    print(
        "rigid median =",
        float(np.median(rigid)),
    )

    print(
        "rigid/base median =",
        float(np.median(ratio)),
    )

    print(
        "rigid/base p25 =",
        float(
            np.quantile(
                ratio,
                0.25,
            )
        ),
    )

    print(
        "rigid/base p75 =",
        float(
            np.quantile(
                ratio,
                0.75,
            )
        ),
    )

    print(
        "pair-distance error "
        "median batch mean =",
        float(
            np.median(
                abs_mm
            )
        ),
        "mm",
    )

    print(
        "objects used =",
        total_objects,
    )

    print(
        "object-frames used =",
        total_frames,
    )

    print(
        "LOSS SCALE AUDIT PASS = True"
    )

    # Restore imported globals.
    dlmod.build_flow_sample = (
        ORIG_BUILD_FLOW_SAMPLE
    )
    pmod.grid_sample_transform = (
        ORIG_GRID_SAMPLE
    )
    pmod.gather = ORIG_GATHER


if __name__ == "__main__":
    main()
