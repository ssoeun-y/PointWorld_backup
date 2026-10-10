#!/usr/bin/env python3

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler

import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pmod

from arguments import parse_args

import diagnostics.behavior_object_mapping_smoke_task0006 as omap

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

from diagnostics.audit_rigidity_loss_scale_v2 import (
    build_model,
    rigidity_loss,
)

from diagnostics.behavior_object_mpjpe_pa_smoke_task0006 import (
    p_mpjpe_single,
)


SOURCE_CKPT = Path(
    "pretrained_checkpoints/large-droid+behavior/model-best.pt"
)

NORM_STATS = "stats/droid_behavior"

TASK_DIRS = {
    "task-0000":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
    "task-0001":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0001",
    "task-0006":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006",
    "task-0008":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0008",
    "task-0010":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0010",
}

SEED = 24680
SPLIT_SEED = 24680

VAL_FRACTION = 0.20

BATCH_SIZE = 1
VAL_BATCHES = 1000

LR = 1.0e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 5.0

MIN_POINTS = 8


# ------------------------------------------------------------
# Preserve official loader functions.
# ------------------------------------------------------------

ORIG_GATHER_SHARDS = dlmod.gather_shard_paths
ORIG_AGGREGATE = dlmod.aggregate_dataset_metadata

SPLIT_MAP = None
CURRENT_SUBSET = None


def parse_local():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--lambda-rigid",
        type=float,
        required=True,
    )

    p.add_argument(
        "--steps",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--tag",
        type=str,
        required=True,
    )

    p.add_argument(
        "--eval-pre",
        action="store_true",
    )

    return p.parse_args()


def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# Fixed 80/20 TRAIN-SHARD holdout.
#
# IMPORTANT:
# Official test split is NEVER used here.
# ============================================================

def make_split_map():
    split = {
        "train": {},
        "val": {},
        "all": {},
    }

    manifest = {}

    for task_i, (task, data_dir) in enumerate(
        TASK_DIRS.items()
    ):
        shards = sorted(
            ORIG_GATHER_SHARDS(
                data_dir,
                "train",
                0,
            )
        )

        if len(shards) < 2:
            raise RuntimeError(
                f"{task}: need >=2 train shards, "
                f"got {len(shards)}"
            )

        rng = np.random.default_rng(
            SPLIT_SEED + task_i
        )

        order = rng.permutation(
            len(shards)
        )

        n_val = max(
            1,
            int(
                round(
                    VAL_FRACTION
                    * len(shards)
                )
            ),
        )

        n_val = min(
            n_val,
            len(shards) - 1,
        )

        val_idx = set(
            order[:n_val].tolist()
        )

        train_shards = [
            s
            for i, s in enumerate(shards)
            if i not in val_idx
        ]

        val_shards = [
            s
            for i, s in enumerate(shards)
            if i in val_idx
        ]

        split["train"][data_dir] = (
            train_shards
        )

        split["val"][data_dir] = (
            val_shards
        )

        split["all"][data_dir] = (
            shards
        )

        manifest[task] = {
            "data_dir":
                data_dir,

            "n_all":
                len(shards),

            "n_train":
                len(train_shards),

            "n_val":
                len(val_shards),

            "train_shards":
                [
                    Path(x).name
                    for x in train_shards
                ],

            "val_shards":
                [
                    Path(x).name
                    for x in val_shards
                ],
        }

    return split, manifest


def patched_gather_shards(
    data_dir,
    splits,
    rank,
):
    global CURRENT_SUBSET

    if (
        CURRENT_SUBSET in ("train", "val")
        and data_dir in SPLIT_MAP[
            CURRENT_SUBSET
        ]
        and splits == "train"
    ):
        return list(
            SPLIT_MAP[
                CURRENT_SUBSET
            ][data_dir]
        )

    return ORIG_GATHER_SHARDS(
        data_dir,
        splits,
        rank,
    )


def patched_aggregate(
    data_dir,
    split,
):
    global CURRENT_SUBSET

    if (
        CURRENT_SUBSET in ("train", "val")
        and data_dir in SPLIT_MAP[
            CURRENT_SUBSET
        ]
        and split == "train"
    ):
        full_count = int(
            ORIG_AGGREGATE(
                data_dir,
                "train",
            )
        )

        n_all = len(
            SPLIT_MAP["all"][data_dir]
        )

        n_selected = len(
            SPLIT_MAP[
                CURRENT_SUBSET
            ][data_dir]
        )

        # Metadata is split-level rather than shard-level.
        # This number controls WebLoader __len__ only.
        # Actual selected shards themselves are exact.
        estimated = int(
            round(
                full_count
                * n_selected
                / n_all
            )
        )

        return max(
            1,
            estimated,
        )

    return ORIG_AGGREGATE(
        data_dir,
        split,
    )


def make_loader_args():
    args = parse_args(
        skip_command_line=True
    )

    args.seed = SEED

    args.data_dirs = list(
        TASK_DIRS.values()
    )

    # Loader domain mapping:
    # five independent BEHAVIOR roots.
    args.domains = [
        "behavior"
        for _ in TASK_DIRS
    ]

    args.norm_stats_path = (
        NORM_STATS
    )

    args.batch_size = BATCH_SIZE

    # Exact label propagation smoke was
    # validated with deterministic train.
    args.deterministic_train = True
    args.deterministic_data = True

    args.num_workers = 0
    args.eval_num_workers = 0

    return args


def install_loader_patches():
    dlmod.build_flow_sample = (
        build_flow_sample_with_object_ids
    )

    pmod.grid_sample_transform = (
        grid_sample_with_object_ids_exact
    )

    pmod.gather = (
        gather_with_object_ids
    )

    dlmod.gather_shard_paths = (
        patched_gather_shards
    )

    dlmod.aggregate_dataset_metadata = (
        patched_aggregate
    )


def restore_loader_patches():
    dlmod.build_flow_sample = (
        ORIG_BUILD_FLOW_SAMPLE
    )

    pmod.grid_sample_transform = (
        ORIG_GRID_SAMPLE
    )

    pmod.gather = (
        ORIG_GATHER
    )

    dlmod.gather_shard_paths = (
        ORIG_GATHER_SHARDS
    )

    dlmod.aggregate_dataset_metadata = (
        ORIG_AGGREGATE
    )


def build_train_val_loaders():
    global CURRENT_SUBSET

    args = make_loader_args()

    install_loader_patches()

    CURRENT_SUBSET = "train"

    train_loader, train_info = (
        dlmod.build_dataloader(
            args=args,
            mode="train",
            rank=0,
            world_size=1,
            override_splits="train",
        )
    )

    CURRENT_SUBSET = "val"

    # Held-out TRAIN shards,
    # but evaluation-time deterministic pipeline.
    val_loader, val_info = (
        dlmod.build_dataloader(
            args=args,
            mode="test",
            rank=0,
            world_size=1,
            override_splits="train",
            force_resampled_eval=False,
        )
    )

    CURRENT_SUBSET = None

    if train_loader is None:
        raise RuntimeError(
            "train loader is None"
        )

    if val_loader is None:
        raise RuntimeError(
            "validation loader is None"
        )

    return (
        train_loader,
        train_info,
        val_loader,
        val_info,
    )


# ============================================================
# Validation MPJPE + PA-MPJPE
#
# Same conventions as current 7,947 evaluator:
# - official valid mask
# - moved mask only selects moving object
# - all valid selected-object points used
# - min 8 points
# - GT geometry rank >= 2
# - PA = similarity Procrustes
# ============================================================

def mask_np(x):
    a = (
        x.detach()
        .cpu()
        .numpy()[0]
    )

    if (
        a.ndim == 3
        and a.shape[-1] == 1
    ):
        a = a[..., 0]

    return a.astype(bool)


@torch.no_grad()
def evaluate_validation(
    model,
    loader,
    max_batches,
):
    model.eval()

    clip_mp = []
    clip_pa = []

    seen = set()
    scanned = 0

    for batch_cpu in loader:
        if scanned >= max_batches:
            break

        scanned += 1

        key = omap.normalize_key(
            batch_cpu["__key__"]
        )

        if key in seen:
            raise RuntimeError(
                f"duplicate validation key: {key}"
            )

        seen.add(key)

        if key not in omap.OBJECT_TABLE:
            raise RuntimeError(
                f"missing OBJECT_TABLE: {key}"
            )

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
            in batch_cpu.items()
        }

        outputs = model(
            batch,
            training=False,
        )

        pred = (
            outputs["scene_flows"][0]
            .detach()
            .cpu()
            .numpy()
        )

        gt = (
            batch[
                "gt_scene_flows"
            ][0]
            .detach()
            .cpu()
            .numpy()
        )

        labels = (
            batch[
                "scene_object_id"
            ][0]
            .detach()
            .cpu()
            .numpy()
        )

        context = mask_np(
            batch[
                "scene_context_mask"
            ]
        )

        exists = mask_np(
            batch[
                "scene_exists"
            ]
        )

        supervised = mask_np(
            batch[
                "scene_supervised_mask"
            ]
        )

        moved = mask_np(
            batch[
                "scene_moved_mask"
            ]
        )

        valid = (
            (~context)
            & exists
            & supervised
        )

        table = (
            omap.OBJECT_TABLE[
                key
            ]
        )

        object_mp = []
        object_pa = []

        for oid, _mesh_name in (
            table.items()
        ):
            oid = int(oid)

            object_mask = (
                labels == oid
            )

            moved_support = int(
                (
                    object_mask
                    & moved
                    & valid
                ).sum()
            )

            if moved_support <= 0:
                continue

            frame_mp = []
            frame_pa = []

            T = min(
                pred.shape[0],
                gt.shape[0],
                object_mask.shape[0],
                valid.shape[0],
            )

            for t in range(T):
                m = (
                    object_mask[t]
                    & valid[t]
                )

                n = int(
                    m.sum()
                )

                if n < MIN_POINTS:
                    continue

                pred_t = pred[
                    t,
                    m,
                ]

                gt_t = gt[
                    t,
                    m,
                ]

                centered_gt = (
                    gt_t
                    - gt_t.mean(
                        axis=0,
                        keepdims=True,
                    )
                )

                if int(
                    np.linalg.matrix_rank(
                        centered_gt,
                        tol=1e-8,
                    )
                ) < 2:
                    continue

                mp = float(
                    np.mean(
                        np.linalg.norm(
                            pred_t
                            - gt_t,
                            axis=-1,
                        )
                    )
                    * 1000.0
                )

                try:
                    pa, _scale = (
                        p_mpjpe_single(
                            pred_t,
                            gt_t,
                        )
                    )

                except Exception:
                    continue

                frame_mp.append(
                    mp
                )

                frame_pa.append(
                    float(pa)
                    * 1000.0
                )

            if frame_mp:
                object_mp.append(
                    float(
                        np.mean(
                            frame_mp
                        )
                    )
                )

                object_pa.append(
                    float(
                        np.mean(
                            frame_pa
                        )
                    )
                )

        if object_mp:
            clip_mp.append(
                float(
                    np.mean(
                        object_mp
                    )
                )
            )

            clip_pa.append(
                float(
                    np.mean(
                        object_pa
                    )
                )
            )

    if not clip_mp:
        raise RuntimeError(
            "No eligible validation clips"
        )

    result = {
        "scanned_batches":
            int(scanned),

        "eligible_clips":
            int(
                len(
                    clip_mp
                )
            ),

        "mpjpe_median_mm":
            float(
                np.median(
                    clip_mp
                )
            ),

        "mpjpe_mean_mm":
            float(
                np.mean(
                    clip_mp
                )
            ),

        "pa_mpjpe_median_mm":
            float(
                np.median(
                    clip_pa
                )
            ),

        "pa_mpjpe_mean_mm":
            float(
                np.mean(
                    clip_pa
                )
            ),
    }

    return result


def main():
    global SPLIT_MAP

    cli = parse_local()

    if cli.steps <= 0:
        raise RuntimeError(
            "--steps must be >0"
        )

    reset_seed()

    root = Path(
        "diagnostics/results/"
        "rigid_generation_ablation"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dir = (
        root
        / cli.tag
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_ckpt = (
        run_dir
        / "model-last.pt"
    )

    if out_ckpt.exists():
        raise RuntimeError(
            f"Refusing to overwrite "
            f"existing checkpoint: "
            f"{out_ckpt}"
        )

    SPLIT_MAP, manifest = (
        make_split_map()
    )

    shared_manifest = (
        root
        / (
            "split_manifest_"
            f"seed{SPLIT_SEED}.json"
        )
    )

    manifest_text = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )

    if shared_manifest.exists():
        existing = (
            shared_manifest
            .read_text()
        )

        if (
            json.loads(existing)
            != manifest
        ):
            raise RuntimeError(
                "Split manifest changed!"
            )

    else:
        shared_manifest.write_text(
            manifest_text
        )

    (
        train_loader,
        train_info,
        val_loader,
        val_info,
    ) = build_train_val_loaders()

    first = next(
        iter(
            train_loader
        )
    )

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
            list(
                first.keys()
            ),
    }

    model = build_model(
        data_info
    )

    # --------------------------------------------------------
    # Controlled fine-tuning:
    # freeze pretrained representation,
    # train ONLY future dynamics head.
    #
    # This matches the head-gradient scale used to derive lambda.
    # --------------------------------------------------------

    for p in model.parameters():
        p.requires_grad = False

    head = (
        model
        .dynamics_predictor
        .dynamics_head
    )

    for p in head.parameters():
        p.requires_grad = True

    head_params = [
        p
        for p in head.parameters()
        if p.requires_grad
    ]

    if not head_params:
        raise RuntimeError(
            "No trainable dynamics-head params"
        )

    optimizer = (
        torch.optim.AdamW(
            head_params,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
    )

    amp_dtype = (
        torch.bfloat16
        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        else torch.float16
    )

    scaler = GradScaler(
        "cuda"
    )

    source_stat = (
        SOURCE_CKPT.stat()
    )

    print("=" * 100)
    print("RIGID GENERATION ABLATION")
    print("=" * 100)

    print(
        "tag =",
        cli.tag,
    )

    print(
        "lambda =",
        cli.lambda_rigid,
    )

    print(
        "steps =",
        cli.steps,
    )

    print(
        "trainable = dynamics_head ONLY"
    )

    print(
        "lr =",
        LR,
    )

    print(
        "train shards / val shards:"
    )

    for task, x in manifest.items():
        print(
            task,
            x["n_train"],
            "/",
            x["n_val"],
        )

    # --------------------------------------------------------
    # PRETRAINED validation only once for lambda=0 run.
    # --------------------------------------------------------

    before = None

    if cli.eval_pre:
        print()
        print(
            "===== PRETRAINED VALIDATION ====="
        )

        before = (
            evaluate_validation(
                model,
                val_loader,
                VAL_BATCHES,
            )
        )

        print(
            json.dumps(
                before,
                indent=2,
            )
        )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.eval()
    head.train()

    train_iter = iter(
        train_loader
    )

    rigid_active = 0

    history = []

    for step in range(
        1,
        cli.steps + 1,
    ):
        try:
            batch = next(
                train_iter
            )

        except StopIteration:
            train_iter = iter(
                train_loader
            )

            batch = next(
                train_iter
            )

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

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
        ):
            outputs = model(
                batch,
                training=True,
            )

            if outputs.get(
                "_has_nan_outputs",
                False,
            ):
                raise RuntimeError(
                    f"NaN output at step {step}"
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
                pair_mm,
                nobj,
                nframe,
            ) = rigidity_loss(
                outputs,
                batch,
            )

            total = (
                base_loss
                + cli.lambda_rigid
                * rigid
            )

        if not torch.isfinite(
            total
        ):
            raise RuntimeError(
                f"non-finite loss "
                f"at step {step}"
            )

        scaler.scale(
            total
        ).backward()

        scaler.unscale_(
            optimizer
        )

        grad = (
            torch.nn.utils
            .clip_grad_norm_(
                head_params,
                max_norm=GRAD_CLIP,
            )
        )

        if not torch.isfinite(
            grad
        ):
            raise RuntimeError(
                f"non-finite grad "
                f"at step {step}"
            )

        scaler.step(
            optimizer
        )

        scaler.update()

        if nobj > 0:
            rigid_active += 1

        row = {
            "step":
                step,

            "base":
                float(
                    base_loss
                    .detach()
                    .float()
                    .cpu()
                ),

            "rigid":
                float(
                    rigid
                    .detach()
                    .float()
                    .cpu()
                ),

            "total":
                float(
                    total
                    .detach()
                    .float()
                    .cpu()
                ),

            "pair_mm":
                float(
                    pair_mm
                    .detach()
                    .float()
                    .cpu()
                ),

            "grad":
                float(
                    grad
                    .detach()
                    .float()
                    .cpu()
                ),

            "objects":
                int(nobj),

            "frames":
                int(nframe),
        }

        history.append(
            row
        )

        if (
            step == 1
            or step % 50 == 0
            or step == cli.steps
        ):
            print(
                f"[{step:05d}/{cli.steps}] "
                f"base={row['base']:+.6f} "
                f"rigid={row['rigid']:.8f} "
                f"pair={row['pair_mm']:.3f}mm "
                f"grad={row['grad']:.3f} "
                f"obj={nobj}"
            )

    # --------------------------------------------------------
    # VALIDATION
    # Rebuild iterator from SAME fixed held-out shards.
    # --------------------------------------------------------

    print()
    print(
        "===== POST-FINETUNE VALIDATION ====="
    )

    after = (
        evaluate_validation(
            model,
            val_loader,
            VAL_BATCHES,
        )
    )

    print(
        json.dumps(
            after,
            indent=2,
        )
    )

    summary = {
        "tag":
            cli.tag,

        "lambda_rigid":
            cli.lambda_rigid,

        "steps":
            cli.steps,

        "lr":
            LR,

        "weight_decay":
            WEIGHT_DECAY,

        "grad_clip":
            GRAD_CLIP,

        "batch_size":
            BATCH_SIZE,

        "trainable":
            "dynamics_head_only",

        "rigid_active_steps":
            rigid_active,

        "pretrained_validation":
            before,

        "post_validation":
            after,
    }

    (
        run_dir
        / "val_metrics.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    (
        run_dir
        / "train_history.json"
    ).write_text(
        json.dumps(
            history,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Save NEW inference checkpoint.
    # Never overwrite source.
    # --------------------------------------------------------

    source = torch.load(
        SOURCE_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    out = {
        "model": {
            k:
                v.detach().cpu()
            for k, v
            in model.state_dict().items()
        },

        "optimizer":
            optimizer.state_dict(),

        "args":
            copy.deepcopy(
                source["args"]
            ),

        "exp_name":
            cli.tag,

        "wandb_id":
            None,

        "batch_count":
            cli.steps,

        "sample_count":
            cli.steps
            * BATCH_SIZE,

        "epoch_count":
            float(
                cli.steps
                * BATCH_SIZE
            )
            / float(
                max(
                    1,
                    train_info[
                        "total_samples"
                    ],
                )
            ),

        "checkpoint_schema_version":
            copy.deepcopy(
                source[
                    "checkpoint_schema_version"
                ]
            ),

        "model_contract":
            copy.deepcopy(
                source[
                    "model_contract"
                ]
            ),

        "data_contract":
            copy.deepcopy(
                source[
                    "data_contract"
                ]
            ),

        "rigid_finetune": {
            "source_checkpoint":
                str(
                    SOURCE_CKPT
                ),

            "lambda_rigid":
                cli.lambda_rigid,

            "steps":
                cli.steps,

            "split_seed":
                SPLIT_SEED,

            "val_fraction":
                VAL_FRACTION,

            "trainable":
                "dynamics_head_only",

            "validation":
                after,
        },
    }

    torch.save(
        out,
        out_ckpt,
    )

    # Exact reload verification.
    saved = torch.load(
        out_ckpt,
        map_location="cpu",
        weights_only=False,
    )

    for k, v in (
        model.state_dict()
        .items()
    ):
        if not torch.equal(
            saved["model"][k],
            v.detach().cpu(),
        ):
            raise RuntimeError(
                f"reload mismatch: {k}"
            )

    source_after = (
        SOURCE_CKPT.stat()
    )

    if (
        source_stat.st_size
        != source_after.st_size
        or
        source_stat.st_mtime_ns
        != source_after.st_mtime_ns
    ):
        raise RuntimeError(
            "SOURCE CHECKPOINT MODIFIED"
        )

    print()
    print(
        "CHECKPOINT RELOAD EXACT = True"
    )

    print(
        "SOURCE CHECKPOINT UNCHANGED = True"
    )

    print(
        "OUTPUT =",
        out_ckpt,
    )

    print(
        "ABLATION RUN PASS = True"
    )

    restore_loader_patches()


if __name__ == "__main__":
    main()
