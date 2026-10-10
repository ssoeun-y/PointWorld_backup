#!/usr/bin/env python3

from pathlib import Path
import os
import copy
import numpy as np
import torch
from torch.amp import GradScaler

from diagnostics.audit_rigidity_loss_scale_v2 import (
    build_loader,
    build_model,
    rigidity_loss,
)

# ------------------------------------------------------------
# Fixed smoke configuration
# ------------------------------------------------------------

SOURCE_CKPT = Path(
    "pretrained_checkpoints/large-droid+behavior/model-best.pt"
)

OUT_DIR = Path(
    "diagnostics/results/"
    "rigid_generation_finetune/"
    "smoke_lambda3435"
)

OUT_CKPT = OUT_DIR / "model-smoke.pt"

SEED = 24680

# Gradient audit:
# ~10% of original dynamics-head gradient.
LAMBDA_RIGID = 3435.5363005687427

# Official fixed-commit base LR / source checkpoint base_lr.
LR = 1.0e-4

WEIGHT_DECAY = 0.01
GRAD_CLIP = 5.0

MAX_STEPS = 20


def check_finite_model(model, context):
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise RuntimeError(
                f"{context}: non-finite parameter {name}"
            )


def main():

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not SOURCE_CKPT.exists():
        raise RuntimeError(
            f"missing source checkpoint: {SOURCE_CKPT}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUT_CKPT.resolve() == SOURCE_CKPT.resolve():
        raise RuntimeError(
            "OUTPUT CHECKPOINT MUST NOT BE SOURCE CHECKPOINT"
        )

    # Verify source checkpoint is never modified.
    source_stat_before = SOURCE_CKPT.stat()

    # --------------------------------------------------------
    # Exact loader already validated against official train
    # output (bitwise-identical scene tensors + parallel IDs).
    # --------------------------------------------------------
    loader, loader_info = build_loader()

    iterator = iter(loader)
    first = next(iterator)

    data_info = {
        "robot_features_dim":
            int(first["robot_features"].shape[-1]),

        "scene_features_dim":
            int(first["scene_features"].shape[-1]),

        "data_keys":
            list(first.keys()),
    }

    # --------------------------------------------------------
    # Exact pretrained model construction + strict checkpoint
    # load. build_model() already fail-fast checks every shape.
    # --------------------------------------------------------
    model = build_model(
        data_info
    )

    model.train()

    check_finite_model(
        model,
        "before training",
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # NEW optimizer.
    #
    # We DO NOT restore source checkpoint optimizer state.
    # This matches fixed commit Trainer._create_optimizer():
    # one AdamW over every trainable parameter.
    # --------------------------------------------------------
    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    if len(optimizer.param_groups) != 1:
        raise RuntimeError(
            "expected exactly one new optimizer param group"
        )

    print("=" * 100)
    print("RIGID GENERATION FINE-TUNING SMOKE")
    print("=" * 100)

    print(
        "source checkpoint =",
        SOURCE_CKPT,
    )

    print(
        "output checkpoint =",
        OUT_CKPT,
    )

    print(
        "lambda_rigid =",
        LAMBDA_RIGID,
    )

    print(
        "lr =",
        LR,
    )

    print(
        "weight_decay =",
        WEIGHT_DECAY,
    )

    print(
        "grad_clip =",
        GRAD_CLIP,
    )

    print(
        "max_steps =",
        MAX_STEPS,
    )

    print(
        "optimizer groups =",
        len(optimizer.param_groups),
    )

    print(
        "trainable parameters =",
        sum(
            p.numel()
            for p in params
        ),
    )

    # A fixed generation-head tensor used only to verify
    # that training actually changed model weights.
    probe = (
        model
        .dynamics_predictor
        .dynamics_head[-1]
        .weight
    )

    probe_before = (
        probe.detach()
        .float()
        .clone()
        .cpu()
    )

    if (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    # Same training machinery as official Trainer.
    scaler = GradScaler("cuda")

    # Include the batch already consumed to infer dimensions.
    batches = [first]

    for _ in range(
        MAX_STEPS - 1
    ):
        batches.append(
            next(iterator)
        )

    rigid_active_steps = 0

    base_history = []
    rigid_history = []
    total_history = []
    grad_history = []

    for step, batch in enumerate(
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
            for k, v in batch.items()
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
                    f"step {step}: "
                    "model reported NaN outputs"
                )

            base_loss, base_dict = (
                model.loss_fn(
                    outputs,
                    batch,
                    training=True,
                )
            )

            (
                rigid_loss,
                pairdist_mm,
                nobj,
                nframe,
            ) = rigidity_loss(
                outputs,
                batch,
            )

            weighted_rigid = (
                LAMBDA_RIGID
                * rigid_loss
            )

            total_loss = (
                base_loss
                + weighted_rigid
            )

        if not torch.isfinite(
            base_loss
        ):
            raise RuntimeError(
                f"step {step}: "
                "non-finite base loss"
            )

        if not torch.isfinite(
            rigid_loss
        ):
            raise RuntimeError(
                f"step {step}: "
                "non-finite rigid loss"
            )

        if not torch.isfinite(
            total_loss
        ):
            raise RuntimeError(
                f"step {step}: "
                "non-finite total loss"
            )

        # Official fixed-commit ordering:
        # scale -> backward -> unscale -> clip -> step.
        scaler.scale(
            total_loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP,
            )
        )

        if not torch.isfinite(
            grad_norm
        ):
            raise RuntimeError(
                f"step {step}: "
                f"non-finite grad norm "
                f"{grad_norm}"
            )

        scaler.step(
            optimizer
        )

        scaler.update()

        check_finite_model(
            model,
            f"after optimizer step {step}",
        )

        b = float(
            base_loss.detach()
            .float().cpu()
        )

        r = float(
            rigid_loss.detach()
            .float().cpu()
        )

        wr = float(
            weighted_rigid.detach()
            .float().cpu()
        )

        t = float(
            total_loss.detach()
            .float().cpu()
        )

        pd = float(
            pairdist_mm.detach()
            .float().cpu()
        )

        gn = float(
            grad_norm.detach()
            .float().cpu()
        )

        base_history.append(b)
        rigid_history.append(r)
        total_history.append(t)
        grad_history.append(gn)

        if nobj > 0:
            rigid_active_steps += 1

        print(
            f"[{step:02d}/{MAX_STEPS}] "
            f"base={b:+.6f} "
            f"rigid={r:.8f} "
            f"lambda*rigid={wr:.6f} "
            f"total={t:+.6f} "
            f"pair={pd:.3f}mm "
            f"objects={nobj} "
            f"frames={nframe} "
            f"grad={gn:.4f}"
        )

    # --------------------------------------------------------
    # Verify actual weight change.
    # --------------------------------------------------------

    probe_after = (
        probe.detach()
        .float()
        .cpu()
    )

    probe_change = (
        probe_after
        - probe_before
    ).abs()

    max_probe_change = float(
        probe_change.max()
    )

    mean_probe_change = float(
        probe_change.mean()
    )

    if max_probe_change <= 0.0:
        raise RuntimeError(
            "dynamics head did not change"
        )

    print()
    print("=" * 100)
    print("20-STEP RESULT")
    print("=" * 100)

    print(
        "rigid-active steps =",
        rigid_active_steps,
        "/",
        MAX_STEPS,
    )

    print(
        "base median =",
        float(
            np.median(
                base_history
            )
        ),
    )

    print(
        "rigid median =",
        float(
            np.median(
                rigid_history
            )
        ),
    )

    print(
        "total median =",
        float(
            np.median(
                total_history
            )
        ),
    )

    print(
        "grad median =",
        float(
            np.median(
                grad_history
            )
        ),
    )

    print(
        "dynamics-head max weight change =",
        max_probe_change,
    )

    print(
        "dynamics-head mean weight change =",
        mean_probe_change,
    )

    # --------------------------------------------------------
    # Save NEW inference-compatible checkpoint.
    #
    # Start from source metadata so canonical model/data
    # contracts are retained exactly, but replace weights and
    # training state with this new fine-tuning run.
    # --------------------------------------------------------

    source = torch.load(
        SOURCE_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    save_args = copy.deepcopy(
        source["args"]
    )

    # Record the optimizer setting actually used here.
    save_args.base_lr = LR
    save_args.weight_decay = WEIGHT_DECAY
    save_args.grad_clip_max_norm = GRAD_CLIP

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
            save_args,

        "exp_name":
            "rigid_generation_smoke_lambda3435",

        "wandb_id":
            None,

        # NEW counters: this is not a resume of the original
        # 324.8-epoch optimizer state.
        "batch_count":
            MAX_STEPS,

        "sample_count":
            MAX_STEPS,

        "epoch_count":
            MAX_STEPS
            / float(
                loader_info[
                    "total_samples"
                ]
            ),

        # Preserve canonical contracts exactly.
        "checkpoint_schema_version":
            source[
                "checkpoint_schema_version"
            ],

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

        # Method metadata.
        "rigid_finetune": {
            "source_checkpoint":
                str(SOURCE_CKPT),

            "lambda_rigid":
                LAMBDA_RIGID,

            "lr":
                LR,

            "weight_decay":
                WEIGHT_DECAY,

            "grad_clip":
                GRAD_CLIP,

            "steps":
                MAX_STEPS,

            "seed":
                SEED,

            "rigid_active_steps":
                rigid_active_steps,

            "max_probe_change":
                max_probe_change,
        },
    }

    torch.save(
        out,
        OUT_CKPT,
    )

    # --------------------------------------------------------
    # Reload verification.
    # --------------------------------------------------------

    saved = torch.load(
        OUT_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    if set(
        saved["model"].keys()
    ) != set(
        model.state_dict().keys()
    ):
        raise RuntimeError(
            "saved checkpoint key mismatch"
        )

    for k, v in model.state_dict().items():

        sv = saved[
            "model"
        ][k]

        if tuple(
            sv.shape
        ) != tuple(
            v.shape
        ):
            raise RuntimeError(
                f"saved shape mismatch: {k}"
            )

        if not torch.equal(
            sv,
            v.detach().cpu(),
        ):
            raise RuntimeError(
                f"saved tensor mismatch: {k}"
            )

    print(
        "SAVED CHECKPOINT RELOAD EXACT = True"
    )

    # --------------------------------------------------------
    # Final safety check:
    # source checkpoint metadata must be unchanged.
    # --------------------------------------------------------

    source_stat_after = (
        SOURCE_CKPT.stat()
    )

    if (
        source_stat_before.st_size
        != source_stat_after.st_size
        or
        source_stat_before.st_mtime_ns
        != source_stat_after.st_mtime_ns
    ):
        raise RuntimeError(
            "SOURCE CHECKPOINT WAS MODIFIED"
        )

    print(
        "SOURCE CHECKPOINT UNCHANGED = True"
    )

    print(
        "OUTPUT =",
        OUT_CKPT,
    )

    print(
        "RIGID FINETUNE SMOKE PASS = True"
    )


if __name__ == "__main__":
    main()
