#!/usr/bin/env python3

import numpy as np
import torch

from diagnostics.audit_rigidity_loss_scale_v2 import (
    build_loader,
    build_model,
    rigidity_loss,
)

SEED = 24680
MAX_BATCHES = 50


def grad_norm(grads):
    sq = None

    for g in grads:
        if g is None:
            continue

        x = g.float().pow(2).sum()

        sq = x if sq is None else sq + x

    if sq is None:
        return 0.0

    return float(torch.sqrt(sq).detach().cpu())


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    loader, info = build_loader()
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

    model = build_model(data_info)
    model.train()

    dyn_params = list(
        model.dynamics_predictor
        .dynamics_head.parameters()
    )

    rows = []

    batches = [first]

    for _ in range(MAX_BATCHES - 1):
        batches.append(next(iterator))

    print("=" * 90)
    print("RIGIDITY GRADIENT SCALE AUDIT")
    print("=" * 90)
    print("NO optimizer")
    print("NO optimizer.step()")
    print("NO checkpoint write")
    print("batches =", MAX_BATCHES)

    for i, batch in enumerate(batches, 1):

        batch = {
            k: (
                v.cuda(non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

        # Full precision intentionally:
        # this is a gradient-scale audit.
        outputs = model(
            batch,
            training=True,
        )

        base_loss, _ = model.loss_fn(
            outputs,
            batch,
            training=True,
        )

        rigid_loss, abs_mm, nobj, nframe = (
            rigidity_loss(
                outputs,
                batch,
            )
        )

        pred_norm = outputs[
            "scene_relative_norm"
        ]

        # ----------------------------------------------------
        # A valid train sample can contain no eligible moved
        # rigid object. In that case rigidity_loss() correctly
        # returns zero. This batch provides no information for
        # rigidity-gradient scaling, so exclude it from the
        # scale estimate rather than treating it as a failure.
        # ----------------------------------------------------
        if nobj == 0 or nframe == 0:
            print(
                f"[{i:02d}/{MAX_BATCHES}] "
                f"SKIP no eligible rigid object "
                f"(objects={nobj}, frames={nframe})"
            )
            continue

        # ----------------------------------------------------
        # 1. Output-space gradient:
        # direct pressure on predicted future displacement.
        # ----------------------------------------------------
        g_base_out = torch.autograd.grad(
            base_loss,
            pred_norm,
            retain_graph=True,
            allow_unused=False,
        )[0]

        g_rigid_out = torch.autograd.grad(
            rigid_loss,
            pred_norm,
            retain_graph=True,
            allow_unused=False,
        )[0]

        base_out_norm = grad_norm(
            [g_base_out]
        )

        rigid_out_norm = grad_norm(
            [g_rigid_out]
        )

        # ----------------------------------------------------
        # 2. Dynamics-head parameter gradient:
        # direct generation head only.
        # Excludes uncertainty-head parameters.
        # ----------------------------------------------------
        g_base_head = torch.autograd.grad(
            base_loss,
            dyn_params,
            retain_graph=True,
            allow_unused=True,
        )

        g_rigid_head = torch.autograd.grad(
            rigid_loss,
            dyn_params,
            retain_graph=False,
            allow_unused=True,
        )

        base_head_norm = grad_norm(
            g_base_head
        )

        rigid_head_norm = grad_norm(
            g_rigid_head
        )

        if rigid_out_norm <= 0:
            raise RuntimeError(
                f"batch {i}: rigid output grad is zero"
            )

        if rigid_head_norm <= 0:
            raise RuntimeError(
                f"batch {i}: rigid head grad is zero"
            )

        lambda_equal_out = (
            base_out_norm
            / rigid_out_norm
        )

        lambda_equal_head = (
            base_head_norm
            / rigid_head_norm
        )

        row = {
            "base_loss":
                float(
                    base_loss.detach().cpu()
                ),

            "rigid_loss":
                float(
                    rigid_loss.detach().cpu()
                ),

            "pairdist_mm":
                float(
                    abs_mm.detach().cpu()
                ),

            "base_out_grad":
                base_out_norm,

            "rigid_out_grad":
                rigid_out_norm,

            "lambda_equal_out":
                lambda_equal_out,

            "base_head_grad":
                base_head_norm,

            "rigid_head_grad":
                rigid_head_norm,

            "lambda_equal_head":
                lambda_equal_head,
        }

        rows.append(row)

        print(
            f"[{i:02d}/{MAX_BATCHES}] "
            f"base={row['base_loss']:.6f} "
            f"rigid={row['rigid_loss']:.8f} "
            f"pair={row['pairdist_mm']:.3f}mm | "
            f"out λeq={lambda_equal_out:.4g} | "
            f"head λeq={lambda_equal_head:.4g} | "
            f"obj={nobj} frames={nframe}"
        )

        del outputs
        del base_loss
        del rigid_loss

    if len(rows) == 0:
        raise RuntimeError(
            "No eligible rigid-object batches found "
            f"among {MAX_BATCHES} scanned batches"
        )

    print()
    print(
        "valid rigidity batches =",
        len(rows),
        "/ scanned =",
        MAX_BATCHES,
    )

    def arr(k):
        return np.asarray(
            [r[k] for r in rows],
            dtype=np.float64,
        )

    out_eq = arr("lambda_equal_out")
    head_eq = arr("lambda_equal_head")

    print()
    print("=" * 90)
    print("FINAL")
    print("=" * 90)

    print(
        "output λ_equal median =",
        float(np.median(out_eq)),
    )

    print(
        "output λ_equal p25/p75 =",
        float(np.quantile(out_eq, 0.25)),
        float(np.quantile(out_eq, 0.75)),
    )

    print(
        "head λ_equal median =",
        float(np.median(head_eq)),
    )

    print(
        "head λ_equal p25/p75 =",
        float(np.quantile(head_eq, 0.25)),
        float(np.quantile(head_eq, 0.75)),
    )

    print()
    print(
        "candidate 10% head-gradient λ =",
        float(
            0.10 * np.median(head_eq)
        ),
    )

    print(
        "candidate 25% head-gradient λ =",
        float(
            0.25 * np.median(head_eq)
        ),
    )

    print(
        "candidate 50% head-gradient λ =",
        float(
            0.50 * np.median(head_eq)
        ),
    )

    print()
    print(
        "GRADIENT SCALE AUDIT PASS = True"
    )


if __name__ == "__main__":
    main()
