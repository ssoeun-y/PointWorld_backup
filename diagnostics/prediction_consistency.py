#!/usr/bin/env python3

import csv
import json
import runpy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from evaluation.tester import Tester


def prepare_mask(batch: dict[str, Any], name: str) -> torch.Tensor:
    mask = batch[name]
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")

    mask = mask.bool()
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)

    if mask.ndim != 3:
        raise ValueError(f"Unexpected {name} shape: {tuple(mask.shape)}")

    return mask


def masked_stats(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    selected = values[mask].detach().float().cpu()

    if selected.numel() == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(selected.numel()),
        "mean": float(selected.mean().item()),
        "median": float(selected.median().item()),
        "p95": float(torch.quantile(selected, 0.95).item()),
        "p99": float(torch.quantile(selected, 0.99).item()),
        "max": float(selected.max().item()),
    }


def safe_rate(count: int, total: int) -> float:
    return float(count / total) if total > 0 else float("nan")


@torch.no_grad()
def diagnostic_accumulate(
    self: Tester,
    dl,
    split: str,
    apply_confidence_mask: bool = False,
):
    """
    One-batch unfiltered Prediction Consistency smoke test.

    Official model, dataloader and loss_fn are reused unchanged.
    Expert confidence filtering is deliberately not injected in this first smoke.
    """

    self.model.eval()

    iterator = iter(dl)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(f"No samples found for split={split}") from exc

    batch = {
        key: value.to(self.device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }

    outputs = self.model(batch, training=False)
    _, official_loss_dict = self.model.loss_fn(outputs, batch, training=False)

    prediction = outputs["scene_flows"]
    gt = batch["gt_scene_flows"]

    if prediction.shape != gt.shape:
        raise RuntimeError(
            f"Prediction/GT shape mismatch: "
            f"prediction={tuple(prediction.shape)}, gt={tuple(gt.shape)}"
        )

    if prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise RuntimeError(
            f"Expected prediction shape (B,T,N,3), got {tuple(prediction.shape)}"
        )

    context = prepare_mask(batch, "scene_context_mask")
    exists = prepare_mask(batch, "scene_exists")
    supervised = prepare_mask(batch, "scene_supervised_mask")
    moved = prepare_mask(batch, "scene_moved_mask")
    static = prepare_mask(batch, "scene_static_mask")

    # Exact official basic evaluation mask.
    valid = (~context) & exists & supervised

    expected_mask_shape = prediction.shape[:3]
    if valid.shape != expected_mask_shape:
        raise RuntimeError(
            f"Mask/prediction mismatch: valid={tuple(valid.shape)}, "
            f"prediction={tuple(prediction.shape)}"
        )

    # ---------------------------------------------------------------
    # 1. Official L2 reproduction
    # ---------------------------------------------------------------
    l2 = torch.linalg.vector_norm(prediction - gt, dim=-1)

    if not valid.any():
        raise RuntimeError("Official valid mask contains zero points")

    recomputed_l2_mean = float(l2[valid].mean().item())

    if "l2/mean" not in official_loss_dict:
        raise KeyError(
            f"Official loss dictionary does not contain l2/mean. "
            f"Available keys: {sorted(official_loss_dict.keys())}"
        )

    official_l2_mean = float(official_loss_dict["l2/mean"])
    l2_abs_difference = abs(recomputed_l2_mean - official_l2_mean)
    l2_match = l2_abs_difference <= 1e-6

    if not l2_match:
        raise RuntimeError(
            "Official L2 reproduction failed: "
            f"official={official_l2_mean:.12f}, "
            f"recomputed={recomputed_l2_mean:.12f}, "
            f"difference={l2_abs_difference:.12e}"
        )

    # ---------------------------------------------------------------
    # 2. Velocity error
    # v(t) = position(t) - position(t-1)
    # ---------------------------------------------------------------
    pred_velocity = prediction[:, 1:] - prediction[:, :-1]
    gt_velocity = gt[:, 1:] - gt[:, :-1]

    velocity_valid = valid[:, 1:] & valid[:, :-1]

    velocity_error = torch.linalg.vector_norm(
        pred_velocity - gt_velocity,
        dim=-1,
    )

    # ---------------------------------------------------------------
    # 3. Acceleration error
    # a(t) = v(t) - v(t-1)
    # ---------------------------------------------------------------
    pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    gt_acceleration = gt_velocity[:, 1:] - gt_velocity[:, :-1]

    acceleration_valid = (
        valid[:, 2:]
        & valid[:, 1:-1]
        & valid[:, :-2]
    )

    acceleration_error = torch.linalg.vector_norm(
        pred_acceleration - gt_acceleration,
        dim=-1,
    )

    # ---------------------------------------------------------------
    # 4. Prediction error variation
    # |e(t) - e(t-1)|, where e(t) is pointwise L2
    # ---------------------------------------------------------------
    error_variation_valid = valid[:, 1:] & valid[:, :-1]
    error_variation = torch.abs(l2[:, 1:] - l2[:, :-1])

    # ---------------------------------------------------------------
    # 5. Trajectory jerk
    # j(t) = a(t) - a(t-1)
    # ---------------------------------------------------------------
    pred_jerk = pred_acceleration[:, 1:] - pred_acceleration[:, :-1]
    gt_jerk = gt_acceleration[:, 1:] - gt_acceleration[:, :-1]

    jerk_valid = (
        valid[:, 3:]
        & valid[:, 2:-1]
        & valid[:, 1:-2]
        & valid[:, :-3]
    )

    pred_jerk_magnitude = torch.linalg.vector_norm(pred_jerk, dim=-1)
    gt_jerk_magnitude = torch.linalg.vector_norm(gt_jerk, dim=-1)
    jerk_error = torch.linalg.vector_norm(pred_jerk - gt_jerk, dim=-1)

    # ---------------------------------------------------------------
    # 6. Direction reversal
    #
    # Consecutive velocity vectors are counted as reversing when:
    # dot(v[t-1], v[t]) < 0
    #
    # The small epsilon only excludes numerically zero velocity vectors.
    # It is not a failure threshold.
    # ---------------------------------------------------------------
    reversal_valid = velocity_valid[:, 1:] & velocity_valid[:, :-1]

    pred_velocity_previous = pred_velocity[:, :-1]
    pred_velocity_next = pred_velocity[:, 1:]
    gt_velocity_previous = gt_velocity[:, :-1]
    gt_velocity_next = gt_velocity[:, 1:]

    eps = torch.finfo(prediction.dtype).eps

    pred_previous_speed = torch.linalg.vector_norm(
        pred_velocity_previous, dim=-1
    )
    pred_next_speed = torch.linalg.vector_norm(
        pred_velocity_next, dim=-1
    )
    gt_previous_speed = torch.linalg.vector_norm(
        gt_velocity_previous, dim=-1
    )
    gt_next_speed = torch.linalg.vector_norm(
        gt_velocity_next, dim=-1
    )

    pred_reversal_defined = (
        reversal_valid
        & (pred_previous_speed > eps)
        & (pred_next_speed > eps)
    )
    gt_reversal_defined = (
        reversal_valid
        & (gt_previous_speed > eps)
        & (gt_next_speed > eps)
    )

    pred_direction_reversal = (
        (pred_velocity_previous * pred_velocity_next).sum(dim=-1) < 0
    )
    gt_direction_reversal = (
        (gt_velocity_previous * gt_velocity_next).sum(dim=-1) < 0
    )

    pred_reversal_count = int(
        (pred_direction_reversal & pred_reversal_defined).sum().item()
    )
    pred_reversal_total = int(pred_reversal_defined.sum().item())

    gt_reversal_count = int(
        (gt_direction_reversal & gt_reversal_defined).sum().item()
    )
    gt_reversal_total = int(gt_reversal_defined.sum().item())

    common_reversal_defined = (
        pred_reversal_defined & gt_reversal_defined
    )

    reversal_disagreement_count = int(
        (
            (pred_direction_reversal != gt_direction_reversal)
            & common_reversal_defined
        ).sum().item()
    )
    common_reversal_total = int(common_reversal_defined.sum().item())

    prediction_only_reversal_count = int(
        (
            pred_direction_reversal
            & (~gt_direction_reversal)
            & common_reversal_defined
        ).sum().item()
    )

    # ---------------------------------------------------------------
    # Per-timestep official L2
    # ---------------------------------------------------------------
    per_timestep_rows: list[dict[str, float | int]] = []

    batch_size, timesteps, num_points, _ = prediction.shape

    for timestep in range(timesteps):
        timestep_valid = valid[:, timestep]
        values = l2[:, timestep][timestep_valid].detach().float().cpu()

        row: dict[str, float | int] = {
            "timestep": timestep,
            "initial_point_count": int(batch_size * num_points),
            "valid_point_count": int(timestep_valid.sum().item()),
            "moved_valid_count": int(
                (timestep_valid & moved[:, timestep]).sum().item()
            ),
            "static_valid_count": int(
                (timestep_valid & static[:, timestep]).sum().item()
            ),
        }

        if values.numel() == 0:
            row.update({
                "l2_mean": float("nan"),
                "l2_median": float("nan"),
                "l2_p95": float("nan"),
            })
        else:
            row.update({
                "l2_mean": float(values.mean().item()),
                "l2_median": float(values.median().item()),
                "l2_p95": float(torch.quantile(values, 0.95).item()),
            })

        per_timestep_rows.append(row)

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        "diagnostics/results",
        f"prediction_consistency_smoke_{timestamp}",
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    keys_raw = batch.get("__key__", [])
    domains_raw = batch.get("__domain__", [])

    if isinstance(keys_raw, (list, tuple)):
        sample_keys = [str(value) for value in keys_raw]
    else:
        sample_keys = [str(keys_raw)]

    if isinstance(domains_raw, (list, tuple)):
        domains = [str(value) for value in domains_raw]
    else:
        domains = [str(domains_raw)]

    raw_data = {
        "prediction_scene_flows": prediction.detach().cpu(),
        "gt_scene_flows": gt.detach().cpu(),
        "official_valid_mask": valid.detach().cpu(),
        "scene_context_mask": context.detach().cpu(),
        "scene_exists": exists.detach().cpu(),
        "scene_supervised_mask": supervised.detach().cpu(),
        "scene_moved_mask": moved.detach().cpu(),
        "scene_static_mask": static.detach().cpu(),
        "sample_keys": sample_keys,
        "domains": domains,
    }

    if "scene_filter_mask" in batch:
        filter_mask = prepare_mask(batch, "scene_filter_mask")
        raw_data["scene_filter_mask"] = filter_mask.detach().cpu()

    torch.save(raw_data, output_dir / "raw_batch.pt")

    summary = {
        "analysis_type": (
            "Official tensors and masks with additional temporal diagnostics"
        ),
        "scope": "DROID one-batch unfiltered smoke test",
        "sample_keys": sample_keys,
        "domains": domains,
        "prediction_shape": list(prediction.shape),
        "gt_shape": list(gt.shape),
        "official_valid_count": int(valid.sum().item()),
        "official_l2_mean": official_l2_mean,
        "recomputed_l2_mean": recomputed_l2_mean,
        "l2_absolute_difference": l2_abs_difference,
        "official_l2_reproduced": l2_match,
        "velocity_error": masked_stats(
            velocity_error,
            velocity_valid,
        ),
        "acceleration_error": masked_stats(
            acceleration_error,
            acceleration_valid,
        ),
        "prediction_error_variation": masked_stats(
            error_variation,
            error_variation_valid,
        ),
        "prediction_jerk_magnitude": masked_stats(
            pred_jerk_magnitude,
            jerk_valid,
        ),
        "gt_jerk_magnitude": masked_stats(
            gt_jerk_magnitude,
            jerk_valid,
        ),
        "jerk_error": masked_stats(
            jerk_error,
            jerk_valid,
        ),
        "direction_reversal": {
            "definition": "dot(v[t-1], v[t]) < 0",
            "prediction_count": pred_reversal_count,
            "prediction_defined_count": pred_reversal_total,
            "prediction_rate": safe_rate(
                pred_reversal_count,
                pred_reversal_total,
            ),
            "gt_count": gt_reversal_count,
            "gt_defined_count": gt_reversal_total,
            "gt_rate": safe_rate(
                gt_reversal_count,
                gt_reversal_total,
            ),
            "common_defined_count": common_reversal_total,
            "prediction_gt_disagreement_count": (
                reversal_disagreement_count
            ),
            "prediction_gt_disagreement_rate": safe_rate(
                reversal_disagreement_count,
                common_reversal_total,
            ),
            "prediction_only_reversal_count": (
                prediction_only_reversal_count
            ),
        },
        "units": {
            "l2": "meter",
            "velocity_error": "meter/timestep",
            "acceleration_error": "meter/timestep^2",
            "jerk": "meter/timestep^3",
        },
    }

    with open(output_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    with open(output_dir / "per_timestep.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(per_timestep_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_timestep_rows)

    with open(output_dir / "run_info.txt", "w") as file:
        file.write(f"output_dir={output_dir}\n")
        file.write(f"split={split}\n")
        file.write(f"sample_keys={sample_keys}\n")
        file.write(f"prediction_shape={tuple(prediction.shape)}\n")
        file.write(f"official_l2_mean={official_l2_mean:.12f}\n")
        file.write(f"recomputed_l2_mean={recomputed_l2_mean:.12f}\n")
        file.write(f"l2_abs_difference={l2_abs_difference:.12e}\n")
        file.write(f"official_l2_reproduced={l2_match}\n")

    print()
    print("=" * 72)
    print("Prediction Consistency one-batch smoke completed")
    print(f"Output directory: {output_dir}")
    print(f"Prediction shape: {tuple(prediction.shape)}")
    print(f"Official L2:     {official_l2_mean:.12f}")
    print(f"Recomputed L2:   {recomputed_l2_mean:.12f}")
    print(f"Absolute diff:   {l2_abs_difference:.12e}")
    print(f"L2 reproduced:   {l2_match}")
    print("=" * 72)
    print()

    # run_evaluation expects a metric dictionary.
    return {
        f"diagnostic/{split}/official_l2_mean": official_l2_mean,
        f"diagnostic/{split}/recomputed_l2_mean": recomputed_l2_mean,
        f"diagnostic/{split}/l2_absolute_difference": l2_abs_difference,
    }


# Patch only in this diagnostics process.
# evaluation/tester.py itself is not modified.
Tester._accumulate_metrics = diagnostic_accumulate

# Execute the official eval.py argument parser and Tester initialization.
runpy.run_path("eval.py", run_name="__main__")
