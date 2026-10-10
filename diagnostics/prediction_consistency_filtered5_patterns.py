#!/usr/bin/env python3

import csv
import json
import math
import runpy
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from evaluation.tester import Tester


def mask3(batch, name):
    x = batch[name].bool()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.ndim != 3:
        raise RuntimeError(f"{name}: unexpected shape {tuple(x.shape)}")
    return x


def window_all(mask, width):
    return mask.unfold(1, width, 1).all(dim=-1)


def tensor_stats(x):
    x = x.detach().float().cpu().reshape(-1)

    if x.numel() == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p95": float(torch.quantile(x, 0.95)),
        "p99": float(torch.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def combined_stats(parts):
    valid_parts = [x.reshape(-1) for x in parts if x.numel() > 0]
    if not valid_parts:
        return tensor_stats(torch.empty(0))
    return tensor_stats(torch.cat(valid_parts))


def safe_rate(count, total):
    return float(count / total) if total > 0 else float("nan")


def compare_metric(actual, expected, name, tolerance=1e-6):
    actual = float(actual)
    expected = float(expected)

    if math.isnan(actual) and math.isnan(expected):
        return

    difference = abs(actual - expected)
    if difference > tolerance:
        raise RuntimeError(
            f"{name} reproduction failed: "
            f"official={expected}, recomputed={actual}, diff={difference}"
        )


def pattern_summary(error_sequences):
    """
    error_sequences: [P, 10]

    Threshold 없이 timestep 간 error 증가/감소 방향만 본다.
    """
    if error_sequences.numel() == 0:
        return {
            "trajectory_count": 0,
            "monotonic_worsening_rate": float("nan"),
            "monotonic_improving_rate": float("nan"),
            "mixed_rate": float("nan"),
            "net_worsening_rate": float("nan"),
            "net_improving_rate": float("nan"),
            "zero_sign_change_rate": float("nan"),
            "one_sign_change_rate": float("nan"),
            "repeated_oscillation_rate": float("nan"),
            "strong_alternation_rate": float("nan"),
            "sign_change_count": tensor_stats(torch.empty(0)),
            "total_variation": tensor_stats(torch.empty(0)),
            "excess_variation": tensor_stats(torch.empty(0)),
            "oscillation_fraction": tensor_stats(torch.empty(0)),
            "mean_error_by_timestep": [],
        }

    x = error_sequences.detach().float().cpu()
    delta = x[:, 1:] - x[:, :-1]

    positive = delta > 0
    negative = delta < 0
    zero = delta == 0

    monotonic_worsening = (~negative).all(dim=1) & positive.any(dim=1)
    monotonic_improving = (~positive).all(dim=1) & negative.any(dim=1)
    constant = zero.all(dim=1)
    mixed = ~(monotonic_worsening | monotonic_improving | constant)

    signs = torch.sign(delta)
    adjacent_defined = (
        (signs[:, 1:] != 0)
        & (signs[:, :-1] != 0)
    )
    sign_changes = (
        (signs[:, 1:] * signs[:, :-1] < 0)
        & adjacent_defined
    ).sum(dim=1)

    net_change = x[:, -1] - x[:, 0]
    total_variation = delta.abs().sum(dim=1)
    excess_variation = (
        total_variation - net_change.abs()
    ).clamp_min(0)

    oscillation_fraction = torch.zeros_like(total_variation)
    nonzero_tv = total_variation > 0
    oscillation_fraction[nonzero_tv] = (
        excess_variation[nonzero_tv]
        / total_variation[nonzero_tv]
    )

    count = x.shape[0]

    return {
        "trajectory_count": int(count),
        "monotonic_worsening_rate": safe_rate(
            int(monotonic_worsening.sum()), count
        ),
        "monotonic_improving_rate": safe_rate(
            int(monotonic_improving.sum()), count
        ),
        "constant_rate": safe_rate(int(constant.sum()), count),
        "mixed_rate": safe_rate(int(mixed.sum()), count),
        "net_worsening_rate": safe_rate(
            int((net_change > 0).sum()), count
        ),
        "net_improving_rate": safe_rate(
            int((net_change < 0).sum()), count
        ),
        "zero_sign_change_rate": safe_rate(
            int((sign_changes == 0).sum()), count
        ),
        "one_sign_change_rate": safe_rate(
            int((sign_changes == 1).sum()), count
        ),
        "repeated_oscillation_rate": safe_rate(
            int((sign_changes >= 2).sum()), count
        ),
        "strong_alternation_rate": safe_rate(
            int((sign_changes >= 4).sum()), count
        ),
        "sign_change_count": tensor_stats(sign_changes.float()),
        "total_variation": tensor_stats(total_variation),
        "excess_variation": tensor_stats(excess_variation),
        "oscillation_fraction": tensor_stats(oscillation_fraction),
        "mean_error_by_timestep": [
            float(v) for v in x.mean(dim=0)
        ],
    }


@torch.no_grad()
def analyze_filtered5(self, dl, split, apply_confidence_mask=False):
    if not apply_confidence_mask:
        raise RuntimeError(
            "Official confidence mask was not enabled. "
            "Stopping to prevent an invalid filtered analysis."
        )

    self.model.eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        "diagnostics/results",
        f"prediction_consistency_droid_filtered5_patterns_{timestamp}",
    )
    raw_dir = output_dir / "raw_sequences"
    raw_dir.mkdir(parents=True, exist_ok=False)

    metric_parts = defaultdict(list)
    pattern_parts = defaultdict(list)
    reversal = defaultdict(int)
    sequence_rows = []

    conf_files = self._confidence_helper.open_conf_files(
        split, mode="r"
    )

    processed = 0

    try:
        for batch in tqdm(dl, desc="Filtered-5 consistency"):
            batch = {
                key: value.to(self.device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }

            self._confidence_helper.inject_confidence_mask(
                batch, conf_files
            )

            outputs = self.model(batch, training=False)
            _, official_metrics = self.model.loss_fn(
                outputs, batch, training=False
            )

            pred = outputs["scene_flows"]
            gt = batch["gt_scene_flows"]

            if pred.shape != gt.shape:
                raise RuntimeError(
                    f"Prediction/GT mismatch: {pred.shape} vs {gt.shape}"
                )

            context = mask3(batch, "scene_context_mask")
            exists = mask3(batch, "scene_exists")
            supervised = mask3(batch, "scene_supervised_mask")
            moved = mask3(batch, "scene_moved_mask")
            static = mask3(batch, "scene_static_mask")
            filter_mask = mask3(batch, "scene_filter_mask")

            official_valid = (~context) & exists & supervised
            expert_valid = official_valid & filter_mask

            l2 = torch.linalg.vector_norm(pred - gt, dim=-1)

            # 공식 기본 L2 재현
            recomputed_l2 = float(l2[official_valid].mean())
            compare_metric(
                recomputed_l2,
                official_metrics["l2/mean"],
                "l2/mean",
            )

            # 공식 filtered L2 재현
            recomputed_filtered = float(l2[expert_valid].mean())
            compare_metric(
                recomputed_filtered,
                official_metrics["filtered_l2/mean"],
                "filtered_l2/mean",
            )

            if (expert_valid & moved).any():
                recomputed_filtered_moved = float(
                    l2[expert_valid & moved].mean()
                )
            else:
                recomputed_filtered_moved = float("nan")

            compare_metric(
                recomputed_filtered_moved,
                official_metrics["filtered_l2_moved/mean"],
                "filtered_l2_moved/mean",
            )

            pred_v = pred[:, 1:] - pred[:, :-1]
            gt_v = gt[:, 1:] - gt[:, :-1]
            velocity_error = torch.linalg.vector_norm(
                pred_v - gt_v, dim=-1
            )

            pred_a = pred_v[:, 1:] - pred_v[:, :-1]
            gt_a = gt_v[:, 1:] - gt_v[:, :-1]
            acceleration_error = torch.linalg.vector_norm(
                pred_a - gt_a, dim=-1
            )

            pred_j = pred_a[:, 1:] - pred_a[:, :-1]
            gt_j = gt_a[:, 1:] - gt_a[:, :-1]

            prediction_jerk = torch.linalg.vector_norm(
                pred_j, dim=-1
            )
            gt_jerk = torch.linalg.vector_norm(gt_j, dim=-1)
            jerk_error = torch.linalg.vector_norm(
                pred_j - gt_j, dim=-1
            )

            error_variation = torch.abs(
                l2[:, 1:] - l2[:, :-1]
            )

            regimes = {
                "official_valid": official_valid,
                "expert_filtered": expert_valid,
            }

            keys = [str(x) for x in batch["__key__"]]

            for regime_name, base_mask in regimes.items():
                moved_position_mask = base_mask & moved
                static_position_mask = base_mask & static

                metric_parts[
                    f"{regime_name}/l2_moved"
                ].append(l2[moved_position_mask].detach().cpu())
                metric_parts[
                    f"{regime_name}/l2_static"
                ].append(l2[static_position_mask].detach().cpu())

                moved2 = window_all(base_mask & moved, 2)
                static2 = window_all(base_mask & static, 2)

                moved3 = window_all(base_mask & moved, 3)
                static3 = window_all(base_mask & static, 3)

                moved4 = window_all(base_mask & moved, 4)
                static4 = window_all(base_mask & static, 4)

                for class_name, mask2, mask3_, mask4 in [
                    ("moved", moved2, moved3, moved4),
                    ("static", static2, static3, static4),
                ]:
                    metric_parts[
                        f"{regime_name}/velocity_error_{class_name}"
                    ].append(
                        velocity_error[mask2].detach().cpu()
                    )
                    metric_parts[
                        f"{regime_name}/error_variation_{class_name}"
                    ].append(
                        error_variation[mask2].detach().cpu()
                    )
                    metric_parts[
                        f"{regime_name}/acceleration_error_{class_name}"
                    ].append(
                        acceleration_error[mask3_].detach().cpu()
                    )
                    metric_parts[
                        f"{regime_name}/prediction_jerk_{class_name}"
                    ].append(
                        prediction_jerk[mask4].detach().cpu()
                    )
                    metric_parts[
                        f"{regime_name}/gt_jerk_{class_name}"
                    ].append(
                        gt_jerk[mask4].detach().cpu()
                    )
                    metric_parts[
                        f"{regime_name}/jerk_error_{class_name}"
                    ].append(
                        jerk_error[mask4].detach().cpu()
                    )

                # 미래 10개 timestep 모두 valid한 trajectory
                future_valid = base_mask[:, 1:].all(dim=1)
                future_moved = (
                    base_mask[:, 1:] & moved[:, 1:]
                ).all(dim=1)
                future_static = (
                    base_mask[:, 1:] & static[:, 1:]
                ).all(dim=1)

                future_errors = l2[:, 1:].permute(0, 2, 1)

                pattern_parts[
                    f"{regime_name}/all"
                ].append(
                    future_errors[future_valid].detach().cpu()
                )
                pattern_parts[
                    f"{regime_name}/moved"
                ].append(
                    future_errors[future_moved].detach().cpu()
                )
                pattern_parts[
                    f"{regime_name}/static"
                ].append(
                    future_errors[future_static].detach().cpu()
                )

                # Direction reversal
                pred_v_prev = pred_v[:, :-1]
                pred_v_next = pred_v[:, 1:]
                gt_v_prev = gt_v[:, :-1]
                gt_v_next = gt_v[:, 1:]

                eps = torch.finfo(pred.dtype).eps

                pred_defined = (
                    window_all(base_mask, 3)
                    & (
                        torch.linalg.vector_norm(
                            pred_v_prev, dim=-1
                        ) > eps
                    )
                    & (
                        torch.linalg.vector_norm(
                            pred_v_next, dim=-1
                        ) > eps
                    )
                )
                gt_defined = (
                    window_all(base_mask, 3)
                    & (
                        torch.linalg.vector_norm(
                            gt_v_prev, dim=-1
                        ) > eps
                    )
                    & (
                        torch.linalg.vector_norm(
                            gt_v_next, dim=-1
                        ) > eps
                    )
                )

                pred_reverse = (
                    (pred_v_prev * pred_v_next).sum(dim=-1) < 0
                )
                gt_reverse = (
                    (gt_v_prev * gt_v_next).sum(dim=-1) < 0
                )

                for class_name, class_mask in [
                    ("moved", window_all(moved, 3)),
                    ("static", window_all(static, 3)),
                ]:
                    pred_class = pred_defined & class_mask
                    gt_class = gt_defined & class_mask
                    common = pred_class & gt_class

                    prefix = f"{regime_name}/{class_name}"

                    reversal[f"{prefix}/pred_count"] += int(
                        (pred_reverse & pred_class).sum()
                    )
                    reversal[f"{prefix}/pred_total"] += int(
                        pred_class.sum()
                    )
                    reversal[f"{prefix}/gt_count"] += int(
                        (gt_reverse & gt_class).sum()
                    )
                    reversal[f"{prefix}/gt_total"] += int(
                        gt_class.sum()
                    )
                    reversal[f"{prefix}/disagreement"] += int(
                        (
                            (pred_reverse != gt_reverse)
                            & common
                        ).sum()
                    )
                    reversal[f"{prefix}/common"] += int(
                        common.sum()
                    )

            for b, key in enumerate(keys):
                sequence_rows.append({
                    "sequence_index": processed,
                    "key": key,
                    "official_valid_count": int(
                        official_valid[b].sum()
                    ),
                    "expert_filtered_count": int(
                        expert_valid[b].sum()
                    ),
                    "official_l2_mean": float(
                        l2[b][official_valid[b]].mean()
                    ),
                    "expert_filtered_l2_mean": float(
                        l2[b][expert_valid[b]].mean()
                    ),
                    "official_continuous_moved_count": int(
                        (
                            official_valid[b, 1:]
                            & moved[b, 1:]
                        ).all(dim=0).sum()
                    ),
                    "expert_continuous_moved_count": int(
                        (
                            expert_valid[b, 1:]
                            & moved[b, 1:]
                        ).all(dim=0).sum()
                    ),
                })

                torch.save(
                    {
                        "prediction_scene_flows": pred[b].detach().cpu(),
                        "gt_scene_flows": gt[b].detach().cpu(),
                        "official_valid_mask": (
                            official_valid[b].detach().cpu()
                        ),
                        "expert_filtered_mask": (
                            expert_valid[b].detach().cpu()
                        ),
                        "scene_filter_mask": (
                            filter_mask[b].detach().cpu()
                        ),
                        "scene_moved_mask": moved[b].detach().cpu(),
                        "scene_static_mask": static[b].detach().cpu(),
                        "key": key,
                    },
                    raw_dir / f"{processed:02d}.pt",
                )

                processed += 1

    finally:
        self._confidence_helper.close_conf_files(conf_files)

    if processed != 5:
        raise RuntimeError(
            f"Expected exactly 5 sequences, processed {processed}"
        )

    summary = {
        "scope": (
            "DROID confidence-covered 5-sample subset: "
            "official-valid vs expert-filtered"
        ),
        "processed_sequences": processed,
        "official_l2_reproduced_for_all_sequences": True,
        "official_filtered_l2_reproduced_for_all_sequences": True,
        "error_pattern_definition": (
            "No correctness threshold. Patterns use only the sign "
            "of consecutive pointwise L2 error changes."
        ),
        "metrics": {},
        "patterns": {},
        "direction_reversal": {},
    }

    for name, parts in metric_parts.items():
        summary["metrics"][name] = combined_stats(parts)

    for name, parts in pattern_parts.items():
        valid_parts = [
            x for x in parts
            if x.ndim == 2 and x.shape[0] > 0
        ]
        sequences = (
            torch.cat(valid_parts, dim=0)
            if valid_parts
            else torch.empty((0, 10))
        )
        summary["patterns"][name] = pattern_summary(sequences)

    for regime_name in [
        "official_valid",
        "expert_filtered",
    ]:
        for class_name in ["moved", "static"]:
            prefix = f"{regime_name}/{class_name}"

            summary["direction_reversal"][prefix] = {
                "prediction_count": reversal[
                    f"{prefix}/pred_count"
                ],
                "prediction_defined_count": reversal[
                    f"{prefix}/pred_total"
                ],
                "prediction_rate": safe_rate(
                    reversal[f"{prefix}/pred_count"],
                    reversal[f"{prefix}/pred_total"],
                ),
                "gt_count": reversal[f"{prefix}/gt_count"],
                "gt_defined_count": reversal[
                    f"{prefix}/gt_total"
                ],
                "gt_rate": safe_rate(
                    reversal[f"{prefix}/gt_count"],
                    reversal[f"{prefix}/gt_total"],
                ),
                "disagreement_count": reversal[
                    f"{prefix}/disagreement"
                ],
                "common_defined_count": reversal[
                    f"{prefix}/common"
                ],
                "disagreement_rate": safe_rate(
                    reversal[f"{prefix}/disagreement"],
                    reversal[f"{prefix}/common"],
                ),
            }

    with open(output_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    with open(
        output_dir / "per_sequence.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(sequence_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(sequence_rows)

    print()
    print("=" * 72)
    print("Filtered-5 Prediction Consistency completed")
    print(f"Output directory: {output_dir}")
    print(f"Processed sequences: {processed}")
    print("Official L2 reproduced: True")
    print("Official filtered L2 reproduced: True")
    print("=" * 72)

    return {
        f"diagnostic/{split}/processed_sequences": processed,
    }


Tester._accumulate_metrics = analyze_filtered5
runpy.run_path("eval.py", run_name="__main__")
