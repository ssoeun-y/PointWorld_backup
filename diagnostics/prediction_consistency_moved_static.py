#!/usr/bin/env python3

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


INPUT_DIR = Path(
    "diagnostics/results/"
    "prediction_consistency_droid_unfiltered_full_20260804_135952"
)
RAW_DIR = INPUT_DIR / "raw_sequences"
OUTPUT_DIR = INPUT_DIR / "moved_static_breakdown"


def stats(values):
    if not values:
        x = torch.empty(0)
    else:
        x = torch.cat(values).float()

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


def window_all(mask, width):
    # [T,N] -> [T-width+1,N]
    return mask.unfold(0, width, 1).all(dim=-1)


def selected(x, mask):
    return x[mask].detach().cpu().reshape(-1)


def safe_mean(x, mask):
    values = x[mask]
    return float(values.mean()) if values.numel() else float("nan")


def safe_rate(count, total):
    return float(count / total) if total else float("nan")


def main():
    files = sorted(RAW_DIR.glob("*.pt"))
    if len(files) != 62:
        raise RuntimeError(f"Expected 62 raw files, found {len(files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    values = defaultdict(list)
    reversal = defaultdict(int)
    rows = []

    audit = {
        "processed_sequences": 0,
        "moved_mask_time_varying_points": 0,
        "static_mask_time_varying_points": 0,
        "moved_static_overlap_count": 0,
        "valid_unclassified_count": 0,
    }

    for index, path in enumerate(files):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        pred = data["prediction_scene_flows"].float()
        gt = data["gt_scene_flows"].float()
        valid = data["official_valid_mask"].bool()
        moved = data["scene_moved_mask"].bool()
        static = data["scene_static_mask"].bool()

        if not (
            pred.shape == gt.shape
            and valid.shape == pred.shape[:2]
            and moved.shape == valid.shape
            and static.shape == valid.shape
        ):
            raise RuntimeError(f"Shape mismatch in {path}")

        # Mask audit
        moved_future = moved[1:]
        static_future = static[1:]

        audit["moved_mask_time_varying_points"] += int(
            (moved_future != moved_future[:1]).any(dim=0).sum()
        )
        audit["static_mask_time_varying_points"] += int(
            (static_future != static_future[:1]).any(dim=0).sum()
        )
        audit["moved_static_overlap_count"] += int(
            (valid & moved & static).sum()
        )
        audit["valid_unclassified_count"] += int(
            (valid & (~moved) & (~static)).sum()
        )

        # Position L2: official timestep masks
        l2 = torch.linalg.vector_norm(pred - gt, dim=-1)

        l2_moved_mask = valid & moved
        l2_static_mask = valid & static

        values["l2_moved"].append(selected(l2, l2_moved_mask))
        values["l2_static"].append(selected(l2, l2_static_mask))

        # Velocity
        pred_v = pred[1:] - pred[:-1]
        gt_v = gt[1:] - gt[:-1]
        velocity_error = torch.linalg.vector_norm(
            pred_v - gt_v,
            dim=-1,
        )

        valid2 = window_all(valid, 2)
        moved2 = window_all(moved, 2)
        static2 = window_all(static, 2)

        velocity_moved_mask = valid2 & moved2
        velocity_static_mask = valid2 & static2

        values["velocity_error_moved"].append(
            selected(velocity_error, velocity_moved_mask)
        )
        values["velocity_error_static"].append(
            selected(velocity_error, velocity_static_mask)
        )

        # Error variation
        error_variation = torch.abs(l2[1:] - l2[:-1])

        values["error_variation_moved"].append(
            selected(error_variation, velocity_moved_mask)
        )
        values["error_variation_static"].append(
            selected(error_variation, velocity_static_mask)
        )

        # Acceleration
        pred_a = pred_v[1:] - pred_v[:-1]
        gt_a = gt_v[1:] - gt_v[:-1]
        acceleration_error = torch.linalg.vector_norm(
            pred_a - gt_a,
            dim=-1,
        )

        valid3 = window_all(valid, 3)
        moved3 = window_all(moved, 3)
        static3 = window_all(static, 3)

        acceleration_moved_mask = valid3 & moved3
        acceleration_static_mask = valid3 & static3

        values["acceleration_error_moved"].append(
            selected(acceleration_error, acceleration_moved_mask)
        )
        values["acceleration_error_static"].append(
            selected(acceleration_error, acceleration_static_mask)
        )

        # Jerk
        pred_j = pred_a[1:] - pred_a[:-1]
        gt_j = gt_a[1:] - gt_a[:-1]

        pred_jerk = torch.linalg.vector_norm(pred_j, dim=-1)
        gt_jerk = torch.linalg.vector_norm(gt_j, dim=-1)
        jerk_error = torch.linalg.vector_norm(pred_j - gt_j, dim=-1)

        valid4 = window_all(valid, 4)
        moved4 = window_all(moved, 4)
        static4 = window_all(static, 4)

        jerk_moved_mask = valid4 & moved4
        jerk_static_mask = valid4 & static4

        for name, tensor in [
            ("prediction_jerk_moved", pred_jerk),
            ("prediction_jerk_static", pred_jerk),
            ("gt_jerk_moved", gt_jerk),
            ("gt_jerk_static", gt_jerk),
            ("jerk_error_moved", jerk_error),
            ("jerk_error_static", jerk_error),
        ]:
            mask = jerk_moved_mask if name.endswith("moved") else jerk_static_mask
            values[name].append(selected(tensor, mask))

        # Direction reversal: two velocities require three positions
        pred_v_prev = pred_v[:-1]
        pred_v_next = pred_v[1:]
        gt_v_prev = gt_v[:-1]
        gt_v_next = gt_v[1:]

        eps = torch.finfo(pred.dtype).eps

        pred_defined = (
            valid3
            & (torch.linalg.vector_norm(pred_v_prev, dim=-1) > eps)
            & (torch.linalg.vector_norm(pred_v_next, dim=-1) > eps)
        )
        gt_defined = (
            valid3
            & (torch.linalg.vector_norm(gt_v_prev, dim=-1) > eps)
            & (torch.linalg.vector_norm(gt_v_next, dim=-1) > eps)
        )

        pred_reverse = (
            (pred_v_prev * pred_v_next).sum(dim=-1) < 0
        )
        gt_reverse = (
            (gt_v_prev * gt_v_next).sum(dim=-1) < 0
        )

        for label, class_mask in [
            ("moved", moved3),
            ("static", static3),
        ]:
            pred_class_defined = pred_defined & class_mask
            gt_class_defined = gt_defined & class_mask
            common = pred_class_defined & gt_class_defined

            reversal[f"{label}_prediction_count"] += int(
                (pred_reverse & pred_class_defined).sum()
            )
            reversal[f"{label}_prediction_total"] += int(
                pred_class_defined.sum()
            )
            reversal[f"{label}_gt_count"] += int(
                (gt_reverse & gt_class_defined).sum()
            )
            reversal[f"{label}_gt_total"] += int(
                gt_class_defined.sum()
            )
            reversal[f"{label}_disagreement_count"] += int(
                ((pred_reverse != gt_reverse) & common).sum()
            )
            reversal[f"{label}_common_total"] += int(common.sum())

        rows.append({
            "sequence_index": index,
            "key": data["key"],
            "l2_moved_mean": safe_mean(l2, l2_moved_mask),
            "l2_static_mean": safe_mean(l2, l2_static_mask),
            "velocity_error_moved_mean": safe_mean(
                velocity_error, velocity_moved_mask
            ),
            "velocity_error_static_mean": safe_mean(
                velocity_error, velocity_static_mask
            ),
            "acceleration_error_moved_mean": safe_mean(
                acceleration_error, acceleration_moved_mask
            ),
            "acceleration_error_static_mean": safe_mean(
                acceleration_error, acceleration_static_mask
            ),
            "error_variation_moved_mean": safe_mean(
                error_variation, velocity_moved_mask
            ),
            "error_variation_static_mean": safe_mean(
                error_variation, velocity_static_mask
            ),
            "prediction_jerk_moved_mean": safe_mean(
                pred_jerk, jerk_moved_mask
            ),
            "prediction_jerk_static_mean": safe_mean(
                pred_jerk, jerk_static_mask
            ),
        })

        audit["processed_sequences"] += 1

    summary = {
        "scope": "DROID 62-sample unfiltered moved/static breakdown",
        "analysis_type": (
            "Official position masks; continuous class masks "
            "for temporal derivatives"
        ),
        "mask_audit": audit,
    }

    for name, tensors in values.items():
        summary[name] = stats(tensors)

    for label in ["moved", "static"]:
        summary[f"direction_reversal_{label}"] = {
            "prediction_count": reversal[f"{label}_prediction_count"],
            "prediction_defined_count": reversal[f"{label}_prediction_total"],
            "prediction_rate": safe_rate(
                reversal[f"{label}_prediction_count"],
                reversal[f"{label}_prediction_total"],
            ),
            "gt_count": reversal[f"{label}_gt_count"],
            "gt_defined_count": reversal[f"{label}_gt_total"],
            "gt_rate": safe_rate(
                reversal[f"{label}_gt_count"],
                reversal[f"{label}_gt_total"],
            ),
            "disagreement_count": reversal[
                f"{label}_disagreement_count"
            ],
            "common_defined_count": reversal[
                f"{label}_common_total"
            ],
            "disagreement_rate": safe_rate(
                reversal[f"{label}_disagreement_count"],
                reversal[f"{label}_common_total"],
            ),
        }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(
        OUTPUT_DIR / "per_sequence.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed sequences: {audit['processed_sequences']}")
    print(f"Output: {OUTPUT_DIR}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
