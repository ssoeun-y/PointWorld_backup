#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import torch


def stats(values):
    x = torch.as_tensor(values, dtype=torch.float32)

    if x.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p95": float(torch.quantile(x, 0.95)),
        "p99": float(torch.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def safe_rate(count, total):
    return float(count / total) if total else None


def analyze_threshold(records, threshold):
    rows = []

    for record in records:
        errors = record["errors"]
        delta = errors[1:] - errors[:-1]

        # +1: material worsening
        # -1: material improvement
        #  0: tolerance 이내의 작은 변화
        direction = torch.zeros_like(delta, dtype=torch.int64)
        direction[delta > threshold] = 1
        direction[delta < -threshold] = -1

        active = direction[direction != 0]

        if active.numel() >= 2:
            sign_changes = int(
                (active[1:] != active[:-1]).sum()
            )
        else:
            sign_changes = 0

        worsening_steps = int((direction == 1).sum())
        improving_steps = int((direction == -1).sum())
        stable_steps = int((direction == 0).sum())
        active_steps = worsening_steps + improving_steps

        no_material_change = active_steps == 0
        monotonic_worsening = (
            active_steps > 0
            and worsening_steps == active_steps
        )
        monotonic_improving = (
            active_steps > 0
            and improving_steps == active_steps
        )
        mixed_direction = (
            worsening_steps > 0
            and improving_steps > 0
        )

        endpoint_change = float(errors[-1] - errors[0])

        net_worsening = endpoint_change > threshold
        net_improving = endpoint_change < -threshold
        net_stable = not net_worsening and not net_improving

        material_mask = delta.abs() > threshold
        material_total_variation = float(
            delta[material_mask].abs().sum()
        )

        direction_string = "".join(
            "W" if x == 1 else "I" if x == -1 else "-"
            for x in direction.tolist()
        )

        row = {
            "sequence_index": record["sequence_index"],
            "key": record["key"],
            "point_index": record["point_index"],
            "threshold_m": threshold,
            "direction_sequence": direction_string,
            "material_change_steps": active_steps,
            "stable_steps": stable_steps,
            "worsening_steps": worsening_steps,
            "improving_steps": improving_steps,
            "sign_change_count": sign_changes,
            "material_total_variation_m": material_total_variation,
            "endpoint_change_m": endpoint_change,
            "no_material_change": no_material_change,
            "monotonic_worsening": monotonic_worsening,
            "monotonic_improving": monotonic_improving,
            "mixed_direction": mixed_direction,
            "repeated_oscillation": sign_changes >= 2,
            "strong_alternation": sign_changes >= 4,
            "net_worsening": net_worsening,
            "net_improving": net_improving,
            "net_stable": net_stable,
        }

        for timestep, value in enumerate(errors.tolist(), start=1):
            row[f"error_t{timestep}_m"] = value

        rows.append(row)

    total = len(rows)

    summary = {
        "threshold_m": threshold,
        "threshold_mm": threshold * 1000,
        "trajectory_count": total,
        "no_material_change_rate": safe_rate(
            sum(r["no_material_change"] for r in rows), total
        ),
        "monotonic_worsening_rate": safe_rate(
            sum(r["monotonic_worsening"] for r in rows), total
        ),
        "monotonic_improving_rate": safe_rate(
            sum(r["monotonic_improving"] for r in rows), total
        ),
        "mixed_direction_rate": safe_rate(
            sum(r["mixed_direction"] for r in rows), total
        ),
        "net_worsening_rate": safe_rate(
            sum(r["net_worsening"] for r in rows), total
        ),
        "net_improving_rate": safe_rate(
            sum(r["net_improving"] for r in rows), total
        ),
        "net_stable_rate": safe_rate(
            sum(r["net_stable"] for r in rows), total
        ),
        "zero_sign_change_rate": safe_rate(
            sum(r["sign_change_count"] == 0 for r in rows), total
        ),
        "one_sign_change_rate": safe_rate(
            sum(r["sign_change_count"] == 1 for r in rows), total
        ),
        "repeated_oscillation_rate": safe_rate(
            sum(r["repeated_oscillation"] for r in rows), total
        ),
        "strong_alternation_rate": safe_rate(
            sum(r["strong_alternation"] for r in rows), total
        ),
        "sign_change_count": stats(
            [r["sign_change_count"] for r in rows]
        ),
        "material_change_steps": stats(
            [r["material_change_steps"] for r in rows]
        ),
        "material_total_variation_m": stats(
            [r["material_total_variation_m"] for r in rows]
        ),
        "mean_error_by_timestep_m": [
            float(torch.stack(
                [r["errors"] for r in records]
            )[:, t].mean())
            for t in range(10)
        ],
    }

    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    raw_dir = input_dir / "raw_sequences"
    summary_path = input_dir / "summary.json"
    output_dir = input_dir / "pattern_tolerance_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_path) as file:
        previous_summary = json.load(file)

    static_p95 = previous_summary["metrics"][
        "expert_filtered/error_variation_static"
    ]["p95"]

    static_p99 = previous_summary["metrics"][
        "expert_filtered/error_variation_static"
    ]["p99"]

    raw_files = sorted(raw_dir.glob("*.pt"))

    if len(raw_files) != 5:
        raise RuntimeError(
            f"Expected 5 raw files, found {len(raw_files)}"
        )

    records = []

    for sequence_index, path in enumerate(raw_files):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        pred = data["prediction_scene_flows"].float()
        gt = data["gt_scene_flows"].float()
        official = data["official_valid_mask"].bool()
        expert = data["expert_filtered_mask"].bool()
        moved = data["scene_moved_mask"].bool()

        if pred.shape != gt.shape:
            raise RuntimeError(f"Prediction/GT mismatch: {path}")

        if expert.shape != pred.shape[:2]:
            raise RuntimeError(f"Mask shape mismatch: {path}")

        if (expert & (~official)).any():
            raise RuntimeError(
                f"Expert mask is not a subset of official mask: {path}"
            )

        l2 = torch.linalg.vector_norm(pred - gt, dim=-1)

        # 미래 10개 timestep 모두 expert-filtered이며,
        # 모든 timestep에서 moved인 point만 사용
        continuous_moved = (
            expert[1:] & moved[1:]
        ).all(dim=0)

        point_indices = torch.nonzero(
            continuous_moved,
            as_tuple=False,
        ).flatten()

        for point_index in point_indices.tolist():
            records.append({
                "sequence_index": sequence_index,
                "key": data["key"],
                "point_index": point_index,
                "errors": l2[1:, point_index].clone(),
            })

    expected_count = previous_summary["patterns"][
        "expert_filtered/moved"
    ]["trajectory_count"]

    if len(records) != expected_count:
        raise RuntimeError(
            f"Trajectory count mismatch: "
            f"previous={expected_count}, current={len(records)}"
        )

    thresholds = {
        "zero_tolerance": 0.0,
        "static_p95_tolerance": float(static_p95),
        "static_p99_tolerance": float(static_p99),
    }

    final_summary = {
        "scope": (
            "Expert-filtered, continuously moved trajectories "
            "from the DROID confidence-covered 5-sample subset"
        ),
        "processed_raw_files": len(raw_files),
        "trajectory_count_reproduced": len(records),
        "previous_trajectory_count": expected_count,
        "threshold_definition": (
            "Consecutive L2-error changes whose absolute value "
            "does not exceed the expert-filtered static "
            "error-variation threshold are treated as stable."
        ),
        "direction_symbols": {
            "W": "error worsened beyond tolerance",
            "I": "error improved beyond tolerance",
            "-": "change remained within tolerance",
        },
        "results": {},
    }

    p95_rows = None

    for name, threshold in thresholds.items():
        result, rows = analyze_threshold(records, threshold)
        final_summary["results"][name] = result

        if name == "static_p95_tolerance":
            p95_rows = rows

    with open(output_dir / "summary.json", "w") as file:
        json.dump(final_summary, file, indent=2)

    if p95_rows:
        p95_rows = sorted(
            p95_rows,
            key=lambda row: (
                row["sign_change_count"],
                row["material_total_variation_m"],
            ),
            reverse=True,
        )

        fieldnames = list(p95_rows[0].keys())

        with open(
            output_dir / "per_trajectory_static_p95.csv",
            "w",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(p95_rows)

    print("=" * 72)
    print("Pattern tolerance audit completed")
    print(f"Raw files: {len(raw_files)}")
    print(f"Continuous expert-filtered moved trajectories: {len(records)}")
    print(f"Static p95 tolerance: {static_p95 * 1000:.4f} mm")
    print(f"Static p99 tolerance: {static_p99 * 1000:.4f} mm")
    print(f"Output: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
