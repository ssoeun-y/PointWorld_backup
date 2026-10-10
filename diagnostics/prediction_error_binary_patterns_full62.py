#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch


THRESHOLDS_M = [0.01, 0.02, 0.05]
COHORT_NAMES = (
    "continuous_valid",
    "continuous_moved",
    "continuous_static",
)


def safe_rate(count, total):
    return float(count / total) if total else None


def pattern_string(code, length):
    bits = format(int(code), f"0{length}b")
    return "".join("C" if bit == "1" else "W" for bit in bits)


def summarize_patterns(binary_sequence):
    """
    binary_sequence: [T, P]
    C(True) = error <= threshold
    W(False) = error > threshold
    """
    if binary_sequence.ndim != 2:
        raise ValueError(binary_sequence.shape)

    length, point_count = binary_sequence.shape

    if point_count == 0:
        return {
            "point_count": 0,
            "flip_hist": [0] * length,
            "correct_hist": [0] * (length + 1),
            "patterns": Counter(),
        }

    correct_count = binary_sequence.sum(dim=0).to(torch.int64)

    if length >= 2:
        flips = (
            binary_sequence[1:] != binary_sequence[:-1]
        ).sum(dim=0).to(torch.int64)
    else:
        flips = torch.zeros(
            point_count,
            dtype=torch.int64,
        )

    flip_hist = torch.bincount(
        flips,
        minlength=length,
    ).tolist()

    correct_hist = torch.bincount(
        correct_count,
        minlength=length + 1,
    ).tolist()

    powers = 2 ** torch.arange(
        length - 1,
        -1,
        -1,
        dtype=torch.int64,
    )

    codes = (
        binary_sequence.T.to(torch.int64) * powers
    ).sum(dim=1)

    unique_codes, unique_counts = torch.unique(
        codes,
        return_counts=True,
    )

    patterns = Counter(
        {
            int(code): int(count)
            for code, count in zip(
                unique_codes.tolist(),
                unique_counts.tolist(),
            )
        }
    )

    return {
        "point_count": int(point_count),
        "flip_hist": flip_hist,
        "correct_hist": correct_hist,
        "patterns": patterns,
    }


def merge_state(state, result):
    state["point_count"] += result["point_count"]

    for index, count in enumerate(result["flip_hist"]):
        state["flip_hist"][index] += count

    for index, count in enumerate(result["correct_hist"]):
        state["correct_hist"][index] += count

    state["patterns"].update(result["patterns"])


def summarize_state(state):
    length = state["length"]
    total = state["point_count"]
    flip_hist = state["flip_hist"]
    correct_hist = state["correct_hist"]

    stable_correct = correct_hist[length]
    stable_wrong = correct_hist[0]
    one_flip = flip_hist[1] if length > 1 else 0
    multi_flip = sum(flip_hist[2:])

    high_alternation_start = math.ceil(
        (length - 1) / 2
    )

    high_alternation = sum(
        flip_hist[high_alternation_start:]
    )

    exact_alternating = (
        flip_hist[length - 1]
        if length > 1
        else 0
    )

    mean_flips = (
        sum(
            index * count
            for index, count in enumerate(flip_hist)
        ) / total
        if total
        else None
    )

    mean_correct_fraction = (
        sum(
            index * count
            for index, count in enumerate(correct_hist)
        ) / (total * length)
        if total
        else None
    )

    return {
        "point_count": total,
        "stable_correct_count": stable_correct,
        "stable_correct_rate": safe_rate(
            stable_correct,
            total,
        ),
        "stable_wrong_count": stable_wrong,
        "stable_wrong_rate": safe_rate(
            stable_wrong,
            total,
        ),
        "one_flip_count": one_flip,
        "one_flip_rate": safe_rate(
            one_flip,
            total,
        ),
        "multi_flip_count": multi_flip,
        "multi_flip_rate": safe_rate(
            multi_flip,
            total,
        ),
        "high_alternation_count": high_alternation,
        "high_alternation_rate": safe_rate(
            high_alternation,
            total,
        ),
        "exact_alternating_count": exact_alternating,
        "exact_alternating_rate": safe_rate(
            exact_alternating,
            total,
        ),
        "mean_flip_count": mean_flips,
        "mean_correct_fraction": mean_correct_fraction,
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source_summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected_sequences",
        type=int,
        default=62,
    )

    args = parser.parse_args()

    raw_files = sorted(args.raw_dir.glob("*.pt"))

    if len(raw_files) != args.expected_sequences:
        raise RuntimeError(
            f"Expected {args.expected_sequences} raw files, "
            f"found {len(raw_files)}"
        )

    source_summary = json.loads(
        args.source_summary.read_text()
    )

    if (
        source_summary.get(
            "official_l2_reproduced_for_all_sequences"
        )
        is not True
    ):
        raise RuntimeError(
            "Original full-run official L2 audit is not True"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    aggregate = {}
    per_sequence_rows = []
    seen_keys = set()

    expected_future_length = None
    total_l2_sum = 0.0
    total_l2_count = 0
    sequence_audits = []

    for file_index, raw_path in enumerate(raw_files):
        raw = torch.load(
            raw_path,
            map_location="cpu",
            weights_only=False,
        )

        required_keys = {
            "prediction_scene_flows",
            "gt_scene_flows",
            "official_valid_mask",
            "scene_moved_mask",
            "scene_static_mask",
            "key",
        }

        missing = required_keys - set(raw.keys())

        if missing:
            raise KeyError(
                f"{raw_path}: missing {sorted(missing)}"
            )

        key = str(raw["key"])

        if key in seen_keys:
            raise RuntimeError(
                f"Duplicate sequence key: {key}"
            )
        seen_keys.add(key)

        prediction = raw[
            "prediction_scene_flows"
        ].float()
        gt = raw["gt_scene_flows"].float()
        valid = raw["official_valid_mask"].bool()
        moved = raw["scene_moved_mask"].bool()
        static = raw["scene_static_mask"].bool()

        if prediction.shape != gt.shape:
            raise RuntimeError(
                f"{key}: prediction/GT shape mismatch"
            )

        if (
            prediction.ndim != 3
            or prediction.shape[-1] != 3
        ):
            raise RuntimeError(
                f"{key}: unexpected flow shape "
                f"{prediction.shape}"
            )

        expected_mask_shape = prediction.shape[:2]

        for name, mask in (
            ("valid", valid),
            ("moved", moved),
            ("static", static),
        ):
            if mask.shape != expected_mask_shape:
                raise RuntimeError(
                    f"{key}: {name} shape {mask.shape}, "
                    f"expected {expected_mask_shape}"
                )

        if (moved & static).any():
            raise RuntimeError(
                f"{key}: moved/static overlap"
            )

        error = torch.linalg.vector_norm(
            prediction - gt,
            dim=-1,
        )

        total_l2_sum += float(
            error[valid].double().sum()
        )
        total_l2_count += int(valid.sum())

        # t=0 context는 official valid가 보통 False이므로,
        # 실제 평가 point가 하나라도 존재하는 미래 timestep만 선택.
        future_indices = torch.where(
            valid.any(dim=1)
        )[0]

        if future_indices.numel() == 0:
            raise RuntimeError(
                f"{key}: no valid future timesteps"
            )

        error_future = error[future_indices]
        valid_future = valid[future_indices]
        moved_future = moved[future_indices]
        static_future = static[future_indices]

        future_length = int(future_indices.numel())

        if expected_future_length is None:
            expected_future_length = future_length
        elif future_length != expected_future_length:
            raise RuntimeError(
                f"{key}: future length {future_length}, "
                f"expected {expected_future_length}"
            )

        cohort_masks = {
            "continuous_valid": (
                valid_future.all(dim=0)
            ),
            "continuous_moved": (
                valid_future
                & moved_future
            ).all(dim=0),
            "continuous_static": (
                valid_future
                & static_future
            ).all(dim=0),
        }

        sequence_audit = {
            "file": raw_path.name,
            "key": key,
            "tensor_shape": list(prediction.shape),
            "future_indices": future_indices.tolist(),
            "official_valid_count": int(valid.sum()),
            "continuous_valid_points": int(
                cohort_masks["continuous_valid"].sum()
            ),
            "continuous_moved_points": int(
                cohort_masks["continuous_moved"].sum()
            ),
            "continuous_static_points": int(
                cohort_masks["continuous_static"].sum()
            ),
        }
        sequence_audits.append(sequence_audit)

        for threshold in THRESHOLDS_M:
            matched = error_future <= threshold

            for cohort_name in COHORT_NAMES:
                point_mask = cohort_masks[cohort_name]
                binary_sequence = matched[:, point_mask]

                result = summarize_patterns(
                    binary_sequence
                )

                aggregate_key = (
                    cohort_name,
                    threshold,
                )

                if aggregate_key not in aggregate:
                    aggregate[aggregate_key] = {
                        "length": future_length,
                        "point_count": 0,
                        "flip_hist": [0] * future_length,
                        "correct_hist": (
                            [0] * (future_length + 1)
                        ),
                        "patterns": Counter(),
                    }

                merge_state(
                    aggregate[aggregate_key],
                    result,
                )

                sequence_state = {
                    "length": future_length,
                    **result,
                }
                sequence_summary = summarize_state(
                    sequence_state
                )

                per_sequence_rows.append(
                    {
                        "sequence_index": file_index,
                        "key": key,
                        "cohort": cohort_name,
                        "threshold_m": threshold,
                        **sequence_summary,
                    }
                )

    if len(seen_keys) != args.expected_sequences:
        raise RuntimeError(
            "Unique key count mismatch"
        )

    category_rows = []
    pattern_rows = []
    final_summary = {}

    for cohort_name in COHORT_NAMES:
        final_summary[cohort_name] = {}

        for threshold in THRESHOLDS_M:
            state = aggregate[
                (cohort_name, threshold)
            ]

            state_summary = summarize_state(state)

            threshold_label = f"{threshold:.3f}"

            final_summary[cohort_name][
                threshold_label
            ] = state_summary

            category_rows.append(
                {
                    "cohort": cohort_name,
                    "threshold_m": threshold,
                    **state_summary,
                }
            )

            total = state["point_count"]

            for rank, (code, count) in enumerate(
                state["patterns"].most_common(20),
                start=1,
            ):
                pattern_rows.append(
                    {
                        "cohort": cohort_name,
                        "threshold_m": threshold,
                        "rank": rank,
                        "pattern": pattern_string(
                            code,
                            state["length"],
                        ),
                        "count": count,
                        "rate": safe_rate(count, total),
                    }
                )

    recomputed_l2 = (
        total_l2_sum / total_l2_count
        if total_l2_count
        else None
    )

    output_summary = {
        "analysis_type": (
            "Binary correct/wrong error-pattern "
            "diagnostic; not an official metric"
        ),
        "correct_definition": (
            "C if pointwise L2 <= threshold; "
            "W otherwise"
        ),
        "thresholds_m": THRESHOLDS_M,
        "processed_sequences": len(raw_files),
        "unique_sequence_keys": len(seen_keys),
        "future_timestep_count": expected_future_length,
        "recomputed_all_official_valid_l2_mean": (
            recomputed_l2
        ),
        "source_official_l2_audit_passed": True,
        "cohorts": final_summary,
        "interpretation": {
            "zero_flips": (
                "always C or always W"
            ),
            "one_flip": (
                "single block transition such as "
                "CCCWW or WWWCC"
            ),
            "multi_flip": (
                "at least two C/W transitions"
            ),
            "high_alternation": (
                "flip count >= half of all adjacent pairs"
            ),
            "exact_alternating": (
                "CWCW... or WCWC..."
            ),
        },
    }

    write_csv(
        args.output_dir / "category_summary.csv",
        category_rows,
    )
    write_csv(
        args.output_dir / "top_patterns.csv",
        pattern_rows,
    )
    write_csv(
        args.output_dir / "per_sequence.csv",
        per_sequence_rows,
    )

    (
        args.output_dir / "summary.json"
    ).write_text(
        json.dumps(
            output_summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    (
        args.output_dir / "audit_per_sequence.json"
    ).write_text(
        json.dumps(
            sequence_audits,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("=" * 88)
    print("FULL-62 BINARY ERROR PATTERN ANALYSIS COMPLETE")
    print("=" * 88)
    print("processed_sequences =", len(raw_files))
    print("unique_keys =", len(seen_keys))
    print("future_timesteps =", expected_future_length)
    print(
        "recomputed_official_valid_l2 =",
        recomputed_l2,
    )

    print("\n[continuous_moved]")
    for threshold in THRESHOLDS_M:
        row = final_summary[
            "continuous_moved"
        ][f"{threshold:.3f}"]

        print(
            f"threshold={threshold * 100:.0f}cm",
            f"points={row['point_count']}",
            f"stable_C={row['stable_correct_rate']}",
            f"stable_W={row['stable_wrong_rate']}",
            f"one_flip={row['one_flip_rate']}",
            f"multi_flip={row['multi_flip_rate']}",
            f"high_alt={row['high_alternation_rate']}",
            f"exact_alt={row['exact_alternating_rate']}",
        )

    print("\noutput_dir =", args.output_dir)
    print("SUCCESS = True")


if __name__ == "__main__":
    main()
