#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch


THRESHOLDS_M = [0.01, 0.02, 0.05]
LAGS = [-2, -1, 0, 1, 2]

# 방향 반전은 너무 작은 움직임의 수치 잡음을 제외한다.
MIN_STEP_M = 0.005
EPS = 1e-8


def finite_values(values):
    values = torch.as_tensor(values).double().flatten()
    return values[torch.isfinite(values)]


def stats(values):
    values = finite_values(values)

    if values.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
        }

    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def safe_rate(count, total):
    return float(count / total) if total else None


def mean_norm(sequence):
    """
    sequence: [T, P, 3]
    returns temporal mean magnitude [P]
    """
    if sequence.shape[0] == 0:
        return torch.full(
            (sequence.shape[1],),
            float("nan"),
        )

    return torch.linalg.vector_norm(
        sequence,
        dim=-1,
    ).mean(dim=0)


def reversal_rate(position):
    """
    A reversal occurs when consecutive velocity vectors
    have a negative dot product. Steps <= 5 mm are excluded.
    """
    velocity = torch.diff(position, dim=0)

    if velocity.shape[0] < 2:
        return torch.full(
            (position.shape[1],),
            float("nan"),
        )

    speed = torch.linalg.vector_norm(
        velocity,
        dim=-1,
    )

    pair_valid = (
        (speed[:-1] > MIN_STEP_M)
        & (speed[1:] > MIN_STEP_M)
    )

    dot = (
        velocity[:-1]
        * velocity[1:]
    ).sum(dim=-1)

    reversal = (dot < 0) & pair_valid

    denominator = pair_valid.sum(dim=0)
    numerator = reversal.sum(dim=0)

    rate = torch.full(
        denominator.shape,
        float("nan"),
        dtype=torch.float64,
    )

    valid_points = denominator > 0

    rate[valid_points] = (
        numerator[valid_points].double()
        / denominator[valid_points].double()
    )

    return rate


def best_velocity_lag(prediction, gt):
    """
    Positive lag means:
      pred_velocity[t] best matches gt_velocity[t-lag]
    Therefore positive lag indicates prediction delay.
    """
    pred_velocity = torch.diff(
        prediction,
        dim=0,
    )
    gt_velocity = torch.diff(
        gt,
        dim=0,
    )

    lag_errors = []

    for lag in LAGS:
        if lag > 0:
            pred_part = pred_velocity[lag:]
            gt_part = gt_velocity[:-lag]
        elif lag < 0:
            shift = -lag
            pred_part = pred_velocity[:-shift]
            gt_part = gt_velocity[shift:]
        else:
            pred_part = pred_velocity
            gt_part = gt_velocity

        error = torch.linalg.vector_norm(
            pred_part - gt_part,
            dim=-1,
        ).mean(dim=0)

        lag_errors.append(error)

    lag_errors = torch.stack(
        lag_errors,
        dim=0,
    )

    best_indices = lag_errors.argmin(dim=0)

    lag_tensor = torch.tensor(
        LAGS,
        dtype=torch.int64,
    )

    best_lags = lag_tensor[best_indices]

    zero_index = LAGS.index(0)
    baseline_error = lag_errors[zero_index]
    best_error = lag_errors.gather(
        0,
        best_indices.unsqueeze(0),
    ).squeeze(0)

    improvement = (
        baseline_error - best_error
    ) / baseline_error.clamp_min(EPS)

    return best_lags, improvement


def crossing_driver_counts(
    prediction,
    gt,
    binary_state,
    selected_points,
):
    """
    Exact two-variable Shapley decomposition of scalar-error change.

    error = ||prediction - GT||

    Prediction contribution and GT contribution sum exactly to
    the adjacent-timestep error change.
    """
    p0 = prediction[:-1]
    p1 = prediction[1:]
    g0 = gt[:-1]
    g1 = gt[1:]

    e00 = torch.linalg.vector_norm(
        p0 - g0,
        dim=-1,
    )
    e10 = torch.linalg.vector_norm(
        p1 - g0,
        dim=-1,
    )
    e01 = torch.linalg.vector_norm(
        p0 - g1,
        dim=-1,
    )
    e11 = torch.linalg.vector_norm(
        p1 - g1,
        dim=-1,
    )

    prediction_contribution = 0.5 * (
        (e10 - e00)
        + (e11 - e01)
    )

    gt_contribution = 0.5 * (
        (e01 - e00)
        + (e11 - e10)
    )

    reconstruction_error = (
        prediction_contribution
        + gt_contribution
        - (e11 - e00)
    ).abs().max()

    if float(reconstruction_error) > 1e-5:
        raise RuntimeError(
            "Shapley decomposition failed: "
            f"{float(reconstruction_error)}"
        )

    crossing = (
        binary_state[1:]
        != binary_state[:-1]
    )

    crossing = (
        crossing
        & selected_points.unsqueeze(0)
    )

    pred_abs = prediction_contribution.abs()[crossing]
    gt_abs = gt_contribution.abs()[crossing]

    tolerance = 1e-12

    prediction_driven = int(
        (pred_abs > gt_abs + tolerance).sum()
    )
    gt_driven = int(
        (gt_abs > pred_abs + tolerance).sum()
    )
    tied = int(
        (
            (pred_abs - gt_abs).abs()
            <= tolerance
        ).sum()
    )

    return {
        "crossing_count": int(crossing.sum()),
        "prediction_driven_count": prediction_driven,
        "gt_driven_count": gt_driven,
        "tied_count": tied,
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

    raw_files = sorted(
        args.raw_dir.glob("*.pt")
    )

    if len(raw_files) != args.expected_sequences:
        raise RuntimeError(
            f"Expected {args.expected_sequences} files, "
            f"found {len(raw_files)}"
        )

    source_summary = json.loads(
        args.source_summary.read_text()
    )

    if source_summary.get(
        "official_l2_reproduced_for_all_sequences"
    ) is not True:
        raise RuntimeError(
            "Official L2 audit was not passed"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    aggregate = {}

    for threshold in THRESHOLDS_M:
        aggregate[threshold] = {
            "multi_flip_point_count": 0,
            "sequence_count_with_multiflip": 0,
            "crossing_count": 0,
            "prediction_driven_count": 0,
            "gt_driven_count": 0,
            "tied_count": 0,
            "pred_speed": [],
            "gt_speed": [],
            "pred_acceleration": [],
            "gt_acceleration": [],
            "pred_jerk": [],
            "gt_jerk": [],
            "pred_reversal": [],
            "gt_reversal": [],
            "best_lags": Counter(),
            "lag_improvement": [],
        }

    per_sequence_rows = []
    seen_keys = set()

    total_valid_error_sum = 0.0
    total_valid_error_count = 0

    for raw_path in raw_files:
        raw = torch.load(
            raw_path,
            map_location="cpu",
            weights_only=False,
        )

        required = {
            "prediction_scene_flows",
            "gt_scene_flows",
            "expert_filtered_mask",
            "scene_moved_mask",
            "key",
        }

        missing = required - set(raw.keys())

        if missing:
            raise KeyError(
                f"{raw_path}: missing {sorted(missing)}"
            )

        key = str(raw["key"])

        if key in seen_keys:
            raise RuntimeError(
                f"Duplicate key: {key}"
            )

        seen_keys.add(key)

        prediction = raw[
            "prediction_scene_flows"
        ].float()
        gt = raw[
            "gt_scene_flows"
        ].float()
        valid = raw[
            "expert_filtered_mask"
        ].bool()
        moved = raw[
            "scene_moved_mask"
        ].bool()

        if prediction.shape != gt.shape:
            raise RuntimeError(
                f"{key}: prediction/GT mismatch"
            )

        error = torch.linalg.vector_norm(
            prediction - gt,
            dim=-1,
        )

        total_valid_error_sum += float(
            error[valid].double().sum()
        )
        total_valid_error_count += int(
            valid.sum()
        )

        future_indices = torch.where(
            valid.any(dim=1)
        )[0]

        if future_indices.numel() == 0:
            raise RuntimeError(
                f"{key}: no valid future timestep"
            )

        prediction = prediction[
            future_indices
        ]
        gt = gt[
            future_indices
        ]
        error = error[
            future_indices
        ]
        valid = valid[
            future_indices
        ]
        moved = moved[
            future_indices
        ]

        continuous_moved = (
            valid & moved
        ).all(dim=0)

        pred_velocity = torch.diff(
            prediction,
            dim=0,
        )
        gt_velocity = torch.diff(
            gt,
            dim=0,
        )

        pred_acceleration = torch.diff(
            pred_velocity,
            dim=0,
        )
        gt_acceleration = torch.diff(
            gt_velocity,
            dim=0,
        )

        pred_jerk = torch.diff(
            pred_acceleration,
            dim=0,
        )
        gt_jerk = torch.diff(
            gt_acceleration,
            dim=0,
        )

        pred_speed_mean = mean_norm(
            pred_velocity
        )
        gt_speed_mean = mean_norm(
            gt_velocity
        )

        pred_acc_mean = mean_norm(
            pred_acceleration
        )
        gt_acc_mean = mean_norm(
            gt_acceleration
        )

        pred_jerk_mean = mean_norm(
            pred_jerk
        )
        gt_jerk_mean = mean_norm(
            gt_jerk
        )

        pred_reversal_rate = reversal_rate(
            prediction
        )
        gt_reversal_rate = reversal_rate(
            gt
        )

        best_lags, lag_improvement = (
            best_velocity_lag(
                prediction,
                gt,
            )
        )

        for threshold in THRESHOLDS_M:
            binary_state = error <= threshold

            flip_count = (
                binary_state[1:]
                != binary_state[:-1]
            ).sum(dim=0)

            multi_flip = (
                continuous_moved
                & (flip_count >= 2)
            )

            selected_count = int(
                multi_flip.sum()
            )

            state = aggregate[threshold]
            state[
                "multi_flip_point_count"
            ] += selected_count

            if selected_count > 0:
                state[
                    "sequence_count_with_multiflip"
                ] += 1

            driver = crossing_driver_counts(
                prediction,
                gt,
                binary_state,
                multi_flip,
            )

            for name in (
                "crossing_count",
                "prediction_driven_count",
                "gt_driven_count",
                "tied_count",
            ):
                state[name] += driver[name]

            selected_pred_speed = (
                pred_speed_mean[multi_flip]
            )
            selected_gt_speed = (
                gt_speed_mean[multi_flip]
            )
            selected_pred_acc = (
                pred_acc_mean[multi_flip]
            )
            selected_gt_acc = (
                gt_acc_mean[multi_flip]
            )
            selected_pred_jerk = (
                pred_jerk_mean[multi_flip]
            )
            selected_gt_jerk = (
                gt_jerk_mean[multi_flip]
            )
            selected_pred_reversal = (
                pred_reversal_rate[multi_flip]
            )
            selected_gt_reversal = (
                gt_reversal_rate[multi_flip]
            )
            selected_lags = best_lags[
                multi_flip
            ]
            selected_improvement = (
                lag_improvement[multi_flip]
            )

            state["pred_speed"].extend(
                selected_pred_speed.tolist()
            )
            state["gt_speed"].extend(
                selected_gt_speed.tolist()
            )
            state["pred_acceleration"].extend(
                selected_pred_acc.tolist()
            )
            state["gt_acceleration"].extend(
                selected_gt_acc.tolist()
            )
            state["pred_jerk"].extend(
                selected_pred_jerk.tolist()
            )
            state["gt_jerk"].extend(
                selected_gt_jerk.tolist()
            )
            state["pred_reversal"].extend(
                selected_pred_reversal.tolist()
            )
            state["gt_reversal"].extend(
                selected_gt_reversal.tolist()
            )
            state["lag_improvement"].extend(
                selected_improvement.tolist()
            )
            state["best_lags"].update(
                selected_lags.tolist()
            )

            common_reversal = (
                torch.isfinite(
                    selected_pred_reversal
                )
                & torch.isfinite(
                    selected_gt_reversal
                )
            )

            per_sequence_rows.append(
                {
                    "key": key,
                    "threshold_m": threshold,
                    "multi_flip_point_count": (
                        selected_count
                    ),
                    "crossing_count": (
                        driver["crossing_count"]
                    ),
                    "prediction_driven_rate": (
                        safe_rate(
                            driver[
                                "prediction_driven_count"
                            ],
                            driver["crossing_count"],
                        )
                    ),
                    "gt_driven_rate": (
                        safe_rate(
                            driver[
                                "gt_driven_count"
                            ],
                            driver["crossing_count"],
                        )
                    ),
                    "median_pred_speed_m": (
                        stats(
                            selected_pred_speed
                        )["median"]
                    ),
                    "median_gt_speed_m": (
                        stats(
                            selected_gt_speed
                        )["median"]
                    ),
                    "pred_jerk_gt_rate": (
                        safe_rate(
                            int(
                                (
                                    selected_pred_jerk
                                    > selected_gt_jerk
                                ).sum()
                            ),
                            selected_count,
                        )
                    ),
                    "pred_reversal_gt_rate": (
                        safe_rate(
                            int(
                                (
                                    selected_pred_reversal[
                                        common_reversal
                                    ]
                                    > selected_gt_reversal[
                                        common_reversal
                                    ]
                                ).sum()
                            ),
                            int(common_reversal.sum()),
                        )
                    ),
                    "positive_best_lag_rate": (
                        safe_rate(
                            int(
                                (selected_lags > 0).sum()
                            ),
                            selected_count,
                        )
                    ),
                    "lag_improvement_gt_5pct_rate": (
                        safe_rate(
                            int(
                                (
                                    selected_improvement
                                    > 0.05
                                ).sum()
                            ),
                            selected_count,
                        )
                    ),
                }
            )

    summary_by_threshold = {}
    threshold_rows = []

    for threshold in THRESHOLDS_M:
        state = aggregate[threshold]

        pred_speed = torch.tensor(
            state["pred_speed"],
            dtype=torch.float64,
        )
        gt_speed = torch.tensor(
            state["gt_speed"],
            dtype=torch.float64,
        )
        pred_acc = torch.tensor(
            state["pred_acceleration"],
            dtype=torch.float64,
        )
        gt_acc = torch.tensor(
            state["gt_acceleration"],
            dtype=torch.float64,
        )
        pred_jerk = torch.tensor(
            state["pred_jerk"],
            dtype=torch.float64,
        )
        gt_jerk = torch.tensor(
            state["gt_jerk"],
            dtype=torch.float64,
        )
        pred_reversal = torch.tensor(
            state["pred_reversal"],
            dtype=torch.float64,
        )
        gt_reversal = torch.tensor(
            state["gt_reversal"],
            dtype=torch.float64,
        )
        lag_improvement = torch.tensor(
            state["lag_improvement"],
            dtype=torch.float64,
        )

        point_count = state[
            "multi_flip_point_count"
        ]
        crossing_count = state[
            "crossing_count"
        ]

        valid_speed_ratio = (
            torch.isfinite(pred_speed)
            & torch.isfinite(gt_speed)
            & (gt_speed > EPS)
        )
        valid_acc_ratio = (
            torch.isfinite(pred_acc)
            & torch.isfinite(gt_acc)
            & (gt_acc > EPS)
        )
        valid_jerk_ratio = (
            torch.isfinite(pred_jerk)
            & torch.isfinite(gt_jerk)
            & (gt_jerk > EPS)
        )
        common_reversal = (
            torch.isfinite(pred_reversal)
            & torch.isfinite(gt_reversal)
        )

        speed_ratio = (
            pred_speed[valid_speed_ratio]
            / gt_speed[valid_speed_ratio]
        )
        acceleration_ratio = (
            pred_acc[valid_acc_ratio]
            / gt_acc[valid_acc_ratio]
        )
        jerk_ratio = (
            pred_jerk[valid_jerk_ratio]
            / gt_jerk[valid_jerk_ratio]
        )

        result = {
            "multi_flip_point_count": point_count,
            "sequence_count_with_multiflip": state[
                "sequence_count_with_multiflip"
            ],
            "crossing_count": crossing_count,
            "prediction_driven_crossing_rate": safe_rate(
                state["prediction_driven_count"],
                crossing_count,
            ),
            "gt_driven_crossing_rate": safe_rate(
                state["gt_driven_count"],
                crossing_count,
            ),
            "median_pred_speed_m": stats(
                pred_speed
            )["median"],
            "median_gt_speed_m": stats(
                gt_speed
            )["median"],
            "median_pred_gt_speed_ratio": stats(
                speed_ratio
            )["median"],
            "median_pred_acceleration_m": stats(
                pred_acc
            )["median"],
            "median_gt_acceleration_m": stats(
                gt_acc
            )["median"],
            "median_pred_gt_acceleration_ratio": stats(
                acceleration_ratio
            )["median"],
            "median_pred_jerk_m": stats(
                pred_jerk
            )["median"],
            "median_gt_jerk_m": stats(
                gt_jerk
            )["median"],
            "median_pred_gt_jerk_ratio": stats(
                jerk_ratio
            )["median"],
            "pred_jerk_greater_than_gt_rate": (
                safe_rate(
                    int(
                        (
                            pred_jerk
                            > gt_jerk
                        ).sum()
                    ),
                    point_count,
                )
            ),
            "median_pred_reversal_rate": stats(
                pred_reversal
            )["median"],
            "median_gt_reversal_rate": stats(
                gt_reversal
            )["median"],
            "pred_reversal_greater_than_gt_rate": (
                safe_rate(
                    int(
                        (
                            pred_reversal[
                                common_reversal
                            ]
                            > gt_reversal[
                                common_reversal
                            ]
                        ).sum()
                    ),
                    int(common_reversal.sum()),
                )
            ),
            "best_lag_minus2_count": int(
                state["best_lags"][-2]
            ),
            "best_lag_minus1_count": int(
                state["best_lags"][-1]
            ),
            "best_lag_zero_count": int(
                state["best_lags"][0]
            ),
            "best_lag_plus1_count": int(
                state["best_lags"][1]
            ),
            "best_lag_plus2_count": int(
                state["best_lags"][2]
            ),
            "positive_best_lag_rate": safe_rate(
                state["best_lags"][1]
                + state["best_lags"][2],
                point_count,
            ),
            "median_lag_improvement": stats(
                lag_improvement
            )["median"],
            "lag_improvement_gt_5pct_rate": (
                safe_rate(
                    int(
                        (
                            lag_improvement
                            > 0.05
                        ).sum()
                    ),
                    point_count,
                )
            ),
        }

        summary_by_threshold[
            f"{threshold:.3f}"
        ] = result

        threshold_rows.append(
            {
                "threshold_m": threshold,
                **result,
            }
        )

    recomputed_l2 = (
        total_valid_error_sum
        / total_valid_error_count
    )

    output_summary = {
        "analysis_type": (
            "Multi-flip cause diagnostic; "
            "not an official PointWorld metric"
        ),
        "processed_sequences": len(raw_files),
        "unique_sequence_keys": len(seen_keys),
        "official_l2_audit_passed": True,
        "recomputed_official_valid_l2": (
            recomputed_l2
        ),
        "thresholds_m": THRESHOLDS_M,
        "continuous_moved_definition": (
            "official-valid and moved at every "
            "evaluated future timestep"
        ),
        "multi_flip_definition": (
            "correct/wrong threshold state changes "
            "at least twice"
        ),
        "crossing_driver_definition": (
            "Exact Shapley decomposition of adjacent "
            "scalar L2 error change into prediction-motion "
            "and GT-motion contributions"
        ),
        "positive_lag_definition": (
            "Prediction velocity best matches an earlier "
            "GT velocity; positive lag indicates delay"
        ),
        "summary_by_threshold": (
            summary_by_threshold
        ),
        "interpretation_rules": {
            "prediction_jitter_signal": (
                "prediction-driven crossings dominate, "
                "prediction jerk/reversal exceed GT"
            ),
            "oversmoothing_or_lag_signal": (
                "GT-driven crossings dominate, prediction "
                "speed/acceleration/jerk are lower than GT, "
                "or positive lag improves velocity alignment"
            ),
            "mixed_signal": (
                "Neither group of indicators dominates"
            ),
        },
    }

    (
        args.output_dir / "summary.json"
    ).write_text(
        json.dumps(
            output_summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    write_csv(
        args.output_dir
        / "threshold_summary.csv",
        threshold_rows,
    )

    write_csv(
        args.output_dir
        / "per_sequence.csv",
        per_sequence_rows,
    )

    print("=" * 88)
    print("MULTI-FLIP CAUSE ANALYSIS COMPLETE")
    print("=" * 88)
    print(
        "processed_sequences =",
        len(raw_files),
    )
    print(
        "unique_keys =",
        len(seen_keys),
    )
    print(
        "recomputed_official_valid_l2 =",
        recomputed_l2,
    )

    for threshold in THRESHOLDS_M:
        row = summary_by_threshold[
            f"{threshold:.3f}"
        ]

        print()
        print(
            f"[threshold={threshold * 100:.0f}cm]"
        )
        print(
            "multi_flip_points =",
            row["multi_flip_point_count"],
        )
        print(
            "prediction_driven_crossing_rate =",
            row[
                "prediction_driven_crossing_rate"
            ],
        )
        print(
            "gt_driven_crossing_rate =",
            row[
                "gt_driven_crossing_rate"
            ],
        )
        print(
            "median_speed_ratio =",
            row[
                "median_pred_gt_speed_ratio"
            ],
        )
        print(
            "median_acceleration_ratio =",
            row[
                "median_pred_gt_acceleration_ratio"
            ],
        )
        print(
            "median_jerk_ratio =",
            row[
                "median_pred_gt_jerk_ratio"
            ],
        )
        print(
            "pred_jerk_gt_rate =",
            row[
                "pred_jerk_greater_than_gt_rate"
            ],
        )
        print(
            "pred_reversal_gt_rate =",
            row[
                "pred_reversal_greater_than_gt_rate"
            ],
        )
        print(
            "positive_best_lag_rate =",
            row[
                "positive_best_lag_rate"
            ],
        )
        print(
            "lag_improvement_gt_5pct_rate =",
            row[
                "lag_improvement_gt_5pct_rate"
            ],
        )

    print()
    print("output_dir =", args.output_dir)
    print("SUCCESS = True")


if __name__ == "__main__":
    main()
