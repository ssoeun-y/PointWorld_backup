from copy import deepcopy
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, spearmanr
from tqdm import tqdm

from arguments import parse_args
from dataset_components.dataloader import build_dataloader
from dataset_components.transforms import compute_flow_derivatives


# ============================================================
# Fixed 5-task paper population
# ============================================================

ROOTS = [
    (
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
        {"task-0000"},
    ),
    (
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
        "diagnostics/data/pointworld_behavior_task0001_restored/behavior/wds",
        {"task-0001"},
    ),
    (
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
        "diagnostics/data/pointworld_behavior_multitask_restored/behavior/wds",
        {"task-0006", "task-0008", "task-0010"},
    ),
]

EXPECTED_MOVED = {
    "task-0000": 377,
    "task-0001": 2322,
    "task-0006": 1782,
    "task-0008": 1827,
    "task-0010": 1679,
}

EXPECTED_TOTAL = 7987

BASE_CSV = Path(
    "diagnostics/results/action_quality_5task_moved_analysis.csv"
)

OUT_DIR = Path(
    "diagnostics/results/action_input_vs_prediction_5task"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def task_from_clip(x):
    m = re.search(r"task-\d{4}", str(x))
    return m.group(0) if m else None


def get_key(batch, b=0):
    x = batch["__key__"][b]
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)


def robot_valid_mask(robot_exists):
    """
    robot_exists: (T,N) or (T,N,1)
    Keep robot points existing through the full sequence.
    """
    x = np.asarray(robot_exists)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    return x.astype(bool).all(axis=0)


# ============================================================
# Metric A: PointWorld official acceleration
# ============================================================

def compute_acceleration_metrics(robot_flows, robot_exists):
    """
    Vector acceleration is EXACTLY PointWorld official:
        compute_flow_derivatives(robot_flows)

    Main scalar:
        RMS Acceleration

    Units:
        meter / frame^2
    because official PointWorld derivative does not divide by dt.
    """

    valid = robot_valid_mask(robot_exists)

    if valid.sum() == 0:
        return {
            "rms_acceleration": np.nan,
            "mean_acceleration": np.nan,
            "peak_acceleration": np.nan,
            "n_robot_points": 0,
        }

    x = np.asarray(robot_flows)[:, valid, :]

    # OFFICIAL POINTWORLD FUNCTION
    _, acceleration = compute_flow_derivatives(x)

    # Euclidean magnitude per timestep / robot point
    a_mag = np.linalg.norm(acceleration, axis=-1)

    return {
        # MAIN acceleration metric
        "rms_acceleration": float(
            np.sqrt(np.mean(a_mag ** 2))
        ),

        # diagnostics only
        "mean_acceleration": float(np.mean(a_mag)),
        "peak_acceleration": float(np.max(a_mag)),
        "n_robot_points": int(valid.sum()),
    }


# ============================================================
# Metric B: FlowMDM CVPR 2024 jerk construction
# ============================================================

def compute_jerk_metrics(robot_flows, robot_exists):
    """
    FlowMDM CVPR 2024 calculate_jerk() construction:

        vel  = x[t+1] - x[t]
        acc  = vel[t+1] - vel[t]
        jerk = acc[t+1] - acc[t]

        L1 over xyz
        max over joints/entities

    FlowMDM's published Peak Jerk is dataset-level.
    Here we use the SAME frame-level jerk construction and
    summarize each PointWorld action sequence separately.

    Main scalar:
        per_sequence_peak_jerk

    Units:
        meter / frame^3
    """

    valid = robot_valid_mask(robot_exists)

    if valid.sum() == 0:
        return {
            "per_sequence_peak_jerk": np.nan,
            "mean_frame_jerk": np.nan,
        }

    poses = np.asarray(robot_flows)[:, valid, :]

    if poses.shape[0] < 4:
        return {
            "per_sequence_peak_jerk": np.nan,
            "mean_frame_jerk": np.nan,
        }

    # EXACT FlowMDM finite difference construction
    velocity = poses[1:] - poses[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    jerk = acceleration[1:] - acceleration[:-1]

    # FlowMDM: L1 norm over xyz
    jerk_l1 = np.sum(np.abs(jerk), axis=-1)

    # FlowMDM: maximum across entity/joint dimension
    frame_jerk = np.max(np.abs(jerk_l1), axis=-1)

    return {
        # MAIN jerk metric for per-sequence correlation
        "per_sequence_peak_jerk": float(
            np.max(frame_jerk)
        ),

        # diagnostic
        "mean_frame_jerk": float(
            np.mean(frame_jerk)
        ),
    }


# ============================================================
# Statistics
# ============================================================

MAIN_METRICS = [
    "rms_acceleration",
    "per_sequence_peak_jerk",
]


def correlations(df):
    rows = []

    # pooled
    for metric in MAIN_METRICS:
        rho, p = spearmanr(
            df[metric],
            df["l2_moved_mm"],
            nan_policy="omit",
        )

        rows.append({
            "scope": "ALL",
            "metric": metric,
            "N": int(
                df[[metric, "l2_moved_mm"]]
                .dropna()
                .shape[0]
            ),
            "spearman_rho": float(rho),
            "p_value": float(p),
        })

    # taskwise
    for task, g in df.groupby("task"):
        for metric in MAIN_METRICS:
            clean = g[[metric, "l2_moved_mm"]].dropna()

            rho, p = spearmanr(
                clean[metric],
                clean["l2_moved_mm"],
            )

            rows.append({
                "scope": task,
                "metric": metric,
                "N": len(clean),
                "spearman_rho": float(rho),
                "p_value": float(p),
            })

    return pd.DataFrame(rows)


def prediction_group_analysis(df, q=0.25):
    """
    Groups are defined ONLY from output prediction error.

    LOW_ERROR_PREDICTION  = bottom 25% within each task
    HIGH_ERROR_PREDICTION = top 25% within each task

    We NEVER call Action itself HQ/LQ.
    """

    pieces = []

    for task, g in df.groupby("task"):
        g = g.copy()

        low_thr = g["l2_moved_mm"].quantile(q)
        high_thr = g["l2_moved_mm"].quantile(1 - q)

        g["prediction_group"] = "MID"

        g.loc[
            g["l2_moved_mm"] <= low_thr,
            "prediction_group"
        ] = "LOW_ERROR_PREDICTION"

        g.loc[
            g["l2_moved_mm"] >= high_thr,
            "prediction_group"
        ] = "HIGH_ERROR_PREDICTION"

        pieces.append(g)

    grouped = pd.concat(pieces, ignore_index=True)

    rows = []

    scopes = [("ALL", grouped)]
    scopes += list(grouped.groupby("task"))

    for scope, g in scopes:

        low = g[
            g["prediction_group"] == "LOW_ERROR_PREDICTION"
        ]

        high = g[
            g["prediction_group"] == "HIGH_ERROR_PREDICTION"
        ]

        for metric in MAIN_METRICS:

            x = low[metric].dropna().to_numpy()
            y = high[metric].dropna().to_numpy()

            u, p = mannwhitneyu(
                y,
                x,
                alternative="two-sided",
            )

            # Positive = high-error group has larger input metric
            cliffs_delta = (
                2.0 * u / (len(y) * len(x))
            ) - 1.0

            low_med = float(np.median(x))
            high_med = float(np.median(y))

            rows.append({
                "scope": scope,
                "metric": metric,

                "N_low_error": len(x),
                "N_high_error": len(y),

                "low_error_median": low_med,
                "high_error_median": high_med,

                "high_minus_low": high_med - low_med,

                "high_over_low":
                    high_med / low_med
                    if low_med != 0
                    else np.nan,

                "mannwhitney_p": float(p),
                "cliffs_delta": float(cliffs_delta),
            })

    return grouped, pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Existing official moved-L2 population
    # --------------------------------------------------------

    assert BASE_CSV.exists(), BASE_CSV

    base = pd.read_csv(BASE_CSV)

    required = {
        "task",
        "clip",
        "l2_moved_mm",
    }

    assert required.issubset(base.columns)

    base = base[
        ["task", "clip", "l2_moved_mm"]
    ].copy()

    assert len(base) == EXPECTED_TOTAL, (
        f"Expected {EXPECTED_TOTAL}, got {len(base)}"
    )

    assert base["clip"].is_unique, (
        "Duplicate clips in existing 5-task table"
    )

    counts = base.groupby("task").size().to_dict()

    assert counts == EXPECTED_MOVED, (
        f"Population mismatch: {counts}"
    )

    target_clips = set(base["clip"])

    print("=== TARGET POPULATION VERIFIED ===")
    print(base.groupby("task").size())
    print("TOTAL =", len(base))

    # --------------------------------------------------------
    # Official PointWorld args
    # --------------------------------------------------------

    args0 = parse_args(skip_command_line=True)

    # Match official evaluation determinism
    args0.seed = 42
    args0.batch_size = 1
    args0.num_workers = 4
    args0.eval_num_workers = 4
    args0.domains = ["behavior"]

    # --------------------------------------------------------
    # Read all 5 tasks through OFFICIAL test dataloader
    # No model forward.
    # --------------------------------------------------------

    metric_rows = []
    seen_target = set()

    for root, expected_tasks in ROOTS:

        print()
        print("==============================================")
        print("ROOT:", root)
        print("EXPECTED TASKS:", sorted(expected_tasks))
        print("==============================================")

        args = deepcopy(args0)
        args.domains = ["behavior"]
        args.data_dirs = [root]

        loader, info = build_dataloader(
            args=args,
            mode="test",
            rank=0,
            world_size=1,
            override_splits="test",
            force_resampled_eval=False,
        )

        assert loader is not None

        root_tasks_seen = set()

        for batch in tqdm(
            loader,
            desc=f"Action only: {sorted(expected_tasks)}",
        ):

            B = len(batch["__key__"])

            for b in range(B):

                clip = get_key(batch, b)
                task = task_from_clip(clip)

                if task is not None:
                    root_tasks_seen.add(task)

                # We need metrics only for the existing
                # official moved population.
                if clip not in target_clips:
                    continue

                if clip in seen_target:
                    raise RuntimeError(
                        f"Duplicate target clip: {clip}"
                    )

                robot_flows = (
                    batch["robot_flows"][b]
                    .detach()
                    .cpu()
                    .numpy()
                )

                robot_exists = (
                    batch["robot_exists"][b]
                    .detach()
                    .cpu()
                    .numpy()
                )

                acc = compute_acceleration_metrics(
                    robot_flows,
                    robot_exists,
                )

                jerk = compute_jerk_metrics(
                    robot_flows,
                    robot_exists,
                )

                metric_rows.append({
                    "clip": clip,
                    "task_loaded": task,
                    **acc,
                    **jerk,
                })

                seen_target.add(clip)

        assert root_tasks_seen == expected_tasks, (
            f"ROOT TASK MISMATCH\n"
            f"expected={expected_tasks}\n"
            f"seen={root_tasks_seen}"
        )

        print(
            "ROOT TASK CHECK OK:",
            sorted(root_tasks_seen)
        )

    # --------------------------------------------------------
    # Mandatory lineage checks BEFORE saving
    # --------------------------------------------------------

    metrics = pd.DataFrame(metric_rows)

    print()
    print("=== EXTRACTION COUNTS ===")
    print(metrics.groupby("task_loaded").size())

    missing = target_clips - seen_target
    extra = seen_target - target_clips

    assert len(missing) == 0, (
        f"Missing target clips: {len(missing)}\n"
        f"Examples: {list(sorted(missing))[:10]}"
    )

    assert len(extra) == 0

    assert len(metrics) == EXPECTED_TOTAL

    loaded_counts = (
        metrics.groupby("task_loaded")
        .size()
        .to_dict()
    )

    assert loaded_counts == EXPECTED_MOVED, (
        f"Loaded population mismatch: {loaded_counts}"
    )

    # --------------------------------------------------------
    # Merge with EXISTING official moved-L2
    # --------------------------------------------------------

    merged = base.merge(
        metrics,
        on="clip",
        how="inner",
        validate="one_to_one",
    )

    assert len(merged) == EXPECTED_TOTAL

    assert (
        merged["task"] == merged["task_loaded"]
    ).all()

    merged = merged.drop(
        columns=["task_loaded"]
    )

    # --------------------------------------------------------
    # Continuous correlation
    # --------------------------------------------------------

    corr = correlations(merged)

    # --------------------------------------------------------
    # Good vs bad prediction comparison
    # --------------------------------------------------------

    grouped, group_stats = prediction_group_analysis(
        merged,
        q=0.25,
    )

    # --------------------------------------------------------
    # SAVE ONLY after every assert passed
    # --------------------------------------------------------

    merged.to_csv(
        OUT_DIR / "per_sample_7987.csv",
        index=False,
    )

    corr.to_csv(
        OUT_DIR / "continuous_correlations.csv",
        index=False,
    )

    grouped.to_csv(
        OUT_DIR / "prediction_groups_q25.csv",
        index=False,
    )

    group_stats.to_csv(
        OUT_DIR / "low_vs_high_error_input_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print()
    print("==============================================")
    print("FINAL 5-TASK LINEAGE: PASS")
    print("==============================================")
    print(merged.groupby("task").size())
    print("TOTAL =", len(merged))

    print()
    print("=== CONTINUOUS CORRELATION ===")
    print(
        corr.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print()
    print("=== LOW-ERROR vs HIGH-ERROR PREDICTION ===")
    print(
        group_stats.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print()
    print("OUTPUT:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()
