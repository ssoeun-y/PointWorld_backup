from copy import deepcopy
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from tqdm import tqdm

from arguments import parse_args
from dataset_components.dataloader import build_dataloader
from dataset_components.transforms import compute_flow_derivatives


# ============================================================
# Fixed 5-task population
# ============================================================

ROOTS = [
    (
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
        {"task-0000"},
        578,
    ),
    (
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
        "diagnostics/data/pointworld_behavior_task0001_restored/behavior/wds",
        {"task-0001"},
        2490,
    ),
    (
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
        "diagnostics/data/pointworld_behavior_multitask_restored/behavior/wds",
        {"task-0006", "task-0008", "task-0010"},
        7184,
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
    "diagnostics/results/established_action_metrics_5task"
)

# Fixed frame interval.
# Physical seconds are not assumed here.
# Dimensionless Jerk is invariant to a constant time-scale factor.
DT = 1.0


# ============================================================
# Helpers
# ============================================================

def task_from_clip(clip):
    m = re.search(r"task-\d{4}", str(clip))
    return m.group(0) if m else None


def decode_key(x):
    if isinstance(x, bytes):
        return x.decode()
    return str(x)


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def xyz_path_length(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)

    if len(xyz) < 2:
        return 0.0

    return float(
        np.linalg.norm(
            np.diff(xyz, axis=0),
            axis=-1,
        ).sum()
    )


def choose_dominant_eef(right_pose, left_pose):
    """
    Choose the end-effector that actually carries more Cartesian motion.

    This is NOT a quality score.
    It only selects the active trajectory in bimanual BEHAVIOR clips.
    """

    right_xyz = np.asarray(
        right_pose,
        dtype=np.float64,
    )[:, :3]

    left_xyz = np.asarray(
        left_pose,
        dtype=np.float64,
    )[:, :3]

    right_len = xyz_path_length(right_xyz)
    left_len = xyz_path_length(left_xyz)

    if right_len >= left_len:
        return (
            "right",
            right_xyz,
            right_len,
            left_len,
        )

    return (
        "left",
        left_xyz,
        right_len,
        left_len,
    )


# ============================================================
# PointWorld official temporal derivative
# ============================================================

def pointworld_velocity_acceleration(xyz):
    """
    Uses the EXACT PointWorld official
    compute_flow_derivatives() implementation.

    xyz: (T,3)
    """

    xyz = np.asarray(xyz, dtype=np.float64)

    flows = xyz[:, None, :]  # (T,1,3)

    velocity, acceleration = compute_flow_derivatives(
        flows
    )

    return (
        velocity[:, 0, :],
        acceleration[:, 0, :],
    )


# ============================================================
# Acceleration metric
# ============================================================

def rms_acceleration(xyz):
    """
    RMS Cartesian acceleration magnitude.

    Under PointWorld's released derivative convention:
        unit = m / frame^2

    No combination with jerk.
    """

    _, acceleration = pointworld_velocity_acceleration(
        xyz
    )

    a_mag = np.linalg.norm(
        acceleration,
        axis=-1,
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(a_mag)
            )
        )
    )


# ============================================================
# ICLR 2024 / Fancy Gym jerk definitions
# ============================================================

def calculate_jerk_profile(velocity_profile, dt=DT):
    """
    Fancy Gym / ICLR 2024 definition:

        jerk = diff(velocity_profile, n=2) / dt^2
    """

    velocity_profile = np.asarray(
        velocity_profile,
        dtype=np.float64,
    )

    return (
        np.diff(
            velocity_profile,
            n=2,
            axis=0,
        )
        / (dt ** 2)
    )


def calculate_mean_squared_jerk(jerk_profile):
    """
    Fancy Gym / ICLR 2024:
        mean(jerk^2)
    """

    return float(
        np.mean(
            np.square(jerk_profile)
        )
    )


def calculate_maximum_jerk(jerk_profile):
    """
    Fancy Gym / ICLR 2024:
        max(abs(jerk))
    """

    return float(
        np.max(
            np.abs(jerk_profile)
        )
    )


def calculate_dimensionless_jerk(
    jerk_profile,
    velocity_profile,
    dt=DT,
):
    """
    Fancy Gym / ICLR 2024 functional form:

        sum_squared_jerk = sum(jerk^2, time)
        duration = len(velocity_profile) * dt
        peak_velocity = max(abs(velocity_profile), time)

        dimensionless_jerk =
            mean(
                sum_squared_jerk
                * duration^3
                / peak_velocity^2
            )

    Zero-motion Cartesian dimensions are ignored to avoid 0/0.

    Larger = less smooth.
    """

    jerk_profile = np.asarray(
        jerk_profile,
        dtype=np.float64,
    )

    velocity_profile = np.asarray(
        velocity_profile,
        dtype=np.float64,
    )

    sum_squared_jerk = np.sum(
        np.square(jerk_profile),
        axis=0,
    )

    duration = (
        len(velocity_profile)
        * dt
    )

    peak_velocity = np.max(
        np.abs(velocity_profile),
        axis=0,
    )

    valid = peak_velocity > 1e-12

    if not np.any(valid):
        return np.nan

    per_dim = (
        sum_squared_jerk[valid]
        * (duration ** 3)
        / np.square(
            peak_velocity[valid]
        )
    )

    return float(
        np.mean(per_dim)
    )


def established_metrics_for_clip(
    right_pose,
    left_pose,
):
    """
    Final independent Action metrics.

    NO averaging between metrics.
    NO composite Action Quality.
    """

    (
        dominant_hand,
        xyz,
        right_path,
        left_path,
    ) = choose_dominant_eef(
        right_pose,
        left_pose,
    )

    velocity, _ = pointworld_velocity_acceleration(
        xyz
    )

    jerk_profile = calculate_jerk_profile(
        velocity,
        DT,
    )

    return {
        "dominant_hand":
            dominant_hand,

        "right_path_length_m":
            right_path,

        "left_path_length_m":
            left_path,

        # Acceleration
        "rms_acceleration":
            rms_acceleration(xyz),

        # ICLR 2024 jerk metrics
        "dimensionless_jerk":
            calculate_dimensionless_jerk(
                jerk_profile,
                velocity,
                DT,
            ),

        "maximum_jerk":
            calculate_maximum_jerk(
                jerk_profile
            ),

        "mean_squared_jerk":
            calculate_mean_squared_jerk(
                jerk_profile
            ),
    }


# ============================================================
# Statistics
# ============================================================

METRICS = [
    "rms_acceleration",
    "dimensionless_jerk",
    "maximum_jerk",
    "mean_squared_jerk",
]


def make_correlations(df):
    rows = []

    # --------------------------------------------
    # Whole population
    # --------------------------------------------

    for metric in METRICS:

        clean = df[
            [metric, "l2_moved_mm"]
        ].dropna()

        rho, p = spearmanr(
            clean[metric],
            clean["l2_moved_mm"],
        )

        rows.append({
            "scope": "ALL_RAW",
            "metric": metric,
            "N": len(clean),
            "spearman_rho": float(rho),
            "p_value": float(p),
        })

    # --------------------------------------------
    # Each task separately
    # --------------------------------------------

    for task, g in df.groupby("task"):

        for metric in METRICS:

            clean = g[
                [metric, "l2_moved_mm"]
            ].dropna()

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

    # --------------------------------------------
    # Pooled, task-adjusted
    #
    # Each metric ranked INDEPENDENTLY.
    # Nothing is averaged together.
    # --------------------------------------------

    error_rank = (
        df.groupby("task")[
            "l2_moved_mm"
        ]
        .rank(pct=True)
    )

    for metric in METRICS:

        metric_rank = (
            df.groupby("task")[metric]
            .rank(pct=True)
        )

        valid = (
            metric_rank.notna()
            & error_rank.notna()
        )

        rho, p = spearmanr(
            metric_rank[valid],
            error_rank[valid],
        )

        rows.append({
            "scope": "ALL_WITHIN_TASK",
            "metric": metric,
            "N": int(valid.sum()),
            "spearman_rho": float(rho),
            "p_value": float(p),
        })

    return pd.DataFrame(rows)


def make_prediction_groups(df, q=0.25):
    """
    Output-based grouping ONLY.

    LOW_ERROR_PREDICTION
        bottom 25% moved-L2 within each task

    HIGH_ERROR_PREDICTION
        top 25% moved-L2 within each task

    We do NOT call Actions HQ/LQ.
    """

    pieces = []

    for task, g in df.groupby("task"):

        g = g.copy()

        low_thr = g[
            "l2_moved_mm"
        ].quantile(q)

        high_thr = g[
            "l2_moved_mm"
        ].quantile(1.0 - q)

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

    grouped = pd.concat(
        pieces,
        ignore_index=True,
    )

    rows = []

    scopes = [("ALL", grouped)]
    scopes += list(
        grouped.groupby("task")
    )

    for scope, g in scopes:

        low = g[
            g["prediction_group"]
            == "LOW_ERROR_PREDICTION"
        ]

        high = g[
            g["prediction_group"]
            == "HIGH_ERROR_PREDICTION"
        ]

        for metric in METRICS:

            x = (
                low[metric]
                .dropna()
                .to_numpy()
            )

            y = (
                high[metric]
                .dropna()
                .to_numpy()
            )

            if len(x) == 0 or len(y) == 0:
                continue

            u, p = mannwhitneyu(
                y,
                x,
                alternative="two-sided",
            )

            low_med = float(
                np.median(x)
            )

            high_med = float(
                np.median(y)
            )

            cliffs_delta = (
                2.0
                * float(u)
                / (len(x) * len(y))
                - 1.0
            )

            rows.append({
                "scope":
                    scope,

                "metric":
                    metric,

                "N_low_error":
                    len(x),

                "N_high_error":
                    len(y),

                "low_error_median":
                    low_med,

                "high_error_median":
                    high_med,

                "high_minus_low":
                    high_med - low_med,

                "high_over_low":
                    (
                        high_med / low_med
                        if low_med != 0
                        else np.nan
                    ),

                "mannwhitney_p":
                    float(p),

                "cliffs_delta":
                    float(cliffs_delta),
            })

    return (
        grouped,
        pd.DataFrame(rows),
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Existing OFFICIAL moved-L2 population
    # --------------------------------------------------------

    assert BASE_CSV.exists(), (
        f"Missing base CSV: {BASE_CSV}"
    )

    old = pd.read_csv(
        BASE_CSV
    )

    base = old[
        [
            "task",
            "clip",
            "l2_moved_mm",
        ]
    ].copy()

    assert len(base) == EXPECTED_TOTAL, (
        f"Expected {EXPECTED_TOTAL}, "
        f"got {len(base)}"
    )

    assert base["clip"].is_unique, (
        "Duplicate clip keys in base CSV"
    )

    counts = (
        base.groupby("task")
        .size()
        .to_dict()
    )

    assert counts == EXPECTED_MOVED, (
        f"Base population mismatch: {counts}"
    )

    print()
    print("======================================")
    print("BASE 5-TASK MOVED POPULATION: PASS")
    print("======================================")
    print(base.groupby("task").size())
    print("TOTAL =", len(base))

    target_clips = set(
        base["clip"]
    )

    # --------------------------------------------------------
    # Official PointWorld args
    # --------------------------------------------------------

    args0 = parse_args(
        skip_command_line=True
    )

    args0.seed = 42
    args0.batch_size = 1
    args0.num_workers = 4
    args0.eval_num_workers = 4
    args0.deterministic_data = True

    metric_rows = []
    seen = set()

    # --------------------------------------------------------
    # Three WDS roots, five tasks.
    #
    # IMPORTANT:
    # each root is read separately because official
    # multi-dataset eval uses RoundRobin(longest=False).
    # --------------------------------------------------------

    for (
        root,
        expected_tasks,
        expected_test_n,
    ) in ROOTS:

        print()
        print("======================================")
        print("ROOT:", root)
        print(
            "EXPECTED TASKS:",
            sorted(expected_tasks),
        )
        print("======================================")

        args = deepcopy(args0)

        args.domains = [
            "behavior"
        ]

        args.data_dirs = [
            root
        ]

        loader, info = build_dataloader(
            args=args,
            mode="test",
            rank=0,
            world_size=1,
            override_splits="test",
            force_resampled_eval=False,
        )

        assert loader is not None

        assert (
            int(info["total_samples"])
            == expected_test_n
        ), (
            f"Test population mismatch "
            f"for {root}: "
            f"{info['total_samples']} "
            f"!= {expected_test_n}"
        )

        root_tasks_seen = set()

        for batch in tqdm(
            loader,
            total=info[
                "batches_per_rank"
            ],
            desc="Action metrics",
        ):

            B = len(
                batch["__key__"]
            )

            for b in range(B):

                clip = decode_key(
                    batch["__key__"][b]
                )

                task = task_from_clip(
                    clip
                )

                if task is not None:
                    root_tasks_seen.add(
                        task
                    )

                # Only fixed moved population
                if clip not in target_clips:
                    continue

                if clip in seen:
                    raise RuntimeError(
                        f"Duplicate target clip: "
                        f"{clip}"
                    )

                right_pose = to_numpy(
                    batch[
                        "right_gripper_pose"
                    ][b]
                )

                left_pose = to_numpy(
                    batch[
                        "left_gripper_pose"
                    ][b]
                )

                metrics = (
                    established_metrics_for_clip(
                        right_pose,
                        left_pose,
                    )
                )

                metric_rows.append({
                    "clip":
                        clip,

                    "task_loaded":
                        task,

                    **metrics,
                })

                seen.add(
                    clip
                )

        assert (
            root_tasks_seen
            == expected_tasks
        ), (
            f"ROOT TASK ERROR\n"
            f"expected={expected_tasks}\n"
            f"seen={root_tasks_seen}"
        )

        print(
            "ROOT TASK CHECK: PASS",
            sorted(root_tasks_seen),
        )

    # --------------------------------------------------------
    # Mandatory 7987 lineage audit
    # --------------------------------------------------------

    metrics = pd.DataFrame(
        metric_rows
    )

    missing = (
        target_clips
        - seen
    )

    extra = (
        seen
        - target_clips
    )

    assert len(missing) == 0, (
        f"Missing {len(missing)} "
        f"target clips.\n"
        f"Examples: "
        f"{sorted(missing)[:10]}"
    )

    assert len(extra) == 0

    assert (
        len(metrics)
        == EXPECTED_TOTAL
    ), (
        f"Expected extracted "
        f"{EXPECTED_TOTAL}, "
        f"got {len(metrics)}"
    )

    loaded_counts = (
        metrics
        .groupby("task_loaded")
        .size()
        .to_dict()
    )

    assert (
        loaded_counts
        == EXPECTED_MOVED
    ), (
        f"Extracted population "
        f"mismatch: {loaded_counts}"
    )

    # --------------------------------------------------------
    # Merge with existing official moved-L2
    # --------------------------------------------------------

    df = base.merge(
        metrics,
        on="clip",
        how="inner",
        validate="one_to_one",
    )

    assert (
        len(df)
        == EXPECTED_TOTAL
    )

    assert (
        df["task"]
        == df["task_loaded"]
    ).all()

    df = df.drop(
        columns=[
            "task_loaded"
        ]
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    corr = make_correlations(
        df
    )

    grouped, group_stats = (
        make_prediction_groups(
            df,
            q=0.25,
        )
    )

    # --------------------------------------------------------
    # Save ONLY after all asserts passed
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT_DIR
        / "per_sample_established_metrics_7987.csv",
        index=False,
    )

    corr.to_csv(
        OUT_DIR
        / "continuous_correlations.csv",
        index=False,
    )

    grouped.to_csv(
        OUT_DIR
        / "prediction_groups_q25.csv",
        index=False,
    )

    group_stats.to_csv(
        OUT_DIR
        / "low_vs_high_error_input_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("======================================")
    print("FINAL 5-TASK LINEAGE: PASS")
    print("======================================")
    print(df.groupby("task").size())
    print("TOTAL =", len(df))

    print()
    print(
        "=== CONTINUOUS CORRELATIONS ==="
    )
    print(
        corr.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6g}",
        )
    )

    print()
    print(
        "=== LOW-ERROR vs "
        "HIGH-ERROR PREDICTION ==="
    )
    print(
        group_stats.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6g}",
        )
    )

    print()
    print(
        "=== METRICS ==="
    )
    print(
        "RMS Acceleration "
        "(PointWorld derivative)"
    )
    print(
        "Dimensionless Jerk "
        "(ICLR 2024 / Fancy Gym)"
    )
    print(
        "Maximum Jerk "
        "(ICLR 2024 / Fancy Gym)"
    )
    print(
        "Mean Squared Jerk "
        "(ICLR 2024 / Fancy Gym)"
    )

    print()
    print(
        "OUTPUT:",
        OUT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
