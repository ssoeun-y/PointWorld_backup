from copy import deepcopy
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

from arguments import parse_args
from dataset_components.dataloader import build_dataloader


ROOTS = [
    (
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
        {"task-0000"},
        578,
    ),
    (
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/diagnostics/data/pointworld_behavior_task0001_restored/behavior/wds",
        {"task-0001"},
        2490,
    ),
    (
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/diagnostics/data/pointworld_behavior_multitask_restored/behavior/wds",
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

METRIC_CSV = Path("diagnostics/results/established_action_metrics_5task/per_sample_established_metrics_7987.csv")
OUT_DIR = Path("diagnostics/results/action_confound_control_5task")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=-1).sum())


def net_displacement(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(xyz[-1] - xyz[0]))


def choose_dominant_eef(right_pose, left_pose):
    right_xyz = np.asarray(right_pose, dtype=np.float64)[:, :3]
    left_xyz = np.asarray(left_pose, dtype=np.float64)[:, :3]

    right_len = xyz_path_length(right_xyz)
    left_len = xyz_path_length(left_xyz)

    if right_len >= left_len:
        return "right", right_xyz
    return "left", left_xyz


def extract_covariates_for_target_population(target_clips):
    args0 = parse_args(skip_command_line=True)
    args0.seed = 42
    args0.batch_size = 1
    args0.num_workers = 4
    args0.eval_num_workers = 4
    args0.deterministic_data = True

    rows = []
    seen = set()

    for root, expected_tasks, expected_test_n in ROOTS:
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

        assert int(info["total_samples"]) == expected_test_n, (root, info["total_samples"], expected_test_n)

        root_tasks_seen = set()

        for batch in tqdm(loader, total=info["batches_per_rank"], desc=f"Covariates: {root.split('/')[-1]}"):
            clip = decode_key(batch["__key__"][0])
            task = task_from_clip(clip)
            if task is not None:
                root_tasks_seen.add(task)

            if clip not in target_clips:
                continue
            if clip in seen:
                raise RuntimeError(f"Duplicate clip: {clip}")

            right_pose = to_numpy(batch["right_gripper_pose"][0])
            left_pose = to_numpy(batch["left_gripper_pose"][0])
            dominant_hand, xyz = choose_dominant_eef(right_pose, left_pose)

            active_path_length_m = xyz_path_length(xyz)
            active_net_displacement_m = net_displacement(xyz)
            active_mean_speed_m_per_frame = active_path_length_m / max(1, (len(xyz) - 1))

            gt_scene = to_numpy(batch["gt_scene_flows"][0])  # (T,N,3)
            def as_bool_mask(x):
                x = to_numpy(x)
                if x.ndim > 0 and x.shape[-1] == 1:
                    x = x[..., 0]
                return x.astype(bool)

            moved = as_bool_mask(batch["scene_moved_mask"][0])
            context = as_bool_mask(batch["scene_context_mask"][0])
            exists = as_bool_mask(batch["scene_exists"][0])
            supervised = as_bool_mask(batch["scene_supervised_mask"][0])

            valid = moved & (~context) & exists & supervised
            disp = np.linalg.norm(gt_scene - gt_scene[0:1], axis=-1)
            gt_moved_motion_mm = float(disp[valid].mean() * 1000.0) if valid.any() else np.nan

            rows.append({
                "clip": clip,
                "task_loaded": task,
                "dominant_hand": dominant_hand,
                "active_path_length_m": active_path_length_m,
                "active_net_displacement_m": active_net_displacement_m,
                "active_mean_speed_m_per_frame": active_mean_speed_m_per_frame,
                "gt_moved_motion_mm": gt_moved_motion_mm,
            })
            seen.add(clip)

        assert root_tasks_seen == expected_tasks, (root, root_tasks_seen, expected_tasks)

    cov = pd.DataFrame(rows)
    assert len(cov) == EXPECTED_TOTAL
    loaded_counts = cov.groupby("task_loaded").size().to_dict()
    assert loaded_counts == EXPECTED_MOVED, loaded_counts
    return cov


def assign_low_high_input_groups(df, metric, q=0.25):
    parts = []
    for task, g in df.groupby("task"):
        g = g.copy()
        lo = g[metric].quantile(q)
        hi = g[metric].quantile(1 - q)
        g["input_group"] = "MID"
        g.loc[g[metric] <= lo, "input_group"] = "LOW_INPUT"
        g.loc[g[metric] >= hi, "input_group"] = "HIGH_INPUT"
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_rank_covariates(df, covariates):
    df = df.copy()
    for c in covariates:
        df[f"{c}__rank"] = df.groupby("task")[c].rank(method="average", pct=True)
    return df


def greedy_match_within_task(df, metric, covariates, caliper=0.05):
    pairs = []

    for task, g in df.groupby("task"):
        g = g.copy()
        low = g[g["input_group"] == "LOW_INPUT"].copy()
        high = g[g["input_group"] == "HIGH_INPUT"].copy()

        low = low.reset_index(drop=True)
        high = high.sort_values(metric, ascending=False).reset_index(drop=True)

        available_low = set(low.index.tolist())

        for _, hrow in high.iterrows():
            candidates = []
            for li in list(available_low):
                lrow = low.loc[li]

                diffs = []
                ok = True
                for c in covariates:
                    d = abs(hrow[f"{c}__rank"] - lrow[f"{c}__rank"])
                    if d > caliper:
                        ok = False
                        break
                    diffs.append(d)

                if ok:
                    candidates.append((sum(diffs), li, lrow))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[0])
            _, li, lrow = candidates[0]
            available_low.remove(li)

            pairs.append({
                "task": task,
                "metric": metric,
                "caliper": caliper,

                "high_clip": hrow["clip"],
                "low_clip": lrow["clip"],

                "high_input_metric": hrow[metric],
                "low_input_metric": lrow[metric],

                "high_l2_moved_mm": hrow["l2_moved_mm"],
                "low_l2_moved_mm": lrow["l2_moved_mm"],

                "high_gt_moved_motion_mm": hrow["gt_moved_motion_mm"],
                "low_gt_moved_motion_mm": lrow["gt_moved_motion_mm"],

                "high_path_length_m": hrow["active_path_length_m"],
                "low_path_length_m": lrow["active_path_length_m"],

                "high_net_displacement_m": hrow["active_net_displacement_m"],
                "low_net_displacement_m": lrow["active_net_displacement_m"],

                "high_mean_speed_m_per_frame": hrow["active_mean_speed_m_per_frame"],
                "low_mean_speed_m_per_frame": lrow["active_mean_speed_m_per_frame"],
            })

    return pd.DataFrame(pairs)


def summarize_pairs(pairs):
    if len(pairs) == 0:
        return None

    diff = pairs["high_l2_moved_mm"] - pairs["low_l2_moved_mm"]
    w = wilcoxon(diff)

    return {
        "metric": pairs["metric"].iloc[0],
        "caliper": float(pairs["caliper"].iloc[0]),
        "N_pairs": int(len(pairs)),
        "low_input_error_median_mm": float(np.median(pairs["low_l2_moved_mm"])),
        "high_input_error_median_mm": float(np.median(pairs["high_l2_moved_mm"])),
        "high_minus_low_median_mm": float(np.median(diff)),
        "high_over_low_ratio": float(np.median(pairs["high_l2_moved_mm"]) / np.median(pairs["low_l2_moved_mm"])),
        "high_worse_fraction": float((diff > 0).mean()),
        "wilcoxon_p": float(w.pvalue),
        "median_abs_gt_motion_gap_mm": float(np.median(np.abs(pairs["high_gt_moved_motion_mm"] - pairs["low_gt_moved_motion_mm"]))),
        "median_abs_path_gap_m": float(np.median(np.abs(pairs["high_path_length_m"] - pairs["low_path_length_m"]))),
        "median_abs_netdisp_gap_m": float(np.median(np.abs(pairs["high_net_displacement_m"] - pairs["low_net_displacement_m"]))),
        "median_abs_speed_gap_m_per_frame": float(np.median(np.abs(pairs["high_mean_speed_m_per_frame"] - pairs["low_mean_speed_m_per_frame"]))),
    }


def main():
    metric_df = pd.read_csv(METRIC_CSV)
    assert len(metric_df) == EXPECTED_TOTAL
    assert metric_df.groupby("task").size().to_dict() == EXPECTED_MOVED

    target_clips = set(metric_df["clip"])

    cov = extract_covariates_for_target_population(target_clips)
    cov.to_csv(OUT_DIR / "per_sample_covariates_7987.csv", index=False)

    df = metric_df.merge(
        cov,
        left_on="clip",
        right_on="clip",
        how="inner",
        validate="one_to_one",
    )

    assert len(df) == EXPECTED_TOTAL
    assert (df["task"] == df["task_loaded"]).all()
    df = df.drop(columns=["task_loaded"])

    covariates = [
        "gt_moved_motion_mm",
        "active_path_length_m",
        "active_net_displacement_m",
        "active_mean_speed_m_per_frame",
    ]

    all_summaries = []
    for metric in ["rms_acceleration", "mean_squared_jerk"]:
        tmp = assign_low_high_input_groups(df, metric, q=0.25)
        tmp = add_rank_covariates(tmp, covariates)

        for caliper in [0.05, 0.10]:
            pairs = greedy_match_within_task(tmp, metric, covariates, caliper=caliper)
            pairs.to_csv(OUT_DIR / f"matched_pairs_{metric}_caliper{int(caliper*100):02d}.csv", index=False)

            summary = summarize_pairs(pairs)
            if summary is not None:
                all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUT_DIR / "matched_pair_summaries.csv", index=False)

    print("=== MATCHED PAIR SUMMARIES ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print()
    print("OUTPUT =", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
