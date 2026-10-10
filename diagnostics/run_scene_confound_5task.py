from copy import deepcopy
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from arguments import parse_args
from dataset_components.dataloader import build_dataloader


ROOTS = [
    (
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
        {"task-0000"}, 578,
    ),
    (
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/diagnostics/data/"
        "pointworld_behavior_task0001_restored/behavior/wds",
        {"task-0001"}, 2490,
    ),
    (
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/diagnostics/data/"
        "pointworld_behavior_multitask_restored/behavior/wds",
        {"task-0006", "task-0008", "task-0010"}, 7184,
    ),
]

METRIC = Path(
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)
MOTION = Path(
    "diagnostics/results/action_confound_control_5task/"
    "per_sample_covariates_7987.csv"
)
OUTDIR = Path(
    "diagnostics/results/action_scene_confound_5task"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def npy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def keystr(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def task_from_clip(x):
    m = re.search(r"task-\d{4}", str(x))
    return m.group(0) if m else None


def mask_t0(x):
    x = npy(x)
    # batch dimension already removed before calling
    x = x[0]
    while x.ndim > 1 and x.shape[-1] == 1:
        x = x[..., 0]
    return x


def safe_frac(values, exists):
    v = np.asarray(values).reshape(-1)
    e = np.asarray(exists).astype(bool).reshape(-1)

    n = min(len(v), len(e))
    v = v[:n]
    e = e[:n]

    if e.sum() == 0:
        return np.nan

    return float(np.asarray(v[e], dtype=float).mean())


def extract_scene(target):
    args0 = parse_args(skip_command_line=True)
    args0.seed = 42
    args0.batch_size = 1
    args0.num_workers = 4
    args0.eval_num_workers = 4
    args0.deterministic_data = True

    rows = []
    seen = set()

    for root, expected_tasks, expected_n in ROOTS:
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
        assert int(info["total_samples"]) == expected_n

        tasks_seen = set()

        for batch in tqdm(loader, total=info["batches_per_rank"],
                          desc=f"Scene: {Path(root).name}"):

            clip = keystr(batch["__key__"][0])
            task = task_from_clip(clip)
            if task:
                tasks_seen.add(task)

            if clip not in target:
                continue

            assert clip not in seen

            exists = mask_t0(batch["scene_exists"][0]).astype(bool)

            visibility = mask_t0(batch["scene_visibility"][0])
            depth_valid = mask_t0(batch["scene_depth_valid_mask"][0])

            n_scene_points_t0 = int(exists.sum())
            visibility_frac_t0 = safe_frac(visibility, exists)
            depth_valid_frac_t0 = safe_frac(depth_valid, exists)

            rows.append({
                "clip": clip,
                "task_loaded": task,
                "n_scene_points_t0": n_scene_points_t0,
                "visibility_frac_t0": visibility_frac_t0,
                "depth_valid_frac_t0": depth_valid_frac_t0,
            })
            seen.add(clip)

        assert tasks_seen == expected_tasks

    out = pd.DataFrame(rows)
    assert len(out) == 7987, len(out)
    return out


def rank01(s):
    return s.rank(method="average", pct=True)


def residualize(y, X):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_rank(g, x, y, controls):
    cols = [x, y] + controls
    d = g[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    for c in cols:
        d[c] = rank01(d[c])

    rx = residualize(d[x], d[controls])
    ry = residualize(d[y], d[controls])
    r, p = pearsonr(rx, ry)
    return len(d), r, p


def main():
    m = pd.read_csv(METRIC)
    motion = pd.read_csv(MOTION)

    assert len(m) == 7987

    target = set(m["clip"])
    scene = extract_scene(target)
    scene.to_csv(OUTDIR / "scene_covariates_7987.csv", index=False)

    df = m.merge(motion, on="clip", validate="one_to_one")
    df = df.merge(scene, on="clip", validate="one_to_one")

    assert len(df) == 7987

    controls = [
        "gt_moved_motion_mm",
        "active_path_length_m",
        "active_net_displacement_m",
        "n_scene_points_t0",
        "visibility_frac_t0",
        "depth_valid_frac_t0",
    ]

    rows = []

    # taskwise
    for task, g in df.groupby("task"):
        for metric in ["rms_acceleration", "mean_squared_jerk"]:
            n, r, p = partial_rank(
                g, metric, "l2_moved_mm", controls
            )
            rows.append({
                "scope": task,
                "metric": metric,
                "N": n,
                "partial_rho_motion_plus_scene": r,
                "p": p,
            })

    # pooled, but rank independently inside each task first
    cols = [
        "rms_acceleration",
        "mean_squared_jerk",
        "l2_moved_mm",
    ] + controls

    parts = []
    for _, g in df.groupby("task"):
        g = g.copy()
        for c in cols:
            g[c] = rank01(g[c])
        parts.append(g)

    d = pd.concat(parts, ignore_index=True)

    for metric in ["rms_acceleration", "mean_squared_jerk"]:
        use = d[[metric, "l2_moved_mm"] + controls].dropna()

        rx = residualize(use[metric], use[controls])
        ry = residualize(use["l2_moved_mm"], use[controls])

        r, p = pearsonr(rx, ry)

        rows.append({
            "scope": "ALL_WITHIN_TASK",
            "metric": metric,
            "N": len(use),
            "partial_rho_motion_plus_scene": r,
            "p": p,
        })

    out = pd.DataFrame(rows)
    out.to_csv(
        OUTDIR / "action_error_partial_motion_plus_scene.csv",
        index=False
    )

    print("=== ACTION ERROR | MOTION + SCENE CONTROLS ===")
    print(out.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    ))

    print("\n=== SCENE COVARIATE SUMMARY ===")
    print(
        scene[
            ["n_scene_points_t0",
             "visibility_frac_t0",
             "depth_valid_frac_t0"]
        ].describe().to_string()
    )

    print("\nOUTPUT =", OUTDIR.resolve())


if __name__ == "__main__":
    main()
