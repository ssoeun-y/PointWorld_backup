from copy import deepcopy
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from arguments import parse_args
from dataset_components.dataloader import build_dataloader

ROOTS = [
    ("/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000", 578),
    ("/home/soenchun/cse/PointWorld_workspace/PointWorld/diagnostics/data/pointworld_behavior_task0001_restored/behavior/wds", 2490),
    ("/home/soenchun/cse/PointWorld_workspace/PointWorld/diagnostics/data/pointworld_behavior_multitask_restored/behavior/wds", 7184),
]

METRIC = Path(
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)
OUTDIR = Path("diagnostics/results/action_confound_control_5task")
OUTDIR.mkdir(parents=True, exist_ok=True)

def arr(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def keystr(x):
    return x.decode() if isinstance(x, bytes) else str(x)

def task_from_clip(x):
    m = re.search(r"task-\d{4}", str(x))
    return m.group(0) if m else None

def path_len(xyz):
    xyz = np.asarray(xyz, float)
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=-1).sum())

def quat_total_rotation(pose):
    pose = np.asarray(pose, float)
    if pose.ndim != 2 or pose.shape[1] < 7 or len(pose) < 2:
        return np.nan
    q = pose[:, 3:7]
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.clip(n, 1e-12, None)
    dots = np.abs(np.sum(q[:-1] * q[1:], axis=1))
    dots = np.clip(dots, 0.0, 1.0)
    return float(np.sum(2.0 * np.arccos(dots)))

def gripper_stats(g):
    g = np.asarray(g, float).reshape(-1)
    if len(g) < 2:
        return 0.0, 0.0
    return (
        float(np.sum(np.abs(np.diff(g)))),
        float(np.max(g) - np.min(g)),
    )

def safe_spearman(x, y):
    d = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(d) < 3 or d["x"].nunique() < 2:
        return np.nan, np.nan
    return spearmanr(d["x"], d["y"])

def extract(target):
    args0 = parse_args(skip_command_line=True)
    args0.seed = 42
    args0.batch_size = 1
    args0.num_workers = 4
    args0.eval_num_workers = 4
    args0.deterministic_data = True

    rows = []
    seen = set()

    for root, expected_n in ROOTS:
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

        for batch in tqdm(
            loader,
            total=info["batches_per_rank"],
            desc=f"Interaction: {Path(root).name}"
        ):
            clip = keystr(batch["__key__"][0])
            if clip not in target:
                continue
            assert clip not in seen

            task = task_from_clip(clip)

            rp = arr(batch["right_gripper_pose"][0])
            lp = arr(batch["left_gripper_pose"][0])
            rg = arr(batch["right_gripper_open"][0])
            lg = arr(batch["left_gripper_open"][0])

            rpath = path_len(rp[:, :3])
            lpath = path_len(lp[:, :3])

            if rpath >= lpath:
                dom_pose, dom_g, dominant = rp, rg, "right"
            else:
                dom_pose, dom_g, dominant = lp, lg, "left"

            gtv, grange = gripper_stats(dom_g)

            max_arm = max(rpath, lpath)
            arm_balance = (
                min(rpath, lpath) / max_arm if max_arm > 1e-12 else 0.0
            )

            base_path = np.nan
            if "base_pose" in batch:
                bp = arr(batch["base_pose"][0])
                if bp.ndim == 2 and bp.shape[1] >= 3:
                    base_path = path_len(bp[:, :3])

            rows.append({
                "clip": clip,
                "task_loaded": task,
                "dominant_hand": dominant,
                "right_path_m": rpath,
                "left_path_m": lpath,
                "arm_balance": arm_balance,
                "dominant_rotation_rad": quat_total_rotation(dom_pose),
                "dominant_gripper_total_variation": gtv,
                "dominant_gripper_range": grange,
                "base_path_m": base_path,
            })
            seen.add(clip)

    out = pd.DataFrame(rows)
    assert len(out) == 7987, len(out)
    return out

def main():
    m = pd.read_csv(METRIC)
    assert len(m) == 7987

    features = extract(set(m["clip"]))
    features.to_csv(
        OUTDIR / "task0010_interaction_features_7987.csv",
        index=False
    )

    df = m.merge(features, on="clip", validate="one_to_one")
    assert (df["task"] == df["task_loaded"]).all()

    candidates = [
        "dominant_gripper_total_variation",
        "dominant_gripper_range",
        "dominant_rotation_rad",
        "arm_balance",
        "base_path_m",
    ]

    rows = []
    for task, g in df.groupby("task"):
        for c in candidates:
            rho, p = safe_spearman(g[c], g["l2_moved_mm"])
            rows.append({
                "task": task,
                "feature": c,
                "N": int(g[[c, "l2_moved_mm"]].dropna().shape[0]),
                "median": float(g[c].median()),
                "rho_to_error": rho,
                "p": p,
            })

    out = pd.DataFrame(rows)
    out.to_csv(
        OUTDIR / "task0010_interaction_correlations.csv",
        index=False
    )

    print("\n=== TASK-0010: OTHER ACTION FEATURES -> ERROR ===")
    print(
        out[out.task == "task-0010"]
        .sort_values("rho_to_error", key=lambda s: s.abs(), ascending=False)
        .to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )

    print("\n=== TASKWISE MEDIANS ===")
    med = df.groupby("task")[candidates].median()
    print(med.to_string(float_format=lambda x: f"{x:.6g}"))

    print("\n=== FOR REFERENCE: RMS/MSJ ===")
    for task, g in df.groupby("task"):
        r1, p1 = safe_spearman(g["rms_acceleration"], g["l2_moved_mm"])
        r2, p2 = safe_spearman(g["mean_squared_jerk"], g["l2_moved_mm"])
        print(
            f"{task}: RMS rho={r1:+.4f}, p={p1:.3g} | "
            f"MSJ rho={r2:+.4f}, p={p2:.3g}"
        )

    print("\nOUTPUT =", OUTDIR.resolve())

if __name__ == "__main__":
    main()
