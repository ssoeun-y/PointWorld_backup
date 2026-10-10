from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from arguments import parse_args
from dataset_components.dataloader import build_dataloader
from dataset_components.transforms import compute_flow_derivatives

COHORT = Path(
    "diagnostics/results/intervention_high_msj_top10_4task_meta.csv"
)

ROOTS = [
    (
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
        578,
    ),
    (
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/diagnostics/data/"
        "pointworld_behavior_task0001_restored/behavior/wds",
        2490,
    ),
    (
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/diagnostics/data/"
        "pointworld_behavior_multitask_restored/behavior/wds",
        7184,
    ),
]

OUT = Path(
    "diagnostics/results/action_intervention_633/"
    "arclength_action_metric_audit.csv"
)
OUT.parent.mkdir(parents=True, exist_ok=True)


def npy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def keystr(x):
    if isinstance(x, bytes):
        return x.decode()
    return str(x)


def path_length(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.linalg.norm(np.diff(x, axis=0), axis=-1).sum())


def dominant_xyz(batch):
    r = npy(batch["right_gripper_pose"][0])[:, :3]
    l = npy(batch["left_gripper_pose"][0])[:, :3]

    if path_length(r) >= path_length(l):
        return r
    return l


# Arc-length / phase canonicalization.
# Preserve the same geometric path and endpoints, but remove
# unnecessary temporal speed variation along that path.
def arclength_time_map(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    T = len(xyz)
    u = np.linspace(0.0, 1.0, T)

    if T < 2:
        return u

    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    total = float(cum[-1])
    if total <= 1e-12:
        return u

    phase = cum / total

    # Handle repeated positions / dwell periods.
    phase_unique, idx = np.unique(phase, return_index=True)
    u_unique = u[idx].copy()

    if len(phase_unique) < 2:
        return u

    # Preserve exact temporal endpoints.
    u_unique[0] = 0.0
    u_unique[-1] = 1.0

    desired_phase = np.linspace(0.0, 1.0, T)
    time_map = np.interp(
        desired_phase,
        phase_unique,
        u_unique,
    )

    time_map[0] = 0.0
    time_map[-1] = 1.0

    if np.any(np.diff(time_map) < -1e-12):
        raise RuntimeError("non-monotonic arc-length time map")

    return time_map

def resample_time(arr, s):
    arr = np.asarray(arr)
    T = arr.shape[0]
    u = np.linspace(0.0, 1.0, T)

    old_shape = arr.shape
    flat = arr.astype(np.float64).reshape(T, -1)
    out = np.empty_like(flat)

    for j in range(flat.shape[1]):
        out[:, j] = np.interp(s, u, flat[:, j])

    return out.reshape(old_shape)


def rms_acceleration(xyz):
    # Same PointWorld derivative operator used previously.
    _, acc = compute_flow_derivatives(
        xyz[:, None, :].astype(np.float32)
    )
    acc = np.asarray(acc)[:, 0, :]
    mag = np.linalg.norm(acc, axis=-1)
    return float(np.sqrt(np.mean(mag ** 2)))


def mean_squared_jerk(xyz):
    # Same dt=1 frame convention used in established-metric analysis.
    velocity = np.diff(
        np.asarray(xyz, dtype=np.float64),
        axis=0
    )

    if len(velocity) < 3:
        return np.nan

    jerk = np.diff(
        velocity,
        n=2,
        axis=0
    )

    return float(
        np.mean(
            np.sum(jerk ** 2, axis=-1)
        )
    )


def main():
    cohort = pd.read_csv(COHORT)
    targets = set(cohort["clip"])

    print("TARGET =", len(targets))
    assert len(targets) == 633

    task_by_clip = dict(zip(cohort["clip"], cohort["task"]))

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
            desc=f"Audit {Path(root).name}",
        ):
            key = keystr(batch["__key__"][0])

            if key not in targets:
                continue

            if key in seen:
                raise RuntimeError(f"duplicate: {key}")

            xyz = dominant_xyz(batch)
            s = arclength_time_map(xyz)
            xyz_s = resample_time(xyz, s)

            # Spatial endpoints MUST be identical.
            assert np.allclose(
                xyz_s[0], xyz[0], atol=1e-10, rtol=0
            )
            assert np.allclose(
                xyz_s[-1], xyz[-1], atol=1e-10, rtol=0
            )

            rms0 = rms_acceleration(xyz)
            rms1 = rms_acceleration(xyz_s)

            msj0 = mean_squared_jerk(xyz)
            msj1 = mean_squared_jerk(xyz_s)

            rows.append({
                "clip": key,
                "task": task_by_clip[key],
                "rms_original": rms0,
                "rms_smooth": rms1,
                "rms_ratio": rms1 / rms0 if rms0 > 0 else np.nan,
                "msj_original": msj0,
                "msj_smooth": msj1,
                "msj_ratio": msj1 / msj0 if msj0 > 0 else np.nan,
                "rms_reduced": rms1 < rms0,
                "msj_reduced": msj1 < msj0,
                "path_original": path_length(xyz),
                "path_canonical": path_length(xyz_s),
                "path_ratio": (
                    path_length(xyz_s) / path_length(xyz)
                    if path_length(xyz) > 1e-12 else np.nan
                ),
                "endpoint_error": float(
                    np.linalg.norm(xyz_s[-1] - xyz[-1])
                ),
            })

            seen.add(key)

    if seen != targets:
        missing = targets - seen
        raise RuntimeError(
            f"Missing {len(missing)} cohort clips"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n=== ARC-LENGTH INTERVENTION ACTION AUDIT ===")
    print("N =", len(df))
    print(
        "RMS median original -> smooth =",
        df.rms_original.median(),
        "->",
        df.rms_smooth.median()
    )
    print(
        "RMS median ratio =",
        df.rms_ratio.median()
    )
    print(
        "RMS reduced fraction =",
        df.rms_reduced.mean()
    )

    print(
        "MSJ median original -> smooth =",
        df.msj_original.median(),
        "->",
        df.msj_smooth.median()
    )
    print(
        "MSJ median ratio =",
        df.msj_ratio.median()
    )
    print(
        "MSJ reduced fraction =",
        df.msj_reduced.mean()
    )

    print(
        "Path median ratio =",
        df.path_ratio.median()
    )
    print(
        "Endpoint max error =",
        df.endpoint_error.max()
    )

    print("\n=== TASKWISE ===")
    print(
        df.groupby("task").agg(
            N=("clip", "size"),
            rms_ratio_median=("rms_ratio", "median"),
            rms_reduced_frac=("rms_reduced", "mean"),
            msj_ratio_median=("msj_ratio", "median"),
            msj_reduced_frac=("msj_reduced", "mean"),
        ).to_string()
    )

    print("\nsaved =", OUT)


if __name__ == "__main__":
    main()
