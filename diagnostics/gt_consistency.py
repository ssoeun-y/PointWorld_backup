import argparse
import csv
import json
from pathlib import Path

import numpy as np

from arguments import parse_args
from dataset_components.dataloader import build_dataloader


def to_np(x, index=0):
    x = x[index]
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_mask(x):
    x = np.asarray(x).astype(bool)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    return x


def describe(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    x = np.concatenate(values)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def collect(store, name, vectors, valid):
    mag = np.linalg.norm(vectors, axis=-1)
    values = mag[valid]
    if values.size:
        store.setdefault(name, []).append(values.astype(np.float64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_batches", type=int, default=-1)
    cli = parser.parse_args()

    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    args = parse_args(skip_command_line=True)
    args.domains = ["droid"]
    args.data_dirs = [str(Path(cli.data_dir).resolve())]
    args.batch_size = 1
    args.num_workers = 0
    args.eval_num_workers = 0
    args.seed = 42
    args.device = "cpu"
    args.deterministic_data = True

    loader, loader_info = build_dataloader(
        args,
        mode="test",
        rank=0,
        world_size=1,
        override_splits="test",
    )

    if loader is None:
        raise RuntimeError("DROID test dataloader 생성 실패")

    metric_values = {}
    coverage_values = []
    reversal_values = []
    rows = []

    for batch_idx, batch in enumerate(loader):
        if cli.max_batches > 0 and batch_idx >= cli.max_batches:
            break

        gt = to_np(batch["gt_scene_flows"]).astype(np.float64)
        exists = squeeze_mask(to_np(batch["scene_exists"]))
        supervised = squeeze_mask(to_np(batch["scene_supervised_mask"]))
        context = squeeze_mask(to_np(batch["scene_context_mask"]))
        static = squeeze_mask(to_np(batch["scene_static_mask"]))
        moved = squeeze_mask(to_np(batch["scene_moved_mask"]))

        valid = exists & supervised
        future = ~context

        key_raw = batch.get("__key__", [f"sample_{batch_idx:04d}"])
        key = str(key_raw[0])

        T, N = gt.shape[:2]
        initial_count = int(exists[0].sum())

        future_t = np.where(future.any(axis=1))[0]
        coverages = [
            float(valid[t].sum() / initial_count)
            for t in future_t
            if initial_count > 0
        ]
        coverage_values.extend(coverages)

        velocity = gt[1:] - gt[:-1]
        acceleration = gt[2:] - 2 * gt[1:-1] + gt[:-2]
        jerk = gt[3:] - 3 * gt[2:-1] + 3 * gt[1:-2] - gt[:-3]

        valid_v = valid[1:] & valid[:-1] & future[1:]
        valid_a = valid[2:] & valid[1:-1] & valid[:-2] & future[2:]
        valid_j = (
            valid[3:]
            & valid[2:-1]
            & valid[1:-2]
            & valid[:-3]
            & future[3:]
        )

        collect(metric_values, "all_velocity", velocity, valid_v)
        collect(metric_values, "all_acceleration", acceleration, valid_a)
        collect(metric_values, "all_jerk", jerk, valid_j)

        collect(
            metric_values,
            "static_velocity",
            velocity,
            valid_v & static[1:],
        )
        collect(
            metric_values,
            "static_acceleration",
            acceleration,
            valid_a & static[2:],
        )
        collect(
            metric_values,
            "static_jerk",
            jerk,
            valid_j & static[3:],
        )

        collect(
            metric_values,
            "moved_velocity",
            velocity,
            valid_v & moved[1:],
        )
        collect(
            metric_values,
            "moved_acceleration",
            acceleration,
            valid_a & moved[2:],
        )
        collect(
            metric_values,
            "moved_jerk",
            jerk,
            valid_j & moved[3:],
        )

        # 방향이 갑자기 반대로 바뀌는 비율
        if velocity.shape[0] >= 2:
            v_prev = velocity[:-1]
            v_next = velocity[1:]

            reversal_valid = (
                valid[2:]
                & valid[1:-1]
                & valid[:-2]
                & future[2:]
            )

            speed_prev = np.linalg.norm(v_prev, axis=-1)
            speed_next = np.linalg.norm(v_next, axis=-1)

            reversal_valid &= speed_prev > 0.001
            reversal_valid &= speed_next > 0.001

            dot = np.sum(v_prev * v_next, axis=-1)

            if reversal_valid.any():
                reversal_values.append(
                    float(np.mean(dot[reversal_valid] < 0))
                )

        final_drift_p95 = None

        if len(future_t):
            final_t = int(future_t[-1])
            always_static = static[future_t].all(axis=0)
            drift_valid = always_static & valid[0] & valid[final_t]

            final_drift = np.linalg.norm(
                gt[final_t] - gt[0],
                axis=-1,
            )

            if drift_valid.any():
                drift_values = final_drift[drift_valid]
                metric_values.setdefault(
                    "always_static_final_drift", []
                ).append(drift_values)

                final_drift_p95 = float(
                    np.quantile(drift_values, 0.95)
                )

        row = {
            "key": key,
            "timesteps": T,
            "points": N,
            "coverage_mean": (
                float(np.mean(coverages)) if coverages else None
            ),
            "coverage_min": (
                float(np.min(coverages)) if coverages else None
            ),
            "coverage_final": (
                float(coverages[-1]) if coverages else None
            ),
            "always_static_final_drift_p95": final_drift_p95,
        }
        rows.append(row)

        print(
            f"[{len(rows):03d}] {key} "
            f"T={T} N={N} "
            f"coverage={row['coverage_mean']}"
        )

    if not rows:
        raise RuntimeError("분석된 sample이 없음")

    summary = {
        "processed_sequences": len(rows),
        "loader_info": loader_info,
        "units": {
            "velocity": "meter/timestep",
            "acceleration": "meter/timestep^2",
            "jerk": "meter/timestep^3",
            "drift": "meter",
        },
        "coverage": describe(
            [np.asarray(coverage_values)]
            if coverage_values else []
        ),
        "direction_reversal_fraction": {
            "mean_across_sequences": (
                float(np.mean(reversal_values))
                if reversal_values else None
            ),
            "median_across_sequences": (
                float(np.median(reversal_values))
                if reversal_values else None
            ),
        },
        "metrics": {
            name: describe(values)
            for name, values in metric_values.items()
        },
    }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out / "per_sequence.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== GT CONSISTENCY SUMMARY =====")
    print("processed_sequences:", len(rows))
    print("coverage:", summary["coverage"])
    print(
        "direction_reversal:",
        summary["direction_reversal_fraction"],
    )

    for name, result in summary["metrics"].items():
        print(name, result)

    print("summary_json:", out / "summary.json")
    print("per_sequence_csv:", out / "per_sequence.csv")


if __name__ == "__main__":
    main()
