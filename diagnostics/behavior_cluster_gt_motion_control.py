#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnostics.behavior_cluster_relative_dynamics_full578 import (
    get_components,
)

QUANTILES = (0.90, 0.95, 0.975)


def edge_median_relative_motion(
    motion,
    valid,
    edges,
):
    vals = []

    for i, j in edges:
        if not (valid[i] and valid[j]):
            continue

        vals.append(
            float(
                torch.linalg.vector_norm(
                    motion[i] - motion[j]
                )
            )
        )

    if not vals:
        return np.nan

    return float(np.median(vals))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--trajectory_dir",
        required=True,
    )
    ap.add_argument(
        "--debris_csv",
        required=True,
    )

    args = ap.parse_args()

    trajectory_dir = Path(
        args.trajectory_dir
    )

    debris = pd.read_csv(
        args.debris_csv
    )

    key_to_path = {}

    for f in trajectory_dir.glob("*.npz"):
        with np.load(
            f,
            allow_pickle=False,
        ) as z:
            key_to_path[
                str(z["sample_key"])
            ] = f

    rows = []

    grouped = debris.groupby(
        ["key", "object_id"],
        sort=True,
    )

    for (key, object_id), group in grouped:
        object_id = int(object_id)

        with np.load(
            key_to_path[str(key)],
            allow_pickle=False,
        ) as z:
            ids = z["scene_object_id"]
            valid = z[
                "official_valid_mask"
            ]
            gt_np = z[
                "gt_scene_flows"
            ]

        global_idx = (
            group.sort_values(
                "global_point_index"
            )["global_point_index"]
            .to_numpy(dtype=np.int64)
        )

        scores = (
            group.sort_values(
                "global_point_index"
            )["outward_stretch_m"]
            .to_numpy(dtype=float)
        )

        expected = np.where(
            (ids[0] == object_id)
            & valid[-1]
        )[0]

        assert np.array_equal(
            global_idx,
            expected,
        )

        gt = torch.from_numpy(
            gt_np[:, global_idx]
        ).float()

        valid_local = valid[
            :, global_idx
        ].astype(bool)

        n = len(global_idx)
        k = min(8, n - 1)

        d0 = torch.cdist(
            gt[0].unsqueeze(0),
            gt[0].unsqueeze(0),
        ).squeeze(0)

        d0.fill_diagonal_(
            float("inf")
        )

        knn = torch.topk(
            d0,
            k=k,
            largest=False,
            dim=1,
        ).indices.cpu().numpy()

        adj = [
            set() for _ in range(n)
        ]
        edges = set()

        for i in range(n):
            for j in map(int, knn[i]):
                adj[i].add(j)
                adj[j].add(i)

                a, b = sorted((i, j))
                edges.add((a, b))

        for q in QUANTILES:
            threshold = np.quantile(
                scores,
                q,
            )

            anomalous = np.where(
                scores >= threshold
            )[0]

            cluster = set(
                get_components(
                    anomalous,
                    adj,
                )[0]
            )

            boundary = [
                e for e in edges
                if (
                    (e[0] in cluster)
                    != (e[1] in cluster)
                )
            ]

            normal = [
                e for e in edges
                if (
                    e[0] not in cluster
                    and e[1] not in cluster
                )
            ]

            vel_gaps = []
            acc_gaps = []

            for t in range(1, 11):
                pair_valid = (
                    valid_local[t - 1]
                    & valid_local[t]
                )

                v_gt = (
                    gt[t]
                    - gt[t - 1]
                )

                b_vel = (
                    edge_median_relative_motion(
                        v_gt,
                        pair_valid,
                        boundary,
                    )
                )

                n_vel = (
                    edge_median_relative_motion(
                        v_gt,
                        pair_valid,
                        normal,
                    )
                )

                if (
                    np.isfinite(b_vel)
                    and np.isfinite(n_vel)
                ):
                    vel_gaps.append(
                        b_vel - n_vel
                    )

                if t >= 2:
                    triple_valid = (
                        valid_local[t - 2]
                        & valid_local[t - 1]
                        & valid_local[t]
                    )

                    a_gt = (
                        (gt[t] - gt[t - 1])
                        - (
                            gt[t - 1]
                            - gt[t - 2]
                        )
                    )

                    b_acc = (
                        edge_median_relative_motion(
                            a_gt,
                            triple_valid,
                            boundary,
                        )
                    )

                    n_acc = (
                        edge_median_relative_motion(
                            a_gt,
                            triple_valid,
                            normal,
                        )
                    )

                    if (
                        np.isfinite(b_acc)
                        and np.isfinite(n_acc)
                    ):
                        acc_gaps.append(
                            b_acc - n_acc
                        )

            rows.append({
                "key": key,
                "object_id":
                    object_id,
                "quantile": q,

                "gt_relvel_gap_mean_m":
                    np.mean(vel_gaps)
                    if vel_gaps
                    else np.nan,

                "gt_relacc_gap_mean_m":
                    np.mean(acc_gaps)
                    if acc_gaps
                    else np.nan,
            })

    df = pd.DataFrame(rows)

    for q in QUANTILES:
        x = df[
            df["quantile"] == q
        ]

        print(
            "\n" + "=" * 72
        )
        print("Q =", q)
        print("objects =", len(x))

        for col in [
            "gt_relvel_gap_mean_m",
            "gt_relacc_gap_mean_m",
        ]:
            v = x[col].dropna().to_numpy()

            print("\n", col)
            print(
                "mean_mm =",
                np.mean(v) * 1000,
            )
            print(
                "median_mm =",
                np.median(v) * 1000,
            )
            print(
                "positive_fraction =",
                np.mean(v > 0),
            )


if __name__ == "__main__":
    main()
