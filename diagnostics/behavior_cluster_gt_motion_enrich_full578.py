#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnostics.behavior_cluster_relative_dynamics_full578 import (
    get_components,
)


def edge_median_relative_value(
    value,
    valid_mask,
    edges,
):
    vals = []

    for i, j in edges:
        if not (
            valid_mask[i]
            and valid_mask[j]
        ):
            continue

        vals.append(
            float(
                torch.linalg.vector_norm(
                    value[i] - value[j]
                )
            )
        )

    if not vals:
        return np.nan, 0

    return float(np.median(vals)), len(vals)


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
    ap.add_argument(
        "--dynamics_csv",
        required=True,
    )
    ap.add_argument(
        "--output_csv",
        required=True,
    )

    args = ap.parse_args()

    trajectory_dir = Path(
        args.trajectory_dir
    )

    debris = pd.read_csv(
        args.debris_csv
    )

    dyn = pd.read_csv(
        args.dynamics_csv
    )

    key_to_path = {}

    for f in trajectory_dir.glob("*.npz"):
        with np.load(
            f,
            allow_pickle=False,
        ) as z:
            key = str(z["sample_key"])

        if key in key_to_path:
            raise RuntimeError(
                f"duplicate key: {key}"
            )

        key_to_path[key] = f

    assert len(key_to_path) == 578

    gt_rows = []

    grouped = debris.groupby(
        ["key", "object_id"],
        sort=True,
    )

    processed = 0

    for (key, object_id), group in grouped:
        key = str(key)
        object_id = int(object_id)

        assert key in key_to_path

        with np.load(
            key_to_path[key],
            allow_pickle=False,
        ) as z:
            ids = z["scene_object_id"]
            valid = z[
                "official_valid_mask"
            ]
            gt_np = z[
                "gt_scene_flows"
            ]

        group = group.sort_values(
            "global_point_index"
        )

        global_idx = group[
            "global_point_index"
        ].to_numpy(dtype=np.int64)

        scores = group[
            "outward_stretch_m"
        ].to_numpy(dtype=float)

        expected_idx = np.where(
            (ids[0] == object_id)
            & valid[-1]
        )[0]

        if not np.array_equal(
            global_idx,
            expected_idx,
        ):
            raise RuntimeError(
                f"[{key} obj={object_id}] "
                "point population mismatch"
            )

        gt = torch.from_numpy(
            gt_np[:, global_idx]
        ).float()

        valid_local = valid[
            :, global_idx
        ].astype(bool)

        n = len(global_idx)
        k = min(8, n - 1)

        # Same GT-t0 kNN used in debris analysis.
        d0 = torch.cdist(
            gt[0].unsqueeze(0),
            gt[0].unsqueeze(0),
            p=2,
        ).squeeze(0)

        d0.fill_diagonal_(
            float("inf")
        )

        knn = torch.topk(
            d0,
            k=k,
            dim=1,
            largest=False,
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

        for q in (0.90, 0.95, 0.975):
            threshold = float(
                np.quantile(
                    scores,
                    q,
                )
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

            boundary_edges = sorted([
                (i, j)
                for i, j in edges
                if (
                    (i in cluster)
                    != (j in cluster)
                )
            ])

            normal_edges = sorted([
                (i, j)
                for i, j in edges
                if (
                    i not in cluster
                    and j not in cluster
                )
            ])

            for t in range(1, 11):
                # -----------------------------
                # GT relative velocity
                # step = t-1 -> t
                # -----------------------------
                pair_valid = (
                    valid_local[t - 1]
                    & valid_local[t]
                )

                gt_vel = (
                    gt[t]
                    - gt[t - 1]
                )

                b_vel, b_vel_n = (
                    edge_median_relative_value(
                        gt_vel,
                        pair_valid,
                        boundary_edges,
                    )
                )

                n_vel, n_vel_n = (
                    edge_median_relative_value(
                        gt_vel,
                        pair_valid,
                        normal_edges,
                    )
                )

                if (
                    np.isfinite(b_vel)
                    and np.isfinite(n_vel)
                ):
                    gt_relvel_gap = (
                        b_vel - n_vel
                    )
                else:
                    gt_relvel_gap = np.nan

                # -----------------------------
                # GT relative acceleration
                # centered on timestep t
                # -----------------------------
                if t >= 2:
                    triple_valid = (
                        valid_local[t - 2]
                        & valid_local[t - 1]
                        & valid_local[t]
                    )

                    gt_acc = (
                        (gt[t] - gt[t - 1])
                        - (
                            gt[t - 1]
                            - gt[t - 2]
                        )
                    )

                    b_acc, b_acc_n = (
                        edge_median_relative_value(
                            gt_acc,
                            triple_valid,
                            boundary_edges,
                        )
                    )

                    n_acc, n_acc_n = (
                        edge_median_relative_value(
                            gt_acc,
                            triple_valid,
                            normal_edges,
                        )
                    )

                    if (
                        np.isfinite(b_acc)
                        and np.isfinite(n_acc)
                    ):
                        gt_relacc_gap = (
                            b_acc - n_acc
                        )
                    else:
                        gt_relacc_gap = np.nan

                else:
                    b_acc = np.nan
                    n_acc = np.nan
                    gt_relacc_gap = np.nan
                    b_acc_n = 0
                    n_acc_n = 0

                gt_rows.append({
                    "key": key,
                    "object_id": object_id,
                    "quantile": q,
                    "timestep": t,

                    "boundary_gt_relvel_median_m":
                        b_vel,
                    "normal_gt_relvel_median_m":
                        n_vel,
                    "gt_relvel_gap_m":
                        gt_relvel_gap,
                    "boundary_gt_relvel_n":
                        b_vel_n,
                    "normal_gt_relvel_n":
                        n_vel_n,

                    "boundary_gt_relacc_median_m":
                        b_acc,
                    "normal_gt_relacc_median_m":
                        n_acc,
                    "gt_relacc_gap_m":
                        gt_relacc_gap,
                    "boundary_gt_relacc_n":
                        b_acc_n,
                    "normal_gt_relacc_n":
                        n_acc_n,
                })

        processed += 1

        if processed % 50 == 0:
            print(
                "processed_objects =",
                processed,
            )

    gt_df = pd.DataFrame(gt_rows)

    assert processed == 377
    assert len(gt_df) == 377 * 3 * 10
    assert len(dyn) == 377 * 3 * 10

    merge_cols = [
        "key",
        "object_id",
        "quantile",
        "timestep",
    ]

    enriched = dyn.merge(
        gt_df,
        on=merge_cols,
        how="left",
        validate="one_to_one",
    )

    if len(enriched) != len(dyn):
        raise RuntimeError(
            "row count changed after merge"
        )

    output = Path(args.output_csv)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 72)
    print("GT MOTION ENRICH DONE")
    print("processed_objects =", processed)
    print("input_rows =", len(dyn))
    print("gt_rows =", len(gt_df))
    print("output_rows =", len(enriched))
    print("output =", output)
    print("SUCCESS = True")
    print("=" * 72)


if __name__ == "__main__":
    main()
