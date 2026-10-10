#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnostics.behavior_cluster_relative_dynamics_full578 import (
    get_components,
)


QUANTILES = (0.90, 0.95, 0.975)


def edge_values(value, valid_mask, edges):
    """Return GT relative magnitude and prediction-error-ready edge indices."""
    vals = []
    kept_edges = []

    for i, j in edges:
        if not (valid_mask[i] and valid_mask[j]):
            continue

        vals.append(
            float(
                torch.linalg.vector_norm(
                    value[i] - value[j]
                )
            )
        )
        kept_edges.append((i, j))

    return (
        np.asarray(vals, dtype=np.float64),
        kept_edges,
    )


def prediction_relative_error(
    gt_value,
    pred_value,
    edges,
):
    vals = []

    for i, j in edges:
        gt_rel = (
            gt_value[i]
            - gt_value[j]
        )
        pred_rel = (
            pred_value[i]
            - pred_value[j]
        )

        vals.append(
            float(
                torch.linalg.vector_norm(
                    pred_rel - gt_rel
                )
            )
        )

    return np.asarray(
        vals,
        dtype=np.float64,
    )


def greedy_unique_match(
    boundary_gt,
    normal_gt,
):
    """
    1:1 nearest matching WITHOUT replacement.

    Match smallest-gap pair iteratively.
    Boundary count is normally much smaller
    than normal-edge count.
    """
    if (
        len(boundary_gt) == 0
        or len(normal_gt) == 0
    ):
        return []

    candidates = []

    for bi, b in enumerate(boundary_gt):
        gaps = np.abs(normal_gt - b)

        # Keep candidate normals sorted by closeness.
        order = np.argsort(gaps)

        for ni in order:
            candidates.append(
                (
                    float(gaps[ni]),
                    int(bi),
                    int(ni),
                )
            )

    candidates.sort(
        key=lambda x: x[0]
    )

    used_b = set()
    used_n = set()
    matches = []

    for gap, bi, ni in candidates:
        if bi in used_b:
            continue
        if ni in used_n:
            continue

        used_b.add(bi)
        used_n.add(ni)

        matches.append(
            (bi, ni, gap)
        )

        if len(used_b) == len(boundary_gt):
            break

    return matches


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
        "--output_root",
        default="diagnostics/results",
    )

    args = ap.parse_args()

    trajectory_dir = Path(
        args.trajectory_dir
    )
    debris = pd.read_csv(
        args.debris_csv
    )

    assert trajectory_dir.is_dir()

    # Key -> trajectory path.
    key_to_path = {}

    for f in trajectory_dir.glob("*.npz"):
        with np.load(
            f,
            allow_pickle=False,
        ) as z:
            key = str(
                z["sample_key"]
            )

        if key in key_to_path:
            raise RuntimeError(
                f"duplicate key: {key}"
            )

        key_to_path[key] = f

    assert len(key_to_path) == 578

    rows = []
    processed = 0

    grouped = debris.groupby(
        ["key", "object_id"],
        sort=True,
    )

    for (key, object_id), group in grouped:
        key = str(key)
        object_id = int(object_id)

        with np.load(
            key_to_path[key],
            allow_pickle=False,
        ) as z:
            ids = z[
                "scene_object_id"
            ]
            valid = z[
                "official_valid_mask"
            ]
            gt_np = z[
                "gt_scene_flows"
            ]
            pred_np = z[
                "pred_scene_flows"
            ]

        group = group.sort_values(
            "global_point_index"
        )

        global_idx = group[
            "global_point_index"
        ].to_numpy(
            dtype=np.int64
        )

        scores = group[
            "outward_stretch_m"
        ].to_numpy(
            dtype=np.float64
        )

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

        pred = torch.from_numpy(
            pred_np[:, global_idx]
        ).float()

        valid_local = (
            valid[:, global_idx]
            .astype(bool)
        )

        n = len(global_idx)
        k = min(8, n - 1)

        # Same GT-t0 kNN definition.
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
            for j in map(
                int,
                knn[i],
            ):
                adj[i].add(j)
                adj[j].add(i)

                a, b = sorted(
                    (i, j)
                )
                edges.add((a, b))

        for q in QUANTILES:
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
                e
                for e in edges
                if (
                    (e[0] in cluster)
                    != (e[1] in cluster)
                )
            ])

            normal_edges = sorted([
                e
                for e in edges
                if (
                    e[0] not in cluster
                    and e[1] not in cluster
                )
            ])

            # ---------------------------------
            # Velocity matching
            # ---------------------------------
            for t in range(1, 11):

                pair_valid = (
                    valid_local[t - 1]
                    & valid_local[t]
                )

                gt_vel = (
                    gt[t]
                    - gt[t - 1]
                )

                pred_vel = (
                    pred[t]
                    - pred[t - 1]
                )

                b_gt, b_edges = edge_values(
                    gt_vel,
                    pair_valid,
                    boundary_edges,
                )

                n_gt, n_edges = edge_values(
                    gt_vel,
                    pair_valid,
                    normal_edges,
                )

                matches = (
                    greedy_unique_match(
                        b_gt,
                        n_gt,
                    )
                )

                if matches:
                    b_sel = [
                        b_edges[m[0]]
                        for m in matches
                    ]
                    n_sel = [
                        n_edges[m[1]]
                        for m in matches
                    ]

                    b_gt_sel = np.asarray([
                        b_gt[m[0]]
                        for m in matches
                    ])
                    n_gt_sel = np.asarray([
                        n_gt[m[1]]
                        for m in matches
                    ])

                    match_abs_gap = np.abs(
                        b_gt_sel
                        - n_gt_sel
                    )

                    b_err = (
                        prediction_relative_error(
                            gt_vel,
                            pred_vel,
                            b_sel,
                        )
                    )

                    n_err = (
                        prediction_relative_error(
                            gt_vel,
                            pred_vel,
                            n_sel,
                        )
                    )

                    rows.append({
                        "key": key,
                        "object_id":
                            object_id,
                        "quantile": q,
                        "metric":
                            "velocity",
                        "timestep": t,

                        "matched_pairs":
                            len(matches),

                        "boundary_gt_median_m":
                            float(
                                np.median(
                                    b_gt_sel
                                )
                            ),

                        "matched_normal_gt_median_m":
                            float(
                                np.median(
                                    n_gt_sel
                                )
                            ),

                        "gt_match_abs_gap_mean_m":
                            float(
                                np.mean(
                                    match_abs_gap
                                )
                            ),

                        "gt_match_abs_gap_median_m":
                            float(
                                np.median(
                                    match_abs_gap
                                )
                            ),

                        "boundary_pred_error_median_m":
                            float(
                                np.median(
                                    b_err
                                )
                            ),

                        "matched_normal_pred_error_median_m":
                            float(
                                np.median(
                                    n_err
                                )
                            ),

                        "matched_error_gap_m":
                            float(
                                np.median(
                                    b_err
                                )
                                - np.median(
                                    n_err
                                )
                            ),
                    })

                # -----------------------------
                # Acceleration matching
                # -----------------------------
                if t < 2:
                    continue

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

                pred_acc = (
                    (pred[t] - pred[t - 1])
                    - (
                        pred[t - 1]
                        - pred[t - 2]
                    )
                )

                b_gt, b_edges = edge_values(
                    gt_acc,
                    triple_valid,
                    boundary_edges,
                )

                n_gt, n_edges = edge_values(
                    gt_acc,
                    triple_valid,
                    normal_edges,
                )

                matches = (
                    greedy_unique_match(
                        b_gt,
                        n_gt,
                    )
                )

                if matches:
                    b_sel = [
                        b_edges[m[0]]
                        for m in matches
                    ]
                    n_sel = [
                        n_edges[m[1]]
                        for m in matches
                    ]

                    b_gt_sel = np.asarray([
                        b_gt[m[0]]
                        for m in matches
                    ])
                    n_gt_sel = np.asarray([
                        n_gt[m[1]]
                        for m in matches
                    ])

                    match_abs_gap = np.abs(
                        b_gt_sel
                        - n_gt_sel
                    )

                    b_err = (
                        prediction_relative_error(
                            gt_acc,
                            pred_acc,
                            b_sel,
                        )
                    )

                    n_err = (
                        prediction_relative_error(
                            gt_acc,
                            pred_acc,
                            n_sel,
                        )
                    )

                    rows.append({
                        "key": key,
                        "object_id":
                            object_id,
                        "quantile": q,
                        "metric":
                            "acceleration",
                        "timestep": t,

                        "matched_pairs":
                            len(matches),

                        "boundary_gt_median_m":
                            float(
                                np.median(
                                    b_gt_sel
                                )
                            ),

                        "matched_normal_gt_median_m":
                            float(
                                np.median(
                                    n_gt_sel
                                )
                            ),

                        "gt_match_abs_gap_mean_m":
                            float(
                                np.mean(
                                    match_abs_gap
                                )
                            ),

                        "gt_match_abs_gap_median_m":
                            float(
                                np.median(
                                    match_abs_gap
                                )
                            ),

                        "boundary_pred_error_median_m":
                            float(
                                np.median(
                                    b_err
                                )
                            ),

                        "matched_normal_pred_error_median_m":
                            float(
                                np.median(
                                    n_err
                                )
                            ),

                        "matched_error_gap_m":
                            float(
                                np.median(
                                    b_err
                                )
                                - np.median(
                                    n_err
                                )
                            ),
                    })

        processed += 1

        if processed % 50 == 0:
            print(
                "processed_objects =",
                processed,
            )

    assert processed == 377

    result = pd.DataFrame(rows)

    out_dir = (
        Path(args.output_root)
        / (
            "behavior_cluster_gt_matched_control_full578_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    result.to_csv(
        out_dir
        / "matched_control.csv",
        index=False,
    )

    print()
    print("=" * 72)
    print("GT-MATCHED CONTROL READY")
    print(
        "processed_objects =",
        processed,
    )
    print(
        "rows =",
        len(result),
    )
    print(
        "output_dir =",
        out_dir,
    )
    print("SUCCESS = True")
    print("=" * 72)


if __name__ == "__main__":
    main()
