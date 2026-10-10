#!/usr/bin/env python3

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


QUANTILES = (0.90, 0.95, 0.975)


def get_components(nodes, adjacency):
    nodes = set(map(int, nodes))
    seen = set()
    comps = []

    for start in sorted(nodes):
        if start in seen:
            continue

        stack = [start]
        seen.add(start)
        comp = []

        while stack:
            u = stack.pop()
            comp.append(u)

            for v in adjacency[u]:
                if v in nodes and v not in seen:
                    seen.add(v)
                    stack.append(v)

        comps.append(sorted(comp))

    comps.sort(key=len, reverse=True)
    return comps


def edge_index_arrays(edges):
    if len(edges) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    return (
        np.asarray([e[0] for e in edges], dtype=np.int64),
        np.asarray([e[1] for e in edges], dtype=np.int64),
    )


def geometry_metrics(
    gt_t,
    pred_t,
    valid_t,
    ii,
    jj,
):
    if len(ii) == 0:
        return {
            "n": 0,
            "median_delta_m": np.nan,
            "mean_delta_m": np.nan,
            "stretched_fraction": np.nan,
            "pred_gt_ratio": np.nan,
        }

    ok = valid_t[ii] & valid_t[jj]
    ii = ii[ok]
    jj = jj[ok]

    if len(ii) == 0:
        return {
            "n": 0,
            "median_delta_m": np.nan,
            "mean_delta_m": np.nan,
            "stretched_fraction": np.nan,
            "pred_gt_ratio": np.nan,
        }

    ii_t = torch.from_numpy(ii).long()
    jj_t = torch.from_numpy(jj).long()

    gt_len = torch.linalg.vector_norm(
        gt_t[ii_t] - gt_t[jj_t],
        dim=-1,
    )
    pred_len = torch.linalg.vector_norm(
        pred_t[ii_t] - pred_t[jj_t],
        dim=-1,
    )

    delta = pred_len - gt_len

    return {
        "n": int(len(ii)),
        "median_delta_m": float(delta.median()),
        "mean_delta_m": float(delta.mean()),
        "stretched_fraction": float(
            (delta > 0).float().mean()
        ),
        "pred_gt_ratio": float(
            pred_len.mean() / gt_len.mean()
        ),
    }


def relative_velocity_error(
    gt_prev,
    gt_now,
    pred_prev,
    pred_now,
    pair_valid,
    ii,
    jj,
):
    if len(ii) == 0:
        return 0, np.nan

    ok = pair_valid[ii] & pair_valid[jj]
    ii = ii[ok]
    jj = jj[ok]

    if len(ii) == 0:
        return 0, np.nan

    ii_t = torch.from_numpy(ii).long()
    jj_t = torch.from_numpy(jj).long()

    v_gt = gt_now - gt_prev
    v_pred = pred_now - pred_prev

    gt_rel = v_gt[ii_t] - v_gt[jj_t]
    pred_rel = v_pred[ii_t] - v_pred[jj_t]

    err = torch.linalg.vector_norm(
        pred_rel - gt_rel,
        dim=-1,
    )

    return int(len(ii)), float(err.median())


def relative_acceleration_error(
    gt0,
    gt1,
    gt2,
    pred0,
    pred1,
    pred2,
    triple_valid,
    ii,
    jj,
):
    if len(ii) == 0:
        return 0, np.nan

    ok = triple_valid[ii] & triple_valid[jj]
    ii = ii[ok]
    jj = jj[ok]

    if len(ii) == 0:
        return 0, np.nan

    ii_t = torch.from_numpy(ii).long()
    jj_t = torch.from_numpy(jj).long()

    a_gt = (
        (gt2 - gt1)
        - (gt1 - gt0)
    )
    a_pred = (
        (pred2 - pred1)
        - (pred1 - pred0)
    )

    gt_rel = a_gt[ii_t] - a_gt[jj_t]
    pred_rel = a_pred[ii_t] - a_pred[jj_t]

    err = torch.linalg.vector_norm(
        pred_rel - gt_rel,
        dim=-1,
    )

    return int(len(ii)), float(err.median())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trajectory_dir",
        required=True,
    )
    parser.add_argument(
        "--debris_csv",
        required=True,
    )
    parser.add_argument(
        "--output_root",
        default="diagnostics/results",
    )

    args = parser.parse_args()

    trajectory_dir = Path(args.trajectory_dir)
    debris_csv = Path(args.debris_csv)

    assert trajectory_dir.is_dir()
    assert debris_csv.is_file()

    out_dir = (
        Path(args.output_root)
        / (
            "behavior_cluster_relative_dynamics_full578_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # -------------------------------------------------
    # Validate trajectory files and build key -> path.
    # -------------------------------------------------
    npz_files = sorted(
        trajectory_dir.glob("*.npz")
    )

    key_to_path = {}
    all_l2_exact = True
    all_temporal_ids_exact = True

    for path in npz_files:
        with np.load(
            path,
            allow_pickle=False,
        ) as z:
            key = str(z["sample_key"])

            if key in key_to_path:
                raise RuntimeError(
                    f"duplicate trajectory key: {key}"
                )

            key_to_path[key] = path

            all_l2_exact &= bool(
                z["official_l2_exact"]
            )
            all_temporal_ids_exact &= bool(
                z["temporal_object_id_exact"]
            )

    # -------------------------------------------------
    # Existing independently defined outward-stretch.
    # No L2 is used for cluster definition here.
    # -------------------------------------------------
    df = pd.read_csv(debris_csv)

    required_columns = {
        "key",
        "object_id",
        "object_name",
        "global_point_index",
        "outward_stretch_m",
    }

    missing_cols = (
        required_columns
        - set(df.columns)
    )

    if missing_cols:
        raise RuntimeError(
            f"missing debris columns: {missing_cols}"
        )

    grouped = list(
        df.groupby(
            ["key", "object_id"],
            sort=True,
        )
    )

    object_rows = []
    timestep_rows = []

    pointset_exact_for_all = True
    missing_trajectory_keys = []

    processed_objects = 0

    # Cache one trajectory sample at a time.
    cached_key = None
    cached = None

    for (key, object_id), group in grouped:
        key = str(key)
        object_id = int(object_id)

        if key not in key_to_path:
            missing_trajectory_keys.append(key)
            continue

        if cached_key != key:
            if cached is not None:
                cached.close()

            cached = np.load(
                key_to_path[key],
                allow_pickle=False,
            )
            cached_key = key

        ids = cached["scene_object_id"]
        valid = cached["official_valid_mask"]
        gt_np = cached["gt_scene_flows"]
        pred_np = cached["pred_scene_flows"]

        assert gt_np.shape == pred_np.shape
        assert gt_np.shape[0] == 11
        assert gt_np.shape[-1] == 3

        T, N, _ = gt_np.shape

        assert ids.shape == (T, N)
        assert valid.shape == (T, N)

        group = group.sort_values(
            "global_point_index"
        )

        global_idx = group[
            "global_point_index"
        ].to_numpy(dtype=np.int64)

        scores = group[
            "outward_stretch_m"
        ].to_numpy(dtype=np.float64)

        if len(np.unique(global_idx)) != len(global_idx):
            raise RuntimeError(
                f"[{key} obj={object_id}] "
                "duplicate global point index"
            )

        expected_idx = np.where(
            (ids[0] == object_id)
            & valid[-1]
        )[0]

        pointset_exact = np.array_equal(
            global_idx,
            expected_idx,
        )

        pointset_exact_for_all &= pointset_exact

        if not pointset_exact:
            raise RuntimeError(
                f"[{key} obj={object_id}] "
                "point population mismatch"
            )

        n = len(global_idx)

        if n < 4:
            raise RuntimeError(
                f"[{key} obj={object_id}] "
                f"too few points: {n}"
            )

        gt = torch.from_numpy(
            gt_np[:, global_idx]
        ).float()

        pred = torch.from_numpy(
            pred_np[:, global_idx]
        ).float()

        valid_local = valid[
            :, global_idx
        ].astype(bool)

        # ---------------------------------------------
        # Exact same GT-t0 kNN definition as the
        # previous debris geometry experiment.
        # ---------------------------------------------
        k = min(8, n - 1)

        d0 = torch.cdist(
            gt[0].unsqueeze(0),
            gt[0].unsqueeze(0),
            p=2,
        ).squeeze(0)

        d0.fill_diagonal_(float("inf"))

        knn = torch.topk(
            d0,
            k=k,
            dim=1,
            largest=False,
        ).indices.cpu().numpy()

        union_adj = [
            set() for _ in range(n)
        ]
        mutual_adj = [
            set() for _ in range(n)
        ]
        edges = set()

        directed = [
            set(map(int, knn[i]))
            for i in range(n)
        ]

        for i in range(n):
            for j in directed[i]:
                union_adj[i].add(j)
                union_adj[j].add(i)

                a, b = sorted((i, j))
                edges.add((a, b))

                if i in directed[j]:
                    mutual_adj[i].add(j)
                    mutual_adj[j].add(i)

        object_name = str(
            group["object_name"].iloc[0]
        )

        for q in QUANTILES:
            threshold = float(
                np.quantile(scores, q)
            )

            anomalous = np.where(
                scores >= threshold
            )[0]

            union_components = get_components(
                anomalous,
                union_adj,
            )

            mutual_components = get_components(
                anomalous,
                mutual_adj,
            )

            if not union_components:
                raise RuntimeError(
                    f"[{key} obj={object_id} q={q}] "
                    "no union component"
                )

            cluster_list = union_components[0]
            cluster = set(cluster_list)

            if mutual_components:
                largest_mutual = mutual_components[0]
                membership_exact = (
                    set(cluster_list)
                    == set(largest_mutual)
                )
                largest_mutual_size = len(
                    largest_mutual
                )
            else:
                membership_exact = False
                largest_mutual_size = 0

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

            b_i, b_j = edge_index_arrays(
                boundary_edges
            )
            n_i, n_j = edge_index_arrays(
                normal_edges
            )

            final_boundary = geometry_metrics(
                gt[-1],
                pred[-1],
                valid_local[-1],
                b_i,
                b_j,
            )

            final_normal = geometry_metrics(
                gt[-1],
                pred[-1],
                valid_local[-1],
                n_i,
                n_j,
            )

            q_timestep_rows = []

            for t in range(1, 11):
                boundary_geom = geometry_metrics(
                    gt[t],
                    pred[t],
                    valid_local[t],
                    b_i,
                    b_j,
                )

                normal_geom = geometry_metrics(
                    gt[t],
                    pred[t],
                    valid_local[t],
                    n_i,
                    n_j,
                )

                pair_valid = (
                    valid_local[t - 1]
                    & valid_local[t]
                )

                b_vel_n, b_vel = (
                    relative_velocity_error(
                        gt[t - 1],
                        gt[t],
                        pred[t - 1],
                        pred[t],
                        pair_valid,
                        b_i,
                        b_j,
                    )
                )

                n_vel_n, n_vel = (
                    relative_velocity_error(
                        gt[t - 1],
                        gt[t],
                        pred[t - 1],
                        pred[t],
                        pair_valid,
                        n_i,
                        n_j,
                    )
                )

                if (
                    np.isfinite(b_vel)
                    and np.isfinite(n_vel)
                ):
                    vel_gap = b_vel - n_vel
                else:
                    vel_gap = np.nan

                if t >= 2:
                    triple_valid = (
                        valid_local[t - 2]
                        & valid_local[t - 1]
                        & valid_local[t]
                    )

                    b_acc_n, b_acc = (
                        relative_acceleration_error(
                            gt[t - 2],
                            gt[t - 1],
                            gt[t],
                            pred[t - 2],
                            pred[t - 1],
                            pred[t],
                            triple_valid,
                            b_i,
                            b_j,
                        )
                    )

                    n_acc_n, n_acc = (
                        relative_acceleration_error(
                            gt[t - 2],
                            gt[t - 1],
                            gt[t],
                            pred[t - 2],
                            pred[t - 1],
                            pred[t],
                            triple_valid,
                            n_i,
                            n_j,
                        )
                    )

                    if (
                        np.isfinite(b_acc)
                        and np.isfinite(n_acc)
                    ):
                        acc_gap = b_acc - n_acc
                    else:
                        acc_gap = np.nan
                else:
                    b_acc_n = 0
                    n_acc_n = 0
                    b_acc = np.nan
                    n_acc = np.nan
                    acc_gap = np.nan

                if (
                    np.isfinite(
                        boundary_geom[
                            "median_delta_m"
                        ]
                    )
                    and np.isfinite(
                        normal_geom[
                            "median_delta_m"
                        ]
                    )
                ):
                    geometry_gap = (
                        boundary_geom[
                            "median_delta_m"
                        ]
                        - normal_geom[
                            "median_delta_m"
                        ]
                    )
                else:
                    geometry_gap = np.nan

                row = {
                    "key": key,
                    "object_id": object_id,
                    "object_name": object_name,
                    "quantile": q,
                    "timestep": t,
                    "cluster_size": len(
                        cluster
                    ),

                    "boundary_geometry_n":
                        boundary_geom["n"],
                    "normal_geometry_n":
                        normal_geom["n"],

                    "boundary_geometry_delta_median_m":
                        boundary_geom[
                            "median_delta_m"
                        ],
                    "normal_geometry_delta_median_m":
                        normal_geom[
                            "median_delta_m"
                        ],
                    "geometry_gap_m":
                        geometry_gap,

                    "boundary_stretched_fraction":
                        boundary_geom[
                            "stretched_fraction"
                        ],
                    "boundary_pred_gt_ratio":
                        boundary_geom[
                            "pred_gt_ratio"
                        ],

                    "boundary_relvel_n":
                        b_vel_n,
                    "normal_relvel_n":
                        n_vel_n,
                    "boundary_relvel_error_median_m":
                        b_vel,
                    "normal_relvel_error_median_m":
                        n_vel,
                    "relvel_gap_m":
                        vel_gap,

                    "boundary_relacc_n":
                        b_acc_n,
                    "normal_relacc_n":
                        n_acc_n,
                    "boundary_relacc_error_median_m":
                        b_acc,
                    "normal_relacc_error_median_m":
                        n_acc,
                    "relacc_gap_m":
                        acc_gap,
                }

                timestep_rows.append(row)
                q_timestep_rows.append(row)

            vel_gaps = np.asarray([
                r["relvel_gap_m"]
                for r in q_timestep_rows
                if np.isfinite(
                    r["relvel_gap_m"]
                )
            ])

            acc_gaps = np.asarray([
                r["relacc_gap_m"]
                for r in q_timestep_rows
                if np.isfinite(
                    r["relacc_gap_m"]
                )
            ])

            geom_gaps = np.asarray([
                r["geometry_gap_m"]
                for r in q_timestep_rows
                if np.isfinite(
                    r["geometry_gap_m"]
                )
            ])

            object_rows.append({
                "key": key,
                "object_id": object_id,
                "object_name": object_name,
                "quantile": q,

                "n_points": n,
                "knn_k": k,
                "threshold_m": threshold,
                "anomalous_points":
                    int(len(anomalous)),

                "largest_union_cluster_size":
                    len(cluster),
                "largest_mutual_cluster_size":
                    largest_mutual_size,
                "union_mutual_membership_exact":
                    membership_exact,

                "boundary_edges":
                    len(boundary_edges),
                "normal_internal_edges":
                    len(normal_edges),

                "final_boundary_delta_median_m":
                    final_boundary[
                        "median_delta_m"
                    ],
                "final_boundary_delta_mean_m":
                    final_boundary[
                        "mean_delta_m"
                    ],
                "final_boundary_stretched_fraction":
                    final_boundary[
                        "stretched_fraction"
                    ],
                "final_boundary_pred_gt_ratio":
                    final_boundary[
                        "pred_gt_ratio"
                    ],

                "final_normal_delta_median_m":
                    final_normal[
                        "median_delta_m"
                    ],

                "relvel_gap_mean_over_valid_steps_m":
                    (
                        float(np.mean(vel_gaps))
                        if len(vel_gaps)
                        else np.nan
                    ),
                "relvel_positive_step_fraction":
                    (
                        float(np.mean(
                            vel_gaps > 0
                        ))
                        if len(vel_gaps)
                        else np.nan
                    ),

                "relacc_gap_mean_over_valid_steps_m":
                    (
                        float(np.mean(acc_gaps))
                        if len(acc_gaps)
                        else np.nan
                    ),
                "relacc_positive_step_fraction":
                    (
                        float(np.mean(
                            acc_gaps > 0
                        ))
                        if len(acc_gaps)
                        else np.nan
                    ),

                "geometry_gap_mean_over_valid_timesteps_m":
                    (
                        float(np.mean(geom_gaps))
                        if len(geom_gaps)
                        else np.nan
                    ),
                "geometry_positive_timestep_fraction":
                    (
                        float(np.mean(
                            geom_gaps > 0
                        ))
                        if len(geom_gaps)
                        else np.nan
                    ),
            })

        processed_objects += 1

        if (
            processed_objects % 25 == 0
            or processed_objects == len(grouped)
        ):
            print(
                f"processed_objects = "
                f"{processed_objects}/"
                f"{len(grouped)}"
            )

    if cached is not None:
        cached.close()

    object_df = pd.DataFrame(object_rows)
    timestep_df = pd.DataFrame(
        timestep_rows
    )

    object_df.to_csv(
        out_dir
        / "per_object_quantile_summary.csv",
        index=False,
    )

    timestep_df.to_csv(
        out_dir
        / "per_object_timestep_dynamics.csv",
        index=False,
    )

    tracked_diff = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    summary = {
        "trajectory_files":
            len(npz_files),
        "unique_trajectory_keys":
            len(key_to_path),

        "debris_point_rows":
            int(len(df)),
        "moved_object_groups":
            len(grouped),
        "processed_objects":
            processed_objects,

        "quantiles":
            list(QUANTILES),

        "object_quantile_rows":
            int(len(object_df)),
        "timestep_rows":
            int(len(timestep_df)),

        "missing_trajectory_keys":
            sorted(set(
                missing_trajectory_keys
            )),

        "official_l2_exact_for_all_trajectories":
            bool(all_l2_exact),

        "temporal_object_ids_exact_for_all_trajectories":
            bool(all_temporal_ids_exact),

        "point_population_exact_for_all_objects":
            bool(pointset_exact_for_all),

        "tracked_official_diff_empty":
            tracked_diff == "",
    }

    success = (
        summary["trajectory_files"]
        == 578
        and summary[
            "unique_trajectory_keys"
        ] == 578
        and summary[
            "moved_object_groups"
        ] == 377
        and summary[
            "processed_objects"
        ] == 377
        and summary[
            "object_quantile_rows"
        ] == 377 * 3
        and summary[
            "timestep_rows"
        ] == 377 * 3 * 10
        and len(
            summary[
                "missing_trajectory_keys"
            ]
        ) == 0
        and summary[
            "official_l2_exact_for_all_trajectories"
        ]
        and summary[
            "temporal_object_ids_exact_for_all_trajectories"
        ]
        and summary[
            "point_population_exact_for_all_objects"
        ]
        and summary[
            "tracked_official_diff_empty"
        ]
    )

    summary["SUCCESS"] = success

    with open(
        out_dir / "summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "FULL CLUSTER RELATIVE-DYNAMICS "
        "ANALYSIS"
    )
    print(
        "trajectory_files =",
        summary[
            "trajectory_files"
        ],
    )
    print(
        "moved_object_groups =",
        summary[
            "moved_object_groups"
        ],
    )
    print(
        "processed_objects =",
        processed_objects,
    )
    print(
        "object_quantile_rows =",
        len(object_df),
    )
    print(
        "timestep_rows =",
        len(timestep_df),
    )
    print(
        "official_l2_exact_for_all =",
        all_l2_exact,
    )
    print(
        "temporal_object_ids_exact_for_all =",
        all_temporal_ids_exact,
    )
    print(
        "point_population_exact_for_all =",
        pointset_exact_for_all,
    )
    print(
        "tracked_official_diff_empty =",
        tracked_diff == "",
    )
    print(
        "output_dir =",
        out_dir,
    )
    print("SUCCESS =", success)
    print("=" * 72)

    if not success:
        raise RuntimeError(
            "full cluster relative-dynamics "
            "validation failed"
        )


if __name__ == "__main__":
    main()
