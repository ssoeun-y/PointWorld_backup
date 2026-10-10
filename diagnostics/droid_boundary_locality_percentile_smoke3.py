#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch


ROOT = Path(
    "diagnostics/results/"
    "prediction_consistency_droid_unfiltered_full_20260804_135952/"
    "raw_sequences"
)

Q = 0.95
K = 8
MAX_FILES = 3


def components(nodes, adj):
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

            for v in adj[u]:
                if v in nodes and v not in seen:
                    seen.add(v)
                    stack.append(v)

        comps.append(sorted(comp))

    comps.sort(
        key=lambda x: (-len(x), x[0])
    )

    return comps


def median_relvel_error(
    gt,
    pred,
    valid,
    edges,
    t,
):
    vals = []

    for i, j in edges:
        if not (
            valid[t - 1, i]
            and valid[t, i]
            and valid[t - 1, j]
            and valid[t, j]
        ):
            continue

        gt_vi = gt[t, i] - gt[t - 1, i]
        gt_vj = gt[t, j] - gt[t - 1, j]

        pr_vi = pred[t, i] - pred[t - 1, i]
        pr_vj = pred[t, j] - pred[t - 1, j]

        gt_rel = gt_vi - gt_vj
        pr_rel = pr_vi - pr_vj

        vals.append(
            float(
                torch.linalg.vector_norm(
                    pr_rel - gt_rel
                )
            )
        )

    if not vals:
        return np.nan, 0

    return float(np.median(vals)), len(vals)


def main():
    files = sorted(ROOT.glob("*.pt"))

    assert len(files) == 62

    files = files[:MAX_FILES]

    print("=" * 80)
    print("DROID LOCAL COHERENCE SMOKE3")
    print("=" * 80)

    processed = 0

    for p in files:
        x = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        pred_all = x[
            "prediction_scene_flows"
        ].float()

        gt_all = x[
            "gt_scene_flows"
        ].float()

        valid_all = x[
            "official_valid_mask"
        ].bool()

        moved_all = x[
            "scene_moved_mask"
        ].bool()

        # Use the same point population throughout:
        # final official-valid + moved points.
        idx = torch.where(
            valid_all[-1] & moved_all[-1]
        )[0]

        n = int(idx.numel())

        if n <= K:
            raise RuntimeError(
                f"{p.name}: too few points n={n}"
            )

        gt = gt_all[:, idx]
        pred = pred_all[:, idx]
        valid = valid_all[:, idx]

        # -------------------------------------------------
        # Fixed GT-t0 local neighborhood.
        # -------------------------------------------------
        d0 = torch.cdist(
            gt[0].unsqueeze(0),
            gt[0].unsqueeze(0),
            p=2,
        ).squeeze(0)

        d0.fill_diagonal_(float("inf"))

        knn = torch.topk(
            d0,
            k=min(K, n - 1),
            dim=1,
            largest=False,
        ).indices

        adj = [set() for _ in range(n)]
        undirected_edges = set()

        for i in range(n):
            for j in knn[i].tolist():
                j = int(j)

                adj[i].add(j)
                adj[j].add(i)

                a, b = sorted((i, j))
                undirected_edges.add((a, b))

        # -------------------------------------------------
        # Independent geometry-break score:
        # mean outward edge stretch at final timestep.
        # -------------------------------------------------
        outward = np.zeros(n, dtype=np.float64)

        for i in range(n):
            js = knn[i]

            gt_edge = torch.linalg.vector_norm(
                gt[-1, js] - gt[-1, i],
                dim=-1,
            )

            pr_edge = torch.linalg.vector_norm(
                pred[-1, js] - pred[-1, i],
                dim=-1,
            )

            outward[i] = float(
                torch.clamp(
                    pr_edge - gt_edge,
                    min=0.0,
                ).mean()
            )

        threshold = float(
            np.quantile(outward, Q)
        )

        anomalous = np.where(
            outward >= threshold
        )[0]

        comps = components(
            anomalous,
            adj,
        )

        if not comps:
            raise RuntimeError(
                f"{p.name}: no Q95 component"
            )

        cluster = set(comps[0])

        boundary_edges = sorted([
            (i, j)
            for i, j in undirected_edges
            if (
                (i in cluster)
                != (j in cluster)
            )
        ])

        normal_edges = sorted([
            (i, j)
            for i, j in undirected_edges
            if (
                i not in cluster
                and j not in cluster
            )
        ])

        # -------------------------------------------------
        # -------------------------------------------------
        # D1 sanity check:
        # GT-t0 distance of Q95 cluster-boundary edges.
        # -------------------------------------------------
        gt_t0_boundary_dist = np.asarray([
            float(
                torch.linalg.vector_norm(
                    gt[0, i] - gt[0, j]
                )
            )
            for i, j in boundary_edges
        ], dtype=np.float64)


        # Compare boundary edges against the same sequence's
        # full undirected kNN-edge distance distribution.
        gt_t0_all_edge_dist = np.asarray([
            float(d0[i, j])
            for i, j in undirected_edges
        ], dtype=np.float64)

        all_edge_p90 = np.quantile(
            gt_t0_all_edge_dist, 0.90
        )
        all_edge_p95 = np.quantile(
            gt_t0_all_edge_dist, 0.95
        )
        all_edge_p99 = np.quantile(
            gt_t0_all_edge_dist, 0.99
        )

        # Final boundary geometry stretch.
        # -------------------------------------------------
        boundary_delta = []

        for i, j in boundary_edges:
            gt_len = float(
                torch.linalg.vector_norm(
                    gt[-1, i] - gt[-1, j]
                )
            )

            pr_len = float(
                torch.linalg.vector_norm(
                    pred[-1, i] - pred[-1, j]
                )
            )

            boundary_delta.append(
                pr_len - gt_len
            )

        boundary_delta = np.asarray(
            boundary_delta,
            dtype=np.float64,
        )

        # -------------------------------------------------
        # Relative velocity prediction error:
        # cluster boundary vs normal local edges.
        # -------------------------------------------------
        step_gaps = []

        print()
        print("-" * 80)
        print("file =", p.name)
        print("key =", x["key"])
        print("final_valid_moved_points =", n)
        print(
            "Q95_threshold_mm =",
            threshold * 1000,
        )
        print(
            "largest_cluster_size =",
            len(cluster),
        )
        print(
            "num_components =",
            len(comps),
        )
        print(
            "boundary_edges =",
            len(boundary_edges),
        )
        print(
            "normal_edges =",
            len(normal_edges),
        )

        if len(boundary_delta):
            print(
                "final_boundary_delta_median_mm =",
                np.median(boundary_delta) * 1000,
            )
            print(
                "final_boundary_stretched_fraction =",
                np.mean(boundary_delta > 0),
            )

        if len(gt_t0_boundary_dist):
            print(
                "gt_t0_all_edge_p90_mm =",
                all_edge_p90 * 1000,
            )
            print(
                "gt_t0_all_edge_p95_mm =",
                all_edge_p95 * 1000,
            )
            print(
                "gt_t0_all_edge_p99_mm =",
                all_edge_p99 * 1000,
            )
            print(
                "boundary_fraction_above_all_p95 =",
                np.mean(
                    gt_t0_boundary_dist > all_edge_p95
                ),
            )
            print(
                "boundary_fraction_above_all_p99 =",
                np.mean(
                    gt_t0_boundary_dist > all_edge_p99
                ),
            )
            print(
                "gt_t0_boundary_edge_count =",
                len(gt_t0_boundary_dist),
            )
            print(
                "gt_t0_boundary_distance_median_mm =",
                np.median(gt_t0_boundary_dist) * 1000,
            )
            print(
                "gt_t0_boundary_distance_p90_mm =",
                np.quantile(gt_t0_boundary_dist, 0.90) * 1000,
            )
            print(
                "gt_t0_boundary_distance_p95_mm =",
                np.quantile(gt_t0_boundary_dist, 0.95) * 1000,
            )
            print(
                "gt_t0_boundary_distance_max_mm =",
                np.max(gt_t0_boundary_dist) * 1000,
            )

        print("timestep relvel boundary_mm normal_mm gap_mm")

        for t in range(1, 11):
            b, bn = median_relvel_error(
                gt,
                pred,
                valid,
                boundary_edges,
                t,
            )

            nrm, nn = median_relvel_error(
                gt,
                pred,
                valid,
                normal_edges,
                t,
            )

            if np.isfinite(b) and np.isfinite(nrm):
                gap = b - nrm
                step_gaps.append(gap)

                print(
                    t,
                    f"{b * 1000:.6f}",
                    f"{nrm * 1000:.6f}",
                    f"{gap * 1000:+.6f}",
                    f"(n={bn}/{nn})",
                )
            else:
                print(
                    t,
                    "NA",
                    "NA",
                    "NA",
                    f"(n={bn}/{nn})",
                )

        step_gaps = np.asarray(
            step_gaps,
            dtype=np.float64,
        )

        if len(step_gaps):
            print(
                "relvel_gap_mean_mm =",
                step_gaps.mean() * 1000,
            )
            print(
                "relvel_gap_median_mm =",
                np.median(step_gaps) * 1000,
            )
            print(
                "positive_step_fraction =",
                np.mean(step_gaps > 0),
            )

        processed += 1

    print()
    print("=" * 80)
    print("processed_sequences =", processed)
    print("SUCCESS = True")
    print("=" * 80)


if __name__ == "__main__":
    main()
