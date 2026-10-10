#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch


ROOT = Path(
    "diagnostics/results/"
    "prediction_consistency_droid_unfiltered_full_20260804_135952/"
    "raw_sequences"
)

K = 8
MAX_FILES = 3


def q(arr, x):
    return float(np.quantile(arr, x))


def main():
    files = sorted(ROOT.glob("*.pt"))

    assert len(files) == 62, len(files)

    files = files[:MAX_FILES]

    print("=" * 80)
    print("DROID GT-T0 POINT SPACING SMOKE3")
    print("=" * 80)

    processed = 0

    for p in files:
        x = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        gt_all = x[
            "gt_scene_flows"
        ].float()

        valid_all = x[
            "official_valid_mask"
        ].bool()

        moved_all = x[
            "scene_moved_mask"
        ].bool()

        # Exactly the same population as local-coherence smoke3.
        idx = torch.where(
            valid_all[-1] & moved_all[-1]
        )[0]

        n = int(idx.numel())

        if n <= K:
            raise RuntimeError(
                f"{p.name}: too few points n={n}"
            )

        gt = gt_all[:, idx]

        # GT t0 pairwise distance.
        d0 = torch.cdist(
            gt[0].unsqueeze(0),
            gt[0].unsqueeze(0),
            p=2,
        ).squeeze(0)

        d0.fill_diagonal_(float("inf"))

        # Sorted local-neighbor distances for every point.
        knn_dist = torch.topk(
            d0,
            k=K,
            dim=1,
            largest=False,
        ).values.cpu().numpy()

        nn1 = knn_dist[:, 0]
        nn4 = knn_dist[:, 3]
        nn8 = knn_dist[:, 7]

        all_knn = knn_dist.reshape(-1)

        print("-" * 80)
        print("file =", p.name)
        print("n_points =", n)

        print(
            "nn1_median_mm =",
            np.median(nn1) * 1000,
        )
        print(
            "nn1_p90_mm =",
            q(nn1, 0.90) * 1000,
        )
        print(
            "nn1_p95_mm =",
            q(nn1, 0.95) * 1000,
        )
        print(
            "nn1_max_mm =",
            np.max(nn1) * 1000,
        )

        print(
            "nn4_median_mm =",
            np.median(nn4) * 1000,
        )
        print(
            "nn4_p90_mm =",
            q(nn4, 0.90) * 1000,
        )
        print(
            "nn4_p95_mm =",
            q(nn4, 0.95) * 1000,
        )
        print(
            "nn4_max_mm =",
            np.max(nn4) * 1000,
        )

        print(
            "nn8_median_mm =",
            np.median(nn8) * 1000,
        )
        print(
            "nn8_p90_mm =",
            q(nn8, 0.90) * 1000,
        )
        print(
            "nn8_p95_mm =",
            q(nn8, 0.95) * 1000,
        )
        print(
            "nn8_max_mm =",
            np.max(nn8) * 1000,
        )

        print(
            "all_knn_median_mm =",
            np.median(all_knn) * 1000,
        )
        print(
            "all_knn_p90_mm =",
            q(all_knn, 0.90) * 1000,
        )
        print(
            "all_knn_p95_mm =",
            q(all_knn, 0.95) * 1000,
        )
        print(
            "all_knn_p99_mm =",
            q(all_knn, 0.99) * 1000,
        )
        print(
            "all_knn_max_mm =",
            np.max(all_knn) * 1000,
        )

        processed += 1

    print("=" * 80)
    print("processed_sequences =", processed)
    print("SUCCESS =", processed == MAX_FILES)


if __name__ == "__main__":
    main()
