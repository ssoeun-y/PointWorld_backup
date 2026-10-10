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


def main():
    files = sorted(ROOT.glob("*.pt"))
    assert len(files) == 62, len(files)

    pooled_nn1 = []
    pooled_nn4 = []
    pooled_nn8 = []
    pooled_all = []

    seq_nn8_median = []
    seq_nn8_p90 = []
    seq_nn8_p95 = []
    seq_all_median = []
    seq_all_p90 = []
    seq_all_p95 = []

    processed = 0

    for p in files:
        x = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        gt_all = x["gt_scene_flows"].float()
        valid_all = x["official_valid_mask"].bool()
        moved_all = x["scene_moved_mask"].bool()

        # Exactly the same point population as smoke3.
        idx = torch.where(
            valid_all[-1] & moved_all[-1]
        )[0]

        n = int(idx.numel())

        if n <= K:
            raise RuntimeError(
                f"{p.name}: too few points n={n}"
            )

        gt0 = gt_all[0, idx]

        d0 = torch.cdist(
            gt0.unsqueeze(0),
            gt0.unsqueeze(0),
            p=2,
        ).squeeze(0)

        d0.fill_diagonal_(float("inf"))

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

        pooled_nn1.append(nn1)
        pooled_nn4.append(nn4)
        pooled_nn8.append(nn8)
        pooled_all.append(all_knn)

        seq_nn8_median.append(
            np.median(nn8)
        )
        seq_nn8_p90.append(
            np.quantile(nn8, 0.90)
        )
        seq_nn8_p95.append(
            np.quantile(nn8, 0.95)
        )

        seq_all_median.append(
            np.median(all_knn)
        )
        seq_all_p90.append(
            np.quantile(all_knn, 0.90)
        )
        seq_all_p95.append(
            np.quantile(all_knn, 0.95)
        )

        processed += 1

    pooled_nn1 = np.concatenate(pooled_nn1)
    pooled_nn4 = np.concatenate(pooled_nn4)
    pooled_nn8 = np.concatenate(pooled_nn8)
    pooled_all = np.concatenate(pooled_all)

    print("=" * 80)
    print("DROID GT-T0 POINT SPACING FULL62")
    print("=" * 80)

    for name, arr in [
        ("nn1", pooled_nn1),
        ("nn4", pooled_nn4),
        ("nn8", pooled_nn8),
        ("all_knn", pooled_all),
    ]:
        print(
            f"{name}_median_mm =",
            np.median(arr) * 1000,
        )
        print(
            f"{name}_p75_mm =",
            np.quantile(arr, 0.75) * 1000,
        )
        print(
            f"{name}_p90_mm =",
            np.quantile(arr, 0.90) * 1000,
        )
        print(
            f"{name}_p95_mm =",
            np.quantile(arr, 0.95) * 1000,
        )
        print(
            f"{name}_p99_mm =",
            np.quantile(arr, 0.99) * 1000,
        )

    print("-" * 80)
    print("SEQUENCE-LEVEL VARIABILITY")

    for name, arr in [
        ("seq_nn8_median", seq_nn8_median),
        ("seq_nn8_p90", seq_nn8_p90),
        ("seq_nn8_p95", seq_nn8_p95),
        ("seq_all_median", seq_all_median),
        ("seq_all_p90", seq_all_p90),
        ("seq_all_p95", seq_all_p95),
    ]:
        arr = np.asarray(arr)

        print(
            f"{name}_median_mm =",
            np.median(arr) * 1000,
        )
        print(
            f"{name}_p90_mm =",
            np.quantile(arr, 0.90) * 1000,
        )
        print(
            f"{name}_max_mm =",
            np.max(arr) * 1000,
        )

    print("-" * 80)
    print("processed_sequences =", processed)
    print("expected_sequences =", len(files))
    print(
        "SUCCESS =",
        processed == 62,
    )


if __name__ == "__main__":
    main()
