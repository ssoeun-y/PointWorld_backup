#!/usr/bin/env python3

import csv
import runpy
from pathlib import Path

import torch
from evaluation.tester import Tester


OUT = Path(
    "diagnostics/results/"
    "behavior_moved_test_inventory.csv"
)
OUT.parent.mkdir(parents=True, exist_ok=True)


def mask3(batch, name):
    x = batch[name].bool()

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise RuntimeError(
            f"{name}: unexpected shape {tuple(x.shape)}"
        )

    return x


def scan_moved_inventory(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):
    rows = []

    for batch in dl:

        raw_keys = batch["__key__"]

        if isinstance(raw_keys, (list, tuple)):
            keys = [str(x) for x in raw_keys]
        else:
            keys = [str(raw_keys)]

        exists = mask3(batch, "scene_exists")
        context = mask3(batch, "scene_context_mask")
        supervised = mask3(
            batch,
            "scene_supervised_mask",
        )
        moved = mask3(
            batch,
            "scene_moved_mask",
        )

        # EXACT PointWorld official valid definition
        valid = (
            (~context)
            & exists
            & supervised
        )

        for i, key in enumerate(keys):

            moved_valid = (
                moved[i] & valid[i]
            )

            T = moved_valid.shape[0]

            row = {
                "clip": key,
                "num_valid":
                    int(valid[i].sum().item()),
                "num_moved_valid":
                    int(moved_valid.sum().item()),
                "has_moved":
                    int(moved_valid.any().item()),
            }

            for t in range(T):
                row[f"num_valid_t{t}"] = int(
                    valid[i, t].sum().item()
                )
                row[f"num_moved_valid_t{t}"] = int(
                    moved_valid[t].sum().item()
                )

            rows.append(row)

    if not rows:
        raise RuntimeError("No test samples found")

    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    n_total = len(rows)
    n_moved = sum(r["has_moved"] for r in rows)

    print("=" * 80)
    print("OFFICIAL MOVED TEST INVENTORY")
    print("=" * 80)
    print("total samples =", n_total)
    print("samples with moved-valid points =", n_moved)
    print(
        "fraction =",
        f"{100.0*n_moved/n_total:.2f}%"
    )

    # No thresholding: just distribution for information.
    vals = sorted(
        r["num_moved_valid"]
        for r in rows
        if r["has_moved"]
    )

    if vals:
        import numpy as np

        vals = np.asarray(vals)

        print(
            "moved-valid count median =",
            f"{np.median(vals):.1f}",
        )
        print(
            "moved-valid count p25/p75 =",
            f"{np.percentile(vals,25):.1f} / "
            f"{np.percentile(vals,75):.1f}",
        )
        print(
            "moved-valid count min/max =",
            f"{vals.min()} / {vals.max()}",
        )

    print("saved =", OUT.resolve())
    print("SUCCESS = True")

    return {
        f"diagnostic/{split}/moved_inventory":
            1.0
    }


# No model forward.
Tester._accumulate_metrics = scan_moved_inventory

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
