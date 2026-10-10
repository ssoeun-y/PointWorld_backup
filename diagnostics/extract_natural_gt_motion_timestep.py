#!/usr/bin/env python3

import csv, runpy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from evaluation.tester import Tester

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "diagnostics/results/behavior_moved_test_inventory.csv"
OUT = ROOT / "diagnostics/results/natural_gt_motion_timestep.csv"

with META.open() as f:
    rows = list(csv.DictReader(f))

TARGET = {
    r["clip"] for r in rows
    if int(r["has_moved"]) == 1
}
assert len(TARGET) == 5288

results = []

def mask3(batch, name):
    x = batch[name].bool()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x

@torch.no_grad()
def capture(self, dl, split, apply_confidence_mask=False):
    found = set()

    for batch in dl:
        keys = [str(x) for x in batch["__key__"]]

        gt = batch["gt_scene_flows"]
        exists = mask3(batch, "scene_exists")
        context = mask3(batch, "scene_context_mask")
        supervised = mask3(batch, "scene_supervised_mask")
        moved = mask3(batch, "scene_moved_mask")

        valid = (~context) & exists & supervised

        # 실제 GT의 timestep 간 이동량
        delta = gt[:, 1:] - gt[:, :-1]
        mag = torch.linalg.vector_norm(delta, dim=-1)

        for i, key in enumerate(keys):
            if key not in TARGET or key in found:
                continue

            row = {"clip": key}

            for k in range(delta.shape[1]):
                t = k + 1

                m = (
                    valid[i, t]
                    & valid[i, t-1]
                    & moved[i, t]
                )

                vals = mag[i, k][m]

                if vals.numel():
                    row[f"gt_motion_mean_t{t}_mm"] = \
                        float(vals.mean()) * 1000.0
                    row[f"gt_motion_p95_t{t}_mm"] = \
                        float(torch.quantile(vals.float(), .95)) * 1000.0
                else:
                    row[f"gt_motion_mean_t{t}_mm"] = np.nan
                    row[f"gt_motion_p95_t{t}_mm"] = np.nan

            results.append(row)
            found.add(key)

        if len(found) % 500 == 0:
            print(f"found = {len(found)} / {len(TARGET)}")

        if len(found) == len(TARGET):
            break

    if len(found) != len(TARGET):
        raise RuntimeError(f"found {len(found)} / {len(TARGET)}")

    pd.DataFrame(results).sort_values("clip").to_csv(OUT, index=False)

    print("SAVED =", OUT)
    print("rows =", len(results))
    print("SUCCESS = True")

    return {f"diagnostic/{split}/gt_motion": 1.0}

Tester._accumulate_metrics = capture

runpy.run_path("eval.py", run_name="__main__")
