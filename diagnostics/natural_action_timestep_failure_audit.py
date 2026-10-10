#!/usr/bin/env python3

import csv
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.tester import Tester

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "diagnostics/results/behavior_moved_test_inventory.csv"
OUT = ROOT / "diagnostics/results/natural_action_timestep_failure.csv"

with META.open() as f:
    rows = list(csv.DictReader(f))

TARGET = {
    r["clip"]
    for r in rows
    if int(r["has_moved"]) == 1
}

assert len(TARGET) == 5288

results = []


def mask3(batch, name):
    x = batch[name].bool()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x


def q95(x):
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float(), 0.95))


@torch.no_grad()
def capture(self, dl, split, apply_confidence_mask=False):
    self.model.eval()
    found = set()

    for batch in dl:

        keys = [str(x) for x in batch["__key__"]]

        selected = [
            i for i, k in enumerate(keys)
            if k in TARGET and k not in found
        ]

        if not selected:
            continue

        batch_gpu = {
            k: (
                v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

        # PointWorld prediction
        outputs = self.model(batch_gpu, training=False)

        pred = outputs["scene_flows"]
        gt = batch_gpu["gt_scene_flows"]

        err = torch.linalg.vector_norm(pred - gt, dim=-1)

        exists = mask3(batch_gpu, "scene_exists")
        context = mask3(batch_gpu, "scene_context_mask")
        supervised = mask3(batch_gpu, "scene_supervised_mask")
        moved = mask3(batch_gpu, "scene_moved_mask")

        valid = (~context) & exists & supervised

        robot = batch_gpu["robot_flows"]
        robot_exists = mask3(batch_gpu, "robot_exists")

        # velocity: interval t -> t+1
        v = robot[:, 1:] - robot[:, :-1]
        valid_v = robot_exists[:, 1:] & robot_exists[:, :-1]

        # acceleration: aligned to destination timestep t=2..10
        a = v[:, 1:] - v[:, :-1]
        valid_a = valid_v[:, 1:] & valid_v[:, :-1]

        # jerk: aligned to destination timestep t=3..10
        j = a[:, 1:] - a[:, :-1]
        valid_j = valid_a[:, 1:] & valid_a[:, :-1]

        for i in selected:
            key = keys[i]
            row = {"clip": key}

            T = err.shape[1]

            # ------------------------------------------------
            # moved L2 at every predicted timestep
            # ------------------------------------------------
            for t in range(T):
                m = valid[i, t] & moved[i, t]

                if m.any():
                    row[f"moved_l2_t{t}_mm"] = (
                        float(err[i, t][m].mean()) * 1000.0
                    )
                else:
                    row[f"moved_l2_t{t}_mm"] = np.nan

            # ------------------------------------------------
            # speed p95: destination timestep 1..10
            # ------------------------------------------------
            for k in range(v.shape[1]):
                vals = torch.linalg.vector_norm(
                    v[i, k], dim=-1
                )[valid_v[i, k]]

                row[f"speed_p95_t{k+1}_mm"] = (
                    q95(vals) * 1000.0
                )

            # ------------------------------------------------
            # acceleration p95: destination timestep 2..10
            # ------------------------------------------------
            for k in range(a.shape[1]):
                vals = torch.linalg.vector_norm(
                    a[i, k], dim=-1
                )[valid_a[i, k]]

                row[f"acc_p95_t{k+2}_mm"] = (
                    q95(vals) * 1000.0
                )

            # ------------------------------------------------
            # jerk p95: destination timestep 3..10
            # ------------------------------------------------
            for k in range(j.shape[1]):
                vals = torch.linalg.vector_norm(
                    j[i, k], dim=-1
                )[valid_j[i, k]]

                row[f"jerk_p95_t{k+3}_mm"] = (
                    q95(vals) * 1000.0
                )

            results.append(row)
            found.add(key)

        if len(found) % 250 == 0:
            print(f"found = {len(found)} / {len(TARGET)}")

        if len(found) == len(TARGET):
            break

    if len(found) != len(TARGET):
        raise RuntimeError(
            f"missing: {len(TARGET)-len(found)}"
        )

    df = pd.DataFrame(results).sort_values("clip")
    df.to_csv(OUT, index=False)

    print("\nSAVED =", OUT)
    print("rows =", len(df))
    print("SUCCESS = True")

    return {
        f"diagnostic/{split}/timestep_failure": 1.0
    }


Tester._accumulate_metrics = capture

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
