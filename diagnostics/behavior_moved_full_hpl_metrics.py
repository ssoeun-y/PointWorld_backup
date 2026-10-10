#!/usr/bin/env python3

import csv
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.tester import Tester


ROOT = Path(__file__).resolve().parents[1]

META = ROOT / (
    "diagnostics/results/"
    "behavior_moved_test_inventory.csv"
)

OUT = ROOT / (
    "diagnostics/results/"
    "behavior_moved_full_hpl_metrics.csv"
)

HPL_ROOT = ROOT / "diagnostics/third_party/HPLFlowNet"

if not hasattr(np, "float"):
    np.float = float

sys.path.insert(0, str(HPL_ROOT))

# 공식 HPLFlowNet 구현 그대로 사용
from evaluation_utils import evaluate_3d


with META.open() as f:
    meta_rows = list(csv.DictReader(f))

TARGET = {
    r["clip"]
    for r in meta_rows
    if int(r["has_moved"]) == 1
}

assert len(TARGET) == 5288

results = []


def mask3(batch, name):
    x = batch[name].bool()

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise RuntimeError(
            f"{name}: unexpected shape {tuple(x.shape)}"
        )

    return x


@torch.no_grad()
def capture(self, dl, split, apply_confidence_mask=False):

    self.model.eval()
    found = set()

    for batch in dl:

        keys = [str(x) for x in batch["__key__"]]

        selected = [
            i for i, key in enumerate(keys)
            if key in TARGET and key not in found
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

        outputs = self.model(
            batch_gpu,
            training=False,
        )

        pred = outputs["scene_flows"]
        gt = batch_gpu["gt_scene_flows"]

        exists = mask3(
            batch_gpu,
            "scene_exists",
        )
        context = mask3(
            batch_gpu,
            "scene_context_mask",
        )
        supervised = mask3(
            batch_gpu,
            "scene_supervised_mask",
        )
        moved = mask3(
            batch_gpu,
            "scene_moved_mask",
        )

        valid = (
            (~context)
            & exists
            & supervised
        )

        T = pred.shape[1]

        for i in selected:

            key = keys[i]

            row = {
                "clip": key,
            }

            # sample 평균 계산용
            epe_list = []
            accs_list = []
            accr_list = []
            outlier_list = []

            for t in range(T):

                # PointWorld 공식 moved subset
                m = (
                    valid[i, t]
                    & moved[i, t]
                )

                row[f"n_moved_t{t}"] = int(m.sum())

                if not m.any():

                    row[f"epe3d_moved_t{t}_m"] = np.nan
                    row[f"acc3ds_moved_t{t}"] = np.nan
                    row[f"acc3dr_moved_t{t}"] = np.nan
                    row[f"outliers3d_moved_t{t}"] = np.nan
                    row[f"gt_flow_mean_t{t}_mm"] = np.nan
                    row[f"gt_flow_p95_t{t}_mm"] = np.nan
                    continue

                sf_pred = (
                    pred[i, t][m]
                    .detach()
                    .cpu()
                    .numpy()
                )

                sf_gt = (
                    gt[i, t][m]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # ---------------------------------------
                # 공식 HPLFlowNet evaluate_3d()
                # ---------------------------------------
                (
                    epe3d,
                    acc3ds,
                    acc3dr,
                    outliers3d,
                ) = evaluate_3d(
                    sf_pred,
                    sf_gt,
                )

                row[f"epe3d_moved_t{t}_m"] = float(epe3d)
                row[f"acc3ds_moved_t{t}"] = float(acc3ds)
                row[f"acc3dr_moved_t{t}"] = float(acc3dr)
                row[f"outliers3d_moved_t{t}"] = float(outliers3d)

                # 이 timestep에서 실제 GT가 얼마나 움직였나
                gt_mag = np.linalg.norm(
                    sf_gt,
                    axis=-1,
                )

                row[f"gt_flow_mean_t{t}_mm"] = (
                    float(np.mean(gt_mag)) * 1000.0
                )

                row[f"gt_flow_p95_t{t}_mm"] = (
                    float(np.quantile(gt_mag, .95)) * 1000.0
                )

                epe_list.append(float(epe3d))
                accs_list.append(float(acc3ds))
                accr_list.append(float(acc3dr))
                outlier_list.append(float(outliers3d))

            row["epe3d_moved_mean_m"] = (
                np.mean(epe_list)
                if epe_list else np.nan
            )

            row["acc3ds_moved_mean"] = (
                np.mean(accs_list)
                if accs_list else np.nan
            )

            row["acc3dr_moved_mean"] = (
                np.mean(accr_list)
                if accr_list else np.nan
            )

            row["outliers3d_moved_mean"] = (
                np.mean(outlier_list)
                if outlier_list else np.nan
            )

            results.append(row)
            found.add(key)

        if len(found) % 250 == 0:
            print(
                f"found = {len(found)} / {len(TARGET)}"
            )

        if len(found) == len(TARGET):
            break

    if len(found) != len(TARGET):
        raise RuntimeError(
            f"found {len(found)} / {len(TARGET)}"
        )

    df = pd.DataFrame(results).sort_values("clip")
    df.to_csv(OUT, index=False)

    print("\nSAVED =", OUT)
    print("rows =", len(df))
    print("SUCCESS = True")

    return {
        f"diagnostic/{split}/moved_hpl":
        1.0
    }


Tester._accumulate_metrics = capture

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
