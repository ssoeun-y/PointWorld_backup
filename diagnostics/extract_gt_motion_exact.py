#!/usr/bin/env python3

import csv
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.tester import Tester


ROOT = Path(__file__).resolve().parents[1]

META = Path(os.environ["PW_META_CSV"])
OUT = Path(os.environ["PW_OUT_CSV"])
EXPECTED = int(os.environ["PW_EXPECTED_SAMPLES"])


with META.open() as f:
    meta_rows = list(csv.DictReader(f))

TARGET = {str(r["clip"]) for r in meta_rows}

assert len(TARGET) == EXPECTED, (
    len(TARGET), EXPECTED
)

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
def capture(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):

    found = set()

    for batch in dl:

        raw_keys = batch["__key__"]

        if isinstance(raw_keys, (list, tuple)):
            keys = [str(x) for x in raw_keys]
        else:
            keys = [str(raw_keys)]

        selected = [
            i for i, key in enumerate(keys)
            if key in TARGET and key not in found
        ]

        if not selected:
            if len(found) >= EXPECTED:
                break
            continue

        # Official PointWorld quantities
        gt_delta = batch["gt_scene_flows_delta"]

        exists = mask3(batch, "scene_exists")
        context = mask3(batch, "scene_context_mask")
        supervised = mask3(batch, "scene_supervised_mask")
        moved = mask3(batch, "scene_moved_mask")

        # EXACT historical official evaluation validity.
        valid = (
            (~context)
            & exists
            & supervised
        )

        T = gt_delta.shape[1]

        for i in selected:

            key = keys[i]

            # -----------------------------------------------
            # EXACT historical fixed temporal support
            # -----------------------------------------------
            moved_any = moved[i, 1:].any(dim=0)
            exists_all = exists[i].all(dim=0)

            supervised_future = (
                supervised[i, 1:].all(dim=0)
            )

            support = (
                moved_any
                & exists_all
                & supervised_future
            )

            row = {
                "clip": key,
                "task": key.split("_episode_")[0],
                "temporal_support_n": int(
                    support.sum().item()
                ),
            }

            # IMPORTANT:
            # sample gt_motion uses ONLY timesteps for which
            # official moved evaluation support is non-empty.
            selected_gt_profile = []

            for t in range(1, T):

                # EXACT historical n_moved_t
                m_eval = (
                    valid[i, t]
                    & moved[i, t]
                )

                n_moved = int(
                    m_eval.sum().item()
                )

                row[f"n_moved_t{t}"] = n_moved

                # Historical temporal fixed support
                m_temporal = (
                    support
                    & (~context[i, t])
                )

                if not m_temporal.any():
                    row[
                        f"gt_inc_mean_t{t}_mm"
                    ] = np.nan

                    if n_moved > 0:
                        raise RuntimeError(
                            f"{key} t={t}: "
                            "n_moved>0 but temporal support empty"
                        )

                    continue

                gt_inc = (
                    gt_delta[i, t][m_temporal]
                )

                gt_inc_mag = torch.linalg.norm(
                    gt_inc,
                    dim=-1,
                )

                g = (
                    gt_inc_mag.mean().item()
                    * 1000.0
                )

                row[
                    f"gt_inc_mean_t{t}_mm"
                ] = g

                # EXACT historical timestep selection rule
                if n_moved > 0:
                    selected_gt_profile.append(g)

            if not selected_gt_profile:
                raise RuntimeError(
                    f"{key}: no selected moved timesteps"
                )

            row["n_selected_timesteps"] = len(
                selected_gt_profile
            )

            row["gt_motion_mm"] = float(
                np.mean(selected_gt_profile)
            )

            results.append(row)
            found.add(key)

        if len(found) >= EXPECTED:
            break

    missing = TARGET - found

    if missing:
        print("FIRST MISSING =", sorted(missing)[:20])
        raise RuntimeError(
            f"found {len(found)} / expected {EXPECTED}"
        )

    df = (
        pd.DataFrame(results)
        .drop_duplicates("clip")
        .sort_values("clip")
        .reset_index(drop=True)
    )

    assert len(df) == EXPECTED
    assert df["clip"].nunique() == EXPECTED
    assert df["gt_motion_mm"].notna().all()

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT,
        index=False,
    )

    print("\n===== GT MOTION EXACT =====")
    print("rows =", len(df))
    print("unique =", df["clip"].nunique())
    print("finite =", df["gt_motion_mm"].notna().sum())

    print("\ntasks:")
    print(df["task"].value_counts().sort_index())

    print("\nselected timestep counts:")
    print(
        df["n_selected_timesteps"]
        .value_counts()
        .sort_index()
    )

    print("\nSAVED =", OUT)
    print("SUCCESS = True")

    return {
        f"diagnostic/{split}/gt_motion_exact": 1.0
    }


Tester._accumulate_metrics = capture

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
