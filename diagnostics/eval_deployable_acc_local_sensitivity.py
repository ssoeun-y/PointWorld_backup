#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# These MUST be set before importing the validated B implementation.
os.environ.setdefault("PW_TARGET_TASKS", "task-0000")
os.environ.setdefault("PW_TAG", "tmp")
os.environ.setdefault("PW_CONDITION", "original")
os.environ.setdefault("PW_MATCH_TOL", "0.05")
os.environ.setdefault("PW_DELTA_FRAC", "0.05")
os.environ.setdefault("PW_MAX_PERTURB_FRAMES", "1.25")
os.environ.setdefault("PW_GRID_N", "161")

from arguments import parse_args
from evaluation.tester import Tester
import dataset_components.dataloader as dlmod

# Re-use the EXACT validated perturbation / split / official-build logic.
from diagnostics import eval_equal_physical_delta_global_heldout as src


ROOT = Path(__file__).resolve().parents[1]

TAG = os.environ["PW_TAG"]

OUTDIR = (
    ROOT
    / "diagnostics/results/"
      "deployable_acc_local_sensitivity"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / f"{TAG}__deployable_acc_sensitivity.csv"


def to_device(batch, device):
    return {
        k: (
            v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


def finite_stats_mm(x, mask):
    """
    x    : [B,T,N] prediction shift in model coordinates
    mask : [B,T,N] inference-available support

    Returns mean / p95 in millimeters.
    """
    vals = x[mask].float()

    if vals.numel() == 0:
        return np.nan, np.nan

    mean = float(
        vals.mean().cpu()
    ) * 1000.0

    p95 = float(
        torch.quantile(
            vals,
            0.95
        ).cpu()
    ) * 1000.0

    return mean, p95


def main():

    args = parse_args()

    tester = Tester(args)
    tester.model.eval()

    # Clear any state left in imported module.
    src.EQUAL_DELTA_META["acc"].clear()

    # ========================================================
    # Only TWO validated pipelines:
    #   base Action
    #   equal-DeltaAcc Action
    #
    # No Jerk loader / forward.
    # ========================================================

    loaders = {}

    for mode in ["base", "acc"]:

        dlmod.build_flow_sample = (
            src.make_builder(mode)
        )

        src.old.reset()

        loader, _ = dlmod.build_dataloader(
            tester.args,
            mode="test",
            rank=tester.rank,
            world_size=tester.world_size,
            override_splits="test",
        )

        loaders[mode] = loader

    # Always restore official builder.
    dlmod.build_flow_sample = (
        src.old.OFFICIAL_BUILD
    )

    rows = []
    found = set()

    with torch.inference_mode():

        for b0, ba in zip(
            loaders["base"],
            loaders["acc"],
        ):

            k0 = str(b0["__key__"][0])
            ka = str(ba["__key__"][0])

            if k0 != ka:
                raise RuntimeError(
                    f"loader mismatch: {k0} vs {ka}"
                )

            key = k0

            if key not in src.TARGET_KEYS:
                continue

            meta = (
                src.EQUAL_DELTA_META["acc"]
                .get(key)
            )

            if meta is None:
                raise RuntimeError(
                    f"{key}: missing Acc metadata"
                )

            matched = bool(
                meta["matched"]
            )

            # Recover TRUE task from established table.
            true_task = str(
                src.METRIC_BY_KEY[key]["task"]
            )

            row = {
                "clip": key,
                "task": true_task,

                "original_action_acc":
                    float(
                        src.METRIC_BY_KEY[key][
                            "rms_acceleration"
                        ]
                    ),

                "original_action_msj":
                    float(
                        src.METRIC_BY_KEY[key][
                            "mean_squared_jerk"
                        ]
                    ),

                "l2_moved_mm":
                    float(
                        src.METRIC_BY_KEY[key][
                            "l2_moved_mm"
                        ]
                    ),

                "acc_boundary_severity":
                    float(
                        src.METRIC_BY_KEY[key][
                            "rms_acceleration"
                        ]
                    ) / src.TA,

                "joint_boundary_severity":
                    max(
                        float(
                            src.METRIC_BY_KEY[key][
                                "rms_acceleration"
                            ]
                        ) / src.TA,

                        float(
                            src.METRIC_BY_KEY[key][
                                "mean_squared_jerk"
                            ]
                        ) / src.TJ,
                    ),

                "requested_delta_acc":
                    src.TARGET_DELTA_ACC,

                "actual_delta_acc":
                    float(
                        meta["actual_delta"]
                    ),

                "acc_match_relerr":
                    float(
                        meta[
                            "relative_match_error"
                        ]
                    ),

                "acc_matched":
                    int(matched),

                "acc_perturb_frames":
                    float(
                        meta["amp_frames"]
                    ),

                "acc_collateral_delta_msj":
                    float(
                        meta[
                            "collateral_delta"
                        ]
                    ),

                "deploy_acc_sensitivity_mean_mm":
                    np.nan,

                "deploy_acc_sensitivity_p95_mm":
                    np.nan,

                "deploy_acc_sensitivity_final_mean_mm":
                    np.nan,
            }

            # ------------------------------------------------
            # If equal physical delta was not achieved,
            # do NOT calculate an incomparable sensitivity.
            # ------------------------------------------------

            if matched:

                g0 = to_device(
                    b0,
                    tester.device
                )

                ga = to_device(
                    ba,
                    tester.device
                )

                src.old.reset()

                p0 = tester.model(
                    g0,
                    training=False
                )["scene_flows"]

                src.old.reset()

                pa = tester.model(
                    ga,
                    training=False
                )["scene_flows"]

                shift = torch.linalg.vector_norm(
                    pa - p0,
                    dim=-1
                )

                # ============================================
                # DEPLOYABLE SUPPORT ONLY
                #
                # scene_exists at t=0 identifies actual
                # currently observed scene points.
                #
                # DO NOT use:
                #   gt_scene_flows
                #   scene_supervised_mask
                #   scene_moved_mask
                #
                # Freeze t0 support across the whole horizon.
                # ============================================

                exists = src.mask3(
                    g0,
                    "scene_exists"
                ).bool()

                support_t0 = (
                    exists[:, 0:1, :]
                    .expand_as(shift)
                )

                mean_mm, p95_mm = (
                    finite_stats_mm(
                        shift,
                        support_t0
                    )
                )

                final_mask = exists[:, 0, :]

                final_vals = (
                    shift[:, -1, :]
                    [final_mask]
                    .float()
                )

                if final_vals.numel():
                    final_mean_mm = (
                        float(
                            final_vals.mean().cpu()
                        )
                        * 1000.0
                    )
                else:
                    final_mean_mm = np.nan

                row[
                    "deploy_acc_sensitivity_mean_mm"
                ] = mean_mm

                row[
                    "deploy_acc_sensitivity_p95_mm"
                ] = p95_mm

                row[
                    "deploy_acc_sensitivity_final_mean_mm"
                ] = final_mean_mm

            rows.append(row)
            found.add(key)

            n = len(found)

            if (
                n <= 5
                or n % 50 == 0
                or n == len(src.TARGET_KEYS)
            ):
                print(
                    f"[{n}/{len(src.TARGET_KEYS)}] "
                    f"{key} | "
                    f"match={row['acc_match_relerr']:.4g} | "
                    f"deploy_mean="
                    f"{row['deploy_acc_sensitivity_mean_mm']}"
                )

            if len(found) == len(
                src.TARGET_KEYS
            ):
                break

    missing = (
        src.TARGET_KEYS - found
    )

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} target clips. "
            f"First few={sorted(missing)[:10]}"
        )

    d = pd.DataFrame(rows)

    if d["clip"].duplicated().any():
        raise RuntimeError(
            "Duplicate clip in output"
        )

    d.to_csv(
        OUT,
        index=False
    )

    print()
    print("=" * 100)
    print("DEPLOYABLE ACC LOCAL SENSITIVITY")
    print("=" * 100)
    print("FOUND =", len(d))
    print(
        "MATCHED =",
        int(d["acc_matched"].sum())
    )
    print()
    print("TASKS:")
    print(
        d["task"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    q = d[
        d["acc_matched"] == 1
    ]

    print()
    print(
        "deploy mean median =",
        q[
            "deploy_acc_sensitivity_mean_mm"
        ].median()
    )
    print(
        "deploy p95 median =",
        q[
            "deploy_acc_sensitivity_p95_mm"
        ].median()
    )
    print(
        "deploy final median =",
        q[
            "deploy_acc_sensitivity_final_mean_mm"
        ].median()
    )

    print()
    print("OUTPUT =", OUT)
    print("SUCCESS = True")
    print("=" * 100)


if __name__ == "__main__":
    main()
