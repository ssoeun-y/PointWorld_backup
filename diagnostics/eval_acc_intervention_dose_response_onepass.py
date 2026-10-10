#!/usr/bin/env python3

import os
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import webdataset as wds

# ============================================================
# Freeze experiment BEFORE importing validated equal-delta code
# ============================================================

os.environ["PW_TARGET_TASKS"] = "task-0006,task-0008,task-0010"
os.environ["PW_TAG"] = "task0006_0008_0010_onepass"
os.environ["PW_CONDITION"] = "original"
os.environ.setdefault("PW_MATCH_TOL", "0.05")
os.environ.setdefault("PW_DELTA_FRAC", "0.05")
os.environ.setdefault("PW_MAX_PERTURB_FRAMES", "1.25")
os.environ.setdefault("PW_GRID_N", "161")

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dlmod

from dataset_components.decoders import decode_data
from dataset_components.cameras import sample_cameras
from dataset_components.robot import canonicalize_gripper_keys_and_flags
from dataset_components.pipeline import apply_release_pipeline_to_sample
from dataset_components.collate import custom_collate_fn
from dataset_components.constants import RELEASE_GRIPPER_ONLY

from robot_sampler import RobotSampler as TorchRobotSampler
from utils import resolve_robot_urdf

# EXACT validated equal-delta implementation.
from diagnostics import eval_equal_physical_delta_global_heldout as src

# Pre-fixed causal/mechanism intervention levels.
# requested_delta = global Acc boundary * fraction.
INTERVENTION_FRACS = (0.05, 0.10, 0.15, 0.20)


ROOT = Path(__file__).resolve().parents[1]

OUTDIR = (
    ROOT
    / "diagnostics/results/acc_intervention_dose_response_onepass"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_COMBINED = (
    OUTDIR
    / "task0006_0008_0010__acc_intervention_dose_response_ONEPASS.csv"
)


# ============================================================
# Utilities
# ============================================================

def to_device(batch, device):

    return {
        k: (
            v.to(
                device,
                non_blocking=True,
            )
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


def finite_stats_mm(shift, support):

    vals = shift[support].float()

    if vals.numel() == 0:
        return np.nan, np.nan

    mean_mm = (
        float(vals.mean().cpu())
        * 1000.0
    )

    p95_mm = (
        float(
            torch.quantile(
                vals,
                0.95,
            ).cpu()
        )
        * 1000.0
    )

    return mean_mm, p95_mm


def release_process(
    sample,
    args,
    domain,
    rank,
):

    # EXACT same camera step as official eval build_dataset().
    sample = sample_cameras(
        sample,
        min_num_cameras=args.eval_min_num_cameras,
        max_num_cameras=args.eval_max_num_cameras,
        deterministic=True,
        seed=(args.seed + rank),
    )

    sample = (
        canonicalize_gripper_keys_and_flags(
            sample
        )
    )

    # Official single-sample equivalent of sample_transform_pipeline().
    sample = apply_release_pipeline_to_sample(
        sample,
        domain=domain,
        mode="test",
        args=args,
        has_bimanual_robot=True,
        rank=rank,
        include_scene_data=False,
        skip_scene_sampling=False,
    )

    return sample


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if len(args.data_dirs) != 1:
        raise RuntimeError(
            "ONEPASS expects exactly one original BEHAVIOR WDS"
        )

    if len(args.domains) != 1:
        raise RuntimeError(
            "ONEPASS expects exactly one domain"
        )

    domain = args.domains[0]
    data_dir = args.data_dirs[0]

    if domain != "behavior":
        raise RuntimeError(
            f"Expected behavior, got {domain}"
        )

    # --------------------------------------------------------
    # Exact target audit
    # --------------------------------------------------------

    expected_by_task = {
        "task-0006": 535,
        "task-0008": 548,
        "task-0010": 504,
    }

    true_counts = (
        src.HELDOUT["task"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("=" * 100)
    print("TARGET AUDIT")
    print("=" * 100)
    print(true_counts)

    assert true_counts == expected_by_task, (
        true_counts,
        expected_by_task,
    )

    assert len(src.TARGET_KEYS) == 1587

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    tester = Tester(args)
    tester.model.eval()

    device = tester.device

    # --------------------------------------------------------
    # EXACT official BEHAVIOR RobotSampler, initialized ONCE.
    # --------------------------------------------------------

    urdf_path = resolve_robot_urdf(domain)

    robot_sampler = TorchRobotSampler(
        urdf_path=urdf_path,
        gripper_only=RELEASE_GRIPPER_ONLY,
        device="cpu",
    )

    # Mirror official eval force-single-arm logic.
    force_single_arm = False

    if getattr(
        args,
        "_eval_override_active",
        False,
    ):
        fs_domains = getattr(
            args,
            "eval_force_single_arm_domains",
            set(),
        )

        force_single_arm = (
            domain in fs_domains
        )

    # --------------------------------------------------------
    # ORIGINAL restored WDS ONLY.
    #
    # No temporary WDS.
    # No new metadata.
    # No base loader + acc loader duplication.
    #
    # WDS shards are traversed ONCE.
    # --------------------------------------------------------

    shard_paths = dlmod.gather_shard_paths(
        data_dir,
        "test",
        rank=tester.rank,
    )

    if not shard_paths:
        raise RuntimeError(
            f"No official test shards: {data_dir}"
        )

    # Mirror official deterministic eval shard ordering.
    random.Random(
        args.seed + tester.rank
    ).shuffle(shard_paths)

    print()
    print("=" * 100)
    print("ORIGINAL WDS ONE-PASS")
    print("=" * 100)
    print("data_dir =", data_dir)
    print("original shards =", len(shard_paths))
    print("targets =", len(src.TARGET_KEYS))
    print("WDS passes = 1")
    print("variants = base + Acc")
    print("Jerk variant = NOT RUN")

    dataset = wds.WebDataset(
        shard_paths,
        shardshuffle=False,
        detshuffle=True,
        resampled=False,
        nodesplitter=(
            wds.split_by_node
            if tester.world_size > 1
            else None
        ),
        seed=(args.seed + tester.rank),
        handler=wds.warn_and_continue,
    )

    rows = []
    found = set()

    scanned = 0
    model_pairs = 0

    with torch.inference_mode():

        for raw in dataset:

            scanned += 1

            raw_key = str(
                raw.get("__key__", "")
            )

            # ----------------------------------------------
            # EARLY FILTER:
            # Non-target samples are NOT decoded,
            # NOT built, and NEVER sent to model.
            # ----------------------------------------------

            if raw_key not in src.TARGET_KEYS:
                continue

            if raw_key in found:
                raise RuntimeError(
                    f"Duplicate target key: {raw_key}"
                )

            # ----------------------------------------------
            # Decode this target ONCE.
            # ----------------------------------------------

            decoded = decode_data(
                raw,
                domain=domain,
            )

            key = str(
                decoded.get(
                    "__key__",
                    raw_key,
                )
            )

            if key != raw_key:
                raise RuntimeError(
                    f"raw/decode key mismatch: "
                    f"{raw_key} != {key}"
                )

            if (
                "right_gripper_pose"
                not in decoded
                or
                "left_gripper_pose"
                not in decoded
            ):
                raise RuntimeError(
                    f"{key}: raw gripper pose missing"
                )

            metric_sample = {
                "right_gripper_pose":
                    np.asarray(
                        decoded[
                            "right_gripper_pose"
                        ]
                    ).copy(),

                "left_gripper_pose":
                    np.asarray(
                        decoded[
                            "left_gripper_pose"
                        ]
                    ).copy(),
            }

            T = len(
                metric_sample[
                    "right_gripper_pose"
                ]
            )

            bp = src.base_phase(
                key,
                T,
            )

            # ----------------------------------------------
            # EXACT validated equal-DeltaAcc search at multiple
            # PRE-FIXED intervention magnitudes.
            #
            # CPU only; same decoded sample and same base phase.
            # No additional WDS pass.
            # ----------------------------------------------

            dose = {}

            for frac in INTERVENTION_FRACS:
                rr = src.search_equal_delta(
                    metric_sample,
                    bp,
                    target_metric="rms_acceleration",
                    collateral_metric="mean_squared_jerk",
                    requested_delta=(
                        src.TA * float(frac)
                    ),
                )
                dose[float(frac)] = rr

            # Preserve legacy +5% fields for compatibility/audit.
            r = dose[0.05]

            matched = bool(
                r["matched"]
            )

            matched_any = any(
                bool(rr["matched"])
                for rr in dose.values()
            )

            metric_row = src.METRIC_BY_KEY[key]

            true_task = str(
                metric_row["task"]
            )

            row = {
                "clip": key,
                "task": true_task,

                "original_action_acc":
                    float(
                        metric_row[
                            "rms_acceleration"
                        ]
                    ),

                "original_action_msj":
                    float(
                        metric_row[
                            "mean_squared_jerk"
                        ]
                    ),

                "l2_moved_mm":
                    float(
                        metric_row[
                            "l2_moved_mm"
                        ]
                    ),

                "acc_outside":
                    int(
                        float(
                            metric_row[
                                "rms_acceleration"
                            ]
                        )
                        > src.TA
                    ),

                "msj_outside":
                    int(
                        float(
                            metric_row[
                                "mean_squared_jerk"
                            ]
                        )
                        > src.TJ
                    ),

                "acc_boundary_excess":
                    float(
                        float(
                            metric_row[
                                "rms_acceleration"
                            ]
                        )
                        / src.TA
                        - 1.0
                    ),

                "msj_boundary_excess":
                    float(
                        float(
                            metric_row[
                                "mean_squared_jerk"
                            ]
                        )
                        / src.TJ
                        - 1.0
                    ),

                "joint_boundary_severity":
                    max(
                        float(
                            metric_row[
                                "rms_acceleration"
                            ]
                        )
                        / src.TA,

                        float(
                            metric_row[
                                "mean_squared_jerk"
                            ]
                        )
                        / src.TJ,
                    ),

                "requested_delta_acc":
                    float(
                        src.TARGET_DELTA_ACC
                    ),

                "actual_delta_acc":
                    float(
                        r["actual_delta"]
                    ),

                "acc_match_relerr":
                    float(
                        r[
                            "relative_match_error"
                        ]
                    ),

                "acc_matched":
                    int(matched),

                "acc_collateral_delta_msj":
                    float(
                        r[
                            "collateral_delta"
                        ]
                    ),

                "acc_perturb_frames":
                    float(
                        r["amp_frames"]
                    ),

                "deploy_acc_sensitivity_mean_mm":
                    np.nan,

                "deploy_acc_sensitivity_p95_mm":
                    np.nan,

                "deploy_acc_sensitivity_final_mean_mm":
                    np.nan,
            }

            # Per-intervention search metadata and model outputs.
            for frac, rr in dose.items():
                tag = int(round(frac * 100.0))

                row[
                    f"acc_{tag:02d}_requested_delta"
                ] = float(
                    src.TA * frac
                )

                row[
                    f"acc_{tag:02d}_actual_delta"
                ] = float(
                    rr["actual_delta"]
                )

                row[
                    f"acc_{tag:02d}_match_relerr"
                ] = float(
                    rr["relative_match_error"]
                )

                row[
                    f"acc_{tag:02d}_matched"
                ] = int(
                    bool(rr["matched"])
                )

                row[
                    f"acc_{tag:02d}_collateral_delta_msj"
                ] = float(
                    rr["collateral_delta"]
                )

                row[
                    f"acc_{tag:02d}_perturb_frames"
                ] = float(
                    rr["amp_frames"]
                )

                row[
                    f"acc_{tag:02d}_prediction_shift_mean_mm"
                ] = np.nan

                row[
                    f"acc_{tag:02d}_prediction_shift_p95_mm"
                ] = np.nan

                row[
                    f"acc_{tag:02d}_prediction_shift_final_mean_mm"
                ] = np.nan

            # ----------------------------------------------
            # Multi-dose intervention inference.
            #
            # - ORIGINAL WDS: still one pass
            # - official build: once/sample
            # - temporal variants: in-memory copies
            # - PointWorld: one batched forward
            # ----------------------------------------------

            if matched_any:

                # Official build exactly ONCE for this sample.
                built = src.old.OFFICIAL_BUILD(
                    decoded,
                    domain=domain,
                    robot_sampler=robot_sampler,
                    max_robot_points=args.max_robot_points,
                    deterministic=True,
                    seed=(
                        args.seed
                        + tester.rank
                    ),
                    force_single_arm=force_single_arm,
                )

                built_key = str(
                    built.get(
                        "__key__",
                        "",
                    )
                )

                if built_key != key:
                    raise RuntimeError(
                        f"{key}: built key mismatch={built_key}"
                    )

                if (
                    built["robot_flows"].shape[0]
                    != T
                ):
                    raise RuntimeError(
                        f"{key}: T mismatch"
                    )

                # ------------------------------------------
                # Base trajectory.
                # ------------------------------------------
                base_unreleased = src.apply_phase(
                    copy.deepcopy(built),
                    bp,
                )

                variants = [
                    (
                        "base",
                        None,
                        base_unreleased,
                    )
                ]

                # ------------------------------------------
                # Matched intervention trajectories.
                # Spatial path/intent is unchanged;
                # temporal phase only is changed.
                # ------------------------------------------
                for frac in INTERVENTION_FRACS:

                    rr = dose[float(frac)]

                    if not bool(
                        rr["matched"]
                    ):
                        continue

                    tag = int(
                        round(
                            frac * 100.0
                        )
                    )

                    v = src.apply_phase(
                        copy.deepcopy(built),
                        rr["phase"],
                    )

                    # Exact start preservation.
                    if not np.allclose(
                        base_unreleased[
                            "robot_flows"
                        ][0],
                        v[
                            "robot_flows"
                        ][0],
                        atol=1e-7,
                        rtol=0,
                    ):
                        raise RuntimeError(
                            f"{key}: "
                            f"+{tag}% start changed"
                        )

                    # Exact endpoint preservation.
                    if not np.allclose(
                        base_unreleased[
                            "robot_flows"
                        ][-1],
                        v[
                            "robot_flows"
                        ][-1],
                        atol=1e-7,
                        rtol=0,
                    ):
                        raise RuntimeError(
                            f"{key}: "
                            f"+{tag}% endpoint changed"
                        )

                    variants.append(
                        (
                            tag,
                            float(frac),
                            v,
                        )
                    )

                # ------------------------------------------
                # Official deterministic release pipeline.
                # gather_features() recomputes robot
                # velocity/acceleration per variant.
                # ------------------------------------------
                released = []

                for label, frac, sample in variants:

                    sample = release_process(
                        sample,
                        args,
                        domain,
                        tester.rank,
                    )

                    released.append(
                        (
                            label,
                            frac,
                            sample,
                        )
                    )

                # ONE collated batch:
                # [base, +5%, +10%, +15%, +20%]
                # with unmatched levels omitted.
                batch = custom_collate_fn(
                    [
                        item[2]
                        for item in released
                    ],
                    args=args,
                )

                # Current observed scene geometry/support
                # must remain identical.
                #
                # scene_features is intentionally NOT
                # equality-constrained because it contains
                # action-conditioned information such as
                # dist2robot.
                for field in [
                    "scene_flows",
                    "scene_exists",
                ]:

                    ref = batch[field][0]

                    for vi in range(
                        1,
                        batch[field].shape[0],
                    ):
                        if not torch.allclose(
                            ref,
                            batch[field][vi],
                            atol=1e-7,
                            rtol=0,
                        ):
                            raise RuntimeError(
                                f"{key}: "
                                f"{field} changed "
                                f"for variant={vi}"
                            )

                g = to_device(
                    batch,
                    device,
                )

                # ONE GPU forward for base + all matched doses.
                src.old.reset()

                pred = tester.model(
                    g,
                    training=False,
                )["scene_flows"]

                model_pairs += (
                    len(released) - 1
                )

                # Base prediction.
                p0 = pred[0:1]

                # DEPLOYABLE support only.
                # scene_exists(t=0); NO GT future mask.
                exists = (
                    g[
                        "scene_exists"
                    ][0:1]
                    .bool()
                )

                if (
                    exists.ndim == 4
                    and
                    exists.shape[-1] == 1
                ):
                    exists = (
                        exists.squeeze(-1)
                    )

                # ------------------------------------------
                # Prediction response at every intervention
                # magnitude.
                # ------------------------------------------
                for vi in range(
                    1,
                    len(released),
                ):

                    tag = int(
                        released[vi][0]
                    )

                    pi = pred[
                        vi:vi + 1
                    ]

                    shift = (
                        torch.linalg.vector_norm(
                            pi - p0,
                            dim=-1,
                        )
                    )

                    support_t0 = (
                        exists[:, 0:1, :]
                        .expand_as(shift)
                    )

                    mean_mm, p95_mm = (
                        finite_stats_mm(
                            shift,
                            support_t0,
                        )
                    )

                    final_vals = (
                        shift[:, -1, :]
                        [
                            exists[:, 0, :]
                        ]
                        .float()
                    )

                    if (
                        final_vals.numel()
                        > 0
                    ):
                        final_mean_mm = (
                            float(
                                final_vals
                                .mean()
                                .cpu()
                            )
                            * 1000.0
                        )
                    else:
                        final_mean_mm = (
                            np.nan
                        )

                    row[
                        f"acc_{tag:02d}_prediction_shift_mean_mm"
                    ] = mean_mm

                    row[
                        f"acc_{tag:02d}_prediction_shift_p95_mm"
                    ] = p95_mm

                    row[
                        f"acc_{tag:02d}_prediction_shift_final_mean_mm"
                    ] = final_mean_mm

                    # Keep existing +5% primary fields
                    # for exact comparison with deployable
                    # sensitivity experiment.
                    if tag == 5:

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
                    f"[{n}/"
                    f"{len(src.TARGET_KEYS)}] "
                    f"{key} | "
                    f"task={true_task} | "
                    f"matched={matched} | "
                    f"mean="
                    f"{row['deploy_acc_sensitivity_mean_mm']}"
                )

            # Stop immediately once all targets found.
            if (
                len(found)
                == len(src.TARGET_KEYS)
            ):
                break

    # ========================================================
    # Final audits
    # ========================================================

    missing = (
        src.TARGET_KEYS
        - found
    )

    if missing:
        raise RuntimeError(
            f"Missing targets="
            f"{len(missing)} | "
            f"first="
            f"{sorted(missing)[:10]}"
        )

    d = pd.DataFrame(rows)

    if d["clip"].duplicated().any():
        raise RuntimeError(
            "Duplicate output clips"
        )

    got_counts = (
        d["task"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    assert got_counts == expected_by_task, (
        got_counts,
        expected_by_task,
    )

    # Combined output.
    d.to_csv(
        OUT_COMBINED,
        index=False,
    )

    # Per-task compatibility outputs.
    filenames = {
        "task-0006":
            "task0006__acc_intervention_dose_response.csv",

        "task-0008":
            "task0008__acc_intervention_dose_response.csv",

        "task-0010":
            "task0010__acc_intervention_dose_response.csv",
    }

    for task, filename in filenames.items():

        q = d[
            d["task"] == task
        ].copy()

        q.to_csv(
            OUTDIR / filename,
            index=False,
        )

    print()
    print("=" * 100)
    print("ONE-PASS ORIGINAL-WDS RESULT")
    print("=" * 100)

    print(
        "RAW WDS SAMPLES SCANNED =",
        scanned,
    )

    print(
        "TARGET FOUND =",
        len(d),
    )

    print(
        "MODEL BASE+ACC PAIRS =",
        model_pairs,
    )

    print()
    print("BY TASK:")

    for task in [
        "task-0006",
        "task-0008",
        "task-0010",
    ]:

        q = d[
            d["task"] == task
        ]

        qm = q[
            q["acc_matched"] == 1
        ]

        print(
            task,
            "| found=",
            len(q),
            "| matched=",
            int(
                q[
                    "acc_matched"
                ].sum()
            ),
            "| median deploy mean=",
            qm[
                "deploy_acc_sensitivity_mean_mm"
            ].median(),
        )

    print()
    print(
        "TOTAL MATCHED =",
        int(
            d[
                "acc_matched"
            ].sum()
        ),
    )

    print(
        "OUTPUT =",
        OUT_COMBINED,
    )

    print(
        "SUCCESS = True"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
