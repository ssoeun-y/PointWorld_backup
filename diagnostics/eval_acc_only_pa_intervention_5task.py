#!/usr/bin/env python3

import os
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import webdataset as wds

# search_equal_delta implementation only.
# Old TARGET_KEYS / old split are NOT used.
os.environ["PW_TARGET_TASKS"] = "task-0006"
os.environ["PW_TAG"] = "acc_only_pa_final"
os.environ["PW_CONDITION"] = "original"
os.environ.setdefault("PW_MATCH_TOL", "0.05")
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

from diagnostics import eval_equal_physical_delta_global_heldout as src
from diagnostics import behavior_object_mapping_smoke_task0006 as omap

from diagnostics.behavior_object_mpjpe_pa_smoke_task0006 import (
    p_mpjpe_single,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "pretrained_checkpoints/large-droid+behavior/model-best.pt"
)

NORM_STATS = (
    ROOT
    / "stats/droid_behavior"
)

TASK_DIRS = {
    "task-0000":
        ROOT.parent / "datasets/behavior/wds_task0000",

    "task-0001":
        ROOT.parent / "datasets/behavior/wds_task0001",

    "task-0006":
        ROOT.parent / "datasets/behavior/wds_task0006",

    "task-0008":
        ROOT.parent / "datasets/behavior/wds_task0008",

    "task-0010":
        ROOT.parent / "datasets/behavior/wds_task0010",
}

EXPECTED_TARGETS = {
    "task-0000": 383,
    "task-0001": 2307,
    "task-0006": 1823,
    "task-0008": 1824,
    "task-0010": 1610,
}

TARGET_CSV = (
    ROOT
    / "diagnostics/results/"
      "object_pa_mpjpe_acc_5task_current10457/"
      "analysis_5task/"
      "merged_per_clip_metrics_5task_clean.csv"
)

OUT_ROOT = (
    ROOT
    / "diagnostics/results/"
      "acc_only_pa_intervention_5task_current10457"
)

DOSES = (
    0.05,
    0.10,
    0.15,
    0.20,
)

MIN_POINTS = 8
SEED = 42


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {
        k: (
            v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


def arr(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    return np.asarray(x)


def mask_array(x):
    x = arr(x)

    if x.shape[-1] == 1:
        x = x[..., 0]

    return x


def short_name(full_name):
    parts = [
        x
        for x in str(full_name).split("__")
        if x
    ]

    for x in parts:
        if x in {
            "World",
            "base_link",
            "visuals",
        }:
            continue

        if x.startswith("scene_"):
            continue

        return x

    return str(full_name)


def release_process(
    sample,
    args,
    domain,
    rank,
):
    sample = sample_cameras(
        sample,
        min_num_cameras=args.eval_min_num_cameras,
        max_num_cameras=args.eval_max_num_cameras,
        deterministic=True,
        seed=args.seed + rank,
    )

    sample = canonicalize_gripper_keys_and_flags(
        sample
    )

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


def same_tensor(a, b):
    if a.dtype.is_floating_point:
        return torch.allclose(
            a,
            b,
            atol=1e-7,
            rtol=0,
        )

    return torch.equal(
        a,
        b,
    )


def save_checkpoint(
    outdir,
    object_rows,
    clip_rows,
    status_rows,
):
    pd.DataFrame(
        object_rows
    ).to_csv(
        outdir / "per_object_intervention.csv",
        index=False,
    )

    pd.DataFrame(
        clip_rows
    ).to_csv(
        outdir / "per_clip_intervention.csv",
        index=False,
    )

    pd.DataFrame(
        status_rows
    ).to_csv(
        outdir / "clip_status.csv",
        index=False,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_DIRS),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    cli = parser.parse_args()

    task = cli.task
    smoke = cli.smoke

    data_dir = TASK_DIRS[task]

    outdir = (
        OUT_ROOT
        / ("smoke" if smoke else task)
    )

    if smoke:
        outdir = (
            OUT_ROOT
            / "smoke"
            / task
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # CURRENT 7,947 ELIGIBLE POPULATION
    # ========================================================

    targets_df = pd.read_csv(
        TARGET_CSV
    )

    targets_df["task"] = (
        targets_df["task"]
        .astype(str)
    )

    targets_df["clip"] = (
        targets_df["clip"]
        .astype(str)
    )

    q = targets_df[
        targets_df["task"] == task
    ].copy()

    target_keys = set(
        q["clip"]
    )

    if (
        len(target_keys)
        != EXPECTED_TARGETS[task]
    ):
        raise RuntimeError(
            f"{task}: target count "
            f"{len(target_keys)} != "
            f"{EXPECTED_TARGETS[task]}"
        )

    print("=" * 100)
    print("FINAL ACC-ONLY PA INTERVENTION")
    print("=" * 100)
    print("TASK =", task)
    print("DATA =", data_dir)
    print("TARGET CLIPS =", len(target_keys))
    print("DOSES = baseline, +5%, +10%, +15%, +20%")
    print("DELTA DEFINITION = fraction of EACH CLIP'S ORIGINAL RMS ACC")
    print("MSJ/JERK = NOT USED")
    print("70/30 SPLIT = NOT USED")
    print("SMOKE =", smoke)
    print("=" * 100)

    # ========================================================
    # Object-ID propagation.
    # ========================================================

    omap.dlmod.build_flow_sample = (
        omap.build_flow_sample_with_object_ids
    )

    omap.pmod.grid_sample_transform = (
        omap.grid_sample_with_object_ids
    )

    omap.pmod.gather = (
        omap.gather_with_object_ids
    )

    # ========================================================
    # PointWorld
    # ========================================================

    args = parse_args(
        skip_command_line=True
    )

    args.seed = SEED
    args.model_path = str(MODEL_PATH)
    args.norm_stats_path = str(NORM_STATS)

    args.domains = ["behavior"]
    args.data_dirs = [str(data_dir)]

    args.batch_size = 1
    args.num_workers = 0

    if hasattr(
        args,
        "eval_num_workers",
    ):
        args.eval_num_workers = 0

    args.device = "cuda"
    args.eval_skip_viz = True

    reset_seed(SEED)

    tester = Tester(args)
    tester.model.eval()

    domain = "behavior"
    device = tester.device

    # ========================================================
    # Robot sampler
    # ========================================================

    robot_sampler = TorchRobotSampler(
        urdf_path=resolve_robot_urdf(domain),
        gripper_only=RELEASE_GRIPPER_ONLY,
        device="cpu",
    )

    force_single_arm = False

    if getattr(
        args,
        "_eval_override_active",
        False,
    ):
        force_single_arm = (
            domain
            in getattr(
                args,
                "eval_force_single_arm_domains",
                set(),
            )
        )

    # ========================================================
    # Direct one-pass WDS
    # ========================================================

    shard_paths = dlmod.gather_shard_paths(
        str(data_dir),
        "test",
        rank=tester.rank,
    )

    if not shard_paths:
        raise RuntimeError(
            f"No test shards: {data_dir}"
        )

    random.Random(
        args.seed + tester.rank
    ).shuffle(
        shard_paths
    )

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
        seed=args.seed + tester.rank,
        handler=wds.warn_and_continue,
    )

    found = set()

    object_rows = []
    clip_rows = []
    status_rows = []

    scanned = 0
    model_forwards = 0
    all_dose_matched = 0

    with torch.inference_mode():

        for raw in dataset:

            scanned += 1

            key = str(
                raw.get("__key__", "")
            )

            if key not in target_keys:
                continue

            if key in found:
                raise RuntimeError(
                    f"duplicate target {key}"
                )

            found.add(key)

            decoded = decode_data(
                raw,
                domain=domain,
            )

            decoded_key = str(
                decoded.get(
                    "__key__",
                    key,
                )
            )

            if decoded_key != key:
                raise RuntimeError(
                    f"{key}: decoded key mismatch"
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

            base_phase = np.linspace(
                0.0,
                1.0,
                T,
            )

            base_metrics = src.pose_metrics(
                metric_sample,
                base_phase,
            )

            base_acc = float(
                base_metrics[
                    "rms_acceleration"
                ]
            )

            status = {
                "task": task,
                "clip": key,
                "base_acc_native": base_acc,
                "base_acc_si_m_s2":
                    base_acc * 25.0,
                "all_doses_matched": 0,
                "inferred": 0,
                "eligible_objects": 0,
            }

            if (
                not np.isfinite(base_acc)
                or base_acc <= 1e-12
            ):
                status[
                    "status"
                ] = "INVALID_BASE_ACC"

                status_rows.append(
                    status
                )
                continue

            # =================================================
            # ACC-ONLY DOSE SEARCH
            #
            # Legacy helper requires a collateral_metric
            # argument, but collateral does NOT affect search.
            # Passing RMS Acc itself ensures Jerk/MSJ is not
            # part of the intervention in any way.
            # =================================================

            dose = {}

            for frac in DOSES:

                requested_delta = (
                    base_acc * frac
                )

                rr = src.search_equal_delta(
                    metric_sample,
                    base_phase,
                    target_metric=(
                        "rms_acceleration"
                    ),
                    collateral_metric=(
                        "rms_acceleration"
                    ),
                    requested_delta=(
                        requested_delta
                    ),
                )

                dose[frac] = rr

                tag = int(
                    round(
                        frac * 100
                    )
                )

                status[
                    f"acc{tag:02d}_matched"
                ] = int(
                    bool(
                        rr["matched"]
                    )
                )

                status[
                    f"acc{tag:02d}_requested_delta_native"
                ] = requested_delta

                status[
                    f"acc{tag:02d}_actual_delta_native"
                ] = float(
                    rr[
                        "actual_delta"
                    ]
                )

                status[
                    f"acc{tag:02d}_match_relerr"
                ] = float(
                    rr[
                        "relative_match_error"
                    ]
                )

            matched = all(
                bool(
                    dose[f][
                        "matched"
                    ]
                )
                for f in DOSES
            )

            status[
                "all_doses_matched"
            ] = int(matched)

            if not matched:
                status[
                    "status"
                ] = "DOSE_MATCH_FAIL"

                status_rows.append(
                    status
                )
                continue

            all_dose_matched += 1

            # =================================================
            # Official build ONCE
            # =================================================

            built = (
                omap.build_flow_sample_with_object_ids(
                    decoded,
                    domain=domain,
                    robot_sampler=robot_sampler,
                    max_robot_points=(
                        args.max_robot_points
                    ),
                    deterministic=True,
                    seed=args.seed + tester.rank,
                    force_single_arm=force_single_arm,
                )
            )

            base_unreleased = (
                src.apply_phase(
                    copy.deepcopy(
                        built
                    ),
                    base_phase,
                )
            )

            # Two identical baseline copies provide a direct
            # estimate of the model's stochastic floor.
            variants = [
                (
                    "base",
                    0.0,
                    copy.deepcopy(
                        base_unreleased
                    ),
                ),
                (
                    "base_repeat",
                    0.0,
                    copy.deepcopy(
                        base_unreleased
                    ),
                ),
            ]

            for frac in DOSES:

                tag = int(
                    round(
                        frac * 100
                    )
                )

                v = src.apply_phase(
                    copy.deepcopy(
                        built
                    ),
                    dose[frac][
                        "phase"
                    ],
                )

                # Hard goal/path endpoint audit.
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
                        f"{key}: +{tag}% start changed"
                    )

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
                        f"{key}: +{tag}% end changed"
                    )

                variants.append(
                    (
                        f"acc{tag:02d}",
                        frac,
                        v,
                    )
                )

            released = []

            for (
                label,
                frac,
                sample,
            ) in variants:

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

            batch_cpu = custom_collate_fn(
                [
                    x[2]
                    for x in released
                ],
                args=args,
            )

            # Scene and object identity must not change.
            for field in [
                "scene_flows",
                "scene_exists",
                "scene_context_mask",
                "scene_supervised_mask",
                "scene_object_id",
                "gt_scene_flows",
            ]:

                if field not in batch_cpu:
                    raise RuntimeError(
                        f"{key}: missing {field}"
                    )

                ref = batch_cpu[field][0]

                for vi in range(
                    1,
                    len(released),
                ):
                    if not same_tensor(
                        ref,
                        batch_cpu[field][vi],
                    ):
                        raise RuntimeError(
                            f"{key}: {field} "
                            f"changed at variant {vi}"
                        )

            batch_gpu = to_device(
                batch_cpu,
                device,
            )

            # =================================================
            # ONE GPU FORWARD:
            # base + repeat + 5 + 10 + 15 + 20
            # =================================================

            outputs = tester.model(
                batch_gpu,
                training=False,
            )

            pred = (
                outputs[
                    "scene_flows"
                ]
                .detach()
                .cpu()
                .numpy()
            )

            model_forwards += 1
            status["inferred"] = 1

            labels = mask_array(
                batch_cpu[
                    "scene_object_id"
                ]
            )[0]

            context = (
                mask_array(
                    batch_cpu[
                        "scene_context_mask"
                    ]
                )[0]
                .astype(bool)
            )

            exists = (
                mask_array(
                    batch_cpu[
                        "scene_exists"
                    ]
                )[0]
                .astype(bool)
            )

            supervised = (
                mask_array(
                    batch_cpu[
                        "scene_supervised_mask"
                    ]
                )[0]
                .astype(bool)
            )

            moved = (
                mask_array(
                    batch_cpu[
                        "scene_moved_mask"
                    ]
                )[0]
                .astype(bool)
            )

            gt = arr(
                batch_cpu[
                    "gt_scene_flows"
                ]
            )[0]

            valid = (
                (~context)
                & exists
                & supervised
            )

            if key not in omap.OBJECT_TABLE:
                raise RuntimeError(
                    f"{key}: object table missing"
                )

            table = omap.OBJECT_TABLE[key]

            eligible_objects = []

            # =================================================
            # RIGID OBJECT LOOP
            # =================================================

            for oid, mesh_name in table.items():

                oid = int(oid)

                object_mask = (
                    labels == oid
                )

                moved_support = int(
                    (
                        object_mask
                        & moved
                        & valid
                    ).sum()
                )

                if moved_support <= 0:
                    continue

                values = {
                    label: []
                    for label, _, _
                    in released
                }

                gt_shape_check = []

                for t in range(
                    1,
                    gt.shape[0],
                ):

                    m = (
                        object_mask[t]
                        & valid[t]
                        & object_mask[0]
                        & exists[0]
                    )

                    n = int(
                        m.sum()
                    )

                    if n < MIN_POINTS:
                        continue

                    gt0 = gt[
                        0,
                        m,
                    ]

                    centered = (
                        gt0
                        - gt0.mean(
                            axis=0,
                            keepdims=True,
                        )
                    )

                    if (
                        np.linalg.matrix_rank(
                            centered,
                            tol=1e-8,
                        )
                        < 2
                    ):
                        continue

                    # GT shape consistency sanity only.
                    try:
                        gt_pa, _ = (
                            p_mpjpe_single(
                                gt[t, m],
                                gt0,
                            )
                        )
                    except Exception:
                        continue

                    gt_shape_check.append(
                        float(gt_pa)
                        * 1000.0
                    )

                    ok = True

                    frame_vals = {}

                    for vi, (
                        label,
                        _,
                        _,
                    ) in enumerate(
                        released
                    ):

                        try:
                            pa, _ = (
                                p_mpjpe_single(
                                    pred[
                                        vi,
                                        t,
                                        m,
                                    ],
                                    gt0,
                                )
                            )
                        except Exception:
                            ok = False
                            break

                        frame_vals[label] = (
                            float(pa)
                            * 1000.0
                        )

                    if not ok:
                        gt_shape_check.pop()
                        continue

                    for label in values:
                        values[
                            label
                        ].append(
                            frame_vals[
                                label
                            ]
                        )

                if (
                    not gt_shape_check
                    or any(
                        len(values[x])
                        != len(
                            gt_shape_check
                        )
                        for x in values
                    )
                ):
                    continue

                row = {
                    "task": task,
                    "clip": key,
                    "object_id": oid,
                    "object_full_name":
                        str(mesh_name),
                    "object_short_name":
                        short_name(
                            mesh_name
                        ),
                    "final_object_point_count":
                        int(
                            object_mask[
                                0
                            ].sum()
                        ),
                    "eligible_future_frames":
                        len(
                            gt_shape_check
                        ),
                    "base_acc_si_m_s2":
                        base_acc * 25.0,
                    "gt_shape_check_pa_mean_mm":
                        float(
                            np.mean(
                                gt_shape_check
                            )
                        ),
                }

                for label in values:
                    row[
                        f"{label}_pa_mean_mm"
                    ] = float(
                        np.mean(
                            values[label]
                        )
                    )

                row[
                    "base_repeat_absdiff_mm"
                ] = abs(
                    row[
                        "base_repeat_pa_mean_mm"
                    ]
                    -
                    row[
                        "base_pa_mean_mm"
                    ]
                )

                for frac in DOSES:

                    tag = int(
                        round(
                            frac * 100
                        )
                    )

                    label = (
                        f"acc{tag:02d}"
                    )

                    row[
                        f"{label}_delta_pa_mm"
                    ] = (
                        row[
                            f"{label}_pa_mean_mm"
                        ]
                        -
                        row[
                            "base_pa_mean_mm"
                        ]
                    )

                    row[
                        f"{label}_pa_ratio"
                    ] = (
                        row[
                            f"{label}_pa_mean_mm"
                        ]
                        /
                        row[
                            "base_pa_mean_mm"
                        ]
                        if row[
                            "base_pa_mean_mm"
                        ] > 0
                        else np.nan
                    )

                    row[
                        f"{label}_actual_acc_si_m_s2"
                    ] = (
                        base_acc
                        +
                        float(
                            dose[frac][
                                "actual_delta"
                            ]
                        )
                    ) * 25.0

                object_rows.append(
                    row
                )

                eligible_objects.append(
                    row
                )

            status[
                "eligible_objects"
            ] = len(
                eligible_objects
            )

            status[
                "status"
            ] = (
                "OK"
                if eligible_objects
                else
                "NO_ELIGIBLE_OBJECT"
            )

            status_rows.append(
                status
            )

            # =================================================
            # ONE CLIP = ONE STATISTICAL UNIT
            # =================================================

            if eligible_objects:

                cr = {
                    "task": task,
                    "clip": key,
                    "eligible_objects":
                        len(
                            eligible_objects
                        ),
                    "base_acc_si_m_s2":
                        base_acc * 25.0,
                }

                labels_to_save = [
                    "base",
                    "base_repeat",
                    "acc05",
                    "acc10",
                    "acc15",
                    "acc20",
                ]

                for label in labels_to_save:

                    cr[
                        f"{label}_pa_mean_mm"
                    ] = float(
                        np.mean(
                            [
                                r[
                                    f"{label}_pa_mean_mm"
                                ]
                                for r
                                in eligible_objects
                            ]
                        )
                    )

                cr[
                    "base_repeat_absdiff_mm"
                ] = abs(
                    cr[
                        "base_repeat_pa_mean_mm"
                    ]
                    -
                    cr[
                        "base_pa_mean_mm"
                    ]
                )

                for frac in DOSES:

                    tag = int(
                        round(
                            frac * 100
                        )
                    )

                    label = (
                        f"acc{tag:02d}"
                    )

                    cr[
                        f"{label}_delta_pa_mm"
                    ] = (
                        cr[
                            f"{label}_pa_mean_mm"
                        ]
                        -
                        cr[
                            "base_pa_mean_mm"
                        ]
                    )

                    cr[
                        f"{label}_pa_ratio"
                    ] = (
                        cr[
                            f"{label}_pa_mean_mm"
                        ]
                        /
                        cr[
                            "base_pa_mean_mm"
                        ]
                        if cr[
                            "base_pa_mean_mm"
                        ] > 0
                        else np.nan
                    )

                    cr[
                        f"{label}_actual_acc_si_m_s2"
                    ] = (
                        base_acc
                        +
                        float(
                            dose[frac][
                                "actual_delta"
                            ]
                        )
                    ) * 25.0

                clip_rows.append(cr)

            if (
                model_forwards <= 5
                or
                model_forwards % 25
                == 0
            ):
                print(
                    f"[{task}] "
                    f"found={len(found)}/"
                    f"{len(target_keys)} "
                    f"matched="
                    f"{all_dose_matched} "
                    f"forward="
                    f"{model_forwards} "
                    f"objects="
                    f"{len(eligible_objects)} "
                    f"{key}"
                )

            if (
                model_forwards % 25
                == 0
            ):
                save_checkpoint(
                    outdir,
                    object_rows,
                    clip_rows,
                    status_rows,
                )

            if (
                smoke
                and model_forwards >= 1
            ):
                break

            if (
                len(found)
                == len(target_keys)
            ):
                break

    save_checkpoint(
        outdir,
        object_rows,
        clip_rows,
        status_rows,
    )

    if not smoke:

        missing = (
            target_keys - found
        )

        if missing:
            raise RuntimeError(
                f"{task}: missing "
                f"{len(missing)} targets, "
                f"first={sorted(missing)[:10]}"
            )

    print()
    print("=" * 100)
    print(
        "SMOKE PASS"
        if smoke
        else
        "FINAL TASK COMPLETE"
    )
    print("=" * 100)
    print("TASK =", task)
    print("TARGET FOUND =", len(found))
    print(
        "ALL-DOSE MATCHED =",
        all_dose_matched,
    )
    print(
        "MODEL FORWARDS =",
        model_forwards,
    )
    print(
        "ELIGIBLE CLIP ROWS =",
        len(clip_rows),
    )
    print(
        "ELIGIBLE OBJECT ROWS =",
        len(object_rows),
    )
    print("OUTPUT =", outdir)
    print("=" * 100)


if __name__ == "__main__":
    main()
