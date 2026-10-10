#!/usr/bin/env python3

import os
import copy
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import webdataset as wds

# ============================================================
# EXPERIMENT FREEZE — MUST happen before importing src
# ============================================================

TASK = os.environ.get(
    "PA_INTERVENTION_TASKS",
    "task-0006",
).strip()

ALLOWED = {
    "task-0006",
    "task-0008",
    "task-0010",
}

if TASK not in ALLOWED:
    raise RuntimeError(
        f"Allowed tasks are {sorted(ALLOWED)}, got {TASK}"
    )

SMOKE = (
    os.environ.get(
        "PA_SMOKE",
        "0",
    ) == "1"
)

os.environ["PW_TARGET_TASKS"] = TASK
os.environ["PW_TAG"] = (
    f"pa_shape_intervention_{TASK}"
)
os.environ["PW_CONDITION"] = "original"
os.environ.setdefault("PW_MATCH_TOL", "0.05")
os.environ.setdefault("PW_DELTA_FRAC", "0.05")
os.environ.setdefault(
    "PW_MAX_PERTURB_FRAMES",
    "1.25",
)
os.environ.setdefault("PW_GRID_N", "161")


# ============================================================
# OFFICIAL / VALIDATED IMPORTS
# ============================================================

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dlmod

from dataset_components.decoders import decode_data
from dataset_components.cameras import sample_cameras
from dataset_components.robot import (
    canonicalize_gripper_keys_and_flags,
)
from dataset_components.pipeline import (
    apply_release_pipeline_to_sample,
)
from dataset_components.collate import (
    custom_collate_fn,
)
from dataset_components.constants import (
    RELEASE_GRIPPER_ONLY,
)

from robot_sampler import (
    RobotSampler as TorchRobotSampler,
)
from utils import resolve_robot_urdf

# EXACT validated intervention implementation.
from diagnostics import (
    eval_equal_physical_delta_global_heldout
    as src
)

# EXACT validated object-ID propagation.
from diagnostics import (
    behavior_object_mapping_smoke_task0006
    as omap
)

# EXACT validated PA-MPJPE implementation.
from diagnostics.behavior_object_mpjpe_pa_smoke_task0006 import (
    p_mpjpe_single,
)


# ============================================================
# FIXED PROTOCOL
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "diagnostics/data/"
      "pointworld_behavior_multitask_restored/"
      "behavior/wds"
)

MODEL_PATH = (
    ROOT
    / "pretrained_checkpoints/"
      "large-droid+behavior/model-best.pt"
)

NORM_STATS = (
    ROOT
    / "stats/droid_behavior"
)

INTERVENTION_FRACS = (
    0.05,
    0.10,
    0.15,
    0.20,
)

MIN_POINTS = 8
EVAL_SEED = 42

BASE_OUT = (
    ROOT
    / "diagnostics/results/"
      "object_pa_acc_intervention_3task"
)

if SMOKE:
    OUTDIR = (
        BASE_OUT
        / "smoke"
        / TASK
    )
else:
    OUTDIR = (
        BASE_OUT
        / TASK
    )

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def np_array(x):
    if isinstance(x, torch.Tensor):
        x = (
            x.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(x)


def squeeze_last_unit(x):
    x = np_array(x)

    if (
        x.ndim >= 1
        and x.shape[-1] == 1
    ):
        x = np.squeeze(
            x,
            axis=-1,
        )

    return x


def short_object_name(full_name):
    parts = [
        p for p in
        str(full_name).split("__")
        if p
    ]

    ignore = {
        "World",
        "base_link",
        "visuals",
    }

    for p in parts:
        if p in ignore:
            continue

        if p.startswith("scene_"):
            continue

        return p

    return str(full_name)


def release_process(
    sample,
    args,
    domain,
    rank,
):
    # EXACT same camera step as validated intervention.
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

    sample = (
        apply_release_pipeline_to_sample(
            sample,
            domain=domain,
            mode="test",
            args=args,
            has_bimanual_robot=True,
            rank=rank,
            include_scene_data=False,
            skip_scene_sampling=False,
        )
    )

    return sample


def tensors_equal(a, b):
    if not (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
    ):
        return False

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


def checkpoint(
    object_rows,
    clip_rows,
    status_rows,
):
    pd.DataFrame(
        object_rows
    ).to_csv(
        OUTDIR / "per_object_intervention.csv",
        index=False,
    )

    pd.DataFrame(
        clip_rows
    ).to_csv(
        OUTDIR / "per_clip_intervention.csv",
        index=False,
    )

    pd.DataFrame(
        status_rows
    ).to_csv(
        OUTDIR / "clip_status.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    reset_seed(EVAL_SEED)

    domain = "behavior"

    # --------------------------------------------------------
    # Exact intervention population from validated held-out set.
    # PW_TARGET_TASKS was frozen BEFORE importing src.
    # --------------------------------------------------------

    target_keys = set(
        str(x)
        for x in src.TARGET_KEYS
    )

    expected_count = len(
        target_keys
    )

    source_counts = (
        src.HELDOUT["task"]
        .value_counts()
        .to_dict()
    )

    print("=" * 100)
    print("PA SHAPE INTERVENTION")
    print("=" * 100)
    print("TASK              =", TASK)
    print("TARGETS           =", expected_count)
    print("SOURCE COUNTS     =", source_counts)
    print("SMOKE             =", SMOKE)
    print(
        "DOSES             =",
        INTERVENTION_FRACS,
    )
    print(
        "REFERENCE         = GT rigid-object shape at t=0"
    )
    print(
        "PRIMARY METRIC    = PA residual of future prediction vs t=0 shape"
    )
    print("=" * 100)

    if expected_count == 0:
        raise RuntimeError(
            "No intervention targets."
        )

    # --------------------------------------------------------
    # Object-ID propagation hooks.
    #
    # Only metadata/labels are propagated.
    # Official scene coordinates are untouched.
    # --------------------------------------------------------

    omap.dlmod.build_flow_sample = (
        omap.build_flow_sample_with_object_ids
    )

    omap.pmod.grid_sample_transform = (
        omap.grid_sample_with_object_ids
    )

    omap.pmod.gather = (
        omap.gather_with_object_ids
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    args = parse_args(
        skip_command_line=True
    )

    args.seed = EVAL_SEED
    args.model_path = str(
        MODEL_PATH
    )
    args.norm_stats_path = str(
        NORM_STATS
    )

    args.domains = [
        "behavior"
    ]

    args.data_dirs = [
        str(DATA_DIR)
    ]

    args.batch_size = 1
    args.num_workers = 0

    if hasattr(
        args,
        "eval_num_workers",
    ):
        args.eval_num_workers = 0

    args.device = "cuda"
    args.eval_skip_viz = True

    tester = Tester(args)
    tester.model.eval()

    device = tester.device

    # --------------------------------------------------------
    # Official RobotSampler initialized once.
    # --------------------------------------------------------

    urdf_path = resolve_robot_urdf(
        domain
    )

    robot_sampler = (
        TorchRobotSampler(
            urdf_path=urdf_path,
            gripper_only=(
                RELEASE_GRIPPER_ONLY
            ),
            device="cpu",
        )
    )

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
    # Exact original restored WDS used by validated intervention.
    # --------------------------------------------------------

    shard_paths = (
        dlmod.gather_shard_paths(
            str(DATA_DIR),
            "test",
            rank=tester.rank,
        )
    )

    if not shard_paths:
        raise RuntimeError(
            f"No test shards: {DATA_DIR}"
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
        seed=(
            args.seed
            + tester.rank
        ),
        handler=wds.warn_and_continue,
    )

    found = set()

    object_rows = []
    clip_rows = []
    status_rows = []

    scanned = 0
    inferred = 0
    all_dose_matched_count = 0

    with torch.inference_mode():

        for raw in dataset:

            scanned += 1

            raw_key = str(
                raw.get(
                    "__key__",
                    "",
                )
            )

            # Early filter.
            if raw_key not in target_keys:
                continue

            if raw_key in found:
                raise RuntimeError(
                    f"Duplicate target: {raw_key}"
                )

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
                    f"key mismatch: "
                    f"{raw_key} != {key}"
                )

            # ------------------------------------------------
            # Exact same temporal intervention search.
            # ------------------------------------------------

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

            dose = {}

            for frac in INTERVENTION_FRACS:

                rr = src.search_equal_delta(
                    metric_sample,
                    bp,
                    target_metric=(
                        "rms_acceleration"
                    ),
                    collateral_metric=(
                        "mean_squared_jerk"
                    ),
                    requested_delta=(
                        src.TA
                        * float(frac)
                    ),
                )

                dose[
                    float(frac)
                ] = rr

            all_matched = all(
                bool(
                    dose[float(frac)][
                        "matched"
                    ]
                )
                for frac
                in INTERVENTION_FRACS
            )

            metric_row = (
                src.METRIC_BY_KEY[
                    key
                ]
            )

            status = {
                "task":
                    TASK,

                "clip":
                    key,

                "original_action_acc_native":
                    float(
                        metric_row[
                            "rms_acceleration"
                        ]
                    ),

                "original_action_acc_si_m_s2":
                    float(
                        metric_row[
                            "rms_acceleration"
                        ]
                    )
                    * 25.0,

                "original_action_msj":
                    float(
                        metric_row[
                            "mean_squared_jerk"
                        ]
                    ),

                "all_doses_matched":
                    int(all_matched),

                "inferred":
                    0,

                "eligible_objects":
                    0,
            }

            for frac in INTERVENTION_FRACS:

                tag = int(
                    round(
                        frac * 100
                    )
                )

                rr = dose[
                    float(frac)
                ]

                status[
                    f"acc_{tag:02d}_matched"
                ] = int(
                    bool(
                        rr["matched"]
                    )
                )

                status[
                    f"acc_{tag:02d}_actual_delta_native"
                ] = float(
                    rr[
                        "actual_delta"
                    ]
                )

                status[
                    f"acc_{tag:02d}_actual_delta_si_m_s2"
                ] = (
                    float(
                        rr[
                            "actual_delta"
                        ]
                    )
                    * 25.0
                )

                status[
                    f"acc_{tag:02d}_match_relerr"
                ] = float(
                    rr[
                        "relative_match_error"
                    ]
                )

                status[
                    f"acc_{tag:02d}_collateral_delta_msj"
                ] = float(
                    rr[
                        "collateral_delta"
                    ]
                )

            found.add(key)

            # Primary paired dose-response requires all 4 doses.
            if not all_matched:

                status_rows.append(
                    status
                )

                if (
                    len(found) % 50 == 0
                ):
                    checkpoint(
                        object_rows,
                        clip_rows,
                        status_rows,
                    )

                continue

            all_dose_matched_count += 1

            # ------------------------------------------------
            # Official build ONCE, with object IDs carried
            # alongside the untouched official point tensors.
            # ------------------------------------------------

            built = (
                omap.build_flow_sample_with_object_ids(
                    decoded,
                    domain=domain,
                    robot_sampler=robot_sampler,
                    max_robot_points=(
                        args.max_robot_points
                    ),
                    deterministic=True,
                    seed=(
                        args.seed
                        + tester.rank
                    ),
                    force_single_arm=(
                        force_single_arm
                    ),
                )
            )

            built_key = str(
                built.get(
                    "__key__",
                    "",
                )
            )

            if built_key != key:
                raise RuntimeError(
                    f"{key}: built key mismatch "
                    f"{built_key}"
                )

            base_unreleased = (
                src.apply_phase(
                    copy.deepcopy(
                        built
                    ),
                    bp,
                )
            )

            variants = [
                (
                    "base",
                    0.0,
                    base_unreleased,
                )
            ]

            for frac in INTERVENTION_FRACS:

                rr = dose[
                    float(frac)
                ]

                tag = int(
                    round(
                        frac * 100
                    )
                )

                v = src.apply_phase(
                    copy.deepcopy(
                        built
                    ),
                    rr["phase"],
                )

                # Hard path endpoint audits.
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
                        f"{key}: +{tag}% "
                        f"start changed"
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
                        f"{key}: +{tag}% "
                        f"endpoint changed"
                    )

                variants.append(
                    (
                        f"acc{tag:02d}",
                        float(frac),
                        v,
                    )
                )

            # ------------------------------------------------
            # Official release pipeline for all variants.
            # ------------------------------------------------

            released = []

            for (
                label,
                frac,
                sample,
            ) in variants:

                sample = (
                    release_process(
                        sample,
                        args,
                        domain,
                        tester.rank,
                    )
                )

                released.append(
                    (
                        label,
                        frac,
                        sample,
                    )
                )

            batch_cpu = (
                custom_collate_fn(
                    [
                        item[2]
                        for item
                        in released
                    ],
                    args=args,
                )
            )

            # ------------------------------------------------
            # Scene / identity must be identical for all doses.
            # ------------------------------------------------

            invariant_fields = [
                "scene_flows",
                "scene_exists",
                "scene_context_mask",
                "scene_supervised_mask",
                "scene_object_id",
                "gt_scene_flows",
            ]

            for field in invariant_fields:

                if field not in batch_cpu:
                    raise RuntimeError(
                        f"{key}: missing field "
                        f"{field}"
                    )

                ref = batch_cpu[
                    field
                ][0]

                for vi in range(
                    1,
                    len(released),
                ):
                    cur = batch_cpu[
                        field
                    ][vi]

                    if not tensors_equal(
                        ref,
                        cur,
                    ):
                        raise RuntimeError(
                            f"{key}: invariant "
                            f"{field} changed at "
                            f"variant {vi}"
                        )

            # ------------------------------------------------
            # ONE GPU forward:
            # base + 5/10/15/20%
            # ------------------------------------------------

            batch_gpu = to_device(
                batch_cpu,
                device,
            )

            pred = tester.model(
                batch_gpu,
                training=False,
            )[
                "scene_flows"
            ]

            pred_np = (
                pred.detach()
                .cpu()
                .numpy()
            )

            inferred += 1
            status[
                "inferred"
            ] = 1

            # ------------------------------------------------
            # Base scene metadata / GT.
            # ------------------------------------------------

            labels_np = squeeze_last_unit(
                batch_cpu[
                    "scene_object_id"
                ]
            )[0]

            context_np = (
                squeeze_last_unit(
                    batch_cpu[
                        "scene_context_mask"
                    ]
                )[0]
                .astype(bool)
            )

            exists_np = (
                squeeze_last_unit(
                    batch_cpu[
                        "scene_exists"
                    ]
                )[0]
                .astype(bool)
            )

            supervised_np = (
                squeeze_last_unit(
                    batch_cpu[
                        "scene_supervised_mask"
                    ]
                )[0]
                .astype(bool)
            )

            if (
                "scene_moved_mask"
                not in batch_cpu
            ):
                raise RuntimeError(
                    f"{key}: "
                    "scene_moved_mask missing"
                )

            moved_np = (
                squeeze_last_unit(
                    batch_cpu[
                        "scene_moved_mask"
                    ]
                )[0]
                .astype(bool)
            )

            gt_np = np_array(
                batch_cpu[
                    "gt_scene_flows"
                ]
            )[0]

            official_valid_np = (
                (~context_np)
                & exists_np
                & supervised_np
            )

            if key not in omap.OBJECT_TABLE:
                raise RuntimeError(
                    f"{key}: object table missing"
                )

            table = (
                omap.OBJECT_TABLE[
                    key
                ]
            )

            eligible_objects = []

            # =================================================
            # RIGID OBJECT LOOP
            # =================================================

            for oid, mesh_name in table.items():

                oid = int(
                    oid
                )

                object_mask = (
                    labels_np
                    == oid
                )

                moved_support = int(
                    (
                        object_mask
                        & moved_np
                        & official_valid_np
                    ).sum()
                )

                # Same rule as main PA experiment:
                # moved mask selects the rigid object only.
                if moved_support <= 0:
                    continue

                frame_data = {
                    label: {
                        "pa": [],
                        "raw": [],
                        "scale": [],
                    }
                    for (
                        label,
                        _,
                        _
                    )
                    in released
                }

                gt_rigid_pa = []

                # Future frames only.
                for t in range(
                    1,
                    gt_np.shape[0],
                ):

                    m = (
                        object_mask[t]
                        & official_valid_np[t]
                        & object_mask[0]
                        & exists_np[0]
                    )

                    n = int(
                        m.sum()
                    )

                    if n < MIN_POINTS:
                        continue

                    # Actual rigid shape at t=0.
                    gt0 = gt_np[
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

                    rank = int(
                        np.linalg.matrix_rank(
                            centered,
                            tol=1e-8,
                        )
                    )

                    if rank < 2:
                        continue

                    # ----------------------------------------
                    # GT rigidity audit:
                    # future GT should match initial GT shape
                    # after similarity alignment.
                    # ----------------------------------------

                    gt_future = gt_np[
                        t,
                        m,
                    ]

                    try:
                        gt_pa, _ = (
                            p_mpjpe_single(
                                gt_future,
                                gt0,
                            )
                        )
                    except Exception:
                        continue

                    gt_rigid_pa.append(
                        float(gt_pa)
                        * 1000.0
                    )

                    # ----------------------------------------
                    # Prediction shape residual for each dose.
                    # SAME points, SAME t=0 reference.
                    # ----------------------------------------

                    for vi, (
                        label,
                        _,
                        _,
                    ) in enumerate(
                        released
                    ):

                        pred_t = (
                            pred_np[
                                vi,
                                t,
                                m,
                            ]
                        )

                        raw_m = float(
                            np.mean(
                                np.linalg.norm(
                                    pred_t
                                    - gt0,
                                    axis=-1,
                                )
                            )
                        )

                        try:
                            (
                                pa_m,
                                fitted_scale,
                            ) = (
                                p_mpjpe_single(
                                    pred_t,
                                    gt0,
                                )
                            )
                        except Exception:
                            continue

                        frame_data[
                            label
                        ]["raw"].append(
                            raw_m
                            * 1000.0
                        )

                        frame_data[
                            label
                        ]["pa"].append(
                            float(pa_m)
                            * 1000.0
                        )

                        frame_data[
                            label
                        ]["scale"].append(
                            float(
                                fitted_scale
                            )
                        )

                # Need a common eligible frame set.
                if (
                    not gt_rigid_pa
                    or
                    any(
                        len(
                            frame_data[
                                label
                            ]["pa"]
                        )
                        != len(
                            gt_rigid_pa
                        )
                        for label
                        in frame_data
                    )
                ):
                    continue

                full_name = str(
                    mesh_name
                )

                short_name = (
                    short_object_name(
                        full_name
                    )
                )

                object_row = {
                    "task":
                        TASK,

                    "clip":
                        key,

                    "object_id":
                        oid,

                    "object_full_name":
                        full_name,

                    "object_short_name":
                        short_name,

                    "final_object_point_count":
                        int(
                            object_mask[
                                0
                            ].sum()
                        ),

                    "official_moved_support_pointframes":
                        moved_support,

                    "eligible_future_frames":
                        len(
                            gt_rigid_pa
                        ),

                    "gt_future_vs_gt0_pa_mean_mm":
                        float(
                            np.mean(
                                gt_rigid_pa
                            )
                        ),

                    "gt_future_vs_gt0_pa_max_mm":
                        float(
                            np.max(
                                gt_rigid_pa
                            )
                        ),

                    "original_action_acc_si_m_s2":
                        status[
                            "original_action_acc_si_m_s2"
                        ],

                    "all_doses_matched":
                        1,
                }

                for label in frame_data:

                    vals_pa = np.asarray(
                        frame_data[
                            label
                        ]["pa"],
                        dtype=float,
                    )

                    vals_raw = np.asarray(
                        frame_data[
                            label
                        ]["raw"],
                        dtype=float,
                    )

                    vals_scale = np.asarray(
                        frame_data[
                            label
                        ]["scale"],
                        dtype=float,
                    )

                    object_row[
                        f"{label}_pa_ref_mean_mm"
                    ] = float(
                        np.mean(
                            vals_pa
                        )
                    )

                    object_row[
                        f"{label}_pa_ref_median_mm"
                    ] = float(
                        np.median(
                            vals_pa
                        )
                    )

                    object_row[
                        f"{label}_raw_ref_mean_mm"
                    ] = float(
                        np.mean(
                            vals_raw
                        )
                    )

                    object_row[
                        f"{label}_scale_mean"
                    ] = float(
                        np.mean(
                            vals_scale
                        )
                    )

                base_pa = object_row[
                    "base_pa_ref_mean_mm"
                ]

                for frac in INTERVENTION_FRACS:

                    tag = int(
                        round(
                            frac * 100
                        )
                    )

                    label = (
                        f"acc{tag:02d}"
                    )

                    dose_pa = (
                        object_row[
                            f"{label}_pa_ref_mean_mm"
                        ]
                    )

                    object_row[
                        f"{label}_delta_pa_mm"
                    ] = (
                        dose_pa
                        - base_pa
                    )

                    object_row[
                        f"{label}_pa_ratio"
                    ] = (
                        dose_pa
                        / base_pa
                        if base_pa > 0
                        else np.nan
                    )

                    object_row[
                        f"{label}_actual_delta_acc_si_m_s2"
                    ] = status[
                        f"acc_{tag:02d}_actual_delta_si_m_s2"
                    ]

                    object_row[
                        f"{label}_total_acc_si_m_s2"
                    ] = (
                        status[
                            "original_action_acc_si_m_s2"
                        ]
                        +
                        status[
                            f"acc_{tag:02d}_actual_delta_si_m_s2"
                        ]
                    )

                    object_row[
                        f"{label}_collateral_delta_msj"
                    ] = status[
                        f"acc_{tag:02d}_collateral_delta_msj"
                    ]

                object_rows.append(
                    object_row
                )

                eligible_objects.append(
                    object_row
                )

            status[
                "eligible_objects"
            ] = len(
                eligible_objects
            )

            status_rows.append(
                status
            )

            # ------------------------------------------------
            # One row per clip, avoiding pseudo-replication
            # across multiple rigid objects.
            # ------------------------------------------------

            if eligible_objects:

                clip_row = {
                    "task":
                        TASK,

                    "clip":
                        key,

                    "eligible_objects":
                        len(
                            eligible_objects
                        ),

                    "original_action_acc_si_m_s2":
                        status[
                            "original_action_acc_si_m_s2"
                        ],

                    "gt_future_vs_gt0_pa_mean_mm":
                        float(
                            np.mean(
                                [
                                    r[
                                        "gt_future_vs_gt0_pa_mean_mm"
                                    ]
                                    for r
                                    in eligible_objects
                                ]
                            )
                        ),
                }

                labels_to_save = [
                    "base",
                    "acc05",
                    "acc10",
                    "acc15",
                    "acc20",
                ]

                for label in labels_to_save:

                    clip_row[
                        f"{label}_pa_ref_mean_mm"
                    ] = float(
                        np.mean(
                            [
                                r[
                                    f"{label}_pa_ref_mean_mm"
                                ]
                                for r
                                in eligible_objects
                            ]
                        )
                    )

                base_pa = (
                    clip_row[
                        "base_pa_ref_mean_mm"
                    ]
                )

                for frac in INTERVENTION_FRACS:

                    tag = int(
                        round(
                            frac * 100
                        )
                    )

                    label = (
                        f"acc{tag:02d}"
                    )

                    clip_row[
                        f"{label}_delta_pa_mm"
                    ] = (
                        clip_row[
                            f"{label}_pa_ref_mean_mm"
                        ]
                        - base_pa
                    )

                    clip_row[
                        f"{label}_pa_ratio"
                    ] = (
                        clip_row[
                            f"{label}_pa_ref_mean_mm"
                        ]
                        / base_pa
                        if base_pa > 0
                        else np.nan
                    )

                    clip_row[
                        f"{label}_actual_delta_acc_si_m_s2"
                    ] = status[
                        f"acc_{tag:02d}_actual_delta_si_m_s2"
                    ]

                clip_rows.append(
                    clip_row
                )

            # ------------------------------------------------
            # Checkpoint frequently.
            # ------------------------------------------------

            if (
                inferred <= 5
                or inferred % 25 == 0
            ):
                print(
                    f"[{TASK}] "
                    f"found={len(found)}/"
                    f"{expected_count} | "
                    f"all-match="
                    f"{all_dose_matched_count} | "
                    f"inferred={inferred} | "
                    f"objects="
                    f"{len(eligible_objects)} | "
                    f"clip={key}"
                )

            if (
                inferred % 25 == 0
            ):
                checkpoint(
                    object_rows,
                    clip_rows,
                    status_rows,
                )

            # Smoke = stop after first full 5-variant inference.
            if (
                SMOKE
                and inferred >= 1
            ):
                break

            if (
                len(found)
                == expected_count
            ):
                break

    # ========================================================
    # FINAL AUDITS
    # ========================================================

    checkpoint(
        object_rows,
        clip_rows,
        status_rows,
    )

    if not SMOKE:

        missing = (
            target_keys
            - found
        )

        if missing:
            raise RuntimeError(
                f"Missing targets="
                f"{len(missing)} "
                f"first="
                f"{sorted(missing)[:10]}"
            )

        if len(found) != expected_count:
            raise RuntimeError(
                f"Found mismatch: "
                f"{len(found)} vs "
                f"{expected_count}"
            )

    print()
    print("=" * 100)

    if SMOKE:
        print("SMOKE PASS")
    else:
        print("INTERVENTION COMPLETE")

    print("=" * 100)
    print("TASK =", TASK)
    print("RAW SCANNED =", scanned)
    print(
        "TARGET FOUND =",
        len(found),
    )
    print(
        "ALL-DOSE MATCHED =",
        all_dose_matched_count,
    )
    print(
        "MODEL FORWARDS =",
        inferred,
    )
    print(
        "ELIGIBLE OBJECT ROWS =",
        len(object_rows),
    )
    print(
        "ELIGIBLE CLIP ROWS =",
        len(clip_rows),
    )
    print("OUTPUT =", OUTDIR)

    if object_rows:
        gt_audit = np.asarray(
            [
                r[
                    "gt_future_vs_gt0_pa_mean_mm"
                ]
                for r
                in object_rows
            ],
            dtype=float,
        )

        print(
            "GT RIGIDITY PA median mm =",
            float(
                np.median(
                    gt_audit
                )
            ),
        )

        print(
            "GT RIGIDITY PA p95 mm =",
            float(
                np.quantile(
                    gt_audit,
                    0.95,
                )
            ),
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
