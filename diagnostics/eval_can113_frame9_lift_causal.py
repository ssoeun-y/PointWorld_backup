#!/usr/bin/env python3

import argparse
import copy
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from scipy.spatial import cKDTree

from arguments import parse_args
from evaluation.tester import Tester
from robot_sampler import RobotSampler as TorchRobotSampler
from utils import resolve_robot_urdf

from dataset_components.constants import RELEASE_GRIPPER_ONLY
from dataset_components.collate import custom_collate_fn

from diagnostics.behavior_object_mpjpe_pa_smoke_task0006 import (
    p_mpjpe_single,
)

# Reuse the already-audited 5-task causal code/functions.
import diagnostics.eval_close_anchor_causal_5task as base


# ============================================================
# FIXED CURRENT POPULATION
# ============================================================

TASK_DIRS = {
    "task-0000":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
    "task-0001":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0001",
    "task-0006":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006",
    "task-0008":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0008",
    "task-0010":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0010",
}

EXPECTED_PA = {
    "task-0000": 383,
    "task-0001": 2307,
    "task-0006": 1823,
    "task-0008": 1824,
    "task-0010": 1610,
}

PA_ROOT = Path(
    "diagnostics/results/"
    "object_pa_mpjpe_acc_5task_current10457"
)

OUT_ROOT = Path(
    "diagnostics/results/"
    "closed_grasp_lift_causal_5task_current10457"
)

MODEL_PATH = (
    "pretrained_checkpoints/"
    "large-droid+behavior/model-best.pt"
)

NORM_STATS = "stats/droid_behavior"

BUILD_SEED = 42
MODEL_SEED = 12345
MIN_POINTS = 8


# ============================================================
# Utilities
# ============================================================

def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def squeeze_mask(x):
    x = np.asarray(x)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    return x


def to_device(batch, device):
    return {
        k: (
            v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


def map_action_index(action_idx, seq_T, action_T):
    if seq_T == action_T:
        return int(action_idx)

    if seq_T == action_T - 1:
        if action_idx < 1:
            return None
        return int(action_idx - 1)

    raise RuntimeError(
        f"Cannot align T: action={action_T}, seq={seq_T}"
    )


def release_sample(sample, args, rank):
    """
    Use the exact established release implementation already used
    by our validated evaluators.
    """
    candidates = [
        getattr(base.old, "release_process", None),
        getattr(
            getattr(base.old, "old", None),
            "release_process",
            None,
        ),
    ]

    for fn in candidates:
        if callable(fn):
            return fn(
                sample,
                args,
                "behavior",
                rank,
            )

    raise RuntimeError(
        "Could not find established release_process"
    )


def active_side_indices(sample, side):
    """
    Active gripper surface indices.

    - both -> all sampled robot/gripper points
    - official single-side sample -> all retained points belong to that side
    - true bimanual sample -> split by nearest gripper center
    """

    robot = np.asarray(
        sample["robot_flows"],
        dtype=np.float64,
    )

    if robot.ndim == 4 and robot.shape[0] == 1:
        robot = robot[0]

    N = int(robot.shape[1])

    if side == "both":
        return np.arange(N, dtype=int)

    def get_flag(name, fallback):
        if name not in sample:
            return bool(fallback)

        x = sample[name]

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = np.asarray(x).reshape(-1)

        if len(x) == 0:
            return bool(fallback)

        return bool(x[0])

    has_r = get_flag(
        "__has_right_gripper__",
        "right_gripper_pose" in sample,
    )

    has_l = get_flag(
        "__has_left_gripper__",
        "left_gripper_pose" in sample,
    )

    # Official sample contains only RIGHT gripper.
    if side == "right" and has_r and not has_l:
        return np.arange(N, dtype=int)

    # Official sample contains only LEFT gripper.
    if side == "left" and has_l and not has_r:
        return np.arange(N, dtype=int)

    # Requested active side genuinely absent.
    if side == "right" and not has_r:
        raise RuntimeError(
            "Active RIGHT gripper absent from official sample"
        )

    if side == "left" and not has_l:
        raise RuntimeError(
            "Active LEFT gripper absent from official sample"
        )

    # True bimanual sample: split sampled points geometrically.
    if (
        "right_gripper_pose" not in sample
        or "left_gripper_pose" not in sample
    ):
        raise RuntimeError(
            "Bimanual sample missing active pose metadata"
        )

    rp = np.asarray(
        sample["right_gripper_pose"],
        dtype=np.float64,
    ).reshape(-1, 7)

    lp = np.asarray(
        sample["left_gripper_pose"],
        dtype=np.float64,
    ).reshape(-1, 7)

    centers = np.stack(
        [
            rp[0, :3],
            lp[0, :3],
        ],
        axis=0,
    )

    d = np.linalg.norm(
        robot[0, :, None, :]
        - centers[None, :, :],
        axis=-1,
    )

    assignment = np.argmin(
        d,
        axis=1,
    )

    wanted = 0 if side == "right" else 1

    idx = np.flatnonzero(
        assignment == wanted
    )

    if len(idx) == 0:
        raise RuntimeError(
            f"No robot points assigned to side={side}"
        )

    return idx.astype(int)


def robot_points(sample, indices, action_idx):
    robot = np.asarray(
        sample["robot_flows"],
        dtype=np.float64,
    )

    t = map_action_index(
        action_idx,
        robot.shape[0],
        robot.shape[0],
    )

    pts = robot[t, indices]

    if "robot_exists" in sample:
        rex = np.asarray(
            sample["robot_exists"]
        )

        if rex.ndim == 3 and rex.shape[-1] == 1:
            rex = rex[..., 0]

        if rex.shape[0] == robot.shape[0]:
            keep = (
                np.asarray(
                    rex[t, indices]
                )
                .reshape(-1)
                .astype(bool)
            )

            if len(keep) == len(pts):
                pts = pts[keep]

    return np.asarray(
        pts,
        dtype=np.float64,
    )


def gap_stats_mm(P, G):
    P = np.asarray(P, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    if len(P) == 0 or len(G) == 0:
        return None

    d, _ = cKDTree(G).query(
        P,
        k=1,
    )

    d = d * 1000.0

    return {
        "min": float(np.min(d)),
        "q10": float(np.percentile(d, 10)),
        "median": float(np.median(d)),
    }


def pa_mm(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if len(A) < MIN_POINTS:
        return np.nan

    try:
        value, _ = p_mpjpe_single(
            A,
            B,
        )
        return float(value) * 1000.0
    except Exception:
        return np.nan


def raw_mpjpe_mm(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    return float(
        np.mean(
            np.linalg.norm(
                A - B,
                axis=1,
            )
        )
        * 1000.0
    )


def build_lift_variant(
    built,
    close_side,
    anchor,
    lift_m,
):
    """
    Important causal intervention:

    - action 0..anchor: EXACT Original
    - from anchor+1 onward:
        add only +Z translation to active gripper surface
    - same open/close commands
    - same orientation/normals
    - same scene
    - same underlying original future trajectory, only vertically offset

    This is an action-space diagnostic on PointWorld's actual robot_flows.
    """
    out = copy.deepcopy(built)

    idx = active_side_indices(
        out,
        close_side,
    )

    rf = np.asarray(
        out["robot_flows"]
    ).copy()

    delta = np.array(
        [0.0, 0.0, lift_m],
        dtype=rf.dtype,
    )

    rf[
        anchor + 1:,
        idx,
        :,
    ] += delta

    out["robot_flows"] = rf

    # Keep pose metadata geometrically consistent when present.
    pose_fields = []

    if close_side in ("right", "both"):
        pose_fields.append(
            "right_gripper_pose"
        )

    if close_side in ("left", "both"):
        pose_fields.append(
            "left_gripper_pose"
        )

    for field in pose_fields:
        if field not in out:
            continue

        pose = np.asarray(
            out[field]
        ).copy()

        if (
            pose.ndim >= 2
            and pose.shape[0] == rf.shape[0]
            and pose.shape[-1] >= 3
        ):
            pose[
                anchor + 1:,
                :3,
            ] += delta

            out[field] = pose

    return out, idx


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_DIRS),
    )

    parser.add_argument(
        "--lift-mm",
        type=float,
        default=10.0,
    )

    args_local = parser.parse_args()

    task = args_local.task
    lift_mm = float(
        args_local.lift_mm
    )
    lift_m = lift_mm / 1000.0

    data_dir = TASK_DIRS[task]

    out_dir = (
        OUT_ROOT
        / "can113_frame9_fixed"
        / f"lift_{lift_mm:g}mm"
        / task
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_csv = (
        out_dir
        / "per_clip_lift_causal_metrics.csv"
    )

    status_csv = (
        out_dir
        / "status.csv"
    )

    # --------------------------------------------------------
    # Exact current PA-eligible rigid-object population
    # --------------------------------------------------------

    pa_file = (
        PA_ROOT
        / task
        / "per_object_metrics.csv"
    )

    pa = pd.read_csv(
        pa_file
    )

    required = [
        "clip",
        "object_id",
    ]

    for x in required:
        if x not in pa.columns:
            raise RuntimeError(
                f"Missing PA column: {x}"
            )

    keep = [
        c for c in [
            "clip",
            "object_id",
            "object_full_name",
            "object_short_name",
        ]
        if c in pa.columns
    ]

    pa = (
        pa[keep]
        .drop_duplicates(
            ["clip", "object_id"]
        )
        .copy()
    )

    eligible_by_clip = {}

    for clip, g in pa.groupby(
        "clip"
    ):
        eligible_by_clip[
            str(clip)
        ] = [
            {
                "object_id":
                    int(r.object_id),

                "object_full_name":
                    str(
                        getattr(
                            r,
                            "object_full_name",
                            "",
                        )
                    ),

                "object_short_name":
                    str(
                        getattr(
                            r,
                            "object_short_name",
                            "",
                        )
                    ),
            }
            for r in g.itertuples()
        ]

    if (
        len(eligible_by_clip)
        != EXPECTED_PA[task]
    ):
        raise RuntimeError(
            f"{task}: PA clips="
            f"{len(eligible_by_clip)} "
            f"expected={EXPECTED_PA[task]}"
        )

    target_keys = set(
        eligible_by_clip
    )

    # ========================================================
    # CAN113 FRAME-9 FIXED CAUSAL PROBE
    # Python/action index 9 -> 10
    # ========================================================
    TARGET_CLIP = (
        "task-0001_episode_00010530-360:371"
    )
    TARGET_OID = 3
    FIXED_ANCHOR = 9

    if task != "task-0001":
        raise RuntimeError(
            "This diagnostic is only for task-0001"
        )

    if TARGET_CLIP not in eligible_by_clip:
        raise RuntimeError(
            f"Target clip not PA-eligible: {TARGET_CLIP}"
        )

    target_objects = [
        obj
        for obj in eligible_by_clip[TARGET_CLIP]
        if int(obj["object_id"]) == TARGET_OID
    ]

    if len(target_objects) != 1:
        raise RuntimeError(
            f"Expected exactly one OID={TARGET_OID} "
            f"for {TARGET_CLIP}, got {len(target_objects)}"
        )

    if (
        str(
            target_objects[0].get(
                "object_short_name",
                "",
            )
        )
        != "can_of_soda_113"
    ):
        raise RuntimeError(
            "OID 3 is not can_of_soda_113"
        )

    eligible_by_clip = {
        TARGET_CLIP: target_objects
    }

    target_keys = {
        TARGET_CLIP
    }

    # --------------------------------------------------------
    # Object ID hooks — same as established PA pipeline
    # --------------------------------------------------------

    base.dlmod.build_flow_sample = (
        base.old.omap
        .build_flow_sample_with_object_ids
    )

    base.pmod.grid_sample_transform = (
        base.old.omap
        .grid_sample_with_object_ids
    )

    base.pmod.gather = (
        base.old.omap
        .gather_with_object_ids
    )

    base.old.omap.dlmod.build_flow_sample = (
        base.old.omap
        .build_flow_sample_with_object_ids
    )

    base.old.omap.pmod.grid_sample_transform = (
        base.old.omap
        .grid_sample_with_object_ids
    )

    base.old.omap.pmod.gather = (
        base.old.omap
        .gather_with_object_ids
    )

    # --------------------------------------------------------
    # PointWorld
    # --------------------------------------------------------

    args = parse_args(
        skip_command_line=True
    )

    args.seed = BUILD_SEED
    args.model_path = MODEL_PATH
    args.norm_stats_path = NORM_STATS
    args.domains = ["behavior"]
    args.data_dirs = [data_dir]

    args.batch_size = 1
    args.num_workers = 0

    if hasattr(
        args,
        "eval_num_workers",
    ):
        args.eval_num_workers = 0

    args.eval_num_batches = -1
    args.device = "cuda"
    args.eval_skip_viz = True

    tester = Tester(args)
    tester.model.eval()

    urdf = resolve_robot_urdf(
        "behavior"
    )

    robot_sampler = (
        TorchRobotSampler(
            urdf_path=urdf,
            gripper_only=
                RELEASE_GRIPPER_ONLY,
            device="cpu",
        )
    )

    force_single_arm = False

    if getattr(
        args,
        "_eval_override_active",
        False,
    ):
        domains = getattr(
            args,
            "eval_force_single_arm_domains",
            set(),
        )

        force_single_arm = (
            "behavior"
            in domains
        )

    # --------------------------------------------------------
    # One-pass WDS
    # --------------------------------------------------------

    shards = (
        base.dlmod.gather_shard_paths(
            data_dir,
            "test",
            rank=tester.rank,
        )
    )

    random.Random(
        args.seed + tester.rank
    ).shuffle(shards)

    dataset = wds.WebDataset(
        shards,
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

    rows = []
    statuses = []

    target_seen = 0
    scored = 0

    print("=" * 100)
    print("CLOSED-GRASP LIFT CAUSAL PROBE")
    print("TASK        =", task)
    print("PA TARGETS  =", len(target_keys))
    print("LIFT        =", lift_mm, "mm +Z")
    print("CONTROL     = OriginalRepeat")
    print("ANCHOR      = FIXED frame 9 -> 10")
    print("TARGET      = can_of_soda_113 / OID 3")
    print("=" * 100)

    def flush():
        if rows:
            pd.DataFrame(
                rows
            ).drop_duplicates(
                "clip",
                keep="last",
            ).to_csv(
                out_csv,
                index=False,
            )

        if statuses:
            pd.DataFrame(
                statuses
            ).drop_duplicates(
                "clip",
                keep="last",
            ).to_csv(
                status_csv,
                index=False,
            )

    for raw in dataset:

        key = str(
            raw.get(
                "__key__",
                "",
            )
        )

        if key not in target_keys:
            continue

        target_seen += 1

        try:
            # ====================================================
            # Decode and close event
            # ====================================================

            decoded = base.old.decode_data(
                raw,
                domain="behavior",
            )

            right_open = np.asarray(
                decoded[
                    "right_gripper_open"
                ]
            ).copy()

            left_open = np.asarray(
                decoded[
                    "left_gripper_open"
                ]
            ).copy()

            ev = base.close_events(
                right_open,
                left_open,
            )

            if ev is None:
                statuses.append({
                    "task": task,
                    "clip": key,
                    "status":
                        "no_open_to_close",
                })
                flush()
                continue

            t_close, close_side = ev

            closed_frames = (
                base.closed_window(
                    t_close,
                    close_side,
                    right_open,
                    left_open,
                )
            )

            # Anchor needs a following still-closed frame.
            candidate_frames = (
                [FIXED_ANCHOR]
                if (
                    FIXED_ANCHOR
                    in closed_frames
                    and
                    FIXED_ANCHOR + 1
                    in closed_frames
                )
                else []
            )

            if not candidate_frames:
                statuses.append({
                    "task": task,
                    "clip": key,
                    "status":
                        "no_closed_anchor_next",
                    "t_close": t_close,
                    "close_side":
                        close_side,
                })
                flush()
                continue

            # ====================================================
            # Official deterministic build
            # ====================================================

            built = (
                base.old.omap
                .build_flow_sample_with_object_ids(
                    decoded,
                    domain="behavior",
                    robot_sampler=
                        robot_sampler,
                    max_robot_points=
                        args.max_robot_points,
                    deterministic=True,
                    seed=(
                        args.seed
                        + tester.rank
                    ),
                    force_single_arm=
                        force_single_arm,
                )
            )

            action_T = int(
                built[
                    "robot_flows"
                ].shape[0]
            )

            # Active close command must genuinely exist
            # in the model input. Never invent it.
            required_fields = []

            if close_side in (
                "right",
                "both",
            ):
                required_fields.append(
                    "right_gripper_open"
                )

            if close_side in (
                "left",
                "both",
            ):
                required_fields.append(
                    "left_gripper_open"
                )

            missing = [
                x
                for x in required_fields
                if x not in built
            ]

            if missing:
                statuses.append({
                    "task": task,
                    "clip": key,
                    "status":
                        "active_close_command_missing_in_built",
                    "missing_fields":
                        ",".join(missing),
                })
                flush()
                continue

            # ====================================================
            # Original release first:
            # choose anchor/object using ONLY Original GT geometry.
            #
            # No model prediction is used for anchor selection.
            # ====================================================

            orig_released = (
                release_sample(
                    copy.deepcopy(
                        built
                    ),
                    args,
                    tester.rank,
                )
            )

            if orig_released is None:
                raise RuntimeError(
                    "Original release failed"
                )

            b0 = custom_collate_fn(
                [orig_released],
                args=args,
            )

            labels = squeeze_mask(
                b0[
                    "scene_object_id"
                ][0]
                .detach().cpu()
                .numpy()
            )

            exists = squeeze_mask(
                b0[
                    "scene_exists"
                ][0]
                .detach().cpu()
                .numpy()
            ).astype(bool)

            context = squeeze_mask(
                b0[
                    "scene_context_mask"
                ][0]
                .detach().cpu()
                .numpy()
            ).astype(bool)

            supervised = squeeze_mask(
                b0[
                    "scene_supervised_mask"
                ][0]
                .detach().cpu()
                .numpy()
            ).astype(bool)

            valid = (
                (~context)
                & exists
                & supervised
            )

            gt = (
                b0[
                    "gt_scene_flows"
                ][0]
                .detach().cpu()
                .numpy()
            )

            scene_T = int(
                gt.shape[0]
            )

            active_idx_orig = (
                active_side_indices(
                    orig_released,
                    close_side,
                )
            )

            best = None

            for anchor in candidate_frames:

                s0 = map_action_index(
                    anchor,
                    scene_T,
                    action_T,
                )

                s1 = map_action_index(
                    anchor + 1,
                    scene_T,
                    action_T,
                )

                if (
                    s0 is None
                    or s1 is None
                ):
                    continue

                grip = robot_points(
                    orig_released,
                    active_idx_orig,
                    anchor,
                )

                for obj in (
                    eligible_by_clip[key]
                ):

                    oid = int(
                        obj["object_id"]
                    )

                    # Fixed support must exist at:
                    # initial, anchor, next.
                    m = (
                        (labels[s0] == oid)
                        & valid[s0]
                        & (labels[s1] == oid)
                        & valid[s1]
                        & (labels[0] == oid)
                        & exists[0]
                    )

                    n = int(m.sum())

                    if n < MIN_POINTS:
                        continue

                    gs = gap_stats_mm(
                        gt[s0, m],
                        grip,
                    )

                    if gs is None:
                        continue

                    # Robust grasp-proximity criterion:
                    # q10 first, min only as tie-break.
                    score = (
                        gs["q10"],
                        gs["min"],
                        anchor,
                        oid,
                    )

                    if (
                        best is None
                        or score
                        < best["score"]
                    ):
                        best = {
                            "score": score,
                            "anchor":
                                int(anchor),
                            "next":
                                int(anchor + 1),
                            "scene_anchor":
                                int(s0),
                            "scene_next":
                                int(s1),
                            "oid":
                                oid,
                            "object_full_name":
                                obj[
                                    "object_full_name"
                                ],
                            "object_short_name":
                                obj[
                                    "object_short_name"
                                ],
                            "gt_gap_min":
                                gs["min"],
                            "gt_gap_q10":
                                gs["q10"],
                        }

            if best is None:
                statuses.append({
                    "task": task,
                    "clip": key,
                    "status":
                        "no_anchor_object_candidate",
                    "t_close": t_close,
                    "close_side":
                        close_side,
                })
                flush()
                continue

            anchor = best["anchor"]
            next_frame = best["next"]
            oid = best["oid"]

            # ====================================================
            # Build causal Lift variant
            # ====================================================

            lift_built, active_idx_built = (
                build_lift_variant(
                    built,
                    close_side,
                    anchor,
                    lift_m,
                )
            )

            # ----------------------------------------------------
            # HARD FAIRNESS AUDITS
            # ----------------------------------------------------

            A = np.asarray(
                built["robot_flows"],
                dtype=np.float64,
            )

            L = np.asarray(
                lift_built[
                    "robot_flows"
                ],
                dtype=np.float64,
            )

            prefix_err = float(
                np.max(
                    np.abs(
                        L[
                            :anchor + 1
                        ]
                        -
                        A[
                            :anchor + 1
                        ]
                    )
                )
            )

            if prefix_err > 1e-8:
                raise RuntimeError(
                    f"Prefix changed "
                    f"before/at anchor: "
                    f"{prefix_err}"
                )

            active_shift = (
                np.linalg.norm(
                    L[
                        next_frame,
                        active_idx_built,
                    ]
                    -
                    A[
                        next_frame,
                        active_idx_built,
                    ],
                    axis=1,
                )
                * 1000.0
            )

            shift_med = float(
                np.median(
                    active_shift
                )
            )

            if not np.isclose(
                shift_med,
                lift_mm,
                atol=1e-4,
                rtol=0,
            ):
                raise RuntimeError(
                    f"Lift dose audit failed: "
                    f"{shift_med} vs {lift_mm}"
                )

            # Discrete command identity.
            for field in [
                "right_gripper_open",
                "left_gripper_open",
            ]:
                if (
                    field in built
                    and
                    field in lift_built
                ):
                    if not np.array_equal(
                        np.asarray(
                            built[field]
                        ),
                        np.asarray(
                            lift_built[field]
                        ),
                    ):
                        raise RuntimeError(
                            f"{field} changed"
                        )

            # ====================================================
            # Release three conditions
            #
            # OriginalRepeat = stochastic floor control.
            # ====================================================

            released = {
                "Original":
                    orig_released,

                "OriginalRepeat":
                    release_sample(
                        copy.deepcopy(
                            built
                        ),
                        args,
                        tester.rank,
                    ),

                "Lift":
                    release_sample(
                        copy.deepcopy(
                            lift_built
                        ),
                        args,
                        tester.rank,
                    ),
            }

            if any(
                x is None
                for x in released.values()
            ):
                raise RuntimeError(
                    "Variant release failed"
                )

            # ====================================================
            # Scene/support must be EXACT
            # ====================================================

            audit = custom_collate_fn(
                [
                    released[
                        "Original"
                    ],
                    released[
                        "OriginalRepeat"
                    ],
                    released[
                        "Lift"
                    ],
                ],
                args=args,
            )

            for field in [
                "scene_flows",
                "gt_scene_flows",
                "scene_exists",
                "scene_context_mask",
                "scene_supervised_mask",
                "scene_object_id",
            ]:
                x = audit[field]

                if isinstance(
                    x,
                    torch.Tensor,
                ):
                    if (
                        not torch.equal(
                            x[0],
                            x[1],
                        )
                        or
                        not torch.equal(
                            x[0],
                            x[2],
                        )
                    ):
                        raise RuntimeError(
                            f"Scene support differs: "
                            f"{field}"
                        )

            # ====================================================
            # Same-seed separate forward
            # ====================================================

            pred = {}
            batches = {}

            for label in [
                "Original",
                "OriginalRepeat",
                "Lift",
            ]:

                batch_cpu = (
                    custom_collate_fn(
                        [
                            released[
                                label
                            ]
                        ],
                        args=args,
                    )
                )

                batches[
                    label
                ] = batch_cpu

                reset_seed(
                    MODEL_SEED
                )

                batch_gpu = to_device(
                    batch_cpu,
                    tester.device,
                )

                with torch.inference_mode():

                    outputs = (
                        tester.model(
                            batch_gpu,
                            training=False,
                        )
                    )

                pred[label] = (
                    outputs[
                        "scene_flows"
                    ][0]
                    .detach().cpu()
                    .numpy()
                )

            pred_T = int(
                pred[
                    "Original"
                ].shape[0]
            )

            pa = map_action_index(
                anchor,
                pred_T,
                action_T,
            )

            pn = map_action_index(
                next_frame,
                pred_T,
                action_T,
            )

            if (
                pa is None
                or pn is None
            ):
                raise RuntimeError(
                    "Prediction alignment failed"
                )

            # Rebuild fixed support on final official batch.
            b_ref = batches[
                "Original"
            ]

            labels2 = squeeze_mask(
                b_ref[
                    "scene_object_id"
                ][0]
                .detach().cpu()
                .numpy()
            )

            exists2 = squeeze_mask(
                b_ref[
                    "scene_exists"
                ][0]
                .detach().cpu()
                .numpy()
            ).astype(bool)

            context2 = squeeze_mask(
                b_ref[
                    "scene_context_mask"
                ][0]
                .detach().cpu()
                .numpy()
            ).astype(bool)

            supervised2 = squeeze_mask(
                b_ref[
                    "scene_supervised_mask"
                ][0]
                .detach().cpu()
                .numpy()
            ).astype(bool)

            valid2 = (
                (~context2)
                & exists2
                & supervised2
            )

            gt2 = (
                b_ref[
                    "gt_scene_flows"
                ][0]
                .detach().cpu()
                .numpy()
            )

            m = (
                (labels2[pa] == oid)
                & valid2[pa]
                & (labels2[pn] == oid)
                & valid2[pn]
                & (labels2[0] == oid)
                & exists2[0]
            )

            n_points = int(
                m.sum()
            )

            if n_points < MIN_POINTS:
                raise RuntimeError(
                    "Final support insufficient"
                )

            gt0 = gt2[
                0,
                m,
            ]

            # ====================================================
            # Per-condition measurements
            # ====================================================

            cond = {}

            for label in [
                "Original",
                "OriginalRepeat",
                "Lift",
            ]:

                P0 = np.asarray(
                    pred[label][
                        pa,
                        m,
                    ],
                    dtype=np.float64,
                )

                P1 = np.asarray(
                    pred[label][
                        pn,
                        m,
                    ],
                    dtype=np.float64,
                )

                idx = (
                    active_side_indices(
                        released[label],
                        close_side,
                    )
                )

                G0 = robot_points(
                    released[label],
                    idx,
                    anchor,
                )

                G1 = robot_points(
                    released[label],
                    idx,
                    next_frame,
                )

                gap0 = gap_stats_mm(
                    P0,
                    G0,
                )

                gap1 = gap_stats_mm(
                    P1,
                    G1,
                )

                c0 = np.mean(
                    P0,
                    axis=0,
                )

                c1 = np.mean(
                    P1,
                    axis=0,
                )

                cond[label] = {
                    "P0": P0,
                    "P1": P1,
                    "c0": c0,
                    "c1": c1,

                    "gap_anchor_q10":
                        gap0["q10"],

                    "gap_next_q10":
                        gap1["q10"],

                    "gap_change_q10":
                        gap1["q10"]
                        -
                        gap0["q10"],

                    "pa_anchor_gt0":
                        pa_mm(
                            P0,
                            gt0,
                        ),

                    "pa_next_gt0":
                        pa_mm(
                            P1,
                            gt0,
                        ),

                    # Shape deformation over one predicted step.
                    "temporal_pa":
                        pa_mm(
                            P1,
                            P0,
                        ),
                }

            O = cond["Original"]
            R = cond[
                "OriginalRepeat"
            ]
            Lc = cond["Lift"]

            # ====================================================
            # Direct causal response
            #
            # Lift - Original isolates added +Z action.
            # ====================================================

            object_response_vec = (
                Lc["c1"]
                -
                O["c1"]
            )

            repeat_response_vec = (
                R["c1"]
                -
                O["c1"]
            )

            object_response_z_mm = float(
                object_response_vec[2]
                * 1000.0
            )

            repeat_response_z_mm = float(
                repeat_response_vec[2]
                * 1000.0
            )

            response_ratio = (
                object_response_z_mm
                / lift_mm
            )

            # Did changing FUTURE action already alter
            # predicted object at the unchanged anchor?
            anchor_raw_lift = (
                raw_mpjpe_mm(
                    Lc["P0"],
                    O["P0"],
                )
            )

            anchor_raw_repeat = (
                raw_mpjpe_mm(
                    R["P0"],
                    O["P0"],
                )
            )

            next_raw_lift = (
                raw_mpjpe_mm(
                    Lc["P1"],
                    O["P1"],
                )
            )

            next_raw_repeat = (
                raw_mpjpe_mm(
                    R["P1"],
                    O["P1"],
                )
            )

            # Shape-only difference between counterfactual
            # and Original at next frame.
            next_shape_pa_lift = (
                pa_mm(
                    Lc["P1"],
                    O["P1"],
                )
            )

            next_shape_pa_repeat = (
                pa_mm(
                    R["P1"],
                    O["P1"],
                )
            )

            row = {
                "task": task,
                "clip": key,

                "object_id": oid,
                "object_full_name":
                    best[
                        "object_full_name"
                    ],
                "object_short_name":
                    best[
                        "object_short_name"
                    ],

                "t_close": t_close,
                "close_side":
                    close_side,

                "anchor_frame":
                    anchor,
                "next_frame":
                    next_frame,
                "anchor_minus_close":
                    anchor
                    - t_close,

                "lift_mm":
                    lift_mm,

                "object_points":
                    n_points,

                # Original GT acquisition geometry.
                "gt_anchor_min_gap_mm":
                    best[
                        "gt_gap_min"
                    ],
                "gt_anchor_q10_gap_mm":
                    best[
                        "gt_gap_q10"
                    ],

                # Hard input audit.
                "prefix_robot_maxerr":
                    prefix_err,
                "actual_active_lift_median_mm":
                    shift_med,

                # Most direct coupling response.
                "object_response_z_mm":
                    object_response_z_mm,
                "object_response_ratio":
                    response_ratio,

                "repeat_response_z_mm":
                    repeat_response_z_mm,

                "object_response_norm_mm":
                    float(
                        np.linalg.norm(
                            object_response_vec
                        )
                        * 1000.0
                    ),

                "repeat_response_norm_mm":
                    float(
                        np.linalg.norm(
                            repeat_response_vec
                        )
                        * 1000.0
                    ),

                # Future-action effect at unchanged anchor.
                "anchor_lift_vs_original_raw_mpjpe_mm":
                    anchor_raw_lift,

                "anchor_repeat_vs_original_raw_mpjpe_mm":
                    anchor_raw_repeat,

                # Total next-frame prediction response.
                "next_lift_vs_original_raw_mpjpe_mm":
                    next_raw_lift,

                "next_repeat_vs_original_raw_mpjpe_mm":
                    next_raw_repeat,

                # Shape-only counterfactual response.
                "next_lift_vs_original_pa_mpjpe_mm":
                    next_shape_pa_lift,

                "next_repeat_vs_original_pa_mpjpe_mm":
                    next_shape_pa_repeat,

                # Gap.
                "original_anchor_q10_gap_mm":
                    O[
                        "gap_anchor_q10"
                    ],

                "original_next_q10_gap_mm":
                    O[
                        "gap_next_q10"
                    ],

                "lift_anchor_q10_gap_mm":
                    Lc[
                        "gap_anchor_q10"
                    ],

                "lift_next_q10_gap_mm":
                    Lc[
                        "gap_next_q10"
                    ],

                "lift_minus_original_next_q10_gap_mm":
                    Lc[
                        "gap_next_q10"
                    ]
                    -
                    O[
                        "gap_next_q10"
                    ],

                "repeat_minus_original_next_q10_gap_mm":
                    R[
                        "gap_next_q10"
                    ]
                    -
                    O[
                        "gap_next_q10"
                    ],

                # PA-MPJPE vs initial rigid GT shape.
                "original_pa_anchor_gt0_mm":
                    O[
                        "pa_anchor_gt0"
                    ],

                "original_pa_next_gt0_mm":
                    O[
                        "pa_next_gt0"
                    ],

                "lift_pa_anchor_gt0_mm":
                    Lc[
                        "pa_anchor_gt0"
                    ],

                "lift_pa_next_gt0_mm":
                    Lc[
                        "pa_next_gt0"
                    ],

                "lift_minus_original_pa_next_gt0_mm":
                    Lc[
                        "pa_next_gt0"
                    ]
                    -
                    O[
                        "pa_next_gt0"
                    ],

                "repeat_minus_original_pa_next_gt0_mm":
                    R[
                        "pa_next_gt0"
                    ]
                    -
                    O[
                        "pa_next_gt0"
                    ],

                # Shape degradation between adjacent frames.
                "original_temporal_pa_mm":
                    O[
                        "temporal_pa"
                    ],

                "lift_temporal_pa_mm":
                    Lc[
                        "temporal_pa"
                    ],

                "repeat_temporal_pa_mm":
                    R[
                        "temporal_pa"
                    ],
            }

            rows.append(row)

            statuses.append({
                "task": task,
                "clip": key,
                "status":
                    "lift_scored",
                "t_close":
                    t_close,
                "close_side":
                    close_side,
                "anchor_frame":
                    anchor,
                "object_id":
                    oid,
            })

            scored += 1

            print(
                f"[{task}] "
                f"N={scored} "
                f"clip={key} "
                f"close={t_close}/{close_side} "
                f"anchor={anchor}->{next_frame} "
                f"GTgap={best['gt_gap_min']:.2f}mm "
                f"resp={object_response_z_mm:+.2f}/"
                f"{lift_mm:.1f}mm "
                f"ratio={response_ratio:+.3f} "
                f"gapΔ="
                f"{row['lift_minus_original_next_q10_gap_mm']:+.2f}mm "
                f"PAΔ="
                f"{row['lift_minus_original_pa_next_gt0_mm']:+.3f}mm",
                flush=True,
            )

            flush()

        except Exception as e:

            statuses.append({
                "task": task,
                "clip": key,
                "status":
                    "unexpected_exception",
                "error_type":
                    type(e).__name__,
                "error":
                    str(e),
            })

            print(
                f"[ERROR] "
                f"task={task} "
                f"target={target_seen}/"
                f"{len(target_keys)} "
                f"clip={key} "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

            traceback.print_exc()
            flush()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

    flush()

    print()
    print("=" * 100)
    print("TASK COMPLETE =", task)
    print("PA TARGETS    =", len(target_keys))
    print("SCORED        =", scored)

    if statuses:
        s = pd.DataFrame(
            statuses
        )

        print()
        print(
            s["status"]
            .value_counts()
            .to_string()
        )

    print("RESULT =", out_csv)
    print("=" * 100)


if __name__ == "__main__":
    main()
