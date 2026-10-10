#!/usr/bin/env python3

import copy
import random
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
from dataset_components.cameras import sample_cameras
from dataset_components.robot import canonicalize_gripper_keys_and_flags
from dataset_components.pipeline import apply_release_pipeline_to_sample
from dataset_components.collate import custom_collate_fn

from diagnostics.behavior_object_mpjpe_pa_smoke_task0006 import (
    p_mpjpe_single,
)

import diagnostics.behavior_object_mapping_smoke_task0006 as omap

# Reuse the EXACT established temporal retiming implementation.
from diagnostics import eval_acc_only_boundary_pa_final as final_pa

apply_phase = final_pa.old.src.apply_phase
decode_data = final_pa.old.decode_data


# ============================================================
# FIXED CAUSAL SMOKE
# ============================================================

TARGET = "task-0001_episode_00010530-360:371"
TASK = "task-0001"
OID = 3
ACTIVE_SIDE = "left"

DATA_ROOT = Path(
    "/home/soenchun/cse/PointWorld_workspace/"
    "datasets/behavior/wds_task0001"
)

# We already verified this target is in this exact test shard.
SHARD = (
    DATA_ROOT
    / "test"
    / "test-rank00-000000.tar"
)

FREE_NPZ = Path(
    "diagnostics/results/"
    "close_anchor_matched_dose_can_smoke/"
    "free_matched.npz"
)

ANCHOR_NPZ = Path(
    "diagnostics/results/"
    "close_anchor_matched_dose_can_smoke/"
    "close_anchor_matched.npz"
)

OUTDIR = Path(
    "diagnostics/results/"
    "close_anchor_causal_eval_can_smoke"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    "pretrained_checkpoints/"
    "large-droid+behavior/model-best.pt"
)

NORM_STATS = "stats/droid_behavior"

MODEL_SEED = 12345
BUILD_SEED = 42


# ============================================================
# Seed
# ============================================================

def reset_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Official single-sample release pipeline
# ============================================================

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


# ============================================================
# Array utilities
# ============================================================

def as_np(x):

    if isinstance(x, torch.Tensor):
        return (
            x.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(x)


def exact_equal(a, b):

    aa = as_np(a)
    bb = as_np(b)

    return (
        aa.shape == bb.shape
        and np.array_equal(
            aa,
            bb,
        )
    )


# ============================================================
# Action index alignment
#
# Raw action:
#   t0 ... t10  => 11
#
# Prediction may be:
#   t1 ... t10  => 10
#
# Hence if sequence has action_T-1 frames:
#   action index k -> sequence index k-1
# ============================================================

def map_action_index(
    action_idx,
    seq_T,
    action_T,
):

    if seq_T == action_T:
        return int(action_idx)

    if seq_T == action_T - 1:

        if action_idx < 1:
            return None

        return int(
            action_idx - 1
        )

    raise RuntimeError(
        "Cannot align action and sequence "
        f"lengths: action_T={action_T}, "
        f"seq_T={seq_T}"
    )


# ============================================================
# Gripper surface side split
#
# Same geometry-based split already used in our grasp code:
# assign each sampled gripper-surface point to the nearest
# right/left gripper center at the first frame.
# ============================================================

def active_gripper_indices(
    sample,
    side,
):

    robot = np.asarray(
        sample["robot_flows"],
        dtype=np.float64,
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

    dist = np.linalg.norm(
        robot[0, :, None, :]
        -
        centers[None, :, :],
        axis=-1,
    )

    assignment = np.argmin(
        dist,
        axis=1,
    )

    wanted = (
        0
        if side == "right"
        else 1
    )

    idx = np.flatnonzero(
        assignment == wanted
    )

    if len(idx) == 0:
        raise RuntimeError(
            f"No robot points assigned to {side}"
        )

    return idx


# ============================================================
# Object ↔ active gripper surface distance
# ============================================================

def gap_stats_mm(
    object_points,
    robot_points,
):

    if (
        len(object_points) == 0
        or
        len(robot_points) == 0
    ):
        return {
            "min": np.nan,
            "q10": np.nan,
            "median": np.nan,
        }

    tree = cKDTree(
        np.asarray(
            robot_points,
            dtype=np.float64,
        )
    )

    d = (
        tree.query(
            np.asarray(
                object_points,
                dtype=np.float64,
            ),
            k=1,
        )[0]
        * 1000.0
    )

    return {
        "min":
            float(
                np.min(d)
            ),

        "q10":
            float(
                np.percentile(
                    d,
                    10,
                )
            ),

        "median":
            float(
                np.median(d)
            ),
    }


# ============================================================
# Load matched phases
# ============================================================

for p in [
    FREE_NPZ,
    ANCHOR_NPZ,
]:

    if not p.exists():
        raise RuntimeError(
            f"Missing NPZ: {p}"
        )

zf = np.load(
    FREE_NPZ,
    allow_pickle=True,
)

za = np.load(
    ANCHOR_NPZ,
    allow_pickle=True,
)

for z, name in [
    (zf, "FREE"),
    (za, "ANCHOR"),
]:

    if str(z["clip"].item()) != TARGET:
        raise RuntimeError(
            f"{name}: target mismatch"
        )

    if str(z["task"].item()) != TASK:
        raise RuntimeError(
            f"{name}: task mismatch"
        )

t_close = int(
    za["t_close"]
)

close_side = str(
    za["close_side"].item()
)

if close_side != ACTIVE_SIDE:
    raise RuntimeError(
        f"Expected {ACTIVE_SIDE}, "
        f"got {close_side}"
    )

if int(zf["t_close"]) != t_close:
    raise RuntimeError(
        "FREE/ANCHOR t_close mismatch"
    )

free_phase_raw = np.asarray(
    zf["phase"],
    dtype=np.float64,
)

anchor_phase_raw = np.asarray(
    za["phase"],
    dtype=np.float64,
)

free_acc_si = (
    float(zf["achieved_acc"])
    * 25.0
)

anchor_acc_si = (
    float(za["achieved_acc"])
    * 25.0
)

original_acc_si = (
    float(za["original_acc"])
    * 25.0
)

if abs(
    free_acc_si
    - anchor_acc_si
) > 1e-6:

    raise RuntimeError(
        "Matched RMS Acc audit failed: "
        f"{free_acc_si} vs "
        f"{anchor_acc_si}"
    )


# ============================================================
# Runtime-only object ID hooks
# ============================================================

omap.dlmod.build_flow_sample = (
    omap.build_flow_sample_with_object_ids
)

omap.pmod.grid_sample_transform = (
    omap.grid_sample_with_object_ids
)

omap.pmod.gather = (
    omap.gather_with_object_ids
)


# ============================================================
# PointWorld
# ============================================================

args = parse_args(
    skip_command_line=True
)

args.model_path = MODEL_PATH
args.norm_stats_path = NORM_STATS

args.domains = [
    "behavior"
]

args.data_dirs = [
    str(DATA_ROOT)
]

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
args.seed = BUILD_SEED

if hasattr(
    args,
    "ptv3_size",
):
    args.ptv3_size = "large"

tester = Tester(args)
tester.model.eval()

domain = "behavior"

urdf_path = resolve_robot_urdf(
    domain
)

robot_sampler = TorchRobotSampler(
    urdf_path=urdf_path,
    gripper_only=RELEASE_GRIPPER_ONLY,
    device="cpu",
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


# ============================================================
# EXACT shard / early key filter
# ============================================================

if not SHARD.exists():

    raise RuntimeError(
        f"Missing shard: {SHARD}"
    )

dataset = wds.WebDataset(
    str(SHARD),
    shardshuffle=False,
    resampled=False,
)

raw = None

for sample in dataset:

    key = str(
        sample.get(
            "__key__",
            "",
        )
    )

    if key != TARGET:
        continue

    raw = sample
    break

if raw is None:

    raise RuntimeError(
        f"Target not found: {TARGET}"
    )

print("=" * 88)
print("FOUND EXACT TARGET =", TARGET)
print("SHARD              =", SHARD.name)
print("=" * 88)


# ============================================================
# Decode ONCE / Build ONCE
# ============================================================

decoded = decode_data(
    raw,
    domain=domain,
)

if str(
    decoded.get(
        "__key__",
        "",
    )
) != TARGET:

    raise RuntimeError(
        "Decoded key mismatch"
    )

raw_left_open = np.asarray(
    decoded[
        "left_gripper_open"
    ]
).reshape(-1).astype(int)

raw_right_open = np.asarray(
    decoded[
        "right_gripper_open"
    ]
).reshape(-1).astype(int)

built = (
    omap.build_flow_sample_with_object_ids(
        decoded,
        domain=domain,
        robot_sampler=robot_sampler,
        max_robot_points=args.max_robot_points,
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

if len(free_phase_raw) != action_T:
    raise RuntimeError(
        "FREE phase length mismatch | "
        f"{len(free_phase_raw)} vs "
        f"{action_T}"
    )

if len(anchor_phase_raw) != action_T:
    raise RuntimeError(
        "ANCHOR phase length mismatch"
    )

identity_phase = np.linspace(
    0.0,
    1.0,
    action_T,
    dtype=np.float64,
)

free_phase = (
    free_phase_raw
    / float(
        action_T - 1
    )
)

anchor_phase = (
    anchor_phase_raw
    / float(
        action_T - 1
    )
)


# ============================================================
# Build three action variants
# ============================================================

unreleased = {
    "Original":
        apply_phase(
            copy.deepcopy(
                built
            ),
            identity_phase,
        ),

    "Free":
        apply_phase(
            copy.deepcopy(
                built
            ),
            free_phase,
        ),

    "Anchor":
        apply_phase(
            copy.deepcopy(
                built
            ),
            anchor_phase,
        ),
}


# ============================================================
# CRITICAL ACTION-INPUT AUDIT
#
# Anchor must be EXACTLY Original at t_close.
#
# Check actual PointWorld action-side fields,
# not only raw XYZ pose.
# ============================================================

audit_fields = [
    "robot_flows",
    "robot_normals",
    "right_gripper_open",
    "left_gripper_open",
    "gripper_positions",
]

print()
print("=" * 88)
print("ACTION INPUT AUDIT @ t_close")
print("=" * 88)

for field in audit_fields:

    if (
        field not in unreleased["Original"]
        or
        field not in unreleased["Anchor"]
    ):
        continue

    a = np.asarray(
        unreleased[
            "Original"
        ][field]
    )

    b = np.asarray(
        unreleased[
            "Anchor"
        ][field]
    )

    if (
        a.ndim < 1
        or
        a.shape[0]
        != action_T
        or
        b.shape != a.shape
    ):
        continue

    close_err = float(
        np.max(
            np.abs(
                np.asarray(
                    b[t_close],
                    dtype=np.float64,
                )
                -
                np.asarray(
                    a[t_close],
                    dtype=np.float64,
                )
            )
        )
    )

    print(
        f"{field:24s} "
        f"ANCHOR close max|Δ| = "
        f"{close_err:.12e}"
    )

    if close_err > 1e-7:
        raise RuntimeError(
            f"Anchor field differs "
            f"at t_close: {field}"
        )

# Free close RobotFlow shift for reference.
free_robot_close = (
    np.linalg.norm(
        np.asarray(
            unreleased["Free"][
                "robot_flows"
            ][t_close],
            dtype=np.float64,
        )
        -
        np.asarray(
            unreleased["Original"][
                "robot_flows"
            ][t_close],
            dtype=np.float64,
        ),
        axis=-1,
    )
    * 1000.0
)

print()
print(
    "FREE t_close robot-surface "
    "mean shift mm =",
    f"{np.mean(free_robot_close):.6f}",
)

print(
    "FREE t_close robot-surface "
    "max shift mm  =",
    f"{np.max(free_robot_close):.6f}",
)

# Hard start/end.
for label in [
    "Free",
    "Anchor",
]:

    for ti, name in [
        (0, "START"),
        (-1, "END"),
    ]:

        if not np.allclose(
            unreleased[
                label
            ][
                "robot_flows"
            ][ti],
            unreleased[
                "Original"
            ][
                "robot_flows"
            ][ti],
            atol=1e-7,
            rtol=0,
        ):

            raise RuntimeError(
                f"{label}: "
                f"{name} changed"
            )


# ============================================================
# Release each sample with exact same deterministic protocol
# ============================================================

released = {}

for label in [
    "Original",
    "Free",
    "Anchor",
]:

    released[label] = final_pa.old.release_process(
        copy.deepcopy(
            unreleased[label]
        ),
        args,
        domain,
        tester.rank,
    )

    if released[label] is None:
        raise RuntimeError(
            f"{label}: release failed"
        )


# ============================================================
# Scene support must be EXACTLY identical
#
# IMPORTANT:
# scene_exists / masks are audited AFTER custom_collate_fn,
# exactly like the validated final PA evaluator.
# ============================================================

audit_batch = final_pa.old.custom_collate_fn(
    [
        released["Original"],
        released["Free"],
        released["Anchor"],
    ],
    args=args,
)

scene_audit_fields = [
    "scene_flows",
    "gt_scene_flows",
    "scene_exists",
    "scene_context_mask",
    "scene_supervised_mask",
    "scene_moved_mask",
    "scene_object_id",
]

for field in scene_audit_fields:

    if field not in audit_batch:
        raise RuntimeError(
            f"Collated batch missing field: {field} | "
            f"keys={list(audit_batch.keys())}"
        )

    x = audit_batch[field]

    if isinstance(x, torch.Tensor):

        if x.shape[0] != 3:
            raise RuntimeError(
                f"{field}: expected 3 variants, "
                f"got shape={tuple(x.shape)}"
            )

        if not torch.equal(x[0], x[1]):
            raise RuntimeError(
                f"{field}: Original != Free"
            )

        if not torch.equal(x[0], x[2]):
            raise RuntimeError(
                f"{field}: Original != Anchor"
            )

    else:

        a = np.asarray(x)

        if a.shape[0] != 3:
            raise RuntimeError(
                f"{field}: expected 3 variants, "
                f"got shape={a.shape}"
            )

        if not np.array_equal(a[0], a[1]):
            raise RuntimeError(
                f"{field}: Original != Free"
            )

        if not np.array_equal(a[0], a[2]):
            raise RuntimeError(
                f"{field}: Original != Anchor"
            )

print()
print("SCENE SUPPORT AUDIT = EXACT AFTER COLLATE")


# ============================================================
# Active left gripper surface indices
# ============================================================

side_idx = {}

for label in [
    "Original",
    "Free",
    "Anchor",
]:

    side_idx[label] = (
        active_gripper_indices(
            released[label],
            ACTIVE_SIDE,
        )
    )

    print(
        f"{label:8s} "
        f"{ACTIVE_SIDE} gripper "
        f"surface points = "
        f"{len(side_idx[label])}"
    )


# ============================================================
# Same-seed separate forward for EACH action.
#
# This is deliberate:
# each condition sees the SAME stochastic seed.
# ============================================================

pred = {}
batches = {}

for label in [
    "Original",
    "Free",
    "Anchor",
]:

    batch_cpu = final_pa.old.custom_collate_fn(
        [
            released[label]
        ],
        args=args,
    )

    batches[label] = batch_cpu

    reset_seed(
        MODEL_SEED
    )

    batch_gpu = to_device(
        batch_cpu,
        tester.device,
    )

    with torch.inference_mode():

        outputs = tester.model(
            batch_gpu,
            training=False,
        )

    if "scene_flows" not in outputs:
        raise RuntimeError(
            f"{label}: no scene_flows"
        )

    pred[label] = (
        outputs[
            "scene_flows"
        ][0]
        .detach()
        .cpu()
        .numpy()
    )

    print(
        f"FORWARD {label:8s} "
        f"shape={pred[label].shape} "
        f"seed={MODEL_SEED}"
    )


# ============================================================
# Reference metadata
# ============================================================

b0 = batches[
    "Original"
]

labels = (
    b0[
        "scene_object_id"
    ][0]
    .detach()
    .cpu()
    .numpy()
)

exists = (
    b0[
        "scene_exists"
    ][0]
    .detach()
    .cpu()
    .numpy()
    .astype(bool)
)

context = (
    b0[
        "scene_context_mask"
    ][0]
    .detach()
    .cpu()
    .numpy()
    .astype(bool)
)

supervised = (
    b0[
        "scene_supervised_mask"
    ][0]
    .detach()
    .cpu()
    .numpy()
    .astype(bool)
)

# Normalize official mask shapes:
# (T,N,1) -> (T,N)
if context.ndim == 3 and context.shape[-1] == 1:
    context = context[..., 0]

if exists.ndim == 3 and exists.shape[-1] == 1:
    exists = exists[..., 0]

if supervised.ndim == 3 and supervised.shape[-1] == 1:
    supervised = supervised[..., 0]

if labels.ndim == 3 and labels.shape[-1] == 1:
    labels = labels[..., 0]

valid = (
    (~context)
    & exists
    & supervised
)

gt = (
    b0[
        "gt_scene_flows"
    ][0]
    .detach()
    .cpu()
    .numpy()
)

pred_T = int(
    pred[
        "Original"
    ].shape[0]
)


# ============================================================
# Release time
# ============================================================

active_open = (
    raw_left_open
    if ACTIVE_SIDE == "left"
    else raw_right_open
)

t_release = None

for tt in range(
    t_close + 1,
    action_T,
):

    if (
        active_open[
            tt - 1
        ] == 0
        and
        active_open[
            tt
        ] == 1
    ):

        t_release = tt
        break

if t_release is None:

    closed_action_frames = list(
        range(
            t_close,
            action_T,
        )
    )

else:

    closed_action_frames = list(
        range(
            t_close,
            t_release,
        )
    )

print()
print("=" * 88)
print("CAUSAL WINDOW")
print("=" * 88)
print(
    "t_close raw index =",
    t_close,
    "| human frame =",
    t_close + 1,
)
print(
    "t_release =",
    t_release,
)
print(
    "closed action indices =",
    closed_action_frames,
)


# ============================================================
# Per-frame contact + PA
# ============================================================

frame_rows = []

for action_idx in closed_action_frames:

    scene_idx = map_action_index(
        action_idx,
        pred_T,
        action_T,
    )

    if scene_idx is None:
        continue

    if (
        scene_idx < 0
        or
        scene_idx >= pred_T
    ):
        continue

    m = (
        (labels[scene_idx] == OID)
        &
        valid[scene_idx]
        &
        (labels[0] == OID)
        &
        exists[0]
    )

    n = int(
        m.sum()
    )

    if n < 8:
        continue

    # Original-data GT gap is only a reference.
    # It is NOT counterfactual GT for Free/Anchor.
    robot_orig = np.asarray(
        released[
            "Original"
        ][
            "robot_flows"
        ],
        dtype=np.float64,
    )

    robot_idx_orig = (
        map_action_index(
            action_idx,
            robot_orig.shape[0],
            action_T,
        )
    )

    rg = robot_orig[
        robot_idx_orig,
        side_idx["Original"],
    ]

    if "robot_exists" in released[
        "Original"
    ]:

        rex = np.asarray(
            released[
                "Original"
            ][
                "robot_exists"
            ]
        )

        rix = map_action_index(
            action_idx,
            rex.shape[0],
            action_T,
        )

        rr = (
            np.asarray(
                rex[
                    rix,
                    side_idx[
                        "Original"
                    ],
                ]
            )
            .reshape(-1)
            .astype(bool)
        )

        if len(rr) == len(rg):
            rg = rg[rr]

    gt_gap = gap_stats_mm(
        gt[
            scene_idx,
            m,
        ],
        rg,
    )

    for label in [
        "Original",
        "Free",
        "Anchor",
    ]:

        robot = np.asarray(
            released[
                label
            ][
                "robot_flows"
            ],
            dtype=np.float64,
        )

        robot_idx = (
            map_action_index(
                action_idx,
                robot.shape[0],
                action_T,
            )
        )

        grip = robot[
            robot_idx,
            side_idx[label],
        ]

        if "robot_exists" in released[
            label
        ]:

            rex = np.asarray(
                released[
                    label
                ][
                    "robot_exists"
                ]
            )

            rix = map_action_index(
                action_idx,
                rex.shape[0],
                action_T,
            )

            rr = (
                np.asarray(
                    rex[
                        rix,
                        side_idx[label],
                    ]
                )
                .reshape(-1)
                .astype(bool)
            )

            if len(rr) == len(grip):
                grip = grip[rr]

        P = pred[
            label
        ][
            scene_idx,
            m,
        ]

        gap = gap_stats_mm(
            P,
            grip,
        )

        gt0 = gt[
            0,
            m,
        ]

        try:

            pa, _ = (
                p_mpjpe_single(
                    P,
                    gt0,
                )
            )

            pa_mm = (
                float(pa)
                * 1000.0
            )

        except Exception:

            pa_mm = np.nan

        frame_rows.append(
            {
                "variant":
                    label,

                "action_index":
                    action_idx,

                "human_action_frame":
                    action_idx + 1,

                "scene_index":
                    scene_idx,

                "object_id":
                    OID,

                "n_points":
                    n,

                "gap_min_mm":
                    gap["min"],

                "gap_q10_mm":
                    gap["q10"],

                "gap_median_mm":
                    gap["median"],

                "original_data_gt_gap_q10_mm":
                    gt_gap["q10"],

                "pa_gt0_mm":
                    pa_mm,
            }
        )


frame_df = pd.DataFrame(
    frame_rows
)

if frame_df.empty:
    raise RuntimeError(
        "No usable post-close frames"
    )


# ============================================================
# q10 drift relative to close
# ============================================================

for label in [
    "Original",
    "Free",
    "Anchor",
]:

    q = frame_df[
        frame_df[
            "variant"
        ] == label
    ].sort_values(
        "action_index"
    )

    close_q10 = float(
        q.iloc[0][
            "gap_q10_mm"
        ]
    )

    idx = (
        frame_df[
            "variant"
        ] == label
    )

    frame_df.loc[
        idx,
        "gap_q10_drift_from_close_mm",
    ] = (
        frame_df.loc[
            idx,
            "gap_q10_mm",
        ]
        - close_q10
    )


# ============================================================
# Goal MPJPE + PA summary
# ============================================================

summary_rows = []

final_scene_idx = (
    pred_T - 1
)

m_final = (
    (labels[
        final_scene_idx
    ] == OID)
    &
    valid[
        final_scene_idx
    ]
    &
    (labels[0] == OID)
    &
    exists[0]
)

if int(
    m_final.sum()
) < 8:

    raise RuntimeError(
        "Too few final object points"
    )

P_orig_final = pred[
    "Original"
][
    final_scene_idx,
    m_final,
]

for label in [
    "Original",
    "Free",
    "Anchor",
]:

    P_final = pred[
        label
    ][
        final_scene_idx,
        m_final,
    ]

    goal_mpjpe_mm = float(
        np.linalg.norm(
            P_final
            -
            P_orig_final,
            axis=-1,
        ).mean()
        * 1000.0
    )

    q = frame_df[
        frame_df[
            "variant"
        ] == label
    ].sort_values(
        "action_index"
    )

    close_gap_q10 = float(
        q.iloc[0][
            "gap_q10_mm"
        ]
    )

    end_gap_q10 = float(
        q.iloc[-1][
            "gap_q10_mm"
        ]
    )

    postclose_drift_q10 = (
        end_gap_q10
        -
        close_gap_q10
    )

    postclose_pa_mean = float(
        q[
            "pa_gt0_mm"
        ].mean()
    )

    postclose_pa_final = float(
        q.iloc[-1][
            "pa_gt0_mm"
        ]
    )

    all_pa = []

    # EXACT same intervention PA semantics:
    # Pred(t) vs GT0 initial rigid geometry.
    for tt in range(
        1,
        pred_T,
    ):

        mt = (
            (labels[tt] == OID)
            &
            valid[tt]
            &
            (labels[0] == OID)
            &
            exists[0]
        )

        if int(
            mt.sum()
        ) < 8:
            continue

        try:

            pa, _ = (
                p_mpjpe_single(
                    pred[
                        label
                    ][
                        tt,
                        mt,
                    ],
                    gt[
                        0,
                        mt,
                    ],
                )
            )

            all_pa.append(
                float(pa)
                * 1000.0
            )

        except Exception:
            pass

    summary_rows.append(
        {
            "variant":
                label,

            "original_rms_acc_si":
                original_acc_si,

            "matched_rms_acc_si":
                (
                    original_acc_si
                    if label == "Original"
                    else (
                        free_acc_si
                        if label == "Free"
                        else anchor_acc_si
                    )
                ),

            "t_close":
                t_close,

            "human_close_frame":
                t_close + 1,

            "close_gap_q10_mm":
                close_gap_q10,

            "end_closed_gap_q10_mm":
                end_gap_q10,

            "postclose_gap_q10_drift_mm":
                postclose_drift_q10,

            "goal_mpjpe_to_original_mm":
                goal_mpjpe_mm,

            "pa_gt0_all_mean_mm":
                (
                    float(
                        np.mean(
                            all_pa
                        )
                    )
                    if all_pa
                    else np.nan
                ),

            "pa_gt0_postclose_mean_mm":
                postclose_pa_mean,

            "pa_gt0_postclose_final_mm":
                postclose_pa_final,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# Save
# ============================================================

FRAME_OUT = (
    OUTDIR
    / "can113_postclose_frame_metrics.csv"
)

SUMMARY_OUT = (
    OUTDIR
    / "can113_causal_summary.csv"
)

frame_df.to_csv(
    FRAME_OUT,
    index=False,
)

summary_df.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ============================================================
# Print result
# ============================================================

print()
print("=" * 88)
print("CAUSAL SUMMARY")
print("=" * 88)

with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    240,
):

    print(
        summary_df.to_string(
            index=False
        )
    )

print()
print("=" * 88)
print("POST-CLOSE FRAME METRICS")
print("=" * 88)

cols = [
    "variant",
    "human_action_frame",
    "gap_q10_mm",
    "gap_q10_drift_from_close_mm",
    "pa_gt0_mm",
]

print(
    frame_df[
        cols
    ]
    .sort_values(
        [
            "human_action_frame",
            "variant",
        ]
    )
    .to_string(
        index=False
    )
)

# No arbitrary threshold classification.
# Print the key continuous contrasts.
def value(
    variant,
    column,
):

    return float(
        summary_df.loc[
            summary_df[
                "variant"
            ] == variant,
            column,
        ].iloc[0]
    )


orig_close = value(
    "Original",
    "close_gap_q10_mm",
)

free_close = value(
    "Free",
    "close_gap_q10_mm",
)

anchor_close = value(
    "Anchor",
    "close_gap_q10_mm",
)

free_drift = value(
    "Free",
    "postclose_gap_q10_drift_mm",
)

anchor_drift = value(
    "Anchor",
    "postclose_gap_q10_drift_mm",
)

print()
print("=" * 88)
print("KEY CAUSAL CONTRASTS")
print("=" * 88)

print(
    "RMS Acc Free / Anchor =",
    f"{free_acc_si:.9f}",
    "/",
    f"{anchor_acc_si:.9f}",
    "m/s^2",
)

print(
    "t_close q10 gap | "
    "Original / Free / Anchor =",
    f"{orig_close:.6f}",
    "/",
    f"{free_close:.6f}",
    "/",
    f"{anchor_close:.6f}",
    "mm",
)

print(
    "post-close q10 drift | "
    "Free / Anchor =",
    f"{free_drift:.6f}",
    "/",
    f"{anchor_drift:.6f}",
    "mm",
)

print()
print(
    "Interpretation:"
)

print(
    "1) Anchor close-gap ~= Original AND "
    "Anchor drift << Free"
)

print(
    "   -> grasp/touch-point shift is "
    "the stronger explanation."
)

print(
    "2) Anchor close-gap ~= Original BUT "
    "Anchor still drifts strongly"
)

print(
    "   -> evidence for PointWorld "
    "contact-maintenance weakness."
)

print(
    "3) Anchor robot pose is exact BUT "
    "Anchor close-gap already differs"
)

print(
    "   -> future/pre-close action timing "
    "already changes the predicted object "
    "state at grasp time."
)

print()
print("FRAME CSV  =", FRAME_OUT)
print("SUMMARY CSV=", SUMMARY_OUT)
print("=" * 88)
