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

from scipy.optimize import minimize
from scipy.spatial import cKDTree

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pmod

from robot_sampler import RobotSampler as TorchRobotSampler
from utils import resolve_robot_urdf

from dataset_components.constants import RELEASE_GRIPPER_ONLY

from diagnostics.run_established_action_metrics_5task import (
    established_metrics_for_clip,
)

import diagnostics.eval_acc_only_boundary_pa_final as final_pa

old = final_pa.old


# ============================================================
# FIXED PROTOCOL
# ============================================================

TASK_DIRS = {
    "task-0000":
        "/home/soenchun/cse/PointWorld_workspace/"
        "datasets/behavior/wds_task0000",

    "task-0001":
        "/home/soenchun/cse/PointWorld_workspace/"
        "datasets/behavior/wds_task0001",

    "task-0006":
        "/home/soenchun/cse/PointWorld_workspace/"
        "datasets/behavior/wds_task0006",

    "task-0008":
        "/home/soenchun/cse/PointWorld_workspace/"
        "datasets/behavior/wds_task0008",

    "task-0010":
        "/home/soenchun/cse/PointWorld_workspace/"
        "datasets/behavior/wds_task0010",
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

ACC_CSV = Path(
    "diagnostics/results/"
    "current_behavior_established_acc_10457/"
    "per_sample_current_established_acc_10457.csv"
)

OUT_ROOT = Path(
    "diagnostics/results/"
    "close_anchor_causal_5task_current10457"
)

MODEL_PATH = (
    "pretrained_checkpoints/"
    "large-droid+behavior/model-best.pt"
)

NORM_STATS = "stats/droid_behavior"

MODEL_SEED = 12345
BUILD_SEED = 42

# Exact same dose definition used in can smoke.
DOSE_FRAC = 0.80

# SI m/s^2.
ACC_MATCH_TOL_SI = 5e-5

MIN_POINTS = 8


# ============================================================
# Helpers
# ============================================================

def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def arr(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_name(x):
    return str(x).replace("/", "_").replace(":", "__")


def squeeze_mask(x):
    x = arr(x)

    if x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    return x


def interp_sequence(a, phase):
    a = np.asarray(a, np.float64)

    T = len(a)
    p = np.asarray(phase, np.float64)

    lo = np.floor(p).astype(int)
    hi = np.ceil(p).astype(int)

    alpha = (
        p - lo
    ).reshape(
        (T,) + (1,) * (a.ndim - 1)
    )

    out = (
        (1.0 - alpha) * a[lo]
        + alpha * a[hi]
    )

    out[0] = a[0]
    out[-1] = a[-1]

    return out


def poses_at_phase(rp, lp, phase):
    r = np.asarray(rp, np.float64).copy()
    l = np.asarray(lp, np.float64).copy()

    r[:, :3] = interp_sequence(
        r[:, :3],
        phase,
    )

    l[:, :3] = interp_sequence(
        l[:, :3],
        phase,
    )

    return r, l


def acc_native(rp, lp, phase):
    r, l = poses_at_phase(
        rp,
        lp,
        phase,
    )

    return float(
        established_metrics_for_clip(
            r,
            l,
        )["rms_acceleration"]
    )


# ============================================================
# Phase construction
# ============================================================

def piecewise_gamma_phase(
    T,
    gamma,
    t_close=None,
):
    identity = np.arange(
        T,
        dtype=np.float64,
    )

    if t_close is None:
        u = np.linspace(
            0.0,
            1.0,
            T,
        )

        p = (
            u ** gamma
        ) * (T - 1)

        p[0] = 0.0
        p[-1] = T - 1

        return p

    p = identity.copy()

    if t_close > 0:
        u = np.linspace(
            0.0,
            1.0,
            t_close + 1,
        )

        p[:t_close + 1] = (
            u ** gamma
        ) * t_close

    if t_close < T - 1:
        n = T - t_close

        u = np.linspace(
            0.0,
            1.0,
            n,
        )

        p[t_close:] = (
            t_close
            +
            (
                u ** gamma
            )
            * (
                T - 1 - t_close
            )
        )

    p[0] = 0.0
    p[t_close] = float(t_close)
    p[-1] = float(T - 1)

    return p


def variable_indices(
    T,
    t_close=None,
):
    idx = list(
        range(
            1,
            T - 1,
        )
    )

    if (
        t_close is not None
        and
        t_close in idx
    ):
        idx.remove(t_close)

    return idx


def unpack_phase(
    x,
    T,
    t_close=None,
):
    identity = np.arange(
        T,
        dtype=np.float64,
    )

    idx = variable_indices(
        T,
        t_close,
    )

    p = identity.copy()
    p[idx] = x

    p[0] = 0.0
    p[-1] = float(T - 1)

    if t_close is not None:
        p[t_close] = float(t_close)

    return p


def starts_for(
    T,
    t_close=None,
    extra=None,
):
    idx = variable_indices(
        T,
        t_close,
    )

    identity = np.arange(
        T,
        dtype=np.float64,
    )

    starts = [
        identity[idx],
    ]

    for gamma in [
        0.70,
        0.85,
        1.15,
        1.30,
    ]:
        p = piecewise_gamma_phase(
            T,
            gamma,
            t_close,
        )

        starts.append(
            p[idx]
        )

    if extra is not None:
        p = np.asarray(
            extra,
            np.float64,
        ).copy()

        if t_close is not None:
            p[t_close] = float(t_close)

        p[0] = 0.0
        p[-1] = float(T - 1)

        starts.insert(
            0,
            p[idx],
        )

    return starts


def find_anchor_min_acc(
    rp,
    lp,
    t_close,
):
    T = len(rp)

    identity = np.arange(
        T,
        dtype=np.float64,
    )

    idx = variable_indices(
        T,
        t_close,
    )

    def objective(x):
        p = unpack_phase(
            x,
            T,
            t_close,
        )

        shift = (
            p - identity
        ) / float(T - 1)

        return (
            acc_native(
                rp,
                lp,
                p,
            )
            +
            1e-5
            * float(
                np.mean(
                    shift ** 2
                )
            )
        )

    constraints = [
        {
            "type": "ineq",
            "fun":
                lambda x:
                np.diff(
                    unpack_phase(
                        x,
                        T,
                        t_close,
                    )
                )
                - 1e-4,
        }
    ]

    best = None

    for x0 in starts_for(
        T,
        t_close,
    ):
        r = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[
                (
                    0.0,
                    float(T - 1),
                )
                for _ in idx
            ],
            constraints=constraints,
            options={
                "maxiter": 700,
                "ftol": 1e-11,
                "disp": False,
            },
        )

        p = unpack_phase(
            r.x,
            T,
            t_close,
        )

        a = acc_native(
            rp,
            lp,
            p,
        )

        feasible = bool(
            np.all(
                np.diff(p)
                >= -1e-8
            )
        )

        candidate = {
            "phase": p,
            "acc": a,
            "success": feasible,
            "optimizer_success":
                bool(r.success),
        }

        if (
            best is None
            or
            (
                candidate["success"]
                and
                not best["success"]
            )
            or
            (
                candidate["success"]
                ==
                best["success"]
                and
                candidate["acc"]
                <
                best["acc"]
            )
        ):
            best = candidate

    return best


def solve_target_acc(
    rp,
    lp,
    target_acc,
    *,
    t_close=None,
    reference_phase=None,
    extra_start=None,
):
    T = len(rp)

    identity = np.arange(
        T,
        dtype=np.float64,
    )

    idx = variable_indices(
        T,
        t_close,
    )

    ref = (
        identity
        if reference_phase is None
        else np.asarray(
            reference_phase,
            np.float64,
        )
    )

    def objective(x):
        p = unpack_phase(
            x,
            T,
            t_close,
        )

        d = (
            p - ref
        ) / float(T - 1)

        dd = (
            np.diff(p)
            -
            np.diff(ref)
        ) / float(T - 1)

        return float(
            np.mean(
                d ** 2
            )
            +
            0.02
            * np.mean(
                dd ** 2
            )
        )

    def current_acc(x):
        p = unpack_phase(
            x,
            T,
            t_close,
        )

        return acc_native(
            rp,
            lp,
            p,
        )

    constraints = [
        {
            "type": "ineq",
            "fun":
                lambda x:
                np.diff(
                    unpack_phase(
                        x,
                        T,
                        t_close,
                    )
                )
                - 1e-4,
        },
        {
            "type": "eq",
            "fun":
                lambda x:
                current_acc(x)
                - target_acc,
        },
    ]

    best = None

    for x0 in starts_for(
        T,
        t_close,
        extra=extra_start,
    ):
        r = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[
                (
                    0.0,
                    float(T - 1),
                )
                for _ in idx
            ],
            constraints=constraints,
            options={
                "maxiter": 800,
                "ftol": 1e-12,
                "disp": False,
            },
        )

        p = unpack_phase(
            r.x,
            T,
            t_close,
        )

        a = acc_native(
            rp,
            lp,
            p,
        )

        err_si = (
            abs(
                a
                - target_acc
            )
            * 25.0
        )

        feasible = bool(
            np.all(
                np.diff(p)
                >= -1e-8
            )
            and
            err_si
            <= ACC_MATCH_TOL_SI
        )

        candidate = {
            "phase": p,
            "acc": a,
            "err_si": err_si,
            "success": feasible,
            "objective":
                float(
                    objective(
                        r.x
                    )
                ),
        }

        if (
            best is None
            or
            (
                candidate["success"]
                and
                not best["success"]
            )
            or
            (
                candidate["success"]
                ==
                best["success"]
                and
                candidate["objective"]
                <
                best["objective"]
            )
        ):
            best = candidate

    return best


# ============================================================
# Close / release
# ============================================================

def close_events(
    right_open,
    left_open,
):
    ro = np.asarray(
        right_open
    ).reshape(-1).astype(int)

    lo = np.asarray(
        left_open
    ).reshape(-1).astype(int)

    events = []

    for side, x in [
        ("right", ro),
        ("left", lo),
    ]:
        for t in range(
            1,
            len(x),
        ):
            if (
                x[t - 1] == 1
                and
                x[t] == 0
            ):
                events.append(
                    (
                        t,
                        side,
                    )
                )
                break

    if not events:
        return None

    t0 = min(
        x[0]
        for x in events
    )

    sides = [
        side
        for t, side in events
        if t == t0
    ]

    side = (
        sides[0]
        if len(sides) == 1
        else "both"
    )

    return (
        int(t0),
        side,
    )


def closed_window(
    t_close,
    side,
    right_open,
    left_open,
):
    ro = np.asarray(
        right_open
    ).reshape(-1).astype(int)

    lo = np.asarray(
        left_open
    ).reshape(-1).astype(int)

    T = len(ro)

    frames = []

    for t in range(
        t_close,
        T,
    ):
        if side == "right":
            closed = (
                ro[t] == 0
            )

        elif side == "left":
            closed = (
                lo[t] == 0
            )

        else:
            closed = (
                ro[t] == 0
                and
                lo[t] == 0
            )

        if not closed:
            break

        frames.append(t)

    return frames


# ============================================================
# Gripper geometry
# ============================================================

def active_robot_indices(
    sample,
    side,
):
    robot = arr(
        sample[
            "robot_flows"
        ]
    )

    if (
        robot.ndim == 4
        and
        robot.shape[0] == 1
    ):
        robot = robot[0]

    if side == "both":
        return np.arange(
            robot.shape[1]
        )

    rp = arr(
        sample[
            "right_gripper_pose"
        ]
    )

    lp = arr(
        sample[
            "left_gripper_pose"
        ]
    )

    rp = rp.reshape(
        -1,
        7,
    )

    lp = lp.reshape(
        -1,
        7,
    )

    centers = np.stack(
        [
            rp[0, :3],
            lp[0, :3],
        ],
        axis=0,
    )

    d = np.linalg.norm(
        robot[
            0,
            :,
            None,
            :
        ]
        -
        centers[
            None,
            :,
            :
        ],
        axis=-1,
    )

    assignment = np.argmin(
        d,
        axis=1,
    )

    wanted = (
        0
        if side == "right"
        else 1
    )

    return np.flatnonzero(
        assignment == wanted
    )


def robot_points(
    sample,
    t,
    idx,
):
    r = arr(
        sample[
            "robot_flows"
        ]
    )

    if (
        r.ndim == 4
        and
        r.shape[0] == 1
    ):
        r = r[0]

    x = r[
        t,
        idx,
    ]

    if "robot_exists" in sample:
        e = arr(
            sample[
                "robot_exists"
            ]
        )

        if (
            e.ndim >= 3
            and
            e.shape[0] == 1
        ):
            e = e[0]

        if (
            e.ndim == 3
            and
            e.shape[-1] == 1
        ):
            e = e[..., 0]

        m = np.asarray(
            e[
                t,
                idx,
            ]
        ).reshape(-1).astype(bool)

        if len(m) == len(x):
            x = x[m]

    return np.asarray(
        x,
        np.float64,
    )


def gap_stats_mm(
    obj,
    grip,
):
    if (
        len(obj) == 0
        or
        len(grip) == 0
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    d = (
        cKDTree(
            grip
        )
        .query(
            obj,
            k=1,
        )[0]
        * 1000.0
    )

    return (
        float(
            np.min(d)
        ),
        float(
            np.percentile(
                d,
                10,
            )
        ),
        float(
            np.median(d)
        ),
    )


# ============================================================
# Main
# ============================================================


# ============================================================
# Fixed-phase N=524 relative-motion evaluation
# ============================================================

_REPLAY_ROOT = Path(
    "diagnostics/results/close_anchor_causal_5task_current10457"
)
_REPLAY_OUT = Path(
    "diagnostics/results/close_anchor_relative_replay_current524"
)



def _pose_local_points(P, pose):
    from scipy.spatial.transform import Rotation

    P = np.asarray(P, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64).reshape(7)

    if P.ndim != 2 or P.shape[1] != 3:
        raise RuntimeError("Invalid object points")

    if not np.isfinite(P).all() or not np.isfinite(pose).all():
        raise RuntimeError("Nonfinite pose or points")

    q = pose[3:]
    norm = np.linalg.norm(q)

    if norm < 0.95 or norm > 1.05:
        raise RuntimeError(f"Invalid quaternion norm: {norm}")

    rot = Rotation.from_quat(q / norm)

    return rot.inv().apply(P - pose[:3]), abs(norm - 1.0)


def _gripper_pose7(sample, t, arm):
    field = f"{arm}_gripper_pose"

    if field not in sample:
        raise RuntimeError(f"Missing {field}")

    poses = np.asarray(
        arr(sample[field]), dtype=np.float64
    ).reshape(-1, 7)

    if not 0 <= t < len(poses):
        raise RuntimeError("Pose timestep out of range")

    return poses[t]


def _relative_motion(P0, Pt, pose0, poset):
    P0 = np.asarray(P0, dtype=np.float64)
    Pt = np.asarray(Pt, dtype=np.float64)

    if P0.shape != Pt.shape or len(P0) < MIN_POINTS:
        raise RuntimeError("Object correspondence mismatch")

    rel0, e0 = _pose_local_points(P0, pose0)
    relt, et = _pose_local_points(Pt, poset)

    delta = relt - rel0
    centroid_delta = delta.mean(axis=0)

    return {
        "centroid_mm": float(
            np.linalg.norm(centroid_delta) * 1000.0
        ),
        "point_median_mm": float(
            np.median(np.linalg.norm(delta, axis=1)) * 1000.0
        ),
        "delta_xyz_m": centroid_delta,
        "quaternion_norm_error": max(e0, et),
    }


def _choose_reference_arm(
    close_side, released, gt, t_close, initial_mask
):
    if close_side in ("left", "right"):
        return close_side

    if close_side != "both":
        raise RuntimeError(f"Unknown close side: {close_side}")

    candidates = []

    for arm in ("right", "left"):
        idx = active_robot_indices(released["Original"], arm)

        gp = robot_points(
            released["Original"], t_close, idx
        )

        if len(gp) == 0:
            continue

        gmin, _, _ = gap_stats_mm(
            gt[t_close, initial_mask], gp
        )

        if np.isfinite(gmin):
            candidates.append((gmin, arm))

    if not candidates:
        raise RuntimeError("Cannot select reference arm")

    candidates.sort()
    return candidates[0][1]




def _export_relative_frames(
    task_out, key, oid, t_close, closed_frames,
    released, side_idx, pred, gt,
    labels, valid, exists, row, expected_row
):
    import csv

    assert int(row["object_id"]) == int(expected_row["object_id"])
    assert int(row["t_close"]) == int(expected_row["t_close"])
    assert str(row["close_side"]) == str(expected_row["close_side"])

    masks = {}
    good_times = []

    for t in closed_frames:
        m = (
            (labels[t] == oid)
            & valid[t]
            & (labels[0] == oid)
            & exists[0]
        )
        masks[t] = m

        if int(m.sum()) >= MIN_POINTS:
            good_times.append(t)

    if len(good_times) < 2:
        raise RuntimeError("Insufficient valid closed frames")

    if good_times[0] != t_close:
        raise RuntimeError("Grasp frame not valid")

    # Use the same object point identities at every evaluated frame.
    common = masks[good_times[0]].copy()
    for t in good_times[1:]:
        common &= masks[t]

    if int(common.sum()) < MIN_POINTS:
        raise RuntimeError(
            f"Insufficient common object points: {common.sum()}"
        )

    reference_arm = _choose_reference_arm(
        str(row["close_side"]),
        released,
        gt,
        t_close,
        common,
    )

    # Verify that the saved Anchor preserves the grasp pose.
    po = _gripper_pose7(
        released["Original"], t_close, reference_arm
    )
    pa = _gripper_pose7(
        released["Anchor"], t_close, reference_arm
    )

    if np.linalg.norm(po[:3] - pa[:3]) > 1e-7:
        raise RuntimeError("Anchor grasp translation changed")

    qo = po[3:] / np.linalg.norm(po[3:])
    qa = pa[3:] / np.linalg.norm(pa[3:])

    if 1.0 - abs(np.dot(qo, qa)) > 1e-6:
        raise RuntimeError("Anchor grasp orientation changed")

    records = []
    first_q10 = {}
    last_q10 = {}

    for t in good_times:
        frame_results = {}

        for condition in ("GT", "Original", "Free", "Anchor"):
            owner = "Original" if condition == "GT" else condition

            P = gt if condition == "GT" else pred[condition]

            # Preserve the ORIGINAL gap definition:
            # nearest distance from object points to selected robot surface.
            Ht = robot_points(
                released[owner], t, side_idx[owner]
            )

            gap_min, gap_q10, _ = gap_stats_mm(
                P[t, masks[t]], Ht
            )

            pose0 = _gripper_pose7(
                released[owner], t_close, reference_arm
            )
            poset = _gripper_pose7(
                released[owner], t, reference_arm
            )

            stats = _relative_motion(
                P[t_close, common],
                P[t, common],
                pose0,
                poset,
            )

            frame_results[condition] = stats

            records.append({
                "task": row["task"],
                "clip": key,
                "object_id": int(oid),
                "reference_arm": reference_arm,
                "t_close": int(t_close),
                "frame": int(t),
                "frames_since_close": int(t - t_close),
                "condition": condition,
                "n_full_points": int(masks[t].sum()),
                "n_common_points": int(common.sum()),
                "absolute_min_gap_mm": gap_min,
                "absolute_q10_gap_mm": gap_q10,
                "relative_centroid_drift_mm": stats["centroid_mm"],
                "relative_point_motion_median_mm":
                    stats["point_median_mm"],
                "quaternion_norm_error":
                    stats["quaternion_norm_error"],
                "relative_delta_x_m": stats["delta_xyz_m"][0],
                "relative_delta_y_m": stats["delta_xyz_m"][1],
                "relative_delta_z_m": stats["delta_xyz_m"][2],
                "original_gt_relative_delta_error_mm":
                    float("nan"),
            })

            if condition != "GT":
                first_q10.setdefault(condition, gap_q10)
                last_q10[condition] = gap_q10

        # Counterfactual GT does not exist for Free/Anchor.
        gt_delta = frame_results["GT"]["delta_xyz_m"]
        orig_delta = frame_results["Original"]["delta_xyz_m"]

        records[-3][
            "original_gt_relative_delta_error_mm"
        ] = float(
            np.linalg.norm(orig_delta - gt_delta) * 1000.0
        )

    # Existing gap values must be reproducible.
    for condition, prefix in (
        ("Original", "original"),
        ("Free", "free"),
        ("Anchor", "anchor"),
    ):
        for suffix, value in (
            ("close", first_q10[condition]),
            ("end", last_q10[condition]),
        ):
            expected = float(
                row[f"{prefix}_{suffix}_gap_q10_mm"]
            )

            error = abs(value - expected)

            if error > 1e-5:
                raise RuntimeError(
                    f"{key}: {condition} {suffix} "
                    f"gap replay mismatch {error:.6f} mm"
                )

    # Frame t_close must have zero relative displacement.
    for rec in records:
        if rec["frame"] == t_close:
            if rec["relative_centroid_drift_mm"] > 1e-5:
                raise RuntimeError("Nonzero grasp-frame drift")

    output = (
        Path(task_out) / "per_frame_relative_metrics.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )

        if output.stat().st_size == 0:
            writer.writeheader()

        writer.writerows(records)

    print(
        f"[RELATIVE_OK] {key} "
        f"frames={len(good_times)} "
        f"common_points={int(common.sum())} "
        f"reference_arm={reference_arm}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
        choices=list(
            TASK_DIRS.keys()
        ),
    )

    parser.add_argument("--limit", type=int, default=0)
    a = parser.parse_args()

    task = a.task
    data_dir = TASK_DIRS[task]

    global OUT_ROOT
    OUT_ROOT = _REPLAY_OUT

    replay_csv = (
        _REPLAY_ROOT / task / "per_clip_causal_metrics.csv"
    )
    replay_df = pd.read_csv(replay_csv)

    if replay_df["clip"].duplicated().any():
        raise RuntimeError("Duplicate cached clips")

    replay_by_key = {
        str(r["clip"]): r
        for _, r in replay_df.iterrows()
    }

    replay_keys = set(replay_by_key)
    assert replay_keys

    print("[REPLAY] cached clips:", len(replay_keys))

    task_out = (
        OUT_ROOT
        / task
    )

    task_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase_dir = (
        task_out
        / "matched_phases"
    )

    phase_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_csv = (
        task_out
        / "per_clip_causal_metrics.csv"
    )

    status_csv = (
        task_out
        / "status.csv"
    )

    # --------------------------------------------------------
    # Current PA-eligible population
    # --------------------------------------------------------

    object_csv = (
        PA_ROOT
        / task
        / "per_object_metrics.csv"
    )

    pa = pd.read_csv(
        object_csv
    )

    pa = (
        pa[
            [
                "clip",
                "object_id",
                "object_full_name",
                "object_short_name",
            ]
        ]
        .drop_duplicates(
            [
                "clip",
                "object_id",
            ]
        )
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
                        r.object_full_name
                    ),

                "object_short_name":
                    str(
                        r.object_short_name
                    ),
            }
            for r in g.itertuples()
        ]

    if (
        len(eligible_by_clip)
        != EXPECTED_PA[task]
    ):
        raise RuntimeError(
            f"{task}: PA population "
            f"{len(eligible_by_clip)} "
            f"!= {EXPECTED_PA[task]}"
        )

    target_keys = set(
        eligible_by_clip
    )

    # --------------------------------------------------------
    # Current established Acc audit
    # --------------------------------------------------------

    ac = pd.read_csv(
        ACC_CSV
    )

    ac = ac[
        ac["task"] == task
    ]

    acc_lookup = {
        str(r.clip):
            float(
                r.rms_acceleration_si_m_s2
            )
        for r in ac.itertuples()
    }

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    rows = []

    if out_csv.exists():
        old_rows = pd.read_csv(
            out_csv
        )

        rows = (
            old_rows
            .to_dict("records")
        )

    done = {
        str(x["clip"])
        for x in rows
    }

    statuses = []

    # ========================================================
    # PER_CLIP_EXCEPTION_GUARD_V2
    #
    # Every normal exclusion and unexpected per-clip error is
    # recorded, printed, and followed by the next clip.
    # ========================================================
    target_seen = 0
    current_target_index = 0
    current_key = ""

    def add_status(record):
        record = dict(record)

        record.setdefault(
            "task",
            task,
        )

        record.setdefault(
            "clip",
            current_key,
        )

        record.setdefault(
            "target_index",
            current_target_index,
        )

        statuses.append(record)

        status = str(
            record.get(
                "status",
                "unknown",
            )
        )

        # Successful samples already have their detailed
        # normal progress print below.
        if status != "matched_scored":

            kind = (
                "ERROR"
                if status == "unexpected_exception"
                else "SKIP"
            )

            detail = []

            for field in [
                "t_close",
                "close_side",
                "missing_fields",
                "error_type",
                "error",
            ]:
                if (
                    field in record
                    and
                    str(record[field]) not in [
                        "",
                        "nan",
                        "None",
                    ]
                ):
                    detail.append(
                        f"{field}={record[field]}"
                    )

            detail_text = (
                " | " + " | ".join(detail)
                if detail
                else ""
            )

            print(
                f"[{kind}] "
                f"task={record.get('task', task)} "
                f"target={record.get('target_index', '?')}/"
                f"{len(target_keys)} "
                f"clip={record.get('clip', current_key)} "
                f"reason={status}"
                f"{detail_text}",
                flush=True,
            )

    if status_csv.exists():
        statuses = (
            pd.read_csv(
                status_csv
            )
            .to_dict("records")
        )

    def flush():
        if rows:
            (
                pd.DataFrame(
                    rows
                )
                .drop_duplicates(
                    "clip",
                    keep="last",
                )
                .to_csv(
                    out_csv,
                    index=False,
                )
            )

        if statuses:
            (
                pd.DataFrame(
                    statuses
                )
                .drop_duplicates(
                    "clip",
                    keep="last",
                )
                .to_csv(
                    status_csv,
                    index=False,
                )
            )

    # --------------------------------------------------------
    # Object-ID hooks
    # --------------------------------------------------------

    dlmod.build_flow_sample = (
        old.omap
        .build_flow_sample_with_object_ids
    )

    pmod.grid_sample_transform = (
        old.omap
        .grid_sample_with_object_ids
    )

    pmod.gather = (
        old.omap
        .gather_with_object_ids
    )

    old.omap.dlmod.build_flow_sample = (
        old.omap
        .build_flow_sample_with_object_ids
    )

    old.omap.pmod.grid_sample_transform = (
        old.omap
        .grid_sample_with_object_ids
    )

    old.omap.pmod.gather = (
        old.omap
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
    # One-pass original WDS
    # --------------------------------------------------------

    shards = (
        dlmod.gather_shard_paths(
            data_dir,
            "test",
            rank=tester.rank,
        )
    )

    random.Random(
        args.seed
        + tester.rank
    ).shuffle(
        shards
    )

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
        handler=
            wds.warn_and_continue,
    )

    scanned = 0
    matched = len(done)

    print("=" * 100)
    print("CLOSE-ANCHOR CAUSAL 5-TASK EVAL")
    print("TASK       =", task)
    print("PA TARGETS =", len(target_keys))
    print("RESUME     =", len(done))
    print("DOSE       =", DOSE_FRAC)
    print("=" * 100)

    for raw in dataset:
        if a.limit and target_seen >= a.limit:
            break
        key = ""
        current_target_index = 0

        try:
            scanned += 1

            key = str(
                raw.get(
                    "__key__",
                    "",
                )
            )

            if key not in replay_keys:
                continue

            if key not in target_keys:
                continue

            if key in done:
                continue

            target_seen += 1
            current_target_index = target_seen
            current_key = key

            decoded = old.decode_data(
                raw,
                domain="behavior",
            )

            if str(
                decoded.get(
                    "__key__",
                    key,
                )
            ) != key:
                raise RuntimeError(
                    f"{key}: decode key mismatch"
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

            ev = close_events(
                right_open,
                left_open,
            )

            if ev is None:
                add_status({
                    "clip": key,
                    "task": task,
                    "status": "no_open_to_close",
                })
                flush()
                continue

            t_close, close_side = ev

            closed_frames = (
                closed_window(
                    t_close,
                    close_side,
                    right_open,
                    left_open,
                )
            )

            if len(closed_frames) < 2:
                add_status({
                    "clip": key,
                    "task": task,
                    "status":
                        "closed_window_too_short",
                    "t_close": t_close,
                    "close_side": close_side,
                })
                flush()
                continue

            rp = np.asarray(
                decoded[
                    "right_gripper_pose"
                ],
                np.float64,
            )

            lp = np.asarray(
                decoded[
                    "left_gripper_pose"
                ],
                np.float64,
            )

            T = len(rp)

            identity = np.arange(
                T,
                dtype=np.float64,
            )

            original_acc = (
                acc_native(
                    rp,
                    lp,
                    identity,
                )
            )

            original_acc_si = (
                original_acc
                * 25.0
            )

            if key not in acc_lookup:
                raise RuntimeError(
                    f"{key}: no current Acc"
                )

            if not np.isclose(
                original_acc_si,
                acc_lookup[key],
                atol=2e-5,
                rtol=2e-5,
            ):
                raise RuntimeError(
                    f"{key}: Acc audit failed | "
                    f"{original_acc_si} "
                    f"vs {acc_lookup[key]}"
                )

            # ----------------------------------------------------
            # 1. Best anchor-feasible dynamics
            # ----------------------------------------------------


            cache_path = (
                _REPLAY_ROOT / task / "matched_phases"
                / (safe_name(key) + ".npz")
            )

            with np.load(cache_path, allow_pickle=False) as z:
                assert str(z["clip"].item()) == key
                assert int(z["t_close"]) == t_close
                assert str(z["close_side"].item()) == close_side

                saved_original_acc = float(z["original_acc"])
                saved_target_acc = float(z["target_acc"])
                saved_free_acc = float(z["free_acc"])
                saved_anchor_acc = float(z["anchor_acc"])

                saved_free_phase = np.asarray(
                    z["free_phase"], np.float64
                ).copy()

                saved_anchor_phase = np.asarray(
                    z["anchor_phase"], np.float64
                ).copy()

            cache_row = replay_by_key[key]

            if not np.isclose(
                original_acc, saved_original_acc,
                atol=1e-8, rtol=1e-6
            ):
                raise RuntimeError("Original Acc mismatch")

            for phase in (saved_free_phase, saved_anchor_phase):
                if phase.shape != identity.shape:
                    raise RuntimeError("Phase shape mismatch")
                if not np.isfinite(phase).all():
                    raise RuntimeError("Nonfinite phase")
                if np.min(np.diff(phase)) < -1e-8:
                    raise RuntimeError("Nonmonotonic phase")
                if abs(phase[0]) > 1e-8:
                    raise RuntimeError("Initial pose changed")
                if abs(phase[-1] - (T-1)) > 1e-8:
                    raise RuntimeError("Final pose changed")

            if abs(saved_anchor_phase[t_close] - t_close) > 1e-8:
                raise RuntimeError("Grasp pose not fixed")

            amin = {
                "success": True,
                "acc": float(
                    cache_row["anchor_min_acc_si"]
                ) / 25.0,
                "phase": saved_anchor_phase,
            }

            if (
                amin is None
                or
                not amin["success"]
                or
                amin["acc"]
                >=
                original_acc
                - 1e-10
            ):
                add_status({
                    "clip": key,
                    "task": task,
                    "status":
                        "no_anchor_reduction",
                    "t_close": t_close,
                    "close_side": close_side,
                })
                flush()
                continue

            target_acc = (
                original_acc
                -
                DOSE_FRAC
                * (
                    original_acc
                    -
                    amin["acc"]
                )
            )

            # ----------------------------------------------------
            # 2. FREE matched target
            # ----------------------------------------------------


            if not np.isclose(
                target_acc, saved_target_acc,
                atol=1e-8, rtol=1e-6
            ):
                raise RuntimeError("Target Acc mismatch")

            free = {
                "success": True,
                "acc": saved_free_acc,
                "phase": saved_free_phase,
            }

            if not free["success"]:
                add_status({
                    "clip": key,
                    "task": task,
                    "status":
                        "free_target_fail",
                    "t_close": t_close,
                    "close_side": close_side,
                })
                flush()
                continue

            # ----------------------------------------------------
            # 3. ANCHOR matched target
            #
            # Reference is FREE phase:
            # minimal additional change necessary to enforce
            # phase[t_close] == t_close at the same RMS Acc.
            # ----------------------------------------------------


            anchor = {
                "success": True,
                "acc": saved_anchor_acc,
                "phase": saved_anchor_phase,
            }

            if not anchor["success"]:
                add_status({
                    "clip": key,
                    "task": task,
                    "status":
                        "anchor_target_fail",
                    "t_close": t_close,
                    "close_side": close_side,
                })
                flush()
                continue

            free_acc_si = (
                free["acc"]
                * 25.0
            )

            anchor_acc_si = (
                anchor["acc"]
                * 25.0
            )

            if (
                abs(
                    free_acc_si
                    -
                    anchor_acc_si
                )
                >
                ACC_MATCH_TOL_SI
            ):
                add_status({
                    "clip": key,
                    "task": task,
                    "status":
                        "free_anchor_acc_mismatch",
                    "free_acc_si":
                        free_acc_si,
                    "anchor_acc_si":
                        anchor_acc_si,
                })
                flush()
                continue

            # ----------------------------------------------------
            # Official deterministic build ONCE
            # ----------------------------------------------------

            built = (
                old.omap
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

            if (
                built[
                    "robot_flows"
                ].shape[0]
                != T
            ):
                raise RuntimeError(
                    f"{key}: T mismatch"
                )

            phases = {
                "Original":
                    identity,

                "Free":
                    free["phase"],

                "Anchor":
                    anchor["phase"],
            }

            samples = {}

            # ----------------------------------------------------
            # Original discrete commands are fixed for EVERY variant.
            #
            # IMPORTANT:
            # Some official built samples legitimately omit one
            # gripper_open key. Never invent a missing model input.
            # For the causal close-anchor experiment, however, the
            # active close side MUST exist in the official built sample.
            # ----------------------------------------------------

            discrete_commands = {}

            for field in [
                "right_gripper_open",
                "left_gripper_open",
            ]:
                if field in built:
                    discrete_commands[field] = (
                        np.asarray(
                            built[field]
                        ).copy()
                    )

            required_close_fields = []

            if close_side in ("right", "both"):
                required_close_fields.append(
                    "right_gripper_open"
                )

            if close_side in ("left", "both"):
                required_close_fields.append(
                    "left_gripper_open"
                )

            missing_close_fields = [
                field
                for field in required_close_fields
                if field not in discrete_commands
            ]

            if missing_close_fields:
                add_status({
                    "clip": key,
                    "task": task,
                    "status":
                        "active_close_command_missing_in_built",
                    "t_close": t_close,
                    "close_side": close_side,
                    "missing_fields":
                        "|".join(
                            missing_close_fields
                        ),
                })
                flush()
                continue

            for label, phase in phases.items():
                s = old.src.apply_phase(
                    copy.deepcopy(
                        built
                    ),
                    phase
                    / float(T - 1),
                )

                # CRITICAL FAIRNESS:
                # Preserve only command fields that genuinely existed
                # in the official built model input.
                for field, value in (
                    discrete_commands.items()
                ):
                    s[field] = value.copy()

                samples[label] = s

            # ----------------------------------------------------
            # Hard input audits
            # ----------------------------------------------------

            # Exact discrete-command equality audit.
            # Only fields present in the official built sample are audited.
            for field in discrete_commands:
                ref = np.asarray(
                    samples["Original"][field]
                )

                for label in [
                    "Free",
                    "Anchor",
                ]:
                    cur = np.asarray(
                        samples[label][field]
                    )

                    if not np.array_equal(
                        ref,
                        cur,
                    ):
                        raise RuntimeError(
                            f"{key}: {label} changed "
                            f"discrete command {field}"
                        )

            anchor_close_robot_err = float(
                np.max(
                    np.abs(
                        np.asarray(
                            samples[
                                "Anchor"
                            ][
                                "robot_flows"
                            ][t_close]
                        )
                        -
                        np.asarray(
                            samples[
                                "Original"
                            ][
                                "robot_flows"
                            ][t_close]
                        )
                    )
                )
            )

            if (
                anchor_close_robot_err
                > 1e-7
            ):
                raise RuntimeError(
                    f"{key}: anchor close robot changed"
                )

            for label in [
                "Free",
                "Anchor",
            ]:
                for ti in [
                    0,
                    -1,
                ]:
                    if not np.allclose(
                        samples[label][
                            "robot_flows"
                        ][ti],
                        samples[
                            "Original"
                        ][
                            "robot_flows"
                        ][ti],
                        atol=1e-7,
                        rtol=0,
                    ):
                        raise RuntimeError(
                            f"{key}: endpoint changed"
                        )

            # ----------------------------------------------------
            # Exact validated release pipeline
            # ----------------------------------------------------

            released = {}

            for label in [
                "Original",
                "Free",
                "Anchor",
            ]:
                released[label] = (
                    old.release_process(
                        copy.deepcopy(
                            samples[
                                label
                            ]
                        ),
                        args,
                        "behavior",
                        tester.rank,
                    )
                )

            audit_batch = (
                old.custom_collate_fn(
                    [
                        released["Original"],
                        released["Free"],
                        released["Anchor"],
                    ],
                    args=args,
                )
            )

            for field in [
                "scene_flows",
                "gt_scene_flows",
                "scene_exists",
                "scene_context_mask",
                "scene_supervised_mask",
                "scene_object_id",
            ]:
                x = audit_batch[field]

                if isinstance(
                    x,
                    torch.Tensor,
                ):
                    if not torch.equal(
                        x[0],
                        x[1],
                    ):
                        raise RuntimeError(
                            f"{key}: "
                            f"{field} Original != Free"
                        )

                    if not torch.equal(
                        x[0],
                        x[2],
                    ):
                        raise RuntimeError(
                            f"{key}: "
                            f"{field} Original != Anchor"
                        )

            # ----------------------------------------------------
            # Three same-seed forwards
            # ----------------------------------------------------

            pred = {}
            batches = {}

            for label in [
                "Original",
                "Free",
                "Anchor",
            ]:
                b = old.custom_collate_fn(
                    [
                        released[label]
                    ],
                    args=args,
                )

                batches[label] = b

                reset_seed(
                    MODEL_SEED
                )

                g = old.to_device(
                    b,
                    tester.device,
                )

                with torch.inference_mode():
                    out = tester.model(
                        g,
                        training=False,
                    )

                pred[label] = (
                    out[
                        "scene_flows"
                    ][0]
                    .detach()
                    .cpu()
                    .numpy()
                )

            # ----------------------------------------------------
            # Masks
            # ----------------------------------------------------

            b0 = batches[
                "Original"
            ]

            labels = squeeze_mask(
                b0[
                    "scene_object_id"
                ]
            ).astype(int)

            exists = squeeze_mask(
                b0[
                    "scene_exists"
                ]
            ).astype(bool)

            context = squeeze_mask(
                b0[
                    "scene_context_mask"
                ]
            ).astype(bool)

            supervised = squeeze_mask(
                b0[
                    "scene_supervised_mask"
                ]
            ).astype(bool)

            valid = (
                (~context)
                & exists
                & supervised
            )

            gt = arr(
                b0[
                    "gt_scene_flows"
                ]
            )[0]

            if (
                pred[
                    "Original"
                ].shape[0]
                != T
            ):
                raise RuntimeError(
                    f"{key}: prediction T "
                    f"{pred['Original'].shape[0]} != {T}"
                )

            # ----------------------------------------------------
            # Active gripper points
            # ----------------------------------------------------

            side_idx = {}

            for label in [
                "Original",
                "Free",
                "Anchor",
            ]:
                side_idx[label] = (
                    active_robot_indices(
                        released[label],
                        close_side,
                    )
                )

            grip_gt_ref = robot_points(
                released["Original"],
                t_close,
                side_idx["Original"],
            )

            # ----------------------------------------------------
            # Evaluation-only object selection:
            #
            # nearest PA-eligible rigid object to active gripper
            # in ORIGINAL GT geometry at close.
            #
            # No threshold and no object ID used by the model.
            # ----------------------------------------------------

            candidates = []

            for meta in (
                eligible_by_clip[key]
            ):
                oid = int(
                    meta[
                        "object_id"
                    ]
                )

                m = (
                    (labels[t_close] == oid)
                    &
                    valid[t_close]
                    &
                    (labels[0] == oid)
                    &
                    exists[0]
                )

                if int(
                    m.sum()
                ) < MIN_POINTS:
                    continue

                gmin, gq10, _ = (
                    gap_stats_mm(
                        gt[
                            t_close,
                            m,
                        ],
                        grip_gt_ref,
                    )
                )

                if not np.isfinite(
                    gmin
                ):
                    continue

                candidates.append(
                    (
                        gmin,
                        gq10,
                        oid,
                        meta,
                    )
                )

            if not candidates:
                add_status({
                    "clip": key,
                    "task": task,
                    "status":
                        "no_close_object_candidate",
                    "t_close": t_close,
                    "close_side": close_side,
                })
                flush()
                continue

            candidates.sort(
                key=lambda x: x[0]
            )

            (
                acquisition_min_mm,
                acquisition_q10_mm,
                oid,
                obj_meta,
            ) = candidates[0]

            # ----------------------------------------------------
            # Per-variant contact trajectory
            # ----------------------------------------------------

            close_gap = {}
            end_gap = {}
            drift = {}
            max_increase = {}
            mean_abs_change = {}

            close_obj_mpjpe = {}
            goal_mpjpe = {}

            pa_all = {}
            pa_post = {}

            for label in [
                "Original",
                "Free",
                "Anchor",
            ]:
                gap_series = []
                pa_post_vals = []

                grip_idx = (
                    side_idx[label]
                )

                for tt in closed_frames:
                    m = (
                        (labels[tt] == oid)
                        &
                        valid[tt]
                        &
                        (labels[0] == oid)
                        &
                        exists[0]
                    )

                    if int(
                        m.sum()
                    ) < MIN_POINTS:
                        continue

                    gp = robot_points(
                        released[label],
                        tt,
                        grip_idx,
                    )

                    _, q10, _ = (
                        gap_stats_mm(
                            pred[label][
                                tt,
                                m,
                            ],
                            gp,
                        )
                    )

                    gap_series.append(
                        (
                            tt,
                            q10,
                        )
                    )

                    gt0 = gt[
                        0,
                        m,
                    ]

                    centered = (
                        gt0
                        -
                        gt0.mean(
                            axis=0,
                            keepdims=True,
                        )
                    )

                    if (
                        np.linalg.matrix_rank(
                            centered,
                            tol=1e-8,
                        )
                        >= 2
                    ):
                        try:
                            paval, _ = (
                                old.p_mpjpe_single(
                                    pred[label][
                                        tt,
                                        m,
                                    ],
                                    gt0,
                                )
                            )

                            pa_post_vals.append(
                                float(
                                    paval
                                )
                                * 1000.0
                            )

                        except Exception:
                            pass

                if len(
                    gap_series
                ) < 2:
                    break

                vals = np.asarray(
                    [
                        x[1]
                        for x in gap_series
                    ],
                    dtype=np.float64,
                )

                close_gap[label] = float(
                    vals[0]
                )

                end_gap[label] = float(
                    vals[-1]
                )

                drift[label] = float(
                    vals[-1]
                    -
                    vals[0]
                )

                max_increase[label] = (
                    float(
                        np.max(
                            vals
                            -
                            vals[0]
                        )
                    )
                )

                mean_abs_change[label] = (
                    float(
                        np.mean(
                            np.abs(
                                vals
                                -
                                vals[0]
                            )
                        )
                    )
                )

                # object shift at close
                m_close = (
                    (labels[t_close] == oid)
                    &
                    valid[t_close]
                    &
                    (labels[0] == oid)
                    &
                    exists[0]
                )

                close_obj_mpjpe[
                    label
                ] = float(
                    np.linalg.norm(
                        pred[label][
                            t_close,
                            m_close,
                        ]
                        -
                        pred[
                            "Original"
                        ][
                            t_close,
                            m_close,
                        ],
                        axis=-1,
                    ).mean()
                    * 1000.0
                )

                # final goal shift
                tf = T - 1

                mf = (
                    (labels[tf] == oid)
                    &
                    valid[tf]
                    &
                    (labels[0] == oid)
                    &
                    exists[0]
                )

                if int(
                    mf.sum()
                ) >= MIN_POINTS:
                    goal_mpjpe[
                        label
                    ] = float(
                        np.linalg.norm(
                            pred[label][
                                tf,
                                mf,
                            ]
                            -
                            pred[
                                "Original"
                            ][
                                tf,
                                mf,
                            ],
                            axis=-1,
                        ).mean()
                        * 1000.0
                    )

                else:
                    goal_mpjpe[
                        label
                    ] = np.nan

                pa_post[label] = (
                    float(
                        np.mean(
                            pa_post_vals
                        )
                    )
                    if pa_post_vals
                    else np.nan
                )

                vals_all = []

                for tt in range(
                    1,
                    T,
                ):
                    m = (
                        (labels[tt] == oid)
                        &
                        valid[tt]
                        &
                        (labels[0] == oid)
                        &
                        exists[0]
                    )

                    if int(
                        m.sum()
                    ) < MIN_POINTS:
                        continue

                    gt0 = gt[
                        0,
                        m,
                    ]

                    centered = (
                        gt0
                        -
                        gt0.mean(
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

                    try:
                        pv, _ = (
                            old.p_mpjpe_single(
                                pred[label][
                                    tt,
                                    m,
                                ],
                                gt0,
                            )
                        )

                        vals_all.append(
                            float(pv)
                            * 1000.0
                        )

                    except Exception:
                        pass

                pa_all[label] = (
                    float(
                        np.mean(
                            vals_all
                        )
                    )
                    if vals_all
                    else np.nan
                )

            if any(
                x not in close_gap
                for x in [
                    "Original",
                    "Free",
                    "Anchor",
                ]
            ):
                add_status({
                    "clip": key,
                    "task": task,
                    "status":
                        "insufficient_postclose_support",
                    "t_close": t_close,
                    "close_side": close_side,
                })
                flush()
                continue

            # ----------------------------------------------------
            # Fairness / causal row
            # ----------------------------------------------------

            free_phase = np.asarray(
                free["phase"]
            )

            anchor_phase = np.asarray(
                anchor["phase"]
            )

            phase_rms_free = float(
                np.sqrt(
                    np.mean(
                        (
                            free_phase
                            -
                            identity
                        ) ** 2
                    )
                )
            )

            phase_rms_anchor = float(
                np.sqrt(
                    np.mean(
                        (
                            anchor_phase
                            -
                            identity
                        ) ** 2
                    )
                )
            )

            phase_free_anchor_rms = (
                float(
                    np.sqrt(
                        np.mean(
                            (
                                free_phase
                                -
                                anchor_phase
                            ) ** 2
                        )
                    )
                )
            )

            row = {
                "task": task,
                "clip": key,

                "object_id": oid,
                "object_full_name":
                    obj_meta[
                        "object_full_name"
                    ],
                "object_short_name":
                    obj_meta[
                        "object_short_name"
                    ],

                "t_close": t_close,
                "close_side": close_side,
                "closed_frames":
                    len(closed_frames),

                "original_acc_si":
                    original_acc_si,

                "anchor_min_acc_si":
                    amin["acc"] * 25.0,

                "target_acc_si":
                    target_acc * 25.0,

                "free_acc_si":
                    free_acc_si,

                "anchor_acc_si":
                    anchor_acc_si,

                "free_anchor_acc_diff_si":
                    abs(
                        free_acc_si
                        -
                        anchor_acc_si
                    ),

                "phase_rms_free_frames":
                    phase_rms_free,

                "phase_rms_anchor_frames":
                    phase_rms_anchor,

                "phase_free_anchor_rms_frames":
                    phase_free_anchor_rms,

                "free_close_phase_shift_frames":
                    abs(
                        free_phase[t_close]
                        -
                        t_close
                    ),

                "anchor_close_phase_shift_frames":
                    abs(
                        anchor_phase[t_close]
                        -
                        t_close
                    ),

                "anchor_close_robot_input_maxerr":
                    anchor_close_robot_err,

                "gt_acquisition_min_gap_mm":
                    acquisition_min_mm,

                "gt_acquisition_q10_gap_mm":
                    acquisition_q10_mm,
            }

            for label, prefix in [
                ("Original", "original"),
                ("Free", "free"),
                ("Anchor", "anchor"),
            ]:
                row[
                    f"{prefix}_close_gap_q10_mm"
                ] = close_gap[label]

                row[
                    f"{prefix}_end_gap_q10_mm"
                ] = end_gap[label]

                row[
                    f"{prefix}_postclose_drift_q10_mm"
                ] = drift[label]

                row[
                    f"{prefix}_max_gap_increase_mm"
                ] = max_increase[label]

                row[
                    f"{prefix}_mean_abs_gap_change_mm"
                ] = mean_abs_change[label]

                row[
                    f"{prefix}_close_object_mpjpe_to_original_mm"
                ] = close_obj_mpjpe[
                    label
                ]

                row[
                    f"{prefix}_goal_mpjpe_to_original_mm"
                ] = goal_mpjpe[
                    label
                ]

                row[
                    f"{prefix}_pa_gt0_all_mean_mm"
                ] = pa_all[
                    label
                ]

                row[
                    f"{prefix}_pa_gt0_postclose_mean_mm"
                ] = pa_post[
                    label
                ]

            # Relative-to-Original causal preservation.
            row[
                "free_close_gap_absdiff_from_original_mm"
            ] = abs(
                close_gap["Free"]
                -
                close_gap["Original"]
            )

            row[
                "anchor_close_gap_absdiff_from_original_mm"
            ] = abs(
                close_gap["Anchor"]
                -
                close_gap["Original"]
            )

            row[
                "free_postclose_drift_absdiff_from_original_mm"
            ] = abs(
                drift["Free"]
                -
                drift["Original"]
            )

            row[
                "anchor_postclose_drift_absdiff_from_original_mm"
            ] = abs(
                drift["Anchor"]
                -
                drift["Original"]
            )

            _export_relative_frames(
                task_out=task_out,
                key=key,
                oid=oid,
                t_close=t_close,
                closed_frames=closed_frames,
                released=released,
                side_idx=side_idx,
                pred=pred,
                gt=gt,
                labels=labels,
                valid=valid,
                exists=exists,
                row=row,
                expected_row=replay_by_key[key],
            )

            rows.append(row)
            done.add(key)
            matched += 1

            np.savez(
                phase_dir
                /
                f"{safe_name(key)}.npz",

                clip=np.array(key),
                task=np.array(task),

                t_close=np.array(
                    t_close
                ),

                close_side=np.array(
                    close_side
                ),

                original_acc=np.array(
                    original_acc
                ),

                target_acc=np.array(
                    target_acc
                ),

                free_acc=np.array(
                    free["acc"]
                ),

                anchor_acc=np.array(
                    anchor["acc"]
                ),

                free_phase=
                    free_phase,

                anchor_phase=
                    anchor_phase,
            )

            add_status({
                "clip": key,
                "task": task,
                "status": "matched_scored",
                "t_close": t_close,
                "close_side": close_side,
            })

            flush()

            print(
                f"[{task}] "
                f"N={matched} "
                f"clip={key} "
                f"close={t_close}/{close_side} "
                f"Acc={free_acc_si:.6f}/"
                f"{anchor_acc_si:.6f} "
                f"closeObj="
                f"{close_obj_mpjpe['Free']:.3f}/"
                f"{close_obj_mpjpe['Anchor']:.3f}mm "
                f"goal="
                f"{goal_mpjpe['Free']:.3f}/"
                f"{goal_mpjpe['Anchor']:.3f}mm",
                flush=True,
            )


        except Exception as e:

            err_key = (
                key
                if key
                else (
                    current_key
                    if current_key
                    else "<unknown>"
                )
            )

            # Never keep a partially completed metric row.
            rows[:] = [
                r
                for r in rows
                if str(
                    r.get(
                        "clip",
                        "",
                    )
                )
                != str(err_key)
            ]

            done.discard(
                str(err_key)
            )

            matched = len(rows)

            add_status({
                "clip":
                    str(err_key),

                "task":
                    task,

                "status":
                    "unexpected_exception",

                "target_index":
                    current_target_index,

                "error_type":
                    type(e).__name__,

                "error":
                    str(e),
            })

            print(
                "[ERROR TRACEBACK]",
                flush=True,
            )

            traceback.print_exc()

            try:
                flush()
            except Exception as flush_error:
                print(
                    f"[ERROR] status flush failed | "
                    f"{type(flush_error).__name__}: "
                    f"{flush_error}",
                    flush=True,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

    flush()

    # ========================================================
    # Final task-level exclusion/error accounting
    # ========================================================
    status_df = pd.DataFrame(
        statuses
    )

    print()
    print("=" * 100)
    print("STATUS SUMMARY =", task)
    print("=" * 100)

    if len(status_df) == 0:
        print("NO STATUS ROWS")

    else:
        if "status" not in status_df:
            raise RuntimeError(
                "status column missing from status table"
            )

        counts = (
            status_df[
                "status"
            ]
            .value_counts(
                dropna=False
            )
        )

        scored_n = int(
            (
                status_df["status"]
                ==
                "matched_scored"
            ).sum()
        )

        error_n = int(
            (
                status_df["status"]
                ==
                "unexpected_exception"
            ).sum()
        )

        skipped_n = int(
            len(status_df)
            -
            scored_n
            -
            error_n
        )

        print(
            "SCORED  =",
            scored_n,
        )

        print(
            "SKIPPED =",
            skipped_n,
        )

        print(
            "ERROR   =",
            error_n,
        )

        print()
        print("REASON COUNTS:")

        for reason, n in counts.items():
            print(
                f"  {reason}: {int(n)}"
            )

        if error_n > 0:
            print()
            print(
                "WARNING: unexpected ERROR clips exist. "
                "Full task was allowed to finish, but these "
                "clips must be audited before paper statistics."
            )

    print()
    print("=" * 100)
    print("TASK COMPLETE =", task)
    print("RAW SCANNED   =", scanned)
    print("PA TARGETS    =", len(target_keys))
    print("SCORED        =", len(rows))
    print("RESULT        =", out_csv)
    print("STATUS        =", status_csv)
    print("=" * 100)


if __name__ == "__main__":
    main()
