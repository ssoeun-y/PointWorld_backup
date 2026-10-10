#!/usr/bin/env python3

from pathlib import Path
from io import BytesIO
import tarfile

import numpy as np
from scipy.optimize import minimize

from diagnostics.build_acc_grasp_anchor_boundary_current10457 import (
    first_open_to_closed,
    acc_at_phase,
    acc_only_metric,
    poses_at_phase,
)

# ============================================================
# FIXED SMOKE CASE
# ============================================================

TARGET = "task-0001_episode_00010530-360:371"
TASK = "task-0001"

WDS = Path(
    "/home/soenchun/cse/PointWorld_workspace/"
    "datasets/behavior/wds_task0001/test"
)

OUT = Path(
    "diagnostics/results/"
    "close_anchor_matched_dose_can_smoke"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FIELDS = [
    "right_gripper_open",
    "left_gripper_open",
    "right_gripper_pose",
    "left_gripper_pose",
]

# We do NOT force the old boundary.
#
# First estimate how much Acc reduction is feasible
# while keeping the close pose anchored.
#
# Then use 80% of that available reduction as the
# common perturbation dose for BOTH variants.
#
# This leaves margin from the numerical feasibility floor.
DOSE_FRAC = 0.80

EPS_MONO = 1e-4

# ============================================================
# EXACT FAST TARGET READ
# ============================================================

def read_exact_target():

    found = {}

    for tar_path in sorted(
        WDS.glob("*.tar")
    ):

        with tarfile.open(
            tar_path,
            "r",
        ) as tf:

            for field in FIELDS:

                name = (
                    f"{TARGET}."
                    f"{field}.npy"
                )

                try:
                    member = tf.getmember(
                        name
                    )
                except KeyError:
                    continue

                f = tf.extractfile(
                    member
                )

                if f is None:
                    continue

                found[field] = np.load(
                    BytesIO(
                        f.read()
                    ),
                    allow_pickle=False,
                )

        if len(found) == len(FIELDS):

            print(
                "FOUND IN =",
                tar_path.name,
                flush=True,
            )

            break

    missing = (
        set(FIELDS)
        - set(found)
    )

    if missing:

        raise RuntimeError(
            f"Missing fields: "
            f"{sorted(missing)}"
        )

    return found


# ============================================================
# COMMON PHASE OBJECTIVE
#
# SAME objective for free and anchored perturbation.
# ============================================================

def phase_objective(
    phase,
    identity,
):

    shift = (
        phase - identity
    ) / float(
        len(identity) - 1
    )

    increments = np.diff(
        phase
    )

    return float(
        np.mean(
            shift ** 2
        )
        +
        0.02
        * np.mean(
            (
                increments
                - 1.0
            ) ** 2
        )
    )


# ============================================================
# INITIALIZATIONS
# ============================================================

def free_starts(T):

    identity = np.arange(
        T,
        dtype=np.float64,
    )

    starts = [
        identity.copy()
    ]

    u = np.linspace(
        0.0,
        1.0,
        T,
    )

    for gamma in [
        0.55,
        0.70,
        0.85,
        1.15,
        1.30,
        1.60,
    ]:

        p = (
            u ** gamma
        ) * float(
            T - 1
        )

        p[0] = 0.0
        p[-1] = float(
            T - 1
        )

        starts.append(p)

    return starts


def anchor_starts(
    T,
    t_close,
):

    identity = np.arange(
        T,
        dtype=np.float64,
    )

    starts = [
        identity.copy()
    ]

    gammas = [
        0.55,
        0.70,
        0.85,
        1.15,
        1.30,
        1.60,
    ]

    for gl in gammas:
        for gr in gammas:

            p = identity.copy()

            # Before close:
            # start and close are exact.
            if t_close > 1:

                u = np.linspace(
                    0.0,
                    1.0,
                    t_close + 1,
                )

                p[:t_close + 1] = (
                    u ** gl
                ) * float(
                    t_close
                )

            # After close:
            # close and final are exact.
            n = (
                T - t_close
            )

            if n > 2:

                u = np.linspace(
                    0.0,
                    1.0,
                    n,
                )

                p[t_close:] = (
                    float(
                        t_close
                    )
                    +
                    (
                        u ** gr
                    )
                    * float(
                        T
                        - 1
                        - t_close
                    )
                )

            p[0] = 0.0
            p[t_close] = float(
                t_close
            )
            p[-1] = float(
                T - 1
            )

            starts.append(p)

    return starts


# ============================================================
# BEST-FOUND ANCHOR MINIMUM RMS ACC
#
# This is only used to define a feasible matched dose.
# It is NOT called a mathematical global minimum.
# ============================================================

def find_anchor_acc_floor(
    right,
    left,
    t_close,
):

    T = len(right)

    identity = np.arange(
        T,
        dtype=np.float64,
    )

    free_idx = np.asarray(
        [
            i
            for i in range(
                1,
                T - 1,
            )
            if i != t_close
        ],
        dtype=int,
    )

    def unpack(x):

        p = identity.copy()

        p[free_idx] = np.asarray(
            x,
            dtype=np.float64,
        )

        p[0] = 0.0
        p[t_close] = float(
            t_close
        )
        p[-1] = float(
            T - 1
        )

        return p

    def acc(x):

        return float(
            acc_at_phase(
                right,
                left,
                unpack(x),
            )[
                "rms_acceleration"
            ]
        )

    constraints = [
        {
            "type": "ineq",
            "fun":
                lambda x:
                np.diff(
                    unpack(x)
                )
                - EPS_MONO,
        },
    ]

    best = None

    for p0 in anchor_starts(
        T,
        t_close,
    ):

        x0 = p0[
            free_idx
        ]

        r = minimize(
            acc,
            x0,
            method="SLSQP",
            bounds=[
                (
                    0.0,
                    float(T - 1),
                )
                for _ in free_idx
            ],
            constraints=
                constraints,
            options={
                "maxiter": 1000,
                "ftol": 1e-12,
                "disp": False,
            },
        )

        p = unpack(
            r.x
        )

        a = float(
            acc_at_phase(
                right,
                left,
                p,
            )[
                "rms_acceleration"
            ]
        )

        monotonic = bool(
            np.all(
                np.diff(p)
                >= -1e-9
            )
        )

        anchor_ok = bool(
            np.isclose(
                p[t_close],
                float(t_close),
                atol=1e-12,
                rtol=0.0,
            )
        )

        if not (
            monotonic
            and
            anchor_ok
        ):
            continue

        cand = {
            "phase": p,
            "acc": a,
            "optimizer_success":
                bool(r.success),
            "message":
                str(r.message),
        }

        if (
            best is None
            or
            cand["acc"]
            <
            best["acc"]
        ):
            best = cand

    if best is None:

        raise RuntimeError(
            "Could not estimate "
            "anchored feasible Acc floor"
        )

    return best


# ============================================================
# MATCHED RMS ACC OPTIMIZER
#
# mode = free
#     start/end fixed only.
#
# mode = anchor
#     start/end + t_close fixed.
#
# SAME:
#     target RMS Acc
#     objective
#     optimizer
#     spatial path representation
# ============================================================

def optimize_matched(
    right,
    left,
    t_close,
    target_acc,
    mode,
):

    T = len(right)

    identity = np.arange(
        T,
        dtype=np.float64,
    )

    if mode == "free":

        free_idx = np.arange(
            1,
            T - 1,
            dtype=int,
        )

        starts = free_starts(
            T
        )

    elif mode == "anchor":

        free_idx = np.asarray(
            [
                i
                for i in range(
                    1,
                    T - 1,
                )
                if i != t_close
            ],
            dtype=int,
        )

        starts = anchor_starts(
            T,
            t_close,
        )

    else:

        raise ValueError(mode)

    def unpack(x):

        p = identity.copy()

        p[free_idx] = np.asarray(
            x,
            dtype=np.float64,
        )

        # start/end exact
        p[0] = 0.0
        p[-1] = float(
            T - 1
        )

        # ONLY anchor variant fixes close pose.
        if mode == "anchor":
            p[t_close] = float(
                t_close
            )

        return p

    def current_acc(x):

        return float(
            acc_at_phase(
                right,
                left,
                unpack(x),
            )[
                "rms_acceleration"
            ]
        )

    def objective(x):

        return phase_objective(
            unpack(x),
            identity,
        )

    constraints = [
        # Same monotonic temporal reparameterization.
        {
            "type": "ineq",
            "fun":
                lambda x:
                np.diff(
                    unpack(x)
                )
                - EPS_MONO,
        },

        # EXACT same RMS Acc target.
        {
            "type": "eq",
            "fun":
                lambda x:
                current_acc(x)
                - target_acc,
        },
    ]

    best = None

    for p0 in starts:

        x0 = p0[
            free_idx
        ]

        r = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[
                (
                    0.0,
                    float(T - 1),
                )
                for _ in free_idx
            ],
            constraints=
                constraints,
            options={
                "maxiter": 1200,
                "ftol": 1e-12,
                "disp": False,
            },
        )

        p = unpack(
            r.x
        )

        a = float(
            acc_at_phase(
                right,
                left,
                p,
            )[
                "rms_acceleration"
            ]
        )

        rel_match = (
            abs(
                a - target_acc
            )
            /
            max(
                abs(target_acc),
                1e-12,
            )
        )

        monotonic = bool(
            np.all(
                np.diff(p)
                >= -1e-9
            )
        )

        anchors_ok = bool(
            np.isclose(
                p[0],
                0.0,
                atol=1e-12,
                rtol=0.0,
            )
            and
            np.isclose(
                p[-1],
                float(T - 1),
                atol=1e-12,
                rtol=0.0,
            )
        )

        if mode == "anchor":

            anchors_ok = (
                anchors_ok
                and
                np.isclose(
                    p[t_close],
                    float(t_close),
                    atol=1e-12,
                    rtol=0.0,
                )
            )

        matched = bool(
            monotonic
            and
            anchors_ok
            and
            rel_match
            <= 1e-4
        )

        cand = {
            "success": matched,
            "phase": p,
            "acc": a,
            "relative_acc_match_error":
                rel_match,
            "objective":
                float(
                    objective(
                        r.x
                    )
                ),
            "optimizer_success":
                bool(r.success),
            "message":
                str(r.message),
        }

        if best is None:

            best = cand

        elif (
            cand["success"]
            and
            not best["success"]
        ):

            best = cand

        elif (
            cand["success"]
            ==
            best["success"]
            and
            cand["objective"]
            <
            best["objective"]
        ):

            best = cand

    return best


# ============================================================
# AUDIT HELPERS
# ============================================================

def phase_stats(
    phase,
):

    identity = np.arange(
        len(phase),
        dtype=np.float64,
    )

    d = (
        np.asarray(
            phase,
            dtype=np.float64,
        )
        - identity
    )

    return {
        "phase_rms_shift_frames":
            float(
                np.sqrt(
                    np.mean(
                        d ** 2
                    )
                )
            ),

        "phase_max_shift_frames":
            float(
                np.max(
                    np.abs(d)
                )
            ),
    }


def pose_audit(
    right,
    left,
    phase,
    t_close,
    close_side,
):

    rp, lp = poses_at_phase(
        right,
        left,
        phase,
    )

    start_err = max(
        float(
            np.max(
                np.abs(
                    rp[0, :3]
                    - right[0, :3]
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    lp[0, :3]
                    - left[0, :3]
                )
            )
        ),
    )

    end_err = max(
        float(
            np.max(
                np.abs(
                    rp[-1, :3]
                    - right[-1, :3]
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    lp[-1, :3]
                    - left[-1, :3]
                )
            )
        ),
    )

    if close_side == "right":

        close_err = float(
            np.linalg.norm(
                rp[
                    t_close,
                    :3
                ]
                -
                right[
                    t_close,
                    :3
                ]
            )
        )

    elif close_side == "left":

        close_err = float(
            np.linalg.norm(
                lp[
                    t_close,
                    :3
                ]
                -
                left[
                    t_close,
                    :3
                ]
            )
        )

    else:

        close_err = max(
            float(
                np.linalg.norm(
                    rp[
                        t_close,
                        :3
                    ]
                    -
                    right[
                        t_close,
                        :3
                    ]
                )
            ),
            float(
                np.linalg.norm(
                    lp[
                        t_close,
                        :3
                    ]
                    -
                    left[
                        t_close,
                        :3
                    ]
                )
            ),
        )

    return {
        "start_error_m":
            start_err,

        "end_error_m":
            end_err,

        "active_close_pose_shift_m":
            close_err,
    }


# ============================================================
# MAIN
# ============================================================

sample = read_exact_target()

right_open = sample[
    "right_gripper_open"
].reshape(-1)

left_open = sample[
    "left_gripper_open"
].reshape(-1)

right = np.asarray(
    sample[
        "right_gripper_pose"
    ],
    dtype=np.float64,
)

left = np.asarray(
    sample[
        "left_gripper_pose"
    ],
    dtype=np.float64,
)

T = len(right)

close_info = (
    first_open_to_closed(
        right_open,
        left_open,
        T,
    )
)

print()
print("=" * 78)
print("MATCHED-DOSE CLOSE-ANCHOR CAUSAL SMOKE")
print("=" * 78)
print("TARGET =", TARGET)

print(
    "RIGHT_OPEN =",
    right_open.astype(int).tolist()
)

print(
    "LEFT_OPEN  =",
    left_open.astype(int).tolist()
)

if close_info is None:

    raise RuntimeError(
        "No open->closed transition"
    )

t_close, close_side = (
    close_info
)

print(
    f"CLOSE = {close_side} | "
    f"t_close={t_close} | "
    f"human frame={t_close + 1}"
)

original = acc_only_metric(
    right,
    left,
)

original_acc = float(
    original[
        "rms_acceleration"
    ]
)

print()
print(
    "ORIGINAL RMS ACC SI =",
    f"{original_acc * 25.0:.6f}"
)

# ------------------------------------------------------------
# 1. Find best-found anchor Acc floor
# ------------------------------------------------------------

floor = find_anchor_acc_floor(
    right,
    left,
    t_close,
)

anchor_floor = float(
    floor["acc"]
)

print(
    "BEST-FOUND ANCHOR MIN ACC SI =",
    f"{anchor_floor * 25.0:.6f}"
)

if not (
    anchor_floor
    <
    original_acc
):

    raise RuntimeError(
        "Anchor optimization cannot "
        "reduce RMS Acc"
    )

# ------------------------------------------------------------
# 2. Define SAME dose for both methods
#
# 80% of the reduction available to the anchored variant.
# ------------------------------------------------------------

target_acc = (
    original_acc
    -
    DOSE_FRAC
    * (
        original_acc
        - anchor_floor
    )
)

print(
    f"MATCHED DOSE = {DOSE_FRAC:.2f} "
    "of anchored feasible reduction"
)

print(
    "COMMON TARGET RMS ACC SI =",
    f"{target_acc * 25.0:.6f}"
)

# ------------------------------------------------------------
# 3. Optimize both variants to EXACT SAME Acc
# ------------------------------------------------------------

free = optimize_matched(
    right,
    left,
    t_close,
    target_acc,
    mode="free",
)

anchor = optimize_matched(
    right,
    left,
    t_close,
    target_acc,
    mode="anchor",
)

if not free["success"]:

    raise RuntimeError(
        "FREE matched-dose optimization failed | "
        f"acc={free['acc']*25.0:.6f} | "
        f"relerr={free['relative_acc_match_error']}"
    )

if not anchor["success"]:

    raise RuntimeError(
        "ANCHOR matched-dose optimization failed | "
        f"acc={anchor['acc']*25.0:.6f} | "
        f"relerr={anchor['relative_acc_match_error']}"
    )

# ------------------------------------------------------------
# 4. Strong fairness audits
# ------------------------------------------------------------

free_stats = phase_stats(
    free["phase"]
)

anchor_stats = phase_stats(
    anchor["phase"]
)

free_pose = pose_audit(
    right,
    left,
    free["phase"],
    t_close,
    close_side,
)

anchor_pose = pose_audit(
    right,
    left,
    anchor["phase"],
    t_close,
    close_side,
)

acc_gap_si = (
    abs(
        free["acc"]
        - anchor["acc"]
    )
    * 25.0
)

print()
print("=" * 78)
print("FAIRNESS AUDIT")
print("=" * 78)

print(
    "FREE RMS ACC SI   =",
    f"{free['acc'] * 25.0:.9f}"
)

print(
    "ANCHOR RMS ACC SI =",
    f"{anchor['acc'] * 25.0:.9f}"
)

print(
    "ABS ACC DIFFERENCE SI =",
    f"{acc_gap_si:.9e}"
)

print()
print(
    "FREE PHASE =",
    np.round(
        free["phase"],
        4,
    ).tolist()
)

print(
    "ANCHOR PHASE =",
    np.round(
        anchor["phase"],
        4,
    ).tolist()
)

print()
print(
    "FREE phase RMS shift frames =",
    f"{free_stats['phase_rms_shift_frames']:.6f}"
)

print(
    "ANCHOR phase RMS shift frames =",
    f"{anchor_stats['phase_rms_shift_frames']:.6f}"
)

print(
    "FREE phase MAX shift frames =",
    f"{free_stats['phase_max_shift_frames']:.6f}"
)

print(
    "ANCHOR phase MAX shift frames =",
    f"{anchor_stats['phase_max_shift_frames']:.6f}"
)

print()
print(
    "FREE active close pose shift mm =",
    f"{free_pose['active_close_pose_shift_m'] * 1000.0:.6f}"
)

print(
    "ANCHOR active close pose shift mm =",
    f"{anchor_pose['active_close_pose_shift_m'] * 1000.0:.6f}"
)

print(
    "FREE start/end error mm =",
    f"{free_pose['start_error_m']*1000.0:.9f}",
    "/",
    f"{free_pose['end_error_m']*1000.0:.9f}",
)

print(
    "ANCHOR start/end error mm =",
    f"{anchor_pose['start_error_m']*1000.0:.9f}",
    "/",
    f"{anchor_pose['end_error_m']*1000.0:.9f}",
)

# ------------------------------------------------------------
# Hard causal invariants
# ------------------------------------------------------------

if (
    anchor_pose[
        "active_close_pose_shift_m"
    ]
    > 1e-10
):

    raise RuntimeError(
        "ANCHOR close pose is NOT exact"
    )

if (
    free_pose[
        "start_error_m"
    ]
    > 1e-10
    or
    free_pose[
        "end_error_m"
    ]
    > 1e-10
    or
    anchor_pose[
        "start_error_m"
    ]
    > 1e-10
    or
    anchor_pose[
        "end_error_m"
    ]
    > 1e-10
):

    raise RuntimeError(
        "Start/end invariant broken"
    )

# Same dynamics dose must be extremely close.
if acc_gap_si > 1e-4:

    raise RuntimeError(
        "Matched RMS Acc dose is not close enough"
    )

# ------------------------------------------------------------
# 5. Save exact phases for next inference smoke
# ------------------------------------------------------------

np.savez_compressed(
    OUT / "free_matched.npz",

    clip=np.array(
        TARGET
    ),

    task=np.array(
        TASK
    ),

    variant=np.array(
        "free_matched"
    ),

    t_close=np.array(
        t_close
    ),

    close_side=np.array(
        close_side
    ),

    phase=np.asarray(
        free["phase"],
        dtype=np.float64,
    ),

    original_acc=np.array(
        original_acc
    ),

    target_acc=np.array(
        target_acc
    ),

    achieved_acc=np.array(
        free["acc"]
    ),
)

np.savez_compressed(
    OUT / "close_anchor_matched.npz",

    clip=np.array(
        TARGET
    ),

    task=np.array(
        TASK
    ),

    variant=np.array(
        "close_anchor_matched"
    ),

    t_close=np.array(
        t_close
    ),

    close_side=np.array(
        close_side
    ),

    phase=np.asarray(
        anchor["phase"],
        dtype=np.float64,
    ),

    original_acc=np.array(
        original_acc
    ),

    target_acc=np.array(
        target_acc
    ),

    achieved_acc=np.array(
        anchor["acc"]
    ),
)

print()
print("=" * 78)
print("SMOKE PASSED")
print("=" * 78)

print(
    "Saved:",
    OUT / "free_matched.npz"
)

print(
    "Saved:",
    OUT / "close_anchor_matched.npz"
)

print()
print(
    "NEXT CAUSAL COMPARISON:"
)

print(
    "Original vs Free-matched vs Close-anchor-matched"
)

print(
    "Metrics: PA-MPJPE / Goal MPJPE / post-close object-gripper drift"
)

print("=" * 78)
