#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from arguments import parse_args
from dataset_components.dataloader import build_dataloader
from diagnostics.run_established_action_metrics_5task import (
    choose_dominant_eef,
    rms_acceleration,
)


# ============================================================
# FIXED PROTOCOL
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

ACC_CSV = Path(
    "diagnostics/results/current_behavior_established_acc_10457/"
    "per_sample_current_established_acc_10457.csv"
)

PA_ROOT = Path(
    "diagnostics/results/object_pa_mpjpe_acc_5task_current10457"
)

OUT_ROOT = Path(
    "diagnostics/results/"
    "acc_grasp_anchor_boundary_radio_quick"
)

# Established GLOBAL RMS-Acceleration boundary.
# Native PointWorld-step unit.
ACC_BOUNDARY_NATIVE = 0.009832167363499603
ACC_BOUNDARY_SI = ACC_BOUNDARY_NATIVE * 25.0

# Same tolerance convention as prior correction.
BOUNDARY_TOL = 1.002


# ============================================================
# Helpers
# ============================================================

def normalize_key(x):
    if isinstance(x, (list, tuple)):
        x = x[0]

    if isinstance(x, bytes):
        x = x.decode()

    try:
        import torch
        if torch.is_tensor(x):
            if x.numel() == 1:
                x = x.detach().cpu().item()
    except Exception:
        pass

    return str(x)


def acc_only_metric(
    right_pose,
    left_pose,
):
    """
    EXACT RMS-Acc branch of the established metric.

    NO jerk/MSJ is computed or used.
    """
    (
        dominant_hand,
        xyz,
        _,
        _,
    ) = choose_dominant_eef(
        np.asarray(right_pose, np.float64),
        np.asarray(left_pose, np.float64),
    )

    return {
        "dominant_hand": dominant_hand,
        "rms_acceleration":
            float(rms_acceleration(xyz)),
    }


def interpolate_sequence(
    arr,
    phase,
):
    arr = np.asarray(
        arr,
        dtype=np.float64,
    )

    T = arr.shape[0]

    phase = np.clip(
        np.asarray(
            phase,
            dtype=np.float64,
        ),
        0.0,
        float(T - 1),
    )

    lo = np.floor(phase).astype(int)
    hi = np.ceil(phase).astype(int)

    alpha = (
        phase - lo
    ).reshape(
        (T,)
        + (1,) * (arr.ndim - 1)
    )

    out = (
        (1.0 - alpha) * arr[lo]
        + alpha * arr[hi]
    )

    # Hard goal preservation.
    out[0] = arr[0]
    out[-1] = arr[-1]

    return out


def poses_at_phase(
    right_pose,
    left_pose,
    phase,
):
    rp = np.asarray(
        right_pose,
        np.float64,
    ).copy()

    lp = np.asarray(
        left_pose,
        np.float64,
    ).copy()

    # Established Acc uses XYZ only.
    rp[:, :3] = interpolate_sequence(
        rp[:, :3],
        phase,
    )

    lp[:, :3] = interpolate_sequence(
        lp[:, :3],
        phase,
    )

    return rp, lp


def acc_at_phase(
    right_pose,
    left_pose,
    phase,
):
    rp, lp = poses_at_phase(
        right_pose,
        left_pose,
        phase,
    )

    return acc_only_metric(
        rp,
        lp,
    )


# ============================================================
# Goal-preserved Acc-only optimization
# ============================================================


def first_open_to_closed(
    right_gripper_open,
    left_gripper_open,
    T,
):
    """
    BEHAVIOR semantics:
        1 / True  = open
        0 / False = closed

    Return:
        (t_close, side)

    t_close:
        Python 0-based timestep of the FIRST
        open -> closed transition across either hand.

    side:
        "right", "left", or "both"
    """

    def to_open_bool(x, name):

        if hasattr(x, "detach"):
            x = (
                x.detach()
                .cpu()
                .numpy()
            )

        x = np.asarray(x)

        # Remove batch=1 dimensions safely.
        while (
            x.ndim > 1
            and
            x.shape[0] == 1
        ):
            x = x[0]

        x = x.reshape(-1)

        if len(x) != T:
            raise RuntimeError(
                f"{name}: gripper_open length "
                f"{len(x)} != T={T}"
            )

        # 1=open, 0=closed
        return (
            x.astype(np.float64)
            > 0.5
        )

    right_open = to_open_bool(
        right_gripper_open,
        "right",
    )

    left_open = to_open_bool(
        left_gripper_open,
        "left",
    )

    events = []

    for side, is_open in [
        ("right", right_open),
        ("left", left_open),
    ]:

        idx = (
            np.flatnonzero(
                is_open[:-1]
                &
                (~is_open[1:])
            )
            + 1
        )

        if len(idx) > 0:
            events.append(
                (
                    int(idx[0]),
                    side,
                )
            )

    if not events:
        return None

    first_t = min(
        t for t, _ in events
    )

    first_sides = [
        side
        for t, side in events
        if t == first_t
    ]

    if len(first_sides) == 2:
        side = "both"
    else:
        side = first_sides[0]

    return (
        int(first_t),
        side,
    )


def optimize_phase_acc_only_grasp_anchor(
    right_pose,
    left_pose,
    tau_acc,
    t_close,
):
    """
    Same spatial path / temporal phase optimization
    as the validated Acc-only boundary method.

    NEW invariant:

        phase[0 : t_close + 1]
        ==
        identity[0 : t_close + 1]

    Therefore:

        start -> t_close : EXACT Original timing
        t_close          : EXACT Original robot pose
        t_close -> end    : temporal retiming allowed
        final             : EXACT Original endpoint

    RMS Acc boundary is unchanged.
    """

    T = len(right_pose)

    if T < 5:
        return {
            "success": False,
            "reason": "TOO_SHORT",
        }

    t_close = int(t_close)

    if not (
        1 <= t_close < T - 1
    ):
        return {
            "success": False,
            "reason":
                f"INVALID_T_CLOSE_{t_close}",
        }

    identity = np.arange(
        T,
        dtype=np.float64,
    )

    # Free phase variables are:
    #
    # t_close+1 ... T-2
    #
    # t_close itself is fixed.
    # final T-1 is also fixed.
    n_free = (
        T
        - t_close
        - 2
    )

    # Nothing remains to retime.
    if n_free <= 0:

        phase = identity.copy()

        corr = acc_at_phase(
            right_pose,
            left_pose,
            phase,
        )

        corr_acc = float(
            corr[
                "rms_acceleration"
            ]
        )

        feasible = bool(
            corr_acc
            <=
            tau_acc
            * BOUNDARY_TOL
        )

        return {
            "success": feasible,
            "optimizer_success": True,
            "optimizer_message":
                "NO_POST_CLOSE_FREE_TIMESTEPS",
            "objective": 0.0,
            "phase": phase,
            "corrected_acc":
                corr_acc,
            "t_close":
                int(t_close),
        }

    def unpack(x):

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        phase = np.concatenate([
            # EVERYTHING through close
            # remains exactly Original.
            identity[
                :t_close + 1
            ],

            # Post-close free timings.
            x,

            # Final endpoint exact.
            [
                float(T - 1)
            ],
        ])

        return phase

    def objective(x):

        phase = unpack(x)

        shift = (
            phase
            - identity
        ) / float(T - 1)

        increments = np.diff(
            phase
        )

        return float(
            np.mean(
                shift ** 2
            )
            +
            0.02
            *
            np.mean(
                (
                    increments
                    - 1.0
                ) ** 2
            )
        )

    def current_acc(x):

        return float(
            acc_at_phase(
                right_pose,
                left_pose,
                unpack(x),
            )[
                "rms_acceleration"
            ]
        )

    constraints = [

        # Monotonic temporal phase.
        {
            "type": "ineq",
            "fun":
                lambda x:
                np.diff(
                    unpack(x)
                )
                - 1e-4,
        },

        # SAME fixed RMS Acc boundary.
        {
            "type": "ineq",
            "fun":
                lambda x:
                tau_acc
                - current_acc(x),
        },
    ]

    # --------------------------------------------------------
    # Multi-start initializations.
    #
    # Same idea as existing boundary optimizer,
    # but only over post-close interval.
    # --------------------------------------------------------

    suffix_len = (
        T - t_close
    )

    u = np.linspace(
        0.0,
        1.0,
        suffix_len,
    )

    starts = []

    # Identity initialization.
    starts.append(
        identity[
            t_close + 1:
            -1
        ].copy()
    )

    # Smoothstep initialization.
    smooth = (
        10 * u**3
        - 15 * u**4
        + 6 * u**5
    )

    smooth = (
        float(t_close)
        +
        smooth
        *
        float(
            T - 1 - t_close
        )
    )

    starts.append(
        smooth[1:-1]
    )

    # Early / late redistributions.
    for gamma in [
        0.70,
        0.85,
        1.15,
        1.30,
    ]:

        p = (
            float(t_close)
            +
            (
                u ** gamma
            )
            *
            float(
                T - 1 - t_close
            )
        )

        starts.append(
            p[1:-1]
        )

    best = None

    for x0 in starts:

        result = minimize(
            objective,
            x0,
            method="SLSQP",

            bounds=[
                (
                    float(t_close),
                    float(T - 1),
                )
                for _ in range(
                    n_free
                )
            ],

            constraints=
                constraints,

            options={
                "maxiter": 600,
                "ftol": 1e-10,
                "disp": False,
            },
        )

        phase = unpack(
            result.x
        )

        corr = acc_at_phase(
            right_pose,
            left_pose,
            phase,
        )

        corr_acc = float(
            corr[
                "rms_acceleration"
            ]
        )

        prefix_exact = bool(
            np.allclose(
                phase[
                    :t_close + 1
                ],
                identity[
                    :t_close + 1
                ],
                atol=1e-12,
                rtol=0.0,
            )
        )

        final_exact = bool(
            np.isclose(
                phase[-1],
                float(T - 1),
                atol=1e-12,
                rtol=0.0,
            )
        )

        monotonic = bool(
            np.all(
                np.diff(phase)
                >= -1e-9
            )
        )

        boundary_ok = bool(
            corr_acc
            <=
            tau_acc
            * BOUNDARY_TOL
        )

        feasible = bool(
            prefix_exact
            and
            final_exact
            and
            monotonic
            and
            boundary_ok
        )

        candidate = {
            "success":
                feasible,

            "optimizer_success":
                bool(
                    result.success
                ),

            "optimizer_message":
                str(
                    result.message
                ),

            "objective":
                float(
                    objective(
                        result.x
                    )
                ),

            "phase":
                phase,

            "corrected_acc":
                corr_acc,

            "t_close":
                int(t_close),
        }

        if best is None:

            best = candidate

        elif (
            candidate[
                "success"
            ]
            and
            not best[
                "success"
            ]
        ):

            best = candidate

        elif (
            candidate[
                "success"
            ]
            ==
            best[
                "success"
            ]
            and
            candidate[
                "objective"
            ]
            <
            best[
                "objective"
            ]
        ):

            best = candidate

    return best


# ============================================================
# Current PA-eligible clip set
# ============================================================

def get_pa_eligible_clips():
    clips = set()

    for p in PA_ROOT.rglob("*.csv"):
        try:
            head = pd.read_csv(
                p,
                nrows=1,
            )
        except Exception:
            continue

        if not {
            "clip",
            "eligible_moving_objects",
        }.issubset(head.columns):
            continue

        d = pd.read_csv(
            p,
            usecols=[
                "clip",
                "eligible_moving_objects",
            ],
        )

        d = d[
            d["eligible_moving_objects"]
            > 0
        ]

        clips.update(
            d["clip"]
            .astype(str)
            .tolist()
        )

    if len(clips) != 7947:
        raise RuntimeError(
            "PA eligible clip audit failed: "
            f"expected 7947, got {len(clips)}"
        )

    return clips


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--tasks",
        default=(
            "task-0000,task-0001,"
            "task-0006,task-0008,"
            "task-0010"
        ),
    )

    ap.add_argument(
        "--smoke",
        action="store_true",
    )

    args_local = ap.parse_args()

    tasks = [
        x.strip()
        for x in args_local.tasks.split(",")
        if x.strip()
    ]

    acc_df = pd.read_csv(
        ACC_CSV
    )

    if len(acc_df) != 10457:
        raise RuntimeError(
            f"ACC CSV expected 10457, "
            f"got {len(acc_df)}"
        )

    if acc_df["clip"].duplicated().any():
        raise RuntimeError(
            "duplicate clip in ACC CSV"
        )

    eligible = (
        get_pa_eligible_clips()
    )

    # ONLY current rigid-object eligible +
    # RMS Acc outside fixed boundary.
    targets = acc_df[
        acc_df["clip"]
        .astype(str)
        .isin(eligible)
        &
        (
            acc_df[
                "rms_acceleration"
            ]
            >
            ACC_BOUNDARY_NATIVE
        )
    ].copy()

    target_lookup = (
        targets
        .set_index("clip")
        .to_dict("index")
    )

    out = (
        OUT_ROOT / "smoke"
        if args_local.smoke
        else OUT_ROOT
    )

    corr_dir = (
        out / "corrected_actions"
    )

    corr_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("ACC-ONLY GOAL-PRESERVED CORRECTION")
    print("ACC boundary native =", ACC_BOUNDARY_NATIVE)
    print("ACC boundary SI     =", ACC_BOUNDARY_SI)
    print("PA eligible         =", len(eligible))
    print("Acc-out targets     =", len(targets))
    print()
    print(
        targets.groupby("task")
        .size()
        .to_string()
    )
    print("=" * 78)

    all_rows = []

    for task in tasks:

        task_targets = {
            str(x)
            for x in targets.loc[
                targets["task"] == task,
                "clip",
            ]
        }

        # QUICK RADIO TEST:
        # only the known radio_89 representative clip.
        QUICK_CLIP = "task-0000_episode_00000010-200:211"

        if task == "task-0000":
            task_targets = (
                task_targets
                &
                {QUICK_CLIP}
            )
        else:
            task_targets = set()

        print()
        print(
            f"===== {task} "
            f"targets={len(task_targets)} ====="
        )

        if not task_targets:
            continue

        pw = parse_args(
            skip_command_line=True
        )

        pw.seed = 42
        pw.domains = ["behavior"]
        pw.data_dirs = [
            TASK_DIRS[task]
        ]
        pw.batch_size = 1
        pw.num_workers = 0
        pw.eval_num_workers = 0

        loader, _ = build_dataloader(
            args=pw,
            mode="test",
            rank=0,
            world_size=1,
            override_splits="test",
            force_resampled_eval=False,
        )

        found = set()

        for batch in loader:

            key = normalize_key(
                batch["__key__"]
            )

            if key not in task_targets:
                continue

            found.add(key)

            right = (
                batch[
                    "right_gripper_pose"
                ][0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            left = (
                batch[
                    "left_gripper_pose"
                ][0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            original = (
                acc_only_metric(
                    right,
                    left,
                )
            )

            csv_acc = float(
                target_lookup[key][
                    "rms_acceleration"
                ]
            )

            # Critical audit:
            # SAME current 10,457 Acc definition.
            if not np.isclose(
                original[
                    "rms_acceleration"
                ],
                csv_acc,
                atol=1e-10,
                rtol=1e-8,
            ):
                raise RuntimeError(
                    f"{key}: BASE ACC AUDIT FAIL "
                    f"runtime="
                    f"{original['rms_acceleration']} "
                    f"csv={csv_acc}"
                )

            # ====================================================
            # Grasp-anchor:
            # find actual first open -> closed transition.
            # Do NOT hard-code frame 5.
            # ====================================================

            t_close = None
            close_side = "none"

            def _print_gripper_state(name, x):
                if hasattr(x, "detach"):
                    x = x.detach().cpu().numpy()

                x = np.asarray(x).reshape(-1)

                print(
                    f"[GRIPPER] {key} "
                    f"{name}="
                    f"{x.astype(int).tolist()}",
                    flush=True,
                )

            _print_gripper_state(
                "RIGHT_OPEN",
                batch["right_gripper_open"],
            )

            _print_gripper_state(
                "LEFT_OPEN",
                batch["left_gripper_open"],
            )

            close_info = (
                first_open_to_closed(
                    batch[
                        "right_gripper_open"
                    ],
                    batch[
                        "left_gripper_open"
                    ],
                    len(right),
                )
            )

            if close_info is None:

                result = {
                    "success": False,
                    "optimizer_success":
                        False,
                    "optimizer_message":
                        "NO_OPEN_TO_CLOSED_TRANSITION",
                    "reason":
                        "NO_OPEN_TO_CLOSED_TRANSITION",
                }

            else:

                (
                    t_close,
                    close_side,
                ) = close_info

                print(
                    f"[CLOSE] {key} "
                    f"side={close_side} "
                    f"t_close={t_close} "
                    f"(human frame={t_close + 1})",
                    flush=True,
                )

                result = (
                    optimize_phase_acc_only_grasp_anchor(
                        right,
                        left,
                        ACC_BOUNDARY_NATIVE,
                        t_close,
                    )
                )

            row = {
                "clip": key,
                "task": task,
                "dominant_hand":
                    original[
                        "dominant_hand"
                    ],
                "t_close":
                    (
                        int(t_close)
                        if t_close is not None
                        else np.nan
                    ),

                "t_close_1based":
                    (
                        int(t_close) + 1
                        if t_close is not None
                        else np.nan
                    ),

                "close_side":
                    str(close_side),

                "original_acc":
                    original[
                        "rms_acceleration"
                    ],
                "original_acc_si":
                    original[
                        "rms_acceleration"
                    ] * 25.0,
                "acc_boundary":
                    ACC_BOUNDARY_NATIVE,
                "acc_boundary_si":
                    ACC_BOUNDARY_SI,
                "success":
                    bool(
                        result[
                            "success"
                        ]
                    ),
                "optimizer_success":
                    bool(
                        result.get(
                            "optimizer_success",
                            False,
                        )
                    ),
                "optimizer_message":
                    str(
                        result.get(
                            "optimizer_message",
                            "",
                        )
                    ),
            }

            if result["success"]:

                phase = result["phase"]

                # =================================================
                # HARD GRASP-ANCHOR AUDIT
                # =================================================

                identity_phase = np.arange(
                    len(phase),
                    dtype=np.float64,
                )

                if t_close is None:
                    raise RuntimeError(
                        f"{key}: success with "
                        "t_close=None"
                    )

                if not np.allclose(
                    phase[
                        :t_close + 1
                    ],
                    identity_phase[
                        :t_close + 1
                    ],
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        f"{key}: PRE-CLOSE "
                        f"PHASE CHANGED | "
                        f"t_close={t_close} | "
                        f"phase={phase}"
                    )

                if not np.isclose(
                    phase[t_close],
                    float(t_close),
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        f"{key}: CLOSE "
                        "ANCHOR BROKEN"
                    )

                if not np.isclose(
                    phase[-1],
                    float(
                        len(phase) - 1
                    ),
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        f"{key}: FINAL "
                        "ANCHOR BROKEN"
                    )

                rp, lp = poses_at_phase(
                    right,
                    left,
                    phase,
                )

                # Close pose must remain exactly Original.
                close_pose_err = max(
                    float(
                        np.max(
                            np.abs(
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
                    ),
                    float(
                        np.max(
                            np.abs(
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
                    ),
                )

                if close_pose_err > 1e-10:
                    raise RuntimeError(
                        f"{key}: close pose changed | "
                        f"t_close={t_close} "
                        f"err={close_pose_err}"
                    )

                row[
                    "close_pose_error"
                ] = close_pose_err

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

                corr_acc = float(
                    result[
                        "corrected_acc"
                    ]
                )

                row.update({
                    "corrected_acc":
                        corr_acc,
                    "corrected_acc_si":
                        corr_acc * 25.0,
                    "acc_ratio":
                        corr_acc
                        /
                        original[
                            "rms_acceleration"
                        ],
                    "start_error":
                        start_err,
                    "endpoint_error":
                        end_err,
                    "max_phase_shift_frames":
                        float(
                            np.max(
                                np.abs(
                                    phase
                                    -
                                    np.arange(
                                        len(phase)
                                    )
                                )
                            )
                        ),
                    "objective":
                        float(
                            result[
                                "objective"
                            ]
                        ),
                })

                safe = (
                    key
                    .replace("/", "_")
                    .replace(":", "__")
                )

                np.savez_compressed(
                    corr_dir
                    / f"{safe}.npz",

                    clip=np.array(key),
                    task=np.array(task),
                    phase=np.asarray(
                        phase,
                        np.float32,
                    ),
                    original_acc=np.array(
                        original[
                            "rms_acceleration"
                        ]
                    ),
                    corrected_acc=np.array(
                        corr_acc
                    ),
                    acc_boundary=np.array(
                        ACC_BOUNDARY_NATIVE
                    ),
                )

            all_rows.append(row)

            print(
                f"[{task}] "
                f"{len(found)}/{len(task_targets)} "
                f"base={original['rms_acceleration']*25:.6f} "
                f"success={result['success']} "
                f"corr="
                f"{result.get('corrected_acc', np.nan)*25:.6f}"
            )

            if (
                len(found)
                == len(task_targets)
            ):
                break

        missing = (
            task_targets - found
        )

        if missing:
            raise RuntimeError(
                f"{task}: missing "
                f"{len(missing)} targets"
            )

    result_df = pd.DataFrame(
        all_rows
    )

    result_df.to_csv(
        out / "correction_results.csv",
        index=False,
    )

    if len(result_df):

        ok = result_df[
            result_df["success"]
            == True
        ]

        print()
        print("=" * 78)
        print("FINAL AUDIT")
        print("N =", len(result_df))
        print(
            "SUCCESS =",
            len(ok),
            "/",
            len(result_df),
        )

        if len(ok):
            print(
                "original Acc SI median =",
                ok["original_acc_si"]
                .median(),
            )

            print(
                "corrected Acc SI median =",
                ok["corrected_acc_si"]
                .median(),
            )

            print(
                "max corrected / boundary =",
                (
                    ok["corrected_acc"]
                    /
                    ACC_BOUNDARY_NATIVE
                ).max(),
            )

            print(
                "max start error =",
                ok["start_error"].max(),
            )

            print(
                "max endpoint error =",
                ok["endpoint_error"].max(),
            )

        print(
            "SAVED =",
            out
            / "correction_results.csv",
        )
        print("=" * 78)


if __name__ == "__main__":
    main()
