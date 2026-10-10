#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import mannwhitneyu

from arguments import parse_args
from dataset_components.dataloader import build_dataloader

# ============================================================
# EXACT metric implementation used for the original 7,987.
# NO reimplementation.
# ============================================================

from diagnostics.run_established_action_metrics_5task import (
    DT,
    choose_dominant_eef,
    established_metrics_for_clip,
    pointworld_velocity_acceleration,
    calculate_jerk_profile,
)


SEED = 20260902

MAIN = Path(
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)

OUT = Path(
    "diagnostics/results/"
    "goal_preserved_range_correction_v2_7987"
)

CORRECTED_DIR = OUT / "corrected_actions"
PLOT_DIR = OUT / "representative_local_plots"

DATA_ROOTS = {
    "task-0000":
        "/home/soenchun/cse/PointWorld_workspace/"
        "datasets/behavior/wds_task0000",

    "task-0001":
        "/home/soenchun/cse/PointWorld_workspace/"
        "PointWorld/diagnostics/data/"
        "pointworld_behavior_task0001_restored/behavior/wds",

    "multitask":
        "/home/soenchun/cse/PointWorld_workspace/"
        "PointWorld/diagnostics/data/"
        "pointworld_behavior_multitask_restored/behavior/wds",
}


# ============================================================
# Generic helpers
# ============================================================

def to_numpy(x):
    try:
        import torch
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def unbatch(x):
    x = to_numpy(x)
    if x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]
    return x


def normalize_key(x):
    try:
        import torch
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.reshape(-1)[0]

    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]

    if isinstance(x, bytes):
        x = x.decode("utf-8")

    return str(x)


def stable_number(text):
    h = hashlib.sha256(
        f"{SEED}|{text}".encode()
    ).digest()

    return int.from_bytes(
        h[:8], "big"
    ) / 2**64


def make_split(df):
    result = pd.Series(
        index=df.index,
        dtype="object"
    )

    for task, g in df.groupby("task"):
        idx = sorted(
            g.index,
            key=lambda i:
                stable_number(
                    str(df.at[i, "clip"])
                )
        )

        ncal = round(len(idx) * 0.70)

        result.loc[idx[:ncal]] = "calibration"
        result.loc[idx[ncal:]] = "validation"

    return result


# ============================================================
# Boundary estimation
# ============================================================

def ols_bic(y, X):
    beta, *_ = np.linalg.lstsq(
        X, y, rcond=None
    )

    resid = y - X @ beta
    sse = max(float(resid @ resid), 1e-15)

    n, k = X.shape

    bic = (
        n * np.log(sse / n)
        + k * np.log(n)
    )

    return beta, bic


def fit_boundary(df, metric):

    xraw = df[metric].to_numpy(float)

    positive = xraw[xraw > 0]

    floor = positive.min() * 0.5

    x = np.log10(
        np.maximum(xraw, floor)
    )

    y = np.log1p(
        df["l2_moved_mm"].to_numpy(float)
    )

    # linear
    Xlin = np.column_stack([
        np.ones(len(df)),
        x,
    ])

    _, biclin = ols_bic(y, Xlin)

    # smooth quadratic competitor
    xc = x - x.mean()

    Xquad = np.column_stack([
        np.ones(len(df)),
        x,
        xc ** 2,
    ])

    _, bicquad = ols_bic(y, Xquad)

    # one-breakpoint segmented model
    candidate_taus = np.unique(
        np.quantile(
            x,
            np.linspace(.20, .90, 71)
        )
    )

    best = None

    for tau in candidate_taus:

        hinge = np.maximum(
            0.0,
            x - tau
        )

        X = np.column_stack([
            np.ones(len(df)),
            x,
            hinge,
        ])

        beta, bic = ols_bic(y, X)

        if best is None or bic < best["bic"]:
            best = {
                "tau": float(tau),
                "bic": float(bic),
                "beta": beta,
            }

    slope_pre = float(
        best["beta"][1]
    )

    slope_post = float(
        best["beta"][1]
        + best["beta"][2]
    )

    return {
        "threshold":
            float(10 ** best["tau"]),

        "delta_bic_vs_linear":
            float(biclin - best["bic"]),

        "delta_bic_vs_quadratic":
            float(bicquad - best["bic"]),

        "slope_pre":
            slope_pre,

        "slope_post":
            slope_post,
    }


def validate_boundary(df, metric, threshold):

    inside = df.loc[
        df[metric] <= threshold,
        "l2_moved_mm"
    ].to_numpy(float)

    outside = df.loc[
        df[metric] > threshold,
        "l2_moved_mm"
    ].to_numpy(float)

    if len(inside) == 0 or len(outside) == 0:
        return {
            "inside_n": len(inside),
            "outside_n": len(outside),
            "inside_med": np.nan,
            "outside_med": np.nan,
            "ratio": np.nan,
            "cliff": np.nan,
            "p": np.nan,
        }

    mw = mannwhitneyu(
        outside,
        inside,
        alternative="two-sided"
    )

    cliff = (
        2.0 * mw.statistic
        / (len(outside) * len(inside))
        - 1.0
    )

    med_in = float(np.median(inside))
    med_out = float(np.median(outside))

    return {
        "inside_n": len(inside),
        "outside_n": len(outside),
        "inside_med": med_in,
        "outside_med": med_out,
        "ratio": med_out / med_in,
        "cliff": float(cliff),
        "p": float(mw.pvalue),
    }


def usable_upper_boundary(fit, val):

    # We specifically require evidence for an UPPER risk boundary.
    # Do not force task-0008 / 0010 into a rule if their shape
    # does not support it.

    return bool(
        fit["delta_bic_vs_linear"] >= 10
        and
        fit["delta_bic_vs_quadratic"] >= 6
        and
        fit["slope_post"] > fit["slope_pre"]
        and
        fit["slope_post"] > 0
        and
        fit["slope_pre"] > -0.10
        and
        val["outside_med"] > val["inside_med"]
        and
        val["cliff"] > 0
    )


# ============================================================
# Official dataloader
# ============================================================

def build_behavior_loader(root):

    args = parse_args(
        skip_command_line=True
    )

    args.seed = SEED
    args.domains = ["behavior"]
    args.data_dirs = [str(root)]

    args.batch_size = 1
    args.num_workers = 4

    if hasattr(args, "eval_num_workers"):
        args.eval_num_workers = 4

    if hasattr(args, "deterministic_data"):
        args.deterministic_data = True

    try:
        loader, info = build_dataloader(
            args,
            mode="test",
            rank=0,
            world_size=1,
            override_splits="test",
        )

    except TypeError:
        loader, info = build_dataloader(
            args,
            mode="test",
            override_splits="test",
        )

    if loader is None:
        raise RuntimeError(
            f"Failed to build loader: {root}"
        )

    return loader


# ============================================================
# Exact original metric
# ============================================================

def exact_metrics(right_pose, left_pose):

    r = established_metrics_for_clip(
        np.asarray(right_pose, np.float64),
        np.asarray(left_pose, np.float64),
    )

    return {
        "dominant_hand":
            r["dominant_hand"],

        "rms_acceleration":
            float(r["rms_acceleration"]),

        "mean_squared_jerk":
            float(r["mean_squared_jerk"]),
    }


# ============================================================
# Temporal reparameterization
# ============================================================

def interpolate_sequence(arr, phase):

    arr = np.asarray(arr, np.float64)

    T = arr.shape[0]

    phase = np.clip(
        np.asarray(phase, float),
        0.0,
        T - 1.0
    )

    lo = np.floor(phase).astype(int)
    hi = np.ceil(phase).astype(int)

    alpha = (
        phase - lo
    ).reshape(
        (T,) + (1,) * (arr.ndim - 1)
    )

    result = (
        (1.0 - alpha) * arr[lo]
        + alpha * arr[hi]
    )

    # hard exact start/end
    result[0] = arr[0]
    result[-1] = arr[-1]

    return result


def metric_poses_from_phase(
    right_pose,
    left_pose,
    phase
):
    """
    Acc/Jerk metrics depend only on XYZ.

    We temporally reparameterize XYZ using the SAME phase
    that will be applied to the full PointWorld robot flow.

    Quaternion values are not used by the established
    Acc/Jerk metrics, so they are left untouched here.
    """

    rp = np.asarray(
        right_pose,
        np.float64
    ).copy()

    lp = np.asarray(
        left_pose,
        np.float64
    ).copy()

    rp[:, :3] = interpolate_sequence(
        rp[:, :3],
        phase
    )

    lp[:, :3] = interpolate_sequence(
        lp[:, :3],
        phase
    )

    return rp, lp


def metrics_at_phase(
    right_pose,
    left_pose,
    phase
):

    rp, lp = metric_poses_from_phase(
        right_pose,
        left_pose,
        phase
    )

    return exact_metrics(
        rp,
        lp
    )


# ============================================================
# Goal-preserved constrained correction
# ============================================================

def optimize_phase(
    right_pose,
    left_pose,
    tau_acc=None,
    tau_msj=None,
):

    T = right_pose.shape[0]

    if T < 5:
        return {
            "success": False,
            "reason": "TOO_SHORT",
        }

    identity = np.arange(
        T,
        dtype=float
    )

    def unpack(x):
        return np.concatenate([
            [0.0],
            np.asarray(x, float),
            [float(T - 1)],
        ])

    def objective(x):

        phase = unpack(x)

        # Minimum necessary temporal change.
        shift = (
            phase - identity
        ) / (T - 1)

        increments = np.diff(phase)

        return float(
            np.mean(shift ** 2)
            +
            0.02
            * np.mean(
                (increments - 1.0) ** 2
            )
        )

    def current_metrics(x):
        return metrics_at_phase(
            right_pose,
            left_pose,
            unpack(x)
        )

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x:
                np.diff(unpack(x))
                - 1e-4,
        }
    ]

    # Acceleration and jerk are NOT combined.
    # They remain separate physical constraints.

    if tau_acc is not None:

        constraints.append({
            "type": "ineq",
            "fun": lambda x:
                tau_acc
                - current_metrics(x)[
                    "rms_acceleration"
                ],
        })

    if tau_msj is not None:

        constraints.append({
            "type": "ineq",
            "fun": lambda x:
                tau_msj
                - current_metrics(x)[
                    "mean_squared_jerk"
                ],
        })

    u = np.linspace(
        0.0,
        1.0,
        T
    )

    starts = []

    # identity
    starts.append(
        identity[1:-1]
    )

    # classical smooth progression
    s = (
        10 * u**3
        - 15 * u**4
        + 6 * u**5
    ) * (T - 1)

    starts.append(
        s[1:-1]
    )

    # early / late redistribution initializations
    for gamma in [0.70, 0.85, 1.15, 1.30]:

        p = (
            u ** gamma
        ) * (T - 1)

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
                (0.0, float(T - 1))
                for _ in range(T - 2)
            ],

            constraints=constraints,

            options={
                "maxiter": 600,
                "ftol": 1e-10,
                "disp": False,
            },
        )

        phase = unpack(
            result.x
        )

        m = metrics_at_phase(
            right_pose,
            left_pose,
            phase
        )

        feasible = bool(
            np.all(
                np.diff(phase) >= -1e-9
            )
        )

        if tau_acc is not None:
            feasible &= (
                m["rms_acceleration"]
                <= tau_acc * 1.002
            )

        if tau_msj is not None:
            feasible &= (
                m["mean_squared_jerk"]
                <= tau_msj * 1.002
            )

        candidate = {
            "success":
                feasible,

            "optimizer_success":
                bool(result.success),

            "optimizer_message":
                str(result.message),

            "objective":
                float(objective(result.x)),

            "phase":
                phase,

            "metrics":
                m,
        }

        if best is None:
            best = candidate

        elif (
            candidate["success"]
            and
            not best["success"]
        ):
            best = candidate

        elif (
            candidate["success"]
            ==
            best["success"]
            and
            candidate["objective"]
            <
            best["objective"]
        ):
            best = candidate

    return best


# ============================================================
# Local diagnostic curves
# ============================================================

def local_profiles(
    right_pose,
    left_pose
):

    (
        hand,
        xyz,
        _,
        _,
    ) = choose_dominant_eef(
        right_pose,
        left_pose
    )

    velocity, acceleration = (
        pointworld_velocity_acceleration(
            xyz
        )
    )

    jerk = calculate_jerk_profile(
        velocity,
        DT
    )

    return {
        "hand": hand,
        "acc":
            np.linalg.norm(
                acceleration,
                axis=-1
            ),

        "jerk":
            np.linalg.norm(
                jerk,
                axis=-1
            ),
    }


def save_profile_plot(
    clip,
    orig_right,
    orig_left,
    corr_right,
    corr_left,
    savepath
):

    a = local_profiles(
        orig_right,
        orig_left
    )

    b = local_profiles(
        corr_right,
        corr_left
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 7)
    )

    axes[0].plot(
        np.arange(len(a["acc"])),
        a["acc"],
        marker="o",
        label="Original",
    )

    axes[0].plot(
        np.arange(len(b["acc"])),
        b["acc"],
        marker="o",
        label="Corrected",
    )

    axes[0].set_ylabel(
        "Acceleration magnitude"
    )

    axes[0].set_title(
        f"{clip} | Acceleration"
    )

    axes[0].legend()
    axes[0].grid(alpha=.2)

    axes[1].plot(
        np.arange(len(a["jerk"])),
        a["jerk"],
        marker="o",
        label="Original",
    )

    axes[1].plot(
        np.arange(len(b["jerk"])),
        b["jerk"],
        marker="o",
        label="Corrected",
    )

    axes[1].set_xlabel(
        "Timestep"
    )

    axes[1].set_ylabel(
        "Jerk magnitude"
    )

    axes[1].set_title(
        "Jerk"
    )

    axes[1].legend()
    axes[1].grid(alpha=.2)

    fig.tight_layout()

    fig.savefig(
        savepath,
        dpi=180
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

# ============================================================
# LOTO 5-TASK MAIN
# ============================================================

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


ROOT = Path(__file__).resolve().parents[1]

LOTO_ROOT = (
    ROOT
    / "diagnostics/results/action_working_range_loto_5task"
)

TARGET_CSV = (
    LOTO_ROOT
    / "loto_correction_targets_all.csv"
)

OUT = (
    ROOT
    / "diagnostics/results/"
      "goal_preserved_loto_correction_5task"
)

CORRECTED_DIR = OUT / "corrected_actions"

OUT.mkdir(parents=True, exist_ok=True)
CORRECTED_DIR.mkdir(parents=True, exist_ok=True)


TASKS = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]

EXPECTED_TASK_N = {
    "task-0000": 162,
    "task-0001": 820,
    "task-0006": 955,
    "task-0008": 797,
    "task-0010": 1097,
}

EXPECTED_TOTAL = sum(EXPECTED_TASK_N.values())


# ============================================================
# One independent correction problem
# ============================================================

def solve_one(payload):

    clip = payload["clip"]
    task = payload["task"]

    right = payload["right_pose"]
    left = payload["left_pose"]

    acc_boundary = float(payload["acc_boundary"])
    msj_boundary = float(payload["msj_boundary"])

    # Exact metric reproduction.
    original = exact_metrics(
        right,
        left,
    )

    acc_violated = (
        original["rms_acceleration"]
        > acc_boundary
    )

    msj_violated = (
        original["mean_squared_jerk"]
        > msj_boundary
    )

    # IMPORTANT:
    # Acceleration and MSJ remain independent constraints.
    tau_acc = (
        acc_boundary
        if acc_violated
        else None
    )

    tau_msj = (
        msj_boundary
        if msj_violated
        else None
    )

    if tau_acc is None and tau_msj is None:
        return {
            "clip": clip,
            "task": task,
            "fatal": True,
            "reason": "TARGET_NO_LONGER_VIOLATES",
        }

    result = optimize_phase(
        right,
        left,
        tau_acc=tau_acc,
        tau_msj=tau_msj,
    )

    phase = result.get(
        "phase",
        np.arange(len(right), dtype=float),
    )

    corrected = result.get(
        "metrics",
        metrics_at_phase(
            right,
            left,
            phase,
        ),
    )

    corr_right, corr_left = (
        metric_poses_from_phase(
            right,
            left,
            phase,
        )
    )

    # Exact endpoint audit in action representation.
    start_error = max(
        float(np.max(
            np.abs(
                corr_right[0, :3]
                - right[0, :3]
            )
        )),
        float(np.max(
            np.abs(
                corr_left[0, :3]
                - left[0, :3]
            )
        )),
    )

    endpoint_error = max(
        float(np.max(
            np.abs(
                corr_right[-1, :3]
                - right[-1, :3]
            )
        )),
        float(np.max(
            np.abs(
                corr_left[-1, :3]
                - left[-1, :3]
            )
        )),
    )

    identity = np.arange(
        len(phase),
        dtype=float,
    )

    max_phase_shift = float(
        np.max(
            np.abs(
                phase - identity
            )
        )
    )

    row = {
        "clip": clip,
        "task": task,

        "success":
            bool(result["success"]),

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
                    ""
                )
            ),

        "objective":
            float(
                result.get(
                    "objective",
                    np.nan,
                )
            ),

        "acc_boundary":
            acc_boundary,

        "msj_boundary":
            msj_boundary,

        "acc_constraint_used":
            bool(acc_violated),

        "msj_constraint_used":
            bool(msj_violated),

        "original_acc":
            float(
                original[
                    "rms_acceleration"
                ]
            ),

        "corrected_acc":
            float(
                corrected[
                    "rms_acceleration"
                ]
            ),

        "original_msj":
            float(
                original[
                    "mean_squared_jerk"
                ]
            ),

        "corrected_msj":
            float(
                corrected[
                    "mean_squared_jerk"
                ]
            ),

        "acc_ratio":
            float(
                corrected[
                    "rms_acceleration"
                ]
                /
                max(
                    original[
                        "rms_acceleration"
                    ],
                    1e-15,
                )
            ),

        "msj_ratio":
            float(
                corrected[
                    "mean_squared_jerk"
                ]
                /
                max(
                    original[
                        "mean_squared_jerk"
                    ],
                    1e-15,
                )
            ),

        "max_phase_shift_frames":
            max_phase_shift,

        "start_error":
            start_error,

        "endpoint_error":
            endpoint_error,

        "phase":
            phase,

        "fatal":
            False,
    }

    return row


# ============================================================
# Load LOTO targets
# ============================================================

targets = pd.read_csv(
    TARGET_CSV
)

if len(targets) != EXPECTED_TOTAL:
    raise RuntimeError(
        f"Expected {EXPECTED_TOTAL} targets, "
        f"got {len(targets)}"
    )

if targets["clip"].duplicated().any():
    raise RuntimeError(
        "Duplicate target clips"
    )

print("=" * 100)
print("5-TASK LOTO GOAL-PRESERVED CORRECTION")
print("=" * 100)

print(
    targets.groupby("heldout_task")
    .size()
    .to_string()
)

print()
print("TOTAL TARGETS =", len(targets))


# ============================================================
# Target lookup
# ============================================================

target_map = {}

for _, r in targets.iterrows():

    task = str(
        r["heldout_task"]
    )

    target_map[
        str(r["clip"])
    ] = {
        "task": task,

        "acc_boundary":
            float(
                r["loto_acc_boundary"]
            ),

        "msj_boundary":
            float(
                r["loto_msj_boundary"]
            ),

        # Original values from established table,
        # used only for reproduction audit.
        "csv_acc":
            float(
                r["rms_acceleration"]
            ),

        "csv_msj":
            float(
                r["mean_squared_jerk"]
            ),
    }


# ============================================================
# Gather target action poses from official dataloaders
# ============================================================

payloads = []
audit_rows = []
found = set()


ROOT_GROUPS = [
    (
        DATA_ROOTS["task-0000"],
        {"task-0000"},
    ),
    (
        DATA_ROOTS["task-0001"],
        {"task-0001"},
    ),
    (
        DATA_ROOTS["multitask"],
        {
            "task-0006",
            "task-0008",
            "task-0010",
        },
    ),
]


for data_root, allowed_tasks in ROOT_GROUPS:

    wanted = {
        k
        for k, v in target_map.items()
        if v["task"] in allowed_tasks
    }

    print()
    print("=" * 100)
    print("SCAN =", data_root)
    print("WANTED =", len(wanted))
    print("=" * 100)

    loader = build_behavior_loader(
        data_root
    )

    for batch in loader:

        key = normalize_key(
            batch["__key__"]
        )

        if key not in wanted:
            continue

        if key in found:
            raise RuntimeError(
                f"Duplicate target in loader: {key}"
            )

        right = unbatch(
            batch[
                "right_gripper_pose"
            ]
        ).astype(
            np.float64
        )

        left = unbatch(
            batch[
                "left_gripper_pose"
            ]
        ).astype(
            np.float64
        )

        meta = target_map[key]

        exact = exact_metrics(
            right,
            left,
        )

        acc_relerr = abs(
            exact["rms_acceleration"]
            - meta["csv_acc"]
        ) / max(
            abs(meta["csv_acc"]),
            1e-15,
        )

        msj_relerr = abs(
            exact["mean_squared_jerk"]
            - meta["csv_msj"]
        ) / max(
            abs(meta["csv_msj"]),
            1e-15,
        )

        audit_rows.append({
            "clip": key,
            "task": meta["task"],

            "csv_acc":
                meta["csv_acc"],

            "exact_acc":
                exact[
                    "rms_acceleration"
                ],

            "acc_relerr":
                acc_relerr,

            "csv_msj":
                meta["csv_msj"],

            "exact_msj":
                exact[
                    "mean_squared_jerk"
                ],

            "msj_relerr":
                msj_relerr,
        })

        payloads.append({
            "clip": key,
            "task": meta["task"],
            "right_pose": right,
            "left_pose": left,
            "acc_boundary":
                meta["acc_boundary"],
            "msj_boundary":
                meta["msj_boundary"],
        })

        found.add(key)

        if len(found) % 200 == 0:
            print(
                "FOUND",
                len(found),
                "/",
                EXPECTED_TOTAL,
            )


missing = (
    set(target_map.keys())
    - found
)

if missing:
    print(
        "MISSING EXAMPLES =",
        sorted(missing)[:20],
    )

    raise RuntimeError(
        f"Missing {len(missing)} "
        "LOTO targets from loaders"
    )


print()
print(
    "ALL TARGETS FOUND =",
    len(found)
)


# ============================================================
# Exact metric reproduction audit BEFORE optimization
# ============================================================

audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    OUT
    / "exact_metric_reproduction_audit.csv",
    index=False,
)

print()
print("=" * 100)
print("EXACT METRIC REPRODUCTION AUDIT")
print("=" * 100)

for c in [
    "acc_relerr",
    "msj_relerr",
]:
    x = audit[c].to_numpy(float)

    print(
        c,
        "| median =",
        np.median(x),
        "| p95 =",
        np.quantile(x, .95),
        "| max =",
        np.max(x),
    )

# STOP rather than silently run mismatched definitions.
if (
    np.quantile(
        audit["acc_relerr"],
        .95,
    ) > 1e-4
    or
    np.quantile(
        audit["msj_relerr"],
        .95,
    ) > 1e-4
):
    raise RuntimeError(
        "Metric reproduction audit failed"
    )


# ============================================================
# Run independent SLSQP problems in parallel
# ============================================================

workers = int(
    os.environ.get(
        "PW_CORR_WORKERS",
        min(16, os.cpu_count() or 1),
    )
)

print()
print("=" * 100)
print("OPTIMIZATION")
print("=" * 100)
print("workers =", workers)
print("targets =", len(payloads))


results = []

ctx = mp.get_context("fork")

with ProcessPoolExecutor(
    max_workers=workers,
    mp_context=ctx,
) as ex:

    futures = [
        ex.submit(
            solve_one,
            p,
        )
        for p in payloads
    ]

    for i, f in enumerate(
        as_completed(futures),
        1,
    ):
        r = f.result()

        if r.get("fatal", False):
            raise RuntimeError(
                f"{r['clip']}: "
                f"{r.get('reason')}"
            )

        results.append(r)

        # Save ONLY successful corrections,
        # matching correction-v2 convention.
        if r["success"]:

            clip_safe = (
                r["clip"]
                .replace("/", "_")
                .replace(":", "__")
            )

            np.savez_compressed(
                CORRECTED_DIR
                / f"{clip_safe}.npz",

                clip=np.asarray(
                    r["clip"]
                ),

                task=np.asarray(
                    r["task"]
                ),

                phase=np.asarray(
                    r["phase"],
                    dtype=np.float64,
                ),

                acc_boundary=np.asarray(
                    r["acc_boundary"]
                ),

                msj_boundary=np.asarray(
                    r["msj_boundary"]
                ),
            )

        if (
            i == 1
            or i % 100 == 0
            or i == len(payloads)
        ):
            suc = sum(
                int(x["success"])
                for x in results
            )

            print(
                f"[{i}/{len(payloads)}] "
                f"success={suc}"
            )


# ============================================================
# Save / summarize
# ============================================================

rows_for_csv = []

for r in results:

    q = dict(r)
    q.pop("phase", None)

    rows_for_csv.append(q)


res = pd.DataFrame(
    rows_for_csv
).sort_values(
    ["task", "clip"]
)

res.to_csv(
    OUT / "correction_results.csv",
    index=False,
)

targets.to_csv(
    OUT / "correction_targets.csv",
    index=False,
)


print()
print("=" * 100)
print("FINAL LOTO CORRECTION SUMMARY")
print("=" * 100)

print(
    "TOTAL =",
    len(res),
)

print(
    "SUCCESS =",
    int(res["success"].sum()),
    "/",
    len(res),
    "=",
    f"{100*res['success'].mean():.2f}%",
)

successful = res[
    res["success"]
].copy()

print()
print("SUCCESSFUL ONLY")
print(
    "median Acc ratio =",
    successful[
        "acc_ratio"
    ].median(),
)

print(
    "median MSJ ratio =",
    successful[
        "msj_ratio"
    ].median(),
)

print(
    "median max phase shift =",
    successful[
        "max_phase_shift_frames"
    ].median(),
)

print(
    "max start error =",
    successful[
        "start_error"
    ].max(),
)

print(
    "max endpoint error =",
    successful[
        "endpoint_error"
    ].max(),
)


print()
print("TASKWISE")
print("-" * 100)

for task in TASKS:

    g = res[
        res["task"] == task
    ]

    s = g[
        g["success"]
    ]

    print()
    print(
        task,
        f"| success {len(s)}/{len(g)} "
        f"({100*len(s)/len(g):.2f}%)"
    )

    if len(s):

        print(
            "  Acc ratio median =",
            s[
                "acc_ratio"
            ].median(),
        )

        print(
            "  MSJ ratio median =",
            s[
                "msj_ratio"
            ].median(),
        )

        print(
            "  phase shift median =",
            s[
                "max_phase_shift_frames"
            ].median(),
        )


summary = {
    "total_targets":
        int(len(res)),

    "success":
        int(res["success"].sum()),

    "success_rate":
        float(res["success"].mean()),

    "taskwise": {},
}

for task in TASKS:

    g = res[
        res["task"] == task
    ]

    s = g[
        g["success"]
    ]

    summary["taskwise"][task] = {
        "target_n":
            int(len(g)),

        "success_n":
            int(len(s)),

        "success_rate":
            float(
                len(s) / len(g)
            ),
    }

with open(
    OUT / "summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print(
    "CORRECTED ACTION FILES =",
    len(
        list(
            CORRECTED_DIR.glob(
                "*.npz"
            )
        )
    ),
)

print("saved =", OUT)
print("SUCCESS = True")
