#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

BOUNDARY_MARGIN = 0.7
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
    "goal_preserved_range_correction_margin70_7987"
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

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CORRECTED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(MAIN)

if len(df) != 7987:
    raise RuntimeError(
        f"Expected N=7987, got {len(df)}"
    )

if df["clip"].duplicated().any():
    raise RuntimeError(
        "Duplicate clip IDs"
    )

df["split"] = make_split(df)


# ------------------------------------------------------------
# 1. Task-specific calibration boundaries
#    Acceleration / Jerk separately.
# ------------------------------------------------------------

boundary_rows = []

for task, g in df.groupby(
    "task",
    sort=True
):

    cal = g[
        g["split"] == "calibration"
    ]

    val = g[
        g["split"] == "validation"
    ]

    for metric in [
        "rms_acceleration",
        "mean_squared_jerk",
    ]:

        fit = fit_boundary(
            cal,
            metric
        )

        validation = validate_boundary(
            val,
            metric,
            fit["threshold"]
        )

        usable = usable_upper_boundary(
            fit,
            validation
        )

        boundary_rows.append({
            "task":
                task,

            "metric":
                metric,

            **fit,

            **{
                f"val_{k}": v
                for k, v
                in validation.items()
            },

            "usable_upper_boundary":
                usable,
        })


boundary_df = pd.DataFrame(
    boundary_rows
)

boundary_df.to_csv(
    OUT /
    "taskwise_boundaries.csv",
    index=False
)


print("\n" + "="*100)
print("TASKWISE BOUNDARIES")
print("="*100)

print(
    boundary_df.to_string(
        index=False
    )
)


boundary_map = {}

for _, r in boundary_df.iterrows():

    if bool(
        r["usable_upper_boundary"]
    ):

        boundary_map[
            (
                r["task"],
                r["metric"]
            )
        ] = float(
            r["threshold"]
        )


print("\n" + "="*100)
print("USABLE UPPER BOUNDARIES")
print("="*100)

for k, v in sorted(
    boundary_map.items()
):
    print(k, "=", v)


# ------------------------------------------------------------
# 2. Validation-only correction targets
# ------------------------------------------------------------

target_rows = []

for _, r in df[
    df["split"] == "validation"
].iterrows():

    task = r["task"]

    ta = boundary_map.get(
        (task, "rms_acceleration")
    )

    tj = boundary_map.get(
        (task, "mean_squared_jerk")
    )

    acc_bad = (
        ta is not None
        and
        r["rms_acceleration"] > ta
    )

    jerk_bad = (
        tj is not None
        and
        r["mean_squared_jerk"] > tj
    )

    if acc_bad or jerk_bad:

        target_rows.append({
            "clip":
                str(r["clip"]),

            "task":
                task,

            "csv_acc":
                float(
                    r["rms_acceleration"]
                ),

            "csv_msj":
                float(
                    r["mean_squared_jerk"]
                ),

            "tau_acc":
                ta,

            "tau_msj":
                tj,

            "acc_violation":
                acc_bad,

            "jerk_violation":
                jerk_bad,
        })


targets = pd.DataFrame(
    target_rows
)

targets.to_csv(
    OUT /
    "correction_targets.csv",
    index=False
)


print("\n" + "="*100)
print("TARGETS")
print("="*100)

print("N =", len(targets))

print(
    targets[
        [
            "task",
            "acc_violation",
            "jerk_violation",
        ]
    ]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 3. Load official processed samples.
# ------------------------------------------------------------

target_lookup = (
    targets
    .set_index("clip")
    .to_dict("index")
)

needed_all = set(
    target_lookup
)

loaded = {}

roots = [
    ("task-0000", DATA_ROOTS["task-0000"]),
    ("task-0001", DATA_ROOTS["task-0001"]),
    ("multitask", DATA_ROOTS["multitask"]),
]


for root_name, root in roots:

    if root_name == "multitask":
        needed = {
            k
            for k in needed_all
            if target_lookup[k]["task"]
            in {
                "task-0006",
                "task-0008",
                "task-0010",
            }
        }

    else:
        needed = {
            k
            for k in needed_all
            if target_lookup[k]["task"]
            == root_name
        }

    if not needed:
        continue

    print(
        f"\nLoading {root_name}: "
        f"{len(needed)} targets"
    )

    loader = build_behavior_loader(
        root
    )

    found_here = 0

    for batch in loader:

        key = normalize_key(
            batch["__key__"]
        )

        if key not in needed:
            continue

        required = [
            "right_gripper_pose",
            "left_gripper_pose",
            "robot_flows",
        ]

        missing = [
            x
            for x in required
            if x not in batch
        ]

        if missing:
            raise RuntimeError(
                f"{key}: missing {missing}"
            )

        loaded[key] = {
            "right_pose":
                unbatch(
                    batch[
                        "right_gripper_pose"
                    ]
                ).astype(
                    np.float64
                ),

            "left_pose":
                unbatch(
                    batch[
                        "left_gripper_pose"
                    ]
                ).astype(
                    np.float64
                ),

            "robot_flows":
                unbatch(
                    batch[
                        "robot_flows"
                    ]
                ).astype(
                    np.float64
                ),
        }

        found_here += 1

        if found_here == len(needed):
            break


missing = sorted(
    needed_all
    - set(loaded)
)

print("\nTARGETS =", len(needed_all))
print("LOADED  =", len(loaded))
print("MISSING =", len(missing))

if missing:
    print(
        "First missing:",
        missing[:20]
    )


# ------------------------------------------------------------
# 4. EXACT metric reproduction audit.
# ------------------------------------------------------------

audit_rows = []

for key, d in loaded.items():

    m = exact_metrics(
        d["right_pose"],
        d["left_pose"]
    )

    ref = target_lookup[key]

    acc_rel = (
        abs(
            m["rms_acceleration"]
            - ref["csv_acc"]
        )
        /
        max(
            abs(ref["csv_acc"]),
            1e-15
        )
    )

    msj_rel = (
        abs(
            m["mean_squared_jerk"]
            - ref["csv_msj"]
        )
        /
        max(
            abs(ref["csv_msj"]),
            1e-15
        )
    )

    audit_rows.append({
        "clip": key,
        "task": ref["task"],

        "csv_acc":
            ref["csv_acc"],

        "recomputed_acc":
            m["rms_acceleration"],

        "relerr_acc":
            acc_rel,

        "csv_msj":
            ref["csv_msj"],

        "recomputed_msj":
            m["mean_squared_jerk"],

        "relerr_msj":
            msj_rel,

        "dominant_hand":
            m["dominant_hand"],
    })


audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    OUT /
    "exact_metric_reproduction_audit.csv",
    index=False
)


print("\n" + "="*100)
print("EXACT METRIC REPRODUCTION AUDIT")
print("="*100)

for c in [
    "relerr_acc",
    "relerr_msj",
]:
    x = audit[c].to_numpy(float)

    print(
        c,
        "median =",
        np.median(x),
        "p95 =",
        np.quantile(x, .95),
        "max =",
        np.max(x),
    )


# Must be essentially the SAME metric implementation.
audit_ok = bool(
    audit["relerr_acc"].median() < 1e-5
    and
    audit["relerr_msj"].median() < 1e-5
    and
    audit["relerr_acc"].quantile(.99) < 1e-4
    and
    audit["relerr_msj"].quantile(.99) < 1e-4
)


if not audit_ok:

    print()
    print(
        "STOP: exact original metric values "
        "were not reproduced."
    )

    print(
        "No correction has been executed."
    )

    print("SUCCESS = False")

    raise SystemExit(3)


print(
    "\nAUDIT PASSED — "
    "using exact original RMS/MSJ definitions."
)


# ------------------------------------------------------------
# 5. Goal-preserved temporal correction.
# ------------------------------------------------------------

results = []
successful_for_plot = []


for count, (
    key,
    d
) in enumerate(
    loaded.items(),
    start=1
):

    ref = target_lookup[key]

    orig = exact_metrics(
        d["right_pose"],
        d["left_pose"]
    )

    ta = ref["tau_acc"]
    tj = ref["tau_msj"]

    if pd.isna(ta):
        ta = None

    if pd.isna(tj):
        tj = None

    # Safety-margin correction.
    # Original boundary is NOT re-estimated.
    # Only the target constraint is moved inward.

    if ta is not None:
        ta = ta * BOUNDARY_MARGIN

    if tj is not None:
        tj = tj * BOUNDARY_MARGIN

    correction = optimize_phase(
        d["right_pose"],
        d["left_pose"],
        tau_acc=ta,
        tau_msj=tj,
    )

    if not correction["success"]:

        results.append({
            "clip": key,
            "task": ref["task"],
            "success": False,
            "reason":
                correction.get(
                    "optimizer_message",
                    correction.get(
                        "reason",
                        "INFEASIBLE"
                    )
                ),

            "original_acc":
                orig["rms_acceleration"],

            "original_msj":
                orig["mean_squared_jerk"],

            "tau_acc": ta,
            "tau_msj": tj,
        })

        continue


    phase = correction["phase"]

    corr_right, corr_left = (
        metric_poses_from_phase(
            d["right_pose"],
            d["left_pose"],
            phase
        )
    )

    corr_metrics = exact_metrics(
        corr_right,
        corr_left
    )

    # SAME temporal redistribution applied
    # to the actual PointWorld Action input.
    corrected_robot_flows = (
        interpolate_sequence(
            d["robot_flows"],
            phase
        )
    )

    # exact start/end in action space
    start_err = float(
        np.max(
            np.abs(
                corrected_robot_flows[0]
                -
                d["robot_flows"][0]
            )
        )
    )

    endpoint_err = float(
        np.max(
            np.abs(
                corrected_robot_flows[-1]
                -
                d["robot_flows"][-1]
            )
        )
    )

    max_phase_shift = float(
        np.max(
            np.abs(
                phase
                -
                np.arange(
                    len(phase)
                )
            )
        )
    )

    row = {
        "clip": key,
        "task": ref["task"],
        "success": True,

        "acc_violation":
            ref["acc_violation"],

        "jerk_violation":
            ref["jerk_violation"],

        "tau_acc": ta,
        "tau_msj": tj,

        "original_acc":
            orig["rms_acceleration"],

        "corrected_acc":
            corr_metrics[
                "rms_acceleration"
            ],

        "acc_ratio":
            corr_metrics[
                "rms_acceleration"
            ]
            /
            orig[
                "rms_acceleration"
            ],

        "original_msj":
            orig[
                "mean_squared_jerk"
            ],

        "corrected_msj":
            corr_metrics[
                "mean_squared_jerk"
            ],

        "msj_ratio":
            corr_metrics[
                "mean_squared_jerk"
            ]
            /
            orig[
                "mean_squared_jerk"
            ],

        "start_error":
            start_err,

        "endpoint_error":
            endpoint_err,

        "max_phase_shift":
            max_phase_shift,

        "objective":
            correction[
                "objective"
            ],
    }

    results.append(row)

    safe = (
        key
        .replace("/", "_")
        .replace(":", "__")
    )

    np.savez_compressed(
        CORRECTED_DIR /
        f"{safe}.npz",

        clip=np.array(key),

        task=np.array(
            ref["task"]
        ),

        phase=
            phase.astype(
                np.float32
            ),

        original_robot_flows=
            d["robot_flows"]
            .astype(
                np.float32
            ),

        corrected_robot_flows=
            corrected_robot_flows
            .astype(
                np.float32
            ),

        original_acc=np.array(
            orig[
                "rms_acceleration"
            ]
        ),

        corrected_acc=np.array(
            corr_metrics[
                "rms_acceleration"
            ]
        ),

        original_msj=np.array(
            orig[
                "mean_squared_jerk"
            ]
        ),

        corrected_msj=np.array(
            corr_metrics[
                "mean_squared_jerk"
            ]
        ),
    )

    severity = max(
        (
            orig["rms_acceleration"]
            / ta
            if ta is not None
            else 0
        ),
        (
            orig["mean_squared_jerk"]
            / tj
            if tj is not None
            else 0
        ),
    )

    successful_for_plot.append(
        (
            severity,
            key,
            d["right_pose"],
            d["left_pose"],
            corr_right,
            corr_left,
        )
    )

    if count % 100 == 0:
        print(
            f"Processed {count}/{len(loaded)}"
        )


result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUT /
    "correction_results.csv",
    index=False
)


# ------------------------------------------------------------
# 6. Representative local plots.
# ------------------------------------------------------------

successful_for_plot = sorted(
    successful_for_plot,
    key=lambda x: x[0],
    reverse=True
)[:10]


for rank, item in enumerate(
    successful_for_plot,
    start=1
):

    (
        severity,
        key,
        orig_r,
        orig_l,
        corr_r,
        corr_l,
    ) = item

    save_profile_plot(
        key,
        orig_r,
        orig_l,
        corr_r,
        corr_l,
        PLOT_DIR /
        f"{rank:02d}_acc_jerk_before_after.png"
    )


# ------------------------------------------------------------
# 7. Summary.
# ------------------------------------------------------------

success = result_df[
    result_df["success"] == True
].copy()


summary = {
    "target_N":
        int(len(targets)),

    "loaded_N":
        int(len(loaded)),

    "metric_audit_passed":
        True,

    "correction_success_N":
        int(len(success)),

    "correction_success_rate":
        float(
            len(success)
            /
            max(len(loaded), 1)
        ),
}


if len(success):

    summary.update({
        "median_original_acc":
            float(
                success[
                    "original_acc"
                ].median()
            ),

        "median_corrected_acc":
            float(
                success[
                    "corrected_acc"
                ].median()
            ),

        "median_acc_ratio":
            float(
                success[
                    "acc_ratio"
                ].median()
            ),

        "median_original_msj":
            float(
                success[
                    "original_msj"
                ].median()
            ),

        "median_corrected_msj":
            float(
                success[
                    "corrected_msj"
                ].median()
            ),

        "median_msj_ratio":
            float(
                success[
                    "msj_ratio"
                ].median()
            ),

        "max_start_error":
            float(
                success[
                    "start_error"
                ].max()
            ),

        "max_endpoint_error":
            float(
                success[
                    "endpoint_error"
                ].max()
            ),

        "median_max_phase_shift_frames":
            float(
                success[
                    "max_phase_shift"
                ].median()
            ),
    })


(
    OUT /
    "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2
    )
)


print("\n" + "="*100)
print("CORRECTION SUMMARY")
print("="*100)

print(
    json.dumps(
        summary,
        indent=2
    )
)


if len(success):

    print("\n" + "="*100)
    print("BY TASK")
    print("="*100)

    print(
        success.groupby("task")[
            [
                "acc_ratio",
                "msj_ratio",
                "endpoint_error",
                "max_phase_shift",
            ]
        ]
        .median()
        .to_string()
    )


print()
print("RESULT_DIR =", OUT)
print("SUCCESS = True")
