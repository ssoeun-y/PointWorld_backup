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
from dataset_components.transforms import compute_flow_derivatives


SEED = 20260902

MAIN = Path(
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)

WORKING = Path(
    "diagnostics/results/action_working_range_v2_7987"
)

OUT = Path(
    "diagnostics/results/"
    "goal_preserved_range_correction_7987"
)

CORRECTED_DIR = OUT / "corrected_actions"

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
# Utilities
# ============================================================

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

    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            x = x[0]

    if isinstance(x, bytes):
        x = x.decode("utf-8")

    return str(x)


def to_numpy(x):

    try:
        import torch

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    return np.asarray(x)


def remove_batch_dim(x):

    x = to_numpy(x)

    if x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]

    return x


def stable_number(text):

    h = hashlib.sha256(
        f"{SEED}|{text}".encode()
    ).digest()

    return int.from_bytes(
        h[:8], "big"
    ) / 2**64


def split_data(df):

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

        n_cal = round(
            len(idx) * 0.70
        )

        result.loc[
            idx[:n_cal]
        ] = "calibration"

        result.loc[
            idx[n_cal:]
        ] = "validation"

    return result


def task_dummies(task):

    return pd.get_dummies(
        task.astype(str),
        drop_first=True,
        dtype=float
    ).to_numpy(float)


def ols_bic(y, X):

    beta, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )

    residual = y - X @ beta

    sse = max(
        float(residual @ residual),
        1e-15
    )

    n, k = X.shape

    bic = (
        n * np.log(sse / n)
        + k * np.log(n)
    )

    return beta, bic


# ============================================================
# Boundary estimation
# ============================================================

def fit_boundary(df, metric):

    x_raw = df[metric].to_numpy(float)

    positive = x_raw[
        x_raw > 0
    ]

    floor = positive.min() * 0.5

    x = np.log10(
        np.maximum(
            x_raw,
            floor
        )
    )

    y = np.log1p(
        df["l2_moved_mm"]
        .to_numpy(float)
    )

    candidate_taus = np.unique(
        np.quantile(
            x,
            np.linspace(
                .20,
                .90,
                71
            )
        )
    )

    X0 = np.column_stack([
        np.ones(len(df)),
        x
    ])

    _, bic_linear = ols_bic(
        y,
        X0
    )

    xc = x - x.mean()

    Xq = np.column_stack([
        np.ones(len(df)),
        x,
        xc ** 2
    ])

    _, bic_quad = ols_bic(
        y,
        Xq
    )

    best = None

    for tau in candidate_taus:

        hinge = np.maximum(
            0,
            x - tau
        )

        X = np.column_stack([
            np.ones(len(df)),
            x,
            hinge
        ])

        beta, bic = ols_bic(
            y,
            X
        )

        if (
            best is None
            or bic < best["bic"]
        ):
            best = {
                "tau": tau,
                "bic": bic,
                "beta": beta
            }

    pre = float(
        best["beta"][1]
    )

    post = float(
        best["beta"][1]
        + best["beta"][2]
    )

    return {
        "threshold":
            float(
                10 ** best["tau"]
            ),

        "delta_bic_linear":
            float(
                bic_linear
                - best["bic"]
            ),

        "delta_bic_quad":
            float(
                bic_quad
                - best["bic"]
            ),

        "slope_pre":
            pre,

        "slope_post":
            post
    }


def validate_boundary(
    df,
    metric,
    threshold
):

    inside = df[
        df[metric] <= threshold
    ]["l2_moved_mm"].to_numpy(float)

    outside = df[
        df[metric] > threshold
    ]["l2_moved_mm"].to_numpy(float)

    if (
        len(inside) == 0
        or len(outside) == 0
    ):
        return {
            "inside_n": len(inside),
            "outside_n": len(outside),
            "inside_med": np.nan,
            "outside_med": np.nan,
            "ratio": np.nan,
            "cliff": np.nan,
            "p": np.nan
        }

    mw = mannwhitneyu(
        outside,
        inside,
        alternative="two-sided"
    )

    cliff = (
        2 * mw.statistic
        / (
            len(outside)
            * len(inside)
        )
        - 1
    )

    med_in = float(
        np.median(inside)
    )

    med_out = float(
        np.median(outside)
    )

    return {
        "inside_n": len(inside),
        "outside_n": len(outside),

        "inside_med": med_in,
        "outside_med": med_out,

        "ratio":
            med_out / med_in,

        "cliff":
            float(cliff),

        "p":
            float(mw.pvalue)
    }


def boundary_is_usable(
    fit,
    val
):
    """
    Upper-boundary interpretation only.

    Do not accept:
    - arbitrary curvature
    - downward post-boundary trend
    - strong low-dynamics U-shape
    - validation reversal
    """

    return bool(
        fit["delta_bic_linear"] >= 10
        and
        fit["delta_bic_quad"] >= 6
        and
        fit["slope_post"] > 0
        and
        fit["slope_post"]
            > fit["slope_pre"]
        and
        fit["slope_pre"] > -0.10
        and
        val["outside_med"]
            > val["inside_med"]
        and
        val["cliff"] > 0
    )


# ============================================================
# Exact dynamics calculation
# ============================================================

def dynamics_metrics(
    robot_flows,
    robot_exists=None
):

    robot_flows = np.asarray(
        robot_flows,
        dtype=np.float64
    )

    velocity, acceleration = (
        compute_flow_derivatives(
            robot_flows
        )
    )

    jerk, _ = (
        compute_flow_derivatives(
            acceleration
        )
    )

    acc_sq = np.sum(
        acceleration ** 2,
        axis=-1
    )

    jerk_sq = np.sum(
        jerk ** 2,
        axis=-1
    )

    if robot_exists is not None:

        mask = np.asarray(
            robot_exists,
            dtype=bool
        )

        if mask.shape != acc_sq.shape:
            raise RuntimeError(
                "robot_exists shape "
                f"{mask.shape} != "
                f"metric shape "
                f"{acc_sq.shape}"
            )

        acc_values = acc_sq[mask]
        jerk_values = jerk_sq[mask]

    else:
        acc_values = acc_sq.reshape(-1)
        jerk_values = jerk_sq.reshape(-1)

    rms_acc = float(
        np.sqrt(
            np.mean(
                acc_values
            )
        )
    )

    msj = float(
        np.mean(
            jerk_values
        )
    )

    local_acc = []

    local_jerk = []

    for t in range(
        robot_flows.shape[0]
    ):

        if robot_exists is not None:

            mt = mask[t]

            if mt.any():

                local_acc.append(
                    np.sqrt(
                        np.mean(
                            acc_sq[t][mt]
                        )
                    )
                )

                local_jerk.append(
                    np.mean(
                        jerk_sq[t][mt]
                    )
                )

            else:
                local_acc.append(
                    np.nan
                )

                local_jerk.append(
                    np.nan
                )

        else:

            local_acc.append(
                np.sqrt(
                    np.mean(
                        acc_sq[t]
                    )
                )
            )

            local_jerk.append(
                np.mean(
                    jerk_sq[t]
                )
            )

    return {
        "rms_acceleration":
            rms_acc,

        "mean_squared_jerk":
            msj,

        "local_acc":
            np.asarray(local_acc),

        "local_jerk":
            np.asarray(local_jerk)
    }


# ============================================================
# Goal-preserved temporal redistribution
# ============================================================

def interpolate_path(
    robot_flows,
    phase
):
    """
    Re-sample ONLY along original robot-flow trajectory.

    phase[0] = 0
    phase[-1] = T-1
    phase is monotonic.

    Therefore:
    - exact start
    - exact endpoint / goal
    - original path ordering preserved
    """

    arr = np.asarray(
        robot_flows,
        dtype=np.float64
    )

    T = arr.shape[0]

    original_t = np.arange(
        T,
        dtype=float
    )

    flat = arr.reshape(
        T,
        -1
    )

    corrected = np.empty_like(
        flat
    )

    for j in range(
        flat.shape[1]
    ):
        corrected[:, j] = np.interp(
            phase,
            original_t,
            flat[:, j]
        )

    return corrected.reshape(
        arr.shape
    )


def optimize_phase(
    robot_flows,
    robot_exists,
    tau_acc=None,
    tau_jerk=None
):

    arr = np.asarray(
        robot_flows,
        dtype=np.float64
    )

    T = arr.shape[0]

    if T < 4:
        return {
            "success": False,
            "reason": "T_TOO_SHORT"
        }

    original_phase = np.arange(
        T,
        dtype=float
    )

    def unpack(x):

        return np.concatenate([
            [0.0],
            np.asarray(x),
            [float(T - 1)]
        ])

    def corrected_from_x(x):

        return interpolate_path(
            arr,
            unpack(x)
        )

    def measure(x):

        corrected = corrected_from_x(
            x
        )

        return dynamics_metrics(
            corrected,
            robot_exists
        )

    def objective(x):

        phase = unpack(x)

        displacement = (
            phase
            - original_phase
        ) / max(
            T - 1,
            1
        )

        increments = np.diff(
            phase
        )

        # minimal correction:
        # stay close to original timing
        # while avoiding violent redistribution
        return float(
            np.mean(
                displacement ** 2
            )
            +
            0.05
            * np.mean(
                (increments - 1.0) ** 2
            )
        )

    constraints = []

    # Monotonic path progression.
    constraints.append({
        "type": "ineq",
        "fun": lambda x:
            np.diff(
                unpack(x)
            )
            - 1e-4
    })

    # IMPORTANT:
    # Acceleration and jerk remain SEPARATE constraints.
    if tau_acc is not None:

        constraints.append({
            "type": "ineq",
            "fun": lambda x:
                tau_acc
                - measure(x)[
                    "rms_acceleration"
                ]
        })

    if tau_jerk is not None:

        constraints.append({
            "type": "ineq",
            "fun": lambda x:
                tau_jerk
                - measure(x)[
                    "mean_squared_jerk"
                ]
        })

    u = np.linspace(
        0,
        1,
        T
    )

    starts = []

    # Original timing.
    starts.append(
        original_phase[1:-1]
    )

    # Minimum-jerk shaped initialization.
    mj = (
        10 * u ** 3
        - 15 * u ** 4
        + 6 * u ** 5
    ) * (T - 1)

    starts.append(
        mj[1:-1]
    )

    # Early / late redistribution starts.
    for gamma in [
        0.70,
        1.30
    ]:

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
                (
                    0,
                    float(T - 1)
                )
                for _ in range(T - 2)
            ],

            constraints=constraints,

            options={
                "maxiter": 500,
                "ftol": 1e-10,
                "disp": False
            }
        )

        phase = unpack(
            result.x
        )

        corrected = interpolate_path(
            arr,
            phase
        )

        metrics = dynamics_metrics(
            corrected,
            robot_exists
        )

        feasible = True

        if tau_acc is not None:

            feasible &= (
                metrics[
                    "rms_acceleration"
                ]
                <= tau_acc * 1.005
            )

        if tau_jerk is not None:

            feasible &= (
                metrics[
                    "mean_squared_jerk"
                ]
                <= tau_jerk * 1.005
            )

        feasible &= bool(
            np.all(
                np.diff(
                    phase
                ) >= -1e-9
            )
        )

        endpoint_error = float(
            np.max(
                np.abs(
                    corrected[-1]
                    - arr[-1]
                )
            )
        )

        start_error = float(
            np.max(
                np.abs(
                    corrected[0]
                    - arr[0]
                )
            )
        )

        feasible &= (
            endpoint_error < 1e-10
            and
            start_error < 1e-10
        )

        candidate = {
            "optimizer_success":
                bool(result.success),

            "optimizer_message":
                str(result.message),

            "objective":
                float(
                    objective(
                        result.x
                    )
                ),

            "feasible":
                bool(feasible),

            "phase":
                phase,

            "corrected":
                corrected,

            "metrics":
                metrics,

            "endpoint_error":
                endpoint_error,

            "start_error":
                start_error
        }

        if best is None:

            best = candidate

        elif (
            candidate["feasible"]
            and
            not best["feasible"]
        ):
            best = candidate

        elif (
            candidate["feasible"]
            ==
            best["feasible"]
            and
            candidate["objective"]
            <
            best["objective"]
        ):
            best = candidate

    return {
        "success":
            bool(
                best["feasible"]
            ),

        **best
    }


# ============================================================
# Dataloader
# ============================================================

def build_behavior_loader(
    data_root
):

    args = parse_args(
        skip_command_line=True
    )

    args.seed = SEED
    args.domains = ["behavior"]
    args.data_dirs = [
        str(data_root)
    ]

    args.batch_size = 1
    args.num_workers = 4

    if hasattr(
        args,
        "eval_num_workers"
    ):
        args.eval_num_workers = 4

    if hasattr(
        args,
        "deterministic_data"
    ):
        args.deterministic_data = True

    try:

        loader, info = build_dataloader(
            args,
            mode="test",
            rank=0,
            world_size=1,
            override_splits="test"
        )

    except TypeError:

        loader, info = build_dataloader(
            args,
            mode="test",
            override_splits="test"
        )

    if loader is None:
        raise RuntimeError(
            f"Failed loader: {data_root}"
        )

    return loader


# ============================================================
# Path / redistribution audits
# ============================================================

def trajectory_path_length(
    arr
):

    arr = np.asarray(
        arr,
        dtype=float
    )

    step = np.diff(
        arr,
        axis=0
    )

    point_step = np.linalg.norm(
        step,
        axis=-1
    )

    return float(
        np.mean(
            np.sum(
                point_step,
                axis=0
            )
        )
    )


def save_local_plot(
    clip,
    original_metrics,
    corrected_metrics,
    tau_acc,
    tau_jerk,
    save_path
):

    T = len(
        original_metrics[
            "local_acc"
        ]
    )

    t = np.arange(T)

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        t,
        original_metrics[
            "local_acc"
        ],
        marker="o",
        label="Original local acceleration"
    )

    ax.plot(
        t,
        corrected_metrics[
            "local_acc"
        ],
        marker="o",
        label="Corrected local acceleration"
    )

    ax.set_xlabel(
        "Timestep"
    )

    ax.set_ylabel(
        "Local acceleration RMS"
    )

    ax.set_title(
        clip
    )

    ax.legend()

    ax.grid(
        alpha=.2
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
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

main = pd.read_csv(
    MAIN
)

if len(main) != 7987:
    raise RuntimeError(
        f"N != 7987: {len(main)}"
    )

main["split"] = split_data(
    main
)


# ------------------------------------------------------------
# 1. Learn Acceleration and Jerk boundaries INDEPENDENTLY
#    inside each task, calibration only.
# ------------------------------------------------------------

boundary_rows = []

metrics = [
    "rms_acceleration",
    "mean_squared_jerk"
]

for task, g in main.groupby(
    "task",
    sort=True
):

    cal = g[
        g["split"]
        ==
        "calibration"
    ]

    val = g[
        g["split"]
        ==
        "validation"
    ]

    for metric in metrics:

        fit = fit_boundary(
            cal,
            metric
        )

        vs = validate_boundary(
            val,
            metric,
            fit["threshold"]
        )

        usable = (
            boundary_is_usable(
                fit,
                vs
            )
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
                in vs.items()
            },

            "usable_upper_boundary":
                usable
        })


boundaries = pd.DataFrame(
    boundary_rows
)

boundaries.to_csv(
    OUT /
    "taskwise_calibrated_boundaries.csv",
    index=False
)


print()
print("=" * 100)
print("TASKWISE CALIBRATED BOUNDARIES")
print("=" * 100)

print(
    boundaries.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 2. Build independent Acc / Jerk rules.
# ------------------------------------------------------------

boundary_map = {}

for _, r in boundaries.iterrows():

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


print()
print("=" * 100)
print("USABLE RULES")
print("=" * 100)

for key, value in sorted(
    boundary_map.items()
):
    print(
        key,
        "->",
        value
    )


# ------------------------------------------------------------
# 3. Select validation Action inputs that violate a rule.
#
# Acc and Jerk remain separate.
# ------------------------------------------------------------

targets = []

for _, r in main[
    main["split"]
    ==
    "validation"
].iterrows():

    task = r["task"]

    ta = boundary_map.get(
        (
            task,
            "rms_acceleration"
        )
    )

    tj = boundary_map.get(
        (
            task,
            "mean_squared_jerk"
        )
    )

    acc_violation = (
        ta is not None
        and
        r[
            "rms_acceleration"
        ] > ta
    )

    jerk_violation = (
        tj is not None
        and
        r[
            "mean_squared_jerk"
        ] > tj
    )

    if (
        acc_violation
        or
        jerk_violation
    ):

        targets.append({
            "clip":
                r["clip"],

            "task":
                task,

            "original_l2_moved_mm":
                r["l2_moved_mm"],

            "csv_rms_acceleration":
                r[
                    "rms_acceleration"
                ],

            "csv_mean_squared_jerk":
                r[
                    "mean_squared_jerk"
                ],

            "tau_acc":
                ta,

            "tau_jerk":
                tj,

            "acc_violation":
                acc_violation,

            "jerk_violation":
                jerk_violation
        })


target_df = pd.DataFrame(
    targets
)

target_df.to_csv(
    OUT /
    "validation_correction_targets.csv",
    index=False
)


print()
print("=" * 100)
print("CORRECTION TARGETS")
print("=" * 100)

print(
    target_df[
        [
            "task",
            "acc_violation",
            "jerk_violation"
        ]
    ]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 4. Load exact official robot_flows.
# ------------------------------------------------------------

target_keys = set(
    target_df[
        "clip"
    ].astype(str)
)

target_lookup = (
    target_df
    .set_index("clip")
    .to_dict("index")
)

loaded = {}


roots = [
    (
        "task-0000",
        DATA_ROOTS[
            "task-0000"
        ]
    ),

    (
        "task-0001",
        DATA_ROOTS[
            "task-0001"
        ]
    ),

    (
        "multitask",
        DATA_ROOTS[
            "multitask"
        ]
    )
]


for root_name, root in roots:

    needed = {
        k
        for k in target_keys
        if (
            root_name
            == "multitask"
            and
            target_lookup[k][
                "task"
            ]
            in {
                "task-0006",
                "task-0008",
                "task-0010"
            }
        )
        or
        (
            root_name
            ==
            target_lookup[k][
                "task"
            ]
        )
    }

    if not needed:
        continue

    print()
    print(
        "Loading root:",
        root_name,
        "targets:",
        len(needed)
    )

    loader = build_behavior_loader(
        root
    )

    for batch in loader:

        if "__key__" not in batch:
            raise RuntimeError(
                "Official batch has no __key__"
            )

        key = normalize_key(
            batch["__key__"]
        )

        if key not in needed:
            continue

        robot_flows = remove_batch_dim(
            batch[
                "robot_flows"
            ]
        )

        robot_exists = None

        if (
            "robot_exists"
            in batch
        ):

            robot_exists = remove_batch_dim(
                batch[
                    "robot_exists"
                ]
            ).astype(bool)

        loaded[key] = {
            "robot_flows":
                robot_flows,

            "robot_exists":
                robot_exists
        }

        if needed.issubset(
            loaded.keys()
        ):
            break


missing = sorted(
    target_keys
    - set(
        loaded.keys()
    )
)

print()
print(
    "TARGETS =",
    len(target_keys)
)

print(
    "LOADED  =",
    len(loaded)
)

print(
    "MISSING =",
    len(missing)
)

if missing:
    print(
        "First missing:",
        missing[:20]
    )


# ------------------------------------------------------------
# 5. CRITICAL METRIC RECOMPUTE AUDIT
#
# If our local implementation does not reproduce
# the established CSV metrics, DO NOT correct anything.
# ------------------------------------------------------------

audit_rows = []

for key, obj in loaded.items():

    original = dynamics_metrics(
        obj[
            "robot_flows"
        ],
        obj[
            "robot_exists"
        ]
    )

    ref = target_lookup[
        key
    ]

    csv_acc = float(
        ref[
            "csv_rms_acceleration"
        ]
    )

    csv_jerk = float(
        ref[
            "csv_mean_squared_jerk"
        ]
    )

    rel_acc = (
        abs(
            original[
                "rms_acceleration"
            ]
            - csv_acc
        )
        /
        max(
            abs(csv_acc),
            1e-12
        )
    )

    rel_jerk = (
        abs(
            original[
                "mean_squared_jerk"
            ]
            - csv_jerk
        )
        /
        max(
            abs(csv_jerk),
            1e-12
        )
    )

    audit_rows.append({
        "clip":
            key,

        "task":
            ref["task"],

        "csv_acc":
            csv_acc,

        "recomputed_acc":
            original[
                "rms_acceleration"
            ],

        "relerr_acc":
            rel_acc,

        "csv_msj":
            csv_jerk,

        "recomputed_msj":
            original[
                "mean_squared_jerk"
            ],

        "relerr_msj":
            rel_jerk
    })


audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    OUT /
    "metric_recompute_audit.csv",
    index=False
)


print()
print("=" * 100)
print("METRIC RECOMPUTE AUDIT")
print("=" * 100)

for metric in [
    "relerr_acc",
    "relerr_msj"
]:

    values = audit_df[
        metric
    ].to_numpy(float)

    print(
        metric,
        "median =",
        np.median(values),
        "p95 =",
        np.quantile(
            values,
            .95
        ),
        "max =",
        np.max(values)
    )


# Hard stop if metric definition is not essentially reproduced.
audit_ok = bool(
    audit_df[
        "relerr_acc"
    ].median()
    < 0.02
    and
    audit_df[
        "relerr_msj"
    ].median()
    < 0.02
)


if not audit_ok:

    print()
    print(
        "STOP: established RMS/MSJ "
        "definition was NOT reproduced "
        "closely enough."
    )

    print(
        "Do not perform correction "
        "with mismatched metrics."
    )

    print(
        "SUCCESS = False"
    )

    raise SystemExit(3)


# ------------------------------------------------------------
# 6. Goal-preserved correction
# ------------------------------------------------------------

results = []

representative = []

for number, (
    key,
    obj
) in enumerate(
    loaded.items(),
    start=1
):

    ref = target_lookup[
        key
    ]

    flows = obj[
        "robot_flows"
    ]

    exists = obj[
        "robot_exists"
    ]

    # If point existence changes over time,
    # do not silently modify such a trajectory.
    if exists is not None:

        constant_exists = bool(
            np.all(
                exists
                ==
                exists[0:1]
            )
        )

    else:
        constant_exists = True

    original = dynamics_metrics(
        flows,
        exists
    )

    tau_acc = ref[
        "tau_acc"
    ]

    tau_jerk = ref[
        "tau_jerk"
    ]

    if (
        pd.isna(tau_acc)
    ):
        tau_acc = None

    if (
        pd.isna(tau_jerk)
    ):
        tau_jerk = None

    if not constant_exists:

        results.append({
            "clip":
                key,

            "task":
                ref["task"],

            "success":
                False,

            "reason":
                "TIME_VARYING_ROBOT_EXISTS"
        })

        continue

    correction = optimize_phase(
        flows,
        exists,
        tau_acc=tau_acc,
        tau_jerk=tau_jerk
    )

    if not correction[
        "success"
    ]:

        results.append({
            "clip":
                key,

            "task":
                ref["task"],

            "success":
                False,

            "reason":
                "OPTIMIZATION_INFEASIBLE",

            "original_acc":
                original[
                    "rms_acceleration"
                ],

            "original_msj":
                original[
                    "mean_squared_jerk"
                ],

            "tau_acc":
                tau_acc,

            "tau_jerk":
                tau_jerk
        })

        continue

    corrected = correction[
        "corrected"
    ]

    corr_metrics = correction[
        "metrics"
    ]

    path_orig = (
        trajectory_path_length(
            flows
        )
    )

    path_corr = (
        trajectory_path_length(
            corrected
        )
    )

    endpoint_err = float(
        np.max(
            np.abs(
                corrected[-1]
                -
                flows[-1]
            )
        )
    )

    start_err = float(
        np.max(
            np.abs(
                corrected[0]
                -
                flows[0]
            )
        )
    )

    phase = correction[
        "phase"
    ]

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

    result_row = {
        "clip":
            key,

        "task":
            ref["task"],

        "success":
            True,

        "acc_violation":
            ref[
                "acc_violation"
            ],

        "jerk_violation":
            ref[
                "jerk_violation"
            ],

        "tau_acc":
            tau_acc,

        "tau_jerk":
            tau_jerk,

        "original_acc":
            original[
                "rms_acceleration"
            ],

        "corrected_acc":
            corr_metrics[
                "rms_acceleration"
            ],

        "acc_ratio":
            corr_metrics[
                "rms_acceleration"
            ]
            /
            original[
                "rms_acceleration"
            ],

        "original_msj":
            original[
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
            original[
                "mean_squared_jerk"
            ],

        "start_error":
            start_err,

        "endpoint_error":
            endpoint_err,

        "path_length_original":
            path_orig,

        "path_length_corrected":
            path_corr,

        "path_length_ratio":
            (
                path_corr
                /
                path_orig
                if path_orig > 0
                else np.nan
            ),

        "max_phase_shift":
            max_phase_shift,

        "objective":
            correction[
                "objective"
            ]
    }

    results.append(
        result_row
    )

    safe_name = (
        key
        .replace("/", "_")
        .replace(":", "__")
    )

    np.savez_compressed(
        CORRECTED_DIR /
        f"{safe_name}.npz",

        clip=np.array(key),

        task=np.array(
            ref["task"]
        ),

        original_robot_flows=
            flows.astype(
                np.float32
            ),

        corrected_robot_flows=
            corrected.astype(
                np.float32
            ),

        phase=
            phase.astype(
                np.float32
            ),

        tau_acc=np.array(
            np.nan
            if tau_acc is None
            else tau_acc
        ),

        tau_jerk=np.array(
            np.nan
            if tau_jerk is None
            else tau_jerk
        )
    )

    representative.append(
        (
            max(
                original[
                    "rms_acceleration"
                ]
                /
                max(
                    tau_acc
                    if tau_acc
                    is not None
                    else np.inf,
                    1e-12
                ),

                original[
                    "mean_squared_jerk"
                ]
                /
                max(
                    tau_jerk
                    if tau_jerk
                    is not None
                    else np.inf,
                    1e-12
                )
            ),
            key,
            original,
            corr_metrics,
            tau_acc,
            tau_jerk
        )
    )

    if number % 100 == 0:
        print(
            "processed",
            number,
            "/",
            len(loaded)
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
# 7. Representative local-timestep plots
# ------------------------------------------------------------

plot_dir = (
    OUT /
    "representative_local_plots"
)

plot_dir.mkdir(
    exist_ok=True
)

representative = sorted(
    representative,
    reverse=True,
    key=lambda x: x[0]
)[:10]


for rank, item in enumerate(
    representative,
    start=1
):

    (
        severity,
        key,
        original,
        corrected,
        ta,
        tj
    ) = item

    save_local_plot(
        key,
        original,
        corrected,
        ta,
        tj,
        plot_dir /
        f"{rank:02d}_local_acc.png"
    )


# ------------------------------------------------------------
# 8. Summary
# ------------------------------------------------------------

success_df = result_df[
    result_df[
        "success"
    ]
    ==
    True
].copy()


summary = {
    "target_N":
        int(
            len(
                target_df
            )
        ),

    "loaded_N":
        int(
            len(
                loaded
            )
        ),

    "correction_success_N":
        int(
            len(
                success_df
            )
        ),

    "correction_success_rate":
        float(
            len(
                success_df
            )
            /
            max(
                len(
                    loaded
                ),
                1
            )
        )
}


if len(
    success_df
):

    summary.update({
        "median_acc_ratio":
            float(
                success_df[
                    "acc_ratio"
                ].median()
            ),

        "median_msj_ratio":
            float(
                success_df[
                    "msj_ratio"
                ].median()
            ),

        "max_endpoint_error":
            float(
                success_df[
                    "endpoint_error"
                ].max()
            ),

        "median_path_length_ratio":
            float(
                success_df[
                    "path_length_ratio"
                ].median()
            ),

        "median_max_phase_shift":
            float(
                success_df[
                    "max_phase_shift"
                ].median()
            )
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


print()
print("=" * 100)
print("GOAL-PRESERVED CORRECTION SUMMARY")
print("=" * 100)

print(
    json.dumps(
        summary,
        indent=2
    )
)


if len(success_df):

    print()
    print("=" * 100)
    print("BY TASK")
    print("=" * 100)

    print(
        success_df.groupby(
            "task"
        )[
            [
                "acc_ratio",
                "msj_ratio",
                "endpoint_error",
                "path_length_ratio",
                "max_phase_shift"
            ]
        ]
        .median()
        .to_string()
    )


print()
print(
    "RESULT_DIR =",
    OUT
)

print(
    "SUCCESS = True"
)
