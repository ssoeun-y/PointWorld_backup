#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq

os.environ.setdefault("PW_TARGET_TASKS", "task-0000")
os.environ.setdefault("PW_TAG", "tmp")

from arguments import parse_args
from evaluation.tester import Tester
import dataset_components.dataloader as dlmod

from diagnostics import eval_prediction_shift_paired as old
from diagnostics.run_established_action_metrics_5task import (
    established_metrics_for_clip,
)


ROOT = Path(__file__).resolve().parents[1]

TASK = os.environ["PW_TARGET_TASKS"].strip()
TAG = os.environ["PW_TAG"]
CONDITION = os.environ["PW_CONDITION"].strip()

# Local physical perturbation:
# target delta = 5% of that task's validated boundary.
DELTA_FRAC = float(
    os.environ.get("PW_DELTA_FRAC", "0.05")
)

# Search only over small monotonic temporal changes.
MAX_PERTURB_FRAMES = float(
    os.environ.get("PW_MAX_PERTURB_FRAMES", "1.25")
)

GRID_N = int(
    os.environ.get("PW_GRID_N", "161")
)

# Actual physical delta must be within 5% of requested delta.
MATCH_TOL = float(
    os.environ.get("PW_MATCH_TOL", "0.05")
)


COMMON = pd.read_csv(
    ROOT
    / "diagnostics/results/"
      "goal_preserved_margin_common_success.csv"
)

TARGET_KEYS = {
    str(x)
    for x in COMMON["clip"]
    if str(x).startswith(TASK + "_")
}

if not TARGET_KEYS:
    raise RuntimeError(
        f"No common-success clips for {TASK}"
    )


# ============================================================
# Load validated ORIGINAL boundaries
# ============================================================

BDF = pd.read_csv(
    ROOT
    / "diagnostics/results/"
      "goal_preserved_range_correction_v2_7987/"
      "taskwise_boundaries.csv"
)

BDF = BDF[
    BDF["usable_upper_boundary"]
    .astype(str)
    .str.lower()
    .eq("true")
]

BOUNDARY = {
    (str(r["task"]), str(r["metric"])):
        float(r["threshold"])
    for _, r in BDF.iterrows()
}

TA = BOUNDARY[
    (TASK, "rms_acceleration")
]

TJ = BOUNDARY[
    (TASK, "mean_squared_jerk")
]

TARGET_DELTA_ACC = DELTA_FRAC * TA
TARGET_DELTA_MSJ = DELTA_FRAC * TJ


# ============================================================
# Load phase for requested margin
# ============================================================

DIRS = {
    "100":
        ROOT / "diagnostics/results/"
        "goal_preserved_range_correction_v2_7987/"
        "corrected_actions",

    "90":
        ROOT / "diagnostics/results/"
        "goal_preserved_range_correction_margin90_7987/"
        "corrected_actions",

    "80":
        ROOT / "diagnostics/results/"
        "goal_preserved_range_correction_margin80_7987/"
        "corrected_actions",

    "70":
        ROOT / "diagnostics/results/"
        "goal_preserved_range_correction_margin70_7987/"
        "corrected_actions",
}


PHASES = {}

# Diagnostic metadata cannot be passed through the official
# PointWorld release pipeline because gather(final_fields)
# intentionally drops unknown sample fields.
# Keep equal-delta diagnostics separately, keyed by clip.
EQUAL_DELTA_META = {
    "acc": {},
    "jerk": {},
}

if CONDITION != "original":

    if CONDITION not in DIRS:
        raise RuntimeError(
            f"Unknown condition: {CONDITION}"
        )

    for p in DIRS[CONDITION].glob("*.npz"):

        z = np.load(
            p,
            allow_pickle=True
        )

        clip = str(
            z["clip"].item()
        )

        if clip not in TARGET_KEYS:
            continue

        PHASES[clip] = np.asarray(
            z["phase"],
            dtype=np.float64
        )

    missing = TARGET_KEYS - set(PHASES)

    if missing:
        raise RuntimeError(
            f"{CONDITION}: missing "
            f"{len(missing)} common phases"
        )


# ============================================================
# Temporal utilities
# ============================================================

def base_phase(key, T):

    if CONDITION == "original":
        return np.linspace(
            0.0, 1.0, T
        )

    p = PHASES[key]

    if len(p) != T:
        raise RuntimeError(
            f"{key}: phase length mismatch"
        )

    return p / float(T - 1)


def perturb_map(T, amp_frames):

    u = np.linspace(
        0.0, 1.0, T
    )

    eps = (
        amp_frames
        / float(T - 1)
    )

    q = (
        u
        + eps
        * np.sin(
            2.0 * np.pi * u
        )
    )

    q[0] = 0.0
    q[-1] = 1.0

    if np.any(
        np.diff(q) <= 1e-8
    ):
        return None

    return q


def compose(base, q):

    T = len(base)

    u = np.linspace(
        0.0, 1.0, T
    )

    out = np.interp(
        q,
        u,
        base
    )

    out[0] = 0.0
    out[-1] = 1.0

    return out


def pose_metrics(
    sample,
    phase_norm,
):

    rp = old.resample(
        np.asarray(
            sample["right_gripper_pose"]
        ),
        phase_norm,
    )

    lp = old.resample(
        np.asarray(
            sample["left_gripper_pose"]
        ),
        phase_norm,
    )

    return established_metrics_for_clip(
        rp,
        lp,
    )


def search_equal_delta(
    sample,
    base,
    target_metric,
    collateral_metric,
    requested_delta,
):
    """
    Match a pre-specified positive physical delta without
    changing the experimental target, tolerance, or locality bound.

    1) Scan the fixed [-MAX_PERTURB_FRAMES, +MAX_PERTURB_FRAMES]
       sinusoidal perturbation family.
    2) If the requested physical delta is crossed between two
       neighboring amplitudes, refine that interval continuously
       with Brent root finding.
    3) If no crossing exists inside the fixed range, retain the
       closest feasible candidate and mark it unmatched unless it
       already satisfies MATCH_TOL.
    """

    base_m = pose_metrics(
        sample,
        base
    )

    base_value = float(
        base_m[target_metric]
    )

    base_collateral = float(
        base_m[collateral_metric]
    )

    T = len(base)

    cache = {}

    def evaluate(amp):
        amp = float(amp)

        # Stable cache key for repeated Brent evaluations.
        ck = round(amp, 14)

        if ck in cache:
            return cache[ck]

        q = perturb_map(
            T,
            amp
        )

        if q is None:
            cache[ck] = None
            return None

        candidate_phase = compose(
            base,
            q
        )

        m = pose_metrics(
            sample,
            candidate_phase
        )

        value = float(
            m[target_metric]
        )

        delta = (
            value
            - base_value
        )

        collateral_delta = (
            float(m[collateral_metric])
            - base_collateral
        )

        relerr = abs(
            delta - requested_delta
        ) / max(
            requested_delta,
            1e-15
        )

        out = {
            "phase":
                candidate_phase,

            "amp_frames":
                amp,

            "actual_delta":
                float(delta),

            "relative_match_error":
                float(relerr),

            "collateral_delta":
                float(collateral_delta),
        }

        cache[ck] = out
        return out


    # --------------------------------------------------------
    # Fixed-range coarse scan: only used to find crossings.
    # --------------------------------------------------------
    amps = np.linspace(
        -MAX_PERTURB_FRAMES,
        MAX_PERTURB_FRAMES,
        GRID_N,
    )

    scanned = []

    best = None

    for amp in amps:

        r = evaluate(amp)

        if r is None:
            continue

        delta = r["actual_delta"]

        if not np.isfinite(delta):
            continue

        objective = (
            delta
            - requested_delta
        )

        scanned.append(
            (
                float(amp),
                float(objective),
            )
        )

        # Preserve the original positive-delta requirement.
        if delta <= 0:
            continue

        if (
            best is None
            or
            r["relative_match_error"]
            <
            best["relative_match_error"]
        ):
            best = r


    # --------------------------------------------------------
    # Continuous refinement wherever target delta is crossed.
    # --------------------------------------------------------
    roots = []

    for (
        a0,
        f0,
    ), (
        a1,
        f1,
    ) in zip(
        scanned[:-1],
        scanned[1:],
    ):

        if not (
            np.isfinite(f0)
            and
            np.isfinite(f1)
        ):
            continue

        if f0 == 0.0:
            rr = evaluate(a0)

            if (
                rr is not None
                and
                rr["actual_delta"] > 0
            ):
                roots.append(rr)

            continue

        if f1 == 0.0:
            rr = evaluate(a1)

            if (
                rr is not None
                and
                rr["actual_delta"] > 0
            ):
                roots.append(rr)

            continue

        # No crossing.
        if f0 * f1 > 0:
            continue

        def objective_fn(a):
            rr = evaluate(a)

            if rr is None:
                raise ValueError(
                    "invalid perturbation during root finding"
                )

            return (
                rr["actual_delta"]
                - requested_delta
            )

        try:
            root = brentq(
                objective_fn,
                a0,
                a1,
                xtol=1e-10,
                rtol=1e-10,
                maxiter=100,
            )
        except (ValueError, RuntimeError):
            continue

        rr = evaluate(root)

        if (
            rr is not None
            and
            rr["actual_delta"] > 0
        ):
            roots.append(rr)


    # Exact/refined roots take priority.
    if roots:
        roots.sort(
            key=lambda x: (
                x["relative_match_error"],
                abs(x["amp_frames"]),
            )
        )

        root_best = roots[0]

        if (
            best is None
            or
            root_best["relative_match_error"]
            <
            best["relative_match_error"]
        ):
            best = root_best


    # --------------------------------------------------------
    # No positive physical increase possible in fixed range.
    # --------------------------------------------------------
    if best is None:

        return {
            "phase":
                base,

            "amp_frames":
                0.0,

            "actual_delta":
                0.0,

            "relative_match_error":
                np.inf,

            "collateral_delta":
                0.0,

            "matched":
                False,
        }


    best["matched"] = bool(
        best[
            "relative_match_error"
        ]
        <= MATCH_TOL
    )

    return best


# ============================================================
# PointWorld action replacement
# ============================================================

def apply_phase(
    out,
    phase_norm,
):

    original = out[
        "robot_flows"
    ].copy()

    out["robot_flows"] = old.resample(
        out["robot_flows"],
        phase_norm
    )

    if not np.allclose(
        out["robot_flows"][0],
        original[0],
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError(
            f"{out.get('__key__')}: start changed"
        )

    if not np.allclose(
        out["robot_flows"][-1],
        original[-1],
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError(
            f"{out.get('__key__')}: endpoint changed"
        )

    if "robot_normals" in out:

        out["robot_normals"] = old.resample(
            out["robot_normals"],
            phase_norm,
            normalize=True,
        )

    T = len(phase_norm)

    for name in [
        "right_gripper_open",
        "left_gripper_open",
        "gripper_positions",
    ]:

        if name not in out:
            continue

        a = np.asarray(
            out[name]
        )

        if (
            a.ndim >= 1
            and
            a.shape[0] == T
        ):

            out[name] = old.resample(
                a,
                phase_norm
            )

    return out


def make_builder(mode):

    def builder(
        sample,
        domain,
        robot_sampler,
        max_robot_points,
        deterministic=False,
        seed=None,
        force_single_arm=False,
    ):

        raw_key = str(
            sample.get("__key__", "")
        )

        metric_sample = None

        if raw_key in TARGET_KEYS:
            if (
                "right_gripper_pose" not in sample
                or "left_gripper_pose" not in sample
            ):
                raise RuntimeError(
                    f"{raw_key}: raw gripper pose missing"
                )

            metric_sample = {
                "right_gripper_pose":
                    np.asarray(
                        sample["right_gripper_pose"]
                    ).copy(),

                "left_gripper_pose":
                    np.asarray(
                        sample["left_gripper_pose"]
                    ).copy(),
            }

        out = old.OFFICIAL_BUILD(
            sample,
            domain=domain,
            robot_sampler=robot_sampler,
            max_robot_points=max_robot_points,
            deterministic=deterministic,
            seed=seed,
            force_single_arm=force_single_arm,
        )

        key = str(
            out.get("__key__", "")
        )

        if key not in TARGET_KEYS:
            return out

        if metric_sample is None:
            raise RuntimeError(
                f"{key}: raw metric sample missing"
            )

        if key != raw_key:
            raise RuntimeError(
                f"{key}: raw/built key mismatch "
                f"(raw={raw_key})"
            )


        T = out[
            "robot_flows"
        ].shape[0]

        bp = base_phase(
            key,
            T
        )


        if mode == "base":

            return apply_phase(
                out,
                bp
            )


        if mode == "acc":

            r = search_equal_delta(
                metric_sample,
                bp,
                target_metric=
                    "rms_acceleration",

                collateral_metric=
                    "mean_squared_jerk",

                requested_delta=
                    TARGET_DELTA_ACC,
            )

            EQUAL_DELTA_META["acc"][key] = {
                "actual_delta": float(r["actual_delta"]),
                "relative_match_error": float(
                    r["relative_match_error"]
                ),
                "matched": bool(r["matched"]),
                "collateral_delta": float(
                    r["collateral_delta"]
                ),
                "amp_frames": float(r["amp_frames"]),
            }

            out = apply_phase(
                out,
                r["phase"]
            )

            out[
                "equal_delta_actual"
            ] = np.array(
                r["actual_delta"],
                dtype=np.float32,
            )

            out[
                "equal_delta_relerr"
            ] = np.array(
                r[
                    "relative_match_error"
                ],
                dtype=np.float32,
            )

            out[
                "equal_delta_matched"
            ] = np.array(
                int(r["matched"]),
                dtype=np.int64,
            )

            out[
                "equal_delta_collateral"
            ] = np.array(
                r[
                    "collateral_delta"
                ],
                dtype=np.float32,
            )

            out[
                "equal_delta_amp_frames"
            ] = np.array(
                r["amp_frames"],
                dtype=np.float32,
            )

            return out


        if mode == "jerk":

            r = search_equal_delta(
                metric_sample,
                bp,
                target_metric=
                    "mean_squared_jerk",

                collateral_metric=
                    "rms_acceleration",

                requested_delta=
                    TARGET_DELTA_MSJ,
            )


            EQUAL_DELTA_META["jerk"][key] = {
                "actual_delta": float(r["actual_delta"]),
                "relative_match_error": float(
                    r["relative_match_error"]
                ),
                "matched": bool(r["matched"]),
                "collateral_delta": float(
                    r["collateral_delta"]
                ),
                "amp_frames": float(r["amp_frames"]),
            }

            out = apply_phase(
                out,
                r["phase"]
            )

            out[
                "equal_delta_actual"
            ] = np.array(
                r["actual_delta"],
                dtype=np.float32,
            )

            out[
                "equal_delta_relerr"
            ] = np.array(
                r[
                    "relative_match_error"
                ],
                dtype=np.float32,
            )

            out[
                "equal_delta_matched"
            ] = np.array(
                int(r["matched"]),
                dtype=np.int64,
            )

            out[
                "equal_delta_collateral"
            ] = np.array(
                r[
                    "collateral_delta"
                ],
                dtype=np.float32,
            )

            out[
                "equal_delta_amp_frames"
            ] = np.array(
                r["amp_frames"],
                dtype=np.float32,
            )

            return out


        raise ValueError(mode)

    return builder


def mask3(batch, name):

    x = batch[name].bool()

    if (
        x.ndim == 4
        and
        x.shape[-1] == 1
    ):
        x = x.squeeze(-1)

    return x


def masked_mean(x, m):

    if not m.any():
        return float("nan")

    return float(
        x[m]
        .float()
        .mean()
        .cpu()
    )


def scalar(batch, name):

    x = batch[name]

    if isinstance(
        x,
        torch.Tensor
    ):
        return float(
            x.reshape(-1)[0]
            .cpu()
        )

    return float(
        np.asarray(x)
        .reshape(-1)[0]
    )


def to_device(batch, device):

    return {
        k: (
            v.to(
                device,
                non_blocking=True
            )
            if isinstance(
                v,
                torch.Tensor
            )
            else v
        )
        for k, v in batch.items()
    }


def main():

    args = parse_args()

    tester = Tester(args)

    tester.model.eval()

    loaders = {}

    for mode in [
        "base",
        "acc",
        "jerk",
    ]:

        dlmod.build_flow_sample = (
            make_builder(mode)
        )

        old.reset()

        loader, _ = dlmod.build_dataloader(
            tester.args,
            mode="test",
            rank=tester.rank,
            world_size=tester.world_size,
            override_splits="test",
        )

        loaders[mode] = loader


    dlmod.build_flow_sample = (
        old.OFFICIAL_BUILD
    )


    rows = []
    found = set()


    with torch.no_grad():

        for b0, ba, bj in zip(
            loaders["base"],
            loaders["acc"],
            loaders["jerk"],
        ):

            keys = [
                str(b0["__key__"][0]),
                str(ba["__key__"][0]),
                str(bj["__key__"][0]),
            ]

            if len(set(keys)) != 1:

                raise RuntimeError(
                    f"loader mismatch {keys}"
                )

            key = keys[0]

            if key not in TARGET_KEYS:
                continue


            for other in [ba, bj]:

                if not torch.allclose(
                    b0["gt_scene_flows"],
                    other[
                        "gt_scene_flows"
                    ],
                    atol=1e-7,
                    rtol=0,
                ):
                    raise RuntimeError(
                        f"{key}: GT changed"
                    )


            acc_meta = EQUAL_DELTA_META["acc"].get(key)

            if acc_meta is None:
                raise RuntimeError(
                    f"{key}: missing equal-DeltaAcc metadata"
                )

            acc_actual = acc_meta["actual_delta"]
            acc_relerr = acc_meta["relative_match_error"]
            acc_matched = acc_meta["matched"]
            acc_collateral = acc_meta["collateral_delta"]
            acc_amp = acc_meta["amp_frames"]


            jerk_meta = EQUAL_DELTA_META["jerk"].get(key)

            if jerk_meta is None:
                raise RuntimeError(
                    f"{key}: missing equal-DeltaJerk metadata"
                )

            jerk_actual = jerk_meta["actual_delta"]
            jerk_relerr = jerk_meta["relative_match_error"]
            jerk_matched = jerk_meta["matched"]
            jerk_collateral = jerk_meta["collateral_delta"]
            jerk_amp = jerk_meta["amp_frames"]


            g0 = to_device(
                b0,
                tester.device
            )

            ga = to_device(
                ba,
                tester.device
            )

            gj = to_device(
                bj,
                tester.device
            )


            old.reset()

            p0 = tester.model(
                g0,
                training=False
            )["scene_flows"]


            old.reset()

            pa = tester.model(
                ga,
                training=False
            )["scene_flows"]


            old.reset()

            pj = tester.model(
                gj,
                training=False
            )["scene_flows"]


            sa = torch.linalg.vector_norm(
                pa - p0,
                dim=-1
            )

            sj = torch.linalg.vector_norm(
                pj - p0,
                dim=-1
            )


            exists = mask3(
                g0,
                "scene_exists"
            )

            context = mask3(
                g0,
                "scene_context_mask"
            )

            supervised = mask3(
                g0,
                "scene_supervised_mask"
            )

            moved = mask3(
                g0,
                "scene_moved_mask"
            )

            valid = (
                (~context)
                & exists
                & supervised
            )

            vm = valid & moved


            row = {
                "clip": key,
                "task": TASK,
                "condition": CONDITION,

                "requested_delta_acc":
                    TARGET_DELTA_ACC,

                "actual_delta_acc":
                    acc_actual,

                "acc_match_relerr":
                    acc_relerr,

                "acc_matched":
                    acc_matched,

                # ΔMSJ produced during
                # the equal-ΔAcc experiment.
                "acc_collateral_delta_msj":
                    acc_collateral,

                "acc_perturb_frames":
                    acc_amp,

                "requested_delta_msj":
                    TARGET_DELTA_MSJ,

                "actual_delta_msj":
                    jerk_actual,

                "jerk_match_relerr":
                    jerk_relerr,

                "jerk_matched":
                    jerk_matched,

                # ΔAcc produced during
                # the equal-ΔJerk experiment.
                "jerk_collateral_delta_acc":
                    jerk_collateral,

                "jerk_perturb_frames":
                    jerk_amp,

                "acc_sensitivity_all_mm":
                    1000.0
                    * masked_mean(
                        sa,
                        valid
                    ),

                "acc_sensitivity_moved_mm":
                    1000.0
                    * masked_mean(
                        sa,
                        vm
                    ),

                "acc_sensitivity_t10_moved_mm":
                    1000.0
                    * masked_mean(
                        sa[:, -1],
                        vm[:, -1]
                    ),

                "jerk_sensitivity_all_mm":
                    1000.0
                    * masked_mean(
                        sj,
                        valid
                    ),

                "jerk_sensitivity_moved_mm":
                    1000.0
                    * masked_mean(
                        sj,
                        vm
                    ),

                "jerk_sensitivity_t10_moved_mm":
                    1000.0
                    * masked_mean(
                        sj[:, -1],
                        vm[:, -1]
                    ),
            }

            rows.append(row)

            found.add(key)

            print(
                f"[{len(found)}/"
                f"{len(TARGET_KEYS)}] "
                f"{key} | "
                f"cond={CONDITION} | "
                f"Acc match="
                f"{acc_relerr:.3f} "
                f"sens="
                f"{row['acc_sensitivity_moved_mm']:.3f} | "
                f"Jerk match="
                f"{jerk_relerr:.3f} "
                f"sens="
                f"{row['jerk_sensitivity_moved_mm']:.3f}"
            )


    OUTDIR = (
        ROOT
        / "diagnostics/results/"
          "equal_physical_delta_margin"
    )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT = (
        OUTDIR
        / f"{TAG}__equal_delta.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        OUT,
        index=False
    )


    print()
    print("=" * 90)
    print("EQUAL PHYSICAL DELTA RESULT")
    print("=" * 90)
    print("TASK =", TASK)
    print("CONDITION =", CONDITION)
    print("TARGETS =", len(TARGET_KEYS))
    print("FOUND =", len(found))
    print(
        "requested ΔAcc =",
        TARGET_DELTA_ACC
    )
    print(
        "requested ΔMSJ =",
        TARGET_DELTA_MSJ
    )
    print("OUTPUT =", OUT)
    print("SUCCESS = True")


if __name__ == "__main__":
    main()
