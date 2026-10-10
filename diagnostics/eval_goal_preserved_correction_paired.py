#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np

# eval_prediction_shift_paired.py requires these at import time
os.environ.setdefault("PW_TARGET_TASKS", "task-0000")
os.environ.setdefault("PW_TAG", "temporary")

from diagnostics import eval_prediction_shift_paired as base


ROOT = Path(__file__).resolve().parents[1]

CORR_ROOT = (
    ROOT
    / "diagnostics/results/"
      "goal_preserved_range_correction_v2_7987/"
      "corrected_actions"
)

TASKS = {
    x.strip()
    for x in os.environ["PW_TARGET_TASKS"].split(",")
    if x.strip()
}

TAG = os.environ["PW_TAG"]


# ============================================================
# Load ONLY successful goal-preserved corrections.
# Each npz exists only for successful correction cases.
# ============================================================

PHASES = {}

for p in sorted(CORR_ROOT.glob("*.npz")):

    z = np.load(p, allow_pickle=True)

    clip = str(z["clip"].item())
    task = str(z["task"].item())

    if task not in TASKS:
        continue

    phase = np.asarray(
        z["phase"],
        dtype=np.float64
    )

    PHASES[clip] = phase


if not PHASES:
    raise RuntimeError(
        f"No corrected phases found for tasks={sorted(TASKS)}"
    )


# Override the target population of the old,
# already-validated paired inference script.
base.TARGET_KEYS = set(PHASES.keys())


OUTDIR = (
    ROOT
    / "diagnostics/results/"
      "goal_preserved_correction_prediction_shift"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

base.OUTDIR = OUTDIR
base.OUT = OUTDIR / f"{TAG}__pred_shift.csv"


# ============================================================
# Correction hook
#
# IMPORTANT:
# This runs immediately after official build_flow_sample().
# Later in the official pipeline, gather_features() recomputes:
#   robot_velocity
#   robot_acceleration
#   dist2robot
# from the corrected robot_flows.
# ============================================================

def corrected_build(
    sample,
    domain,
    robot_sampler,
    max_robot_points,
    deterministic=False,
    seed=None,
    force_single_arm=False,
):

    out = base.OFFICIAL_BUILD(
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

    if key not in PHASES:
        return out

    phase = PHASES[key]

    T = out["robot_flows"].shape[0]

    if len(phase) != T:
        raise RuntimeError(
            f"{key}: phase length {len(phase)} != T {T}"
        )

    original = out["robot_flows"].copy()

    # Same goal-preserving temporal redistribution
    # discovered in the correction experiment.
    out["robot_flows"] = base.resample(
        out["robot_flows"],
        phase / float(T - 1),
    )

    # Exact start / endpoint preservation audit.
    if not np.allclose(
        out["robot_flows"][0],
        original[0],
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError(
            f"{key}: corrected t0 changed"
        )

    if not np.allclose(
        out["robot_flows"][-1],
        original[-1],
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError(
            f"{key}: corrected endpoint changed"
        )

    # Keep time-dependent robot quantities synchronized.
    if "robot_normals" in out:
        out["robot_normals"] = base.resample(
            out["robot_normals"],
            phase / float(T - 1),
            normalize=True,
        )

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
            out[name] = base.resample(
                a,
                phase / float(T - 1),
            )

    return out


base.corrected_build = corrected_build


print("=" * 90)
print("GOAL-PRESERVED CORRECTION → POINTWORLD")
print("=" * 90)
print("TASKS =", sorted(TASKS))
print("SUCCESSFUL CORRECTED TARGETS =", len(PHASES))
print("OUTPUT =", base.OUT)
print()


base.main()
