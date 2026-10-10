#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import wilcoxon

# Needed because the already-validated paired script reads these at import.
os.environ.setdefault("PW_TARGET_TASKS", "task-0000")
os.environ.setdefault("PW_TAG", "tmp")

from diagnostics import eval_prediction_shift_paired as base
from arguments import parse_args
from evaluation.tester import Tester
import dataset_components.dataloader as dlmod


ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    x.strip()
    for x in os.environ["PW_TARGET_TASKS"].split(",")
    if x.strip()
}

TAG = os.environ["PW_TAG"]

# Small deterministic timing disturbance.
# Max shift = 0.25 frame.
PERTURB_FRAMES = float(
    os.environ.get("PW_PERTURB_FRAMES", "0.25")
)

CORR_ROOT = (
    ROOT
    / "diagnostics/results/"
      "goal_preserved_range_correction_v2_7987/"
      "corrected_actions"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/"
      "goal_preserved_action_robustness"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT = OUTDIR / f"{TAG}__robustness.csv"


# ============================================================
# Load successful corrected phases only
# ============================================================

PHASES = {}

for p in sorted(CORR_ROOT.glob("*.npz")):

    z = np.load(
        p,
        allow_pickle=True
    )

    clip = str(z["clip"].item())
    task = str(z["task"].item())

    if task not in TASKS:
        continue

    PHASES[clip] = np.asarray(
        z["phase"],
        dtype=np.float64
    )


if not PHASES:
    raise RuntimeError(
        f"No successful corrected actions "
        f"for {sorted(TASKS)}"
    )


TARGET_KEYS = set(PHASES)


# ============================================================
# Same small perturbation for original / corrected action
# ============================================================

def perturb_time_map(T):

    u = np.linspace(
        0.0,
        1.0,
        T
    )

    eps = (
        PERTURB_FRAMES
        /
        max(T - 1, 1)
    )

    # Smooth zero-endpoint time warp.
    q = (
        u
        +
        eps
        * np.sin(
            2.0 * np.pi * u
        )
    )

    q[0] = 0.0
    q[-1] = 1.0

    if np.any(
        np.diff(q) <= 0
    ):
        raise RuntimeError(
            f"Perturbation became non-monotonic "
            f"for T={T}"
        )

    return q


def compose_corrected_phase(
    corrected_phase,
    q
):

    T = len(corrected_phase)

    u = np.linspace(
        0.0,
        1.0,
        T
    )

    corrected_norm = (
        corrected_phase
        /
        float(T - 1)
    )

    # Apply the SAME execution-time perturbation q
    # around the corrected trajectory.
    out = np.interp(
        q,
        u,
        corrected_norm
    )

    out[0] = 0.0
    out[-1] = 1.0

    return out


# ============================================================
# Modify actual PointWorld robot action
# ============================================================

def apply_phase(
    out,
    phase_norm,
):

    T = out["robot_flows"].shape[0]

    phase_norm = np.asarray(
        phase_norm,
        dtype=np.float64
    )

    if len(phase_norm) != T:
        raise RuntimeError(
            "Phase length mismatch"
        )

    original = out[
        "robot_flows"
    ].copy()

    out["robot_flows"] = base.resample(
        out["robot_flows"],
        phase_norm
    )

    # Exact goal preservation.
    if not np.allclose(
        out["robot_flows"][0],
        original[0],
        atol=1e-7,
        rtol=0
    ):
        raise RuntimeError(
            f"{out.get('__key__')}: t0 changed"
        )

    if not np.allclose(
        out["robot_flows"][-1],
        original[-1],
        atol=1e-7,
        rtol=0
    ):
        raise RuntimeError(
            f"{out.get('__key__')}: endpoint changed"
        )

    if "robot_normals" in out:

        out["robot_normals"] = base.resample(
            out["robot_normals"],
            phase_norm,
            normalize=True
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

        if key not in TARGET_KEYS:
            return out

        T = out[
            "robot_flows"
        ].shape[0]

        u = np.linspace(
            0.0,
            1.0,
            T
        )

        # -------------------------------
        # Original action
        # -------------------------------
        if mode == "original":
            return out

        # -------------------------------
        # Original + small perturbation
        # -------------------------------
        if mode == "original_perturbed":

            q = perturb_time_map(T)

            return apply_phase(
                out,
                q
            )

        # -------------------------------
        # Corrected action
        # -------------------------------
        corr_phase = PHASES[key]

        if len(corr_phase) != T:
            raise RuntimeError(
                f"{key}: corrected phase "
                f"length {len(corr_phase)} != {T}"
            )

        corr_norm = (
            corr_phase
            /
            float(T - 1)
        )

        if mode == "corrected":

            return apply_phase(
                out,
                corr_norm
            )

        # -------------------------------
        # Corrected + SAME perturbation
        # -------------------------------
        if mode == "corrected_perturbed":

            q = perturb_time_map(T)

            pert_corr = (
                compose_corrected_phase(
                    corr_phase,
                    q
                )
            )

            return apply_phase(
                out,
                pert_corr
            )

        raise ValueError(mode)

    return builder


# ============================================================
# Helpers
# ============================================================

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


def to_device(batch, device):

    return {
        k: (
            v.to(
                device,
                non_blocking=True
            )
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    tester = Tester(args)

    tester.model.eval()


    # --------------------------------------------
    # Build four deterministic versions
    # --------------------------------------------

    loaders = {}

    modes = [
        "original",
        "original_perturbed",
        "corrected",
        "corrected_perturbed",
    ]

    for mode in modes:

        if mode == "original":
            dlmod.build_flow_sample = (
                base.OFFICIAL_BUILD
            )
        else:
            dlmod.build_flow_sample = (
                make_builder(mode)
            )

        base.reset()

        loader, _ = dlmod.build_dataloader(
            tester.args,
            mode="test",
            rank=tester.rank,
            world_size=tester.world_size,
            override_splits="test",
        )

        loaders[mode] = loader


    # Restore global function.
    dlmod.build_flow_sample = (
        base.OFFICIAL_BUILD
    )


    rows = []
    found = set()


    with torch.no_grad():

        for (
            b_orig,
            b_orig_p,
            b_corr,
            b_corr_p,
        ) in zip(
            loaders["original"],
            loaders["original_perturbed"],
            loaders["corrected"],
            loaders["corrected_perturbed"],
        ):

            keys = [
                str(b_orig["__key__"][0]),
                str(b_orig_p["__key__"][0]),
                str(b_corr["__key__"][0]),
                str(b_corr_p["__key__"][0]),
            ]

            if len(set(keys)) != 1:
                raise RuntimeError(
                    f"Loader mismatch: {keys}"
                )

            key = keys[0]

            if key not in TARGET_KEYS:
                continue


            # Same factual scene / correspondence.
            for other in [
                b_orig_p,
                b_corr,
                b_corr_p,
            ]:

                if not torch.allclose(
                    b_orig["gt_scene_flows"],
                    other["gt_scene_flows"],
                    atol=1e-7,
                    rtol=0,
                ):
                    raise RuntimeError(
                        f"{key}: GT scene changed"
                    )


            g0 = to_device(
                b_orig,
                tester.device
            )

            g0p = to_device(
                b_orig_p,
                tester.device
            )

            gc = to_device(
                b_corr,
                tester.device
            )

            gcp = to_device(
                b_corr_p,
                tester.device
            )


            # Deterministic forward runs.
            base.reset()
            p0 = tester.model(
                g0,
                training=False
            )["scene_flows"]

            base.reset()
            p0p = tester.model(
                g0p,
                training=False
            )["scene_flows"]

            base.reset()
            pc = tester.model(
                gc,
                training=False
            )["scene_flows"]

            base.reset()
            pcp = tester.model(
                gcp,
                training=False
            )["scene_flows"]


            # ----------------------------------------
            # LOCAL SENSITIVITY
            #
            # Original:
            #   P(A + delta) - P(A)
            #
            # Corrected:
            #   P(A* + delta) - P(A*)
            # ----------------------------------------

            sens_original = (
                torch.linalg.vector_norm(
                    p0p - p0,
                    dim=-1
                )
            )

            sens_corrected = (
                torch.linalg.vector_norm(
                    pcp - pc,
                    dim=-1
                )
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

            vm = (
                valid
                & moved
            )


            orig_all = (
                1000.0
                * masked_mean(
                    sens_original,
                    valid
                )
            )

            corr_all = (
                1000.0
                * masked_mean(
                    sens_corrected,
                    valid
                )
            )

            orig_moved = (
                1000.0
                * masked_mean(
                    sens_original,
                    vm
                )
            )

            corr_moved = (
                1000.0
                * masked_mean(
                    sens_corrected,
                    vm
                )
            )


            orig_t10 = (
                1000.0
                * masked_mean(
                    sens_original[:, -1],
                    vm[:, -1]
                )
            )

            corr_t10 = (
                1000.0
                * masked_mean(
                    sens_corrected[:, -1],
                    vm[:, -1]
                )
            )


            row = {
                "clip":
                    key,

                "perturb_frames":
                    PERTURB_FRAMES,

                "original_sensitivity_all_mm":
                    orig_all,

                "corrected_sensitivity_all_mm":
                    corr_all,

                "original_sensitivity_moved_mm":
                    orig_moved,

                "corrected_sensitivity_moved_mm":
                    corr_moved,

                "original_sensitivity_t10_moved_mm":
                    orig_t10,

                "corrected_sensitivity_t10_moved_mm":
                    corr_t10,
            }


            for label in [
                "all",
                "moved",
                "t10_moved",
            ]:

                o = row[
                    f"original_sensitivity_{label}_mm"
                ]

                c = row[
                    f"corrected_sensitivity_{label}_mm"
                ]

                row[
                    f"sensitivity_delta_{label}_mm"
                ] = (
                    c - o
                )

                row[
                    f"sensitivity_ratio_{label}"
                ] = (
                    c / o
                    if o > 1e-12
                    else np.nan
                )


            rows.append(row)

            found.add(key)


            print(
                f"[{len(found)}/{len(TARGET_KEYS)}] "
                f"{key} | "
                f"moved: "
                f"{orig_moved:.3f} -> "
                f"{corr_moved:.3f} mm | "
                f"ratio="
                f"{row['sensitivity_ratio_moved']:.3f}"
            )


    result = pd.DataFrame(rows)

    result.to_csv(
        OUT,
        index=False
    )


    print()
    print("=" * 90)
    print("ROBUSTNESS RESULT")
    print("=" * 90)

    print("TARGETS =", len(TARGET_KEYS))
    print("FOUND   =", len(found))
    print("OUTPUT  =", OUT)
    print("SUCCESS = True")


if __name__ == "__main__":
    main()
