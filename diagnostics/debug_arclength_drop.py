#!/usr/bin/env python3

import csv
import os
import random
import runpy
from pathlib import Path

import numpy as np
import torch

import dataset_components.dataloader as dlmod
from evaluation.tester import Tester


SEED = 42

ROOT = Path(__file__).resolve().parents[1]

COHORT = ROOT / (
    "diagnostics/results/"
    "intervention_high_msj_top10_4task_meta.csv"
)

TASKS = {
    x.strip()
    for x in os.environ["PW_TARGET_TASKS"].split(",")
    if x.strip()
}

TAG = os.environ["PW_TAG"]

OUTDIR = ROOT / (
    "diagnostics/results/"
    "action_intervention_633/prediction"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / f"{TAG}__arclength.csv"


def reset_rng(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Cohort: selected ONLY from input MSJ, not prediction outcome.
# ------------------------------------------------------------
import pandas as pd

meta = pd.read_csv(COHORT)

target_df = meta[
    meta["task"].isin(TASKS)
].copy()

TARGET_KEYS = set(target_df["clip"])

if not TARGET_KEYS:
    raise RuntimeError(
        f"No cohort clips for tasks={sorted(TASKS)}"
    )

print("TARGET TASKS =", sorted(TASKS))
print("TARGET N     =", len(TARGET_KEYS))


# ============================================================
# Arc-length / phase canonicalization
# ============================================================
def path_length(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)

    if len(xyz) < 2:
        return 0.0

    return float(
        np.linalg.norm(
            np.diff(xyz, axis=0),
            axis=-1,
        ).sum()
    )


def dominant_xyz(sample):
    candidates = []

    for side in ["right", "left"]:
        key = f"{side}_gripper_pose"

        if key not in sample:
            continue

        pose = np.asarray(
            sample[key],
            dtype=np.float64,
        )

        if pose.ndim != 2 or pose.shape[1] < 3:
            raise RuntimeError(
                f"{key}: unexpected shape {pose.shape}"
            )

        xyz = pose[:, :3]

        candidates.append(
            (path_length(xyz), side, xyz)
        )

    if not candidates:
        raise RuntimeError(
            "No valid left/right gripper pose"
        )

    # If both arms exist, use the arm with the longer path.
    # If only one exists, use that arm.
    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return candidates[0][2]


def arclength_time_map(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)

    T = len(xyz)

    u = np.linspace(
        0.0,
        1.0,
        T,
        dtype=np.float64,
    )

    if T < 2:
        return u

    seg = np.linalg.norm(
        np.diff(xyz, axis=0),
        axis=-1,
    )

    cum = np.concatenate([
        [0.0],
        np.cumsum(seg),
    ])

    total = float(cum[-1])

    if total <= 1e-12:
        return u

    phase = cum / total

    phase_unique, idx = np.unique(
        phase,
        return_index=True,
    )

    u_unique = u[idx].copy()

    if len(phase_unique) < 2:
        return u

    u_unique[0] = 0.0
    u_unique[-1] = 1.0

    desired_phase = np.linspace(
        0.0,
        1.0,
        T,
    )

    s = np.interp(
        desired_phase,
        phase_unique,
        u_unique,
    )

    s[0] = 0.0
    s[-1] = 1.0

    if np.any(np.diff(s) < -1e-12):
        raise RuntimeError(
            "Non-monotonic phase map"
        )

    return s


def resample_time(
    arr,
    s,
    normalize_vectors=False,
):
    arr = np.asarray(arr)

    T = arr.shape[0]

    if T != len(s):
        raise RuntimeError(
            f"time mismatch {T} vs {len(s)}"
        )

    u = np.linspace(0.0, 1.0, T)

    original_shape = arr.shape

    flat = (
        arr.astype(np.float64)
        .reshape(T, -1)
    )

    out = np.empty_like(flat)

    for j in range(flat.shape[1]):
        out[:, j] = np.interp(
            s,
            u,
            flat[:, j],
        )

    out = (
        out.reshape(original_shape)
        .astype(arr.dtype)
    )

    if normalize_vectors:
        norm = np.linalg.norm(
            out,
            axis=-1,
            keepdims=True,
        )

        out = out / np.maximum(
            norm,
            1e-12,
        )

    return out


# ============================================================
# Wrap official Action construction.
# ============================================================
OFFICIAL_BUILD_FLOW_SAMPLE = (
    dlmod.build_flow_sample
)


def build_flow_sample_arclength(
    sample,
    domain,
    robot_sampler,
    max_robot_points,
    deterministic=False,
    seed=None,
    force_single_arm=False,
):
    out = OFFICIAL_BUILD_FLOW_SAMPLE(
        sample,
        domain=domain,
        robot_sampler=robot_sampler,
        max_robot_points=max_robot_points,
        deterministic=deterministic,
        seed=seed,
        force_single_arm=force_single_arm,
    )

    key = str(out.get("__key__", ""))

    # Absolutely no modification outside selected cohort.
    if key not in TARGET_KEYS:
        return out

    xyz = dominant_xyz(out)

    T = out["robot_flows"].shape[0]

    if len(xyz) != T:
        raise RuntimeError(
            f"{key}: EEF/robot T mismatch"
        )

    s = arclength_time_map(xyz)

    original_robot = (
        out["robot_flows"].copy()
    )

    out["robot_flows"] = resample_time(
        out["robot_flows"],
        s,
    )

    # Exact endpoints.
    if not np.allclose(
        out["robot_flows"][0],
        original_robot[0],
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError(
            f"{key}: t0 changed"
        )

    if not np.allclose(
        out["robot_flows"][-1],
        original_robot[-1],
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError(
            f"{key}: final changed"
        )

    if "robot_normals" in out:
        out["robot_normals"] = (
            resample_time(
                out["robot_normals"],
                s,
                normalize_vectors=True,
            )
        )

    for name in [
        "right_gripper_open",
        "left_gripper_open",
        "gripper_positions",
    ]:
        if name not in out:
            continue

        arr = np.asarray(out[name])

        if (
            arr.ndim >= 1
            and arr.shape[0] == T
        ):
            out[name] = resample_time(
                arr,
                s,
            )

    return out


dlmod.build_flow_sample = (
    build_flow_sample_arclength
)


# ============================================================
# Official moved-L2 capture
# ============================================================
def mask3(batch, name):
    x = batch[name].bool()

    if (
        x.ndim == 4
        and x.shape[-1] == 1
    ):
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise RuntimeError(
            f"{name}: {tuple(x.shape)}"
        )

    return x


def masked_mean(x, mask):
    if not mask.any():
        return float("nan")

    return float(
        x[mask]
        .float()
        .mean()
        .detach()
        .cpu()
    )


results = []


@torch.no_grad()
def capture(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):
    self.model.eval()

    found = set()

    for batch in dl:

        raw_keys = batch["__key__"]

        if isinstance(
            raw_keys,
            (list, tuple),
        ):
            keys = [
                str(x)
                for x in raw_keys
            ]
        else:
            keys = [str(raw_keys)]

        selected = [
            i
            for i, key in enumerate(keys)
            if key in TARGET_KEYS
        ]

        if not selected:
            continue

        batch_gpu = {
            k: (
                v.to(
                    self.device,
                    non_blocking=True,
                )
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

        outputs = self.model(
            batch_gpu,
            training=False,
        )

        pred = outputs["scene_flows"]
        gt = batch_gpu["gt_scene_flows"]

        err = torch.linalg.vector_norm(
            pred - gt,
            dim=-1,
        )

        exists = mask3(
            batch_gpu,
            "scene_exists",
        )

        context = mask3(
            batch_gpu,
            "scene_context_mask",
        )

        supervised = mask3(
            batch_gpu,
            "scene_supervised_mask",
        )

        moved = mask3(
            batch_gpu,
            "scene_moved_mask",
        )

        valid = (
            (~context)
            & exists
            & supervised
        )

        T = err.shape[1]

        for i in selected:
            key = keys[i]

            if key in found:
                raise RuntimeError(
                    f"duplicate {key}"
                )

            found.add(key)

            ei = err[i]
            vi = valid[i]
            mi = moved[i]

            row = {
                "clip": key,
                "variant": "arclength",
                "l2_moved_mm":
                    1000.0
                    * masked_mean(
                        ei,
                        vi & mi,
                    ),
            }

            for t in range(T):
                row[
                    f"l2_t{t}_mm"
                ] = (
                    1000.0
                    * masked_mean(
                        ei[t],
                        vi[t],
                    )
                )

            results.append(row)

            print(
                f"[{len(found)}/"
                f"{len(TARGET_KEYS)}] "
                f"{key} | "
                f"moved="
                f"{row['l2_moved_mm']:.4f} mm"
            )

        if len(found) == len(
            TARGET_KEYS
        ):
            break

    missing = (
        TARGET_KEYS - found
    )

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} "
            f"target clips"
        )

    results.sort(
        key=lambda x: x["clip"]
    )

    with OUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    print("saved =", OUT)
    print("SUCCESS = True")

    return {
        f"diagnostic/{split}/"
        "arclength_capture":
        float(len(found))
    }


Tester._accumulate_metrics = capture

reset_rng(SEED)

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
