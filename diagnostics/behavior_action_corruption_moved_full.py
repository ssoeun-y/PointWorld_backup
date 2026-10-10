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
VARIANT = os.environ.get("PW_ACTION_VARIANT", "baseline")

VALID_VARIANTS = {
    "baseline",
    "speed_osc",
    "accel_spike",
    "reversal",
}

if VARIANT not in VALID_VARIANTS:
    raise RuntimeError(VARIANT)

ROOT = Path(__file__).resolve().parents[1]

META = ROOT / (
    "diagnostics/results/"
    "behavior_moved_test_inventory.csv"
)

OUT = ROOT / (
    "diagnostics/results/"
    f"action_corruption_moved_full_{VARIANT}.csv"
)


def reset_rng(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ================================================================
# Exact previously validated temporal maps
# ================================================================
def make_time_map(T, variant):
    t = np.linspace(0.0, 1.0, T, dtype=np.float64)

    if variant == "baseline":
        return t

    if variant == "speed_osc":
        s = t + 0.08 * np.sin(2.0 * np.pi * t)

    elif variant == "accel_spike":
        u = np.linspace(0.0, 1.0, 2001)

        v = 1.0 + 4.0 * np.exp(
            -0.5 * ((u - 0.55) / 0.07) ** 2
        )

        du = np.diff(u)

        integ = np.concatenate([
            [0.0],
            np.cumsum(
                0.5 * (v[:-1] + v[1:]) * du
            ),
        ])

        integ /= integ[-1]

        s = np.interp(t, u, integ)

    elif variant == "reversal":
        kt = np.array(
            [0.0, 0.20, 0.40, 0.55, 0.65, 0.75, 1.0]
        )

        ks = np.array(
            [0.0, 0.20, 0.45, 0.60, 0.45, 0.70, 1.0]
        )

        s = np.interp(t, kt, ks)

    else:
        raise AssertionError(variant)

    if abs(float(s[0])) > 1e-12:
        raise RuntimeError("t0 not preserved")

    if abs(float(s[-1]) - 1.0) > 1e-12:
        raise RuntimeError("final endpoint not preserved")

    return s


def resample_time(arr, s, normalize_vectors=False):
    arr = np.asarray(arr)

    T = arr.shape[0]
    u = np.linspace(0.0, 1.0, T)

    old_shape = arr.shape
    flat = arr.astype(np.float64).reshape(T, -1)

    out = np.empty_like(flat)

    for j in range(flat.shape[1]):
        out[:, j] = np.interp(
            s,
            u,
            flat[:, j],
        )

    out = out.reshape(old_shape).astype(arr.dtype)

    if normalize_vectors:
        norm = np.linalg.norm(
            out,
            axis=-1,
            keepdims=True,
        )
        out = out / np.maximum(norm, 1e-12)

    return out


# ================================================================
# Keep ORIGINAL official function.
# Only wrap Action construction.
# ================================================================
OFFICIAL_BUILD_FLOW_SAMPLE = dlmod.build_flow_sample


def build_flow_sample_action_corruption(
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

    # ------------------------------------------------------------
    # BASELINE = byte-for-byte official output.
    # Do NOT even interpolate baseline.
    # ------------------------------------------------------------
    if VARIANT == "baseline":
        return out

    T = out["robot_flows"].shape[0]
    s = make_time_map(T, VARIANT)

    original_robot = out["robot_flows"].copy()

    out["robot_flows"] = resample_time(
        out["robot_flows"],
        s,
    )

    # Endpoints MUST remain identical.
    if not np.allclose(
        out["robot_flows"][0],
        original_robot[0],
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError("robot t0 changed")

    if not np.allclose(
        out["robot_flows"][-1],
        original_robot[-1],
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError("robot final endpoint changed")

    if "robot_normals" in out:
        out["robot_normals"] = resample_time(
            out["robot_normals"],
            s,
            normalize_vectors=True,
        )

    for key in [
        "right_gripper_open",
        "left_gripper_open",
        "gripper_positions",
    ]:
        if key in out:
            arr = np.asarray(out[key])

            if arr.ndim >= 1 and arr.shape[0] == T:
                out[key] = resample_time(
                    arr,
                    s,
                )

    return out


dlmod.build_flow_sample = build_flow_sample_action_corruption


# ================================================================
# Official moved-sample inventory.
# Selection uses ONLY GT-derived PointWorld official masks.
# No prediction outcome is used.
# ================================================================
with META.open() as f:
    meta_rows = list(csv.DictReader(f))

TARGET_KEYS = sorted(
    r["clip"]
    for r in meta_rows
    if int(r["has_moved"]) == 1
)

if len(TARGET_KEYS) != 5288:
    raise RuntimeError(
        f"Expected 5288 official moved samples, got {len(TARGET_KEYS)}"
    )

print("variant =", VARIANT)
print("official moved target samples =", len(TARGET_KEYS))


def mask3(batch, name):
    x = batch[name].bool()

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise RuntimeError(
            f"{name}: {tuple(x.shape)}"
        )

    return x


def masked_mean(x, m):
    if not m.any():
        return float("nan")

    return float(
        x[m].float().mean().detach().cpu()
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

        if isinstance(raw_keys, (list, tuple)):
            keys = [str(x) for x in raw_keys]
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
                v.to(self.device, non_blocking=True)
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
        static = mask3(
            batch_gpu,
            "scene_static_mask",
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
                    f"duplicate key: {key}"
                )

            found.add(key)

            vi = valid[i]
            mi = moved[i]
            si = static[i]
            ei = err[i]

            row = {
                "clip": key,
                "variant": VARIANT,
                "l2_mm":
                    1000.0 * masked_mean(
                        ei,
                        vi,
                    ),
                "l2_moved_mm":
                    1000.0 * masked_mean(
                        ei,
                        vi & mi,
                    ),
                "l2_static_mm":
                    1000.0 * masked_mean(
                        ei,
                        vi & si,
                    ),
            }

            for t in range(T):
                row[f"l2_t{t}_mm"] = (
                    1000.0
                    * masked_mean(
                        ei[t],
                        vi[t],
                    )
                )

            results.append(row)

            print(
                f"[{len(found):02d}/{len(TARGET_KEYS)}] "
                f"{key} | "
                f"t10={row[f'l2_t{T-1}_mm']:.4f} mm"
            )

        if len(found) == len(TARGET_KEYS):
            break

    missing = set(TARGET_KEYS) - found

    if missing:
        raise RuntimeError(
            f"missing {len(missing)} samples"
        )

    results.sort(
        key=lambda x: x["clip"]
    )

    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print("saved =", OUT)
    print("SUCCESS = True")

    return {
        f"diagnostic/{split}/action_corruption":
            1.0
    }


Tester._accumulate_metrics = capture

reset_rng(SEED)

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
