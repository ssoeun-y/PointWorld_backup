#!/usr/bin/env python3
import os, random, csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arguments import parse_args
from evaluation.tester import Tester
import dataset_components.dataloader as dlmod

ROOT = Path(__file__).resolve().parents[1]

POP = pd.read_csv(
    ROOT / "diagnostics/results/established_action_metrics_5task/"
           "per_sample_established_metrics_7987.csv"
)

TASKS = {
    x.strip()
    for x in os.environ["PW_TARGET_TASKS"].split(",")
    if x.strip()
}
TAG = os.environ["PW_TAG"]

TARGET_KEYS = set(
    POP.loc[POP["task"].isin(TASKS), "clip"]
)

OUTDIR = (
    ROOT / "diagnostics/results/"
           "action_prediction_sensitivity_all7987"
)
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / f"{TAG}__pred_shift.csv"

OFFICIAL_BUILD = dlmod.build_flow_sample


def reset(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def path_length(xyz):
    xyz = np.asarray(xyz, np.float64)
    if len(xyz) < 2:
        return 0.0
    return float(
        np.linalg.norm(np.diff(xyz, axis=0), axis=-1).sum()
    )


def dominant_xyz(sample):
    cand = []

    for side in ["right", "left"]:
        key = f"{side}_gripper_pose"
        if key not in sample:
            continue

        pose = np.asarray(sample[key], np.float64)
        xyz = pose[:, :3]
        cand.append((path_length(xyz), xyz))

    if not cand:
        raise RuntimeError(
            f"{sample.get('__key__')}: no gripper pose"
        )

    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][1]


def arc_map(xyz):
    xyz = np.asarray(xyz, np.float64)
    T = len(xyz)
    u = np.linspace(0., 1., T)

    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    cum = np.concatenate([[0.], np.cumsum(seg)])

    if cum[-1] <= 1e-12:
        return u

    phase = cum / cum[-1]
    pu, idx = np.unique(phase, return_index=True)
    uu = u[idx].copy()

    if len(pu) < 2:
        return u

    uu[0], uu[-1] = 0., 1.

    s = np.interp(
        np.linspace(0., 1., T),
        pu, uu
    )
    s[0], s[-1] = 0., 1.
    return s


def resample(arr, s, normalize=False):
    arr = np.asarray(arr)
    T = arr.shape[0]
    u = np.linspace(0., 1., T)

    shape = arr.shape
    flat = arr.astype(np.float64).reshape(T, -1)
    out = np.empty_like(flat)

    for j in range(flat.shape[1]):
        out[:, j] = np.interp(s, u, flat[:, j])

    out = out.reshape(shape).astype(arr.dtype)

    if normalize:
        n = np.linalg.norm(out, axis=-1, keepdims=True)
        out = out / np.maximum(n, 1e-12)

    return out


def corrected_build(
    sample,
    domain,
    robot_sampler,
    max_robot_points,
    deterministic=False,
    seed=None,
    force_single_arm=False,
):
    out = OFFICIAL_BUILD(
        sample,
        domain=domain,
        robot_sampler=robot_sampler,
        max_robot_points=max_robot_points,
        deterministic=deterministic,
        seed=seed,
        force_single_arm=force_single_arm,
    )

    key = str(out.get("__key__", ""))

    if key not in TARGET_KEYS:
        return out

    xyz = dominant_xyz(out)
    T = out["robot_flows"].shape[0]
    s = arc_map(xyz)

    original = out["robot_flows"].copy()

    out["robot_flows"] = resample(
        out["robot_flows"], s
    )

    if not np.allclose(
        out["robot_flows"][0], original[0],
        atol=1e-7, rtol=0
    ):
        raise RuntimeError(f"{key}: t0 changed")

    if not np.allclose(
        out["robot_flows"][-1], original[-1],
        atol=1e-7, rtol=0
    ):
        raise RuntimeError(f"{key}: endpoint changed")

    if "robot_normals" in out:
        out["robot_normals"] = resample(
            out["robot_normals"], s, normalize=True
        )

    for name in [
        "right_gripper_open",
        "left_gripper_open",
        "gripper_positions",
    ]:
        if name in out:
            a = np.asarray(out[name])
            if a.ndim >= 1 and a.shape[0] == T:
                out[name] = resample(a, s)

    return out


def mask3(batch, name):
    x = batch[name].bool()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x


def masked_mean(x, m):
    if not m.any():
        return float("nan")
    return float(x[m].float().mean().cpu())


def main():
    args = parse_args()
    tester = Tester(args)
    tester.model.eval()

    # Baseline pipeline.
    dlmod.build_flow_sample = OFFICIAL_BUILD
    reset()
    base_dl, _ = dlmod.build_dataloader(
        tester.args,
        mode="test",
        rank=tester.rank,
        world_size=tester.world_size,
        override_splits="test",
    )

    # Retimed-action pipeline.
    dlmod.build_flow_sample = corrected_build
    reset()
    corr_dl, _ = dlmod.build_dataloader(
        tester.args,
        mode="test",
        rank=tester.rank,
        world_size=tester.world_size,
        override_splits="test",
    )

    rows = []
    found = set()

    with torch.no_grad():
        for b0, b1 in zip(base_dl, corr_dl):

            k0 = str(b0["__key__"][0])
            k1 = str(b1["__key__"][0])

            if k0 != k1:
                raise RuntimeError(
                    f"loader mismatch: {k0} != {k1}"
                )

            if k0 not in TARGET_KEYS:
                continue

            # Same scene / same point correspondence is mandatory.
            if not torch.allclose(
                b0["gt_scene_flows"],
                b1["gt_scene_flows"],
                atol=1e-7,
                rtol=0,
            ):
                raise RuntimeError(
                    f"{k0}: scene correspondence changed"
                )

            g0 = {
                k: (
                    v.to(tester.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in b0.items()
            }
            g1 = {
                k: (
                    v.to(tester.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in b1.items()
            }

            reset()
            p0 = tester.model(
                g0, training=False
            )["scene_flows"]

            reset()
            p1 = tester.model(
                g1, training=False
            )["scene_flows"]

            shift = torch.linalg.vector_norm(
                p1 - p0,
                dim=-1,
            )

            exists = mask3(g0, "scene_exists")
            context = mask3(g0, "scene_context_mask")
            supervised = mask3(
                g0, "scene_supervised_mask"
            )
            moved = mask3(g0, "scene_moved_mask")

            valid = (
                (~context)
                & exists
                & supervised
            )

            vm = valid & moved

            row = {
                "clip": k0,
                "prediction_shift_all_mm":
                    1000.0 * masked_mean(
                        shift, valid
                    ),
                "prediction_shift_moved_mm":
                    1000.0 * masked_mean(
                        shift, vm
                    ),
                "prediction_shift_t10_moved_mm":
                    1000.0 * masked_mean(
                        shift[:, -1],
                        vm[:, -1],
                    ),
            }

            rows.append(row)
            found.add(k0)

            print(
                f"[{len(found)}/{len(TARGET_KEYS)}] "
                f"{k0} | "
                f"shift_moved="
                f"{row['prediction_shift_moved_mm']:.4f} mm"
            )

    missing = TARGET_KEYS - found
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} targets"
        )

    pd.DataFrame(rows).sort_values(
        "clip"
    ).to_csv(OUT, index=False)

    print("saved =", OUT)
    print("SUCCESS = True")


if __name__ == "__main__":
    main()
