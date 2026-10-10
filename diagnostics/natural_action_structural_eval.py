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

META = Path(
    os.environ.get(
        "PW_META_CSV",
        str(
            ROOT
            / "diagnostics/results/"
            / "natural_action_quality_per_sample.csv"
        ),
    )
)

RUN_TAG = os.environ.get("PW_RUN_TAG", "default")

OUT = ROOT / (
    "diagnostics/results/"
    f"natural_action_structural_5task_{RUN_TAG}_{VARIANT}.csv"
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
# Load previous native official results
# ================================================================
with META.open() as f:
    meta_rows = list(csv.DictReader(f))

TARGET_KEYS = sorted(
    set(r["clip"] for r in meta_rows)
)


print("variant =", VARIANT)
print("unique target samples =", len(TARGET_KEYS))


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



# ================================================================
# Local 3D structural consistency diagnostic
#
# Reuses the previously validated DROID local-coherence definition:
#   K = 8
#   locality gate = P95 of GT-t0 kNN edge distances
#
# Main quantity:
#   outward stretch = max(predicted edge length - GT edge length, 0)
#
# This is a diagnostic structural metric, NOT an official PointWorld
# evaluation metric.
# ================================================================

LOCAL_K = 8
LOCAL_EDGE_GATE_Q = 0.95


def local_structure_outward_mm(
    pred_xyz,
    gt_xyz,
    point_mask,
):
    """
    pred_xyz, gt_xyz: (T,N,3)
    point_mask:        (N,) bool

    Neighborhoods are fixed using GT at t0.
    Geometry is compared at the final timestep.
    """

    idx = torch.where(point_mask)[0]

    # Need enough points to form a meaningful local neighborhood.
    if idx.numel() < 3:
        return None

    gt_sel = gt_xyz[:, idx]
    pred_sel = pred_xyz[:, idx]

    n = gt_sel.shape[1]
    k = min(LOCAL_K, n - 1)

    # Fixed GT-t0 neighborhood.
    d0 = torch.cdist(
        gt_sel[0].unsqueeze(0),
        gt_sel[0].unsqueeze(0),
        p=2,
    ).squeeze(0)

    d0.fill_diagonal_(float("inf"))

    knn = torch.topk(
        d0,
        k=k,
        dim=1,
        largest=False,
    ).indices

    # Same P95 adaptive locality gate as the previous
    # local-coherence implementation.
    raw_edges = set()

    for a in range(n):
        for b in knn[a].tolist():
            x, y = sorted((a, int(b)))
            raw_edges.add((x, y))

    if not raw_edges:
        return None

    raw_dist = torch.stack([
        d0[a, b]
        for a, b in raw_edges
    ])

    edge_gate = torch.quantile(
        raw_dist.float(),
        LOCAL_EDGE_GATE_Q,
    )

    outward = []

    for a in range(n):
        js = knn[a]

        keep = d0[a, js] <= edge_gate
        js = js[keep]

        if js.numel() == 0:
            continue

        gt_edge = torch.linalg.vector_norm(
            gt_sel[-1, js] - gt_sel[-1, a],
            dim=-1,
        )

        pred_edge = torch.linalg.vector_norm(
            pred_sel[-1, js] - pred_sel[-1, a],
            dim=-1,
        )

        outward_i = torch.clamp(
            pred_edge - gt_edge,
            min=0.0,
        ).mean()

        outward.append(outward_i)

    if not outward:
        return None

    outward = torch.stack(outward).float()

    return {
        "local_struct_n_points": int(n),
        "local_struct_mean_mm":
            float(outward.mean().detach().cpu()) * 1000.0,
        "local_struct_median_mm":
            float(outward.median().detach().cpu()) * 1000.0,
        "local_struct_p95_mm":
            float(
                torch.quantile(outward, 0.95)
                .detach().cpu()
            ) * 1000.0,
    }


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

        # -------------------------------------------------------------
        # HPLFlowNet official 3D scene-flow metrics
        #
        # PointWorld pred/GT above are absolute XYZ trajectories.
        # Convert both to displacement relative to t0.
        # Do NOT use gt_scene_flows_relative.
        # -------------------------------------------------------------

        pred_flow = pred - pred[:, 0:1]
        gt_flow = gt - gt[:, 0:1]

        official_valid = (
            (~context)
            & exists
            & supervised
            & moved
        )

        # t0 is zero displacement by construction; exclude it.
        official_valid[:, 0] = False

        flow_err = torch.linalg.vector_norm(
            pred_flow - gt_flow,
            dim=-1,
        )

        gt_flow_norm = torch.linalg.vector_norm(
            gt_flow,
            dim=-1,
        )

        relative_err = flow_err / (gt_flow_norm + 1e-4)

        acc3ds_mask = (
            (flow_err < 0.05)
            | (relative_err < 0.05)
        )

        acc3dr_mask = (
            (flow_err < 0.10)
            | (relative_err < 0.10)
        )

        outlier3d_mask = (
            (flow_err > 0.30)
            | (relative_err > 0.10)
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

            # HPLFlowNet official scene-flow evaluation support:
            # PointWorld official valid & moved points, excluding t0.
            hpl_mask = official_valid[i]

            epe3d_m = masked_mean(
                flow_err[i],
                hpl_mask,
            )

            acc3ds = masked_mean(
                acc3ds_mask[i].float(),
                hpl_mask,
            )

            acc3dr = masked_mean(
                acc3dr_mask[i].float(),
                hpl_mask,
            )

            outliers3d = masked_mean(
                outlier3d_mask[i].float(),
                hpl_mask,
            )

            n_hpl_points = int(
                hpl_mask.sum().item()
            )

            # ---------------------------------------------------------
            # Whole moved-point local structural consistency.
            #
            # Use points that are officially valid+moved at the final
            # prediction timestep. Neighborhood itself is defined from
            # GT t0 inside local_structure_outward_mm().
            # ---------------------------------------------------------
            local_struct = local_structure_outward_mm(
                pred[i],
                gt[i],
                official_valid[i, -1],
            )

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

                # Official HPLFlowNet 3D metrics
                "n_hpl_points": n_hpl_points,
                "EPE3D_m": epe3d_m,
                "EPE3D_mm": 1000.0 * epe3d_m,
                "Acc3DS": acc3ds,
                "Acc3DR": acc3dr,
                "Outliers3D": outliers3d,

                # Whole moved 3D state structural diagnostic
                "local_struct_n_points":
                    (
                        local_struct["local_struct_n_points"]
                        if local_struct is not None else 0
                    ),
                "local_struct_mean_mm":
                    (
                        local_struct["local_struct_mean_mm"]
                        if local_struct is not None
                        else float("nan")
                    ),
                "local_struct_median_mm":
                    (
                        local_struct["local_struct_median_mm"]
                        if local_struct is not None
                        else float("nan")
                    ),
                "local_struct_p95_mm":
                    (
                        local_struct["local_struct_p95_mm"]
                        if local_struct is not None
                        else float("nan")
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
