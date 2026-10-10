#!/usr/bin/env python3

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# eval_prediction_shift_paired requires these at import time.
os.environ.setdefault("PW_TARGET_TASKS", "task-0000")
os.environ.setdefault("PW_TAG", "temporary")

from diagnostics import eval_prediction_shift_paired as base
from arguments import parse_args
from evaluation.tester import Tester
import dataset_components.dataloader as dlmod


ROOT = Path(__file__).resolve().parents[1]

CORR_ROOT = (
    ROOT
    / "diagnostics/results/"
      "goal_preserved_global_boundary_correction_7987/"
      "corrected_actions"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/"
      "goal_preserved_global_boundary_pointcloud_structure"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

TASKS = {
    x.strip()
    for x in os.environ["PW_TARGET_TASKS"].split(",")
    if x.strip()
}

TAG = os.environ["PW_TAG"]

# Fixed before looking at results.
K = 8
EDGE_GATE_Q = 0.95
KNN_CHUNK = 512

OFFICIAL_BUILD = dlmod.build_flow_sample


# ============================================================
# Successful goal-preserved corrections only
# ============================================================

PHASES = {}

for p in sorted(CORR_ROOT.glob("*.npz")):
    z = np.load(p, allow_pickle=True)

    clip = str(z["clip"].item())
    task = str(z["task"].item())

    if task not in TASKS:
        continue

    PHASES[clip] = np.asarray(
        z["phase"],
        dtype=np.float64,
    )

if not PHASES:
    raise RuntimeError(
        f"No successful corrections for {sorted(TASKS)}"
    )

TARGET_KEYS = set(PHASES.keys())

OUT = OUTDIR / f"{TAG}__pointcloud_structure.csv"


# ============================================================
# Reproducibility
# ============================================================

def reset(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Exact same correction used in validated correction evaluator
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

    if key not in PHASES:
        return out

    phase = PHASES[key]

    T = out["robot_flows"].shape[0]

    if len(phase) != T:
        raise RuntimeError(
            f"{key}: phase length {len(phase)} != T {T}"
        )

    original = out["robot_flows"].copy()

    out["robot_flows"] = base.resample(
        out["robot_flows"],
        phase / float(T - 1),
    )

    # Hard endpoint constraints.
    if not np.allclose(
        out["robot_flows"][0],
        original[0],
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError(f"{key}: t0 changed")

    if not np.allclose(
        out["robot_flows"][-1],
        original[-1],
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError(f"{key}: endpoint changed")

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

        a = np.asarray(out[name])

        if a.ndim >= 1 and a.shape[0] == T:
            out[name] = base.resample(
                a,
                phase / float(T - 1),
            )

    return out


# ============================================================
# Masks
# ============================================================

def mask3(batch, name):
    x = batch[name].bool()

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise RuntimeError(
            f"{name}: unexpected shape {tuple(x.shape)}"
        )

    return x


# ============================================================
# Fixed t0 local graph
#
# No future GT geometry is used.
#
# 1) choose the benchmark moved-point support
# 2) positions for graph construction are observed t0 points
# 3) kNN identities are fixed at t0
# 4) P95 locality gate removes unusually long local edges
# ============================================================

def build_fixed_local_graph(
    x0,
    support,
    k=K,
    gate_q=EDGE_GATE_Q,
):
    idx = torch.where(support)[0]

    if int(idx.numel()) <= k:
        return None

    pts = x0[idx].float()
    n = int(pts.shape[0])

    if not torch.isfinite(pts).all():
        finite = torch.isfinite(pts).all(dim=-1)
        idx = idx[finite]
        pts = pts[finite]
        n = int(pts.shape[0])

    if n <= k:
        return None

    k_eff = min(k, n - 1)

    raw_edges = set()

    # Exact chunked kNN:
    # avoids allocating an N x N matrix for the whole cloud.
    for start in range(0, n, KNN_CHUNK):
        end = min(start + KNN_CHUNK, n)

        d = torch.cdist(
            pts[start:end],
            pts,
            p=2,
        )

        rows = torch.arange(
            end - start,
            device=pts.device,
        )
        cols = torch.arange(
            start,
            end,
            device=pts.device,
        )

        d[rows, cols] = float("inf")

        knn = torch.topk(
            d,
            k=k_eff,
            dim=1,
            largest=False,
        ).indices.cpu().numpy()

        for r in range(end - start):
            i = start + r

            for j in knn[r]:
                j = int(j)

                a, b = sorted((i, j))
                raw_edges.add((a, b))

    if not raw_edges:
        return None

    raw_edges = sorted(raw_edges)

    li = torch.as_tensor(
        [a for a, _ in raw_edges],
        device=pts.device,
        dtype=torch.long,
    )
    lj = torch.as_tensor(
        [b for _, b in raw_edges],
        device=pts.device,
        dtype=torch.long,
    )

    d0 = torch.linalg.vector_norm(
        pts[li] - pts[lj],
        dim=-1,
    )

    gate = torch.quantile(
        d0,
        gate_q,
    )

    keep = d0 <= gate

    li = li[keep]
    lj = lj[keep]
    d0 = d0[keep]

    if int(li.numel()) == 0:
        return None

    # Convert local support indices back to global point indices.
    gi = idx[li]
    gj = idx[lj]

    return {
        "i": gi,
        "j": gj,
        "d0": d0,
        "num_points": n,
        "num_edges": int(gi.numel()),
        "edge_gate_mm": float(gate.cpu()) * 1000.0,
    }


# ============================================================
# Structural quality
#
# local_rigidity:
#   | predicted local edge length - observed t0 edge length |
#
# local_jitter:
#   frame-to-frame change of each predicted local edge length
#
# Lower = structurally more stable.
# ============================================================

def structural_scores(
    pred,
    valid,
    graph,
):
    if graph is None:
        return None

    i = graph["i"]
    j = graph["j"]
    d0 = graph["d0"]

    # pred: [T, N, 3]
    lens = torch.linalg.vector_norm(
        pred[:, i] - pred[:, j],
        dim=-1,
    )

    edge_valid = (
        valid[:, i]
        & valid[:, j]
    )

    # ------------------------------------------
    # Local rigidity distortion from observed t0
    # ------------------------------------------
    distortion = torch.abs(
        lens - d0.unsqueeze(0)
    ) * 1000.0

    vals = distortion[edge_valid]

    if int(vals.numel()) == 0:
        return None

    # ------------------------------------------
    # Final-frame structural distortion
    # ------------------------------------------
    final_vals = distortion[-1][
        edge_valid[-1]
    ]

    # ------------------------------------------
    # Temporal edge-length jitter
    # Only pairs of consecutive predicted-valid
    # timesteps are scored.
    # ------------------------------------------
    jitter = torch.abs(
        lens[1:] - lens[:-1]
    ) * 1000.0

    jitter_valid = (
        edge_valid[1:]
        & edge_valid[:-1]
    )

    jitter_vals = jitter[jitter_valid]

    def stat_mean(x):
        if int(x.numel()) == 0:
            return float("nan")
        return float(x.mean().cpu())

    def stat_median(x):
        if int(x.numel()) == 0:
            return float("nan")
        return float(x.median().cpu())

    def stat_p95(x):
        if int(x.numel()) == 0:
            return float("nan")
        return float(
            torch.quantile(
                x.float(),
                0.95,
            ).cpu()
        )

    return {
        "rigidity_mean_mm": stat_mean(vals),
        "rigidity_median_mm": stat_median(vals),
        "rigidity_p95_mm": stat_p95(vals),

        "final_rigidity_mean_mm":
            stat_mean(final_vals),

        "final_rigidity_p95_mm":
            stat_p95(final_vals),

        "jitter_mean_mm":
            stat_mean(jitter_vals),

        "jitter_p95_mm":
            stat_p95(jitter_vals),
    }


# ============================================================
# Main paired inference
# ============================================================

@torch.no_grad()
def main():
    args = parse_args()

    tester = Tester(args)
    tester.model.eval()

    # --------------------------------------------------------
    # Original Action loader
    # --------------------------------------------------------
    dlmod.build_flow_sample = OFFICIAL_BUILD

    reset()

    base_dl, _ = dlmod.build_dataloader(
        tester.args,
        mode="test",
        rank=tester.rank,
        world_size=tester.world_size,
        override_splits="test",
    )

    # --------------------------------------------------------
    # Corrected Action loader
    # --------------------------------------------------------
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

    print("=" * 100)
    print("GOAL-PRESERVED CORRECTION → POINT-CLOUD STRUCTURE")
    print("=" * 100)
    print("TASKS =", sorted(TASKS))
    print("TARGETS =", len(TARGET_KEYS))
    print("K =", K)
    print("EDGE_GATE_Q =", EDGE_GATE_Q)
    print("OUTPUT =", OUT)
    print()

    for b0, b1 in zip(base_dl, corr_dl):

        k0 = str(b0["__key__"][0])
        k1 = str(b1["__key__"][0])

        if k0 != k1:
            raise RuntimeError(
                f"loader mismatch: {k0} != {k1}"
            )

        if k0 not in TARGET_KEYS:
            continue

        # Scene must be exactly identical.
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
                v.to(
                    tester.device,
                    non_blocking=True,
                )
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in b0.items()
        }

        g1 = {
            k: (
                v.to(
                    tester.device,
                    non_blocking=True,
                )
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in b1.items()
        }

        # ----------------------------------------------------
        # Frozen PointWorld: Original Action
        # ----------------------------------------------------
        reset()

        p0 = tester.model(
            g0,
            training=False,
        )["scene_flows"]

        # ----------------------------------------------------
        # Frozen PointWorld: Corrected Action
        # ----------------------------------------------------
        reset()

        p1 = tester.model(
            g1,
            training=False,
        )["scene_flows"]

        # ----------------------------------------------------
        # Official prediction-valid support
        # ----------------------------------------------------
        exists = mask3(
            g0,
            "scene_exists",
        )[0]

        context = mask3(
            g0,
            "scene_context_mask",
        )[0]

        supervised = mask3(
            g0,
            "scene_supervised_mask",
        )[0]

        moved = mask3(
            g0,
            "scene_moved_mask",
        )[0]

        valid = (
            (~context)
            & exists
            & supervised
        )

        # Same fixed point population for original/corrected.
        # t0 exists + final official-valid + benchmark moved.
        support = (
            exists[0]
            & valid[-1]
            & moved[-1]
        )

        # Observed initial geometry only.
        x0 = g0["scene_flows"][0, 0]

        graph = build_fixed_local_graph(
            x0=x0,
            support=support,
        )

        row = {
            "clip": k0,
            "task":
                str(
                    next(
                        (
                            t for t in TASKS
                            if k0.startswith(t)
                        ),
                        list(TASKS)[0],
                    )
                ),
            "num_support_points":
                int(support.sum().item()),
        }

        if graph is None:
            row["status"] = "INSUFFICIENT_GRAPH"
            rows.append(row)
            found.add(k0)

            print(
                f"[{len(found)}/{len(TARGET_KEYS)}] "
                f"{k0} | insufficient graph"
            )
            continue

        s0 = structural_scores(
            p0[0],
            valid,
            graph,
        )

        s1 = structural_scores(
            p1[0],
            valid,
            graph,
        )

        if s0 is None or s1 is None:
            row["status"] = "NO_VALID_STRUCTURAL_SCORE"
            rows.append(row)
            found.add(k0)
            continue

        row.update({
            "status": "OK",
            "num_graph_points":
                graph["num_points"],
            "num_graph_edges":
                graph["num_edges"],
            "edge_gate_mm":
                graph["edge_gate_mm"],
        })

        for name in s0:
            row[f"original_{name}"] = s0[name]
            row[f"corrected_{name}"] = s1[name]

            row[f"delta_{name}"] = (
                s1[name] - s0[name]
            )

            if (
                np.isfinite(s0[name])
                and
                s0[name] > 0
            ):
                row[f"ratio_{name}"] = (
                    s1[name] / s0[name]
                )
            else:
                row[f"ratio_{name}"] = np.nan

        rows.append(row)
        found.add(k0)

        print(
            f"[{len(found)}/{len(TARGET_KEYS)}] "
            f"{k0} | "
            f"rigidity "
            f"{s0['rigidity_mean_mm']:.4f}"
            f" → "
            f"{s1['rigidity_mean_mm']:.4f} mm | "
            f"jitter "
            f"{s0['jitter_mean_mm']:.4f}"
            f" → "
            f"{s1['jitter_mean_mm']:.4f} mm"
        )

    missing = TARGET_KEYS - found

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} targets"
        )

    df = pd.DataFrame(rows).sort_values(
        "clip"
    )

    df.to_csv(
        OUT,
        index=False,
    )

    ok = df[df["status"] == "OK"]

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("found =", len(df))
    print("structurally scored =", len(ok))

    for metric in [
        "rigidity_mean_mm",
        "rigidity_p95_mm",
        "final_rigidity_mean_mm",
        "jitter_mean_mm",
        "jitter_p95_mm",
    ]:
        a = ok[f"original_{metric}"].to_numpy(
            dtype=float
        )
        b = ok[f"corrected_{metric}"].to_numpy(
            dtype=float
        )

        good = np.isfinite(a) & np.isfinite(b)

        a = a[good]
        b = b[good]

        if len(a) == 0:
            continue

        print()
        print(metric)
        print(
            "  N =",
            len(a),
        )
        print(
            "  original median =",
            float(np.median(a)),
        )
        print(
            "  corrected median =",
            float(np.median(b)),
        )
        print(
            "  median paired ratio =",
            float(np.median(b / np.maximum(a, 1e-12))),
        )
        print(
            "  corrected lower fraction =",
            float(np.mean(b < a)),
        )

    print()
    print("saved =", OUT)
    print("SUCCESS = True")


if __name__ == "__main__":
    main()
