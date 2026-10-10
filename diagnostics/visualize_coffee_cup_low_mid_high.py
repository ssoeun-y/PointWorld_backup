#!/usr/bin/env python3

import copy
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import webdataset as wds

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dlmod
from dataset_components.decoders import decode_data
from dataset_components.cameras import sample_cameras
from dataset_components.robot import canonicalize_gripper_keys_and_flags
from dataset_components.pipeline import apply_release_pipeline_to_sample
from dataset_components.collate import custom_collate_fn
from dataset_components.constants import RELEASE_GRIPPER_ONLY

from robot_sampler import RobotSampler as TorchRobotSampler
from utils import resolve_robot_urdf

from diagnostics import behavior_object_mapping_smoke_task0006 as omap
from diagnostics.behavior_object_mpjpe_pa_smoke_task0006 import p_mpjpe_single


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT.parent / "datasets/behavior/wds_task0010"
MODEL_PATH = ROOT / "pretrained_checkpoints/large-droid+behavior/model-best.pt"
NORM_STATS = ROOT / "stats/droid_behavior"

OUTDIR = (
    ROOT
    / "diagnostics/results/object_pa_mpjpe_acc_5task_current10457/"
      "representative_visuals_coffee_cup"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MIN_POINTS = 8
OBJECT_NAME = "coffee_cup_209"

# established result values used only for figure labels
TARGETS = {
    "LOW": {
        "clip": "task-0010_episode_00100560-1025:1036",
        "acc": 0.070628,
        "pa": 0.644294,
    },
    "MID": {
        "clip": "task-0010_episode_00100890-1020:1031",
        "acc": 0.220185,
        "pa": 1.497410,
    },
    "HIGH": {
        "clip": "task-0010_episode_00102970-1365:1376",
        "acc": 0.465105,
        "pa": 3.063917,
    },
}

TARGET_KEYS = {
    v["clip"] for v in TARGETS.values()
}


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def arr(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_mask(x):
    x = arr(x)
    if x.ndim >= 1 and x.shape[-1] == 1:
        x = x[..., 0]
    return x


def to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in batch.items()
    }


def release_process(sample, args, domain, rank):
    sample = sample_cameras(
        sample,
        min_num_cameras=args.eval_min_num_cameras,
        max_num_cameras=args.eval_max_num_cameras,
        deterministic=True,
        seed=args.seed + rank,
    )

    sample = canonicalize_gripper_keys_and_flags(sample)

    sample = apply_release_pipeline_to_sample(
        sample,
        domain=domain,
        mode="test",
        args=args,
        has_bimanual_robot=True,
        rank=rank,
        include_scene_data=False,
        skip_scene_sampling=False,
    )

    return sample


def pa_align(predicted, target):
    """
    EXACT same similarity-Procrustes convention as p_mpjpe_single.
    Returns aligned prediction, PA error, scale.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    muX = np.mean(target, axis=0, keepdims=True)
    muY = np.mean(predicted, axis=0, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2))
    normY = np.sqrt(np.sum(Y0 ** 2))

    if normX <= 0 or normY <= 0:
        raise RuntimeError("Degenerate geometry")

    X0 /= normX
    Y0 /= normY

    H = X0.T @ Y0
    U, s, Vt = np.linalg.svd(H)
    V = Vt.T

    R = V @ U.T

    sign_detR = np.sign(np.linalg.det(R))
    if sign_detR == 0:
        sign_detR = 1.0

    V[:, -1] *= sign_detR
    s[-1] *= sign_detR

    R = V @ U.T

    tr = np.sum(s)
    scale = tr * normX / normY

    translation = muX - scale * (muY @ R)

    aligned = scale * (predicted @ R) + translation

    error = np.mean(
        np.linalg.norm(
            aligned - target,
            axis=-1
        )
    )

    # hard audit against established metric function
    ref_error, ref_scale = p_mpjpe_single(
        predicted,
        target,
    )

    if abs(error - ref_error) > 1e-10:
        raise RuntimeError(
            f"PA mismatch: {error} vs {ref_error}"
        )

    if abs(scale - ref_scale) > 1e-10:
        raise RuntimeError(
            f"scale mismatch: {scale} vs {ref_scale}"
        )

    return aligned, float(error), float(scale)


def equal_axes(ax, xyz):
    xyz = np.asarray(xyz)

    center = (
        xyz.max(axis=0) +
        xyz.min(axis=0)
    ) / 2.0

    span = np.max(
        xyz.max(axis=0) -
        xyz.min(axis=0)
    )

    if span <= 0:
        span = 1.0

    r = span * 0.58

    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)

    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    # same camera for every panel
    ax.view_init(elev=22, azim=-60)


def find_oid(key):
    table = omap.OBJECT_TABLE[key]

    matches = [
        (int(oid), str(name))
        for oid, name in table.items()
        if OBJECT_NAME in str(name)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{key}: expected one {OBJECT_NAME}, "
            f"found {matches}"
        )

    return matches[0]


def main():

    # ------------------------------------------------------------
    # Object-ID propagation hooks
    # ------------------------------------------------------------
    omap.dlmod.build_flow_sample = (
        omap.build_flow_sample_with_object_ids
    )
    omap.pmod.grid_sample_transform = (
        omap.grid_sample_with_object_ids
    )
    omap.pmod.gather = (
        omap.gather_with_object_ids
    )

    args = parse_args(skip_command_line=True)

    args.seed = SEED
    args.model_path = str(MODEL_PATH)
    args.norm_stats_path = str(NORM_STATS)

    args.domains = ["behavior"]
    args.data_dirs = [str(DATA_DIR)]

    args.batch_size = 1
    args.num_workers = 0

    if hasattr(args, "eval_num_workers"):
        args.eval_num_workers = 0

    args.device = "cuda"
    args.eval_skip_viz = True

    reset_seed(SEED)

    tester = Tester(args)
    tester.model.eval()

    domain = "behavior"
    device = tester.device

    robot_sampler = TorchRobotSampler(
        urdf_path=resolve_robot_urdf(domain),
        gripper_only=RELEASE_GRIPPER_ONLY,
        device="cpu",
    )

    force_single_arm = False

    if getattr(args, "_eval_override_active", False):
        force_single_arm = (
            domain
            in getattr(
                args,
                "eval_force_single_arm_domains",
                set(),
            )
        )

    shard_paths = dlmod.gather_shard_paths(
        str(DATA_DIR),
        "test",
        rank=tester.rank,
    )

    random.Random(
        args.seed + tester.rank
    ).shuffle(shard_paths)

    dataset = wds.WebDataset(
        shard_paths,
        shardshuffle=False,
        detshuffle=True,
        resampled=False,
        nodesplitter=None,
        seed=args.seed + tester.rank,
        handler=wds.warn_and_continue,
    )

    results = {}
    found = set()

    with torch.inference_mode():

        for raw in dataset:

            key = str(raw.get("__key__", ""))

            if key not in TARGET_KEYS:
                continue

            print("\nPROCESSING =", key)

            found.add(key)

            decoded = decode_data(
                raw,
                domain=domain,
            )

            # ----------------------------------------------------
            # Build before release sampling.
            # Keep this copy for the dense/raw object panel.
            # ----------------------------------------------------
            built = omap.build_flow_sample_with_object_ids(
                decoded,
                domain=domain,
                robot_sampler=robot_sampler,
                max_robot_points=args.max_robot_points,
                deterministic=True,
                seed=args.seed + tester.rank,
                force_single_arm=force_single_arm,
            )

            oid, full_name = find_oid(key)

            # ----------------------------------------------------
            # Dense/raw object for visualization:
            # before official camera sampling, scene points remain
            # separated by camera. Collect the target rigid object
            # from all available cameras at t=0.
            # ----------------------------------------------------
            dense_parts = []

            for cam in ["camera_head", "camera_left", "camera_right"]:
                flow_key = f"{cam}_scene_flows"
                id_key = f"{cam}_scene_object_id"

                if flow_key not in built or id_key not in built:
                    continue

                pts = arr(built[flow_key])
                ids = squeeze_mask(built[id_key])

                # Expected raw BEHAVIOR layout is [T, N, 3].
                if pts.ndim == 3:
                    pts0 = pts[0]
                elif pts.ndim == 2:
                    pts0 = pts
                else:
                    raise RuntimeError(
                        f"{key}: unexpected {flow_key} shape={pts.shape}"
                    )

                if ids.ndim >= 2:
                    ids0 = ids[0]
                else:
                    ids0 = ids

                if len(pts0) != len(ids0):
                    raise RuntimeError(
                        f"{key}: {cam} point/id length mismatch "
                        f"{len(pts0)} vs {len(ids0)}"
                    )

                m = (ids0 == oid)

                if np.any(m):
                    dense_parts.append(
                        pts0[m]
                    )

                    print(
                        cam,
                        "dense object points =",
                        int(m.sum()),
                    )

            if not dense_parts:
                raise RuntimeError(
                    f"{key}: no dense camera points found "
                    f"for object id={oid}, name={full_name}"
                )

            raw_object = np.concatenate(
                dense_parts,
                axis=0,
            )

            print(
                "TOTAL DENSE OBJECT POINTS =",
                len(raw_object),
            )

            print(
                "OBJECT =",
                full_name,
                "RAW POINTS =",
                len(raw_object),
            )

            # ----------------------------------------------------
            # Official release pipeline / official sparse sample
            # ----------------------------------------------------
            released = release_process(
                copy.deepcopy(built),
                args,
                domain,
                tester.rank,
            )

            batch_cpu = custom_collate_fn(
                [released],
                args=args,
            )

            batch_gpu = to_device(
                batch_cpu,
                device,
            )

            outputs = tester.model(
                batch_gpu,
                training=False,
            )

            pred = (
                outputs["scene_flows"][0]
                .detach()
                .cpu()
                .numpy()
            )

            gt = arr(
                batch_cpu["gt_scene_flows"]
            )[0]

            labels = squeeze_mask(
                batch_cpu["scene_object_id"]
            )[0]

            context = squeeze_mask(
                batch_cpu["scene_context_mask"]
            )[0].astype(bool)

            exists = squeeze_mask(
                batch_cpu["scene_exists"]
            )[0].astype(bool)

            supervised = squeeze_mask(
                batch_cpu["scene_supervised_mask"]
            )[0].astype(bool)

            valid = (
                (~context)
                & exists
                & supervised
            )

            object_mask = (
                labels == oid
            )

            # official sampled object at t=0
            m0 = (
                object_mask[0]
                & valid[0]
            )

            sampled_gt0 = gt[
                0,
                m0,
            ]

            # ----------------------------------------------------
            # Compute PA for every valid future frame.
            # Choose the frame CLOSEST TO THIS CLIP'S MEAN PA.
            # Not the worst frame.
            # ----------------------------------------------------
            frame_rows = []

            for t in range(
                1,
                gt.shape[0],
            ):
                # EXACT same per-frame object/valid mask
                # used by the successful main PA-MPJPE evaluator.
                m = (
                    object_mask[t]
                    & valid[t]
                )

                n = int(m.sum())

                if n < MIN_POINTS:
                    continue

                target = gt[t, m]
                prediction = pred[t, m]

                centered = (
                    target
                    - target.mean(
                        axis=0,
                        keepdims=True,
                    )
                )

                if (
                    np.linalg.matrix_rank(
                        centered,
                        tol=1e-8,
                    )
                    < 2
                ):
                    continue

                aligned, pa, scale = pa_align(
                    prediction,
                    target,
                )

                mpjpe = np.mean(
                    np.linalg.norm(
                        prediction - target,
                        axis=-1,
                    )
                )

                frame_rows.append({
                    "t": t,
                    "n": n,
                    "mpjpe_mm":
                        float(mpjpe * 1000.0),
                    "pa_mm":
                        float(pa * 1000.0),
                    "scale": scale,
                    "gt": target,
                    "pred": prediction,
                    "aligned": aligned,
                })

            if not frame_rows:
                raise RuntimeError(
                    f"{key}: no valid frames"
                )

            fresh_mean_pa = float(
                np.mean(
                    [r["pa_mm"] for r in frame_rows]
                )
            )

            chosen = min(
                frame_rows,
                key=lambda r:
                    abs(
                        r["pa_mm"]
                        - fresh_mean_pa
                    )
            )

            group = next(
                g
                for g, v in TARGETS.items()
                if v["clip"] == key
            )

            results[group] = {
                "key": key,
                "oid": oid,
                "full_name": full_name,

                "raw": raw_object,
                "sampled_gt0": sampled_gt0,

                "gt": chosen["gt"],
                "pred": chosen["pred"],
                "aligned": chosen["aligned"],

                "frame": chosen["t"],
                "frame_pa_mm": chosen["pa_mm"],
                "frame_mpjpe_mm":
                    chosen["mpjpe_mm"],
                "fresh_mean_pa_mm":
                    fresh_mean_pa,
                "scale": chosen["scale"],

                "established_acc":
                    TARGETS[group]["acc"],
                "established_pa":
                    TARGETS[group]["pa"],
            }

            print(
                group,
                "chosen frame =",
                chosen["t"],
                "fresh mean PA =",
                fresh_mean_pa,
                "chosen PA =",
                chosen["pa_mm"],
            )

            if found == TARGET_KEYS:
                break

    missing = TARGET_KEYS - found

    if missing:
        raise RuntimeError(
            f"missing clips: {missing}"
        )

    # ============================================================
    # 3 rows × 4 columns
    # ============================================================
    groups = ["LOW", "MID", "HIGH"]

    fig = plt.figure(
        figsize=(15, 11)
    )

    for ri, group in enumerate(groups):

        r = results[group]

        # ------------------ 1. Raw dense object ------------------
        ax = fig.add_subplot(
            3, 4, ri * 4 + 1,
            projection="3d",
        )

        raw_pts = r["raw"]

        # plotting only: cap huge raw clouds
        if len(raw_pts) > 6000:
            idx = np.linspace(
                0,
                len(raw_pts) - 1,
                6000,
            ).astype(int)
            raw_pts = raw_pts[idx]

        ax.scatter(
            raw_pts[:, 0],
            raw_pts[:, 1],
            raw_pts[:, 2],
            s=1.5,
            alpha=0.65,
        )

        equal_axes(ax, raw_pts)

        ax.set_title(
            "Raw / dense object"
        )

        ax.text2D(
            -0.12,
            0.50,
            (
                f"{group}\n"
                f"RMS Acc = "
                f"{r['established_acc']:.3f} m/s²\n"
                f"PA-MPJPE = "
                f"{r['established_pa']:.3f} mm"
            ),
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=11,
        )

        # ---------------- 2. Official sampled GT0 ----------------
        ax = fig.add_subplot(
            3, 4, ri * 4 + 2,
            projection="3d",
        )

        s0 = r["sampled_gt0"]

        ax.scatter(
            s0[:, 0],
            s0[:, 1],
            s0[:, 2],
            s=20,
        )

        equal_axes(ax, s0)

        ax.set_title(
            f"Official sampled points\nN={len(s0)}"
        )

        # ---------------- 3. Before PA alignment -----------------
        ax = fig.add_subplot(
            3, 4, ri * 4 + 3,
            projection="3d",
        )

        gt = r["gt"]
        pred = r["pred"]

        ax.scatter(
            gt[:, 0],
            gt[:, 1],
            gt[:, 2],
            s=24,
            label="GT",
        )

        ax.scatter(
            pred[:, 0],
            pred[:, 1],
            pred[:, 2],
            s=24,
            marker="^",
            label="Pred",
        )

        both = np.concatenate(
            [gt, pred],
            axis=0,
        )

        equal_axes(ax, both)

        ax.set_title(
            "GT vs Pred\n"
            f"MPJPE={r['frame_mpjpe_mm']:.2f} mm"
        )

        ax.legend(
            loc="upper right",
            fontsize=7,
        )

        # ------------------ 4. After PA --------------------------
        ax = fig.add_subplot(
            3, 4, ri * 4 + 4,
            projection="3d",
        )

        aligned = r["aligned"]

        ax.scatter(
            gt[:, 0],
            gt[:, 1],
            gt[:, 2],
            s=24,
            label="GT",
        )

        ax.scatter(
            aligned[:, 0],
            aligned[:, 1],
            aligned[:, 2],
            s=24,
            marker="^",
            label="PA-aligned Pred",
        )

        both_pa = np.concatenate(
            [gt, aligned],
            axis=0,
        )

        equal_axes(ax, both_pa)

        ax.set_title(
            "After similarity alignment\n"
            f"PA={r['frame_pa_mm']:.2f} mm"
        )

        ax.legend(
            loc="upper right",
            fontsize=7,
        )

    fig.suptitle(
        "Same rigid object: coffee_cup_209 — low / mid / high Action RMS Acceleration",
        fontsize=15,
        y=0.995,
    )

    fig.tight_layout(
        rect=[0.04, 0.02, 1, 0.97]
    )

    png = (
        OUTDIR
        / "coffee_cup_low_mid_high_3x4.png"
    )

    pdf = (
        OUTDIR
        / "coffee_cup_low_mid_high_3x4.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    summary = []

    for group in groups:
        r = results[group]

        summary.append({
            "group": group,
            "clip": r["key"],
            "object": OBJECT_NAME,
            "established_rms_acc_m_s2":
                r["established_acc"],
            "established_clip_pa_mpjpe_mm":
                r["established_pa"],
            "revisualization_mean_pa_mm":
                r["fresh_mean_pa_mm"],
            "visualized_frame":
                r["frame"],
            "visualized_frame_mpjpe_mm":
                r["frame_mpjpe_mm"],
            "visualized_frame_pa_mm":
                r["frame_pa_mm"],
            "pa_scale":
                r["scale"],
            "raw_dense_points":
                len(r["raw"]),
            "official_sampled_points_t0":
                len(r["sampled_gt0"]),
        })

    pd.DataFrame(summary).to_csv(
        OUTDIR / "visualization_summary.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("VISUALIZATION COMPLETE")
    print("=" * 100)

    print(
        pd.DataFrame(summary).to_string(
            index=False
        )
    )

    print("\nPNG =", png)
    print("PDF =", pdf)
    print(
        "SUMMARY =",
        OUTDIR / "visualization_summary.csv",
    )


if __name__ == "__main__":
    main()
