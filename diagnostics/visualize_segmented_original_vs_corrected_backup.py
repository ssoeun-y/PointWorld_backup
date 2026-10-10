from pathlib import Path
import argparse
import copy
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arguments import parse_args
from evaluation.tester import Tester
import dataset_components.dataloader as dlmod

import diagnostics.behavior_object_mapping_smoke_task0006 as omap
from diagnostics.behavior_object_mpjpe_pa_smoke_task0006 import (
    MODEL_PATH,
    NORM_STATS,
)

TASK_DIRS = {
    "task-0000": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
    "task-0001": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0001",
    "task-0006": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006",
    "task-0008": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0008",
    "task-0010": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0010",
}

CORR_ROOT = Path(
    "diagnostics/results/"
    "acc_only_boundary_correction_current10457/"
    "corrected_actions"
)

OUTDIR = Path(
    "diagnostics/results/"
    "segmented_original_vs_corrected"
)


def reset_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_key(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)


def resample(a, phase, normalize=False):
    """Temporal interpolation using phase in original-frame coordinates."""
    a = np.asarray(a)
    T = a.shape[0]

    p = np.clip(np.asarray(phase, float), 0.0, T - 1)
    lo = np.floor(p).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = p - lo

    shape = (T,) + (1,) * (a.ndim - 1)
    w = w.reshape(shape)

    out = (
        (1.0 - w) * a[lo]
        + w * a[hi]
    )

    if normalize:
        n = np.linalg.norm(out, axis=-1, keepdims=True)
        out = out / np.maximum(n, 1e-12)

    return out.astype(a.dtype, copy=False)


def mask3(batch, name):
    x = batch[name][0]
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).astype(bool)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    return x


def to_device(batch, device):
    return {
        k: (
            v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }


parser = argparse.ArgumentParser()
parser.add_argument("--npz", default=None)
parser.add_argument("--t", type=int, default=10)
args_local = parser.parse_args()

OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Choose CURRENT successful corrected clip
# ------------------------------------------------------------
if args_local.npz is None:
    files = sorted(CORR_ROOT.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No current correction NPZ: {CORR_ROOT}")
    npz_path = files[0]
else:
    npz_path = Path(args_local.npz)

z = np.load(npz_path, allow_pickle=True)

required = {"clip", "task", "phase"}
missing = required - set(z.files)
if missing:
    raise RuntimeError(
        f"{npz_path} missing fields {missing}; fields={z.files}"
    )

TARGET = str(z["clip"].item())
TASK = str(z["task"].item())
PHASE = np.asarray(z["phase"], dtype=np.float64)

if TASK not in TASK_DIRS:
    raise RuntimeError(f"Unknown task {TASK}")

print("=" * 80)
print("TARGET =", TARGET)
print("TASK   =", TASK)
print("NPZ    =", npz_path)
print("PHASE  =", np.round(PHASE, 4))
print("=" * 80)

# ------------------------------------------------------------
# 2. Object-ID propagation
#    Same validated metadata mapping used in PA analysis.
# ------------------------------------------------------------
OBJECT_BUILD = omap.build_flow_sample_with_object_ids

omap.pmod.grid_sample_transform = omap.grid_sample_with_object_ids
omap.pmod.gather = omap.gather_with_object_ids


def corrected_build(
    sample,
    domain,
    robot_sampler,
    max_robot_points,
    deterministic=False,
    seed=None,
    force_single_arm=False,
):
    out = OBJECT_BUILD(
        sample,
        domain=domain,
        robot_sampler=robot_sampler,
        max_robot_points=max_robot_points,
        deterministic=deterministic,
        seed=seed,
        force_single_arm=force_single_arm,
    )

    key = normalize_key(out.get("__key__", ""))

    if key != TARGET:
        return out

    T = out["robot_flows"].shape[0]

    if len(PHASE) != T:
        raise RuntimeError(
            f"phase={len(PHASE)}, robot T={T}"
        )

    original = out["robot_flows"].copy()

    out["robot_flows"] = resample(
        out["robot_flows"],
        PHASE,
    )

    # exact start / endpoint preservation
    assert np.allclose(
        original[0], out["robot_flows"][0],
        atol=1e-7, rtol=0
    )
    assert np.allclose(
        original[-1], out["robot_flows"][-1],
        atol=1e-7, rtol=0
    )

    # keep time-dependent action features synchronized
    if "robot_normals" in out:
        out["robot_normals"] = resample(
            out["robot_normals"],
            PHASE,
            normalize=True,
        )

    for name in [
        "right_gripper_open",
        "left_gripper_open",
        "gripper_positions",
    ]:
        if name in out:
            a = np.asarray(out[name])
            if a.ndim >= 1 and a.shape[0] == T:
                out[name] = resample(a, PHASE)

    return out


# ------------------------------------------------------------
# 3. PointWorld
# ------------------------------------------------------------
pw = parse_args(skip_command_line=True)

pw.model_path = MODEL_PATH
pw.norm_stats_path = NORM_STATS
pw.domains = ["behavior"]
pw.data_dirs = [TASK_DIRS[TASK]]

pw.batch_size = 1
pw.num_workers = 0
pw.eval_num_workers = 0
pw.eval_num_batches = -1
pw.device = "cuda"
pw.eval_skip_viz = True

tester = Tester(pw)
tester.model.eval()

# ------------------------------------------------------------
# 4. Two IDENTICAL scene loaders:
#    A = original Action
#    B = corrected Action
# ------------------------------------------------------------
dlmod.build_flow_sample = OBJECT_BUILD
reset_seed(42)
base_loader, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

dlmod.build_flow_sample = corrected_build
reset_seed(42)
corr_loader, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

found = False

for b0, b1 in zip(base_loader, corr_loader):

    k0 = normalize_key(b0["__key__"])
    k1 = normalize_key(b1["__key__"])

    if k0 != k1:
        raise RuntimeError(f"loader mismatch: {k0} != {k1}")

    if k0 != TARGET:
        continue

    found = True

    # Scene/object correspondence must be identical.
    if not torch.equal(
        b0["scene_object_id"],
        b1["scene_object_id"],
    ):
        raise RuntimeError("object segmentation changed")

    if not torch.allclose(
        b0["gt_scene_flows"],
        b1["gt_scene_flows"],
        atol=1e-7,
        rtol=0,
    ):
        raise RuntimeError("scene correspondence changed")

    g0 = to_device(b0, tester.device)
    g1 = to_device(b1, tester.device)

    # Same RNG before both forward passes.
    # Makes the qualitative comparison action-controlled.
    reset_seed(12345)
    with torch.no_grad():
        pred0 = tester.model(
            g0, training=False
        )["scene_flows"][0].detach().cpu().numpy()

    reset_seed(12345)
    with torch.no_grad():
        pred1 = tester.model(
            g1, training=False
        )["scene_flows"][0].detach().cpu().numpy()

    labels = (
        b0["scene_object_id"][0]
        .detach().cpu().numpy()
    )

    gt = (
        b0["gt_scene_flows"][0]
        .detach().cpu().numpy()
    )

    context = mask3(b0, "scene_context_mask")
    exists = mask3(b0, "scene_exists")
    supervised = mask3(b0, "scene_supervised_mask")
    moved = mask3(b0, "scene_moved_mask")

    valid = (~context) & exists & supervised

    t = args_local.t
    if t < 1 or t >= pred0.shape[0]:
        raise RuntimeError(
            f"--t must be 1..{pred0.shape[0]-1}"
        )

    # --------------------------------------------------------
    # Same moving-rigid support as current PA evaluator:
    # object exists both at t0 and t.
    # --------------------------------------------------------
    object_ids = []

    for oid in sorted(np.unique(labels[t]).tolist()):
        oid = int(oid)

        if oid <= 0:
            continue

        m = (
            (labels[t] == oid)
            & valid[t]
            & moved[t]
            & (labels[0] == oid)
            & exists[0]
        )

        if int(m.sum()) >= 8:
            object_ids.append(oid)

    if not object_ids:
        raise RuntimeError(
            f"No eligible moving rigid object at t={t}. "
            "Try --t 5 or --t 8."
        )

    table = omap.OBJECT_TABLE.get(TARGET, {})

    print("OBJECT IDS =", object_ids)

    for oid in object_ids:
        print(
            f"  id={oid}",
            "name=",
            table.get(oid, "unknown"),
        )

    # --------------------------------------------------------
    # 5. Build 3-panel segmented point-cloud figure
    # --------------------------------------------------------
    fig = plt.figure(figsize=(18, 6))

    axes = [
        fig.add_subplot(131, projection="3d"),
        fig.add_subplot(132, projection="3d"),
        fig.add_subplot(133, projection="3d"),
    ]

    titles = [
        "Initial rigid-object reference (t=0)",
        f"Original Action prediction (t={t})",
        f"Corrected Action prediction (t={t})",
    ]

    cmap = plt.get_cmap("tab10")

    all_xyz = []

    for j, oid in enumerate(object_ids):

        m_t = (
            (labels[t] == oid)
            & valid[t]
            & moved[t]
            & (labels[0] == oid)
            & exists[0]
        )

        # Same physical point indices at t0.
        idx = np.where(m_t)[0]

        x_ref = gt[0, idx]
        x_org = pred0[t, idx]
        x_cor = pred1[t, idx]

        color = cmap(j % 10)

        name = str(table.get(oid, f"object_{oid}"))
        short = name.split("__")[-1]

        for ax, xyz in zip(
            axes,
            [x_ref, x_org, x_cor],
        ):
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s=14,
                color=color,
                alpha=0.9,
                label=short,
            )

        all_xyz.extend([x_ref, x_org, x_cor])

    # identical world-coordinate bounds for fair comparison
    all_xyz = np.concatenate(all_xyz, axis=0)

    lo = all_xyz.min(axis=0)
    hi = all_xyz.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max((hi - lo).max() / 2.0, 0.02) * 1.15

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=24, azim=-60)
        ax.set_box_aspect((1,1,1))

    axes[-1].legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
    )

    fig.suptitle(
        f"{TARGET}\n"
        f"Object-segmented PointWorld prediction | "
        f"Original vs Corrected Action",
        fontsize=13,
    )

    fig.tight_layout()

    safe = (
        TARGET
        .replace("/", "_")
        .replace(":", "__")
    )

    out = OUTDIR / f"{safe}__t{t}_segmented_compare.png"

    fig.savefig(
        out,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    print()
    print("SAVED =", out)
    print("START/END = preserved by correction NPZ")
    print("DONE")
    break

if not found:
    raise RuntimeError(
        f"Target clip not found in official loader: {TARGET}"
    )

# restore
dlmod.build_flow_sample = OBJECT_BUILD
