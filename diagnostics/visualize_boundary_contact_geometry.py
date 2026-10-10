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
    a = np.asarray(a)
    T = a.shape[0]

    p = np.clip(
        np.asarray(phase, dtype=float),
        0.0,
        T - 1,
    )

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
        n = np.linalg.norm(
            out,
            axis=-1,
            keepdims=True,
        )
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
parser.add_argument("--object_id", type=int, default=None)

args_local = parser.parse_args()

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# Choose corrected clip
# ============================================================

if args_local.npz is None:
    files = sorted(CORR_ROOT.glob("*.npz"))

    if not files:
        raise RuntimeError(
            f"No corrected action NPZ: {CORR_ROOT}"
        )

    npz_path = files[0]

else:
    npz_path = Path(args_local.npz)

z = np.load(
    npz_path,
    allow_pickle=True,
)

TARGET = str(
    z["clip"].item()
)

TASK = str(
    z["task"].item()
)

PHASE = np.asarray(
    z["phase"],
    dtype=np.float64,
)

print("=" * 80)
print("TARGET =", TARGET)
print("TASK   =", TASK)
print("PHASE  =", np.round(PHASE, 3))
print("=" * 80)

# ============================================================
# Object-ID mapping
# ============================================================

OBJECT_BUILD = (
    omap.build_flow_sample_with_object_ids
)

omap.pmod.grid_sample_transform = (
    omap.grid_sample_with_object_ids
)

omap.pmod.gather = (
    omap.gather_with_object_ids
)

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

    key = normalize_key(
        out.get("__key__", "")
    )

    if key != TARGET:
        return out

    original = out[
        "robot_flows"
    ].copy()

    out["robot_flows"] = resample(
        out["robot_flows"],
        PHASE,
    )

    # start/end must be exactly preserved
    assert np.allclose(
        original[0],
        out["robot_flows"][0],
        atol=1e-7,
        rtol=0,
    )

    assert np.allclose(
        original[-1],
        out["robot_flows"][-1],
        atol=1e-7,
        rtol=0,
    )

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

        if name not in out:
            continue

        a = np.asarray(out[name])

        if (
            a.ndim >= 1
            and
            a.shape[0] == len(PHASE)
        ):
            out[name] = resample(
                a,
                PHASE,
            )

    return out

# ============================================================
# PointWorld
# ============================================================

pw = parse_args(
    skip_command_line=True
)

pw.model_path = MODEL_PATH
pw.norm_stats_path = NORM_STATS

pw.domains = [
    "behavior"
]

pw.data_dirs = [
    TASK_DIRS[TASK]
]

pw.batch_size = 1
pw.num_workers = 0
pw.eval_num_workers = 0
pw.eval_num_batches = -1
pw.device = "cuda"
pw.eval_skip_viz = True

tester = Tester(pw)

tester.model.eval()

# ============================================================
# Original loader
# ============================================================

dlmod.build_flow_sample = OBJECT_BUILD

reset_seed(42)

base_loader, _ = (
    tester._build_eval_loader(
        "test",
        enable_mask=False,
    )
)

# ============================================================
# Corrected loader
# ============================================================

dlmod.build_flow_sample = corrected_build

reset_seed(42)

corr_loader, _ = (
    tester._build_eval_loader(
        "test",
        enable_mask=False,
    )
)

found = False

for b0, b1 in zip(
    base_loader,
    corr_loader,
):

    k0 = normalize_key(
        b0["__key__"]
    )

    k1 = normalize_key(
        b1["__key__"]
    )

    if k0 != k1:
        raise RuntimeError(
            f"loader mismatch: {k0} != {k1}"
        )

    if k0 != TARGET:
        continue

    found = True

    if not torch.equal(
        b0["scene_object_id"],
        b1["scene_object_id"],
    ):
        raise RuntimeError(
            "Object IDs changed."
        )

    g0 = to_device(
        b0,
        tester.device,
    )

    g1 = to_device(
        b1,
        tester.device,
    )

    # Same stochastic seed
    reset_seed(12345)

    with torch.no_grad():
        pred_original = (
            tester.model(
                g0,
                training=False,
            )["scene_flows"][0]
            .detach()
            .cpu()
            .numpy()
        )

    reset_seed(12345)

    with torch.no_grad():
        pred_corrected = (
            tester.model(
                g1,
                training=False,
            )["scene_flows"][0]
            .detach()
            .cpu()
            .numpy()
        )

    object_id = (
        b0["scene_object_id"][0]
        .detach()
        .cpu()
        .numpy()
    )

    context = mask3(
        b0,
        "scene_context_mask",
    )

    exists = mask3(
        b0,
        "scene_exists",
    )

    supervised = mask3(
        b0,
        "scene_supervised_mask",
    )

    moved = mask3(
        b0,
        "scene_moved_mask",
    )

    valid = (
        (~context)
        & exists
        & supervised
    )

    # ============================================================
    # BOUNDARY CONTACT GEOMETRY DIAGNOSTIC
    # Diagnostic only. Final method will use soft distance weights,
    # NOT a hard contact threshold.
    # ============================================================

    from scipy.spatial import cKDTree
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    def _contact_as_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    robot_corrected = _contact_as_numpy(b1["robot_flows"])

    if robot_corrected.ndim == 4:
        robot_corrected = robot_corrected[0]

    if robot_corrected.ndim != 3:
        raise RuntimeError(
            f"Unexpected robot_flows shape: {robot_corrected.shape}"
        )

    valid_np_contact = np.asarray(valid).astype(bool)
    Tdiag = int(pred_corrected.shape[0])

    for gname in ["right_gripper_open", "left_gripper_open"]:
        if gname not in b1:
            continue

        gv = _contact_as_numpy(b1[gname])

        if gv.ndim >= 3:
            gv = gv[0]

        print(
            f"[CONTACT_DIAG] {gname} = "
            f"{np.asarray(gv).reshape(Tdiag, -1).tolist()}"
        )

    diagnostic_radii_m = [0.015, 0.020, 0.030]

    contact_rows = []
    nearest_distances = {}

    for tt in range(Tdiag):
        scene_idx = np.flatnonzero(valid_np_contact[tt])

        if len(scene_idx) == 0:
            continue

        scene_xyz_contact = pred_corrected[tt, scene_idx]
        robot_xyz_contact = robot_corrected[tt]

        tree = cKDTree(robot_xyz_contact)
        dist, nn = tree.query(scene_xyz_contact, k=1)

        nearest_distances[tt] = (scene_idx, dist, nn)

        row = {
            "frame": int(tt),
            "n_valid_scene": int(len(scene_idx)),
            "min_mm": float(np.min(dist) * 1000.0),
            "p01_mm": float(np.quantile(dist, 0.01) * 1000.0),
            "p05_mm": float(np.quantile(dist, 0.05) * 1000.0),
            "p10_mm": float(np.quantile(dist, 0.10) * 1000.0),
            "median_mm": float(np.median(dist) * 1000.0),
        }

        for rr in diagnostic_radii_m:
            rr_mm = int(round(rr * 1000.0))
            row[f"n_within_{rr_mm}mm"] = int(np.sum(dist <= rr))

        contact_rows.append(row)

    contact_diag_df = pd.DataFrame(contact_rows)

    print("\n[CONTACT_DIAG] corrected prediction")
    print(contact_diag_df.to_string(index=False))

    contact_diag_root = Path(
        "diagnostics/results/boundary_contact_geometry"
    )
    contact_diag_root.mkdir(parents=True, exist_ok=True)

    safe_target_contact = (
        str(TARGET)
        .replace("/", "_")
        .replace(":", "__")
    )

    contact_csv_path = (
        contact_diag_root
        / f"{safe_target_contact}__contact_geometry.csv"
    )

    contact_diag_df.to_csv(contact_csv_path, index=False)

    print("[CONTACT_DIAG] saved:", contact_csv_path)

    td_contact = int(args_local.t)

    scene_idx_contact, dist_contact, _ = nearest_distances[td_contact]

    scene_xyz_contact = pred_corrected[
        td_contact, scene_idx_contact
    ]
    robot_xyz_contact = robot_corrected[td_contact]

    display_mask_contact = dist_contact <= 0.060
    candidate_mask_contact = dist_contact <= 0.030

    fig_contact = plt.figure(figsize=(8, 7))
    ax_contact = fig_contact.add_subplot(111, projection="3d")

    if np.any(display_mask_contact):
        xyz = scene_xyz_contact[display_mask_contact]
        ax_contact.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=8, c="lightgray", alpha=0.45,
            label="Corrected PW scene (local)",
        )

    if np.any(candidate_mask_contact):
        xyz = scene_xyz_contact[candidate_mask_contact]
        ax_contact.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=25, c="orange", alpha=0.9,
            label="<=30 mm diagnostic candidates",
        )

    ax_contact.scatter(
        robot_xyz_contact[:, 0],
        robot_xyz_contact[:, 1],
        robot_xyz_contact[:, 2],
        s=14,
        c="black",
        alpha=0.9,
        label="Corrected gripper flow",
    )

    vis = [robot_xyz_contact]

    if np.any(display_mask_contact):
        vis.append(scene_xyz_contact[display_mask_contact])

    vis = np.concatenate(vis, axis=0)

    mins = vis.min(axis=0)
    maxs = vis.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins) / 2.0), 1e-3)

    ax_contact.set_xlim(center[0] - radius, center[0] + radius)
    ax_contact.set_ylim(center[1] - radius, center[1] + radius)
    ax_contact.set_zlim(center[2] - radius, center[2] + radius)

    ax_contact.set_xlabel("X [m]")
    ax_contact.set_ylabel("Y [m]")
    ax_contact.set_zlabel("Z [m]")

    ax_contact.set_title(
        f"Boundary-only contact geometry\n{TARGET}, frame={td_contact}"
    )
    ax_contact.legend(loc="best")

    fig_contact.tight_layout()

    contact_fig_path = (
        contact_diag_root
        / f"{safe_target_contact}__t{td_contact}__contact_geometry.png"
    )

    fig_contact.savefig(
        contact_fig_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig_contact)

    print("[CONTACT_DIAG] figure:", contact_fig_path)

    t = args_local.t

    # ========================================================
    # Find eligible moving rigid objects
    # ========================================================

    candidates = []

    for oid in sorted(
        np.unique(
            object_id[t]
        ).tolist()
    ):

        oid = int(oid)

        if oid <= 0:
            continue

        m = (
            (object_id[t] == oid)
            & valid[t]
            & moved[t]
            & (object_id[0] == oid)
            & exists[0]
        )

        n = int(m.sum())

        if n >= 8:
            candidates.append(
                (n, oid)
            )

    if not candidates:
        raise RuntimeError(
            f"No eligible moving object at t={t}"
        )

    candidates.sort(
        reverse=True
    )

    if args_local.object_id is None:
        oid = candidates[0][1]
    else:
        oid = args_local.object_id

    m = (
        (object_id[t] == oid)
        & valid[t]
        & moved[t]
        & (object_id[0] == oid)
        & exists[0]
    )

    idx = np.where(m)[0]

    original_xyz = (
        pred_original[t, idx]
    )

    corrected_xyz = (
        pred_corrected[t, idx]
    )

    print(
        "OBJECT ID =",
        oid,
        "POINTS =",
        len(idx),
    )

    # ========================================================
    # PROCRUSTES PROCESS VISUALIZATION
    #
    # Gray = initial rigid reference
    # Blue = Original Action prediction
    # Red  = Corrected Action prediction
    # ========================================================

    reference_xyz = (
        b0["gt_scene_flows"][0, 0, idx]
        .detach()
        .cpu()
        .numpy()
    )

    def center_points(x):
        mu = x.mean(axis=0, keepdims=True)
        return x - mu, mu

    def scale_to_reference(x_centered, ref_centered):
        nx = np.linalg.norm(x_centered)
        nr = np.linalg.norm(ref_centered)

        if nx < 1e-12 or nr < 1e-12:
            raise RuntimeError("Degenerate point cloud")

        return x_centered * (nr / nx)

    def rotate_to_reference(x, ref):
        """
        Proper Kabsch rotation.
        x/ref are already centered and scaled.
        """
        H = x.T @ ref

        U, S, Vt = np.linalg.svd(H)

        R = U @ Vt

        # No reflection
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        return x @ R

    # --------------------------------------------------------
    # STEP 0: RAW
    # --------------------------------------------------------
    ref_raw = reference_xyz.copy()
    org_raw = original_xyz.copy()
    cor_raw = corrected_xyz.copy()

    # --------------------------------------------------------
    # STEP 1: TRANSLATION / CENTERING
    # Put all clouds at the same centroid.
    # --------------------------------------------------------
    ref_center, ref_mu = center_points(ref_raw)
    org_center, _ = center_points(org_raw)
    cor_center, _ = center_points(cor_raw)

    # Display around reference centroid
    ref_center_vis = ref_center + ref_mu
    org_center_vis = org_center + ref_mu
    cor_center_vis = cor_center + ref_mu

    # --------------------------------------------------------
    # STEP 2: SCALE
    # Match overall point-cloud size to reference.
    # --------------------------------------------------------
    org_scale = scale_to_reference(
        org_center,
        ref_center,
    )

    cor_scale = scale_to_reference(
        cor_center,
        ref_center,
    )

    ref_scale_vis = ref_center + ref_mu
    org_scale_vis = org_scale + ref_mu
    cor_scale_vis = cor_scale + ref_mu

    # --------------------------------------------------------
    # STEP 3: ROTATION
    # Kabsch rotation -> final aligned shapes
    # --------------------------------------------------------
    org_rot = rotate_to_reference(
        org_scale,
        ref_center,
    )

    cor_rot = rotate_to_reference(
        cor_scale,
        ref_center,
    )

    ref_final = ref_center + ref_mu
    org_final = org_rot + ref_mu
    cor_final = cor_rot + ref_mu

    # Residual after alignment
    org_pa_mm = (
        np.linalg.norm(
            org_final - ref_final,
            axis=-1,
        ).mean()
        * 1000.0
    )

    cor_pa_mm = (
        np.linalg.norm(
            cor_final - ref_final,
            axis=-1,
        ).mean()
        * 1000.0
    )

    print()
    print("Original aligned residual =", org_pa_mm, "mm")
    print("Corrected aligned residual =", cor_pa_mm, "mm")

    # ========================================================
    # DRAW PROCESS + SCENE CONTEXT
    # ========================================================

    # --------------------------------------------------------
    # Real object/mesh name
    # OBJECT_TABLE: object_id -> full BEHAVIOR mesh name
    # ex) World__scene_0__can_of_soda_114__base_link__visuals
    # --------------------------------------------------------
    table = omap.OBJECT_TABLE.get(TARGET, {})
    object_full_name = str(
        table.get(oid, f"object_{oid}")
    )

    parts = object_full_name.split("__")
    object_name = object_full_name

    for i, part in enumerate(parts):
        if (
            part.startswith("scene_")
            and i + 1 < len(parts)
        ):
            object_name = parts[i + 1]
            break

    # --------------------------------------------------------
    # Full initial scene
    # --------------------------------------------------------
    full_gt = (
        b0["gt_scene_flows"][0]
        .detach()
        .cpu()
        .numpy()
    )

    scene_mask0 = (
        exists[0]
        & (~context[0])
    )

    scene0 = full_gt[0][scene_mask0]

    if len(scene0) == 0:
        scene0 = full_gt[0][exists[0]]

    if len(scene0) == 0:
        scene0 = full_gt[0]

    # --------------------------------------------------------
    # Nearby scene context around target object
    # --------------------------------------------------------
    object_center = ref_raw.mean(axis=0)

    dist = np.linalg.norm(
        scene0 - object_center[None, :],
        axis=1,
    )

    # show nearest scene points so object identity/context is visible
    n_context = min(
        2500,
        len(scene0),
    )

    nearest_idx = np.argsort(dist)[:n_context]
    scene_near = scene0[nearest_idx]

    def equal_axis(ax, clouds, pad=1.15):
        clouds = [
            x for x in clouds
            if x is not None and len(x) > 0
        ]

        q = np.concatenate(
            clouds,
            axis=0,
        )

        lo = q.min(axis=0)
        hi = q.max(axis=0)

        center = (lo + hi) / 2.0

        radius = max(
            (hi - lo).max() / 2.0,
            0.005,
        ) * pad

        ax.set_xlim(
            center[0] - radius,
            center[0] + radius,
        )
        ax.set_ylim(
            center[1] - radius,
            center[1] + radius,
        )
        ax.set_zlim(
            center[2] - radius,
            center[2] + radius,
        )

        ax.set_box_aspect((1, 1, 1))
        ax.view_init(
            elev=22,
            azim=-60,
        )

    fig = plt.figure(
        figsize=(22, 12)
    )

    # ========================================================
    # 1. TARGET OBJECT + LOCAL SCENE CONTEXT
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 1,
        projection="3d",
    )

    ax.scatter(
        scene_near[:, 0],
        scene_near[:, 1],
        scene_near[:, 2],
        s=4,
        color="gray",
        alpha=0.10,
    )

    ax.scatter(
        ref_raw[:, 0],
        ref_raw[:, 1],
        ref_raw[:, 2],
        s=22,
        color="orange",
        alpha=0.95,
        label=object_name,
    )

    ax.set_title(
        f"1. Target object in scene\n"
        f"{object_name}",
        fontsize=14,
    )

    ax.legend(fontsize=10)

    equal_axis(
        ax,
        [scene_near, ref_raw],
        pad=1.08,
    )

    # ========================================================
    # 2. RAW SEGMENTED OBJECT
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 2,
        projection="3d",
    )

    ax.scatter(
        ref_raw[:,0],
        ref_raw[:,1],
        ref_raw[:,2],
        s=10,
        color="gray",
        alpha=0.22,
        label="Initial reference",
    )

    ax.scatter(
        org_raw[:,0],
        org_raw[:,1],
        org_raw[:,2],
        s=14,
        color="blue",
        alpha=0.65,
        label="Original Action",
    )

    ax.scatter(
        cor_raw[:,0],
        cor_raw[:,1],
        cor_raw[:,2],
        s=14,
        color="red",
        alpha=0.55,
        label="Corrected Action",
    )

    ax.set_title(
        "2. Segmented object\nRaw prediction",
        fontsize=14,
    )

    ax.legend(fontsize=9)

    equal_axis(
        ax,
        [ref_raw, org_raw, cor_raw],
    )

    # ========================================================
    # 3. TRANSLATION
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 3,
        projection="3d",
    )

    ax.scatter(
        ref_center_vis[:,0],
        ref_center_vis[:,1],
        ref_center_vis[:,2],
        s=9,
        color="gray",
        alpha=0.20,
    )

    ax.scatter(
        org_center_vis[:,0],
        org_center_vis[:,1],
        org_center_vis[:,2],
        s=14,
        color="blue",
        alpha=0.65,
    )

    ax.scatter(
        cor_center_vis[:,0],
        cor_center_vis[:,1],
        cor_center_vis[:,2],
        s=14,
        color="red",
        alpha=0.50,
    )

    ax.set_title(
        "3. Translation removed\nSame center",
        fontsize=14,
    )

    equal_axis(
        ax,
        [
            ref_center_vis,
            org_center_vis,
            cor_center_vis,
        ],
    )

    # ========================================================
    # 4. SCALE
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 4,
        projection="3d",
    )

    ax.scatter(
        ref_scale_vis[:,0],
        ref_scale_vis[:,1],
        ref_scale_vis[:,2],
        s=9,
        color="gray",
        alpha=0.20,
    )

    ax.scatter(
        org_scale_vis[:,0],
        org_scale_vis[:,1],
        org_scale_vis[:,2],
        s=14,
        color="blue",
        alpha=0.65,
    )

    ax.scatter(
        cor_scale_vis[:,0],
        cor_scale_vis[:,1],
        cor_scale_vis[:,2],
        s=14,
        color="red",
        alpha=0.50,
    )

    ax.set_title(
        "4. Scale normalized\nSame overall size",
        fontsize=14,
    )

    equal_axis(
        ax,
        [
            ref_scale_vis,
            org_scale_vis,
            cor_scale_vis,
        ],
    )

    # ========================================================
    # 5. ROTATION / FINAL PA
    # Different markers so overlap is visible
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 5,
        projection="3d",
    )

    ax.scatter(
        ref_final[:,0],
        ref_final[:,1],
        ref_final[:,2],
        s=8,
        color="gray",
        alpha=0.16,
        label="Initial reference",
    )

    ax.scatter(
        org_final[:,0],
        org_final[:,1],
        org_final[:,2],
        s=20,
        marker="x",
        color="blue",
        alpha=0.85,
        label="Original Action",
    )

    ax.scatter(
        cor_final[:,0],
        cor_final[:,1],
        cor_final[:,2],
        s=24,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=0.8,
        alpha=0.80,
        label="Corrected Action",
    )

    ax.set_title(
        "5. Rotation aligned\nFinal PA alignment",
        fontsize=14,
    )

    ax.legend(fontsize=9)

    equal_axis(
        ax,
        [
            ref_final,
            org_final,
            cor_final,
        ],
    )

    # ========================================================
    # 6. DIRECT ORIGINAL vs CORRECTED
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 6,
        projection="3d",
    )

    ax.scatter(
        org_final[:,0],
        org_final[:,1],
        org_final[:,2],
        s=22,
        marker="x",
        color="blue",
        alpha=0.90,
        label="Original Action",
    )

    ax.scatter(
        cor_final[:,0],
        cor_final[:,1],
        cor_final[:,2],
        s=28,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=0.9,
        alpha=0.85,
        label="Corrected Action",
    )

    ax.set_title(
        "6. Original vs Corrected\n"
        "Blue × = Original   Red ○ = Corrected",
        fontsize=14,
    )

    ax.legend(fontsize=9)

    equal_axis(
        ax,
        [
            org_final,
            cor_final,
        ],
    )

    ax.text2D(
        0.02,
        -0.13,
        (
            f"Original PA residual:  {org_pa_mm:.3f} mm\n"
            f"Corrected PA residual: {cor_pa_mm:.3f} mm"
        ),
        transform=ax.transAxes,
        fontsize=11,
    )

    for ax in fig.axes:
        if hasattr(ax, "set_xlabel"):
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

    fig.suptitle(
        f"{TARGET}\n"
        f"Object: {object_name}  |  frame={t}  |  object_id={oid}",
        fontsize=17,
        y=0.985,
    )

    plt.tight_layout(
        rect=[0.01, 0.04, 0.99, 0.92],
        h_pad=4.0,
        w_pad=2.5,
    )

    safe = (
        TARGET
        .replace("/", "_")
        .replace(":", "__")
    )

    out = (
        OUTDIR /
        f"{safe}__t{t}__object_context_pa_process_v2.png"
    )

    fig.savefig(
        out,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("OBJECT FULL NAME =", object_full_name)
    print("OBJECT NAME      =", object_name)
    print("OBJECT ID        =", oid)
    print("POINTS           =", len(ref_raw))
    print("Original PA      =", org_pa_mm, "mm")
    print("Corrected PA     =", cor_pa_mm, "mm")
    print("SAVED            =", out)
    print("DONE")

    break

if not found:
    raise RuntimeError(
        f"Target clip not found: {TARGET}"
    )
