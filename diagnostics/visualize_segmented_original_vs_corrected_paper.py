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

    # ========================================================
    # PAPER VISUALIZATION
    # Metric / PA above remains unchanged.
    # Only visualization below is replaced.
    # ========================================================

    from matplotlib.patches import Rectangle

    def _np0(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim > 0 and x.shape[0] == 1:
            x = x[0]
        return x

    def _fps(points, k=40):
        p = np.asarray(points, dtype=np.float64)
        n = len(p)

        if n <= k:
            return np.arange(n, dtype=int)

        c = p.mean(axis=0)
        first = int(np.argmax(np.linalg.norm(p - c, axis=1)))

        selected = [first]
        d2 = np.sum((p - p[first]) ** 2, axis=1)

        for _ in range(1, k):
            j = int(np.argmax(d2))
            selected.append(j)

            new_d2 = np.sum((p - p[j]) ** 2, axis=1)
            d2 = np.minimum(d2, new_d2)

        return np.asarray(selected, dtype=int)

    def _pca_basis(x):
        xc = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        return vt[:2].T

    def _p2(x, basis):
        return np.asarray(x) @ basis

    def _kabsch(src, ref):
        h = src.T @ ref
        u, _, vt = np.linalg.svd(h)
        r = u @ vt

        if np.linalg.det(r) < 0:
            vt[-1] *= -1
            r = u @ vt

        return r

    def _hull(points):
        p = np.unique(np.asarray(points, dtype=float), axis=0)

        if len(p) <= 2:
            return p

        p = p[np.lexsort((p[:, 1], p[:, 0]))]

        def cross(o, a, b):
            return (
                (a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0])
            )

        lo = []
        for q in p:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
                lo.pop()
            lo.append(q)

        hi = []
        for q in p[::-1]:
            while len(hi) >= 2 and cross(hi[-2], hi[-1], q) <= 0:
                hi.pop()
            hi.append(q)

        return np.asarray(lo[:-1] + hi[:-1])

    def _outline(ax, p, color, lw=1.5, alpha=0.7):
        h = _hull(p)

        if len(h) >= 3:
            h = np.vstack([h, h[0]])
            ax.plot(
                h[:, 0], h[:, 1],
                color=color,
                linewidth=lw,
                alpha=alpha,
            )

    def _draw(
        ax,
        ref,
        org,
        cor,
        landmarks,
        title,
        connections=True,
    ):
        # full shape
        ax.scatter(
            ref[:, 0], ref[:, 1],
            s=8, c="0.55", alpha=0.18,
        )
        ax.scatter(
            org[:, 0], org[:, 1],
            s=8, c="blue", alpha=0.10,
        )
        ax.scatter(
            cor[:, 0], cor[:, 1],
            s=8, c="red", alpha=0.10,
        )

        _outline(ax, ref, "0.45", 1.4, 0.65)
        _outline(ax, org, "blue", 1.7, 0.75)
        _outline(ax, cor, "red", 1.7, 0.75)

        ri = ref[landmarks]
        oi = org[landmarks]
        ci = cor[landmarks]

        # representative corresponding points
        ax.scatter(
            ri[:, 0], ri[:, 1],
            s=20, c="0.35",
            zorder=5,
        )
        ax.scatter(
            oi[:, 0], oi[:, 1],
            s=32, marker="x",
            c="blue", linewidths=1.3,
            label="Original",
            zorder=6,
        )
        ax.scatter(
            ci[:, 0], ci[:, 1],
            s=34, marker="o",
            facecolors="none",
            edgecolors="red",
            linewidths=1.3,
            label="Corrected",
            zorder=6,
        )

        if connections:
            for r, o, c in zip(ri, oi, ci):
                ax.plot(
                    [r[0], o[0]], [r[1], o[1]],
                    color="blue",
                    linewidth=0.5,
                    alpha=0.15,
                )
                ax.plot(
                    [r[0], c[0]], [r[1], c[1]],
                    color="red",
                    linewidth=0.5,
                    alpha=0.15,
                )

        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    def _project(points, K, E, H, W):
        p = np.asarray(points, dtype=np.float64)

        ph = np.concatenate(
            [p, np.ones((len(p), 1))],
            axis=1,
        )

        # PointWorld extrinsic: transformed world -> camera
        cam = (E @ ph.T).T[:, :3]

        pix = (K @ cam.T).T
        z = pix[:, 2]

        ok = np.isfinite(z) & (z > 1e-6)

        uv = np.zeros((len(p), 2), dtype=float)
        uv[ok] = pix[ok, :2] / z[ok, None]

        inside = (
            ok
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )

        return uv, inside

    # --------------------------------------------------------
    # FULL object for visualization.
    # Metric still uses original idx/ref_raw/org_raw/cor_raw above.
    # --------------------------------------------------------

    full_mask = (
        (object_id[0] == oid)
        & (object_id[t] == oid)
        & exists[0]
        & exists[t]
    )

    full_idx = np.where(full_mask)[0]

    if len(full_idx) < 8:
        full_idx = idx.copy()

    ref_full = (
        b0["gt_scene_flows"][0, 0, full_idx]
        .detach().cpu().numpy()
    )

    org_full = pred_original[t, full_idx]
    cor_full = pred_corrected[t, full_idx]

    # --------------------------------------------------------
    # PA display transforms estimated using the SAME metric points
    # --------------------------------------------------------

    ref_mu2 = ref_raw.mean(axis=0, keepdims=True)
    org_mu2 = org_raw.mean(axis=0, keepdims=True)
    cor_mu2 = cor_raw.mean(axis=0, keepdims=True)

    ref_c2 = ref_raw - ref_mu2
    org_c2 = org_raw - org_mu2
    cor_c2 = cor_raw - cor_mu2

    ref_full_c = ref_full - ref_mu2
    org_full_c = org_full - org_mu2
    cor_full_c = cor_full - cor_mu2

    nr = max(np.linalg.norm(ref_c2), 1e-12)
    no = max(np.linalg.norm(org_c2), 1e-12)
    nc = max(np.linalg.norm(cor_c2), 1e-12)

    so = nr / no
    sc = nr / nc

    ref_s2 = ref_c2
    org_s2 = org_c2 * so
    cor_s2 = cor_c2 * sc

    ref_full_s = ref_full_c
    org_full_s = org_full_c * so
    cor_full_s = cor_full_c * sc

    Ro = _kabsch(org_s2, ref_s2)
    Rc = _kabsch(cor_s2, ref_s2)

    ref_full_r = ref_full_s
    org_full_r = org_full_s @ Ro
    cor_full_r = cor_full_s @ Rc

    # --------------------------------------------------------
    # PCA is visualization ONLY. Metric remains full 3D above.
    # --------------------------------------------------------

    basis = _pca_basis(ref_c2)

    raw_ref2 = _p2(ref_full - ref_mu2, basis)
    raw_org2 = _p2(org_full - ref_mu2, basis)
    raw_cor2 = _p2(cor_full - ref_mu2, basis)

    cen_ref2 = _p2(ref_full_c, basis)
    cen_org2 = _p2(org_full_c, basis)
    cen_cor2 = _p2(cor_full_c, basis)

    scl_ref2 = _p2(ref_full_s, basis)
    scl_org2 = _p2(org_full_s, basis)
    scl_cor2 = _p2(cor_full_s, basis)

    rot_ref2 = _p2(ref_full_r, basis)
    rot_org2 = _p2(org_full_r, basis)
    rot_cor2 = _p2(cor_full_r, basis)

    landmarks = _fps(
        ref_full_c,
        k=min(40, len(ref_full_c)),
    )

    # --------------------------------------------------------
    # Object name
    # --------------------------------------------------------

    if (
        TARGET == "task-0000_episode_00001310-260:271"
        and int(oid) == 2
    ):
        vis_object_name = "radio_89"
    else:
        vis_object_name = f"object_{oid}"

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig = plt.figure(figsize=(18, 10.5))

    fig.suptitle(
        f"{TARGET}\n"
        f"Target: {vis_object_name} | frame={t} | object_id={oid}",
        fontsize=15,
        y=0.985,
    )

    axes = [
        fig.add_subplot(2, 3, i + 1)
        for i in range(6)
    ]

    # --------------------------------------------------------
    # PANEL 1: actual RGB + radio box
    # --------------------------------------------------------

    object0_idx = np.where(
        (object_id[0] == oid)
        & exists[0]
    )[0]

    object0_xyz = (
        b0["gt_scene_flows"][0, 0, object0_idx]
        .detach().cpu().numpy()
    )

    camera_views = []

    for key in b0.keys():
        if not key.endswith("_initial_rgb"):
            continue

        prefix = key[:-len("_initial_rgb")]
        ikey = prefix + "_intrinsic"
        ekey = prefix + "_extrinsic"

        if ikey not in b0 or ekey not in b0:
            continue

        rgb = _np0(b0[key])
        K = _np0(b0[ikey])
        E = _np0(b0[ekey])

        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            continue

        H, W = rgb.shape[:2]

        uv, inside = _project(
            object0_xyz,
            K,
            E,
            H,
            W,
        )

        camera_views.append(
            (
                int(inside.sum()),
                prefix,
                rgb,
                uv,
                inside,
            )
        )

    ax = axes[0]

    if camera_views:
        camera_views.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        nvis, cam_name, rgb, uv, inside = camera_views[0]

        rgb = np.asarray(rgb)

        if rgb.dtype != np.uint8:
            if np.nanmax(rgb) <= 1.5:
                rgb = (
                    np.clip(rgb, 0, 1) * 255
                ).astype(np.uint8)
            else:
                rgb = np.clip(
                    rgb, 0, 255
                ).astype(np.uint8)

        ax.imshow(rgb)

        pts = uv[inside]

        if len(pts) >= 3:
            x0, y0 = np.percentile(
                pts, 2, axis=0
            )
            x1, y1 = np.percentile(
                pts, 98, axis=0
            )

            pad = 10

            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(rgb.shape[1] - 1, x1 + pad)
            y1 = min(rgb.shape[0] - 1, y1 + pad)

            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="orange",
                    linewidth=3,
                )
            )

            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=8,
                c="orange",
                alpha=0.65,
            )

            ax.annotate(
                "radio_89",
                xy=(x0, y0),
                xytext=(x0, max(15, y0 - 18)),
                fontsize=12,
                color="white",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="orange",
                    ec="orange",
                    alpha=0.95,
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color="orange",
                    lw=2,
                ),
            )

        ax.set_title(
            f"1. Target object in RGB\n"
            f"{vis_object_name} | {cam_name}",
            fontsize=12,
        )
        ax.axis("off")

    else:
        ax.text(
            0.5, 0.5,
            "RGB camera payload not found",
            ha="center",
            va="center",
        )
        ax.set_title(
            f"1. Target object\n{vis_object_name}"
        )
        ax.axis("off")

    # --------------------------------------------------------
    # PANELS 2~6
    # --------------------------------------------------------

    _draw(
        axes[1],
        raw_ref2,
        raw_org2,
        raw_cor2,
        landmarks,
        "2. Full segmented radio\nRaw prediction",
    )
    axes[1].legend(
        fontsize=8,
        frameon=False,
    )

    _draw(
        axes[2],
        cen_ref2,
        cen_org2,
        cen_cor2,
        landmarks,
        "3. Translation removed\nSame center",
    )

    _draw(
        axes[3],
        scl_ref2,
        scl_org2,
        scl_cor2,
        landmarks,
        "4. Scale normalized\nSame overall size",
    )

    _draw(
        axes[4],
        rot_ref2,
        rot_org2,
        rot_cor2,
        landmarks,
        "5. Rotation aligned\nFinal PA alignment",
    )

    _draw(
        axes[5],
        rot_ref2,
        rot_org2,
        rot_cor2,
        landmarks,
        "6. Original vs Corrected\n"
        "Blue × = Original | Red ○ = Corrected",
    )

    axes[5].text(
        0.02,
        -0.08,
        (
            f"Original PA residual:  {org_pa_mm:.3f} mm\n"
            f"Corrected PA residual: {cor_pa_mm:.3f} mm"
        ),
        transform=axes[5].transAxes,
        fontsize=10,
        va="top",
    )

    axes[5].legend(
        fontsize=8,
        frameon=False,
    )

    plt.tight_layout(
        rect=(0, 0.03, 1, 0.94)
    )

    safe = (
        TARGET
        .replace(":", "__")
        .replace("/", "_")
    )

    outpath = (
        OUTDIR
        / f"{safe}__t{t}__paper_procrustes.png"
    )

    fig.savefig(
        outpath,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("=" * 80)
    print("SAVED        =", outpath)
    print("OBJECT       =", vis_object_name)
    print("OBJECT ID    =", oid)
    print("FULL POINTS  =", len(ref_full))
    print("LANDMARKS    =", len(landmarks))
    print("Original PA  =", org_pa_mm, "mm")
    print("Corrected PA =", cor_pa_mm, "mm")
    print("=" * 80)

    break
if not found:
    raise RuntimeError(
        f"Target clip not found: {TARGET}"
    )
