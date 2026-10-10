#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from arguments import parse_args
import dataset_components.dataloader as dlmod
from dataset_components.transforms import compute_flow_derivatives


ROOT = Path(__file__).resolve().parents[1]

TARGET_TASKS = [
    x.strip()
    for x in os.environ.get(
        "PW_TARGET_TASKS",
        "task-0000,task-0001,task-0006,task-0008,task-0010"
    ).split(",")
    if x.strip()
]

TAG = os.environ.get(
    "PW_TAG",
    "_".join(t.replace("task-", "") for t in TARGET_TASKS)
)

KNN_K = int(os.environ.get("PW_KNN_K", "8"))

ACC_BOUNDARY = 0.009832167363499603
MSJ_BOUNDARY = 8.152857428804615e-05

METRICS_CSV = (
    ROOT
    / "diagnostics/results/established_action_metrics_5task/"
      "per_sample_established_metrics_7987.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/"
      "dynamics_to_geometry_failure_analysis"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

if not METRICS_CSV.exists():
    raise FileNotFoundError(METRICS_CSV)


# ============================================================
# Load official 7,987 moved-sample action metrics
# ============================================================

mdf = pd.read_csv(METRICS_CSV)

required = {
    "clip",
    "task",
    "rms_acceleration",
    "mean_squared_jerk",
}

missing = required - set(mdf.columns)

if missing:
    raise RuntimeError(
        f"Missing columns in {METRICS_CSV}: {sorted(missing)}"
    )

mdf["task"] = mdf["task"].astype(str)
mdf["clip"] = mdf["clip"].astype(str)

mdf = mdf[
    mdf["task"].isin(TARGET_TASKS)
].copy()

if len(mdf) == 0:
    raise RuntimeError(
        f"No metric rows for TARGET_TASKS={TARGET_TASKS}"
    )


def canonical_metric_key(row):
    c = str(row["clip"])
    t = str(row["task"])

    if c.startswith("task-"):
        return c

    return f"{t}_{c}"


mdf["canonical_key"] = mdf.apply(
    canonical_metric_key,
    axis=1,
)

if mdf["canonical_key"].duplicated().any():
    dup = mdf.loc[
        mdf["canonical_key"].duplicated(),
        "canonical_key"
    ].head().tolist()

    raise RuntimeError(
        f"Duplicate canonical metric keys: {dup}"
    )

METRIC_BY_KEY = {
    str(r["canonical_key"]): r
    for _, r in mdf.iterrows()
}

TARGET_KEYS = set(METRIC_BY_KEY)

# For resolving a raw __key__ that might not carry the task prefix.
SUFFIX_LOOKUP = {}

for k in TARGET_KEYS:
    if "_" in k:
        suffix = k.split("_", 1)[1]
        SUFFIX_LOOKUP.setdefault(
            suffix,
            []
        ).append(k)


def resolve_key(raw_key):
    raw_key = str(raw_key)

    if raw_key in TARGET_KEYS:
        return raw_key

    if len(TARGET_TASKS) == 1:
        c = f"{TARGET_TASKS[0]}_{raw_key}"
        if c in TARGET_KEYS:
            return c

    matches = SUFFIX_LOOKUP.get(
        raw_key,
        []
    )

    if len(matches) == 1:
        return matches[0]

    return None


print("=" * 90)
print("DYNAMICS -> GEOMETRY FAILURE ANALYSIS")
print("tasks:", TARGET_TASKS)
print("target moved samples:", len(TARGET_KEYS))
print("Global Acc boundary:", ACC_BOUNDARY)
print("Global MSJ boundary:", MSJ_BOUNDARY)
print("=" * 90)


# ============================================================
# Capture raw BEHAVIOR contact metadata before official gather()
# drops non-release fields.
# ============================================================

CONTACT_META = {}

OFFICIAL_BUILD = dlmod.build_flow_sample


def _scalar(x, default=np.nan):
    if x is None:
        return default

    try:
        a = np.asarray(x).reshape(-1)
        if len(a) == 0:
            return default
        return a[0].item()
    except Exception:
        return x


def contact_value(sample, key, default=np.nan):
    return _scalar(
        sample.get(key, default),
        default=default,
    )


def build_with_contact_capture(
    sample,
    domain,
    robot_sampler,
    max_robot_points,
    deterministic=False,
    seed=None,
    force_single_arm=False,
):
    raw_key = str(
        sample.get("__key__", "")
    )

    key = resolve_key(raw_key)

    if (
        key is not None
        and "behavior" in domain
    ):
        lc = bool(
            contact_value(
                sample,
                "__has_left_gripper_finger_collision__",
                False,
            )
        )

        rc = bool(
            contact_value(
                sample,
                "__has_right_gripper_finger_collision__",
                False,
            )
        )

        ld = float(
            contact_value(
                sample,
                "__left_min_distance_to_all_objects__",
                np.nan,
            )
        )

        rd = float(
            contact_value(
                sample,
                "__right_min_distance_to_all_objects__",
                np.nan,
            )
        )

        finite_dist = [
            x for x in [ld, rd]
            if np.isfinite(x)
        ]

        min_dist = (
            float(min(finite_dist))
            if finite_dist
            else np.nan
        )

        CONTACT_META[key] = {
            "left_collision": lc,
            "right_collision": rc,
            "contact_any": bool(lc or rc),
            "left_min_object_distance_m": ld,
            "right_min_object_distance_m": rd,
            "min_object_distance_m": min_dist,
        }

    return OFFICIAL_BUILD(
        sample,
        domain=domain,
        robot_sampler=robot_sampler,
        max_robot_points=max_robot_points,
        deterministic=deterministic,
        seed=seed,
        force_single_arm=force_single_arm,
    )


dlmod.build_flow_sample = build_with_contact_capture


# ============================================================
# Helpers
# ============================================================

def mask3(batch, name):
    x = batch[name]

    if isinstance(x, torch.Tensor):
        x = x.bool()
    else:
        x = torch.as_tensor(x).bool()

    if (
        x.ndim == 4
        and x.shape[-1] == 1
    ):
        x = x.squeeze(-1)

    return x


def nan_rms(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(
        np.sqrt(
            np.mean(x ** 2)
        )
    )


def nan_mean(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(
        np.mean(x)
    )


def nan_p95(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(
        np.percentile(
            x,
            95,
        )
    )


# ============================================================
# B5: GT scene higher-order dynamics
# ============================================================

def scene_temporal_metrics(
    gt,
    valid,
    moved,
):
    """
    gt:    (T,N,3) positions
    valid: (T,N)
    moved: (T,N)

    Acceleration uses PointWorld official compute_flow_derivatives().
    Scene jerk is the midpoint temporal derivative of that acceleration.

    Metrics are measured only on official valid & moved points.
    """

    velocity, acceleration = (
        compute_flow_derivatives(gt)
    )

    # derivative(acceleration) = scene jerk
    jerk, _ = compute_flow_derivatives(
        acceleration
    )

    vm = valid & moved

    acc_mag = np.linalg.norm(
        acceleration,
        axis=-1,
    )

    jerk_mag = np.linalg.norm(
        jerk,
        axis=-1,
    )

    acc_vals = acc_mag[vm]
    jerk_vals = jerk_mag[vm]

    # Scene motion magnitude controls
    step = (
        gt[1:]
        - gt[:-1]
    )

    step_mag = np.linalg.norm(
        step,
        axis=-1,
    )

    transition_valid = (
        valid[1:]
        & valid[:-1]
    )

    transition_moved = (
        moved[1:]
        | moved[:-1]
    )

    tm = (
        transition_valid
        & transition_moved
    )

    step_vals = step_mag[tm]

    return {
        "scene_acc_rms":
            nan_rms(acc_vals),

        "scene_acc_mean":
            nan_mean(acc_vals),

        "scene_acc_p95":
            nan_p95(acc_vals),

        # Mean Squared Scene Jerk
        "scene_msj":
            float(
                np.mean(
                    jerk_vals ** 2
                )
            )
            if len(jerk_vals)
            else np.nan,

        "scene_jerk_rms":
            nan_rms(jerk_vals),

        "scene_jerk_p95":
            nan_p95(jerk_vals),

        "scene_step_mean_m":
            nan_mean(step_vals),

        "scene_step_p95_m":
            nan_p95(step_vals),
    }


# ============================================================
# B7 GT side:
# local structural preservation / GT rigidity eligibility
#
# Same core principle as our Local Isometry metric:
# construct KNN graph at t0 and measure edge-length distortion.
# ============================================================

def gt_local_structure_metrics(
    gt,
    exists,
    valid,
    moved,
    k=8,
):
    support = (
        exists[0]
        & valid[-1]
        & moved[-1]
    )

    idx_support = np.flatnonzero(
        support
    )

    n = len(idx_support)

    if n < 3:
        return {
            "gt_structure_points": n,
            "gt_local_isometry_mean_mm": np.nan,
            "gt_local_isometry_p95_mm": np.nan,
            "gt_temporal_jitter_mean_mm": np.nan,
            "gt_temporal_jitter_p95_mm": np.nan,
        }

    pts = gt[
        :,
        idx_support,
        :
    ].astype(
        np.float64,
        copy=False,
    )

    kk = min(
        int(k),
        n - 1,
    )

    tree = cKDTree(
        pts[0]
    )

    _, neigh = tree.query(
        pts[0],
        k=kk + 1,
    )

    neigh = np.asarray(
        neigh
    )[:, 1:]

    src = np.repeat(
        np.arange(n)[:, None],
        kk,
        axis=1,
    )

    p0a = pts[0, src]
    p0b = pts[0, neigh]

    d0 = np.linalg.norm(
        p0a - p0b,
        axis=-1,
    )

    all_distortion = []
    all_jitter = []

    prev_d = d0

    for t in range(1, len(pts)):
        pa = pts[t, src]
        pb = pts[t, neigh]

        dt = np.linalg.norm(
            pa - pb,
            axis=-1,
        )

        all_distortion.append(
            np.abs(
                dt - d0
            ).reshape(-1)
        )

        all_jitter.append(
            np.abs(
                dt - prev_d
            ).reshape(-1)
        )

        prev_d = dt

    if all_distortion:
        distortion = np.concatenate(
            all_distortion
        )
    else:
        distortion = np.array([])

    if all_jitter:
        jitter = np.concatenate(
            all_jitter
        )
    else:
        jitter = np.array([])

    return {
        "gt_structure_points": n,

        "gt_local_isometry_mean_mm":
            1000.0
            * nan_mean(
                distortion
            ),

        "gt_local_isometry_p95_mm":
            1000.0
            * nan_p95(
                distortion
            ),

        "gt_temporal_jitter_mean_mm":
            1000.0
            * nan_mean(
                jitter
            ),

        "gt_temporal_jitter_p95_mm":
            1000.0
            * nan_p95(
                jitter
            ),
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if int(args.batch_size) != 1:
        raise RuntimeError(
            "Use --batch_size=1 for deterministic per-clip extraction."
        )

    loader, info = dlmod.build_dataloader(
        args,
        mode="test",
        rank=0,
        world_size=1,
        override_splits="test",
    )

    if loader is None:
        raise RuntimeError(
            "Could not build test dataloader."
        )

    rows = []
    found = set()

    for batch in loader:
        raw_key = str(
            batch["__key__"][0]
        )

        key = resolve_key(
            raw_key
        )

        if (
            key is None
            or key not in TARGET_KEYS
            or key in found
        ):
            continue

        metric = METRIC_BY_KEY[
            key
        ]

        gt = (
            batch["gt_scene_flows"][0]
            .cpu()
            .numpy()
            .astype(
                np.float64,
                copy=False,
            )
        )

        exists = (
            mask3(
                batch,
                "scene_exists",
            )[0]
            .cpu()
            .numpy()
            .astype(bool)
        )

        context = (
            mask3(
                batch,
                "scene_context_mask",
            )[0]
            .cpu()
            .numpy()
            .astype(bool)
        )

        supervised = (
            mask3(
                batch,
                "scene_supervised_mask",
            )[0]
            .cpu()
            .numpy()
            .astype(bool)
        )

        moved = (
            mask3(
                batch,
                "scene_moved_mask",
            )[0]
            .cpu()
            .numpy()
            .astype(bool)
        )

        valid = (
            (~context)
            & exists
            & supervised
        )

        temporal = scene_temporal_metrics(
            gt,
            valid,
            moved,
        )

        structure = gt_local_structure_metrics(
            gt,
            exists,
            valid,
            moved,
            k=KNN_K,
        )

        acc = float(
            metric[
                "rms_acceleration"
            ]
        )

        msj = float(
            metric[
                "mean_squared_jerk"
            ]
        )

        row = {
            "clip": key,
            "task": str(
                metric["task"]
            ),

            "action_rms_acceleration":
                acc,

            "action_msj":
                msj,

            "acc_boundary":
                ACC_BOUNDARY,

            "msj_boundary":
                MSJ_BOUNDARY,

            "acc_outside":
                bool(
                    acc
                    > ACC_BOUNDARY
                ),

            "msj_outside":
                bool(
                    msj
                    > MSJ_BOUNDARY
                ),

            "acc_excess_ratio":
                max(
                    acc
                    / ACC_BOUNDARY
                    - 1.0,
                    0.0,
                ),

            "msj_excess_ratio":
                max(
                    msj
                    / MSJ_BOUNDARY
                    - 1.0,
                    0.0,
                ),
        }

        # Preserve official moved-L2 if present
        for candidate in [
            "filtered_l2_moved",
            "moved_l2",
            "l2_moved",
        ]:
            if candidate in metric.index:
                row[
                    "official_moved_l2"
                ] = float(
                    metric[candidate]
                )
                break

        row.update(
            temporal
        )

        row.update(
            structure
        )

        contact = CONTACT_META.get(
            key,
            {}
        )

        row.update({
            "contact_any":
                contact.get(
                    "contact_any",
                    np.nan,
                ),

            "left_collision":
                contact.get(
                    "left_collision",
                    np.nan,
                ),

            "right_collision":
                contact.get(
                    "right_collision",
                    np.nan,
                ),

            "min_object_distance_m":
                contact.get(
                    "min_object_distance_m",
                    np.nan,
                ),

            "left_min_object_distance_m":
                contact.get(
                    "left_min_object_distance_m",
                    np.nan,
                ),

            "right_min_object_distance_m":
                contact.get(
                    "right_min_object_distance_m",
                    np.nan,
                ),
        })

        rows.append(
            row
        )

        found.add(
            key
        )

        if (
            len(found) <= 10
            or len(found) % 100 == 0
            or len(found) == len(TARGET_KEYS)
        ):
            print(
                f"[{len(found)}/{len(TARGET_KEYS)}] "
                f"{key} | "
                f"sceneAcc={row['scene_acc_rms']:.6g} | "
                f"sceneMSJ={row['scene_msj']:.6g} | "
                f"contact={row['contact_any']} | "
                f"GTiso={row['gt_local_isometry_mean_mm']:.4f}mm"
            )

        if len(found) == len(
            TARGET_KEYS
        ):
            break

    out = pd.DataFrame(
        rows
    )

    OUT = (
        OUTDIR
        / f"{TAG}__gt_contact_structure.csv"
    )

    out.to_csv(
        OUT,
        index=False,
    )

    print()
    print("=" * 90)
    print("DONE")
    print("saved:", OUT)
    print(
        "found:",
        len(found),
        "/",
        len(TARGET_KEYS),
    )

    missing_keys = (
        TARGET_KEYS
        - found
    )

    print(
        "missing:",
        len(missing_keys)
    )

    if len(missing_keys):
        print(
            "first missing:",
            sorted(
                missing_keys
            )[:10]
        )

    if len(out):
        print()
        print(
            "By task:"
        )

        for task, g in out.groupby(
            "task"
        ):
            print(
                task,
                "N=",
                len(g),
                "| AccOutside=",
                f"{100*g['acc_outside'].mean():.2f}%",
                "| MSJOutside=",
                f"{100*g['msj_outside'].mean():.2f}%",
                "| Contact=",
                (
                    f"{100*pd.to_numeric(g['contact_any'], errors='coerce').mean():.2f}%"
                    if g["contact_any"].notna().any()
                    else "NA"
                ),
            )

    print("=" * 90)


if __name__ == "__main__":
    main()
