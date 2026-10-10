#!/usr/bin/env python3

import os
import runpy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pmod

from evaluation.tester import Tester
import diagnostics.behavior_object_mapping_smoke as objmap


ROOT = Path(__file__).resolve().parents[1]

TARGET_CSV = Path(os.environ["PW_OBJECT_TARGET_CSV"])
OUT = Path(os.environ["PW_OBJECT_OUT"])
EXPECTED = int(os.environ["PW_EXPECTED_SAMPLES"])

MIN_POINTS = 20
PAIRWISE_CAP = 256


target_df = pd.read_csv(TARGET_CSV)
TARGET = set(target_df["clip"].astype(str))

if len(TARGET) != EXPECTED:
    raise RuntimeError(
        f"TARGET={len(TARGET)} EXPECTED={EXPECTED}"
    )


def mask3(batch, name):
    x = batch[name].bool()

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise RuntimeError(
            f"{name}: bad shape {tuple(x.shape)}"
        )

    return x


def kabsch_residual_mm(x0, x1):
    """
    Best rigid SE(3) alignment x0 -> x1.
    Returns per-point residual in mm.
    """

    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)

    if len(x0) < 3:
        return None

    c0 = x0.mean(axis=0)
    c1 = x1.mean(axis=0)

    X = x0 - c0
    Y = x1 - c1

    H = X.T @ Y

    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    aligned = (R @ X.T).T + c1

    residual = np.linalg.norm(
        aligned - x1,
        axis=1,
    ) * 1000.0

    return residual


# ============================================================
# Published Multi-Body Neural Scene Flow consistency
# Exact implementation already validated in
# behavior_radio_top9_published_isometry.py
# ============================================================

D_THR = 0.03
POWER_ITERATIONS = 10


def mbnsf_pairwise_distance_matrix(x):
    diff = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def mbnsf_affinity(source, projected, d_thr=D_THR):
    d = mbnsf_pairwise_distance_matrix(source)
    d_hat = mbnsf_pairwise_distance_matrix(projected)

    A = 1.0 - ((d - d_hat) ** 2) / (d_thr ** 2)
    A = np.maximum(A, 0.0)

    return A


def mbnsf_score_eq8(A):
    return float(A.mean())


def mbnsf_score_eq10(A, n_iter=POWER_ITERATIONS):
    n = A.shape[0]

    v = np.ones(n, dtype=np.float64)

    for _ in range(n_iter):
        Av = A @ v
        norm = np.linalg.norm(Av)

        if norm == 0:
            return 0.0

        v = Av / norm

    return float((v.T @ A @ v) / n)



def deterministic_subset(n, key, cap=PAIRWISE_CAP):

    if n <= cap:
        return np.arange(n)

    seed = int(
        hashlib.sha1(
            key.encode("utf-8")
        ).hexdigest()[:8],
        16,
    )

    rng = np.random.default_rng(seed)

    return np.sort(
        rng.choice(
            n,
            size=cap,
            replace=False,
        )
    )


def pairwise_change_mm(x0, x1, key):
    """
    Absolute preservation error of pairwise distances.
    Diagnostic cross-check.
    """

    n = len(x0)

    if n < 3:
        return None

    idx = deterministic_subset(
        n,
        key,
    )

    a = np.asarray(x0[idx], np.float64)
    b = np.asarray(x1[idx], np.float64)

    da = np.linalg.norm(
        a[:, None, :] - a[None, :, :],
        axis=-1,
    )

    db = np.linalg.norm(
        b[:, None, :] - b[None, :, :],
        axis=-1,
    )

    tri = np.triu_indices(
        len(idx),
        k=1,
    )

    change = (
        np.abs(db[tri] - da[tri])
        * 1000.0
    )

    return change


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
            if key in TARGET
            and key not in found
        ]

        if not selected:
            if len(found) >= EXPECTED:
                break
            continue

        if "scene_object_id" not in batch:
            raise RuntimeError(
                "scene_object_id missing"
            )

        # --------------------------------------------
        # Keep object labels on CPU.
        # --------------------------------------------
        object_ids = batch["scene_object_id"]

        if isinstance(object_ids, torch.Tensor):
            object_ids_np = (
                object_ids.detach()
                .cpu()
                .numpy()
            )
        else:
            object_ids_np = np.asarray(
                object_ids
            )

        if (
            object_ids_np.ndim == 4
            and object_ids_np.shape[-1] == 1
        ):
            object_ids_np = (
                object_ids_np.squeeze(-1)
            )

        if object_ids_np.ndim != 3:
            raise RuntimeError(
                f"object IDs shape "
                f"{object_ids_np.shape}"
            )

        # --------------------------------------------
        # Official model forward.
        # --------------------------------------------
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

        pred = (
            outputs["scene_flows"]
            .detach()
            .cpu()
            .numpy()
        )

        gt = (
            batch_gpu["gt_scene_flows"]
            .detach()
            .cpu()
            .numpy()
        )

        exists = (
            mask3(
                batch,
                "scene_exists",
            )
            .cpu()
            .numpy()
        )

        context = (
            mask3(
                batch,
                "scene_context_mask",
            )
            .cpu()
            .numpy()
        )

        supervised = (
            mask3(
                batch,
                "scene_supervised_mask",
            )
            .cpu()
            .numpy()
        )

        moved = (
            mask3(
                batch,
                "scene_moved_mask",
            )
            .cpu()
            .numpy()
        )

        valid = (
            (~context)
            & exists
            & supervised
        )

        for i in selected:

            key = keys[i]

            ids = object_ids_np[i]

            temporal_exact = np.array_equal(
                ids,
                np.broadcast_to(
                    ids[0:1],
                    ids.shape,
                ),
            )

            if not temporal_exact:
                raise RuntimeError(
                    f"{key}: object IDs changed "
                    "over time"
                )

            table = objmap.OBJECT_TABLE.get(
                key,
                {},
            )

            unique_ids = np.unique(
                ids[0]
            )

            # ==================================================
            # Dominant moved object:
            # object with largest number of official
            # valid & moved point-timestep observations.
            # ==================================================
            ranked = []

            for oid in unique_ids:

                obj_mask = (
                    ids[0] == oid
                )

                moved_valid_count = int(
                    (
                        valid[i, 1:, obj_mask]
                        & moved[i, 1:, obj_mask]
                    ).sum()
                )

                point_count_t0 = int(
                    (
                        obj_mask
                        & exists[i, 0]
                    ).sum()
                )

                ranked.append({
                    "object_id": int(oid),
                    "object_name":
                        str(
                            table.get(
                                int(oid),
                                "",
                            )
                        ),
                    "moved_valid_count":
                        moved_valid_count,
                    "point_count_t0":
                        point_count_t0,
                })

            ranked.sort(
                key=lambda r: (
                    -r["moved_valid_count"],
                    -r["point_count_t0"],
                    r["object_id"],
                )
            )

            if not ranked:
                raise RuntimeError(
                    f"{key}: no objects"
                )

            dominant = ranked[0]
            oid = dominant["object_id"]

            # Same final-valid object identities,
            # matching prior rigid-object analysis.
            obj_final = (
                (ids[0] == oid)
                & valid[i, -1]
            )

            n = int(
                obj_final.sum()
            )

            row = {
                "clip": key,
                "task":
                    key.split(
                        "_episode_"
                    )[0],
                "object_id": oid,
                "object_name":
                    dominant[
                        "object_name"
                    ],
                "moved_valid_count":
                    dominant[
                        "moved_valid_count"
                    ],
                "object_points_t0":
                    dominant[
                        "point_count_t0"
                    ],
                "n_final_valid":
                    n,
                "temporal_object_id_exact":
                    temporal_exact,
            }

            if n < MIN_POINTS:

                row["status"] = (
                    "too_few_points"
                )

                results.append(row)
                found.add(key)
                continue

            x0 = gt[i, 0, obj_final]
            xgt = gt[i, -1, obj_final]
            xpred = pred[i, -1, obj_final]

            # Published MBNSF physical/isometry consistency.
            # EXACT same object identities for GT and prediction.
            A_gt = mbnsf_affinity(
                x0,
                xgt,
                d_thr=D_THR,
            )

            A_pred = mbnsf_affinity(
                x0,
                xpred,
                d_thr=D_THR,
            )

            gt_mbnsf_eq8 = mbnsf_score_eq8(A_gt)
            pred_mbnsf_eq8 = mbnsf_score_eq8(A_pred)

            gt_mbnsf_eq10 = mbnsf_score_eq10(
                A_gt,
                POWER_ITERATIONS,
            )

            pred_mbnsf_eq10 = mbnsf_score_eq10(
                A_pred,
                POWER_ITERATIONS,
            )


            gt_res = kabsch_residual_mm(
                x0,
                xgt,
            )

            pred_res = kabsch_residual_mm(
                x0,
                xpred,
            )

            # IMPORTANT:
            # GT and Prediction must use EXACTLY the same
            # deterministic object-point subset.
            gt_pair = pairwise_change_mm(
                x0,
                xgt,
                key,
            )

            pred_pair = pairwise_change_mm(
                x0,
                xpred,
                key,
            )

            row.update({
                "status": "ok",

                "gt_mbnsf_eq8":
                    gt_mbnsf_eq8,

                "pred_mbnsf_eq8":
                    pred_mbnsf_eq8,

                "mbnsf_eq8_gap":
                    gt_mbnsf_eq8
                    - pred_mbnsf_eq8,

                "gt_mbnsf_eq10":
                    gt_mbnsf_eq10,

                "pred_mbnsf_eq10":
                    pred_mbnsf_eq10,

                "mbnsf_eq10_gap":
                    gt_mbnsf_eq10
                    - pred_mbnsf_eq10,


                "gt_se3_mean_mm":
                    float(
                        np.mean(gt_res)
                    ),

                "gt_se3_median_mm":
                    float(
                        np.median(gt_res)
                    ),

                "gt_se3_p95_mm":
                    float(
                        np.quantile(
                            gt_res,
                            .95,
                        )
                    ),

                "pred_se3_mean_mm":
                    float(
                        np.mean(pred_res)
                    ),

                "pred_se3_median_mm":
                    float(
                        np.median(pred_res)
                    ),

                "pred_se3_p95_mm":
                    float(
                        np.quantile(
                            pred_res,
                            .95,
                        )
                    ),

                "excess_se3_mean_mm":
                    float(
                        np.mean(pred_res)
                        - np.mean(gt_res)
                    ),

                "excess_se3_p95_mm":
                    float(
                        np.quantile(
                            pred_res,
                            .95,
                        )
                        - np.quantile(
                            gt_res,
                            .95,
                        )
                    ),

                "gt_pair_median_mm":
                    float(
                        np.median(gt_pair)
                    ),

                "gt_pair_p95_mm":
                    float(
                        np.quantile(
                            gt_pair,
                            .95,
                        )
                    ),

                "pred_pair_median_mm":
                    float(
                        np.median(pred_pair)
                    ),

                "pred_pair_p95_mm":
                    float(
                        np.quantile(
                            pred_pair,
                            .95,
                        )
                    ),

                "excess_pair_median_mm":
                    float(
                        np.median(pred_pair)
                        - np.median(gt_pair)
                    ),

                "excess_pair_p95_mm":
                    float(
                        np.quantile(
                            pred_pair,
                            .95,
                        )
                        - np.quantile(
                            gt_pair,
                            .95,
                        )
                    ),
            })

            results.append(row)
            found.add(key)

        print(
            f"processed {len(found)}/{EXPECTED}"
        )

        if len(found) >= EXPECTED:
            break

    missing = TARGET - found

    if missing:
        print(
            "FIRST MISSING =",
            sorted(missing)[:20],
        )
        raise RuntimeError(
            f"found={len(found)} "
            f"expected={EXPECTED}"
        )

    df = (
        pd.DataFrame(results)
        .drop_duplicates("clip")
        .sort_values("clip")
    )

    if len(df) != EXPECTED:
        raise RuntimeError(
            f"rows={len(df)} "
            f"expected={EXPECTED}"
        )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT,
        index=False,
    )

    print()
    print("=" * 70)
    print("OBJECT STRUCTURE RESULT")
    print("=" * 70)

    print(
        df["status"]
        .value_counts(
            dropna=False
        )
    )

    ok = df[
        df["status"] == "ok"
    ]

    print(
        "OK =",
        len(ok),
    )

    if len(ok):

        print(
            "GT SE3 mean median =",
            ok[
                "gt_se3_mean_mm"
            ].median(),
        )

        print(
            "PRED SE3 mean median =",
            ok[
                "pred_se3_mean_mm"
            ].median(),
        )

        print(
            "GT pair p95 median =",
            ok[
                "gt_pair_p95_mm"
            ].median(),
        )

        print(
            "PRED pair p95 median =",
            ok[
                "pred_pair_p95_mm"
            ].median(),
        )

    print(
        "SAVED =",
        OUT,
    )
    print(
        "OBJECT_STRUCTURE_EVAL_SUCCESS=True"
    )

    return {
        f"diagnostic/{split}/"
        "object_structure_eval": 1.0
    }


# ============================================================
# Runtime-only validated object-ID augmentation.
# Official tracked code remains untouched.
# ============================================================

dlmod.build_flow_sample = (
    objmap.build_flow_sample_with_object_ids
)

pmod.grid_sample_transform = (
    objmap.grid_sample_with_object_ids
)

pmod.gather = (
    objmap.gather_with_object_ids
)

Tester._accumulate_metrics = capture


runpy.run_path(
    "eval.py",
    run_name="__main__",
)
