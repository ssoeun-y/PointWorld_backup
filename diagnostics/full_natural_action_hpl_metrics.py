#!/usr/bin/env python3

"""
Full natural-action PointWorld 3D scene-flow evaluation.

Official components reused:
  - PointWorld arguments.parse_args
  - PointWorld evaluation.tester.Tester
  - PointWorld checkpoint loading
  - PointWorld official test dataloader
  - PointWorld official prediction:
        outputs["scene_flows"]
  - PointWorld GT absolute scene positions:
        batch["gt_scene_flows"]
  - PointWorld official valid supervision:
        (~scene_context_mask) & scene_exists & scene_supervised_mask
  - PointWorld official moved mask:
        scene_moved_mask

External standard metric:
  HPLFlowNet CVPR 2019 evaluation_utils.py::evaluate_3d
    EPE3D
    Acc3DS
    Acc3DR
    Outliers3D

IMPORTANT:
  We DO NOT use batch["gt_scene_flows_relative"] for metric GT.
  Scene-flow vectors are reconstructed directly from raw absolute GT XYZ:

      gt_flow(t) = gt_xyz(t) - gt_xyz(t0)

  and prediction:

      pred_flow(t) = pred_xyz(t) - pred_xyz(t0)

This avoids the previously audited relative-flow clipping problem.

Tracked PointWorld source files are not modified.
"""

import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from arguments import parse_args
from evaluation.tester import Tester
from dataset_components.constants import RELEASE_EVAL_SPLITS


# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------

QUALITY_CSV = os.environ.get(
    "QUALITY_CSV",
    "diagnostics/results/natural_action_quality_per_sample.csv",
)

OUT_CSV = os.environ.get(
    "OUT_CSV",
    "diagnostics/results/natural_action_official_hpl3d_full.csv",
)

EXPECTED_MOVED_SAMPLES = int(
    os.environ.get("EXPECTED_MOVED_SAMPLES", "5288")
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def reset_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# HPLFlowNet CVPR 2019 official evaluate_3d definition
#
# Source:
# https://github.com/laoreja/HPLFlowNet/blob/master/evaluation_utils.py
#
# Only np.float -> float conversion is modernized because np.float was
# removed from newer NumPy. Numerical definition is unchanged.
# ---------------------------------------------------------------------

def evaluate_3d_hpl_official(sf_pred: np.ndarray,
                             sf_gt: np.ndarray):
    """
    HPLFlowNet official evaluate_3d metric.

    Inputs:
        sf_pred : (N, 3), meters
        sf_gt   : (N, 3), meters

    Returns:
        EPE3D
        Acc3DS
        Acc3DR
        Outliers3D
    """

    sf_pred = np.asarray(sf_pred)
    sf_gt = np.asarray(sf_gt)

    if sf_pred.ndim != 2 or sf_pred.shape[-1] != 3:
        raise ValueError(
            f"sf_pred must be (N,3), got {sf_pred.shape}"
        )

    if sf_gt.shape != sf_pred.shape:
        raise ValueError(
            f"Shape mismatch: pred={sf_pred.shape}, gt={sf_gt.shape}"
        )

    if sf_pred.shape[0] == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    # Official HPLFlowNet code.
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)

    EPE3D = l2_norm.mean()

    sf_norm = np.linalg.norm(sf_gt, axis=-1)

    relative_err = l2_norm / (sf_norm + 1e-4)

    acc3d_strict = (
        np.logical_or(
            l2_norm < 0.05,
            relative_err < 0.05,
        )
        .astype(float)
        .mean()
    )

    acc3d_relax = (
        np.logical_or(
            l2_norm < 0.10,
            relative_err < 0.10,
        )
        .astype(float)
        .mean()
    )

    outlier = (
        np.logical_or(
            l2_norm > 0.30,
            relative_err > 0.10,
        )
        .astype(float)
        .mean()
    )

    return (
        float(EPE3D),
        float(acc3d_strict),
        float(acc3d_relax),
        float(outlier),
    )


# ---------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------

def bool_mask(x: torch.Tensor) -> torch.Tensor:
    """
    Convert PointWorld masks to (B,T,N) bool.

    Some collated masks are (B,T,N,1), others may already be (B,T,N).
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    x = x.bool()

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise ValueError(
            f"Expected mask shape (B,T,N), got {tuple(x.shape)}"
        )

    return x


def build_official_valid_moved_mask(batch):
    """
    Reproduce PointWorld official evaluation supervision support.

    tester.py / losses.py:
        pred = ~context
        pred_exists_supervised = pred & exists & supervised

    Then restrict to official moved points.
    """

    context = bool_mask(batch["scene_context_mask"])
    exists = bool_mask(batch["scene_exists"])
    supervised = bool_mask(batch["scene_supervised_mask"])
    moved = bool_mask(batch["scene_moved_mask"])

    valid = (~context) & exists & supervised

    valid_moved = valid & moved

    return valid_moved


# ---------------------------------------------------------------------
# Absolute XYZ -> scene flow
# ---------------------------------------------------------------------

def absolute_xyz_to_flow(xyz: torch.Tensor) -> torch.Tensor:
    """
    xyz: (B,T,N,3)

    Raw displacement relative to t0:
        flow[t] = xyz[t] - xyz[0]
    """

    if xyz.ndim != 4 or xyz.shape[-1] != 3:
        raise ValueError(
            f"Expected (B,T,N,3), got {tuple(xyz.shape)}"
        )

    return xyz - xyz[:, 0:1]


# ---------------------------------------------------------------------
# Quality CSV
# ---------------------------------------------------------------------

def detect_column(fieldnames, candidates, required=True):
    for name in candidates:
        if name in fieldnames:
            return name

    if required:
        raise RuntimeError(
            f"Could not find any of {candidates}. "
            f"CSV columns={fieldnames}"
        )

    return None


def load_quality_table(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Natural Action quality CSV not found: {path}"
        )

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(f"No header in {path}")

        fields = list(reader.fieldnames)

        key_col = detect_column(
            fields,
            [
                "sample_key",
                "clip",
                "__key__",
                "key",
                "clip_key",
            ],
        )

        task_col = detect_column(
            fields,
            [
                "task",
                "task_id",
                "task_name",
            ],
            required=False,
        )

        irregularity_col = detect_column(
            fields,
            [
                "irregularity_score",
                "action_irregularity_core",
                "action_irregularity_core_pct",
                "action_irregularity_score",
                "quality_score",
                "core_irregularity",
            ],
            required=False,
        )

        group_col = detect_column(
            fields,
            [
                "quality_group",
                "action_quality_group",
                "group",
            ],
            required=False,
        )

        raw_rows = []

        for row in reader:
            key = str(row[key_col]).strip()

            if not key:
                continue

            task = (
                str(row.get(task_col, "")).strip()
                if task_col is not None else ""
            )

            irregularity = float(row[irregularity_col])

            raw_rows.append(
                {
                    "key": key,
                    "task": task,
                    "irregularity_score": irregularity,
                }
            )

        # Reproduce the existing Natural Action definition:
        # within each task:
        #   bottom quartile irregularity = HQ
        #   top quartile irregularity    = LQ
        #   middle 50%                   = MID
        by_task = {}

        for r in raw_rows:
            by_task.setdefault(r["task"], []).append(
                r["irregularity_score"]
            )

        quartiles = {}

        for task, vals in by_task.items():
            vals = np.asarray(vals, dtype=float)

            q25 = float(np.quantile(vals, 0.25))
            q75 = float(np.quantile(vals, 0.75))

            quartiles[task] = (q25, q75)

            print(
                f"[quality] {task}: "
                f"N={len(vals)}, q25={q25:.6f}, q75={q75:.6f}"
            )

        table = {}

        for r in raw_rows:
            key = r["key"]
            task = r["task"]
            score = r["irregularity_score"]

            if key in table:
                raise RuntimeError(
                    f"Duplicate sample key in quality CSV: {key}"
                )

            q25, q75 = quartiles[task]

            if score <= q25:
                group = "HQ"
            elif score >= q75:
                group = "LQ"
            else:
                group = "MID"

            table[key] = {
                "task": task,
                "irregularity_score": score,
                "quality_group": group,
            }

    print(
        f"[quality] loaded {len(table)} rows "
        f"from {path}"
    )

    print(
        "[quality] columns:",
        {
            "key": key_col,
            "task": task_col,
            "irregularity": irregularity_col,
            "group": group_col,
        },
    )

    return table


# ---------------------------------------------------------------------
# Per-sample metric
# ---------------------------------------------------------------------

def evaluate_one_sample(pred_xyz,
                        gt_xyz,
                        valid_moved):
    """
    Inputs are one sample:
        pred_xyz    : (T,N,3)
        gt_xyz      : (T,N,3)
        valid_moved : (T,N)

    HPL metric is evaluated on all official valid+moved future
    point/timestep pairs pooled within the sample.

    t0 is explicitly excluded.
    """

    if pred_xyz.shape != gt_xyz.shape:
        raise ValueError(
            f"pred/gt shape mismatch: "
            f"{tuple(pred_xyz.shape)} vs {tuple(gt_xyz.shape)}"
        )

    if pred_xyz.ndim != 3 or pred_xyz.shape[-1] != 3:
        raise ValueError(
            f"Expected pred XYZ (T,N,3), "
            f"got {tuple(pred_xyz.shape)}"
        )

    if valid_moved.shape != pred_xyz.shape[:2]:
        raise ValueError(
            f"mask/XYZ mismatch: "
            f"mask={tuple(valid_moved.shape)}, "
            f"xyz={tuple(pred_xyz.shape)}"
        )

    # -----------------------------------------------------------------
    # IMPORTANT:
    # Do not use gt_scene_flows_relative here.
    #
    # Reconstruct flow from raw absolute XYZ.
    # -----------------------------------------------------------------

    pred_flow = pred_xyz - pred_xyz[0:1]
    gt_flow = gt_xyz - gt_xyz[0:1]

    # t0 displacement is zero by definition and is not a prediction
    # horizon for scene-flow evaluation.
    pred_flow = pred_flow[1:]
    gt_flow = gt_flow[1:]
    valid_moved = valid_moved[1:]

    mask = valid_moved.bool()

    n_eval_points = int(mask.sum().item())

    if n_eval_points == 0:
        return None

    sf_pred = (
        pred_flow[mask]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    sf_gt = (
        gt_flow[mask]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    (
        epe3d,
        acc3ds,
        acc3dr,
        outliers3d,
    ) = evaluate_3d_hpl_official(
        sf_pred,
        sf_gt,
    )

    return {
        "n_eval_points": n_eval_points,
        "EPE3D_m": epe3d,
        "EPE3D_mm": epe3d * 1000.0,
        "Acc3DS": acc3ds,
        "Acc3DR": acc3dr,
        "Outliers3D": outliers3d,
    }


# ---------------------------------------------------------------------
# Full evaluator
# ---------------------------------------------------------------------

class NaturalActionHPL3DEvaluator(Tester):

    @torch.no_grad()
    def run_natural_action_hpl(self):

        quality = load_quality_table(QUALITY_CSV)

        out_path = Path(OUT_CSV)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        output_fields = [
            "sample_key",
            "domain",
            "task",
            "irregularity_score",
            "quality_group",
            "n_eval_points",
            "EPE3D_m",
            "EPE3D_mm",
            "Acc3DS",
            "Acc3DR",
            "Outliers3D",
        ]

        matched = 0
        skipped_not_quality = 0
        skipped_no_moved = 0
        total_loader_samples = 0

        seen_keys = set()

        self.model.eval()

        with out_path.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=output_fields,
            )

            writer.writeheader()

            # Official release evaluation split(s).
            for split in RELEASE_EVAL_SPLITS:

                print(
                    f"\n[run] official PointWorld "
                    f"split={split}"
                )

                # Reuse official Tester loader.
                dl, _ = self._build_eval_loader(
                    split,
                    enable_mask=False,
                )

                total_batches = (
                    self.args.eval_num_batches
                    if self.args.eval_num_batches > 0
                    else None
                )

                iterator = tqdm(
                    dl,
                    desc=f"HPL3D [{split}]",
                    total=total_batches,
                )

                for i, batch in enumerate(iterator):

                    if (
                        total_batches is not None
                        and i >= total_batches
                    ):
                        break

                    # Exact pattern from PointWorld official Tester.
                    batch = {
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

                    keys = [
                        str(k)
                        for k in batch["__key__"]
                    ]

                    domains = [
                        str(d)
                        for d in batch["__domain__"]
                    ]

                    batch_size = len(keys)
                    total_loader_samples += batch_size

                    # -------------------------------------------------
                    # Optimization only:
                    # If no sample in this batch belongs to the 5288
                    # natural moved table, do not waste a forward pass.
                    # -------------------------------------------------

                    wanted = [
                        key in quality
                        for key in keys
                    ]

                    if not any(wanted):
                        skipped_not_quality += batch_size
                        continue

                    # -------------------------------------------------
                    # Official PointWorld forward.
                    # -------------------------------------------------

                    outputs = self.model(
                        batch,
                        training=False,
                    )

                    if bool(
                        outputs.get(
                            "_has_nan_outputs",
                            False,
                        )
                    ):
                        raise RuntimeError(
                            f"NaN model output at "
                            f"batch {i}, keys={keys}"
                        )

                    # -------------------------------------------------
                    # CRITICAL CONTRACT
                    #
                    # Prediction absolute XYZ:
                    #   outputs["scene_flows"]
                    #
                    # Raw GT absolute XYZ:
                    #   batch["gt_scene_flows"]
                    #
                    # NOT batch["scene_flows"].
                    # NOT batch["gt_scene_flows_relative"].
                    # -------------------------------------------------

                    pred_xyz_all = outputs[
                        "scene_flows"
                    ]

                    gt_xyz_all = batch[
                        "gt_scene_flows"
                    ]

                    if (
                        pred_xyz_all.shape
                        != gt_xyz_all.shape
                    ):
                        raise RuntimeError(
                            "Official pred/GT shapes differ: "
                            f"pred={tuple(pred_xyz_all.shape)}, "
                            f"gt={tuple(gt_xyz_all.shape)}"
                        )

                    valid_moved_all = (
                        build_official_valid_moved_mask(
                            batch
                        )
                    )

                    for b in range(batch_size):

                        key = keys[b]

                        if key not in quality:
                            skipped_not_quality += 1
                            continue

                        if key in seen_keys:
                            raise RuntimeError(
                                f"Duplicate evaluated key: {key}"
                            )

                        q = quality[key]

                        result = evaluate_one_sample(
                            pred_xyz_all[b],
                            gt_xyz_all[b],
                            valid_moved_all[b],
                        )

                        if result is None:
                            skipped_no_moved += 1
                            continue

                        row = {
                            "sample_key": key,
                            "domain": domains[b],
                            "task": q["task"],
                            "irregularity_score": (
                                q["irregularity_score"]
                            ),
                            "quality_group": (
                                q["quality_group"]
                            ),
                        }

                        row.update(result)

                        writer.writerow(row)
                        f.flush()

                        seen_keys.add(key)
                        matched += 1

                    iterator.set_postfix(
                        matched=matched,
                        target=len(quality),
                    )

        # -------------------------------------------------------------
        # Final integrity checks.
        # -------------------------------------------------------------

        quality_keys = set(quality.keys())
        missing_keys = sorted(
            quality_keys - seen_keys
        )

        print("\n========================================")
        print("FULL NATURAL ACTION HPL3D COMPLETE")
        print("========================================")
        print(
            f"quality rows       : {len(quality)}"
        )
        print(
            f"evaluated matched  : {matched}"
        )
        print(
            f"loader samples seen: {total_loader_samples}"
        )
        print(
            f"not in quality CSV : {skipped_not_quality}"
        )
        print(
            f"zero moved support : {skipped_no_moved}"
        )
        print(
            f"missing quality key: {len(missing_keys)}"
        )
        print(
            f"output             : {out_path}"
        )

        if missing_keys:
            miss_path = out_path.with_name(
                out_path.stem
                + "_missing_keys.txt"
            )

            with miss_path.open("w") as mf:
                for key in missing_keys:
                    mf.write(key + "\n")

            print(
                f"missing-key list   : {miss_path}"
            )

        # The current natural-action table is expected to contain
        # exactly 5,288 moved samples.
        if (
            EXPECTED_MOVED_SAMPLES > 0
            and len(quality) != EXPECTED_MOVED_SAMPLES
        ):
            raise RuntimeError(
                "Quality CSV row count differs from "
                f"expected {EXPECTED_MOVED_SAMPLES}: "
                f"got {len(quality)}"
            )

        if matched != len(quality):
            raise RuntimeError(
                "Did not evaluate every natural-action "
                f"quality sample: matched={matched}, "
                f"quality={len(quality)}, "
                f"missing={len(missing_keys)}, "
                f"zero_moved={skipped_no_moved}"
            )

        print(
            "\nSUCCESS: every Natural Action sample "
            "was joined and evaluated."
        )


# ---------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------

def sanity_check_hpl():
    """
    Perfect prediction must produce:
        EPE3D     = 0
        Acc3DS    = 1
        Acc3DR    = 1
        Outliers  = 0
    """

    gt = np.asarray(
        [
            [0.01, 0.00, 0.00],
            [0.02, 0.01, 0.00],
            [0.00, 0.00, 0.03],
        ],
        dtype=np.float64,
    )

    pred = gt.copy()

    result = evaluate_3d_hpl_official(
        pred,
        gt,
    )

    expected = (
        0.0,
        1.0,
        1.0,
        0.0,
    )

    if not np.allclose(
        np.asarray(result),
        np.asarray(expected),
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(
            f"HPL sanity failed: {result}"
        )

    print(
        "[sanity] HPL perfect prediction:",
        result,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    # Tester itself enforces release eval seed=42 unless the special
    # missing-confidence path is used. We explicitly reset here too,
    # matching the project's reproducibility protocol.
    reset_seed(42)

    sanity_check_hpl()

    args = parse_args()

    evaluator = NaturalActionHPL3DEvaluator(
        args
    )

    evaluator.run_natural_action_hpl()
