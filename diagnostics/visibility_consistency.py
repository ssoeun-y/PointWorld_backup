#!/usr/bin/env python3
"""PointWorld DROID visibility–consistency diagnostic.

Purpose
-------
Join the official DROID test dataloader masks/actions with previously saved
prediction-consistency raw sequence files by ``__key__``.  This script does
NOT run the PointWorld model and does NOT modify official source files.

Primary outputs
---------------
summary.json
per_timestep.csv
per_sequence.csv
mask_transition.csv
cohort_metrics.json
correlations.csv
action_visibility_bins.csv
manifest.json

Run from the PointWorld repository root.  See ``--help`` for arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as exc:  # pragma: no cover - checked at runtime on server
    raise RuntimeError("scipy is required: pip/conda install scipy") from exc


EPS = 1e-12
EXPECTED_TIMESTEPS = 11

RAW_ALIASES: dict[str, tuple[str, ...]] = {
    "key": ("__key__", "key", "sample_key", "sequence_key"),
    "pred": (
        "prediction_scene_flows",
        "prediction",
        "pred",
        "pred_scene_flows",
        "scene_flows_pred",
        "predicted_scene_flows",
        "pred_positions",
    ),
    "gt": (
        "gt",
        "gt_scene_flows",
        "scene_flows_gt",
        "target",
        "target_scene_flows",
        "gt_positions",
    ),
    "valid": (
        "official_valid_mask",
        "official_valid",
        "valid_mask",
        "eval_valid_mask",
        "pred_exists_mask",
    ),
    "moved": ("scene_moved_mask", "moved_mask", "official_moved_mask"),
    "static": ("scene_static_mask", "static_mask", "official_static_mask"),
    "official_l2": (
        "official_l2",
        "official_l2_mean",
        "l2",
        "l2_mean",
        "recomputed_l2",
    ),
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Analyze DROID visibility/mask churn versus saved PointWorld "
            "prediction consistency without model inference."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="DROID WDS root used by the official PointWorld test dataloader.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="Directory containing the 62 saved raw prediction .pt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Result directory. A timestamped diagnostics/results directory is used if omitted.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("test", "val", "train"),
        help="Official split to iterate.",
    )
    parser.add_argument(
        "--expected_sequences",
        type=int,
        default=62,
        help="Hard audit target. The run fails if this many unique samples are not processed.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Dataloader workers. Zero is safest for deterministic diagnostics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dataset sampling seed.",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Optional smoke-test limit. Set expected_sequences consistently when used.",
    )
    parser.add_argument(
        "--l2_tolerance",
        type=float,
        default=1e-7,
        help="Absolute tolerance for stored official L2 reproduction.",
    )
    parser.add_argument(
        "--tensor_tolerance",
        type=float,
        default=1e-6,
        help="Max absolute tolerance for raw GT versus dataloader GT audit.",
    )
    parser.add_argument(
        "--allow_missing_raw_reference_l2",
        action="store_true",
        help=(
            "Allow raw files without a stored official-L2 scalar. The script still computes "
            "the official-mask L2, but will not label it independently reproduced."
        ),
    )
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Run synthetic helper tests without importing PointWorld and exit.",
    )
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


def run_git(args: Sequence[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def official_git_diff() -> dict[str, Any]:
    """Report tracked modifications outside diagnostics/."""
    code, output = run_git(["diff", "--name-only"])
    names = [line.strip() for line in output.splitlines() if line.strip()] if code == 0 else []
    official = [name for name in names if not name.startswith("diagnostics/")]
    return {
        "git_command_ok": code == 0,
        "all_modified_tracked_files": names,
        "official_modified_tracked_files": official,
        "official_files_modified": bool(official),
    }


def normalize_key(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one key, got {len(value)}: {value!r}")
        return normalize_key(value[0])
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected scalar key array, got shape {value.shape}")
        return normalize_key(value.reshape(-1)[0])
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar key tensor, got shape {tuple(value.shape)}")
        return normalize_key(value.detach().cpu().item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def recursive_lookup(obj: Any, aliases: Sequence[str], depth: int = 0) -> Any | None:
    if not isinstance(obj, Mapping) or depth > 3:
        return None
    for alias in aliases:
        if alias in obj:
            return obj[alias]
    for value in obj.values():
        if isinstance(value, Mapping):
            found = recursive_lookup(value, aliases, depth + 1)
            if found is not None:
                return found
    return None


def to_numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def strip_batch(arr: np.ndarray, expected_ndim: int) -> np.ndarray:
    while arr.ndim > expected_ndim and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def as_bool_mask(value: Any, name: str) -> np.ndarray:
    arr = strip_batch(to_numpy(value), 2)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected [T,N] mask, got {arr.shape}")
    return arr.astype(bool, copy=False)


def as_xyz(value: Any, name: str) -> np.ndarray:
    arr = strip_batch(to_numpy(value, dtype=np.float64), 3)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{name}: expected [T,N,3], got {arr.shape}")
    return arr


def as_timeseries(value: Any, name: str) -> np.ndarray:
    arr = to_numpy(value, dtype=np.float64)
    # All diagnostics run with batch_size=1. Scalar time series are often
    # collated as [1,T], while vector time series are [1,T,D].
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected [T,D], got {arr.shape}")
    return arr


def scalar_or_none(value: Any | None) -> float | None:
    if value is None:
        return None
    arr = to_numpy(value, dtype=np.float64).reshape(-1)
    if arr.size != 1:
        return None
    result = float(arr[0])
    return result if math.isfinite(result) else None


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def nanmedian(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan")


# ---------------------------------------------------------------------------
# Raw prediction indexing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawRecord:
    key: str
    path: Path


def load_pt(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def index_raw_files(raw_dir: Path) -> tuple[dict[str, RawRecord], list[str], list[dict[str, Any]]]:
    files = sorted(raw_dir.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found under {raw_dir}")

    index: dict[str, RawRecord] = {}
    duplicates: list[str] = []
    unresolved: list[dict[str, Any]] = []

    for path in files:
        obj = load_pt(path)
        key_value = recursive_lookup(obj, RAW_ALIASES["key"])
        if key_value is None:
            unresolved.append(
                {
                    "path": str(path),
                    "top_level_keys": sorted(obj.keys()) if isinstance(obj, Mapping) else [],
                }
            )
            continue
        key = normalize_key(key_value)
        if key in index:
            duplicates.append(key)
        else:
            index[key] = RawRecord(key=key, path=path)

    return index, sorted(set(duplicates)), unresolved


def load_official_l2_by_key() -> dict[str, float]:
    csv_path = Path(
        "diagnostics/results/"
        "prediction_consistency_droid_unfiltered_full_20260804_135952/"
        "per_sequence.csv"
    )

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    table = pd.read_csv(csv_path)

    required = {"key", "official_l2_mean"}
    missing = required - set(table.columns)
    if missing:
        raise KeyError(
            f"{csv_path}: missing required columns {sorted(missing)}"
        )

    if table["key"].duplicated().any():
        duplicates = sorted(
            table.loc[table["key"].duplicated(), "key"]
            .astype(str)
            .unique()
        )
        raise RuntimeError(
            f"Duplicate keys in official L2 reference: {duplicates}"
        )

    result = {
        str(row["key"]): float(row["official_l2_mean"])
        for _, row in table.iterrows()
    }

    if len(result) != 62:
        raise RuntimeError(
            f"Expected 62 official L2 references, found {len(result)}"
        )

    return result


def extract_raw_arrays(path: Path) -> dict[str, Any]:
    obj = load_pt(path)
    if not isinstance(obj, Mapping):
        raise TypeError(f"Raw file must contain a mapping: {path} -> {type(obj)!r}")

    values: dict[str, Any] = {}
    missing_required: list[str] = []
    for canonical, aliases in RAW_ALIASES.items():
        value = recursive_lookup(obj, aliases)
        values[canonical] = value
        if canonical in {"key", "pred", "gt"} and value is None:
            missing_required.append(canonical)

    if missing_required:
        raise KeyError(
            f"{path}: missing raw fields {missing_required}. "
            f"Top-level keys: {sorted(obj.keys())}"
        )

    result: dict[str, Any] = {
        "key": normalize_key(values["key"]),
        "pred": as_xyz(values["pred"], "raw prediction"),
        "gt": as_xyz(values["gt"], "raw GT"),
        "valid": as_bool_mask(values["valid"], "raw official valid") if values["valid"] is not None else None,
        "moved": as_bool_mask(values["moved"], "raw moved") if values["moved"] is not None else None,
        "static": as_bool_mask(values["static"], "raw static") if values["static"] is not None else None,
        "official_l2": scalar_or_none(values["official_l2"]),
        "top_level_keys": sorted(obj.keys()),
    }
    return result


# ---------------------------------------------------------------------------
# Metric accumulators
# ---------------------------------------------------------------------------


@dataclass
class ScalarAccumulator:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def add(self, values: np.ndarray | Sequence[float] | float) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.total += float(arr.sum())
        self.total_sq += float(np.square(arr).sum())
        self.minimum = min(self.minimum, float(arr.min()))
        self.maximum = max(self.maximum, float(arr.max()))

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class GlobalCohortAccumulator:
    point_count: int = 0
    l2: ScalarAccumulator = field(default_factory=ScalarAccumulator)
    error_variation: ScalarAccumulator = field(default_factory=ScalarAccumulator)
    velocity_error: ScalarAccumulator = field(default_factory=ScalarAccumulator)
    acceleration_error: ScalarAccumulator = field(default_factory=ScalarAccumulator)
    pred_jerk: ScalarAccumulator = field(default_factory=ScalarAccumulator)
    gt_jerk: ScalarAccumulator = field(default_factory=ScalarAccumulator)
    jerk_error: ScalarAccumulator = field(default_factory=ScalarAccumulator)
    oscillation_point_count: int = 0
    repeated_oscillation_count: int = 0
    strong_alternation_count: int = 0
    sign_change_count: ScalarAccumulator = field(default_factory=ScalarAccumulator)

    def update(self, pred: np.ndarray, gt: np.ndarray, mask_tn: np.ndarray) -> dict[str, float]:
        """Update using a fixed point cohort represented at every supplied timestep."""
        if pred.shape != gt.shape:
            raise ValueError(f"Pred/GT mismatch: {pred.shape} vs {gt.shape}")
        if mask_tn.shape != pred.shape[:2]:
            raise ValueError(f"Mask mismatch: {mask_tn.shape} vs {pred.shape[:2]}")

        fixed = mask_tn.all(axis=0)
        n_points = int(fixed.sum())
        self.point_count += n_points
        if n_points == 0:
            return empty_metric_row()

        p = pred[:, fixed]
        g = gt[:, fixed]
        error = np.linalg.norm(p - g, axis=-1)  # [T,P]
        self.l2.add(error)

        if p.shape[0] >= 2:
            err_var = np.abs(np.diff(error, axis=0))
            vel_err = np.linalg.norm(np.diff(p, axis=0) - np.diff(g, axis=0), axis=-1)
            self.error_variation.add(err_var)
            self.velocity_error.add(vel_err)
        else:
            err_var = np.empty((0, n_points))
            vel_err = np.empty((0, n_points))

        if p.shape[0] >= 3:
            acc_err = np.linalg.norm(np.diff(p, n=2, axis=0) - np.diff(g, n=2, axis=0), axis=-1)
            self.acceleration_error.add(acc_err)
        else:
            acc_err = np.empty((0, n_points))

        if p.shape[0] >= 4:
            pred_jerk = np.linalg.norm(np.diff(p, n=3, axis=0), axis=-1)
            gt_jerk = np.linalg.norm(np.diff(g, n=3, axis=0), axis=-1)
            jerk_err = np.linalg.norm(
                np.diff(p, n=3, axis=0) - np.diff(g, n=3, axis=0), axis=-1
            )
            self.pred_jerk.add(pred_jerk)
            self.gt_jerk.add(gt_jerk)
            self.jerk_error.add(jerk_err)
        else:
            pred_jerk = gt_jerk = jerk_err = np.empty((0, n_points))

        sign_counts, repeated, strong = error_oscillation(error)
        self.oscillation_point_count += n_points
        self.repeated_oscillation_count += int(repeated.sum())
        self.strong_alternation_count += int(strong.sum())
        self.sign_change_count.add(sign_counts)

        return {
            "point_count": n_points,
            "l2": float(error.mean()),
            "error_variation": float(err_var.mean()) if err_var.size else float("nan"),
            "velocity_error": float(vel_err.mean()) if vel_err.size else float("nan"),
            "acceleration_error": float(acc_err.mean()) if acc_err.size else float("nan"),
            "pred_jerk": float(pred_jerk.mean()) if pred_jerk.size else float("nan"),
            "gt_jerk": float(gt_jerk.mean()) if gt_jerk.size else float("nan"),
            "jerk_error": float(jerk_err.mean()) if jerk_err.size else float("nan"),
            "repeated_oscillation_rate": float(repeated.mean()),
            "strong_alternation_rate": float(strong.mean()),
            "mean_sign_changes": float(sign_counts.mean()),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "fixed_point_count_summed_across_sequences": self.point_count,
            "l2": self.l2.summary(),
            "error_variation": self.error_variation.summary(),
            "velocity_error": self.velocity_error.summary(),
            "acceleration_error": self.acceleration_error.summary(),
            "prediction_jerk": self.pred_jerk.summary(),
            "gt_jerk": self.gt_jerk.summary(),
            "jerk_error": self.jerk_error.summary(),
            "oscillation_point_count": self.oscillation_point_count,
            "repeated_oscillation_rate": safe_div(
                self.repeated_oscillation_count, self.oscillation_point_count
            ),
            "strong_alternation_rate": safe_div(
                self.strong_alternation_count, self.oscillation_point_count
            ),
            "sign_change_count": self.sign_change_count.summary(),
        }


def empty_metric_row() -> dict[str, float]:
    return {
        "point_count": 0,
        "l2": float("nan"),
        "error_variation": float("nan"),
        "velocity_error": float("nan"),
        "acceleration_error": float("nan"),
        "pred_jerk": float("nan"),
        "gt_jerk": float("nan"),
        "jerk_error": float("nan"),
        "repeated_oscillation_rate": float("nan"),
        "strong_alternation_rate": float("nan"),
        "mean_sign_changes": float("nan"),
    }


def error_oscillation(error_tp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sign-change count and two explicitly defined oscillation flags.

    repeated_oscillation: at least two sign changes in consecutive non-zero
    error increments.

    strong_alternation: sign changes on at least half of all possible adjacent
    increment pairs.  This definition is stored in summary.json so it is not
    confused with any prior script's implementation.
    """
    if error_tp.ndim != 2:
        raise ValueError(f"Expected [T,P] error, got {error_tp.shape}")
    if error_tp.shape[0] < 3:
        zeros = np.zeros(error_tp.shape[1], dtype=np.int64)
        return zeros, zeros.astype(bool), zeros.astype(bool)

    increments = np.diff(error_tp, axis=0)
    signs = np.sign(increments)
    # Exact zeros break adjacency instead of being forced positive/negative.
    adjacent_nonzero = (signs[:-1] != 0) & (signs[1:] != 0)
    changes = adjacent_nonzero & (signs[:-1] != signs[1:])
    count = changes.sum(axis=0).astype(np.int64)
    max_possible = max(error_tp.shape[0] - 2, 1)
    repeated = count >= 2
    strong = (count / max_possible) >= 0.5
    return count, repeated, strong


def masked_l2(error_tn: np.ndarray, mask_tn: np.ndarray) -> float:
    values = error_tn[mask_tn]
    return float(values.mean()) if values.size else float("nan")


def dynamic_temporal_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    class_mask: np.ndarray,
) -> dict[str, float]:
    """Temporal metrics with interval-wise continuity masks."""
    error = np.linalg.norm(pred - gt, axis=-1)
    point_mask = valid & class_mask

    result: dict[str, float] = {
        "l2": masked_l2(error, point_mask),
        "point_observation_count": int(point_mask.sum()),
    }

    if pred.shape[0] >= 2:
        pair = point_mask[:-1] & point_mask[1:]
        vel_err = np.linalg.norm(np.diff(pred, axis=0) - np.diff(gt, axis=0), axis=-1)
        err_var = np.abs(np.diff(error, axis=0))
        result["velocity_error"] = masked_l2(vel_err, pair)
        result["error_variation"] = masked_l2(err_var, pair)
        result["velocity_interval_count"] = int(pair.sum())
    else:
        result.update(
            velocity_error=float("nan"),
            error_variation=float("nan"),
            velocity_interval_count=0,
        )

    if pred.shape[0] >= 3:
        triple = point_mask[:-2] & point_mask[1:-1] & point_mask[2:]
        acc_err = np.linalg.norm(
            np.diff(pred, n=2, axis=0) - np.diff(gt, n=2, axis=0), axis=-1
        )
        result["acceleration_error"] = masked_l2(acc_err, triple)
        result["acceleration_interval_count"] = int(triple.sum())
    else:
        result.update(acceleration_error=float("nan"), acceleration_interval_count=0)

    if pred.shape[0] >= 4:
        quad = (
            point_mask[:-3]
            & point_mask[1:-2]
            & point_mask[2:-1]
            & point_mask[3:]
        )
        pred_j = np.linalg.norm(np.diff(pred, n=3, axis=0), axis=-1)
        gt_j = np.linalg.norm(np.diff(gt, n=3, axis=0), axis=-1)
        jerk_e = np.linalg.norm(
            np.diff(pred, n=3, axis=0) - np.diff(gt, n=3, axis=0), axis=-1
        )
        result["prediction_jerk"] = masked_l2(pred_j, quad)
        result["gt_jerk"] = masked_l2(gt_j, quad)
        result["jerk_error"] = masked_l2(jerk_e, quad)
        result["jerk_interval_count"] = int(quad.sum())
    else:
        result.update(
            prediction_jerk=float("nan"),
            gt_jerk=float("nan"),
            jerk_error=float("nan"),
            jerk_interval_count=0,
        )
    return result


# ---------------------------------------------------------------------------
# Masks, transitions, action descriptors
# ---------------------------------------------------------------------------


def transition_row(
    key: str,
    mask_name: str,
    previous_timestep: int,
    current_timestep: int,
    previous: np.ndarray,
    current: np.ndarray,
) -> dict[str, Any]:
    intersection = previous & current
    dropped = previous & ~current
    gained = ~previous & current
    union = previous | current
    prev_count = int(previous.sum())
    curr_count = int(current.sum())
    keep_count = int(intersection.sum())
    drop_count = int(dropped.sum())
    gain_count = int(gained.sum())
    return {
        "key": key,
        "mask": mask_name,
        "previous_timestep": previous_timestep,
        "current_timestep": current_timestep,
        "previous_count": prev_count,
        "current_count": curr_count,
        "retained_count": keep_count,
        "dropped_count": drop_count,
        "gained_count": gain_count,
        "retention_rate": safe_div(keep_count, prev_count),
        "drop_rate": safe_div(drop_count, prev_count),
        "gain_rate_relative_to_current": safe_div(gain_count, curr_count),
        "jaccard": safe_div(keep_count, int(union.sum())),
        "churn_rate": safe_div(drop_count + gain_count, int(union.sum())),
    }


def cohort_transition_error_rows(
    key: str,
    timestep_prev: int,
    timestep_curr: int,
    supervised_prev: np.ndarray,
    supervised_curr: np.ndarray,
    exists_prev: np.ndarray,
    exists_curr: np.ndarray,
    error_prev: np.ndarray,
    error_curr: np.ndarray,
) -> list[dict[str, Any]]:
    eligible_prev = exists_prev
    eligible_curr = exists_curr
    continuing = supervised_prev & supervised_curr & eligible_prev & eligible_curr
    dropped = supervised_prev & ~supervised_curr & eligible_prev
    entered = ~supervised_prev & supervised_curr & eligible_curr

    cohorts = {
        "continuing_supervised_error_before": (continuing, error_prev),
        "dropped_supervised_error_before": (dropped, error_prev),
        "entered_supervised_error_after": (entered, error_curr),
    }
    rows: list[dict[str, Any]] = []
    for cohort, (mask, values) in cohorts.items():
        selected = values[mask]
        rows.append(
            {
                "key": key,
                "previous_timestep": timestep_prev,
                "current_timestep": timestep_curr,
                "cohort": cohort,
                "count": int(selected.size),
                "mean_l2": float(selected.mean()) if selected.size else float("nan"),
                "median_l2": float(np.median(selected)) if selected.size else float("nan"),
            }
        )
    return rows


def action_descriptors(batch: Mapping[str, Any], valid_points: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}

    joints_value = batch.get("joint_positions")
    if joints_value is not None:
        joints = as_timeseries(joints_value, "joint_positions")
        if joints.shape[0] >= 2:
            steps = np.linalg.norm(np.diff(joints, axis=0), axis=-1)
            result.update(
                joint_step_mean=float(steps.mean()),
                joint_step_max=float(steps.max()),
                joint_path_length=float(steps.sum()),
                joint_endpoint_displacement=float(np.linalg.norm(joints[-1] - joints[0])),
            )

    gripper_value = batch.get("gripper_positions")
    if gripper_value is None:
        # Official DROID batches often expose left/right open values instead.
        pieces = []
        for name in ("left_gripper_open", "right_gripper_open"):
            if batch.get(name) is not None:
                pieces.append(as_timeseries(batch[name], name))
        if pieces:
            gripper = np.concatenate(pieces, axis=-1)
        else:
            gripper = None
    else:
        gripper = as_timeseries(gripper_value, "gripper_positions")

    if gripper is not None and gripper.shape[0] >= 2:
        steps = np.linalg.norm(np.diff(gripper, axis=0), axis=-1)
        result.update(
            gripper_step_mean=float(steps.mean()),
            gripper_step_max=float(steps.max()),
            gripper_path_length=float(steps.sum()),
            gripper_endpoint_displacement=float(np.linalg.norm(gripper[-1] - gripper[0])),
        )

    robot_value = batch.get("robot_flows")
    robot_exists_value = batch.get("robot_exists")
    if robot_value is not None:
        robot = as_xyz(robot_value, "robot_flows")
        if robot_exists_value is not None:
            robot_exists = as_bool_mask(robot_exists_value, "robot_exists")
        else:
            robot_exists = np.ones(robot.shape[:2], dtype=bool)
        if robot.shape[0] >= 2:
            step_vec = np.diff(robot, axis=0)
            step_mag = np.linalg.norm(step_vec, axis=-1)
            pair_exists = robot_exists[:-1] & robot_exists[1:]
            selected = step_mag[pair_exists]
            if selected.size:
                result.update(
                    robot_flow_step_mean=float(selected.mean()),
                    robot_flow_step_max=float(selected.max()),
                    robot_flow_step_p95=float(np.quantile(selected, 0.95)),
                )

            # Mean per-point path length among points existing at least once.
            point_path = np.where(pair_exists, step_mag, 0.0).sum(axis=0)
            point_valid = pair_exists.any(axis=0)
            if point_valid.any():
                result["robot_flow_point_path_mean"] = float(point_path[point_valid].mean())
                result["robot_flow_point_path_max"] = float(point_path[point_valid].max())

    # Number of actual scene points is useful for checking action/scene padding.
    result["scene_point_count"] = float(valid_points.sum())
    return result


def mask_churn(mask: np.ndarray) -> float:
    if mask.shape[0] < 2:
        return float("nan")
    values = []
    for i in range(1, mask.shape[0]):
        union = mask[i - 1] | mask[i]
        changed = mask[i - 1] ^ mask[i]
        values.append(safe_div(int(changed.sum()), int(union.sum())))
    return nanmean(values)


# ---------------------------------------------------------------------------
# Correlations and grouped summaries
# ---------------------------------------------------------------------------


def correlation_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    pairs = [
        ("visibility_drop_ratio", "moved_l2"),
        ("visibility_drop_ratio", "moved_error_variation"),
        ("supervised_drop_ratio", "moved_l2"),
        ("supervised_mask_churn_rate", "moved_error_variation"),
        ("supervised_mask_churn_rate", "continuous_moved_repeated_oscillation_rate"),
        ("continuous_supervised_ratio", "moved_velocity_error"),
        ("continuous_supervised_ratio", "continuous_moved_velocity_error"),
        ("robot_flow_step_mean", "visibility_drop_ratio"),
        ("robot_flow_step_mean", "supervised_drop_ratio"),
        ("robot_flow_step_mean", "supervised_mask_churn_rate"),
        ("robot_flow_step_mean", "moved_l2"),
        ("robot_flow_step_mean", "moved_error_variation"),
        ("joint_path_length", "visibility_drop_ratio"),
        ("joint_path_length", "moved_l2"),
        ("gripper_path_length", "visibility_drop_ratio"),
    ]

    rows: list[dict[str, Any]] = []
    for x_name, y_name in pairs:
        if x_name not in df.columns or y_name not in df.columns:
            continue
        pair = df[[x_name, y_name]].replace([np.inf, -np.inf], np.nan).dropna()
        row: dict[str, Any] = {
            "x": x_name,
            "y": y_name,
            "n": int(len(pair)),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "status": "ok",
        }
        if len(pair) < 3:
            row["status"] = "insufficient_n"
        elif pair[x_name].nunique() < 2 or pair[y_name].nunique() < 2:
            row["status"] = "constant_input"
        else:
            sr = spearmanr(pair[x_name].to_numpy(), pair[y_name].to_numpy())
            pr = pearsonr(pair[x_name].to_numpy(), pair[y_name].to_numpy())
            row.update(
                spearman_rho=float(sr.statistic),
                spearman_p=float(sr.pvalue),
                pearson_r=float(pr.statistic),
                pearson_p=float(pr.pvalue),
            )
        rows.append(row)
    return rows


def action_bins(df: pd.DataFrame) -> pd.DataFrame:
    magnitude = "robot_flow_step_mean"
    if magnitude not in df.columns or df[magnitude].notna().sum() < 3:
        return pd.DataFrame(
            columns=[
                "action_descriptor",
                "action_bin",
                "sequence_count",
                "descriptor_min",
                "descriptor_max",
            ]
        )

    valid = df[df[magnitude].notna()].copy()
    try:
        codes = pd.qcut(valid[magnitude], q=3, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    unique_codes = sorted(int(x) for x in pd.unique(codes.dropna()))
    label_map = {
        code: ("low" if i == 0 else "high" if i == len(unique_codes) - 1 else "medium")
        for i, code in enumerate(unique_codes)
    }
    valid["action_bin"] = codes.map(label_map)

    metrics = [
        "visibility_drop_ratio",
        "supervised_drop_ratio",
        "supervised_mask_churn_rate",
        "continuous_supervised_ratio",
        "moved_l2",
        "moved_velocity_error",
        "moved_acceleration_error",
        "moved_error_variation",
        "continuous_moved_repeated_oscillation_rate",
    ]
    rows: list[dict[str, Any]] = []
    for label, group in valid.groupby("action_bin", observed=True):
        row: dict[str, Any] = {
            "action_descriptor": magnitude,
            "action_bin": label,
            "sequence_count": int(len(group)),
            "descriptor_min": float(group[magnitude].min()),
            "descriptor_max": float(group[magnitude].max()),
            "descriptor_mean": float(group[magnitude].mean()),
            "descriptor_median": float(group[magnitude].median()),
        }
        for metric in metrics:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
                row[f"{metric}_median"] = float(values.median()) if len(values) else float("nan")
        rows.append(row)
    order = {"low": 0, "medium": 1, "high": 2}
    return pd.DataFrame(rows).sort_values(
        "action_bin", key=lambda s: s.map(order), ignore_index=True
    )


# ---------------------------------------------------------------------------
# PointWorld dataloader construction
# ---------------------------------------------------------------------------


def build_official_dataloader(cli: argparse.Namespace):
    try:
        from arguments import parse_args as parse_pointworld_args
        from dataset_components.dataloader import build_dataloader
    except Exception as exc:
        raise RuntimeError(
            "Could not import PointWorld modules. Run this script from the official "
            "PointWorld repository root with PYTHONPATH=$(pwd)."
        ) from exc

    # Use the repository's complete default argument namespace, then override only
    # data/determinism fields needed for this read-only diagnostic.
    pw_args = parse_pointworld_args(skip_command_line=True)
    pw_args.seed = cli.seed
    pw_args.domains = ["droid"]
    pw_args.data_dirs = [str(cli.data_dir)]
    pw_args.batch_size = 1
    pw_args.num_workers = cli.num_workers
    pw_args.eval_num_workers = cli.num_workers
    pw_args.deterministic_data = True
    if hasattr(pw_args, "deterministic_train"):
        pw_args.deterministic_train = False

    # Match standard DROID evaluation camera configuration unless the installed
    # repository defaults already define a stricter fixed choice.
    if hasattr(pw_args, "eval_min_num_cameras"):
        pw_args.eval_min_num_cameras = 2
    if hasattr(pw_args, "eval_max_num_cameras"):
        pw_args.eval_max_num_cameras = 2

    try:
        dataloader, dataset_info = build_dataloader(
            pw_args,
            mode="test",
            override_splits=cli.split,
        )
    except TypeError:
        # Compatibility with versions whose split argument is positional/absent.
        dataloader, dataset_info = build_dataloader(pw_args, mode=cli.split)
    return pw_args, dataloader, dataset_info


# ---------------------------------------------------------------------------
# Main sequence analysis
# ---------------------------------------------------------------------------


def choose_future_indices(context: np.ndarray) -> np.ndarray:
    if context.ndim != 2:
        raise ValueError(context.shape)
    # A timestep is future if at least one real point is not context there.
    future = np.flatnonzero((~context).any(axis=1))
    if future.size == 0:
        # Explicit fallback for expected t0 context + ten future timesteps.
        future = np.arange(1, context.shape[0])
    return future


def analyze_sequence(
    key: str,
    batch: Mapping[str, Any],
    raw: Mapping[str, Any],
    global_cohorts: dict[str, GlobalCohortAccumulator],
    per_timestep_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    transition_error_rows: list[dict[str, Any]],
    cli: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    visibility = as_bool_mask(batch["scene_visibility"], "scene_visibility")
    depth_valid = as_bool_mask(batch["scene_depth_valid_mask"], "scene_depth_valid_mask")
    supervised = as_bool_mask(batch["scene_supervised_mask"], "scene_supervised_mask")
    exists = as_bool_mask(batch["scene_exists"], "scene_exists")
    context = as_bool_mask(batch["scene_context_mask"], "scene_context_mask")
    moved = as_bool_mask(batch["scene_moved_mask"], "scene_moved_mask")
    static = as_bool_mask(batch["scene_static_mask"], "scene_static_mask")
    gt_batch = as_xyz(batch["gt_scene_flows"], "batch gt_scene_flows")

    shapes = {
        "visibility": visibility.shape,
        "depth_valid": depth_valid.shape,
        "supervised": supervised.shape,
        "exists": exists.shape,
        "context": context.shape,
        "moved": moved.shape,
        "static": static.shape,
        "gt_batch": gt_batch.shape[:2],
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"{key}: dataloader mask/GT shape mismatch: {shapes}")
    if visibility.shape[0] != EXPECTED_TIMESTEPS:
        raise ValueError(
            f"{key}: expected {EXPECTED_TIMESTEPS} timesteps, got {visibility.shape[0]}"
        )

    pred = raw["pred"]
    gt = raw["gt"]
    if pred.shape != gt.shape:
        raise ValueError(f"{key}: raw pred/GT mismatch {pred.shape} vs {gt.shape}")
    if pred.shape[:2] != visibility.shape:
        raise ValueError(
            f"{key}: raw point/timestep shape {pred.shape[:2]} != dataloader {visibility.shape}"
        )

    gt_max_abs_diff = float(np.max(np.abs(gt - gt_batch)))
    gt_matches = gt_max_abs_diff <= cli.tensor_tolerance

    supervised_formula = visibility & depth_valid
    supervised_formula_mismatch = int(np.count_nonzero(supervised ^ supervised_formula))
    official_valid = (~context) & exists & supervised
    error = np.linalg.norm(pred - gt, axis=-1)
    recomputed_official_l2 = masked_l2(error, official_valid)

    raw_valid_matches: bool | None = None
    raw_valid_mismatch = None
    if raw["valid"] is not None:
        if raw["valid"].shape != official_valid.shape:
            raise ValueError(
                f"{key}: raw valid shape {raw['valid'].shape} != official {official_valid.shape}"
            )
        raw_valid_mismatch = int(np.count_nonzero(raw["valid"] ^ official_valid))
        raw_valid_matches = raw_valid_mismatch == 0

    raw_moved_matches: bool | None = None
    raw_static_matches: bool | None = None
    if raw["moved"] is not None:
        raw_moved_matches = bool(np.array_equal(raw["moved"], moved))
    if raw["static"] is not None:
        raw_static_matches = bool(np.array_equal(raw["static"], static))

    reference_l2 = raw["official_l2"]
    l2_abs_diff = (
        abs(recomputed_official_l2 - reference_l2)
        if reference_l2 is not None and math.isfinite(recomputed_official_l2)
        else None
    )
    l2_reproduced = l2_abs_diff is not None and l2_abs_diff <= cli.l2_tolerance

    future_idx = choose_future_indices(context)
    if future_idx.tolist() != list(range(1, EXPECTED_TIMESTEPS)):
        raise ValueError(f"{key}: unexpected future timesteps {future_idx.tolist()}")

    f_visibility = visibility[future_idx]
    f_depth = depth_valid[future_idx]
    f_supervised = supervised[future_idx]
    f_exists = exists[future_idx]
    f_context = context[future_idx]
    f_moved = moved[future_idx]
    f_static = static[future_idx]
    f_official = official_valid[future_idx]
    f_pred = pred[future_idx]
    f_gt = gt[future_idx]
    f_error = error[future_idx]

    # A point slot is considered real if it exists at least once in the sequence.
    real_point = exists.any(axis=0)
    action = action_descriptors(batch, real_point)

    # Timestep counts and weighted error sums.
    for local_i, t in enumerate(future_idx):
        masks = {
            "scene_exists": f_exists[local_i],
            "scene_visibility": f_visibility[local_i] & f_exists[local_i],
            "scene_depth_valid_mask": f_depth[local_i] & f_exists[local_i],
            "scene_supervised_mask": f_supervised[local_i] & f_exists[local_i],
            "official_valid": f_official[local_i],
            "moved_valid": f_official[local_i] & f_moved[local_i],
            "static_valid": f_official[local_i] & f_static[local_i],
        }
        row: dict[str, Any] = {"key": key, "timestep": int(t), "is_future": True}
        for name, mask in masks.items():
            row[f"{name}_count"] = int(mask.sum())
            row[f"{name}_ratio_of_real_points"] = safe_div(int(mask.sum()), int(real_point.sum()))
        row["official_l2"] = masked_l2(f_error[local_i], masks["official_valid"])
        row["moved_l2"] = masked_l2(f_error[local_i], masks["moved_valid"])
        row["static_l2"] = masked_l2(f_error[local_i], masks["static_valid"])
        per_timestep_rows.append(row)

    # Transition accounting over future timesteps only.
    for local_i in range(1, len(future_idx)):
        prev_t = int(future_idx[local_i - 1])
        curr_t = int(future_idx[local_i])
        for name, mask in (
            ("visibility", f_visibility & f_exists),
            ("depth_valid", f_depth & f_exists),
            ("supervised", f_supervised & f_exists),
            ("official_valid", f_official),
        ):
            transition_rows.append(
                transition_row(key, name, prev_t, curr_t, mask[local_i - 1], mask[local_i])
            )
        transition_error_rows.extend(
            cohort_transition_error_rows(
                key,
                prev_t,
                curr_t,
                f_supervised[local_i - 1],
                f_supervised[local_i],
                f_exists[local_i - 1],
                f_exists[local_i],
                f_error[local_i - 1],
                f_error[local_i],
            )
        )

    continuous_visible = (f_visibility & f_exists).all(axis=0)
    continuous_depth = (f_depth & f_exists).all(axis=0)
    continuous_supervised = (f_supervised & f_exists).all(axis=0)
    continuous_official = f_official.all(axis=0)
    continuous_moved = continuous_official & f_moved.all(axis=0)
    continuous_static = continuous_official & f_static.all(axis=0)

    cohort_masks = {
        "continuous_visible": np.broadcast_to(continuous_visible, f_supervised.shape),
        "continuous_depth_valid": np.broadcast_to(continuous_depth, f_supervised.shape),
        "continuous_supervised": np.broadcast_to(continuous_supervised, f_supervised.shape),
        "continuous_official_valid": np.broadcast_to(continuous_official, f_supervised.shape),
        "continuous_moved": np.broadcast_to(continuous_moved, f_supervised.shape),
        "continuous_static": np.broadcast_to(continuous_static, f_supervised.shape),
    }
    per_sequence_cohorts: dict[str, dict[str, float]] = {}
    for cohort_name, cohort_mask in cohort_masks.items():
        per_sequence_cohorts[cohort_name] = global_cohorts[cohort_name].update(
            f_pred, f_gt, cohort_mask
        )

    moved_dynamic = dynamic_temporal_metrics(f_pred, f_gt, f_official, f_moved)
    static_dynamic = dynamic_temporal_metrics(f_pred, f_gt, f_official, f_static)

    vis_counts = (f_visibility & f_exists).sum(axis=1)
    supervised_counts = (f_supervised & f_exists).sum(axis=1)
    initial_vis = int(vis_counts[0])
    final_vis = int(vis_counts[-1])
    initial_sup = int(supervised_counts[0])
    final_sup = int(supervised_counts[-1])

    seq: dict[str, Any] = {
        "key": key,
        "raw_path": str(raw.get("path", "")),
        "timesteps": int(visibility.shape[0]),
        "future_timesteps": int(len(future_idx)),
        "scene_point_count": int(real_point.sum()),
        "initial_visible_count": initial_vis,
        "final_visible_count": final_vis,
        "visibility_drop_count": initial_vis - final_vis,
        "visibility_drop_ratio": safe_div(initial_vis - final_vis, initial_vis),
        "minimum_visible_ratio": safe_div(int(vis_counts.min()), initial_vis),
        "initial_supervised_count": initial_sup,
        "final_supervised_count": final_sup,
        "supervised_drop_count": initial_sup - final_sup,
        "supervised_drop_ratio": safe_div(initial_sup - final_sup, initial_sup),
        "visibility_mask_churn_rate": mask_churn(f_visibility & f_exists),
        "supervised_mask_churn_rate": mask_churn(f_supervised & f_exists),
        "continuous_visible_count": int(continuous_visible.sum()),
        "continuous_depth_valid_count": int(continuous_depth.sum()),
        "continuous_supervised_count": int(continuous_supervised.sum()),
        "continuous_supervised_ratio": safe_div(int(continuous_supervised.sum()), initial_sup),
        "continuous_official_valid_count": int(continuous_official.sum()),
        "continuous_moved_count": int(continuous_moved.sum()),
        "continuous_static_count": int(continuous_static.sum()),
        "official_l2_recomputed": recomputed_official_l2,
        "official_l2_reference": reference_l2,
        "official_l2_abs_diff": l2_abs_diff,
        "official_l2_reproduced": l2_reproduced,
        "raw_valid_mask_available": raw["valid"] is not None,
        "raw_valid_mask_matches": raw_valid_matches,
        "raw_valid_mask_mismatch_count": raw_valid_mismatch,
        "raw_moved_mask_matches": raw_moved_matches,
        "raw_static_mask_matches": raw_static_matches,
        "raw_gt_matches_dataloader": gt_matches,
        "raw_gt_max_abs_diff": gt_max_abs_diff,
        "supervised_formula_mismatch_count": supervised_formula_mismatch,
        "moved_static_overlap_count": int(np.count_nonzero(moved & static)),
        "valid_unclassified_count": int(np.count_nonzero(official_valid & ~(moved | static))),
        **action,
    }

    prefix_map = {"moved": moved_dynamic, "static": static_dynamic}
    for prefix, metrics in prefix_map.items():
        for name, value in metrics.items():
            seq[f"{prefix}_{name}"] = value

    for cohort_name, metrics in per_sequence_cohorts.items():
        for name, value in metrics.items():
            seq[f"{cohort_name}_{name}"] = value

    audits = {
        "key": key,
        "supervised_formula_mismatch_count": supervised_formula_mismatch,
        "raw_gt_max_abs_diff": gt_max_abs_diff,
        "raw_valid_mask_mismatch_count": raw_valid_mismatch,
        "official_l2_abs_diff": l2_abs_diff,
    }
    return seq, audits


def aggregate_timestep(per_timestep: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    count_cols = [c for c in per_timestep.columns if c.endswith("_count")]
    for timestep, group in per_timestep.groupby("timestep", sort=True):
        row: dict[str, Any] = {
            "timestep": int(timestep),
            "sequence_count": int(group["key"].nunique()),
        }
        for col in count_cols:
            row[col] = int(group[col].sum())
        # Sequence-level means are explicitly named as such to avoid pretending
        # they are pooled point-wise metrics.
        for col in ("official_l2", "moved_l2", "static_l2"):
            values = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{col}_sequence_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{col}_sequence_median"] = float(values.median()) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_transition_error(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if df.empty:
        return result
    for cohort, group in df.groupby("cohort"):
        valid = group[group["count"] > 0].copy()
        total_count = int(valid["count"].sum())
        weighted = (
            float((valid["mean_l2"] * valid["count"]).sum() / total_count)
            if total_count > 0
            else float("nan")
        )
        result[str(cohort)] = {
            "point_observation_count": total_count,
            "pooled_mean_l2": weighted,
            "transition_row_count": int(len(group)),
        }
    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test() -> None:
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(10, 8, 3))
    gt = pred.copy()
    gt[:, 0, 0] += np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    mask = np.ones((10, 8), dtype=bool)

    acc = GlobalCohortAccumulator()
    metrics = acc.update(pred, gt, mask)
    assert metrics["point_count"] == 8
    assert metrics["repeated_oscillation_rate"] > 0

    prev = np.array([1, 1, 0, 0], dtype=bool)
    curr = np.array([1, 0, 1, 0], dtype=bool)
    row = transition_row("k", "m", 1, 2, prev, curr)
    assert row["retained_count"] == 1
    assert row["dropped_count"] == 1
    assert row["gained_count"] == 1
    assert math.isclose(row["jaccard"], 1 / 3)

    dyn = dynamic_temporal_metrics(pred, gt, mask, mask)
    assert dyn["point_observation_count"] == 80
    print("SELF TEST PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    cli = parse_cli()
    if cli.self_test:
        self_test()
        return 0

    repo_root = Path.cwd().resolve()
    raw_dir = cli.raw_dir.expanduser().resolve()
    data_dir = cli.data_dir.expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        cli.output_dir.expanduser().resolve()
        if cli.output_dir is not None
        else repo_root / "diagnostics" / "results" / f"visibility_consistency_droid_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    git_before = official_git_diff()

    print(f"Repository: {repo_root}")
    print(f"DROID data: {data_dir}")
    print(f"Raw predictions: {raw_dir}")
    print(f"Output: {output_dir}")

    raw_index, duplicate_raw_keys, unresolved_raw = index_raw_files(raw_dir)
    official_l2_by_key = load_official_l2_by_key()
    if duplicate_raw_keys:
        raise RuntimeError(f"Duplicate raw keys: {duplicate_raw_keys}")
    if unresolved_raw:
        dump_json(output_dir / "unresolved_raw_files.json", unresolved_raw)
        raise RuntimeError(
            f"{len(unresolved_raw)} raw files lacked an embedded key; see unresolved_raw_files.json"
        )
    print(f"Indexed raw sequences: {len(raw_index)}")

    pw_args, dataloader, dataset_info = build_official_dataloader(cli)

    per_timestep_rows: list[dict[str, Any]] = []
    per_sequence_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    transition_error_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    duplicate_dataloader_keys: list[str] = []
    missing_raw_keys: list[str] = []
    matched_raw_keys: set[str] = set()

    global_cohorts = {
        name: GlobalCohortAccumulator()
        for name in (
            "continuous_visible",
            "continuous_depth_valid",
            "continuous_supervised",
            "continuous_official_valid",
            "continuous_moved",
            "continuous_static",
        )
    }

    for batch_index, batch in enumerate(dataloader):
        if cli.max_sequences is not None and len(seen_keys) >= cli.max_sequences:
            break
        if not isinstance(batch, Mapping):
            raise TypeError(f"Batch must be a mapping, got {type(batch)!r}")
        if "__key__" not in batch:
            raise KeyError(f"Batch {batch_index} has no __key__. Keys: {sorted(batch.keys())}")
        key = normalize_key(batch["__key__"])
        if key in seen_keys:
            duplicate_dataloader_keys.append(key)
            continue
        seen_keys.add(key)

        record = raw_index.get(key)
        if record is None:
            missing_raw_keys.append(key)
            print(f"[MISSING RAW] {key}")
            continue

        raw = extract_raw_arrays(record.path)
        raw["path"] = record.path

        if key not in official_l2_by_key:
            raise KeyError(
                f"Missing official L2 reference for key: {key}"
            )

        raw["official_l2"] = official_l2_by_key[key]
        if raw["key"] != key:
            raise RuntimeError(f"Raw key changed after load: index={key}, file={raw['key']}")

        seq, audit = analyze_sequence(
            key,
            batch,
            raw,
            global_cohorts,
            per_timestep_rows,
            transition_rows,
            transition_error_rows,
            cli,
        )
        per_sequence_rows.append(seq)
        audits.append(audit)
        matched_raw_keys.add(key)
        print(
            f"[{len(per_sequence_rows):02d}] {key} | "
            f"vis_drop={seq['visibility_drop_ratio']:.4f} | "
            f"moved_l2={seq['moved_l2']:.6f}"
        )

    unmatched_raw_keys = sorted(set(raw_index) - matched_raw_keys)
    processed = len(per_sequence_rows)

    per_sequence_df = pd.DataFrame(per_sequence_rows)
    per_timestep_df = pd.DataFrame(per_timestep_rows)
    transition_df = pd.DataFrame(transition_rows)
    transition_error_df = pd.DataFrame(transition_error_rows)

    per_sequence_df.to_csv(output_dir / "per_sequence.csv", index=False)
    aggregate_timestep(per_timestep_df).to_csv(output_dir / "per_timestep.csv", index=False)
    transition_df.to_csv(output_dir / "mask_transition.csv", index=False)
    transition_error_df.to_csv(output_dir / "transition_cohort_errors.csv", index=False)

    corr_df = pd.DataFrame(correlation_rows(per_sequence_df))
    corr_df.to_csv(output_dir / "correlations.csv", index=False)
    bins_df = action_bins(per_sequence_df)
    bins_df.to_csv(output_dir / "action_visibility_bins.csv", index=False)

    cohort_summary = {
        "definitions": {
            "future_timesteps": "t=1..10; t=0 context excluded",
            "continuous_visible": "scene_exists & scene_visibility true at every future timestep",
            "continuous_depth_valid": "scene_exists & scene_depth_valid_mask true at every future timestep",
            "continuous_supervised": "scene_exists & scene_supervised_mask true at every future timestep",
            "continuous_official_valid": "(~scene_context_mask) & scene_exists & scene_supervised_mask true at every future timestep",
            "continuous_moved": "continuous_official_valid and scene_moved_mask true at every future timestep",
            "continuous_static": "continuous_official_valid and scene_static_mask true at every future timestep",
            "repeated_oscillation": "at least 2 sign changes in consecutive non-zero pointwise L2-error increments",
            "strong_alternation": "sign changes in at least 50% of possible adjacent error-increment pairs",
        },
        "fixed_cohorts": {name: acc.summary() for name, acc in global_cohorts.items()},
        "transition_cohorts": aggregate_transition_error(transition_error_df),
    }
    dump_json(output_dir / "cohort_metrics.json", cohort_summary)

    git_after = official_git_diff()
    all_l2_refs = bool(processed) and per_sequence_df["official_l2_reference"].notna().all()
    all_l2_reproduced = bool(processed) and per_sequence_df["official_l2_reproduced"].all()
    all_gt_match = bool(processed) and per_sequence_df["raw_gt_matches_dataloader"].all()
    all_raw_valid_match = (
        bool(processed)
        and per_sequence_df["raw_valid_mask_available"].all()
        and per_sequence_df["raw_valid_mask_matches"].fillna(False).all()
    )
    supervised_formula_total_mismatch = int(
        per_sequence_df["supervised_formula_mismatch_count"].sum()
    ) if processed else 0

    expected_ok = processed == cli.expected_sequences
    missing_ok = not missing_raw_keys
    duplicate_ok = not duplicate_dataloader_keys and not duplicate_raw_keys
    official_files_clean = not git_after["official_files_modified"]
    l2_ok = all_l2_reproduced or (
        cli.allow_missing_raw_reference_l2 and not all_l2_refs
    )

    success = all(
        [
            expected_ok,
            missing_ok,
            duplicate_ok,
            len(unmatched_raw_keys) == max(len(raw_index) - cli.expected_sequences, 0),
            supervised_formula_total_mismatch == 0,
            all_gt_match,
            all_raw_valid_match,
            l2_ok,
            official_files_clean,
        ]
    )

    summary = {
        "scope": "DROID actual clips: visibility/mask consistency and action-magnitude relation; no model inference",
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repository": str(repo_root),
        "data_dir": str(data_dir),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "processed_sequences": processed,
        "expected_sequences": cli.expected_sequences,
        "dataloader_unique_keys_seen": len(seen_keys),
        "raw_file_count": len(raw_index),
        "matched_prediction_raw_files": len(matched_raw_keys),
        "missing_keys": sorted(missing_raw_keys),
        "duplicate_dataloader_keys": sorted(set(duplicate_dataloader_keys)),
        "duplicate_raw_keys": duplicate_raw_keys,
        "unmatched_raw_keys": unmatched_raw_keys,
        "expected_timesteps": EXPECTED_TIMESTEPS,
        "supervised_formula": "scene_visibility & scene_depth_valid_mask",
        "official_valid_formula": "(~scene_context_mask) & scene_exists & scene_supervised_mask",
        "supervised_formula_total_mismatch": supervised_formula_total_mismatch,
        "raw_gt_matches_dataloader_for_all": all_gt_match,
        "raw_valid_mask_matches_for_all": all_raw_valid_match,
        "official_l2_reference_available_for_all": all_l2_refs,
        "official_l2_reproduced_for_all": all_l2_reproduced,
        "official_files_modified": git_after["official_files_modified"],
        "official_modified_tracked_files": git_after["official_modified_tracked_files"],
        "success": success,
        "notes": [
            "Spearman correlations are relationship analyses, not causal evidence.",
            "Action bins use robot_flow_step_mean tertiles and are not semantic action labels.",
            "Official PointWorld metrics and added temporal diagnostics remain separate.",
        ],
    }
    dump_json(output_dir / "summary.json", summary)
    dump_json(output_dir / "audit_per_sequence.json", audits)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "argv": sys.argv,
        "pointworld_dataset_info": str(dataset_info),
        "pointworld_overrides": {
            "seed": cli.seed,
            "domains": ["droid"],
            "data_dirs": [str(data_dir)],
            "batch_size": 1,
            "num_workers": cli.num_workers,
            "deterministic_data": True,
            "split": cli.split,
        },
        "git_before": git_before,
        "git_after": git_after,
        "output_files": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
    }
    dump_json(output_dir / "manifest.json", manifest)

    print("\n=== AUDIT ===")
    print(f"Processed sequences: {processed}")
    print(f"Matched prediction raw files: {len(matched_raw_keys)}")
    print(f"Missing keys: {len(missing_raw_keys)}")
    print(f"Duplicate keys: {len(set(duplicate_dataloader_keys)) + len(duplicate_raw_keys)}")
    print(f"Supervised formula mismatches: {supervised_formula_total_mismatch}")
    print(f"Raw GT matches dataloader for all: {all_gt_match}")
    print(f"Raw valid mask matches for all: {all_raw_valid_match}")
    print(f"Official L2 reference available for all: {all_l2_refs}")
    print(f"Official L2 reproduced for all: {all_l2_reproduced}")
    print(f"Official files modified: {git_after['official_files_modified']}")
    print(f"SUCCESS: {success}")
    print(f"Summary: {output_dir / 'summary.json'}")

    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
