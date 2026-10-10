#!/usr/bin/env python3

import csv
import hashlib
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

HPL_ROOT = ROOT / "diagnostics/third_party/HPLFlowNet"
HPL_EVAL = HPL_ROOT / "evaluation_utils.py"

META = ROOT / (
    "diagnostics/results/"
    "real_action_state_link_official_l2.csv"
)

PRED_DIR = ROOT / (
    "diagnostics/results/"
    "real_action_state_link_predictions"
)

OUT = ROOT / (
    "diagnostics/results/"
    "real_action_state_link_hpl_metrics.csv"
)


# ------------------------------------------------------------------
# Compatibility ONLY.
#
# Official 2019 HPLFlowNet code uses deprecated np.float.
# Modern NumPy removed that alias.
#
# This restores the old alias only; no metric formula, threshold,
# mask, or arithmetic is changed.
# ------------------------------------------------------------------
if not hasattr(np, "float"):
    np.float = float


if not HPL_EVAL.exists():
    raise FileNotFoundError(
        f"Official HPLFlowNet evaluation_utils.py not found: {HPL_EVAL}"
    )

sys.path.insert(0, str(HPL_ROOT))

# DIRECT import from official HPLFlowNet repository.
from evaluation_utils import evaluate_3d  # noqa: E402


def git_head(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(key):
    return key.replace("/", "_").replace(":", "_")


def mask2(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x).astype(bool)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    if x.ndim != 2:
        raise RuntimeError(
            f"Unexpected mask shape: {x.shape}"
        )

    return x


print("HPLFlowNet commit =", git_head(HPL_ROOT))
print("evaluation_utils sha256 =", sha256(HPL_EVAL))


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------
with META.open() as f:
    rows = list(csv.DictReader(f))

if len(rows) != 90:
    raise RuntimeError(
        f"Expected 90 metadata rows, got {len(rows)}"
    )

unique_keys = sorted(set(r["clip"] for r in rows))

print("metadata rows =", len(rows))
print("unique prediction samples =", len(unique_keys))


# ------------------------------------------------------------------
# Evaluate each unique PointWorld prediction.
#
# IMPORTANT:
# HPLFlowNet metric is computed independently at EACH PointWorld
# timestep. We do not invent a new temporal aggregation metric.
# ------------------------------------------------------------------
sample_results = {}

for n, key in enumerate(unique_keys, 1):

    pt_path = PRED_DIR / f"{safe_name(key)}.pt"

    if not pt_path.exists():
        raise FileNotFoundError(pt_path)

    d = torch.load(
        pt_path,
        map_location="cpu",
        weights_only=False,
    )

    pred = d["pred_scene_flows"].detach().cpu().numpy()
    gt = d["gt_scene_flows"].detach().cpu().numpy()

    if pred.shape != gt.shape:
        raise RuntimeError(
            f"{key}: pred={pred.shape}, gt={gt.shape}"
        )

    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise RuntimeError(
            f"{key}: unexpected flow shape={pred.shape}"
        )

    exists = mask2(d["scene_exists"])
    context = mask2(d["scene_context_mask"])
    supervised = mask2(d["scene_supervised_mask"])

    # Exact PointWorld official valid supervision definition.
    valid = (~context) & exists & supervised

    T = pred.shape[0]

    res = {
        "clip": key,
        "num_timesteps": T,
    }

    for t in range(T):

        m = valid[t]

        res[f"n_valid_t{t}"] = int(m.sum())

        if not m.any():
            res[f"epe3d_t{t}_m"] = float("nan")
            res[f"acc3ds_t{t}"] = float("nan")
            res[f"acc3dr_t{t}"] = float("nan")
            res[f"outliers3d_t{t}"] = float("nan")
            continue

        sf_pred = pred[t, m]
        sf_gt = gt[t, m]

        # ----------------------------------------------------------
        # OFFICIAL HPLFlowNet evaluate_3d().
        # No local reimplementation.
        # ----------------------------------------------------------
        (
            epe3d,
            acc3ds,
            acc3dr,
            outliers3d,
        ) = evaluate_3d(
            sf_pred,
            sf_gt,
        )

        res[f"epe3d_t{t}_m"] = float(epe3d)
        res[f"acc3ds_t{t}"] = float(acc3ds)
        res[f"acc3dr_t{t}"] = float(acc3dr)
        res[f"outliers3d_t{t}"] = float(outliers3d)

    sample_results[key] = res

    last = T - 1

    print(
        f"[{n:02d}/{len(unique_keys)}] {key} | "
        f"t{last}: "
        f"EPE3D={1000*res[f'epe3d_t{last}_m']:.3f} mm | "
        f"AccS={res[f'acc3ds_t{last}']:.4f} | "
        f"AccR={res[f'acc3dr_t{last}']:.4f} | "
        f"Outlier={res[f'outliers3d_t{last}']:.4f}"
    )


# ------------------------------------------------------------------
# Expand back to 90 candidate rows.
# ------------------------------------------------------------------
expanded = []

for meta in rows:
    row = dict(meta)
    row.update(sample_results[meta["clip"]])
    expanded.append(row)


with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(expanded[0].keys()),
    )
    writer.writeheader()
    writer.writerows(expanded)


# ------------------------------------------------------------------
# Paired comparison.
# This is descriptive statistics on the OFFICIAL metrics,
# not a new metric.
# ------------------------------------------------------------------
T = int(expanded[0]["num_timesteps"])
last = T - 1

print("\n" + "=" * 110)
print(
    "NATIVE ACTION EXTREME vs CONTROL "
    "- OFFICIAL HPLFlowNet METRICS"
)
print("Primary readout = final PointWorld timestep")
print("=" * 110)


outcomes = [
    (f"epe3d_t{last}_m", "higher"),
    (f"acc3ds_t{last}", "lower"),
    (f"acc3dr_t{last}", "lower"),
    (f"outliers3d_t{last}", "higher"),
]


for action_metric in [
    "speed_cv",
    "jerk_p95",
    "path_inefficiency",
]:

    print(f"\n[{action_metric}]")

    rr = [
        r for r in expanded
        if r["metric"] == action_metric
    ]

    pairs = defaultdict(dict)

    for r in rr:
        pairs[r["pair_id"]][r["group"]] = r

    for outcome, worse_direction in outcomes:

        ext = []
        ctl = []
        extreme_worse = []

        for pid, p in sorted(pairs.items()):

            if "extreme" not in p or "control" not in p:
                continue

            e = float(p["extreme"][outcome])
            c = float(p["control"][outcome])

            if not (math.isfinite(e) and math.isfinite(c)):
                continue

            ext.append(e)
            ctl.append(c)

            if worse_direction == "higher":
                extreme_worse.append(e > c)
            else:
                extreme_worse.append(e < c)

        ext = np.asarray(ext, dtype=np.float64)
        ctl = np.asarray(ctl, dtype=np.float64)

        if len(ext) == 0:
            continue

        if outcome.startswith("epe3d"):
            scale = 1000.0
            unit = "mm"
        else:
            scale = 1.0
            unit = ""

        print(
            f"{outcome:20s} | "
            f"N={len(ext):2d} | "
            f"EXT_med={np.median(ext)*scale:9.4f}{unit} | "
            f"CTL_med={np.median(ctl)*scale:9.4f}{unit} | "
            f"EXT_worse={100*np.mean(extreme_worse):6.2f}%"
        )


# ------------------------------------------------------------------
# Also print each timestep, but do NOT collapse them into a custom
# temporal score.
# ------------------------------------------------------------------
print("\n" + "=" * 110)
print("PER-TIMESTEP EXTREME-WORSE RATE")
print("=" * 110)

for action_metric in [
    "speed_cv",
    "jerk_p95",
    "path_inefficiency",
]:

    rr = [
        r for r in expanded
        if r["metric"] == action_metric
    ]

    pairs = defaultdict(dict)

    for r in rr:
        pairs[r["pair_id"]][r["group"]] = r

    print(f"\n[{action_metric}]")

    for t in range(T):

        vals = {}

        for metric_name, direction in [
            ("epe3d", "higher"),
            ("acc3ds", "lower"),
            ("acc3dr", "lower"),
            ("outliers3d", "higher"),
        ]:

            outcome = (
                f"{metric_name}_t{t}_m"
                if metric_name == "epe3d"
                else f"{metric_name}_t{t}"
            )

            flags = []

            for _, p in pairs.items():

                if "extreme" not in p or "control" not in p:
                    continue

                e = float(p["extreme"][outcome])
                c = float(p["control"][outcome])

                if not (
                    math.isfinite(e)
                    and math.isfinite(c)
                ):
                    continue

                flags.append(
                    e > c if direction == "higher"
                    else e < c
                )

            vals[metric_name] = (
                100.0 * np.mean(flags)
                if flags else float("nan")
            )

        print(
            f"t{t:02d} | "
            f"EPE worse={vals['epe3d']:6.2f}% | "
            f"AccS worse={vals['acc3ds']:6.2f}% | "
            f"AccR worse={vals['acc3dr']:6.2f}% | "
            f"Outlier worse={vals['outliers3d']:6.2f}%"
        )


print("\nsaved =", OUT.resolve())
print("SUCCESS = True")
