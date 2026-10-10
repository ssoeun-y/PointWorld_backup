#!/usr/bin/env python3

import csv
import math
import os
import random
import runpy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluation.tester import Tester


CANDIDATE_CSV = Path(
    "diagnostics/results/real_action_state_link_candidates.csv"
)

OUT_CSV = Path(
    "diagnostics/results/real_action_state_link_official_l2.csv"
)

PRED_DIR = Path(
    "diagnostics/results/real_action_state_link_predictions"
)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def reset_rng(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Candidate metadata
# ---------------------------------------------------------------------
candidate_rows = []

with CANDIDATE_CSV.open() as f:
    candidate_rows = list(csv.DictReader(f))

if len(candidate_rows) != 90:
    raise RuntimeError(
        f"Expected 90 candidate rows, got {len(candidate_rows)}"
    )

# Reconstruct 45 extreme-control pair IDs.
pair_count = defaultdict(int)
active_pair = {}

for r in candidate_rows:
    group_key = (r["task"], r["metric"])

    if r["group"] == "extreme":
        pair_count[group_key] += 1
        active_pair[group_key] = pair_count[group_key]
    elif r["group"] == "control":
        if group_key not in active_pair:
            raise RuntimeError(
                f"Control before extreme: {group_key}"
            )
    else:
        raise RuntimeError(r["group"])

    r["pair_id"] = (
        f"{r['task']}__{r['metric']}__"
        f"{active_pair[group_key]:02d}"
    )

TARGET_KEYS = {r["clip"] for r in candidate_rows}

print("candidate rows =", len(candidate_rows))
print("unique target samples =", len(TARGET_KEYS))


def mask3(batch, name):
    x = batch[name].bool()

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise RuntimeError(
            f"{name}: unexpected shape={tuple(x.shape)}"
        )

    return x


def masked_mean(x, mask):
    if not mask.any():
        return float("nan")

    return float(
        x[mask].float().mean().detach().cpu()
    )


def safe_name(key):
    return (
        key.replace("/", "_")
           .replace(":", "_")
    )


# One inference result per UNIQUE sample.
sample_results = {}


@torch.no_grad()
def capture_selected_samples(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):
    if apply_confidence_mask:
        raise RuntimeError(
            "BEHAVIOR simulation evaluation should not "
            "require confidence filtering."
        )

    self.model.eval()

    found = set()

    for batch_idx, batch in enumerate(dl):

        raw_keys = batch["__key__"]

        if isinstance(raw_keys, (list, tuple)):
            keys = [str(x) for x in raw_keys]
        else:
            keys = [str(raw_keys)]

        selected_indices = [
            i for i, key in enumerate(keys)
            if key in TARGET_KEYS
        ]

        # Important: no GPU/model forward for irrelevant samples.
        if not selected_indices:
            continue

        batch_gpu = {
            k: (
                v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

        # Native action / official loader.
        # No perturbation and no forced-camera patch.
        outputs = self.model(
            batch_gpu,
            training=False,
        )

        pred = outputs["scene_flows"]
        gt = batch_gpu["gt_scene_flows"]

        if pred.shape != gt.shape:
            raise RuntimeError(
                f"pred/GT mismatch: {pred.shape} vs {gt.shape}"
            )

        l2 = torch.linalg.vector_norm(
            pred - gt,
            dim=-1,
        )

        exists = mask3(batch_gpu, "scene_exists")
        context = mask3(
            batch_gpu,
            "scene_context_mask",
        )
        supervised = mask3(
            batch_gpu,
            "scene_supervised_mask",
        )

        moved = mask3(
            batch_gpu,
            "scene_moved_mask",
        )
        static = mask3(
            batch_gpu,
            "scene_static_mask",
        )

        # Exact official definition from pointworld/losses.py
        valid = (~context) & exists & supervised

        B, T, Ns = l2.shape

        for i in selected_indices:

            key = keys[i]

            if key in found:
                raise RuntimeError(
                    f"Duplicate test key encountered: {key}"
                )

            found.add(key)

            vi = valid[i]
            mi = moved[i]
            si = static[i]
            ei = l2[i]

            result = {
                "clip": key,

                # Official aggregate definitions.
                "l2_mm":
                    1000.0 * masked_mean(
                        ei, vi
                    ),

                "l2_moved_mm":
                    1000.0 * masked_mean(
                        ei, vi & mi
                    ),

                "l2_static_mm":
                    1000.0 * masked_mean(
                        ei, vi & si
                    ),

                "num_valid":
                    int(vi.sum().item()),

                "num_moved_valid":
                    int((vi & mi).sum().item()),

                "num_static_valid":
                    int((vi & si).sum().item()),
            }

            # Per-timestep diagnostic using SAME official masks.
            for t in range(T):

                result[f"l2_t{t}_mm"] = (
                    1000.0
                    * masked_mean(
                        ei[t],
                        vi[t],
                    )
                )

                result[f"l2_moved_t{t}_mm"] = (
                    1000.0
                    * masked_mean(
                        ei[t],
                        vi[t] & mi[t],
                    )
                )

                result[f"l2_static_t{t}_mm"] = (
                    1000.0
                    * masked_mean(
                        ei[t],
                        vi[t] & si[t],
                    )
                )

            sample_results[key] = result

            # ----------------------------------------------------------
            # Save exactly what we will need for later structural
            # consistency analysis, avoiding another model inference.
            # ----------------------------------------------------------
            save_dict = {
                "__key__": key,
                "pred_scene_flows":
                    pred[i].detach().cpu(),

                "gt_scene_flows":
                    gt[i].detach().cpu(),

                "scene_exists":
                    batch_gpu["scene_exists"][i]
                    .detach().cpu(),

                "scene_context_mask":
                    batch_gpu["scene_context_mask"][i]
                    .detach().cpu(),

                "scene_supervised_mask":
                    batch_gpu["scene_supervised_mask"][i]
                    .detach().cpu(),

                "scene_moved_mask":
                    batch_gpu["scene_moved_mask"][i]
                    .detach().cpu(),

                "scene_static_mask":
                    batch_gpu["scene_static_mask"][i]
                    .detach().cpu(),
            }

            # Keep processed scene trajectory/input if available.
            for field in [
                "scene_flows",
                "robot_flows",
                "__shift_amount__",
            ]:
                if (
                    field in batch_gpu
                    and isinstance(
                        batch_gpu[field],
                        torch.Tensor,
                    )
                ):
                    save_dict[field] = (
                        batch_gpu[field][i]
                        .detach().cpu()
                    )

            torch.save(
                save_dict,
                PRED_DIR / f"{safe_name(key)}.pt",
            )

            print(
                f"[{len(found):02d}/{len(TARGET_KEYS)}] "
                f"{key} | "
                f"L2={result['l2_mm']:.3f} mm | "
                f"moved={result['l2_moved_mm']:.3f} mm"
            )

        if len(found) == len(TARGET_KEYS):
            print("All selected samples captured.")
            break

    missing = TARGET_KEYS - found

    if missing:
        print("\nMISSING KEYS:")
        for key in sorted(missing):
            print(" ", key)

        raise RuntimeError(
            f"Missing {len(missing)} / "
            f"{len(TARGET_KEYS)} selected samples"
        )

    # --------------------------------------------------------------
    # Expand predictions back to 90 candidate rows.
    # Same sample may participate in >1 action-metric comparison.
    # --------------------------------------------------------------
    expanded = []

    for meta in candidate_rows:
        result = sample_results[meta["clip"]]

        row = dict(meta)
        row.update(result)
        expanded.append(row)

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(expanded[0].keys()),
        )
        writer.writeheader()
        writer.writerows(expanded)

    # --------------------------------------------------------------
    # Paired extreme vs control summary.
    # --------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PAIRED ACTION -> STATE L2 SUMMARY")
    print("=" * 100)

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

        for outcome in [
            "l2_mm",
            "l2_moved_mm",
            "l2_static_mm",
            "l2_t10_mm",
            "l2_moved_t10_mm",
        ]:

            ext = []
            ctl = []
            delta = []

            for pair_id, p in pairs.items():

                if (
                    "extreme" not in p
                    or "control" not in p
                ):
                    continue

                e = float(p["extreme"][outcome])
                c = float(p["control"][outcome])

                if not (
                    math.isfinite(e)
                    and math.isfinite(c)
                ):
                    continue

                ext.append(e)
                ctl.append(c)
                delta.append(e - c)

            ext = np.asarray(ext, dtype=np.float64)
            ctl = np.asarray(ctl, dtype=np.float64)
            delta = np.asarray(
                delta,
                dtype=np.float64,
            )

            if len(delta) == 0:
                continue

            print(
                f"{action_metric:18s} | "
                f"{outcome:20s} | "
                f"N={len(delta):2d} | "
                f"EXT_med={np.median(ext):8.3f} | "
                f"CTL_med={np.median(ctl):8.3f} | "
                f"delta_med={np.median(delta):+8.3f} | "
                f"EXT>CTL={np.mean(delta > 0)*100:6.2f}%"
            )

    print("\nsaved CSV =", OUT_CSV.resolve())
    print("prediction dir =", PRED_DIR.resolve())
    print("SUCCESS = True")

    return {
        f"diagnostic/{split}/real_action_state_link":
            1.0
    }


# Diagnostic hook only.
Tester._accumulate_metrics = capture_selected_samples

reset_rng()

# Reuse official eval.py model/loader/checkpoint path.
runpy.run_path(
    "eval.py",
    run_name="__main__",
)
