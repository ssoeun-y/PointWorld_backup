#!/usr/bin/env python3

import csv
import json
import runpy
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from evaluation.tester import Tester


def mask3(batch, name):
    x = batch[name].bool()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    assert x.ndim == 3, (name, x.shape)
    return x


def stats(x):
    x = x.detach().float().cpu().reshape(-1)
    if x.numel() == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p95": float(torch.quantile(x, 0.95)),
        "p99": float(torch.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def rate(a, b):
    return float(a / b) if b else float("nan")


@torch.no_grad()
def full_prediction_consistency(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):
    # 이 분석은 DROID 62개 unfiltered 분석이다.
    # confidence mask를 의도적으로 주입하지 않는다.
    self.model.eval()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        "diagnostics/results",
        f"prediction_consistency_droid_unfiltered_full_{stamp}",
    )
    raw_dir = out_dir / "raw_sequences"
    raw_dir.mkdir(parents=True, exist_ok=False)

    metric_values = defaultdict(list)
    timestep_l2 = defaultdict(list)
    timestep_valid_counts = defaultdict(int)
    timestep_moved_counts = defaultdict(int)
    timestep_static_counts = defaultdict(int)

    per_sequence = []

    reversal_totals = {
        "prediction_count": 0,
        "prediction_defined_count": 0,
        "gt_count": 0,
        "gt_defined_count": 0,
        "common_defined_count": 0,
        "disagreement_count": 0,
        "prediction_only_count": 0,
    }

    processed = 0
    max_batches = (
        self.args.eval_num_batches
        if self.args.eval_num_batches > 0
        else None
    )

    iterator = tqdm(dl, desc=f"Prediction consistency [{split}]")

    for batch_index, batch in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break

        batch = {
            k: v.to(self.device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
        }

        outputs = self.model(batch, training=False)
        _, official_dict = self.model.loss_fn(
            outputs,
            batch,
            training=False,
        )

        pred = outputs["scene_flows"]
        gt = batch["gt_scene_flows"]

        assert pred.shape == gt.shape
        assert pred.ndim == 4 and pred.shape[-1] == 3

        context = mask3(batch, "scene_context_mask")
        exists = mask3(batch, "scene_exists")
        supervised = mask3(batch, "scene_supervised_mask")
        moved = mask3(batch, "scene_moved_mask")
        static = mask3(batch, "scene_static_mask")

        valid = (~context) & exists & supervised
        l2 = torch.linalg.vector_norm(pred - gt, dim=-1)

        recomputed_l2 = float(l2[valid].mean())
        official_l2 = float(official_dict["l2/mean"])
        l2_difference = abs(recomputed_l2 - official_l2)

        if l2_difference > 1e-6:
            raise RuntimeError(
                f"L2 reproduction failed at batch {batch_index}: "
                f"official={official_l2}, "
                f"recomputed={recomputed_l2}, "
                f"diff={l2_difference}"
            )

        pred_v = pred[:, 1:] - pred[:, :-1]
        gt_v = gt[:, 1:] - gt[:, :-1]
        velocity_valid = valid[:, 1:] & valid[:, :-1]
        velocity_error = torch.linalg.vector_norm(
            pred_v - gt_v,
            dim=-1,
        )

        pred_a = pred_v[:, 1:] - pred_v[:, :-1]
        gt_a = gt_v[:, 1:] - gt_v[:, :-1]
        acceleration_valid = (
            valid[:, 2:]
            & valid[:, 1:-1]
            & valid[:, :-2]
        )
        acceleration_error = torch.linalg.vector_norm(
            pred_a - gt_a,
            dim=-1,
        )

        error_variation_valid = valid[:, 1:] & valid[:, :-1]
        error_variation = torch.abs(
            l2[:, 1:] - l2[:, :-1]
        )

        pred_j = pred_a[:, 1:] - pred_a[:, :-1]
        gt_j = gt_a[:, 1:] - gt_a[:, :-1]
        jerk_valid = (
            valid[:, 3:]
            & valid[:, 2:-1]
            & valid[:, 1:-2]
            & valid[:, :-3]
        )

        pred_jerk_mag = torch.linalg.vector_norm(pred_j, dim=-1)
        gt_jerk_mag = torch.linalg.vector_norm(gt_j, dim=-1)
        jerk_error = torch.linalg.vector_norm(
            pred_j - gt_j,
            dim=-1,
        )

        metric_values["velocity_error"].append(
            velocity_error[velocity_valid].detach().cpu()
        )
        metric_values["acceleration_error"].append(
            acceleration_error[acceleration_valid].detach().cpu()
        )
        metric_values["prediction_error_variation"].append(
            error_variation[error_variation_valid].detach().cpu()
        )
        metric_values["prediction_jerk_magnitude"].append(
            pred_jerk_mag[jerk_valid].detach().cpu()
        )
        metric_values["gt_jerk_magnitude"].append(
            gt_jerk_mag[jerk_valid].detach().cpu()
        )
        metric_values["jerk_error"].append(
            jerk_error[jerk_valid].detach().cpu()
        )

        reversal_valid = (
            velocity_valid[:, 1:]
            & velocity_valid[:, :-1]
        )

        pred_v_prev = pred_v[:, :-1]
        pred_v_next = pred_v[:, 1:]
        gt_v_prev = gt_v[:, :-1]
        gt_v_next = gt_v[:, 1:]

        eps = torch.finfo(pred.dtype).eps

        pred_defined = (
            reversal_valid
            & (
                torch.linalg.vector_norm(
                    pred_v_prev, dim=-1
                ) > eps
            )
            & (
                torch.linalg.vector_norm(
                    pred_v_next, dim=-1
                ) > eps
            )
        )

        gt_defined = (
            reversal_valid
            & (
                torch.linalg.vector_norm(
                    gt_v_prev, dim=-1
                ) > eps
            )
            & (
                torch.linalg.vector_norm(
                    gt_v_next, dim=-1
                ) > eps
            )
        )

        pred_reversal = (
            (pred_v_prev * pred_v_next).sum(dim=-1) < 0
        )
        gt_reversal = (
            (gt_v_prev * gt_v_next).sum(dim=-1) < 0
        )

        common_defined = pred_defined & gt_defined

        pred_rev_count = int(
            (pred_reversal & pred_defined).sum()
        )
        pred_defined_count = int(pred_defined.sum())
        gt_rev_count = int(
            (gt_reversal & gt_defined).sum()
        )
        gt_defined_count = int(gt_defined.sum())
        disagreement_count = int(
            (
                (pred_reversal != gt_reversal)
                & common_defined
            ).sum()
        )
        pred_only_count = int(
            (
                pred_reversal
                & (~gt_reversal)
                & common_defined
            ).sum()
        )

        reversal_totals["prediction_count"] += pred_rev_count
        reversal_totals[
            "prediction_defined_count"
        ] += pred_defined_count
        reversal_totals["gt_count"] += gt_rev_count
        reversal_totals[
            "gt_defined_count"
        ] += gt_defined_count
        reversal_totals[
            "common_defined_count"
        ] += int(common_defined.sum())
        reversal_totals[
            "disagreement_count"
        ] += disagreement_count
        reversal_totals[
            "prediction_only_count"
        ] += pred_only_count

        B, T, N, _ = pred.shape
        keys = [str(x) for x in batch["__key__"]]
        domains = [str(x) for x in batch["__domain__"]]

        for t in range(T):
            m = valid[:, t]
            timestep_valid_counts[t] += int(m.sum())
            timestep_moved_counts[t] += int(
                (m & moved[:, t]).sum()
            )
            timestep_static_counts[t] += int(
                (m & static[:, t]).sum()
            )
            if m.any():
                timestep_l2[t].append(
                    l2[:, t][m].detach().cpu()
                )

        # 현재 실행은 batch_size=1 기준
        for b in range(B):
            seq_valid = valid[b]
            seq_velocity_valid = velocity_valid[b]
            seq_acceleration_valid = acceleration_valid[b]
            seq_error_variation_valid = error_variation_valid[b]
            seq_jerk_valid = jerk_valid[b]

            row = {
                "sequence_index": processed,
                "key": keys[b],
                "domain": domains[b],
                "official_l2_mean": official_l2,
                "recomputed_l2_mean": recomputed_l2,
                "l2_difference": l2_difference,
                "valid_count": int(seq_valid.sum()),
                "velocity_error_mean": float(
                    velocity_error[b][seq_velocity_valid].mean()
                ),
                "acceleration_error_mean": float(
                    acceleration_error[b][
                        seq_acceleration_valid
                    ].mean()
                ),
                "error_variation_mean": float(
                    error_variation[b][
                        seq_error_variation_valid
                    ].mean()
                ),
                "prediction_jerk_mean": float(
                    pred_jerk_mag[b][seq_jerk_valid].mean()
                ),
                "gt_jerk_mean": float(
                    gt_jerk_mag[b][seq_jerk_valid].mean()
                ),
                "jerk_error_mean": float(
                    jerk_error[b][seq_jerk_valid].mean()
                ),
                "prediction_reversal_rate": rate(
                    pred_rev_count,
                    pred_defined_count,
                ),
                "gt_reversal_rate": rate(
                    gt_rev_count,
                    gt_defined_count,
                ),
                "reversal_disagreement_rate": rate(
                    disagreement_count,
                    int(common_defined.sum()),
                ),
            }
            per_sequence.append(row)

            torch.save(
                {
                    "prediction_scene_flows": pred[b].detach().cpu(),
                    "gt_scene_flows": gt[b].detach().cpu(),
                    "official_valid_mask": valid[b].detach().cpu(),
                    "scene_moved_mask": moved[b].detach().cpu(),
                    "scene_static_mask": static[b].detach().cpu(),
                    "key": keys[b],
                    "domain": domains[b],
                },
                raw_dir / f"{processed:04d}.pt",
            )

            processed += 1

        iterator.set_postfix(processed=processed)

    if processed == 0:
        raise RuntimeError("No sequences processed")

    combined = {
        name: torch.cat(values)
        for name, values in metric_values.items()
    }

    per_timestep_rows = []
    for t in sorted(timestep_valid_counts):
        values = (
            torch.cat(timestep_l2[t])
            if timestep_l2[t]
            else torch.empty(0)
        )
        s = stats(values)
        per_timestep_rows.append({
            "timestep": t,
            "valid_point_count": timestep_valid_counts[t],
            "moved_valid_count": timestep_moved_counts[t],
            "static_valid_count": timestep_static_counts[t],
            "l2_mean": s["mean"],
            "l2_median": s["median"],
            "l2_p95": s["p95"],
            "l2_p99": s["p99"],
        })

    summary = {
        "analysis_type": (
            "Official tensors and masks with additional "
            "temporal diagnostics"
        ),
        "scope": "DROID 62-sample unfiltered subset",
        "processed_sequences": processed,
        "official_l2_reproduced_for_all_sequences": True,
        "velocity_error": stats(combined["velocity_error"]),
        "acceleration_error": stats(
            combined["acceleration_error"]
        ),
        "prediction_error_variation": stats(
            combined["prediction_error_variation"]
        ),
        "prediction_jerk_magnitude": stats(
            combined["prediction_jerk_magnitude"]
        ),
        "gt_jerk_magnitude": stats(
            combined["gt_jerk_magnitude"]
        ),
        "jerk_error": stats(combined["jerk_error"]),
        "direction_reversal": {
            **reversal_totals,
            "prediction_rate": rate(
                reversal_totals["prediction_count"],
                reversal_totals[
                    "prediction_defined_count"
                ],
            ),
            "gt_rate": rate(
                reversal_totals["gt_count"],
                reversal_totals["gt_defined_count"],
            ),
            "disagreement_rate": rate(
                reversal_totals["disagreement_count"],
                reversal_totals[
                    "common_defined_count"
                ],
            ),
            "definition": "dot(v[t-1], v[t]) < 0",
        },
        "units": {
            "l2": "meter",
            "velocity_error": "meter/timestep",
            "acceleration_error": "meter/timestep^2",
            "jerk": "meter/timestep^3",
        },
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(
        out_dir / "per_sequence.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(per_sequence[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_sequence)

    with open(
        out_dir / "per_timestep.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(per_timestep_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_timestep_rows)

    print()
    print("=" * 72)
    print("Full Prediction Consistency completed")
    print(f"Output directory: {out_dir}")
    print(f"Processed sequences: {processed}")
    print("Official L2 reproduced for all sequences: True")
    print("=" * 72)

    return {
        f"diagnostic/{split}/processed_sequences": processed,
    }


Tester._accumulate_metrics = full_prediction_consistency
runpy.run_path("eval.py", run_name="__main__")
