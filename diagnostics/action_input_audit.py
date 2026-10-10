#!/usr/bin/env python3

import json
import math
import runpy
from datetime import datetime
from pathlib import Path

import torch

from evaluation.tester import Tester


TOKENS = (
    "action",
    "proprio",
    "joint",
    "eef",
    "end_effector",
    "robot",
    "command",
    "control",
    "trajectory",
)


def summarize_tensor(x):
    x_cpu = x.detach().cpu()

    result = {
        "type": "tensor",
        "shape": list(x_cpu.shape),
        "dtype": str(x_cpu.dtype),
        "device_before_move": str(x.device),
        "numel": int(x_cpu.numel()),
    }

    if x_cpu.numel() == 0:
        return result

    if torch.is_floating_point(x_cpu):
        finite = torch.isfinite(x_cpu)
        result["finite_rate"] = float(finite.float().mean())

        if finite.any():
            values = x_cpu[finite].float()
            result.update({
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
            })

    elif x_cpu.dtype == torch.bool:
        result["true_count"] = int(x_cpu.sum())
        result["true_rate"] = float(x_cpu.float().mean())

    elif x_cpu.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        result.update({
            "min": int(x_cpu.min()),
            "max": int(x_cpu.max()),
        })

    return result


def summarize_value(value):
    if isinstance(value, torch.Tensor):
        return summarize_tensor(value)

    if isinstance(value, (list, tuple)):
        preview = [str(x)[:100] for x in value[:3]]
        return {
            "type": type(value).__name__,
            "length": len(value),
            "preview": preview,
        }

    return {
        "type": type(value).__name__,
        "value": str(value)[:300],
    }


@torch.no_grad()
def audit_inputs(self, dl, split, apply_confidence_mask=False):
    self.model.eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        "diagnostics/results",
        f"action_input_audit_{timestamp}",
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    iterator = iter(dl)
    batch = next(iterator)

    batch_schema = {
        key: summarize_value(value)
        for key, value in sorted(batch.items())
    }

    candidate_keys = [
        key
        for key in sorted(batch)
        if any(token in key.lower() for token in TOKENS)
    ]

    device_batch = {
        key: value.to(self.device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }

    outputs = self.model(device_batch, training=False)

    output_schema = {
        key: summarize_value(value)
        for key, value in sorted(outputs.items())
    }

    result = {
        "split": split,
        "sample_key": str(batch.get("__key__", "unknown")),
        "candidate_action_related_keys": candidate_keys,
        "batch_schema": batch_schema,
        "output_schema": output_schema,
    }

    with open(output_dir / "audit.json", "w") as file:
        json.dump(result, file, indent=2, sort_keys=True)

    print()
    print("=" * 72)
    print("Action input audit completed")
    print(f"Output directory: {output_dir}")
    print(f"Sample key: {result['sample_key']}")
    print()
    print("Candidate action-related keys:")
    for key in candidate_keys:
        info = batch_schema[key]
        print(
            f"  {key}: "
            f"shape={info.get('shape')} "
            f"dtype={info.get('dtype')} "
            f"min={info.get('min')} "
            f"max={info.get('max')} "
            f"mean={info.get('mean')}"
        )

    if not candidate_keys:
        print("  NONE FOUND — inspect full audit.json and code search.")

    print()
    print("Model output keys:")
    for key, info in output_schema.items():
        print(
            f"  {key}: "
            f"shape={info.get('shape')} "
            f"dtype={info.get('dtype')}"
        )

    print("=" * 72)

    return {
        f"diagnostic/{split}/audited_batches": 1,
    }


Tester._accumulate_metrics = audit_inputs
runpy.run_path("eval.py", run_name="__main__")
