#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch


def compare_exact(a, b):
    return {
        "shape_equal": a.shape == b.shape,
        "dtype_equal": a.dtype == b.dtype,
        "exact_equal": torch.equal(a, b),
        "max_abs_difference": (
            float((a.float() - b.float()).abs().max())
            if a.numel() > 0
            else 0.0
        ),
    }


def compare_close(a, b, tolerance):
    if a.shape != b.shape:
        return {
            "shape_equal": False,
            "dtype_equal": a.dtype == b.dtype,
            "allclose": False,
            "max_abs_difference": None,
        }

    difference = (a.float() - b.float()).abs()

    return {
        "shape_equal": True,
        "dtype_equal": a.dtype == b.dtype,
        "allclose": bool(torch.all(difference <= tolerance)),
        "max_abs_difference": (
            float(difference.max())
            if difference.numel() > 0
            else 0.0
        ),
        "tolerance": tolerance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = args.run_dir / "baseline.pt"
    identity_path = args.run_dir / "identity.pt"

    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)

    if not identity_path.exists():
        raise FileNotFoundError(identity_path)

    baseline = torch.load(
        baseline_path,
        map_location="cpu",
        weights_only=False,
    )
    identity = torch.load(
        identity_path,
        map_location="cpu",
        weights_only=False,
    )

    if baseline["sample_key"] != identity["sample_key"]:
        raise RuntimeError(
            "Baseline and identity used different samples: "
            f"{baseline['sample_key']} vs {identity['sample_key']}"
        )

    if identity["hook_state"]["call_count"] < 1:
        raise RuntimeError("Identity hook call count is zero")

    report = {
        "sample_key": baseline["sample_key"],
        "identity_hook_state": identity["hook_state"],
        "batch_comparison": {},
        "output_comparison": {},
    }

    # These must be bitwise identical.
    for key in baseline["batch"]:
        if key not in identity["batch"]:
            raise RuntimeError(
                f"Identity capture missing batch key: {key}"
            )

        report["batch_comparison"][key] = compare_exact(
            baseline["batch"][key],
            identity["batch"][key],
        )

    # GPU output comparison allows only tiny numerical tolerance.
    for key in baseline["outputs"]:
        if key not in identity["outputs"]:
            raise RuntimeError(
                f"Identity capture missing output key: {key}"
            )

        report["output_comparison"][key] = compare_close(
            baseline["outputs"][key],
            identity["outputs"][key],
            tolerance=1e-6,
        )

    batch_success = all(
        item["exact_equal"]
        for item in report["batch_comparison"].values()
    )

    output_success = all(
        item["allclose"]
        for item in report["output_comparison"].values()
    )

    hook_success = (
        identity["hook_state"]["call_count"] >= 1
        and identity["hook_state"]["joint_copy_exact"]
        and identity["hook_state"]["gripper_copy_exact"]
    )

    report["success"] = (
        batch_success
        and output_success
        and hook_success
    )

    output_path = args.run_dir / "comparison.json"

    with open(output_path, "w") as file:
        json.dump(report, file, indent=2)

    print("=" * 72)
    print("Action identity comparison")
    print(f"Sample: {report['sample_key']}")
    print(f"Identity hook calls: {identity['hook_state']['call_count']}")
    print()

    print("Batch tensors:")
    for key, item in report["batch_comparison"].items():
        print(
            f"  {key}: "
            f"exact={item['exact_equal']} "
            f"max_diff={item['max_abs_difference']}"
        )

    print()
    print("Model outputs:")
    for key, item in report["output_comparison"].items():
        print(
            f"  {key}: "
            f"allclose={item['allclose']} "
            f"max_diff={item['max_abs_difference']}"
        )

    print()
    print(f"IDENTITY VALIDATION SUCCESS: {report['success']}")
    print(f"Report: {output_path}")
    print("=" * 72)

    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
