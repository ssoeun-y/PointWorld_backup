#!/usr/bin/env python3

import json
import random
import runpy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from evaluation.tester import Tester


REPEATS = 5
RESET_SEED = 42


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stats(x):
    x = x.detach().float().cpu().reshape(-1)

    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p95": float(torch.quantile(x, 0.95)),
        "p99": float(torch.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def get_mask(batch, key):
    x = batch[key].bool()

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    return x


@torch.no_grad()
def audit_repeatability(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):
    self.model.eval()

    batch = next(iter(dl))
    batch = {
        key: value.to(self.device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }

    valid = (
        (~get_mask(batch, "scene_context_mask"))
        & get_mask(batch, "scene_exists")
        & get_mask(batch, "scene_supervised_mask")
    )

    gt = batch["gt_scene_flows"]

    # 첫 실행은 GPU warm-up 용도로만 사용하고 버린다.
    set_all_seeds(RESET_SEED)
    _ = self.model(batch, training=False)
    torch.cuda.synchronize()

    results = {}

    for mode in ["natural", "reseeded"]:
        outputs = []
        official_l2 = []

        for repeat in range(REPEATS):
            if mode == "reseeded":
                # 매 forward 전에 동일한 RNG 상태를 복원한다.
                set_all_seeds(RESET_SEED)

            output = self.model(batch, training=False)
            torch.cuda.synchronize()

            scene_flow = output["scene_flows"].detach().cpu()
            outputs.append(scene_flow)

            l2 = torch.linalg.vector_norm(
                output["scene_flows"] - gt,
                dim=-1,
            )
            official_l2.append(float(l2[valid].mean()))

        reference = outputs[0]
        comparisons = []

        for index, current in enumerate(outputs[1:], start=1):
            coordinate_difference = torch.abs(current - reference)
            point_difference = torch.linalg.vector_norm(
                current - reference,
                dim=-1,
            )

            comparisons.append({
                "repeat_index": index,
                "exact_equal": bool(torch.equal(current, reference)),
                "allclose_1e-6": bool(
                    torch.allclose(
                        current,
                        reference,
                        atol=1e-6,
                        rtol=0,
                    )
                ),
                "coordinate_abs_difference": stats(
                    coordinate_difference
                ),
                "point_l2_difference_m": stats(
                    point_difference
                ),
            })

        l2_tensor = torch.tensor(official_l2)

        results[mode] = {
            "repeat_count": REPEATS,
            "official_l2_each_run_m": official_l2,
            "official_l2_range_m": float(
                l2_tensor.max() - l2_tensor.min()
            ),
            "official_l2_std_m": float(
                l2_tensor.std(unbiased=False)
            ),
            "comparisons_to_first": comparisons,
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        "diagnostics/results",
        f"action_forward_repeatability_{timestamp}",
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    summary = {
        "sample_key": list(batch["__key__"]),
        "repeats": REPEATS,
        "seed": RESET_SEED,
        "definition": {
            "natural": "Repeated forward calls without resetting RNG",
            "reseeded": (
                "Python, NumPy, CPU and CUDA seeds reset "
                "before every forward call"
            ),
        },
        "results": results,
    }

    with open(output_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    print()
    print("=" * 72)
    print("Action forward repeatability completed")
    print(f"Output: {output_dir}")
    print(f"Sample: {summary['sample_key']}")

    for mode in ["natural", "reseeded"]:
        comparisons = results[mode]["comparisons_to_first"]
        max_point_diff = max(
            item["point_l2_difference_m"]["max"]
            for item in comparisons
        )
        mean_point_diff = max(
            item["point_l2_difference_m"]["mean"]
            for item in comparisons
        )

        print()
        print(mode)
        print(
            "  all exact:",
            all(item["exact_equal"] for item in comparisons),
        )
        print(
            "  max mean point difference:",
            mean_point_diff,
            "m",
        )
        print(
            "  max point difference:",
            max_point_diff,
            "m",
        )
        print(
            "  official L2 range:",
            results[mode]["official_l2_range_m"],
            "m",
        )

    print("=" * 72)

    return {
        f"diagnostic/{split}/repeatability_checked": 1,
    }


Tester._accumulate_metrics = audit_repeatability
runpy.run_path("eval.py", run_name="__main__")
