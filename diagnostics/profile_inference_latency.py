#!/usr/bin/env python3

import os
import json
import runpy
from pathlib import Path

import numpy as np
import torch

from evaluation.tester import Tester


TARGET_KEY = os.environ.get(
    "PW_TARGET_KEY",
    "task-0000_episode_00002530-300:311",
)

WARMUP = int(os.environ.get("PW_LATENCY_WARMUP", "10"))
ITERS = int(os.environ.get("PW_LATENCY_ITERS", "30"))

OUT = Path(
    os.environ.get(
        "PW_LATENCY_OUT",
        "diagnostics/results/latency_profile_02530.json",
    )
)


def norm_key(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)


def stats(x):
    x = np.asarray(x, dtype=np.float64)

    return {
        "mean_ms": float(x.mean()),
        "median_ms": float(np.median(x)),
        "p10_ms": float(np.percentile(x, 10)),
        "p90_ms": float(np.percentile(x, 90)),
        "min_ms": float(x.min()),
        "max_ms": float(x.max()),
    }


def cuda_bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    values = []

    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.cuda.synchronize()

        values.append(start.elapsed_time(end))

    return values


@torch.no_grad()
def latency_capture(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):
    self.model.eval()

    for batch in dl:

        key = norm_key(batch["__key__"])

        if key != TARGET_KEY:
            continue

        batch = {
            k: (
                v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

        core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

        print("=" * 78)
        print("POINTWORLD LATENCY PROFILE")
        print("target =", TARGET_KEY)
        print("device =", self.device)
        print(
            "scene points =",
            int(batch["scene_exists"][:, 0].sum().item()),
        )
        print(
            "robot points/timestep mean =",
            float(
                batch["robot_exists"]
                .sum(dim=-1)
                .float()
                .mean()
                .item()
            ),
        )
        print("warmup =", WARMUP)
        print("iters  =", ITERS)
        print("=" * 78)

        # ----------------------------------------------------------
        # A. Official model forward: image + action -> 10-step output
        # ----------------------------------------------------------
        full_times = cuda_bench(
            lambda: core(batch, training=False),
            WARMUP,
            ITERS,
        )

        # ----------------------------------------------------------
        # B. Scene/image encoding only
        # ----------------------------------------------------------
        scene_times = cuda_bench(
            lambda: core.encode_scene_features(batch),
            WARMUP,
            ITERS,
        )

        # Cache image/scene feature exactly through official API.
        encoded_scene = core.encode_scene_features(batch).detach()
        torch.cuda.synchronize()

        # ----------------------------------------------------------
        # C. Action-conditioned forward with scene feature cached
        # ----------------------------------------------------------
        cached_times = cuda_bench(
            lambda: core(
                batch,
                training=False,
                encoded_scene_feat0=encoded_scene,
            ),
            WARMUP,
            ITERS,
        )

        # ----------------------------------------------------------
        # D. Stage-level CUDA-event profiling
        # Event hooks avoid synchronizing inside nested modules.
        # ----------------------------------------------------------
        stage_modules = {
            "scene_feature_encoder":
                core.scene_feature_encoder,

            "robot_projection":
                core.robot_proj,

            "time_embedding":
                core.time_embed,

            "dynamics_total":
                core.dynamics_predictor,

            "ptv3":
                core.dynamics_predictor.predictor_model,

            "dynamics_head":
                core.dynamics_predictor.dynamics_head,

            "uncertainty_head":
                core.dynamics_predictor.log_var_head,
        }

        current = {}
        pending = {}
        stage_values = {
            name: []
            for name in stage_modules
        }

        handles = []

        def make_pre(name):
            def hook(module, inputs):
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                current[name] = ev
            return hook

        def make_post(name):
            def hook(module, inputs, output):
                end = torch.cuda.Event(enable_timing=True)
                end.record()

                if name not in current:
                    raise RuntimeError(
                        f"missing start event for {name}"
                    )

                pending[name] = (
                    current.pop(name),
                    end,
                )
            return hook

        for name, module in stage_modules.items():
            handles.append(
                module.register_forward_pre_hook(
                    make_pre(name)
                )
            )
            handles.append(
                module.register_forward_hook(
                    make_post(name)
                )
            )

        # Small warmup before collecting hook measurements.
        for _ in range(3):
            core(batch, training=False)

        torch.cuda.synchronize()

        stage_total = []

        for _ in range(ITERS):
            current.clear()
            pending.clear()

            total_start = torch.cuda.Event(
                enable_timing=True
            )
            total_end = torch.cuda.Event(
                enable_timing=True
            )

            total_start.record()
            core(batch, training=False)
            total_end.record()

            torch.cuda.synchronize()

            stage_total.append(
                total_start.elapsed_time(total_end)
            )

            for name in stage_modules:
                if name not in pending:
                    raise RuntimeError(
                        f"stage did not execute: {name}"
                    )

                a, b = pending[name]

                stage_values[name].append(
                    a.elapsed_time(b)
                )

        for h in handles:
            h.remove()

        # ----------------------------------------------------------
        # Results
        # ----------------------------------------------------------
        result = {
            "target_key": TARGET_KEY,
            "warmup": WARMUP,
            "iters": ITERS,

            "scene_points_t0":
                int(batch["scene_exists"][:, 0].sum()),

            "full_forward": stats(full_times),

            "scene_encode": stats(scene_times),

            "cached_forward": stats(cached_times),

            "cache_speedup_median": (
                float(
                    np.median(full_times)
                    / np.median(cached_times)
                )
            ),

            "hooked_total": stats(stage_total),

            "stages": {
                name: stats(vals)
                for name, vals in stage_values.items()
            },
        }

        OUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with OUT.open("w") as f:
            json.dump(
                result,
                f,
                indent=2,
                sort_keys=True,
            )

        print("\n" + "=" * 78)
        print("MODEL-ONLY LATENCY")
        print("=" * 78)

        print(
            "FULL forward median      =",
            f"{result['full_forward']['median_ms']:.3f} ms",
        )

        print(
            "SCENE encode median      =",
            f"{result['scene_encode']['median_ms']:.3f} ms",
        )

        print(
            "CACHED-action median     =",
            f"{result['cached_forward']['median_ms']:.3f} ms",
        )

        print(
            "scene-cache speedup      =",
            f"{result['cache_speedup_median']:.2f}x",
        )

        print("\n" + "=" * 78)
        print("STAGE BREAKDOWN")
        print("=" * 78)

        for name in [
            "scene_feature_encoder",
            "robot_projection",
            "time_embedding",
            "dynamics_total",
            "ptv3",
            "dynamics_head",
            "uncertainty_head",
        ]:
            print(
                f"{name:24s} "
                f"{result['stages'][name]['median_ms']:9.3f} ms"
            )

        print("\nsaved =", OUT)
        print("=" * 78)

        return {
            f"diagnostic/{split}/latency_profile": 1.0
        }

    raise RuntimeError(
        f"target sample not found: {TARGET_KEY}"
    )


Tester._accumulate_metrics = latency_capture

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
