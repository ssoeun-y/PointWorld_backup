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

CANDIDATES = [
    int(x)
    for x in os.environ.get(
        "PW_CANDIDATES", "1,8,32"
    ).split(",")
]

WARMUP = int(
    os.environ.get("PW_CAND_WARMUP", "3")
)

ITERS = int(
    os.environ.get("PW_CAND_ITERS", "10")
)

OUT = Path(
    os.environ.get(
        "PW_CAND_OUT",
        "diagnostics/results/candidate_latency_02530.json",
    )
)


def normalize_key(x):
    if isinstance(x, (list, tuple)):
        x = x[0]

    if isinstance(x, bytes):
        x = x.decode()

    return str(x)


def stat_ms(values):
    x = np.asarray(values, dtype=np.float64)

    return {
        "mean_ms": float(x.mean()),
        "median_ms": float(np.median(x)),
        "p90_ms": float(np.percentile(x, 90)),
        "min_ms": float(x.min()),
        "max_ms": float(x.max()),
    }


def timed(fn, warmup, iters):
    for _ in range(warmup):
        out = fn()
        del out

    torch.cuda.synchronize()

    vals = []

    for _ in range(iters):
        start = torch.cuda.Event(
            enable_timing=True
        )
        end = torch.cuda.Event(
            enable_timing=True
        )

        start.record()

        out = fn()

        end.record()
        torch.cuda.synchronize()

        vals.append(
            float(start.elapsed_time(end))
        )

        del out

    return vals


def make_minimal_batch(batch):
    """
    Cached-scene PointWorld forward only requires these fields.
    No RGB/depth duplication for candidate batching.
    """
    required = [
        "scene_flows",
        "scene_features",
        "scene_exists",
        "robot_flows",
        "robot_features",
        "robot_exists",
        "__domain__",
    ]

    out = {}

    for k in required:
        if k not in batch:
            raise RuntimeError(
                f"required field missing: {k}"
            )

        out[k] = batch[k]

    return out


def repeat_batch(batch, n):
    """
    Repeat one already-processed candidate to B=n.

    For latency profiling, candidate numerical values do not
    affect tensor shapes/compute materially. This measures
    candidate throughput capacity of the official forward.
    """
    out = {}

    for k, v in batch.items():

        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                out[k] = v
                continue

            if v.shape[0] != 1:
                raise RuntimeError(
                    f"{k}: expected source B=1, "
                    f"got {tuple(v.shape)}"
                )

            reps = [n] + [1] * (v.ndim - 1)

            out[k] = v.repeat(*reps)

        elif k == "__domain__":
            if not isinstance(v, (list, tuple)):
                raise RuntimeError(
                    "__domain__ must be list/tuple"
                )

            if len(v) != 1:
                raise RuntimeError(
                    f"expected one domain, got {v}"
                )

            out[k] = [v[0]] * n

        else:
            out[k] = v

    return out


def repeat_encoded(encoded, n):
    if not isinstance(encoded, torch.Tensor):
        raise RuntimeError(
            f"encoded scene expected tensor, "
            f"got {type(encoded)}"
        )

    if encoded.shape[0] != 1:
        raise RuntimeError(
            f"encoded source B != 1: {encoded.shape}"
        )

    reps = [n] + [1] * (encoded.ndim - 1)

    return encoded.repeat(*reps)


@torch.no_grad()
def candidate_latency_capture(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):
    self.model.eval()

    for batch in dl:

        key = normalize_key(
            batch["__key__"]
        )

        if key != TARGET_KEY:
            continue

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

        core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

        base = make_minimal_batch(batch)

        # --------------------------------------------------
        # Cache scene once.
        # --------------------------------------------------
        for _ in range(3):
            encoded = core.encode_scene_features(
                batch
            )

        torch.cuda.synchronize()

        start = torch.cuda.Event(
            enable_timing=True
        )
        end = torch.cuda.Event(
            enable_timing=True
        )

        start.record()
        encoded = core.encode_scene_features(
            batch
        )
        end.record()

        torch.cuda.synchronize()

        encode_once_ms = float(
            start.elapsed_time(end)
        )

        encoded = encoded.detach()

        print("=" * 86)
        print("POINTWORLD CANDIDATE-ACTION LATENCY")
        print("target =", TARGET_KEY)
        print(
            "scene points =",
            int(
                batch["scene_exists"][:, 0]
                .sum()
                .item()
            ),
        )
        print(
            "candidate counts =",
            CANDIDATES,
        )
        print(
            "scene encode once =",
            f"{encode_once_ms:.3f} ms",
        )
        print("=" * 86)

        results = {
            "target_key": TARGET_KEY,
            "scene_encode_once_ms":
                encode_once_ms,
            "warmup": WARMUP,
            "iters": ITERS,
            "candidates": {},
        }

        for n in CANDIDATES:

            print(
                f"\n--- candidates={n} ---"
            )

            # ==============================================
            # A. Sequential:
            # cached scene, N separate PTv3 calls.
            # ==============================================
            def sequential_fn():
                out = None

                for _ in range(n):
                    out = core(
                        base,
                        training=False,
                        encoded_scene_feat0=encoded,
                    )

                return out

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            seq_times = timed(
                sequential_fn,
                WARMUP,
                ITERS,
            )

            seq_peak_gb = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            seq = stat_ms(seq_times)

            # ==============================================
            # B. True batch:
            # B=N candidates in ONE official forward.
            # ==============================================
            batch_status = "ok"
            bat = None
            bat_peak_gb = None

            try:
                candidate_batch = repeat_batch(
                    base,
                    n,
                )

                encoded_batch = repeat_encoded(
                    encoded,
                    n,
                )

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                def batched_fn():
                    return core(
                        candidate_batch,
                        training=False,
                        encoded_scene_feat0=encoded_batch,
                    )

                bat_times = timed(
                    batched_fn,
                    WARMUP,
                    ITERS,
                )

                bat_peak_gb = (
                    torch.cuda.max_memory_allocated()
                    / 1024**3
                )

                bat = stat_ms(bat_times)

                del candidate_batch
                del encoded_batch

            except torch.cuda.OutOfMemoryError:
                batch_status = "oom"

                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    batch_status = "oom"
                    torch.cuda.empty_cache()
                else:
                    raise

            seq_median = seq["median_ms"]

            print(
                "sequential total median =",
                f"{seq_median:.3f} ms",
            )

            print(
                "sequential per candidate =",
                f"{seq_median/n:.3f} ms",
            )

            print(
                "sequential candidates/s  =",
                f"{1000*n/seq_median:.2f}",
            )

            if bat is not None:

                bat_median = bat["median_ms"]

                print(
                    "BATCH total median      =",
                    f"{bat_median:.3f} ms",
                )

                print(
                    "BATCH per candidate     =",
                    f"{bat_median/n:.3f} ms",
                )

                print(
                    "BATCH candidates/s      =",
                    f"{1000*n/bat_median:.2f}",
                )

                print(
                    "batch speedup vs seq     =",
                    f"{seq_median/bat_median:.2f}x",
                )

                print(
                    "batch peak allocated     =",
                    f"{bat_peak_gb:.2f} GB",
                )

            else:
                print(
                    "BATCH status             = OOM"
                )

            results["candidates"][str(n)] = {
                "sequential": seq,
                "sequential_per_candidate_ms":
                    seq_median / n,
                "sequential_candidates_per_sec":
                    1000.0 * n / seq_median,
                "sequential_peak_allocated_gb":
                    seq_peak_gb,

                "batch_status":
                    batch_status,
                "batched":
                    bat,
                "batched_peak_allocated_gb":
                    bat_peak_gb,
            }

            if bat is not None:
                results["candidates"][str(n)][
                    "batched_per_candidate_ms"
                ] = bat["median_ms"] / n

                results["candidates"][str(n)][
                    "batched_candidates_per_sec"
                ] = (
                    1000.0
                    * n
                    / bat["median_ms"]
                )

                results["candidates"][str(n)][
                    "batch_speedup_vs_sequential"
                ] = (
                    seq_median
                    / bat["median_ms"]
                )

        OUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with OUT.open("w") as f:
            json.dump(
                results,
                f,
                indent=2,
                sort_keys=True,
            )

        print("\n" + "=" * 86)
        print("saved =", OUT)
        print("=" * 86)

        return {
            f"diagnostic/{split}/candidate_latency":
                1.0
        }

    raise RuntimeError(
        f"target not found: {TARGET_KEY}"
    )


Tester._accumulate_metrics = (
    candidate_latency_capture
)

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
