#!/usr/bin/env python3

import csv
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.tester import Tester


ROOT = Path(__file__).resolve().parents[1]

META = ROOT / "diagnostics/results/behavior_moved_test_inventory.csv"
OUT = ROOT / "diagnostics/results/action_perturbation_realism_audit.csv"


with META.open() as f:
    rows = list(csv.DictReader(f))

TARGET = {
    r["clip"]
    for r in rows
    if int(r["has_moved"]) == 1
}

assert len(TARGET) == 5288


def time_map(T, variant):
    t = np.linspace(0.0, 1.0, T)

    if variant == "baseline":
        return t

    if variant.startswith("speed_"):
        amp = float(variant.split("_")[1])
        s = t + amp * np.sin(2*np.pi*t)

    elif variant == "accel_spike":
        u = np.linspace(0.0, 1.0, 2001)
        v = 1.0 + 4.0 * np.exp(
            -0.5 * ((u - 0.55) / 0.07)**2
        )
        du = np.diff(u)
        integ = np.concatenate([
            [0.0],
            np.cumsum(
                0.5 * (v[:-1] + v[1:]) * du
            )
        ])
        integ /= integ[-1]
        s = np.interp(t, u, integ)

    elif variant == "reversal":
        kt = np.array(
            [0.0, .20, .40, .55, .65, .75, 1.0]
        )
        ks = np.array(
            [0.0, .20, .45, .60, .45, .70, 1.0]
        )
        s = np.interp(t, kt, ks)

    else:
        raise ValueError(variant)

    assert abs(s[0]) < 1e-12
    assert abs(s[-1] - 1.0) < 1e-12
    return s


def resample(x, s):
    T = x.shape[0]
    u = np.linspace(0.0, 1.0, T)

    flat = x.reshape(T, -1).astype(np.float64)
    out = np.empty_like(flat)

    for j in range(flat.shape[1]):
        out[:, j] = np.interp(s, u, flat[:, j])

    return out.reshape(x.shape).astype(np.float32)


def quality(x, exists):
    # x: T,N,3
    # same finite-difference definition used previously

    v = x[1:] - x[:-1]
    valid_v = exists[1:] & exists[:-1]

    a = v[1:] - v[:-1]
    valid_a = valid_v[1:] & valid_v[:-1]

    j = a[1:] - a[:-1]
    valid_j = valid_a[1:] & valid_a[:-1]

    acc = np.linalg.norm(a, axis=-1)
    jerk = np.linalg.norm(j, axis=-1)

    av = acc[valid_a]
    jv = jerk[valid_j]

    return {
        "p95_acc_mm_step2":
            float(np.quantile(av, .95) * 1000)
            if av.size else np.nan,

        "p95_jerk_mm_step3":
            float(np.quantile(jv, .95) * 1000)
            if jv.size else np.nan,
    }


variants = [
    "baseline",
    "speed_0.04",
    "speed_0.08",
    "speed_0.12",
    "accel_spike",
    "reversal",
]

results = []


@torch.no_grad()
def capture(self, dl, split, apply_confidence_mask=False):
    found = set()

    for batch in dl:
        keys = [str(x) for x in batch["__key__"]]

        flows = batch["robot_flows"]
        exists = batch["robot_exists"]

        for i, key in enumerate(keys):
            if key not in TARGET or key in found:
                continue

            x0 = flows[i].detach().cpu().numpy()
            e0 = exists[i].detach().cpu().numpy()

            if e0.ndim == 3 and e0.shape[-1] == 1:
                e0 = e0[..., 0]

            e0 = e0.astype(bool)

            T = x0.shape[0]

            for variant in variants:
                s = time_map(T, variant)

                if variant == "baseline":
                    x = x0
                else:
                    x = resample(x0, s)

                    if not np.allclose(
                        x[0], x0[0], atol=1e-6
                    ):
                        raise RuntimeError("t0 changed")

                    if not np.allclose(
                        x[-1], x0[-1], atol=1e-6
                    ):
                        raise RuntimeError("final changed")

                q = quality(x, e0)

                results.append({
                    "clip": key,
                    "variant": variant,
                    **q,
                })

            found.add(key)

        if len(found) == len(TARGET):
            break

    if len(found) != len(TARGET):
        raise RuntimeError(
            f"found {len(found)} / {len(TARGET)}"
        )

    df = pd.DataFrame(results)

    # --------------------------------------------------
    # Percentile relative to NATURAL baseline distribution
    # --------------------------------------------------
    for col in [
        "p95_acc_mm_step2",
        "p95_jerk_mm_step3",
    ]:
        base = (
            df[df.variant == "baseline"][col]
            .dropna()
            .sort_values()
            .to_numpy()
        )

        def percentile(v):
            if not np.isfinite(v):
                return np.nan
            return (
                np.searchsorted(base, v, side="right")
                / len(base) * 100.0
            )

        df[col + "_natural_percentile"] = (
            df[col].map(percentile)
        )

    df.to_csv(OUT, index=False)

    print("\n" + "="*95)
    print("NATURAL BASELINE DISTRIBUTION")
    print("="*95)

    b = df[df.variant == "baseline"]

    for col in [
        "p95_acc_mm_step2",
        "p95_jerk_mm_step3",
    ]:
        x = b[col].dropna()

        print(f"\n{col}")
        for q in [.50, .90, .95, .99]:
            print(
                f"  natural p{int(q*100):02d} "
                f"= {x.quantile(q):.4f}"
            )

    print("\n" + "="*95)
    print("PERTURBATION REALISM")
    print("="*95)

    base_idx = (
        df[df.variant == "baseline"]
        .set_index("clip")
    )

    for variant in variants[1:]:
        d = (
            df[df.variant == variant]
            .set_index("clip")
        )

        print(f"\n[{variant}]")

        for col, label in [
            ("p95_acc_mm_step2", "ACC"),
            ("p95_jerk_mm_step3", "JERK"),
        ]:
            common = d.index.intersection(base_idx.index)

            x = d.loc[common, col]
            x0 = base_idx.loc[common, col]

            ratio = x / x0.replace(0, np.nan)

            pct = d.loc[
                common,
                col + "_natural_percentile"
            ]

            natural95 = (
                b[col].dropna().quantile(.95)
            )
            natural99 = (
                b[col].dropna().quantile(.99)
            )

            print(
                f" {label}: "
                f"median={x.median():.4f}, "
                f"median_ratio_vs_original="
                f"{ratio.median():.2f}x, "
                f"median_natural_percentile="
                f"{pct.median():.1f}%, "
                f"within_natural_p95="
                f"{(x <= natural95).mean()*100:.1f}%, "
                f"within_natural_p99="
                f"{(x <= natural99).mean()*100:.1f}%"
            )

    print("\nsaved =", OUT)
    print("SUCCESS = True")

    return {
        f"diagnostic/{split}/realism": 1.0
    }


Tester._accumulate_metrics = capture

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
