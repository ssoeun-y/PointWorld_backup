#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]

OUT_ROOT = (
    ROOT
    / "diagnostics/results/"
      "online_rigidity_reflex_4way_5task"
)

TASKS = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]


dfs = []

for task in TASKS:

    p = (
        OUT_ROOT
        / task
        / "per_clip.csv"
    )

    if not p.exists():
        raise FileNotFoundError(p)

    d = pd.read_csv(p)

    if len(d) == 0:
        raise RuntimeError(
            f"{task}: empty"
        )

    dfs.append(d)


df = pd.concat(
    dfs,
    ignore_index=True,
)

df.to_csv(
    OUT_ROOT
    / "combined_5task_per_clip.csv",
    index=False,
)


def rel_change(new, old):
    return (
        (new - old)
        / old
        * 100.0
    )


def report(name, x):

    original = x[
        "original_pa_mm"
    ]

    print()
    print("-" * 78)
    print(name)
    print("N =", len(x))

    for label, col in [
        (
            "Original",
            "original_pa_mm",
        ),
        (
            "Always slow10",
            "slow10_pa_mm",
        ),
        (
            "Rigidity reflex",
            "reflex_pa_mm",
        ),
        (
            "Acc boundary",
            "boundary_pa_mm",
        ),
        (
            "Oracle O/S",
            "oracle_pa_mm",
        ),
    ]:

        vals = x[col]

        print(
            f"{label:18s} "
            f"median={vals.median():.6f} mm"
        )

        if col != "original_pa_mm":

            rc = rel_change(
                vals,
                original,
            )

            print(
                f"{'':18s} "
                f"paired rel median="
                f"{rc.median():+.3f}% | "
                f"lower="
                f"{(vals < original).mean()*100:.1f}%"
            )

    print()

    print(
        "Reflex triggered slow = %.1f%%"
        % (
            x[
                "trigger_slow"
            ].mean()
            * 100
        )
    )

    print(
        "Reflex selection == O/S oracle = %.1f%%"
        % (
            x[
                "selection_equals_oracle"
            ].mean()
            * 100
        )
    )

    print(
        "Score-change sign correct = %.1f%%"
        % (
            x[
                "score_sign_correct"
            ].mean()
            * 100
        )
    )

    print(
        "Reflex lower than Always-slow = %.1f%%"
        % (
            (
                x["reflex_pa_mm"]
                <
                x["slow10_pa_mm"]
            ).mean()
            * 100
        )
    )

    print(
        "Reflex lower than Boundary = %.1f%%"
        % (
            (
                x["reflex_pa_mm"]
                <
                x["boundary_pa_mm"]
            ).mean()
            * 100
        )
    )

    d_ref_slow = (
        x["reflex_pa_mm"]
        - x["slow10_pa_mm"]
    )

    d_ref_B = (
        x["reflex_pa_mm"]
        - x["boundary_pa_mm"]
    )

    print(
        "Reflex - AlwaysSlow "
        "paired delta median = "
        f"{d_ref_slow.median():+.6f} mm"
    )

    print(
        "Reflex - Boundary "
        "paired delta median = "
        f"{d_ref_B.median():+.6f} mm"
    )


for task in TASKS:

    report(
        task,
        df[
            df["task"] == task
        ],
    )


report(
    "ALL 5 TASKS",
    df,
)


rho, p = spearmanr(
    df["delta_score"],
    df["delta_pa_slow_mm"],
)

print()
print("=" * 78)
print("GLOBAL SCORE VALIDATION")
print("=" * 78)

print(
    "delta rigidity score vs delta PA: "
    f"rho={rho:.6f}, p={p:.3e}"
)

triggered = df[
    df["trigger_slow"] == 1
]

print(
    "When reflex triggered slow, "
    "slow actually reduced PA = %.1f%%"
    % (
        (
            triggered[
                "delta_pa_slow_mm"
            ] < 0
        ).mean()
        * 100
    )
)

print(
    "Triggered delta PA median = %.6f mm"
    % triggered[
        "delta_pa_slow_mm"
    ].median()
)

print()
print(
    "SAVED =",
    OUT_ROOT
    / "combined_5task_per_clip.csv",
)
