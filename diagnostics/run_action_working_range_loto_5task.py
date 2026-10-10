#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd

# Reuse the EXACT already-validated fitting machinery.
from diagnostics import analyze_action_working_range_v2 as wr


ROOT = Path(__file__).resolve().parents[1]

MAIN = (
    ROOT
    / "diagnostics/results/established_action_metrics_5task/"
      "per_sample_established_metrics_7987.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/action_working_range_loto_5task"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]

EXPECTED = {
    "task-0000": 377,
    "task-0001": 2322,
    "task-0006": 1782,
    "task-0008": 1827,
    "task-0010": 1679,
}

METRICS = [
    ("rms_acceleration", "ACC"),
    ("mean_squared_jerk", "MSJ"),
]

# Exact same bootstrap count as established analysis.
wr.BOOTSTRAP = 300


df = pd.read_csv(MAIN)

required = [
    "clip",
    "task",
    "l2_moved_mm",
    "rms_acceleration",
    "mean_squared_jerk",
]

missing = set(required) - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns: {sorted(missing)}")

df = df[df["task"].isin(TASKS)].copy()

for task, n in EXPECTED.items():
    got = int((df["task"] == task).sum())
    if got != n:
        raise RuntimeError(
            f"{task}: expected {n}, got {got}"
        )

if len(df) != 7987:
    raise RuntimeError(
        f"Expected total 7987, got {len(df)}"
    )


boundary_rows = []
all_targets = []

print("=" * 100)
print("5-TASK LEAVE-ONE-TASK-OUT WORKING RANGE")
print("=" * 100)
print("TOTAL =", len(df))
print()


for heldout in TASKS:

    train = df[df["task"] != heldout].copy()
    test = df[df["task"] == heldout].copy()

    print()
    print("=" * 100)
    print("HELD OUT =", heldout)
    print("=" * 100)
    print("train N =", len(train))
    print("held-out N =", len(test))

    fold_thresholds = {}

    for metric, short in METRICS:

        # --------------------------------------------------
        # Fit boundary using OTHER FOUR TASKS ONLY.
        # --------------------------------------------------
        fit = wr.fit_models(
            train,
            metric,
        )

        threshold = float(
            fit["threshold_raw"]
        )

        ci_lo, ci_hi, boot_n = (
            wr.bootstrap_threshold(
                train,
                metric,
            )
        )

        # --------------------------------------------------
        # Held-out evaluation only.
        # Absolutely no refitting from these results.
        # --------------------------------------------------
        val = wr.validate_threshold(
            test,
            metric,
            threshold,
        )

        train_breakpoint_supported = (
            fit["delta_bic_seg_vs_linear"] >= 10.0
            and
            fit["delta_bic_seg_vs_quadratic"] >= 6.0
            and
            fit["slope_post"] > fit["slope_pre"]
        )

        heldout_direction = (
            np.isfinite(val["val_med_in_mm"])
            and
            np.isfinite(val["val_med_out_mm"])
            and
            val["val_med_out_mm"]
                > val["val_med_in_mm"]
            and
            np.isfinite(val["val_cliff_delta"])
            and
            val["val_cliff_delta"] > 0
        )

        # Do not use this flag to decide whether correction runs.
        heldout_p_lt_005 = (
            np.isfinite(val["val_p"])
            and
            val["val_p"] < 0.05
        )

        fold_thresholds[metric] = threshold

        # SI conversion for reporting only.
        if metric == "rms_acceleration":
            threshold_si = threshold * 25.0
            ci_lo_si = ci_lo * 25.0
            ci_hi_si = ci_hi * 25.0
            si_unit = "m/s^2"
        else:
            threshold_si = threshold * 15625.0
            ci_lo_si = ci_lo * 15625.0
            ci_hi_si = ci_hi * 15625.0
            si_unit = "m^2/s^6"

        row = {
            "heldout_task": heldout,
            "metric": metric,

            "train_n": len(train),
            "heldout_n": len(test),

            "threshold_native": threshold,
            "bootstrap_ci_low_native": ci_lo,
            "bootstrap_ci_high_native": ci_hi,
            "bootstrap_success_n": boot_n,

            "threshold_si": threshold_si,
            "bootstrap_ci_low_si": ci_lo_si,
            "bootstrap_ci_high_si": ci_hi_si,
            "si_unit": si_unit,

            "delta_bic_seg_vs_linear":
                fit["delta_bic_seg_vs_linear"],

            "delta_bic_seg_vs_quadratic":
                fit["delta_bic_seg_vs_quadratic"],

            "slope_pre":
                fit["slope_pre"],

            "slope_post":
                fit["slope_post"],

            "train_breakpoint_supported":
                train_breakpoint_supported,

            **val,

            "heldout_direction":
                heldout_direction,

            "heldout_p_lt_005":
                heldout_p_lt_005,
        }

        boundary_rows.append(row)

        print()
        print(short)
        print(
            f"  boundary native = {threshold:.10g}"
        )
        print(
            f"  boundary SI     = {threshold_si:.6g} "
            f"{si_unit}"
        )
        print(
            f"  bootstrap CI    = "
            f"[{ci_lo:.10g}, {ci_hi:.10g}]"
        )
        print(
            "  ΔBIC seg-linear = "
            f"{fit['delta_bic_seg_vs_linear']:+.3f}"
        )
        print(
            "  ΔBIC seg-quad   = "
            f"{fit['delta_bic_seg_vs_quadratic']:+.3f}"
        )
        print(
            "  slopes          = "
            f"{fit['slope_pre']:+.4f} "
            f"-> {fit['slope_post']:+.4f}"
        )
        print(
            "  train breakpoint supported =",
            train_breakpoint_supported,
        )
        print(
            f"  held-out in/out N = "
            f"{val['val_n_in']} / {val['val_n_out']}"
        )
        print(
            f"  held-out error median = "
            f"{val['val_med_in_mm']:.4f} "
            f"-> {val['val_med_out_mm']:.4f} mm"
        )
        print(
            f"  held-out ratio = "
            f"{val['val_out_in_ratio']:.4f}"
        )
        print(
            f"  held-out Cliff = "
            f"{val['val_cliff_delta']:+.4f}"
        )
        print(
            f"  held-out p = "
            f"{val['val_p']:.6g}"
        )

    # ======================================================
    # Correction targets for THIS held-out task.
    #
    # Selection uses ONLY boundaries learned from other tasks.
    # We do NOT look at held-out prediction-error outcome.
    # ======================================================

    acc_b = fold_thresholds[
        "rms_acceleration"
    ]

    msj_b = fold_thresholds[
        "mean_squared_jerk"
    ]

    target = test.copy()

    target["loto_acc_boundary"] = acc_b
    target["loto_msj_boundary"] = msj_b

    target["acc_violated"] = (
        target["rms_acceleration"] > acc_b
    )

    target["msj_violated"] = (
        target["mean_squared_jerk"] > msj_b
    )

    target["either_violated"] = (
        target["acc_violated"]
        |
        target["msj_violated"]
    )

    target["both_violated"] = (
        target["acc_violated"]
        &
        target["msj_violated"]
    )

    target["acc_excess_frac"] = (
        target["rms_acceleration"] / acc_b
        - 1.0
    )

    target["msj_excess_frac"] = (
        target["mean_squared_jerk"] / msj_b
        - 1.0
    )

    selected = target[
        target["either_violated"]
    ].copy()

    selected["heldout_task"] = heldout

    selected.to_csv(
        OUTDIR
        / f"{heldout}__correction_targets.csv",
        index=False,
    )

    all_targets.append(selected)

    print()
    print(
        "CORRECTION TARGETS =",
        len(selected),
        "/",
        len(test),
        f"({100*len(selected)/len(test):.2f}%)",
    )

    print(
        "  Acc violated  =",
        int(target["acc_violated"].sum()),
    )

    print(
        "  MSJ violated  =",
        int(target["msj_violated"].sum()),
    )

    print(
        "  Both violated =",
        int(target["both_violated"].sum()),
    )


boundary_df = pd.DataFrame(
    boundary_rows
)

boundary_df.to_csv(
    OUTDIR / "loto_boundary_summary.csv",
    index=False,
)

targets_df = pd.concat(
    all_targets,
    ignore_index=True,
)

targets_df.to_csv(
    OUTDIR / "loto_correction_targets_all.csv",
    index=False,
)

# Compact fold summary.
compact = []

for task in TASKS:

    b = boundary_df[
        boundary_df["heldout_task"] == task
    ]

    t = targets_df[
        targets_df["heldout_task"] == task
    ]

    compact.append({
        "heldout_task": task,

        "heldout_n":
            EXPECTED[task],

        "correction_targets":
            len(t),

        "target_fraction":
            len(t) / EXPECTED[task],

        "acc_boundary":
            float(
                b.loc[
                    b["metric"]
                    == "rms_acceleration",
                    "threshold_native"
                ].iloc[0]
            ),

        "msj_boundary":
            float(
                b.loc[
                    b["metric"]
                    == "mean_squared_jerk",
                    "threshold_native"
                ].iloc[0]
            ),
    })

pd.DataFrame(compact).to_csv(
    OUTDIR / "loto_fold_compact.csv",
    index=False,
)

summary = {
    "seed": wr.SEED,
    "bootstrap": wr.BOOTSTRAP,
    "tasks": TASKS,
    "total_population": int(len(df)),
    "total_loto_targets": int(len(targets_df)),
    "principle": (
        "Each held-out task is excluded completely "
        "from boundary fitting. Correction target "
        "selection uses only the boundary fitted on "
        "the other four tasks."
    ),
}

with open(
    OUTDIR / "summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print()
print("=" * 100)
print("FINAL")
print("=" * 100)

print(
    boundary_df[
        [
            "heldout_task",
            "metric",
            "threshold_native",
            "train_breakpoint_supported",
            "val_n_in",
            "val_n_out",
            "val_out_in_ratio",
            "val_cliff_delta",
            "val_p",
        ]
    ].to_string(index=False)
)

print()
print("TARGET COUNTS")
print(
    pd.DataFrame(compact).to_string(
        index=False
    )
)

print()
print("saved =", OUTDIR)
print("SUCCESS = True")
