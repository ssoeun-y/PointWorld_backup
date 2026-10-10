#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]

INFILE = (
    ROOT
    / "diagnostics/results/equal_physical_delta_global_heldout/"
      "global_heldout_task0000__equal_delta.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/equal_physical_delta_global_heldout/analysis"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INFILE)

print("=" * 110)
print("[B4] EQUAL-DELTA LOCAL SENSITIVITY")
print("=" * 110)
print("Total N =", len(df))
print()


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    gt = 0
    lt = 0

    for a in x:
        gt += np.sum(a > y)
        lt += np.sum(a < y)

    return (gt - lt) / (len(x) * len(y))


def analyze_axis(
    axis,
    matched_col,
    outside_col,
    severity_col,
    actual_delta_col,
    requested_delta_col,
    collateral_col,
    sensitivity_cols,
):

    d = df[
        df[matched_col].astype(bool)
    ].copy()

    print()
    print("=" * 110)
    print(axis)
    print("=" * 110)

    print(
        f"matched = {len(d)} / {len(df)} "
        f"({100*len(d)/len(df):.2f}%)"
    )

    if len(d) == 0:
        print("NO MATCHED SAMPLES")
        return []

    d["regime"] = np.where(
        d[outside_col].astype(bool),
        "Outside",
        "Inside",
    )

    print(
        "Inside N =",
        int((d["regime"] == "Inside").sum())
    )
    print(
        "Outside N =",
        int((d["regime"] == "Outside").sum())
    )

    print()
    print(
        "actual delta median =",
        pd.to_numeric(
            d[actual_delta_col],
            errors="coerce"
        ).median()
    )

    print(
        "requested delta =",
        pd.to_numeric(
            d[requested_delta_col],
            errors="coerce"
        ).median()
    )

    print(
        "collateral delta median =",
        pd.to_numeric(
            d[collateral_col],
            errors="coerce"
        ).median()
    )

    rows = []

    for col in sensitivity_cols:

        z = d[
            [
                "regime",
                severity_col,
                col,
                actual_delta_col,
            ]
        ].copy()

        for c in [
            severity_col,
            col,
            actual_delta_col,
        ]:
            z[c] = pd.to_numeric(
                z[c],
                errors="coerce"
            )

        z = z.dropna()

        inside = z.loc[
            z["regime"] == "Inside",
            col,
        ].to_numpy()

        outside = z.loc[
            z["regime"] == "Outside",
            col,
        ].to_numpy()

        print()
        print("-" * 110)
        print(col)

        print(
            f"  Inside : N={len(inside)}, "
            f"median={np.median(inside) if len(inside) else np.nan:.6f} mm"
        )

        print(
            f"  Outside: N={len(outside)}, "
            f"median={np.median(outside) if len(outside) else np.nan:.6f} mm"
        )

        ratio = np.nan

        if (
            len(inside)
            and np.median(inside) != 0
        ):
            ratio = (
                np.median(outside)
                / np.median(inside)
            )

        print(
            f"  Outside/Inside median ratio = {ratio:.3f}x"
        )

        if len(inside) >= 2 and len(outside) >= 2:
            mw = mannwhitneyu(
                outside,
                inside,
                alternative="two-sided",
            )

            delta = cliffs_delta(
                outside,
                inside,
            )

            print(
                f"  Mann-Whitney p = {mw.pvalue:.3e}"
            )

            print(
                f"  Cliff delta = {delta:+.4f}"
            )

        else:
            mw = None
            delta = np.nan

        # ----------------------------------------------------
        # Local gain:
        # prediction shift per unit actually realized delta
        # ----------------------------------------------------

        z["local_gain"] = (
            z[col]
            / np.abs(
                z[actual_delta_col]
            )
        )

        gain_inside = z.loc[
            z["regime"] == "Inside",
            "local_gain",
        ]

        gain_outside = z.loc[
            z["regime"] == "Outside",
            "local_gain",
        ]

        print(
            f"  local gain Inside median  = "
            f"{gain_inside.median():.6g}"
        )

        print(
            f"  local gain Outside median = "
            f"{gain_outside.median():.6g}"
        )

        # ----------------------------------------------------
        # Among outside samples:
        # more boundary excess -> more sensitivity?
        # ----------------------------------------------------

        zo = z[
            z["regime"] == "Outside"
        ].copy()

        if len(zo) >= 10:

            rho, p = spearmanr(
                zo[severity_col],
                zo[col],
                nan_policy="omit",
            )

            gain_rho, gain_p = spearmanr(
                zo[severity_col],
                zo["local_gain"],
                nan_policy="omit",
            )

            print(
                f"  Outside severity -> shift: "
                f"rho={rho:+.4f}, p={p:.3e}"
            )

            print(
                f"  Outside severity -> local gain: "
                f"rho={gain_rho:+.4f}, p={gain_p:.3e}"
            )

        else:
            rho = p = gain_rho = gain_p = np.nan

        rows.append({
            "axis": axis,
            "metric": col,
            "N_matched": len(z),
            "N_inside": len(inside),
            "N_outside": len(outside),

            "inside_median_mm":
                np.median(inside)
                if len(inside) else np.nan,

            "outside_median_mm":
                np.median(outside)
                if len(outside) else np.nan,

            "outside_inside_ratio": ratio,

            "mw_p":
                mw.pvalue
                if mw is not None else np.nan,

            "cliff_delta": delta,

            "inside_local_gain_median":
                gain_inside.median(),

            "outside_local_gain_median":
                gain_outside.median(),

            "severity_shift_rho": rho,
            "severity_shift_p": p,

            "severity_gain_rho": gain_rho,
            "severity_gain_p": gain_p,
        })

    return rows


all_rows = []

all_rows += analyze_axis(
    axis="ACC",
    matched_col="acc_matched",
    outside_col="acc_outside",
    severity_col="acc_boundary_excess",
    actual_delta_col="actual_delta_acc",
    requested_delta_col="requested_delta_acc",
    collateral_col="acc_collateral_delta_msj",
    sensitivity_cols=[
        "acc_sensitivity_all_mm",
        "acc_sensitivity_moved_mm",
        "acc_sensitivity_t10_moved_mm",
    ],
)

all_rows += analyze_axis(
    axis="MSJ",
    matched_col="jerk_matched",
    outside_col="msj_outside",
    severity_col="msj_boundary_excess",
    actual_delta_col="actual_delta_msj",
    requested_delta_col="requested_delta_msj",
    collateral_col="jerk_collateral_delta_acc",
    sensitivity_cols=[
        "jerk_sensitivity_all_mm",
        "jerk_sensitivity_moved_mm",
        "jerk_sensitivity_t10_moved_mm",
    ],
)

out = pd.DataFrame(all_rows)

OUTFILE = (
    OUTDIR
    / "task0000_equal_delta_local_sensitivity_summary.csv"
)

out.to_csv(
    OUTFILE,
    index=False,
)

print()
print("=" * 110)
print("SAVED =", OUTFILE)
print("SUCCESS = True")
print("=" * 110)
