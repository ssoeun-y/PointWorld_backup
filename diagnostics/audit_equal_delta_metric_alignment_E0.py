#!/usr/bin/env python3

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260902
MATCH_TOL = 0.05

ROOT = Path(__file__).resolve().parents[1]

EST = (
    ROOT /
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)

BDIR = (
    ROOT /
    "diagnostics/results/equal_physical_delta_global_heldout"
)

TASKS = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]


# ============================================================
# Reproduce the fixed task-controlled 70/30 split
# ============================================================

def stable_number(text):
    h = hashlib.sha256(
        f"{SEED}|{text}".encode()
    ).digest()

    return int.from_bytes(
        h[:8], "big"
    ) / 2**64


def exact_split(df):
    out = pd.Series(
        index=df.index,
        dtype="object",
    )

    for task, g in df.groupby("task"):

        idx = sorted(
            g.index,
            key=lambda i:
                stable_number(
                    str(df.at[i, "clip"])
                )
        )

        n_cal = round(
            0.70 * len(idx)
        )

        out.loc[idx[:n_cal]] = "calibration"
        out.loc[idx[n_cal:]] = "validation"

    return out


# ============================================================
# Load established official analysis table
# ============================================================

e = pd.read_csv(EST)

required_est = [
    "clip",
    "task",
    "rms_acceleration",
    "mean_squared_jerk",
    "l2_moved_mm",
]

missing = [
    c for c in required_est
    if c not in e.columns
]

if missing:
    raise RuntimeError(
        f"Established CSV missing columns: {missing}\n"
        f"Available={list(e.columns)}"
    )

assert len(e) == 7987
assert e["clip"].nunique() == 7987

e["split_reproduced"] = exact_split(e)

print("=" * 100)
print("ESTABLISHED DATA")
print("=" * 100)
print("N total       =", len(e))
print(
    "N calibration =",
    int((e["split_reproduced"] == "calibration").sum())
)
print(
    "N validation  =",
    int((e["split_reproduced"] == "validation").sum())
)

print()
print("Validation by task:")
print(
    e.loc[
        e["split_reproduced"] == "validation",
        "task"
    ].value_counts().sort_index().to_string()
)


# ============================================================
# Load ONLY raw per-clip Equal-delta outputs.
# Never use old summary tables.
# ============================================================

raw_files = sorted(
    BDIR.glob(
        "global_heldout_task*__equal_delta.csv"
    )
)

if not raw_files:
    raise RuntimeError(
        "No raw Equal-delta per-clip CSVs found."
    )

frames = []

print()
print("=" * 100)
print("RAW B FILES")
print("=" * 100)

for p in raw_files:
    d = pd.read_csv(p)

    if "clip" not in d.columns:
        continue

    print(
        p.name,
        "| rows =", len(d),
        "| unique =", d["clip"].nunique()
    )

    d["source_file"] = p.name
    frames.append(d)

b = pd.concat(
    frames,
    ignore_index=True
)

if b["clip"].duplicated().any():
    z = b.loc[
        b["clip"].duplicated(False),
        ["clip", "source_file"]
    ]
    raise RuntimeError(
        "Duplicate clips across raw B files:\n"
        + z.to_string(index=False)
    )


# ============================================================
# Merge with established table
# ============================================================

m = b.merge(
    e[
        [
            "clip",
            "task",
            "split_reproduced",
            "rms_acceleration",
            "mean_squared_jerk",
            "l2_moved_mm",
        ]
    ],
    on="clip",
    how="left",
    suffixes=("_b", "_est"),
    validate="one_to_one",
)

print()
print("=" * 100)
print("IDENTITY / SPLIT CHECK")
print("=" * 100)

print("B rows              =", len(b))
print("Merged rows         =", len(m))
print(
    "Missing established =",
    int(m["rms_acceleration"].isna().sum())
)

assert len(m) == len(b)
assert m["rms_acceleration"].notna().all()

# Task identity
task_equal = (
    m["task_b"].astype(str)
    == m["task_est"].astype(str)
)

print(
    "Task identity       =",
    f"{task_equal.mean()*100:.6f}%"
)

assert task_equal.all()

# Every Equal-delta row must belong to fixed held-out split.
is_val = (
    m["split_reproduced"]
    == "validation"
)

print(
    "Held-out identity   =",
    f"{is_val.mean()*100:.6f}%"
)

assert is_val.all()


# ============================================================
# Exact metric-alignment audit
# ============================================================

def metric_check(name, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    absdiff = np.abs(x - y)

    denom = np.maximum(
        np.maximum(
            np.abs(x),
            np.abs(y)
        ),
        1e-15
    )

    reldiff = absdiff / denom

    print()
    print(name)
    print(
        "  max abs diff =",
        f"{np.nanmax(absdiff):.12g}"
    )
    print(
        "  median abs   =",
        f"{np.nanmedian(absdiff):.12g}"
    )
    print(
        "  max rel diff =",
        f"{np.nanmax(reldiff):.12g}"
    )
    print(
        "  median rel   =",
        f"{np.nanmedian(reldiff):.12g}"
    )

    return absdiff, reldiff


print()
print("=" * 100)
print("ACTION METRIC ALIGNMENT")
print("=" * 100)

acc_abs, acc_rel = metric_check(
    "RMS ACC",
    m["original_action_acc"],
    m["rms_acceleration"],
)

msj_abs, msj_rel = metric_check(
    "MSJ",
    m["original_action_msj"],
    m["mean_squared_jerk"],
)

# Serialization-level numerical equality expected.
acc_ok = (
    np.nanmax(acc_rel) <= 1e-6
)
msj_ok = (
    np.nanmax(msj_rel) <= 1e-6
)

print()
print("ACC aligned =", acc_ok)
print("MSJ aligned =", msj_ok)


# ============================================================
# Perturbation matching semantics
# ============================================================

print()
print("=" * 100)
print("MATCHING SEMANTICS")
print("=" * 100)

acc_relerr = pd.to_numeric(
    m["acc_match_relerr"],
    errors="coerce",
).to_numpy(float)

jerk_relerr = pd.to_numeric(
    m["jerk_match_relerr"],
    errors="coerce",
).to_numpy(float)

acc_expected = (
    np.isfinite(acc_relerr)
    & (acc_relerr <= MATCH_TOL)
)

jerk_expected = (
    np.isfinite(jerk_relerr)
    & (jerk_relerr <= MATCH_TOL)
)

acc_flag = (
    pd.to_numeric(
        m["acc_matched"],
        errors="coerce"
    ).fillna(0).to_numpy(int)
    == 1
)

jerk_flag = (
    pd.to_numeric(
        m["jerk_matched"],
        errors="coerce"
    ).fillna(0).to_numpy(int)
    == 1
)

print(
    "ACC matched        =",
    f"{acc_flag.sum()}/{len(m)} "
    f"({100*acc_flag.mean():.2f}%)"
)

print(
    "JERK matched       =",
    f"{jerk_flag.sum()}/{len(m)} "
    f"({100*jerk_flag.mean():.2f}%)"
)

print(
    "ACC flag agreement =",
    f"{(acc_expected == acc_flag).mean()*100:.6f}%"
)

print(
    "JERK flag agreement=",
    f"{(jerk_expected == jerk_flag).mean()*100:.6f}%"
)

assert np.array_equal(
    acc_expected,
    acc_flag
)

assert np.array_equal(
    jerk_expected,
    jerk_flag
)


# ============================================================
# Requested delta sanity
# ============================================================

print()
print("=" * 100)
print("DELTA SANITY")
print("=" * 100)

for c in [
    "requested_delta_acc",
    "requested_delta_msj",
]:
    x = pd.to_numeric(
        m[c],
        errors="coerce"
    ).to_numpy(float)

    print(
        f"{c:25s}",
        "unique finite =",
        np.unique(
            x[np.isfinite(x)]
        )
    )


# ============================================================
# Crucial population counts for future E2
# ============================================================

print()
print("=" * 100)
print("E2 AVAILABLE POPULATIONS")
print("=" * 100)

for task in sorted(
    m["task_b"].unique()
):
    q = m[
        m["task_b"] == task
    ]

    print(
        f"{task}: "
        f"N={len(q)} | "
        f"ACC matched={int(q['acc_matched'].sum())} | "
        f"JERK matched={int(q['jerk_matched'].sum())} | "
        f"BOTH matched={int(((q['acc_matched']==1) & (q['jerk_matched']==1)).sum())}"
    )

print()
print(
    "TOTAL:",
    "N =", len(m),
    "| ACC matched =", int(m["acc_matched"].sum()),
    "| JERK matched =", int(m["jerk_matched"].sum()),
    "| BOTH matched =",
    int(
        (
            (m["acc_matched"] == 1)
            &
            (m["jerk_matched"] == 1)
        ).sum()
    )
)

print()
print("=" * 100)

if acc_ok and msj_ok:
    print(
        "PASS: Equal-delta Action metrics match the "
        "established boundary metrics."
    )
else:
    print(
        "FAIL: metric definitions are not numerically aligned. "
        "Do NOT proceed to E2."
    )

print("=" * 100)
