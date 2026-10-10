from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PROX = (
    ROOT
    / "diagnostics/results/scene_action_proximity_probe/"
      "scene_action_proximity_5task_2397.csv"
)

OOF = (
    ROOT
    / "diagnostics/results/context_incremental_fair_oof/"
      "oof_predictions.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/context_vs_fixed_correction_gate"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "context_vs_fixed_gate_2067.csv"

# Frozen global boundaries
TA = 0.009832167363499603
TJ = 8.152857428804615e-05

p = pd.read_csv(PROX)
o = pd.read_csv(OOF)

# Exact fair population used in the context OOF experiment.
p = p[p["acc_matched"] == 1].copy()

need = [
    "clip",
    "task",
    "original_action_acc",
    "original_action_msj",
    "l2_moved_mm",
    "scene_action_dist_p05_native",
]

missing = [c for c in need if c not in p.columns]
if missing:
    raise RuntimeError(f"Missing PROX columns: {missing}")

need_oof = [
    "clip",
    "task",
    "high_risk",
    "task_boundary_oof",
    "task_context_oof",
]

missing = [c for c in need_oof if c not in o.columns]
if missing:
    raise RuntimeError(f"Missing OOF columns: {missing}")

q = p[need].merge(
    o[need_oof],
    on=["clip", "task"],
    how="inner",
    validate="one_to_one",
)

assert len(q) == 2067, len(q)
assert not q["clip"].duplicated().any()

# ------------------------------------------------------------
# FIXED gate = original frozen global envelope.
# ------------------------------------------------------------
q["fixed_gate"] = (
    (q["original_action_acc"] > TA)
    |
    (q["original_action_msj"] > TJ)
).astype(int)

K = int(q["fixed_gate"].sum())

# ------------------------------------------------------------
# CONTEXT gate:
# same correction budget K, using only cross-fitted context scores.
#
# Highest risk = correction target.
# Stable tie-breaking by clip.
# ------------------------------------------------------------
ranked = q.sort_values(
    ["task_context_oof", "clip"],
    ascending=[False, True],
    kind="mergesort",
).copy()

selected = set(
    ranked.iloc[:K]["clip"].tolist()
)

q["context_gate"] = (
    q["clip"].isin(selected)
).astype(int)

assert int(q["context_gate"].sum()) == K

q["gate_group"] = np.select(
    [
        (q["fixed_gate"] == 1) & (q["context_gate"] == 1),
        (q["fixed_gate"] == 1) & (q["context_gate"] == 0),
        (q["fixed_gate"] == 0) & (q["context_gate"] == 1),
    ],
    [
        "BOTH",
        "FIXED_ONLY",
        "CONTEXT_ONLY",
    ],
    default="NEITHER",
)

# ------------------------------------------------------------
# Descriptive selection quality only.
# This is NOT correction outcome yet.
# ------------------------------------------------------------
def describe(mask):
    g = q[mask]
    return {
        "n": len(g),
        "median_l2_moved_mm":
            float(g["l2_moved_mm"].median()) if len(g) else np.nan,
        "high_risk_rate":
            float(g["high_risk"].mean()) if len(g) else np.nan,
        "median_context_score":
            float(g["task_context_oof"].median()) if len(g) else np.nan,
    }

fixed_desc = describe(q["fixed_gate"] == 1)
ctx_desc = describe(q["context_gate"] == 1)

print("\n" + "=" * 100)
print("CONTEXT VS FIXED — SAME CORRECTION BUDGET")
print("=" * 100)

print("FAIR POPULATION =", len(q))
print("FIXED TARGETS   =", K)
print("CONTEXT TARGETS =", int(q["context_gate"].sum()))

print("\nOVERLAP:")
print(q["gate_group"].value_counts().to_string())

print("\nFIXED TARGET QUALITY:")
print(fixed_desc)

print("\nCONTEXT TARGET QUALITY:")
print(ctx_desc)

print("\nBY TASK:")
print(
    q.groupby("task")[
        ["fixed_gate", "context_gate"]
    ].sum().to_string()
)

q.to_csv(OUT, index=False)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
