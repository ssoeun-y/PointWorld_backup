from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

GATE = (
    ROOT
    / "diagnostics/results/context_vs_fixed_correction_gate/"
      "context_vs_fixed_gate_2067.csv"
)

CORR = (
    ROOT
    / "diagnostics/results/"
      "goal_preserved_global_boundary_correction_taskcontrolled_7987/"
      "correction_results.csv"
)

STRUCT = (
    ROOT
    / "diagnostics/results/"
      "goal_preserved_global_boundary_taskcontrolled_pointcloud_structure/"
      "global_fixed_correct_5task__pointcloud_structure.csv"
)

TA = 0.009832167363499603
TJ = 8.152857428804615e-05

g = pd.read_csv(GATE)
c = pd.read_csv(CORR)
s = pd.read_csv(STRUCT)

assert len(g) == 2067
assert not g["clip"].duplicated().any()

c = c.drop_duplicates("clip")
s = s.drop_duplicates("clip")

q = g.merge(
    c[
        [
            "clip",
            "success",
            "corrected_acc",
            "corrected_msj",
            "acc_ratio",
            "msj_ratio",
        ]
    ],
    on="clip",
    how="left",
)

q = q.merge(
    s[
        [
            "clip",
            "ratio_jitter_mean_mm",
            "ratio_jitter_p95_mm",
            "ratio_rigidity_mean_mm",
        ]
    ],
    on="clip",
    how="left",
)

q["has_fixed_correction"] = q["success"].notna()
q["fixed_success"] = q["success"].fillna(False).astype(bool)
q["has_structure"] = q["ratio_jitter_mean_mm"].notna()

q["acc_rel_boundary"] = q["original_action_acc"] / TA
q["msj_rel_boundary"] = q["original_action_msj"] / TJ
q["fixed_severity"] = np.maximum(
    q["acc_rel_boundary"],
    q["msj_rel_boundary"],
)

print("=" * 100)
print("CONTEXT CORRECTION REUSE AUDIT")
print("=" * 100)

for group in [
    "BOTH",
    "FIXED_ONLY",
    "CONTEXT_ONLY",
    "NEITHER",
]:
    x = q[q["gate_group"] == group]

    print(f"\n[{group}]")
    print("N =", len(x))
    print(
        "existing fixed correction rows =",
        int(x["has_fixed_correction"].sum()),
    )
    print(
        "fixed correction success =",
        int(x["fixed_success"].sum()),
    )
    print(
        "existing structure scored =",
        int(x["has_structure"].sum()),
    )
    print(
        "median fixed severity =",
        float(x["fixed_severity"].median()),
    )

    if len(x):
        print(
            "inside global envelope =",
            int((x["fixed_severity"] <= 1.0).sum()),
        )
        print(
            "outside global envelope =",
            int((x["fixed_severity"] > 1.0).sum()),
        )

out = (
    ROOT
    / "diagnostics/results/context_vs_fixed_correction_gate/"
      "context_correction_reuse_audit.csv"
)

q.to_csv(out, index=False)

print("\nOUTPUT =", out)
print("SUCCESS = True")
