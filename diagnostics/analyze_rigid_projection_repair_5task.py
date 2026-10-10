from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

ROOT = Path(
    "diagnostics/results/"
    "rigid_projection_repair_current10457"
)

rows = []

required = {
    "task",
    "clip",
    "frame",
    "repair_raw_mpjpe_mm",
    "rigid_repair_mpjpe_mm",
    "rigid_residual_mm",
    "repair_delta_mpjpe_mm",
}

for p in sorted(ROOT.glob("task-*/*.csv")):
    try:
        d = pd.read_csv(p)
    except Exception:
        continue

    if required.issubset(d.columns):
        print("FOUND:", p)
        rows.append(d)

if not rows:
    raise RuntimeError("No repair frame CSVs found")

d = pd.concat(rows, ignore_index=True)

dedup_cols = [
    c for c in ["task", "clip", "object_id", "frame"]
    if c in d.columns
]

d = d.drop_duplicates(subset=dedup_cols)

m = (
    np.isfinite(d["repair_raw_mpjpe_mm"])
    & np.isfinite(d["rigid_repair_mpjpe_mm"])
    & np.isfinite(d["rigid_residual_mm"])
)
d = d[m].copy()

d["delta_mm"] = (
    d["rigid_repair_mpjpe_mm"]
    - d["repair_raw_mpjpe_mm"]
)

d["rel_pct"] = (
    100.0 * d["delta_mm"]
    / d["repair_raw_mpjpe_mm"]
)

print("\n===== FRAME LEVEL =====")
print("rows =", len(d))
print("clips =", d[["task","clip"]].drop_duplicates().shape[0])
print("raw median = %.6f mm" %
      d["repair_raw_mpjpe_mm"].median())
print("repair median = %.6f mm" %
      d["rigid_repair_mpjpe_mm"].median())
print("paired delta median = %.6f mm" %
      d["delta_mm"].median())
print("paired relative median = %.3f%%" %
      d["rel_pct"].median())
print("repair lower = %.2f%%" %
      (100.0 * (
          d["rigid_repair_mpjpe_mm"]
          < d["repair_raw_mpjpe_mm"]
      ).mean()))
print("rigid residual median = %.6f mm" %
      d["rigid_residual_mm"].median())

c = (
    d.groupby(["task","clip"], as_index=False)
    .agg(
        raw_mp=("repair_raw_mpjpe_mm","mean"),
        repaired_mp=("rigid_repair_mpjpe_mm","mean"),
        residual_mm=("rigid_residual_mm","mean"),
    )
)

c["delta_mm"] = c["repaired_mp"] - c["raw_mp"]
c["rel_pct"] = (
    100.0 * c["delta_mm"] / c["raw_mp"]
)

print("\n===== CLIP LEVEL — MAIN =====")
print("N =", len(c))
print("raw median = %.6f mm" %
      c["raw_mp"].median())
print("repair median = %.6f mm" %
      c["repaired_mp"].median())
print("aggregate reduction = %.3f%%" %
      (100.0 * (
          c["repaired_mp"].median()
          / c["raw_mp"].median()
          - 1.0
      )))
print("paired delta median = %.6f mm" %
      c["delta_mm"].median())
print("paired relative median = %.3f%%" %
      c["rel_pct"].median())
print("repair lower clips = %.2f%%" %
      (100.0 * (
          c["repaired_mp"] < c["raw_mp"]
      ).mean()))

try:
    w = wilcoxon(
        c["repaired_mp"],
        c["raw_mp"]
    )
    print("Wilcoxon p = %.6e" % w.pvalue)
except Exception as e:
    print("Wilcoxon failed:", repr(e))

print("\n===== TASKWISE =====")

for task, g in c.groupby("task"):
    print(
        task,
        "N=", len(g),
        "raw=%.6f" % g["raw_mp"].median(),
        "repair=%.6f" % g["repaired_mp"].median(),
        "paired_rel=%.3f%%" % g["rel_pct"].median(),
        "lower=%.2f%%" % (
            100.0 * (
                g["repaired_mp"] < g["raw_mp"]
            ).mean()
        ),
    )

c["benefit_mm"] = -c["delta_mm"]

rho, p = spearmanr(
    c["residual_mm"],
    c["benefit_mm"]
)

print("\n===== RESIDUAL -> BENEFIT =====")
print("Spearman rho = %.6f" % rho)
print("p = %.6e" % p)

c.to_csv(
    ROOT / "combined_5task_repair_per_clip.csv",
    index=False,
)

print("\nSAVED =", ROOT)
