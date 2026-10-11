from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

TASKS = ["0000", "0001", "0006", "0008", "0010"]
ROOT = Path("diagnostics/results/object_pa_mpjpe_acc_5task_current10457")
OUT = Path("diagnostics/results/cause_screen_20261011")
OUT.mkdir(parents=True, exist_ok=True)

files = [ROOT/f"task-{t}"/"per_frame_metrics.csv" for t in TASKS]
for p in files:
    if not p.is_file():
        raise FileNotFoundError(f"MAIN PA FILE MISSING: {p}")

parts = []
for task, p in zip(TASKS, files):
    d = pd.read_csv(p)
    needed = {
        "clip", "object_id", "frame",
        "pa_mpjpe_mm", "mpjpe_mm",
        "valid_point_count",
        "rms_acceleration_si_m_s2"
    }
    missing = needed-set(d.columns)
    if missing:
        raise RuntimeError(f"{p}: missing {sorted(missing)}")
    d["task"] = task
    parts.append(d)

raw = pd.concat(parts, ignore_index=True)
for c in [
    "frame", "pa_mpjpe_mm", "mpjpe_mm",
    "valid_point_count", "rms_acceleration_si_m_s2"
]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

raw = raw.dropna(subset=[
    "pa_mpjpe_mm", "mpjpe_mm", "frame"
])

cf = raw.groupby(
    ["task", "clip", "frame"], as_index=False
).agg(
    pa=("pa_mpjpe_mm", "mean"),
    mpjpe=("mpjpe_mm", "mean"),
    acc=("rms_acceleration_si_m_s2", "first")
)

clips = cf.groupby(["task","clip"], as_index=False).agg(
    pa=("pa", "median"),
    mpjpe=("mpjpe", "median"),
    acc=("acc","first")
)

n = len(clips)
print("\n=== POPULATION AUDIT ===")
print("Clips:", n, "Expected: 7947")
print("Clip-frames:", len(cf), "Expected: 79470")
print("Object rows:", len(raw), "Expected: 139760")
print(clips.groupby("task").size().to_string())

if n != 7947 or len(cf) != 79470 or len(raw) != 139760:
    raise RuntimeError("MAIN PA POPULATION MISMATCH")

# 1. Rigidity: object shape PA error and deformation after
# removing similarity transformation.
print("\n=== H1: OBJECT RIGIDITY ===")
print(
    "Object PA-MPJPE pooled median (mm):",
    round(float(clips.pa.median()), 6)
)
print(
    "Raw MPJPE pooled median (mm):",
    round(float(clips.mpjpe.median()), 6)
)
print("PA residual measures shape distortion after alignment.")

# 2. Displacement hypothesis:
# Acceleration association does NOT establish correspondence failure.
print("\n=== H2: DISPLACEMENT / CORRESPONDENCE ===")
for task, g in clips.groupby("task"):
    low = g[g.acc <= g.acc.quantile(.25)].pa
    high = g[g.acc >= g.acc.quantile(.75)].pa
    print(
        task,
        "Low25 PA", round(float(low.median()),4),
        "High25 PA",round(float(high.median()),4),
        "ratio",round(float(high.median()/max(low.median(),1e-12)),3)
    )
print(
    "CAUTION: acceleration association is not a direct "
    "displacement/correspondence mechanism test."
)

# 3. Point-count proxy for possible training-loss imbalance.
# NOT direct gradient contribution or background-vs-object loss.
print("\n=== H3: OBJECT SUPPORT / LOSS IMBALANCE PROXY ===")
obj = raw.groupby(
    ["task","clip","object_id"],as_index=False
).agg(
    pa=("pa_mpjpe_mm","mean"),
    points=("valid_point_count","median")
)
support_rows = []
for task,g in obj.groupby("task"):
    g = g[(g.points>=3)&np.isfinite(g.pa)]
    rho,p = spearmanr(np.log(g.points),g.pa)
    support_rows.append({
        "task":task,"objects":len(g),
        "support_vs_pa_rho":rho,"p":p
    })
    print(
        task, "objects",len(g),
        "Spearman(point_count,PA)",
        round(float(rho),4),"p",f"{p:.3g}"
    )
pd.DataFrame(support_rows).to_csv(
    OUT/"object_support_proxy.csv",index=False
)
print("CAUTION: point count is NOT measured loss gradient share.")

# 4. Temporal error accumulation.
print("\n=== H4: TEMPORAL ERROR ACCUMULATION ===")
start = int(cf.frame.min())
end = int(cf.frame.max())
wide = cf.pivot(
    index=["task","clip"],columns="frame",values="pa"
)
paired = wide[[start,end]].dropna()
delta = paired[end]-paired[start]
try:
    p_time = wilcoxon(delta).pvalue
except ValueError:
    p_time = float("nan")
print(
    "PA last-first median(mm)",round(float(delta.median()),6),
    "worsened(%)",round(float(100*(delta>0).mean()),2),
    "paired p",p_time
)
time_rows=[]
for task,g in cf.groupby("task"):
    w=g.pivot(index="clip",columns="frame",values="pa")
    if start not in w or end not in w:
        continue
    v=(w[end]-w[start]).dropna()
    row={
        "task":task,"clips":len(v),
        "delta_pa_mm_median":float(v.median()),
        "worse_pct":float(100*(v>0).mean())
    }
    time_rows.append(row)
    print(task,row)
pd.DataFrame(time_rows).to_csv(
    OUT/"temporal_drift_5task.csv",index=False
)
print(
    "CAUTION: later-frame growth does not prove a "
    "slow-motion prior."
)

# 5. Architecture / representation audit.
print("\n=== H5: OBJECT-LEVEL REPRESENTATION ===")
base = Path("pointworld/base.py").read_text()
loss = Path("pointworld/losses.py").read_text()
pipe = Path("dataset_components/pipeline.py").read_text()
pointwise_head = (
    "dynamics_head_output = self.dynamics_head(padded_scene_feat)"
    in base
)
object_id_passed = "scene_object_id" in pipe
explicit_rigid_loss = any(
    x in loss for x in [
        "rigidity_loss", "rigid_loss",
        "pairwise_distance_loss", "procrustes_loss"
    ]
)
print("Per-point displacement head:",pointwise_head)
print("Original pipeline object IDs:",object_id_passed)
print("Named rigidity loss:",explicit_rigid_loss)
print(
    "CAUTION: architecture audit establishes absence of an "
    "explicit guarantee, not a proven unique cause."
)

# Final task summary.
rows=[]
for task,g in clips.groupby("task"):
    rows.append({
        "task":task,
        "clips":len(g),
        "pa_median_mm":float(g.pa.median()),
        "mpjpe_median_mm":float(g.mpjpe.median()),
    })
rows.append({
    "task":"ALL",
    "clips":len(clips),
    "pa_median_mm":float(clips.pa.median()),
    "mpjpe_median_mm":float(clips.mpjpe.median()),
})
pd.DataFrame(rows).to_csv(
    OUT/"baseline_5task.csv",index=False
)

print("\n=== PRIORITIZATION ===")
if pointwise_head and not object_id_passed and not explicit_rigid_loss:
    print(
        "PRIMARY STRUCTURAL HYPOTHESIS: "
        "LACK OF OBJECT-LEVEL RIGID-MOTION CONSTRAINT"
    )
else:
    print(
        "ARCHITECTURAL AUDIT INCONCLUSIVE: inspect model manually"
    )
print(
    "H2 correspondence and H3 loss dominance remain "
    "unproven without targeted interventions."
)
print("OUTPUT:", OUT.resolve())
print("CAUSE_SCREEN_COMPLETE")
