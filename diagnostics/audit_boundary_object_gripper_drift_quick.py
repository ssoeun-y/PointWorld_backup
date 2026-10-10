from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("diagnostics/results/acc_boundary_rigid_support_v10_1_current10457")
TASKS = ["task-0000", "task-0001", "task-0006", "task-0008", "task-0010"]

dfs = []
for task in TASKS:
    p = ROOT / task / "per_object_endpoint_goal_metrics.csv"
    if not p.exists():
        print(f"[MISS] {p}")
        continue

    df = pd.read_csv(p)
    df["task"] = task
    dfs.append(df)

if not dfs:
    raise RuntimeError("No endpoint CSVs found.")

df = pd.concat(dfs, ignore_index=True)

need = [
    "task", "clip", "object_id",
    "gripper_q10_base_mm",
    "gripper_q10_corr_mm",
    "gripper_q10_delta_mm",
    "gripper_median_base_mm",
    "gripper_median_corr_mm",
    "centroid_gripper_base_mm",
    "centroid_gripper_corr_mm",
]
missing = [c for c in need if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

for c in need[3:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=[
    "gripper_q10_base_mm",
    "gripper_q10_corr_mm",
])

# ---------------------------------------------------------
# QUICK SCREEN ONLY.
# base = Original-action PointWorld prediction
# corr = Boundary-retimed action PointWorld prediction
#
# No V10.1 repair is used here.
#
# q10 <= 10 mm:
# at least ~10% of object points lie within 10 mm of gripper surface.
# This is only a quick fixed-distance screening criterion.
# ---------------------------------------------------------
base_contact = df["gripper_q10_base_mm"] <= 10.0
g = df[base_contact].copy()

g["q10_delta_mm"] = (
    g["gripper_q10_corr_mm"] - g["gripper_q10_base_mm"]
)

g["contact_break_10mm"] = (
    g["gripper_q10_corr_mm"] > 10.0
)

g["large_separation_20mm"] = (
    g["gripper_q10_corr_mm"] > 20.0
)

g["drift_gt_5mm"] = g["q10_delta_mm"] > 5.0
g["drift_gt_10mm"] = g["q10_delta_mm"] > 10.0

print()
print("======================================================")
print(" BOUNDARY-ONLY OBJECT ↔ GRIPPER DRIFT QUICK AUDIT")
print(" Original prediction vs Boundary-retimed prediction")
print(" V10.1 REPAIR: NOT USED")
print("======================================================")
print(f"All rigid-object rows       : {len(df):,}")
print(f"Original near-contact rows  : {len(g):,}")

if len(g) == 0:
    raise RuntimeError("No original near-contact rows found.")

print()
print("[POOLED]")
print(
    f"q10 distance median: "
    f"{g['gripper_q10_base_mm'].median():.3f} "
    f"-> {g['gripper_q10_corr_mm'].median():.3f} mm"
)
print(
    f"paired q10 delta median: "
    f"{g['q10_delta_mm'].median():+.3f} mm"
)
print(
    f"Boundary farther than Original: "
    f"{(g['q10_delta_mm'] > 0).mean()*100:.2f}%"
)
print(
    f"Contact break (>10 mm): "
    f"{g['contact_break_10mm'].sum():,}/{len(g):,} "
    f"= {g['contact_break_10mm'].mean()*100:.2f}%"
)
print(
    f"Large separation (>20 mm): "
    f"{g['large_separation_20mm'].sum():,}/{len(g):,} "
    f"= {g['large_separation_20mm'].mean()*100:.2f}%"
)
print(
    f"q10 drift > +5 mm: "
    f"{g['drift_gt_5mm'].sum():,}/{len(g):,} "
    f"= {g['drift_gt_5mm'].mean()*100:.2f}%"
)
print(
    f"q10 drift > +10 mm: "
    f"{g['drift_gt_10mm'].sum():,}/{len(g):,} "
    f"= {g['drift_gt_10mm'].mean()*100:.2f}%"
)

print()
print("[TASKWISE]")
for task, x in g.groupby("task", sort=True):
    print(
        f"{task} | N={len(x):4d} | "
        f"q10 {x['gripper_q10_base_mm'].median():6.2f}"
        f"->{x['gripper_q10_corr_mm'].median():6.2f} mm | "
        f"break={x['contact_break_10mm'].mean()*100:6.2f}% | "
        f">+5mm={x['drift_gt_5mm'].mean()*100:6.2f}% | "
        f">+10mm={x['drift_gt_10mm'].mean()*100:6.2f}%"
    )

# Clip-level: one or more originally-contacting rigid objects break.
clip = (
    g.groupby(["task", "clip"], as_index=False)
     .agg(
         n_contact_objects=("object_id", "size"),
         any_break=("contact_break_10mm", "max"),
         any_gt5=("drift_gt_5mm", "max"),
         any_gt10=("drift_gt_10mm", "max"),
         median_q10_delta_mm=("q10_delta_mm", "median"),
     )
)

print()
print("[CLIP LEVEL]")
print(f"Original-contact clips: {len(clip):,}")
print(
    f"Any contact break: "
    f"{clip['any_break'].sum():,}/{len(clip):,} "
    f"= {clip['any_break'].mean()*100:.2f}%"
)
print(
    f"Any >+5 mm drift: "
    f"{clip['any_gt5'].sum():,}/{len(clip):,} "
    f"= {clip['any_gt5'].mean()*100:.2f}%"
)
print(
    f"Any >+10 mm drift: "
    f"{clip['any_gt10'].sum():,}/{len(clip):,} "
    f"= {clip['any_gt10'].mean()*100:.2f}%"
)

out = ROOT / "boundary_only_object_gripper_drift_quick.csv"
g.to_csv(out, index=False)

clip_out = ROOT / "boundary_only_object_gripper_drift_quick_clip.csv"
clip.to_csv(clip_out, index=False)

print()
print("[SAVE]", out)
print("[SAVE]", clip_out)
