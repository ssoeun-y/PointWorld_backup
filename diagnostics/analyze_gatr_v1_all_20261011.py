from pathlib import Path
import numpy as np
import pandas as pd

R = Path("diagnostics/results")
root = R / "acc_gatr_v1_local_eval_20261011"
master = (
    R / "gatr_v1_local_20261011/master.log"
).read_text()

tasks = ("0000", "0001", "0006", "0008", "0010")
metrics = (
    "pa_mpjpe_mm",
    "se3_residual_mm",
    "pairwise_error_mm"
)

all_data = []

for t in tasks:
    tag = f"FINISHED GATR V1 task-{t}"
    if tag not in master:
        print(f"PENDING task-{t}")
        continue

    path = root / f"task-{t}"
    a = pd.read_csv(
        path / "mechanism_action_audit.csv"
    )
    g = pd.read_csv(
        path / "mechanism_geometry_per_frame.csv"
    )

    assert not a["clip"].duplicated().any()

    f = g.groupby(
        ["clip", "frame", "variant"],
        as_index=False
    )[list(metrics)].mean()

    p = f.pivot(
        index=["clip", "frame"],
        columns="variant",
        values=list(metrics)
    )
    p.columns = [
        f"{m}_{v}" for m, v in p.columns
    ]

    required = [
        f"{m}_{v}"
        for m in metrics
        for v in ("base", "base_repeat", "corrected")
    ]
    p = p.dropna(subset=required)

    counts = p.groupby(level="clip").size()
    valid_clips = counts[counts == 10].index
    p = p[
        p.index.get_level_values("clip").isin(valid_clips)
    ]

    c = p.groupby(level="clip").mean()
    z = a.set_index("clip").join(
        c, how="inner", validate="one_to_one"
    ).reset_index()

    z["task"] = f"task-{t}"
    z["applied"] = (
        z["gatr_applied"].astype(str)
        .str.lower().eq("true")
    )
    z["boundary_valid"] = (
        z["gatr_boundary_valid"].astype(str)
        .str.lower().eq("true")
    )

    all_data.append(z)

if not all_data:
    raise SystemExit("NO COMPLETED TASKS")

D = pd.concat(all_data, ignore_index=True)

out = R / "gatr_v1_local_20261011"
D.to_csv(
    out / "completed_task_clip_metrics.csv",
    index=False
)

print("\n===== COMPLETED TASKS =====")
print(D.groupby("task").agg(
    n=("clip", "size"),
    applied=("applied", "sum"),
    boundary_valid=("boundary_valid", "sum")
).to_string())

rng = np.random.default_rng(20261011)

def report(label, q, m):
    if q.empty:
        return

    gain = (
        q[f"{m}_base"].to_numpy()
        - q[f"{m}_corrected"].to_numpy()
    )
    noise = np.abs(
        q[f"{m}_base"].to_numpy()
        - q[f"{m}_base_repeat"].to_numpy()
    )

    boots = np.median(
        rng.choice(
            gain, size=(2000, len(gain)),
            replace=True
        ),
        axis=1
    )
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(
        f"{label} N={len(q)}",
        f"median_gain={np.median(gain):+.6f}mm",
        f"95%CI=[{lo:+.6f},{hi:+.6f}]",
        f"improved={100*np.mean(gain>0):.2f}%",
        f"above_2x_noise={100*np.mean(gain>2*noise):.2f}%"
    )

print("\n===== OVERALL =====")
print("Clips:", len(D))
print("Local applied:", int(D["applied"].sum()))
print("Boundary valid:", int(D["boundary_valid"].sum()))

for m in metrics:
    print(f"\n===== {m} =====")
    report("ALL", D, m)
    report("APPLIED", D[D["applied"]], m)
    report("UNCHANGED", D[~D["applied"]], m)

print("\n===== TASKWISE PA =====")
for task, q in D.groupby("task"):
    report(
        task,
        q[q["applied"]],
        "pa_mpjpe_mm"
    )

bad = D[~D["boundary_valid"]]
if not bad.empty:
    cols = [
        c for c in (
            "task", "clip", "gatr_applied",
            "gatr_fixed_acc_si",
            "modified_acc_si",
            "gatr_boundary_valid"
        )
        if c in bad.columns
    ]
    print("\n===== BOUNDARY VIOLATIONS =====")
    print(bad[cols].to_string(index=False))

print("\nANALYSIS COMPLETE")
