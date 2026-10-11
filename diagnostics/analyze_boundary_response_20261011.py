from pathlib import Path
import numpy as np
import pandas as pd

R = Path("diagnostics/results")
SRC = R / "acc_causal_geometry_boundary_20261010"
OUT = R / "acc_boundary_response_20261011"
OUT.mkdir(parents=True, exist_ok=True)

B = 0.24580418408749
RNG = np.random.default_rng(42)

def ci_median(x, boots=1000):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    draws = RNG.choice(x, size=(boots, len(x)), replace=True)
    return tuple(np.percentile(np.median(draws, axis=1), [2.5, 97.5]))

def make_quartiles(s):
    return pd.qcut(
        s.rank(method="first"), 4,
        labels=[1, 2, 3, 4]
    ).astype(int)

rows = []
tasks = []

for taskdir in sorted(SRC.glob("task-*")):
    ap = taskdir / "mechanism_action_audit.csv"
    gp = taskdir / "mechanism_geometry_per_frame.csv"

    if not ap.exists() or not gp.exists():
        continue

    try:
        a = pd.read_csv(ap)
        g = pd.read_csv(gp)
    except (OSError, pd.errors.ParserError) as e:
        print("SKIP incomplete:", taskdir.name, str(e))
        continue

    required_a = {
        "task", "clip", "original_acc_si",
        "modified_acc_si", "original_speed_rms",
        "saved_runtime_absdiff_si"
    }
    required_g = {
        "task", "clip", "object_id", "frame",
        "variant", "pa_mpjpe_mm"
    }
    assert required_a <= set(a), required_a - set(a)
    assert required_g <= set(g), required_g - set(g)

    # Clip-frame-object 동일 위치에서 세 변형 비교.
    p = g.pivot_table(
        index=["task", "clip", "object_id", "frame"],
        columns="variant",
        values="pa_mpjpe_mm",
        aggfunc="first"
    )

    assert {"base", "base_repeat", "corrected"} <= set(p)

    p = p[["base", "base_repeat", "corrected"]].dropna()
    p["benefit"] = p["base"] - p["corrected"]
    p["repeat_noise"] = abs(p["base"] - p["base_repeat"])

    clip = p.groupby(level=["task", "clip"]).agg(
        pa_base=("base", "mean"),
        pa_corrected=("corrected", "mean"),
        pa_benefit_mm=("benefit", "mean"),
        noise_mm=("repeat_noise", "mean"),
        n_object_frames=("benefit", "size")
    ).reset_index()

    if a[["task", "clip"]].duplicated().any():
        raise RuntimeError(f"Duplicate action audit: {taskdir}")

    merged = clip.merge(
        a, on=["task", "clip"],
        validate="one_to_one"
    )

    rows.append(merged)
    tasks.append(taskdir.name)

if not rows:
    raise RuntimeError("No completed boundary tasks found")

d = pd.concat(rows, ignore_index=True)
d = d.replace([np.inf, -np.inf], np.nan)

# 따로 기록된 correction과 현재 재평가한 Action이 일치하는지 확인.
d["reproduced"] = d.saved_runtime_absdiff_si <= 1e-6

# 정확한 Boundary 충족과 수치 허용오차 충족은 별도 기록.
d["exact_boundary"] = d.modified_acc_si <= B
d["tolerant_boundary"] = d.modified_acc_si <= B + 1e-6

d["rel_benefit_pct"] = np.where(
    d.pa_base > 1e-9,
    100 * d.pa_benefit_mm / d.pa_base,
    np.nan
)
d["improved"] = d.pa_benefit_mm > 0
d["robust_improved"] = d.pa_benefit_mm > 2 * d.noise_mm

d.to_csv(OUT / "all_boundary_response.csv", index=False)

print("===== COHORT AUDIT =====")
print("COMPLETED TASKS:", tasks)
print("ALL CLIPS:", len(d))
print("REPRODUCED:", int(d.reproduced.sum()))
print("EXACT BOUNDARY:", int(d.exact_boundary.sum()))
print("REPRODUCED + EXACT:", int(
    (d.reproduced & d.exact_boundary).sum()
))
print("REPRODUCED + TOLERANT:", int(
    (d.reproduced & d.tolerant_boundary).sum()
))

# Primary conservative analysis: actual runtime Boundary strictly satisfied.
z = d[
    d.reproduced
    & d.exact_boundary
].dropna(subset=[
    "original_acc_si",
    "original_speed_rms",
    "pa_benefit_mm",
    "noise_mm"
]).copy()

if len(z) < 20:
    raise RuntimeError("Too few strictly valid clips")

# Task-specific velocity and acceleration quartiles.
z["vel_q"] = z.groupby("task")[
    "original_speed_rms"
].transform(make_quartiles)

z["acc_q"] = z.groupby("task")[
    "original_acc_si"
].transform(make_quartiles)

summary = []

def append_group(name, group, task="ALL", acc_q="ALL", vel_q="ALL"):
    x = group.pa_benefit_mm.to_numpy(float)
    lo, hi = ci_median(x)
    summary.append({
        "group": name,
        "task": task,
        "acc_q": acc_q,
        "vel_q": vel_q,
        "n": len(group),
        "median_benefit_mm": float(np.median(x)),
        "ci95_low_mm": lo,
        "ci95_high_mm": hi,
        "median_relative_benefit_pct":
            group.rel_benefit_pct.median(),
        "improved_pct": 100 * group.improved.mean(),
        "robust_improved_pct":
            100 * group.robust_improved.mean(),
        "median_base_PA_mm": group.pa_base.median(),
        "median_noise_mm": group.noise_mm.median(),
    })

append_group("ALL", z)

for task, g in z.groupby("task"):
    append_group("TASK", g, task=task)

# Global comparison by within-task quartile.
for (aq, vq), g in z.groupby(["acc_q", "vel_q"]):
    append_group(
        "ACC_X_VEL", g,
        acc_q=int(aq), vel_q=int(vq)
    )

# More diagnostic: within each task and acceleration bin.
for (task, aq, vq), g in z.groupby(
    ["task", "acc_q", "vel_q"]
):
    if len(g) >= 10:
        append_group(
            "TASK_ACC_X_VEL", g, task=task,
            acc_q=int(aq), vel_q=int(vq)
        )

s = pd.DataFrame(summary)
s.to_csv(OUT / "response_summary.csv", index=False)
z.to_csv(OUT / "strict_response_clips.csv", index=False)

print("\n===== OVERALL CORRECTION RESPONSE =====")
print(s[s.group == "ALL"].to_string(index=False))

print("\n===== VELOCITY × ACCELERATION =====")
cols = [
    "acc_q", "vel_q", "n",
    "median_benefit_mm",
    "ci95_low_mm", "ci95_high_mm",
    "improved_pct", "robust_improved_pct",
    "median_noise_mm"
]
print(
    s[s.group == "ACC_X_VEL"][cols]
    .sort_values(["acc_q", "vel_q"])
    .round(4).to_string(index=False)
)

print("\n===== TASKWISE =====")
print(
    s[s.group == "TASK"][
        ["task", "n", "median_benefit_mm",
         "improved_pct", "robust_improved_pct"]
    ].round(4).to_string(index=False)
)

print("\nSAVED:", OUT)
print("COMPLETE")
