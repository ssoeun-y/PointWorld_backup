import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODE = sys.argv[1]
R = Path("diagnostics/results")
O = R / "acc_boundary_failure_mechanisms_20261011"
O.mkdir(parents=True, exist_ok=True)
TASKS = ["0000", "0001", "0006", "0008", "0010"]

def med(x):
    return float(np.nanmedian(x)) if len(x) else np.nan

def corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 10:
        return np.nan
    return float(spearmanr(x[good], y[good]).statistic)

# Original predictions: actual Pred(t) vs GT(t) metrics.
files = [
    R / "object_pa_mpjpe_acc_5task_current10457"
    / f"task-{t}" / "per_frame_metrics.csv"
    for t in TASKS
]
raw = pd.concat([pd.read_csv(p) for p in files],
                ignore_index=True)
assert len(raw) == 139760
raw["frame"] = raw["frame"].astype(int)

f = raw.groupby(["task", "clip", "frame"], as_index=False).agg(
    acc=("rms_acceleration_si_m_s2", "first"),
    pa=("pa_mpjpe_mm", "mean"),
    mp=("mpjpe_mm", "mean"),
    npts=("valid_point_count", "sum"),
)
assert len(f) == 79470

p = f.pivot(index=["task", "clip"], columns="frame", values="pa")
n = f.pivot(index=["task", "clip"], columns="frame", values="npts")
a = f.groupby(["task", "clip"]).acc.first()
d = pd.DataFrame({
    "acc": a,
    "pa_first": p[1],
    "pa_last": p[10],
    "points_first": n[1],
}).reset_index()
d["pa_growth"] = d.pa_last - d.pa_first
assert len(d) == 7947

# Within-task acceleration quartiles.
d["acc_q"] = d.groupby("task")["acc"].transform(
    lambda x: pd.qcut(
        x.rank(method="first"), 4,
        labels=[1, 2, 3, 4]
    ).astype(int)
)

if MODE == "A":
    # Measured robot velocity is available for the successful
    # Boundary cohort; do not infer it from acceleration.
    root = R / "acc_causal_geometry_boundary_20261010"
    files = sorted(root.glob("task-*/mechanism_action_audit.csv"))
    if not files:
        raise RuntimeError("Missing boundary action audits")

    act = pd.concat(
        [pd.read_csv(p) for p in files],
        ignore_index=True
    )
    assert not act[["task", "clip"]].duplicated().any()

    z = d.merge(
        act[[
            "task", "clip",
            "original_acc_si",
            "original_speed_rms"
        ]],
        on=["task", "clip"],
        validate="one_to_one"
    )

    z["vel_q"] = z.groupby("task")["original_speed_rms"].transform(
        lambda x: pd.qcut(
            x.rank(method="first"), 4,
            labels=[1, 2, 3, 4]
        ).astype(int)
    )

    # Same task and acceleration band; compare speed strata.
    rows = []
    for (task, aq), g in z.groupby(["task", "acc_q"]):
        low = g[g.vel_q == 1]
        high = g[g.vel_q == 4]
        rows.append({
            "task": task,
            "acc_q": aq,
            "n": len(g),
            "n_low_speed": len(low),
            "n_high_speed": len(high),
            "low_speed_growth": med(low.pa_growth),
            "high_speed_growth": med(high.pa_growth),
            "high_minus_low_mm":
                med(high.pa_growth) - med(low.pa_growth),
            "acc_growth_rho": corr(g.acc, g.pa_growth),
            "vel_growth_rho":
                corr(g.original_speed_rms, g.pa_growth),
        })

    result = pd.DataFrame(rows)
    z.to_csv(O/"A_acc_velocity_clip_data.csv", index=False)
    result.to_csv(O/"A_acc_velocity_strata.csv", index=False)
    print("A — ACC × VELOCITY", flush=True)
    print("CLIPS:", len(z), "TASKS:", z.task.nunique())
    print(result.to_string(index=False), flush=True)
    print("NOTE: observational, velocity strata may have "
          "different acceleration distributions.")

elif MODE == "B":
    # Same initial PA removed to expose additional growth.
    x = f.merge(
        d[["task","clip","acc_q","pa_first"]],
        on=["task","clip"],
        validate="many_to_one"
    )
    x["pa_excess_since_first"] = x.pa - x.pa_first

    curves = x.groupby(
        ["task","acc_q","frame"], as_index=False
    ).agg(
        n=("clip","nunique"),
        median_pa=("pa","median"),
        median_excess=("pa_excess_since_first","median")
    )

    # Clip-level association of acceleration with growth,
    # stratified by initial error to reduce difficulty confounding.
    d["initial_pa_q"] = d.groupby("task")["pa_first"].transform(
        lambda s: pd.qcut(
            s.rank(method="first"), 4,
            labels=[1,2,3,4]
        ).astype(int)
    )
    summary = []
    for (task, q), g in d.groupby(["task", "initial_pa_q"]):
        summary.append({
            "task":task,
            "initial_pa_q":q,
            "n":len(g),
            "rho_acc_growth":corr(g.acc,g.pa_growth),
            "growth_median_mm":med(g.pa_growth),
        })

    curves.to_csv(O/"B_PA_frame_curves.csv", index=False)
    pd.DataFrame(summary).to_csv(
        O/"B_growth_at_matched_initial_difficulty.csv",index=False
    )
    print("B — TEMPORAL PA GROWTH", flush=True)
    print(pd.DataFrame(summary).to_string(index=False),
          flush=True)
    print("Saved frame curves; trajectory-level acceleration "
          "does not localize an acceleration spike in time.")

elif MODE == "C":
    # Deployable geometry feature: initial visible point count.
    # Initial PA is a confound check, NOT deployable.
    d["points_q"] = d.groupby("task")["points_first"].transform(
        lambda s: pd.qcut(
            s.rank(method="first"), 4,
            labels=[1,2,3,4]
        ).astype(int)
    )

    rows = []
    for (task, aq), g in d.groupby(["task", "acc_q"]):
        for pq in (1,4):
            h = g[g.points_q == pq]
            rows.append({
                "task":task,
                "acc_q":aq,
                "points_q":pq,
                "n":len(h),
                "point_count_median":med(h.points_first),
                "initial_pa_median":med(h.pa_first),
                "growth_median_mm":med(h.pa_growth),
                "final_pa_median_mm":med(h.pa_last),
            })

    result = pd.DataFrame(rows)
    result.to_csv(O/"C_geometry_acc_interaction.csv", index=False)
    d.to_csv(O/"C_geometry_clip_data.csv", index=False)
    print("C — INITIAL GEOMETRY × ACC", flush=True)
    print(result.to_string(index=False), flush=True)
    print("NOTE: point count is a visibility/coverage proxy, "
          "not a complete measure of geometric vulnerability.")

else:
    raise ValueError(MODE)

print("MECHANISM_" + MODE + "_COMPLETE", flush=True)
