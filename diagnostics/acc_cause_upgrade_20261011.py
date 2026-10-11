import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODE = sys.argv[1]
R = Path("diagnostics/results")
O = R / "acc_cause_upgrade_20261011"
O.mkdir(parents=True, exist_ok=True)

TASKS = ("0000", "0001", "0006", "0008", "0010")

def rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan
    return spearmanr(x[m], y[m]).statistic

if MODE == "competition":
    # Actual original Action speed and acceleration from
    # completed Boundary mechanism-audit tasks.
    root = R / "acc_causal_geometry_boundary_20261010"

    afiles = sorted(root.glob("task-*/mechanism_action_audit.csv"))
    gfiles = sorted(root.glob("task-*/mechanism_geometry_per_frame.csv"))

    if not afiles or len(afiles) != len(gfiles):
        raise RuntimeError("Boundary outputs incomplete or mismatched")

    a = pd.concat([pd.read_csv(p) for p in afiles], ignore_index=True)
    g = pd.concat([pd.read_csv(p) for p in gfiles], ignore_index=True)

    base = g[g.variant == "base"]

    early = base[base.frame == 1].groupby(
        ["task", "clip"]
    ).pa_mpjpe_mm.mean().rename("pa_f1")

    late = base[base.frame == 10].groupby(
        ["task", "clip"]
    ).pa_mpjpe_mm.mean().rename("pa_f10")

    d = a.drop_duplicates(["task", "clip"]).set_index(
        ["task", "clip"]
    ).join(early).join(late).reset_index()

    d["pa_growth"] = d.pa_f10 - d.pa_f1

    # Different targets: average error and subsequent growth.
    means = base.groupby(
        ["task", "clip"]
    ).pa_mpjpe_mm.mean().rename("pa_mean")

    d = d.set_index(["task", "clip"]).join(means).reset_index()

    features = {
        "RMS Acceleration": "original_acc_si",
        "RMS Velocity": "original_speed_rms",
    }

    print("COHORT:", len(d), "TASKS:", d.task.nunique())

    for target in ("pa_mean", "pa_growth"):
        print("\nTARGET:", target)
        for name, col in features.items():
            print("  ", name,
                  "rho:", round(rho(
                      d[col].to_numpy(float),
                      d[target].to_numpy(float)
                  ), 5))

        print("TASKWISE:")
        for task, z in d.groupby("task"):
            print(task, "N", len(z),
                  "Acc", round(rho(
                      z.original_acc_si.to_numpy(float),
                      z[target].to_numpy(float)
                  ), 5),
                  "Vel", round(rho(
                      z.original_speed_rms.to_numpy(float),
                      z[target].to_numpy(float)
                  ), 5))

    # Held-out feature ablation (predictive, not causal).
    from sklearn.model_selection import KFold
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[
        "original_acc_si", "original_speed_rms",
        "pa_growth", "pa_f1"
    ]).copy()

    task_features = pd.get_dummies(
        d.task, prefix="task", dtype=float
    )
    base_cols = task_features.copy()
    base_cols["initial_pa"] = d.pa_f1.values

    variants = {
        "TASK_INITIAL": base_cols,
        "+ACC": base_cols.assign(
            acc=d.original_acc_si.values
        ),
        "+VEL": base_cols.assign(
            vel=d.original_speed_rms.values
        ),
        "+ACC_VEL": base_cols.assign(
            acc=d.original_acc_si.values,
            vel=d.original_speed_rms.values
        ),
    }

    y = d.pa_growth.to_numpy(float)
    cv = KFold(5, shuffle=True, random_state=42)

    for name, xdf in variants.items():
        x = xdf.to_numpy(float)
        pred = np.full(len(y), np.nan)

        for tr, te in cv.split(x):
            imp = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            xx = scaler.fit_transform(
                imp.fit_transform(x[tr])
            )
            xt = scaler.transform(
                imp.transform(x[te])
            )
            model = Ridge(alpha=10)
            model.fit(xx, y[tr])
            pred[te] = model.predict(xt)

        rmse = np.sqrt(mean_squared_error(y, pred))
        print("HELDOUT", name, "RMSE(mm)", round(rmse, 6))

    d.to_csv(O / "competition_clip_table.csv", index=False)

elif MODE == "temporal":
    files = [
        R / "object_pa_mpjpe_acc_5task_current10457"
        / f"task-{t}" / "per_frame_metrics.csv"
        for t in TASKS
    ]

    d = pd.concat(
        [pd.read_csv(f) for f in files],
        ignore_index=True
    )

    assert len(d) == 139760

    # Object-frame averaged to clip-frame.
    f = d.groupby(
        ["task", "clip", "frame"], as_index=False
    ).agg(
        acc=("rms_acceleration_si_m_s2", "first"),
        pa=("pa_mpjpe_mm", "mean"),
        mp=("mpjpe_mm", "mean"),
    )

    assert len(f) == 79470

    acc = f.groupby(["task", "clip"]).acc.first()
    med = acc.groupby(level=0).transform("median")

    # Within-task comparison prevents task-mixture effects.
    f = f.join(
        (acc >= med).rename("high_acc"),
        on=["task", "clip"]
    )

    curves = f.groupby(
        ["task", "high_acc", "frame"],
        as_index=False
    ).agg(
        pa_median=("pa", "median"),
        mp_median=("mp", "median"),
        count=("clip", "nunique"),
    )

    print("CLIPS:", len(acc))
    print("TASKWISE LOW/HIGH ACC PA CURVES")
    print(curves.to_string(index=False))

    curves.to_csv(O / "temporal_curves.csv", index=False)

    p = f.pivot(
        index=["task", "clip"],
        columns="frame",
        values="pa"
    )
    growth = (p[10]-p[1]).rename("growth")
    z = pd.concat([acc.rename("acc"), growth], axis=1).dropna()

    print("\nPA GROWTH VS ACC")
    for task, q in z.reset_index().groupby("task"):
        print(task, len(q), "rho", round(
            rho(q.acc.to_numpy(), q.growth.to_numpy()), 5
        ))

    z.to_csv(O / "temporal_growth.csv")

else:
    raise ValueError(MODE)

print("COMPLETED", MODE, flush=True)
