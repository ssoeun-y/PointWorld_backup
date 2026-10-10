import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODE = sys.argv[1]
ROOT = Path("diagnostics/results")
OUT = ROOT / "acc_cause_parallel_20261010"
OUT.mkdir(parents=True, exist_ok=True)
TASKS = ["0000", "0001", "0006", "0008", "0010"]

def rho(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    return float(spearmanr(x[m], y[m]).statistic)

def report(df, x, y):
    for task, d in [("ALL", df), *list(df.groupby("task"))]:
        z = d[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        q1, q3 = z[x].quantile([.25, .75])
        lo = z.loc[z[x] <= q1, y]
        hi = z.loc[z[x] >= q3, y]
        print(
            task, "N", len(z),
            "rho", round(rho(z[x], z[y]), 5),
            "Low25", round(lo.median(), 6),
            "High25", round(hi.median(), 6),
            flush=True,
        )

def frames():
    d = pd.concat([
        pd.read_csv(
            ROOT / "object_pa_mpjpe_acc_5task_current10457"
            / f"task-{t}" / "per_frame_metrics.csv"
        )
        for t in TASKS
    ], ignore_index=True)
    assert len(d) == 139760, len(d)
    d["frame"] = pd.to_numeric(d["frame"]).astype(int)
    return d

print("MODE", MODE, flush=True)

if MODE == "A":
    d = frames()
    f = d.groupby(["task", "clip", "frame"], as_index=False).agg(
        acc=("rms_acceleration_si_m_s2", "first"),
        pa=("pa_mpjpe_mm", "mean"),
        mp=("mpjpe_mm", "mean"),
    )
    assert len(f) == 79470, len(f)
    p = f.pivot(index=["task", "clip"], columns="frame", values="pa")
    m = f.pivot(index=["task", "clip"], columns="frame", values="mp")
    a = f.groupby(["task", "clip"])["acc"].first()
    z = pd.DataFrame({
        "acc": a, "pa_f1": p[1], "pa_f10": p[10],
        "mp_f1": m[1], "mp_f10": m[10],
    }).reset_index()
    z["pa_growth"] = z.pa_f10 - z.pa_f1
    z["mp_growth"] = z.mp_f10 - z.mp_f1
    assert len(z) == 7947, len(z)
    z.to_csv(OUT / "A_growth_7947.csv", index=False)
    print("PA GROWTH")
    report(z, "acc", "pa_growth")
    print("MP GROWTH")
    report(z, "acc", "mp_growth")
    print("INITIAL PA")
    report(z, "acc", "pa_f1")

elif MODE == "B":
    d = frames()
    s = pd.to_numeric(d.fitted_scale, errors="coerce")
    d = d.loc[np.isfinite(s) & (s > 0)].copy()
    d["scale_abslog"] = abs(np.log(d.fitted_scale))
    z = d.groupby(["task", "clip"], as_index=False).agg(
        acc=("rms_acceleration_si_m_s2", "first"),
        scale_abslog=("scale_abslog", "median"),
    )
    z.to_csv(OUT / "B_scale_by_clip.csv", index=False)
    print("SCALE DEVIATION (abs log fitted scale)")
    report(z, "acc", "scale_abslog")
    print("FRAME MEDIAN SCALE DEVIATION")
    print(d.groupby("frame").scale_abslog.median().to_string())

elif MODE == "C":
    p = (ROOT / "rigid_projection_repair_current10457"
         / "boundary_repair_joined_7947.csv")
    d = pd.read_csv(p)
    assert len(d) == 7947, len(d)
    d["acc"] = d["rms_acceleration_si_m_s2"]
    print("RIGID SE(3) RESIDUAL (mm)")
    report(d, "acc", "residual_mm")
    d[["task", "clip", "acc", "residual_mm"]].to_csv(
        OUT / "C_rigid_residual_7947.csv", index=False
    )

elif MODE == "D":
    p = (ROOT / "acc_shape_mechanism_20261010"
         / "strict2677_mechanism_joined.csv")
    d = pd.read_csv(p)
    assert len(d) == 2677, len(d)
    print("STRICT CORRECTION: BENEFIT vs INITIAL RESIDUAL")
    report(d, "residual_mm", "benefit_mm")
    print("STRICT CORRECTION: BENEFIT vs PA GROWTH")
    report(d, "pa_growth", "benefit_mm")
    if "base_repeat_absdiff_mm" in d:
        z = d.copy()
        z["robust_gain"] = (
            z.benefit_mm > 2 * z.base_repeat_absdiff_mm
        ).astype(float)
        print("ROBUST GAIN vs INITIAL RESIDUAL")
        report(z, "residual_mm", "robust_gain")
    print("Note: association only; not independent causality.")

else:
    raise ValueError(MODE)

print(f"CAUSE_{MODE}_COMPLETE", flush=True)
