from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = Path("diagnostics/results")
OUT = R / "acc_dose_response_final_20261011"
OUT.mkdir(parents=True, exist_ok=True)

METRIC = "pa_mpjpe_mm"
KEY = ["task", "clip", "object_id", "frame"]
CKEY = ["task", "clip"]
rng = np.random.default_rng(20261011)

all_doses = []

for dose in [25, 50, 75, 100]:
    folder = (
        f"acc_causal_geometry_dose{dose}_20261011"
        if dose < 100 else
        "acc_causal_geometry_boundary_20261010"
    )
    root = R / folder
    aa, gg = [], []

    for task in ["0000","0001","0006","0008","0010"]:
        td = root / f"task-{task}"
        aa.append(pd.read_csv(
            td / "mechanism_action_audit.csv"))
        gg.append(pd.read_csv(
            td / "mechanism_geometry_per_frame.csv"))

    a = pd.concat(aa, ignore_index=True)
    g = pd.concat(gg, ignore_index=True)
    assert not a[CKEY].duplicated().any()

    # Pair every variant at the same object and future frame.
    p = g.pivot_table(
        index=KEY, columns="variant",
        values=METRIC, aggfunc="first"
    )
    assert {"base","base_repeat","corrected"} <= set(p)
    p = p[["base","base_repeat","corrected"]].dropna()
    p = p.reset_index()

    # Equal weight per clip-frame, not per object count.
    f = p.groupby(CKEY+["frame"],as_index=False)[
        ["base","base_repeat","corrected"]
    ].mean()
    c = f.groupby(CKEY,as_index=False)[
        ["base","base_repeat","corrected"]
    ].mean()
    c["dose"] = dose
    c["benefit_mm"] = c.base - c.corrected
    c["repeat_noise_mm"] = abs(c.base-c.base_repeat)

    # Repeat noise must also be evaluated per-frame:
    # avoid cancellation when averaging signed differences.
    nf = f.assign(
        repeat_abs=(f.base-f.base_repeat).abs()
    ).groupby(CKEY,as_index=False).repeat_abs.mean()
    c = c.merge(nf,on=CKEY,validate="one_to_one")

    c = c.merge(
        a[CKEY+[
            "original_acc_si","modified_acc_si",
            "original_speed_rms","modified_speed_rms",
            "max_phase_shift_frames"
        ]],
        on=CKEY,validate="one_to_one"
    )
    c["acc_reduction_pct"] = 100*(
        1-c.modified_acc_si /
        c.original_acc_si.clip(lower=1e-9)
    )
    c["speed_change_pct"] = 100*(
        c.modified_speed_rms /
        c.original_speed_rms.clip(lower=1e-9)-1
    )
    all_doses.append(c)
    print("LOADED DOSE",dose,"CLIPS",len(c),flush=True)

d = pd.concat(all_doses,ignore_index=True)
assert d.groupby(CKEY).dose.nunique().eq(4).all()
assert d.groupby(CKEY).size().eq(4).all()

# Baseline fluctuations across separately evaluated doses.
w = d.pivot(index=CKEY,columns="dose",
            values=["base","corrected",
                    "benefit_mm","acc_reduction_pct"])

baseline_spread = (
    w["base"].max(axis=1)-w["base"].min(axis=1)
)
d = d.merge(
    baseline_spread.rename("baseline_spread_mm").reset_index(),
    on=CKEY,validate="many_to_one"
)

print("\nBASELINE ACROSS RUNS")
print("Median base spread mm:",
      baseline_spread.median())
print("90th percentile mm:",
      baseline_spread.quantile(.9))

# Dose effect must be compared using a common baseline:
# compare corrected PA directly within the matched clips.
wide = d.pivot(index=CKEY,columns="dose",
               values="corrected")
win = wide.idxmin(axis=1).rename("best_dose")
d = d.merge(win.reset_index(),on=CKEY,
            validate="many_to_one")
d.to_csv(OUT/"all_dose_clip_data.csv",index=False)

print("\n===== DOSE SUMMARY =====")
for dose,z in d.groupby("dose"):
    x = z.benefit_mm.to_numpy(float)
    ix = rng.integers(0,len(x),size=(1000,len(x)))
    ci = np.percentile(
        np.median(x[ix],axis=1),[2.5,97.5]
    )
    print(
        "DOSE",dose,"N",len(z),
        "actual_acc_reduction_median_pct",
        round(z.acc_reduction_pct.median(),3),
        "PA_benefit_median_mm",
        round(np.median(x),5),
        "CI95",np.round(ci,5),
        "improved_pct",round(100*(x>0).mean(),2),
        "above_noise_pct",
        round(100*(x>2*z.repeat_abs.to_numpy()).mean(),2)
    )

print("\n===== MATCHED CORRECTED PA =====")
for dose in [25,50,75,100]:
    print("DOSE",dose,
          "median corrected PA mm:",
          round(wide[dose].median(),6),
          "median paired corrected-minus-100 mm:",
          round((wide[dose]-wide[100]).median(),6))

print("\n===== BEST DOSE (DESCRIPTIVE ONLY) =====")
print(win.value_counts().sort_index().to_string())

for task,z in d.groupby("task"):
    wtask=z.pivot(index="clip",columns="dose",
                  values="corrected")
    print(
        "TASK",task,"N",len(wtask),
        "best dose counts",
        wtask.idxmin(axis=1).value_counts().to_dict()
    )

print("\nNOTE: best dose uses evaluated PA and is an "
      "oracle descriptive result, NOT deployable selection.")
print("NOTE: compare baseline spread and repeat noise before "
      "claiming meaningful differences between doses.")
print("COMPLETE:",OUT)
