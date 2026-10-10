from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import traceback

R = Path("diagnostics/results")
O = R / "cvpr_paperpack_20261010_v2"
O.mkdir(parents=True, exist_ok=True)
TASKS = ("0000", "0001", "0006", "0008", "0010")
status = []

def load(path):
    d = pd.read_csv(path)
    if "task" in d.columns:
        d["task"] = d.task.astype(str).str.extract(
            r"(\d{1,4})$", expand=False
        ).str.zfill(4)
    if "clip" in d.columns:
        d["clip"] = d["clip"].astype(str)
    return d

def save(d, name):
    d.to_csv(O / name, index=False)
    print("[TABLE]", name, len(d), flush=True)

def figure(fig, name):
    fig.tight_layout()
    fig.savefig(O / (name + ".png"), dpi=300)
    fig.savefig(O / (name + ".pdf"))
    plt.close(fig)
    print("[FIGURE]", name, flush=True)

def execute(name, fn):
    try:
        fn()
        status.append([name, "OK", ""])
    except Exception as e:
        traceback.print_exc()
        status.append([name, "FAILED", repr(e)])

def main_boundary():
    d = load(
        R / "acc_only_boundary_pa_final_current10457"
        / "combined_5task_per_clip_metrics.csv"
    )
    assert len(d) == 2683, f"Expected 2683, got {len(d)}"
    assert not d.duplicated(["task", "clip"]).any()

    rows = []
    for name, z in [("ALL", d)] + [
        (t, d[d.task == t]) for t in TASKS
    ]:
        a = z.base_pa_mean_mm.to_numpy(float)
        b = z.corrected_pa_mean_mm.to_numpy(float)
        rows.append({
            "task": name,
            "N": len(z),
            "Original_PA_mm": np.median(a),
            "Ours_PA_mm": np.median(b),
            "Difference_of_medians_mm": np.median(b)-np.median(a),
            "Median_paired_gain_mm": np.median(a-b),
            "Better_pct": 100*np.mean(b<a),
        })
    table = pd.DataFrame(rows)
    save(table, "T1_main_PA_taskwise.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    vals = table.iloc[0][["Original_PA_mm", "Ours_PA_mm"]].to_numpy(float)
    bars = ax.bar(["Original", "Boundary"], vals)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.025,
                f"{v:.4f}", ha="center")
    ax.set_ylim(0, max(vals)*1.2)
    ax.set_ylabel("PA-MPJPE (mm)")
    ax.set_title("Matched Boundary Correction, N=2683")
    figure(fig, "F1_main_PA")

def dynamics_pa():
    root = R / "object_pa_mpjpe_acc_5task_current10457"
    clips, frames = [], []
    for task in TASKS:
        clips.append(load(root/f"task-{task}"/"per_clip_metrics.csv"))
        frames.append(load(root/f"task-{task}"/"per_frame_metrics.csv"))

    c = pd.concat(clips, ignore_index=True)
    f = pd.concat(frames, ignore_index=True)
    assert len(c) == 7947, f"Expected 7947, got {len(c)}"
    assert not c.duplicated(["task", "clip"]).any()

    acc = "rms_acceleration_si_m_s2"
    c["q1"] = c.groupby("task")[acc].transform(lambda x:x.quantile(.25))
    c["q3"] = c.groupby("task")[acc].transform(lambda x:x.quantile(.75))
    c["group"] = np.where(c[acc]<=c.q1, "Low25",
                  np.where(c[acc]>=c.q3, "High25", "Other"))
    selected = c[c.group!="Other"]

    bars = selected.groupby("group",as_index=False).agg(
        N=("clip","size"),
        PA_mm=("pa_mpjpe_objectmean_mm","median")
    )
    save(bars, "T2_lowhigh_PA.csv")

    # Multiple rigid objects -> one clip-frame first
    f = f.groupby(["task","clip","frame"],as_index=False).agg(
        PA_mm=("pa_mpjpe_mm","mean")
    )
    f = f.merge(selected[["task","clip","group"]],
                on=["task","clip"],validate="many_to_one")
    curves = f.groupby(["group","frame"],as_index=False).agg(
        median_PA_mm=("PA_mm","median")
    )
    save(curves, "T3_PA_frame_curves.csv")

    fig, ax = plt.subplots(1,2,figsize=(10,4))
    for g in ("Low25","High25"):
        v = bars[bars.group==g].PA_mm.iloc[0]
        z = curves[curves.group==g]
        ax[0].bar(g,v)
        ax[1].plot(z.frame,z.median_PA_mm,marker="o",label=g)
    ax[0].set_ylabel("PA-MPJPE (mm)")
    ax[0].set_title("(a) Dynamics")
    ax[1].set_xlabel("Future frame")
    ax[1].set_ylabel("PA-MPJPE (mm)")
    ax[1].set_title("(b) Error growth")
    ax[1].legend()
    figure(fig,"F2_lowhigh_PA")

def temporal():
    root = R/"temporal_consistency_5task_current10457"
    c = load(root/"combined_5task_clip_temporal_summary.csv")
    t = load(root/"combined_5task_clip_transition_metrics.csv")
    assert not c.duplicated(["task","clip"]).any()
    assert not t.duplicated(["task","clip","transition_end"]).any()

    acc = "rms_acceleration_si_m_s2"
    c["q1"] = c.groupby("task")[acc].transform(lambda x:x.quantile(.25))
    c["q3"] = c.groupby("task")[acc].transform(lambda x:x.quantile(.75))
    c["group"] = np.where(c[acc]<=c.q1,"Low25",
                  np.where(c[acc]>=c.q3,"High25","Other"))
    s = c[c.group!="Other"]

    b = s.groupby("group",as_index=False).agg(
        N=("clip","size"),
        temporal_PA_mm=("temporal_pa_median_mm","median"),
        GT_temporal_PA_mm=("gt_temporal_pa_median_mm","median")
    )
    save(b,"T4_temporal_lowhigh.csv")

    t = t.merge(s[["task","clip","group"]],
                on=["task","clip"],validate="many_to_one")
    curves = t.groupby(["group","transition_end"],as_index=False).agg(
        predicted_mm=("pred_temporal_pa_mpjpe_mm","median"),
        GT_mm=("gt_temporal_pa_mpjpe_mm","median")
    )
    save(curves,"T5_temporal_frame_curves.csv")

    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for g in ("Low25","High25"):
        v=b[b.group==g].temporal_PA_mm.iloc[0]
        z=curves[curves.group==g]
        ax[0].bar(g,v)
        ax[1].plot(z.transition_end,z.predicted_mm,
                   marker="o",label=g)
    ax[0].set_title("(a) Temporal PA")
    ax[0].set_ylabel("Temporal PA-MPJPE (mm)")
    ax[1].set_title("(b) Transition-wise consistency")
    ax[1].set_xlabel("Transition end frame")
    ax[1].set_ylabel("Temporal PA-MPJPE (mm)")
    ax[1].legend()
    figure(fig,"F3_temporal_PA")

    root=R/"boundary_temporal_consistency_current10457"
    save(load(root/"boundary_temporal_primary_summary.csv"),
         "T6_boundary_temporal_overall.csv")
    save(load(root/"taskwise_boundary_temporal_summary.csv"),
         "T7_boundary_temporal_taskwise.csv")

def droid():
    d=load(R/"droid_established_dynamics_1679"
           /"droid_rms_msj_filtered_l2_1679.csv")
    assert len(d)==1679
    d=d.dropna(subset=["site","rms_acceleration",
                       "filtered_l2_moved_mm"])
    pooled=spearmanr(d.rms_acceleration,d.filtered_l2_moved_mm)

    sites=d[d.groupby("site").clip.transform("size")>=8].copy()
    a=sites.groupby("site").rms_acceleration.rank(pct=True)
    b=sites.groupby("site").filtered_l2_moved_mm.rank(pct=True)
    controlled=spearmanr(a,b)

    save(pd.DataFrame([{
        "N":len(d),
        "Sites":d.site.nunique(),
        "Pooled_rho":pooled.statistic,
        "Within_site_rho":controlled.statistic,
        "Pooled_p":pooled.pvalue,
        "Within_site_p_descriptive":controlled.pvalue
    }]),"T8_DROID_site_control.csv")

def alignment_probe():
    a=load(R/"framewise_acc_pa_5task_current10457"
           /"combined_raw_frame_object_rows.csv")
    t=load(R/"temporal_consistency_5task_current10457"
           /"combined_5task_clip_transition_metrics.csv")

    spread=a.groupby(["task","clip","frame"]).local_acc_mps2.agg(
        ["min","max"])
    assert (spread["max"]-spread["min"]).max()<1e-4
    a=a.groupby(["task","clip","frame"],as_index=False).agg(
        local_acc_mps2=("local_acc_mps2","first"))

    rows=[]
    for lag in (0,1,2):
        z=t[["task","clip","transition_end",
             "pred_temporal_pa_mpjpe_mm"]].copy()
        z["frame"]=z.transition_end-lag
        z=z.merge(a,on=["task","clip","frame"],
                  validate="many_to_one")
        cors=[]
        for _,sub in z.groupby(["task","clip"]):
            if len(sub)<4 or sub.local_acc_mps2.nunique()<3:
                continue
            rho=spearmanr(sub.local_acc_mps2,
                          sub.pred_temporal_pa_mpjpe_mm).statistic
            if np.isfinite(rho):
                cors.append(rho)
        rows.append({
            "lag":lag,
            "aligned_rows":len(z),
            "N_clips":len(cors),
            "median_rho":np.median(cors) if cors else np.nan,
            "status":"EXPLORATORY - timestep audit required"
        })
    save(pd.DataFrame(rows),"AUDIT_acc_temporal_lags.csv")

def candidate_audit():
    base=load(R/"acc_only_boundary_pa_final_current10457"
              /"combined_5task_per_clip_metrics.csv")
    names=[
        "acc_boundary_phase_grasp_current10457",
        "acc_boundary_adaptive_contact_v6_current10457",
        "acc_boundary_rigid_support_v9_2_current10457",
        "acc_boundary_rigid_support_v9_3_current10457",
        "acc_boundary_rigid_support_v10_1_current10457",
    ]
    rows=[]
    for name in names:
        p=R/name/"combined_5task_main_pa_metrics.csv"
        if not p.exists():
            continue
        other=load(p)
        z=base.merge(other,on=["task","clip"],
                     suffixes=("_base","_alt"))
        rows.append({
            "candidate":name,
            "N_candidate":len(other),
            "N_common":len(z),
            "baseline_difference_mm":(
                np.median(abs(z.base_pa_mean_mm_base-
                              z.base_pa_mean_mm_alt))
                if len(z) else np.nan
            ),
            "status":"Protocol matching not yet verified"
        })
    save(pd.DataFrame(rows),"AUDIT_upgrade_candidates.csv")

for name,fn in [
    ("Main_PA",main_boundary),
    ("Dynamics_PA",dynamics_pa),
    ("Temporal_PA",temporal),
    ("DROID",droid),
    ("Frame_alignment",alignment_probe),
    ("Upgrade_candidates",candidate_audit)
]:
    execute(name,fn)

save(pd.DataFrame(status,columns=["section","status","error"]),
     "RUN_STATUS.csv")
print("[CVPR PAPERPACK FINISHED]",flush=True)
