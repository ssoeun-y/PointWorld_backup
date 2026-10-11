from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path("diagnostics/results/vagg_20261011")
KEY = ["task","clip","object_id","frame"]
MODES = ["vagg","speed_l2","ungated"]

PAIRS = {
    "pa": ("base_pa_mpjpe_mm","vagg_pa_mpjpe_mm"),
    "l2": ("base_gt_l2_mm","vagg_gt_l2_mm"),
    "endpoint": ("base_endpoint_mm","vagg_endpoint_mm"),
}

data = {}
rng = np.random.default_rng(20261011)

for mode in MODES:
    folder = ROOT / mode
    full = folder / "object_pa_per_frame.csv"
    partial = folder / "object_pa_partial_frames.csv"

    source = full if full.is_file() else partial
    if not source.is_file():
        print("NOT_READY", mode)
        continue

    d = pd.read_csv(
        source, dtype={"task":str,"clip":str}
    )
    d["task"] = d.task.str.zfill(4)
    d["object_id"] = d.object_id.astype(str)

    required = KEY + [
        x for pair in PAIRS.values() for x in pair
    ]
    missing = set(required)-set(d.columns)
    if missing:
        raise RuntimeError((mode,"MISSING",missing))

    if d.duplicated(KEY).any():
        raise RuntimeError((mode,"DUPLICATED OBJECT FRAME"))

    d = d[required]
    data[mode] = d

    print(
        "LOADED",mode,
        "FINAL" if source==full else "PARTIAL",
        "object-frames",len(d),
        "clips",d[["task","clip"]].drop_duplicates().shape[0],
        "tasks",sorted(d.task.unique())
    )

def clip_metrics(d):
    numeric = [x for pair in PAIRS.values() for x in pair]
    f = d.groupby(
        ["task","clip","frame"],as_index=False
    )[numeric].mean()
    return f.groupby(
        ["task","clip"],as_index=False
    )[numeric].mean()

def clustered_ci(clip, gain):
    keys = clip["clip"].astype(str).str.extract(
        r"(task-\d{4}_episode_\d+)"
    )[0].fillna(clip["clip"].astype(str))
    groups = [
        np.flatnonzero(keys.to_numpy()==k)
        for k in keys.unique()
    ]
    if len(groups)<8:
        return [float("nan")]*2
    boots = []
    for _ in range(300):
        draw = rng.integers(0,len(groups),len(groups))
        ix = np.concatenate([groups[j] for j in draw])
        boots.append(np.median(gain[ix]))
    return np.percentile(boots,[2.5,97.5])

if "vagg" in data:
    print("\n=== POINTWORLD vs VAGG ===")
    d = clip_metrics(data["vagg"])
    for task in sorted(d.task.unique())+["ALL"]:
        a = d if task=="ALL" else d[d.task==task]
        print("\nTASK",task,"CLIPS",len(a))
        for name,(old,new) in PAIRS.items():
            v=a[[old,new]].dropna()
            if len(v)==0:
                continue
            x=v[old].to_numpy()
            y=v[new].to_numpy()
            pct=100*(np.median(x)-np.median(y))/max(
                np.median(x),1e-12
            )
            print(
                name,
                f"{np.median(x):.5f} -> {np.median(y):.5f} mm",
                f"median-gain={np.median(x-y):+.5f} mm",
                f"median-reduction={pct:+.2f}%",
                f"improved={100*np.mean(x>y):.2f}%"
            )

for control,label in [
    ("speed_l2","GEOMETRY SUPERVISION"),
    ("ungated","VELOCITY CONDITIONING")
]:
    if "vagg" not in data or control not in data:
        continue

    print("\n===",label,"VAGG vs",control,"===")

    d = data["vagg"].merge(
        data[control],on=KEY,
        how="inner",validate="one_to_one",
        suffixes=("_v","_c")
    )

    coverage = len(d)/max(
        len(data["vagg"]),len(data[control]),1
    )
    print(
        "EXACT MATCHED OBJECT-FRAMES",
        len(d),"coverage",round(coverage,4)
    )

    if coverage < .95:
        print(
            "WARNING: MATCH COVERAGE BELOW 95%; "
            "DO NOT DRAW CAUSAL CONCLUSIONS"
        )
        continue

    numeric = [
        name+"_"+suffix
        for pair in PAIRS.values()
        for name in pair
        for suffix in ("v","c")
    ]
    cf=d.groupby(
        ["task","clip","frame"],as_index=False
    )[numeric].mean()
    clips=cf.groupby(
        ["task","clip"],as_index=False
    )[numeric].mean()

    for task in sorted(clips.task.unique())+["ALL"]:
        a=clips if task=="ALL" else clips[
            clips.task==task
        ]
        print("\nTASK",task,"MATCHED_CLIPS",len(a))

        for metric,(old,new) in PAIRS.items():
            c=a[[
                old+"_v",new+"_v",
                old+"_c",new+"_c"
            ]].dropna()
            if not len(c):
                continue

            # Positive value = VAGG improves more than control.
            gain=(
                c[old+"_v"].to_numpy()
                - c[new+"_v"].to_numpy()
                - c[old+"_c"].to_numpy()
                + c[new+"_c"].to_numpy()
            )

            baseline_gap=np.abs(
                c[old+"_v"].to_numpy()
                -c[old+"_c"].to_numpy()
            )
            interval=clustered_ci(
                a.loc[c.index,["clip"]],gain
            )

            print(
                metric,
                f"VAGG-extra-gain={np.median(gain):+.5f} mm",
                f"better-than-control={100*np.mean(gain>0):.2f}%",
                f"episode-bootstrap-CI=[{interval[0]:+.5f},"
                f"{interval[1]:+.5f}]",
                f"baseline-mismatch-median={np.median(baseline_gap):.5f}"
            )

print("\nIMPORTANT: mesh-level object IDs; audit against")
print("the official rigid-object grouping before paper claims.")
print("ABLATION_ANALYSIS_COMPLETE")
