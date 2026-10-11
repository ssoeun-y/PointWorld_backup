from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("diagnostics/results/vagg_20261011")
OUT = ROOT / "vagg_object_pa_comparison.csv"
TASKS = {"0000","0001","0006","0008","0010"}

rng = np.random.default_rng(20261011)
summary = []

for mode in ("vagg","ungated","speed_l2"):
    file = ROOT / mode / "object_pa_per_clip.csv"
    print("\n==========",mode,"==========")

    if not file.exists():
        print("OBJECT PA RESULT NOT READY")
        continue

    df = pd.read_csv(file, dtype={"task":str, "clip":str})
    df["task"] = df["task"].str.zfill(4)

    found = set(df["task"].unique())
    print("TASKS:",sorted(found))
    print("CLIPS:",len(df))

    if found != TASKS:
        print("WARNING: INCOMPLETE FIVE-TASK COVERAGE")
        continue

    if df.duplicated(["task","clip"]).any():
        raise RuntimeError(mode + ": DUPLICATED CLIPS")

    for task in sorted(TASKS)+["ALL"]:
        sub = df if task=="ALL" else df[df.task==task]

        for metric in (
            "pa_mpjpe_mm",
            "gt_l2_mm",
            "endpoint_mm"
        ):
            before = "base_"+metric
            after = "vagg_"+metric

            if before not in sub or after not in sub:
                raise RuntimeError(
                    f"MISSING METRIC: {mode} {metric}"
                )

            a = sub[[before,after]].replace(
                [np.inf,-np.inf],np.nan
            ).dropna()

            if len(a)==0:
                continue

            old = a[before].to_numpy(float)
            new = a[after].to_numpy(float)
            gain = old-new

            # Paired clip-level median difference
            med_gain = float(np.median(gain))

            # Relative change using medians
            old_med = float(np.median(old))
            new_med = float(np.median(new))
            improvement_pct = (
                100*(old_med-new_med)/old_med
                if old_med>0 else float("nan")
            )

            # Nonparametric bootstrap CI
            boots = []
            for _ in range(500):
                ix = rng.integers(0,len(gain),len(gain))
                boots.append(np.median(gain[ix]))
            ci = np.percentile(boots,[2.5,97.5])

            rec = {
                "mode":mode,
                "task":task,
                "metric":metric,
                "clips":len(a),
                "base_median_mm":old_med,
                "new_median_mm":new_med,
                "median_gain_mm":med_gain,
                "median_reduction_pct":improvement_pct,
                "improved_clips_pct":float(100*np.mean(gain>0)),
                "gain_ci95_low_mm":float(ci[0]),
                "gain_ci95_high_mm":float(ci[1])
            }
            summary.append(rec)

            print(
                f"{task:4s} {metric:14s} N={len(a):5d} "
                f"{old_med:.4f} -> {new_med:.4f} mm | "
                f"median paired gain={med_gain:+.4f} mm | "
                f"medians reduction={improvement_pct:+.2f}% | "
                f"better={100*np.mean(gain>0):.1f}% | "
                f"CI=[{ci[0]:+.4f},{ci[1]:+.4f}]"
            )

if summary:
    pd.DataFrame(summary).to_csv(OUT,index=False)
    print("\nSAVED:",OUT)
else:
    print("\nNO COMPLETE FIVE-TASK OBJECT PA RESULTS YET")
