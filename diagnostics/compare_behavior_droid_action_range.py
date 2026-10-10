from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

B = pd.read_csv(
    "diagnostics/results/action_quality_5task_moved_analysis.csv"
)
D = pd.read_csv(
    "diagnostics/results/droid_temporal_quality_final_1679.csv"
)

TEMP = ["speed_cv","accel_p95","jerk_p95","reversal_fraction"]

print("="*90)
print("1. ACTION RANGE: BEHAVIOR vs DROID")
print("="*90)

for c in TEMP:
    print(f"\n[{c}]")
    stats={}
    for name,df in [("BEHAVIOR",B),("DROID",D)]:
        x=df[c].to_numpy(float)
        x=x[np.isfinite(x)]
        q10,q25,q50,q75,q90=np.percentile(x,[10,25,50,75,90])
        width=q90-q10
        stats[name]=width
        print(
            f"{name:8s} "
            f"p10={q10:.6g} p25={q25:.6g} med={q50:.6g} "
            f"p75={q75:.6g} p90={q90:.6g} "
            f"p90-p10={width:.6g}"
        )
    print(
        "DROID/BEHAVIOR robust-range ratio = "
        f"{stats['DROID']/stats['BEHAVIOR']:.3f}x"
        if stats["BEHAVIOR"]>0 else "NA"
    )

# ---------------------------------------------------------
# 2. DROID site-controlled temporal score
# Same four primitives, but percentile ranks within each site.
# ---------------------------------------------------------
print("\n"+"="*90)
print("2. DROID WITHIN-SITE QUALITY")
print("="*90)

print("SITE COUNTS:")
print(D["site"].value_counts().to_string())

pct=[]
for c in TEMP:
    cc=f"{c}_site_pct"
    D[cc]=D.groupby("site")[c].rank(method="average",pct=True)
    pct.append(cc)

D["_raw_site"]=D[pct].mean(axis=1)
D["temporal_score_site"]=(
    D.groupby("site")["_raw_site"]
     .rank(method="average",pct=True)
)

hq=D["temporal_score_site"]<=.25
lq=D["temporal_score_site"]>=.75

H=D.loc[hq,"filtered_l2_moved_mm"]
L=D.loc[lq,"filtered_l2_moved_mm"]

rho,p=spearmanr(
    D["temporal_score_site"],
    D["filtered_l2_moved_mm"]
)

print(
    f"\nOVERALL within-site: "
    f"HQ={H.median():.4f} mm, "
    f"LQ={L.median():.4f} mm, "
    f"ratio={L.median()/H.median():.4f}x, "
    f"increase={(L.median()/H.median()-1)*100:.2f}%"
)
print(f"rho={rho:+.4f}, p={p:.3e}")

print("\nSITEWISE:")
for site,g in D.groupby("site"):
    h=g[g["temporal_score_site"]<=.25]["filtered_l2_moved_mm"]
    l=g[g["temporal_score_site"]>=.75]["filtered_l2_moved_mm"]
    if len(h) and len(l):
        print(
            f"{site:12s} N={len(g):4d} "
            f"HQ={h.median():.3f} "
            f"LQ={l.median():.3f} "
            f"ratio={l.median()/h.median():.3f}x"
        )

# ---------------------------------------------------------
# 3. Episode-level + site-controlled
# ---------------------------------------------------------
ep=(
    D.groupby(["site","episode"],as_index=False)
     .agg(
        score=("temporal_score_site","median"),
        err=("filtered_l2_moved_mm","median")
     )
)

ep["q"]=ep.groupby("site")["score"].rank(method="average",pct=True)

eh=ep["q"]<=.25
el=ep["q"]>=.75

EH=ep.loc[eh,"err"]
EL=ep.loc[el,"err"]

erho,epv=spearmanr(ep["q"],ep["err"])

print("\nEPISODE + SITE CONTROL:")
print(
    f"N={len(ep)}, "
    f"HQ={EH.median():.4f}, "
    f"LQ={EL.median():.4f}, "
    f"ratio={EL.median()/EH.median():.4f}x, "
    f"increase={(EL.median()/EH.median()-1)*100:.2f}%"
)
print(f"rho={erho:+.4f}, p={epv:.3e}")

print("\nSUCCESS = True")
