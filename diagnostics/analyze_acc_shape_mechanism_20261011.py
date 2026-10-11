from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

R = Path("diagnostics/results")
SRC = R / "acc_causal_geometry_boundary_20261010"
OUT = R / "acc_shape_mechanism_final_20261011"
OUT.mkdir(parents=True, exist_ok=True)

B = 0.24580418408749
METRICS = [
    "pa_mpjpe_mm",
    "se3_residual_mm",
    "pairwise_error_mm",
]
rng = np.random.default_rng(20261011)

def median_ci(x, reps=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return (np.nan, np.nan)
    boot = []
    for _ in range(reps):
        boot.append(np.median(rng.choice(
            x, size=len(x), replace=True
        )))
    return tuple(np.percentile(boot, [2.5, 97.5]))

def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)

frames = []
audits = []

for task in ("0000", "0001", "0006", "0008", "0010"):
    root = SRC / f"task-{task}"
    gp = root / "mechanism_geometry_per_frame.csv"
    ap = root / "mechanism_action_audit.csv"
    if not gp.exists() or not ap.exists():
        raise FileNotFoundError(root)

    frames.append(pd.read_csv(gp))
    audits.append(pd.read_csv(ap))

G = pd.concat(frames, ignore_index=True)
A = pd.concat(audits, ignore_index=True)

KEY = ["task", "clip", "object_id", "frame"]
CLIP = ["task", "clip"]

assert not A[CLIP].duplicated().any()
assert set(G.variant.unique()) >= {
    "base", "base_repeat", "corrected"
}

# 동일 object / 동일 frame에서 비교.
P = G.pivot_table(
    index=KEY,
    columns="variant",
    values=METRICS,
    aggfunc="first",
)

P = P.dropna()
assert len(P) > 0

obj = pd.DataFrame({
    k: P.index.get_level_values(k).to_numpy()
    for k in KEY
})

for m in METRICS:
    obj[m + "_base"] = P[(m, "base")].to_numpy()
    obj[m + "_corr"] = P[(m, "corrected")].to_numpy()
    obj[m + "_repeat"] = P[(m, "base_repeat")].to_numpy()

    obj[m + "_benefit"] = (
        obj[m + "_base"] - obj[m + "_corr"]
    )
    obj[m + "_noise"] = abs(
        obj[m + "_base"] - obj[m + "_repeat"]
    )

# 객체 수가 많은 clip이 과도하게 가중되지 않도록
# clip-frame 내 객체 평균.
numeric = [
    c for c in obj.columns
    if c not in KEY
    and pd.api.types.is_numeric_dtype(obj[c])
]

F = obj.groupby(
    ["task", "clip", "frame"],
    as_index=False
)[numeric].mean()

F = F.merge(
    A, on=CLIP, validate="many_to_one"
)

F["reproduced"] = (
    F.saved_runtime_absdiff_si <= 1e-6
)

F["exact_valid"] = (
    F.reproduced
    & (F.modified_acc_si <= B)
)

F["tolerant_valid"] = (
    F.reproduced
    & (F.modified_acc_si <= B + 1e-6)
)

F.to_csv(
    OUT / "mechanism_clip_frame.csv",
    index=False
)

print("===== AUDIT =====")
print("ACTION CLIPS:", len(A))
print("OBJECT FRAME ROWS:", len(obj))
print("CLIP FRAME ROWS:", len(F))
print("EXACT CLIPS:",
      F[F.exact_valid][CLIP].drop_duplicates().shape[0])
print("TOLERANT CLIPS:",
      F[F.tolerant_valid][CLIP].drop_duplicates().shape[0])

for cohort, mask in (
    ("EXACT", F.exact_valid),
    ("TOLERANT", F.tolerant_valid),
):
    D = F[mask].copy()
    C = D.groupby(CLIP, as_index=False).agg({
        **{
            m + "_benefit": "mean"
            for m in METRICS
        },
        **{
            m + "_noise": "mean"
            for m in METRICS
        },
        "original_acc_si": "first",
        "modified_acc_si": "first",
        "original_speed_rms": "first",
        "modified_speed_rms": "first",
        "max_phase_shift_frames": "first",
    })

    C["acc_reduction_pct"] = 100 * (
        1 - C.modified_acc_si
        / C.original_acc_si.clip(lower=1e-9)
    )

    # 초기 구간(1-3)과 후기 구간(8-10)의
    # correction benefit 변화.
    early = D[D.frame.between(1,3)].groupby(CLIP)[
        [m + "_benefit" for m in METRICS]
    ].mean()
    late = D[D.frame.between(8,10)].groupby(CLIP)[
        [m + "_benefit" for m in METRICS]
    ].mean()

    delta = (late - early).add_suffix("_late_minus_early")
    C = C.merge(
        delta.reset_index(),
        on=CLIP,
        validate="one_to_one"
    )
    C.to_csv(
        OUT / f"{cohort.lower()}_clip_results.csv",
        index=False
    )

    print(f"\n===== {cohort}: N={len(C)} =====")

    for m in METRICS:
        benefit = C[m + "_benefit"].to_numpy()
        noise = C[m + "_noise"].to_numpy()
        lo, hi = median_ci(benefit)
        late_delta = C[
            m + "_benefit_late_minus_early"
        ].to_numpy()

        print(f"\n{m}")
        print("  median benefit mm:",
              round(np.median(benefit), 6))
        print("  bootstrap CI:",
              round(lo,6), round(hi,6))
        print("  improved %:",
              round(100*np.mean(benefit > 0),2))
        print("  above 2x repeat noise %:",
              round(100*np.mean(
                  benefit > 2*noise
              ),2))
        print("  late-minus-early benefit mm:",
              round(np.median(late_delta),6))
        print("  Wilcoxon p:",
              wilcoxon(benefit).pvalue)

    print("\n===== MECHANISM ASSOCIATIONS =====")

    pa = C["pa_mpjpe_mm_benefit"]
    se3 = C["se3_residual_mm_benefit"]
    pair = C["pairwise_error_mm_benefit"]

    print("PA vs SE3 benefit rho:", rho(pa,se3))
    print("PA vs Pairwise benefit rho:", rho(pa,pair))
    print("Acc reduction vs PA benefit rho:",
          rho(C.acc_reduction_pct,pa))
    print("Phase shift vs PA benefit rho:",
          rho(C.max_phase_shift_frames,pa))

    print("\n===== TASKWISE =====")
    for task, q in C.groupby("task"):
        p = q["pa_mpjpe_mm_benefit"]
        s = q["se3_residual_mm_benefit"]
        n = q["pa_mpjpe_mm_noise"]
        print(task, "N", len(q),
              "PA benefit", round(p.median(),5),
              "SE3 benefit", round(s.median(),5),
              "PA improve %",
              round(100*(p>0).mean(),2),
              "robust %",
              round(100*(p>2*n).mean(),2))

    print("\n===== PA FRAME CURVE =====")
    curve = D.groupby("frame",as_index=False).agg(
        n=("clip","nunique"),
        pa_benefit=("pa_mpjpe_mm_benefit","median"),
        se3_benefit=("se3_residual_mm_benefit","median"),
        pair_benefit=("pairwise_error_mm_benefit","median")
    )
    print(curve.to_string(index=False))
    curve.to_csv(
        OUT / f"{cohort.lower()}_frame_curve.csv",
        index=False
    )

print("\nMECHANISM ANALYSIS COMPLETE")
print("SAVED:", OUT)
