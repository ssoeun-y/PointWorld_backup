from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.stats import spearmanr, rankdata, pearsonr

FRAME_CSV = Path(
    "diagnostics/results/framewise_acc_pa_5task_current10457/"
    "combined_one_row_per_clip_frame.csv"
)

ACC_CSV = Path(
    "diagnostics/results/current_behavior_established_acc_10457/"
    "per_sample_current_established_acc_10457.csv"
)

OUT = Path(
    "diagnostics/results/action_frequency_vs_pa_current10457"
)
OUT.mkdir(parents=True, exist_ok=True)


def pick(df, candidates, label):
    lower = {c.lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in lower:
            return lower[x.lower()]

    print(f"\n[{label}] 후보를 못 찾았습니다.")
    print("columns:")
    print(list(df.columns))
    raise RuntimeError(label)


def pick_contains(df, required, label):
    cand = []
    for c in df.columns:
        s = c.lower()
        if all(x in s for x in required):
            if pd.api.types.is_numeric_dtype(df[c]):
                cand.append(c)

    if not cand:
        print(f"\n[{label}] 후보를 못 찾았습니다.")
        print(list(df.columns))
        raise RuntimeError(label)

    # SI가 명시된 것을 우선
    cand.sort(
        key=lambda x: (
            "si" not in x.lower(),
            "mps" not in x.lower(),
            len(x)
        )
    )
    return cand[0]


frame = pd.read_csv(FRAME_CSV)
accdf = pd.read_csv(ACC_CSV)

print("======================================================")
print("FRAME CSV columns")
print(list(frame.columns))
print("\nACC CSV columns")
print(list(accdf.columns))
print("======================================================")

task_f = pick(
    frame,
    ["task", "task_name"],
    "frame task"
)

key_f = pick(
    frame,
    [
        "key",
        "sample_key",
        "clip_key",
        "clip_id",
        "sample_id"
    ],
    "frame clip key"
)

time_f = pick(
    frame,
    [
        "frame",
        "frame_idx",
        "future_frame",
        "future_idx",
        "t",
        "time_idx"
    ],
    "frame index"
)

pa_f = None
for c in frame.columns:
    s = c.lower()
    if (
        "pa" in s
        and ("mpjpe" in s or "mm" in s)
        and pd.api.types.is_numeric_dtype(frame[c])
    ):
        pa_f = c
        break

if pa_f is None:
    pa_f = pick_contains(
        frame,
        ["pa"],
        "PA-MPJPE"
    )

# frame-wise Acc scalar 찾기
acc_f = None

preferred_acc = [
    "acc",
    "acc_si",
    "acc_mag",
    "acc_magnitude",
    "frame_acc",
    "frame_acc_si",
    "acceleration",
    "acceleration_si",
    "acc_mps2",
    "acceleration_mps2",
]

lower = {c.lower(): c for c in frame.columns}
for c in preferred_acc:
    if c.lower() in lower:
        cc = lower[c.lower()]
        if pd.api.types.is_numeric_dtype(frame[cc]):
            acc_f = cc
            break

if acc_f is None:
    candidates = []
    for c in frame.columns:
        s = c.lower()
        if (
            "acc" in s
            and "rms" not in s
            and "history" not in s
            and "past" not in s
            and pd.api.types.is_numeric_dtype(frame[c])
        ):
            candidates.append(c)

    if len(candidates) == 1:
        acc_f = candidates[0]
    else:
        print("\nframe-wise Acc column을 자동 확정하지 못했습니다.")
        print("Acc 후보:", candidates)
        print("전체 columns:", list(frame.columns))
        raise RuntimeError("frame-wise Acc column")

task_a = pick(
    accdf,
    ["task", "task_name"],
    "established task"
)

key_a = pick(
    accdf,
    [
        "key",
        "sample_key",
        "clip_key",
        "clip_id",
        "sample_id"
    ],
    "established clip key"
)

# established RMS Acc
rms_candidates = []
for c in accdf.columns:
    s = c.lower()
    if (
        "rms" in s
        and "acc" in s
        and pd.api.types.is_numeric_dtype(accdf[c])
    ):
        rms_candidates.append(c)

if not rms_candidates:
    raise RuntimeError(
        "Established RMS Acc column을 찾지 못함: "
        + str(list(accdf.columns))
    )

rms_candidates.sort(
    key=lambda x: (
        "si" not in x.lower(),
        "mps" not in x.lower(),
        len(x)
    )
)
rms_a = rms_candidates[0]

print("\n===== AUTO-DETECTED =====")
print("task(frame) =", task_f)
print("key(frame)  =", key_f)
print("frame       =", time_f)
print("frame Acc   =", acc_f)
print("PA          =", pa_f)
print("task(acc)   =", task_a)
print("key(acc)    =", key_a)
print("RMS Acc     =", rms_a)
print("=========================\n")

frame = frame[
    [task_f, key_f, time_f, acc_f, pa_f]
].copy()

frame.columns = [
    "task",
    "key",
    "frame",
    "acc_t",
    "pa_mm"
]

accsmall = accdf[
    [task_a, key_a, rms_a]
].copy()

accsmall.columns = [
    "task",
    "key",
    "rms_acc"
]

frame["task"] = frame["task"].astype(str)
frame["key"] = frame["key"].astype(str)

accsmall["task"] = accsmall["task"].astype(str)
accsmall["key"] = accsmall["key"].astype(str)

frame["acc_t"] = pd.to_numeric(
    frame["acc_t"], errors="coerce"
)
frame["pa_mm"] = pd.to_numeric(
    frame["pa_mm"], errors="coerce"
)

# 각 clip에서 실제 Acc 값이 있는 길이 확인
counts = (
    frame.dropna(subset=["acc_t"])
    .groupby(["task", "key"])
    .size()
)

mode_n = int(counts.mode().iloc[0])

print("Most common valid Acc-series length =", mode_n)
print(counts.value_counts().sort_index())

rows = []

for (task, key), g in frame.groupby(
    ["task", "key"],
    sort=False
):
    g = g.sort_values("frame")

    pa = g["pa_mm"].to_numpy(dtype=float)
    tt = g["frame"].to_numpy(dtype=float)

    good_pa = np.isfinite(pa) & np.isfinite(tt)

    if good_pa.sum() < 5:
        continue

    # PA rollout slope
    slope = np.polyfit(
        tt[good_pa],
        pa[good_pa],
        1
    )[0]

    # early PA control
    gp = g.loc[good_pa].sort_values("frame")
    early_pa = float(
        gp["pa_mm"].iloc[:3].mean()
    )

    a = g["acc_t"].to_numpy(dtype=float)
    a = a[np.isfinite(a)]

    # 동일 길이 population만 사용
    if len(a) != mode_n:
        continue

    # 시간 변화 '형태'를 보기 위해 mean 제거
    centered = a - np.mean(a)

    C = dct(
        centered,
        type=2,
        norm="ortho"
    )

    E = C ** 2

    # k=0은 mean 제거 때문에 사실상 DC
    Ef = E[1:]

    total = float(Ef.sum())

    if not np.isfinite(total) or total <= 1e-16:
        continue

    frac = Ef / total

    k = np.arange(1, len(C), dtype=float)

    spectral_centroid = float(
        np.sum(k * Ef)
        / np.sum(Ef)
        / k.max()
    )

    r = {
        "task": task,
        "key": key,
        "pa_slope_mm_per_frame": slope,
        "early_pa_mm": early_pa,
        "spectral_centroid": spectral_centroid,
        "acc_series_n": len(a),
    }

    for j, v in enumerate(frac, start=1):
        r[f"dct_energy_frac_k{j}"] = float(v)

    rows.append(r)

feat = pd.DataFrame(rows)

feat = feat.merge(
    accsmall,
    on=["task", "key"],
    how="inner",
    validate="one_to_one"
)

print("\nMatched clips =", len(feat))
print(feat["task"].value_counts().sort_index())

feat.to_csv(
    OUT / "per_clip_frequency_features.csv",
    index=False
)


def residualize_rank(y, controls):
    y = rankdata(y)

    Xs = []

    for c in controls:
        Xs.append(rankdata(c))

    X = np.column_stack(Xs)

    # intercept
    X = np.column_stack([
        np.ones(len(X)),
        X
    ])

    beta = np.linalg.lstsq(
        X, y, rcond=None
    )[0]

    return y - X @ beta


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    out = np.empty(n)
    out[order] = q
    return out


# task를 one-hot controls로 추가
task_dummy = pd.get_dummies(
    feat["task"],
    drop_first=True,
    dtype=float
)

features = [
    c for c in feat.columns
    if c.startswith("dct_energy_frac_k")
]

features += ["spectral_centroid"]

results = []

y = feat["pa_slope_mm_per_frame"].to_numpy(float)
rms = feat["rms_acc"].to_numpy(float)
early = feat["early_pa_mm"].to_numpy(float)

task_controls = [
    task_dummy[c].to_numpy(float)
    for c in task_dummy.columns
]

for f in features:
    x = feat[f].to_numpy(float)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(rms)
        & np.isfinite(early)
    )

    xx = x[mask]
    yy = y[mask]
    rr = rms[mask]
    ee = early[mask]

    tc = [
        z[mask]
        for z in task_controls
    ]

    rho_simple, p_simple = spearmanr(
        xx, yy
    )

    # 핵심:
    # task + established RMS Acc + early PA 통제
    rx = residualize_rank(
        xx,
        [rr, ee] + tc
    )

    ry = residualize_rank(
        yy,
        [rr, ee] + tc
    )

    rho_partial, p_partial = pearsonr(
        rx, ry
    )

    results.append({
        "feature": f,
        "N": len(xx),
        "spearman_rho": rho_simple,
        "spearman_p": p_simple,
        "partial_rho_task_rms_earlyPA":
            rho_partial,
        "partial_p":
            p_partial,
    })

res = pd.DataFrame(results)

res["partial_fdr_q"] = bh_fdr(
    res["partial_p"].to_numpy(float)
)

res["abs_partial_rho"] = (
    res["partial_rho_task_rms_earlyPA"]
    .abs()
)

res = res.sort_values(
    "abs_partial_rho",
    ascending=False
)

res.to_csv(
    OUT / "frequency_vs_pa_slope_summary.csv",
    index=False
)

print("\n")
print("======================================================")
print("FREQUENCY COMPOSITION -> PA ROLLOUT SLOPE")
print("Controls: task + established RMS Acc + early PA")
print("======================================================")

show_cols = [
    "feature",
    "N",
    "spearman_rho",
    "partial_rho_task_rms_earlyPA",
    "partial_p",
    "partial_fdr_q",
]

print(
    res[show_cols]
    .to_string(index=False)
)

print("\n===== Significant after FDR q<0.05 =====")

sig = res[
    res["partial_fdr_q"] < 0.05
]

if len(sig):
    print(
        sig[show_cols]
        .to_string(index=False)
    )
else:
    print("NONE")

print("\nOutputs:")
print(
    OUT / "per_clip_frequency_features.csv"
)
print(
    OUT / "frequency_vs_pa_slope_summary.csv"
)
