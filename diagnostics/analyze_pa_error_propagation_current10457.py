from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr, spearmanr

FRAME_CSV = Path(
    "diagnostics/results/framewise_acc_pa_5task_current10457/"
    "combined_one_row_per_clip_frame.csv"
)

ACC_CSV = Path(
    "diagnostics/results/current_behavior_established_acc_10457/"
    "per_sample_current_established_acc_10457.csv"
)

OUT = Path(
    "diagnostics/results/pa_error_propagation_current10457"
)
OUT.mkdir(parents=True, exist_ok=True)


def find_col(df, exacts=None, contains=None, label="col"):
    exacts = exacts or []
    lower = {c.lower(): c for c in df.columns}

    for x in exacts:
        if x.lower() in lower:
            return lower[x.lower()]

    if contains:
        cand = []
        for c in df.columns:
            s = c.lower()
            if all(x.lower() in s for x in contains):
                cand.append(c)
        if cand:
            return cand[0]

    raise RuntimeError(
        f"Cannot find {label}\ncolumns={list(df.columns)}"
    )


def residualize_rank(y, controls):
    y = rankdata(np.asarray(y, float))
    X = [np.ones(len(y))]

    for c in controls:
        c = np.asarray(c, float)
        if np.nanstd(c) > 0:
            X.append(rankdata(c))

    X = np.column_stack(X)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


frame = pd.read_csv(FRAME_CSV)
acc = pd.read_csv(ACC_CSV)

task_f = find_col(
    frame,
    exacts=["task", "task_name"],
    label="frame task"
)

key_f = find_col(
    frame,
    exacts=["clip", "key", "sample_key", "clip_key", "clip_id", "sample_id"],
    label="frame key"
)

time_f = find_col(
    frame,
    exacts=["frame", "frame_idx", "future_frame", "future_idx", "t"],
    label="frame index"
)

# PA column
pa_candidates = [
    c for c in frame.columns
    if "pa" in c.lower()
    and pd.api.types.is_numeric_dtype(frame[c])
]

if not pa_candidates:
    raise RuntimeError(
        "No PA column found:\n" + str(list(frame.columns))
    )

pa_candidates.sort(
    key=lambda c: (
        "mpjpe" not in c.lower(),
        "mm" not in c.lower(),
        len(c)
    )
)
pa_f = pa_candidates[0]

task_a = find_col(
    acc,
    exacts=["task", "task_name"],
    label="acc task"
)

key_a = find_col(
    acc,
    exacts=["clip", "key", "sample_key", "clip_key", "clip_id", "sample_id"],
    label="acc key"
)

rms_candidates = [
    c for c in acc.columns
    if "rms" in c.lower()
    and "acc" in c.lower()
    and pd.api.types.is_numeric_dtype(acc[c])
]

if not rms_candidates:
    raise RuntimeError(
        "No established RMS Acc column:\n" + str(list(acc.columns))
    )

rms_candidates.sort(
    key=lambda c: (
        "si" not in c.lower(),
        "mps" not in c.lower(),
        len(c)
    )
)
rms_a = rms_candidates[0]

print("===== DETECTED COLUMNS =====")
print("task:", task_f)
print("key :", key_f)
print("frame:", time_f)
print("PA  :", pa_f)
print("RMS :", rms_a)
print("============================")


d = frame[
    [task_f, key_f, time_f, pa_f]
].copy()

d.columns = [
    "task",
    "key",
    "frame",
    "pa_mm",
]

a = acc[
    [task_a, key_a, rms_a]
].copy()

a.columns = [
    "task",
    "key",
    "rms_acc",
]

for df in [d, a]:
    df["task"] = df["task"].astype(str)
    df["key"] = df["key"].astype(str)

d["frame"] = pd.to_numeric(
    d["frame"], errors="coerce"
)
d["pa_mm"] = pd.to_numeric(
    d["pa_mm"], errors="coerce"
)
a["rms_acc"] = pd.to_numeric(
    a["rms_acc"], errors="coerce"
)

d = d.dropna(
    subset=["frame", "pa_mm"]
)

# task/frame별 일반적인 PA 증가 추세 제거.
# 즉 단순히 "뒤 frame이라 PA가 큼"을 빼고
# 해당 clip이 같은 task/같은 frame의 다른 clip보다
# 유난히 더 틀어져 있는지를 본다.
tf_med = (
    d.groupby(["task", "frame"])["pa_mm"]
    .median()
    .rename("task_frame_pa_median")
    .reset_index()
)

d = d.merge(
    tf_med,
    on=["task", "frame"],
    how="left"
)

d["pa_excess"] = (
    d["pa_mm"]
    - d["task_frame_pa_median"]
)

d = d.merge(
    a,
    on=["task", "key"],
    how="inner"
)

print("\nRows:", len(d))
print("Clips:", d[["task", "key"]].drop_duplicates().shape[0])


# =====================================================
# 1. LAG PROPAGATION
# 현재 frame의 '평균보다 큰 shape error'가
# 1/2/3 frame 뒤에도 남아있는지 확인
# =====================================================

lag_results = []

for lag in [1, 2, 3, 4]:
    left = d[
        ["task", "key", "frame", "pa_excess", "rms_acc"]
    ].copy()

    right = d[
        ["task", "key", "frame", "pa_excess"]
    ].copy()

    right["frame"] = right["frame"] - lag
    right = right.rename(
        columns={"pa_excess": "future_pa_excess"}
    )

    m = left.merge(
        right,
        on=["task", "key", "frame"],
        how="inner"
    )

    m = m.replace([np.inf, -np.inf], np.nan).dropna()

    x = m["pa_excess"].to_numpy(float)
    y = m["future_pa_excess"].to_numpy(float)
    rms = m["rms_acc"].to_numpy(float)
    fr = m["frame"].to_numpy(float)

    task_dummy = pd.get_dummies(
        m["task"],
        drop_first=True,
        dtype=float
    )

    controls = [
        rms,
        fr
    ] + [
        task_dummy[c].to_numpy(float)
        for c in task_dummy.columns
    ]

    simple_rho, simple_p = spearmanr(x, y)

    rx = residualize_rank(x, controls)
    ry = residualize_rank(y, controls)

    pr, pp = pearsonr(rx, ry)

    lag_results.append({
        "lag_frames": lag,
        "N": len(m),
        "simple_spearman": simple_rho,
        "simple_p": simple_p,
        "partial_rank_rho_controlling_task_frame_rms": pr,
        "partial_p": pp,
    })


lag_df = pd.DataFrame(lag_results)

lag_df.to_csv(
    OUT / "lag_error_propagation.csv",
    index=False
)


# =====================================================
# 2. EARLY ERROR -> LATE ERROR
# 초반에 평균보다 더 틀어진 clip이
# 후반에도 평균보다 더 틀어지는지 확인
# =====================================================

rows = []

for (task, key), g in d.groupby(
    ["task", "key"],
    sort=False
):
    g = g.sort_values("frame")

    vals = g["pa_excess"].to_numpy(float)
    frames = g["frame"].to_numpy(float)

    if len(g) < 8:
        continue

    # 실제 frame 순서를 기준으로 3등분
    idx = np.argsort(frames)
    vals = vals[idx]

    n = len(vals)

    early = vals[:3]
    mid = vals[n//2-1:n//2+2]
    late = vals[-3:]

    if (
        len(early) < 3
        or len(mid) < 2
        or len(late) < 3
    ):
        continue

    rows.append({
        "task": task,
        "key": key,
        "rms_acc": float(g["rms_acc"].iloc[0]),
        "early_excess_pa": float(np.mean(early)),
        "mid_excess_pa": float(np.mean(mid)),
        "late_excess_pa": float(np.mean(late)),
    })

clip = pd.DataFrame(rows)

clip.to_csv(
    OUT / "per_clip_early_mid_late.csv",
    index=False
)

task_dummy = pd.get_dummies(
    clip["task"],
    drop_first=True,
    dtype=float
)

task_controls = [
    task_dummy[c].to_numpy(float)
    for c in task_dummy.columns
]

rms = clip["rms_acc"].to_numpy(float)

pairs = [
    ("early_excess_pa", "late_excess_pa"),
    ("mid_excess_pa", "late_excess_pa"),
]

summary = []

for xcol, ycol in pairs:
    x = clip[xcol].to_numpy(float)
    y = clip[ycol].to_numpy(float)

    simple_rho, simple_p = spearmanr(x, y)

    controls = [rms] + task_controls

    rx = residualize_rank(x, controls)
    ry = residualize_rank(y, controls)

    pr, pp = pearsonr(rx, ry)

    summary.append({
        "relation": f"{xcol} -> {ycol}",
        "N": len(clip),
        "simple_spearman": simple_rho,
        "simple_p": simple_p,
        "partial_rank_rho_controlling_task_rms": pr,
        "partial_p": pp,
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUT / "early_mid_to_late_summary.csv",
    index=False
)


print("\n=====================================================")
print("1. CURRENT ERROR -> FUTURE ERROR")
print("PA excess already removes task/frame median trend.")
print("Controls: task + frame + established RMS Acc")
print("=====================================================")

print(
    lag_df.to_string(index=False)
)

print("\n=====================================================")
print("2. EARLY/MID ERROR -> LATE ERROR")
print("Controls: task + established RMS Acc")
print("=====================================================")

print(
    summary_df.to_string(index=False)
)


# =====================================================
# 3. EASY EFFECT SIZE
# 같은 task 안에서 mid-rollout excess가
# 상위/하위 25%인 clip들의 late excess 비교
# =====================================================

effect_rows = []

for task, g in clip.groupby("task"):
    q25 = g["mid_excess_pa"].quantile(0.25)
    q75 = g["mid_excess_pa"].quantile(0.75)

    lo = g[g["mid_excess_pa"] <= q25]
    hi = g[g["mid_excess_pa"] >= q75]

    effect_rows.append({
        "task": task,
        "N_low": len(lo),
        "N_high": len(hi),
        "late_excess_low_median":
            float(lo["late_excess_pa"].median()),
        "late_excess_high_median":
            float(hi["late_excess_pa"].median()),
    })

effect = pd.DataFrame(effect_rows)

effect.to_csv(
    OUT / "mid_error_quartile_late_effect.csv",
    index=False
)

print("\n=====================================================")
print("3. MID-ROLLOUT ERROR QUARTILES -> LATE ERROR")
print("=====================================================")

print(
    effect.to_string(index=False)
)

print("\nSaved to:", OUT)
