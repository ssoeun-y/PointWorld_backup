from pathlib import Path
import numpy as np
import pandas as pd

R = Path("diagnostics/results")
SRC = R / "object_pa_mpjpe_acc_5task_current10457"
OUT = R / "acc_failure_pairs_20261011"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["0000", "0001", "0006", "0008", "0010"]

frames = []

for t in TASKS:
    p = SRC / f"task-{t}" / "per_frame_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    d = pd.read_csv(p)

    need = {
        "task", "clip", "frame",
        "pa_mpjpe_mm",
        "rms_acceleration_si_m_s2"
    }
    assert need <= set(d.columns), need - set(d.columns)

    frames.append(d)

raw = pd.concat(frames, ignore_index=True)
assert len(raw) == 139760, len(raw)

# 여러 물체가 있는 clip은 object 평균 후 frame 평균.
f = raw.groupby(
    ["task", "clip", "frame"], as_index=False
).agg(
    pa=("pa_mpjpe_mm", "mean"),
    acc=("rms_acceleration_si_m_s2", "first")
)

assert len(f) == 79470, len(f)

p = f.pivot(
    index=["task", "clip"],
    columns="frame",
    values="pa"
)

assert list(p.columns) == list(range(1, 11))

acc = f.groupby(["task", "clip"]).acc.first()

d = pd.DataFrame({
    "acc": acc,
    "pa_initial": p[1],
    "pa_final": p[10],
    "pa_mean": p.mean(axis=1),
    "pa_growth": p[10] - p[1],
}).reset_index()

assert len(d) == 7947
assert d[["task","clip"]].duplicated().sum() == 0

# 동일 Task 안에서 고가속도 clip 선별.
d["acc_q"] = d.groupby("task").acc.transform(
    lambda s: pd.qcut(
        s.rank(method="first"),
        4, labels=[1,2,3,4]
    ).astype(int)
)

pairs = []

for task, group in d.groupby("task"):
    high = group[group.acc_q == 4].copy()

    # 초기 예측 난이도까지 비슷하게 비교.
    high["initial_bin"] = pd.qcut(
        high.pa_initial.rank(method="first"),
        4, labels=False
    )

    candidates = []

    for ib, g in high.groupby("initial_bin"):
        g = g.sort_values("pa_growth")

        k = min(30, len(g) // 4)
        if k < 1:
            continue

        stable = g.head(k)
        failure = g.tail(k)

        # 실패 사례별로 가장 유사한 초기 PA / 가속도 찾기.
        for _, bad in failure.iterrows():
            for _, good in stable.iterrows():
                acc_gap = abs(
                    bad.acc-good.acc
                ) / max((bad.acc+good.acc)/2, 1e-9)

                pa_gap = abs(
                    bad.pa_initial-good.pa_initial
                ) / max(
                    (bad.pa_initial+good.pa_initial)/2,
                    0.1
                )

                growth_gap = bad.pa_growth-good.pa_growth

                if acc_gap > 0.10 or pa_gap > 0.20:
                    continue

                if growth_gap <= 0:
                    continue

                candidates.append({
                    "task": task,
                    "initial_bin": int(ib),
                    "failure_clip": bad.clip,
                    "stable_clip": good.clip,
                    "failure_acc": bad.acc,
                    "stable_acc": good.acc,
                    "acc_gap_pct": 100*acc_gap,
                    "initial_pa_gap_pct": 100*pa_gap,
                    "failure_initial_PA": bad.pa_initial,
                    "stable_initial_PA": good.pa_initial,
                    "failure_final_PA": bad.pa_final,
                    "stable_final_PA": good.pa_final,
                    "failure_PA_growth": bad.pa_growth,
                    "stable_PA_growth": good.pa_growth,
                    "growth_gap_mm": growth_gap,
                })

    if not candidates:
        print(task, "NO PAIRS under matching limits")
        continue

    c = pd.DataFrame(candidates)

    # 가장 큰 실패 차이를 우선하되 매칭 불량은 페널티.
    c["score"] = (
        c.growth_gap_mm
        - 0.02*c.acc_gap_pct
        - 0.01*c.initial_pa_gap_pct
    )
    c = c.sort_values("score", ascending=False)

    used_bad = set()
    used_good = set()
    selected = []

    for row in c.itertuples(index=False):
        if row.failure_clip in used_bad:
            continue
        if row.stable_clip in used_good:
            continue

        selected.append(row._asdict())
        used_bad.add(row.failure_clip)
        used_good.add(row.stable_clip)

        if len(selected) == 10:
            break

    pairs.extend(selected)
    print(task, "SELECTED:", len(selected))

d.to_csv(OUT / "all_7947_clip_metrics.csv", index=False)

result = pd.DataFrame(pairs)
if result.empty:
    raise RuntimeError(
        "No matched pairs; inspect cohort and matching limits"
    )

result.to_csv(
    OUT / "high_acc_failure_vs_stable_pairs.csv",
    index=False
)

print("\n===== MATCHED FAILURE vs STABLE =====")
print("PAIRS:", len(result))
print("TASKS:", result.task.nunique())
print("ACC GAP median %:",
      result.acc_gap_pct.median())
print("INITIAL PA GAP median %:",
      result.initial_pa_gap_pct.median())
print("GROWTH GAP median mm:",
      result.growth_gap_mm.median())

print("\n===== TASKWISE =====")
print(result.groupby("task").agg(
    n=("growth_gap_mm","size"),
    median_acc_gap_pct=("acc_gap_pct","median"),
    median_initial_pa_gap_pct=("initial_pa_gap_pct","median"),
    median_growth_gap_mm=("growth_gap_mm","median")
).round(4).to_string())

print("\nSAVED:", OUT)
print("SELECTION COMPLETE")
