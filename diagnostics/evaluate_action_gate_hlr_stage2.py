from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model"
)

GATE_ROOT = ROOT / "uncertainty_gate_5task"
TASKS = ["0000", "0001", "0006", "0008", "0010"]
EXPECTED = {"0000":383, "0001":2312, "0006":1855, "0008":1824, "0010":1614}
RATES = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

OUT = ROOT / "stage2_action_gate_hlr"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Existing episode-disjoint OOF Action-only gate
# ------------------------------------------------------------
gate = pd.read_csv(GATE_ROOT / "gate_oof_predictions.csv")

gate = gate[
    (gate["model"] == "RandomForest") &
    (gate["features"] == "Action-only")
].copy()

gate["task"] = gate["task"].astype(str).str.zfill(4)

# Must be one OOF prediction per sample
assert len(gate) == 7988
assert gate[["task","clip"]].duplicated().sum() == 0

# ------------------------------------------------------------
# 2. Load ACTUAL HLR 3-camera errors
# ------------------------------------------------------------
hlr_frames = []

print("=" * 100)
print("HLR AUDIT")
print("=" * 100)

for task in TASKS:
    f = (
        ROOT /
        f"fixed_camera_pair_task{task}" /
        f"task{task}_pair_HLR__per_sample.csv"
    )

    h = pd.read_csv(f)

    metric = (
        "behavior/l2_moved/mean"
        if "behavior/l2_moved/mean" in h.columns
        else "l2_moved/mean"
    )

    h = h[["clip", metric]].copy()
    h["task"] = task
    h["hlr_error_mm"] = h[metric] * 1000.0
    h = h[["task", "clip", "hlr_error_mm"]]

    print(
        f"task-{task}: HLR={len(h)} | "
        f"expected={EXPECTED[task]} | "
        f"{'OK' if len(h)==EXPECTED[task] else 'MISMATCH'}"
    )

    hlr_frames.append(h)

hlr = pd.concat(hlr_frames, ignore_index=True)

# ------------------------------------------------------------
# 3. Merge gate + HLR
# ------------------------------------------------------------
df = gate.merge(
    hlr,
    on=["task", "clip"],
    how="inner",
    validate="one_to_one",
)

print("-" * 100)
print("MERGED =", len(df))
assert len(df) == 7988

# Positive = going from current 2-view to HLR improves prediction.
df["hlr_benefit_mm"] = (
    df["base_error_mm"] - df["hlr_error_mm"]
)

base_mean = df["base_error_mm"].mean()
hlr_mean = df["hlr_error_mm"].mean()
mean_benefit = df["hlr_benefit_mm"].mean()
helps = (df["hlr_benefit_mm"] > 0).mean()

print("\n" + "=" * 100)
print("ACTUAL 2-VIEW -> 3-VIEW EFFECT")
print("=" * 100)
print(f"Always 2-view : {base_mean:.4f} mm")
print(f"Always 3-view : {hlr_mean:.4f} mm")
print(f"Mean benefit  : {mean_benefit:+.4f} mm")
print(
    f"Always-3 relative change vs 2-view: "
    f"{100*(base_mean-hlr_mean)/base_mean:+.3f}%"
)
print(f"HLR actually helps: {helps*100:.2f}% of samples")

# ------------------------------------------------------------
# 4. Query-rate curves
#
# IMPORTANT:
# queried sample ALWAYS uses actual HLR error.
# No min(base, HLR), no oracle fallback.
# ------------------------------------------------------------
rows = []

def eval_ranking(name, score, model=""):
    order = np.argsort(-np.asarray(score))

    base = df["base_error_mm"].to_numpy()
    hlr_err = df["hlr_error_mm"].to_numpy()
    benefit = df["hlr_benefit_mm"].to_numpy()

    for rate in RATES:
        k = int(round(rate * len(df)))

        final = base.copy()

        if k > 0:
            idx = order[:k]
            final[idx] = hlr_err[idx]

            positive_rate = np.mean(benefit[idx] > 0)
            selected_benefit = benefit[idx].mean()
        else:
            positive_rate = np.nan
            selected_benefit = np.nan

        rows.append({
            "method": name,
            "model": model,
            "query_rate": rate,
            "query_count": k,
            "final_moved_l2_mm": final.mean(),
            "improvement_vs_2view_pct":
                100.0 * (base.mean() - final.mean()) / base.mean(),
            "queried_positive_benefit_fraction": positive_rate,
            "mean_selected_benefit_mm": selected_benefit,
        })

# Ours: real OOF Action gate
eval_ranking(
    "Action Gate",
    df["predicted_risk"].to_numpy(),
    "RandomForest Action-only",
)

# Perfect knowledge of current error, but NOT HLR outcome
eval_ranking(
    "True-Risk Oracle",
    df["base_error_mm"].to_numpy(),
)

# Absolute upper bound: knows actual 3-view benefit
eval_ranking(
    "Benefit Oracle",
    df["hlr_benefit_mm"].to_numpy(),
)

# Expected random querying
for rate in RATES:
    expected = base_mean - rate * mean_benefit

    rows.append({
        "method": "Random Expected",
        "model": "",
        "query_rate": rate,
        "query_count": int(round(rate * len(df))),
        "final_moved_l2_mm": expected,
        "improvement_vs_2view_pct":
            100.0 * (base_mean - expected) / base_mean,
        "queried_positive_benefit_fraction": helps,
        "mean_selected_benefit_mm": mean_benefit,
    })

curve = pd.DataFrame(rows)

print("\n" + "=" * 100)
print("STAGE 2: ACTUAL HLR QUERY-RATE CURVE")
print("=" * 100)

print(
    curve[
        [
            "method",
            "query_rate",
            "final_moved_l2_mm",
            "improvement_vs_2view_pct",
            "queried_positive_benefit_fraction",
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 5. Taskwise Action Gate results
# ------------------------------------------------------------
task_rows = []

print("\n" + "=" * 100)
print("TASKWISE ACTION-GATE RESULTS")
print("=" * 100)

for task in TASKS:
    g = df[df["task"] == task].copy()

    base = g["base_error_mm"].to_numpy()
    h3 = g["hlr_error_mm"].to_numpy()
    score = g["predicted_risk"].to_numpy()

    order = np.argsort(-score)

    for rate in [0.10, 0.20, 0.30]:
        k = max(1, int(round(rate * len(g))))

        final = base.copy()
        idx = order[:k]
        final[idx] = h3[idx]

        gain = 100 * (base.mean() - final.mean()) / base.mean()

        task_rows.append({
            "task": task,
            "query_rate": rate,
            "base_2view_mm": base.mean(),
            "action_gate_final_mm": final.mean(),
            "gain_pct": gain,
            "positive_hlr_fraction_selected":
                np.mean((base[idx] - h3[idx]) > 0),
        })

task_df = pd.DataFrame(task_rows)

print(task_df.to_string(index=False))

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
df.to_csv(
    OUT / "stage2_per_sample.csv",
    index=False,
)

curve.to_csv(
    OUT / "stage2_query_rate_curve.csv",
    index=False,
)

task_df.to_csv(
    OUT / "stage2_taskwise_action_gate.csv",
    index=False,
)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
