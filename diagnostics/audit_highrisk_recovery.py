from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("diagnostics/results/action_conditioned_active_world_model")
GROOT = ROOT / "uncertainty_gate_5task"

TASKS = ["0000","0001","0006","0008","0010"]
RATES = [0.10, 0.20, 0.30]

# ------------------------------------------------------------------
# Base / alternative 2-view errors
# ------------------------------------------------------------------
data = pd.read_csv(GROOT / "gate_analysis_input.csv")
data["task"] = data["task"].astype(str).str.zfill(4)

# ------------------------------------------------------------------
# Actual HLR errors
# ------------------------------------------------------------------
hs = []
for t in TASKS:
    p = ROOT / f"fixed_camera_pair_task{t}" / f"task{t}_pair_HLR__per_sample.csv"
    h = pd.read_csv(p)

    metric = (
        "behavior/l2_moved/mean"
        if "behavior/l2_moved/mean" in h.columns
        else "l2_moved/mean"
    )

    h = h[["clip", metric]].copy()
    h["task"] = t
    h["hlr_error_mm"] = h[metric] * 1000.0
    hs.append(h[["task","clip","hlr_error_mm"]])

hlr = pd.concat(hs, ignore_index=True)

data = data.merge(
    hlr,
    on=["task","clip"],
    validate="one_to_one"
)

# ------------------------------------------------------------------
# Existing OOF risk scores
# ------------------------------------------------------------------
pred = pd.read_csv(GROOT / "gate_oof_predictions.csv")
pred["task"] = pred["task"].astype(str).str.zfill(4)

pred = pred[
    pred["model"].eq("RandomForest")
    & pred["features"].isin(["Action-only","Scene+Action"])
][["task","clip","features","predicted_risk"]]

print("="*100)
print("HIGH-RISK RECOVERY UPPER-BOUND AUDIT")
print("="*100)

rows = []

for feat in ["Action-only","Scene+Action"]:

    s = pred[pred["features"] == feat][
        ["task","clip","predicted_risk"]
    ]

    d = data.merge(
        s,
        on=["task","clip"],
        validate="one_to_one"
    )

    base = d["base_error_mm"].to_numpy()
    alt  = d["alt_error_mm"].to_numpy()
    h3   = d["hlr_error_mm"].to_numpy()
    risk = d["predicted_risk"].to_numpy()

    order = np.argsort(-risk)

    print(f"\n[{feat}]")
    print(f"Overall base = {base.mean():.4f} mm")

    for rate in RATES:

        k = int(round(rate * len(d)))
        idx = order[:k]

        # actual error among selected high-risk samples
        b = base[idx]
        a = alt[idx]
        c = h3[idx]

        # Upper bound if any available recovery could be chosen perfectly.
        oracle = np.minimum.reduce([b, a, c])

        def overall_after(replacement):
            final = base.copy()
            final[idx] = replacement
            return final.mean()

        alt_final = overall_after(a)
        h3_final = overall_after(c)
        oracle_final = overall_after(oracle)

        base_mean = base.mean()

        alt_gain = 100*(base_mean-alt_final)/base_mean
        h3_gain = 100*(base_mean-h3_final)/base_mean
        oracle_gain = 100*(base_mean-oracle_final)/base_mean

        alt_help = np.mean(a < b)
        h3_help = np.mean(c < b)
        any_help = np.mean(oracle < b)

        # Which source is oracle-best?
        stack = np.stack([b,a,c], axis=1)
        src = np.argmin(stack, axis=1)

        base_best = np.mean(src == 0)
        alt_best  = np.mean(src == 1)
        hlr_best  = np.mean(src == 2)

        print(
            f"rate={rate:.1f} | "
            f"selected base={b.mean():.3f} mm | "
            f"ALT gain={alt_gain:+.3f}% | "
            f"HLR gain={h3_gain:+.3f}% | "
            f"ORACLE recovery={oracle_gain:+.3f}% | "
            f"any recovery helps={any_help*100:.1f}%"
        )

        print(
            f"          oracle source: "
            f"keep-base={base_best*100:.1f}% | "
            f"alt-2view={alt_best*100:.1f}% | "
            f"HLR={hlr_best*100:.1f}%"
        )

        rows.append({
            "gate": feat,
            "risk_rate": rate,
            "selected_base_error_mm": b.mean(),
            "alt2view_gain_pct": alt_gain,
            "hlr_gain_pct": h3_gain,
            "oracle_recovery_gain_pct": oracle_gain,
            "alt_helps_fraction": alt_help,
            "hlr_helps_fraction": h3_help,
            "any_recovery_helps_fraction": any_help,
            "oracle_keep_base_fraction": base_best,
            "oracle_alt2view_fraction": alt_best,
            "oracle_hlr_fraction": hlr_best,
        })

out = ROOT / "reliability_router_final" / "highrisk_recovery_audit.csv"
pd.DataFrame(rows).to_csv(out, index=False)

print("\nOUTPUT =", out)
print("SUCCESS = True")
