from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "uncertainty_gate_5task/robustness"
)

FILE = ROOT / "loto_action_vs_sceneaction_predictions.csv"
N_BOOT = 2000
rng = np.random.default_rng(42)

df = pd.read_csv(FILE)
df["task"] = df["task"].astype(str).str.zfill(4)

TASKS = ["0000", "0001", "0006", "0008", "0010"]

def rho(y, p):
    return spearmanr(y, p).statistic

for base in ["HL", "HR"]:

    d = df[df["base_pair"] == base].copy()

    # ----------------------------------------------------------
    # Original macro LOTO difference
    # Δρ = Scene+Action - Action-only
    # ----------------------------------------------------------
    r_a, r_sa = [], []

    for task in TASKS:
        g = d[d["task"] == task]

        r_a.append(
            rho(
                g["true_error_mm"].to_numpy(),
                g["pred_Action-only"].to_numpy()
            )
        )

        r_sa.append(
            rho(
                g["true_error_mm"].to_numpy(),
                g["pred_Scene+Action"].to_numpy()
            )
        )

    orig_a = np.mean(r_a)
    orig_sa = np.mean(r_sa)
    orig_delta = orig_sa - orig_a

    # ----------------------------------------------------------
    # Prepare episode index arrays once
    # ----------------------------------------------------------
    prepared = {}

    for task in TASKS:

        g = d[d["task"] == task].reset_index(drop=True)

        episode_indices = [
            idx.to_numpy()
            for _, idx in g.groupby("group_id").groups.items()
        ]

        prepared[task] = {
            "y": g["true_error_mm"].to_numpy(),
            "a": g["pred_Action-only"].to_numpy(),
            "sa": g["pred_Scene+Action"].to_numpy(),
            "episodes": episode_indices,
        }

    boot = np.empty(N_BOOT)

    for b in range(N_BOOT):

        deltas = []

        for task in TASKS:

            z = prepared[task]
            eps = z["episodes"]

            sampled = rng.integers(
                0, len(eps), size=len(eps)
            )

            idx = np.concatenate([
                eps[i] for i in sampled
            ])

            y = z["y"][idx]
            a = z["a"][idx]
            sa = z["sa"][idx]

            deltas.append(
                rho(y, sa) - rho(y, a)
            )

        boot[b] = np.mean(deltas)

    lo, med, hi = np.percentile(
        boot, [2.5, 50, 97.5]
    )

    p = 2 * min(
        np.mean(boot <= 0),
        np.mean(boot >= 0)
    )
    p = min(p, 1.0)

    print("=" * 70)
    print(f"BASE = {base}")
    print("=" * 70)
    print(f"Action-only macro rho  : {orig_a:.4f}")
    print(f"Scene+Action macro rho : {orig_sa:.4f}")
    print(f"Delta rho (SA - A)     : {orig_delta:+.4f}")
    print(f"Bootstrap median       : {med:+.4f}")
    print(f"95% CI                 : [{lo:+.4f}, {hi:+.4f}]")
    print(f"Bootstrap p            : {p:.6g}")

    if hi < 0:
        print("RESULT: Action-only significantly better.")
    elif lo > 0:
        print("RESULT: Scene+Action significantly better.")
    else:
        print("RESULT: No significant difference.")

print("\nSUCCESS = True")
