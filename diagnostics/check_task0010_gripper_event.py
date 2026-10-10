from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

METRIC = Path(
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)
FEAT = Path(
    "diagnostics/results/action_confound_control_5task/"
    "task0010_interaction_features_7987.csv"
)

m = pd.read_csv(METRIC)
f = pd.read_csv(FEAT)

df = m.merge(f, on="clip", validate="one_to_one")
df = df[df["task"] == "task-0010"].copy()

# numerical noise를 제외하고 실제 gripper 변화가 있었는지 판정
THR = 1e-6
df["gripper_event"] = (
    (df["dominant_gripper_range"] > THR) |
    (df["dominant_gripper_total_variation"] > THR)
)

no = df.loc[~df.gripper_event, "l2_moved_mm"].dropna().to_numpy()
yes = df.loc[df.gripper_event, "l2_moved_mm"].dropna().to_numpy()

u, p = mannwhitneyu(yes, no, alternative="two-sided")
cliff = 2 * u / (len(yes) * len(no)) - 1

print("=== TASK-0010 GRIPPER EVENT SPLIT ===")
print("No event N =", len(no))
print("Event N    =", len(yes))
print()
print("No-event moved-L2 median =", np.median(no), "mm")
print("Event moved-L2 median    =", np.median(yes), "mm")
print("Event / No-event ratio   =", np.median(yes) / np.median(no))
print("Median difference        =", np.median(yes) - np.median(no), "mm")
print("Mann-Whitney p           =", p)
print("Cliff's delta            =", cliff)
