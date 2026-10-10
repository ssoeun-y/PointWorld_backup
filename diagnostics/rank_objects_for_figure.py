import pandas as pd
from scipy.stats import spearmanr
R = "diagnostics/results/object_pa_mpjpe_acc_5task_current10457"
D = []
for t in ["task-0000", "task-0001", "task-0006", "task-0008", "task-0010"]:
    D.append(pd.read_csv(f"{R}/{t}/per_object_metrics.csv", low_memory=False))
d = pd.concat(D)
d = d[(d["eligible_future_frames"].astype(float) == 10) & (d["final_object_point_count"].astype(float) >= 300)]
rows = []
for (t, o), g in d.groupby(["task", "object_short_name"]):
    if len(g) < 60:
        continue
    rho = spearmanr(g["rms_acceleration_si_m_s2"], g["pa_mpjpe_mean_mm"])[0]
    rows.append((t, o, len(g), g["final_object_point_count"].astype(float).median(),
                 g["rms_acceleration_si_m_s2"].quantile(0.9), g["pa_mpjpe_mean_mm"].quantile(0.1),
                 g["pa_mpjpe_mean_mm"].quantile(0.9), rho))
out = pd.DataFrame(rows, columns=["task", "object", "n_clips", "pts_med", "acc_p90", "pa_p10", "pa_p90", "spearman"])
print(out.sort_values("spearman", ascending=False).round(3).to_string(index=False))
