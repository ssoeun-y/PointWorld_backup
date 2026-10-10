import numpy as np, pandas as pd
R = "diagnostics/results"
O = pd.read_csv(f"{R}/object_pa_mpjpe_acc_5task_current10457/task-0000/per_object_metrics.csv", low_memory=False)
O = O[(O["object_short_name"] == "radio_89") & (O["eligible_future_frames"].astype(float) == 10)
      & (O["final_object_point_count"].astype(float) >= 300)]
H = pd.read_csv(f"{R}/action_conditioned_active_world_model/fixed_camera_pair_task0000/task0000_pair_HL__per_sample.csv",
                usecols=["clip", "l2_moved/mean"])
H["l2"] = H["l2_moved/mean"] * 1000
df = O.merge(H[["clip", "l2"]], on="clip").rename(
    columns={"rms_acceleration_si_m_s2": "acc", "pa_mpjpe_mean_mm": "pa"}).reset_index(drop=True)
df["pct"] = df["acc"].rank(pct=True)
cho = pd.read_csv(f"{R}/problem_acc_levels/chosen_clips.csv")
mid_clip = cho.loc[cho["level"] == "MID", "clip"].iloc[0]
mid = df[df["clip"] == mid_clip].iloc[0]
b = df[(df["pct"] - 0.90).abs() <= 0.05]
mp, ml = b["pa"].median(), b["l2"].median()
c = b[(b["pa"] <= 1.6 * mp) & (b["l2"] <= 1.6 * ml) & (b["pa"] > mid["pa"]) & (b["l2"] > mid["l2"])]
print(f"HIGH band median PA={mp:.2f} L2={ml:.2f} | 후보 {len(c)}개 (MID PA={mid['pa']:.2f}, L2={mid['l2']:.2f})")
if len(c) == 0:
    raise SystemExit("후보 0개: 조건 완화 필요")
r = c.sort_values("pa", ascending=False).iloc[0]
print(f"HIGH 선택: {r['clip']} acc={r['acc']:.3f} PA={r['pa']:.3f} L2={r['l2']:.2f}")
cho.loc[cho["level"] == "HIGH", ["clip", "object_id", "rms_acc_si", "pa_mpjpe_mean_mm", "points"]] = [
    r["clip"], r["object_id"], r["acc"], r["pa"], r["final_object_point_count"]]
cho.to_csv(f"{R}/problem_acc_levels/chosen_clips.csv", index=False)
print(cho[["level", "clip", "rms_acc_si", "pa_mpjpe_mean_mm"]].to_string(index=False))
