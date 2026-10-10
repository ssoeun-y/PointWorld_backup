#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
TASKS = ["task-0000", "task-0001", "task-0006", "task-0008", "task-0010"]
PA_ROOT = Path("diagnostics/results/object_pa_mpjpe_acc_5task_current10457")
OUT_DIR = Path("diagnostics/results/problem_intro_panels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUT_DIR / "figure1_problem_final.png"
PDF_OUT = OUT_DIR / "figure1_problem_final.pdf"
TXT_OUT = OUT_DIR / "figure1_problem_summary.txt"

# ------------------------------------------------------------
# Load per-frame PA data and aggregate object rows -> clip-frame mean
# ------------------------------------------------------------
def load_frame_pa():
    parts = []
    for task in TASKS:
        csv_path = PA_ROOT / task / "per_frame_metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing: {csv_path}")

        df = pd.read_csv(csv_path)
        need = {"task", "clip", "frame", "pa_mpjpe_mm"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{csv_path} missing columns: {miss}")

        # aggregate raw rigid-object rows -> one row per (task, clip, frame)
        g = (
            df.groupby(["task", "clip", "frame"], as_index=False)["pa_mpjpe_mm"]
              .mean()
        )
        parts.append(g)

    frame_df = pd.concat(parts, ignore_index=True)
    frame_df["frame"] = frame_df["frame"].astype(int)

    # sanity
    n_rows = len(frame_df)
    n_clips = frame_df[["task", "clip"]].drop_duplicates().shape[0]
    print(f"[PA] aggregated clip-frame rows = {n_rows}")
    print(f"[PA] unique clips = {n_clips}")
    print(f"[PA] frame range = {frame_df['frame'].min()}..{frame_df['frame'].max()}")

    return frame_df


# ------------------------------------------------------------
# Find established RMS Acc CSV(s)
# We look for csvs containing at least: clip, rms_acceleration_si_m_s2
# task may be in a column or inferred from the path.
# ------------------------------------------------------------
TASK_RE = re.compile(r"task-\d{4}")

def infer_task_from_path(path: Path):
    for p in path.parts:
        if TASK_RE.fullmatch(p):
            return p
    return None

def load_rms_acc():
    search_roots = [Path("stats"), Path("diagnostics/results")]
    acc_parts = []

    for root in search_roots:
        if not root.exists():
            continue
        for csv_path in root.rglob("*.csv"):
            try:
                head = pd.read_csv(csv_path, nrows=5)
            except Exception:
                continue

            cols = set(head.columns)
            if "clip" not in cols or "rms_acceleration_si_m_s2" not in cols:
                continue

            usecols = [c for c in ["task", "clip", "rms_acceleration_si_m_s2"] if c in head.columns]
            try:
                df = pd.read_csv(csv_path, usecols=usecols)
            except Exception:
                continue

            if "task" not in df.columns:
                inferred = infer_task_from_path(csv_path)
                if inferred is None:
                    continue
                df["task"] = inferred

            df = df[["task", "clip", "rms_acceleration_si_m_s2"]].copy()
            df = df[df["task"].isin(TASKS)].copy()

            if len(df) == 0:
                continue

            acc_parts.append(df)
            print(f"[ACC] found candidate: {csv_path}  rows={len(df)}")

    if not acc_parts:
        raise RuntimeError(
            "Could not find any RMS Acc CSV with columns [clip, rms_acceleration_si_m_s2]."
        )

    acc = pd.concat(acc_parts, ignore_index=True).dropna()
    acc["task"] = acc["task"].astype(str)
    acc["clip"] = acc["clip"].astype(str)

    # Drop exact duplicates
    acc = acc.drop_duplicates()

    # Resolve duplicates per (task, clip)
    dup_counts = acc.groupby(["task", "clip"]).size().reset_index(name="n")
    dup_keys = dup_counts[dup_counts["n"] > 1][["task", "clip"]]
    if len(dup_keys) > 0:
        # keep the first unique value if all values equal; otherwise median
        acc = (
            acc.groupby(["task", "clip"], as_index=False)["rms_acceleration_si_m_s2"]
               .median()
        )
    else:
        acc = acc.drop_duplicates(subset=["task", "clip"])

    print(f"[ACC] merged clip-level rows = {len(acc)}")
    return acc


# ------------------------------------------------------------
# Build low/high groups using within-task quartiles
# ------------------------------------------------------------
def assign_low_high(frame_df, acc_df):
    merged = frame_df.merge(acc_df, on=["task", "clip"], how="inner")
    if len(merged) == 0:
        raise RuntimeError("Merge between PA frame data and RMS Acc data failed.")

    clip_acc = merged[["task", "clip", "rms_acceleration_si_m_s2"]].drop_duplicates()

    stats = []
    group_map_parts = []

    for task in TASKS:
        sub = clip_acc[clip_acc["task"] == task].copy()
        if len(sub) == 0:
            raise RuntimeError(f"No merged clips for {task}")

        q25 = sub["rms_acceleration_si_m_s2"].quantile(0.25)
        q75 = sub["rms_acceleration_si_m_s2"].quantile(0.75)

        sub["group"] = "mid"
        sub.loc[sub["rms_acceleration_si_m_s2"] <= q25, "group"] = "low"
        sub.loc[sub["rms_acceleration_si_m_s2"] >= q75, "group"] = "high"

        low_n = (sub["group"] == "low").sum()
        high_n = (sub["group"] == "high").sum()

        print(f"[{task}] q25={q25:.6f} q75={q75:.6f} lowN={low_n} highN={high_n}")

        stats.append({
            "task": task,
            "q25": q25,
            "q75": q75,
            "lowN": low_n,
            "highN": high_n,
        })
        group_map_parts.append(sub[["task", "clip", "group"]])

    group_map = pd.concat(group_map_parts, ignore_index=True)
    merged = merged.merge(group_map, on=["task", "clip"], how="left")
    merged = merged[merged["group"].isin(["low", "high"])].copy()

    return merged, pd.DataFrame(stats)


# ------------------------------------------------------------
# Panel (a): clip-level median PA for low/high groups
# ------------------------------------------------------------
def compute_bar_stats(df_lh):
    clip_level = (
        df_lh.groupby(["task", "clip", "group"], as_index=False)["pa_mpjpe_mm"]
             .median()
             .rename(columns={"pa_mpjpe_mm": "clip_median_pa_mpjpe_mm"})
    )

    pooled = (
        clip_level.groupby("group")["clip_median_pa_mpjpe_mm"]
                  .median()
                  .to_dict()
    )

    low_med = float(pooled["low"])
    high_med = float(pooled["high"])
    ratio = high_med / low_med if low_med > 0 else np.nan

    # taskwise summary
    taskwise = (
        clip_level.groupby(["task", "group"])["clip_median_pa_mpjpe_mm"]
                  .median()
                  .unstack()
                  .reset_index()
    )

    return clip_level, taskwise, low_med, high_med, ratio


# ------------------------------------------------------------
# Panel (b): rollout curves with percentile bands
# ------------------------------------------------------------
def compute_rollout_stats(df_lh):
    rows = []
    for group in ["low", "high"]:
        sub = df_lh[df_lh["group"] == group]
        for frame in sorted(sub["frame"].unique()):
            vals = sub.loc[sub["frame"] == frame, "pa_mpjpe_mm"].to_numpy()
            rows.append({
                "group": group,
                "frame": int(frame),
                "p10": np.percentile(vals, 10),
                "p25": np.percentile(vals, 25),
                "p50": np.percentile(vals, 50),
                "p75": np.percentile(vals, 75),
                "p90": np.percentile(vals, 90),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def make_figure(low_med, high_med, ratio, rollout_stats):
    fig = plt.figure(figsize=(12.6, 5.6), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.28)

    # ---------------- panel (a)
    ax1 = fig.add_subplot(gs[0, 0])

    x = np.array([0, 1])
    vals = np.array([low_med, high_med])

    bars = ax1.bar(x, vals, width=0.56)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Low RMS Acc\n(bottom 25%)", "High RMS Acc\n(top 25%)"], fontsize=11)
    ax1.set_ylabel("Clip-level median PA-MPJPE (mm)", fontsize=12)
    ax1.set_title("(a) Higher-acceleration actions yield larger rigid-object error",
                  fontsize=14, fontweight="bold", pad=10)
    ax1.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")

    ymax = max(vals) * 1.35
    ax1.set_ylim(0, ymax)

    ax1.text(
        0.5, ymax * 0.92,
        f"{ratio:.2f}× higher",
        ha="center", va="center",
        fontsize=13, fontweight="bold"
    )

    # ---------------- panel (b)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("(b) The gap widens over rollout time",
                  fontsize=14, fontweight="bold", pad=10)

    color_map = {"low": "#1f77b4", "high": "#d62728"}
    label_map = {"low": "Low RMS Acc", "high": "High RMS Acc"}

    for group in ["low", "high"]:
        sub = rollout_stats[rollout_stats["group"] == group].sort_values("frame")
        x = sub["frame"].to_numpy()

        ax2.fill_between(x, sub["p10"], sub["p90"], alpha=0.12, color=color_map[group])
        ax2.fill_between(x, sub["p25"], sub["p75"], alpha=0.25, color=color_map[group])
        ax2.plot(x, sub["p50"], marker="o", linewidth=2.6, label=label_map[group], color=color_map[group])

    ax2.set_xlabel("Rollout frame", fontsize=12)
    ax2.set_ylabel("PA-MPJPE (mm)", fontsize=12)
    ax2.set_xticks(sorted(rollout_stats["frame"].unique()))
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, fontsize=11, loc="upper left")

    ax2.text(
        0.99, 0.03,
        "Bands: 10–90% and 25–75%; line: median",
        transform=ax2.transAxes,
        ha="right", va="bottom", fontsize=10, color="0.35"
    )

    fig.suptitle(
        "High-dynamics actions degrade rigid-object shape consistency",
        fontsize=16, fontweight="bold", y=0.995
    )

    fig.savefig(PNG_OUT, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    frame_df = load_frame_pa()
    acc_df = load_rms_acc()
    df_lh, quartile_df = assign_low_high(frame_df, acc_df)
    clip_level, taskwise, low_med, high_med, ratio = compute_bar_stats(df_lh)
    rollout_stats = compute_rollout_stats(df_lh)
    make_figure(low_med, high_med, ratio, rollout_stats)

    with open(TXT_OUT, "w") as f:
        f.write("=== Intro Problem Figure Summary ===\n")
        f.write(f"Low median PA-MPJPE  = {low_med:.10f} mm\n")
        f.write(f"High median PA-MPJPE = {high_med:.10f} mm\n")
        f.write(f"High / Low ratio     = {ratio:.10f}x\n\n")

        f.write("=== Taskwise quartiles ===\n")
        f.write(quartile_df.to_string(index=False))
        f.write("\n\n=== Taskwise clip-level medians (mm) ===\n")
        f.write(taskwise.to_string(index=False))
        f.write("\n\n=== Rollout stats ===\n")
        f.write(rollout_stats.to_string(index=False))

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"PNG = {PNG_OUT}")
    print(f"PDF = {PDF_OUT}")
    print(f"TXT = {TXT_OUT}")
    print("====================================================")
    print(f"Low median PA-MPJPE  = {low_med:.10f} mm")
    print(f"High median PA-MPJPE = {high_med:.10f} mm")
    print(f"High / Low ratio     = {ratio:.10f}x")
    print("====================================================")


if __name__ == "__main__":
    main()
