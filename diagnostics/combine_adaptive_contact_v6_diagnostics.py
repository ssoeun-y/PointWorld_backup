from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "diagnostics/results/"
    "acc_boundary_adaptive_contact_v6_current10457"
)

TASKS = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]

dfs = []

for task in TASKS:

    p = (
        ROOT
        / task
        / "per_clip_adaptive_contact_v6_diagnostics.csv"
    )

    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)

    if "task" not in df:
        df["task"] = task

    dfs.append(df)

all_df = pd.concat(
    dfs,
    ignore_index=True,
)

if len(all_df) != 2683:
    raise RuntimeError(
        f"Expected 2683 clips, got {len(all_df)}"
    )

out = (
    ROOT
    / "combined_5task_adaptive_contact_v6_diagnostics.csv"
)

all_df.to_csv(
    out,
    index=False,
)

lines = []

lines.append(
    "ADAPTIVE CONTACT V6 - 5 TASK DIAGNOSTICS"
)
lines.append(
    f"N = {len(all_df)}"
)

with_memory = (
    all_df["n_memories"] > 0
)

release = (
    all_df["release_frames"] > 0
)

finite_sep = (
    np.isfinite(
        all_df[
            "separation_before_mean_mm"
        ]
    )
    &
    np.isfinite(
        all_df[
            "separation_after_mean_mm"
        ]
    )
)

lines.append(
    f"clips_with_contact_memory = "
    f"{int(with_memory.sum())} "
    f"({100*with_memory.mean():.2f}%)"
)

lines.append(
    f"clips_with_release = "
    f"{int(release.sum())} "
    f"({100*release.mean():.2f}%)"
)

lines.append(
    f"clips_with_separation_eval = "
    f"{int(finite_sep.sum())}"
)

if finite_sep.any():

    a = all_df.loc[
        finite_sep,
        "separation_before_mean_mm",
    ].to_numpy(float)

    b = all_df.loc[
        finite_sep,
        "separation_after_mean_mm",
    ].to_numpy(float)

    lines.append(
        "separation_before_median_mm = "
        f"{np.median(a):.6f}"
    )

    lines.append(
        "separation_after_median_mm = "
        f"{np.median(b):.6f}"
    )

    lines.append(
        "separation_lower_fraction = "
        f"{np.mean(b < a):.6f}"
    )

    good = a > 0

    if good.any():
        rel = (
            (b[good] - a[good])
            / a[good]
            * 100.0
        )

        lines.append(
            "paired_separation_relative_median_pct = "
            f"{np.median(rel):.6f}"
        )

lines.append(
    "final_residual_median_mm = "
    f"{np.nanmedian(all_df['final_residual_mean_mm']):.6f}"
)

lines.append(
    "max_residual_change_median_mm = "
    f"{np.nanmedian(all_df['max_residual_change_mean_mm']):.6f}"
)

lines.append("")
lines.append("TASKWISE")

for task in TASKS:

    d = all_df[
        all_df["task"] == task
    ]

    fs = (
        np.isfinite(
            d[
                "separation_before_mean_mm"
            ]
        )
        &
        np.isfinite(
            d[
                "separation_after_mean_mm"
            ]
        )
    )

    msg = (
        f"{task}: N={len(d)}, "
        f"memory={(d['n_memories'] > 0).mean()*100:.2f}%, "
        f"release={(d['release_frames'] > 0).mean()*100:.2f}%"
    )

    if fs.any():

        bef = d.loc[
            fs,
            "separation_before_mean_mm",
        ].to_numpy(float)

        aft = d.loc[
            fs,
            "separation_after_mean_mm",
        ].to_numpy(float)

        msg += (
            f", sep_med="
            f"{np.median(bef):.4f}"
            f"->{np.median(aft):.4f}mm"
            f", lower="
            f"{np.mean(aft < bef)*100:.2f}%"
        )

    lines.append(msg)

report = (
    ROOT
    / "FINAL_V6_DIAGNOSTICS.txt"
)

report.write_text(
    "\n".join(lines) + "\n"
)

print("\n".join(lines))
print("SAVED =", out)
print("REPORT =", report)
