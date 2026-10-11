from pathlib import Path
import numpy as np
import pandas as pd

R = Path("diagnostics/results")
OUT = R / "acc_gatr_probe_20261011"
OUT.mkdir(parents=True, exist_ok=True)

KEY = ["task", "clip"]
OBJECT = KEY + ["object_id", "frame"]
FRAMES = KEY + ["frame"]
B = 0.24580418408749

all_rows = []

for dose in (25, 50, 75, 100):
    name = (
        f"acc_causal_geometry_dose{dose}_20261011"
        if dose != 100
        else "acc_causal_geometry_boundary_20261010"
    )
    root = R / name
    actions, geometry = [], []

    for task in ("0000", "0001", "0006", "0008", "0010"):
        tp = root / f"task-{task}"
        actions.append(pd.read_csv(
            tp / "mechanism_action_audit.csv"
        ))
        geometry.append(pd.read_csv(
            tp / "mechanism_geometry_per_frame.csv"
        ))

    a = pd.concat(actions, ignore_index=True)
    g = pd.concat(geometry, ignore_index=True)

    assert not a[KEY].duplicated().any()

    p = g.pivot_table(
        index=OBJECT,
        columns="variant",
        values=["pairwise_error_mm", "pa_mpjpe_mm"],
        aggfunc="first"
    )

    needed = [
        (m, v)
        for m in ("pairwise_error_mm", "pa_mpjpe_mm")
        for v in ("base", "base_repeat", "corrected")
    ]

    assert all(x in p.columns for x in needed)

    p = p[needed].dropna()
    p.columns = [
        f"{m}_{v}" for m, v in p.columns
    ]
    p = p.reset_index()

    # One vote per clip-frame rather than per object.
    f = p.groupby(
        FRAMES, as_index=False
    ).mean(numeric_only=True)

    f["pw_repeat_abs"] = abs(
        f.pairwise_error_mm_base
        - f.pairwise_error_mm_base_repeat
    )
    f["pa_repeat_abs"] = abs(
        f.pa_mpjpe_mm_base
        - f.pa_mpjpe_mm_base_repeat
    )
    f["pw_late"] = np.where(
        f.frame >= 8,
        f.pairwise_error_mm_corrected,
        np.nan
    )
    f["pw_early"] = np.where(
        f.frame <= 3,
        f.pairwise_error_mm_corrected,
        np.nan
    )

    c = f.groupby(KEY, as_index=False).agg(
        pair_score=("pairwise_error_mm_corrected", "mean"),
        pair_late=("pw_late", "mean"),
        pair_early=("pw_early", "mean"),
        pair_repeat_noise=("pw_repeat_abs", "mean"),
        pa_repeat_noise=("pa_repeat_abs", "mean"),
        pa_corrected=("pa_mpjpe_mm_corrected", "mean"),
        pa_original=("pa_mpjpe_mm_base", "mean"),
        frame_count=("frame", "nunique")
    )

    assert c.frame_count.eq(10).all()

    c = c.merge(
        a[KEY + [
            "original_acc_si",
            "modified_acc_si",
            "original_speed_rms",
            "modified_speed_rms",
            "max_phase_shift_frames"
        ]],
        on=KEY,
        validate="one_to_one"
    )

    c["dose"] = dose

    # Reliability score: no future GT, no PA used.
    # Lower means more stable geometry.
    c["score"] = (
        c.pair_score
        + 0.5 * np.maximum(
            c.pair_late - c.pair_early, 0
        )
        + 0.02 * c.max_phase_shift_frames
    )

    c["acc_reduction_pct"] = 100 * (
        1 - c.modified_acc_si /
        c.original_acc_si.clip(lower=1e-9)
    )
    c["speed_delta_pct"] = 100 * (
        c.modified_speed_rms /
        c.original_speed_rms.clip(lower=1e-9) - 1
    )

    # Prototype action trust region.
    # These are not certified robot hardware limits.
    c["valid"] = (
        np.isfinite(c.score)
        & (c.modified_acc_si <= c.original_acc_si + 1e-6)
        & (c.max_phase_shift_frames <= 2)
        & (c.speed_delta_pct.abs() <= 10)
    )

    all_rows.append(c)
    print(
        "LOADED", dose,
        "CLIPS", len(c),
        "ACTION VALID", int(c.valid.sum()),
        flush=True
    )

D = pd.concat(all_rows, ignore_index=True)
counts = D.groupby(KEY).dose.nunique()
assert len(counts) == 2677
assert counts.eq(4).all()

D.to_csv(
    OUT / "candidate_scores.csv",
    index=False
)

selected = []

for (task, clip), z in D.groupby(KEY, sort=False):
    fixed = z.loc[z.dose == 100].iloc[0]
    options = z.loc[
        z.valid & (z.dose != 100)
    ]

    chosen = fixed
    threshold = 0.0

    if len(options):
        candidate = options.loc[
            options.score.idxmin()
        ]

        gain = fixed.score - candidate.score

        # Require gain above observed repeat variability.
        threshold = max(
            0.01,
            2 * fixed.pair_repeat_noise,
            2 * candidate.pair_repeat_noise
        )

        if np.isfinite(gain) and gain > threshold:
            chosen = candidate

    selected.append({
        "task": task,
        "clip": clip,
        "selected_dose": int(chosen.dose),
        "fixed_score": float(fixed.score),
        "selected_score": float(chosen.score),
        "score_gain": float(
            fixed.score - chosen.score
        ),
        "threshold_mm": float(threshold),
        "fixed_PA_mm": float(fixed.pa_corrected),
        "selected_PA_mm": float(chosen.pa_corrected),
        "pa_gain_mm": float(
            fixed.pa_corrected - chosen.pa_corrected
        ),
        "selected_acc": float(chosen.modified_acc_si),
        "selected_speed_delta_pct": float(
            chosen.speed_delta_pct
        ),
        "selected_above_B": bool(
            chosen.modified_acc_si > B + 1e-6
        )
    })

S = pd.DataFrame(selected)
assert not S[KEY].duplicated().any()

S.to_csv(
    OUT / "gatr_selections.csv",
    index=False
)

print("\n===== GATR V0 PROBE =====")
print("CLIPS:", len(S))
print("ALTERNATIVE SELECTED:",
      int((S.selected_dose != 100).sum()))
print("DOSE DISTRIBUTION:")
print(S.selected_dose.value_counts().sort_index())
print("SELECTED ABOVE FIXED B:",
      int(S.selected_above_B.sum()))

print("\nCACHED PA GAIN VS FIXED:")
print("Median mm:", S.pa_gain_mm.median())
print("Mean mm:", S.pa_gain_mm.mean())
print("Improved %:",
      100 * (S.pa_gain_mm > 0).mean())

print("\nTASKWISE:")
print(S.groupby("task").agg(
    n=("clip", "size"),
    alternate=("selected_dose",
               lambda x: int((x != 100).sum())),
    mean_pa_gain=("pa_gain_mm", "mean")
).to_string())

print("\nNOTE: Cached comparison is optimistic.")
print("NOTE: Object identity is oracle in this pilot.")
print("PROBE_COMPLETE", flush=True)
