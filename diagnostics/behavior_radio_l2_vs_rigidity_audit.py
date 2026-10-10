import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANK_CSV = "diagnostics/results/behavior_radio_rigidity_rank.csv"

NPZ_ROOT = (
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

OUTDIR = (
    "diagnostics/results/"
    "behavior_radio_l2_vs_rigidity_audit"
)
os.makedirs(OUTDIR, exist_ok=True)

rank_df = pd.read_csv(RANK_CSV)

assert len(rank_df) == 554


def spearman_np(a, b):
    a = pd.Series(a).rank(method="average").to_numpy(float)
    b = pd.Series(b).rank(method="average").to_numpy(float)

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


rows = []

for _, r in rank_df.iterrows():

    key = str(r["sample_key"])
    object_id = int(r["object_id"])

    path = os.path.join(
        NPZ_ROOT,
        key.replace(":", "_") + ".npz"
    )

    assert os.path.exists(path)

    d = np.load(path, allow_pickle=True)

    assert d["official_l2_exact"].item()
    assert d["temporal_object_id_exact"].item()

    gt = d["gt_scene_flows"].astype(np.float64)
    pred = d["pred_scene_flows"].astype(np.float64)
    obj = d["scene_object_id"]
    valid = d["official_valid_mask"]

    # same radio identity
    radio_t0 = (obj[0] == object_id)

    # --------------------------------------------------------
    # 1) official scene-level scalar stored in NPZ
    # --------------------------------------------------------
    official_l2_mm = (
        float(d["official_l2"].item()) * 1000.0
    )

    # --------------------------------------------------------
    # 2) radio-only L2 across all future official-valid points
    #    t0 is context, so use t=1..10
    # --------------------------------------------------------
    err = np.linalg.norm(pred - gt, axis=-1)

    radio_valid = (
        (obj == object_id)
        & valid
    )

    future_mask = radio_valid.copy()
    future_mask[0] = False

    assert future_mask.sum() > 0

    radio_allfuture_l2_mm = (
        err[future_mask].mean() * 1000.0
    )

    # --------------------------------------------------------
    # 3) radio t10 L2 on same final-valid point set used
    #    for rigidity analysis
    # --------------------------------------------------------
    final_mask = (
        radio_t0
        & valid[-1]
    )

    assert final_mask.sum() == int(r["n_points"])

    radio_t10_l2_mm = (
        err[-1, final_mask].mean() * 1000.0
    )

    rows.append({
        "sample_key": key,
        "n_points": int(final_mask.sum()),

        "official_scene_l2_mm":
            official_l2_mm,

        "radio_allfuture_l2_mm":
            radio_allfuture_l2_mm,

        "radio_t10_l2_mm":
            radio_t10_l2_mm,

        "rigid_residual_mean_mm":
            float(r["pred_mean_mm"]),
    })


out = pd.DataFrame(rows)

assert len(out) == 554

csv_path = os.path.join(
    OUTDIR,
    "radio_l2_vs_rigidity_per_sample.csv"
)
out.to_csv(csv_path, index=False)


rig = out["rigid_residual_mean_mm"].to_numpy()

metrics = [
    "official_scene_l2_mm",
    "radio_allfuture_l2_mm",
    "radio_t10_l2_mm",
]

print("=" * 100)
print("L2 vs RIGIDITY AUDIT")
print("N =", len(out))
print("=" * 100)

for m in metrics:
    x = out[m].to_numpy()

    rho = spearman_np(x, rig)
    pear = float(np.corrcoef(x, rig)[0, 1])

    print(
        f"{m:28s} "
        f"Spearman={rho:+.6f} "
        f"Pearson={pear:+.6f}"
    )


# ============================================================
# Find systematic "similar L2, different rigidity" candidates.
#
# Use radio all-future L2, not scene-level L2.
# Pair samples only when L2 differs by <= 0.25 mm.
# Report pair with largest rigidity difference.
# ============================================================

L2_TOL_MM = 0.25

best = None

vals = out[
    [
        "sample_key",
        "radio_allfuture_l2_mm",
        "rigid_residual_mean_mm",
    ]
].sort_values(
    "radio_allfuture_l2_mm"
).reset_index(drop=True)

for i in range(len(vals)):
    for j in range(i + 1, len(vals)):

        l2_gap = (
            vals.loc[j, "radio_allfuture_l2_mm"]
            - vals.loc[i, "radio_allfuture_l2_mm"]
        )

        if l2_gap > L2_TOL_MM:
            break

        rig_gap = abs(
            vals.loc[j, "rigid_residual_mean_mm"]
            - vals.loc[i, "rigid_residual_mean_mm"]
        )

        if best is None or rig_gap > best["rig_gap"]:
            best = {
                "i": i,
                "j": j,
                "l2_gap": float(l2_gap),
                "rig_gap": float(rig_gap),
            }


print("\n" + "=" * 100)
print(
    "SIMILAR-L2 / DIFFERENT-RIGIDITY CANDIDATE"
)
print(
    "L2 tolerance <=",
    L2_TOL_MM,
    "mm"
)

if best is not None:
    a = vals.loc[best["i"]]
    b = vals.loc[best["j"]]

    print("\nA:")
    print(a.to_string())

    print("\nB:")
    print(b.to_string())

    print(
        "\nL2_GAP_MM =",
        best["l2_gap"]
    )
    print(
        "RIGIDITY_GAP_MM =",
        best["rig_gap"]
    )


# ============================================================
# Scatter
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    out["radio_allfuture_l2_mm"],
    out["rigid_residual_mean_mm"],
    s=20,
    alpha=0.65,
)

plt.xlabel(
    "Radio positional L2, all future valid points (mm)"
)
plt.ylabel(
    "Final non-rigid residual after best rigid alignment (mm)"
)

plt.title(
    "Does positional L2 capture rigid-object integrity?"
)

fig_path = os.path.join(
    OUTDIR,
    "radio_l2_vs_rigidity_scatter.png"
)

plt.tight_layout()
plt.savefig(
    fig_path,
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\nCSV =", csv_path)
print("FIGURE =", fig_path)

print(
    "FILES_EXIST =",
    os.path.exists(csv_path)
    and os.path.exists(fig_path)
)

print(
    "L2_VS_RIGIDITY_AUDIT_SUCCESS = True"
)
