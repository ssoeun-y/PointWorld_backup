from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ============================================================
# Fixed target
# ============================================================

SAMPLE_KEY = "task-0000_episode_00002530-300:311"

TRAJ_ROOT = Path(
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

LOC_ROOT = Path(
    "diagnostics/results/"
    "behavior_radio_top1_mbnsf_pairwise_localization_smoke"
)

TRAJ_PATH = (
    TRAJ_ROOT /
    (SAMPLE_KEY.replace(":", "_") + ".npz")
)

LOC_PATH = (
    LOC_ROOT /
    "top1_mbnsf_pairwise_matrices.npz"
)

OUT_CSV = (
    LOC_ROOT /
    "top1_final_mbnsf_pair_relative_velocity_precursor.csv"
)

OUT_PNG = (
    LOC_ROOT /
    "top1_final_mbnsf_pair_relative_velocity_precursor.png"
)


# ============================================================
# 1. Load existing saved results only
# ============================================================

traj = np.load(TRAJ_PATH, allow_pickle=True)
loc = np.load(LOC_PATH, allow_pickle=False)

traj_sample = str(traj["sample_key"].item())
loc_sample = str(loc["sample_key"].item())

assert traj_sample == SAMPLE_KEY, (
    traj_sample,
    SAMPLE_KEY,
)

assert loc_sample == SAMPLE_KEY, (
    loc_sample,
    SAMPLE_KEY,
)

assert bool(traj["official_l2_exact"].item())
assert bool(traj["temporal_object_id_exact"].item())

object_id = int(loc["object_id"].item())

gt = traj["gt_scene_flows"]
pred = traj["pred_scene_flows"]
obj = traj["scene_object_id"]
valid = traj["official_valid_mask"]

assert gt.shape == pred.shape
assert gt.shape == (11, 7875, 3), gt.shape
assert obj.shape == (11, 7875)
assert valid.shape == (11, 7875)


# ============================================================
# 2. EXACT SAME fixed point population as temporal MBNSF
# ============================================================

mask = (
    (obj[0] == object_id)
    & valid[-1]
)

selected_idx = np.flatnonzero(mask)

assert len(selected_idx) == 425, len(selected_idx)

global_idx = loc["global_idx"]

assert global_idx.shape == (425,)

global_idx_exact = np.array_equal(
    selected_idx,
    global_idx,
)

assert global_idx_exact, (
    selected_idx[:20],
    global_idx[:20],
)


gt_local = gt[:, mask]
pred_local = pred[:, mask]
valid_local = valid[:, mask]

assert gt_local.shape == (11, 425, 3)
assert pred_local.shape == (11, 425, 3)
assert valid_local.shape == (11, 425)


# ============================================================
# 3. x0 reproduction against localization NPZ
# ============================================================

x0_saved = loc["x0"]
x0_traj = gt_local[0].astype(np.float64)

x0_max_abs_diff = float(
    np.max(
        np.abs(
            x0_saved.astype(np.float64)
            - x0_traj
        )
    )
)

assert x0_max_abs_diff < 1e-7, x0_max_abs_diff


# ============================================================
# 4. Final MBNSF pair labels at t10
#
# pred_adj is the official MBNSF compatibility matrix
# reconstructed/saved by the previous audited localization.
#
# Only unique undirected pairs i < j are used.
#
# final broken:
#       pred_adj[i,j] == 0
#
# comparison:
#       pred_adj[i,j] > 0
#
# No new threshold is introduced here.
# ============================================================

pred_adj = loc["pred_adj"]

assert pred_adj.shape == (425, 425)

symmetry_max_diff = float(
    np.max(
        np.abs(
            pred_adj - pred_adj.T
        )
    )
)

assert symmetry_max_diff < 1e-6, symmetry_max_diff

ii, jj = np.triu_indices(
    425,
    k=1,
)

n_pairs = len(ii)

assert n_pairs == 90100, n_pairs

pair_adj = pred_adj[ii, jj]

broken = pair_adj == 0
nonbroken = pair_adj > 0

n_broken = int(broken.sum())
n_nonbroken = int(nonbroken.sum())

assert n_broken == 39103, n_broken
assert n_nonbroken == 50997, n_nonbroken
assert n_broken + n_nonbroken == 90100


# ============================================================
# 5. Existing relative-velocity-error definition
#
# v_gt   = gt_now   - gt_prev
# v_pred = pred_now - pred_prev
#
# gt_rel   = v_gt[i]   - v_gt[j]
# pred_rel = v_pred[i] - v_pred[j]
#
# err = || pred_rel - gt_rel ||
#
# Keep float32 torch path, matching the existing diagnostic.
# ============================================================

rows = []

for t in range(2, 11):

    gt_prev = torch.from_numpy(
        gt_local[t - 1]
    )

    gt_now = torch.from_numpy(
        gt_local[t]
    )

    pred_prev = torch.from_numpy(
        pred_local[t - 1]
    )

    pred_now = torch.from_numpy(
        pred_local[t]
    )

    v_gt = gt_now - gt_prev
    v_pred = pred_now - pred_prev

    ii_t = torch.from_numpy(ii).long()
    jj_t = torch.from_numpy(jj).long()

    gt_rel = (
        v_gt[ii_t]
        - v_gt[jj_t]
    )

    pred_rel = (
        v_pred[ii_t]
        - v_pred[jj_t]
    )

    err_m = torch.linalg.vector_norm(
        pred_rel - gt_rel,
        dim=-1,
    ).numpy()

    # Same pair-valid logic as the existing full578
    # relative-dynamics implementation.
    point_valid = (
        valid_local[t - 1]
        & valid_local[t]
    )

    pair_valid = (
        point_valid[ii]
        & point_valid[jj]
    )

    b_ok = pair_valid & broken
    n_ok = pair_valid & nonbroken

    b_err_mm = err_m[b_ok] * 1000.0
    n_err_mm = err_m[n_ok] * 1000.0

    assert len(b_err_mm) > 0
    assert len(n_err_mm) > 0

    b_median = float(
        np.median(b_err_mm)
    )

    n_median = float(
        np.median(n_err_mm)
    )

    row = {
        "timestep": t,
        "transition": f"t{t-1}->t{t}",

        "broken_pair_total": n_broken,
        "nonbroken_pair_total": n_nonbroken,

        "broken_pair_valid_n": int(len(b_err_mm)),
        "nonbroken_pair_valid_n": int(len(n_err_mm)),

        "broken_relvel_median_mm": b_median,
        "nonbroken_relvel_median_mm": n_median,

        "median_gap_mm": (
            b_median
            - n_median
        ),

        "broken_relvel_mean_mm": float(
            np.mean(b_err_mm)
        ),
        "nonbroken_relvel_mean_mm": float(
            np.mean(n_err_mm)
        ),

        "broken_relvel_p75_mm": float(
            np.quantile(
                b_err_mm,
                0.75,
            )
        ),
        "nonbroken_relvel_p75_mm": float(
            np.quantile(
                n_err_mm,
                0.75,
            )
        ),

        "broken_relvel_p90_mm": float(
            np.quantile(
                b_err_mm,
                0.90,
            )
        ),
        "nonbroken_relvel_p90_mm": float(
            np.quantile(
                n_err_mm,
                0.90,
            )
        ),

        "broken_relvel_p95_mm": float(
            np.quantile(
                b_err_mm,
                0.95,
            )
        ),
        "nonbroken_relvel_p95_mm": float(
            np.quantile(
                n_err_mm,
                0.95,
            )
        ),
    }

    rows.append(row)


out = pd.DataFrame(rows)

out.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# 6. Visualization
#
# Descriptive smoke only.
# No pair-level significance testing because pairs inside one
# object are highly dependent.
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

ax.plot(
    out["timestep"],
    out["broken_relvel_median_mm"],
    marker="o",
    label="Final MBNSF-zero pairs",
)

ax.plot(
    out["timestep"],
    out["nonbroken_relvel_median_mm"],
    marker="o",
    label="Final MBNSF-positive pairs",
)

ax.set_xlabel(
    "Future timestep / velocity transition"
)

ax.set_ylabel(
    "Median relative-velocity error (mm / timestep)"
)

ax.set_title(
    "Top-1 severe radio: final structural failure vs earlier relative dynamics"
)

ax.set_xticks(
    np.arange(1, 11)
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT_PNG,
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 7. Console audit
# ============================================================

print("=" * 100)
print("TOP-1 MBNSF FINAL-PAIR RELATIVE-VELOCITY PRECURSOR SMOKE")
print("=" * 100)

print("sample_key =", SAMPLE_KEY)
print("object_id =", object_id)

print()
print("trajectory_shape =", gt.shape)
print("fixed_radio_points =", len(selected_idx))

print(
    "global_idx_exact =",
    global_idx_exact,
)

print(
    "x0_max_abs_diff =",
    x0_max_abs_diff,
)

print()
print("unique_pairs =", n_pairs)
print("final_broken_pairs =", n_broken)
print("final_nonbroken_pairs =", n_nonbroken)

print(
    "pred_adj_symmetry_max_diff =",
    symmetry_max_diff,
)

print()
print(
    out[
        [
            "transition",
            "broken_pair_valid_n",
            "nonbroken_pair_valid_n",
            "broken_relvel_median_mm",
            "nonbroken_relvel_median_mm",
            "median_gap_mm",
            "broken_relvel_p90_mm",
            "nonbroken_relvel_p90_mm",
        ]
    ].to_string(
        index=False
    )
)

print()
print("saved_csv =", OUT_CSV)
print("saved_png =", OUT_PNG)

print()
print("official_l2_exact =", bool(
    traj["official_l2_exact"].item()
))

print(
    "temporal_object_id_exact =",
    bool(
        traj["temporal_object_id_exact"].item()
    )
)

print("NO INFERENCE RUN")
print("NO NEW FAILURE THRESHOLD")
print("NO PAIR-LEVEL SIGNIFICANCE TEST")
print("SUCCESS = True")
print("=" * 100)
