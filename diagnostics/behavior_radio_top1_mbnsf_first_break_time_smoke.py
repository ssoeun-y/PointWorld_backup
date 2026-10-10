from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ============================================================
# FIXED, PRE-AUDITED SETTINGS
# ============================================================

SAMPLE_KEY = "task-0000_episode_00002530-300:311"

TRAJ_PATH = Path(
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237/"
    + SAMPLE_KEY.replace(":", "_")
    + ".npz"
)

LOC_ROOT = Path(
    "diagnostics/results/"
    "behavior_radio_top1_mbnsf_pairwise_localization_smoke"
)

LOC_PATH = (
    LOC_ROOT /
    "top1_mbnsf_pairwise_matrices.npz"
)

OUT_ONSET = (
    LOC_ROOT /
    "top1_mbnsf_zero_onset_summary.csv"
)

OUT_EVENT = (
    LOC_ROOT /
    "top1_mbnsf_persistent_onset_relvel_event_time.csv"
)

OUT_PAIR = (
    LOC_ROOT /
    "top1_mbnsf_pair_zero_onsets.npz"
)

OUT_ONSET_PNG = (
    LOC_ROOT /
    "top1_mbnsf_zero_onset_distribution.png"
)

OUT_EVENT_PNG = (
    LOC_ROOT /
    "top1_mbnsf_persistent_onset_relvel_gap.png"
)


# ============================================================
# 1. LOAD EXISTING RESULTS ONLY
# ============================================================

traj = np.load(
    TRAJ_PATH,
    allow_pickle=True,
)

loc = np.load(
    LOC_PATH,
    allow_pickle=False,
)

assert str(traj["sample_key"].item()) == SAMPLE_KEY
assert str(loc["sample_key"].item()) == SAMPLE_KEY

assert bool(
    traj["official_l2_exact"].item()
)

assert bool(
    traj["temporal_object_id_exact"].item()
)

object_id = int(
    loc["object_id"].item()
)

d_thre = float(
    loc["d_thre"].item()
)

assert abs(d_thre - 0.03) < 1e-12, d_thre


# ============================================================
# 2. EXACT SAME TRAJECTORY / POINT POPULATION
#
# IMPORTANT:
# Match the previously successful localization path:
#
# stored float32
# -> astype(float64)
# -> torch.from_numpy(...).float()
#
# ============================================================

gt_np = traj[
    "gt_scene_flows"
].astype(np.float64)

pred_np = traj[
    "pred_scene_flows"
].astype(np.float64)

obj = traj[
    "scene_object_id"
]

valid = traj[
    "official_valid_mask"
]

assert gt_np.shape == pred_np.shape
assert gt_np.shape == (11, 7875, 3)

mask = (
    (obj[0] == object_id)
    & valid[-1]
)

global_idx = np.where(mask)[0]

assert len(global_idx) == 425

assert np.array_equal(
    global_idx,
    loc["global_idx"],
)

gt_local_np = gt_np[:, mask]
pred_local_np = pred_np[:, mask]

valid_local = valid[:, mask]

assert gt_local_np.shape == (11, 425, 3)
assert pred_local_np.shape == (11, 425, 3)

# Previously audited PointWorld semantics:
# t0 is context, so official_valid=False there.
# All 425 points are valid at future t1..t10.
assert int(valid_local[0].sum()) == 0
assert np.all(valid_local[1:])


# Exact x0 population reproduction.
x0_diff = float(
    np.max(
        np.abs(
            gt_local_np[0]
            - loc["x0"].astype(np.float64)
        )
    )
)

assert x0_diff == 0.0, x0_diff


# ============================================================
# 3. EXACT SUCCESSFUL MBNSF ADJACENCY PATH
#
# This is copied from the already-audited:
# behavior_radio_top1_mbnsf_pairwise_localization_smoke.py
#
# DO NOT replace with torch.cdist.
# ============================================================

x0 = (
    torch.from_numpy(
        gt_local_np[0]
    )
    .float()
    .unsqueeze(0)
)

src_dist = torch.norm(
    x0[:, :, None, :]
    - x0[:, None, :, :],
    dim=-1,
)

temporal_adj = []
temporal_cross_dist = []

with torch.no_grad():

    for t in range(1, 11):

        tgt = (
            torch.from_numpy(
                pred_local_np[t]
            )
            .float()
            .unsqueeze(0)
        )

        tgt_dist = torch.norm(
            tgt[:, :, None, :]
            - tgt[:, None, :, :],
            dim=-1,
        )

        cross_dist = torch.abs(
            src_dist - tgt_dist
        )

        adj = torch.clamp(
            1.0
            - cross_dist ** 2
            / d_thre ** 2,
            min=0,
        )

        temporal_cross_dist.append(
            cross_dist[0]
            .cpu()
            .numpy()
            .copy()
        )

        temporal_adj.append(
            adj[0]
            .cpu()
            .numpy()
            .copy()
        )


temporal_cross_dist = np.stack(
    temporal_cross_dist,
    axis=0,
)

temporal_adj = np.stack(
    temporal_adj,
    axis=0,
)

assert temporal_adj.shape == (
    10,
    425,
    425,
)


# ============================================================
# 4. HARD REPRODUCTION GATE
#
# t10 MUST reproduce the already-saved successful localization.
# If not, STOP.
# ============================================================

saved_pred_adj = loc[
    "pred_adj"
]

saved_pred_cross = loc[
    "pred_cross_dist"
]

adj_exact = np.array_equal(
    temporal_adj[-1],
    saved_pred_adj,
)

cross_exact = np.array_equal(
    temporal_cross_dist[-1],
    saved_pred_cross,
)

adj_max_abs_diff = float(
    np.max(
        np.abs(
            temporal_adj[-1]
            - saved_pred_adj
        )
    )
)

cross_max_abs_diff = float(
    np.max(
        np.abs(
            temporal_cross_dist[-1]
            - saved_pred_cross
        )
    )
)

zero_mask_exact = np.array_equal(
    temporal_adj[-1] == 0,
    saved_pred_adj == 0,
)

print("=" * 100)
print("EXACT T10 REPRODUCTION GATE")
print("=" * 100)
print("adj_exact =", adj_exact)
print("adj_max_abs_diff =", adj_max_abs_diff)
print("cross_exact =", cross_exact)
print("cross_max_abs_diff =", cross_max_abs_diff)
print("zero_mask_exact =", zero_mask_exact)
print()

assert adj_exact, (
    "t10 adjacency did not reproduce exactly",
    adj_max_abs_diff,
)

assert cross_exact, (
    "t10 cross-distance did not reproduce exactly",
    cross_max_abs_diff,
)

assert zero_mask_exact


# ============================================================
# 5. UNIQUE UNDIRECTED PAIRS
# ============================================================

ii, jj = np.triu_indices(
    425,
    k=1,
)

n_pairs = len(ii)

assert n_pairs == 90100

pair_adj = temporal_adj[
    :,
    ii,
    jj,
]
# shape = [10, 90100]
# row0 = t1
# row9 = t10

zero = (
    pair_adj == 0
)

final_zero = zero[-1]

assert int(
    final_zero.sum()
) == 39103

final_positive = ~final_zero


# ============================================================
# 6. FIRST-ZERO / PERSISTENT-ZERO ONSET
#
# first_zero_t:
# first t where pair adjacency becomes 0.
#
# persistent_zero_t:
# earliest t such that adjacency remains 0
# for ALL remaining timesteps through t10.
#
# This is NOT an "official MBNSF metric".
# It is a temporal diagnostic derived from the
# exact official MBNSF compatibility formulation.
# ============================================================

first_zero_t = np.full(
    n_pairs,
    -1,
    dtype=np.int16,
)

persistent_zero_t = np.full(
    n_pairs,
    -1,
    dtype=np.int16,
)


for p in range(n_pairs):

    z = zero[:, p]

    zero_where = np.flatnonzero(z)

    if len(zero_where) > 0:
        first_zero_t[p] = int(
            zero_where[0] + 1
        )

    # Only pairs that are zero at t10
    # can have a persistent-zero onset.
    if z[-1]:

        positive_where = np.flatnonzero(
            ~z
        )

        if len(positive_where) == 0:
            # zero throughout t1..t10
            persistent_zero_t[p] = 1
        else:
            last_positive_idx = int(
                positive_where[-1]
            )

            # array index 0=t1
            # if last positive index k,
            # persistent zero starts at timestep k+2.
            persistent_zero_t[p] = (
                last_positive_idx + 2
            )


assert np.all(
    persistent_zero_t[final_zero] >= 1
)

assert np.all(
    persistent_zero_t[final_positive] == -1
)


# ============================================================
# 7. RECOVERY / STABLE-CONTROL INVENTORY
# ============================================================

ever_zero = np.any(
    zero,
    axis=0,
)

never_zero = ~ever_zero

recovered_by_t10 = (
    ever_zero
    & final_positive
)

print("PAIR INVENTORY")
print("--------------")
print("unique_pairs =", n_pairs)
print(
    "final_zero_pairs =",
    int(final_zero.sum()),
)
print(
    "final_positive_pairs =",
    int(final_positive.sum()),
)
print(
    "never_zero_pairs =",
    int(never_zero.sum()),
)
print(
    "recovered_by_t10_pairs =",
    int(recovered_by_t10.sum()),
)
print()


# ============================================================
# 8. ONSET DISTRIBUTION
# ============================================================

onset_rows = []

for t in range(1, 11):

    onset_rows.append(
        {
            "timestep": t,

            "zero_pairs_at_t":
                int(
                    zero[t - 1].sum()
                ),

            "zero_fraction_at_t":
                float(
                    zero[t - 1].mean()
                ),

            "first_zero_at_t":
                int(
                    (
                        first_zero_t == t
                    ).sum()
                ),

            "persistent_zero_onset_at_t":
                int(
                    (
                        persistent_zero_t == t
                    ).sum()
                ),
        }
    )


onset_df = pd.DataFrame(
    onset_rows
)

onset_df.to_csv(
    OUT_ONSET,
    index=False,
)


# ============================================================
# 9. RELATIVE-VELOCITY ERROR
#
# EXACT existing definition:
#
# v_gt   = gt_now - gt_prev
# v_pred = pred_now - pred_prev
#
# err =
# || (v_pred_i-v_pred_j)
#    -(v_gt_i-v_gt_j) ||
#
# Primary future transitions:
# t1->t2 ... t9->t10.
#
# t0->t1 excluded because t0 is context and
# official_valid=False by PointWorld evaluation semantics.
# ============================================================

ii_t = torch.from_numpy(
    ii
).long()

jj_t = torch.from_numpy(
    jj
).long()

relvel_mm = {}


for t in range(2, 11):

    gt_prev = torch.from_numpy(
        gt_local_np[t - 1]
    ).float()

    gt_now = torch.from_numpy(
        gt_local_np[t]
    ).float()

    pred_prev = torch.from_numpy(
        pred_local_np[t - 1]
    ).float()

    pred_now = torch.from_numpy(
        pred_local_np[t]
    ).float()

    v_gt = (
        gt_now - gt_prev
    )

    v_pred = (
        pred_now - pred_prev
    )

    gt_rel = (
        v_gt[ii_t]
        - v_gt[jj_t]
    )

    pred_rel = (
        v_pred[ii_t]
        - v_pred[jj_t]
    )

    err = torch.linalg.vector_norm(
        pred_rel - gt_rel,
        dim=-1,
    )

    relvel_mm[t] = (
        err.cpu().numpy()
        * 1000.0
    )


# ============================================================
# 10. PERSISTENT-ONSET EVENT-TIME ANALYSIS
#
# For pairs whose persistent MBNSF-zero onset is t:
#
# lag -2 = velocity transition ending 2 steps before onset
# lag -1 = velocity transition immediately before onset
# lag  0 = velocity transition entering onset timestep
#
# Control:
# pairs that NEVER become MBNSF-zero at any t1..t10.
#
# This avoids using "recovered" pairs as stable controls.
#
# No pair-level p-values:
# 90k pairs inside one object are not independent samples.
# ============================================================

event_rows = []

for onset_t in range(2, 11):

    cases = (
        persistent_zero_t
        == onset_t
    )

    case_n = int(
        cases.sum()
    )

    if case_n == 0:
        continue

    controls = never_zero

    control_n = int(
        controls.sum()
    )

    assert control_n > 0

    for lag in (
        -2,
        -1,
        0,
    ):

        transition_end = (
            onset_t + lag
        )

        # Only t1->t2 ... t9->t10.
        if (
            transition_end < 2
            or transition_end > 10
        ):
            continue

        case_err = relvel_mm[
            transition_end
        ][cases]

        control_err = relvel_mm[
            transition_end
        ][controls]

        case_median = float(
            np.median(
                case_err
            )
        )

        control_median = float(
            np.median(
                control_err
            )
        )

        event_rows.append(
            {
                "persistent_zero_onset_t":
                    onset_t,

                "lag":
                    lag,

                "transition":
                    f"t{transition_end-1}->t{transition_end}",

                "case_pair_n":
                    case_n,

                "control_pair_n":
                    control_n,

                "case_relvel_median_mm":
                    case_median,

                "control_relvel_median_mm":
                    control_median,

                "median_gap_mm":
                    case_median
                    - control_median,

                "case_relvel_p75_mm":
                    float(
                        np.quantile(
                            case_err,
                            0.75,
                        )
                    ),

                "control_relvel_p75_mm":
                    float(
                        np.quantile(
                            control_err,
                            0.75,
                        )
                    ),

                "case_relvel_p90_mm":
                    float(
                        np.quantile(
                            case_err,
                            0.90,
                        )
                    ),

                "control_relvel_p90_mm":
                    float(
                        np.quantile(
                            control_err,
                            0.90,
                        )
                    ),
            }
        )


event_df = pd.DataFrame(
    event_rows
)

assert len(event_df) > 0

event_df.to_csv(
    OUT_EVENT,
    index=False,
)


# ============================================================
# 11. EVENT-TIME SUMMARY
#
# Compare WITHIN each absolute onset time first,
# then summarize the case-control gap across onset groups.
#
# This is more meaningful than pooling all pair observations
# across different absolute timesteps.
# ============================================================

event_summary = (
    event_df
    .groupby(
        "lag",
        as_index=False,
    )
    .agg(
        onset_groups=(
            "persistent_zero_onset_t",
            "nunique",
        ),

        median_case_relvel_mm=(
            "case_relvel_median_mm",
            "median",
        ),

        median_control_relvel_mm=(
            "control_relvel_median_mm",
            "median",
        ),

        median_gap_across_onsets_mm=(
            "median_gap_mm",
            "median",
        ),

        min_gap_across_onsets_mm=(
            "median_gap_mm",
            "min",
        ),

        max_gap_across_onsets_mm=(
            "median_gap_mm",
            "max",
        ),
    )
    .sort_values("lag")
)


# ============================================================
# 12. SAVE PAIR-LEVEL ONSET LABELS
# ============================================================

np.savez_compressed(
    OUT_PAIR,

    sample_key=np.array(
        SAMPLE_KEY
    ),

    object_id=np.array(
        object_id
    ),

    ii=ii,
    jj=jj,

    first_zero_t=first_zero_t,

    persistent_zero_t=
        persistent_zero_t,

    final_zero=final_zero,

    never_zero=never_zero,

    recovered_by_t10=
        recovered_by_t10,

    d_thre=np.array(
        d_thre
    ),
)


# ============================================================
# 13. VISUALIZATION A:
# ZERO / PERSISTENT-ONSET DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.4)
)

ax.plot(
    onset_df["timestep"],
    onset_df["zero_pairs_at_t"],
    marker="o",
    label="Pairs with MBNSF compatibility = 0",
)

ax.plot(
    onset_df["timestep"],
    onset_df[
        "persistent_zero_onset_at_t"
    ],
    marker="o",
    label="New persistent-zero onsets",
)

ax.set_xlabel(
    "Future timestep"
)

ax.set_ylabel(
    "Number of unique point pairs"
)

ax.set_xticks(
    np.arange(
        1,
        11,
    )
)

ax.set_title(
    "Top-1 severe radio: temporal onset of pairwise structural incompatibility"
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT_ONSET_PNG,
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 14. VISUALIZATION B:
# CASE-CONTROL RELATIVE-VELOCITY GAP AROUND ONSET
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.8, 5.2)
)

ax.plot(
    event_summary["lag"],
    event_summary[
        "median_gap_across_onsets_mm"
    ],
    marker="o",
)

ax.axhline(
    0.0,
    linestyle="--",
    alpha=0.5,
)

ax.axvline(
    0.0,
    linestyle="--",
    alpha=0.35,
)

ax.set_xticks(
    [-2, -1, 0]
)

ax.set_xticklabels(
    [
        "2 transitions before",
        "1 transition before",
        "Onset transition",
    ]
)

ax.set_ylabel(
    "Median relative-velocity error gap\n"
    "(onset pairs - never-zero controls, mm/timestep)"
)

ax.set_title(
    "Top-1 severe radio: dynamics around persistent structural-break onset"
)

ax.grid(
    True,
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUT_EVENT_PNG,
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 15. FINAL AUDIT / OUTPUT
# ============================================================

print("=" * 100)
print("TOP-1 MBNSF FIRST/PERSISTENT BREAK-TIME ANALYSIS")
print("=" * 100)

print("sample_key =", SAMPLE_KEY)
print("object_id =", object_id)
print("d_thre =", d_thre)

print()
print("official_l2_exact =", bool(
    traj["official_l2_exact"].item()
))
print("temporal_object_id_exact =", bool(
    traj["temporal_object_id_exact"].item()
))

print()
print("fixed_points =", len(global_idx))
print("x0_max_abs_diff =", x0_diff)

print()
print("t10_adj_exact =", adj_exact)
print("t10_cross_dist_exact =", cross_exact)
print("t10_zero_mask_exact =", zero_mask_exact)

print()
print("unique_pairs =", n_pairs)
print(
    "final_zero_pairs =",
    int(final_zero.sum()),
)
print(
    "final_positive_pairs =",
    int(final_positive.sum()),
)
print(
    "never_zero_pairs =",
    int(never_zero.sum()),
)
print(
    "recovered_by_t10_pairs =",
    int(recovered_by_t10.sum()),
)

print()
print("===== ONSET DISTRIBUTION =====")
print(
    onset_df.to_string(
        index=False
    )
)

print()
print("===== EVENT-TIME DETAIL =====")
print(
    event_df[
        [
            "persistent_zero_onset_t",
            "lag",
            "transition",
            "case_pair_n",
            "control_pair_n",
            "case_relvel_median_mm",
            "control_relvel_median_mm",
            "median_gap_mm",
        ]
    ].to_string(
        index=False
    )
)

print()
print("===== EVENT-TIME SUMMARY =====")
print(
    event_summary.to_string(
        index=False
    )
)

print()
print("saved =", OUT_ONSET)
print("saved =", OUT_EVENT)
print("saved =", OUT_PAIR)
print("saved =", OUT_ONSET_PNG)
print("saved =", OUT_EVENT_PNG)

print()
print("NO NEW INFERENCE")
print("NO NEW FAILURE THRESHOLD")
print("NO PAIR-LEVEL SIGNIFICANCE CLAIM")
print("SUCCESS = True")
print("=" * 100)
