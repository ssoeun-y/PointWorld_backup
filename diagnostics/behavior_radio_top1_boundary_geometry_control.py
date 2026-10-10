from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from scipy.spatial import cKDTree
from scipy.stats import spearmanr


# ============================================================
# FIXED INPUTS
# ============================================================

SAMPLE_KEY = "task-0000_episode_00002530-300:311"

ROOT = Path(
    "diagnostics/results/"
    "behavior_radio_top1_mbnsf_pairwise_localization_smoke"
)

TRAJ_PATH = Path(
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237/"
    + SAMPLE_KEY.replace(":", "_")
    + ".npz"
)

LOC_PATH = ROOT / "top1_mbnsf_pairwise_matrices.npz"

ONSET_PATH = ROOT / "top1_mbnsf_pair_zero_onsets.npz"

POINT_CSV = ROOT / "per_point_mbnsf_localization_diagnostic.csv"

RAW_EVENT_CSV = (
    ROOT /
    "top1_mbnsf_persistent_onset_relvel_event_time.csv"
)

OUT_POINT = ROOT / "top1_boundary_geometry_point_diagnostic.csv"
OUT_MATCH = ROOT / "top1_boundary_geometry_match_balance.csv"
OUT_EVENT = ROOT / "top1_boundary_geometry_matched_event_time.csv"
OUT_SUMMARY = ROOT / "top1_boundary_geometry_control_summary.csv"

OUT_SPATIAL_PNG = (
    ROOT /
    "top1_boundary_geometry_vs_incompatibility.png"
)

OUT_GAP_PNG = (
    ROOT /
    "top1_boundary_geometry_matched_relvel_gap.png"
)

K_LIST = [8, 12, 16]
PRIMARY_K = 12


# ============================================================
# 1. LOAD + HARD AUDIT
# ============================================================

traj = np.load(TRAJ_PATH, allow_pickle=True)
loc = np.load(LOC_PATH, allow_pickle=False)
onset = np.load(ONSET_PATH, allow_pickle=False)
point_df = pd.read_csv(POINT_CSV)
raw_event_df = pd.read_csv(RAW_EVENT_CSV)

assert str(traj["sample_key"].item()) == SAMPLE_KEY
assert str(loc["sample_key"].item()) == SAMPLE_KEY
assert str(onset["sample_key"].item()) == SAMPLE_KEY

assert bool(traj["official_l2_exact"].item())
assert bool(traj["temporal_object_id_exact"].item())

object_id = int(loc["object_id"].item())

gt = traj["gt_scene_flows"].astype(np.float64)
pred = traj["pred_scene_flows"].astype(np.float64)
obj = traj["scene_object_id"]
valid = traj["official_valid_mask"]

mask = (
    (obj[0] == object_id)
    & valid[-1]
)

global_idx = np.flatnonzero(mask)

assert len(global_idx) == 425
assert np.array_equal(global_idx, loc["global_idx"])

gt_local = gt[:, mask]
pred_local = pred[:, mask]
valid_local = valid[:, mask]

assert np.all(valid_local[1:])
assert int(valid_local[0].sum()) == 0

x0 = gt_local[0]

assert np.array_equal(
    x0.astype(np.float64),
    loc["x0"].astype(np.float64),
)

ii, jj = np.triu_indices(425, k=1)

assert len(ii) == 90100
assert np.array_equal(ii, onset["ii"])
assert np.array_equal(jj, onset["jj"])

persistent_zero_t = onset["persistent_zero_t"]
final_zero = onset["final_zero"]
never_zero = onset["never_zero"]

assert int(final_zero.sum()) == 39103
assert int(never_zero.sum()) == 45064

case_pair_idx = np.flatnonzero(persistent_zero_t >= 2)
control_pair_idx = np.flatnonzero(never_zero)

assert len(case_pair_idx) == 39103
assert len(control_pair_idx) == 45064


# ============================================================
# 2. EXISTING RELATIVE-VELOCITY ERROR
# ============================================================

ii_t = torch.from_numpy(ii).long()
jj_t = torch.from_numpy(jj).long()

relvel_mm = {}

for t in range(2, 11):

    gt_prev = torch.from_numpy(gt_local[t - 1]).float()
    gt_now = torch.from_numpy(gt_local[t]).float()

    pred_prev = torch.from_numpy(pred_local[t - 1]).float()
    pred_now = torch.from_numpy(pred_local[t]).float()

    v_gt = gt_now - gt_prev
    v_pred = pred_now - pred_prev

    gt_rel = v_gt[ii_t] - v_gt[jj_t]
    pred_rel = v_pred[ii_t] - v_pred[jj_t]

    err = torch.linalg.vector_norm(
        pred_rel - gt_rel,
        dim=-1,
    )

    relvel_mm[t] = (
        err.cpu().numpy() * 1000.0
    )


# ============================================================
# 3. LOCAL GEOMETRY AT t0
#
# Continuous diagnostics only.
#
# density:
#   mean distance to k nearest neighbours.
#
# boundary_gap:
#   largest angular gap between neighbours after
#   projecting the local neighbourhood to its PCA
#   tangent plane.
#
#   Large value = neighbourhood occupies a smaller
#   angular sector -> boundary-like / missing-neighbour
#   geometry.
#
# curvature:
#   smallest PCA eigenvalue / eigenvalue sum.
#
# We run k=8,12,16 rather than relying on one k.
# ============================================================

point_dist = np.linalg.norm(
    x0[:, None, :] - x0[None, :, :],
    axis=-1,
)

np.fill_diagonal(point_dist, np.inf)

geometry_by_k = {}

point_rows = []

for k in K_LIST:

    nbr_idx = np.argsort(
        point_dist,
        axis=1,
    )[:, :k]

    density = np.zeros(425, dtype=np.float64)
    boundary_gap_deg = np.zeros(425, dtype=np.float64)
    curvature = np.zeros(425, dtype=np.float64)

    for p in range(425):

        nbr = nbr_idx[p]

        offsets = (
            x0[nbr] - x0[p]
        )

        d = np.linalg.norm(
            offsets,
            axis=1,
        )

        density[p] = float(
            d.mean()
        )

        cov = (
            offsets.T @ offsets
        ) / float(k)

        eigval, eigvec = np.linalg.eigh(cov)

        eigval = np.maximum(
            eigval,
            0.0,
        )

        denom = float(eigval.sum())

        if denom > 0:
            curvature[p] = float(
                eigval[0] / denom
            )

        tangent = eigvec[:, [2, 1]]

        proj = offsets @ tangent

        angles = np.arctan2(
            proj[:, 1],
            proj[:, 0],
        )

        angles = np.sort(
            np.mod(
                angles,
                2.0 * np.pi,
            )
        )

        wrapped = np.concatenate(
            [
                angles,
                angles[:1] + 2.0 * np.pi,
            ]
        )

        gaps = np.diff(wrapped)

        boundary_gap_deg[p] = float(
            np.degrees(
                gaps.max()
            )
        )

    geometry_by_k[k] = {
        "density": density,
        "boundary_gap_deg": boundary_gap_deg,
        "curvature": curvature,
    }

    for p in range(425):

        point_rows.append(
            {
                "k": k,
                "local_index": p,
                "global_point_index": int(global_idx[p]),
                "density_m": density[p],
                "boundary_gap_deg": boundary_gap_deg[p],
                "curvature": curvature[p],
            }
        )


geom_point_df = pd.DataFrame(point_rows)

# Attach the already-computed final per-point incompatibility.
point_df = (
    point_df
    .sort_values("local_index")
    .reset_index(drop=True)
)

assert np.array_equal(
    point_df["local_index"].to_numpy(),
    np.arange(425),
)

incompat = point_df[
    "mean_mbnsf_incompatibility"
].to_numpy(dtype=np.float64)

for k in K_LIST:

    g = geometry_by_k[k]

    rho_boundary, _ = spearmanr(
        g["boundary_gap_deg"],
        incompat,
    )

    rho_density, _ = spearmanr(
        g["density"],
        incompat,
    )

    rho_curvature, _ = spearmanr(
        g["curvature"],
        incompat,
    )

    print(
        f"k={k}: "
        f"Spearman incompat~boundary={rho_boundary:.6f}, "
        f"density={rho_density:.6f}, "
        f"curvature={rho_curvature:.6f}"
    )

geom_point_df.to_csv(
    OUT_POINT,
    index=False,
)


# ============================================================
# 4. BASELINE PAIR GEOMETRY
#
# Matching variables are ALL determined from x0,
# before looking at relative-velocity outcome.
#
# - initial pair distance
# - mean/max endpoint density
# - mean/max endpoint boundary-gap
# - mean/max endpoint curvature
# ============================================================

pair_distance = np.linalg.norm(
    x0[ii] - x0[jj],
    axis=-1,
)

raw_summary = (
    raw_event_df
    .groupby("lag", as_index=False)
    .agg(
        raw_gap_mm=(
            "median_gap_mm",
            "median",
        )
    )
)

match_balance_rows = []
matched_event_rows = []
summary_rows = []


def smd(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))

    denom = np.sqrt(
        (va + vb) / 2.0
    )

    if denom == 0:
        return 0.0

    return float(
        (np.mean(a) - np.mean(b))
        / denom
    )


# ============================================================
# 5. OUTCOME-BLIND GEOMETRY MATCHING
#
# Matching WITH replacement:
# every persistent-zero pair gets its nearest never-zero
# control in standardized baseline-geometry space.
#
# We report:
# - full matched population
# - closest 25% by baseline match distance
#
# Match quality selection does NOT use dynamics outcome.
# ============================================================

for k in K_LIST:

    g = geometry_by_k[k]

    density = g["density"]
    boundary = g["boundary_gap_deg"]
    curvature = g["curvature"]

    feature_names = [
        "pair_distance_m",
        "density_mean_m",
        "density_max_m",
        "boundary_mean_deg",
        "boundary_max_deg",
        "curvature_mean",
        "curvature_max",
    ]

    features = np.column_stack(
        [
            pair_distance,

            (
                density[ii]
                + density[jj]
            ) / 2.0,

            np.maximum(
                density[ii],
                density[jj],
            ),

            (
                boundary[ii]
                + boundary[jj]
            ) / 2.0,

            np.maximum(
                boundary[ii],
                boundary[jj],
            ),

            (
                curvature[ii]
                + curvature[jj]
            ) / 2.0,

            np.maximum(
                curvature[ii],
                curvature[jj],
            ),
        ]
    )

    case_x = features[
        case_pair_idx
    ]

    control_x = features[
        control_pair_idx
    ]

    pooled = np.concatenate(
        [
            case_x,
            control_x,
        ],
        axis=0,
    )

    mu = pooled.mean(
        axis=0,
    )

    sigma = pooled.std(
        axis=0,
    )

    sigma[sigma == 0] = 1.0

    case_z = (
        case_x - mu
    ) / sigma

    control_z = (
        control_x - mu
    ) / sigma

    tree = cKDTree(
        control_z
    )

    match_dist, match_local_idx = tree.query(
        case_z,
        k=1,
    )

    matched_control_pair_idx = (
        control_pair_idx[
            match_local_idx
        ]
    )

    unique_matched_controls = int(
        np.unique(
            matched_control_pair_idx
        ).size
    )

    reuse_counts = np.bincount(
        match_local_idx,
        minlength=len(control_pair_idx),
    )

    max_control_reuse = int(
        reuse_counts.max()
    )

    q25 = float(
        np.quantile(
            match_dist,
            0.25,
        )
    )

    subsets = {
        "all_matches": np.ones(
            len(case_pair_idx),
            dtype=bool,
        ),
        "closest25pct": (
            match_dist <= q25
        ),
    }

    # Balance audit.
    for f, name in enumerate(feature_names):

        before = smd(
            case_x[:, f],
            control_x[:, f],
        )

        after = smd(
            case_x[:, f],
            features[
                matched_control_pair_idx,
                f,
            ],
        )

        match_balance_rows.append(
            {
                "k": k,
                "feature": name,
                "smd_before": before,
                "smd_after_all_matches": after,
                "abs_smd_before": abs(before),
                "abs_smd_after_all_matches": abs(after),
            }
        )

    print()
    print(f"===== MATCH k={k} =====")
    print("case pairs =", len(case_pair_idx))
    print("raw controls =", len(control_pair_idx))
    print("unique matched controls =", unique_matched_controls)
    print("max control reuse =", max_control_reuse)
    print(
        "match distance median/p90 =",
        float(np.median(match_dist)),
        float(np.quantile(match_dist, 0.90)),
    )
    print(
        "closest25 threshold =",
        q25,
    )

    # --------------------------------------------------------
    # Matched event-time analysis
    # --------------------------------------------------------

    for subset_name, keep in subsets.items():

        kept_cases = case_pair_idx[keep]

        kept_controls = matched_control_pair_idx[keep]

        kept_onsets = persistent_zero_t[
            kept_cases
        ]

        detail_rows = []

        for onset_t in range(2, 11):

            in_group = (
                kept_onsets == onset_t
            )

            if not np.any(in_group):
                continue

            cidx = kept_cases[in_group]
            midx = kept_controls[in_group]

            for lag in [-2, -1, 0]:

                transition_end = (
                    onset_t + lag
                )

                if (
                    transition_end < 2
                    or transition_end > 10
                ):
                    continue

                case_err = relvel_mm[
                    transition_end
                ][cidx]

                matched_err = relvel_mm[
                    transition_end
                ][midx]

                pair_gap = (
                    case_err
                    - matched_err
                )

                detail_rows.append(
                    {
                        "k": k,
                        "subset": subset_name,
                        "persistent_zero_onset_t": onset_t,
                        "lag": lag,
                        "transition": (
                            f"t{transition_end-1}"
                            f"->t{transition_end}"
                        ),
                        "case_pair_n": int(len(cidx)),
                        "median_case_relvel_mm": float(
                            np.median(case_err)
                        ),
                        "median_matched_control_relvel_mm": float(
                            np.median(matched_err)
                        ),
                        "median_pairwise_gap_mm": float(
                            np.median(pair_gap)
                        ),
                    }
                )

        detail_df = pd.DataFrame(
            detail_rows
        )

        matched_event_rows.extend(
            detail_rows
        )

        for lag in [-2, -1, 0]:

            z = detail_df[
                detail_df["lag"] == lag
            ]

            if len(z) == 0:
                continue

            # summarize across absolute onset groups
            matched_gap = float(
                np.median(
                    z[
                        "median_pairwise_gap_mm"
                    ]
                )
            )

            raw_gap_row = raw_summary[
                raw_summary["lag"] == lag
            ]

            raw_gap = float(
                raw_gap_row[
                    "raw_gap_mm"
                ].iloc[0]
            )

            summary_rows.append(
                {
                    "k": k,
                    "subset": subset_name,
                    "lag": lag,
                    "onset_groups": int(
                        z[
                            "persistent_zero_onset_t"
                        ].nunique()
                    ),
                    "matched_median_gap_mm": matched_gap,
                    "raw_median_gap_mm": raw_gap,
                    "gap_retained_fraction": (
                        matched_gap / raw_gap
                        if raw_gap != 0
                        else np.nan
                    ),
                    "median_match_distance": float(
                        np.median(
                            match_dist[keep]
                        )
                    ),
                    "p90_match_distance": float(
                        np.quantile(
                            match_dist[keep],
                            0.90,
                        )
                    ),
                    "n_case_pairs": int(
                        keep.sum()
                    ),
                    "unique_matched_controls": int(
                        np.unique(
                            matched_control_pair_idx[keep]
                        ).size
                    ),
                }
            )


balance_df = pd.DataFrame(
    match_balance_rows
)

event_df = pd.DataFrame(
    matched_event_rows
)

summary_df = pd.DataFrame(
    summary_rows
)

balance_df.to_csv(
    OUT_MATCH,
    index=False,
)

event_df.to_csv(
    OUT_EVENT,
    index=False,
)

summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
)


# ============================================================
# 6. VISUALIZATION A
#
# SAME t0 positions:
# boundary proxy vs density vs final incompatibility.
# Primary display k=12; k=8/16 remain sensitivity.
# ============================================================

g = geometry_by_k[
    PRIMARY_K
]

fig = plt.figure(
    figsize=(15.5, 5.3)
)

data_panels = [
    (
        g["boundary_gap_deg"],
        "t0 boundary proxy\n(largest local angular gap)",
    ),
    (
        g["density"] * 1000.0,
        "t0 local sparsity\n(mean kNN distance, mm)",
    ),
    (
        incompat,
        "Final MBNSF-derived\npoint incompatibility",
    ),
]

for idx, (value, title) in enumerate(
    data_panels,
    start=1,
):

    ax = fig.add_subplot(
        1,
        3,
        idx,
        projection="3d",
    )

    sc = ax.scatter(
        x0[:, 0],
        x0[:, 1],
        x0[:, 2],
        c=value,
        cmap="viridis",
        s=20,
        depthshade=False,
    )

    ax.set_title(
        title,
        fontsize=12,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=22,
        azim=-58,
    )

    ax.set_box_aspect(
        (1, 1, 1)
    )

    fig.colorbar(
        sc,
        ax=ax,
        fraction=0.035,
        pad=0.03,
    )

fig.suptitle(
    "Top-1 radio: is structural incompatibility explained by t0 boundary/sparsity?",
    fontsize=15,
)

fig.tight_layout()

fig.savefig(
    OUT_SPATIAL_PNG,
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 7. VISUALIZATION B
#
# Raw vs geometry-matched event-time gap.
# Primary k=12.
# ============================================================

primary = summary_df[
    summary_df["k"] == PRIMARY_K
]

raw_plot = (
    primary[
        primary["subset"] == "all_matches"
    ]
    .sort_values("lag")
)

all_plot = (
    primary[
        primary["subset"] == "all_matches"
    ]
    .sort_values("lag")
)

best_plot = (
    primary[
        primary["subset"] == "closest25pct"
    ]
    .sort_values("lag")
)

fig, ax = plt.subplots(
    figsize=(8.2, 5.3)
)

ax.plot(
    raw_plot["lag"],
    raw_plot["raw_median_gap_mm"],
    marker="o",
    label="Raw: onset pairs - never-zero controls",
)

ax.plot(
    all_plot["lag"],
    all_plot["matched_median_gap_mm"],
    marker="o",
    label="Geometry-matched controls (all)",
)

ax.plot(
    best_plot["lag"],
    best_plot["matched_median_gap_mm"],
    marker="o",
    linestyle="--",
    label="Geometry-matched controls (closest 25%)",
)

ax.axhline(
    0,
    linestyle="--",
    alpha=0.4,
)

ax.set_xticks(
    [-2, -1, 0]
)

ax.set_xticklabels(
    [
        "2 before",
        "1 before",
        "Onset",
    ]
)

ax.set_ylabel(
    "Median relative-velocity error gap (mm/timestep)"
)

ax.set_title(
    "Top-1 radio: boundary/local-geometry controlled dynamics gap"
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT_GAP_PNG,
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 8. FINAL REPORT
# ============================================================

print()
print("=" * 105)
print("TOP-1 BOUNDARY / LOCAL-GEOMETRY CONFOUND CONTROL")
print("=" * 105)

print("sample_key =", SAMPLE_KEY)
print("object_id =", object_id)

print()
print("official_l2_exact =", bool(
    traj["official_l2_exact"].item()
))
print("temporal_object_id_exact =", bool(
    traj["temporal_object_id_exact"].item()
))

print()
print("fixed_points =", len(global_idx))
print("unique_pairs =", len(ii))
print("persistent-zero cases =", len(case_pair_idx))
print("never-zero controls =", len(control_pair_idx))

print()
print("===== MATCH BALANCE =====")
print(
    balance_df.to_string(
        index=False
    )
)

print()
print("===== CONTROLLED EVENT-TIME SUMMARY =====")
print(
    summary_df.to_string(
        index=False
    )
)

print()
print("saved =", OUT_POINT)
print("saved =", OUT_MATCH)
print("saved =", OUT_EVENT)
print("saved =", OUT_SUMMARY)
print("saved =", OUT_SPATIAL_PNG)
print("saved =", OUT_GAP_PNG)

print()
print("NO NEW INFERENCE")
print("MATCHING USES t0 GEOMETRY ONLY")
print("NO DYNAMICS OUTCOME USED FOR MATCHING")
print("K SENSITIVITY = 8, 12, 16")
print("NO PAIR-LEVEL SIGNIFICANCE CLAIM")
print("SUCCESS = True")
print("=" * 105)
