import os
import re
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch


# ============================================================
# FIXED / PRE-AUDITED SETTINGS
# ============================================================

MBNSF_ROOT = "/home/viglab/cse/MBNSF_official"
D_THRE = 0.03
MAX_POINTS = 3000
TOP_K = 5

FULL_CSV = Path(
    "diagnostics/results/"
    "behavior_radio_full554_mbnsf_vs_l2/"
    "radio_full554_mbnsf_vs_l2.csv"
)

NPZ_ROOT = Path(
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

OUTDIR = Path(
    "diagnostics/results/"
    "behavior_radio_top5_episode_disjoint_repeatability"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_SUMMARY = OUTDIR / "top5_episode_disjoint_structural_summary.csv"
OUT_TEMPORAL = OUTDIR / "top5_episode_disjoint_temporal_mbnsf.csv"

OUT_SCORE_PNG = OUTDIR / "top5_gt_vs_pred_final_mbnsf.png"
OUT_TEMPORAL_PNG = OUTDIR / "top5_pred_temporal_mbnsf.png"
OUT_ZERO_PNG = OUTDIR / "top5_pred_zero_pair_fraction.png"


# ============================================================
# 1. Load exact official MBNSF implementation
# ============================================================

SC_UTILS_PATH = os.path.join(
    MBNSF_ROOT,
    "utils",
    "sc_utils.py",
)

assert os.path.exists(
    SC_UTILS_PATH
), SC_UTILS_PATH

spec = importlib.util.spec_from_file_location(
    "mbnsf_official_sc_utils",
    SC_UTILS_PATH,
)

mbnsf_sc = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    mbnsf_sc
)

spatial_consistency_loss = (
    mbnsf_sc.spatial_consistency_loss
)


# ============================================================
# 2. Episode parsing
# ============================================================

def episode_id_from_key(key):
    m = re.search(
        r"(task-\d+_episode_\d+)",
        key,
    )

    assert m is not None, key

    return m.group(1)


# ============================================================
# 3. Select severe cases objectively
#
# Rank by already-computed official prediction MBNSF score.
#
# Primary repeatability population is episode-disjoint:
# take the first row from each new episode until 5 episodes.
#
# No outcome beyond the existing official MBNSF score is used.
# ============================================================

full = pd.read_csv(
    FULL_CSV
)

assert len(full) == 554

required_cols = [
    "sample_key",
    "object_id",
    "mbnsf_pred_sc_score",
    "mbnsf_gt_sc_score",
]

for c in required_cols:
    assert c in full.columns, c

full = (
    full
    .sort_values(
        "mbnsf_pred_sc_score",
        ascending=True,
    )
    .reset_index(drop=True)
)

full["global_severity_rank"] = (
    np.arange(len(full)) + 1
)

full["episode_id"] = full[
    "sample_key"
].map(
    episode_id_from_key
)

selected_rows = []
seen_episode = set()

for _, row in full.iterrows():

    ep = row["episode_id"]

    if ep in seen_episode:
        continue

    selected_rows.append(
        row
    )

    seen_episode.add(
        ep
    )

    if len(selected_rows) == TOP_K:
        break

assert len(selected_rows) == TOP_K

selected = pd.DataFrame(
    selected_rows
).reset_index(drop=True)

selected["severe_case_rank"] = (
    np.arange(TOP_K) + 1
)

assert selected["episode_id"].nunique() == TOP_K


print("=" * 100)
print("SELECTED EPISODE-DISJOINT TOP-5")
print("=" * 100)

print(
    selected[
        [
            "severe_case_rank",
            "global_severity_rank",
            "sample_key",
            "episode_id",
            "object_id",
            "mbnsf_pred_sc_score",
            "mbnsf_gt_sc_score",
        ]
    ].to_string(index=False)
)


# ============================================================
# Helper: explicit pairwise distance path
# matching previous successful localization
# ============================================================

def pair_dist(x):
    return torch.norm(
        x[:, :, None, :]
        - x[:, None, :, :],
        dim=-1,
    )


def mbnsf_from_arrays(src_np, tgt_np):

    src = (
        torch.from_numpy(
            src_np.astype(np.float64)
        )
        .float()
        .unsqueeze(0)
    )

    tgt = (
        torch.from_numpy(
            tgt_np.astype(np.float64)
        )
        .float()
        .unsqueeze(0)
    )

    with torch.no_grad():

        loss = spatial_consistency_loss(
            src,
            tgt,
            d_thre=D_THRE,
            max_points=MAX_POINTS,
        )

        loss_value = float(
            loss.item()
        )

        score = float(
            np.exp(
                -loss_value
            )
        )

        src_dist = pair_dist(src)
        tgt_dist = pair_dist(tgt)

        cross_dist = torch.abs(
            src_dist - tgt_dist
        )

        adj = torch.clamp(
            1.0
            - cross_dist ** 2
            / D_THRE ** 2,
            min=0,
        )

    return {
        "loss": loss_value,
        "score": score,
        "cross_dist": (
            cross_dist[0]
            .cpu()
            .numpy()
        ),
        "adj": (
            adj[0]
            .cpu()
            .numpy()
        ),
    }


# ============================================================
# 4. Analyze selected cases
# ============================================================

summary_rows = []
temporal_rows = []

for _, sel in selected.iterrows():

    case_rank = int(
        sel["severe_case_rank"]
    )

    global_rank = int(
        sel["global_severity_rank"]
    )

    key = str(
        sel["sample_key"]
    )

    episode_id = str(
        sel["episode_id"]
    )

    object_id = int(
        sel["object_id"]
    )

    old_pred_score = float(
        sel["mbnsf_pred_sc_score"]
    )

    old_gt_score = float(
        sel["mbnsf_gt_sc_score"]
    )

    path = (
        NPZ_ROOT /
        (
            key.replace(
                ":",
                "_",
            )
            + ".npz"
        )
    )

    assert path.exists(), path

    d = np.load(
        path,
        allow_pickle=True,
    )

    assert str(
        d["sample_key"].item()
    ) == key

    assert bool(
        d["official_l2_exact"].item()
    )

    assert bool(
        d["temporal_object_id_exact"].item()
    )

    gt = d[
        "gt_scene_flows"
    ].astype(np.float64)

    pred = d[
        "pred_scene_flows"
    ].astype(np.float64)

    obj = d[
        "scene_object_id"
    ]

    valid = d[
        "official_valid_mask"
    ]

    assert gt.shape == pred.shape
    assert gt.shape[0] == 11

    # Exact same population rule as previous radio MBNSF.
    mask = (
        (obj[0] == object_id)
        & valid[-1]
    )

    n = int(
        mask.sum()
    )

    assert n > 1

    # No MBNSF random subsampling allowed in this diagnostic.
    assert n < MAX_POINTS, (
        key,
        n,
    )

    x0 = gt[
        0,
        mask,
    ]

    gt10 = gt[
        -1,
        mask,
    ]

    pred10 = pred[
        -1,
        mask,
    ]

    # --------------------------------------------------------
    # t10 reproduction
    # --------------------------------------------------------

    gt_res = mbnsf_from_arrays(
        x0,
        gt10,
    )

    pred_res = mbnsf_from_arrays(
        x0,
        pred10,
    )

    gt_score_diff = abs(
        gt_res["score"]
        - old_gt_score
    )

    pred_score_diff = abs(
        pred_res["score"]
        - old_pred_score
    )

    assert gt_score_diff < 1e-6, (
        key,
        gt_res["score"],
        old_gt_score,
        gt_score_diff,
    )

    assert pred_score_diff < 1e-6, (
        key,
        pred_res["score"],
        old_pred_score,
        pred_score_diff,
    )

    # --------------------------------------------------------
    # Unique pairs
    # --------------------------------------------------------

    ii, jj = np.triu_indices(
        n,
        k=1,
    )

    n_pairs = len(ii)

    assert n_pairs == (
        n * (n - 1) // 2
    )

    gt_cross_mm = (
        gt_res[
            "cross_dist"
        ][ii, jj]
        * 1000.0
    )

    pred_cross_mm = (
        pred_res[
            "cross_dist"
        ][ii, jj]
        * 1000.0
    )

    gt_pair_adj = (
        gt_res[
            "adj"
        ][ii, jj]
    )

    pred_pair_adj = (
        pred_res[
            "adj"
        ][ii, jj]
    )

    gt_zero_fraction = float(
        np.mean(
            gt_pair_adj == 0
        )
    )

    pred_zero_fraction = float(
        np.mean(
            pred_pair_adj == 0
        )
    )

    # --------------------------------------------------------
    # Temporal MBNSF t1..t10
    # --------------------------------------------------------

    temporal_pred_scores = []

    for t in range(1, 11):

        gt_t = gt[
            t,
            mask,
        ]

        pred_t = pred[
            t,
            mask,
        ]

        gt_t_res = mbnsf_from_arrays(
            x0,
            gt_t,
        )

        pred_t_res = mbnsf_from_arrays(
            x0,
            pred_t,
        )

        temporal_pred_scores.append(
            pred_t_res["score"]
        )

        p_adj = (
            pred_t_res["adj"][
                ii,
                jj,
            ]
        )

        g_adj = (
            gt_t_res["adj"][
                ii,
                jj,
            ]
        )

        temporal_rows.append(
            {
                "case_rank": case_rank,
                "global_severity_rank": global_rank,
                "sample_key": key,
                "episode_id": episode_id,
                "object_id": object_id,
                "n_points": n,
                "timestep": t,
                "gt_mbnsf_score": gt_t_res["score"],
                "pred_mbnsf_score": pred_t_res["score"],
                "gt_zero_pair_fraction": float(
                    np.mean(
                        g_adj == 0
                    )
                ),
                "pred_zero_pair_fraction": float(
                    np.mean(
                        p_adj == 0
                    )
                ),
            }
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows.append(
        {
            "case_rank": case_rank,
            "global_severity_rank": global_rank,
            "sample_key": key,
            "episode_id": episode_id,
            "object_id": object_id,
            "n_points": n,
            "n_unique_pairs": n_pairs,

            "official_gt_mbnsf_score": old_gt_score,
            "recomputed_gt_mbnsf_score": gt_res["score"],
            "gt_score_abs_diff": gt_score_diff,

            "official_pred_mbnsf_score": old_pred_score,
            "recomputed_pred_mbnsf_score": pred_res["score"],
            "pred_score_abs_diff": pred_score_diff,

            "gt_zero_pair_fraction_t10": gt_zero_fraction,
            "pred_zero_pair_fraction_t10": pred_zero_fraction,

            "gt_pair_change_median_mm": float(
                np.median(
                    gt_cross_mm
                )
            ),
            "gt_pair_change_p95_mm": float(
                np.quantile(
                    gt_cross_mm,
                    0.95,
                )
            ),
            "gt_pair_change_max_mm": float(
                np.max(
                    gt_cross_mm
                )
            ),

            "pred_pair_change_median_mm": float(
                np.median(
                    pred_cross_mm
                )
            ),
            "pred_pair_change_p95_mm": float(
                np.quantile(
                    pred_cross_mm,
                    0.95,
                )
            ),
            "pred_pair_change_max_mm": float(
                np.max(
                    pred_cross_mm
                )
            ),

            "pred_min_temporal_mbnsf_score": float(
                np.min(
                    temporal_pred_scores
                )
            ),

            "pred_t10_mbnsf_score": float(
                temporal_pred_scores[-1]
            ),

            "official_l2_exact": bool(
                d["official_l2_exact"].item()
            ),

            "temporal_object_id_exact": bool(
                d["temporal_object_id_exact"].item()
            ),
        }
    )

    print()
    print(
        f"[case {case_rank}] "
        f"global_rank={global_rank} "
        f"n={n} "
        f"GT={gt_res['score']:.6f} "
        f"Pred={pred_res['score']:.6f} "
        f"Pred zero={pred_zero_fraction:.3f}"
    )


summary = pd.DataFrame(
    summary_rows
).sort_values(
    "case_rank"
)

temporal = pd.DataFrame(
    temporal_rows
).sort_values(
    [
        "case_rank",
        "timestep",
    ]
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

temporal.to_csv(
    OUT_TEMPORAL,
    index=False,
)


# ============================================================
# 5. Visualization A:
# final GT vs prediction MBNSF score
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.3)
)

x = np.arange(
    1,
    TOP_K + 1,
)

ax.plot(
    x,
    summary[
        "official_gt_mbnsf_score"
    ],
    marker="o",
    label="GT t10",
)

ax.plot(
    x,
    summary[
        "official_pred_mbnsf_score"
    ],
    marker="o",
    label="Prediction t10",
)

ax.set_xticks(x)

ax.set_xlabel(
    "Episode-disjoint severe case rank"
)

ax.set_ylabel(
    "Official MBNSF spatial-consistency score"
)

ax.set_ylim(
    0,
    1.03,
)

ax.set_title(
    "Top-5 episode-disjoint radio: GT rigidity vs predicted structural integrity"
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT_SCORE_PNG,
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 6. Visualization B:
# prediction temporal MBNSF
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.0, 5.5)
)

for case_rank in range(
    1,
    TOP_K + 1,
):

    z = temporal[
        temporal[
            "case_rank"
        ] == case_rank
    ]

    ax.plot(
        z["timestep"],
        z["pred_mbnsf_score"],
        marker="o",
        label=f"Case {case_rank}",
    )

ax.set_xticks(
    np.arange(
        1,
        11,
    )
)

ax.set_ylim(
    0,
    1.03,
)

ax.set_xlabel(
    "Future timestep"
)

ax.set_ylabel(
    "Official MBNSF spatial-consistency score"
)

ax.set_title(
    "Top-5 episode-disjoint radio: temporal structural consistency"
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    ncol=2,
)

fig.tight_layout()

fig.savefig(
    OUT_TEMPORAL_PNG,
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 7. Visualization C:
# t10 zero-compatible pair fraction
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.2)
)

ax.bar(
    x,
    summary[
        "pred_zero_pair_fraction_t10"
    ] * 100.0,
)

ax.set_xticks(x)

ax.set_xlabel(
    "Episode-disjoint severe case rank"
)

ax.set_ylabel(
    "Predicted MBNSF-zero pair fraction at t10 (%)"
)

ax.set_title(
    "Top-5 episode-disjoint radio: fraction of structurally incompatible relations"
)

ax.grid(
    True,
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    OUT_ZERO_PNG,
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 8. Final audit
# ============================================================

print()
print("=" * 110)
print("TOP-5 EPISODE-DISJOINT RADIO STRUCTURAL REPEATABILITY")
print("=" * 110)

print(
    summary[
        [
            "case_rank",
            "global_severity_rank",
            "sample_key",
            "n_points",
            "official_gt_mbnsf_score",
            "official_pred_mbnsf_score",
            "gt_zero_pair_fraction_t10",
            "pred_zero_pair_fraction_t10",
            "gt_pair_change_p95_mm",
            "pred_pair_change_p95_mm",
            "pred_pair_change_max_mm",
        ]
    ].to_string(
        index=False
    )
)

print()
print("n_cases =", len(summary))
print(
    "unique_episodes =",
    summary[
        "episode_id"
    ].nunique(),
)

print(
    "official_l2_exact_all =",
    bool(
        summary[
            "official_l2_exact"
        ].all()
    ),
)

print(
    "temporal_object_id_exact_all =",
    bool(
        summary[
            "temporal_object_id_exact"
        ].all()
    ),
)

print(
    "official_mbnsf_t10_reproduced_all =",
    bool(
        (
            summary[
                "pred_score_abs_diff"
            ] < 1e-6
        ).all()
        and
        (
            summary[
                "gt_score_abs_diff"
            ] < 1e-6
        ).all()
    ),
)

print()
print("saved =", OUT_SUMMARY)
print("saved =", OUT_TEMPORAL)
print("saved =", OUT_SCORE_PNG)
print("saved =", OUT_TEMPORAL_PNG)
print("saved =", OUT_ZERO_PNG)

print()
print("NO NEW INFERENCE")
print("OFFICIAL MBNSF IMPLEMENTATION REUSED")
print("EPISODE-DISJOINT SELECTION")
print("NO MANUAL CASE SELECTION")
print("SUCCESS = True")
print("=" * 110)
