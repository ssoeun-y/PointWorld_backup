import os
import math
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
    "behavior_radio_full554_sparsity_test"
)
os.makedirs(OUTDIR, exist_ok=True)

K = 8
TOP_FRAC = 0.20
BOTTOM_FRAC = 0.20
BOOT_N = 10000
SEED = 0

rank_df = pd.read_csv(RANK_CSV).copy()

assert len(rank_df) == 554, len(rank_df)


def rigid_align(src, dst):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    sc = src.mean(axis=0)
    dc = dst.mean(axis=0)

    X = src - sc
    Y = dst - dc

    U, S, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = dc - sc @ R.T
    return src @ R.T + t


def local_sparsity_knn(xyz, k=8):
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    np.fill_diagonal(dist, np.inf)

    knn = np.partition(
        dist,
        kth=k - 1,
        axis=1
    )[:, :k]

    # robust local spacing
    return np.median(knn, axis=1)


def spearman_np(a, b):
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)

    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan

    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_median_ci(values, n_boot=10000, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    rng = np.random.default_rng(seed)

    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )
        boot[i] = np.median(sample)

    return (
        float(np.median(values)),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def binomial_upper_tail(k, n, p=0.5):
    # P[X >= k] under Binomial(n,p)
    return sum(
        math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        for i in range(k, n + 1)
    )


rows = []

for idx, row in rank_df.iterrows():

    key = str(row["sample_key"])
    object_id = int(row["object_id"])

    path = os.path.join(
        NPZ_ROOT,
        key.replace(":", "_") + ".npz"
    )

    assert os.path.exists(path), path

    d = np.load(path, allow_pickle=True)

    assert d["official_l2_exact"].item()
    assert d["temporal_object_id_exact"].item()

    gt = d["gt_scene_flows"].astype(np.float64)
    pred = d["pred_scene_flows"].astype(np.float64)
    obj = d["scene_object_id"]
    valid = d["official_valid_mask"]

    mask = (
        (obj[0] == object_id)
        & valid[-1]
    )

    x0 = gt[0, mask]
    xgt = gt[-1, mask]
    xpred = pred[-1, mask]

    n = len(x0)

    assert n == int(row["n_points"])
    assert n >= 20

    # --------------------------------------------------------
    # final non-rigid residual
    # --------------------------------------------------------
    pred_aligned = rigid_align(xpred, xgt)

    residual_mm = (
        np.linalg.norm(
            pred_aligned - xgt,
            axis=1
        ) * 1000.0
    )

    # --------------------------------------------------------
    # GT t0 local sparsity
    # --------------------------------------------------------
    sparsity_mm = (
        local_sparsity_knn(x0, K)
        * 1000.0
    )

    q_hi = np.quantile(
        residual_mm,
        1.0 - TOP_FRAC
    )

    q_lo = np.quantile(
        residual_mm,
        BOTTOM_FRAC
    )

    high = residual_mm >= q_hi
    low = residual_mm <= q_lo

    high_sparse = float(
        np.median(sparsity_mm[high])
    )

    low_sparse = float(
        np.median(sparsity_mm[low])
    )

    gap = high_sparse - low_sparse

    ratio = (
        high_sparse / low_sparse
        if low_sparse > 0
        else np.nan
    )

    rho = spearman_np(
        residual_mm,
        sparsity_mm
    )

    # same episode can have multiple temporal windows
    episode = key.rsplit("-", 1)[0]

    rows.append({
        "sample_key": key,
        "episode": episode,
        "n_points": n,
        "pred_rigid_mean_mm":
            float(np.mean(residual_mm)),
        "high_sparsity_median_mm":
            high_sparse,
        "low_sparsity_median_mm":
            low_sparse,
        "sparsity_gap_mm":
            gap,
        "sparsity_ratio_high_over_low":
            ratio,
        "spearman_residual_vs_sparsity":
            rho,
    })


out = pd.DataFrame(rows)

assert len(out) == 554

sample_csv = os.path.join(
    OUTDIR,
    "radio_full554_sparsity_per_sample.csv"
)
out.to_csv(sample_csv, index=False)


# ============================================================
# SAMPLE-LEVEL SUMMARY
# ============================================================

positive_gap = int(
    (out["sparsity_gap_mm"] > 0).sum()
)

positive_rho = int(
    (
        out["spearman_residual_vs_sparsity"]
        > 0
    ).sum()
)

sample_ratio_med = float(
    out[
        "sparsity_ratio_high_over_low"
    ].median()
)

sample_rho_med = float(
    out[
        "spearman_residual_vs_sparsity"
    ].median()
)


# ============================================================
# EPISODE-LEVEL ROBUSTNESS
#
# Multiple windows from the same episode are not treated as
# independent. Aggregate each episode first.
# ============================================================

episode_df = (
    out.groupby("episode", as_index=False)
    .agg({
        "sparsity_gap_mm": "median",
        "sparsity_ratio_high_over_low": "median",
        "spearman_residual_vs_sparsity": "median",
        "pred_rigid_mean_mm": "median",
        "sample_key": "count",
    })
    .rename(columns={
        "sample_key": "n_windows"
    })
)

episode_csv = os.path.join(
    OUTDIR,
    "radio_sparsity_per_episode.csv"
)
episode_df.to_csv(
    episode_csv,
    index=False
)

n_episode = len(episode_df)

episode_positive_gap = int(
    (episode_df["sparsity_gap_mm"] > 0).sum()
)

episode_positive_rho = int(
    (
        episode_df[
            "spearman_residual_vs_sparsity"
        ] > 0
    ).sum()
)

ratio_med, ratio_lo, ratio_hi = (
    bootstrap_median_ci(
        episode_df[
            "sparsity_ratio_high_over_low"
        ],
        n_boot=BOOT_N,
        seed=SEED,
    )
)

rho_med, rho_lo, rho_hi = (
    bootstrap_median_ci(
        episode_df[
            "spearman_residual_vs_sparsity"
        ],
        n_boot=BOOT_N,
        seed=SEED + 1,
    )
)

sign_p = binomial_upper_tail(
    episode_positive_gap,
    n_episode,
    0.5
)


print("=" * 110)
print("FULL RADIO SPARSITY TEST")
print("K =", K)
print("RADIO_WINDOWS =", len(out))
print("UNIQUE_EPISODES =", n_episode)
print(
    "HIGH = top 20% residual; "
    "LOW = bottom 20% residual"
)

print("\nSAMPLE-LEVEL")
print(
    "POSITIVE_SPARSITY_GAP =",
    positive_gap,
    "/",
    len(out),
    f"({100*positive_gap/len(out):.2f}%)"
)
print(
    "POSITIVE_SPEARMAN =",
    positive_rho,
    "/",
    len(out),
    f"({100*positive_rho/len(out):.2f}%)"
)
print(
    "MEDIAN_SPARSITY_RATIO =",
    sample_ratio_med
)
print(
    "MEDIAN_SPEARMAN =",
    sample_rho_med
)

print("\nEPISODE-LEVEL ROBUSTNESS")
print(
    "POSITIVE_EPISODE_GAP =",
    episode_positive_gap,
    "/",
    n_episode,
    f"({100*episode_positive_gap/n_episode:.2f}%)"
)
print(
    "POSITIVE_EPISODE_SPEARMAN =",
    episode_positive_rho,
    "/",
    n_episode,
    f"({100*episode_positive_rho/n_episode:.2f}%)"
)

print(
    "EPISODE_MEDIAN_SPARSITY_RATIO =",
    ratio_med
)
print(
    "BOOTSTRAP_95CI_RATIO =",
    f"[{ratio_lo:.6f}, {ratio_hi:.6f}]"
)

print(
    "EPISODE_MEDIAN_SPEARMAN =",
    rho_med
)
print(
    "BOOTSTRAP_95CI_SPEARMAN =",
    f"[{rho_lo:.6f}, {rho_hi:.6f}]"
)

print(
    "EPISODE_SIGN_TEST_P =",
    sign_p
)


# ============================================================
# FIGURE: sample-balanced distribution
# ============================================================

plt.figure(figsize=(9, 6))

ratios = out[
    "sparsity_ratio_high_over_low"
].dropna()

plt.hist(
    ratios,
    bins=40
)

plt.axvline(
    1.0,
    linestyle="--",
    linewidth=2,
    label="No sparsity difference (ratio=1)"
)

plt.axvline(
    sample_ratio_med,
    linestyle="-",
    linewidth=2,
    label=f"Median={sample_ratio_med:.3f}"
)

plt.xlabel(
    "Local sparsity ratio "
    "(high deformation / low deformation)"
)
plt.ylabel("Number of radio windows")
plt.title(
    "Radio-wide association between local sparsity "
    "and non-rigid deformation"
)
plt.legend()

fig_path = os.path.join(
    OUTDIR,
    "radio_full554_sparsity_ratio_hist.png"
)

plt.tight_layout()
plt.savefig(
    fig_path,
    dpi=200,
    bbox_inches="tight"
)
plt.close()


print("\nSAMPLE_CSV =", sample_csv)
print("EPISODE_CSV =", episode_csv)
print("FIGURE =", fig_path)

print(
    "FILES_EXIST =",
    os.path.exists(sample_csv)
    and os.path.exists(episode_csv)
    and os.path.exists(fig_path)
)

print(
    "FULL554_SPARSITY_TEST_SUCCESS = True"
)
