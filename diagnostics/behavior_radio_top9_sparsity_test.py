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
    "behavior_radio_top9_sparsity_test"
)
os.makedirs(OUTDIR, exist_ok=True)

K = 8
TOP_FRAC = 0.20
BOTTOM_FRAC = 0.20

rank_df = pd.read_csv(RANK_CSV).head(9).copy()


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
    """
    각 point에서 자기 자신을 제외한 k개 최근접 GT t0 point까지의
    거리 median.
    값이 클수록 local point support가 sparse함.
    """
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    # self distance 제외
    np.fill_diagonal(dist, np.inf)

    knn = np.partition(dist, kth=k-1, axis=1)[:, :k]

    return np.median(knn, axis=1)


def rankdata_average(x):
    """
    scipy 없이 average-tie rank 계산.
    """
    return pd.Series(x).rank(
        method="average"
    ).to_numpy(dtype=np.float64)


def spearman_np(a, b):
    ra = rankdata_average(a)
    rb = rankdata_average(b)

    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan

    return np.corrcoef(ra, rb)[0, 1]


rows = []

for rank_idx, row in rank_df.iterrows():

    key = row["sample_key"]
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

    mask = (obj[0] == object_id) & valid[-1]

    x0 = gt[0, mask]
    xgt = gt[-1, mask]
    xpred = pred[-1, mask]

    n = len(x0)

    assert n == int(row["n_points"])
    assert n > K

    # --------------------------------------------------------
    # Final non-rigid deformation
    # --------------------------------------------------------
    pred_aligned = rigid_align(xpred, xgt)

    residual_mm = (
        np.linalg.norm(pred_aligned - xgt, axis=1)
        * 1000.0
    )

    # --------------------------------------------------------
    # GT t0 local sparsity
    # --------------------------------------------------------
    sparsity_mm = (
        local_sparsity_knn(x0, k=K)
        * 1000.0
    )

    # --------------------------------------------------------
    # sample 내부 residual quantile comparison
    # --------------------------------------------------------
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

    high_sparse = np.median(sparsity_mm[high])
    low_sparse = np.median(sparsity_mm[low])

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

    rows.append({
        "rank": rank_idx + 1,
        "sample_key": key,
        "n_points": n,
        "high_n": int(high.sum()),
        "low_n": int(low.sum()),

        "high_residual_sparsity_median_mm":
            high_sparse,

        "low_residual_sparsity_median_mm":
            low_sparse,

        "sparsity_gap_mm":
            gap,

        "sparsity_ratio_high_over_low":
            ratio,

        "spearman_residual_vs_sparsity":
            rho,
    })


out = pd.DataFrame(rows)

csv_path = os.path.join(
    OUTDIR,
    "top9_sparsity_comparison.csv"
)

out.to_csv(csv_path, index=False)

print("=" * 110)
print("TOP-9 LOCAL SPARSITY TEST")
print("K =", K)
print(
    "HIGH = top 20% non-rigid residual, "
    "LOW = bottom 20%"
)
print("=" * 110)

for _, r in out.iterrows():
    print(
        f"rank={int(r['rank']):2d} "
        f"n={int(r['n_points']):4d} | "
        f"high_sparse={r['high_residual_sparsity_median_mm']:.3f}mm "
        f"low_sparse={r['low_residual_sparsity_median_mm']:.3f}mm | "
        f"gap={r['sparsity_gap_mm']:+.3f}mm "
        f"ratio={r['sparsity_ratio_high_over_low']:.3f} | "
        f"rho={r['spearman_residual_vs_sparsity']:+.3f}"
    )

positive_gap = (
    out["sparsity_gap_mm"] > 0
).sum()

positive_rho = (
    out["spearman_residual_vs_sparsity"] > 0
).sum()

print("\n" + "=" * 110)

print(
    "POSITIVE_SPARSITY_GAP =",
    positive_gap,
    "/",
    len(out)
)

print(
    "POSITIVE_SPEARMAN =",
    positive_rho,
    "/",
    len(out)
)

print(
    "MEDIAN_SPARSITY_RATIO =",
    out[
        "sparsity_ratio_high_over_low"
    ].median()
)

print(
    "MEDIAN_SPEARMAN =",
    out[
        "spearman_residual_vs_sparsity"
    ].median()
)

# ------------------------------------------------------------
# 간단한 sample-wise figure
# ratio > 1이면 high-deformation point가 더 sparse
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))

plt.bar(
    out["rank"].astype(str),
    out["sparsity_ratio_high_over_low"]
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Rigidity-failure rank")
plt.ylabel(
    "Local sparsity ratio\n"
    "(high residual / low residual)"
)

plt.title(
    "Are highly deformed radio points locally sparser?"
)

png_path = os.path.join(
    OUTDIR,
    "top9_sparsity_ratio.png"
)

plt.tight_layout()
plt.savefig(
    png_path,
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\nCSV =", csv_path)
print("FIGURE =", png_path)
print("CSV_EXISTS =", os.path.exists(csv_path))
print("FIGURE_EXISTS =", os.path.exists(png_path))
print("TOP9_SPARSITY_TEST_SUCCESS = True")
