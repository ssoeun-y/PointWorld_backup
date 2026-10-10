from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr, wilcoxon


# ============================================================
# Load the established LOTO structural evaluator unchanged
# ============================================================

HERE = Path(__file__).resolve().parent
BASE_FILE = HERE / "eval_goal_preserved_loto_pointcloud_structure.py"

spec = importlib.util.spec_from_file_location(
    "loto_structural_base",
    BASE_FILE,
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


# Preserve established implementations
_original_build_graph = base.build_fixed_local_graph
_original_structural_scores = base.structural_scores


# ============================================================
# Attach observed t0 coordinates to the SAME established graph
# ============================================================

def build_graph_with_x0(x0, support, k=base.K, gate_q=base.EDGE_GATE_Q):
    graph = _original_build_graph(
        x0,
        support,
        k=k,
        gate_q=gate_q,
    )

    if graph is not None:
        # Used only during the current sample evaluation.
        graph["_kabsch_x0"] = x0.detach()

    return graph


# ============================================================
# Best-fit local SE(3) residual
#
# For each graph node:
#   center + its graph neighbors = one local patch.
#
# Fit:
#   observed t0 local patch -> predicted local patch
# by best rigid transform (Kabsch).
#
# Score:
#   RMS point residual after best SE(3) alignment.
#
# Lower = local geometry is better explained by rigid motion.
# ============================================================

def _fit_kabsch_rms_mm(src, dst):
    # src, dst: [M, 3]
    if src.shape[0] < 3:
        return None

    src = src.float()
    dst = dst.float()

    if (
        not torch.isfinite(src).all()
        or not torch.isfinite(dst).all()
    ):
        return None

    src_c = src - src.mean(dim=0, keepdim=True)
    dst_c = dst - dst.mean(dim=0, keepdim=True)

    # Skip degenerate local geometry.
    # Three non-collinear points have centered rank >= 2.
    rank = torch.linalg.matrix_rank(src_c)

    if int(rank.item()) < 2:
        return None

    H = src_c.transpose(0, 1) @ dst_c

    try:
        U, _, Vh = torch.linalg.svd(H)
    except RuntimeError:
        return None

    R = U @ Vh

    # Enforce proper rotation: det(R) = +1, no reflection.
    if float(torch.linalg.det(R).item()) < 0.0:
        U_fix = U.clone()
        U_fix[:, -1] *= -1.0
        R = U_fix @ Vh

    aligned = src_c @ R

    residual = torch.linalg.vector_norm(
        aligned - dst_c,
        dim=-1,
    )

    rms_m = torch.sqrt(
        torch.mean(residual ** 2)
    )

    return float(rms_m.detach().cpu()) * 1000.0


def kabsch_scores(pred, valid, graph):
    x0 = graph["_kabsch_x0"]

    gi = graph["i"].detach().cpu().tolist()
    gj = graph["j"].detach().cpu().tolist()

    # Undirected adjacency from the exact established local graph.
    adjacency = {}

    for a, b in zip(gi, gj):
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))

    all_scores = []
    final_scores = []

    T = int(pred.shape[0])

    for center, nbrs in adjacency.items():
        nodes = [center] + sorted(nbrs)

        if len(nodes) < 3:
            continue

        nodes = torch.as_tensor(
            nodes,
            device=pred.device,
            dtype=torch.long,
        )

        for t in range(T):
            keep = valid[t, nodes]
            ids = nodes[keep]

            if int(ids.numel()) < 3:
                continue

            score = _fit_kabsch_rms_mm(
                x0[ids],
                pred[t, ids],
            )

            if score is None or not np.isfinite(score):
                continue

            all_scores.append(score)

            if t == T - 1:
                final_scores.append(score)

    if not all_scores:
        return {
            "kabsch_se3_mean_mm": float("nan"),
            "kabsch_se3_median_mm": float("nan"),
            "kabsch_se3_p95_mm": float("nan"),
            "final_kabsch_se3_mean_mm": float("nan"),
        }

    a = np.asarray(all_scores, dtype=float)

    if final_scores:
        f = np.asarray(final_scores, dtype=float)
        final_mean = float(np.mean(f))
    else:
        final_mean = float("nan")

    return {
        "kabsch_se3_mean_mm":
            float(np.mean(a)),
        "kabsch_se3_median_mm":
            float(np.median(a)),
        "kabsch_se3_p95_mm":
            float(np.quantile(a, 0.95)),
        "final_kabsch_se3_mean_mm":
            final_mean,
    }


def structural_scores_with_kabsch(pred, valid, graph):
    scores = _original_structural_scores(
        pred,
        valid,
        graph,
    )

    if scores is None:
        return None

    scores.update(
        kabsch_scores(
            pred,
            valid,
            graph,
        )
    )

    return scores


# ============================================================
# Patch only the two functions used by base.main()
# ============================================================

base.build_fixed_local_graph = build_graph_with_x0
base.structural_scores = structural_scores_with_kabsch

# Never overwrite the established structural CSV.
base.OUT = (
    base.OUTDIR
    / f"{base.TAG}__pointcloud_structure_kabsch.csv"
)


# ============================================================
# Statistical sanity summary
# ============================================================

def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[good]
    y = y[good]

    if len(x) < 3:
        return len(x), np.nan, np.nan

    r = spearmanr(x, y)

    return len(x), float(r.statistic), float(r.pvalue)


def report():
    df = pd.read_csv(base.OUT)

    ok = df[df["status"] == "OK"].copy()

    metric = "kabsch_se3_mean_mm"

    a = ok[f"original_{metric}"].to_numpy(float)
    b = ok[f"corrected_{metric}"].to_numpy(float)

    good = np.isfinite(a) & np.isfinite(b)
    a = a[good]
    b = b[good]

    print()
    print("=" * 100)
    print("KABSCH SE(3) SANITY VALIDATION")
    print("=" * 100)

    print("N =", len(a))

    print()
    print("PAIRED KABSCH RESIDUAL")
    print("  original median =", float(np.median(a)))
    print("  corrected median =", float(np.median(b)))

    ratio = b / np.maximum(a, 1e-12)

    print(
        "  median paired ratio =",
        float(np.median(ratio)),
    )
    print(
        "  corrected lower fraction =",
        float(np.mean(b < a)),
    )

    diff = b - a

    nz = diff != 0

    if int(nz.sum()) > 0:
        w = wilcoxon(
            b[nz],
            a[nz],
            alternative="two-sided",
        )
        print(
            "  Wilcoxon p =",
            float(w.pvalue),
        )
    else:
        print("  Wilcoxon p = all paired differences zero")

    print()
    print("CORRELATION WITH LOCAL ISOMETRY ERROR")
    print("(existing rigidity_mean_mm)")

    n, rho, p = safe_spearman(
        ok["original_rigidity_mean_mm"],
        ok["original_kabsch_se3_mean_mm"],
    )
    print(
        f"  Original  : N={n}, rho={rho:.6f}, p={p:.6g}"
    )

    n, rho, p = safe_spearman(
        ok["corrected_rigidity_mean_mm"],
        ok["corrected_kabsch_se3_mean_mm"],
    )
    print(
        f"  Corrected : N={n}, rho={rho:.6f}, p={p:.6g}"
    )

    # Does improvement in our local-isometry diagnostic agree
    # with improvement in independently fitted SE(3) residual?
    iso_improve = (
        ok["original_rigidity_mean_mm"]
        - ok["corrected_rigidity_mean_mm"]
    )

    kab_improve = (
        ok["original_kabsch_se3_mean_mm"]
        - ok["corrected_kabsch_se3_mean_mm"]
    )

    n, rho, p = safe_spearman(
        iso_improve,
        kab_improve,
    )

    print(
        f"  Improvement correlation: "
        f"N={n}, rho={rho:.6f}, p={p:.6g}"
    )

    print()
    print("TASKWISE KABSCH")

    for task, g in ok.groupby("task"):
        x = g["original_kabsch_se3_mean_mm"].to_numpy(float)
        y = g["corrected_kabsch_se3_mean_mm"].to_numpy(float)

        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]

        if len(x) == 0:
            continue

        print(
            task,
            "| N =", len(x),
            "| orig =", float(np.median(x)),
            "| corr =", float(np.median(y)),
            "| ratio =",
            float(np.median(y / np.maximum(x, 1e-12))),
            "| lower =",
            float(np.mean(y < x)),
        )

    print()
    print("saved =", base.OUT)


if __name__ == "__main__":
    base.main()
    report()
