from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

METRIC = Path(
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)
COV = Path(
    "diagnostics/results/action_confound_control_5task/"
    "per_sample_covariates_7987.csv"
)
OUT = Path(
    "diagnostics/results/action_confound_control_5task/"
    "partial_action_dynamics.csv"
)

def rank01(s):
    return s.rank(method="average", pct=True)

def residualize(y, X):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta

def partial_rank_corr(g, x, y, controls):
    cols = [x, y] + controls
    d = g[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    # Spearman-style: rank first
    for c in cols:
        d[c] = rank01(d[c])

    rx = residualize(d[x], d[controls])
    ry = residualize(d[y], d[controls])

    r, p = pearsonr(rx, ry)
    return len(d), r, p

def main():
    m = pd.read_csv(METRIC)
    c = pd.read_csv(COV)

    df = m.merge(c, on="clip", how="inner", validate="one_to_one")

    if "task_loaded" in df.columns:
        assert (df["task"] == df["task_loaded"]).all()

    assert len(df) == 7987, len(df)

    # mean speed deliberately omitted:
    # with fixed frame count it is proportional to path length.
    motion_controls = [
        "gt_moved_motion_mm",
        "active_path_length_m",
        "active_net_displacement_m",
    ]

    rows = []

    scopes = [("ALL", df)]
    scopes += [(task, g.copy()) for task, g in df.groupby("task")]

    for scope, g in scopes:
        # For ALL, rank within task before pooling to remove task-level scale.
        if scope == "ALL":
            ranked_parts = []
            needed = [
                "mean_squared_jerk",
                "rms_acceleration",
                "l2_moved_mm",
            ] + motion_controls
            for _, gt in g.groupby("task"):
                gt = gt.copy()
                for col in needed:
                    gt[col] = rank01(gt[col])
                ranked_parts.append(gt)
            gp = pd.concat(ranked_parts, ignore_index=True)

            def calc(x, controls):
                d = gp[[x, "l2_moved_mm"] + controls].dropna()
                rx = residualize(d[x], d[controls])
                ry = residualize(d["l2_moved_mm"], d[controls])
                r, p = pearsonr(rx, ry)
                return len(d), r, p

            # raw task-adjusted rank association between MSJ and RMS
            r_mr, p_mr = pearsonr(
                gp["mean_squared_jerk"],
                gp["rms_acceleration"]
            )
            rows.append({
                "scope": scope,
                "analysis": "MSJ_vs_RMS",
                "N": len(gp),
                "partial_rho": r_mr,
                "p": p_mr,
            })

            n, r, p = calc(
                "mean_squared_jerk",
                ["rms_acceleration"] + motion_controls
            )
            rows.append({
                "scope": scope,
                "analysis": "MSJ_to_error_control_RMS+motion",
                "N": n,
                "partial_rho": r,
                "p": p,
            })

            n, r, p = calc(
                "rms_acceleration",
                ["mean_squared_jerk"] + motion_controls
            )
            rows.append({
                "scope": scope,
                "analysis": "RMS_to_error_control_MSJ+motion",
                "N": n,
                "partial_rho": r,
                "p": p,
            })

        else:
            rr = g[
                ["mean_squared_jerk", "rms_acceleration"]
            ].dropna()
            r_mr, p_mr = pearsonr(
                rank01(rr["mean_squared_jerk"]),
                rank01(rr["rms_acceleration"])
            )
            rows.append({
                "scope": scope,
                "analysis": "MSJ_vs_RMS",
                "N": len(rr),
                "partial_rho": r_mr,
                "p": p_mr,
            })

            n, r, p = partial_rank_corr(
                g,
                "mean_squared_jerk",
                "l2_moved_mm",
                ["rms_acceleration"] + motion_controls
            )
            rows.append({
                "scope": scope,
                "analysis": "MSJ_to_error_control_RMS+motion",
                "N": n,
                "partial_rho": r,
                "p": p,
            })

            n, r, p = partial_rank_corr(
                g,
                "rms_acceleration",
                "l2_moved_mm",
                ["mean_squared_jerk"] + motion_controls
            )
            rows.append({
                "scope": scope,
                "analysis": "RMS_to_error_control_MSJ+motion",
                "N": n,
                "partial_rho": r,
                "p": p,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print("=== PARTIAL ACTION DYNAMICS ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nOUTPUT =", OUT.resolve())

if __name__ == "__main__":
    main()
