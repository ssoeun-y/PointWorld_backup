#!/usr/bin/env python3
"""
v2 render from the CACHED case_LOW/MID/HIGH.npz (no inference):
  * shaded solid look (normal-lit splats) instead of loose dots -> the object reads as an object
  * row 1: true geometry, colored by per-point residual (mm)
  * row 2: SAME residuals, deviation from GT exaggerated xK (K printed on the figure) -> the distortion is visible
  * panel titles use RMS Acc and its ratio to the working boundary (no 'low/mid/high' words)
  python diagnostics/render_acc_levels_v2.py [--elev 60 --azim -25 --amp 8 --world --flip]
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_acc_level_problem_figure as M

B_DEFAULT = 0.24580418
LIGHT = np.array([-0.35, 0.45, 0.82]); LIGHT /= np.linalg.norm(LIGHT)

def view_R(elev, azim):
    a, phi = np.radians(azim), np.radians(-(90 - elev))
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    return Rx @ Rz

def normals(P, k=14):
    d2 = ((P[:, None] - P[None]) ** 2).sum(-1)
    idx = np.argsort(d2, 1)[:, 1:k + 1]
    N = np.zeros_like(P)
    for i in range(len(P)):
        Q = P[idx[i]] - P[idx[i]].mean(0)
        N[i] = np.linalg.eigh(Q.T @ Q)[1][:, 0]
    return N, float(np.median(np.sqrt(np.sort(d2, 1)[:, 1:4].mean(1))))

def splat(ax, P, rgb, elev, azim, size, N=None, alpha=1.0, z=1):
    Rv = view_R(elev, azim); Q = P @ Rv.T
    rgb = np.asarray(rgb, float)[:, :3]
    if N is not None:
        Nq = N @ Rv.T; Nq[Nq[:, 2] < 0] *= -1
        rgb = np.clip(rgb * (0.62 + 0.38 * np.clip(Nq @ LIGHT, 0, 1))[:, None], 0, 1)
    o = np.argsort(Q[:, 2])
    ax.scatter(Q[o, 0], Q[o, 1], s=size, c=rgb[o], linewidths=0, alpha=alpha, zorder=z)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elev", type=float, default=60); ap.add_argument("--azim", type=float, default=-25)
    ap.add_argument("--amp", type=float); ap.add_argument("--vmax", type=float)
    ap.add_argument("--boundary", type=float, default=B_DEFAULT)
    ap.add_argument("--world", action="store_true", help="use the LOW clip's scene pose instead of PCA face-on frame")
    ap.add_argument("--flip", action="store_true", help="view from the opposite side")
    ap.add_argument("--out", default=str(M.OUTDIR / "figure_problem_acc_levels_v2.png"))
    a = ap.parse_args()
    for lv in M.LEVELS:
        if not (M.OUTDIR / f"case_{lv}.npz").exists():
            raise SystemExit(f"missing {M.OUTDIR}/case_{lv}.npz  -> run diagnostics/make_acc_level_problem_figure.py first")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch

    D = M.prepare(M.load_cases(), 1.0, not a.world)          # vis_c = aligned prediction (true geometry), ref_c = GT
    elev = -a.elev if a.flip else a.elev
    K = len(D)
    unit = [c["unit"] for c in M.load_cases()][0]
    size3d = float(np.linalg.norm(np.ptp(D[0]["ref_c"], axis=0)))
    if a.amp is None:
        p90 = np.percentile(np.concatenate([d["res"] for d in D]), 90) / unit
        a.amp = float(np.clip(round(0.08 * size3d / p90), 4, 20))
    for d in D:
        d["exa_c"] = d["ref_c"] + a.amp * (d["vis_c"] - d["ref_c"])
    vmax = a.vmax or float(np.ceil(np.percentile(np.concatenate([d["res"] for d in D]), 97)))
    norm, cmap = Normalize(0, vmax), plt.get_cmap("turbo")

    Rv = view_R(elev, a.azim)
    xy = np.concatenate([(d[k] @ Rv.T)[:, :2] for d in D for k in ("ref_c", "exa_c")])
    half = 0.5 * np.ptp(xy, axis=0).max() * 1.10; mid = 0.5 * (xy.max(0) + xy.min(0))

    W = 3.5 * K + 1.3
    fig = plt.figure(figsize=(W, 8.0), dpi=200)
    gs = fig.add_gridspec(2, K + 1, width_ratios=[1] * K + [0.05], left=0.09, right=0.93, top=0.80, bottom=0.05, wspace=0.04, hspace=0.16)
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(K)] for r in range(2)])
    for r in range(2):
        for c, d in enumerate(D):
            ax = axs[r, c]
            N, nn = normals(d["ref_c"] if r == 0 else d["exa_c"])
            ax.set_xlim(mid[0] - half, mid[0] + half); ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_aspect("equal"); ax.axis("off")
            pts_per_unit = ax.get_position().width * W * 72 / (2 * half)
            size = (1.25 * nn * pts_per_unit) ** 2
            col = cmap(norm(d["res"]))
            if r == 0:
                splat(ax, d["vis_c"], col, elev, a.azim, size, N=N)
            else:
                splat(ax, d["ref_c"], np.tile([0.72, 0.72, 0.72], (len(col), 1)), elev, a.azim, size, N=N, alpha=0.55, z=1)
                P0 = d["ref_c"] @ Rv.T; P1 = d["exa_c"] @ Rv.T
                ax.add_collection(LineCollection(np.stack([P0[:, :2], P1[:, :2]], 1), colors=col, linewidths=0.7, alpha=0.75, zorder=2))
                splat(ax, d["exa_c"], col, elev, a.azim, size * 0.55, N=N, z=3)
            if r == 0:
                ax.set_title(f"({chr(97+c)}) RMS Acc = {d['acc']:.2f} m/s²\n{d['acc'] / a.boundary:.1f}× working boundary",
                             fontsize=13, fontweight="bold", pad=4)
                ax.text(0.5, 0.0, f"PA-MPJPE = {d['res'].mean():.2f} mm  (t={d['frame']})", transform=ax.transAxes,
                        ha="center", va="top", fontsize=12)
    fig.text(0.03, 0.5 * (axs[0, 0].get_position().y0 + axs[0, 0].get_position().y1), "as predicted\n(true geometry)", rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
    fig.text(0.03, 0.5 * (axs[1, 0].get_position().y0 + axs[1, 0].get_position().y1), f"deviation from GT\n×{a.amp:g} (exaggerated)", rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_subplot(gs[:, K]))
    cb.set_label("true per-point shape residual (mm)", fontsize=11)
    x0, x1 = axs[0, 0].get_position().x0, axs[0, -1].get_position().x1
    fig.add_artist(FancyArrowPatch((x0 + 0.02, 0.925), (x1 - 0.02, 0.925), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, lw=2.2, color="#444"))
    fig.text(0.5 * (x0 + x1), 0.935, "increasing action dynamics (trajectory RMS acceleration)", ha="center", va="bottom", fontsize=13, fontweight="bold", color="#333")
    fig.savefig(a.out, dpi=200, facecolor="white"); fig.savefig(a.out.replace(".png", ".pdf"), facecolor="white"); plt.close(fig)

    # view sheet (shaded) to pick the best angle
    elevs, azims = [30, 55, 80, -55], [-135, -90, -45, 0, 45, 90, 135, 180]
    fs, sx = plt.subplots(len(elevs), len(azims), figsize=(2.6 * len(azims), 2.6 * len(elevs)), dpi=100)
    P = D[0]["ref_c"]; N, nn = normals(P); h = 0.5 * np.ptp(P, axis=0).max() * 1.2
    for i, e in enumerate(elevs):
        for j, az in enumerate(azims):
            ax = sx[i, j]; ax.set_xlim(-h, h); ax.set_ylim(-h, h); ax.set_aspect("equal"); ax.axis("off")
            splat(ax, P - P.mean(0), np.tile([0.55, 0.55, 0.6], (len(P), 1)), e, az, (1.25 * nn * 2.6 * 72 / (2 * h)) ** 2, N=N)
            ax.set_title(f"--elev {e} --azim {az}", fontsize=8)
    fs.tight_layout(); fs.savefig(M.OUTDIR / "view_sheet_v2.png"); plt.close(fs)
    print("SAVED", a.out, "| view sheet:", M.OUTDIR / "view_sheet_v2.png", f"| amp x{a.amp:g} vmax={vmax:g} mm")

if __name__ == "__main__":
    main()
