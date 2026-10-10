#!/usr/bin/env python3
"""
One command: low/mid/high RMS-Acc clips of the SAME object -> PointWorld inference (via your existing
make_problem_failure_qualitative_final.py, patched in memory: your files are NOT modified) ->
similarity-aligned prediction colored by per-point residual (shared color scale).
All three radios are registered into ONE common orientation (the LOW clip's own upright scene pose),
so the radio keeps the same recognizable shape in every panel.

  python diagnostics/make_acc_level_problem_figure.py                       # dump (cached) + figure + view_sheet.png
  python diagnostics/make_acc_level_problem_figure.py --elev 35 --azim -120 # re-render only (npz cached)
"""
import argparse, csv, itertools, os, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "diagnostics" / "results"
OUTDIR = RES / "problem_acc_levels"
CHOSEN = OUTDIR / "chosen_clips.csv"
PER_FRAME_ROOT = RES / "object_pa_mpjpe_acc_5task_current10457"
TARGET = REPO / "diagnostics" / "make_problem_failure_qualitative_final.py"
LIT_CSV, LIT_OUT = "representative_highdyn_choice.csv", "problem_failure_qualitative_final"
LEVELS = ["LOW", "MID", "HIGH"]
FAILED = OUTDIR / "failed_clips.txt"      # clips the inference script rejected (never retried)


# ------------------------------------------------------------------ geometry
def to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)

def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    n = len(X)
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / n)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (D * np.diag(E)).sum() / ((Xc ** 2).sum() / n)
    return s, R, my - s * R @ mx

def pa_align(pred, ref):
    s, R, t = umeyama(pred, ref)
    return s * pred @ R.T + t

def kabsch(A, B):
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mb - R @ ma

def _rot24():
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, sg)):
                M[i, p] = s
            if np.linalg.det(M) > 0:
                out.append(M)
    return out
ROT24 = _rot24()

def _sub(P, k=250):
    return P[np.linspace(0, len(P) - 1, min(len(P), k)).astype(int)]

def register(src, dst, keep=0.8, iters=25):
    """rigid (R,t): R@src+t ~ dst. partial-overlap safe (trimmed ICP), 24 axis-aligned starts. returns (score,R,t)."""
    s_, d_ = _sub(src), _sub(dst)
    best = None
    for R0 in ROT24:
        R, t = R0, d_.mean(0) - R0 @ s_.mean(0)
        for _ in range(iters):
            X = s_ @ R.T + t
            d2 = ((X[:, None] - d_[None]) ** 2).sum(-1)
            j = d2.argmin(1); dm = d2[np.arange(len(X)), j]
            sel = dm <= np.quantile(dm, keep)
            if sel.sum() < 10:
                break
            R, t = kabsch(s_[sel], d_[j[sel]])
        X = s_ @ R.T + t
        d2 = ((X[:, None] - d_[None]) ** 2).sum(-1)
        sc = (np.sqrt(np.sort(d2.min(1))[:int(keep * len(X))]).mean()
              + np.sqrt(np.sort(d2.min(0))[:int(keep * len(d_))]).mean())
        if best is None or sc < best[0]:
            best = (sc, R, t)
    return best

def canonical_rot(ref):
    c = ref - ref.mean(0)
    R = np.linalg.svd(c, full_matrices=False)[2].copy()
    proj = c @ R.T
    for k in range(3):
        if (proj[:, k] ** 3).mean() < 0:
            R[k] *= -1
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return R

def view_project(P, elev, azim):
    a, phi = np.radians(azim), np.radians(-(90 - elev))
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    Q = P @ (Rx @ Rz).T
    return Q[:, :2], Q[:, 2]


# ------------------------------------------------------------------ step 1: dump (runs YOUR inference script, patched in memory)
def _write_choice(path, choice):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(choice)); w.writeheader(); w.writerow(choice)

def _load_cached_fd(path, clip=None):
    z = np.load(path)
    if clip is not None and "meta__clip" in z.files and str(z["meta__clip"]) != clip:
        return None
    fd = {}
    for k in z.files:
        parts = k.split("__")
        if len(parts) >= 3 and parts[0] == "frame_data" and parts[1].isdigit():
            fd.setdefault(int(parts[1]), {})["__".join(parts[2:])] = z[k]
    return fd or None

def dump_level(level, T=10, force=False):
    row = next(r for r in csv.DictReader(open(CHOSEN)) if r["level"] == level)
    task, clip, oid = row["task"], row["clip"], int(float(row["object_id"]))
    pf = [r for r in csv.DictReader(open(PER_FRAME_ROOT / task / "per_frame_metrics.csv"))
          if r["clip"] == clip and int(float(r["object_id"])) == oid]
    if len(pf) != 10:
        raise SystemExit(f"[{level}] expected 10 per-frame rows for {clip} obj {oid}, got {len(pf)}")
    pa = {int(r["frame"]): float(r["pa_mpjpe_mm"]) for r in pf}
    mp = {int(r["frame"]): float(r["mpjpe_mm"]) for r in pf}
    pts = np.array([float(r["valid_point_count"]) for r in pf]); acc = np.array([float(r["rms_acceleration_si_m_s2"]) for r in pf])
    pav = np.array([pa[k] for k in sorted(pa)])
    choice = dict(task=task, clip=clip, object_id=oid, object_short_name=row["object_short_name"], n_frames=10,
                  points_median=float(np.median(pts)), points_mean=float(pts.mean()), acc_median=float(np.median(acc)),
                  acc_mean=float(acc.mean()), pa_t1=pa[1], pa_t5=pa[5], pa_t10=pa[10], pa_delta_1_10=pa[10] - pa[1],
                  pa_ratio_10_over_1=pa[10] / pa[1], pa_slope=float(np.polyfit(np.arange(1, 11), pav, 1)[0]),
                  pa_median=float(np.median(pav)), pa_mean=float(pav.mean()),
                  monotonic_fraction=float((np.diff(pav) >= 0).mean()), repr_score=0.0)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTDIR / f"raw_frame_data_{level}.npz"
    fd = None
    if raw_path.exists() and not force:
        fd = _load_cached_fd(raw_path, clip)
        if fd is not None:
            print(f"[{level}] using CACHED inference {raw_path.name} (add --force to re-run the model)")
    if fd is None:
        src = TARGET.read_text()
        for lit in (LIT_CSV, LIT_OUT):
            if lit not in src:
                raise SystemExit(f"'{lit}' not found in {TARGET.name}; refusing to run (would overwrite existing figure outputs).")
        src = src.replace(LIT_CSV, f"acc_level_{level}.csv").replace(LIT_OUT, f"problem_acc_levels_raw_{level}")
        _write_choice(RES / "problem_failure_selection" / f"acc_level_{level}.csv", choice)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
        sys.path.insert(0, str(REPO))
        ns = None
        for attempt in range(3):
            ns = {"__name__": "__main__", "__file__": str(TARGET)}
            sys.argv = [str(TARGET)]
            try:
                exec(compile(src, str(TARGET), "exec"), ns)
                break
            except SystemExit:
                break
            except FileNotFoundError as e:
                fn = getattr(e, "filename", None)
                if fn and "acc_level_" in str(fn) and attempt < 2:
                    _write_choice(fn, choice); continue
                raise
        fd = ns.get("frame_data")
        cache = {}
        def _walk(prefix, obj, depth=0):
            if depth > 4:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _walk(f"{prefix}__{k}", v, depth + 1)
            elif isinstance(obj, (list, tuple)) and len(obj) < 64:
                for i, v in enumerate(obj):
                    _walk(f"{prefix}__{i}", v, depth + 1)
            else:
                try:
                    a_ = to_np(obj)
                    if a_.ndim >= 1 and a_.size > 0:
                        cache[prefix] = a_
                except Exception:
                    pass
        if fd is None:
            print(f"[{level}] 'frame_data' is not a global of the script. array/dict/list globals:")
            for k, v in ns.items():
                if k.startswith("__"):
                    continue
                if isinstance(v, (dict, list, tuple)) or hasattr(v, "shape"):
                    print("   ", k, type(v).__name__, getattr(v, "shape", len(v) if hasattr(v, "__len__") else ""))
            raise SystemExit(f"[{level}] cannot find frame_data")
        _walk("frame_data", fd)
        if cache:
            cache["meta__clip"] = np.array(clip)
            np.savez(raw_path, **cache)          # cache FIRST: never pay for inference twice
        fd = _load_cached_fd(raw_path, clip) or fd

    # ---- pick frame T
    d = fd.get(T) if isinstance(fd, dict) else None
    if not isinstance(d, dict) or "gt" not in d:
        raise SystemExit(f"[{level}] cannot find frame {T} with a 'gt' entry; keys={list(fd) if isinstance(fd, dict) else type(fd).__name__}")
    gt = to_np(d["gt"])

    # ---- pick the prediction that pairs with 'gt' (ignore *_vis display copies)
    cands = []
    for k, v in d.items():
        if k in ("gt", "idx"):
            continue
        try:
            c = to_np(v)
        except Exception:
            continue
        if c.shape != gt.shape:
            continue
        raw_ = np.linalg.norm(pa_align(c, gt) - gt, axis=1).mean()
        if raw_ <= 0:
            continue
        ratio_ = pa[T] / raw_
        u_ = 1000.0 if 300 < ratio_ < 3000 else (1.0 if 0.3 < ratio_ < 3 else 1000.0)
        cands.append(dict(key=k, arr=c, unit=u_, pa_mm=raw_ * u_, mp_mm=np.linalg.norm(c - gt, axis=1).mean() * u_,
                          vis="vis" in str(k).lower()))
        print(f"[{level}]   candidate '{k}': PA={raw_ * u_:.3f} mm  direct MPJPE vs gt={cands[-1]['mp_mm']:.3f} mm"
              f"  (csv PA={pa[T]:.3f}, csv MPJPE={mp[T]:.3f})")
    pool = [c for c in cands if not c["vis"]] or cands
    if not pool:
        raise SystemExit(f"[{level}] no same-shape prediction array next to 'gt'; keys={list(d)}")
    pick = min(pool, key=lambda c: abs(c["mp_mm"] - mp[T]) / mp[T])
    kind = "raw prediction"
    if abs(pick["mp_mm"] - mp[T]) / mp[T] > 0.3:       # raw prediction is not stored; the stored one is already aligned to gt
        pick = min(pool, key=lambda c: abs(c["pa_mm"] - pa[T]) / pa[T])
        kind = "prediction already similarity-aligned to gt (raw one is not stored; fine for this figure)"
    if abs(pick["pa_mm"] - pa[T]) / pa[T] > 0.5:
        raise SystemExit(f"[{level}] PA of every candidate is >50% off the csv PA ({pa[T]:.3f} mm): wrong frame/key? see candidates above")
    pred, unit, mine = pick["arr"], pick["unit"], pick["pa_mm"]
    flag = "OK" if abs(mine - pa[T]) / pa[T] < 0.15 else "WARNING >15% off (fresh inference / different point subset)"
    print(f"[{level}] using key='{pick['key']}' ({kind}) N={len(gt)} unit_scale={unit:g} | PA mine={mine:.3f} mm vs per_frame csv={pa[T]:.3f} mm -> {flag}")
    np.savez(OUTDIR / f"case_{level}.npz", ref=gt, pred=pred, rms_acc=float(row["rms_acc_si"]), frame=T,
             unit_scale=unit, pa_csv=pa[T], label=f"{row['object_short_name']} {clip}")


def _chosen_rows():
    return list(csv.DictReader(open(CHOSEN)))

def _set_choice(level, cand):
    rows = _chosen_rows()
    for r in rows:
        if r["level"] == level:
            r.update(clip=cand["clip"], object_id=cand["object_id"], rms_acc_si=cand["acc"], pa_mpjpe_mean_mm=cand["pa"], points=cand["points"])
    with open(CHOSEN, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def candidates(level, band=0.05):
    """clips of the same object in the same Acc band, ordered by closeness to the band's MEDIAN PA (cherry-pick safe)."""
    cur = next(r for r in _chosen_rows() if r["level"] == level)
    q = {"LOW": 0.10, "MID": 0.50, "HIGH": 0.90}[level]
    rows = [r for r in csv.DictReader(open(PER_FRAME_ROOT / cur["task"] / "per_object_metrics.csv"))
            if r["object_short_name"] == cur["object_short_name"] and int(float(r["eligible_future_frames"])) == 10
            and int(float(r["final_object_point_count"])) >= 300]
    acc = np.array([float(r["rms_acceleration_si_m_s2"]) for r in rows]); pa_ = np.array([float(r["pa_mpjpe_mean_mm"]) for r in rows])
    lo, hi = np.quantile(acc, [max(q - band, 0), min(q + band, 1)])
    idx = np.where((acc >= lo) & (acc <= hi))[0]
    med = np.median(pa_[idx])
    idx = idx[np.argsort(np.abs(pa_[idx] - med))]
    return [dict(clip=rows[i]["clip"], object_id=rows[i]["object_id"], acc=acc[i], pa=pa_[i], points=rows[i]["final_object_point_count"]) for i in idx]


# ------------------------------------------------------------------ step 2: render
def load_cases():
    cs = []
    for lv in LEVELS:
        z = np.load(OUTDIR / f"case_{lv}.npz")
        cs.append(dict(ref=z["ref"], pred=z["pred"], acc=float(z["rms_acc"]), frame=int(z["frame"]),
                       unit=float(z["unit_scale"]), label=str(z["label"])))
    return cs

def prepare(cases, amp, use_pca):
    """align, residuals, and put all clips in ONE common orientation (the first clip's scene pose)."""
    out = []
    for c in cases:
        al = pa_align(c["pred"], c["ref"])
        res = np.linalg.norm(al - c["ref"], axis=1) * c["unit"]
        out.append(dict(ref=c["ref"], vis=c["ref"] + amp * (al - c["ref"]), res=res, acc=c["acc"], frame=c["frame"], label=c["label"]))
    m0 = out[0]["ref"].mean(0)
    R0 = canonical_rot(out[0]["ref"]) if use_pca else np.eye(3)
    for i, d in enumerate(out):
        if i == 0:
            R, t, mi = R0, np.zeros(3), m0
        else:
            mi = d["ref"].mean(0)
            sc, R, t = register(d["ref"] - mi, (out[0]["ref"] - m0) @ R0.T)
            size = np.linalg.norm(np.ptp(d["ref"], axis=0))
            print(f"  registration {LEVELS[i]} -> LOW: residual / object size = {sc / size:.3f}  (small = good, >0.10 = check the view sheet)")
        f = (lambda P, R=R, t=t, mi=mi: (P - mi) @ R.T + t) if i else (lambda P, R=R, mi=mi: (P - mi) @ R.T)
        d["ref_c"], d["vis_c"] = f(d["ref"]), f(d["vis"])
    return out

def view_sheet(P, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    elevs, azims = [15, 35, 60], [-135, -90, -45, 0, 45, 90, 135, 180]
    fig, axs = plt.subplots(len(elevs), len(azims), figsize=(2.6 * len(azims), 2.6 * len(elevs)), dpi=110)
    for i, e in enumerate(elevs):
        for j, a in enumerate(azims):
            xy, z = view_project(P, e, a); o = np.argsort(z)
            axs[i, j].scatter(*xy[o].T, s=6, c=z[o], cmap="Greys", linewidths=0)
            axs[i, j].set_aspect("equal"); axs[i, j].axis("off"); axs[i, j].set_title(f"elev={e} azim={a}", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def render(cases, out, amp=1.0, elev=30, azim=-60, vmax=None, use_pca=False, cmap="turbo"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch
    D = prepare(cases, amp, use_pca)
    view_sheet(D[0]["ref_c"], OUTDIR / "view_sheet.png")
    if vmax is None:
        vmax = float(np.ceil(np.percentile(np.concatenate([d["res"] for d in D]), 97)))
    norm, cm_ = Normalize(0, vmax), plt.get_cmap(cmap)
    xy = np.concatenate([view_project(d[k], elev, azim)[0] for d in D for k in ("ref_c", "vis_c")])
    half = 0.5 * np.ptp(xy, axis=0).max() * 1.10; mid = 0.5 * (xy.max(0) + xy.min(0))
    K = len(D)
    fig = plt.figure(figsize=(3.6 * K + 1.0, 4.6), dpi=200)
    gs = fig.add_gridspec(1, K + 1, width_ratios=[1] * K + [0.05], left=0.02, right=0.93, top=0.73, bottom=0.09, wspace=0.05)
    axs = []
    names = ["Low", "Medium", "High"]
    for i, d in enumerate(D):
        ax = fig.add_subplot(gs[0, i]); axs.append(ax)
        g_xy, g_z = view_project(d["ref_c"], elev, azim); p_xy, p_z = view_project(d["vis_c"], elev, azim)
        o = np.argsort(g_z); ax.scatter(*g_xy[o].T, s=9, c="#b8b8b8", alpha=0.55, linewidths=0, zorder=1)
        if amp != 1.0:
            ax.add_collection(LineCollection(np.stack([g_xy, p_xy], 1), colors=cm_(norm(d["res"])), linewidths=0.6, alpha=0.8, zorder=2))
        o = np.argsort(p_z); zn = (p_z[o] - p_z.min()) / (np.ptp(p_z) + 1e-9)
        ax.scatter(*p_xy[o].T, s=12 + 10 * zn, c=d["res"][o], cmap=cm_, norm=norm, linewidths=0.25, edgecolors="k", zorder=3)
        ax.set_xlim(mid[0] - half, mid[0] + half); ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"({chr(97+i)}) {names[i]}\nRMS Acc = {d['acc']:.2f} m/s²", fontsize=13, fontweight="bold", pad=4)
        ax.text(0.5, 0.0, f"PA-MPJPE = {d['res'].mean():.2f} mm  (t={d['frame']})", transform=ax.transAxes, ha="center", va="top", fontsize=12)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm_), cax=fig.add_subplot(gs[0, K]))
    cb.set_label("per-point shape residual (mm)", fontsize=11)
    x0, x1 = axs[0].get_position().x0, axs[-1].get_position().x1
    fig.add_artist(FancyArrowPatch((x0 + 0.02, 0.90), (x1 - 0.02, 0.90), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, lw=2.2, color="#444"))
    fig.text(0.5 * (x0 + x1), 0.915, "increasing action dynamics (trajectory RMS acceleration)", ha="center", va="bottom", fontsize=13, fontweight="bold", color="#333")
    if amp != 1.0:
        fig.text(0.5 * (x0 + x1), 0.01, f"deviation from GT exaggerated ×{amp:g} for visibility", ha="center", fontsize=9, color="0.35")
    fig.savefig(out, dpi=200, facecolor="white"); fig.savefig(str(out).replace(".png", ".pdf"), facecolor="white"); plt.close(fig)
    print("SAVED", out, "| SAVED", OUTDIR / "view_sheet.png", f"| vmax={vmax:g} mm | PA:", [round(float(d['res'].mean()), 2) for d in D])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--_dump"); ap.add_argument("--force", action="store_true"); ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--amp", type=float, default=1.0); ap.add_argument("--vmax", type=float)
    ap.add_argument("--elev", type=float, default=30); ap.add_argument("--azim", type=float, default=-60)
    ap.add_argument("--pca", action="store_true", help="use PCA-canonical orientation instead of the LOW clip's scene pose")
    ap.add_argument("--out", default=str(OUTDIR / "figure_problem_acc_levels.png"))
    a = ap.parse_args()
    os.chdir(REPO)
    if a._dump:
        dump_level(a._dump, force=a.force); return
    for lv in LEVELS:
        if a.force or not (OUTDIR / f"case_{lv}.npz").exists():
            tried = set(FAILED.read_text().split()) if FAILED.exists() else set()
            for attempt in range(1 + a.retries):
                cur = next(r for r in _chosen_rows() if r["level"] == lv)["clip"]
                if cur in tried:                                   # known-bad clip: jump to the next best candidate
                    nxt = next((c for c in candidates(lv) if c["clip"] not in tried), None)
                    if nxt is None:
                        raise SystemExit(f"[{lv}] no candidate left (all rejected)")
                    print(f"[{lv}] skipping known-bad {cur} -> {nxt['clip']} (RMS Acc {nxt['acc']:.3f}, clip PA {nxt['pa']:.2f} mm)")
                    _set_choice(lv, nxt); cur = nxt["clip"]
                print(f"--- inference: {lv}  clip={cur}  (attempt {attempt + 1})")
                r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_dump", lv] + (["--force"] if a.force else []))
                if r.returncode == 0:
                    break
                tried.add(cur)
                OUTDIR.mkdir(parents=True, exist_ok=True)
                with open(FAILED, "a") as f:
                    f.write(cur + "\n")
                if attempt == a.retries:
                    raise SystemExit(f"[{lv}] all {attempt + 1} attempts failed; see the log above")
                print(f"[{lv}] FAILED on {cur}; will try the next candidate")
    for c in load_cases():
        print("USED:", c["label"], f"| RMS Acc {c['acc']:.3f} m/s2")
    render(load_cases(), Path(a.out), a.amp, a.elev, a.azim, a.vmax, a.pca)

if __name__ == "__main__":
    main()
