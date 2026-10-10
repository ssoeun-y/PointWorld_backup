#!/usr/bin/env python3
"""Pick low/mid/high RMS-Acc clips of ONE object (default radio_89, task-0000), cherry-pick safe:
per Acc band, the clip whose clip-level PA is closest to that band's MEDIAN PA."""
import argparse, csv
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="task-0000")
ap.add_argument("--object", default="radio_89")
ap.add_argument("--root", default="diagnostics/results/object_pa_mpjpe_acc_5task_current10457")
ap.add_argument("--min_points", type=int, default=300)
ap.add_argument("--quantiles", type=float, nargs=3, default=[0.10, 0.50, 0.90])
ap.add_argument("--band", type=float, default=0.05)
ap.add_argument("--out", default="diagnostics/results/problem_acc_levels/chosen_clips.csv")
a = ap.parse_args()

src = Path(a.root) / a.task / "per_object_metrics.csv"
rows = [r for r in csv.DictReader(open(src)) if r["object_short_name"] == a.object
        and int(float(r["eligible_future_frames"])) == 10 and int(float(r["final_object_point_count"])) >= a.min_points]
if len(rows) < 30:
    raise SystemExit(f"only {len(rows)} usable rows for {a.object} in {src}; lower --min_points?")
acc = np.array([float(r["rms_acceleration_si_m_s2"]) for r in rows])
pa = np.array([float(r["pa_mpjpe_mean_mm"]) for r in rows])

def rank(x): return np.argsort(np.argsort(x))
print(f"{a.object} / {a.task}: {len(rows)} clips | acc range {acc.min():.3f}-{acc.max():.3f} m/s2 | "
      f"within-object Spearman(acc, PA) = {np.corrcoef(rank(acc), rank(pa))[0,1]:+.3f}")

out = []
for name, q in zip(["LOW", "MID", "HIGH"], a.quantiles):
    lo, hi = np.quantile(acc, [max(q - a.band, 0), min(q + a.band, 1)])
    idx = np.where((acc >= lo) & (acc <= hi))[0]
    i = int(idx[np.argmin(np.abs(pa[idx] - np.median(pa[idx])))])
    r = rows[i]
    print(f"{name:4s} acc={acc[i]:.3f}  PA={pa[i]:.2f} mm  (band median PA {np.median(pa[idx]):.2f}, n_band={len(idx)})  "
          f"pts={r['final_object_point_count']}  {r['clip']}  object_id={r['object_id']}")
    out.append(dict(level=name, task=r["task"], clip=r["clip"], object_id=r["object_id"], object_short_name=r["object_short_name"],
                    rms_acc_si=acc[i], pa_mpjpe_mean_mm=pa[i], points=r["final_object_point_count"]))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print("WROTE", a.out)
