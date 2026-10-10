from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

dump = Path(
    "diagnostics/results/"
    "acc_only_boundary_gripper_retention_current10457/"
    "candidate_vis/task0001_can113_candidate.npz"
)

z = np.load(dump, allow_pickle=True)

bo = list(z["base_obj_frames"])
co = list(z["corr_obj_frames"])
bg = list(z["base_grip_frames"])
cg = list(z["corr_grip_frames"])
bs = list(z["base_scene_frames"])
cs = list(z["corr_scene_frames"])
frames = np.asarray(z["used_frames"]).astype(int)

def split_two_hands(G):
    x = G[:, 0]
    med = np.median(x)
    g1 = G[x <= med]
    g2 = G[x > med]
    if len(g1) == 0 or len(g2) == 0:
        half = len(G) // 2
        g1, g2 = G[:half], G[half:]
    return g1, g2

def centroid(A):
    return A.mean(axis=0)

def min_dist(A, B):
    d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) ** 0.5
    return float(d.min()) * 1000.0

def scene_clearance(obj_pts, scene_pts, xy_radius=0.12):
    c = centroid(obj_pts)
    dx = scene_pts[:, 0] - c[0]
    dy = scene_pts[:, 1] - c[1]
    near = scene_pts[(dx*dx + dy*dy) ** 0.5 < xy_radius]
    if len(near) < 20:
        near = scene_pts
    obj_bottom = float(obj_pts[:, 2].min())
    support_top = float(np.percentile(near[:, 2], 95))
    return (obj_bottom - support_top) * 1000.0

rows = []
for i, f in enumerate(frames):
    for tag, O, G, S in [
        ("orig", bo[i], bg[i], bs[i]),
        ("corr", co[i], cg[i], cs[i]),
    ]:
        O = np.asarray(O, np.float32)
        G = np.asarray(G, np.float32)
        S = np.asarray(S, np.float32)

        g1, g2 = split_two_hands(G)
        d1 = min_dist(O, g1)
        d2 = min_dist(O, g2)
        dc1 = np.linalg.norm(centroid(O) - centroid(g1)) * 1000.0
        dc2 = np.linalg.norm(centroid(O) - centroid(g2)) * 1000.0
        clr = scene_clearance(O, S)

        rows.append({
            "frame": int(f + 1),
            "tag": tag,
            "min_hand1_mm": d1,
            "min_hand2_mm": d2,
            "cent_hand1_mm": dc1,
            "cent_hand2_mm": dc2,
            "clearance_mm": clr,
        })

# print table
print("\nFRAMEWISE METRICS")
print("=" * 90)
print(f"{'frame':>5s} {'tag':>5s} {'min_h1':>10s} {'min_h2':>10s} {'cent_h1':>10s} {'cent_h2':>10s} {'clear':>10s}")
for r in rows:
    print(f"{r['frame']:5d} {r['tag']:>5s} "
          f"{r['min_hand1_mm']:10.3f} {r['min_hand2_mm']:10.3f} "
          f"{r['cent_hand1_mm']:10.3f} {r['cent_hand2_mm']:10.3f} "
          f"{r['clearance_mm']:10.3f}")

# plot corrected vs original
fr = sorted(set(r["frame"] for r in rows))
def series(tag, key):
    m = {r["frame"]: r[key] for r in rows if r["tag"] == tag}
    return [m[f] for f in fr]

fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(311)
ax1.plot(fr, series("orig", "min_hand1_mm"), label="orig hand1")
ax1.plot(fr, series("orig", "min_hand2_mm"), label="orig hand2")
ax1.plot(fr, series("corr", "min_hand1_mm"), label="corr hand1")
ax1.plot(fr, series("corr", "min_hand2_mm"), label="corr hand2")
ax1.set_title("Can-to-hand minimum distance [mm]")
ax1.set_xlabel("frame")
ax1.set_ylabel("mm")
ax1.legend()

ax2 = fig.add_subplot(312)
ax2.plot(fr, series("orig", "cent_hand1_mm"), label="orig hand1")
ax2.plot(fr, series("orig", "cent_hand2_mm"), label="orig hand2")
ax2.plot(fr, series("corr", "cent_hand1_mm"), label="corr hand1")
ax2.plot(fr, series("corr", "cent_hand2_mm"), label="corr hand2")
ax2.set_title("Can-centroid to hand-centroid distance [mm]")
ax2.set_xlabel("frame")
ax2.set_ylabel("mm")
ax2.legend()

ax3 = fig.add_subplot(313)
ax3.plot(fr, series("orig", "clearance_mm"), label="orig clearance")
ax3.plot(fr, series("corr", "clearance_mm"), label="corr clearance")
ax3.axhline(0.0)
ax3.set_title("Can bottom to nearby scene/table clearance [mm]")
ax3.set_xlabel("frame")
ax3.set_ylabel("mm")
ax3.legend()

out = Path(
    "diagnostics/results/"
    "acc_only_boundary_gripper_retention_current10457/"
    "candidate_vis/task0001_can113_detach_metrics.png"
)
plt.tight_layout()
plt.savefig(out, dpi=180)
print("\nSAVED:", out)
