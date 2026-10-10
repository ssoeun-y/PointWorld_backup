from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
frames = z["used_frames"]
clip = str(z["clip"])

# Same zoom limits for Original / Corrected / all frames.
pts = []
for arrs in [bo, co, bg, cg]:
    for a in arrs:
        pts.append(np.asarray(a, np.float32))

P = np.concatenate(pts, axis=0)
mn, mx = P.min(axis=0), P.max(axis=0)
ctr = (mn + mx) / 2.0

# Tight crop around gripper + can.
span = np.max(mx - mn)
rad = max(float(span * 0.60), 0.08)

fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

def setup(ax, title):
    ax.set_title(title)
    ax.set_xlim(ctr[0]-rad, ctr[0]+rad)
    ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
    ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=-60)

def update(i):
    ax1.cla()
    ax2.cla()

    setup(ax1, f"Original | frame {int(frames[i])+1}")
    setup(ax2, f"Corrected | frame {int(frames[i])+1}")

    Gb = np.asarray(bg[i], np.float32)
    Gc = np.asarray(cg[i], np.float32)
    Ob = np.asarray(bo[i], np.float32)
    Oc = np.asarray(co[i], np.float32)

    ax1.scatter(Gb[:,0], Gb[:,1], Gb[:,2],
                s=8, alpha=0.55, label="gripper")
    ax1.scatter(Ob[:,0], Ob[:,1], Ob[:,2],
                s=14, alpha=0.95, label="can")

    ax2.scatter(Gc[:,0], Gc[:,1], Gc[:,2],
                s=8, alpha=0.55, label="gripper")
    ax2.scatter(Oc[:,0], Oc[:,1], Oc[:,2],
                s=14, alpha=0.95, label="can")

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    fig.suptitle(clip)

ani = FuncAnimation(
    fig, update,
    frames=len(frames),
    interval=900,
    repeat=True
)

out = Path(
    "diagnostics/results/"
    "acc_only_boundary_gripper_retention_current10457/"
    "candidate_vis/task0001_can113_zoom.gif"
)

ani.save(out, writer=PillowWriter(fps=1))
print("SAVED:", out)
