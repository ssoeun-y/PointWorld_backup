from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

dump = Path(
    "diagnostics/results/"
    "acc_only_boundary_gripper_retention_current10457/"
    "candidate_vis/task0000_radio89_candidate.npz"
)

z = np.load(dump, allow_pickle=True)

base_obj = list(z["base_obj_frames"])
corr_obj = list(z["corr_obj_frames"])
base_grip = list(z["base_grip_frames"])
corr_grip = list(z["corr_grip_frames"])
frames = z["used_frames"]
clip = str(z["clip"])

all_pts = []
for arrs in [base_obj, corr_obj, base_grip, corr_grip]:
    for a in arrs:
        all_pts.append(np.asarray(a, np.float32))
P = np.concatenate(all_pts, axis=0)

mn = P.min(axis=0)
mx = P.max(axis=0)
ctr = (mn + mx) / 2.0
rad = float(np.max(mx - mn) / 2.0) * 1.15
if rad <= 0:
    rad = 0.1

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

def setup(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_xlim(ctr[0]-rad, ctr[0]+rad)
    ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
    ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def update(i):
    ax1.cla()
    ax2.cla()
    setup(ax1, f"Original | frame {int(frames[i])+1}")
    setup(ax2, f"Corrected | frame {int(frames[i])+1}")

    Gb = np.asarray(base_grip[i], np.float32)
    Gc = np.asarray(corr_grip[i], np.float32)
    Ob = np.asarray(base_obj[i], np.float32)
    Oc = np.asarray(corr_obj[i], np.float32)

    ax1.scatter(Gb[:,0], Gb[:,1], Gb[:,2], s=3, alpha=0.35, label="gripper")
    ax1.scatter(Ob[:,0], Ob[:,1], Ob[:,2], s=7, alpha=0.95, label="radio")

    ax2.scatter(Gc[:,0], Gc[:,1], Gc[:,2], s=3, alpha=0.35, label="gripper")
    ax2.scatter(Oc[:,0], Oc[:,1], Oc[:,2], s=7, alpha=0.95, label="radio")

    ax1.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle(clip, fontsize=10)

setup(ax1, "Original")
setup(ax2, "Corrected")

ani = FuncAnimation(fig, update, frames=len(frames), interval=800, repeat=True)

out = Path(
    "diagnostics/results/"
    "acc_only_boundary_gripper_retention_current10457/"
    "candidate_vis/task0000_radio89_original_vs_corrected.gif"
)
ani.save(out, writer=PillowWriter(fps=1))
print("SAVED:", out)
