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

base_obj = list(z["base_obj_frames"])
corr_obj = list(z["corr_obj_frames"])
base_grip = list(z["base_grip_frames"])
corr_grip = list(z["corr_grip_frames"])
base_scene = list(z["base_scene_frames"])
corr_scene = list(z["corr_scene_frames"])
frames = z["used_frames"]
clip = str(z["clip"])

all_pts = []
for arrs in [base_obj, corr_obj, base_grip, corr_grip, base_scene, corr_scene]:
    for a in arrs:
        all_pts.append(np.asarray(a, np.float32))
P = np.concatenate(all_pts, axis=0)

mn = P.min(axis=0)
mx = P.max(axis=0)
ctr = (mn + mx) / 2.0
rad = float(np.max(mx - mn) / 2.0) * 1.12
if rad <= 0:
    rad = 0.1

fig = plt.figure(figsize=(11, 5))
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
    ax.view_init(elev=20, azim=-60)

def update(i):
    ax1.cla()
    ax2.cla()
    setup(ax1, f"Original | frame {int(frames[i])+1}")
    setup(ax2, f"Corrected | frame {int(frames[i])+1}")

    Sb = np.asarray(base_scene[i], np.float32)
    Sc = np.asarray(corr_scene[i], np.float32)
    Gb = np.asarray(base_grip[i], np.float32)
    Gc = np.asarray(corr_grip[i], np.float32)
    Ob = np.asarray(base_obj[i], np.float32)
    Oc = np.asarray(corr_obj[i], np.float32)

    ax1.scatter(Sb[:,0], Sb[:,1], Sb[:,2], s=1, alpha=0.08, label="scene")
    ax1.scatter(Gb[:,0], Gb[:,1], Gb[:,2], s=3, alpha=0.35, label="gripper")
    ax1.scatter(Ob[:,0], Ob[:,1], Ob[:,2], s=8, alpha=0.95, label="can")

    ax2.scatter(Sc[:,0], Sc[:,1], Sc[:,2], s=1, alpha=0.08, label="scene")
    ax2.scatter(Gc[:,0], Gc[:,1], Gc[:,2], s=3, alpha=0.35, label="gripper")
    ax2.scatter(Oc[:,0], Oc[:,1], Oc[:,2], s=8, alpha=0.95, label="can")

    ax1.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle(clip, fontsize=10)

setup(ax1, "Original")
setup(ax2, "Corrected")

ani = FuncAnimation(fig, update, frames=len(frames), interval=800, repeat=True)

out = Path(
    "diagnostics/results/"
    "acc_only_boundary_gripper_retention_current10457/"
    "candidate_vis/task0001_can113_with_scene_original_vs_corrected.gif"
)
ani.save(out, writer=PillowWriter(fps=1))
print("SAVED:", out)
