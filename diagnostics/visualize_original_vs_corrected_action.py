from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

DT = 0.2

p = argparse.ArgumentParser()
p.add_argument("--npz", type=str, default=None)
p.add_argument(
    "--root",
    type=str,
    default="diagnostics/results/goal_preserved_range_correction_v2_7987/corrected_actions"
)
p.add_argument("--out", type=str, default="diagnostics/results/original_vs_corrected_action.png")
args = p.parse_args()

root = Path(args.root)

if args.npz is None:
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(root)
    path = files[0]
else:
    path = Path(args.npz)

d = np.load(path, allow_pickle=True)

orig = d["original_robot_flows"].astype(float)
corr = d["corrected_robot_flows"].astype(float)
phase = d["phase"].astype(float)

# timestep 사이에서 robot point들이 얼마나 움직였는지
def step_motion(x):
    dx = x[1:] - x[:-1]
    return np.sqrt(np.mean(np.sum(dx**2, axis=-1), axis=-1))

orig_motion = step_motion(orig)
corr_motion = step_motion(corr)

t = np.arange(len(phase))
tm = np.arange(1, len(phase))

fig, ax = plt.subplots(2, 1, figsize=(9, 7))

# 1) 실제 retiming
ax[0].plot(t, t, "--", label="Original timing")
ax[0].plot(t, phase, "o-", label="Corrected timing")
ax[0].set_xlabel("Output timestep")
ax[0].set_ylabel("Original trajectory phase")
ax[0].set_title("Temporal Retiming")
ax[0].legend()
ax[0].grid(alpha=0.3)

# 2) 원본 vs 보정 action 변화량
ax[1].plot(tm, orig_motion * 1000, "o-", label="Original action")
ax[1].plot(tm, corr_motion * 1000, "o-", label="Corrected action")
ax[1].set_xlabel("Timestep")
ax[1].set_ylabel("Mean robot motion / step (mm)")
ax[1].set_title("Original vs Corrected Action")
ax[1].legend()
ax[1].grid(alpha=0.3)

fig.suptitle(f"{path.stem}\nStart/endpoint preserved", fontsize=12)
fig.tight_layout()

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")

print("NPZ :", path)
print("SAVE:", out)
print("phase:", np.round(phase, 3))
print("start diff:", np.max(np.abs(orig[0] - corr[0])))
print("end diff  :", np.max(np.abs(orig[-1] - corr[-1])))
