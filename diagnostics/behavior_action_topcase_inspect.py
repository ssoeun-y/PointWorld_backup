import csv, io, tarfile
from pathlib import Path
import numpy as np

ROOT = Path("/home/viglab/cse/PointWorld_workspace/datasets/behavior/wds_task0000")
CSV = Path("diagnostics/results/action_consistency_task0000.csv")

# 서로 다른 현상을 대표하는 후보.
TARGETS = [
    "task-0000_episode_00002090-390:401", # speed CV
    "task-0000_episode_00002010-0:11",    # accel/jerk
    "task-0000_episode_00001040-610:621", # reversal/turn
    "task-0000_episode_00002990-70:81",   # path inefficiency, test
    "task-0000_episode_00001750-325:336", # turn, test
]

rows = {}
with CSV.open() as f:
    for r in csv.DictReader(f):
        if r["clip"] in TARGETS:
            rows[r["clip"]] = r

def load_pose(clip, hand):
    for split in ["train", "test"]:
        for tp in sorted((ROOT/split).glob("*.tar")):
            name = f"{clip}.{hand}_gripper_pose.npy"
            with tarfile.open(tp) as tf:
                try:
                    m = tf.getmember(name)
                except KeyError:
                    continue
                return np.load(io.BytesIO(tf.extractfile(m).read())), split
    raise RuntimeError(clip)

for clip in TARGETS:
    r = rows[clip]
    hand = r["dominant_hand"]
    pose, split = load_pose(clip, hand)
    xyz = pose[:, :3].astype(float)

    v = np.diff(xyz, axis=0)
    speed = np.linalg.norm(v, axis=1)
    a = np.diff(v, axis=0)
    jerk = np.diff(a, axis=0)

    angles = []
    for x, y in zip(v[:-1], v[1:]):
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if nx > 1e-12 and ny > 1e-12:
            c = np.clip(np.dot(x,y)/(nx*ny), -1, 1)
            angles.append(np.degrees(np.arccos(c)))
        else:
            angles.append(np.nan)

    print("\n" + "="*100)
    print(clip, "|", split, "|", hand)
    print("path_length_m =", r["path_length_m"])
    print("net_displacement_m =", r["net_displacement_m"])
    print("speed_cv =", r["speed_cv"])
    print("reversal_fraction =", r["reversal_fraction"])

    print("speed m/step =", np.round(speed, 5))
    print("turn deg      =", np.round(angles, 1))
    print("accel norm    =", np.round(np.linalg.norm(a,axis=1), 5))
    print("jerk norm     =", np.round(np.linalg.norm(jerk,axis=1), 5))

    print("xyz:")
    for i,p in enumerate(xyz):
        print(f"t{i:02d}", np.round(p,5))

print("\nSUCCESS = True")
