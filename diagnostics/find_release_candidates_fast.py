from pathlib import Path
import tarfile
import io
import numpy as np

ROOT = Path("/home/soenchun/cse/PointWorld_workspace/PointWorld")
DATA = Path("/home/soenchun/cse/PointWorld_workspace/datasets/behavior")

CORR = (
    ROOT
    / "diagnostics/results/"
      "acc_only_boundary_correction_current10457/"
      "corrected_actions"
)

corr_npzs = list(CORR.rglob("*.npz"))

print("corrected NPZ count =", len(corr_npzs))

if not corr_npzs:
    raise RuntimeError("No corrected_actions NPZ files found")


def find_corr(key):
    # Raw WDS key may contain ":" while saved corrected-action
    # filenames use "__". Match both representations.
    variants = {
        str(key),
        str(key).replace(":", "__"),
    }

    for p in corr_npzs:
        name = p.name
        if any(v in name for v in variants):
            return p

    return None


def load_npy_member(tf, member):
    f = tf.extractfile(member)
    if f is None:
        return None
    return np.load(
        io.BytesIO(f.read()),
        allow_pickle=False,
    )


def release_frames(open_arr):
    """
    PointWorld/BEHAVIOR convention observed here:
      True  = open
      False = closed

    release = closed -> open
    """
    a = np.asarray(open_arr).reshape(-1).astype(bool)

    if len(a) < 2:
        return []

    idx = np.flatnonzero(
        (~a[:-1]) & a[1:]
    ) + 1

    # Need at least one future frame after release,
    # so exclude release on final timestep.
    return [
        int(x)
        for x in idx
        if x < len(a) - 1
    ]


found = []
seen = set()

task_dirs = sorted(
    DATA.glob("wds_task*")
)

for task_dir in task_dirs:

    tar_paths = sorted(
        task_dir.rglob("*.tar")
    )

    if not tar_paths:
        continue

    print(
        f"\nSCAN {task_dir.name}:",
        len(tar_paths),
        "tar(s)",
    )

    for tar_path in tar_paths:

        pending = {}

        with tarfile.open(
            tar_path,
            "r:*",
        ) as tf:

            for member in tf:

                if not member.isfile():
                    continue

                base = Path(
                    member.name
                ).name

                side = None

                for s in [
                    "right",
                    "left",
                ]:
                    suffix = (
                        f".{s}_gripper_open.npy"
                    )

                    if base.endswith(
                        suffix
                    ):
                        side = s
                        key = base[
                            :-len(suffix)
                        ]
                        break

                if side is None:
                    continue

                if key in seen:
                    continue

                arr = load_npy_member(
                    tf,
                    member,
                )

                if arr is None:
                    continue

                pending.setdefault(
                    key,
                    {},
                )[side] = arr

                # Evaluate as soon as both sides are available.
                if (
                    "right" not in pending[key]
                    or
                    "left" not in pending[key]
                ):
                    continue

                rr = release_frames(
                    pending[key]["right"]
                )

                lr = release_frames(
                    pending[key]["left"]
                )

                seen.add(key)

                if not rr and not lr:
                    del pending[key]
                    continue

                corr = find_corr(key)

                # We only want clips that actually belong to the
                # 2,683 successful boundary-correction population.
                if corr is None:
                    del pending[key]
                    continue

                found.append(
                    (
                        key,
                        rr,
                        lr,
                        corr,
                    )
                )

                print(
                    f"\nCANDIDATE {len(found)}"
                )
                print(" key   =", key)
                print(" right release frames =", rr)
                print(" left  release frames =", lr)
                print(" npz   =", corr)

                del pending[key]

                if len(found) >= 8:
                    break

        if len(found) >= 8:
            break

    if len(found) >= 8:
        break


print("\n======================================")
print("FOUND correction-paired release clips =", len(found))
print("======================================")

for i, (key, rr, lr, corr) in enumerate(
    found,
    1,
):
    print(
        f"{i:02d} | {key}"
        f" | R={rr}"
        f" | L={lr}"
    )
