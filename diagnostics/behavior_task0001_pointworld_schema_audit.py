from pathlib import Path
import h5py
import numpy as np
import json
import re

ROOT = Path(
    "diagnostics/data/"
    "pointworld_behavior_task0001_restored/"
    "behavior/flows/task-0001"
)

FILES = sorted(ROOT.glob("episode_*.hdf5"))

assert len(FILES) == 200, (
    f"Expected 200 restored episodes, got {len(FILES)}"
)

TARGET = re.compile(
    r"can[_\s-]*of[_\s-]*soda|"
    r"mesh|object|scene|trajectory|point|name",
    re.I,
)


def decode(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, str):
        return x
    return None


def flatten_json(obj, prefix=""):
    out = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten_json(v, p))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            out.extend(flatten_json(v, p))

    elif isinstance(obj, str):
        out.append((prefix, obj))

    return out


def inspect_string(label, s, hits):
    if not s:
        return

    if TARGET.search(label) or TARGET.search(s):
        hits.append((label, s))

    st = s.strip()

    if (
        (st.startswith("{") and st.endswith("}"))
        or
        (st.startswith("[") and st.endswith("]"))
    ):
        try:
            obj = json.loads(st)
        except Exception:
            return

        for p, v in flatten_json(obj):
            lab = f"{label} JSON:{p}"

            if TARGET.search(lab) or TARGET.search(v):
                hits.append((lab, v))


# ============================================================
# 1. Exact schema of first restored PointWorld episode
# ============================================================

first = FILES[0]

print("=" * 110)
print("POINTWORLD TASK-0001 HDF5 SCHEMA AUDIT")
print("=" * 110)
print("n_episode_files =", len(FILES))
print("first_file      =", first)
print()

schema_rows = []
metadata_hits = []

with h5py.File(first, "r") as f:

    print("ROOT KEYS =", list(f.keys()))
    print()

    for k, v in f.attrs.items():
        try:
            arr = np.asarray(v)

            if arr.ndim == 0:
                s = decode(arr.item())
                if s is not None:
                    inspect_string(f"/ @ {k}", s, metadata_hits)
        except Exception:
            pass

    def visit(name, obj):

        if isinstance(obj, h5py.Group):
            schema_rows.append(
                f"GROUP   {name}"
            )

        elif isinstance(obj, h5py.Dataset):
            schema_rows.append(
                f"DATASET {name} "
                f"shape={obj.shape} "
                f"dtype={obj.dtype}"
            )

        # Attributes
        for k, v in obj.attrs.items():
            try:
                arr = np.asarray(v)
            except Exception:
                continue

            vals = []

            if arr.ndim == 0:
                s = decode(arr.item())
                if s is not None:
                    vals.append(s)

            elif (
                arr.size <= 5000
                and arr.dtype.kind in ("S", "U", "O")
            ):
                for q in arr.reshape(-1):
                    s = decode(q)
                    if s is not None:
                        vals.append(s)

            for s in vals:
                inspect_string(
                    f"{name or '/'} @ {k}",
                    s,
                    metadata_hits,
                )

        # String datasets only
        if isinstance(obj, h5py.Dataset):

            if (
                obj.dtype.kind in ("S", "U", "O")
                and obj.size <= 10000
            ):
                try:
                    arr = np.asarray(obj[()])
                except Exception:
                    return

                for q in arr.reshape(-1):
                    s = decode(q)

                    if s is not None:
                        inspect_string(
                            name,
                            s,
                            metadata_hits,
                        )

    f.visititems(visit)


print("=" * 110)
print("EXACT HDF5 STRUCTURE")
print("=" * 110)

for row in schema_rows:
    print(row)


print()
print("=" * 110)
print("FIRST-EPISODE RELEVANT METADATA")
print("=" * 110)

seen = set()

for label, val in metadata_hits:

    key = (label, val)

    if key in seen:
        continue

    seen.add(key)

    val2 = val.replace("\n", " ")

    if len(val2) > 600:
        val2 = val2[:600] + " ..."

    print(f"[{label}] {val2}")


# ============================================================
# 2. Search all 200 episodes specifically for soda-can strings
# ============================================================

print()
print("=" * 110)
print("SEARCHING ALL 200 POINTWORLD EPISODES FOR SODA-CAN IDENTIFIERS")
print("=" * 110)

CAN_RE = re.compile(
    r"can[_\s-]*of[_\s-]*soda",
    re.I,
)

episode_hits = []

for idx, path in enumerate(FILES):

    hits = []

    with h5py.File(path, "r") as f:

        def scan(name, obj):

            # attrs
            for k, v in obj.attrs.items():
                try:
                    arr = np.asarray(v)
                except Exception:
                    continue

                if (
                    arr.ndim == 0
                    and arr.dtype.kind in ("S", "U", "O")
                ):
                    vals = [arr.item()]

                elif (
                    arr.size <= 10000
                    and arr.dtype.kind in ("S", "U", "O")
                ):
                    vals = arr.reshape(-1)

                else:
                    vals = []

                for q in vals:
                    s = decode(q)

                    if s is not None and CAN_RE.search(s):
                        hits.append(
                            (f"{name or '/'} @ {k}", s)
                        )

            # string datasets
            if (
                isinstance(obj, h5py.Dataset)
                and obj.dtype.kind in ("S", "U", "O")
                and obj.size <= 10000
            ):
                try:
                    vals = np.asarray(obj[()]).reshape(-1)
                except Exception:
                    return

                for q in vals:
                    s = decode(q)

                    if s is not None and CAN_RE.search(s):
                        hits.append((name, s))

        # root attrs too
        for k, v in f.attrs.items():

            try:
                arr = np.asarray(v)
            except Exception:
                continue

            if (
                arr.ndim == 0
                and arr.dtype.kind in ("S", "U", "O")
            ):
                s = decode(arr.item())

                if s is not None and CAN_RE.search(s):
                    hits.append((f"/ @ {k}", s))

        f.visititems(scan)

    if hits:
        episode_hits.append(
            (path.name, hits)
        )


print("episodes_with_literal_can_of_soda =", len(episode_hits))

for filename, hits in episode_hits[:10]:

    print()
    print("FILE =", filename)

    done = set()

    for label, s in hits:

        k = (label, s)

        if k in done:
            continue

        done.add(k)

        s2 = s.replace("\n", " ")

        if len(s2) > 500:
            s2 = s2[:500] + " ..."

        print(f"  [{label}] {s2}")


print()
print("=" * 110)
print("AUDIT SUMMARY")
print("=" * 110)
print("episode_files_exact_200 =", len(FILES) == 200)
print(
    "episodes_with_literal_can_of_soda =",
    len(episode_hits),
)
print("NO INFERENCE")
print("NO MBNSF")
print("NO FILE MODIFICATION")
print("POINTWORLD OFFICIAL RESTORED HDF5 ONLY")
print("SUCCESS = True")
print("=" * 110)
