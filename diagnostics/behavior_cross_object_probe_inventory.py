from pathlib import Path
import json
import re

import h5py
import numpy as np


ROOT = Path(
    "diagnostics/data/behavior_cross_object_probe_raw"
)

FILES = sorted(ROOT.rglob("*.hdf5"))

assert len(FILES) == 3, FILES

OUT = ROOT / "cross_object_probe_inventory.txt"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def decode_scalar(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")

    if isinstance(x, str):
        return x

    if np.isscalar(x):
        try:
            return str(x.item())
        except Exception:
            return str(x)

    return None


def flatten_strings(x, prefix=""):
    out = []

    if isinstance(x, dict):
        for k, v in x.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten_strings(v, p))

    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            p = f"{prefix}[{i}]"
            out.extend(flatten_strings(v, p))

    elif isinstance(x, str):
        out.append((prefix, x))

    return out


def try_parse_json(text):
    text = text.strip()

    if not text:
        return []

    if not (
        (text.startswith("{") and text.endswith("}"))
        or
        (text.startswith("[") and text.endswith("]"))
    ):
        return []

    try:
        obj = json.loads(text)
    except Exception:
        return []

    return flatten_strings(obj)


INTEREST = re.compile(
    r"""
    object|instance|category|model|scene|task|bddl|
    cup|mug|bottle|can|box|bowl|plate|jar|container|
    radio|table|sofa|cabinet|drawer|door|shelf|
    pot|pan|kettle|pitcher|glass|dish|tray
    """,
    re.I | re.X,
)


# ------------------------------------------------------------
# Scan
# ------------------------------------------------------------

lines = []

lines.append("=" * 110)
lines.append("BEHAVIOR CROSS-OBJECT PROBE INVENTORY")
lines.append("=" * 110)
lines.append(f"n_files = {len(FILES)}")
lines.append("")

for path in FILES:

    lines.append("")
    lines.append("#" * 110)
    lines.append(f"FILE = {path}")
    lines.append("#" * 110)

    text_hits = []
    paths = []

    with h5py.File(path, "r") as f:

        lines.append("")
        lines.append("ROOT KEYS:")
        lines.append(str(list(f.keys())))

        if len(f.attrs):
            lines.append("")
            lines.append("ROOT ATTRIBUTES:")

            for k, v in f.attrs.items():

                try:
                    arr = np.asarray(v)

                    if arr.ndim == 0:
                        val = decode_scalar(arr.item())

                    elif arr.size <= 100:
                        vals = []

                        for q in arr.reshape(-1):
                            s = decode_scalar(q)

                            if s is not None:
                                vals.append(s)

                        val = " | ".join(vals)

                    else:
                        val = (
                            f"<array shape={arr.shape} "
                            f"dtype={arr.dtype}>"
                        )

                except Exception:
                    val = repr(v)

                lines.append(f"  {k} = {val}")

        def visit(name, obj):

            if isinstance(obj, h5py.Group):
                paths.append(
                    f"GROUP   {name}"
                )
            else:
                paths.append(
                    f"DATASET {name} "
                    f"shape={obj.shape} "
                    f"dtype={obj.dtype}"
                )

            # ---------- attributes ----------
            for key, val in obj.attrs.items():

                try:
                    arr = np.asarray(val)
                except Exception:
                    continue

                values = []

                if arr.ndim == 0:
                    s = decode_scalar(arr.item())

                    if s is not None:
                        values.append(s)

                elif (
                    arr.size <= 1000
                    and arr.dtype.kind in ("S", "U", "O")
                ):
                    for q in arr.reshape(-1):
                        s = decode_scalar(q)

                        if s is not None:
                            values.append(s)

                for s in values:

                    label = f"{name or '/'} @ {key}"

                    if (
                        INTEREST.search(label)
                        or INTEREST.search(s)
                    ):
                        text_hits.append(
                            (label, s)
                        )

                    for jp, js in try_parse_json(s):
                        jl = f"{label} JSON:{jp}"

                        if (
                            INTEREST.search(jl)
                            or INTEREST.search(js)
                        ):
                            text_hits.append(
                                (jl, js)
                            )

            # ---------- small string datasets ----------
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

                        s = decode_scalar(q)

                        if s is None:
                            continue

                        label = name

                        if (
                            INTEREST.search(label)
                            or INTEREST.search(s)
                        ):
                            text_hits.append(
                                (label, s)
                            )

                        for jp, js in try_parse_json(s):
                            jl = f"{label} JSON:{jp}"

                            if (
                                INTEREST.search(jl)
                                or INTEREST.search(js)
                            ):
                                text_hits.append(
                                    (jl, js)
                                )

        f.visititems(visit)

    lines.append("")
    lines.append("HDF5 STRUCTURE:")
    for x in paths[:300]:
        lines.append("  " + x)

    if len(paths) > 300:
        lines.append(
            f"  ... ({len(paths)-300} more)"
        )

    lines.append("")
    lines.append("OBJECT / SCENE / TASK TEXT HITS:")

    # de-duplicate
    seen = set()
    dedup = []

    for label, text in text_hits:

        key = (label, text)

        if key in seen:
            continue

        seen.add(key)
        dedup.append(
            (label, text)
        )

    if not dedup:
        lines.append("  NONE")
    else:
        for label, text in dedup[:500]:

            text = text.replace(
                "\n",
                " ",
            )

            if len(text) > 1000:
                text = text[:1000] + " ..."

            lines.append(
                f"  [{label}] {text}"
            )

        if len(dedup) > 500:
            lines.append(
                f"  ... ({len(dedup)-500} more hits)"
            )


OUT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\n".join(lines))
print()
print("=" * 110)
print("saved =", OUT)
print("NO INFERENCE")
print("METADATA / HDF5 INSPECTION ONLY")
print("SUCCESS = True")
print("=" * 110)
