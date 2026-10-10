import os
import re
from pathlib import Path

import pandas as pd
import torch

from arguments import parse_args
from evaluation.tester import Tester

TASK = os.environ["PW_TASK"]

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model"
)

D = ROOT / f"fixed_camera_pair_task{TASK}"

FEATURE_FILE = (
    D / f"task{TASK}_head_scene_action_query_features.csv"
)

OUT = (
    D / f"task{TASK}_pair_HLR__per_sample.csv"
)

# ---------------------------------------------------------------------
# Only evaluate the exact clips already used in Stage 1.
# ---------------------------------------------------------------------

feat = pd.read_csv(FEATURE_FILE)

targets = feat["clip"].astype(str).tolist()

alias_to_target = {}
for c in targets:
    alias_to_target[c] = c
    alias_to_target[c.split("/")[-1]] = c

target_set = set(targets)

# Resume safely if partially completed.
rows = []
done = set()

if OUT.exists():
    old = pd.read_csv(OUT)

    if "clip" in old.columns:
        rows = old.to_dict("records")
        done = set(old["clip"].astype(str))

    if len(done) == len(target_set):
        print(
            f"TASK = {TASK} | rows = {len(done)} | "
            f"already complete"
        )
        print(f"output = {OUT}")
        print("SUCCESS = True")
        raise SystemExit(0)

print(
    f"TASK = {TASK} | "
    f"targets = {len(target_set)} | "
    f"already_done = {len(done)}"
)

# ---------------------------------------------------------------------
# Official PointWorld Tester.
# ---------------------------------------------------------------------

args = parse_args()

tester = Tester(args)

# Enforce H + L + R.
# BEHAVIOR has exactly these 3 decoded cameras.
tester.args.eval_min_num_cameras = 3
tester.args.eval_max_num_cameras = 3

dl, info = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

tester.model.eval()

printed_camera_audit = False
seen_raw = 0
new_count = 0

def normalize_key(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)

def resolve_target(raw_key):
    if raw_key in alias_to_target:
        return alias_to_target[raw_key]

    base = raw_key.split("/")[-1]

    if base in alias_to_target:
        return alias_to_target[base]

    return None

@torch.no_grad()
def main():

    global printed_camera_audit
    global seen_raw
    global new_count
    global rows

    for batch in dl:

        seen_raw += 1

        raw_key = normalize_key(
            batch["__key__"][0]
        )

        clip = resolve_target(raw_key)

        # Important: no GPU forward for clips outside Stage-1 set.
        if clip is None:
            continue

        if clip in done:
            continue

        # Audit once that the loader really supplied 3 cameras.
        if not printed_camera_audit:

            cams = sorted({
                m.group(1)
                for k in batch.keys()
                for m in [re.match(r"(cam\d+)_", str(k))]
                if m is not None
            })

            print("Detected camera prefixes =", cams)

            if len(cams) != 3:
                raise RuntimeError(
                    f"Expected exactly 3 cameras, got {cams}"
                )

            printed_camera_audit = True

        batch = {
            k: (
                v.to(
                    tester.device,
                    non_blocking=True,
                )
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

        outputs = tester.model(
            batch,
            training=False,
        )

        _, loss_dict = tester.model.loss_fn(
            outputs,
            batch,
            training=False,
        )

        metric = None

        for candidate in [
            "behavior/l2_moved/mean",
            "l2_moved/mean",
        ]:
            if candidate in loss_dict:
                metric = candidate
                break

        if metric is None:
            raise KeyError(
                "Could not find moved-L2 metric. "
                f"Available keys: {sorted(loss_dict.keys())}"
            )

        value = loss_dict[metric]

        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        else:
            value = float(value)

        rows.append({
            "clip": clip,
            "behavior/l2_moved/mean": value,
        })

        done.add(clip)
        new_count += 1

        if len(done) % 200 == 0:
            print(
                f"{TASK} HLR: {len(done)}/{len(target_set)}"
            )

            pd.DataFrame(rows).drop_duplicates(
                "clip",
                keep="last",
            ).to_csv(
                OUT,
                index=False,
            )

        if len(done) == len(target_set):
            break

    result = (
        pd.DataFrame(rows)
        .drop_duplicates("clip", keep="last")
    )

    result.to_csv(
        OUT,
        index=False,
    )

    missing = target_set - set(
        result["clip"].astype(str)
    )

    print("=" * 80)
    print(
        f"TASK = {TASK} | "
        f"rows = {len(result)} | "
        f"expected = {len(target_set)}"
    )
    print("missing =", len(missing))
    print("output =", OUT)

    if missing:
        print("first missing =", sorted(missing)[:10])
        raise RuntimeError(
            f"Missing {len(missing)} target clips"
        )

    print("SUCCESS = True")


if __name__ == "__main__":
    main()
