#!/usr/bin/env python3

import json
import re
import tarfile
from pathlib import Path


SOURCE_ROOT = Path(
    "diagnostics/data/droid_confidence_covered_5"
)
OUTPUT_ROOT = Path(
    "diagnostics/data/visibility_single_sample_wds_20260805"
)

KEY_PATTERN = re.compile(
    r"^(.*?_flows-\d+:\d+)\."
)


def extract_sample_key(member_name: str):
    name = Path(member_name).name
    match = KEY_PATTERN.match(name)
    return match.group(1) if match else None


source_tars = sorted(SOURCE_ROOT.rglob("*.tar"))

if len(source_tars) != 1:
    raise RuntimeError(
        f"Expected exactly one source tar, found {len(source_tars)}"
    )

source_tar = source_tars[0]

members_by_key = {}

with tarfile.open(source_tar, "r") as archive:
    members = [
        member
        for member in archive.getmembers()
        if member.isfile()
    ]

    for member in members:
        key = extract_sample_key(member.name)

        if key is None:
            continue

        members_by_key.setdefault(
            key,
            [],
        ).append(member)

if len(members_by_key) != 5:
    raise RuntimeError(
        f"Expected 5 sample keys, found {len(members_by_key)}"
    )

if OUTPUT_ROOT.exists():
    raise FileExistsError(
        f"Refusing to overwrite: {OUTPUT_ROOT}"
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

manifest = []

with tarfile.open(source_tar, "r") as source_archive:
    for index, key in enumerate(
        sorted(members_by_key),
        start=1,
    ):
        clip_id = f"clip{index:02d}"

        clip_dir = OUTPUT_ROOT / clip_id
        clip_dir.mkdir()

        output_tar = clip_dir / "shard-000000.tar"
        selected_members = members_by_key[key]

        with tarfile.open(output_tar, "w") as output_archive:
            for member in selected_members:
                file_object = source_archive.extractfile(member)

                if file_object is None:
                    raise RuntimeError(
                        f"Could not read member: {member.name}"
                    )

                output_archive.addfile(
                    member,
                    file_object,
                )

        manifest.append(
            {
                "clip_id": clip_id,
                "sample_key": key,
                "member_count": len(selected_members),
                "tar_path": str(output_tar),
            }
        )

manifest_path = OUTPUT_ROOT / "manifest.json"

with open(manifest_path, "w") as file:
    json.dump(
        manifest,
        file,
        indent=2,
    )

print("source_tar =", source_tar)
print("clip_count =", len(manifest))

for item in manifest:
    print(
        item["clip_id"],
        item["member_count"],
        item["sample_key"],
    )

print("manifest =", manifest_path)
