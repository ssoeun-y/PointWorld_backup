from huggingface_hub import HfApi


REPO = "nvidia/PointWorld-BEHAVIOR"
TASKS = [
    "task-0001",   # picking_up_trash -> soda cans
    "task-0003",   # cleaning_up_plates_and_food -> bowl / plate
]


def human_bytes(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)

    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0


api = HfApi()

print("=" * 90)
print("POINTWORLD-BEHAVIOR CROSS-OBJECT PACKAGE SIZE AUDIT")
print("=" * 90)
print("repo =", REPO)
print()

# files_metadata=True is important: gives actual file sizes
info = api.dataset_info(
    REPO,
    files_metadata=True,
)

all_files = info.siblings

grand_total = 0

for task in TASKS:

    prefix = f"behavior/flows/{task}/"

    parts = []

    for f in all_files:

        name = getattr(f, "rfilename", "")

        if (
            name.startswith(prefix)
            and "package.tar.zst.part-" in name
        ):
            size = getattr(f, "size", None)

            assert size is not None, (
                "Missing size metadata",
                name,
            )

            parts.append(
                (name, int(size))
            )

    parts = sorted(parts)

    print("-" * 90)
    print("TASK =", task)
    print("n_package_parts =", len(parts))

    assert len(parts) > 0, (
        "No packaged PointWorld data found for",
        task,
    )

    total = sum(
        size for _, size in parts
    )

    grand_total += total

    print(
        "compressed_download_size =",
        human_bytes(total),
    )

    print()
    print("PARTS:")

    for name, size in parts:
        print(
            f"  {human_bytes(size):>12s}  {name}"
        )

    print()


print("=" * 90)
print(
    "task-0001 + task-0003 compressed total =",
    human_bytes(grand_total),
)
print()
print("NO DOWNLOAD")
print("NO INFERENCE")
print("NO MBNSF")
print("POINTWORLD OFFICIAL DATA REPO ONLY")
print("SUCCESS = True")
print("=" * 90)
