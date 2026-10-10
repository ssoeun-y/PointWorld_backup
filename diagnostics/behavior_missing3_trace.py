import webdataset as wds

# warn_and_continue 대신 실제 exception을 그대로 올림.
wds.warn_and_continue = wds.reraise_exception

from arguments import parse_args
from dataset_components.dataloader import build_dataloader

args = parse_args(skip_command_line=True)

args.domains = ["behavior"]
args.data_dirs = [
    "/mnt/hdd/sunjaeyoon/cse/PointWorld_workspace/"
    "datasets/behavior/wds_task0000"
]
args.batch_size = 1
args.num_workers = 0
args.eval_num_workers = 0
args.seed = 42
args.deterministic_data = True

targets = {
    "task-0000_episode_00001470-0:11",
    "task-0000_episode_00001590-0:11",
    "task-0000_episode_00002740-0:11",
}

dl, _ = build_dataloader(
    args,
    mode="test",
    rank=0,
    world_size=1,
    override_splits="test",
)

count = 0

for batch in dl:
    count += 1

    key = batch["__key__"]
    if isinstance(key, (list, tuple)):
        key = key[0]

    print("YIELD", count, key)

    if key in targets:
        print("TARGET SURVIVED:", key)

print("TOTAL YIELDED =", count)
