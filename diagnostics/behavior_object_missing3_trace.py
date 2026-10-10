import webdataset as wds

# 반드시 pipeline 생성 전에 silent skip을 끈다.
wds.warn_and_continue = wds.reraise_exception

from arguments import parse_args
import dataset_components.dataloader as dlmod
import dataset_components.decoders as decmod
import dataset_components.pipeline as pmod

import diagnostics.behavior_object_mapping_smoke as objmap


# 우리가 full 분석에서 사용한 동일 object-ID diagnostic patch
dlmod.build_flow_sample = objmap.build_flow_sample_with_object_ids
decmod.build_flow_sample = objmap.build_flow_sample_with_object_ids
pmod.grid_sample_transform = objmap.grid_sample_with_object_ids
pmod.gather = objmap.gather_with_object_ids


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


dl, _ = dlmod.build_dataloader(
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
    if isinstance(key, bytes):
        key = key.decode()

    if key in targets:
        print("TARGET SURVIVED:", key)

print("TOTAL YIELDED =", count)
