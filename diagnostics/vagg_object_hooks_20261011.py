import zlib
import numpy as np
from scipy.spatial import cKDTree

_INSTALLED = False

def install():
    global _INSTALLED
    if _INSTALLED:
        return

    from dataset_components import decoders, pipeline

    old_decode = decoders.SimDataDecoder.decode_wds
    old_flatten = decoders._flatten_camera_scene_data
    old_grid = pipeline.grid_sample_transform
    old_gather = pipeline.gather

    def decode_with_ids(self, sample):
        out = old_decode(self, sample)

        for camera, info in out.items():
            raw = sample.get(camera + "_local_scene_points")
            if not isinstance(raw, dict) or not raw:
                raise RuntimeError("MISSING RAW MESH DATA: " + camera)

            names = list(raw.keys())
            T, N = info["scene_flows"].shape[:2]

            labels = np.concatenate([
                np.full(
                    len(raw[name]),
                    (zlib.crc32(str(name).encode()) & 0x7fffffff) + 1,
                    dtype=np.int64
                )
                for name in names
            ])

            if len(labels) != N:
                raise RuntimeError(
                    f"OBJECT ID SIZE MISMATCH {camera}: {len(labels)} != {N}"
                )

            info["scene_object_id"] = np.broadcast_to(
                labels[None, :, None], (T, N, 1)
            ).copy()

        return out

    def flatten_with_ids(scene, sample):
        out = old_flatten(scene, sample)
        for camera, data in scene.items():
            out[camera + "_scene_object_id"] = data["scene_object_id"]
        return out

    def grid_with_ids(sample, grid_size, mode):
        if "scene_object_id" not in sample:
            return old_grid(sample, grid_size, mode)

        xyz_before = sample["scene_flows"][0].copy()
        ids_before = sample["scene_object_id"].copy()

        out = old_grid(sample, grid_size, mode)

        distance, index = cKDTree(xyz_before).query(
            out["scene_flows"][0], k=1
        )

        if np.any(distance > 1e-6):
            raise RuntimeError("OBJECT ID GRID ALIGNMENT FAILED")

        out["scene_object_id"] = ids_before[:, index, :].copy()

        if out["scene_object_id"].shape[:2] != out["scene_flows"].shape[:2]:
            raise RuntimeError("OBJECT ID SHAPE MISALIGNMENT")

        return out

    def gather_with_ids(sample, fields, domain):
        out = old_gather(
            sample, list(fields) + ["scene_object_id"], domain
        )
        if "behavior" in str(domain) and "scene_object_id" not in out:
            raise RuntimeError("OBJECT ID LOST DURING GATHER")
        return out

    decoders.SimDataDecoder.decode_wds = decode_with_ids
    decoders._flatten_camera_scene_data = flatten_with_ids
    pipeline.grid_sample_transform = grid_with_ids
    pipeline.gather = gather_with_ids

    _INSTALLED = True
    print("OBJECT_ID_PIPELINE_INSTALLED", flush=True)
