from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

BASE = Path("/home/soenchun/cse/PointWorld_workspace")
CACHE = BASE / "PointWorld/diagnostics/results/droid_object_eval_cache_20261010"
OUT = BASE / "PointWorld/diagnostics/results/droid_sam2_20261010"
WEIGHT = BASE / "third_party/checkpoints/sam2.1_hiera_tiny.pt"

files = sorted(p for p in CACHE.glob("*.npz")
               if not p.name.endswith(".part.npz"))
assert files, "DROID cache missing"
assert WEIGHT.is_file(), WEIGHT
assert torch.cuda.is_available(), "GPU not available"

torch.set_grad_enabled(False)

model = build_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    str(WEIGHT),
    device="cuda",
    apply_postprocessing=False,
)

generator = SAM2AutomaticMaskGenerator(
    model,
    points_per_side=24,
    points_per_batch=16,
    pred_iou_thresh=0.70,
    stability_score_thresh=0.88,
    crop_n_layers=0,
    min_mask_region_area=0,
)

canvas = Image.new("RGB", (1280, 720), "white")

with np.load(files[0], allow_pickle=False) as z:
    clip = str(z["clip"])
    print("CLIP:", clip, flush=True)

    for ci in range(2):
        rgb = z[f"cam{ci}_initial_rgb"].astype(np.uint8)

        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            records = generator.generate(rgb)

        print(
            f"CAM{ci}: {len(records)} SAM masks",
            flush=True
        )

        masks = np.stack(
            [r["segmentation"] for r in records],
            axis=0
        ).astype(bool) if records else np.zeros(
            (0, 180, 320), dtype=bool
        )

        np.savez_compressed(
            OUT / f"sam2_cam{ci}_candidate_masks.npz",
            masks=masks,
            predicted_iou=np.array(
                [r["predicted_iou"] for r in records]
            ),
            stability=np.array(
                [r["stability_score"] for r in records]
            ),
            area=np.array(
                [r["area"] for r in records]
            ),
            clip=np.array(clip),
        )

        rgb_big = Image.fromarray(rgb).resize((640, 360))
        overlay = rgb.copy()

        # Visualization only: not yet accepted object masks.
        selected = []
        occupied = np.zeros(rgb.shape[:2], dtype=bool)

        order = sorted(
            range(len(records)),
            key=lambda i: records[i]["predicted_iou"],
            reverse=True,
        )

        colors = [
            (255, 60, 60), (50, 220, 80),
            (255, 170, 30), (150, 70, 255),
            (30, 210, 255), (255, 70, 210),
        ]

        for j in order:
            m = masks[j]
            area = int(m.sum())

            if not 25 <= area <= 12000:
                continue

            overlap = int((m & occupied).sum())
            if overlap / max(area, 1) > 0.50:
                continue

            selected.append(j)
            occupied |= m

            if len(selected) >= 20:
                break

        for k, j in enumerate(selected):
            color = np.array(colors[k % len(colors)])
            m = masks[j]
            overlay[m] = (
                0.55 * overlay[m] + 0.45 * color
            ).astype(np.uint8)

        overlay_img = Image.fromarray(overlay).resize((640, 360))
        draw = ImageDraw.Draw(overlay_img)

        for k, j in enumerate(selected):
            ys, xs = np.where(masks[j])
            if len(xs) == 0:
                continue

            x = int(np.median(xs) * 2)
            y = int(np.median(ys) * 2)
            draw.text(
                (x, y),
                str(k + 1),
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )

        canvas.paste(rgb_big, (0, ci * 360))
        canvas.paste(overlay_img, (640, ci * 360))

        print(
            f"CAM{ci}: displayed {len(selected)} candidates",
            flush=True
        )

output = OUT / "sam2_first_clip_visual_audit.png"
canvas.save(output)

print("SAVED:", output, flush=True)
print("SAM2_FIRST_CLIP_COMPLETE", flush=True)
