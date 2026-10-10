import csv
import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from arguments import parse_args
from evaluation.tester import Tester


def squeeze_bool(x):
    x = x.bool()
    if x.dim() == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x


@torch.no_grad()
def get_visibility_count(model, batch):
    """
    Copy of the visibility decision used in official scene_featurizer.py.
    Returns number of conditioning cameras (0/1/2) contributing DINO
    features to each initial scene point.
    """
    enc = model.scene_feature_encoder.scene_encoder

    scene_coord = batch["scene_flows"][:, 0]
    camera_data = enc._extract_camera_data(batch)

    cams = sorted(camera_data.keys())
    if len(cams) != 2:
        raise RuntimeError(f"Expected checkpoint's 2-camera setting, got {cams}")

    rgb = enc._stack_camera_tensors(camera_data, "rgb")
    depth = enc._stack_camera_tensors(camera_data, "depth")
    intr = enc._stack_camera_tensors(camera_data, "intrinsic")
    extr = enc._stack_camera_tensors(camera_data, "extrinsic")

    B, C, H, W, _ = rgb.shape
    Ns = scene_coord.shape[1]
    device = scene_coord.device

    if all("exists" in camera_data[c] for c in cams):
        cam_exists = enc._stack_camera_tensors(camera_data, "exists").bool()
    else:
        cam_exists = torch.ones((B, C), dtype=torch.bool, device=device)

    # ---- Exact official projection path ----
    ones = torch.ones((B, Ns, 1), device=device, dtype=scene_coord.dtype)
    pts_h = torch.cat([scene_coord, ones], dim=-1)
    pts_h = pts_h.unsqueeze(1).expand(-1, C, -1, -1)

    with torch.autocast("cuda", enabled=False):
        world2cam = extr.float()
        intr_f = intr.float()
        pts_h_f = pts_h.float()

        pts_cam = torch.matmul(
            world2cam, pts_h_f.transpose(-2, -1)
        ).transpose(-2, -1)[..., :3]

        pix_h = torch.matmul(
            intr_f, pts_cam.transpose(-2, -1)
        ).transpose(-2, -1)

        z = pix_h[..., 2:3]
        eps = 1e-6
        valid_z = torch.isfinite(z) & (z.abs() > eps)
        safe_z = torch.where(valid_z, z, torch.ones_like(z))

        pixels = pix_h[..., :2] / safe_z
        z_safe = torch.where(valid_z, z, torch.zeros_like(z))
        proj_depth = z_safe.squeeze(-1)

    in_img = (
        (pixels[..., 0] >= 0)
        & (pixels[..., 0] < W)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < H)
        & (proj_depth > 0)
        & valid_z.squeeze(-1)
    )

    norm_xy_depth = torch.stack(
        [
            2 * pixels[..., 0] / (W - 1) - 1,
            2 * pixels[..., 1] / (H - 1) - 1,
        ],
        dim=-1,
    )

    depth_imgs = depth.unsqueeze(2)

    depth_samp = F.grid_sample(
        depth_imgs.view(B * C, 1, H, W),
        norm_xy_depth.view(B * C, Ns, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(-1).squeeze(-1)

    depth_samp = depth_samp.view(B, C, Ns)

    thr_behind = enc.args.depth_threshold
    thr_front = 0.5 * enc.args.depth_threshold

    depth_ok = (
        (proj_depth <= depth_samp + thr_behind)
        & (proj_depth >= depth_samp - thr_front)
    )

    visible = in_img & depth_ok
    visible = visible & cam_exists.unsqueeze(-1)

    # ---- Exact DINO token coverage condition ----
    patch_size = 16
    patch_h = H // patch_size
    patch_w = W // patch_size

    W_eff = float(patch_w * patch_size)
    H_eff = float(patch_h * patch_size)

    in_feat_x = (pixels[..., 0] >= 0.0) & (pixels[..., 0] < W_eff)
    in_feat_y = (pixels[..., 1] >= 0.0) & (pixels[..., 1] < H_eff)
    in_feat = in_feat_x & in_feat_y

    valid_sample = visible & in_feat

    return valid_sample.sum(dim=1)  # (B,Ns), value ∈ {0,1,2}


def finite_mean(x):
    if x.numel() == 0:
        return float("nan")
    return float(x.mean().item())


def main():
    args = parse_args()
    tester = Tester(args)
    model = tester.model
    model.eval()

    dl, _ = tester._build_eval_loader("test", enable_mask=False)

    rows = []

    sanity_l2 = []
    sanity_moved = []

    for bi, batch in enumerate(dl):
        if args.eval_num_batches > 0 and bi >= args.eval_num_batches:
            break

        batch = {
            k: v.to(tester.device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
        }

        outputs = model(batch, training=False)

        # IMPORTANT:
        # Official losses.py compares predicted scene_flows to gt_scene_flows.
        pred = outputs["scene_flows"]
        gt = batch["gt_scene_flows"]

        l2 = torch.norm(pred - gt, dim=-1)  # (B,T,N)

        context = squeeze_bool(batch["scene_context_mask"])
        exists = squeeze_bool(batch["scene_exists"])
        supervised = squeeze_bool(batch["scene_supervised_mask"])
        moved = squeeze_bool(batch["scene_moved_mask"])

        valid = (~context) & exists & supervised

        # Initial observation visibility; replicate through future horizon.
        vis0 = get_visibility_count(model, batch)
        vis = vis0.unsqueeze(1).expand(-1, l2.shape[1], -1)

        # Exact raw-L2 sanity quantities used by official metric.
        sanity_l2.append(finite_mean(l2[valid]))
        sanity_moved.append(finite_mean(l2[valid & moved]))

        key_obj = batch.get("__key__", ["unknown"])
        if isinstance(key_obj, (list, tuple)):
            key = str(key_obj[0])
        else:
            key = str(key_obj)

        for time_scope in ["all_future", "t10"]:
            if time_scope == "all_future":
                l2_s = l2
                valid_s = valid
                moved_s = moved
                vis_s = vis
            else:
                l2_s = l2[:, -1:]
                valid_s = valid[:, -1:]
                moved_s = moved[:, -1:]
                vis_s = vis[:, -1:]

            for scope in ["all", "moved"]:
                base_mask = valid_s
                if scope == "moved":
                    base_mask = base_mask & moved_s

                for nview in [0, 1, 2]:
                    mask = base_mask & (vis_s == nview)
                    vals = l2_s[mask] * 1000.0

                    rows.append({
                        "key": key,
                        "time_scope": time_scope,
                        "scope": scope,
                        "visible_views": nview,
                        "n": int(vals.numel()),
                        "mean_l2_mm": finite_mean(vals),
                    })

    print("\n=== OFFICIAL L2 SANITY ===")
    valid_l2 = [x for x in sanity_l2 if math.isfinite(x)]
    valid_mov = [x for x in sanity_moved if math.isfinite(x)]

    mean_l2_mm = 1000.0 * sum(valid_l2) / len(valid_l2)
    mean_moved_mm = 1000.0 * sum(valid_mov) / len(valid_mov)

    print(f"mean L2       = {mean_l2_mm:.6f} mm")
    print(f"mean moved L2 = {mean_moved_mm:.6f} mm")

    print("\n=== VISIBILITY SUMMARY ===")

    for time_scope in ["all_future", "t10"]:
        print(f"\n[{time_scope}]")
        for scope in ["all", "moved"]:
            print(f"  {scope}:")
            for nview in [0, 1, 2]:
                rr = [
                    r for r in rows
                    if r["time_scope"] == time_scope
                    and r["scope"] == scope
                    and r["visible_views"] == nview
                    and r["n"] > 0
                    and math.isfinite(r["mean_l2_mm"])
                ]

                seq_mean = (
                    sum(r["mean_l2_mm"] for r in rr) / len(rr)
                    if rr else float("nan")
                )
                total_n = sum(r["n"] for r in rr)

                print(
                    f"    views={nview}: "
                    f"sequences={len(rr)}, n={total_n}, "
                    f"mean_sequence_L2={seq_mean:.4f} mm"
                )

    out = "diagnostics/results/droid_visibility_quickcheck.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "key", "time_scope", "scope",
            "visible_views", "n", "mean_l2_mm"
        ])
        w.writeheader()
        w.writerows(rows)

    print("\nSAVED:", out)


if __name__ == "__main__":
    main()
