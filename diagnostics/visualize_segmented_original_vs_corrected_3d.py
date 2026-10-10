"""
Original vs Corrected action의 rigid-object PA-alignment 과정을
3D dense scatter로 시각화.

- 전체 object point(예: radio_89, 431개)를 그대로 사용 (landmark subset 사용 안 함)
- line으로 잇지 않고 scatter만 사용 -> 점 밀도로 물체 실루엣 표현
- PA alignment(translation/scale/rotation)는 3D에서 그대로 계산
- 5개 panel: RGB context / Raw / Translation removed / Scale normalized / Rotation aligned(Final PA)
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D projection 등록용)


# ============================================================
# 1) PA (Procrustes) alignment -- self-contained, umeyama 알고리즘
# ============================================================

def umeyama_alignment(X, Y, with_scale=True):
    """
    X를 Y에 similarity alignment (s, R, t 추정).
    X, Y: (N,3) 대응되는 point pair.
    반환: s(float), R(3,3), t(3,), X_aligned(N,3)
    """
    assert X.shape == Y.shape and X.shape[1] == 3

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    # covariance
    Sigma = (Yc.T @ Xc) / X.shape[0]

    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scale:
        var_x = (Xc ** 2).sum() / X.shape[0]
        s = np.trace(np.diag(D) @ S) / var_x
    else:
        s = 1.0

    t = mu_y - s * R @ mu_x
    X_aligned = (s * (R @ X.T).T) + t

    return s, R, t, X_aligned


def pa_mpjpe(X, Y, with_scale=True):
    """PA-MPJPE(mm 단위 가정): X를 Y에 alignment 후 residual."""
    _, _, _, X_aligned = umeyama_alignment(X, Y, with_scale=with_scale)
    return float(np.linalg.norm(X_aligned - Y, axis=1).mean()), X_aligned


# ============================================================
# 2) 시각화 -- self-contained, 3D dense scatter
# ============================================================

def _scatter3d(ax, points, color, label=None, s=10, alpha=0.75,
               hollow=False):
    kwargs = dict(s=s, alpha=alpha, linewidths=1.0, label=label)
    if hollow:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   facecolors="none", edgecolors=color, marker="o", **kwargs)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=color, marker="o", **kwargs)


def draw_stage_3d(ax, ref, org, cor, title, view=(20, -60),
                   show_legend=False, residual_text=None):
    """
    ref/org/cor : (N,3) 전체 point 3D 좌표, 같은 correspondence 순서 유지.
    ref = GT0(initial reference)  -> 옅은 회색, 배경
    org = Original prediction     -> 파랑, 채운 점
    cor = Corrected prediction    -> 빨강, hollow circle
    """
    _scatter3d(ax, ref, color="0.75", label="Initial reference (GT0)", s=8, alpha=0.5)
    _scatter3d(ax, org, color="blue", label="Original", s=12, alpha=0.85)
    _scatter3d(ax, cor, color="red", label="Corrected", s=12, alpha=0.85, hollow=True)

    ax.set_title(title, fontsize=12)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()

    if show_legend:
        ax.legend(fontsize=8, frameon=False, loc="upper left")

    if residual_text:
        ax.text2D(0.02, -0.06, residual_text, transform=ax.transAxes,
                  fontsize=9, va="top")


def best_view_from_pca(points):
    """
    GT0 point cloud의 PCA 주축을 이용해서 물체가 가장 넓게 퍼져 보이는
    각도를 자동으로 하나 추정. 맘에 안 들면 elev/azim을 수동으로 바꿔서 재실행.
    """
    centered = points - points.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[2]  # 분산이 가장 작은 축 = 물체를 "내려다보는" 축에 가까움
    azim = np.degrees(np.arctan2(normal[1], normal[0]))
    elev = np.degrees(np.arcsin(np.clip(normal[2], -1, 1)))
    return float(elev + 90 if abs(elev) < 5 else elev), float(azim)


# ============================================================
# 3) 전체 pipeline
# ============================================================

def make_figure(ref, org_raw, cor_raw, title_suffix, out_path, rgb_image=None,
                view=None):
    """
    ref      : (N,3) GT0 initial rigid reference (t=0)
    org_raw  : (N,3) Original action prediction (raw, alignment 전)
    cor_raw  : (N,3) Corrected action prediction (raw, alignment 전)
    rgb_image: (H,W,3) or None -- panel 0용 실제 RGB (없으면 panel 생략)
    """
    if view is None:
        view = best_view_from_pca(ref)

    # ---- stage 1: raw (원본 좌표, alignment 전) ----
    raw_ref, raw_org, raw_cor = ref, org_raw, cor_raw

    # ---- stage 2: translation only (평행이동만 제거, scale/rotation 전) ----
    def center(p):
        return p - p.mean(axis=0)
    cen_ref, cen_org, cen_cor = center(ref), center(org_raw), center(cor_raw)

    # ---- stage 3: scale normalized (uniform scale까지, rotation 전) ----
    def scale_norm(p):
        c = center(p)
        rms = np.sqrt((c ** 2).sum() / c.shape[0])
        return c / rms
    scl_ref, scl_org, scl_cor = scale_norm(ref), scale_norm(org_raw), scale_norm(cor_raw)

    # ---- stage 4: full PA alignment (s,R,t 전부) ----
    org_pa_mm, rot_org = pa_mpjpe(org_raw, ref, with_scale=True)
    cor_pa_mm, rot_cor = pa_mpjpe(cor_raw, ref, with_scale=True)
    rot_ref = ref  # reference는 그대로 (org/cor가 ref에 맞춰짐)

    n_panels = 5 if rgb_image is not None else 4
    fig = plt.figure(figsize=(5 * n_panels, 5))

    panel_idx = 1
    if rgb_image is not None:
        ax0 = fig.add_subplot(1, n_panels, panel_idx)
        ax0.imshow(rgb_image)
        ax0.set_title("1. Target object in RGB", fontsize=12)
        ax0.axis("off")
        panel_idx += 1

    ax1 = fig.add_subplot(1, n_panels, panel_idx, projection="3d")
    draw_stage_3d(ax1, raw_ref, raw_org, raw_cor,
                  f"{panel_idx}. Raw prediction", view=view, show_legend=True)
    panel_idx += 1

    ax2 = fig.add_subplot(1, n_panels, panel_idx, projection="3d")
    draw_stage_3d(ax2, cen_ref, cen_org, cen_cor,
                  f"{panel_idx}. Translation removed", view=view)
    panel_idx += 1

    ax3 = fig.add_subplot(1, n_panels, panel_idx, projection="3d")
    draw_stage_3d(ax3, scl_ref, scl_org, scl_cor,
                  f"{panel_idx}. Scale normalized", view=view)
    panel_idx += 1

    ax4 = fig.add_subplot(1, n_panels, panel_idx, projection="3d")
    residual_text = (
        f"Original PA residual: {org_pa_mm:.3f} mm\n"
        f"Corrected PA residual: {cor_pa_mm:.3f} mm"
    )
    draw_stage_3d(ax4, rot_ref, rot_org, rot_cor,
                  f"{panel_idx}. Rotation aligned (Final PA)",
                  view=view, show_legend=True, residual_text=residual_text)

    fig.suptitle(title_suffix, fontsize=14)
    plt.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Original PA residual:  {org_pa_mm:.3f} mm")
    print(f"Corrected PA residual: {cor_pa_mm:.3f} mm")
    print(f"SAVED: {out_path}")


# ============================================================
# 4) 여기부터 연결 -- 데이터 로딩은 너네 기존 스크립트에서 그대로 가져오기
# ============================================================

def load_clip_arrays(npz_path, t):
    """
    ★★★ 여기 함수 바디만 기존
    diagnostics/visualize_segmented_original_vs_corrected_landmark.py 에서
    복붙해서 채우면 됨. ★★★

    필요한 것 (핸드오프 문서 §9~11, §31 기준):
      1. official PointWorld dataloader로 npz의 clip을 찾음
      2. Original action / Corrected action(NPZ의 corrected_robot_flows)
         각각으로 inference 돌려서 future 3D point 얻음
      3. scene_object_id == 2 (radio_89) mask로 필터링
      4. t=0 시점 GT0 reference, t 시점 Original/Corrected prediction 추출
      5. (선택) t=0 RGB 프레임(cam0)

    반환:
      ref      : (N,3) float, GT0 initial rigid reference, N=431 근처
      org_raw  : (N,3) float, Original action prediction at time t (align 전, raw)
      cor_raw  : (N,3) float, Corrected action prediction at time t (align 전, raw)
      rgb_image: (H,W,3) uint8 or None
      clip_name: str, figure 제목용
    """
    raise NotImplementedError(
        "이 함수 바디를 기존 landmark.py의 데이터 로딩 부분으로 교체하세요. "
        "반환 shape은 docstring 참고."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--out", default=None,
                        help="출력 경로. 생략 시 npz 이름 기반 자동 생성")
    args = parser.parse_args()

    ref, org_raw, cor_raw, rgb_image, clip_name = load_clip_arrays(args.npz, args.t)

    if args.out is None:
        import os
        base = os.path.splitext(os.path.basename(args.npz))[0]
        out_dir = "diagnostics/results/segmented_original_vs_corrected"
        os_makedirs = __import__("os").makedirs
        os_makedirs(out_dir, exist_ok=True)
        args.out = f"{out_dir}/{base}__t{args.t}__3d_dense_pa_process.png"

    title = f"{clip_name}\nTarget: radio_89 | frame={args.t} | object_id=2"
    make_figure(ref, org_raw, cor_raw, title, args.out, rgb_image=rgb_image)


if __name__ == "__main__":
    main()
