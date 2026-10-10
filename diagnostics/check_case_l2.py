import numpy as np

D = "diagnostics/results/problem_acc_levels"

def align(P, Q, scale):
    mp, mq = P.mean(0), Q.mean(0)
    P0, Q0 = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(P0.T @ Q0)
    d = np.sign(np.linalg.det(U @ Vt))
    Dm = np.diag([1, 1, d])
    R = U @ Dm @ Vt
    s = (S * np.diag(Dm)).sum() / (P0 ** 2).sum() if scale else 1.0
    return s * (P0 @ R) + mq

def stats(gt, pred):
    l2 = np.linalg.norm(pred - gt, axis=1).mean()
    pr = np.linalg.norm(align(pred, gt, False) - gt, axis=1).mean()
    ps = np.linalg.norm(align(pred, gt, True) - gt, axis=1).mean()
    return l2, pr, ps

print("== case_*.npz vs 저장된 pa_csv ==")
for lv in ["LOW", "MID", "HIGH"]:
    z = np.load(f"{D}/case_{lv}.npz", allow_pickle=True)
    l2, pr, ps = stats(z["ref"], z["pred"])
    print(f"{lv}: frame={int(z['frame'])} rms_acc={float(z['rms_acc']):.4f} unit_scale={float(z['unit_scale']):.6g} pa_csv={float(z['pa_csv']):.4f}")
    print(f"     x1000 -> L2={l2*1000:.3f}  PA_rigid={pr*1000:.3f}  PA_scale={ps*1000:.3f}")

print("\n== raw_frame_data: frame별 (x1000) ==")
for lv in ["LOW", "MID", "HIGH"]:
    z = np.load(f"{D}/raw_frame_data_{lv}.npz", allow_pickle=True)
    for t in [1, 5, 10]:
        gt, pd_ = z[f"frame_data__{t}__gt"], z[f"frame_data__{t}__pred"]
        l2, pr, ps = stats(gt, pd_)
        print(f"{lv} t={t:2d} n={len(gt)} L2={l2*1000:.3f} PA_rigid={pr*1000:.3f} PA_scale={ps*1000:.3f}")
