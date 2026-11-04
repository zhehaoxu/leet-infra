"""SVD-LLM 的 truncation-aware 白化压缩与闭式更新。

官方 SVDLLM.py 的 per-Linear 压缩管线（whitening + local_update 类）：
校准 H=Σinpᵀinp 的 Cholesky 因子 L 已由上游产出（profiling 阶段，不入判题域），
本题从 L 出发：

1. 白化域 SVD：W_scale = W @ L，svd(W_scale)——白化保证「截断小奇异值 ⇔ 低
   压缩损失」的直接映射（truncation-aware 核心）；
2. 参数预算：k = int(M·N·ratio/(M+N))（压缩后参数量 ≈ ratio 倍）；
3. 因子拆分：svd_u = U[:,:k] @ √diag(S[:k])，svd_v = √diag(S[:k]) @ (VT[:k,:] @ L⁻¹)
   ——v 侧右乘 L⁻¹ 把白化抵消回去；direct_update=True 跳过白化（v 侧无 L⁻¹）；
4. 闭式更新（layer-wise closed-form update）：x' = x @ svd_v_coreᵀ @ Σ_k，
   updated_uT = lstsq(x', y).solution（最小二乘补偿截断损失）；
5. fasterprune 因子：append_u = updated_uTᵀ @ √Σ_k，append_v = √Σ_k @ truc_v；
6. 相对误差：err = ‖y − x@Ŵᵀ‖_F / ‖y‖_F（截断与更新后各一份）。

L 非满秩时官方防御：+1e-6·I 后再求逆。
"""

import math

import torch


def _validate_2d(t, name, rows=None, cols=None):
    if not isinstance(t, torch.Tensor) or t.dim() != 2 \
            or not torch.is_floating_point(t):
        raise ValueError(f"{name} 必须是 2 维浮点张量")
    if rows is not None and t.shape[0] != rows:
        raise ValueError(f"{name} 必须是 ({rows}, {cols}) 形状")
    if cols is not None and t.shape[1] != cols:
        raise ValueError(f"{name} 必须是 ({rows}, {cols}) 形状")


def _invert(matrix):
    """官方防御: 求逆失败 -> +1e-6*I 再逆."""
    try:
        return torch.linalg.inv(matrix)
    except Exception:
        eye = torch.eye(matrix.shape[0], dtype=matrix.dtype,
                        device=matrix.device)
        return torch.linalg.inv(matrix + 1e-6 * eye)


def solve(weight, scaling_diag_matrix, x, y, ratio=0.2, direct_update=False):
    _validate_2d(weight, "weight")
    M, N = weight.shape
    use_white = not direct_update
    if use_white:
        _validate_2d(scaling_diag_matrix, "scaling_diag_matrix", N, N)
    else:
        if scaling_diag_matrix is not None:
            raise ValueError("direct_update=True 时不需要 scaling_diag_matrix")
    _validate_2d(x, "x", None, N)
    _validate_2d(y, "y", x.shape[0], M)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) \
            or not math.isfinite(ratio) or not (0.0 < ratio <= 1.0):
        raise ValueError("ratio 必须是 (0, 1] 内的有限数值")

    wd = weight.double()
    if use_white:
        L = scaling_diag_matrix.double()
        L_inv = _invert(L)
        W_scale = wd @ L
    else:
        W_scale = wd
    U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)

    k = int(M * N * ratio / (M + N))
    if k < 1:
        raise ValueError(f"压缩率过小: k ({k}) 必须 >= 1")
    # k 上界无需校验: ratio<=1 时 k=int(MN/(M+N)) < min(M,N) 恒成立

    truc_u = U[:, :k]
    truc_s = S[:k]
    if use_white:
        truc_v = VT[:k, :] @ L_inv
    else:
        truc_v = VT[:k, :]
    sigma = torch.diag(truc_s)
    sqrt_sigma = torch.sqrt(sigma)
    svd_u = truc_u @ sqrt_sigma
    svd_v = sqrt_sigma @ truc_v
    new_w = truc_u @ sigma @ truc_v

    x_d = x.double()
    y_d = y.double()
    x_proj = (x_d @ truc_v.t()) @ sigma
    updated_u_t = torch.linalg.lstsq(x_proj, y_d).solution

    append_u = updated_u_t.t() @ sqrt_sigma
    append_v = sqrt_sigma @ truc_v

    def rel_err(w_mat):
        out_hat = x_d @ w_mat.t()
        return float(torch.sqrt(torch.sum((y_d - out_hat) ** 2))
                     / torch.linalg.matrix_norm(y_d, ord="fro"))

    return {
        "k": k,
        "svd_u": svd_u,
        "svd_v": svd_v,
        "new_w": new_w,
        "updated_u_t": updated_u_t,
        "append_u": append_u,
        "append_v": append_v,
        "error": rel_err(new_w),
        "updated_error": rel_err(append_u @ append_v),
    }
