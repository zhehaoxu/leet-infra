"""FRoD (arXiv:2512.23485, AAAI 2026) 全秩旋转自由度微调 — 判题复现。

HF PEFT tuners/frod。三段账全确定性：

A. shared_projection —— 同类别各层 base weight (out,in) 行拼接（float32 CPU）
   后的跨层联合分解：瘦阵（行 < 列）走 SVD 取 vh.T；否则 QR 分解，逐层
   inv(QᵀQ + α·I) 取平均后 eigh，V = Rᵀ @ 特征向量。V 为 in_dim 方阵、
   同类别各层共享。
B. sparse_indices —— 稀疏 S 的 COO 位置：in_dim² 网格**排除对角**，
   nnz = max(1, int(in_dim²·sparse_rate))（sparse_rate>0 时），
   torch.Generator(prng_key) 播种 randperm 采样（判题 CPU 确定性）。
C. 层内（float64）—— bi = solve(V, Wᵀ)ᵀ（非方阵回退 pinv），λ_L = bi 列范数、
   U = bi 列归一化（W = U·diag(λ)·Vᵀ 在分解域精确成立）；forward delta =
   (xV)(S+L)ᵀUᵀ − xWᵀ（官方 _sparse_activation_mm 实为 z @ Sᵀ！）；
   ΔW = U(S + diag(λ))Vᵀ − W（参数化的是适配后权重本身，delta 是与 base 之差）；
   init_weights=True 时 S values 全零 → ΔW 精确为 0。

关键恒等（锚点互证）：S=0 ⇒ ΔW=0；x @ ΔWᵀ == forward_delta。
"""
import torch


def shared_projection(matrices, regularization_alpha):
    """跨层联合分解出共享投影 V（float32，官方 dtype 路径）。"""
    stacked = torch.cat([m.detach().to(torch.float32).cpu() for m in matrices], dim=0)
    if stacked.shape[0] < stacked.shape[1]:
        _, _, vh = torch.linalg.svd(stacked, full_matrices=True)
        return vh.T
    q_matrix, r_matrix = torch.linalg.qr(stacked)
    q_slices = []
    start = 0
    for m in matrices:
        rows = m.shape[0]
        q_slices.append(q_matrix[start:start + rows, :])
        start += rows
    dim = r_matrix.shape[1]
    t_pi = torch.zeros((dim, dim), dtype=r_matrix.dtype)
    for q in q_slices:
        t_pi += torch.linalg.inv(q.T @ q + regularization_alpha * torch.eye(dim, dtype=r_matrix.dtype))
    t_pi /= len(q_slices)
    _, eigvec = torch.linalg.eigh(t_pi)
    return r_matrix.T @ eigvec


def sparse_indices(in_dim, sparse_rate, prng_key):
    """非对角 COO 位置：prng randperm 确定性采样。"""
    generator = torch.Generator(device="cpu").manual_seed(prng_key)
    rows, cols = torch.meshgrid(torch.arange(in_dim), torch.arange(in_dim), indexing="ij")
    mask_indices = torch.stack([rows.flatten(), cols.flatten()], dim=1)
    non_diag = mask_indices[mask_indices[:, 0] != mask_indices[:, 1]]
    nnz = min(int(in_dim * in_dim * sparse_rate), non_diag.shape[0])
    if sparse_rate > 0 and non_diag.shape[0] > 0:
        nnz = max(1, nnz)
    if nnz:
        perm = torch.randperm(non_diag.shape[0], generator=generator)[:nnz]
        return non_diag[perm].t().contiguous()
    return torch.empty(2, 0, dtype=torch.long)


def solve_u_lambda(w, v):
    """层内分解：bi=solve(V,Wᵀ)ᵀ（pinv 回退），λ=列范数，U=列归一化。"""
    try:
        bi = torch.linalg.solve(v, w.T).T
    except Exception:
        bi = w @ torch.linalg.pinv(v, rtol=1e-6).T
    lambda_l = torch.linalg.norm(bi, dim=0)
    u = torch.zeros_like(bi)
    nonzero = lambda_l > 0
    u[:, nonzero] = bi[:, nonzero] / lambda_l[nonzero]
    return u, lambda_l


def sparse_dense(values, indices, size):
    """COO → 稠密 S。"""
    return torch.sparse_coo_tensor(indices, values, size).coalesce().to_dense()


def forward_delta(x, w, v, s_values, s_indices, s_size, lambda_l, u):
    """adapter delta 输出：(xV)(S+L)ᵀUᵀ − xWᵀ。"""
    z = x @ v
    s = sparse_dense(s_values, s_indices, s_size)
    z_s = z @ s.T  # 官方 sparse.mm(S, z.t()).t()
    z_l = z * lambda_l
    out_add = (z_s + z_l) @ u.T
    return out_add - x @ w.T


def delta_weight(w, u, v, s_values, s_indices, s_size, lambda_l):
    """ΔW = U(S + diag(λ))Vᵀ − W（S=0 时精确为 0）。"""
    s = sparse_dense(s_values, s_indices, s_size)
    l_mat = torch.diag_embed(lambda_l)
    return u @ (s + l_mat) @ v.T - w


def solve(mode=None, **kw):
    if mode == "shared_projection":
        return shared_projection(kw["matrices"], kw["regularization_alpha"])
    if mode == "sparse_indices":
        return sparse_indices(kw["in_dim"], kw["sparse_rate"], kw["prng_key"])
    if mode == "solve_u_lambda":
        u, lam = solve_u_lambda(kw["w"], kw["v"])
        return u, lam
    if mode == "sparse_dense":
        return sparse_dense(kw["values"], kw["indices"], kw["size"])
    if mode == "forward_delta":
        return forward_delta(kw["x"], kw["w"], kw["v"], kw["s_values"],
                             kw["s_indices"], kw["s_size"], kw["lambda_l"], kw["u"])
    if mode == "delta_weight":
        return delta_weight(kw["w"], kw["u"], kw["v"], kw["s_values"],
                            kw["s_indices"], kw["s_size"], kw["lambda_l"])
    raise ValueError(f"unknown mode: {mode}")
