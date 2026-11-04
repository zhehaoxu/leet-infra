"""D2-MoE 的 delta 解压缩账（Fisher 合并 + 白化 SVD 低秩 + 共享 V）。

回源: lliai/D2MoE model/merge_mixtral.py
（论文 D2-MoE: Delta Decompression for MoE-based LLMs Compression,
arXiv:2502.17298）。机制：把 MoE 各专家权重分解为「共享 base 权重 +
专家独有 delta」，delta 按低秩性质用白化 SVD 压缩，base 侧另做半动
态结构化剪枝（依赖动态 probe 输入，不在本题账内）。全程零训练。

Fisher 合并（merge_experts 的 fisher 分支，w1/w3/w2 各自独立）:
  coef_j = fisher_j * freq_j / Σ_l (fisher_l * freq_l)
  W_base = Σ_j coef_j * W_j
  delta_j = W_j - W_base        # fisher=0 的专家不贡献 base

delta 的参数预算秩（svd_delta）:
  k = int(r * c * ratio / (r + c))
  # (r+c)*k ≈ ratio*r*c —— ratio=1 时 k=rc/(r+c) 仍 < min(r,c)，
  # 该映射是「压缩后参数量占比」而非满秩重建

svdllm 白化 SVD + sqrt(Σ) 拆分:
  W_s = M @ scale;  SVD -> u_k, Σ_k, vt_k
  svd_u = u_k @ diag(sqrt(Σ_k))
  svd_v = diag(sqrt(Σ_k)) @ vt_k @ inv(scale)
  svd_u @ svd_v = (M @ scale)_k @ inv(scale)   # 白化域截断重建去白化

共享 V（delta_share_V）: 组内 delta 按行堆叠 (m·r, c) 一次 SVD，
V_shared = diag(sqrt(Σ)) @ vt_k，各专家只留行切片 U_j =
us[j·r:(j+1)·r, :k] @ diag(sqrt(Σ))——堆叠低秩重建的行切片
（U_j @ V_shared == Us_k diag(Σ_k) vt_k 的第 j 段）。

合并前向（SwiGLU）:
  gate_j = x @ (W1_base + U1_j V1).T
  up_j   = x @ (W3_base + U3_j V3).T
  h      = relu(gate_j) * up_j
  out_j  = h @ (W2_base + U2_j V2).T
路由权重、top-k 选择沿用原 MoE，在 adapter 外乘。

SVD/Cholesky 因子全部显式入参（判题不现跑分解）。
"""

import numpy as np


def fisher_merge(weights, fisher, freq):
    """Fisher 加权合并: coef = fisher*freq 归一化；base 与 deltas。

    weights: [(out, in)]；fisher/freq: 每专家标量。
    返回 (base, deltas, coef)。
    """
    coef = np.asarray(fisher) * np.asarray(freq)
    coef = coef / coef.sum()
    base = sum(coef[j] * np.asarray(weights[j]) for j in range(len(weights)))
    deltas = [np.asarray(W) - base for W in weights]
    return base, deltas, coef


def delta_rank(shape, ratio):
    """参数预算秩: k = int(r*c*ratio/(r+c))（官方 svd_delta）。"""
    r, c = shape
    return int(r * c * ratio / (r + c))


def ta_svd_split(u_k, sigma_k, vt_k, scale_inv):
    """svdllm 白化 SVD 的 sqrt(Σ) 拆分（官方 svd_delta）。

    svd_u = u_k @ diag(sqrt(Σ))；svd_v = diag(sqrt(Σ)) @ vt_k @ inv(scale)。
    乘积恢复白化域截断重建去白化：(M @ scale)_k @ inv(scale)。
    """
    sq = np.diag(np.sqrt(np.asarray(sigma_k)))
    svd_u = np.asarray(u_k) @ sq
    svd_v = sq @ np.asarray(vt_k) @ np.asarray(scale_inv)
    return svd_u, svd_v


def shared_v_split(us, sigma_s, vts_s, seg, num_basis):
    """共享 V 账（官方 delta_share_V）: 行堆叠 SVD 因子拆给各专家。

    us: (m*r, min) 左因子；vts_s: 右因子转置；seg: 每专家行数 r。
    返回 (V_shared (k, c), U_list [(r, k)])：
    V = diag(sqrt(Σ_k)) @ vts_k；U_j = us[j*r:(j+1)*r, :k] @ diag(sqrt(Σ_k))。
    """
    us = np.asarray(us)
    sq = np.diag(np.sqrt(np.asarray(sigma_s)[:num_basis]))
    V = sq @ np.asarray(vts_s)[:num_basis]
    m = us.shape[0] // seg
    U_list = [us[j * seg:(j + 1) * seg, :num_basis] @ sq for j in range(m)]
    return V, U_list


def forward_merged(x, w1_base, w3_base, w2_base, du1, dv1, du3, dv3, du2, dv2):
    """合并 SwiGLU 前向: base 路径 + delta 低秩路径相加（官方 shared_infer）。

    gate = x @ (W1b + du1@dv1).T；up = x @ (W3b + du3@dv3).T；
    h = relu(gate) * up；out = h @ (W2b + du2@dv2).T。
    """
    gate = x @ (np.asarray(w1_base) + du1 @ dv1).T
    up = x @ (np.asarray(w3_base) + du3 @ dv3).T
    h = np.maximum(gate, 0.0) * up
    return h @ (np.asarray(w2_base) + du2 @ dv2).T


def solve(mode=None, **kw):
    if mode == "fisher_merge":
        return fisher_merge(kw["weights"], kw["fisher"], kw["freq"])
    if mode == "delta_rank":
        return delta_rank(kw["shape"], kw["ratio"])
    if mode == "ta_svd_split":
        return ta_svd_split(kw["u_k"], kw["sigma_k"], kw["vt_k"],
                            kw["scale_inv"])
    if mode == "shared_v_split":
        return shared_v_split(kw["us"], kw["sigma_s"], kw["vts_s"],
                              kw["seg"], kw["num_basis"])
    if mode == "forward_merged":
        return forward_merged(kw["x"], kw["w1_base"], kw["w3_base"],
                              kw["w2_base"], kw["du1"], kw["dv1"],
                              kw["du3"], kw["dv3"], kw["du2"], kw["dv2"])
    raise ValueError("unknown mode: %s" % mode)
