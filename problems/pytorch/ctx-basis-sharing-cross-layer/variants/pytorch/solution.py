"""Basis Sharing 的跨层共享基低秩压缩账。

回源: TUDa-HWAI/Basis_Sharing group.py / calib.py
（论文 Basis Sharing: Cross-Layer Parameter Sharing for Large
Language Model Compression, arXiv:2410.03765, ICLR 2025）。
机制：不同层的同名投影权重不做各自独立的低秩分解，而是**拼成一张
宽矩阵后联合分解**——一组共享基向量（basis）被组内所有层复用，
每层只保留自己的窄系数矩阵，参数量从 m*out*in 降到 in*k + m*k*out。

官方压缩主干（Group._init_basis_coefficient，llama 路径权重先转置）:
  W_cat = [W_1.T | W_2.T | ... ]            # (in, m*out)
  S    = chol(H).T                          # H 为校准 Hessian（SPD）
  w    = S @ W_cat                          # SVD-LLM 白化预处理
  u, sigma, v = svd(w)                      # thin SVD
  basis = invs @ u[:, :k] @ diag(sigma[:k]) # (in, k) 共享基
  C_l   = v.T[:, seg_l][:k, :]              # (k, out_l) 每层系数
重建 W_hat_l = (basis @ C_l).T；前向 y = x @ basis @ C_l。

官方系数更新（update_model，basis 被再训练/替换后重解系数）:
  C_cat = (W_cat.T @ H @ B) @ (B.T @ H @ B)^-1   # 最小二乘闭式解
  按层列段切分。关键恒等: H = S.T @ S 时该目标等价
  ||S W_cat - U_k diag(sigma_k) C||_F^2，其最优解恰为初建 SVD
  系数——两阶段在 SVD basis 下数值恒等；basis 变化后 LS 重解。

校准数据复用映射（同输入投影共享 Hessian，官方 get_calib_data /
update_model 的 basis_name 分支）: q/v 投影用 k 投影的 H、gate 用
up 的 H、其余投影用自身。SVD/Cholesky 因子全部显式入参（不现跑）。

七入口复刻:
- scaled_concat / build_shared_basis / initial_coefficients /
  least_squares_update / reconstruct / forward_shared / calib_source
"""

import numpy as np


def scaled_concat(weights, s):
    """组内拼接 + 白化: W_cat = cat([W_l.T], -1)，w = S @ W_cat。

    weights: [(out_l, in)]；s: (in, in) 上三角 Cholesky 因子。
    返回 (w_cat, w)，均 (in, sum out_l)。
    """
    w_cat = np.concatenate([np.asarray(W).T for W in weights], axis=-1)
    return w_cat, np.asarray(s) @ w_cat


def build_shared_basis(u, sigma, invs, num_basis):
    """共享基: invs @ u[:, :k] @ diag(sigma[:k])，(in, k)。"""
    u = np.asarray(u)
    k = num_basis
    return np.asarray(invs) @ u[:, :k] @ np.diag(np.asarray(sigma)[:k])


def initial_coefficients(vt, out_dim, num_basis):
    """初建系数: v.T 按层列段切分，各取前 k 行。

    vt: v.T (in, m*out)（v 为 thin SVD 右因子）；层数 = vt.shape[1]//out_dim。
    返回 [(k, out_l)]。
    """
    vt = np.asarray(vt)
    m = vt.shape[1] // out_dim
    coefs = []
    for i in range(m):
        seg = vt[:, i * out_dim:(i + 1) * out_dim]
        coefs.append(seg[:num_basis, :])
    return coefs


def least_squares_update(w_cat, xtx, basis, out_dim):
    """系数闭式解（官方 update_model）: (W_cat.T @ H @ B)(B.T @ H @ B)^-1。

    w_cat: (in, m*out)；xtx: H (in, in)；basis: B (in, k)。
    返回按层列段切分的 [(k, out_l)]。
    """
    w_cat = np.asarray(w_cat)
    B = np.asarray(basis)
    inv = np.linalg.inv(B.T @ np.asarray(xtx) @ B)
    cat = (w_cat.T @ np.asarray(xtx) @ B @ inv).T      # (k, m*out)
    m = cat.shape[1] // out_dim
    return [cat[:, i * out_dim:(i + 1) * out_dim] for i in range(m)]


def reconstruct(basis, coefs):
    """每层重建权重: W_hat_l = (B @ C_l).T，[(out_l, in)]。"""
    B = np.asarray(basis)
    return [(B @ np.asarray(C)).T for C in coefs]


def forward_shared(x, basis, coefs):
    """共享前向: 每层 y_l = x @ B @ C_l（等价 x @ W_hat_l.T）。"""
    B = np.asarray(basis)
    return [np.asarray(x) @ B @ np.asarray(C) for C in coefs]


def calib_source(name):
    """同输入投影的校准复用映射（官方 get_calib_data / update_model）。

    q/v 用 k 的 H（同吃 hidden_states）、gate 用 up 的 H，其余用自身。
    """
    if name == "mlp.gate_proj":
        return "mlp.up_proj"
    if name in ("self_attn.q_proj", "self_attn.v_proj"):
        return "self_attn.k_proj"
    return name


def solve(mode=None, **kw):
    if mode == "scaled_concat":
        return scaled_concat(kw["weights"], kw["s"])
    if mode == "build_shared_basis":
        return build_shared_basis(kw["u"], kw["sigma"], kw["invs"],
                                  kw["num_basis"])
    if mode == "initial_coefficients":
        return initial_coefficients(kw["vt"], kw["out_dim"], kw["num_basis"])
    if mode == "least_squares_update":
        return least_squares_update(kw["w_cat"], kw["xtx"], kw["basis"],
                                    kw["out_dim"])
    if mode == "reconstruct":
        return reconstruct(kw["basis"], kw["coefs"])
    if mode == "forward_shared":
        return forward_shared(kw["x"], kw["basis"], kw["coefs"])
    if mode == "calib_source":
        return calib_source(kw["name"])
    raise ValueError("unknown mode: %s" % mode)
