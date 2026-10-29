import math

import torch


def _rms(x, eps):
    return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _tensor(name, value, ndim):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dim() != ndim:
        raise ValueError(f"{name} 必须是 {ndim} 维张量")
    return value


def solve(hidden, embeddings, W, eps=1e-6):
    """MTP 的多深度预测：逐层把「上一层隐状态 + 下一个 token 的输入嵌入」融合再投影。

    hidden:(B, S, d)     主干最后一层的隐状态 h^0
    embeddings:(B, L, d) 输入嵌入序列，L 必须至少 S + D（要能看到向前 D 个 token）
    W:(D, 2d, d)         第 k 层（k 从 1 计）的融合投影，作用在 [rms(h^{k-1}); rms(Emb_{:,k:k+S})]
    eps: RMSNorm 的数值稳定项

    递推：
      h^0 = hidden
      h^k = W_k · [ RMS(h^{k-1}) ; RMS(embeddings[:, k : k+S]) ]     k = 1..D

    RMS 沿最后一维（特征维）计算。返回长度为 D 的 tuple，元素为 float32 的 (B, S, d)，
    依次是 h^1 .. h^D（不包含输入的 h^0）。
    """
    hidden = _tensor("hidden", hidden, 3)
    embeddings = _tensor("embeddings", embeddings, 3)
    W = _tensor("W", W, 3)

    batch, seq, dim = hidden.shape
    if embeddings.shape[0] != batch or embeddings.shape[2] != dim:
        raise ValueError("embeddings 的 batch 与特征维必须与 hidden 一致")

    depth, in_dim, out_dim = W.shape
    if depth < 1:
        raise ValueError("深度 D 必须至少为 1")
    if out_dim != dim or in_dim != 2 * dim:
        raise ValueError("W 必须是 (D, 2d, d)")
    if embeddings.shape[1] < seq + depth:
        raise ValueError("embeddings 的长度必须至少为 S + D")

    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise ValueError("eps 必须是实数")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps 必须是有限正实数")

    h = hidden.to(torch.float64)
    emb = embeddings.to(torch.float64)

    states = []
    for k in range(depth):
        # 第 k+1 层读的是 embeddings[:, k+1 : k+1+S]
        fused = torch.cat([_rms(h, eps), _rms(emb[:, k + 1 : k + 1 + seq], eps)], dim=-1)
        h = torch.einsum("bsf,fe->bse", fused, W[k].to(torch.float64))
        states.append(h.to(torch.float32))

    return tuple(states)
