import math

import torch


def _tensor(name, value, ndim):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dim() != ndim:
        raise ValueError(f"{name} 必须是 {ndim} 维张量")
    return value


def solve(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, scale):
    """MLA 解码时的「权重吸收」注意力。

    q_nope:(B, H, d_h)  每头的非 RoPE 查询分量
    q_pe:  (B, H, d_r)  每头已施加 RoPE 的查询分量
    c_kv:  (B, S, d_c)  潜在 KV 缓存
    k_pe:  (B, S, d_r)  所有头共享、已施加 RoPE 的键分量
    W_UK:  (H, d_h, d_c) 把潜在向量升维成 k_nope 的每头投影
    W_UV:  (H, d_h, d_c) 把潜在向量升维成 v 的每头投影
    scale: softmax 前的缩放因子

    吸收形式（全程在 d_c 维潜在空间里做注意力）：
      q_abs = q_nope^T W_UK        -> (B, H, d_c)
      s     = (q_abs · c_kv + q_pe · k_pe) * scale
      p     = softmax(s, dim=-1)
      z     = p · c_kv             -> (B, H, d_c)   加权求和后的潜在向量
      out   = W_UV z               -> (B, H, d_h)   每头一次升维

    返回 float32 张量，形状 (B, H, d_h)。
    """
    q_nope = _tensor("q_nope", q_nope, 3)
    q_pe = _tensor("q_pe", q_pe, 3)
    c_kv = _tensor("c_kv", c_kv, 3)
    k_pe = _tensor("k_pe", k_pe, 3)
    W_UK = _tensor("W_UK", W_UK, 3)
    W_UV = _tensor("W_UV", W_UV, 3)

    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("scale 必须是实数")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale 必须是有限正实数")

    batch, heads, head_dim = q_nope.shape
    seq, latent_dim = c_kv.shape[1], c_kv.shape[2]
    rope_dim = q_pe.shape[2]

    if q_pe.shape[:2] != (batch, heads):
        raise ValueError("q_pe 的 batch / 头数必须与 q_nope 一致")
    if k_pe.shape != (batch, seq, rope_dim):
        raise ValueError("k_pe 必须是 (B, S, d_r)，且与 c_kv 的 batch / 长度一致")
    if c_kv.shape[0] != batch:
        raise ValueError("c_kv 的 batch 必须与查询一致")
    if W_UK.shape != (heads, head_dim, latent_dim):
        raise ValueError("W_UK 必须是 (H, d_h, d_c)")
    if W_UV.shape != (heads, head_dim, latent_dim):
        raise ValueError("W_UV 必须是 (H, d_h, d_c)")

    qn = q_nope.to(torch.float64)
    qp = q_pe.to(torch.float64)
    latent = c_kv.to(torch.float64)
    kp = k_pe.to(torch.float64)
    uk = W_UK.to(torch.float64)
    uv = W_UV.to(torch.float64)

    # 把 W_UK 吸收进查询：(B,H,d_h) x (H,d_h,d_c) -> (B,H,d_c)
    q_abs = torch.einsum("bhk,hkc->bhc", qn, uk)

    # QK：潜在部分用吸收后的查询直接与 c_kv 内积，RoPE 部分单独算
    scores = torch.einsum("bhc,bsc->bhs", q_abs, latent)
    scores = scores + torch.einsum("bhr,bsr->bhs", qp, kp)
    probs = torch.softmax(scores * scale, dim=-1)

    # 先在潜在空间做加权求和，再一次性升维到 d_h（等价于先逐 token 升维再加权）
    pooled = torch.einsum("bhs,bsc->bhc", probs, latent)
    out = torch.einsum("bhc,hkc->bhk", pooled, uv)

    return out.to(torch.float32)
