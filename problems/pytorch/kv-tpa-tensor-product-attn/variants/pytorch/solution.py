import math

import torch


def _as_float_tensor(name, value, dtype):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dim() != 4:
        raise ValueError(f"{name} 必须是四维张量")
    if value.dtype != dtype:
        raise ValueError("aq/ak/av/bq/bk/bv 必须同 dtype（float32/float64）")
    return value.to(torch.float64)


def solve(
    aq,
    ak,
    av,
    bq,
    bk,
    bv,
    is_causal=False,
    attn_mask=None,
    scale=None,
    scale_q=None,
    scale_k=None,
    scale_v=None,
):
    """TPA（Tensor Product Attention）的可掩码注意力：张量积因子收缩 + sigmoid 门。"""
    if not isinstance(is_causal, bool):
        raise ValueError("is_causal 必须是 bool")
    if is_causal and attn_mask is not None:
        raise ValueError("is_causal 与 attn_mask 不能同时给出")

    if not isinstance(aq, torch.Tensor):
        raise ValueError("aq 必须是 torch.Tensor")
    if aq.dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype 必须是 float32/float64")
    dtype = aq.dtype
    aq64 = _as_float_tensor("aq", aq, dtype)
    ak64 = _as_float_tensor("ak", ak, dtype)
    av64 = _as_float_tensor("av", av, dtype)
    bq64 = _as_float_tensor("bq", bq, dtype)
    bk64 = _as_float_tensor("bk", bk, dtype)
    bv64 = _as_float_tensor("bv", bv, dtype)

    batch, n, heads, rank = aq64.shape
    m, s = ak64.shape[1], ak64.shape[3]
    s_v = av64.shape[3]
    d = bq64.shape[3]
    e = bv64.shape[2]

    if min(batch, n, m, heads, rank, s, s_v, d, e) < 1:
        raise ValueError("所有维度必须 >= 1")
    if ak64.shape[0] != batch or av64.shape[0] != batch:
        raise ValueError("batch 不一致")
    if bq64.shape[0] != batch or bk64.shape[0] != batch or bv64.shape[0] != batch:
        raise ValueError("batch 不一致")
    if ak64.shape[1] != m or av64.shape[1] != m:
        raise ValueError("key 长度 M 不一致")
    if bk64.shape[1] != m or bv64.shape[1] != m:
        raise ValueError("key 长度 M 不一致")
    if ak64.shape[2] != heads or av64.shape[2] != heads:
        raise ValueError("head 数不一致")
    if bq64.shape[1] != n or bq64.shape[2] != rank:
        raise ValueError("bq 形状必须是 (B, N, R, D)")
    if bk64.shape[2] != d or bk64.shape[3] != s:
        raise ValueError("bk 形状必须是 (B, M, D, S)")
    if av64.shape[3] != s_v or bv64.shape[2] != e or bv64.shape[3] != s_v:
        raise ValueError("av/bv 的 S_V 或 E 不一致")

    def _scale(name, value, default):
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} 必须是实数")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} 必须有限")
        return float(value)

    scale = _scale("scale", scale, d ** -0.5)
    scale_q = _scale("scale_q", scale_q, 1.0 / rank)
    scale_k = _scale("scale_k", scale_k, 1.0 / s)
    scale_v = _scale("scale_v", scale_v, 1.0 / s_v)

    bias = torch.zeros((n, m), dtype=torch.float64)
    if is_causal:
        keep = torch.ones((n, m), dtype=torch.bool).tril(diagonal=0)
        bias.masked_fill_(~keep, float("-inf"))
    if attn_mask is not None:
        if not isinstance(attn_mask, torch.Tensor):
            raise ValueError("attn_mask 必须是 torch.Tensor")
        if tuple(attn_mask.shape) != (n, m):
            raise ValueError("attn_mask 形状必须是 (N, M)")
        if attn_mask.dtype == torch.bool:
            bias.masked_fill_(~attn_mask, float("-inf"))
        elif attn_mask.dtype == dtype:
            bias = bias + attn_mask.to(torch.float64)
        else:
            raise ValueError("attn_mask 必须是 bool 或与输入同 dtype")

    q = torch.einsum("bnhr,bnrd->bnhd", torch.sigmoid(aq64), bq64) * scale_q
    k = torch.einsum("bmhs,bmds->bmhd", torch.sigmoid(ak64), bk64) * scale_k
    v = torch.einsum("bmhs,bmes->bmhe", torch.sigmoid(av64), bv64) * scale_v

    scores = torch.einsum("bnhd,bmhd->bhnm", q, k) * scale + bias
    prob = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhnm,bmhe->bnhe", prob, v)

    return {"out": out.to(dtype), "prob": prob.to(dtype)}
