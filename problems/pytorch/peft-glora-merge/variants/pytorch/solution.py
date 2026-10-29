"""参考答案：GLoRA 权重合并（多适配器累加、偏置旁路）与精确逆解。"""

import torch


def _low_rank(adapter, prefix, out_dim, in_dim):
    """(d_out,r)@(r,d_in) -> 稠密 (d_out,d_in)。"""
    for suffix in ("d", "u"):
        key = f"{prefix}_{suffix}"
        if key not in adapter:
            raise ValueError(f"adapter 缺少字段 {key!r}")
        if adapter[key].dim() != 2:
            raise ValueError(f"{key} 必须是二维张量")
    d = adapter[f"{prefix}_d"].to(torch.float64)
    u = adapter[f"{prefix}_u"].to(torch.float64)
    if u.shape[0] != d.shape[1] or d.shape[1] < 1:
        raise ValueError(f"{prefix} 的低秩 r 不匹配或为 0")
    if d.shape[0] != out_dim or u.shape[1] != in_dim:
        raise ValueError(f"{prefix} 形状与 W0 不匹配")
    return d @ u


def _bias_vector(adapter, key, size):
    if key not in adapter:
        raise ValueError(f"带偏置的层必须提供字段 {key!r}")
    value = adapter[key]
    if value.dim() != 1 or value.numel() != size:
        raise ValueError(f"{key} 形状必须是 ({size},)")
    return value.to(torch.float64)


def solve(x, W0, b0, adapters):
    if W0.dim() != 2:
        raise ValueError("W0 必须是二维张量")
    out_dim, in_dim = W0.shape
    if x.dim() < 1 or x.shape[-1] != in_dim:
        raise ValueError("x 的最后一维必须等于 d_in")
    if not adapters:
        raise ValueError("adapters 不能为空")
    if b0 is not None and (b0.dim() != 1 or b0.numel() != out_dim):
        raise ValueError("b0 形状必须是 (d_out,)")

    w0 = W0.to(torch.float64)
    bias0 = b0.to(torch.float64) if b0 is not None else None
    merged_w = w0.clone()
    merged_b = bias0.clone() if bias0 is not None else None

    for adapter in adapters:
        a = _low_rank(adapter, "A", out_dim, in_dim)
        b = _low_rank(adapter, "B", out_dim, in_dim)
        merged_w = merged_w + w0 * a + b
        if merged_b is not None:
            c = _bias_vector(adapter, "C", in_dim)
            d = _bias_vector(adapter, "D", out_dim)
            e = _bias_vector(adapter, "E", out_dim)
            merged_b = merged_b + bias0 * d + e + (w0 @ c)

    y = x.to(torch.float64) @ merged_w.T
    if merged_b is not None:
        y = y + merged_b
    merged_b32 = merged_b.to(torch.float32) if merged_b is not None else None
    return y.to(torch.float32), merged_w.to(torch.float32), merged_b32
