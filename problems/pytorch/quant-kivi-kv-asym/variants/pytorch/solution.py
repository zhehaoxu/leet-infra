import torch


def _asym(x, dim, qmax):
    mn = x.amin(dim=dim, keepdim=True)
    mx = x.amax(dim=dim, keepdim=True)
    span = mx - mn
    scale = torch.where(span > 0, span / qmax, torch.ones_like(span))
    q = ((x - mn) / scale).round().clamp(0, qmax)
    return q * scale + mn, scale.squeeze(dim), mn.squeeze(dim)


def solve(K, V, bits):
    """KIVI 非对称 KV 量化（float64 内部、float32 输出）。"""
    if not (2 <= int(bits) <= 8):
        raise ValueError("bits 必须在 [2, 8] 内")
    qmax = 2 ** int(bits) - 1
    k64, v64 = K.to(torch.float64), V.to(torch.float64)
    k_deq, k_scale, k_zero = _asym(k64, 0, qmax)   # 每个通道一套
    v_deq, v_scale, v_zero = _asym(v64, 1, qmax)   # 每个 token 一套
    return (k_deq.to(torch.float32), v_deq.to(torch.float32),
            k_scale.to(torch.float32), k_zero.to(torch.float32),
            v_scale.to(torch.float32), v_zero.to(torch.float32))
