import math

import torch

_ROPE_TYPES = ("yarn", "linear", "dynamic_ntk")


def _int(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    if value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}")
    return value


def _num(value, name, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须有限")
    if minimum is not None:
        if strict and not value > minimum:
            raise ValueError(f"{name} 必须 > {minimum}")
        if not strict and value < minimum:
            raise ValueError(f"{name} 必须 >= {minimum}")
    return value


def _correction_dim(num_rotations, dim, base, max_position):
    return (dim * math.log(max_position / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(base))


def _mscale(scale, mscale=1.0):
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def solve(
    head_dim,
    base,
    factor,
    original_max_position_embeddings,
    position_ids,
    beta_fast=32.0,
    beta_slow=1.0,
    truncate=True,
    mscale=None,
    mscale_all_dim=None,
    attention_factor=None,
    rope_type="yarn",
    seq_len=None,
):
    head_dim = _int(head_dim, "head_dim", 4)
    if head_dim % 2:
        raise ValueError("head_dim 必须是偶数")
    base = _num(base, "base", 0.0, strict=True)
    if base <= 1.0:
        raise ValueError("base 必须 > 1")
    factor = _num(factor, "factor", 0.0, strict=True)
    orig = _int(original_max_position_embeddings, "original_max_position_embeddings", 2)
    beta_fast = _num(beta_fast, "beta_fast", 0.0, strict=True)
    beta_slow = _num(beta_slow, "beta_slow", 0.0, strict=True)
    if not beta_fast > beta_slow:
        raise ValueError("beta_fast 必须大于 beta_slow")
    if not isinstance(truncate, bool):
        raise ValueError("truncate 必须是 bool")
    if rope_type not in _ROPE_TYPES:
        raise ValueError(f"未知的 rope_type: {rope_type!r}")
    if not torch.is_tensor(position_ids):
        raise ValueError("position_ids 必须是张量")
    if position_ids.dim() != 2:
        raise ValueError("position_ids 必须是 (B, S) 二维张量")
    if position_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("position_ids 必须是整数张量")
    if position_ids.numel() == 0:
        raise ValueError("position_ids 不能为空")
    if int(position_ids.min()) < 0:
        raise ValueError("position_ids 不能为负")

    if rope_type != "yarn" and (
        attention_factor is not None or mscale is not None or mscale_all_dim is not None
    ):
        raise ValueError("attention_factor / mscale / mscale_all_dim 只在 yarn 下生效")

    dim = head_dim
    half = dim // 2
    idx = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    pos_freqs = base**idx
    inv_extrap = 1.0 / pos_freqs
    inv_interp = inv_extrap / factor

    if rope_type == "linear":
        inv_freq = inv_interp
        gamma = 1.0
    elif rope_type == "dynamic_ntk":
        length = orig if seq_len is None else max(_int(seq_len, "seq_len", 1), orig)
        scaled_base = base * ((factor * length / orig) - (factor - 1.0)) ** (dim / (dim - 2.0))
        inv_freq = 1.0 / (scaled_base**idx)
        gamma = 1.0
    else:
        if attention_factor is not None:
            gamma = _num(attention_factor, "attention_factor", 0.0, strict=True)
        elif mscale is not None and mscale_all_dim is not None:
            mscale = _num(mscale, "mscale")
            mscale_all_dim = _num(mscale_all_dim, "mscale_all_dim")
            gamma = _mscale(factor, mscale) / _mscale(factor, mscale_all_dim)
        else:
            gamma = _mscale(factor)

        low = _correction_dim(beta_fast, dim, base, orig)
        high = _correction_dim(beta_slow, dim, base, orig)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        low = max(low, 0.0)
        high = min(high, dim - 1.0)
        if low == high:
            high = high + 0.001
        ramp = ((torch.arange(half, dtype=torch.float64) - low) / (high - low)).clamp(0.0, 1.0)
        extrap_weight = 1.0 - ramp
        inv_freq = inv_interp * (1.0 - extrap_weight) + inv_extrap * extrap_weight

    freqs = position_ids.to(torch.float64).unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * gamma).to(torch.float32)
    sin = (emb.sin() * gamma).to(torch.float32)
    return inv_freq.to(torch.float32), float(gamma), cos, sin
