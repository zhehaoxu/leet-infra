"""参考答案：FLUX.1 二维 RoPE —— 三轴频率表（float64 计算、float32 落表）+ 相邻对旋转。

口径对齐 diffusers 的 transformer_flux.FluxPosEmbed + embeddings.get_1d_rotary_pos_embed
（use_real=True、repeat_interleave_real=True）与 apply_rotary_emb 的相邻对（interleaved）旋转。
"""

import math
import numbers
import torch


def _to_int(value, *, allow_tensor=False):
    """把 value 当作 Python 整数返回；bool 与非法类型返回 None，交由调用方报错。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if allow_tensor and isinstance(value, torch.Tensor) and value.numel() == 1:
        if value.dtype == torch.bool:
            return None
        as_float = float(value.reshape(-1)[0].item())
        return int(as_float) if as_float.is_integer() else None
    return None


def _validate_axes_dim(axes_dim):
    if isinstance(axes_dim, (str, bytes)) or axes_dim is None:
        raise ValueError("axes_dim must be a length-3 sequence of positive even ints")
    try:
        values = list(axes_dim)
    except TypeError as exc:
        raise ValueError("axes_dim must be a length-3 sequence of positive even ints") from exc
    if len(values) != 3:
        raise ValueError("axes_dim must have exactly 3 axes")
    dims = []
    for value in values:
        dim = _to_int(value, allow_tensor=True)
        if dim is None or dim <= 0 or dim % 2 != 0:
            raise ValueError(f"every axes_dim entry must be a positive even int, got {value!r}")
        dims.append(dim)
    return dims


def _validate_theta(theta):
    if isinstance(theta, bool) or not isinstance(theta, numbers.Real):
        raise ValueError("theta must be a real number")
    theta = float(theta)
    if not math.isfinite(theta) or theta < 1.0:
        raise ValueError("theta must be finite and >= 1.0")
    return theta


def _validate_image_grid(image_grid):
    if isinstance(image_grid, (str, bytes)) or image_grid is None:
        raise ValueError("image_grid must be a length-2 sequence of ints >= 1")
    try:
        values = list(image_grid)
    except TypeError as exc:
        raise ValueError("image_grid must be a length-2 sequence of ints >= 1") from exc
    if len(values) != 2:
        raise ValueError("image_grid must have exactly 2 entries (latent_h, latent_w)")
    sizes = []
    for value in values:
        size = _to_int(value, allow_tensor=True)
        if size is None or size < 1:
            raise ValueError(f"image_grid entries must be ints >= 1, got {value!r}")
        sizes.append(size)
    return sizes[0], sizes[1]


def _validate_text_len(text_len):
    size = _to_int(text_len)
    if size is None or size < 0:
        raise ValueError(f"text_len must be a non-negative Python int, got {text_len!r}")
    return size


def _build_ids(text_len, height, width, device):
    """上游 _prepare_latent_image_ids 布局：轴 0 恒 0，轴 1 是行、轴 2 是列，行主序展平。"""
    image_ids = torch.zeros(height, width, 3, dtype=torch.float64, device=device)
    image_ids[..., 1] = image_ids[..., 1] + torch.arange(height, dtype=torch.float64, device=device)[:, None]
    image_ids[..., 2] = image_ids[..., 2] + torch.arange(width, dtype=torch.float64, device=device)[None, :]
    image_ids = image_ids.reshape(height * width, 3)
    text_ids = torch.zeros(text_len, 3, dtype=torch.float64, device=device)
    return torch.cat((text_ids, image_ids), dim=0)


def solve(query, image_grid, text_len, axes_dim=(16, 56, 56), theta=10000.0):
    dims = _validate_axes_dim(axes_dim)
    theta = _validate_theta(theta)
    height, width = _validate_image_grid(image_grid)
    text_len = _validate_text_len(text_len)
    if not isinstance(query, torch.Tensor) or query.ndim != 3:
        raise ValueError("query must be a 3-D tensor (B, S, head_dim)")
    head_dim = sum(dims)
    batch, seq_len, last_dim = query.shape
    if last_dim != head_dim:
        raise ValueError(f"query last dim {last_dim} != sum(axes_dim) {head_dim}")
    expected_len = text_len + height * width
    if seq_len != expected_len:
        raise ValueError(f"query sequence length {seq_len} != text_len + h*w = {expected_len}")

    device = query.device
    ids = _build_ids(text_len, height, width, device)          # (S, 3) float64
    pos = ids.float()                                          # 上游 pos = ids.float()

    cos_parts, sin_parts, angle_parts, inv_freq_parts = [], [], [], []
    for axis_index, axis_dim in enumerate(dims):
        # 指数分母是「该轴自己的 dim」，不是 head_dim；freqs 长度 = axis_dim // 2。
        inv_freq = theta ** (
            -torch.arange(0, axis_dim, 2, dtype=torch.float64, device=device) / axis_dim
        )
        angles = torch.outer(pos[:, axis_index], inv_freq)      # (S, axis_dim//2) float64
        # float64 里取 cos/sin，再 repeat_interleave 复制成 2 倍宽，最后 .float() 落表。
        cos_parts.append(angles.cos().repeat_interleave(2, dim=-1).float())
        sin_parts.append(angles.sin().repeat_interleave(2, dim=-1).float())
        angle_parts.append(angles)
        inv_freq_parts.append(inv_freq)

    freqs_cos = torch.cat(cos_parts, dim=-1)                    # (S, head_dim) float32
    freqs_sin = torch.cat(sin_parts, dim=-1)                    # (S, head_dim) float32
    angles = torch.cat(angle_parts, dim=-1)                     # (S, head_dim//2) float64
    inv_freqs = torch.cat(inv_freq_parts, dim=-1)               # (head_dim//2,) float64

    # 因 repeat_interleave 有 cos[2m] == cos[2m+1]，取每对的第一个元素即该对角度余弦。
    half = head_dim // 2
    cos_pair = freqs_cos.reshape(seq_len, half, 2)[..., 0]
    sin_pair = freqs_sin.reshape(seq_len, half, 2)[..., 0]
    even = query[..., 0::2]
    odd = query[..., 1::2]
    rotated = torch.stack(
        (even * cos_pair - odd * sin_pair, even * sin_pair + odd * cos_pair), dim=-1
    ).reshape(batch, seq_len, head_dim)
    return rotated.to(query.dtype), freqs_cos, freqs_sin, ids, angles, inv_freqs
