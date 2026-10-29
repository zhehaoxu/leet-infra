import math

import torch
import torch.nn.functional as F


def _scalar(name, value, low=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数")
    if low is not None and value < low:
        raise ValueError(f"{name} 不能小于 {low}")
    return value


def _kernel_size(sigma):
    """上游的奇数化公式：ceil(6σ) 取奇数后加一。"""
    raw = math.ceil(6.0 * sigma)
    return raw + 1 - raw % 2


def solve(query, blur_sigma, blur_threshold_inf=9999.0):
    """SEG 的查询平滑：方阵高斯模糊，或退化为整块均值。

    query 形如 (B, S, D) 的 query 投影（第 0 维 batch，第 1 维 token，第 2 维通道）。
    按方形图像假设，取前 n*n 个 token（n = int(sqrt(S))，向下取整）重排成 (B, D, n, n)
    的二维特征图，逐通道做平滑后写回原位；剩下的 S - n*n 个 token 原样保留。

    blur_sigma > blur_threshold_inf（严格大于）走「无限模糊」分支：
      k0 = ceil(6σ) + 1 - ceil(6σ) % 2                （保证奇数）
      k  = min(k0, n - (n % 2 - 1))                   （n 偶时上限 n+1，n 奇时上限 n）
      h  = (k - 1) / 2
      x  = linspace(-h, h, k)
      p  = exp(-0.5 * (x / blur_sigma)^2)，归一化成 1-D 高斯核
      kernel2d = outer(p, p)，按通道广播成 depthwise 卷积核 (D, 1, k, k)
      特征图四周各 pad k//2（reflect 模式）后做分组卷积
    否则走均值分支：每个 (b, d) 通道的特征图被其 n*n 个元素的均值整体替换。

    返回 float32 张量，形状与 query 相同；输入不被就地修改。
    """
    if not isinstance(query, torch.Tensor):
        raise ValueError("query 必须是 torch.Tensor")
    if query.dim() != 3:
        raise ValueError("query 必须是三维张量 (B, S, D)")
    sigma = _scalar("blur_sigma", blur_sigma, 0.0)
    threshold = _scalar("blur_threshold_inf", blur_threshold_inf, 0.0)

    batch, seq_len, dim = query.shape
    grid = int(math.sqrt(seq_len))
    num_square = grid * grid
    if num_square < 1:
        raise ValueError("query 的 token 数至少为 1")

    values = query.to(torch.float64)
    patch = values[:, :num_square, :].permute(0, 2, 1).reshape(batch, dim, grid, grid).clone()

    if sigma > threshold:
        size = min(_kernel_size(sigma), grid - (grid % 2 - 1))
        half = (size - 1) / 2.0
        axis = torch.linspace(-half, half, size, dtype=torch.float64)
        pdf = torch.exp(-0.5 * (axis / sigma) ** 2)
        kernel1d = pdf / pdf.sum()
        kernel2d = torch.outer(kernel1d, kernel1d).expand(dim, 1, size, size)
        pad = size // 2
        padded = F.pad(patch, [pad, pad, pad, pad], mode="reflect")
        patch = F.conv2d(padded, kernel2d, groups=dim)
    else:
        patch = patch.mean(dim=(-2, -1), keepdim=True).expand_as(patch).clone()

    smoothed = patch.reshape(batch, dim, num_square).permute(0, 2, 1)
    out = values.clone()
    out[:, :num_square, :] = smoothed
    return out.to(torch.float32)
