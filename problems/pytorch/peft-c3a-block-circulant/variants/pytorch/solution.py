"""参考答案：C3A 块循环卷积适配器。"""

import torch


def _float_tensor(name, value, ndim_min, ndim_max=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} 必须是 float32/float64 张量")
    if value.dim() < ndim_min:
        raise ValueError(f"{name} 维度不足")
    if ndim_max is not None and value.dim() > ndim_max:
        raise ValueError(f"{name} 维度过高")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} 含 NaN/Inf")
    return value


def solve(x, W0, kernel, b0=None):
    x = _float_tensor("x", x, 1)
    W0 = _float_tensor("W0", W0, 2, 2)
    kernel = _float_tensor("kernel", kernel, 3, 3)

    out_dim, in_dim = W0.shape
    if in_dim == 0 or out_dim == 0:
        raise ValueError("W0 不能有空维度")
    if x.shape[-1] != in_dim:
        raise ValueError("x 的最后一维必须等于 d_in")

    block = kernel.shape[-1]
    if block <= 0:
        raise ValueError("block size 必须为正")
    if in_dim % block != 0 or out_dim % block != 0:
        raise ValueError("d_in 与 d_out 都必须能被 block size 整除")
    if kernel.shape[0] != out_dim // block or kernel.shape[1] != in_dim // block:
        raise ValueError("kernel 形状必须是 (d_out//bs, d_in//bs, bs)")

    if b0 is not None:
        b0 = _float_tensor("b0", b0, 1, 1)
        if b0.numel() != out_dim:
            raise ValueError("b0 长度必须等于 d_out")

    x64 = x.to(torch.float64)
    weight = W0.to(torch.float64)
    k = kernel.to(torch.float64)

    offset = (torch.arange(block).unsqueeze(0) - torch.arange(block).unsqueeze(1)) % block
    blocks = k[:, :, offset]
    delta = blocks.permute(0, 2, 1, 3).reshape(out_dim, in_dim) / in_dim

    y = x64 @ weight.T
    if b0 is not None:
        y = y + b0.to(torch.float64)
    y = y + x64 @ delta.T
    return y.to(torch.float32), delta.to(torch.float32)
