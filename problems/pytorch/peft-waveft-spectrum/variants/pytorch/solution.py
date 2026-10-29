"""参考答案：WaveFT 稀疏频谱增量权重（direct 路径）。"""

import torch


def solve(x, W0, b0, spectrum, n_frequency, random_loc_seed, scaling):
    if W0.dim() != 2:
        raise ValueError("W0 必须是二维张量")
    out_dim, in_dim = W0.shape
    if x.dim() < 1 or x.shape[-1] != in_dim:
        raise ValueError("x 的最后一维必须等于 d_in")
    if b0 is not None and (b0.dim() != 1 or b0.numel() != out_dim):
        raise ValueError("b0 形状必须是 (d_out,)")
    if n_frequency <= 0:
        raise ValueError("n_frequency 必须是正整数")
    if n_frequency > out_dim * in_dim:
        raise ValueError("n_frequency 不能超过 d_out * d_in")
    if spectrum.dim() != 1 or spectrum.numel() != n_frequency:
        raise ValueError("spectrum 必须是一维且长度等于 n_frequency")
    scale = float(scaling)
    if not torch.isfinite(torch.tensor(scale)):
        raise ValueError("scaling 必须是有限值")

    generator = torch.Generator().manual_seed(int(random_loc_seed))
    flat = torch.randperm(out_dim * in_dim, generator=generator)[:n_frequency]

    delta = torch.zeros(out_dim * in_dim, dtype=torch.float64)
    delta[flat] = spectrum.to(torch.float64)
    delta = delta.reshape(out_dim, in_dim) * scale

    x64 = x.to(torch.float64)
    y = x64 @ W0.to(torch.float64).T
    if b0 is not None:
        y = y + b0.to(torch.float64)
    y = y + x64 @ delta.T
    return y.to(torch.float32), delta.to(torch.float32)
