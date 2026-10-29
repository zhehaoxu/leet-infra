import torch


def solve(packed, scales, zeros, group_size, N):
    """per-group 非对称 uint4 解包：低 4 位偶数索引，高 4 位奇数索引。"""
    M = packed.shape[0]
    low = (packed & 0xF).to(torch.int64)
    high = ((packed >> 4) & 0xF).to(torch.int64)
    interleaved = torch.stack([low, high], dim=2).view(M, -1)
    q = interleaved[:, :N]
    group_idx = torch.arange(N, device=packed.device) // group_size
    s = scales[:, group_idx]
    z = zeros[:, group_idx]
    return s.to(torch.float32) * (q.to(torch.float32) - z.to(torch.float32))
