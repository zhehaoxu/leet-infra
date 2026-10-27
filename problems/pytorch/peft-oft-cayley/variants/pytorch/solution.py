import math

import torch


def _block_size(n_elem):
    """n_elem = h(h-1)/2 -> h。"""
    return int(round((1.0 + math.sqrt(1.0 + 8.0 * n_elem)) / 2.0))


def solve(blocks, W0):
    """OFT Cayley 参数化：每块 v -> 反对称 Q -> R_b=(I+Q)(I-Q)^{-1}，拼块对角 R，W'=R@W0。
    blocks:(n_blocks, h*(h-1)/2)（triu 行主序），W0:(n_blocks*h, d_in)。
    返回 (R, W')，R:(n_blocks*h, n_blocks*h)，W':(n_blocks*h, d_in)，均 float32。"""
    n_elem = int(blocks.shape[1])
    h = _block_size(n_elem)
    tri = torch.triu_indices(h, h, 1)
    eye = torch.eye(h, dtype=torch.float64)
    r_blocks = []
    for b in range(blocks.shape[0]):
        q = torch.zeros(h, h, dtype=torch.float64)
        q[tri[0], tri[1]] = blocks[b].to(torch.float64)
        q = q - q.T                              # 反对称
        r_blocks.append((eye + q) @ torch.linalg.inv(eye - q))   # Cayley
    n = h * len(r_blocks)
    r_big = torch.zeros(n, n, dtype=torch.float64)
    for b, rb in enumerate(r_blocks):
        r_big[b * h:(b + 1) * h, b * h:(b + 1) * h] = rb
    return (r_big.to(torch.float32), (r_big @ W0.to(torch.float64)).to(torch.float32))
