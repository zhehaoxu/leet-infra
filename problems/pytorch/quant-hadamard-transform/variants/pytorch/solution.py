import torch


def _hadamard_matrix(k):
    """递归构造 (k, k) Sylvester Hadamard 矩阵，k 为 2 的幂。"""
    if k == 1:
        return torch.ones(1, 1, dtype=torch.float32)
    H = _hadamard_matrix(k // 2)
    top = torch.cat([H, H], dim=1)
    bottom = torch.cat([H, -H], dim=1)
    return torch.cat([top, bottom], dim=0)


def _decompose(n):
    """把 n 贪心拆成从大到小的 2 的幂之和。"""
    blocks = []
    while n > 0:
        k = 1 << (n.bit_length() - 1)
        blocks.append(k)
        n -= k
    return blocks


def _random_signs(n, seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return torch.randint(0, 2, (n,), generator=g, device=device, dtype=torch.float32) * 2 - 1


def solve(n, x, seed=0):
    """构造归一化 Hadamard / block-diagonal Hadamard 并应用随机符号变换。"""
    if x.shape[-1] != n:
        raise ValueError("x 的最后一维必须等于 n")
    blocks = _decompose(n)
    H_blocks = [_hadamard_matrix(k) / (k ** 0.5) for k in blocks]
    H = torch.block_diag(*H_blocks)
    signs = _random_signs(n, seed, x.device)
    if x.dim() == 1:
        y = H @ (signs * x)
    else:
        y = (H @ (signs * x).T).T
    return H, y
