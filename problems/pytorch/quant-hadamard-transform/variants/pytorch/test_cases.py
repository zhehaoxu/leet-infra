import torch


def _hadamard_matrix(k):
    if k == 1:
        return torch.ones(1, 1, dtype=torch.float32)
    H = _hadamard_matrix(k // 2)
    top = torch.cat([H, H], dim=1)
    bottom = torch.cat([H, -H], dim=1)
    return torch.cat([top, bottom], dim=0)


def _decompose(n):
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


def _oracle(n, x, seed=0):
    blocks = _decompose(n)
    H_blocks = [_hadamard_matrix(k) / (k ** 0.5) for k in blocks]
    H = torch.block_diag(*H_blocks)
    signs = _random_signs(n, seed, x.device)
    if x.dim() == 1:
        y = H @ (signs * x)
    else:
        y = (H @ (signs * x).T).T
    return H, y


def check_example(solve):
    n = 4
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    H, y = solve(n, x, seed=7)
    exp_H, exp_y = _oracle(n, x, seed=7)
    assert torch.allclose(H, exp_H, atol=1e-5, rtol=1e-5), "H 与重算不符"
    assert torch.allclose(y, exp_y, atol=1e-5, rtol=1e-5), f"y got {y.tolist()}"


check_example.example = True


def check_orthogonality_power_of_two(solve):
    """2 的幂时 H.T @ H 应为单位阵。"""
    for n in [2, 4, 8, 16]:
        H, _ = solve(n, torch.ones(n), seed=0)
        I = H.T @ H
        assert torch.allclose(I, torch.eye(n), atol=1e-5, rtol=1e-5), f"n={n} 不正交"


def check_block_diagonal_non_power_of_two(solve):
    """非 2 幂时 H 应为 block-diagonal，且子块分别正交。"""
    n = 6
    H, _ = solve(n, torch.ones(n), seed=0)
    blocks = _decompose(n)
    start = 0
    for k in blocks:
        block = H[start:start + k, start:start + k]
        assert torch.allclose(block.T @ block, torch.eye(k), atol=1e-5, rtol=1e-5)
        # 非对角块应为 0
        if start > 0:
            assert H[start:start + k, :start].abs().max() < 1e-5
        if start + k < n:
            assert H[start:start + k, start + k:].abs().max() < 1e-5
        start += k


def check_random_sign_changes(solve):
    """不同 seed 产生不同符号，从而得到不同 y。"""
    n = 8
    x = torch.randn(n)
    _, y1 = solve(n, x, seed=1)
    _, y2 = solve(n, x, seed=2)
    assert not torch.allclose(y1, y2, atol=1e-4, rtol=1e-4), "不同 seed 应产生不同 y"


def check_batch_input(solve):
    """支持 (b, n) 批量输入。"""
    n = 4
    x = torch.randn(3, n)
    H, y = solve(n, x, seed=11)
    _, exp_y = _oracle(n, x, seed=11)
    assert y.shape == x.shape
    assert torch.allclose(y, exp_y, atol=1e-5, rtol=1e-5)


def check_entries_pm_one_over_sqrt(solve):
    """H 的每个非零元绝对值应等于 1/sqrt(块大小)。"""
    n = 10
    H, _ = solve(n, torch.ones(n), seed=3)
    blocks = _decompose(n)
    start = 0
    for k in blocks:
        block = H[start:start + k, start:start + k]
        assert torch.allclose(block.abs(), torch.ones_like(block) / (k ** 0.5), atol=1e-6, rtol=1e-6)
        start += k
