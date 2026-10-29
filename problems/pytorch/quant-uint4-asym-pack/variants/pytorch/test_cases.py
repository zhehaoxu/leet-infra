import torch


def _pack(q):
    """把 (M, N) uint4 整数按低 4 位偶数、高 4 位奇数打包成 uint8。"""
    M, N = q.shape
    pad = (2 - N % 2) % 2
    q = torch.nn.functional.pad(q, (0, pad), value=0)
    even = q[:, 0::2] & 0xF
    odd = (q[:, 1::2] & 0xF) << 4
    return (even | odd).to(torch.uint8)


def _oracle(packed, scales, zeros, group_size, N):
    """逐元素解包 + 反量化。"""
    M = packed.shape[0]
    W = torch.zeros(M, N, dtype=torch.float64)
    for i in range(M):
        q = []
        for k in range(packed.shape[1]):
            b = int(packed[i, k])
            q.append(b & 0xF)
            q.append((b >> 4) & 0xF)
        q = q[:N]
        for j in range(N):
            g = j // group_size
            W[i, j] = float(scales[i, g]) * (q[j] - int(zeros[i, g]))
    return W.to(torch.float32)


def check_example(solve):
    packed = torch.tensor([[0x73, 0x1C]], dtype=torch.uint8)
    scales = torch.tensor([[2.0, 0.5]], dtype=torch.float32)
    zeros = torch.tensor([[1, 4]], dtype=torch.int64)
    W = solve(packed, scales, zeros, 2, 4)
    exp = torch.tensor([[4.0, 12.0, 4.0, -1.5]], dtype=torch.float32)
    assert torch.allclose(W, exp, atol=1e-5, rtol=1e-5), f"got {W.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(43)
    M, N, group_size = 3, 64, 32
    q = torch.randint(0, 16, (M, N), dtype=torch.int64)
    packed = _pack(q)
    scales = torch.rand(M, N // group_size, dtype=torch.float32) + 0.1
    zeros = torch.randint(0, 16, (M, N // group_size), dtype=torch.int64)
    W = solve(packed, scales, zeros, group_size, N)
    exp = _oracle(packed, scales, zeros, group_size, N)
    assert torch.allclose(W, exp, atol=1e-5, rtol=1e-5), "与逐元素重算不符"


def check_odd_columns(solve):
    """N 为奇数时忽略最后一个 uint8 的高 4 位。"""
    M, N, group_size = 1, 5, 5
    q = torch.tensor([[3, 7, 12, 1, 9]], dtype=torch.int64)
    packed = _pack(q)  # [0x73, 0x19] -> high nibble of last byte is padding
    scales = torch.tensor([[0.5]], dtype=torch.float32)
    zeros = torch.tensor([[2]], dtype=torch.int64)
    W = solve(packed, scales, zeros, group_size, N)
    exp = 0.5 * (q - 2).to(torch.float32)
    assert torch.allclose(W, exp, atol=1e-5, rtol=1e-5), f"got {W.tolist()}"


def check_multi_group(solve):
    """多 group、非平凡 scale/zero 仍能精确还原。"""
    torch.manual_seed(47)
    M, N, group_size = 2, 128, 16
    q = torch.randint(0, 16, (M, N), dtype=torch.int64)
    packed = _pack(q)
    scales = torch.arange(M * (N // group_size), dtype=torch.float32).reshape(M, N // group_size) * 0.1 + 0.1
    zeros = torch.randint(0, 16, (M, N // group_size), dtype=torch.int64)
    W = solve(packed, scales, zeros, group_size, N)
    exp = _oracle(packed, scales, zeros, group_size, N)
    assert torch.allclose(W, exp, atol=1e-5, rtol=1e-5)


def check_zero_scale(solve):
    """scale=0 时输出应为 0。"""
    packed = torch.full((1, 2), 0xAB, dtype=torch.uint8)
    scales = torch.zeros(1, 1, dtype=torch.float32)
    zeros = torch.tensor([[5]], dtype=torch.int64)
    W = solve(packed, scales, zeros, 4, 4)
    assert torch.equal(W, torch.zeros(1, 4, dtype=torch.float32))
