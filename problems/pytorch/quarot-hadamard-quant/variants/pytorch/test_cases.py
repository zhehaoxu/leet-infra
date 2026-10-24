import math

import torch


def _oracle(W, bits):
    """独立纯 Python：构造归一化 Hadamard，W·H → per-tensor 对称 RTN → ·H 还原。"""
    O, C = W.shape
    M = [[1.0]]
    n = 1
    while n < C:
        k = len(M)
        new = [[0.0] * (2 * k) for _ in range(2 * k)]
        for i in range(k):
            for j in range(k):
                new[i][j] = M[i][j]
                new[i][j + k] = M[i][j]
                new[i + k][j] = M[i][j]
                new[i + k][j + k] = -M[i][j]
        M = new
        n *= 2
    norm = math.sqrt(C)
    H = [[M[i][j] / norm for j in range(C)] for i in range(C)]
    Wl = W.tolist()
    Wr = [[sum(Wl[o][k] * H[i][k] for k in range(C)) for i in range(C)] for o in range(O)]
    qmax = 2 ** (bits - 1) - 1
    m = max(abs(v) for row in Wr for v in row)
    scale = m if m > 0 else 1.0
    Wrq = [[max(-qmax - 1, min(qmax, round(v / scale * qmax))) / qmax * scale for v in row] for row in Wr]
    out = [[sum(Wrq[o][k] * H[i][k] for k in range(C)) for i in range(C)] for o in range(O)]
    return out


def check_example(solve):
    W = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]])
    out = solve(W, 8)
    exp = torch.tensor([[1.0039, 1.9882, 3.0118, 3.9961],
                        [0.4921, -1.0039, 1.9882, 0.0197]])
    assert torch.allclose(out, exp, atol=1e-3), out


def check_matches_oracle(solve):
    torch.manual_seed(11)
    for _ in range(30):
        O = int(torch.randint(1, 6, (1,)))
        C = 2 ** int(torch.randint(0, 4, (1,)))   # C ∈ {1,2,4,8}
        W = torch.randn(O, C) * 2
        bits = int(torch.randint(2, 9, (1,)))
        out = solve(W, bits)
        exp = _oracle(W, bits)
        assert torch.allclose(out, torch.tensor(exp), atol=1e-4), (out, exp)


def check_outlier_flattened(solve):
    # 单个离群列被 Hadamard 摊平：2-bit 反旋转后最大误差仍很小
    W = torch.tensor([[100.0, 1.0, 1.0, 1.0]])
    out = solve(W, 2)
    assert out.shape == W.shape
    assert float((W - out).abs().max()) < 10.0, (W, out)


def check_high_bits_near_identity(solve):
    W = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]])
    out = solve(W, 16)   # 高位宽旋转→量化→反旋转近似恒等
    assert torch.allclose(out, W, atol=1e-2), out


check_example.example = True
