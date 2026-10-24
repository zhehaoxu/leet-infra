import torch


def _inv(A):
    """纯 Python 高斯-约当求逆（独立于 torch.linalg.inv）。"""
    n = len(A)
    M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                if f != 0.0:
                    M[r] = [M[r][k] - f * M[c][k] for k in range(2 * n)]
    return [row[n:] for row in M]


def _oracle(W, H, bits, damp):
    """独立纯 Python：per-tensor 对称量化，逐列沿 H^{-1} 行向后补偿。"""
    O, C = W.shape
    qmax = 2 ** (bits - 1) - 1
    Wl = W.tolist()
    m = max(abs(v) for row in Wl for v in row)
    scale = m if m > 0 else 1.0
    A = [[float(H[i][j]) + (damp if i == j else 0.0) for j in range(C)] for i in range(C)]
    Hinv = _inv(A)
    Wc = [[float(Wl[o][j]) for j in range(C)] for o in range(O)]
    out = [[0.0] * C for _ in range(O)]
    for i in range(C):
        col = [Wc[o][i] for o in range(O)]
        di = Hinv[i][i]
        for o in range(O):
            q = max(-qmax - 1, min(qmax, round(col[o] / scale * qmax)))
            dq = q / qmax * scale
            out[o][i] = dq
            e = col[o] - dq
            for j in range(i + 1, C):
                Wc[o][j] -= (e / di) * Hinv[i][j]
    return out


def check_example(solve):
    W = torch.tensor([[1.0, 2.0, 0.5], [0.3, 1.0, 1.5]])
    H = torch.tensor([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])
    out = solve(W, H, 2, 0.1)
    exp = torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 2.0]])
    assert torch.allclose(out, exp, atol=1e-6), out


def check_matches_oracle(solve):
    torch.manual_seed(5)
    for _ in range(30):
        C = int(torch.randint(1, 6, (1,)))
        O = int(torch.randint(1, 5, (1,)))
        W = torch.randn(O, C)
        A = torch.randn(C, C)
        H = A @ A.t() + C * torch.eye(C)     # 对称正定
        bits = int(torch.randint(2, 7, (1,)))
        damp = float(0.05 + torch.rand(1) * 0.5)
        out = solve(W, H, bits, damp)
        exp = _oracle(W, H, bits, damp)
        assert torch.allclose(out, torch.tensor(exp), atol=1e-3), (out, exp)


def check_high_bits_near_identity(solve):
    W = torch.tensor([[1.0, 2.0, 0.5], [0.3, 1.0, 1.5]])
    H = torch.tensor([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])
    out = solve(W, H, 16, 0.1)   # 高位宽几乎无损
    assert torch.allclose(out, W, atol=1e-2), out


def check_output_on_grid(solve):
    # 补偿只改变"量化哪一列取值"，最终每列仍落在 per-tensor 合法格点上
    W = torch.tensor([[0.7, 1.3, -0.9, 0.2], [1.1, -0.4, 0.8, 1.5]])
    H = torch.tensor([[2.0, 0.5, 0.1, 0.0], [0.5, 1.8, 0.3, 0.1],
                      [0.1, 0.3, 1.6, 0.2], [0.0, 0.1, 0.2, 1.9]])
    out = solve(W, H, 3, 0.1)
    qmax = 2 ** 2 - 1
    scale = float(W.abs().max())
    lvl = {round(q / qmax * scale, 5) for q in range(-qmax - 1, qmax + 1)}
    vals = {round(float(v), 5) for v in out.flatten().tolist()}
    assert vals <= lvl, (vals, lvl)


check_example.example = True
