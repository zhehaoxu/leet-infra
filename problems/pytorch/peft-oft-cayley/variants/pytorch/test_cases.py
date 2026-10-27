import torch

BLOCKS = torch.tensor([[0.1, 0.2, -0.1],
                       [0.05, -0.15, 0.2]])            # (2,3) -> h=3, n_blocks=2
W0 = torch.tensor([[0.5, -0.2, 0.1, 0.3],
                   [0.2, 0.4, -0.1, 0.0],
                   [-0.3, 0.1, 0.5, 0.2],
                   [0.1, 0.3, 0.0, -0.2],
                   [0.4, -0.1, 0.2, 0.1],
                   [-0.2, 0.0, 0.3, 0.5]])             # (6,4)

EXP_R = torch.tensor([
    [0.90566039, 0.22641510, 0.35849056, 0.0, 0.0, 0.0],
    [-0.15094340, 0.96226412, -0.22641510, 0.0, 0.0, 0.0],
    [-0.39622641, 0.15094340, 0.90566039, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.95305163, 0.15023474, -0.26291081],
    [0.0, 0.0, 0.0, -0.03755868, 0.92018777, 0.38967136],
    [0.0, 0.0, 0.0, 0.30046949, -0.36150235, 0.88262910]])

EXP_W = torch.tensor([
    [0.39056602, -0.05471698, 0.24716981, 0.34339625],
    [0.18490566, 0.39245284, -0.22452831, -0.09056604],
    [-0.43962264, 0.23018868, 0.39811319, 0.06226415],
    [0.20798123, 0.27089202, -0.04882630, -0.30704227],
    [0.28638497, -0.10328639, 0.30093896, 0.29436621],
    [-0.29107982, 0.12629108, 0.19248827, 0.34507042]])


def _oracle(blocks, w0):
    """独立重算：解线性方程组 (I-Q)R = I+Q 而非矩阵求逆（结构异于参考解）。"""
    import math
    n_elem = int(blocks.shape[1])
    h = int(round((1.0 + math.sqrt(1.0 + 8.0 * n_elem)) / 2.0))
    tri = torch.triu_indices(h, h, 1)
    eye = torch.eye(h, dtype=torch.float64)
    nb = int(blocks.shape[0])
    r_big = torch.zeros(nb * h, nb * h, dtype=torch.float64)
    for b in range(nb):
        q = torch.zeros(h, h, dtype=torch.float64)
        q[tri[0], tri[1]] = blocks[b].to(torch.float64)
        q = q - q.T
        r_big[b * h:(b + 1) * h, b * h:(b + 1) * h] = torch.linalg.solve(eye - q, eye + q)
    return (r_big.to(torch.float32),
            (r_big @ w0.to(torch.float64)).to(torch.float32))


def check_example(solve):
    r, w = solve(BLOCKS, W0)
    assert torch.allclose(r, EXP_R, atol=1e-5, rtol=1e-5), f"R got {r}"
    assert torch.allclose(w, EXP_W, atol=1e-5, rtol=1e-5), f"W' got {w}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(17)
    blocks = torch.randn(3, 6)          # h=4, n_blocks=3
    w0 = torch.randn(12, 5)
    r, w = solve(blocks, w0)
    r_o, w_o = _oracle(blocks, w0)
    assert torch.allclose(r, r_o, atol=1e-5, rtol=1e-5), "R 与独立重算不符"
    assert torch.allclose(w, w_o, atol=1e-5, rtol=1e-5), "W' 与独立重算不符"


def check_zero_skew_is_identity(solve):
    r, w = solve(torch.zeros(2, 3), W0)
    assert torch.allclose(r, torch.eye(6), atol=1e-6, rtol=1e-6), "零参数 R 应为恒等"
    assert torch.allclose(w, W0, atol=1e-6, rtol=1e-6), "零参数不应扰动基座权重"


def check_orthogonality_frobenius(solve):
    r, w = solve(BLOCKS, W0)
    eye = torch.eye(6)
    assert torch.allclose(r @ r.T, eye, atol=1e-5, rtol=1e-5), "R@R.T 应为 I"
    assert torch.allclose(r.T @ r, eye, atol=1e-5, rtol=1e-5), "R.T@R 应为 I"
    assert abs(w.norm() - W0.norm()) < 1e-5, "正交旋转保持 Frobenius 范数"


def check_block_diagonal_structure(solve):
    r, _ = solve(BLOCKS, W0)
    off = r.clone()
    for b in range(2):
        off[b * 3:(b + 1) * 3, b * 3:(b + 1) * 3] = 0.0
    assert torch.count_nonzero(off) == 0, "非对角块必须全零"
