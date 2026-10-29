import torch


def _oracle(weight, mask, Hinv):
    w = [row[:] for row in weight.tolist()]
    m = mask.tolist()
    h = Hinv.tolist()
    r, c = len(w), len(w[0])
    for j in range(c):
        cur = [w[i][j] for i in range(r)]
        q = [cur[i] * m[i][j] for i in range(r)]
        e = [(cur[i] - q[i]) / h[j][j] for i in range(r)]
        for i in range(r):
            w[i][j] = q[i]
        for k in range(j + 1, c):
            for i in range(r):
                w[i][k] -= e[i] * h[j][k]
    return torch.tensor(w, dtype=torch.float32)


def _spd_inv(c, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(c, c, generator=g)
    hess = a @ a.t() + 0.7 * torch.eye(c)
    return torch.linalg.inv(hess)


W = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
M = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
H = torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])


def check_example(solve):
    out = solve(W, M, H)
    exp = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 6.0]])
    assert torch.allclose(out, exp, atol=1e-6), f"got {out.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(51)
    for r, c in [(3, 4), (5, 6), (2, 3)]:
        w = torch.randn(r, c)
        m = (torch.rand(r, c) > 0.4).to(torch.float32)
        hinv = _spd_inv(c, 51 + c)
        out = solve(w, m, hinv)
        exp = _oracle(w, m, hinv)
        assert torch.allclose(out, exp, atol=1e-5, rtol=1e-5), f"{r}x{c}"


def check_mask_all_ones_unchanged(solve):
    torch.manual_seed(52)
    w = torch.randn(4, 5)
    hinv = _spd_inv(5, 52)
    out = solve(w, torch.ones(4, 5), hinv)
    assert torch.allclose(out, w, atol=1e-6), "全保留时误差为 0，结果应不变"


def check_mask_all_zeros_zeroes(solve):
    torch.manual_seed(53)
    w = torch.randn(3, 4)
    hinv = _spd_inv(4, 53)
    out = solve(w, torch.zeros(3, 4), hinv)
    assert torch.all(out == 0), out.tolist()


def check_identity_hessian_is_plain_mask(solve):
    torch.manual_seed(54)
    w = torch.randn(4, 4)
    m = (torch.rand(4, 4) > 0.5).to(torch.float32)
    out = solve(w, m, torch.eye(4))
    assert torch.allclose(out, w * m, atol=1e-6), "Hinv=I 时无列间耦合，应恰好等于逐元素掩码"


def check_pruned_positions_stay_zero(solve):
    torch.manual_seed(55)
    w = torch.randn(6, 7)
    m = (torch.rand(6, 7) > 0.3).to(torch.float32)
    hinv = _spd_inv(7, 55)
    out = solve(w, m, hinv)
    assert torch.all(out[m == 0] == 0), "被剪位置必须为 0"


def check_error_absorbed(solve):
    torch.manual_seed(56)
    c, n = 6, 5
    X = torch.randn(c, n)
    hess = X @ X.t() + 0.5 * torch.eye(c)
    hinv = torch.linalg.inv(hess)
    w = torch.randn(5, c)
    m = (torch.rand(c, 5).t() > 0.5).to(torch.float32)
    out = solve(w, m, hinv)
    err_out = float((out @ X - w @ X).norm())
    err_naive = float(((w * m) @ X - w @ X).norm())
    assert err_out < err_naive, (err_out, err_naive)


def check_output_dtype_shape(solve):
    w = torch.randn(2, 3)
    m = torch.ones(2, 3)
    out = solve(w, m, torch.eye(3))
    assert out.shape == (2, 3) and out.dtype == torch.float32, (out.shape, out.dtype)
