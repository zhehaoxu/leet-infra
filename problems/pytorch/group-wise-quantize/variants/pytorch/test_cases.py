import torch


def _ref(W, gs, bits):
    """向量化独立重算：用 index_reduce_ 按组求 absmax（与解法的逐组切片是不同路径）。"""
    qmax = 2 ** (bits - 1) - 1
    n = W.shape[-1]
    flat = W.reshape(-1, n)
    rows = flat.shape[0]
    ng = (n + gs - 1) // gs
    gid = torch.arange(n) // gs
    amax = torch.zeros(rows, ng, dtype=W.dtype)
    amax.index_reduce_(1, gid, flat.abs(), "amax", include_self=True)
    scale_map = amax / qmax
    scale = scale_map[:, gid]                       # [rows, n] 逐元素 scale
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(flat / safe), -qmax, qmax)
    return (q * scale).reshape(W.shape)


def _assert(solve, W, gs, bits):
    out = solve(W, gs, bits)
    exp = _ref(W, gs, bits)
    assert out.shape == W.shape and out.dtype == W.dtype
    assert torch.allclose(out, exp, atol=1e-5), f"\n{out}\n期望\n{exp}"


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example_int8(solve):
    W = torch.tensor([[1.0, -2.0, 0.5, 0.25, 4.0, -1.0]])
    out = solve(W, 4, 8)
    expected = torch.tensor([[1.00787, -2.0, 0.50394, 0.25197, 4.0, -1.00787]])
    assert torch.allclose(out, expected, atol=1e-4), out.tolist()


def check_example_int4(solve):
    W = torch.tensor([[1.0, 3.0, -7.0, 0.5, 2.0, 2.0, 2.0, 2.0]])
    out = solve(W, 4, 4)
    expected = torch.tensor([[1.0, 3.0, -7.0, 0.0, 2.0, 2.0, 2.0, 2.0]])
    assert torch.allclose(out, expected, atol=1e-6), out.tolist()


def check_remainder_group(solve):
    # n=7 gs=4 -> 组0[0:4] 组1[4:7] 短尾，独立估 scale
    torch.manual_seed(0)
    W = torch.randn(3, 7) * 5
    _assert(solve, W, 4, 8)


def check_group_size_gt_last_dim(solve):
    torch.manual_seed(1)
    W = torch.randn(2, 3)
    _assert(solve, W, 10, 8)  # 整行一组


def check_zero_group(solve):
    W = torch.tensor([[0.0, 0.0, 3.0, -3.0]])
    out = solve(W, 2, 8)
    assert torch.isfinite(out).all() and out[0, 0].item() == 0.0 and out[0, 1].item() == 0.0


def check_random_varied_bits(solve):
    torch.manual_seed(2)
    W = torch.randn(4, 12) * 3
    for gs in (1, 3, 4, 5, 12):
        for bits in (4, 8):
            _assert(solve, W, gs, bits)


def check_3d(solve):
    torch.manual_seed(3)
    W = torch.randn(2, 3, 8)
    _assert(solve, W, 4, 8)


def check_error_within_half_step(solve):
    torch.manual_seed(4)
    W = torch.randn(1, 6) * 4
    gs = 3
    qmax = 127
    out = solve(W, gs, 8)
    for s in range(0, 6, gs):
        g = W[0, s:s + gs]
        step = g.abs().max() / qmax
        assert ((out[0, s:s + gs] - g).abs() <= step / 2 + 1e-4).all()


def check_does_not_mutate_input(solve):
    W = torch.randn(2, 5)
    snap = W.clone()
    solve(W, 3, 8)
    assert torch.equal(W, snap)


check_example_int8.example = True
check_example_int4.example = True
