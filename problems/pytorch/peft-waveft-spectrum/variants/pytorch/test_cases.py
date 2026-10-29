import torch

W0 = torch.tensor([[0.5, -0.2, 0.1, 0.3],
                   [0.2, 0.4, -0.1, 0.0],
                   [-0.3, 0.1, 0.5, 0.2]])           # (3,4)
B0 = torch.tensor([0.1, -0.2, 0.05])                 # (3,)
X = torch.tensor([[1.0, 0.5, -0.3, 0.2],
                  [0.1, -0.4, 0.6, 0.3]])            # (2,4)
SPECTRUM = torch.tensor([0.02, -0.01, 0.03, 0.015])  # (4,)
N_FREQ, SEED, SCALING = 4, 777, 25.0


def _positions(n_frequency, random_loc_seed, out_dim, in_dim):
    generator = torch.Generator().manual_seed(random_loc_seed)
    flat = torch.randperm(out_dim * in_dim, generator=generator)[:n_frequency]
    return flat


def _oracle(x, w0, b0, spectrum, n_frequency, random_loc_seed, scaling):
    """独立重算：先取坐标，再用 (row,col) 逐点写入增量，前向走合并权重。"""
    out_dim, in_dim = w0.shape
    flat = _positions(n_frequency, random_loc_seed, out_dim, in_dim)
    delta = torch.zeros(out_dim, in_dim, dtype=torch.float64)
    for k in range(n_frequency):
        index = int(flat[k])
        delta[index // in_dim, index % in_dim] = spectrum[k].to(torch.float64)
    merged = w0.to(torch.float64) + delta * scaling
    y = x.to(torch.float64) @ merged.T
    if b0 is not None:
        y = y + b0.to(torch.float64)
    return y, delta * scaling


def check_example(solve):
    y, delta = solve(X, W0, B0, SPECTRUM, N_FREQ, SEED, SCALING)
    exp_delta = torch.tensor([[0.0, 0.0, 0.0, 0.75],
                              [0.0, 0.0, 0.0, -0.25],
                              [0.375, 0.0, 0.0, 0.5]])
    exp_y = torch.tensor([[0.68, 0.18, 0.165],
                          [0.605, -0.475, 0.5275]])
    assert delta.shape == (3, 4), f"delta 形状错: {tuple(delta.shape)}"
    assert torch.allclose(delta, exp_delta, atol=1e-6, rtol=1e-6), f"delta 错: {delta.tolist()}"
    assert torch.allclose(y, exp_y, atol=1e-5, rtol=1e-5), f"y 错: {y.tolist()}"


check_example.example = True


def check_index_selection(solve):
    # 坐标必须是 randperm(out*in)[:n] 的扁平下标，按 row=idx//d_in、col=idx%d_in 落位
    for n_frequency, seed, shape in ((4, 777, (3, 4)), (7, 13, (4, 5)), (1, 2024, (2, 6))):
        out_dim, in_dim = shape
        spectrum = torch.linspace(0.1, 0.1 * n_frequency, n_frequency)
        _, delta = solve(torch.zeros(1, in_dim), torch.zeros(out_dim, in_dim), None,
                         spectrum, n_frequency, seed, 1.0)
        flat = _positions(n_frequency, seed, out_dim, in_dim)
        nonzero = (delta != 0).nonzero(as_tuple=False)
        assert len(nonzero) == n_frequency, f"{shape} 非零元个数应为 {n_frequency}，实得 {len(nonzero)}"
        got = {(int(r) * in_dim + int(c)) for r, c in nonzero}
        assert got == {int(i) for i in flat}, f"{shape} 选中坐标与 randperm 不一致"
        for k in range(n_frequency):
            index = int(flat[k])
            assert torch.allclose(delta[index // in_dim, index % in_dim], spectrum[k], atol=1e-6, rtol=0)


def check_forward_is_merged_linear(solve):
    torch.manual_seed(3)
    w0 = torch.randn(3, 5)
    b0 = torch.randn(3)
    spectrum = torch.randn(6) * 0.1
    x = torch.randn(2, 4, 5)                       # 3D 批量
    y, delta = solve(x, w0, b0, spectrum, 6, 99, 3.5)
    assert y.dtype == torch.float32 and delta.dtype == torch.float32, "返回必须是 float32"
    merged = w0.to(torch.float64) + delta.to(torch.float64)
    ref = (x.to(torch.float64) @ merged.T + b0.to(torch.float64)).to(torch.float32)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5), "前向必须等于合并权重后的普通线性层"
    # 无偏置
    y2, d2 = solve(x, w0, None, spectrum, 6, 99, 3.5)
    ref2 = (x.to(torch.float64) @ (w0.to(torch.float64) + d2.to(torch.float64)).T).to(torch.float32)
    assert torch.allclose(y2, ref2, atol=1e-5, rtol=1e-5)


def check_zero_spectrum_identity(solve):
    # peft 初始化频谱为 0 => 适配器什么都不做
    _, delta = solve(X, W0, B0, torch.zeros(N_FREQ), N_FREQ, SEED, SCALING)
    assert torch.equal(delta, torch.zeros(3, 4)), "零频谱必须给出全零增量"
    y, _ = solve(X, W0, B0, torch.zeros(N_FREQ), N_FREQ, SEED, SCALING)
    assert torch.allclose(y, X @ W0.T + B0, atol=1e-6, rtol=0), "零频谱时前向必须等于基座前向"


def check_scaling_is_linear(solve):
    _, d1 = solve(X, W0, B0, SPECTRUM, N_FREQ, SEED, SCALING)
    _, d2 = solve(X, W0, B0, SPECTRUM, N_FREQ, SEED, 2.0 * SCALING)
    assert torch.allclose(d2, 2.0 * d1, atol=1e-5, rtol=1e-5), "scaling 应线性缩放增量"
    y1, _ = solve(X, W0, B0, SPECTRUM, N_FREQ, SEED, SCALING)
    y2, _ = solve(X, W0, B0, SPECTRUM, N_FREQ, SEED, 2.0 * SCALING)
    assert torch.allclose(y2 - y1, X @ (2.0 * d1 - d1).T, atol=1e-5, rtol=1e-5), "前向增量应同步线性"


def check_full_budget_covers_every_cell(solve):
    # n_frequency == d_out*d_in 时 randperm 是一个置换：每个格子恰好落一个系数
    out_dim, in_dim = 3, 4
    spectrum = torch.arange(1, out_dim * in_dim + 1, dtype=torch.float32)
    _, delta = solve(torch.zeros(1, in_dim), torch.zeros(out_dim, in_dim), None,
                     spectrum, out_dim * in_dim, SEED, 1.0)
    assert torch.all(delta != 0), "满预算时不应有空格子"
    assert torch.allclose(delta.flatten().sort().values, spectrum.sort().values, atol=1e-5, rtol=0), \
        "满预算增量应是频谱的一个置换"


def check_seed_changes_support(solve):
    def support(seed):
        _, delta = solve(torch.zeros(1, 4), torch.zeros(3, 4), None, SPECTRUM, N_FREQ, seed, 1.0)
        return {tuple(p) for p in (delta != 0).nonzero(as_tuple=False).tolist()}

    assert support(1) != support(777), "不同种子应选中不同坐标"


def check_invalid(solve):
    def expect(fn):
        try:
            fn()
        except ValueError:
            return
        except Exception as err:  # noqa: BLE001
            raise AssertionError(f"应抛 ValueError，实得 {type(err).__name__}: {err}")
        raise AssertionError("应抛 ValueError，但没有报错")

    expect(lambda: solve(X, W0, B0, SPECTRUM, 0, SEED, SCALING))
    expect(lambda: solve(X, W0, B0, torch.zeros(13), 13, SEED, SCALING))
    expect(lambda: solve(X, W0, B0, SPECTRUM[:3], N_FREQ, SEED, SCALING))
    expect(lambda: solve(X[:, :3], W0, B0, SPECTRUM, N_FREQ, SEED, SCALING))
    expect(lambda: solve(X, W0, B0[:2], SPECTRUM, N_FREQ, SEED, SCALING))
    expect(lambda: solve(X, W0[0], B0, SPECTRUM, N_FREQ, SEED, SCALING))
    expect(lambda: solve(X, W0, B0, SPECTRUM.reshape(2, 2), N_FREQ, SEED, SCALING))
