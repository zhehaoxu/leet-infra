import numpy as np
import torch


def _kernel(up, down, taps, beta):
    fc = 0.5 / max(up, down)
    center = (taps - 1) // 2
    j = np.arange(taps, dtype=np.float64) - center
    h = 2.0 * fc * np.sinc(2.0 * fc * j) * np.kaiser(taps, beta)
    return h * (up / h.sum()), center


def _oracle(x, up, down, taps=63, beta=8.0):
    """独立 oracle：先补零上采样、再整段卷积、最后按步长抽取。"""
    h, center = _kernel(up, down, taps, beta)
    xs = np.zeros((len(x) - 1) * up + 1, dtype=np.float64)
    xs[::up] = x
    full = np.convolve(xs, h, mode="full")
    n_out = ((len(x) - 1) * up) // down + 1
    return full[center : center + n_out * down : down]


def check_example(solve):
    torch.manual_seed(0)
    x = torch.tensor([1.0, 0.5, -0.5, 0.25])
    y = solve(x, 2, 1, 7, 5.0)
    want = torch.tensor([1.01441, 0.754998, 0.507205, -0.009878, -0.507205, -0.129126, 0.253602])
    assert tuple(y.shape) == (7,), f"2 倍上采样后应有 7 个点，got {tuple(y.shape)}"
    assert torch.allclose(y, want, atol=1e-5), f"重采样结果不对：{y.tolist()}"

    z = solve(x, 1, 2, 7, 5.0)
    want_z = torch.tensor([0.631391, -0.065841])
    assert tuple(z.shape) == (2,), f"2 倍下采样后应有 2 个点，got {tuple(z.shape)}"
    assert torch.allclose(z, want_z, atol=1e-5), f"下采样结果不对：{z.tolist()}"


check_example.example = True


def check_lengths(solve):
    """输出长度只由 ((T-1)*up)//down + 1 决定。"""
    torch.manual_seed(1)
    x = torch.randn(9)
    for up, down in ((1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (3, 2), (2, 3), (4, 5)):
        y = solve(x, up, down)
        assert y.shape[0] == ((9 - 1) * up) // down + 1, f"up={up} down={down} 长度不对：{y.shape[0]}"
    assert solve(torch.tensor([2.0]), 3, 2).shape[0] == 1, "单点输入应输出 1 个点"


def check_dc_gain(solve):
    """常数信号在内部区域应原样重建（各相位的滤波器系数和都是 1，允许 ~1% 的截止边缘纹波）。"""
    torch.manual_seed(2)
    x = torch.ones(80) * 3.0
    for up, down in ((1, 1), (2, 1), (1, 2), (3, 2)):
        y = solve(x, up, down)
        core = y[15:-15]
        assert torch.allclose(core, torch.full_like(core, 3.0), atol=3e-2), (
            f"up={up} down={down} 直流增益不对：{core.min()}..{core.max()}"
        )

    # 上采样时偶数位恰好等于原信号（零相位对齐），奇数位是插值
    y = solve(x, 2, 1)
    assert torch.allclose(y[0::2], x, atol=1e-3), "隔点抽样应还原原信号"
    assert torch.allclose(y[15:-15:2], torch.full_like(y[15:-15:2], 3.0), atol=3e-2), "插值点也应接近 3"


def check_random_vs_oracle(solve):
    """随机信号 × 多种比例，对拍「补零上采样 + 卷积 + 抽取」的独立实现。"""
    torch.manual_seed(3)
    rng = np.random.default_rng(11)
    for up, down in ((1, 2), (2, 1), (3, 2), (2, 3), (1, 1), (4, 3)):
        for taps, beta in ((15, 4.0), (63, 8.0)):
            x = rng.normal(size=11)
            got = solve(torch.tensor(x, dtype=torch.float32), up, down, taps, beta)
            want = _oracle(x, up, down, taps, beta)
            assert got.shape[0] == want.shape[0], f"长度 {got.shape[0]} vs {want.shape[0]}"
            assert np.abs(got.numpy() - want).max() < 1e-4, f"up={up} down={down} taps={taps} 不一致"


def check_invalid_inputs(solve):
    good = torch.randn(6)
    for bad in (
        lambda: solve(torch.randn(2, 3), 2, 1),
        lambda: solve(good, 0, 1),
        lambda: solve(good, 2, 0),
        lambda: solve(good, 2, 1, 8),
        lambda: solve(good, 2, 1, 1),
        lambda: solve(good, 2, 1, 7, -1.0),
        lambda: solve(torch.tensor([]), 2, 1),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
