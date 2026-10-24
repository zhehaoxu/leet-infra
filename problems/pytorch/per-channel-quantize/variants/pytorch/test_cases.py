import torch


def _oracle(x, axis):
    axis = axis % x.dim()
    reduce_dims = tuple(i for i in range(x.dim()) if i != axis)
    absmax = x.abs().amax(dim=reduce_dims) if reduce_dims else x.abs()
    scales = (absmax / 127.0).to(torch.float64).to(x.dtype)
    zps = torch.zeros(x.size(axis), dtype=torch.int32)
    return torch.fake_quantize_per_channel_affine(x, scales, zps, axis, -127, 127)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    x = torch.tensor([[1.0, -4.0, 2.0, 0.5],
                      [0.0, 0.0, 0.0, 0.0],
                      [3.0, 1.5, -6.0, -0.75]])
    out = solve(x, 0)
    expected = torch.tensor([[1.00787, -4.0, 2.01575, 0.50394],
                             [0.0, 0.0, 0.0, 0.0],
                             [3.02362, 1.51181, -6.0, -0.75591]])
    assert torch.allclose(out, expected, atol=1e-4), out.tolist()


def check_zero_channel_returns_zero(solve):
    # 全零通道必须输出 0（不是 NaN）
    x = torch.tensor([[5.0, -5.0], [0.0, 0.0]])
    out = solve(x, 0)
    assert torch.isfinite(out).all(), out
    assert out[1].tolist() == [0.0, 0.0], out[1].tolist()


def check_matches_oracle_2d(solve):
    torch.manual_seed(42)
    for axis in (0, 1):
        x = torch.randn(5, 7) * (torch.rand(()) * 4 + 0.1)
        assert torch.allclose(solve(x, axis), _oracle(x, axis), atol=1e-5), axis


def check_matches_oracle_conv3d(solve):
    torch.manual_seed(1)
    w = torch.randn(4, 3, 5, 5)  # conv 权重 [O,I,kh,kw]，per output-channel axis=0
    assert torch.allclose(solve(w, 0), _oracle(w, 0), atol=1e-5)


def check_negative_axis(solve):
    torch.manual_seed(2)
    x = torch.randn(6, 4)
    assert torch.allclose(solve(x, -1), _oracle(x, -1), atol=1e-5)
    assert torch.allclose(solve(x, -2), _oracle(x, -2), atol=1e-5)


def check_error_within_half_step(solve):
    torch.manual_seed(3)
    x = torch.randn(10, 16) * 3
    axis = 1
    absmax = x.abs().amax(dim=0)
    scale = (absmax / 127.0).view(1, -1)
    err = (solve(x, axis) - x).abs()
    assert (err <= scale / 2 + 1e-4).all(), err.max()


def check_dtype_preserved(solve):
    torch.manual_seed(4)
    x = torch.randn(4, 5, dtype=torch.float64)
    assert solve(x, 0).dtype == torch.float64


def check_matches_oracle_random(solve):
    # 随机多维 + 随机 axis 的对拍
    torch.manual_seed(5)
    for shape in [(8, 3), (2, 5, 4), (3, 2, 2, 6)]:
        x = torch.randn(*shape)
        for axis in range(len(shape)):
            assert torch.allclose(solve(x, axis), _oracle(x, axis), atol=1e-5), (shape, axis)


check_example.example = True
check_zero_channel_returns_zero.example = True
