import torch


def _oracle(latents, scale, shift, mean, std, inverse):
    """独立 oracle：逐元素按 (batch, channel) 索引取统计量，用 python float 重算。"""
    rows = []
    for i in range(latents.shape[0]):
        block = latents[i]
        plane = []
        for c in range(block.shape[0]):
            mu = 0.0 if mean is None else mean[c]
            sd = 1.0 if std is None else std[c]
            chan = []
            for v in block[c].flatten().tolist():
                if inverse:
                    raw = (v - shift) * scale
                    chan.append((raw - mu) / sd if std is not None else raw)
                else:
                    raw = v * sd + mu if std is not None else v
                    chan.append(raw / scale + shift)
            plane.append(torch.tensor(chan).reshape(block[c].shape))
        rows.append(torch.stack(plane))
    return torch.stack(rows)


def check_example(solve):
    z = torch.tensor([[0.5, -1.0], [2.0, 0.0]])
    mean, std = [0.2, -0.3], [1.5, 0.5]

    got = solve(z, 0.18215, 0.0609, mean, std)
    want = [[5.276382, -4.331085], [17.628838, -1.586094]]
    assert [[round(float(v), 6) for v in row] for row in got.tolist()] == want, got.tolist()

    plain = solve(z, 0.18215, 0.0609)
    want_plain = [[2.805890, -5.429081], [11.040861, 0.060900]]
    assert [[round(float(v), 6) for v in row] for row in plain.tolist()] == want_plain, plain.tolist()
    assert got.dtype == torch.float32 and got.shape == z.shape


check_example.example = True


def check_roundtrip(solve):
    torch.manual_seed(0)
    z = torch.randn(2, 3, 4, 4, dtype=torch.float64)
    mean = torch.tensor([0.1, -0.2, 0.35], dtype=torch.float64)
    std = torch.tensor([1.4, 0.6, 2.5], dtype=torch.float64)

    for kwargs in (
        {"scaling_factor": 0.18215, "shift_factor": 0.0609},
        {"scaling_factor": 0.13025},
        {"scaling_factor": 0.3611, "shift_factor": 0.1159, "channel_mean": mean, "channel_std": std},
        {"channel_mean": mean, "channel_std": std},
        {"scaling_factor": 1.0, "shift_factor": -0.4, "channel_mean": 0.25, "channel_std": 2.0},
    ):
        x = solve(z, **kwargs)
        back = solve(x, inverse=True, **kwargs)
        assert torch.allclose(back, z.to(torch.float32), atol=1e-5), f"{kwargs} 往返失败"
        assert x.dtype == torch.float32


def check_order_of_operations(solve):
    z = torch.tensor([[1.0, -2.0]])
    scale, shift = 0.5, 0.25

    # 解码：先 /scale 再 +shift（顺序写反会得到 (1/0.5 + 0.25) 之外的数）
    got = solve(z, scale, shift)
    want = torch.tensor([[1.0 / scale + shift, -2.0 / scale + shift]])
    assert torch.allclose(got, want, atol=1e-6), got.tolist()

    # 编码：先减 shift 再乘 scale
    x = torch.tensor([[2.25, -3.75]])
    back = solve(x, scale, shift, inverse=True)
    assert torch.allclose(back, torch.tensor([[1.0, -2.0]]), atol=1e-6), back.tolist()

    # 默认参数是恒等
    torch.manual_seed(1)
    rand = torch.randn(2, 3, 2)
    assert torch.allclose(solve(rand), rand, atol=1e-7), "scale=1、shift=0 时应恒等"
    assert torch.allclose(solve(rand, inverse=True), rand, atol=1e-7)


def check_per_channel_stats(solve):
    mean = torch.tensor([0.2, -0.3, 1.1], dtype=torch.float64)
    std = torch.tensor([1.5, 0.5, 2.0], dtype=torch.float64)
    scale, shift = 0.2, 0.1

    zeros = torch.zeros(1, 3, 2)
    out = solve(zeros, scale, shift, mean, std)
    want = (mean / scale + shift).to(torch.float32).reshape(1, 3, 1).expand_as(out)
    assert torch.allclose(out, want, atol=1e-6), "z=0 应映到每通道的 mean/scale+shift"

    torch.manual_seed(2)
    z = torch.randn(1, 3, 5, dtype=torch.float64)
    out = solve(z, scale, shift, mean, std).to(torch.float64)
    # 通道内的差分只被 std/scale 放大
    diff_out = out - out.mean(dim=2, keepdim=True)
    diff_z = z - z.mean(dim=2, keepdim=True)
    ratio = (std / scale).reshape(1, 3, 1)
    assert torch.allclose(diff_out, diff_z * ratio, atol=1e-5), "每通道的尺度因子不对"

    # 标量统计量等价于整条向量
    flat_mean = torch.full((3,), 0.5, dtype=torch.float64)
    flat_std = torch.full((3,), 2.0, dtype=torch.float64)
    assert torch.allclose(
        solve(z, scale, shift, 0.5, 2.0), solve(z, scale, shift, flat_mean, flat_std), atol=1e-6
    ), "标量与等值向量应一致"


def check_oracle_and_invalid(solve):
    torch.manual_seed(3)
    z = torch.randn(2, 2, 3, dtype=torch.float64) * 2.0
    mean = [0.15, -0.6]
    std = [0.9, 1.7]

    for scale, shift, stats in (
        (0.18215, 0.0609, True),
        (0.13025, 0.0, False),
        (1.0, -0.25, True),
    ):
        kwargs = {"channel_mean": mean, "channel_std": std} if stats else {}
        got = solve(z, scale, shift, **kwargs)
        want = _oracle(z, scale, shift, mean if stats else None, std if stats else None, False)
        assert torch.allclose(got, want.to(torch.float32), atol=1e-6), f"解码方向与 oracle 不一致"

        back = solve(got, scale, shift, inverse=True, **kwargs)
        want_back = _oracle(got.to(torch.float64), scale, shift, mean if stats else None, std if stats else None, True)
        assert torch.allclose(back, want_back.to(torch.float32), atol=1e-5), "编码方向与 oracle 不一致"

    good = torch.zeros(2, 2)
    for bad in (
        lambda: solve([[0.0, 1.0]], 0.5),
        lambda: solve(torch.zeros(4), 0.5),
        lambda: solve(good, 0.0),
        lambda: solve(good, -0.5),
        lambda: solve(good, 0.5, float("nan")),
        lambda: solve(good, 0.5, 0.0, channel_mean=[0.1, 0.2]),
        lambda: solve(good, 0.5, 0.0, channel_std=[1.0, 1.0]),
        lambda: solve(good, 0.5, 0.0, [0.1, 0.2, 0.3], [1.0, 1.0, 1.0]),
        lambda: solve(good, 0.5, 0.0, [[0.1], [0.2]], [1.0, 1.0]),
        lambda: solve(good, 0.5, 0.0, [0.1, 0.2], [1.0, 0.0]),
        lambda: solve(good, 0.5, 0.0, [0.1, 0.2], [-1.0, 1.0]),
        lambda: solve(good, 0.5, 0.0, [0.1, float("nan")], [1.0, 1.0]),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
