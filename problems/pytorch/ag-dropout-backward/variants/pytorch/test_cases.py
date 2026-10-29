import torch

# 示例：p=0.5 ⇒ 保留位放大 2 倍；mask=[[1,0],[0,1]]、dY=[[1,2],[3,4]]
EXAMPLE_DY = [[1.0, 2.0], [3.0, 4.0]]
EXAMPLE_MASK = [[1, 0], [0, 1]]
EXAMPLE_OUT = [[2.0, 0.0], [0.0, 8.0]]


def check_example(solve):
    out = solve(torch.tensor(EXAMPLE_DY), torch.tensor(EXAMPLE_MASK), 0.5)
    assert torch.allclose(out, torch.tensor(EXAMPLE_OUT), atol=1e-6)


check_example.example = True


def check_matches_manual_forward(solve):
    # 独立写出前向 y = x·mask/(1−p)，用 autograd 产生参考梯度
    torch.manual_seed(1201)
    for p in (0.1, 0.25, 0.5):
        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        mask = (torch.rand(4, 6, dtype=torch.float64) > p).to(torch.float64)
        y = x * mask / (1.0 - p)
        dY = torch.randn_like(y)
        (ref,) = torch.autograd.grad((y * dY).sum(), x)
        got = solve(dY, mask, p)
        assert got.shape == x.shape, p
        assert torch.allclose(got, ref, atol=1e-12), p


def check_eval_is_identity(solve):
    torch.manual_seed(1202)
    dY = torch.randn(3, 5, dtype=torch.float64)
    mask = (torch.rand(3, 5, dtype=torch.float64) > 0.3).to(torch.float64)
    out = solve(dY, mask, 0.3, train=False)
    assert torch.allclose(out, dY, atol=0.0)
    # eval 态与 mask/p 完全无关
    assert torch.allclose(solve(dY, torch.zeros(3, 5, dtype=torch.float64), 0.9,
                                train=False), dY, atol=0.0)


def check_p_zero_and_full_mask(solve):
    dY = torch.tensor([[1.5, -2.0], [0.25, 7.0]], dtype=torch.float64)
    ones = torch.ones(2, 2, dtype=torch.float64)
    assert torch.allclose(solve(dY, ones, 0.0), dY, atol=1e-12)
    zeros = torch.zeros(2, 2, dtype=torch.float64)
    assert torch.count_nonzero(solve(dY, zeros, 0.5)) == 0
    # bool 掩码同样可用
    assert torch.allclose(solve(dY, ones.bool(), 0.0), dY, atol=1e-12)
    # 期望守恒：E[Σdx] = ΣdY（mask 按 (1−p) 的保留率给出）
    torch.manual_seed(1203)
    big = torch.ones(64, 64, dtype=torch.float64)
    m = (torch.rand(64, 64, dtype=torch.float64) > 0.4).to(torch.float64)
    got = solve(big, m, 0.4)
    assert abs(float(got.sum()) / float(big.numel()) - 1.0) < 0.05


def check_broadcast_channel_mask(solve):
    # dropout2d：mask 形状 (1, C, 1, 1)，整个通道一起丢/一起放
    x = torch.ones(2, 3, 4, 5, dtype=torch.float64)
    mask = torch.tensor([1.0, 0.0, 1.0]).reshape(1, 3, 1, 1)
    got = solve(x, mask, 0.5)
    assert got.shape == x.shape
    assert torch.allclose(got[:, 0], torch.full((2, 4, 5), 2.0, dtype=torch.float64))
    assert torch.count_nonzero(got[:, 1]) == 0
    assert torch.allclose(got[:, 2], torch.full((2, 4, 5), 2.0, dtype=torch.float64))


def check_dtype_and_invalid(solve):
    dY = torch.randn(2, 3)
    mask = torch.ones(2, 3)
    assert solve(dY, mask, 0.2).dtype == torch.float32
    assert solve(dY.double(), mask.double(), 0.2).dtype == torch.float64
    bad = [
        (dY, mask, 1.0, True),        # p=1 除零
        (dY, mask, -0.1, True),       # p 负
        (dY, torch.ones(4, 3), 0.2, True),   # mask 不可广播
        (dY, mask, 0.2, "yes"),       # train 非 bool
    ]
    for dy, mm, pp, tr in bad:
        try:
            solve(dy, mm, pp, train=tr)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
