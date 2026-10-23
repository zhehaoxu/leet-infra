import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = solve(x, 50.0)
    exp = torch.softmax(50.0 * torch.tanh(x / 50.0), dim=-1)
    assert torch.allclose(y, exp, atol=1e-6)
    assert torch.allclose(y.sum(-1), torch.ones(1), atol=1e-6)


def check_matches_reference(solve):
    torch.manual_seed(0)
    s = torch.randn(2, 4, 8, 8) * 30
    for cap in (15.0, 30.0, 50.0):
        y = solve(s, cap)
        exp = torch.softmax(cap * torch.tanh(s / cap), dim=-1)
        assert torch.allclose(y, exp, atol=1e-5), f"cap={cap}"
        assert torch.allclose(y.sum(-1), torch.ones_like(y.sum(-1)), atol=1e-5)


def check_bounded_logits(solve):
    # 极端大 logit 被 tanh 封顶到 ±cap，仍是合法概率分布、无 NaN/Inf
    s = torch.tensor([[1e6, -1e6, 0.0]])
    y = solve(s, 30.0)
    assert torch.isfinite(y).all(), "极端 logits 不应产生 NaN/Inf"
    assert (y >= 0).all() and torch.allclose(y.sum(-1), torch.ones(1), atol=1e-6)
    assert y.max(-1).values.item() > 0.99, "最大项经封顶后概率仍应接近 1"


def check_monotonic_small_cap(solve):
    # cap 小：softmax 前 logits 被压缩但保序，概率随 logit 单调递增
    s = torch.tensor([[0.0, 1.0, 2.0]])
    y = solve(s, 1.0)
    assert bool((y[0, 2] > y[0, 1] > y[0, 0]).item()), "输出应随输入 logit 单调"


check_basic.example = True
