import torch


def _ref_mask(shape, p, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=g, dtype=dtype) >= p).to(dtype)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.ones(1, 8)
    y = solve(x, p=0.5, training=True, seed=1234)
    mask = _ref_mask((1, 8), 0.5, 1234)
    expected = x * mask / 0.5
    assert torch.allclose(y, expected, atol=1e-7), f"got {y.tolist()}, expected {expected.tolist()}"
    assert set(y[0].tolist()) <= {0.0, 2.0}, "inverted dropout 输出只能是 0 或 x/(1-p)"


def check_eval_identity(solve):
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    y = solve(x, p=0.5, training=False)
    assert torch.equal(y, x), "eval 模式必须恒等"
    y2 = solve(x, p=0.9, training=False)
    assert torch.equal(y2, x), "高 p 下 eval 也必须恒等"


def check_inverted_scaling(solve):
    # 保留元素必须放大 1/(1-p)，使期望与原值一致
    x = torch.full((8, 128), 3.0)
    y = solve(x, p=0.25, training=True, seed=7)
    kept = y != 0
    assert torch.allclose(y[kept], torch.tensor(4.0), atol=1e-6), "保留值应为 3/(1-0.25)=4"
    ratio = kept.float().mean().item()
    assert abs(ratio - 0.75) < 0.05, f"保留率应≈0.75, got {ratio}"
    assert abs(y.mean().item() - 3.0) < 0.1, "期望应保持"


def check_zero_prob(solve):
    torch.manual_seed(1)
    x = torch.randn(3, 5)
    assert torch.equal(solve(x, p=0.0, training=True), x), "p=0 应恒等"


def check_seed_determinism_and_independence(solve):
    # 同 seed 同掩码；且不得触碰全局 RNG
    x = torch.ones(4, 4)
    y1 = solve(x, p=0.5, training=True, seed=99)
    y2 = solve(x, p=0.5, training=True, seed=99)
    assert torch.equal(y1, y2), "同 seed 应得同掩码"
    y3 = solve(x, p=0.5, training=True, seed=100)
    assert not torch.equal(y1, y3), "不同 seed 应得不同掩码"
    # 关键：用固定基准序列验证全局 RNG 未被消费
    torch.manual_seed(42)
    a = torch.rand(3)
    torch.manual_seed(42)
    solve(x, p=0.5, training=True, seed=7)
    b = torch.rand(3)
    assert torch.equal(a, b), "solve 不应消费全局 RNG 状态"


def check_float64(solve):
    x = torch.ones(2, 10, dtype=torch.float64)
    y = solve(x, p=0.5, training=True, seed=1234)
    mask = _ref_mask((2, 10), 0.5, 1234, dtype=torch.float64)
    assert y.dtype == torch.float64
    assert torch.allclose(y, x * mask / 0.5, atol=1e-12)


check_basic.example = True
