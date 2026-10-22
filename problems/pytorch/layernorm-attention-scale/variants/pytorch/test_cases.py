import math

import torch


def _rmsnorm(t, w, eps):
    return t / torch.sqrt(t.pow(2).mean(-1, keepdim=True) + eps) * w


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    q = torch.tensor([[[1.0, 1.0]]])
    k = torch.tensor([[[1.0, 1.0]]])
    wq = torch.ones(2)
    wk = torch.ones(2)
    y = solve(q, k, wq, wk, eps=1e-6)
    expected = 2.0 / math.sqrt(2)  # ≈1.4142
    assert torch.allclose(y, torch.tensor([[[expected]]]), atol=1e-5), f"got {y.tolist()}"


def check_matches_manual(solve):
    torch.manual_seed(42)
    q = torch.randn(2, 3, 16)
    k = torch.randn(2, 4, 16)
    wq = torch.randn(16)
    wk = torch.randn(16)
    y = solve(q, k, wq, wk, eps=1e-6)
    exp = _rmsnorm(q, wq, 1e-6) @ _rmsnorm(k, wk, 1e-6).transpose(-2, -1) / math.sqrt(16)
    assert torch.allclose(y, exp, atol=1e-5), "与手推公式不一致"
    assert y.shape == (2, 3, 4)


def check_scale_invariance(solve):
    # 核心卖点：Q/K 幅度整体放大不改变归一化后的分数（防注意力 logits 爆炸）
    torch.manual_seed(0)
    q = torch.randn(1, 4, 32)
    k = torch.randn(1, 4, 32)
    ones = torch.ones(32)
    y1 = solve(q, k, ones, ones)
    y2 = solve(q * 1000.0, k * 1000.0, ones, ones)
    assert torch.allclose(y1, y2, atol=1e-4), "分数应对 Q/K 幅度不变"
    assert y1.abs().max() < 2.0, "归一化后分数应有界（|score|<=sqrt(d)量级）"


def check_weight_effect(solve):
    # weight 全 1 时行范数恒为 sqrt(d)：score 满足 |q·k| <= d（柯西-施瓦茨）
    torch.manual_seed(1)
    q = torch.randn(8, 64)
    k = torch.randn(8, 64)
    ones = torch.ones(64)
    s = solve(q, k, ones, ones, eps=1e-6)
    assert (s.abs() <= math.sqrt(64) + 1e-3).all(), "分数应被 sqrt(d) 缩放后有界"


def check_asymmetric_weights(solve):
    # Q/K 用不同 weight：结果必须随之改变
    torch.manual_seed(2)
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 5, 8)
    ones = torch.ones(8)
    half = torch.full((8,), 0.5)
    y1 = solve(q, k, ones, ones)
    y2 = solve(q, k, ones, half)
    assert not torch.allclose(y1, y2, atol=1e-4), "K 的 weight 应生效"
    exp2 = _rmsnorm(q, ones, 1e-6) @ (_rmsnorm(k, half, 1e-6)).transpose(-2, -1) / math.sqrt(8)
    assert torch.allclose(y2, exp2, atol=1e-5)


def check_float64(solve):
    q = torch.randn(2, 4, dtype=torch.float64)
    k = torch.randn(3, 4, dtype=torch.float64)
    wq = torch.randn(4, dtype=torch.float64)
    wk = torch.randn(4, dtype=torch.float64)
    y = solve(q, k, wq, wk)
    assert y.dtype == torch.float64
    exp = _rmsnorm(q, wq, 1e-6) @ _rmsnorm(k, wk, 1e-6).T / math.sqrt(4)
    assert torch.allclose(y, exp, atol=1e-10)


check_basic.example = True
