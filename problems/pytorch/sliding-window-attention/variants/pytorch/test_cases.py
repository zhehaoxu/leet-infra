import math

import torch


def _ref(q, k, v, window):
    b, l, d = q.shape
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    idx = torch.arange(l)
    mask = (idx.view(l, 1) >= idx.view(1, l)) & ((idx.view(l, 1) - idx.view(1, l)) < window)
    s = s.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(s, -1), v)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    assert torch.allclose(solve(q, k, v, 2), _ref(q, k, v, 2), atol=1e-5)


def check_window_full_equals_causal(solve):
    torch.manual_seed(1)
    q = torch.randn(1, 5, 8)
    k = torch.randn(1, 5, 8)
    v = torch.randn(1, 5, 8)
    # window >= L 时等价于纯因果注意力
    assert torch.allclose(solve(q, k, v, 5), _ref(q, k, v, 5), atol=1e-5)
    # 第 0 行只看到自己 → 输出恰为 v0
    y = solve(q, k, v, 2)
    assert torch.allclose(y[:, 0], v[:, 0], atol=1e-6)


def check_locality(solve):
    # window=2：token i 只依赖 k/v 的 i、i-1
    torch.manual_seed(2)
    l, d = 6, 8
    q = torch.randn(1, l, d)
    k = torch.randn(1, l, d)
    v = torch.randn(1, l, d)
    y = solve(q, k, v, 2)
    v2 = v.clone()
    v2[:, 0] += 100.0
    y2 = solve(q, k, v2, 2)
    # 扰动位置 0 的 v，不应改变位置 3 及之后的输出（回看范围受限）
    assert torch.allclose(y[:, 3:], y2[:, 3:], atol=1e-6), "滑动窗口应限制回看范围"
    # 但位置 1 会受影响
    assert not torch.allclose(y[:, 1], y2[:, 1], atol=1e-4)


def check_batch_and_shape(solve):
    torch.manual_seed(3)
    q = torch.randn(3, 7, 16)
    k = torch.randn(3, 7, 16)
    v = torch.randn(3, 7, 16)
    y = solve(q, k, v, 3)
    assert y.shape == q.shape
    assert torch.allclose(y, _ref(q, k, v, 3), atol=1e-5)


check_basic.example = True
