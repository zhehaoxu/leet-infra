import math

import torch


def _std(q, k, v):
    d = q.shape[-1]
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    return torch.matmul(torch.softmax(s, -1), v)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    y = solve(q, k, v, block=2)
    assert torch.allclose(y, _std(q, k, v), atol=1e-5)


def check_matches_standard_any_block(solve):
    torch.manual_seed(1)
    q = torch.randn(2, 6, 16)
    k = torch.randn(2, 6, 16)
    v = torch.randn(2, 6, 16)
    for blk in (1, 2, 3, 4, 8):
        assert torch.allclose(solve(q, k, v, blk), _std(q, k, v), atol=1e-5), f"block={blk}"


def check_uneven_and_lk_neq_lq(solve):
    # 长度非 block 整数倍、且 K/V 长度 ≠ Q 长度
    torch.manual_seed(2)
    q = torch.randn(1, 5, 8)
    k = torch.randn(1, 7, 8)
    v = torch.randn(1, 7, 8)
    y = solve(q, k, v, block=3)
    assert y.shape == q.shape
    assert torch.allclose(y, _std(q, k, v), atol=1e-5)


def check_large_values_stability(solve):
    # 数值稳定：logits 很大时也要与标准 softmax 等价（在线 max 归约的作用）
    torch.manual_seed(3)
    q = torch.randn(1, 4, 8) * 20
    k = torch.randn(1, 4, 8) * 20
    v = torch.randn(1, 4, 8)
    assert torch.allclose(solve(q, k, v, block=2), _std(q, k, v), atol=1e-4)
    assert torch.isfinite(solve(q, k, v, block=2)).all(), "大 logits 不应溢出 NaN/Inf"


check_basic.example = True
