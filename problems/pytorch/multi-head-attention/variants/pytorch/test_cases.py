import math

import torch


def _ref_mha(q, k, v, h, mask=None):
    # 逐头调用单头注意力参考实现（与 F.scaled_dot_product_attention 同语义）
    b, lq, d = q.shape
    lk = k.shape[1]
    dh = d // h
    outs = []
    for i in range(h):
        qi = q[:, :, i * dh:(i + 1) * dh]
        ki = k[:, :, i * dh:(i + 1) * dh]
        vi = v[:, :, i * dh:(i + 1) * dh]
        s = torch.matmul(qi, ki.transpose(-2, -1)) / math.sqrt(dh)
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))  # bool: True=可见, False=屏蔽
        a = torch.softmax(s, dim=-1)
        outs.append(torch.matmul(a, vi))
    return torch.cat(outs, dim=-1)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    torch.manual_seed(0)
    q = torch.randn(1, 3, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)
    y = solve(q, k, v, num_heads=2)
    exp = _ref_mha(q, k, v, 2)
    assert torch.allclose(y, exp, atol=1e-5), f"got {y.tolist()}, expected {exp.tolist()}"


def check_matches_per_head_loop(solve):
    torch.manual_seed(42)
    q = torch.randn(2, 5, 16)
    k = torch.randn(2, 5, 16)
    v = torch.randn(2, 5, 16)
    for h in (1, 2, 4, 8, 16):
        y = solve(q, k, v, h)
        assert torch.allclose(y, _ref_mha(q, k, v, h), atol=1e-5), f"H={h} 与逐头循环不一致"


def check_head_split_order(solve):
    # 维度顺序陷阱：头是按「连续块」切分的（view 语义），合头时必须是 [head0 | head1]
    torch.manual_seed(1)
    q = torch.randn(1, 2, 4)
    k = torch.randn(1, 2, 4)
    v = torch.randn(1, 2, 4)
    y = solve(q, k, v, 2)
    exp = _ref_mha(q, k, v, 2)
    assert y.shape == (1, 2, 4)
    assert torch.allclose(y, exp, atol=1e-5), "切头/合头维度顺序错误"
    # 交换两个头的拼接顺序必须能被检出（两头输出不同）
    swapped = torch.cat([exp[:, :, 2:], exp[:, :, :2]], dim=-1)
    assert not torch.allclose(y, swapped, atol=1e-4), "头顺序不应可交换"


def check_eq_single_head(solve):
    # H=1 应与普通缩放点积注意力一致
    torch.manual_seed(2)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 6, 8)
    v = torch.randn(2, 6, 8)
    y = solve(q, k, v, 1)
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(8)
    exp = torch.matmul(torch.softmax(s, -1), v)
    assert torch.allclose(y, exp, atol=1e-5)


def check_inductive_bias_shape(solve):
    # 不同头应有不同输出（随机权重下）；且输出形状=输入 q 形状
    torch.manual_seed(3)
    q = torch.randn(2, 6, 12)
    k = torch.randn(2, 6, 12)
    v = torch.randn(2, 6, 12)
    y = solve(q, k, v, 3)
    assert y.shape == q.shape
    heads = y.reshape(2, 6, 3, 4)
    assert not torch.allclose(heads[:, :, 0], heads[:, :, 1], atol=1e-4), "不同头输出应不同"


check_basic.example = True
