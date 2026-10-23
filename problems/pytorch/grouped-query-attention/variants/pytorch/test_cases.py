import math

import torch


def _ref(q, k, v, h, kvh):
    b, lq, hd = q.shape
    dh = hd // h
    lk = k.shape[1]
    qh = q.view(b, lq, h, dh).transpose(1, 2)
    kh = k.view(b, lk, kvh, dh).transpose(1, 2)
    vh = v.view(b, lk, kvh, dh).transpose(1, 2)
    kh = kh.repeat_interleave(h // kvh, dim=1)
    vh = vh.repeat_interleave(h // kvh, dim=1)
    s = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(dh)
    a = torch.softmax(s, dim=-1)
    return torch.matmul(a, vh).transpose(1, 2).reshape(b, lq, hd)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    torch.manual_seed(0)
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    # num_kv == num_heads → 与标准 MHA 一致
    y = solve(q, k, v, 2, 2)
    assert torch.allclose(y, _ref(q, k, v, 2, 2), atol=1e-5)
    assert y.shape == q.shape


def check_gqa(solve):
    torch.manual_seed(1)
    dh = 8
    for h, kvh in [(4, 2), (4, 1)]:
        q = torch.randn(2, 5, h * dh)
        k = torch.randn(2, 5, kvh * dh)
        v = torch.randn(2, 5, kvh * dh)
        y = solve(q, k, v, h, kvh)
        assert torch.allclose(y, _ref(q, k, v, h, kvh), atol=1e-5), f"h={h},kvh={kvh}"


def check_mqa(solve):
    # 多查询=单 kv 头；所有 query 头共享同一组 kv
    torch.manual_seed(2)
    q = torch.randn(1, 6, 24)
    k = torch.randn(1, 6, 6)
    v = torch.randn(1, 6, 6)
    y = solve(q, k, v, 4, 1)
    assert y.shape == q.shape
    assert torch.allclose(y, _ref(q, k, v, 4, 1), atol=1e-5)


def check_group_assignment(solve):
    # 分组归属：第 i 个 query 头匹配 kv 头 i // (h//kvh)（连续成组），非循环取模
    torch.manual_seed(3)
    h, kvh, dh = 4, 2, 4
    q = torch.randn(1, 3, h * dh)
    k = torch.randn(1, 3, kvh * dh)
    v = torch.randn(1, 3, kvh * dh)
    # 给两组 kv 各自叠加不同常数，放大组间差异（按 kv 头分块广播）
    v = (v.view(1, 3, kvh, dh) + torch.arange(kvh).view(1, 1, kvh, 1) * 5.0).reshape(1, 3, kvh * dh)
    y = solve(q, k, v, h, kvh)
    heads = y.reshape(1, 3, h, dh)
    # 头 0,1 属 kv0；头 2,3 属 kv1 —— 不同组输出块不应相同
    assert not torch.allclose(heads[:, :, 0], heads[:, :, 2], atol=1e-4), "不同 kv 组输出应不同"
    assert torch.allclose(y, _ref(q, k, v, h, kvh), atol=1e-5)


check_basic.example = True
