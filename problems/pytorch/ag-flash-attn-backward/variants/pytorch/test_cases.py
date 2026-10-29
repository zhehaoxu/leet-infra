import math

import torch

# 示例：B=H=1、Sq=1、Sk=2、D=2；Q=[0,1]、K=[[1,0],[0,0]] ⇒ S = [0, 0]
# P = [0.5, 0.5] ⇒ LSE = log 2；V = [[0,0],[1,1]] ⇒ O = [0.5, 0.5]；dO = [1, 0]
# dV = Pᵀ·dO = [[0.5,0],[0.5,0]]；dP = [0,1]；D = dO·O = 0.5
# dS = P∘(dP−D) = [−0.25, 0.25]；dQ = dS@K = [−0.25, 0]；dK = dSᵀ∘Q = [[0,−0.25],[0,0.25]]
EXAMPLE_Q = [[[[0.0, 1.0]]]]
EXAMPLE_K = [[[[1.0, 0.0], [0.0, 0.0]]]]
EXAMPLE_V = [[[[0.0, 0.0], [1.0, 1.0]]]]
EXAMPLE_O = [[[[0.5, 0.5]]]]
EXAMPLE_DO = [[[[1.0, 0.0]]]]
EXAMPLE_DQ = [[[[ -0.25, 0.0]]]]
EXAMPLE_DK = [[[[0.0, -0.25], [0.0, 0.25]]]]
EXAMPLE_DV = [[[[0.5, 0.0], [0.5, 0.0]]]]


def _lse_example():
    return torch.tensor([[[math.log(2.0)]]])


def _naive(q, k, v, scale, mask):
    """测试内独立实现的前向：返回 (O, LSE)，并保留 autograd 追踪。"""
    s = (q @ k.transpose(-1, -2)) * scale
    if mask is not None:
        s = s + mask
    p = torch.softmax(s, dim=-1)
    return p @ v, torch.logsumexp(s, dim=-1)


def _refs(q, k, v, do, scale, mask):
    q = q.clone().requires_grad_(True)
    k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    o, _ = _naive(q, k, v, scale, mask)
    (dq, dk, dv) = torch.autograd.grad((o * do).sum(), [q, k, v])
    return o.detach(), dq.detach(), dk.detach(), dv.detach()


def check_example(solve):
    q = torch.tensor(EXAMPLE_Q)
    k = torch.tensor(EXAMPLE_K)
    v = torch.tensor(EXAMPLE_V)
    o = torch.tensor(EXAMPLE_O)
    do = torch.tensor(EXAMPLE_DO)
    out = solve(q, k, v, o, _lse_example(), do)
    assert isinstance(out, tuple) and len(out) == 3
    dq, dk, dv = out
    assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape
    assert torch.allclose(dq, torch.tensor(EXAMPLE_DQ), atol=1e-6), dq
    assert torch.allclose(dk, torch.tensor(EXAMPLE_DK), atol=1e-6), dk
    assert torch.allclose(dv, torch.tensor(EXAMPLE_DV), atol=1e-6), dv


check_example.example = True


def check_matches_naive_autograd(solve):
    torch.manual_seed(301)
    b, h, sq, sk, d = 2, 3, 5, 7, 4
    q = torch.randn(b, h, sq, d, dtype=torch.float64)
    k = torch.randn(b, h, sk, d, dtype=torch.float64)
    v = torch.randn(b, h, sk, d, dtype=torch.float64)
    do = torch.randn(b, h, sq, d, dtype=torch.float64)
    scale = 0.35
    o, dq, dk, dv = _refs(q, k, v, do, scale, None)
    lse = torch.logsumexp((q @ k.transpose(-1, -2)) * scale, dim=-1)
    got = solve(q, k, v, o, lse, do, scale=scale)
    for g, r in zip(got, (dq, dk, dv)):
        assert g.dtype == r.dtype and g.shape == r.shape
        assert torch.allclose(g, r, atol=1e-9), (g - r).abs().max()


def check_causal_mask(solve):
    # 掩码必须由前向的 S 与 LSE 一致地重建：漏掉 mask 的实现会在上三角给出非零梯度
    torch.manual_seed(302)
    b, h, s, d = 1, 2, 6, 3
    q = torch.randn(b, h, s, d, dtype=torch.float64)
    k = torch.randn(b, h, s, d, dtype=torch.float64)
    v = torch.randn(b, h, s, d, dtype=torch.float64)
    do = torch.randn(b, h, s, d, dtype=torch.float64)
    neg = torch.full((s, s), float('-inf'), dtype=torch.float64)
    mask = torch.triu(neg, diagonal=1)                      # (Sq, Sk) 广播
    scale = 1.0 / math.sqrt(d)
    o, dq, dk, dv = _refs(q, k, v, do, scale, mask)
    sfull = (q @ k.transpose(-1, -2)) * scale + mask
    lse = torch.logsumexp(sfull, dim=-1)
    got = solve(q, k, v, o, lse, do, scale=scale, mask=mask)
    for g, r in zip(got, (dq, dk, dv)):
        assert torch.allclose(g, r, atol=1e-9), (g - r).abs().max()
    # 被屏蔽的位置不贡献 dK：第 j 行只应收到 i >= j 的 dS
    assert torch.isfinite(torch.stack(got)).all()


def check_d_term_uses_dot_with_o(solve):
    # D = rowsum(dO ∘ O) 而不是 rowsum(dP)：把 O 沿常数平移时，只要 dO 每行和为 0，
    # D 就不变，梯度也必须一字不差；用 dP 的行和（或漏掉归一化项）做不到这一点。
    torch.manual_seed(303)
    b, h, sq, sk, d = 2, 2, 4, 6, 3
    q = torch.randn(b, h, sq, d, dtype=torch.float64)
    k = torch.randn(b, h, sk, d, dtype=torch.float64)
    v = torch.randn(b, h, sk, d, dtype=torch.float64)
    do = torch.randn(b, h, sq, d, dtype=torch.float64)
    do = do - do.mean(dim=-1, keepdim=True)          # 每行和为 0
    assert torch.allclose(do.sum(dim=-1), torch.zeros(b, h, sq, dtype=torch.float64), atol=1e-12)
    o, dq, dk, dv = _refs(q, k, v, do, 1.0, None)
    lse = torch.logsumexp(q @ k.transpose(-1, -2), dim=-1)
    base = solve(q, k, v, o, lse, do)
    for g, r in zip(base, (dq, dk, dv)):
        assert torch.allclose(g, r, atol=1e-9), (g - r).abs().max()
    shifted = solve(q, k, v, o + 100.0, lse, do)
    for g0, g1 in zip(base, shifted):
        assert torch.allclose(g0, g1, atol=1e-12), "dO 行和为 0 时 O 的常数平移不该改变 D"
    # 反过来：dO 行和不为 0 时，O 平移必然改变 D，梯度也要跟着变
    do1 = torch.ones(b, h, sq, d, dtype=torch.float64)
    o2, dq2, dk2, dv2 = _refs(q, k, v, do1, 1.0, None)
    a = solve(q, k, v, o2, lse, do1)[0]
    bb = solve(q, k, v, o2 + 100.0, lse, do1)[0]
    assert not torch.allclose(a, bb, atol=1e-6), "D = dO·O 随 O 平移而变，缺 D 项的实现这里会漏"


def check_lse_shift_scaling(solve):
    # 单独给 LSE 加常数 c ⇒ P 缩放 e^(−c)，三个梯度整体按 e^(−c) 缩放（漏减 LSE 或改减 max 都不满足）
    torch.manual_seed(304)
    b, h, sq, sk, d = 1, 2, 3, 5, 2
    q = torch.randn(b, h, sq, d, dtype=torch.float64)
    k = torch.randn(b, h, sk, d, dtype=torch.float64)
    v = torch.randn(b, h, sk, d, dtype=torch.float64)
    do = torch.randn(b, h, sq, d, dtype=torch.float64)
    o = torch.softmax(q @ k.transpose(-1, -2), dim=-1) @ v
    lse = torch.logsumexp(q @ k.transpose(-1, -2), dim=-1)
    base = solve(q, k, v, o, lse, do)
    c = 1.75
    shifted = solve(q, k, v, o, lse + c, do)
    f = math.exp(-c)
    for g0, g1 in zip(base, shifted):
        assert torch.allclose(g1, g0 * f, atol=1e-9), (g1 - g0 * f).abs().max()
    # mask 加同一个常数（与 S 同平移）而 LSE 也加 c ⇒ 结果不变
    mask = torch.full((1, 1, sq, sk), c, dtype=torch.float64)
    same = solve(q, k, v, o, lse + c, do, mask=mask)
    for g0, g1 in zip(base, same):
        assert torch.allclose(g0, g1, atol=1e-9)


def check_dtype_and_batch(solve):
    torch.manual_seed(305)
    q = torch.randn(1, 1, 3, 8)                    # float32 进 → float32 出
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    s = q @ k.transpose(-1, -2)
    o = torch.softmax(s, dim=-1) @ v
    lse = torch.logsumexp(s, dim=-1)
    do = torch.randn_like(o)
    got = solve(q, k, v, o, lse, do)
    for g, src in zip(got, (q, k, v)):
        assert g.dtype == src.dtype, (g.dtype, src.dtype)
    # Sk != Sq 且 head_dim = 1 的退化形状
    q2 = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    k2 = torch.zeros(1, 1, 5, 1, dtype=torch.float64)
    v2 = torch.arange(5.0, dtype=torch.float64).view(1, 1, 5, 1)
    s2 = q2 @ k2.transpose(-1, -2)
    o2 = torch.softmax(s2, dim=-1) @ v2
    lse2 = torch.logsumexp(s2, dim=-1)
    do2 = torch.ones(1, 1, 2, 1, dtype=torch.float64)
    gq2, gk2, gv2 = solve(q2, k2, v2, o2, lse2, do2)
    assert gq2.shape == q2.shape and gk2.shape == k2.shape
    # S 全 0 ⇒ P = 1/5；Sq = 2 行、dO 全 1 ⇒ dV[j] = Σ_i P[i,j]·dO[i] = 0.2 + 0.2 = 0.4
    assert torch.allclose(gv2, torch.full_like(gv2, 0.4), atol=1e-9), gv2


def check_invalid_args(solve):
    torch.manual_seed(306)
    q = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    k = torch.randn(1, 2, 5, 4, dtype=torch.float64)
    v = torch.randn(1, 2, 5, 4, dtype=torch.float64)
    s = q @ k.transpose(-1, -2)
    o = torch.softmax(s, dim=-1) @ v
    lse = torch.logsumexp(s, dim=-1)
    do = torch.randn_like(o)
    bad = [
        (q.squeeze(0), k, v, o, lse, do, 1.0, None),          # Q 三维
        (q, k, v.transpose(-1, -2), o, lse, do, 1.0, None),   # V 形状不对
        (q, k, v, o, lse[:, :, 1:], do, 1.0, None),           # LSE 的 Sq 少一格
        (q, k, v, o, lse.unsqueeze(-1), do, 1.0, None),       # LSE 多一轴
        (q, k, v, o[:, :, 1:], lse, do, 1.0, None),           # O 的 Sq 少一格
        (q, k, v, o, lse, do[:, :, 1:], 1.0, None),           # dO 的 Sq 少一格
        (q, k, v[:, :1], o, lse, do, 1.0, None),              # V 的 Sk 与 K 不符
        (q, k, v, o, lse, do, 1.0, torch.zeros(2, 3, 5, 7, dtype=torch.float64)),  # mask 不可广播
    ]
    for qq, kk, vv, oo, ll, dd, sc, mk in bad:
        try:
            solve(qq, kk, vv, oo, ll, dd, scale=sc, mask=mk)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
