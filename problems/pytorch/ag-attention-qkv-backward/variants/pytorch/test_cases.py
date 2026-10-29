import torch

# 示例：单头、Sq=1、Sk=2、D=1，scale=0.5
# dP = dO·Vᵀ = 2·[1,3] = [2,6]；ΣP·dP = 0.5·2+0.5·6 = 4
# dS = P∘(dP−4)·scale = [0.5·(−2)·0.5, 0.5·2·0.5] = [−0.5, 0.5]
# dQ = Σ dS·K = −0.5·1+0.5·2 = 0.5；dK = dS·Q = [−2,2]；dV = P·dO = [1,1]
EXAMPLE_Q = [[[[4.0]]]]
EXAMPLE_K = [[[[1.0], [2.0]]]]
EXAMPLE_V = [[[[1.0], [3.0]]]]
EXAMPLE_P = [[[[0.5, 0.5]]]]
EXAMPLE_DO = [[[[2.0]]]]
EXAMPLE_DQ = [[[[0.5]]]]
EXAMPLE_DK = [[[[ -2.0], [2.0]]]]
EXAMPLE_DV = [[[[ 1.0], [1.0]]]]


def _t(x):
    return torch.tensor(x, dtype=torch.float64)


def check_example(solve):
    dQ, dK, dV = solve(_t(EXAMPLE_Q), _t(EXAMPLE_K), _t(EXAMPLE_V),
                       _t(EXAMPLE_P), _t(EXAMPLE_DO), scale=0.5)
    assert torch.allclose(dQ, _t(EXAMPLE_DQ), atol=1e-9)
    assert torch.allclose(dK, _t(EXAMPLE_DK), atol=1e-9)
    assert torch.allclose(dV, _t(EXAMPLE_DV), atol=1e-9)


check_example.example = True


def _fwd(Q, K, V, scale, mask=None):
    g = Q.shape[1] // K.shape[1]
    kh = K.repeat_interleave(g, dim=1)
    vh = V.repeat_interleave(g, dim=1)
    s = (Q @ kh.transpose(-1, -2)) * scale
    if mask is not None:
        s = s + mask
    p = torch.softmax(s, dim=-1)
    return p @ vh, p


def check_matches_autograd_mha(solve):
    torch.manual_seed(801)
    b, h, sq, sk, d = 2, 3, 4, 4, 5
    Q = torch.randn(b, h, sq, d, dtype=torch.float64, requires_grad=True)
    K = torch.randn(b, h, sk, d, dtype=torch.float64, requires_grad=True)
    V = torch.randn(b, h, sk, d, dtype=torch.float64, requires_grad=True)
    dO = torch.randn(b, h, sq, d, dtype=torch.float64)
    O, P = _fwd(Q, K, V, 1.0 / (2 * d) ** 0.5)
    refQ, refK, refV = torch.autograd.grad((O * dO).sum(), [Q, K, V])
    gQ, gK, gV = solve(Q.detach(), K.detach(), V.detach(), P.detach(), dO,
                       scale=1.0 / (2 * d) ** 0.5)
    assert torch.allclose(gQ, refQ, atol=1e-9)
    assert torch.allclose(gK, refK, atol=1e-9)
    assert torch.allclose(gV, refV, atol=1e-9)


def check_gqa_head_folding(solve):
    # Hq=4、Hkv=2：dK/dV 形状是 (B,2,Sk,D)，且等于同组两个头之和
    torch.manual_seed(802)
    b, hq, hkv, sq, sk, d = 2, 4, 2, 3, 5, 4
    Q = torch.randn(b, hq, sq, d, dtype=torch.float64, requires_grad=True)
    K = torch.randn(b, hkv, sk, d, dtype=torch.float64, requires_grad=True)
    V = torch.randn(b, hkv, sk, d, dtype=torch.float64, requires_grad=True)
    dO = torch.randn(b, hq, sq, d, dtype=torch.float64)
    O, P = _fwd(Q, K, V, 0.25)
    refQ, refK, refV = torch.autograd.grad((O * dO).sum(), [Q, K, V])
    gQ, gK, gV = solve(Q.detach(), K.detach(), V.detach(), P.detach(), dO, scale=0.25)
    assert gQ.shape == (b, hq, sq, d)
    assert gK.shape == (b, hkv, sk, d) and gV.shape == (b, hkv, sk, d)
    assert torch.allclose(gQ, refQ, atol=1e-9)
    assert torch.allclose(gK, refK, atol=1e-9)
    assert torch.allclose(gV, refV, atol=1e-9)
    # 折头口径：dV 的每个 kv 头 = 同组各头 Pᵀ@dO 之和
    vh = V.detach().repeat_interleave(hq // hkv, dim=1)
    per = torch.einsum('bhik,bhid->bhkd', P.detach(), dO)
    folded = per.view(b, hkv, hq // hkv, sk, d).sum(dim=2)
    assert torch.allclose(gV, folded.to(gV.dtype), atol=1e-9)
    assert vh.shape == (b, hq, sk, d)


def check_causal_mask_and_scale_linearity(solve):
    torch.manual_seed(803)
    b, h, s, d = 1, 2, 4, 3
    Q = torch.randn(b, h, s, d, dtype=torch.float64, requires_grad=True)
    K = torch.randn(b, h, s, d, dtype=torch.float64, requires_grad=True)
    V = torch.randn(b, h, s, d, dtype=torch.float64, requires_grad=True)
    dO = torch.randn(b, h, s, d, dtype=torch.float64)
    mask = torch.full((s, s), float('-inf'), dtype=torch.float64).triu(1)
    O, P = _fwd(Q, K, V, 0.5, mask)
    refQ, refK, refV = torch.autograd.grad((O * dO).sum(), [Q, K, V])
    gQ, gK, gV = solve(Q.detach(), K.detach(), V.detach(), P.detach(), dO, scale=0.5)
    assert torch.allclose(gQ, refQ, atol=1e-9)
    assert torch.allclose(gK, refK, atol=1e-9)
    assert torch.allclose(gV, refV, atol=1e-9)
    # 上三角被掩掉 ⇒ P=0 ⇒ 对分数的梯度也必为 0
    dP = dO @ V.detach().transpose(-1, -2)
    dS = (dP - (dP * P.detach()).sum(-1, keepdim=True)) * P.detach()
    assert torch.count_nonzero(dS.tril(-1) if False else dS.masked_select(
        torch.triu(torch.ones(s, s, dtype=torch.bool), 1))) == 0
    # scale 只影响 dQ/dK，不影响 dV
    a = solve(Q.detach(), K.detach(), V.detach(), P.detach(), dO, scale=0.5)
    c = solve(Q.detach(), K.detach(), V.detach(), P.detach(), dO, scale=1.0)
    assert torch.allclose(c[0], a[0] * 2, atol=1e-9) and torch.allclose(c[2], a[2], atol=1e-9)


def check_shapes_dtype_float32(solve):
    torch.manual_seed(804)
    b, hq, hkv, sq, sk, d = 2, 3, 3, 4, 6, 8
    Q = torch.softmax(torch.randn(b, hq, sq, d), dim=-1) * 0 + torch.randn(b, hq, sq, d)
    K = torch.randn(b, hkv, sk, d)
    V = torch.randn(b, hkv, sk, d)
    P = torch.softmax(torch.randn(b, hq, sq, sk), dim=-1)
    dO = torch.randn(b, hq, sq, d)
    gQ, gK, gV = solve(Q, K, V, P, dO)
    assert gQ.dtype == Q.dtype and gK.dtype == K.dtype and gV.dtype == V.dtype
    assert gQ.shape == Q.shape and gK.shape == K.shape and gV.shape == V.shape
    # dO 整体加常数会经 Σ_k P_k dP_k 这一项影响每一行的梯度（P 行和为 1 时并不抵消）
    shift = dO + 0.7
    hQ, _, _ = solve(Q, K, V, P, shift)
    assert not torch.allclose(hQ, gQ, atol=1e-6)


def check_invalid_args(solve):
    Q = torch.randn(1, 4, 2, 3)
    K = torch.randn(1, 3, 2, 3)
    V = torch.randn(1, 3, 2, 3)
    P = torch.rand(1, 4, 2, 2)
    dO = torch.randn(1, 4, 2, 3)
    bad = [
        (Q, K, V, P, dO),          # 4 % 3 != 0
        (Q, K, V.transpose(-1, -2).reshape(1, 3, 3, 2), P, dO),  # V 与 K 不同形
        (Q, K, V, torch.rand(1, 4, 2, 5), dO),                   # P 的 Sk 不符
        (Q, K, V, P, torch.randn(1, 4, 5, 3)),                   # dO 形状不符
        (Q.squeeze(0), K, V, P, dO),                             # Q 少一维
    ]
    for q, k, v, p, o in bad:
        try:
            solve(q, k, v, p, o)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
