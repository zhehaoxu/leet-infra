import math
import random

import torch


# ---------- 纯 Python oracle（与 solution 同公式，独立实现） ----------

def _oracle(q, k, v, beta, g, scale=None, initial_state=None):
    B, T, H, K = len(q), len(q[0]), len(q[0][0]), len(q[0][0][0])
    V = len(v[0][0][0])
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    o = [[[[0.0] * V for _ in range(T)] for _ in range(H)] for _ in range(B)]
    h = [[[[0.0] * V for _ in range(K)] for _ in range(H)] for _ in range(B)]
    if initial_state is not None:
        for b in range(B):
            for hh in range(H):
                for i in range(K):
                    for j in range(V):
                        h[b][hh][i][j] = initial_state[b][hh][i][j]
    for b in range(B):
        for hh in range(H):
            for t in range(T):
                decay = math.exp(g[b][t][hh])
                for i in range(K):
                    for j in range(V):
                        h[b][hh][i][j] *= decay
                # v' = (v - h·k) * beta
                hk = [sum(h[b][hh][i][j] * k[b][t][hh][i] for i in range(K)) for j in range(V)]
                bv = [(v[b][t][hh][j] - hk[j]) * beta[b][t][hh] for j in range(V)]
                for i in range(K):
                    for j in range(V):
                        h[b][hh][i][j] += k[b][t][hh][i] * bv[j]
                # o = (q*scale)·h
                for j in range(V):
                    o[b][hh][t][j] = sum(
                        q[b][t][hh][i] * scale * h[b][hh][i][j] for i in range(K))
    # 转回 [B,T,H,·]
    o_t = [[[[o[b][hh][t][j] for j in range(V)] for hh in range(H)] for t in range(T)] for b in range(B)]
    return o_t, h


def _py(t):
    return t.detach().float().tolist()


def _close(a, b, tol=1e-4):
    fa, fb = [], []

    def rec(x, y):
        if isinstance(x, list):
            for u, v in zip(x, y):
                rec(u, v)
        else:
            fa.append(x)
            fb.append(y)
    rec(a, b)
    return all(abs(p - q) <= tol * max(1.0, abs(q)) for p, q in zip(fa, fb))


def _rand_case(rng, T=None):
    B, H = rng.randint(1, 2), rng.randint(1, 3)
    T = T or rng.randint(1, 6)
    K, V = rng.randint(1, 4), rng.randint(1, 4)
    q = torch.randn(B, T, H, K) * 0.7
    k = torch.randn(B, T, H, K) * 0.7
    v = torch.randn(B, T, H, V) * 0.7
    beta = torch.rand(B, T, H)  # [0,1)
    g = -torch.rand(B, T, H) * 0.8  # 负 log 衰减
    return q, k, v, beta, g


def check_random(solve):
    rng = random.Random(651)
    for _ in range(25):
        q, k, v, beta, g = _rand_case(rng)
        led = solve(q, k, v, beta, g)
        o, h = _oracle(_py(q), _py(k), _py(v), _py(beta), _py(g))
        assert _close(_py(led["o"]), o)
        assert _close(_py(led["final_state"]), h)


def check_default_scale(solve):
    q, k, v, beta, g = _rand_case(random.Random(1651), T=3)
    led = solve(q, k, v, beta, g)
    K = q.shape[-1]
    led2 = solve(q, k, v, beta, g, scale=1.0 / math.sqrt(K))
    assert torch.allclose(led["o"], led2["o"], atol=1e-6)
    # 显式 scale=1 与默认不同
    led3 = solve(q, k, v, beta, g, scale=1.0)
    assert not torch.allclose(led["o"], led3["o"], atol=1e-3)


def check_zero_gate_identity(solve):
    # g=0（无衰减）且 beta=0（无更新）时：h 不变，o = q*scale·h0
    B, T, H, K, V = 1, 4, 2, 3, 2
    gq = torch.Generator().manual_seed(2651)
    q = torch.randn(B, T, H, K, generator=gq)
    k = torch.randn(B, T, H, K, generator=gq)
    v = torch.randn(B, T, H, V, generator=gq)
    beta = torch.zeros(B, T, H)
    g = torch.zeros(B, T, H)
    h0 = torch.randn(B, H, K, V, generator=gq)
    led = solve(q, k, v, beta, g, initial_state=h0)
    # o 全程 = q*scale·h0
    for t in range(T):
        for hh in range(H):
            expected = torch.einsum('k,kv->v', q[0, t, hh] / math.sqrt(K), h0[0, hh])
            assert torch.allclose(led["o"][0, t, hh], expected, atol=1e-5), (t, hh)
    assert torch.allclose(led["final_state"], h0, atol=1e-5)


def check_delta_rule_corrects(solve):
    # beta=1、g=-inf 近似（取 g 很负）时：h 一步后被 v 方向替换；用 k 与 v 正交的构造验证
    # 更直接：单头单步 K=V=1 标量账手算
    q = torch.tensor([[[[2.0]]]])       # B=1,T=1,H=1,K=1
    k = torch.tensor([[[[1.0]]]])
    v = torch.tensor([[[[3.0]]]])
    beta = torch.tensor([[[0.5]]])
    g = torch.tensor([[[-0.1]]])
    h0 = torch.tensor([[[[4.0]]]])      # K=1, V=1
    led = solve(q, k, v, beta, g, scale=1.0, initial_state=h0)
    # h = 4*exp(-0.1) = 3.6398;  v' = (3 - 3.6398*1)*0.5 = -0.3199;  h = 3.6398 + 1*(-0.3199) = 3.3199
    # o = q*h = 2 * 3.3199 = 6.6398
    exp_h = 4.0 * math.exp(-0.1)
    exp_h2 = exp_h + (3.0 - exp_h) * 0.5
    assert abs(_py(led["final_state"])[0][0][0][0] - exp_h2) < 1e-5
    assert abs(_py(led["o"])[0][0][0][0] - 2.0 * exp_h2) < 1e-5


def check_initial_state_used(solve):
    q, k, v, beta, g = _rand_case(random.Random(3651), T=2)
    led0 = solve(q, k, v, beta, g)
    h0 = torch.randn(q.shape[0], q.shape[2], q.shape[3], v.shape[-1])
    led1 = solve(q, k, v, beta, g, initial_state=h0)
    assert not torch.allclose(led0["o"], led1["o"], atol=1e-4)
    o, h = _oracle(_py(q), _py(k), _py(v), _py(beta), _py(g), initial_state=_py(h0))
    assert _close(_py(led1["final_state"]), h)


def check_gating_order(solve):
    # 衰减必须在 delta 修正「之前」作用于 h：g 很负时 h≈0，delta 项只剩 beta*v
    B, T, H, K, V = 1, 1, 1, 2, 2
    q = torch.zeros(B, T, H, K)
    k = torch.tensor([[[[1.0, 0.0]]]])
    v = torch.tensor([[[[5.0, 7.0]]]])
    beta = torch.tensor([[[1.0]]])
    g = torch.full((B, T, H), -50.0)  # exp(g)≈0
    h0 = torch.randn(B, H, K, V)
    led = solve(q, k, v, beta, g, scale=1.0, initial_state=h0)
    # h ≈ 0 + k⊗v -> [[5,7],[0,0]]
    fs = _py(led["final_state"])
    assert abs(fs[0][0][0][0] - 5.0) < 1e-4
    assert abs(fs[0][0][0][1] - 7.0) < 1e-4
    assert abs(fs[0][0][1][0]) < 1e-4


def check_shape_and_dtype(solve):
    q, k, v, beta, g = _rand_case(random.Random(4651), T=3)
    led = solve(q, k, v, beta, g)
    assert led["o"].shape == tuple(v.shape)
    assert led["final_state"].shape == (q.shape[0], q.shape[2], q.shape[3], v.shape[-1])
    assert led["o"].dtype == torch.float32
    qd = q.to(torch.float64)
    led2 = solve(qd, k.to(torch.float64), v.to(torch.float64), beta.to(torch.float64), g.to(torch.float64))
    assert led2["o"].dtype == torch.float32  # 上游 float32 化计算


def check_noncontiguous(solve):
    q, k, v, beta, g = _rand_case(random.Random(5651), T=4)
    big = torch.randn(2, 8, q.shape[2], q.shape[3])
    q_nc = big[:, ::4][:, :q.shape[1]]
    assert not q_nc.is_contiguous()
    led = solve(q_nc, k, v, beta, g)
    led_ref = solve(q_nc.contiguous(), k, v, beta, g)
    assert torch.allclose(led["o"], led_ref["o"], atol=1e-6)


def check_invalid_shapes(solve):
    gq = torch.Generator().manual_seed(6651)
    q = torch.randn(1, 2, 1, 3, generator=gq)
    k = torch.randn(1, 2, 1, 3, generator=gq)
    v = torch.randn(1, 2, 1, 2, generator=gq)
    beta = torch.rand(1, 2, 1, generator=gq)
    g = -torch.rand(1, 2, 1, generator=gq)
    k_bad = k[..., :2]  # K=2 vs q 的 K=3：广播与 einsum 都无法对齐
    try:
        solve(q, k_bad, v, beta, g)
    except (RuntimeError, IndexError, ValueError):
        pass
    else:
        raise AssertionError("expected an error for mismatched K")
