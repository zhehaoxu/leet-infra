import math

import torch


def _py(q, k, v, g, chunk_size, h0=None, scale=None):
    """独立纯 Python oracle：朴素递推（fla naive_recurrent_gla 口径）。"""
    T, K = len(q), len(q[0])
    V = len(v[0])
    if scale is None:
        scale = K ** -0.5
    if h0 is None:
        h = [[0.0] * V for _ in range(K)]
    else:
        h = [[float(h0[i][j]) for j in range(V)] for i in range(K)]
    o = []
    states = []
    for t in range(T):
        for i in range(K):
            decay = math.exp(float(g[t][i]))
            for j in range(V):
                h[i][j] = h[i][j] * decay + float(k[t][i]) * float(v[t][j])
        row = []
        for j in range(V):
            acc = 0.0
            for i in range(K):
                acc += float(q[t][i]) * scale * h[i][j]
            row.append(acc)
        o.append(row)
        if (t + 1) % chunk_size == 0 or t == T - 1:
            states.append([row[:] for row in h])
    return {"o": o, "final_state": [row[:] for row in h], "chunk_states": states}


def _close(actual, expect, atol=1e-8, rtol=1e-6):
    if torch.is_tensor(actual):
        actual = actual.tolist()

    def flat(x):
        if isinstance(x, list):
            out = []
            for it in x:
                out.extend(flat(it))
            return out
        return [x]

    fa, fe = flat(actual), flat(expect)
    assert len(fa) == len(fe), f"length mismatch {len(fa)} vs {len(fe)}"
    for av, ev in zip(fa, fe):
        assert abs(av - ev) <= atol + rtol * abs(ev), f"{av} vs {ev}"


def _compare(solve, q, k, v, g, chunk_size, h0=None, scale=None):
    led = solve(q, k, v, g, chunk_size, h0, scale)
    ref = _py(
        q.tolist(), k.tolist(), v.tolist(), g.tolist(), chunk_size,
        None if h0 is None else h0.tolist(), scale,
    )
    _close(led["o"], ref["o"])
    _close(led["final_state"], ref["final_state"])
    assert len(led["chunk_states"]) == len(ref["chunk_states"])
    for a, e in zip(led["chunk_states"], ref["chunk_states"]):
        _close(a, e)
    return led


def check_anchor_single_chunk(solve):
    # 手算锚点：T=2, K=1, V=1, 单 chunk, h0=None
    # g = [-a, -b] → S = [-a, -a-b]
    # o0 = q0·s · k0 v0 = 1·0.5 = 0.5  (scale=1, q0=1, k0=1, v0=0.5)
    # o1 = q1·s·[exp(S1-S0)·k0 v0 + exp(0)·k1 v1] = 1·[exp(-b)·0.5 + 2]
    a, b = 0.3, 0.7
    dt = dict(dtype=torch.float64)
    q = torch.tensor([[1.0], [1.0]], **dt)
    k = torch.tensor([[1.0], [1.0]], **dt)
    v = torch.tensor([[0.5], [2.0]], **dt)
    g = torch.tensor([[-a], [-b]], **dt)
    led = _compare(solve, q, k, v, g, 2, scale=1.0)
    assert abs(led["o"][0][0].item() - 0.5) < 1e-12
    assert abs(led["o"][1][0].item() - (math.exp(-b) * 0.5 + 2.0)) < 1e-12
    # 终态：h = exp(S1)·0 + exp(S1-S0)·k0v0 + k1v1 = exp(-b)·0.5 + 2
    assert abs(led["final_state"][0][0].item() - (math.exp(-b) * 0.5 + 2.0)) < 1e-12


def check_anchor_inter_chunk(solve):
    # 两 chunk：第二段 o1 含段间项 (q·exp(S))·h_prev
    # T=2, K=1, V=1, chunk=1：每段单步
    # t0: o0 = q0·k0v0 = 4；h1 = exp(g0)·0 + k0v0 = 2
    # t1: o1 = q1·[exp(g1)·h1 + k1 v1] = 1·[0.5·2 + 1] = 2
    dt = dict(dtype=torch.float64)
    q = torch.tensor([[2.0], [1.0]], **dt)
    k = torch.tensor([[1.0], [1.0]], **dt)
    v = torch.tensor([[2.0], [1.0]], **dt)
    g = torch.tensor([[-0.4], [-0.6931471805599453]], **dt)  # exp ≈ 0.5
    led = _compare(solve, q, k, v, g, 1, scale=1.0)
    exp_g1 = math.exp(-0.6931471805599453)
    assert abs(led["o"][0][0].item() - 4.0) < 1e-12
    assert abs(led["o"][1][0].item() - (exp_g1 * 2.0 + 1.0)) < 1e-12
    assert abs(led["final_state"][0][0].item() - (exp_g1 * 2.0 + 1.0)) < 1e-12
    assert abs(led["chunk_states"][0][0][0].item() - 2.0) < 1e-12


def check_h0_contribution(solve):
    # h0 非 None：t0 的段间项 = q0·scale·exp(g0)·h0
    dt = dict(dtype=torch.float64)
    q = torch.tensor([[1.0]], **dt)
    k = torch.tensor([[1.0]], **dt)
    v = torch.tensor([[1.0]], **dt)
    g = torch.tensor([[-0.5]], **dt)
    h0 = torch.tensor([[4.0]], **dt)
    led = _compare(solve, q, k, v, g, 4, h0=h0, scale=1.0)
    expect = math.exp(-0.5) * 4.0 + 1.0
    assert abs(led["o"][0][0].item() - expect) < 1e-12


def check_oracle_identity_random(solve):
    # chunk 并行形式 == 朴素递推（含不均匀尾 chunk）
    torch.manual_seed(23)
    T, K, V = 7, 3, 2
    q = torch.randn(T, K)
    k = torch.randn(T, K) * 0.7
    v = torch.randn(T, V)
    g = -torch.rand(T, K) * 1.5
    led = _compare(solve, q, k, v, g, 3)
    # chunk_states 数量 = ceil(7/3) = 3
    assert len(led["chunk_states"]) == 3


def check_chunk_size_variants(solve):
    torch.manual_seed(5)
    T, K, V = 6, 2, 2
    q = torch.randn(T, K)
    k = torch.randn(T, K) * 0.5
    v = torch.randn(T, V)
    g = -torch.rand(T, K)
    ref = _py(q.tolist(), k.tolist(), v.tolist(), g.tolist(), 6)
    for cs in (1, 2, 4, 6, 9):  # 9 > T：单 chunk
        led = solve(q, k, v, g, cs)
        _close(led["o"], ref["o"])
        _close(led["final_state"], ref["final_state"])


def check_default_scale(solve):
    # 缺省 scale = K^-0.5
    torch.manual_seed(9)
    T, K, V = 5, 4, 1
    q = torch.randn(T, K)
    k = torch.randn(T, K) * 0.5
    v = torch.randn(T, V)
    g = -torch.rand(T, K)
    led = _compare(solve, q, k, v, g, 2)  # _py 缺省同口径
    led2 = solve(q, k, v, g, 2, scale=1.0)
    ratio = led["o"][0][0].item() / led2["o"][0][0].item()
    assert abs(ratio - K ** -0.5) < 1e-9


def check_invalid_inputs(solve):
    q = torch.randn(3, 2)
    k = torch.randn(3, 2)
    v = torch.randn(3, 2)
    g = -torch.rand(3, 2)
    bad = [
        lambda: solve(q, k, torch.randn(4, 2), g, 2),
        lambda: solve(q, k, v, g[:2], 2),
        lambda: solve(q, k, v, g, 0),
        lambda: solve(q, k, v, g, -1),
        lambda: solve(q, k, v, g, 2.5),
        lambda: solve(q, k, v, g, True),
        lambda: solve(torch.randn(3), k, v, g, 2),
        lambda: solve(q, k, v, g, 2, h0=torch.randn(3, 2)),
        lambda: solve(q, k, v, g, 2, scale=-0.1),
        lambda: solve(torch.zeros(0, 2), torch.zeros(0, 2), torch.zeros(0, 2),
                      torch.zeros(0, 2), 2),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
