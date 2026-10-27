import torch
import math

PK = torch.tensor([[2.0]])                                # (1,1)
PV = torch.tensor([[3.0]])
K = torch.tensor([[5.0]])
V = torch.tensor([[7.0]])
Q = torch.tensor([[1.0]])


def _oracle(P_k, P_v, K, V, Q):
    """独立重算：纯 python 拼接 + 逐标量 softmax 点积。"""
    L, d = P_k.shape
    S = K.shape[0]
    K2 = [[float(P_k[i, j]) for j in range(d)] for i in range(L)] + \
         [[float(K[i, j]) for j in range(d)] for i in range(S)]
    V2 = [[float(P_v[i, j]) for j in range(d)] for i in range(L)] + \
         [[float(V[i, j]) for j in range(d)] for i in range(S)]
    nq = Q.shape[0]
    out = []
    for q in range(nq):
        logits = [sum(float(Q[q, j]) * K2[t][j] for j in range(d)) / math.sqrt(d)
                  for t in range(L + S)]
        m = max(logits)
        ex = [math.exp(v - m) for v in logits]
        z = sum(ex)
        w = [e / z for e in ex]
        out.append([sum(w[t] * V2[t][j] for t in range(L + S)) for j in range(d)])
    h = d
    return (torch.tensor(out, dtype=torch.float64).to(torch.float32),
            int(2 * L * d),
            int(L * d + (d * h + h) + (h * (2 * L * d) + 2 * L * d)))


def check_example(solve):
    attn, n_direct, n_reparam = solve(PK, PV, K, V, Q)
    e2, e5 = math.exp(2.0), math.exp(5.0)
    exp_attn = (e2 * 3.0 + e5 * 7.0) / (e2 + e5)
    assert torch.allclose(attn, torch.tensor([[exp_attn]]), atol=1e-5, rtol=1e-5), (
        f"got {attn.tolist()}")
    assert n_direct == 2, f"n_direct 应为 2，got {n_direct}"
    assert n_reparam == 7, f"n_reparam 应为 7，got {n_reparam}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(51)
    d, L, S, nq = 4, 2, 5, 3
    pk, pv = torch.randn(L, d), torch.randn(L, d)
    k, v = torch.randn(S, d), torch.randn(S, d)
    q = torch.randn(nq, d)
    attn, n_direct, n_reparam = solve(pk, pv, k, v, q)
    exp_attn, e_nd, e_nr = _oracle(pk, pv, k, v, q)
    assert torch.allclose(attn, exp_attn, atol=1e-5, rtol=1e-5), "attn 与独立重算不符"
    assert (n_direct, n_reparam) == (e_nd, e_nr), "参数量与独立重算不符"


def check_prefix_zero_degenerates(solve):
    """L=0：虚拟前缀为空，应退化为标准 attention，且 n_direct=0。"""
    torch.manual_seed(61)
    d, S, nq = 4, 6, 2
    k, v = torch.randn(S, d), torch.randn(S, d)
    q = torch.randn(nq, d)
    attn, n_direct, n_reparam = solve(torch.zeros(0, d), torch.zeros(0, d), k, v, q)
    std = torch.softmax(q.to(torch.float64) @ k.to(torch.float64).T / math.sqrt(d), dim=-1) \
        @ v.to(torch.float64)
    assert torch.allclose(attn, std.to(torch.float32), atol=1e-5, rtol=1e-5), (
        "L=0 应等于标准注意力")
    assert n_direct == 0, f"n_direct 应为 0，got {n_direct}"
    assert n_reparam == d * d + d, f"n_reparam 应为 {d*d+d}，got {n_reparam}"


def check_param_counts(solve):
    torch.manual_seed(71)
    for L, d, S in ((3, 2, 4), (5, 8, 2)):
        pk, pv = torch.randn(L, d), torch.randn(L, d)
        k, v = torch.randn(S, d), torch.randn(S, d)
        q = torch.randn(2, d)
        _, n_direct, n_reparam = solve(pk, pv, k, v, q)
        assert n_direct == 2 * L * d, f"n_direct 应为 {2*L*d}，got {n_direct}"
        exp_nr = L * d + (d * d + d) + (d * (2 * L * d) + 2 * L * d)
        assert n_reparam == exp_nr, f"n_reparam 应为 {exp_nr}，got {n_reparam}"


def check_prefix_dominates(solve):
    """Q 与 P_k 强对齐、与 K 正交时输出逼近 P_v：虚拟 KV 是可学习的软提示 KV。"""
    pk = torch.tensor([[20.0, 0.0]])
    pv = torch.tensor([[100.0, 100.0]])
    k = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    v = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    q = torch.tensor([[1.0, 0.0]])
    attn, _, _ = solve(pk, pv, k, v, q)
    assert torch.allclose(attn, pv, atol=1e-4, rtol=1e-4), f"got {attn.tolist()}"
