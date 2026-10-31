import math
import random

import torch


def _softmax(v, T):
    m = max(x / T for x in v)
    es = [math.exp(x / T - m) for x in v]
    s = sum(es)
    return [e / s for e in es]


def _oracle(hs, pw, pb, fl, T):
    L, B, D = len(hs), len(hs[0]), len(hs[0][0])
    V = len(fl[0])
    logits = []
    for l in range(L):
        rows = []
        for b in range(B):
            rows.append([sum(pw[l][v][d] * hs[l][b][d] for d in range(D)) + pb[l][v]
                         for v in range(V)])
        logits.append(rows)
    target = [max(range(V), key=lambda v: (fl[b][v], -v)) for b in range(B)]
    ce = []
    for l in range(L):
        row = []
        for b in range(B):
            lp = logits[l][b][target[b]] / T
            m = max(x / T for x in logits[l][b])
            lse = m + math.log(sum(math.exp(x / T - m) for x in logits[l][b]))
            row.append(-(lp - lse))
        ce.append(row)
    kl = []
    pf = [_softmax(fl[b], T) for b in range(B)]
    for l in range(L):
        row = []
        for b in range(B):
            pl = _softmax(logits[l][b], T)
            row.append(sum(pf[b][v] * (math.log(pf[b][v]) - math.log(pl[v]))
                           for v in range(V) if pf[b][v] > 0))
        kl.append(row)
    match = [[max(range(V), key=lambda v: (logits[l][b][v], -v)) == target[b] for b in range(B)]
             for l in range(L)]
    return logits, ce, kl, match


def _py(t):
    return t.detach().float().tolist()


def _close(a, b, tol=1e-5):
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


def _rand_case(rng, L=4, B=2, D=6, V=9):
    g = torch.Generator().manual_seed(rng.randrange(10**9))
    hs = torch.randn(L, B, D, generator=g) * 0.8
    pw = torch.randn(L, V, D, generator=g) * 0.5
    pb = torch.randn(L, V, generator=g) * 0.1
    fl = torch.randn(B, V, generator=g) * 2.0
    return hs, pw, pb, fl


def check_random(solve):
    rng = random.Random(653)
    for _ in range(20):
        hs, pw, pb, fl = _rand_case(rng)
        led = solve(hs, pw, pb, fl)
        logits, ce, kl, match = _oracle(_py(hs), _py(pw), _py(pb), _py(fl), 1.0)
        assert _close(_py(led["layer_logits"]), logits)
        assert _close(_py(led["layer_ce"]), ce)
        assert _close(_py(led["layer_kl"]), kl)
        assert _py(led["argmax_match"].int()) == [[int(m) for m in row] for row in match]


def check_anchor_hand(solve):
    # V=2, B=1, L=2 手算：probe0 给 logits [ln2, ln2]（均匀），final 命中 1
    T = 1.0
    hs = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])  # L=2,B=1,D=2
    pw = torch.tensor([[[1.0, 0.0], [1.0, 0.0]],
                       [[0.0, 1.0], [0.0, 1.0]]])     # 每层两行同向 -> 两词表 logit 相同
    pb = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    fl = torch.tensor([[-1.0, 3.0]])
    led = solve(hs, pw, pb, fl, T)
    ll = _py(led["layer_logits"])
    assert abs(ll[0][0][0] - 1.0) < 1e-9 and abs(ll[0][0][1] - 1.0) < 1e-9
    assert abs(ll[1][0][0] - 1.0) < 1e-9 and abs(ll[1][0][1] - 1.0) < 1e-9
    # 均匀分布对 target=1 的 CE = -ln(0.5) = ln2
    for l in range(2):
        assert abs(_py(led["layer_ce"])[l][0] - math.log(2.0)) < 1e-6
    # KL(p_final || uniform)：p_final = softmax([-1,3]) 非均匀，KL = ln V + sum p ln p
    z = math.exp(-1.0) + math.exp(3.0)
    p0, p1 = math.exp(-1.0) / z, math.exp(3.0) / z
    kl_exp = math.log(2.0) + p0 * math.log(p0) + p1 * math.log(p1)
    for l in range(2):
        assert abs(_py(led["layer_kl"])[l][0] - kl_exp) < 1e-6


def check_kl_zero_on_identity(solve):
    # probe = final LM head：最后一层隐态的 logits 与 final 完全相同 -> KL=0；
    # 更早的层用同一探针但隐态不同 -> KL>0（这正是 tuned lens 度量的「层间差」）
    g = torch.Generator().manual_seed(1653)
    D, V, B = 5, 7, 2
    W = torch.randn(V, D, generator=g)
    h = torch.randn(3, B, D, generator=g)
    fl = h[-1] @ W.T
    led = solve(h, W.unsqueeze(0).expand(3, V, D).contiguous(), torch.zeros(3, V), fl)
    kl = _py(led["layer_kl"])
    for b in range(B):
        assert abs(kl[-1][b]) < 1e-9
        for l in range(2):
            assert kl[l][b] > 1e-6
    # 最后一层 argmax 必一致
    assert all(bool(v) for v in _py(led["argmax_match"].int())[-1])


def check_temperature_effect(solve):
    # T 越大 KL 越小（分布被抹平）；T 变化必改变 KL 与 CE
    hs, pw, pb, fl = _rand_case(random.Random(2653), L=2)
    led1 = solve(hs, pw, pb, fl, temperature=1.0)
    led4 = solve(hs, pw, pb, fl, temperature=4.0)
    assert not torch.allclose(led1["layer_kl"], led4["layer_kl"], atol=1e-9)
    assert not torch.allclose(led1["layer_ce"], led4["layer_ce"], atol=1e-9)


def check_probe_rank(solve):
    # 探针秩 = min(V, D)（随机满秩）与秩亏（重复行）
    hs, pw, pb, fl = _rand_case(random.Random(3653), L=2, D=4, V=6)
    pw = pw.clone()
    pw[1, 3:] = pw[1, :3]  # 层 1 秩亏：后三行复制前三行
    led = solve(hs, pw, pb, fl)
    ranks = _py(led["probe_rank"])
    assert ranks[0] == 4  # min(6,4)
    assert ranks[1] == 3


def check_ce_monotone_target(solve):
    # CE 对 target 是手算可验的：单层两词表 logit [0, a] 的 CE = ln(1+e^-a)
    hs = torch.tensor([[[0.0, 0.0]]])
    pw = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    pb = torch.tensor([[0.0, 2.0]])
    fl = torch.tensor([[0.0, 2.0]])
    led = solve(hs, pw, pb, fl)
    a = 2.0
    expected = math.log(1 + math.exp(-a))
    assert abs(_py(led["layer_ce"])[0][0] - expected) < 1e-6


def check_noncontiguous(solve):
    hs, pw, pb, fl = _rand_case(random.Random(4653), L=3, B=2, D=6)
    big = torch.randn(6, 2, 12)
    nc = big[::2, :, ::2][:, :, :6]
    assert not nc.is_contiguous()
    led = solve(nc, pw, pb, fl)
    led_ref = solve(nc.contiguous(), pw, pb, fl)
    assert torch.allclose(led["layer_logits"], led_ref["layer_logits"], atol=1e-8)


def check_invalid_temperature(solve):
    hs, pw, pb, fl = _rand_case(random.Random(5653), L=1)
    for bad in (0.0, -1.0):
        try:
            solve(hs, pw, pb, fl, temperature=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for T={bad}")
