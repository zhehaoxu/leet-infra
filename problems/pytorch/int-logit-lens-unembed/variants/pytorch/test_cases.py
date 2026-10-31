import math

import torch


def _py(hiddens, lnf_w, lnf_b, unembed, apply_norm=True, eps=1e-5):
    """独立纯 Python oracle。"""
    L, D = len(hiddens), len(hiddens[0])
    V = len(unembed[0])
    logits_rows = []
    for l in range(L):
        h = [float(x) for x in hiddens[l]]
        if apply_norm:
            mu = sum(h) / D
            var = sum((x - mu) ** 2 for x in h) / D
            n = [
                (h[d] - mu) / math.sqrt(var + eps) * float(lnf_w[d]) + float(lnf_b[d])
                for d in range(D)
            ]
        else:
            n = h
        row = [sum(n[d] * float(unembed[d][v]) for d in range(D)) for v in range(V)]
        logits_rows.append(row)
    argmax = []
    for row in logits_rows:
        mx = max(row)
        argmax.append(min(i for i, v in enumerate(row) if v == mx))
    final_argmax = argmax[-1]
    agree = [a == final_argmax for a in argmax]
    rank = [1 + sum(1 for v in row if v > row[final_argmax]) for row in logits_rows]
    top = [max(row) for row in logits_rows]
    return {
        "logits": logits_rows,
        "argmax": argmax,
        "final_argmax": final_argmax,
        "agree": agree,
        "rank_of_final": rank,
        "top_logit": top,
    }


def _close(actual, expect, atol=1e-9, rtol=1e-9):
    if torch.is_tensor(actual):
        actual = actual.tolist()
    if isinstance(expect, torch.Tensor):
        expect = expect.tolist()

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


def _compare(solve, hiddens, lnf_w, lnf_b, unembed, apply_norm=True, eps=1e-5):
    led = solve(hiddens, lnf_w, lnf_b, unembed, apply_norm, eps)
    ref = _py(hiddens.tolist(), lnf_w.tolist(), lnf_b.tolist(), unembed.tolist(),
              apply_norm, eps)
    _close(led["logits"], ref["logits"])
    assert led["argmax"] == ref["argmax"]
    assert led["final_argmax"] == ref["final_argmax"]
    assert led["agree"] == ref["agree"]
    assert led["rank_of_final"] == ref["rank_of_final"]
    _close(led["top_logit"], ref["top_logit"])
    return led


def check_anchor_layernorm(solve):
    # h=[4,0]：mean=2、有偏 var=4 → normed=[1,-1]；unembed 取 D 前两列 → logits=[1,-1,0]
    hiddens = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    lnf_w = torch.tensor([1.0, 1.0])
    lnf_b = torch.tensor([0.0, 0.0])
    unembed = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    led = _compare(solve, hiddens, lnf_w, lnf_b, unembed)
    e = 2.0 / math.sqrt(4.0 + 1e-5)  # normed = ±2/sqrt(var+eps)
    assert abs(led["logits"][0][0] - e) < 1e-9
    assert abs(led["logits"][0][1] - (-e)) < 1e-9
    assert abs(led["logits"][0][2]) < 1e-12
    assert led["argmax"] == [0, 1]
    assert led["final_argmax"] == 1
    assert led["agree"] == [False, True]
    # 层 0 中 token1 的 logit=-e，严格大于它的有 2 个 → rank 3
    assert led["rank_of_final"] == [3, 1]
    assert abs(led["top_logit"][0] - e) < 1e-9 and abs(led["top_logit"][1] - e) < 1e-9


def check_norm_vs_raw(solve):
    # LayerNorm 是读出的一部分：对 normed h 敏感
    hiddens = torch.tensor([[4.0, 0.0]])
    lnf_w = torch.tensor([1.0, 3.0])
    lnf_b = torch.tensor([0.5, -0.5])
    unembed = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    led = _compare(solve, hiddens, lnf_w, lnf_b, unembed)
    raw = solve(hiddens, lnf_w, lnf_b, unembed, apply_norm=False)
    assert led["logits"][0] != raw["logits"][0]
    assert raw["argmax"] == [int((torch.tensor([4.0, 0.0]).double() @ unembed.double()).argmax())]


def check_tie_smallest_index(solve):
    # 常数隐态：normed = lnf_b（var=0 走 eps）；logits 全并列时取最小下标
    hiddens = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    lnf_w = torch.tensor([0.0, 0.0])
    lnf_b = torch.tensor([0.0, 0.0])
    unembed = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    led = solve(hiddens, lnf_w, lnf_b, unembed)
    assert led["logits"][0] == [0.0, 0.0, 0.0]
    assert led["argmax"] == [0, 0]
    assert led["rank_of_final"] == [1, 1]
    # 部分并列：最大值并列两处
    unembed2 = torch.tensor([[2.0, 2.0, 1.0], [1.0, 1.0, 0.0]])
    led2 = solve(torch.tensor([[3.0, 0.0]]), torch.tensor([1.0, 1.0]),
                 torch.tensor([0.0, 0.0]), unembed2)
    # normed=[1,-1] → logits=[1,1,1] 全并列 → 0
    assert led2["argmax"] == [0]


def check_rank_and_agreement(solve):
    torch.manual_seed(19)
    L, D, V = 5, 6, 9
    hiddens = torch.randn(L, D)
    lnf_w = torch.rand(D) + 0.5
    lnf_b = torch.randn(D) * 0.1
    unembed = torch.randn(D, V)
    led = _compare(solve, hiddens, lnf_w, lnf_b, unembed)
    assert led["rank_of_final"][-1] == 1
    assert led["agree"][-1] is True
    # rank 与 argmax 自洽：rank==1 当且仅当该层 argmax == final_argmax
    for l in range(L):
        assert (led["rank_of_final"][l] == 1) == led["agree"][l]


def check_affine_effect(solve):
    # w/b 改变读出：b 直接加到 logits 的对应线性组合
    hiddens = torch.tensor([[1.0, -1.0]])
    lnf_w = torch.tensor([1.0, 1.0])
    unembed = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    led0 = solve(hiddens, lnf_w, torch.tensor([0.0, 0.0]), unembed)
    led1 = solve(hiddens, lnf_w, torch.tensor([5.0, 0.0]), unembed)
    # normed=[1,-1]/sqrt(1+1e-5)：logits0 = [e,-e]；b[0]=5 → [e+5,-e]
    e = 1.0 / math.sqrt(1.0 + 1e-5)
    assert abs(led0["logits"][0][0] - e) < 1e-9
    assert abs(led1["logits"][0][0] - (e + 5.0)) < 1e-9


def check_eps_effect(solve):
    # 常数隐态：var=0，读出只由 eps 与 b 决定
    h = torch.tensor([[3.0, 3.0]])
    w = torch.tensor([1.0, 1.0])
    b = torch.tensor([0.0, 0.0])
    u = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    led = _compare(solve, h, w, b, u, eps=1e-5)
    assert abs(led["logits"][0][0]) < 1e-9  # normed = 0/sqrt(eps) = 0
    led2 = _compare(solve, h, w, b, u, eps=0.25)
    assert abs(led2["logits"][0][0]) < 1e-12


def check_invalid_inputs(solve):
    h = torch.randn(2, 4)
    w = torch.randn(4)
    b = torch.randn(4)
    u = torch.randn(4, 7)
    bad = [
        lambda: solve(torch.randn(2, 3), w, b, u),
        lambda: solve(torch.randn(4, 2), w, b, u),
        lambda: solve(h, torch.randn(5), b, u),
        lambda: solve(h, w, torch.randn(4, 1), u),
        lambda: solve(h, w, b, torch.randn(5, 7)),
        lambda: solve(h, w, b, u, apply_norm=1),
        lambda: solve(h, w, b, u, eps=0.0),
        lambda: solve(h, w, b, u, eps=-1e-5),
        lambda: solve(h, w, b, u, eps=True),
        lambda: solve(torch.zeros(0, 4), w, b, u),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
