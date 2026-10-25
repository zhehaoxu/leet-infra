import math

import torch


def _oracle(logits, bias_tokens, bias_values, banned, u):
    """独立重算：纯 Python 列表逐条累加 + math.exp 手工 softmax，不碰任何张量运算。"""
    lg = [float(x) for x in logits.reshape(-1).tolist()]
    v = len(lg)
    toks = [int(t) for t in bias_tokens.reshape(-1).tolist()]
    vals = [float(b) for b in bias_values.reshape(-1).tolist()]
    for tid, val in zip(toks, vals):
        if 0 <= tid < v:
            lg[tid] += val
    for tid in [int(t) for t in banned.reshape(-1).tolist()]:
        if 0 <= tid < v:
            lg[tid] = None                      # None 表示被强制掩码（-inf）
    mx = max(x for x in lg if x is not None)
    ex = [0.0 if x is None else math.exp(x - mx) for x in lg]
    tot = math.fsum(ex)
    p = [x / tot for x in ex]
    acc, c = 0.0, v - 1
    uv = float(u.reshape(-1)[0]) if u.dim() else float(u)
    for i, x in enumerate(p):
        acc += x
        if acc >= uv:
            c = i
            break
    return (torch.tensor([-math.inf if x is None else x for x in lg]),
            torch.tensor(p), torch.tensor(c, dtype=torch.long))


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([1.0, 0.5, 0.0, -0.5, -1.0])
    biased, filtered, chosen = solve(
        logits, torch.tensor([1, 1, 3]), torch.tensor([0.5, 1.5, -2.0]),
        torch.tensor([4]), torch.tensor(0.6))
    assert torch.allclose(biased[:4], torch.tensor([1.0, 2.5, 0.0, -2.5]), atol=1e-6)
    assert biased[4].item() == float("-inf")
    exp = torch.tensor([0.170075, 0.762222, 0.062567, 0.005136, 0.0])
    assert torch.allclose(filtered, exp, atol=1e-5, rtol=1e-5), filtered.tolist()
    assert chosen.item() == 1, chosen.item()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(40):
        logits = torch.randn(7)
        nb = torch.randint(0, 4, ()).item()
        # 含越界/负 id 的条目，验证「匹配不到就无副作用」
        bt = torch.randint(-2, 10, (max(nb, 1),)).float().long()
        bv = torch.randn(max(nb, 1)) * 2
        if nb == 0:
            bt = torch.tensor([0], dtype=torch.long)
            bv = torch.tensor([0.0])
        banned = torch.randperm(7)[:torch.randint(0, 3, ()).item()]
        u = torch.rand(1)
        got = solve(logits, bt, bv, banned, u)
        exp = _oracle(logits, bt, bv, banned, u)
        assert torch.allclose(got[0], exp[0], atol=1e-5, rtol=1e-5, equal_nan=True), bt.tolist()
        assert torch.allclose(got[1], exp[1], atol=1e-5, rtol=1e-5), bt.tolist()
        assert torch.equal(got[2], exp[2]), bt.tolist()


def check_duplicate_bias_accumulates(solve):
    # 同一 token 出现两次 → 偏置累加：[+0.5, +1.5] == +2.0
    logits = torch.zeros(4)
    a, _, _ = solve(logits, torch.tensor([2, 2]), torch.tensor([0.5, 1.5]),
                    torch.tensor([], dtype=torch.long), torch.tensor(0.5))
    b, _, _ = solve(logits, torch.tensor([2]), torch.tensor([2.0]),
                    torch.tensor([], dtype=torch.long), torch.tensor(0.5))
    assert torch.allclose(a, b, atol=1e-6), (a.tolist(), b.tolist())
    assert a[2].item() == 2.0 and a[0].item() == 0.0, a.tolist()


def check_banned_is_exact_zero(solve):
    # 被 ban 的项概率恰为 0，且 chosen 绝不会落在被 ban 的位置上
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0])
    _, filtered, _ = solve(logits, torch.tensor([], dtype=torch.long),
                           torch.tensor([]), torch.tensor([0]), torch.tensor(0.5))
    assert filtered[0].item() == 0.0, filtered.tolist()
    assert torch.allclose(filtered[1:], torch.tensor([0.665241, 0.244728, 0.090031]),
                          atol=1e-5, rtol=1e-5), filtered.tolist()
    picks = {solve(logits, torch.tensor([], dtype=torch.long), torch.tensor([]),
                   torch.tensor([0, 1]), torch.tensor(uu))[2].item() for uu in
             (1e-6, 0.3, 0.5, 0.9, 0.999999)}
    assert picks <= {2, 3}, picks     # u>0 时绝不会选中概率为 0 的禁令位


def check_out_of_range_entries_noop(solve):
    # 越界 / 负 id 的 bias 与 banned 条目都不能改变结果，也不能报错
    logits = torch.tensor([1.0, 0.0, -1.0])
    base = solve(logits, torch.tensor([], dtype=torch.long), torch.tensor([]),
                 torch.tensor([], dtype=torch.long), torch.tensor(0.2))
    got = solve(logits, torch.tensor([-1, 7, 999]), torch.tensor([5.0, -5.0, 100.0]),
                torch.tensor([-3, 42]), torch.tensor(0.2))
    assert torch.equal(got[0], base[0]) and torch.equal(got[1], base[1])
    assert torch.equal(got[2], base[2])


def check_empty_bias_identity(solve):
    # 无偏置无禁令 → biased_logits 就是原 logits，filtered 就是 softmax
    torch.manual_seed(3)
    logits = torch.randn(3, 6)
    u = torch.rand(3)
    biased, filtered, chosen = solve(logits, torch.tensor([], dtype=torch.long),
                                     torch.tensor([]), torch.tensor([], dtype=torch.long), u)
    assert torch.equal(biased, logits)
    assert torch.allclose(filtered, torch.softmax(logits, dim=-1), atol=1e-6)
    assert torch.allclose(filtered.sum(-1), torch.ones(3), atol=1e-6)
    assert chosen.shape == (3,) and bool((chosen >= 0).all()) and bool((chosen < 6).all())


check_example.example = True
