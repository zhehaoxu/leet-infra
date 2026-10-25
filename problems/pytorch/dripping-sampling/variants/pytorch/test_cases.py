import math

import torch


def _oracle(logits, mode, drip_amount, min_keep, u):
    """独立重算：逐行纯 Python 求均值/总体标准差与头部剔除，与 solution 的张量路径不同。"""
    probs = torch.softmax(logits, dim=-1)
    v = probs.shape[-1]
    flat = probs.reshape(-1, v)
    us = u.reshape(-1)
    n = flat.shape[0]
    out = torch.zeros_like(flat)
    cut_counts = torch.zeros(n, dtype=torch.long)
    chosen = []
    for r in range(n):
        p = [float(x) for x in flat[r]]
        valid = [x for x in p if x > 1e-35]
        nv = max(len(valid), 1)
        mean = sum(valid) / nv
        var = sum(x * x for x in valid) / nv - mean * mean
        std = math.sqrt(max(var, 0.0))
        if str(mode) == "mean":
            thr = float(drip_amount) * mean
        else:
            thr = mean + float(drip_amount) * std
        cut = [x > thr for x in p]
        n_cut = sum(cut)
        fired = n_cut > 0 and (v - n_cut) >= int(min_keep)
        if not fired:
            cut = [False] * v
        keep = [0.0 if c else 1.0 for c in cut]
        tot = math.fsum(p[i] * keep[i] for i in range(v))
        row = [p[i] * keep[i] / tot for i in range(v)]
        uv = float(us[min(r, us.numel() - 1)])
        acc, c = 0.0, v - 1
        for i, x in enumerate(row):
            acc += x
            if acc >= uv:
                c = i
                break
        out[r] = torch.tensor(row)
        cut_counts[r] = sum(1 for x in cut if x)
        chosen.append(c)
    return (out.reshape(probs.shape),
            torch.tensor(chosen).reshape(probs.shape[:-1]),
            cut_counts.reshape(probs.shape[:-1]))


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5])
    filtered, chosen, n_cut = solve(logits, "std", 1.0, 1, torch.tensor(0.55))
    exp = torch.tensor([0.0, 0.4287, 0.2600, 0.1577, 0.0956, 0.0580])
    assert torch.allclose(filtered, exp, atol=1e-4), filtered.tolist()
    assert chosen.item() == 2, chosen.item()
    assert n_cut.item() == 1, n_cut.item()

    f_mean, _, n_mean = solve(logits, "mean", 1.0, 1, torch.tensor(0.55))
    assert n_mean.item() == 2, (n_mean.item(), f_mean.tolist())


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(25):
        logits = torch.randn(4, 11)
        for mode in ("mean", "std"):
            for amount in (0.2, 0.5, 1.0, 3.0):
                for min_keep in (1, 5, 9):
                    u = torch.rand(4)
                    got = solve(logits, mode, amount, min_keep, u)
                    exp = _oracle(logits, mode, amount, min_keep, u)
                    assert torch.allclose(got[0], exp[0], atol=1e-5, rtol=1e-5), (mode, amount, min_keep)
                    assert torch.equal(got[1], exp[1]), (mode, amount, min_keep)
                    assert torch.equal(got[2], exp[2]), (mode, amount, min_keep)


def check_head_not_tail(solve):
    # 与 min-p 相反：被清零的是头部（最高概率项），尾部质量保留
    logits = torch.tensor([6.0, 1.0, 0.8, 0.6, 0.4])
    f, _, n_cut = solve(logits, "mean", 0.5, 1, torch.tensor(0.9))
    assert n_cut.item() >= 1, n_cut.item()
    assert f[0].item() == 0.0, f.tolist()
    assert bool((f[2:] > 0).all()), f.tolist()


def check_uniform_is_noop(solve):
    # 完全平坦：没有任何项「高于均值」→ 不触发，filtered == 原始 softmax
    logits = torch.zeros(5, 8)
    probs = torch.softmax(logits, dim=-1)
    u = torch.rand(5)
    for mode in ("mean", "std"):
        f, _, n_cut = solve(logits, mode, 1.0, 1, u)
        assert torch.equal(n_cut, torch.zeros(5, dtype=torch.long)), (mode, n_cut)
        assert torch.allclose(f, probs, atol=1e-6), (mode, f.tolist())


def check_min_keep_rescue(solve):
    # 存活数不足 min_keep → 整行不砍（兜底优先于「砍头」）
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5])
    f, c, n_cut = solve(logits, "mean", 0.1, 5, torch.tensor(0.5))
    assert n_cut.item() == 0, n_cut.item()
    assert torch.allclose(f, torch.softmax(logits, dim=-1), atol=1e-6)
    assert torch.allclose(f.sum(), torch.tensor(1.0), atol=1e-6)


def check_larger_amount_cuts_less(solve):
    torch.manual_seed(3)
    logits = torch.randn(7)
    u = torch.tensor(0.5)
    _, _, loose = solve(logits, "std", 0.2, 1, u)
    _, _, tight = solve(logits, "std", 2.0, 1, u)
    assert int(loose) >= int(tight), (int(loose), int(tight))


def check_row_sums_to_one(solve):
    torch.manual_seed(4)
    logits = torch.randn(3, 10)
    f, c = solve(logits, "std", 0.5, 1, torch.rand(3))[:2]
    assert torch.allclose(f.sum(-1), torch.ones(3), atol=1e-5)
    assert bool((f.gather(-1, c.view(-1, 1)).squeeze(-1) > 0).all())


check_example.example = True
