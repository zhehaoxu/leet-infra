import math

import torch


def _oracle(logits, eps, renorm, u):
    """独立重算：逐行纯 Python 阈值/两种 renorm 口径 + 朴素累加采样。"""
    probs = torch.softmax(logits, dim=-1)
    v = probs.shape[-1]
    flat = probs.reshape(-1, v)
    us = u.reshape(-1)
    n = flat.shape[0]
    out = torch.zeros_like(flat)
    chosen = []
    for r in range(n):
        p = [float(x) for x in flat[r]]
        thr = float(eps) * max(p)
        kept = [x if x >= thr else 0.0 for x in p]
        tot = math.fsum(kept)
        if bool(renorm):
            row = [x / tot for x in kept]
        else:
            row = list(kept)
            top = 0
            for i in range(1, v):
                if p[i] > p[top]:
                    top = i
            row[top] += 1.0 - tot
        uv = float(us[min(r, us.numel() - 1)])
        acc, c = 0.0, v - 1
        for i, x in enumerate(row):
            acc += x
            if acc >= uv:
                c = i
                break
        out[r] = torch.tensor(row)
        chosen.append(c)
    return out.reshape(probs.shape), torch.tensor(chosen).reshape(probs.shape[:-1])


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0])
    # eps=0.15 → thr=0.15*0.6027=0.0904 → 保留前 3 个；不 renorm 时把 0.0411 补给 argmax
    filtered, chosen = solve(logits, 0.15, False, torch.tensor(0.95))
    exp = torch.tensor([0.6438, 0.2217, 0.1345, 0.0, 0.0])
    assert torch.allclose(filtered, exp, atol=1e-4), filtered.tolist()
    assert chosen.item() == 2, chosen.item()

    filtered_r, chosen_r = solve(logits, 0.15, True, torch.tensor(0.95))
    exp_r = torch.tensor([0.6285, 0.2312, 0.1402, 0.0, 0.0])
    assert torch.allclose(filtered_r, exp_r, atol=1e-4), filtered_r.tolist()
    assert chosen_r.item() == 2, chosen_r.item()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(40):
        logits = torch.randn(4, 9)
        for renorm in (True, False):
            for eps in (0.05, 0.2, 0.5, 0.9):
                u = torch.rand(4)
                f, c = solve(logits, eps, renorm, u)
                ef, ec = _oracle(logits, eps, renorm, u)
                assert torch.allclose(f, ef, atol=1e-5, rtol=1e-5), (eps, renorm)
                assert torch.equal(c, ec), (eps, renorm, c, ec)


def check_zero_eps_is_identity(solve):
    # eps=0 → 阈值 0，全保留；两种口径都应等于原始 softmax
    torch.manual_seed(1)
    logits = torch.randn(6)
    probs = torch.softmax(logits, dim=-1)
    u = torch.tensor(0.4)
    for renorm in (True, False):
        f, _ = solve(logits, 0.0, renorm, u)
        assert torch.allclose(f, probs, atol=1e-6), (renorm, f.tolist())


def check_eps_one_is_one_hot(solve):
    # eps=1 → 只有 argmax 达到阈值 → 两种口径都是 one-hot
    torch.manual_seed(2)
    logits = torch.randn(5, 8)
    u = torch.rand(5)
    for renorm in (True, False):
        f, c = solve(logits, 1.0, renorm, u)
        assert torch.equal(c, logits.argmax(-1)), c
        assert torch.allclose(f.max(-1).values, torch.ones(5), atol=1e-6)
        assert torch.allclose(f.sum(-1), torch.ones(5), atol=1e-6)


def check_no_renorm_keeps_raw_mass(solve):
    # 不 renorm：非 argmax 的保留项概率**原样不动**，被砍质量全落到 argmax
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0])
    probs = torch.softmax(logits, dim=-1)
    f, _ = solve(logits, 0.15, False, torch.tensor(0.3))
    assert torch.allclose(f[1], probs[1], atol=1e-7), (f[1].item(), probs[1].item())
    assert torch.allclose(f[2], probs[2], atol=1e-7)
    assert torch.equal(f[3:], torch.zeros(2))
    assert f[0].item() > probs[0].item()
    assert abs(float(f.sum()) - 1.0) < 1e-6, f.sum().item()


def check_filtered_valid_and_chosen_kept(solve):
    torch.manual_seed(3)
    logits = torch.randn(3, 7)
    f, c = solve(logits, 0.3, True, torch.rand(3))
    assert torch.allclose(f.sum(-1), torch.ones(3), atol=1e-5)
    assert bool((f.gather(-1, c.view(-1, 1)).squeeze(-1) > 0).all())


check_example.example = True
