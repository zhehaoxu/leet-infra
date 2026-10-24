import math

import torch

_TINY = 1e-30


def _oracle(logits, tau, eta, mu):
    """独立重算：逐行 python 排序截断 + 归一化 + 原顺序逆变换 + mu 更新。"""
    probs = torch.softmax(logits, dim=-1)
    n = probs.shape[-1]
    flat = probs.reshape(-1, n)
    out_f = torch.zeros_like(flat)
    out_mu = []
    for r in range(flat.shape[0]):
        p = [float(x) for x in flat[r]]
        order = sorted(range(n), key=lambda i: (-p[i], i))  # 概率降序，并列按原索引升序
        size = n
        for j, i in enumerate(order):
            if -math.log2(max(p[i], _TINY)) > mu:
                size = j
                break
        size = max(size, 1)
        kept_idx = order[:size]
        kept_p = [p[i] for i in kept_idx]
        s = sum(kept_p)
        renorm = {i: v / s for i, v in zip(kept_idx, kept_p)}
        row = [renorm.get(i, 0.0) for i in range(n)]
        out_f[r] = torch.tensor(row)
        out_mu.append(mu)
    return out_f.reshape(probs.shape)


def _oracle_full(logits, tau, eta, mu, u):
    f = _oracle(logits, tau, eta, mu)
    # 原顺序逆变换采样
    uf = u.reshape(-1)
    ff = f.reshape(-1, f.shape[-1])
    chosen = []
    mu_new = []
    for r in range(ff.shape[0]):
        cdf = 0.0
        pick = ff.shape[-1] - 1
        for i in range(ff.shape[-1]):
            cdf += float(ff[r, i])
            if cdf >= float(uf[r]):
                pick = i
                break
        chosen.append(pick)
        obs = -math.log2(max(float(ff[r, pick]), _TINY))
        mu_new.append(mu - eta * (obs - tau))
    return f, torch.tensor(chosen).reshape(f.shape[:-1]), torch.tensor(mu_new).reshape(f.shape[:-1])


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    f, c, mu_new = solve(logits, 5.0, 0.1, 4.0, torch.tensor(0.5))
    exp = torch.tensor([0.628532, 0.231224, 0.140244, 0.0, 0.0, 0.0])
    assert torch.allclose(f, exp, atol=1e-5), f.tolist()
    assert c.item() == 0, c.item()
    assert abs(float(mu_new) - 4.433006) < 1e-4, float(mu_new)


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for mu in (1.0, 3.0, 6.0, 12.0):
        for _ in range(20):
            logits = torch.randn(torch.randint(4, 16, (1,)).item())
            f, c, mn = solve(logits, 5.0, 0.1, mu, torch.tensor(0.5))
            of, oc, omn = _oracle_full(logits, 5.0, 0.1, mu, torch.tensor(0.5))
            assert torch.allclose(f, of, atol=1e-6), (mu, f.tolist(), of.tolist())
            assert c.item() == oc.item(), (mu, c.item(), oc.item())
            assert abs(float(mn) - float(omn)) < 1e-5, (mu, float(mn), float(omn))


def check_sum_and_in_set(solve):
    torch.manual_seed(1)
    logits = torch.randn(5, 12)
    f, c, mn = solve(logits, 5.0, 0.1, 4.0, torch.rand(5))
    assert torch.allclose(f.sum(-1), torch.ones(5), atol=1e-6)
    assert bool((f.gather(-1, c.view(-1, 1)) > 0).all())
    assert mn.shape == (5,), mn.shape


def check_large_mu_keeps_all(solve):
    # mu 很大 → 无 token 越界 → 保留全部 → filtered == softmax
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    f, _, _ = solve(logits, 5.0, 0.1, 50.0, torch.tensor(0.5))
    assert torch.allclose(f, torch.softmax(logits, dim=-1), atol=1e-6), f.tolist()


def check_mu_feedback_direction(solve):
    # obs < tau（抽中高概率 token）→ mu_new > mu（放宽）
    logits = torch.tensor([5.0, 0.0, 0.0, 0.0])
    _, c, mn = solve(logits, 3.0, 0.1, 2.0, torch.tensor(0.1))
    assert c.item() == 0
    assert float(mn) > 2.0, float(mn)


check_example.example = True
