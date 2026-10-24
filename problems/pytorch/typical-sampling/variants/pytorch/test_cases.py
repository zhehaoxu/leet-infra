import math

import torch

_TINY = 1e-30


def _oracle(logits, tau):
    """独立重算：逐行用 python 排序 + 循环累加，与 solution 的向量化 gather 路径不同。"""
    probs = torch.softmax(logits, dim=-1)
    lp = probs.clamp_min(_TINY).log()
    H = -(probs * lp).sum(dim=-1, keepdim=True)
    d = (-lp - H).abs()
    n = probs.shape[-1]
    out = torch.zeros_like(probs)
    rows = probs.reshape(-1, n)
    ds = d.reshape(-1, n)
    os_ = out.reshape(-1, n)
    for r in range(rows.shape[0]):
        order = sorted(range(n), key=lambda i: (float(ds[r, i]), i))
        c = 0.0
        for i in order:
            c += float(rows[r, i])
            os_[r, i] = 1.0
            if c > tau:
                break
    keep = out
    f = probs * keep
    return f / f.sum(dim=-1, keepdim=True)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([3.0, 2.95, -5, -5, -5, -5, -5, -5, -5, -5])
    filtered, chosen = solve(logits, 0.4, torch.tensor(0.3))
    exp = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(filtered, exp, atol=1e-6), filtered.tolist()
    assert chosen.item() == 1, chosen.item()


def check_drops_argmax(solve):
    # 标志行为：典型采样把概率最高的 argmax（索引0）丢弃
    logits = torch.tensor([3.0, 2.95, -5, -5, -5, -5, -5, -5, -5, -5])
    f, _ = solve(logits, 0.4, torch.tensor(0.3))
    assert float(f[0]) == 0.0 and float(f[1]) == 1.0, f.tolist()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for tau in (0.3, 0.6, 0.9):
        for _ in range(30):
            logits = torch.randn(torch.randint(6, 16, (1,)).item())
            f, _ = solve(logits, tau, torch.tensor(0.3))
            assert torch.allclose(f, _oracle(logits, tau), atol=1e-6), tau


def check_sum_and_in_set(solve):
    torch.manual_seed(1)
    logits = torch.randn(5, 12)
    f, c = solve(logits, 0.7, torch.rand(5))
    assert torch.allclose(f.sum(-1), torch.ones(5), atol=1e-6)
    assert bool((f.gather(-1, c.view(-1, 1)) > 0).all())
    assert int((f[0] > 0).sum()) >= 1


def check_tau_near_one_keeps_most(solve):
    # tau→1 基本不截断
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    f, _ = solve(logits, 0.99, torch.tensor(0.3))
    assert int((f > 0).sum()) >= 3, f.tolist()


check_example.example = True
