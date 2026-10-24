import math

import torch


def _oracle(logits, p):
    """独立重算：走 logit 域阈值 logit_max + ln(p)（与 solution 的概率域不同路径）。"""
    probs = torch.softmax(logits, dim=-1)
    thr = logits.max(dim=-1, keepdim=True).values + math.log(float(p))
    keep = (logits >= thr).to(probs.dtype)
    f = probs * keep
    return f / f.sum(dim=-1, keepdim=True)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    filtered, chosen = solve(logits, 0.2, torch.tensor(0.9))
    exp = torch.tensor([0.628532, 0.231224, 0.140244, 0.0, 0.0, 0.0])
    assert torch.allclose(filtered, exp, atol=1e-5), filtered.tolist()
    assert chosen.item() == 2, chosen.item()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for p in (0.05, 0.2, 0.5):
        for _ in range(30):
            logits = torch.randn(torch.randint(6, 16, (1,)).item())
            f, _ = solve(logits, p, torch.tensor(0.3))
            assert torch.allclose(f, _oracle(logits, p), atol=1e-6), p


def check_keeps_top(solve):
    # 最高概率者永远保留；每行和为 1
    torch.manual_seed(1)
    logits = torch.randn(4, 11)
    f, c = solve(logits, 0.3, torch.rand(4))
    assert torch.allclose(f.sum(-1), torch.ones(4), atol=1e-6)
    amax = logits.argmax(-1)
    assert bool((f.gather(-1, amax.view(-1, 1)) > 0).all()), "argmax 必被保留"


def check_high_p_approaches_greedy(solve):
    # p 接近 1 时几乎只剩最高概率 token
    torch.manual_seed(2)
    logits = torch.randn(3, 10)
    f, c = solve(logits, 0.99, torch.rand(3))
    assert torch.equal(c, logits.argmax(-1)), c.tolist()


def check_inverse_cdf(solve):
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])  # filtered 前 3 项
    _, lo = solve(logits, 0.2, torch.tensor(0.1))
    _, hi = solve(logits, 0.2, torch.tensor(0.97))
    assert lo.item() == 0 and hi.item() == 2, (lo.item(), hi.item())


check_example.example = True
