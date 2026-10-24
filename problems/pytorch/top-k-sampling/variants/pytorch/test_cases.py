import torch


def _oracle(logits, k):
    """独立重算：按 logit 降序取前 k（并列按原索引升序），归一化。与 solution 的
    阈值法走不同代码路径。"""
    probs = torch.softmax(logits, dim=-1)
    kk = max(1, min(int(k), logits.shape[-1]))
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    keep = order[..., :kk]
    mask = torch.zeros_like(probs).scatter_(-1, keep, 1.0)
    f = probs * mask
    return f / f.sum(dim=-1, keepdim=True)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    filtered, chosen = solve(logits, 2, torch.tensor(0.9))
    exp = torch.tensor([0.731059, 0.268941, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(filtered, exp, atol=1e-5), filtered.tolist()
    assert chosen.item() == 1, chosen.item()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for m in range(6, 15):
        logits = torch.randn(m)
        for k in (1, 3, m):
            f, _ = solve(logits, k, torch.tensor(0.3))
            assert torch.allclose(f, _oracle(logits, k), atol=1e-6), (m, k)


def check_sum_and_zeros(solve):
    # 每行和为 1；非 top-k 位置严格为 0；保留的恰是 k 个
    torch.manual_seed(1)
    logits = torch.randn(5, 12)
    f, c = solve(logits, 4, torch.rand(5))
    assert torch.allclose(f.sum(-1), torch.ones(5), atol=1e-6)
    assert int((f[0] > 0).sum()) == 4
    assert bool((f.gather(-1, c.view(-1, 1)) > 0).all())


def check_greedy_when_k1(solve):
    # k=1 退化为贪心 argmax
    torch.manual_seed(2)
    logits = torch.randn(3, 9)
    f, c = solve(logits, 1, torch.rand(3))
    assert torch.equal(c, logits.argmax(-1)), c.tolist()
    assert torch.allclose(f.max(-1).values, torch.ones(3), atol=1e-6)


def check_inverse_cdf(solve):
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])  # k=2 → [0.7311, 0.2689]
    _, lo = solve(logits, 2, torch.tensor(0.1))
    _, hi = solve(logits, 2, torch.tensor(0.9))
    assert lo.item() == 0 and hi.item() == 1, (lo.item(), hi.item())


check_example.example = True
