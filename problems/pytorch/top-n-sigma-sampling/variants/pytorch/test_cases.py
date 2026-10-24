import statistics

import torch


def _oracle(logits, n):
    """独立重算：逐行用 python statistics.pstdev 计算总体标准差 + 阈值筛，
    与 solution 的张量化 sub/mean/sqrt 路径不同。"""
    probs = torch.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    lflat = logits.reshape(-1, logits.shape[-1])
    out = torch.zeros_like(flat)
    for r in range(flat.shape[0]):
        z = [float(x) for x in lflat[r]]
        mx = max(z)
        sd = statistics.pstdev(z)
        cut = mx - n * sd
        keep = [1.0 if v >= cut else 0.0 for v in z]
        row = [flat[r, i].item() * keep[i] for i in range(len(z))]
        s = sum(row)
        out[r] = torch.tensor([v / s for v in row])
    return out.reshape(probs.shape)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([3.0, 2.5, 0.0, -0.2, -0.5, -4.0])
    filtered, chosen = solve(logits, 1.0, torch.tensor(0.8))
    exp = torch.tensor([0.622459, 0.377541, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(filtered, exp, atol=1e-5), filtered.tolist()
    assert chosen.item() == 1, chosen.item()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for n in (0.5, 1.0, 2.0, 3.0):
        for _ in range(30):
            logits = torch.randn(torch.randint(6, 16, (1,)).item())
            f, _ = solve(logits, n, torch.tensor(0.3))
            assert torch.allclose(f, _oracle(logits, n), atol=1e-6), n


def check_sum_and_in_set(solve):
    torch.manual_seed(1)
    logits = torch.randn(5, 12)
    f, c = solve(logits, 1.5, torch.rand(5))
    assert torch.allclose(f.sum(-1), torch.ones(5), atol=1e-6)
    assert bool((f.gather(-1, c.view(-1, 1)) > 0).all())
    assert int((f[0] > 0).sum()) >= 1


def check_argmax_always_kept(solve):
    # argmax 的 logit == max >= max - n*std，必被保留
    torch.manual_seed(2)
    logits = torch.randn(4, 10)
    f, _ = solve(logits, 1.0, torch.rand(4))
    am = logits.argmax(-1)
    assert bool((f.gather(-1, am.view(-1, 1)) > 0).all())


def check_large_n_keeps_all(solve):
    # n 很大时阈值远低于 min，全保留 → filtered == softmax
    logits = torch.tensor([3.0, 2.5, 0.0, -0.2, -0.5, -4.0])
    f, _ = solve(logits, 100.0, torch.tensor(0.3))
    assert torch.allclose(f, torch.softmax(logits, dim=-1), atol=1e-6), f.tolist()


check_example.example = True
