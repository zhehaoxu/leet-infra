import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0])
    filtered, chosen = solve(logits, 0.8, torch.tensor(0.05))
    exp = torch.tensor([0.7311, 0.2689, 0.0, 0.0, 0.0])
    assert torch.allclose(filtered, exp, atol=1e-3), filtered.tolist()
    assert chosen.item() == 0, chosen.item()


def check_nucleus_mass(solve):
    # p=0.99 几乎保留全部；每行和为 1
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    u = torch.rand(4) * 0.99
    filtered, chosen = solve(logits, 0.99, u)
    assert torch.allclose(filtered.sum(-1), torch.ones(4), atol=1e-5)
    assert bool((chosen >= 0).all() and (chosen < 10).all())


def check_small_p_keeps_top(solve):
    # p 很小 → 核只剩概率最大的 1 个 → filtered 是它的 one-hot，chosen 即 argmax
    torch.manual_seed(1)
    logits = torch.randn(3, 8)
    u = torch.tensor([0.5, 0.5, 0.5])
    filtered, chosen = solve(logits, 0.01, u)
    assert torch.equal(chosen, logits.argmax(-1)), "极小 p 应退化为贪心 argmax"
    assert torch.allclose(filtered.max(-1).values, torch.ones(3), atol=1e-5)


def check_chosen_in_nucleus(solve):
    # chosen 处概率必 >0（落在核内）
    torch.manual_seed(2)
    logits = torch.randn(5, 12)
    u = torch.rand(5)
    filtered, chosen = solve(logits, 0.7, u)
    got = filtered.gather(-1, chosen.view(-1, 1)).squeeze(-1)
    assert bool((got > 0).all()), "采样结果必须在核内"


def check_inverse_cdf(solve):
    # u 落在核内 CDF 的不同区间应选到不同 token
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0])   # 核 [0.7311, 0.2689]
    _, c_lo = solve(logits, 0.8, torch.tensor(0.1))       # u<0.7311 → idx0
    _, c_hi = solve(logits, 0.8, torch.tensor(0.9))       # u>0.7311 → idx1
    assert c_lo.item() == 0 and c_hi.item() == 1, (c_lo.item(), c_hi.item())


check_basic.example = True
