import torch


def _oracle(logits, history, pr, pf, pp, last_n):
    """独立重算：用 bincount 统计窗口计数 + 张量化 where 分支，与 solution 的 dict 循环不同。"""
    v = logits.shape[0]
    win = list(history) if last_n < 0 else list(history)[-int(last_n):]
    if win:
        counts = torch.bincount(torch.tensor(win, dtype=torch.long), minlength=v).float()
    else:
        counts = torch.zeros(v)
    present = (counts > 0).float()
    out = logits.clone()
    scaled = torch.where(out <= 0, out * pr, out / pr)
    # 仅对出现过的 token 施加缩放
    out = torch.where(present > 0, scaled, out)
    out = out - counts * pf - present * pp
    return out


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.tensor([1.0, -2.0, 0.5, 3.0, -0.5])
    out = solve(logits, [0, 1, 0, 2, 0], 1.1, 0.1, 0.2, -1)
    exp = torch.tensor([0.409091, -2.5, 0.154545, 3.0, -0.5])
    assert torch.allclose(out, exp, atol=1e-5), out.tolist()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    V = 9
    for _ in range(40):
        logits = torch.randn(V) * 3 - 1
        hlen = int(torch.randint(1, 12, (1,)).item())
        history = torch.randint(0, V, (hlen,)).tolist()
        last_n = int(torch.randint(-1, 13, (1,)).item())
        got = solve(logits, history, 1.15, 0.2, 0.3, last_n)
        assert torch.allclose(got, _oracle(logits, history, 1.15, 0.2, 0.3, last_n), atol=1e-6)


def check_identity_when_disabled(solve):
    logits = torch.tensor([1.0, -2.0, 0.5, 3.0, -0.5])
    out = solve(logits, [0, 1, 0, 2, 0], 1.0, 0.0, 0.0, -1)
    assert torch.allclose(out, logits, atol=1e-7), out.tolist()


def check_window_limits_counts(solve):
    # penalty_last_n=2 只统计末尾两个 token [2, 0]，token1 不受罚
    logits = torch.tensor([1.0, -2.0, 0.5, 3.0, -0.5])
    out = solve(logits, [0, 1, 0, 2, 0], 1.1, 0.1, 0.2, 2)
    assert torch.allclose(out, _oracle(logits, [0, 1, 0, 2, 0], 1.1, 0.1, 0.2, 2), atol=1e-6)
    assert out[1].item() == -2.0, out.tolist()  # token1 不在窗口内 → 原样


def check_penalized_decreases(solve):
    # 出现过的 token 惩罚后 logit 必然下降
    logits = torch.tensor([2.0, -1.0, 0.0, 5.0])
    out = solve(logits, [0, 1, 2], 1.2, 0.1, 0.1, -1)
    for t in (0, 1, 2):
        assert out[t] < logits[t], (t, out.tolist())
    assert out[3] == logits[3]  # 未出现 → 不变


check_example.example = True
