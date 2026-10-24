import math

import torch

_FLOAT_MAX_LOG = 88.7228391


def _oracle(logits, history, mult, base, allowed, last_n):
    """独立重算：朴素最长公共前缀（不用 Z-box）逐个 shift 求重复长，与 solution 的
    Z-algorithm 路径不同。"""
    out = logits.clone()
    allowed = int(allowed)
    last_n = int(last_n)
    last = list(history) if last_n < 0 else list(history)[-last_n:] if last_n > 0 else []
    N = len(last)
    if mult == 0.0 or base < 1.0 or last_n == 0 or N <= allowed:
        return out
    rev = last[::-1]
    max_tok = {}
    for k in range(1, N):
        n = 0
        while k + n < N and rev[n] == rev[k + n]:
            n += 1
        if n >= allowed:
            tok = last[N - k]
            if n > max_tok.get(tok, 0):
                max_tok[tok] = n
    max_exp = int(_FLOAT_MAX_LOG / math.log(base)) if base > 1.000001 else 0
    for tok, rl in max_tok.items():
        if tok < out.shape[0]:
            exp = rl - allowed
            if max_exp and exp > max_exp:
                exp = max_exp
            out[tok] = out[tok] - mult * (base ** exp)
    return out


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    logits = torch.zeros(8)
    out = solve(logits, [5, 6, 7, 5, 6, 7], 1.0, 1.5, 2, -1)
    exp = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 0.0])
    assert torch.allclose(out, exp, atol=1e-6), out.tolist()


def check_example_period2(solve):
    logits = torch.zeros(8)
    out = solve(logits, [0, 1, 0, 1, 0, 1], 1.0, 1.5, 2, -1)
    exp = torch.tensor([-2.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(out, exp, atol=1e-6), out.tolist()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    cases = [[5, 6, 7, 5, 6, 7], [1, 2, 3, 4, 2, 3, 4, 5], [0, 1, 0, 1, 0, 1],
             [2, 2, 2, 2, 2, 2, 2], [1, 2, 3, 1, 2, 3, 1, 2, 3], [4, 4, 1, 4, 4]]
    for _ in range(200):
        n = int(torch.randint(1, 15, (1,)).item())
        cases.append(torch.randint(0, 5, (n,)).tolist())
    logits = torch.randn(8)
    for hist in cases:
        last_n = int(torch.randint(-1, len(hist) + 2, (1,)).item())
        for allowed in (1, 2, 3):
            for base in (1.0, 1.5, 2.0):
                got = solve(logits, hist, 1.0, base, allowed, last_n)
                exp = _oracle(logits, hist, 1.0, base, allowed, last_n)
                assert torch.allclose(got, exp, atol=1e-6), (hist, allowed, base, last_n)


def check_no_repeat_no_penalty(solve):
    # 无重复且 allowed 较大 → 不罚
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    out = solve(logits, [1, 2, 3, 4], 1.0, 1.5, 2, -1)
    assert torch.allclose(out, logits, atol=1e-7), out.tolist()


def check_early_exit_guards(solve):
    logits = torch.tensor([1.0, -2.0, 0.5, 3.0])
    assert torch.allclose(solve(logits, [0, 0, 0, 0], 0.0, 1.5, 1, -1), logits)   # mult=0
    assert torch.allclose(solve(logits, [0, 0, 0, 0], 1.0, 0.9, 1, -1), logits)   # base<1
    assert torch.allclose(solve(logits, [0, 0, 0, 0], 1.0, 1.5, 5, -1), logits)   # N<=allowed


def check_longer_repeat_penalizes_harder(solve):
    # 同一 token，重复越长罚越重（base>1 幂次更高）：比较短循环 vs 长循环
    logits = torch.zeros(6)
    short = solve(logits, [1, 1], 1.0, 2.0, 1, -1)   # 末尾 token1，最大重复长 1 → 罚 2^0=1
    long = solve(logits, [1, 1, 1, 1], 1.0, 2.0, 1, -1)  # 最大重复长 3 → 罚 2^2=4
    assert short[1].item() < 0.0 and long[1].item() < 0.0, (short.tolist(), long.tolist())
    assert abs(long[1].item()) > abs(short[1].item()), (short[1].item(), long[1].item())


check_example.example = True
