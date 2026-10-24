"""参考解：DRY 惩罚（反向 Z-algorithm，对齐 llama.cpp `dry` 的无-breaker 版本）。

流程：
1. 取窗口 last = history[-dry_penalty_last_n:]（<0 用整段），N = len(last)。
2. 早退：mult==0 / base<1 / last_n==0 / N<=allowed → 原样返回。
3. 对 last 的逆序 rev 跑 Z-algorithm，得每个 shift k∈[1,N) 的后缀重复长 z[k]（上限 N）。
4. 对 z[k] >= allowed 的 k，候选惩罚 token = last[N-k]（时间正序），同一 token 取最大 z。
5. logits[t] -= mult * base**(R-allowed)（R>=allowed 恒成立），指数按 max_exponent 截断防溢出。

返回惩罚后的 logits（形状 (V,)），不做采样。
"""

import math

import torch

_FLOAT_MAX_LOG = 88.7228391


def solve(logits, history, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n):
    out = logits.clone()
    allowed = int(dry_allowed_length)
    last_n = int(dry_penalty_last_n)
    last = list(history) if last_n < 0 else list(history)[-last_n:] if last_n > 0 else []
    N = len(last)
    if dry_multiplier == 0.0 or dry_base < 1.0 or last_n == 0 or N <= allowed:
        return out

    rev = last[::-1]
    z = [0] * N
    lt = rt = 0
    for k in range(1, N):
        if k > rt:
            n = 0
            while k + n < N and rev[n] == rev[k + n]:
                n += 1
            z[k] = min(n, N)
            if n > 0:
                lt, rt = k, k + n - 1
        else:
            p = k - lt
            right = rt - k + 1
            if z[p] < right:
                z[k] = min(z[p], N)
            else:
                i = rt + 1
                while i < N and rev[i] == rev[i - k]:
                    i += 1
                z[k] = min(i - k, N)
                lt, rt = k, i - 1

    max_tok = {}
    for k in range(1, N):
        if z[k] >= allowed:
            tok = last[N - k]
            if z[k] > max_tok.get(tok, 0):
                max_tok[tok] = z[k]

    max_exp = int(_FLOAT_MAX_LOG / math.log(dry_base)) if dry_base > 1.000001 else 0
    for tok, rl in max_tok.items():
        if tok < out.shape[0]:
            exp = rl - allowed
            if max_exp and exp > max_exp:
                exp = max_exp
            out[tok] = out[tok] - dry_multiplier * (dry_base ** exp)
    return out
