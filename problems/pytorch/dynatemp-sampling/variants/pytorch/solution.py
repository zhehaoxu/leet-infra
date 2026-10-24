"""参考解：Dynatemp 动态温度（温度随分布熵自适应；u 显式注入保证采样确定性）。

按每行的归一化熵 C = entropy / max_entropy（max_entropy = ln n）把温度插值到窗口
[T_min, T_max] = [max(0, temp-delta), temp+delta]：
    dyn_temp = T_min + (T_max - T_min) * C ** exponent
再用 logits / dyn_temp 做 softmax 得 filtered，u 逆变换采样得 chosen。
delta <= 0 时禁用动态、退化为固定温度 temp（dyn_temp == temp）。

返回 (filtered, chosen, dyn_temp)。
"""

import math

import torch

_TINY = 1e-30


def _inverse_cdf(filtered, u):
    if u.dim() < filtered.dim():
        u = u.view(u.shape + (1,) * (filtered.dim() - u.dim()))
    cdf = torch.cumsum(filtered, dim=-1)
    ge = cdf >= u
    pos = ge.to(torch.long).argmax(dim=-1)
    pos = torch.where(ge.any(dim=-1), pos, torch.full_like(pos, filtered.shape[-1] - 1))
    return pos


def solve(logits, temp, delta, exponent, u):
    probs = torch.softmax(logits, dim=-1)
    n = probs.shape[-1]
    lead = probs.shape[:-1]
    temp = float(temp)
    delta = float(delta)
    if delta <= 0.0:
        dyn = torch.full(lead, temp, dtype=probs.dtype, device=probs.device)
        filtered = torch.softmax(logits / temp, dim=-1)
    else:
        max_entropy = -math.log(1.0 / n)  # = ln n
        entropy = -(probs * probs.clamp_min(_TINY).log()).sum(dim=-1)
        norm = entropy / max_entropy
        t_min = max(0.0, temp - delta)
        t_max = temp + delta
        dyn = t_min + (t_max - t_min) * norm.pow(float(exponent))
        filtered = torch.softmax(logits / dyn.unsqueeze(-1), dim=-1)
    chosen = _inverse_cdf(filtered, u)
    return filtered, chosen, dyn
