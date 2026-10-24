"""参考解：Adaptive-p 采样（llama.cpp `adaptive-p`；EMA 状态显式入/出）。

思想：不截断，而是把分布**变形**——用一条以「目标概率 adapted_target」为中心的钟形函数
重排每个 token 的 logit，让概率最接近 adapted_target 的 token 拿到最高新 logit。adapted_target
由历史采样命中概率的指数滑动平均（EMA）反馈得到：近期总抽到高概率（分布过尖）就抬高目标、
鼓励更平的分布，反之压低。

流程：
1. probs = softmax(logits)（这份「原始概率」要留给 EMA 更新）。
2. adapted_target = clamp(total_weight==0 ? target : 2*target - weighted_sum/total_weight, 0, 1)。
3. dist = |probs - adapted_target| * INV_WIDTH；new_logit = PEAK - SHARPNESS * dist²/(1+dist)。
4. probs2 = softmax(new_logit)（这就是返回的 filtered，和为 1）。
5. u 在 probs2（原顺序）CDF 上逆变换采样得 chosen。
6. EMA 用**原始 probs[chosen]** 更新：weighted_sum' = probs[chosen] + decay*weighted_sum；
   total_weight' = 1 + decay*total_weight。

返回 (filtered, chosen, weighted_sum_new, total_weight_new)。
"""

import torch

_DISTRIBUTION_WIDTH = 0.3
_PEAK_LOGIT_VALUE = 5.0
_SHARPNESS = 10.0
_INV_WIDTH = 1.0 / _DISTRIBUTION_WIDTH


def _inverse_cdf(filtered, u):
    if u.dim() < filtered.dim():
        u = u.view(u.shape + (1,) * (filtered.dim() - u.dim()))
    cdf = torch.cumsum(filtered, dim=-1)
    ge = cdf >= u
    pos = ge.to(torch.long).argmax(dim=-1)
    pos = torch.where(ge.any(dim=-1), pos, torch.full_like(pos, filtered.shape[-1] - 1))
    return pos


def solve(logits, target, decay, weighted_sum, total_weight, u):
    probs = torch.softmax(logits, dim=-1)
    lead = probs.shape[:-1]
    ws = torch.as_tensor(float(weighted_sum), dtype=probs.dtype, device=probs.device)
    wt = torch.as_tensor(float(total_weight), dtype=probs.dtype, device=probs.device)

    if float(wt) == 0.0:
        adapted = torch.full(lead, float(target), dtype=probs.dtype, device=probs.device)
    else:
        adapted = (2.0 * float(target) - ws / wt).clamp(0.0, 1.0)
    adapted = adapted.clamp(0.0, 1.0).reshape(lead + (1,))

    dist = (probs - adapted).abs() * _INV_WIDTH
    new_logit = _PEAK_LOGIT_VALUE - _SHARPNESS * dist * dist / (1.0 + dist)
    filtered = torch.softmax(new_logit, dim=-1)

    chosen = _inverse_cdf(filtered, u)
    sel = probs.gather(-1, chosen.view(-1, 1)).reshape(lead) if probs.dim() > 1 \
        else probs[chosen]
    weighted_sum_new = sel + float(decay) * ws
    total_weight_new = 1.0 + float(decay) * wt
    return filtered, chosen, weighted_sum_new, total_weight_new
