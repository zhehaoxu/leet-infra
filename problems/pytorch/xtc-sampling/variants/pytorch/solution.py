"""参考答案：XTC（Exclude Top Choices）采样（两处随机性均作显式入参，保证判题确定性）。

对齐 llama.cpp `src/llama-sampler.cpp::llama_sample_xtc_apply`：
1. 早退条件（任一命中即不触发）：probability <= 0、threshold > 0.5、候选数 < 2。
2. 概率门：`u_gate <= probability` 才执行剔除（C++ 里 `chance > probability` 直接 return）。
3. 把 probs=softmax(logits) 按概率降序（并列按索引升序）排布，统计**连续前缀**中满足
   p >= threshold 的元素个数 cnt，pos_last = cnt - 1 即前缀里最后一个达标元素的秩
   （softmax 单调 ⇒ 达标集合必是降序数组的前缀，等价于 C++ 的「遇到第一个 p < threshold 就 break」）。
4. 仅当 pos_last > 0 且存活数 V - pos_last >= min_keep 时，剔除秩 [0, pos_last)
   —— 砍掉最头部那批高概率候选，但保留它们中最不高的那一个 + 整个尾部。
   这就是 XTC「把截断反过来用」：top-p/min-p 砍尾部，XTC 砍头部。
5. 存活集合重新归一化得 filtered（原顺序，被剔除处置 0）；用 u 在 filtered 原顺序 CDF 上
   逆变换采样得 chosen。

返回 (filtered, chosen, applied)，applied 是布尔张量 (...,)，标记该行剔除是否真的发生。
"""

import torch


def _inverse_cdf(filtered, u):
    if u.dim() < filtered.dim():
        u = u.view(u.shape + (1,) * (filtered.dim() - u.dim()))
    cdf = torch.cumsum(filtered, dim=-1)
    ge = cdf >= u
    pos = ge.to(torch.long).argmax(dim=-1)
    pos = torch.where(ge.any(dim=-1), pos, torch.full_like(pos, filtered.shape[-1] - 1))
    return pos


def solve(logits, probability, threshold, min_keep, u_gate, u):
    probs = torch.softmax(logits, dim=-1)
    v = probs.shape[-1]
    shape = probs.shape[:-1]

    fired = torch.zeros(shape, dtype=torch.bool)
    if float(probability) > 0.0 and float(threshold) <= 0.5 and v >= 2:
        order = torch.argsort(probs, dim=-1, descending=True, stable=True)
        sorted_p = probs.gather(-1, order)
        cnt = (sorted_p >= float(threshold)).sum(dim=-1)
        pos_last = cnt - 1

        gate = u_gate.reshape(-1) <= float(probability)
        enough = (v - pos_last) >= int(min_keep)
        fired = gate.reshape(shape) & (pos_last > 0) & enough

        ranks = torch.arange(v, device=probs.device).expand_as(order)
        drop_rank = (ranks < pos_last.unsqueeze(-1)) & fired.unsqueeze(-1)
        keep = torch.ones_like(probs)
        keep.scatter_(-1, order, (~drop_rank).to(probs.dtype))
    else:
        keep = torch.ones_like(probs)

    filtered = probs * keep
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return filtered, _inverse_cdf(filtered, u), fired
