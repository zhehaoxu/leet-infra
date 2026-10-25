"""参考答案：logit-bias + banned tokens（随机性以显式入参 u 注入）。

对齐 llama.cpp `src/llama-sampler.cpp::llama_sampler_logit_bias_apply` 与
`common/sampling.cpp::common_sampler_init` 的 bias 装配：
1. 加性偏置按**列表顺序逐条 `logit += bias`**。同一个 token 在串里出现多次 → 自然累加
   （llama.cpp 的 `--logit-bias` 只是往 vector 里 push_back，不做合并），故
   `[1,+2],[1,+3]` 等价于 `[1,+5]`（浮点上仍要注意相加顺序）。
2. 越界/为负的 token id 匹配不到任何候选 → 该条**无副作用**（C++ 里 `lb.token >= 0` 的快路径
   和后续 `id == token` 的线性查找都不会命中）。
3. banned tokens = 强制掩码：`logit = -inf`。对应 llama.cpp 把模型 vocab 的 suppress tokens
   以 `{token, -INFINITY}` 追加进 logit_bias 串（`-ignore-eos` 亦即 `--logit-bias EOS-inf`）。
4. 对偏置后的 logits 做 softmax：被 ban 的项概率**恰好为 0**，其余按存活集合归一化。
5. 用 u 在 filtered 原顺序 CDF 上逆变换采样得 chosen（必然落在未 ban 集合内）。

返回 (biased_logits, filtered, chosen)；biased_logits 形状同 logits，含 -inf。
"""

import torch

NEG_INF = float("-inf")


def _inverse_cdf(filtered, u):
    if u.dim() < filtered.dim():
        u = u.view(u.shape + (1,) * (filtered.dim() - u.dim()))
    cdf = torch.cumsum(filtered, dim=-1)
    ge = cdf >= u
    pos = ge.to(torch.long).argmax(dim=-1)
    pos = torch.where(ge.any(dim=-1), pos, torch.full_like(pos, filtered.shape[-1] - 1))
    return pos


def solve(logits, bias_tokens, bias_values, banned, u):
    out = logits.clone()
    v = out.shape[-1]
    for t, b in zip(list(bias_tokens), list(bias_values)):
        tid = int(t)
        if 0 <= tid < v:                      # 匹配不到候选的条目无副作用
            out[..., tid] = out[..., tid] + float(b)
    for t in list(banned):
        tid = int(t)
        if 0 <= tid < v:
            out[..., tid] = NEG_INF

    # 手写 softmax：exp(-inf - max) = 0，被 ban 项精确为 0（整行全 ban 不在约束内）
    shifted = out - out.max(dim=-1, keepdim=True).values
    exps = torch.where(torch.isinf(shifted), torch.zeros_like(shifted), torch.exp(shifted))
    filtered = exps / exps.sum(dim=-1, keepdim=True)
    return out, filtered, _inverse_cdf(filtered, u)
