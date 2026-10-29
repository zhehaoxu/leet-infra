import torch


def solve(x, dim=None, keepdim=False):
    """logsumexp：torch.logsumexp 等价实现，log Σ exp(x) 的数值稳定版。

    入参：
    - x: float32 张量，可含 -inf（表示概率 0）与 +inf；内部 float64 计算，返回 float32。
    - dim: None / int / int 元组。None 表示把整个张量一起归约（返回 0 维），是本题在
      torch 之外的扩展口径；int 或元组时与 torch.logsumexp 一致，负数按 x.dim() 折算，
      越界、重复轴或空元组抛 ValueError。
    - keepdim: 保留被归约轴（长度 1）。

    实现要求（面试考点）：
    - 必须用「先提最大值 m」的稳定式：log Σ exp(x) = m + log Σ exp(x − m)，
      否则 x~1e3 时 exp 上溢成 inf、x~−1e3 时下溢全零导致 log(0) = −inf。
      朴素写法 log(sum(exp(x))) 在这两端都会挂，这也是本题唯一的硬约束。
    - 退化段必须显式分支：段内全为 −inf 时结果是 −inf（掩码把一整行 logits 全置 −inf
      就会遇到）；段内至少一个 +inf 时结果是 +inf；段内含 NaN 时结果传播 NaN。
      直接算 x − m 在 m = ±inf 处会产生 NaN 并污染整段，所以 ±inf 要单独取值。
    - 与 log-softmax 题的分工：那题要逐元素 x − logsumexp(x)，本题只交 logsumexp 本身。
    """
    # TODO
    raise NotImplementedError
