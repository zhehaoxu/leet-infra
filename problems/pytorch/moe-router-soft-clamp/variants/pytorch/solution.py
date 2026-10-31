"""参考答案：先软限幅、再 sigmoid、bias 只加在「选择分数」上，权重始终由不带 bias 的 s 归一化。

三个容易写反的地方：
1. 限幅必须发生在 sigmoid **之前**。tanh 是严格单调的，所以只要浮点上还分得开，
   限幅就不会改变排序；它改变的是梯度的大小与饱和段的「并列」。
2. 归一化用的是 s，不是 s+bias。把 bias 带进权重里，就等于把「负载均衡的补偿项」
   变成了「专家输出的实际系数」，这正是 aux-free 路由反复强调要避免的事。
3. 选中项之和可能因为 sigmoid 下溢而恰好为 0（tau=0 且 logits 里有 -800 这种值），
   此时按契约给均匀 1/k，而不是让 0/0 变成 NaN。

排序用 argsort(descending, stable)：并列时保留原索引顺序，即小编号优先，
与判题里纯 Python 的 sorted(key=(-score, index)) 完全一致。
"""

import math

import torch


def _check(logits, bias, k, tau):
    if not isinstance(logits, torch.Tensor):
        raise ValueError("logits must be a torch.Tensor")
    if logits.dtype != torch.float64 or logits.device.type != "cpu":
        raise ValueError("logits must be a CPU float64 tensor")
    if logits.dim() != 2:
        raise ValueError("logits must be 2-D [T, E]")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= logits.shape[1]:
        raise ValueError("k must be a true int with 1 <= k <= E")
    if isinstance(tau, bool) or not isinstance(tau, (int, float)):
        raise ValueError("tau must be an int or float")
    if math.isnan(tau) or math.isinf(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or bias.dtype != torch.float64:
            raise ValueError("bias must be None or a CPU float64 tensor")
        if bias.device.type != "cpu" or bias.dim() != 1 or bias.shape[0] != logits.shape[1]:
            raise ValueError("bias must have shape [E]")


def solve(logits, bias, k, tau):
    _check(logits, bias, k, tau)
    if tau > 0:
        clamped = torch.tanh(logits / tau) * tau
    else:
        clamped = logits
    scores = torch.sigmoid(clamped)
    selection = scores if bias is None else scores + bias
    indices = torch.argsort(selection, dim=-1, descending=True, stable=True)[:, :k]
    picked = scores.gather(-1, indices)
    total = picked.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(picked, 1.0 / k)
    weights = torch.where(total > 0, picked / total.clamp_min(1e-300), uniform)
    return weights, indices, scores
