"""参考答案：固定轮数的 Sinkhorn–Knopp，列归一化在前、行归一化在后，eps 只加在除数上。

三个容易被写错的地方：
1. 交替方向。行归一化会把列和推离 1，列归一化又会把行和推离 1，所以「一轮」必须是
   一次列归一化加一次行归一化，而且以行归一化收尾——返回的 Q 每一行才是一个 token
   在专家上的分配，行和为 1 才是可采样的。反过来以列收尾，行和就不恒定了。
2. eps 加在除数上，不是被除数上。`q / (q.sum() + eps)` 与 `(q + eps) / q.sum()`
   给的是完全不同的矩阵：前者保留 0 元（0 除以正数还是 0），后者会把死列抬成 eps，
   凭空给零负载专家分走质量。真实实现里 SuperGlue 与 DeepSeek 的多路残差（mHC）
   都是前一种写法。
3. eps=0 时死列必须让 0/0 如实变成 NaN。Sinkhorn 的收敛前提是存在严格正的双随机
   标度；一列全零意味着某个专家在整个 batch 上的概率下溢为 0，此时矩阵没有有限的
   解，返回 NaN 比返回一个「看起来归一化过」的假矩阵更容易被发现。

固定轮数（而不是 `allclose(col_sum, 1, rtol=tol)` 提前停）是刻意的：提前停会让
迭代次数依赖浮点比较的顺序，同一份输入换个后端或换个归约顺序就可能少跑一轮，
判题就无法给出逐位一致的期望值。
"""

import math

import torch


def _check(logits, iters, tau, square, eps):
    if not isinstance(logits, torch.Tensor):
        raise ValueError("logits must be a torch.Tensor")
    if logits.dtype != torch.float64 or logits.device.type != "cpu":
        raise ValueError("logits must be a CPU float64 tensor")
    if logits.dim() != 2 or min(logits.shape) < 1:
        raise ValueError("logits must be a non-empty 2-D [T, E] tensor")
    if isinstance(iters, bool) or not isinstance(iters, int) or not 0 <= iters <= 16:
        raise ValueError("iters must be a true int with 0 <= iters <= 16")
    if not isinstance(square, bool):
        raise ValueError("square must be a true Python bool")
    for name, value in (("tau", tau), ("eps", eps)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be an int or float")
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{name} must be finite")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if eps < 0:
        raise ValueError("eps must be non-negative")


def solve(logits, iters, tau=1.0, square=False, eps=0.0):
    _check(logits, iters, tau, square, eps)
    q = torch.softmax(logits / tau, dim=-1)
    for _ in range(iters):
        q = q / (q.sum(dim=0, keepdim=True) + eps)   # 列归一化：压平专家负载
        q = q / (q.sum(dim=1, keepdim=True) + eps)   # 行归一化：每个 token 重新分配
    if square:
        q = q * q
        q = q / (q.sum(dim=1, keepdim=True) + eps)
    return q, q.sum(dim=0)
