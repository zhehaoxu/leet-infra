import torch


def solve(y, grad, dim=-1, log_input=False):
    """softmax 反向：给定前向输出与上游梯度，求损失对 logits 的梯度（不构造 Jacobian）。

    入参：
    - y: 前向输出张量（float32 或 float64）。
      * log_input=False 时 y = softmax(x, dim)，即逐点概率，沿 dim 的和为 1；
      * log_input=True  时 y = log_softmax(x, dim)，即对数概率，概率为 exp(y)。
    - grad: 与 y 同形的上游梯度 dL/dy。
    - dim: 归一化所在轴，默认 -1；负数按 y.dim() 折算，越界抛 ValueError。
    - log_input: bool，声明前向是 softmax 还是 log_softmax。两条路的公式**不同**：
      * softmax:     dx_i = y_i * (grad_i - Σ_k y_k * grad_k)
      * log_softmax: dx_i = grad_i - exp(y_i) * Σ_k grad_k
      两者都只依赖「沿 dim 的一次点积」，禁止写成 n×n 的雅可比矩阵 (δ_ij - y_i y_j) 再乘 grad。

    返回与 y 同形、同 dtype 的梯度张量。要求：
    - y 与 grad 形状不一致抛 ValueError；log_input 非 bool 抛 ValueError。
    - 若 grad 沿 dim 是常数（例如全 1），softmax 支路的结果必须为 0（softmax 的平移不变性）。
    - 沿 dim 求和后的梯度满足 Σ_i dx_i * y_i = Σ_i grad_i * y_i - (Σ_i y_i grad_i)（可用于自检）。
    """
    # TODO
    raise NotImplementedError
