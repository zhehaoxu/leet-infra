import torch


def solve(forward, backward, inputs, eps=None):
    """把自定义前向/反向包进 autograd.Function，用中心差分做梯度校验，返回逐输入的最大相对误差。

    入参：
    - forward: callable(*inputs) -> **0 维** Tensor（标量损失）。
    - backward: callable(*inputs, grad_output) -> 与 inputs 一一对应的梯度 tuple/list；
      grad_output 是 0 维张量（对标量输出就是 1.0）。
    - inputs: tuple/list 的 torch.Tensor，必须全部是 float64；不要求预先 requires_grad。
    - eps: 步长基数，默认 torch.finfo(torch.float64).eps；第 i 个元素的步长
      δ_i = sqrt(eps) · (1 + |x_i|)。

    口径：
    - 解析梯度**必须**经过 autograd.Function：forward 里算前向、backward 里调用给定的
      backward，再 torch.autograd.grad 取梯度。若改成对 forward 直接 autograd.grad，
      就等于相信了 PyTorch 的反向，永远校验不到自定义 backward（本题专门抓这一点）。
    - 数值梯度对**每个输入的每个元素**都做中心差分 (L(x+δ) − L(x−δ)) / (2δ)，
      只查第一个输入、或只查前几个元素都不算通过。
    - 相对误差 rel = |a − n| / max(|a|, |n|, 1e-8)。返回 list[float]，长度 = len(inputs)，
      第 k 项是 inputs[k] 所有元素 rel 的最大值。
    - 合法输入不得抛异常；以下情况一律抛 ValueError：inputs 里有非 float64 张量、
      forward 返回的不是 0 维张量、backward 返回的个数与 inputs 不符。
    """
    # TODO
    raise NotImplementedError
