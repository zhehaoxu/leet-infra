"""参考答案：Function 包一层拿解析梯度，再逐元素中心差分，两者比相对误差。

三个容易糊过去的点：
1) 解析梯度只能来自 autograd.Function —— 直接对 forward 求 autograd 得到的永远是
   PyTorch 自己的反向，自定义 backward 写错了也测不出来。
2) 步长要逐元素取 δ = sqrt(eps)·(1+|x|)：绝对步长对大数太小、对小数太大；
   sqrt(eps)（float64 下约 1.5e-8）是中心差分的经典最优量级。
3) 每个输入的每个元素都得动一遍：多输入算子里最容易漏的就是「第二个输入根本没查」。
"""

import torch


def solve(forward, backward, inputs, eps=None):
    inputs = list(inputs)
    if not inputs:
        raise ValueError("inputs 不能为空")
    for t in inputs:
        if not isinstance(t, torch.Tensor):
            raise ValueError("inputs 必须全是 Tensor")
        if t.dtype != torch.float64:
            raise ValueError("gradcheck 必须在 float64 下做，float32 的差分噪声会淹没判定")

    base = eps if eps is not None else torch.finfo(torch.float64).eps
    delta0 = float(base) ** 0.5

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            ctx.save_for_backward(*args)
            with torch.no_grad():
                y = forward(*args)
            if not isinstance(y, torch.Tensor) or y.dim() != 0:
                raise ValueError("forward 必须返回 0 维张量（标量损失）")
            return y

        @staticmethod
        def backward(ctx, grad_output):
            grads = backward(*ctx.saved_tensors, grad_output)
            if not isinstance(grads, (tuple, list)):
                grads = (grads,)
            if len(grads) != len(ctx.saved_tensors):
                raise ValueError("backward 返回的梯度个数必须与 inputs 一致")
            return tuple(grads)

    params = [t.detach().clone().requires_grad_(True) for t in inputs]
    y = _Fn.apply(*params)
    try:
        analytic = torch.autograd.grad(y, params)
    except ValueError as exc:
        raise exc
    except RuntimeError as exc:                       # 返回个数/形状不对时 autograd 会抛这里
        raise ValueError(str(exc))

    def loss(vals):
        with torch.no_grad():
            out = forward(*vals)
        if not isinstance(out, torch.Tensor) or out.dim() != 0:
            raise ValueError("forward 必须返回 0 维张量（标量损失）")
        return float(out)

    errors = []
    for k in range(len(inputs)):
        vals = [t.detach().clone() for t in inputs]
        worst = 0.0
        flat = vals[k].view(-1)
        n_elem = flat.numel()
        for j in range(n_elem):
            xj = float(flat[j])
            d = delta0 * (1.0 + abs(xj))
            flat[j] = xj + d
            lp = loss(vals)
            flat[j] = xj - d
            lm = loss(vals)
            flat[j] = xj
            numeric = (lp - lm) / (2.0 * d)
            a = float(analytic[k].reshape(-1)[j])
            rel = abs(a - numeric) / max(abs(a), abs(numeric), 1e-8)
            if rel > worst:
                worst = rel
        errors.append(worst)
    return errors
