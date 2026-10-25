"""参考答案：新版 setup_context API 的 autograd.Function，含不可微第二输出与调用计数账。"""

import torch


class FusedSquareTanh(torch.autograd.Function):
    fwd_calls = 0
    bwd_calls = 0
    extra_grad_is_none = False
    needs_input_grad_seen = []

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, c = inputs
        ctx.save_for_backward(x)
        ctx.c = c

    @staticmethod
    def forward(x, c):
        FusedSquareTanh.fwd_calls += 1
        y = c * x * x * torch.tanh(x)
        neg_count = int((x < 0).sum().item())  # 纯 Python int：不可微对象原样返回
        return y, neg_count

    @staticmethod
    def backward(ctx, g, g_extra):
        FusedSquareTanh.bwd_calls += 1
        FusedSquareTanh.extra_grad_is_none = g_extra is None
        FusedSquareTanh.needs_input_grad_seen = [bool(b) for b in ctx.needs_input_grad]
        (x,) = ctx.saved_tensors
        c = ctx.c
        t = torch.tanh(x)
        gx = c * (2 * x * t + x * x * (1 - t * t)) * g
        return gx, None  # c 不可微：按 needs_input_grad 返回 None


def solve():
    FusedSquareTanh.fwd_calls = 0
    FusedSquareTanh.bwd_calls = 0

    x = torch.tensor([-2.5, -0.75, 0.5, 1.25, 3.0], requires_grad=True)
    c = torch.tensor(2.0)  # 不 requires_grad
    g = torch.arange(1, 6, dtype=torch.float32)

    y, neg_count = FusedSquareTanh.apply(x, c)
    (grad_x,) = torch.autograd.grad(y, x, grad_outputs=g)

    # 对照：同样的函数用标准算子组合式书写，autograd 自动求的局部雅可比
    xd = x.detach().clone().requires_grad_(True)
    y_ref = c * xd * xd * torch.tanh(xd)
    (grad_x_ref,) = torch.autograd.grad(y_ref, xd, grad_outputs=g)

    # 第二输出是 int：没有 .backward / 不进图；再反传一次会因图已释放而报错
    second_backward_raises = False
    try:
        y2, _ = FusedSquareTanh.apply(x, c)
        torch.autograd.grad(y2, x, grad_outputs=g)
        torch.autograd.grad(y2, x, grad_outputs=g)
    except RuntimeError:
        second_backward_raises = True

    return {
        "y_flat": y.detach().tolist(),
        "neg_count": neg_count,
        "neg_count_type": type(neg_count).__name__,
        "y_has_grad_fn": y.grad_fn is not None,
        "grad_x_flat": grad_x.tolist(),
        "grad_x_ref_flat": grad_x_ref.tolist(),
        "grads_match_ref": bool(torch.allclose(grad_x, grad_x_ref, atol=1e-5, rtol=1e-5)),
        "extra_grad_is_none": FusedSquareTanh.extra_grad_is_none,
        "needs_input_grad": FusedSquareTanh.needs_input_grad_seen,
        "fwd_calls": FusedSquareTanh.fwd_calls,
        "bwd_calls": FusedSquareTanh.bwd_calls,
        "second_backward_raises": second_backward_raises,
    }
