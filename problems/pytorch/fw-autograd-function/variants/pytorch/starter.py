import torch


def solve():
    """用新版 `torch.autograd.Function`（setup_context API）实现融合算子并返回行为账。

    算子定义：输入可微张量 \(x\) 与非可微标量 \(c\)，
    前向 \\(y = c\\,x^2\\tanh(x)\\)；同时返回**不可微第二输出**
    `neg_count = int((x < 0).sum())`（纯 Python int，原样透出，不要包成张量）。

    必须实现的 Function 类（新版 API，forward 签名不带 ctx）::

        class FusedSquareTanh(torch.autograd.Function):
            fwd_calls / bwd_calls: 类属性计数器，forward、backward 第一行各自 +1
            extra_grad_is_none:    backward 第一行记录 g_extra is None
            needs_input_grad_seen: backward 第一行记录 [bool(b) for b in ctx.needs_input_grad]

            setup_context(ctx, inputs, output): x, c = inputs；ctx.save_for_backward(x)；ctx.c = c
            forward(x, c): fwd_calls += 1；返回 (c*x*x*tanh(x), neg_count)
            backward(ctx, g, g_extra): bwd_calls += 1；先记 extra_grad_is_none 和
                needs_input_grad_seen，再取 ctx.saved_tensors，解析梯度
                gx = c*(2x*tanh(x) + x^2*(1 - tanh(x)^2)) * g；返回 (gx, None)

    solve() 固定流程::

        1. 类属性计数器复位为 0（只复位 fwd_calls/bwd_calls）
        2. x = tensor([-2.5, -0.75, 0.5, 1.25, 3.0], requires_grad=True)
           c = tensor(2.0)（不 requires_grad）；g = arange(1,6).float()
        3. y, neg_count = FusedSquareTanh.apply(x, c)
        4. grad_x = torch.autograd.grad(y, x, grad_outputs=g)[0]
        5. 对照路径：xd = x.detach().clone().requires_grad_(True)，
           y_ref = c*xd*xd*tanh(xd)（纯标准算子组合），
           grad_x_ref = autograd.grad(y_ref, xd, g)[0]
        6. 重传检查：再 apply 得 y2，连续两次 autograd.grad(y2, x, g)；
           第二次应抛 RuntimeError（save_for_backward 的张量已被释放），捕获记 True。
           注意：入口计数器仍会被第三次调用累加

    返回 dict（键与语义）::

        y_flat: list[float]                 # y 展平
        neg_count: int                       # 第二输出原值
        neg_count_type: str                  # type(neg_count).__name__
        y_has_grad_fn: bool
        grad_x_flat: list[float]             # 手写 backward 的梯度
        grad_x_ref_flat: list[float]         # 对照 autograd 的梯度
        grads_match_ref: bool                # 前两者 allclose(atol=1e-5, rtol=1e-5)
        extra_grad_is_none: bool             # backward 对 int 输出是否收到 None
        needs_input_grad: list[bool]         # backward 内所见 ctx.needs_input_grad
        fwd_calls: int                       # apply 次数
        bwd_calls: int                       # backward 入口被调用次数（含报错那次）
        second_backward_raises: bool
    """
    # TODO: 按 docstring 实现 Function 类与固定流程，汇总全部账目
    raise NotImplementedError
