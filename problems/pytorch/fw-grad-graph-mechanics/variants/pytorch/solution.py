import torch


def solve(x):
    """按 starter 注释的 12 步建图，逐步记录 autograd 机制账本。"""
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a torch.Tensor")
    if x.dim() != 1:
        raise ValueError("x must be 1-D")
    if not x.is_floating_point():
        raise ValueError("x must be a floating-point tensor")

    ledger = {}

    # 步 2：叶子必须自己造，不能弄脏入参
    w = x.clone().requires_grad_(True)
    ledger["w_is_leaf"] = w.is_leaf
    ledger["w_copies_input"] = w.data_ptr() != x.data_ptr()

    # 步 3：非叶中间量
    y = w * 2 + 1
    ledger["y_is_leaf"] = y.is_leaf
    ledger["y_grad_fn"] = type(y.grad_fn).__name__ if y.grad_fn is not None else "None"

    # 步 4：detach = 同 storage 的新叶子
    d = y.detach()
    ledger["d_is_leaf"] = d.is_leaf
    ledger["d_requires_grad"] = d.requires_grad
    ledger["d_shared_storage"] = d.data_ptr() == y.data_ptr()

    # 步 5：no_grad 产物不是叶子、不可导
    with torch.no_grad():
        n = w * 3
    ledger["n_requires_grad"] = n.requires_grad

    # 步 6：到叶子的 grad_fn 链（输出端 -> 叶子端）
    loss = (w * 2 + 1).sum()
    chain = []
    node = loss.grad_fn
    while node is not None:
        chain.append(type(node).__name__)
        node = node.next_functions[0][0] if node.next_functions else None
    ledger["chain"] = chain
    ledger["leaf_accumulator"] = "AccumulateGrad"

    # 步 7：inference_mode 产物是 inference tensor，且不可导
    with torch.inference_mode():
        t = w * 3
    ledger["t_requires_grad"] = t.requires_grad
    ledger["t_is_inference"] = t.is_inference()

    # 步 8：inference tensor 不能事后开 requires_grad
    try:
        t.requires_grad_(True)
        ledger["t_rg_error"] = False
    except RuntimeError:
        ledger["t_rg_error"] = True

    # 步 9/10：非标量 backward 用 gradient 参数；默认 retain_graph=False，
    # 第一次必须显式 retain，梯度在叶子上累加
    y.backward(torch.ones_like(y), retain_graph=True)
    ledger["w_grad_first"] = w.grad.clone()
    y.backward(torch.ones_like(y))
    ledger["w_grad_second"] = w.grad.clone()

    # 步 11/12：detach 产物可以重新作为叶子建第二张图
    d.requires_grad_(True)
    (d * 3).sum().backward()
    ledger["d_grad"] = d.grad.clone()

    return ledger
