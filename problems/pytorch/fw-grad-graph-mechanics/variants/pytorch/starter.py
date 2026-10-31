def solve(x):
    """按题面步骤在入参 x 上建一张 autograd 图，返回每一步的机制账本。

    参数
    ----
    x : 一维浮点张量（requires_grad=False），作为数据来源。禁止修改它：
        不得原地写、不得让它的 requires_grad 变成 True。

    返回
    ----
    dict，键与含义（全部由下面 8 步产生，不得走捷径手填）：
        w_is_leaf            bool   步 2 的叶子 w 是否是叶子
        w_copies_input       bool   w 是否复制了 x 的存储（clone 与 detach 的
                                    分界：clone 新开 storage，detach 共享）
        y_is_leaf            bool   步 3 的 y 是否是叶子
        y_grad_fn            str    y.grad_fn 的类名（非叶时），叶时为 "None"
        chain                list   步 6 的 loss 图上从输出端到叶子端的
                                    grad_fn 类名序列（含两端），如
                                    ["SumBackward0", "MulBackward0", "AccumulateGrad"]
        leaf_accumulator     str    叶子在图输入端对应的节点类名
        d_is_leaf            bool   步 4 的 d 是否是叶子
        d_requires_grad      bool   刚 detach 完（尚未 requires_grad_）的 d
        d_shared_storage     bool   d 与 y 是否共享同一 storage
        n_requires_grad      bool   步 5 的 n.requires_grad
        t_requires_grad      bool   步 7 的 t.requires_grad
        t_is_inference       bool   步 7 的 t.is_inference()
        t_rg_error           bool   步 8 对 t 调 requires_grad_(True) 是否
                                    抛出 RuntimeError（捕获后不外泄）
        w_grad_first         tensor 步 9 第一次 backward 后 w.grad（clone）
        w_grad_second        tensor 步 10 第二次 backward 后 w.grad（clone），
                                    用以证明梯度是累加而非覆盖
        d_grad               tensor 步 12 后 d.grad（clone）

    步骤
    ----
    1. 校验 x：必须是一维浮点张量（float32/float64），否则抛 ValueError。
    2. w = x.clone().requires_grad_(True)：不得直接对 x 开 requires_grad。
    3. y = w * 2 + 1。
    4. d = y.detach()。
    5. n = w * 3，在 torch.no_grad() 里计算。
    6. loss = (w * 2 + 1).sum()，记录它到叶子的 grad_fn 链。
    7. t = w * 3，在 torch.inference_mode() 里计算。
    8. 捕获 t.requires_grad_(True) 的异常。
    9. y.backward(torch.ones_like(y), retain_graph=True)。
    10. 再次 y.backward(torch.ones_like(y))（第二次不得再传 retain_graph）。
    11. d.requires_grad_(True)。
    12. (d * 3).sum().backward()。
    """
    raise NotImplementedError
