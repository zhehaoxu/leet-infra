import torch


def _step_all(opt, grads_seq):
    """按梯度序列驱动优化器：把每步梯度挂到参数对象上再 step。"""
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_novograd(params, grads_seq, lr, betas, eps, weight_decay, grad_averaging):
    """独立向量化重算：逐层标量二阶矩 EMA + 归一 + 纯动量，无偏差修正。"""
    b1, b2 = betas
    ps = [p.clone() for p in params]
    s = [None] * len(ps)
    m = [None] * len(ps)
    for gs in grads_seq:
        for i, (p, g) in enumerate(zip(ps, gs)):
            gn = g.clone()
            g2 = gn.pow(2).sum()
            first = s[i] is None
            if first:
                s[i] = g2
            else:
                s[i] = b2 * s[i] + (1 - b2) * g2
            de_nom = torch.sqrt(s[i]) + eps
            gn = gn / de_nom
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)
            if first:
                m[i] = gn.clone()
            else:
                if grad_averaging:
                    gn = gn * (1 - b1)
                m[i] = b1 * m[i] + gn
            p -= lr * m[i]
    return ps


def check_example(solve):
    ps = [torch.tensor([1.0, -2.0, 0.5])]
    grads = [
        [torch.tensor([0.1, -0.2, 0.3])],
        [torch.tensor([-0.4, 0.05, 0.2])],
        [torch.tensor([0.15, -0.1, -0.25])],
    ]
    opt = solve(ps, lr=0.01, betas=(0.95, 0.98), eps=1e-8, weight_decay=0.01, grad_averaging=False)
    _step_all(opt, grads)
    exp = torch.tensor([1.0088257789611816, -1.9840788841247559, 0.47327882051467896])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_matches_oracle_mixed_dims(solve):
    torch.manual_seed(42)
    p0 = [torch.randn(3, 4), torch.randn(5)]  # 2D 权重（整层标量归一）+ 1D 偏置混合
    gseq = [[torch.randn_like(p) * 0.3 for p in p0] for _ in range(20)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=0.01, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.05, grad_averaging=False)
    _step_all(opt, gseq)
    exp = _oracle_novograd([p.clone() for p in p0], gseq, 0.01, (0.9, 0.98), 1e-8, 0.05, False)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参考:\n{a}\nvs\n{b}"


def check_grad_averaging(solve):
    torch.manual_seed(3)
    p0 = [torch.randn(2, 3)]
    gseq = [[torch.randn_like(p) * 0.4 for p in p0] for _ in range(15)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=0.02, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, grad_averaging=True)
    _step_all(opt, gseq)
    exp = _oracle_novograd([p.clone() for p in p0], gseq, 0.02, (0.9, 0.95), 1e-8, 0.01, True)
    assert torch.allclose(ps[0], exp[0], atol=1e-5, rtol=1e-5)


def check_none_grad_param_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, betas=(0.95, 0.98), eps=1e-8, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()  # 第二个参数无梯度，含权重衰减在内都应保持不变
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动: {ps[1].tolist()}"


def check_returns_optimizer_instance(solve):
    ps = [torch.tensor([1.0, 2.0])]
    opt = solve(ps, lr=0.1)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
