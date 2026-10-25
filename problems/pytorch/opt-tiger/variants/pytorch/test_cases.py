import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_tiger(params, grads_seq, lr, beta, weight_decay):
    """独立重算：EMA 动量 + 符号更新 + 解耦权重衰减。"""
    ps = [p.clone() for p in params]
    ms = [torch.zeros_like(p) for p in ps]
    for gs in grads_seq:
        for i, (p, g) in enumerate(zip(ps, gs)):
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)
            ms[i] = beta * ms[i] + (1 - beta) * g
            p -= lr * torch.sign(ms[i])
    return ps


def check_example(solve):
    ps = [torch.tensor([1.0, -2.0, 0.5])]
    grads = [
        [torch.tensor([0.1, -0.2, 0.3])],
        [torch.tensor([-0.4, 0.05, 0.2])],
        [torch.tensor([0.15, -0.1, -0.25])],
    ]
    opt = solve(ps, lr=0.01, beta=0.965, weight_decay=0.01)
    _step_all(opt, grads)
    exp = torch.tensor([1.0097010135650635, -1.9694030284881592, 0.469853013753891])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_matches_oracle_mixed_dims(solve):
    torch.manual_seed(23)
    p0 = [torch.randn(5, 2), torch.randn(4)]
    gseq = [[torch.randn_like(p) * 0.4 for p in p0] for _ in range(20)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=0.02, beta=0.9, weight_decay=0.05)
    _step_all(opt, gseq)
    exp = _oracle_tiger([p.clone() for p in p0], gseq, 0.02, 0.9, 0.05)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参考:\n{a}\nvs\n{b}"


def check_sign_update_is_lr_magnitude(solve):
    # 无权重衰减时单步：m=(1−β)·g 与 g 同号，故每个非零坐标恰移动 lr
    p = torch.tensor([2.0, -3.0, 0.0])
    g = torch.tensor([0.7, -0.1, 0.5])
    ps = [p.clone()]
    opt = solve(ps, lr=0.05, beta=0.9, weight_decay=0.0)
    _step_all(opt, [[g]])
    assert torch.allclose(ps[0], p - 0.05 * torch.sign(g), atol=1e-6, rtol=0), f"got {ps[0].tolist()}"


def check_none_grad_param_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, beta=0.965, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动: {ps[1].tolist()}"


def check_returns_optimizer_instance(solve):
    ps = [torch.tensor([1.0, 2.0])]
    opt = solve(ps, lr=0.1)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
