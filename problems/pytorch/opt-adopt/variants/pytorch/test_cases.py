import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_adopt(params, grads_seq, lr, betas, eps, weight_decay):
    """独立重算：前一步 v 归一 + 裁剪 + 无偏差动量更新，末了并入 g²。"""
    b1, b2 = betas
    ps = [p.clone() for p in params]
    ms = [torch.zeros_like(p) for p in ps]
    vs = [torch.zeros_like(p) for p in ps]
    for t, gs in enumerate(grads_seq, start=1):
        for i, (p, g) in enumerate(zip(ps, gs)):
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)
            if t == 1:
                vs[i] = vs[i] + g * g
                continue
            de_nom = torch.clamp(torch.sqrt(vs[i]), min=eps)
            normed = torch.clamp(g / de_nom, -(t ** 0.25), t ** 0.25)
            ms[i] = b1 * ms[i] + (1 - b1) * normed
            p -= lr * ms[i]
            vs[i] = b2 * vs[i] + (1 - b2) * g * g
    return ps


def check_example(solve):
    ps = [torch.tensor([1.0, -2.0, 0.5])]
    grads = [
        [torch.tensor([0.1, -0.2, 0.3])],
        [torch.tensor([-0.4, 0.05, 0.2])],
        [torch.tensor([0.15, -0.1, -0.25])],
        [torch.tensor([0.05, 0.3, -0.1])],
        [torch.tensor([-0.2, -0.1, 0.4])],
    ]
    opt = solve(ps, lr=0.01, betas=(0.9, 0.9999), eps=1e-6, weight_decay=0.01)
    _step_all(opt, grads)
    exp = torch.tensor([1.0005689859390259, -2.0006916522979736, 0.4990157186985016])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_first_step_only_inits_v(solve):
    # 首步（无权重衰减时）只初始化二阶矩，参数必须完全不动
    ps = [torch.tensor([1.0, -2.0, 0.5])]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.9999), eps=1e-6, weight_decay=0.0)
    _step_all(opt, [[torch.tensor([0.3, -0.2, 0.7])]])
    assert torch.equal(ps[0], torch.tensor([1.0, -2.0, 0.5])), f"首步竟更新了参数: {ps[0].tolist()}"


def check_matches_oracle_mixed_dims(solve):
    torch.manual_seed(31)
    p0 = [torch.randn(3, 3), torch.randn(7)]
    gseq = [[torch.randn_like(p) * 0.5 for p in p0] for _ in range(20)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=0.02, betas=(0.9, 0.9999), eps=1e-6, weight_decay=0.05)
    _step_all(opt, gseq)
    exp = _oracle_adopt([p.clone() for p in p0], gseq, 0.02, (0.9, 0.9999), 1e-6, 0.05)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参考:\n{a}\nvs\n{b}"


def check_none_grad_param_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.9999), eps=1e-6, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动: {ps[1].tolist()}"


def check_returns_optimizer_instance(solve):
    ps = [torch.tensor([1.0, 2.0])]
    opt = solve(ps, lr=0.1)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
