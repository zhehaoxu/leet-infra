import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_yogi(params, grads_seq, lr, betas, initial_accumulator, eps, weight_decay):
    """独立向量化重算：符号控制二阶矩 + Adam 式偏差修正更新。"""
    b1, b2 = betas
    ps = [p.clone() for p in params]
    ms = [torch.full_like(p, initial_accumulator) for p in ps]
    vs = [torch.full_like(p, initial_accumulator) for p in ps]
    for t, gs in enumerate(grads_seq, start=1):
        bc1 = 1 - b1 ** t
        bc2sq = (1 - b2 ** t) ** 0.5
        step_size = lr / bc1
        for i, (p, g) in enumerate(zip(ps, gs)):
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)
            gp2 = g * g
            ms[i] = b1 * ms[i] + (1 - b1) * g
            vs[i] = vs[i] - (1 - b2) * torch.sign(vs[i] - gp2) * gp2
            de_nom = torch.sqrt(vs[i]) / bc2sq + eps
            p -= step_size * ms[i] / de_nom
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
    opt = solve(ps, lr=1e-2, betas=(0.9, 0.999), initial_accumulator=1e-6, eps=1e-3, weight_decay=0.01)
    _step_all(opt, grads)
    exp = torch.tensor([1.0012454986572266, -1.9801409244537354, 0.4729447662830353])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_matches_oracle_mixed_dims(solve):
    torch.manual_seed(11)
    p0 = [torch.randn(4, 3), torch.randn(6)]
    gseq = [[torch.randn_like(p) * 0.5 for p in p0] for _ in range(20)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=1e-2, betas=(0.9, 0.999), initial_accumulator=1e-4, eps=1e-3, weight_decay=0.02)
    _step_all(opt, gseq)
    exp = _oracle_yogi([p.clone() for p in p0], gseq, 1e-2, (0.9, 0.999), 1e-4, 1e-3, 0.02)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参考:\n{a}\nvs\n{b}"


def check_zero_decay(solve):
    torch.manual_seed(5)
    p0 = [torch.randn(3, 3)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(12)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=2e-2, betas=(0.9, 0.99), initial_accumulator=1e-6, eps=1e-3, weight_decay=0.0)
    _step_all(opt, gseq)
    exp = _oracle_yogi([p.clone() for p in p0], gseq, 2e-2, (0.9, 0.99), 1e-6, 1e-3, 0.0)
    assert torch.allclose(ps[0], exp[0], atol=1e-5, rtol=1e-5)


def check_none_grad_param_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=1e-2, betas=(0.9, 0.999), eps=1e-3, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动: {ps[1].tolist()}"


def check_returns_optimizer_instance(solve):
    ps = [torch.tensor([1.0, 2.0])]
    opt = solve(ps, lr=1e-2)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
