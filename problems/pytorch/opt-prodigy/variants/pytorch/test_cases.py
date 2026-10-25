import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_prodigy(params, grads_seq, lr, betas, d0, d_coef, growth_rate, weight_decay, eps):
    """独立重算：list 版两趟 D-Adaptation，显式标量 num/den。"""
    b1, b2 = betas
    beta3 = b2 ** 0.5
    ps = [p.clone() for p in params]
    d, d_max = d0, d0
    num = torch.tensor([0.0])
    ss = [torch.zeros_like(p) for p in ps]
    p0s = [p.clone() for p in ps]
    ms = [torch.zeros_like(p) for p in ps]
    vs = [torch.zeros_like(p) for p in ps]
    for gs in grads_seq:
        d_lr = d * lr
        den = torch.tensor([0.0])
        num = num * beta3
        for i, (p, g) in enumerate(zip(ps, gs)):
            num = num + (d / d0) * d_lr * torch.dot(g.flatten(), (p0s[i] - p).flatten())
            ms[i] = b1 * ms[i] + d * (1 - b1) * g
            vs[i] = b2 * vs[i] + d * d * (1 - b2) * g * g
            ss[i] = beta3 * ss[i] + (d / d0) * d_lr * g
            den = den + ss[i].abs().sum()
        if den.item() == 0:
            continue
        d_hat = (d_coef * num / den).item()
        if d == d0:
            d = max(d, d_hat)
        d_max = max(d_max, d_hat)
        d = min(d_max, d * growth_rate)
        for i, p in enumerate(ps):
            if weight_decay != 0:
                p.mul_(1 - d_lr * weight_decay)
            de_nom = torch.sqrt(vs[i]) + d * eps
            p -= d_lr * ms[i] / de_nom
    return ps


def check_example(solve):
    torch.manual_seed(0)
    ps = [torch.tensor([1.0, -2.0, 0.5])]
    grads = [[torch.randn_like(ps[0]) * 0.5] for _ in range(30)]
    opt = solve(ps, lr=1.0, betas=(0.9, 0.999), d0=1e-2, d_coef=1.0, growth_rate=1.1,
                weight_decay=0.0, eps=1e-8)
    _step_all(opt, grads)
    exp = torch.tensor([0.8528182506561279, -2.713569402694702, 0.25577741861343384])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_matches_oracle_defaults(solve):
    torch.manual_seed(2)
    p0 = [torch.randn(4, 3), torch.randn(6)]
    gseq = [[torch.randn_like(p) * 0.5 for p in p0] for _ in range(25)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=1.0, betas=(0.9, 0.999), d0=1e-6, d_coef=1.0, growth_rate=float("inf"),
                weight_decay=0.0, eps=1e-8)
    _step_all(opt, gseq)
    exp = _oracle_prodigy([p.clone() for p in p0], gseq, 1.0, (0.9, 0.999), 1e-6, 1.0,
                          float("inf"), 0.0, 1e-8)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参考:\n{a}\nvs\n{b}"


def check_matches_oracle_growth_capped(solve):
    torch.manual_seed(9)
    p0 = [torch.randn(3, 3)]
    gseq = [[torch.randn_like(p) * 0.5 for p in p0] for _ in range(20)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=1.0, betas=(0.9, 0.999), d0=1e-2, d_coef=1.0, growth_rate=1.2,
                weight_decay=0.01, eps=1e-8)
    _step_all(opt, gseq)
    exp = _oracle_prodigy([p.clone() for p in p0], gseq, 1.0, (0.9, 0.999), 1e-2, 1.0, 1.2, 0.01, 1e-8)
    assert torch.allclose(ps[0], exp[0], atol=1e-5, rtol=1e-5)


def check_none_grad_param_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=1.0, betas=(0.9, 0.999), d0=1e-2, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动: {ps[1].tolist()}"


def check_returns_optimizer_instance(solve):
    ps = [torch.tensor([1.0, 2.0])]
    opt = solve(ps, lr=1.0)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
