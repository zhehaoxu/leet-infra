import math

import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_adabelief(params, grads_seq, lr, betas, eps, wd):
    """独立参照：向量化重算（与 Optimizer.step 不同的实现路径）。"""
    b1, b2 = betas
    ps = [p.clone() for p in params]
    ms = [torch.zeros_like(p) for p in ps]
    ss = [torch.zeros_like(p) for p in ps]
    for t, gs in enumerate(grads_seq, start=1):
        bc1 = 1.0 - b1 ** t
        bc2sq = math.sqrt(1.0 - b2 ** t)
        step_size = lr / bc1
        for i, (p, g) in enumerate(zip(ps, gs)):
            if wd != 0:
                p.mul_(1 - lr * wd)
            ms[i] = b1 * ms[i] + (1 - b1) * g
            diff = g - ms[i]
            ss[i] = b2 * ss[i] + (1 - b2) * diff * diff + eps
            denom = torch.sqrt(ss[i]) / bc2sq + eps
            p.sub_(step_size * ms[i] / denom)
    return ps


def check_example(solve):
    ps = [torch.tensor([1.0, -2.0, 0.5])]
    grads = [
        [torch.tensor([0.1, -0.2, 0.3])],
        [torch.tensor([-0.4, 0.05, 0.2])],
        [torch.tensor([0.15, -0.1, -0.25])],
    ]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-16, weight_decay=0.01)
    _step_all(opt, grads)
    exp = torch.tensor([0.9667389392852783, -1.765122890472412, 0.24798846244812012])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_matches_oracle_mixed_dims(solve):
    torch.manual_seed(42)
    p0 = [torch.randn(3, 4), torch.randn(5)]  # 2D 权重 + 1D 偏置混合
    gseq = [[torch.randn_like(p) * 0.4 for p in p0] for _ in range(20)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=2e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0.05)
    _step_all(opt, gseq)
    exp = _oracle_adabelief([p.clone() for p in p0], gseq, 2e-3, (0.9, 0.999), 1e-16, 0.05)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参照:\n{a}\nvs\n{b}"


def check_zero_decay(solve):
    torch.manual_seed(7)
    p0 = [torch.randn(4, 2)]
    gseq = [[torch.randn_like(p) * 0.3 for p in p0] for _ in range(12)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=5e-3, betas=(0.9, 0.98), eps=1e-16, weight_decay=0.0)
    _step_all(opt, gseq)
    exp = _oracle_adabelief([p.clone() for p in p0], gseq, 5e-3, (0.9, 0.98), 1e-16, 0.0)
    assert torch.allclose(ps[0], exp[0], atol=1e-5, rtol=1e-5)


def check_none_grad_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-16, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), "无梯度参数不应被改动"


def check_returns_optimizer_instance(solve):
    opt = solve([torch.randn(3, 3)], lr=0.01)
    assert isinstance(opt, torch.optim.Optimizer)
