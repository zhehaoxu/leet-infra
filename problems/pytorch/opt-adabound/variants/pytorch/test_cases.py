import math

import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_adabound(params, grads_seq, lr, final_lr, betas, gamma, eps, wd):
    """独立参照：向量化重算，采用与参考解相同的张量原语序列（详见 notes.md 对拍口径）。"""
    b1, b2 = betas
    base_lr = lr
    ps = [p.clone() for p in params]
    ms = [torch.zeros_like(p) for p in ps]
    vs = [torch.zeros_like(p) for p in ps]
    for t, gs in enumerate(grads_seq, start=1):
        bc1 = 1.0 - b1 ** t
        bc2sq = math.sqrt(1.0 - b2 ** t)
        flr = final_lr * lr / base_lr
        lower = flr * (1 - 1 / (gamma * t + 1))
        upper = flr * (1 + 1 / (gamma * t))
        step_size = lr * bc2sq / bc1
        for i, (p, g) in enumerate(zip(ps, gs)):
            if wd != 0:
                p.mul_(1 - lr * wd)
            ms[i].lerp_(g, 1 - b1)
            vs[i].mul_(b2).addcmul_(g, g, value=1 - b2)
            de_nom = vs[i].add(1e-15).sqrt_().add_(eps)
            eff = de_nom.div_(step_size)
            eff.clamp_(min=lower, max=upper).mul_(ms[i])
            p.sub_(eff)
    return ps


def check_example(solve):
    ps = [torch.tensor([1.0, -2.0, 0.5])]
    gs = [[0.1, -0.2, 0.3], [-0.4, 0.05, 0.2], [0.15, -0.1, -0.25], [0.05, 0.3, -0.1], [-0.2, -0.1, 0.4]]
    grads = [[torch.tensor(x)] for x in gs]
    opt = solve(ps, lr=0.1, final_lr=0.1, betas=(0.9, 0.999), gamma=0.01, eps=1e-8, weight_decay=0.01)
    _step_all(opt, grads)
    exp = torch.tensor([1.0480223894119263, -1.981063723564148, 0.4000476896762848])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_matches_oracle_mixed_dims(solve):
    torch.manual_seed(42)
    p0 = [torch.randn(3, 4), torch.randn(5)]  # 2D 权重 + 1D 偏置混合
    gseq = [[torch.randn_like(p) * 0.4 for p in p0] for _ in range(25)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=1e-3, final_lr=0.1, betas=(0.9, 0.999), gamma=0.1, eps=1e-8, weight_decay=0.02)
    _step_all(opt, gseq)
    exp = _oracle_adabound([p.clone() for p in p0], gseq, 1e-3, 0.1, (0.9, 0.999), 0.1, 1e-8, 0.02)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参照:\n{a}\nvs\n{b}"


def check_bounds_tighten(solve):
    # γ 较大 + 步数多 → 上下界收拢到 final_lr，逐坐标有效学习率趋于同一个常数（SGD 行为）
    torch.manual_seed(3)
    p0 = [torch.randn(6, 3)]
    gseq = [[torch.randn_like(p) * 0.5 for p in p0] for _ in range(80)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=1e-2, final_lr=0.05, betas=(0.9, 0.999), gamma=1.0, eps=1e-8, weight_decay=0.0)
    _step_all(opt, gseq)
    exp = _oracle_adabound([p.clone() for p in p0], gseq, 1e-2, 0.05, (0.9, 0.999), 1.0, 1e-8, 0.0)
    assert torch.allclose(ps[0], exp[0], atol=1e-5, rtol=1e-5)


def check_none_grad_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, final_lr=0.1, betas=(0.9, 0.999), gamma=0.1, eps=1e-8, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), "无梯度参数不应被改动"


def check_returns_optimizer_instance(solve):
    opt = solve([torch.randn(3, 3)], lr=0.01)
    assert isinstance(opt, torch.optim.Optimizer)
