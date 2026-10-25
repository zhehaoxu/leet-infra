import math

import torch
from torch.nn.functional import cosine_similarity


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _channel_view(x):
    return x.view(x.size()[0], -1)


def _layer_view(x):
    return x.view(1, -1)


def _cos_by_view(x, y, eps, view_func):
    x = view_func(x)
    y = view_func(y)
    return cosine_similarity(x, y, dim=1, eps=eps).abs_()


def _projection(p, grad, perturb, delta, wd_ratio, eps):
    """与参考解共享的投影判据（离散分支须一致）。"""
    wd = 1.0
    expand_size = [-1] + [1] * (len(p.shape) - 1)
    for view_func in (_channel_view, _layer_view):
        cosine_sim = _cos_by_view(grad, p, eps, view_func)
        if cosine_sim.max() < delta / math.sqrt(view_func(p).size()[1]):
            p_n = p / view_func(p).norm(dim=1).view(expand_size).add_(eps)
            radial = view_func(p_n * perturb).sum(dim=1).view(expand_size)
            perturb = perturb - p_n * radial
            wd = wd_ratio
            return perturb, wd
    return perturb, wd


def _oracle_adamp(params, grads_seq, lr, betas, eps, wd, delta, wd_ratio):
    """独立参照：向量化重算 Adam 矩（显式代数）+ 共享投影判据。"""
    b1, b2 = betas
    ps = [p.clone() for p in params]
    ms = [torch.zeros_like(p) for p in ps]
    vs = [torch.zeros_like(p) for p in ps]
    for t, gs in enumerate(grads_seq, start=1):
        bc1 = 1.0 - b1 ** t
        bc2sq = math.sqrt(1.0 - b2 ** t)
        step_size = lr / bc1
        for i, (p, g) in enumerate(zip(ps, gs)):
            ms[i] = b1 * ms[i] + (1 - b1) * g
            vs[i] = b2 * vs[i] + (1 - b2) * g * g
            perturb = bc2sq / (torch.sqrt(vs[i]) + eps) * ms[i]
            ratio = 1.0
            if p.dim() > 1:
                perturb, ratio = _projection(p, g, perturb, delta, wd_ratio, eps)
            if wd != 0:
                p.mul_(1 - lr * wd * ratio)
            p.sub_(step_size * perturb)
    return ps


def check_example(solve):
    ps = [torch.tensor([[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]])]
    gs = [
        [[0.2, -0.1, 0.05], [0.1, 0.3, -0.2]],
        [[-0.15, 0.2, 0.1], [0.05, -0.1, 0.3]],
        [[0.1, 0.05, -0.2], [-0.3, 0.15, 0.1]],
        [[0.25, -0.05, 0.1], [0.1, -0.2, -0.15]],
        [[-0.1, 0.3, -0.05], [0.2, 0.1, 0.2]],
    ]
    grads = [[torch.tensor(x)] for x in gs]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, delta=0.1, wd_ratio=0.1)
    _step_all(opt, grads)
    exp = torch.tensor([[0.26863083243370056, -0.252073734998703, -0.08219829946756363],
                        [0.12958168983459473, 0.17102918028831482, -0.09375680238008499]])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_matches_oracle_mixed_dims(solve):
    torch.manual_seed(42)
    p0 = [torch.randn(3, 4), torch.randn(5), torch.randn(2, 3, 4)]  # 2D + 1D(不投影) + 高维
    gseq = [[torch.randn_like(p) * 0.4 for p in p0] for _ in range(20)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, delta=0.1, wd_ratio=0.1)
    _step_all(opt, gseq)
    exp = _oracle_adamp([p.clone() for p in p0], gseq, 2e-3, (0.9, 0.999), 1e-8, 0.05, 0.1, 0.1)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参照:\n{a}\nvs\n{b}"


def check_1d_param_not_projected(solve):
    # 一维参数跳过投影，r 恒为 1.0，等价纯 Adam 矩 + 完整权重衰减
    torch.manual_seed(7)
    p0 = [torch.randn(6)]
    gseq = [[torch.randn_like(p) * 0.3 for p in p0] for _ in range(12)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=5e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02, delta=0.1, wd_ratio=0.1)
    _step_all(opt, gseq)
    exp = _oracle_adamp([p.clone() for p in p0], gseq, 5e-3, (0.9, 0.999), 1e-8, 0.02, 0.1, 0.1)
    assert torch.allclose(ps[0], exp[0], atol=1e-5, rtol=1e-5)


def check_none_grad_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), "无梯度参数不应被改动"


def check_returns_optimizer_instance(solve):
    opt = solve([torch.randn(3, 3)], lr=0.01)
    assert isinstance(opt, torch.optim.Optimizer)
