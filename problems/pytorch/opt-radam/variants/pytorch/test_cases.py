import math

import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_radam(params, grads_seq, lr, betas, eps, wd, thr=5):
    """独立参照：向量化重算（与 Optimizer.step 不同的实现路径）。"""
    b1, b2 = betas
    ps = [p.clone() for p in params]
    ms = [torch.zeros_like(p) for p in ps]
    vs = [torch.zeros_like(p) for p in ps]
    n_sma_max = 2.0 / (1.0 - b2) - 1.0
    for t, gs in enumerate(grads_seq, start=1):
        b2t = b2 ** t
        n_sma = n_sma_max - 2 * t * b2t / (1.0 - b2t)
        rect = n_sma >= thr
        rt = math.sqrt((1 - b2t) * (n_sma - 4) / (n_sma_max - 4) * (n_sma - 2) / n_sma * n_sma_max / (n_sma_max - 2)) if rect else -1.0
        step_size = lr * rt / (1.0 - b1 ** t)
        for i, (p, g) in enumerate(zip(ps, gs)):
            if not (step_size > 0 or rect):
                continue
            if wd != 0:
                p.mul_(1 - lr * wd)
            ms[i] = b1 * ms[i] + (1 - b1) * g
            vs[i] = b2 * vs[i] + (1 - b2) * g * g
            if rect:
                p.sub_(step_size * ms[i] / (torch.sqrt(vs[i]) + eps))
            elif step_size > 0:
                p.sub_(step_size * ms[i])
    return ps


def check_example(solve):
    ps = [torch.tensor([1.0, -2.0, 0.5])]
    gs = [[0.1, -0.2, 0.3], [-0.4, 0.05, 0.2], [0.15, -0.1, -0.25], [0.05, 0.3, -0.1],
          [-0.2, -0.1, 0.4], [0.3, 0.1, -0.2], [-0.1, 0.25, 0.05], [0.2, -0.15, -0.3]]
    grads = [[torch.tensor(x)] for x in gs]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, n_sma_threshold=5)
    _step_all(opt, grads)
    exp = torch.tensor([0.9929537773132324, -1.9984153509140015, 0.5030925273895264])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_warmup_is_noop(solve):
    # β2=0.999 时前 4 步 n_t<5 → 完全 no-op（含权重衰减也不做），p 应逐元素不变
    p0 = torch.tensor([1.0, -2.0, 0.5])
    ps = [p0.clone()]
    grads = [[torch.tensor([0.1, -0.2, 0.3])] for _ in range(4)]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1, n_sma_threshold=5)
    _step_all(opt, grads)
    assert torch.equal(ps[0], p0), f"前 4 步应完全不动，实际 {ps[0].tolist()}"


def check_matches_oracle(solve):
    torch.manual_seed(42)
    p0 = [torch.randn(3, 4), torch.randn(5)]
    gseq = [[torch.randn_like(p) * 0.4 for p in p0] for _ in range(20)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, n_sma_threshold=5)
    _step_all(opt, gseq)
    exp = _oracle_radam([p.clone() for p in p0], gseq, 2e-3, (0.9, 0.999), 1e-8, 0.05)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参照:\n{a}\nvs\n{b}"


def check_none_grad_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), "无梯度参数不应被改动"


def check_returns_optimizer_instance(solve):
    opt = solve([torch.randn(3, 3)], lr=0.01)
    assert isinstance(opt, torch.optim.Optimizer)
