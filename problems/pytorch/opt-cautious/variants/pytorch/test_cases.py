import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _adamw_direction(m, v, bc1, bc2, eps):
    denom = torch.sqrt(v) / (bc2 ** 0.5) + eps
    return (m / bc1) / denom


def _oracle_cautious(params, grads_seq, lr, betas, eps, weight_decay):
    """独立重算：AdamW 方向 + 谨慎符号掩码。"""
    b1, b2 = betas
    ps = [p.clone() for p in params]
    ms = [torch.zeros_like(p) for p in ps]
    vs = [torch.zeros_like(p) for p in ps]
    for t, gs in enumerate(grads_seq, start=1):
        bc1 = 1 - b1 ** t
        bc2 = 1 - b2 ** t
        for i, (p, g) in enumerate(zip(ps, gs)):
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)
            ms[i] = b1 * ms[i] + (1 - b1) * g
            vs[i] = b2 * vs[i] + (1 - b2) * g * g
            update = _adamw_direction(ms[i], vs[i], bc1, bc2, eps)
            mask = (update * g > 0).to(g.dtype)
            mask = mask * (mask.numel() / (mask.sum() + 1))
            p -= lr * update * mask
    return ps


def _oracle_adamw_nomask(params, grads_seq, lr, betas, eps, weight_decay):
    """不加掩码的 AdamW，用于对照谨慎掩码是否真的改变了结果。"""
    b1, b2 = betas
    ps = [p.clone() for p in params]
    ms = [torch.zeros_like(p) for p in ps]
    vs = [torch.zeros_like(p) for p in ps]
    for t, gs in enumerate(grads_seq, start=1):
        bc1 = 1 - b1 ** t
        bc2 = 1 - b2 ** t
        for i, (p, g) in enumerate(zip(ps, gs)):
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)
            ms[i] = b1 * ms[i] + (1 - b1) * g
            vs[i] = b2 * vs[i] + (1 - b2) * g * g
            update = _adamw_direction(ms[i], vs[i], bc1, bc2, eps)
            p -= lr * update
    return ps


def check_example(solve):
    ps = [torch.tensor([1.0, -2.0, 0.5])]
    grads = [
        [torch.tensor([0.1, -0.2, 0.3])],
        [torch.tensor([-0.4, 0.05, 0.2])],
        [torch.tensor([0.15, -0.1, -0.25])],
    ]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    _step_all(opt, grads)
    exp = torch.tensor([0.9780473709106445, -1.8283348083496094, 0.3267132341861725])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"


check_example.example = True


def check_matches_oracle_mixed_dims(solve):
    torch.manual_seed(8)
    p0 = [torch.randn(6, 4), torch.randn(5)]
    gseq = [[torch.randn_like(p) * 0.4 for p in p0] for _ in range(25)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=5e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)
    _step_all(opt, gseq)
    exp = _oracle_cautious([p.clone() for p in p0], gseq, 5e-3, (0.9, 0.999), 1e-8, 0.02)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参考:\n{a}\nvs\n{b}"


def check_cautious_mask_changes_result(solve):
    torch.manual_seed(123)
    p0 = [torch.randn(50)]
    gseq = [[torch.randn_like(p) * 0.5 for p in p0] for _ in range(20)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=0.02, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    _step_all(opt, gseq)
    nomask = _oracle_adamw_nomask([p.clone() for p in p0], gseq, 0.02, (0.9, 0.999), 1e-8, 0.0)
    assert not torch.allclose(ps[0], nomask[0], atol=1e-4), "掩码未生效：结果应与不加掩码的 AdamW 不同"


def check_none_grad_param_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.1])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动: {ps[1].tolist()}"


def check_returns_optimizer_instance(solve):
    ps = [torch.tensor([1.0, 2.0])]
    opt = solve(ps, lr=0.1)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
