import torch


def _step_all(opt, grads_seq):
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _oracle_clipaccum(params, grads_seq, lr, momentum, accum, max_norm):
    """独立重算：list 版，裁剪系数用 Python float 的 min(...)（结构异于参考解的张量 clamp）。"""
    ps = [p.clone() for p in params]
    acc = [torch.zeros_like(p) for p in ps]
    buf = [torch.zeros_like(p) for p in ps]
    have = [False] * len(ps)
    micro = 0
    for gs in grads_seq:
        for i, g in enumerate(gs):
            if g is None:
                continue
            acc[i] = acc[i] + g
            have[i] = True
        micro += 1
        if micro < accum:
            continue
        idxs = [i for i in range(len(ps)) if have[i]]
        sq = 0.0
        for i in idxs:
            sq += float((acc[i] / accum).pow(2).sum())
        total = sq ** 0.5
        coef = min(max_norm / (total + 1e-6), 1.0)
        for i in idxs:
            ge = (acc[i] / accum) * coef
            buf[i] = momentum * buf[i] + ge
            ps[i] = ps[i] - lr * buf[i]
            acc[i] = torch.zeros_like(acc[i])
        micro = 0
    return ps


def check_example(solve):
    torch.manual_seed(7)
    p0 = [torch.randn(3, 2), torch.randn(4)]
    gseq = [[torch.randn_like(p) * 2.0 for p in p0] for _ in range(12)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=0.1, momentum=0.9, accum_steps=4, max_norm=1.0)
    _step_all(opt, gseq)
    exp1 = torch.tensor([[-0.21124641597270966, 0.8275884985923767],
                         [0.988394558429718, -0.9523070454597473],
                         [1.6206506490707397, -0.9328624606132507]])
    exp2 = torch.tensor([-0.5245463848114014, 1.2102464437484741, 0.03376564010977745, -1.750956654548645])
    assert torch.allclose(ps[0], exp1, atol=1e-5, rtol=1e-5), f"p1 got {ps[0].tolist()}"
    assert torch.allclose(ps[1], exp2, atol=1e-5, rtol=1e-5), f"p2 got {ps[1].tolist()}"


check_example.example = True


def check_matches_oracle_clipped(solve):
    torch.manual_seed(11)
    p0 = [torch.randn(2, 3), torch.randn(5), torch.randn(1, 1, 4)]
    gseq = [[torch.randn_like(p) * 3.0 for p in p0] for _ in range(10)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=0.05, momentum=0.9, accum_steps=3, max_norm=1.0)
    _step_all(opt, gseq)
    exp = _oracle_clipaccum([p.clone() for p in p0], gseq, 0.05, 0.9, 3, 1.0)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离参考:\n{a}\nvs\n{b}"


def check_partial_micro_no_update(solve):
    p0 = [torch.tensor([1.0, 2.0, 3.0])]
    gseq = [[torch.tensor([0.5, -0.5, 1.0])] for _ in range(2)]  # 只喂 2 个微批次，accum=3
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=0.1, momentum=0.9, accum_steps=3, max_norm=1.0)
    _step_all(opt, gseq)
    assert torch.equal(ps[0], p0[0]), f"未凑满 accum_steps 不应更新: got {ps[0].tolist()}"


def check_none_grad_param_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, momentum=0.0, accum_steps=1, max_norm=1.0)
    opt.param_groups[0]["params"][0].grad = torch.tensor([0.1, -0.2])
    opt.step()
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动: {ps[1].tolist()}"


def check_returns_optimizer_instance(solve):
    ps = [torch.tensor([1.0, 2.0])]
    opt = solve(ps, lr=0.1)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
