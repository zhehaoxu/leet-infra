import torch


def _step_all(opt, grads_seq, hess_seq):
    """按 (梯度, Hessian 估计) 序列驱动优化器：两者都挂好后 step(hessians=...)。

    hess_seq[i] 为 None 表示第 i 步不做 Hessian 估算（H 冻结）。
    """
    params = opt.param_groups[0]["params"]
    for gs, hs in zip(grads_seq, hess_seq):
        for p, g in zip(params, gs):
            p.grad = g
        opt.step(hessians=None if hs is None else [h.clone() for h in hs])


def _oracle_sophia(params0, grads_seq, hess_seq, lr, betas, eps, wd, rho):
    """独立纯 torch 重算：不用 Optimizer 类，按 Algorithm 3 逐元素写（m/H 用新张量、
    除法与夹紧分两步），与参考解的算子顺序不同。"""
    ps = [p.clone() for p in params0]
    b1, b2 = betas
    m = [torch.zeros_like(p) for p in ps]
    h = [torch.zeros_like(p) for p in ps]
    for gs, hs in zip(grads_seq, hess_seq):
        for i, p in enumerate(ps):
            if wd != 0:
                p.mul_(1.0 - lr * wd)
            m[i] = b1 * m[i] + (1.0 - b1) * gs[i]
            if hs is not None:
                h[i] = b2 * h[i] + (1.0 - b2) * hs[i]
            u = m[i] / torch.clamp(h[i], min=eps)
            p -= lr * u.clamp(-rho, rho)
    return ps


def check_example(solve):
    ps = [torch.tensor([1.0, -2.0, 0.5])]
    grads = [
        [torch.tensor([0.1, -0.2, 0.3])],
        [torch.tensor([-0.4, 0.05, 0.2])],
        [torch.tensor([0.15, -0.1, -0.25])],
    ]
    hess = [
        [torch.tensor([0.5, 1.0, 0.25])],
        [torch.tensor([0.5, 0.5, 0.5])],
        [torch.tensor([2.0, 1.0, 0.1])],
    ]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1, rho=0.5)
    _step_all(opt, grads, hess)
    exp = torch.tensor([1.0140093564987183, -1.7920931577682495, 0.3366444706916809])
    assert torch.allclose(ps[0], exp, atol=1e-5, rtol=1e-5), f"got {ps[0].tolist()}"
    ref = _oracle_sophia([torch.tensor([1.0, -2.0, 0.5])], grads, hess, 0.1, (0.9, 0.99), 1e-8, 0.1, 0.5)
    assert torch.allclose(ps[0], ref[0], atol=1e-7, rtol=1e-7), f"与独立重算不符: {ps[0]} vs {ref[0]}"


check_example.example = True


def check_matches_oracle_mixed_dims(solve):
    torch.manual_seed(42)
    p0 = [torch.randn(3, 4), torch.randn(5)]
    n = 20
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(n)]
    # 每 3 步才估算一次 Hessian，其余步 H 冻结
    hseq = []
    for i in range(n):
        if i % 3 == 0:
            hseq.append([torch.rand_like(p) * 2.0 + 0.05 for p in p0])
        else:
            hseq.append(None)
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=5e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01, rho=0.3)
    _step_all(opt, gseq, hseq)
    exp = _oracle_sophia([p.clone() for p in p0], gseq, hseq, 5e-2, (0.9, 0.99), 1e-8, 0.01, 0.3)
    for a, b in zip(ps, exp):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), f"偏离独立重算:\n{a}\nvs\n{b}"


def check_hessian_is_linear_not_sqrt(solve):
    """分母是 H 的一次方：把 Ĥ 整体翻倍 ⇒ 未夹紧时的位移精确减半（√H 会得到 1/√2）。"""
    ps_a = [torch.tensor([1.0, -1.0, 2.0])]
    ps_b = [torch.tensor([1.0, -1.0, 2.0])]
    g = [[torch.tensor([0.6, -0.3, 0.9])]]
    h1 = [[torch.tensor([0.5, 2.0, 1.0])]]
    h2 = [[torch.tensor([1.0, 4.0, 2.0])]]
    kw = dict(lr=0.1, betas=(0.0, 0.5), eps=1e-8, weight_decay=0.0, rho=1e9)
    _step_all(solve(ps_a, **kw), g, h1)
    _step_all(solve(ps_b, **kw), g, h2)
    da = ps_a[0] - torch.tensor([1.0, -1.0, 2.0])
    db = ps_b[0] - torch.tensor([1.0, -1.0, 2.0])
    ratio = (da / db).abs()
    assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=1e-5, rtol=1e-5), \
        f"H 翻倍应让位移减半（比值 2），实得 {ratio.tolist()}（若写成 √H 会得 1.414）"


def check_clip_binds_and_bounds_step(solve):
    """eps 兜底 ⇒ H≈0 时比值爆掉、被 ρ 夹住：位移恰为 lr·ρ 且方向由 m 的符号决定。"""
    p0 = torch.tensor([1.0, 1.0, -1.0, -1.0])
    ps = [p0.clone()]
    grads = [[torch.tensor([0.5, -0.5, 0.5, -0.5])]]
    hess = [[torch.tensor([0.0, 0.0, 0.0, 0.0])]]
    opt = solve(ps, lr=0.2, betas=(0.0, 0.9), eps=1e-6, weight_decay=0.0, rho=0.05)
    _step_all(opt, grads, hess)
    delta = (ps[0] - p0).abs()
    assert torch.allclose(delta, torch.full_like(delta, 0.01), atol=1e-6, rtol=1e-6), f"位移应为 lr·rho: {delta}"
    assert torch.all((ps[0] - p0) * torch.tensor([-1.0, 1.0, -1.0, 1.0]) > 0), \
        f"更新方向应逆着 m 的符号: {(ps[0]-p0).tolist()}"

    # 不变量：任意步 |Δp| 不超过 lr·rho（无权重衰减时）
    torch.manual_seed(3)
    q0 = [torch.randn(6)]
    gs = [[torch.randn(6) * 5.0] for _ in range(10)]
    hs = [[torch.rand(6) * 0.02] for _ in range(10)]
    before = q0[0].clone()
    o2 = solve(q0, lr=0.03, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, rho=0.1)
    _step_all(o2, gs, hs)
    assert (q0[0] - before).abs().max().item() <= 10 * 0.03 * 0.1 + 1e-5, "每步位移必须被 lr·rho 一致限界（float32 舍入留 1e-5 余量）"


def check_degenerates_to_sgd(solve):
    """β1=0（无动量）、β2=0（H 直接等于喂入的 Ĥ≡1，不经 EMA）、ρ 极大、wd=0 时逐位退化为原生 SGD。

    注意必须取 β2=0：EMA 从 0 起步，β2=0.99 时 H 前几步只有 0.01 量级，退化不成立。
    """
    torch.manual_seed(11)
    p0 = [torch.randn(4, 3)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(15)]
    hseq = [[torch.ones_like(p) for p in p0] for _ in range(15)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, lr=0.05, betas=(0.0, 0.0), eps=1e-8, weight_decay=0.0, rho=1e12)
    _step_all(opt, gseq, hseq)
    ref = [p.clone() for p in p0]
    sgd = torch.optim.SGD(ref, lr=0.05)
    for gs in gseq:
        for p, g in zip(ref, gs):
            p.grad = g
        sgd.step()
    assert torch.allclose(ps[0], ref[0], atol=1e-7, rtol=1e-7), f"应等价于 SGD:\n{ps[0]}\nvs\n{ref[0]}"


def check_hessian_frozen_when_not_supplied(solve):
    """hessians=None 的那一步：H 保持不动，但 m 与 p 照常更新（论文的 k 步复用）。"""
    grads = [[torch.tensor([0.4, -0.4])], [torch.tensor([0.0, 0.0])], [torch.tensor([0.0, 0.0])]]
    hess_given = [[torch.tensor([0.5, 0.5])], None, None]
    ps = [torch.tensor([1.0, 1.0])]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, rho=1.0)
    _step_all(opt, grads, hess_given)
    # H 只被估算一次：H=(1-b2)*0.5，之后两步冻结；m 继续按 β1 衰减
    ref = _oracle_sophia([torch.tensor([1.0, 1.0])], grads, hess_given, 0.1, (0.9, 0.99), 1e-8, 0.0, 1.0)
    assert torch.allclose(ps[0], ref[0], atol=1e-7, rtol=1e-7), f"{ps[0]} vs {ref[0]}"
    assert ps[0][0].item() < 1.0, "只估算一次 Hessian 后仍应继续下降"


def check_none_grad_param_skipped(solve):
    ps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    opt = solve(ps, lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1, rho=1.0)
    params = opt.param_groups[0]["params"]
    params[0].grad = torch.tensor([0.1, -0.1])
    opt.step(hessians=[torch.tensor([0.5, 0.5]), torch.zeros(2)])
    assert torch.equal(ps[1], torch.tensor([3.0, 4.0])), f"无梯度参数被改动: {ps[1].tolist()}"


def check_returns_optimizer_instance(solve):
    opt = solve([torch.tensor([1.0, 2.0])], lr=0.1)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
