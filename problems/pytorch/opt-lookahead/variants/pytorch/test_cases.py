import math

import torch


def _drive(opt, grads_seq):
    """按固定梯度序列驱动优化器：每步把梯度挂到参数对象上再 step()（None 表示该参数这步无梯度）。"""
    for gs in grads_seq:
        for p, g in zip(opt.param_groups[0]["params"], gs):
            p.grad = g
        opt.step()


def _native_trace(base, params, grads_seq, lr, momentum, betas, eps):
    """裸 base 优化器跑同一条梯度序列，返回 (优化器, 每一步后的参数快照列表)。"""
    ps = [p.clone() for p in params]
    if base == "sgd":
        opt = torch.optim.SGD(ps, lr=lr, momentum=momentum)
    else:
        opt = torch.optim.Adam(ps, lr=lr, betas=betas, eps=eps)
    traj = []
    for gs in grads_seq:
        for p, g in zip(ps, gs):
            p.grad = g
        opt.step()
        traj.append([p.detach().clone() for p in ps])
    return opt, traj


def _native_base(base, params, grads_seq, lr, momentum, betas, eps):
    """裸 base 优化器（不套 Lookahead）跑同一条梯度序列，返回 (优化器, 最终参数克隆列表)。"""
    opt, traj = _native_trace(base, params, grads_seq, lr, momentum, betas, eps)
    return opt, traj[-1]


def _base_deltas(base, params, grads_seq, lr, momentum, betas, eps):
    """独立手算 base 优化器的**逐步增量** d_t（纯 Python 循环，不建任何 Optimizer 对象）。

    本题的 base 不带权重衰减、不带 nesterov/dampening、Adam 不开 amsgrad，
    于是 d_t 只由梯度序列与该参数的有效步数决定、与当前参数值无关——这正是能把整条
    d 序列先单独算出来、再做 Lookahead 账本的原因（与参考解「真优化器 + 就地写回」结构不同）。
    """
    b1, b2 = betas
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    seen = [0] * len(params)
    deltas = [[None] * len(grads_seq) for _ in params]
    for t, gs in enumerate(grads_seq):
        for i, g in enumerate(gs):
            if g is None:
                continue  # base 跳过无梯度参数：既不更新值，也不推进它自己的 step 计数
            seen[i] += 1
            if base == "sgd":
                m[i] = momentum * m[i] + g
                deltas[i][t] = -lr * m[i]
            else:
                bc1 = 1.0 - b1 ** seen[i]
                bc2 = 1.0 - b2 ** seen[i]
                m[i] = b1 * m[i] + (1.0 - b1) * g
                v[i] = b2 * v[i] + (1.0 - b2) * (g * g)
                denom = torch.sqrt(v[i]) / math.sqrt(bc2) + eps
                deltas[i][t] = -(lr / bc1) * (m[i] / denom)
    return deltas


def _oracle_lookahead(params, grads_seq, base="sgd", lr=0.05, momentum=0.9, k=2, alpha=0.5,
                      betas=(0.9, 0.999), eps=1e-8):
    """独立重算 Lookahead：先算增量序列，再用「非就地 + 张量重绑定」的方式走同步账本。"""
    deltas = _base_deltas(base, params, grads_seq, lr, momentum, betas, eps)
    fast = [p.detach().clone() for p in params]
    slow = [p.detach().clone() for p in params]
    for t in range(len(grads_seq)):
        for i in range(len(params)):
            if deltas[i][t] is not None:
                fast[i] = fast[i] + deltas[i][t]
        if k > 0 and (t + 1) % k == 0:
            for i in range(len(params)):
                slow[i] = slow[i] + alpha * (fast[i] - slow[i])
                fast[i] = slow[i].clone()
    return fast, slow


def _oracle_snapshots(params, grads_seq, **cfg):
    """独立重算的**逐步**快照：返回第 t 步后的 (训练权重列表, 慢权重列表)。"""
    deltas = _base_deltas(cfg.get("base", "sgd"), params, grads_seq, cfg.get("lr", 0.05),
                          cfg.get("momentum", 0.9), cfg.get("betas", (0.9, 0.999)), cfg.get("eps", 1e-8))
    k, alpha = cfg.get("k", 2), cfg.get("alpha", 0.5)
    fast = [p.detach().clone() for p in params]
    slow = [p.detach().clone() for p in params]
    shots = []
    for t in range(len(grads_seq)):
        for i in range(len(params)):
            if deltas[i][t] is not None:
                fast[i] = fast[i] + deltas[i][t]
        if k > 0 and (t + 1) % k == 0:
            for i in range(len(params)):
                slow[i] = slow[i] + alpha * (fast[i] - slow[i])
                fast[i] = slow[i].clone()
        shots.append(([f.clone() for f in fast], [s.clone() for s in slow]))
    return shots


def _close(a, b, atol=1e-5, rtol=1e-5):
    return torch.allclose(a, b, atol=atol, rtol=rtol)


_EXAMPLE_P0 = [1.0, -2.0, 0.5]
_EXAMPLE_G = [
    [0.4, -0.1, 0.25],
    [-0.2, 0.5, -0.1],
    [0.3, 0.2, -0.35],
    [-0.5, -0.3, 0.15],
]


def _example_grads():
    return [[torch.tensor(g)] for g in _EXAMPLE_G]


def check_example(solve):
    ps = [torch.tensor(_EXAMPLE_P0)]
    grads = _example_grads()
    opt = solve(ps, base="sgd", lr=0.1, momentum=0.9, k=2, alpha=0.5)
    _drive(opt, grads)
    exp = torch.tensor([0.9548200368881226, -2.0545549392700195, 0.4963124990463257])
    assert _close(ps[0], exp), f"示例结果不符: got {ps[0].tolist()}"
    ref, _ = _oracle_lookahead([torch.tensor(_EXAMPLE_P0)], grads, "sgd", 0.1, 0.9, 2, 0.5)
    assert _close(ps[0], ref[0]), f"示例与独立重算不符: {ps[0].tolist()} vs {ref[0].tolist()}"


check_example.example = True


def check_matches_oracle_sgd_trajectory(solve):
    torch.manual_seed(3)
    p0 = [torch.randn(5), torch.randn(3, 2)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(13)]
    ps = [p.clone() for p in p0]
    cfg = dict(base="sgd", lr=0.05, momentum=0.9, k=3, alpha=0.4)
    opt = solve(ps, **cfg)
    _drive(opt, gseq)
    ref, slow_ref = _oracle_lookahead([p.clone() for p in p0], gseq, **cfg)
    for i, (a, b) in enumerate(zip(ps, ref)):
        assert _close(a, b), f"参数 {i} 轨迹偏离独立重算:\n{a}\nvs\n{b}"
    for i, (a, b) in enumerate(zip(opt.slow_weights, slow_ref)):
        assert _close(a, b), f"慢权重 {i} 偏离独立重算:\n{a}\nvs\n{b}"


def check_matches_oracle_adam_trajectory(solve):
    torch.manual_seed(5)
    p0 = [torch.randn(4), torch.randn(2, 4)]
    gseq = [[torch.randn_like(p) * 0.5 for p in p0] for _ in range(11)]
    ps = [p.clone() for p in p0]
    cfg = dict(base="adam", lr=0.05, momentum=0.0, k=4, alpha=0.5, betas=(0.9, 0.99), eps=1e-8)
    opt = solve(ps, **cfg)
    _drive(opt, gseq)
    ref, slow_ref = _oracle_lookahead([p.clone() for p in p0], gseq, **cfg)
    for i, (a, b) in enumerate(zip(ps, ref)):
        assert _close(a, b), f"Adam 基座参数 {i} 偏离独立重算:\n{a}\nvs\n{b}"
    for i, (a, b) in enumerate(zip(opt.slow_weights, slow_ref)):
        assert _close(a, b), f"Adam 基座慢权重 {i} 偏离独立重算:\n{a}\nvs\n{b}"
    assert isinstance(opt.base_optimizer, torch.optim.Adam), "base='adam' 须内建 torch.optim.Adam"


def check_sync_lands_after_the_kth_fast_step(solve):
    """下标约定护栏：逐步对拍。同步必须落在第 k 个 fast 步**之后**，而不是下一次 step 的开头——
    只在末步恰好是同步点时才能发现的「错位一步」写法，会被这里每一步的比对抓到。"""
    torch.manual_seed(47)
    p0 = [torch.randn(4)]
    gseq = [[torch.randn_like(p0[0])] for _ in range(11)]  # k=4 → 同步发生在 t=4、8，末步 11 不是同步点
    cfg = dict(base="sgd", lr=0.05, momentum=0.9, k=4, alpha=0.5)
    ps = [p.clone() for p in p0]
    opt = solve(ps, **cfg)
    shots = _oracle_snapshots([p.clone() for p in p0], gseq, **cfg)
    for t, gs in enumerate(gseq, start=1):
        _drive(opt, [gs])
        fast_ref, slow_ref = shots[t - 1]
        assert _close(ps[0], fast_ref[0]), f"第 {t} 步的训练权重与独立重算不符: {ps[0].tolist()} vs {fast_ref[0].tolist()}"
        assert _close(opt.slow_weights[0], slow_ref[0]), \
            f"第 {t} 步的慢权重与独立重算不符: {opt.slow_weights[0].tolist()} vs {slow_ref[0].tolist()}"
        on_sync = t % 4 == 0
        same = torch.equal(ps[0], opt.slow_weights[0])
        assert same == on_sync, f"第 {t} 步{'是' if on_sync else '不是'}同步点，训练权重{'应' if on_sync else '不应'}逐位等于慢权重"


def check_base_state_survives_sync(solve):
    """核心探针：同步绝不重置 base 自身的状态（动量缓冲 / 二阶矩 / 它自己的 step 计数）。"""
    torch.manual_seed(17)
    p0 = [torch.randn(6)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(9)]  # k=3 → 同步 3 次

    ps = [p.clone() for p in p0]
    opt = solve(ps, base="sgd", lr=0.05, momentum=0.9, k=3, alpha=0.5)
    _drive(opt, gseq)
    native, _ = _native_base("sgd", p0, gseq, 0.05, 0.9, (0.9, 0.999), 1e-8)
    nt_p = native.param_groups[0]["params"][0]
    mine_buf = opt.base_optimizer.state[ps[0]]["momentum_buffer"]
    assert torch.count_nonzero(mine_buf) > 0, "SGD 动量缓冲被清零了：同步不该重置 base 状态"
    assert _close(mine_buf, native.state[nt_p]["momentum_buffer"], 1e-6, 1e-6), \
        f"动量缓冲与裸 SGD 不符:\n{mine_buf}\nvs\n{native.state[nt_p]['momentum_buffer']}"

    ps = [p.clone() for p in p0]
    opt = solve(ps, base="adam", lr=0.05, k=3, alpha=0.5)
    _drive(opt, gseq)
    native, _ = _native_base("adam", p0, gseq, 0.05, 0.0, (0.9, 0.999), 1e-8)
    nt_p = native.param_groups[0]["params"][0]
    st = opt.base_optimizer.state[ps[0]]
    assert _close(st["exp_avg"], native.state[nt_p]["exp_avg"], 1e-6, 1e-6), "Adam 一阶矩被同步污染"
    assert _close(st["exp_avg_sq"], native.state[nt_p]["exp_avg_sq"], 1e-6, 1e-6), "Adam 二阶矩被同步污染"
    assert float(st["step"]) == 9.0, f"Adam 自身 step 计数应仍是 9（未被同步打断），得 {st['step']}"


def check_no_sync_when_k_exceeds_steps(solve):
    """退化 1：k 大于总步数 → 从不同步，轨迹须逐点等于裸 base，慢权重仍是初值。"""
    torch.manual_seed(23)
    p0 = [torch.randn(4), torch.randn(2, 2)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(7)]
    ps = [p.clone() for p in p0]
    opt = solve(ps, base="sgd", lr=0.05, momentum=0.9, k=50, alpha=0.5)
    _drive(opt, gseq)
    _, ref = _native_base("sgd", p0, gseq, 0.05, 0.9, (0.9, 0.999), 1e-8)
    for i, (a, b) in enumerate(zip(ps, ref)):
        assert _close(a, b, 1e-6, 1e-6), f"k 大于步数时不该同步，参数 {i}: {a} vs {b}"
    for i, (s, p_init) in enumerate(zip(opt.slow_weights, p0)):
        assert torch.equal(s, p_init), f"没到同步点时慢权重应仍是初值（参数 {i}）: {s} vs {p_init}"


def check_alpha_one_equals_base(solve):
    """退化 2：alpha=1 → 慢更新把 s 抬到 θ、再写回同一个 θ，同步对训练轨迹是恒等。"""
    torch.manual_seed(29)
    p0 = [torch.randn(5), torch.randn(2, 3)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(8)]  # k=4 → 末步恰是同步点
    ps = [p.clone() for p in p0]
    opt = solve(ps, base="sgd", lr=0.05, momentum=0.9, k=4, alpha=1.0)
    _drive(opt, gseq)
    _, ref = _native_base("sgd", p0, gseq, 0.05, 0.9, (0.9, 0.999), 1e-8)
    for i, (a, b) in enumerate(zip(ps, ref)):
        assert _close(a, b, 1e-6, 1e-6), f"alpha=1 应与裸 base 一致，参数 {i}: {a} vs {b}"
    for i, (s, p) in enumerate(zip(opt.slow_weights, ps)):
        assert torch.equal(s, p), f"alpha=1 且刚同步过时慢权重应逐位等于训练权重（参数 {i}）"
    ref_opt, _ = _native_base("sgd", p0, gseq, 0.05, 0.9, (0.9, 0.999), 1e-8)
    assert _close(opt.base_optimizer.state[ps[0]]["momentum_buffer"],
                  ref_opt.state[ref_opt.param_groups[0]["params"][0]]["momentum_buffer"], 1e-6, 1e-6), \
        "alpha=1 时 base 状态同样不该被动过"


def check_alpha_zero_rolls_back(solve):
    """退化 3：alpha=0 → 慢权重永不吸收新东西，同步等于把 θ 硬回滚到上一次同步点。"""
    torch.manual_seed(31)
    p0 = [torch.randn(4)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(9)]
    init = p0[0].clone()

    ps = [p.clone() for p in p0]
    opt = solve(ps, base="sgd", lr=0.05, momentum=0.9, k=3, alpha=0.0)
    _drive(opt, gseq)  # 9 = 3 个整周期，末步恰是同步点
    assert torch.equal(ps[0], init), f"alpha=0 走满整周期后 θ 应被回滚到上一次同步点（=初值）: {ps[0].tolist()}"
    assert torch.count_nonzero(opt.base_optimizer.state[ps[0]]["momentum_buffer"]) > 0, \
        "回滚不该顺手清掉 base 的动量缓冲"
    for s in opt.slow_weights:
        assert torch.equal(s, init), f"alpha=0 时慢权重应始终等于初值: {s.tolist()}"

    ps = [p.clone() for p in p0]
    cfg = dict(base="sgd", lr=0.05, momentum=0.9, k=3, alpha=0.0)
    g8 = gseq[:6] + gseq[6:8]  # 8 步 = 2 个整周期 + 2 步零头
    opt = solve(ps, **cfg)
    _drive(opt, g8)
    ref, _ = _oracle_lookahead(p0, g8, **cfg)
    assert _close(ps[0], ref[0]), f"alpha=0 的锯齿尾巴与独立重算不符: {ps[0].tolist()} vs {ref[0].tolist()}"
    assert not _close(ps[0], init), "走完零头步后 θ 应已离开初值（同步只发生在整周期末）"


def check_k_one_scales_base_steps(solve):
    """退化 4：k=1 → 每步同步，等价于把 base 的每步增量整体缩放 alpha（θ_T = θ_0 + α·Σ d_t）。"""
    torch.manual_seed(37)
    lr, mom, alpha = 0.05, 0.9, 0.35
    p0 = [torch.randn(3), torch.randn(4, 1)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(12)]

    _, base_traj = _native_trace("sgd", p0, gseq, lr, mom, (0.9, 0.999), 1e-8)
    prev = [p.clone() for p in p0]
    acc = [p.clone() for p in p0]
    for snap in base_traj:  # d_t 只依赖梯度序列，可直接从裸 base 的轨迹差分里取
        for i, cur in enumerate(snap):
            acc[i] = acc[i] + alpha * (cur - prev[i])
            prev[i] = cur.clone()

    ps = [p.clone() for p in p0]
    opt = solve(ps, base="sgd", lr=lr, momentum=mom, k=1, alpha=alpha)
    _drive(opt, gseq)
    for i, (a, b) in enumerate(zip(ps, acc)):
        assert _close(a, b, 1e-6, 1e-6), f"k=1 应退化为步长缩放 α，参数 {i}: {a} vs {b}"
    for i, (s, p) in enumerate(zip(opt.slow_weights, ps)):
        assert torch.equal(s, p), f"k=1 时刚同步过的慢权重应逐位等于训练权重（参数 {i}）"


def check_slow_sequence_monotone_and_contained(solve):
    """不变量：同号梯度下慢权重序列单调，且慢权重始终落在训练权重取过值的凸包（min/max）内。"""
    torch.manual_seed(41)
    p0 = [torch.zeros(6) + 1.0]
    gseq = [[torch.rand_like(p0[0]) + 0.05] for _ in range(12)]  # 梯度恒正 → θ 单调下降
    ps = [p.clone() for p in p0]
    opt = solve(ps, base="sgd", lr=0.05, momentum=0.9, k=3, alpha=0.5)
    fast_seen = [ps[0].clone()]
    slow_at_sync = []
    for step_i, gs in enumerate(gseq, start=1):
        _drive(opt, [gs])
        fast_seen.append(ps[0].clone())
        if step_i % 3 == 0:
            slow_at_sync.append(opt.slow_weights[0].clone())
    assert len(slow_at_sync) == 4, f"12 步、k=3 应同步 4 次，实得 {len(slow_at_sync)}"
    for a, b in zip(slow_at_sync, slow_at_sync[1:]):
        assert bool((b < a).all()), f"同号梯度下慢权重应单调下降: {a.tolist()} -> {b.tolist()}"
    lo = torch.stack(fast_seen).amin(dim=0)
    hi = torch.stack(fast_seen).amax(dim=0)
    for s in slow_at_sync + [opt.slow_weights[0]]:
        inside = bool((s >= lo - 1e-6).all().item()) and bool((s <= hi + 1e-6).all().item())
        assert inside, f"慢权重跑出训练权重取过的范围（凸包）: {s.tolist()}"
    assert bool((fast_seen[-1] < fast_seen[0]).all()), "恒正梯度下 θ 应整体下行"


def check_none_grad_param_still_synced(solve):
    """同步由步计数驱动，与某个参数这一步有没有梯度无关：梯度断供的参数照样被拉回慢权重。"""
    torch.manual_seed(43)
    p0 = [torch.randn(4), torch.randn(3)]
    gseq = [[torch.randn_like(p) for p in p0] for _ in range(6)]
    for t in (4, 5):
        gseq[t][1] = None  # 第二个参数从第 5 步起没有梯度，但 t=6 仍是同步点（k=3）
    ps = [p.clone() for p in p0]
    cfg = dict(base="adam", lr=0.05, momentum=0.0, k=3, alpha=0.5, betas=(0.9, 0.99), eps=1e-8)
    opt = solve(ps, **cfg)
    _drive(opt, gseq)
    ref, slow_ref = _oracle_lookahead([p.clone() for p in p0], gseq, **cfg)
    for i, (a, b) in enumerate(zip(ps, ref)):
        assert _close(a, b), f"参数 {i} 与独立重算不符:\n{a}\nvs\n{b}"
    for i, (a, b) in enumerate(zip(opt.slow_weights, slow_ref)):
        assert _close(a, b), f"慢权重 {i} 与独立重算不符:\n{a}\nvs\n{b}"
    st = opt.base_optimizer.state[ps[1]]
    assert float(st["step"]) == 4.0, f"无梯度参数的 base 自身计数应停在 4，得 {st['step']}"
    # 「只同步这一步有梯度的参数」这一常见错法会让该参数在第 6 步的同步里被漏掉、停在第 5 步的值上
    frozen = _oracle_lookahead([p.clone() for p in p0], gseq[:5], **cfg)[0][1]
    assert not _close(ps[1], frozen, 1e-5, 1e-5), \
        f"无梯度参数未被同步拉回：现值 {ps[1].tolist()} 与「同步漏掉它」的 {frozen.tolist()} 相同"


def check_returns_optimizer_instance(solve):
    ps = [torch.tensor([1.0, 2.0])]
    opt = solve(ps)
    assert isinstance(opt, torch.optim.Optimizer), "solve 须返回 torch.optim.Optimizer 实例"
    assert isinstance(opt.base_optimizer, torch.optim.Optimizer), "须暴露内部 base 优化器 .base_optimizer"
    assert len(opt.slow_weights) == len(ps), ".slow_weights 须与 params 一一对应"
    assert opt.slow_weights[0].shape == ps[0].shape and opt.slow_weights[0].dtype == ps[0].dtype
    bad = False
    try:
        solve([torch.tensor([1.0])], base="lion")
    except Exception:
        bad = True
    assert bad, "未知 base 名应抛异常"
