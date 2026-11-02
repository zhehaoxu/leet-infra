import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_plain_first_step(solve):
    p = torch.tensor([1.0, 1.0], dtype=torch.float64)
    g = torch.tensor([0.5, -0.5], dtype=torch.float64)
    out = solve(p, [g], lr=0.1, momentum=0.9)
    assert torch.allclose(out["params"], torch.tensor([0.95, 1.05], dtype=torch.float64), atol=1e-12), out["params"]
    assert torch.allclose(out["buffers"], g, atol=1e-12), out["buffers"]
    assert len(out["path"]) == 2 and torch.allclose(out["path"][0], p)
    assert torch.allclose(out["dirs"][0], g)


def check_first_step_ignores_dampening(solve):
    # 首次建缓冲：buf = g（不是 (1-dampening)*g），dampening=0.5 在这一步不起作用
    p = torch.tensor([1.0, 1.0], dtype=torch.float64)
    g = torch.tensor([1.0, -2.0], dtype=torch.float64)
    out = solve(p, [g], lr=0.1, momentum=0.9, dampening=0.5)
    assert torch.allclose(out["buffers"], g, atol=1e-12), f"buf {out['buffers']}"
    assert torch.allclose(out["dirs"][0], g, atol=1e-12), f"dir {out['dirs'][0]}"
    assert torch.allclose(out["params"], torch.tensor([0.9, 1.2], dtype=torch.float64), atol=1e-12), out["params"]


def check_zero_buffer_is_history(solve):
    # 调用方显式传全零缓冲 = 「已有历史状态」，首个动量步照样吃 dampening
    p = torch.tensor([1.0, 1.0], dtype=torch.float64)
    g = torch.tensor([1.0, -2.0], dtype=torch.float64)
    z = torch.zeros(2, dtype=torch.float64)
    out = solve(p, [g], lr=0.1, momentum=0.9, dampening=0.5, buffers=z)
    assert torch.allclose(out["buffers"], 0.5 * g, atol=1e-12), f"buf {out['buffers']}"
    assert torch.allclose(out["dirs"][0], 0.5 * g, atol=1e-12), f"dir {out['dirs'][0]}"
    assert torch.allclose(out["params"], torch.tensor([0.95, 1.1], dtype=torch.float64), atol=1e-12), out["params"]


def check_dampening_second_step(solve):
    p = torch.tensor([1.0, -1.0], dtype=torch.float64)
    g0 = torch.tensor([1.0, -2.0], dtype=torch.float64)
    g1 = torch.tensor([0.5, 0.5], dtype=torch.float64)
    out = solve(p, [g0, g1], lr=0.5, momentum=0.9, dampening=0.5)
    # buf1 = 0.9*[1,-2] + 0.5*[0.5,0.5] = [1.15, -1.55]
    assert torch.allclose(out["buffers"], torch.tensor([1.15, -1.55], dtype=torch.float64), atol=1e-12), out["buffers"]
    assert torch.allclose(out["params"], torch.tensor([-0.075, 0.775], dtype=torch.float64), atol=1e-12), out["params"]


def check_nesterov_direction(solve):
    p = torch.tensor([1.0, 1.0], dtype=torch.float64)
    g = torch.tensor([1.0, 1.0], dtype=torch.float64)
    out = solve(p, [g], lr=0.1, momentum=0.5, nesterov=True)
    # step = g + mu*buf = [1.5, 1.5]
    assert torch.allclose(out["dirs"][0], torch.tensor([1.5, 1.5], dtype=torch.float64), atol=1e-12), out["dirs"][0]
    assert torch.allclose(out["params"], torch.tensor([0.85, 0.85], dtype=torch.float64), atol=1e-12), out["params"]
    plain = solve(p, [g], lr=0.1, momentum=0.5, nesterov=False)
    assert torch.allclose(plain["params"], torch.tensor([0.9, 0.9], dtype=torch.float64), atol=1e-12), plain["params"]


def check_maximize_before_weight_decay(solve):
    # maximize 先取负，再并 weight_decay：dir = -g + wd*p（不是 -(g + wd*p)）
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    g = torch.tensor([1.0, 1.0], dtype=torch.float64)
    out = solve(p, [g], lr=1.0, weight_decay=0.5, maximize=True)
    assert torch.allclose(out["dirs"][0], torch.tensor([-0.5, -2.0], dtype=torch.float64), atol=1e-12), out["dirs"][0]
    assert torch.allclose(out["params"], torch.tensor([1.5, 0.0], dtype=torch.float64), atol=1e-12), out["params"]


def check_weight_decay_into_first_buffer(solve):
    # wd 并进梯度后才建缓冲：首步 buf = g + wd*p
    p = torch.tensor([1.0, -1.0], dtype=torch.float64)
    g = torch.zeros(2, dtype=torch.float64)
    out = solve(p, [g], lr=0.1, momentum=0.9, dampening=0.3, weight_decay=0.1)
    assert torch.allclose(out["buffers"], torch.tensor([0.1, -0.1], dtype=torch.float64), atol=1e-12), out["buffers"]
    assert torch.allclose(out["params"], torch.tensor([0.99, -0.99], dtype=torch.float64), atol=1e-12), out["params"]


def check_skip_none_creates_no_state(solve):
    # 跳过步不建缓冲：后面第一个真梯度仍走「首建」分支，dampening 被忽略
    p = torch.tensor([1.0, 2.0], dtype=torch.float64)
    g = torch.tensor([1.0, 1.0], dtype=torch.float64)
    out = solve(p, [None, g], lr=0.1, momentum=0.9, dampening=0.5)
    assert out["dirs"][0] is None, "跳过步的 dirs 应为 None"
    assert torch.allclose(out["path"][1], p, atol=0), "跳过步不应改参数"
    assert torch.allclose(out["buffers"], g, atol=1e-12), f"buf {out['buffers']}"
    assert torch.allclose(out["params"], torch.tensor([0.9, 1.9], dtype=torch.float64), atol=1e-12), out["params"]


def check_skip_none_mid_path(solve):
    p = torch.zeros(2, dtype=torch.float64)
    g0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    g2 = torch.tensor([0.0, 2.0], dtype=torch.float64)
    out = solve(p, [g0, None, g2], lr=0.1, momentum=0.5, dampening=0.5)
    assert len(out["path"]) == 4, len(out["path"])
    assert torch.allclose(out["path"][1], out["path"][2], atol=0)
    # buf2 = 0.5*[1,1] + 0.5*[0,2] = [0.5, 1.5]
    assert torch.allclose(out["buffers"], torch.tensor([0.5, 1.5], dtype=torch.float64), atol=1e-12), out["buffers"]
    assert torch.allclose(out["params"], torch.tensor([-0.15, -0.25], dtype=torch.float64), atol=1e-12), out["params"]


def check_zero_lr_still_advances_buffer(solve):
    p = torch.tensor([1.0, 1.0], dtype=torch.float64)
    g = torch.tensor([1.0, 0.0], dtype=torch.float64)
    out = solve(p, [g, g], lr=0.0, momentum=0.9, dampening=0.5)
    assert torch.allclose(out["params"], p, atol=0), "lr=0 时参数不应变化"
    # buf2 = 0.9*[1,0] + 0.5*[1,0] = [1.4, 0]
    assert torch.allclose(out["buffers"], torch.tensor([1.4, 0.0], dtype=torch.float64), atol=1e-12), out["buffers"]


def check_momentum_zero_ignores_dampening(solve):
    p = torch.tensor([1.0, 1.0], dtype=torch.float64)
    g = torch.tensor([1.0, 2.0], dtype=torch.float64)
    out = solve(p, [g], lr=0.1, momentum=0.0, dampening=3.0)
    assert torch.allclose(out["params"], torch.tensor([0.9, 0.8], dtype=torch.float64), atol=1e-12), out["params"]
    assert torch.equal(out["buffers"], torch.zeros(2, dtype=torch.float64)), "momentum=0 时缓冲应为全零"


def check_validation_errors(solve):
    p = torch.zeros(2, dtype=torch.float64)
    g = torch.ones(2, dtype=torch.float64)

    def bad(exc_sub, **kw):
        try:
            solve(p, [g], **kw)
        except ValueError as exc:
            assert exc_sub in str(exc), f"{exc_sub!r} not in {exc!r}"
            return
        raise AssertionError(f"应当抛 ValueError: {kw}")

    bad("learning rate", lr=-0.1)
    bad("momentum", lr=0.1, momentum=-0.1)
    bad("weight_decay", lr=0.1, weight_decay=-0.1)
    bad("Nesterov", lr=0.1, momentum=0.0, nesterov=True)
    bad("Nesterov", lr=0.1, momentum=0.5, dampening=0.1, nesterov=True)
    bad("Nesterov", lr=0.1, momentum=0.5, dampening=-0.1, nesterov=True)
    # dampening 本身不做取值范围校验；momentum=0 时也不报错
    solve(p, [g], lr=0.1, momentum=0.5, dampening=5.0)
    solve(p, [g], lr=0.0, momentum=0.0, dampening=2.0)


def check_validation_order(solve):
    p = torch.zeros(2, dtype=torch.float64)
    g = torch.ones(2, dtype=torch.float64)
    try:
        solve(p, [g], lr=-1.0, momentum=-1.0, weight_decay=-1.0)
    except ValueError as exc:
        assert "learning rate" in str(exc), f"先报 lr，got {exc}"
    else:
        raise AssertionError("应当抛 ValueError")
    try:
        solve(p, [g], lr=0.1, momentum=-1.0, weight_decay=-1.0)
    except ValueError as exc:
        assert "momentum" in str(exc), f"再报 momentum，got {exc}"
    else:
        raise AssertionError("应当抛 ValueError")


def check_no_inplace(solve):
    p = torch.tensor([1.0, 1.0], dtype=torch.float64)
    g = torch.tensor([0.5, 0.5], dtype=torch.float64)
    b = torch.zeros(2, dtype=torch.float64)
    out = solve(p, [g], lr=0.1, momentum=0.9, buffers=b)
    assert torch.equal(p, torch.tensor([1.0, 1.0], dtype=torch.float64)), "不应原地改 params"
    assert torch.equal(g, torch.tensor([0.5, 0.5], dtype=torch.float64)), "不应原地改 grads"
    assert torch.equal(b, torch.zeros(2, dtype=torch.float64)), "不应原地改 buffers"
    out["dirs"][0].add_(1.0)
    assert torch.equal(b, torch.zeros(2, dtype=torch.float64)), "dirs 不应与入参 buffers 共享存储"
    assert torch.allclose(out["buffers"], torch.tensor([0.5, 0.5], dtype=torch.float64), atol=1e-12), \
        "返回的账本字段应各自独立：就地改 dirs 不能改到 buffers"


def check_resume_matches_continuous(solve):
    p = torch.tensor([0.5, -0.5], dtype=torch.float64)
    gs = [torch.tensor([1.0, 0.5], dtype=torch.float64),
          torch.tensor([-0.5, 1.0], dtype=torch.float64),
          torch.tensor([0.25, -0.25], dtype=torch.float64),
          torch.tensor([1.0, 1.0], dtype=torch.float64)]
    full = solve(p, gs, lr=0.1, momentum=0.9, dampening=0.25, weight_decay=0.01)
    first = solve(p, gs[:2], lr=0.1, momentum=0.9, dampening=0.25, weight_decay=0.01)
    second = solve(first["params"], gs[2:], lr=0.1, momentum=0.9, dampening=0.25,
                   weight_decay=0.01, buffers=first["buffers"])
    assert torch.allclose(second["params"], full["params"], atol=1e-12), f"{second['params']} vs {full['params']}"
    assert torch.allclose(second["buffers"], full["buffers"], atol=1e-12)


def _oracle(params, grads, **kw):
    p = params.clone().requires_grad_(True)
    opt = torch.optim.SGD([p], **kw)
    path = [p.detach().clone()]
    for g in grads:
        p.grad = None if g is None else g.clone()
        opt.step()
        path.append(p.detach().clone())
    buf = opt.state[p].get("momentum_buffer")
    if buf is None:
        buf = torch.zeros_like(params)
    return path, buf


def check_matches_torch_optim(solve):
    configs = [
        dict(lr=0.1),
        dict(lr=0.05, momentum=0.9),
        dict(lr=0.05, momentum=0.9, dampening=0.25, weight_decay=0.01),
        dict(lr=0.2, momentum=0.5, nesterov=True, weight_decay=0.1),
        dict(lr=0.1, momentum=0.8, dampening=0.4, weight_decay=0.05, maximize=True),
        dict(lr=0.1, momentum=0.6, maximize=True),
        dict(lr=0.0, momentum=0.9, dampening=0.1, weight_decay=0.02),
        dict(lr=0.3, momentum=0.95, dampening=1.0),
        dict(lr=0.07, momentum=0.99, dampening=0.9, maximize=True, nesterov=False),
    ]
    torch.manual_seed(20261102)
    for cfg in configs:
        for trial in range(3):
            p0 = torch.randn(4, dtype=torch.float64)
            gs = [torch.randn(4, dtype=torch.float64) for _ in range(7)]
            if trial == 1:
                gs[0] = None
            if trial == 2:
                gs[3] = None
            path_ref, buf_ref = _oracle(p0, gs, **cfg)
            out = solve(p0, gs, **cfg)
            assert len(out["path"]) == len(path_ref), (len(out["path"]), len(path_ref))
            for step, (a, b) in enumerate(zip(out["path"], path_ref)):
                assert torch.allclose(a, b, atol=1e-12), f"{cfg} trial {trial} step {step}: {a} vs {b}"
            assert torch.allclose(out["buffers"], buf_ref, atol=1e-12), f"{cfg} trial {trial} buf"


check_plain_first_step.example = True
check_first_step_ignores_dampening.example = True
