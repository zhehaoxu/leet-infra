import math

import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_first_step_hand(solve):
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    g = torch.tensor([0.5, -1.0], dtype=torch.float64)
    out = solve(p, [g], lr=0.1)
    # m = 0.1g, v = 0.001g^2, bc1 = 0.1, bc2 = 0.001
    assert torch.allclose(out["exp_avg"], torch.tensor([0.05, -0.1], dtype=torch.float64), atol=1e-12), out["exp_avg"]
    assert torch.allclose(out["exp_avg_sq"], torch.tensor([2.5e-4, 1e-3], dtype=torch.float64), atol=1e-15), out["exp_avg_sq"]
    # denom = |g| + eps，step_size = 1 → delta = -g/(|g| + 1e-8)
    assert torch.allclose(out["deltas"][0], torch.tensor([-0.09999999800000003, 0.09999999900000002],
                                                         dtype=torch.float64), atol=1e-12), out["deltas"][0]
    assert torch.allclose(out["params"], torch.tensor([0.90000000199999997, -1.900000001],
                                                      dtype=torch.float64), atol=1e-12), out["params"]
    assert out["steps"] == [1], out["steps"]
    assert torch.equal(out["max_exp_avg_sq"], torch.zeros(2, dtype=torch.float64)), "amsgrad=False 时 max 应为全零"


def check_second_step_bias_correction(solve):
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    g1 = torch.tensor([0.5, -1.0], dtype=torch.float64)
    g2 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    out = solve(p, [g1, g2], lr=0.1)
    # m2 = 0.9m1 + 0.1g2, v2 = 0.999v1 + 0.001g2^2
    assert torch.allclose(out["exp_avg"], torch.tensor([0.14499999999999996, 0.009999999999999995],
                                                      dtype=torch.float64), atol=1e-12), out["exp_avg"]
    assert torch.allclose(out["exp_avg_sq"], torch.tensor([0.001249750000000001, 0.0019990000000000016],
                                                          dtype=torch.float64), atol=1e-15), out["exp_avg_sq"]
    assert out["steps"] == [1, 2], out["steps"]
    assert torch.allclose(out["params"], torch.tensor([0.8034818006385094, -1.9052631588421052],
                                                      dtype=torch.float64), atol=1e-12), out["params"]


def check_skip_none_does_not_advance_step(solve):
    # 无梯度的步不建 state、也不推进 step 计数 → 偏差修正仍从 1 开始
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    g1 = torch.tensor([0.5, -1.0], dtype=torch.float64)
    g2 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    out = solve(p, [None, g1, None, g2], lr=0.1)
    assert out["steps"] == [None, 1, None, 2], out["steps"]
    assert torch.allclose(out["path"][1], p, atol=0), "跳过步不应改参数"
    assert torch.allclose(out["path"][3], out["path"][2], atol=0), "第二个跳过步同样不应改参数"
    plain = solve(p, [g1, g2], lr=0.1)
    assert torch.allclose(out["params"], plain["params"], atol=0), "跳过步不应影响后续数值"
    assert out["deltas"][0] is None and out["deltas"][2] is None


def check_amsgrad_keeps_running_max(solve):
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    g1 = torch.tensor([2.0, 1.0], dtype=torch.float64)
    g2 = torch.tensor([0.05, 0.05], dtype=torch.float64)
    out = solve(p, [g1, g2], lr=0.1, amsgrad=True)
    # 第一维 v2 = 0.0039985 < v1 = 0.004 → max 停在 0.004；第二维 v 变小不了，max 跟上
    assert torch.allclose(out["exp_avg_sq"], torch.tensor([0.003998500000000004, 0.0010015000000000009],
                                                          dtype=torch.float64), atol=1e-15), out["exp_avg_sq"]
    assert torch.allclose(out["max_exp_avg_sq"], torch.tensor([0.0040000000000000036, 0.0010015000000000009],
                                                              dtype=torch.float64), atol=1e-15), out["max_exp_avg_sq"]
    plain = solve(p, [g1, g2], lr=0.1)
    assert not torch.allclose(out["params"], plain["params"], atol=1e-9), "max 进分母后参数应与普通 Adam 不同"
    assert torch.allclose(out["params"], torch.tensor([0.8311673448066528, -2.170640036064937],
                                                      dtype=torch.float64), atol=1e-12), out["params"]


def check_amsgrad_first_step_equals_v(solve):
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    g = torch.tensor([0.5, -1.0], dtype=torch.float64)
    out = solve(p, [g], lr=0.1, amsgrad=True)
    assert torch.allclose(out["max_exp_avg_sq"], out["exp_avg_sq"], atol=0), \
        f"首步 max(0, v) 应等于 v：{out['max_exp_avg_sq']} vs {out['exp_avg_sq']}"
    assert torch.allclose(out["params"], solve(p, [g], lr=0.1)["params"], atol=1e-15)


def check_weight_decay_l2_enters_moments(solve):
    # Adam 的 weight_decay 是 L2：λp 先并进梯度，再同时进两个矩（不是 AdamW 的解耦衰减）
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    g = torch.tensor([0.5, -1.0], dtype=torch.float64)
    wd = 0.5
    out = solve(p, [g], lr=0.1, weight_decay=wd)
    g_eff = g + wd * p
    assert torch.allclose(out["exp_avg"], 0.1 * g_eff, atol=1e-12), out["exp_avg"]
    assert torch.allclose(out["exp_avg_sq"], 0.001 * g_eff * g_eff, atol=1e-15), out["exp_avg_sq"]
    denom = (0.001 * g_eff * g_eff).sqrt() / (0.001 ** 0.5) + 1e-8
    assert torch.allclose(out["deltas"][0], -(0.1 * g_eff) / denom, atol=1e-12), out["deltas"][0]


def check_maximize_before_weight_decay(solve):
    # maximize 先取负再并 wd：-g + λp，等于 0 时这一步完全不动
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    g = torch.tensor([0.5, -1.0], dtype=torch.float64)
    zero = solve(p, [g], lr=0.1, weight_decay=0.5, maximize=True)
    assert torch.allclose(zero["deltas"][0], torch.zeros(2, dtype=torch.float64), atol=1e-15), zero["deltas"][0]
    assert torch.allclose(zero["params"], p, atol=1e-15), "梯度恰为 0 时参数不应移动"
    assert torch.equal(zero["exp_avg"], torch.zeros(2, dtype=torch.float64))
    # 非退化实例：g_eff = -g + λp 与 -(g + λp) 结果不同
    g2 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    out = solve(p, [g2], lr=0.1, weight_decay=0.5, maximize=True)
    g_eff = -g2 + 0.5 * p
    assert torch.allclose(out["exp_avg"], 0.1 * g_eff, atol=1e-12), out["exp_avg"]
    assert not torch.allclose(out["exp_avg"], -0.1 * (g2 + 0.5 * p), atol=1e-3)


def check_eps_enters_denom_unscaled(solve):
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    g = torch.tensor([0.5, -1.0], dtype=torch.float64)
    out = solve(p, [g], lr=0.1, eps=0.0)
    # eps=0 且单步：denom = |g|, step_size = 1 → delta = -g/|g| * lr
    assert torch.allclose(out["deltas"][0], torch.tensor([-0.1, 0.1], dtype=torch.float64), atol=1e-15), out["deltas"][0]
    # eps 没有乘 sqrt(bias_correction2)：差值是 2e-9 量级，而不是 6e-10
    with_eps = solve(p, [g], lr=0.1, eps=1e-8)
    diff = (with_eps["deltas"][0] - out["deltas"][0]).abs()
    assert torch.allclose(diff, torch.tensor([2e-9, 1e-9], dtype=torch.float64), atol=1e-18), diff


def check_beta_zero_edges(solve):
    # beta=0：m = g、v = g^2、两个偏差修正都等于 1
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    g = torch.tensor([0.5, -1.0], dtype=torch.float64)
    out = solve(p, [g], lr=0.1, betas=(0.0, 0.0))
    assert torch.allclose(out["exp_avg"], g, atol=0), out["exp_avg"]
    assert torch.allclose(out["exp_avg_sq"], g * g, atol=0), out["exp_avg_sq"]
    assert torch.allclose(out["deltas"][0], -0.1 * g / (g.abs() + 1e-8), atol=1e-12), out["deltas"][0]


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
    bad("learning rate", lr=float("nan"))
    bad("epsilon", lr=0.1, eps=-1e-8)
    bad("epsilon", lr=0.1, eps=float("nan"))
    bad("index 0", lr=0.1, betas=(-0.1, 0.999))
    bad("index 0", lr=0.1, betas=(1.0, 0.999))
    bad("index 1", lr=0.1, betas=(0.9, 1.0))
    bad("weight_decay", lr=0.1, weight_decay=-1.0)
    bad("betas", lr=0.1, betas=(0.9,))
    # 边界合法：eps=0、beta=0、lr=0
    solve(p, [g], lr=0.0, eps=0.0, betas=(0.0, 0.0))


def check_validation_order(solve):
    p = torch.zeros(2, dtype=torch.float64)
    g = torch.ones(2, dtype=torch.float64)
    cases = [((dict(lr=-1.0, eps=-1.0)), "learning rate"),
             ((dict(lr=0.1, eps=-1.0, betas=(-1.0, 2.0))), "epsilon"),
             ((dict(lr=0.1, betas=(-1.0, 2.0), weight_decay=-1.0)), "index 0")]
    for kw, sub in cases:
        try:
            solve(p, [g], **kw)
        except ValueError as exc:
            assert sub in str(exc), f"{kw} 应先报 {sub}，got {exc}"
        else:
            raise AssertionError(f"应当抛 ValueError: {kw}")


def check_state_resume(solve):
    p = torch.tensor([0.5, -0.5], dtype=torch.float64)
    gs = [torch.tensor([1.0, 0.5], dtype=torch.float64),
          torch.tensor([-0.5, 1.0], dtype=torch.float64),
          torch.tensor([0.25, -0.25], dtype=torch.float64),
          torch.tensor([1.0, 1.0], dtype=torch.float64)]
    kw = dict(lr=0.1, betas=(0.8, 0.95), weight_decay=0.02, amsgrad=True)
    full = solve(p, gs, **kw)
    first = solve(p, gs[:2], **kw)
    state = {"exp_avg": first["exp_avg"], "exp_avg_sq": first["exp_avg_sq"],
             "max_exp_avg_sq": first["max_exp_avg_sq"], "step": first["steps"][-1]}
    second = solve(first["params"], gs[2:], state=state, **kw)
    assert torch.allclose(second["params"], full["params"], atol=1e-12), f"{second['params']} vs {full['params']}"
    assert torch.allclose(second["exp_avg"], full["exp_avg"], atol=1e-12)
    assert second["steps"] == [3, 4], second["steps"]


def check_no_inplace(solve):
    p = torch.tensor([1.0, 1.0], dtype=torch.float64)
    g = torch.tensor([0.5, 0.5], dtype=torch.float64)
    state = {"exp_avg": torch.zeros(2, dtype=torch.float64),
             "exp_avg_sq": torch.zeros(2, dtype=torch.float64),
             "max_exp_avg_sq": torch.zeros(2, dtype=torch.float64),
             "step": 0}
    out = solve(p, [g], lr=0.1, amsgrad=True, state=state)
    assert torch.equal(p, torch.tensor([1.0, 1.0], dtype=torch.float64)), "不应原地改 params"
    assert torch.equal(g, torch.tensor([0.5, 0.5], dtype=torch.float64)), "不应原地改 grads"
    assert torch.equal(state["exp_avg"], torch.zeros(2, dtype=torch.float64)), "不应原地改 state"
    before = out["params"].clone()
    out["deltas"][0].add_(1.0)
    assert torch.equal(out["params"], before), "deltas 不应与 params 共享存储"
    assert torch.equal(out["path"][-1], before), "deltas 不应与 path 共享存储"


def check_zero_steps(solve):
    p = torch.tensor([1.0, -2.0], dtype=torch.float64)
    out = solve(p, [], lr=0.1)
    assert len(out["path"]) == 1 and out["steps"] == [] and out["deltas"] == []
    assert torch.equal(out["params"], p)
    assert torch.equal(out["exp_avg"], torch.zeros(2, dtype=torch.float64))


def _oracle(params, grads, **kw):
    p = params.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], **kw)
    path = [p.detach().clone()]
    for g in grads:
        p.grad = None if g is None else g.clone()
        opt.step()
        path.append(p.detach().clone())
    st = opt.state[p]
    if not st:
        z = torch.zeros_like(params)
        return path, z, z, z
    return path, st["exp_avg"], st["exp_avg_sq"], st.get("max_exp_avg_sq", torch.zeros_like(params))


def check_matches_torch_optim(solve):
    configs = [
        dict(lr=0.1),
        dict(lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05),
        dict(lr=0.05, betas=(0.5, 0.9), amsgrad=True),
        dict(lr=0.2, betas=(0.0, 0.0)),
        dict(lr=0.1, betas=(0.9, 0.999), weight_decay=0.1, maximize=True),
        dict(lr=0.3, betas=(0.99, 0.9999), eps=0.0),
        dict(lr=0.0, betas=(0.9, 0.95)),
        dict(lr=0.07, betas=(0.8, 0.99), amsgrad=True, maximize=True, weight_decay=0.02),
    ]
    torch.manual_seed(20261102)
    for cfg in configs:
        for trial in range(4):
            p0 = torch.randn(4, dtype=torch.float64)
            gs = [torch.randn(4, dtype=torch.float64) for _ in range(8)]
            if trial == 1:
                gs[0] = None
            if trial == 2:
                gs[3] = None
            if trial == 3:
                gs = [None] * 8
            path_ref, m_ref, v_ref, mx_ref = _oracle(p0, gs, **cfg)
            out = solve(p0, gs, **cfg)
            assert len(out["path"]) == len(path_ref), (len(out["path"]), len(path_ref))
            for step, (a, b) in enumerate(zip(out["path"], path_ref)):
                assert torch.allclose(a, b, atol=1e-12), f"{cfg} trial {trial} step {step}: {a} vs {b}"
            assert torch.allclose(out["exp_avg"], m_ref, atol=1e-12), f"{cfg} trial {trial} m"
            assert torch.allclose(out["exp_avg_sq"], v_ref, atol=1e-12), f"{cfg} trial {trial} v"
            assert torch.allclose(out["max_exp_avg_sq"], mx_ref, atol=1e-12), f"{cfg} trial {trial} max"


def check_lr_scale_linearity(solve):
    # lr 线性进 step_size（不是 sqrt(1-b2) 那种缩放），double lr 恰好 double 位移
    p = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    gs = [torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64),
          torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64)]
    a = solve(p, gs, lr=0.1, betas=(0.9, 0.99))
    b = solve(p, gs, lr=0.05, betas=(0.9, 0.99))
    assert torch.allclose((a["params"] - p), 2.0 * (b["params"] - p), atol=1e-12), \
        f"{a['params'] - p} vs 2*{b['params'] - p}"


check_first_step_hand.example = True
check_skip_none_does_not_advance_step.example = True
