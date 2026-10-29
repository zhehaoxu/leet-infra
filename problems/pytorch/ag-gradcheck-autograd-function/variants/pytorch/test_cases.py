import torch

# 示例：L = Σ x²，解析梯度 2x。二次函数的中心差分是**精确**的
# （(x+δ)² − (x−δ)² = 4xδ ⇒ /2δ = 2x），所以最大相对误差只剩浮点舍入量级。
EXAMPLE_X = [[1.0, -2.0]]


def _quad_fwd(x):
    return (x * x).sum()


def _quad_bwd(x, g):
    return (2.0 * x * g,)


def check_example(solve):
    x = torch.tensor(EXAMPLE_X, dtype=torch.float64)
    out = solve(_quad_fwd, _quad_bwd, (x,))
    assert isinstance(out, (list, tuple)) and len(out) == 1
    e = float(out[0])
    assert 0.0 <= e < 1e-10, e


check_example.example = True


def _softmax_ce_fwd(logits, target_oh):
    p = torch.softmax(logits, dim=-1)
    return -(target_oh * torch.log(p)).sum()


def _softmax_ce_bwd(logits, target_oh, g):
    p = torch.softmax(logits, dim=-1)
    return ((p - target_oh) * g, -g * torch.log(p))


def check_matches_autograd_engine(solve):
    torch.manual_seed(501)
    logits = torch.randn(4, 5, dtype=torch.float64)
    tgt = torch.zeros(4, 5, dtype=torch.float64)
    tgt[torch.arange(4), torch.randint(0, 5, (4,))] = 1.0
    errs = solve(_softmax_ce_fwd, _softmax_ce_bwd, (logits, tgt))
    assert len(errs) == 2
    for k, e in enumerate(errs):
        assert e < 1e-6, (k, e)


def check_detects_wrong_backward(solve):
    # 只有第二个输入的解析梯度错了一倍：out[1] 必须很大、out[0] 仍然很小。
    # 若实现是「直接对 forward 求 autograd」，这里会报两个 0 —— 本题就靠这条抓住它。
    torch.manual_seed(502)
    x = torch.randn(3, 2, dtype=torch.float64)
    w = torch.randn(3, 2, dtype=torch.float64)

    def fwd(x, w):
        return (x * w).sum() + torch.tanh(x).sum()

    def bad_bwd(x, w, g):
        return (w + (1.0 - torch.tanh(x) ** 2) * g, 2.0 * x * g)   # dw 少了一半/错了一倍

    errs = solve(fwd, bad_bwd, (x, w))
    assert errs[0] < 1e-6, errs
    assert errs[1] > 0.3, errs


def check_relu_kink(solve):
    # x = 0 处 relu 的次梯度约定是 0，而中心差分给 0.5 ⇒ rel ≈ 1
    x = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float64)
    errs = solve(lambda z: torch.relu(z).sum(), lambda z, g: (torch.where(z > 0, g, torch.zeros_like(g)),), (x,))
    assert errs[0] > 0.5, errs
    # 同一个算子避开折点就应当通过
    y = torch.tensor([[1.5, 2.0, -1.0, -3.0]], dtype=torch.float64)
    errs2 = solve(lambda z: torch.relu(z).sum(), lambda z, g: (torch.where(z > 0, g, torch.zeros_like(g)),), (y,))
    assert errs2[0] < 1e-9, errs2


def check_every_input_is_perturbed(solve):
    # 三个输入，只有最后一个的解析梯度写错 ⇒ 错误必须落在最后一项上
    x = torch.tensor([0.5, -0.25], dtype=torch.float64)
    a = torch.tensor([1.5], dtype=torch.float64)
    b = torch.tensor([-2.0, 0.75, 3.0], dtype=torch.float64)

    def fwd(x, a, b):
        return (x * x).sum() + a[0] ** 3 + (b * b).sum()

    def bwd(x, a, b, g):
        return (2.0 * x * g, 3.0 * a * a * g, 3.0 * b * g)      # db 系数写错

    errs = solve(fwd, bwd, (x, a, b))
    assert len(errs) == 3
    assert errs[0] < 1e-6 and errs[1] < 1e-6, errs
    assert errs[2] > 0.3, errs


def check_step_size_sensitivity(solve):
    # 自定义 eps 时步长随之变化，但对正确梯子的判定不应翻转
    torch.manual_seed(503)
    x = torch.randn(2, 3, dtype=torch.float64)

    def fwd(x):
        return torch.sin(x).sum() + torch.exp(x * 0.1).sum()

    def bwd(x, g):
        return (torch.cos(x) + 0.1 * torch.exp(x * 0.1)) * g

    for eps in (torch.finfo(torch.float64).eps, 1e-9, 1e-12):
        errs = solve(fwd, bwd, (x,), eps=eps)
        assert errs[0] < 1e-5, (eps, errs)


def check_invalid_args(solve):
    x = torch.randn(2, 2, dtype=torch.float64)
    bad = [
        (torch.randn(2, 2), _quad_bwd, "float32 输入"),
        (torch.randn(2, 2, dtype=torch.float16), _quad_bwd, "float16 输入"),
    ]
    for xx, bwd, msg in bad:
        try:
            solve(_quad_fwd, bwd, (xx,))
        except ValueError:
            continue
        raise AssertionError(msg + " 应抛 ValueError")
    # forward 返回非 0 维
    try:
        solve(lambda z: z * z, _quad_bwd, (x,))
        raise AssertionError("非标量前向应抛 ValueError")
    except ValueError:
        pass
    # backward 返回个数不符
    try:
        solve(_quad_fwd, lambda z, g: (g, g), (x,))
        raise AssertionError("梯度个数不符应抛 ValueError")
    except ValueError:
        pass
