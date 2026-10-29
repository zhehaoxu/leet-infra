import torch
import torch.nn.functional as F

# 示例：logits 全 0（故 p = 1/C），N=2、C=4、target=[1,0]、ε=0、reduction='mean'
# 有效行数 2 ⇒ 非目标列 (1/4)/2 = 0.125，目标列 (1/4 − 1)/2 = −0.375
EXAMPLE_LOGITS = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
EXAMPLE_TARGET = [1, 0]
EXAMPLE_OUT = [[0.125, -0.375, 0.125, 0.125], [-0.375, 0.125, 0.125, 0.125]]


def check_example(solve):
    logits = torch.tensor(EXAMPLE_LOGITS)
    target = torch.tensor(EXAMPLE_TARGET, dtype=torch.long)
    out = solve(logits, target)
    assert out.shape == (2, 4)
    assert torch.allclose(out, torch.tensor(EXAMPLE_OUT), atol=1e-6)


check_example.example = True


def check_matches_autograd(solve):
    torch.manual_seed(201)
    logits = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    target = torch.randint(0, 5, (6,), dtype=torch.long)
    loss = F.cross_entropy(logits, target)
    (ref,) = torch.autograd.grad(loss, logits)
    got = solve(logits.detach(), target)
    assert got.dtype == logits.dtype
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-9), (got - ref).abs().max()


def check_label_smoothing(solve):
    torch.manual_seed(202)
    for eps in (0.0, 0.05, 0.25):
        logits = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
        target = torch.randint(0, 4, (7,), dtype=torch.long)
        loss = F.cross_entropy(logits, target, label_smoothing=eps)
        (ref,) = torch.autograd.grad(loss, logits)
        assert torch.allclose(solve(logits.detach(), target, label_smoothing=eps),
                              ref, atol=1e-9), eps
    # ε=0.25、C=4、logits 全 0 时梯度是精确值：非目标 0.1875、目标 −0.5625（再除有效行数 2）
    z = torch.zeros(2, 4, dtype=torch.float64)
    t = torch.tensor([1, 0], dtype=torch.long)
    got = solve(z, t, label_smoothing=0.25)
    assert torch.allclose(got, torch.tensor([[0.09375, -0.28125, 0.09375, 0.09375],
                                             [-0.28125, 0.09375, 0.09375, 0.09375]],
                                            dtype=torch.float64), atol=1e-12)


def check_ignore_index_denominator(solve):
    # 4 行里 2 行被忽略：mean 的分母必须是 2，且忽略行梯度为 0
    torch.manual_seed(203)
    logits = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0, -100, 2, -100], dtype=torch.long)
    loss = F.cross_entropy(logits, target, ignore_index=-100)
    (ref,) = torch.autograd.grad(loss, logits)
    got = solve(logits.detach(), target)
    assert torch.allclose(got, ref, atol=1e-9)
    assert torch.count_nonzero(got[[1, 3]]) == 0
    # 全部忽略：分母为 0，梯度整块为 0 且不得出现 NaN
    allig = torch.full((4,), -100, dtype=torch.long)
    g2 = solve(logits.detach(), allig)
    assert torch.isfinite(g2).all() and torch.count_nonzero(g2) == 0


def check_reduction_sum_and_large_c(solve):
    torch.manual_seed(204)
    logits = torch.randn(5, 9, dtype=torch.float64, requires_grad=True)
    target = torch.randint(0, 9, (5,), dtype=torch.long)
    loss = F.cross_entropy(logits, target, reduction='sum')
    (ref,) = torch.autograd.grad(loss, logits)
    assert torch.allclose(solve(logits.detach(), target, reduction='sum'), ref, atol=1e-9)
    # sum 与 mean 只差一个有效行数因子
    sm = solve(logits.detach(), target, reduction='mean')
    assert torch.allclose(sm * 5, ref, atol=1e-9)


def check_invalid_args(solve):
    good = torch.zeros(2, 3)
    tgt = torch.tensor([0, 1], dtype=torch.long)
    bad = [
        (good.unsqueeze(0), tgt, 0.0, -100, 'mean'),   # logits 三维
        (good, tgt.unsqueeze(1), 0.0, -100, 'mean'),   # target 二维
        (good, torch.tensor([0], dtype=torch.long), 0.0, -100, 'mean'),  # 长度不符
        (good, tgt, 1.0, -100, 'mean'),                # ε 越界
        (good, tgt, -0.1, -100, 'mean'),               # ε 负
        (good, tgt, 0.0, -100, 'none'),                # 不支持的 reduction
    ]
    for logits, target, eps, ig, red in bad:
        try:
            solve(logits, target, label_smoothing=eps, ignore_index=ig, reduction=red)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
