import torch
import torch.nn.functional as F

# 示例：logits=[[-2,-0.5,0,1,3]], targets=[[1,0,1,0,1]]（实跑参考答案回填）
EXAMPLE_LOGITS = [[-2.0, -0.5, 0.0, 1.0, 3.0]]
EXAMPLE_TARGETS = [[1.0, 0.0, 1.0, 0.0, 1.0]]
EXAMPLE_NONE = [2.1269280910491943, 0.4740769863128662, 0.6931471824645996,
                1.31326162815094, 0.04858735203742981]
EXAMPLE_MEAN = 0.9312002658843994
EXAMPLE_POS_WEIGHT_SUM = 7.524663925170898


def check_example(solve):
    x = torch.tensor(EXAMPLE_LOGITS)
    y = torch.tensor(EXAMPLE_TARGETS)
    assert torch.allclose(solve(x, y, reduction="none"),
                          torch.tensor(EXAMPLE_NONE), atol=1e-6)
    assert abs(float(solve(x, y)) - EXAMPLE_MEAN) < 1e-6
    got = solve(x, y, reduction="sum", pos_weight=torch.tensor([2.0]))
    assert abs(float(got) - EXAMPLE_POS_WEIGHT_SUM) < 1e-5


check_example.example = True


def check_matches_torch(solve):
    torch.manual_seed(501)
    logits = torch.randn(3, 5) * 3.0
    targets = (torch.rand(3, 5) > 0.4).float()
    weight = torch.rand(3, 5)
    for w in [None, weight]:
        for pw in [None, torch.tensor([2.0]), torch.rand(5)]:
            for red in ["none", "sum", "mean"]:
                ref = F.binary_cross_entropy_with_logits(
                    logits, targets, weight=w, reduction=red, pos_weight=pw)
                got = solve(logits, targets, weight=w, reduction=red, pos_weight=pw)
                assert got.shape == ref.shape, (red, w is not None, got.shape)
                assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5), (red, w is not None,
                                                                       pw is not None)
                assert got.dtype == torch.float32


def check_no_sigmoid_then_log(solve):
    # 先 sigmoid 再 log 会在 x=-1e3 处 sigmoid 下溢成 0 → log 0 = -inf
    x = torch.tensor([-1000.0, -100.0, 0.0, 100.0, 1000.0])
    y = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
    got = solve(x, y, reduction="none")
    assert torch.isfinite(got).all(), got.tolist()
    ref = F.binary_cross_entropy_with_logits(x, y, reduction="none")
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
    # x=0、y=0 的解析值就是 log 2
    assert abs(float(solve(torch.tensor([0.0]), torch.tensor([0.0]))) - 0.6931471805599453) < 1e-6


def check_mean_denominator(solve):
    # mean 的分母是元素个数，而不是 Σweight —— 这是最容易写错的一条
    torch.manual_seed(502)
    x = torch.randn(2, 4)
    y = (torch.rand(2, 4) > 0.5).float()
    w = torch.full((2, 4), 3.0)
    total = solve(x, y, weight=w, reduction="sum")
    mean = solve(x, y, weight=w, reduction="mean")
    assert abs(float(mean) - float(total) / 8.0) < 1e-6, (float(mean), float(total))
    assert abs(float(mean) - 3.0 * float(solve(x, y, reduction="mean"))) < 1e-5
    # 全 1 权重时与默认口径一致
    ones = torch.ones(2, 4)
    assert torch.allclose(solve(x, y, weight=ones, reduction="mean"),
                          solve(x, y, reduction="mean"), atol=1e-6)


def check_pos_weight_math(solve):
    # pos_weight=w 只作用于正类项：与手写公式 -[w·y·log σ(x) + (1-y)·log(1-σ(x))] 对齐
    torch.manual_seed(503)
    x = torch.randn(20) * 2.0
    y = (torch.rand(20) > 0.5).float()
    for wv in [1.0, 2.5, 0.5]:
        ref = -(wv * y * F.logsigmoid(x) + (1.0 - y) * F.logsigmoid(-x)).sum()
        got = solve(x, y, reduction="sum", pos_weight=torch.tensor([wv]))
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), wv
        assert torch.allclose(got,
                              F.binary_cross_entropy_with_logits(
                                  x, y, reduction="sum", pos_weight=torch.tensor([wv])),
                              atol=1e-5, rtol=1e-5)


def check_invalid_reduction(solve):
    x = torch.randn(3, 3)
    y = (torch.rand(3, 3) > 0.5).float()
    for red in ["median", "MEAN", "", None]:
        try:
            solve(x, y, reduction=red)
        except ValueError:
            continue
        raise AssertionError(f"reduction={red!r} 应抛 ValueError")
