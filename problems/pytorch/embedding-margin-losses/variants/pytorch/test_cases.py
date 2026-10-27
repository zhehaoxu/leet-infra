import torch
import torch.nn.functional as F

# 示例期望值（reduction="none"）：cosine 用 2×3 手写数据、默认 margin=0.0；
# hinge 用 2 个手写分数、默认 margin=1.0（均实测回填）
EX_X1 = [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]
EX_X2 = [[1.0, 1.0, 0.0], [0.0, -1.0, 2.0]]
EX_TARGET = [1, -1]
EX_SCORES = [2.5, -0.5]
EXAMPLE_COSINE = [0.6837722063, 0.3162277639]  # 实测回填
EXAMPLE_HINGE = [2.5, 1.5]                     # 实测回填


def check_example(solve):
    cos = solve(torch.tensor(EX_X1), torch.tensor(EX_X2), torch.tensor(EX_TARGET),
                mode="cosine", reduction="none")
    hin = solve(torch.tensor(EX_SCORES), None, torch.tensor(EX_TARGET), mode="hinge",
                reduction="none")
    assert cos.shape == (2,) and hin.shape == (2,)
    assert torch.allclose(cos, torch.tensor(EXAMPLE_COSINE), atol=1e-6), cos.tolist()
    assert torch.allclose(hin, torch.tensor(EXAMPLE_HINGE), atol=1e-6), hin.tolist()


check_example.example = True


def check_cosine_matches_torch(solve):
    torch.manual_seed(21)
    x1 = torch.randn(16, 7)
    x2 = torch.randn(16, 7)
    target = torch.tensor([1, -1] * 8)
    for margin in (0.0, 0.2, 0.7):
        for red in ("none", "mean", "sum"):
            got = solve(x1, x2, target, mode="cosine", margin=margin, reduction=red)
            ref = F.cosine_embedding_loss(x1, x2, target, margin=margin, reduction=red)
            assert torch.allclose(got, ref, atol=1e-6), (margin, red, got.tolist(), ref.tolist())


def check_hinge_matches_torch(solve):
    torch.manual_seed(22)
    scores = torch.randn(20)
    target = torch.tensor([1, -1] * 10)
    for margin in (1.0, 0.5):
        for red in ("none", "mean", "sum"):
            got = solve(scores, None, target, mode="hinge", margin=margin, reduction=red)
            ref = F.hinge_embedding_loss(scores, target, margin=margin, reduction=red)
            assert torch.allclose(got, ref, atol=1e-6), (margin, red, got.tolist(), ref.tolist())


def check_cosine_negative_cos_probe(solve):
    # 反向平行 cos=−1 与同向 cos=1，margin=0
    x1 = torch.tensor([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    x2 = torch.tensor([[-3.0, 0.0], [4.0, 0.0], [-1.0, 0.0]])
    target = torch.tensor([1, -1, -1])
    got = solve(x1, x2, target, mode="cosine", margin=0.0, reduction="none")
    # cos = [-1, 1, -1]；target=1 支 1−cos=2；target=−1 支 max(0, cos−0)：
    # cos>0 的样本贡献 cos=1，cos<0 的样本被 clamp 成 0
    assert torch.allclose(got, torch.tensor([2.0, 1.0, 0.0]), atol=1e-7), got.tolist()
    assert torch.allclose(
        got, F.cosine_embedding_loss(x1, x2, target, margin=0.0, reduction="none"), atol=1e-6)
    # target=1 且 cos<0 的样本损失 = 1 − cos = 1 + |cos| > 1
    x3 = torch.tensor([[1.0, 1.0]])
    x4 = torch.tensor([[-1.0, -0.5]])
    got2 = solve(x3, x4, torch.tensor([1]), mode="cosine", reduction="none")
    cos = float((x3 * x4).sum()) / float(torch.norm(x3) * torch.norm(x4))
    assert abs(got2.item() - (1.0 - cos)) < 1e-6 and got2.item() > 1.0
    # margin>0 时 target=−1 支：cos−margin>0 才有损失
    got3 = solve(x3, x3, torch.tensor([-1]), mode="cosine", margin=0.5, reduction="none")
    assert abs(got3.item() - 0.5) < 1e-6, got3.item()  # cos=1 → 1−0.5


def check_hinge_all_negative_targets(solve):
    # 全 −1：全部走 max(0, margin − x) 支
    scores = torch.tensor([2.0, -0.5, 0.0, 1.0])
    target = torch.tensor([-1, -1, -1, -1])
    got = solve(scores, None, target, mode="hinge", margin=1.0, reduction="none")
    assert torch.allclose(got, torch.tensor([0.0, 1.5, 1.0, 0.0]), atol=1e-7), got.tolist()
    assert torch.allclose(
        got, F.hinge_embedding_loss(scores, target, margin=1.0, reduction="none"), atol=1e-6)
    got_sum = solve(scores, None, target, mode="hinge", margin=1.0, reduction="sum")
    assert torch.allclose(got_sum, torch.tensor(2.5), atol=1e-6)
    # 全 1：全部走恒等支 x
    got2 = solve(scores, None, torch.ones(4, dtype=torch.long), mode="hinge", margin=1.0,
                 reduction="none")
    assert torch.equal(got2, scores)


def check_empty_input(solve):
    x0 = torch.zeros(0, 3)
    t0 = torch.zeros(0, dtype=torch.long)
    cos_none = solve(x0, torch.zeros(0, 3), t0, mode="cosine", reduction="none")
    assert cos_none.shape == (0,)
    cos_mean = solve(x0, torch.zeros(0, 3), t0, mode="cosine", reduction="mean")
    assert cos_mean.shape == () and cos_mean.item() == 0.0, "空集 mean 兜底返回 0"
    cos_sum = solve(x0, torch.zeros(0, 3), t0, mode="cosine", reduction="sum")
    assert cos_sum.shape == () and cos_sum.item() == 0.0
    s0 = torch.zeros(0)
    hin_none = solve(s0, None, t0, mode="hinge", reduction="none")
    assert hin_none.shape == (0,)
    hin_mean = solve(s0, None, t0, mode="hinge", reduction="mean")
    assert hin_mean.shape == () and hin_mean.item() == 0.0
    hin_sum = solve(s0, None, t0, mode="hinge", reduction="sum")
    assert hin_sum.shape == () and hin_sum.item() == 0.0


def check_zero_vector_denominator(solve):
    # 零向量：分母 clamp_min(1e-8) 兜底，cos=0，target=1 支损失恰为 1
    x1 = torch.tensor([[0.0, 0.0]])
    x2 = torch.tensor([[1.0, 0.0]])
    got = solve(x1, x2, torch.tensor([1]), mode="cosine", reduction="none")
    assert abs(got.item() - 1.0) < 1e-7, got.item()
    assert torch.allclose(
        got, F.cosine_embedding_loss(x1, x2, torch.tensor([1]), reduction="none"), atol=1e-6)
    # 双零向量同样有限
    got2 = solve(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([-1]),
                 mode="cosine", reduction="none")
    assert got2.item() == 0.0 and torch.isfinite(got2)


def check_invalid_args(solve):
    x1 = torch.randn(3, 2)
    x2 = torch.randn(3, 2)
    t = torch.tensor([1, -1, 1])
    try:
        solve(x1, x2, t, mode="euclidean")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 mode 应抛 ValueError")
    try:
        solve(x1, None, t, mode="cosine")
    except ValueError:
        pass
    else:
        raise AssertionError("cosine 模式 x2=None 应抛 ValueError")
    try:
        solve(x1, x2, t, mode="cosine", reduction="avg")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 reduction 应抛 ValueError")
