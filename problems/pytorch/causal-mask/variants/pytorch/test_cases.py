import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    scores = torch.ones(3, 3)
    y = solve(scores)
    assert y[0, 0] == 1.0 and torch.isinf(y[0, 1]) and y[0, 1] < 0
    assert y[2, 0] == 1.0 and y[2, 2] == 1.0
    expected = torch.tensor([[1.0, float("-inf"), float("-inf")],
                             [1.0, 1.0, float("-inf")],
                             [1.0, 1.0, 1.0]])
    assert torch.equal(y.masked_fill(~torch.isinf(y), 0), expected.masked_fill(~torch.isinf(expected), 0)) \
        and torch.isinf(y).eq(torch.isinf(expected)).all(), f"got {y.tolist()}"


def check_matches_tril_fill(solve):
    torch.manual_seed(42)
    scores = torch.randn(2, 4, 5, 5)
    y = solve(scores)
    keep = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    exp = scores.masked_fill(~keep, float("-inf"))
    assert torch.equal(y, exp), "与 tril + masked_fill(-inf) 语义不一致"


def check_softmax_semantics(solve):
    # 掩码后 softmax：每行只在下三角有概率；第 0 行必须全部集中到自己
    torch.manual_seed(0)
    scores = torch.randn(6, 6)
    y = solve(scores)
    attn = torch.softmax(y, dim=-1)
    assert torch.allclose(attn[0], torch.tensor([1.0] + [0.0] * 5), atol=1e-7), "第 0 行只能看自己"
    assert torch.allclose(attn[5], torch.softmax(scores[5], -1), atol=1e-7), "最后一行不受影响"
    upper = attn.triu(diagonal=1)
    assert (upper == 0).all(), "上三角概率必须为 0"
    assert torch.allclose(attn.sum(-1), torch.ones(6), atol=1e-6), "每行概率和为 1"


def check_batch_broadcastable_shape(solve):
    # 对任意前置 batch/head 维都可应用
    torch.manual_seed(1)
    scores = torch.randn(2, 8, 4, 4)
    y = solve(scores)
    assert y.shape == scores.shape
    assert torch.isfinite(y[:, :, 3]).all(), "最后一行不应被掩"
    assert torch.isinf(y[:, :, 0, 3]).all() and (y[:, :, 0, 3] < 0).all(), "角落应为 -inf"


def check_dtype_preserved(solve):
    scores = torch.randn(3, 3, dtype=torch.float64)
    y = solve(scores)
    assert y.dtype == torch.float64, "dtype 应保持"
    assert torch.isfinite(y.tril()).all() if hasattr(y, "tril") else True


check_basic.example = True
