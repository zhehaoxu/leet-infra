"""Test cases for numerically-stable-softmax.

每个 check_* 函数接收用户的 solve 可调用对象，断言失败即该用例不通过。
"""

import torch


def _ref(logits, mask=None, dim=-1):
    """float64 参考实现（数学精确，允许使用原生 softmax 逻辑）。"""
    x = logits.double()
    if mask is None:
        m = x.amax(dim=dim, keepdim=True)
        e = torch.exp(x - m)
        return e / e.sum(dim=dim, keepdim=True)
    neg_inf = torch.tensor(float("-inf"), dtype=x.dtype)
    m = torch.where(mask, x, neg_inf).amax(dim=dim, keepdim=True)
    all_masked = torch.isinf(m) & (m < 0)
    m = torch.where(all_masked, torch.zeros_like(m), m)
    e = torch.exp(torch.where(mask, x - m, neg_inf))
    s = e.sum(dim=dim, keepdim=True)
    out = e / s
    return torch.where(all_masked, torch.zeros_like(out), out)


def check_basic_probability_simplex(solve):
    torch.manual_seed(0)
    logits = torch.randn(3, 5)
    out = solve(logits)
    assert out.shape == logits.shape, f"shape {tuple(out.shape)} != {tuple(logits.shape)}"
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-6), f"row sums = {sums}"
    assert bool((out >= 0).all()), f"negative probabilities: {out}"


def check_overflow_safe_no_nan(solve):
    logits = torch.tensor([[1e4, -1e4, 0.0], [500.0, 400.0, 300.0], [-1e4, -1e4, -1e4]])
    out = solve(logits)
    assert torch.isfinite(out).all(), f"non-finite output: {out}"
    expected = _ref(logits)
    assert torch.allclose(out.double(), expected, atol=1e-6), f"got {out}, expected {expected}"


def check_mask_positions_zero_and_renormalized(solve):
    torch.manual_seed(1)
    logits = torch.randn(2, 4) * 10
    mask = torch.tensor([[True, False, True, True], [False, False, True, False]])
    out = solve(logits, mask)
    for pos in [(0, 1), (1, 0), (1, 3)]:
        assert out[pos].item() == 0.0, f"masked position {pos} = {out[pos].item()}"
    valid_sums = (out * mask).sum(dim=-1)
    assert torch.allclose(valid_sums, torch.ones(2), atol=1e-6), f"valid sums = {valid_sums}"
    assert torch.allclose(out.double(), _ref(logits, mask), atol=1e-6), "mismatch with reference"


def check_fully_masked_group_returns_zeros(solve):
    torch.manual_seed(2)
    logits = torch.randn(2, 3) * 20
    mask = torch.tensor([[False, False, False], [True, False, True]])
    out = solve(logits, mask)
    assert torch.isfinite(out).all(), f"nan/inf in output: {out}"
    assert out[0].tolist() == [0.0, 0.0, 0.0], f"fully masked row = {out[0]}"
    assert torch.allclose(out[1].double(), _ref(logits, mask)[1], atol=1e-6)


def check_dim_zero_with_mask(solve):
    torch.manual_seed(3)
    logits = torch.randn(4, 3) * 5
    mask = torch.tensor(
        [[True, True, False], [False, True, True], [True, True, True], [True, False, False]]
    )
    out = solve(logits, mask, dim=0)
    assert out.shape == logits.shape
    assert torch.allclose(out.double(), _ref(logits, mask, dim=0), atol=1e-6), "dim=0 mismatch"


def check_matches_torch_softmax_no_mask(solve):
    torch.manual_seed(4)
    logits = torch.randn(5, 7)
    out = solve(logits)
    assert torch.allclose(out, torch.softmax(logits, dim=-1), atol=1e-6), (
        "与 torch.softmax 结果不一致"
    )


# 样例用例：点「运行」只执行这些；「提交」执行全部用例
check_basic_probability_simplex.example = True
check_matches_torch_softmax_no_mask.example = True
