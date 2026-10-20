"""Test cases for layernorm-forward."""

import torch
import torch.nn.functional as F


def check_matches_functional_with_affine(solve):
    torch.manual_seed(0)
    x = torch.randn(4, 8) * 3 + 2
    w = torch.randn(8) * 0.5 + 1
    b = torch.randn(8) * 0.2
    out = solve(x, w, b)
    ref = F.layer_norm(x, (8,), w, b, eps=1e-5)
    assert torch.allclose(out, ref, atol=1e-5), f"与 F.layer_norm 最大差 {(out - ref).abs().max()}"


def check_no_affine_params(solve):
    torch.manual_seed(1)
    x = torch.randn(16, 32)
    out = solve(x)
    ref = F.layer_norm(x, (32,), eps=1e-5)
    assert torch.allclose(out, ref, atol=1e-5), "weight/bias=None 时应等价于纯归一化"


def check_constant_row_no_nan(solve):
    x = torch.full((3, 6), 7.5)
    out = solve(x)
    assert torch.isfinite(out).all(), f"常数行出现 nan/inf: {out}"
    assert torch.allclose(out, torch.zeros(3, 6), atol=1e-4), f"常数行输出应≈0: {out}"


def check_3d_input_normalizes_last_dim(solve):
    torch.manual_seed(2)
    x = torch.randn(2, 5, 10) * 2
    out = solve(x)
    assert out.shape == x.shape
    ref = F.layer_norm(x, (10,), eps=1e-5)
    assert torch.allclose(out, ref, atol=1e-5), "3D 输入应沿最后一维归一化"
    row = out[0, 0]
    assert abs(row.mean().item()) < 1e-5, f"归一化后均值应为 0: {row.mean()}"


def check_eps_is_respected(solve):
    torch.manual_seed(3)
    x = torch.randn(4, 8)
    out = solve(x, eps=1.0)
    ref = F.layer_norm(x, (8,), eps=1.0)
    assert torch.allclose(out, ref, atol=1e-5), "eps 大值时必须参与除法"


def check_input_not_modified_and_dtype_kept(solve):
    x = torch.randn(3, 7, dtype=torch.float64)
    snapshot = x.clone()
    out = solve(x)
    assert torch.equal(x, snapshot), "输入张量不得被 in-place 修改"
    assert out.dtype == torch.float64, f"dtype 应保持 float64，实际 {out.dtype}"


# 样例用例：点「运行」只执行这些；「提交」执行全部用例
check_matches_functional_with_affine.example = True
check_no_affine_params.example = True
