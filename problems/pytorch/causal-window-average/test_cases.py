"""Test cases for causal-window-average."""

import torch


def _naive(x, kernel):
    """朴素三重循环参考实现（t-i<0 补 0）。"""
    B, L = x.shape
    W = kernel.numel()
    s = kernel.sum().item()
    out = torch.zeros(B, L, dtype=x.dtype)
    for b in range(B):
        for t in range(L):
            acc = 0.0
            for i in range(W):
                if t - i >= 0:
                    acc += kernel[i].item() * x[b, t - i].item()
            out[b, t] = acc / s
    return out


def check_identity_kernel(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 10)
    out = solve(x, torch.tensor([1.0]))
    assert torch.allclose(out, x, atol=1e-6), "kernel=[1] 必须原样输出"


def check_uniform_moving_average(solve):
    torch.manual_seed(1)
    x = torch.randn(3, 12)
    kernel = torch.ones(4)
    out = solve(x, kernel)
    ref = _naive(x, kernel)
    assert out.shape == x.shape, f"shape {tuple(out.shape)} != {tuple(x.shape)}"
    assert torch.allclose(out, ref, atol=1e-5), "与朴素循环的移动平均不一致"


def check_weighted_kernel_vs_naive(solve):
    torch.manual_seed(2)
    x = torch.randn(4, 20) * 5
    kernel = torch.rand(5) + 0.1
    out = solve(x, kernel)
    ref = _naive(x, kernel)
    assert torch.allclose(out, ref, atol=1e-5), "任意加权 kernel 与朴素实现不一致"


def _assert_prefix_causal(out, ref):
    assert torch.allclose(out[:, :6], ref[:, :6], atol=1e-6), (
        "修改未来输入影响了过去的输出——违反因果性"
    )


def check_strict_causality(solve):
    torch.manual_seed(3)
    x = torch.randn(2, 10)
    kernel = torch.rand(3) + 0.1
    out_full = solve(x, kernel)
    x_future_zeroed = x.clone()
    x_future_zeroed[:, 6:] = 0.0
    out_prefix = solve(x_future_zeroed, kernel)
    _assert_prefix_causal(out_full, out_prefix)


def check_left_zero_padding_prefix(solve):
    x = torch.ones(1, 5)
    kernel = torch.ones(3)
    out = solve(x, kernel)
    expected = torch.tensor([[1 / 3, 2 / 3, 1.0, 1.0, 1.0]])
    assert torch.allclose(out, expected, atol=1e-5), f"窗口起始段应被零填充: {out}"


def check_zero_kernel_sum_raises(solve):
    x = torch.randn(1, 4)
    try:
        solve(x, torch.tensor([1.0, -1.0]))
    except ValueError:
        return
    raise AssertionError("kernel 全和为 0 时应抛出 ValueError")


# 样例用例：点「运行」只执行这些；「提交」执行全部用例
check_identity_kernel.example = True
check_uniform_moving_average.example = True
