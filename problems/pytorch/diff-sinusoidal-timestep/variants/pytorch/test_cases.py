import math

import torch


def _oracle(t, dim):
    """独立重算：Python math 逐频率填 sin 半段 / cos 半段（结构异于参考解的向量化 cat）。"""
    half = dim // 2
    out = torch.zeros(dim)
    for i in range(half):
        angle = t / (10000.0 ** (i / half))
        out[i] = math.sin(angle)
        out[half + i] = math.cos(angle)
    return out


def check_example(solve):
    y = solve(7, 8)
    exp = torch.tensor([0.6569865942001343, 0.6442176103591919, 0.06994284689426422,
                        0.006999942008405924, 0.7539022564888, 0.7648422122001648,
                        0.9975510239601135, 0.9999755024909973])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    for t, dim in [(0, 6), (1, 16), (17, 12), (123, 32), (999, 4)]:
        y = solve(t, dim)
        assert torch.allclose(y, _oracle(t, dim), atol=1e-5, rtol=1e-5), (t, dim)


def check_t_zero_boundary(solve):
    y = solve(0, 6)
    assert torch.allclose(y, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), atol=1e-6, rtol=1e-6), \
        "t=0 时 sin 半段应全 0、cos 半段应全 1"


def check_sin_cos_order(solve):
    # sin 半段绝对值 <=1 且首项=sin(t)，cos 半段首项=cos(t)（t 最 freq=1 那项）
    y = solve(7, 8)
    assert abs(y[0].item() - math.sin(7.0)) < 1e-5
    assert abs(y[4].item() - math.cos(7.0)) < 1e-5


def check_shape_and_dtype(solve):
    y = solve(5, 20)
    assert y.shape == (20,), tuple(y.shape)
    assert y.dtype == torch.float32, y.dtype
