import math

import torch


def _oracle(T, kind):
    """独立重算：逐 t 用 math 计算 beta 并顺序累乘（结构异于参考解的向量化 cumprod）。"""
    prev = 1.0
    ab = torch.ones(T)
    for t in range(T):
        if kind == "linear":
            b = 1e-4 + (0.02 - 1e-4) * t / (T - 1)
        else:
            s = 0.008

            def ff(u):
                return math.cos(((u / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            ac_t = ff(t + 1) / ff(0)
            ac_p = ff(t) / ff(0)
            b = min(max(1 - ac_t / ac_p, 0.0), 0.999)
        prev *= (1 - b)
        ab[t] = prev
    return ab


def check_example(solve):
    lin = solve(4, "linear")
    exp_lin = torch.tensor([0.9998999834060669, 0.9931672811508179, 0.9798920154571533, 0.9602941870689392])
    assert torch.allclose(lin, exp_lin, atol=1e-5, rtol=1e-5), f"linear got {lin.tolist()}"
    cos = solve(4, "cosine")
    exp_cos = torch.tensor([0.8470122218132019, 0.49384361505508423, 0.1442720741033554, 0.00014427020505536348])
    assert torch.allclose(cos, exp_cos, atol=1e-5, rtol=1e-5), f"cosine got {cos.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    for kind in ["linear", "cosine"]:
        for T in [2, 5, 50, 1000]:
            assert torch.allclose(solve(T, kind), _oracle(T, kind), atol=1e-5, rtol=1e-5), (kind, T)


def check_monotone_nonincreasing(solve):
    for kind in ["linear", "cosine"]:
        ab = solve(20, kind)
        assert bool((ab[1:] <= ab[:-1] + 1e-7).all()), f"{kind} 应单调不增"
        assert ab[0].item() < 1.0 and ab[-1].item() <= ab[0].item()


def check_first_alpha_bar(solve):
    # alpha_bar[0] = 1 - beta_1；linear 首 beta=1e-4
    ab = solve(10, "linear")
    assert abs(ab[0].item() - (1 - 1e-4)) < 1e-6, ab[0].item()


def check_length_and_dtype(solve):
    for kind in ["linear", "cosine"]:
        ab = solve(37, kind)
        assert ab.shape == (37,) and ab.dtype == torch.float32, (tuple(ab.shape), ab.dtype)
