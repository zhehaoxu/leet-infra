import math

import torch

REW = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 6.0])
G = 3


def _oracle(rewards, g):
    """独立重算：逐组纯 Python 求均值、总体标准差(ddof=0)再标准化。"""
    out = torch.zeros_like(rewards)
    n = rewards.numel()
    for start in range(0, n, g):
        grp = [rewards[start + k].item() for k in range(g)]
        m = sum(grp) / g
        var = sum((z - m) ** 2 for z in grp) / g
        s = math.sqrt(var)
        for k in range(g):
            out[start + k] = (grp[k] - m) / s
    return out


def check_example(solve):
    y = solve(REW, G)
    exp = torch.tensor([-1.2247447967529297, 0.0, 1.2247447967529297,
                        -0.7071067690849304, -0.7071067690849304, 1.4142135381698608])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(23)
    for n, g in [(12, 4), (9, 3), (20, 5), (6, 2)]:
        rewards = torch.rand(n) * 10 + 1
        assert torch.allclose(solve(rewards, g), _oracle(rewards, g), atol=1e-5, rtol=1e-5), (n, g)


def check_group_mean_zero_std_one(solve):
    y = solve(REW, G).view(-1, G)
    assert torch.allclose(y.mean(dim=1), torch.zeros(y.shape[0]), atol=1e-5), "每组优势均值应为 0"
    gstd = y.std(dim=1, unbiased=False)
    assert torch.allclose(gstd, torch.ones(y.shape[0]), atol=1e-5), "每组总体标准差应为 1"


def check_groups_are_independent(solve):
    # 缩放其中一组不影响另一组的标准化结果（组内归一化的独立性）
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    y = solve(rewards, 3)
    g0 = y[:3]
    g0_scaled = solve(torch.tensor([1.0, 2.0, 3.0, -999.0, 5.0, 77.0]), 3)[:3]
    assert torch.allclose(g0, g0_scaled, atol=1e-5, rtol=1e-5), "另一组内容不应影响本组优势"


def check_shape_dtype(solve):
    y = solve(REW, G)
    assert y.shape == REW.shape and y.dtype == torch.float32
