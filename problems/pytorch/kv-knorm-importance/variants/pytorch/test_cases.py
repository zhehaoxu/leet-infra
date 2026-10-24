import math

import torch


def _oracle(keys):
    """独立：逐元素 sqrt(sum(x^2))。"""
    H, L, D = keys.shape
    out = torch.zeros(H, L, dtype=torch.float32)
    for h in range(H):
        for l in range(L):
            s = 0.0
            for d in range(D):
                v = float(keys[h, l, d])
                s += v * v
            out[h, l] = -math.sqrt(s)
    return out


def check_example(solve):
    keys = torch.tensor([
        [[1.0, 2.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
        [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]],
    ])
    out = solve(keys)
    exp = torch.tensor([[-3.0, 0.0, -5.0], [-0.8660254, -1.7320508, -2.0]])
    assert torch.allclose(out, exp, atol=1e-5), out


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(30):
        H = int(torch.randint(1, 5, (1,)))
        L = int(torch.randint(1, 12, (1,)))
        D = int(torch.randint(1, 16, (1,)))
        keys = torch.randn(H, L, D)
        assert torch.allclose(solve(keys), _oracle(keys), atol=1e-5)


def check_zero_key_is_zero(solve):
    keys = torch.zeros(2, 3, 4)
    out = solve(keys)
    assert torch.allclose(out, torch.zeros(2, 3)), out


def check_scale_linear(solve):
    # keys 乘 α → norm 乘 |α| → 分数乘 |α|
    keys = torch.tensor([[[1.0, 2.0, 2.0], [0.5, -0.5, 1.0]]])
    base = solve(keys)
    scaled = solve(keys * 3.0)
    assert torch.allclose(base * 3.0, scaled, atol=1e-5), (base, scaled)


def check_shape(solve):
    keys = torch.randn(3, 5, 8)
    out = solve(keys)
    assert out.shape == (3, 5), out.shape


check_example.example = True
