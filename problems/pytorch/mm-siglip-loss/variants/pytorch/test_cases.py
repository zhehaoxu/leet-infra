import math

import torch


def _oracle(sim, scale, bias):
    b = sim.shape[0]
    total = 0.0
    for i in range(b):
        for j in range(b):
            m = (1.0 if i == j else -1.0) * (scale * sim[i, j].item() + bias)
            total += -math.log1p(math.exp(-m))
    return -total / b


def check_example(solve):
    sim = torch.tensor([[1.0, 0.2], [0.3, 0.9]])
    y = solve(sim, 5.0, -1.0)
    assert abs(y.item() - 0.857562243938446) < 1e-5, f"got {y.item()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(41)
    for b in [2, 4, 5]:
        sim = torch.randn(b, b)
        for scale, bias in [(5.0, -1.0), (1.0, 0.0), (0.3, 2.0)]:
            assert abs(solve(sim, scale, bias).item() - _oracle(sim, scale, bias)) < 1e-5


def check_scale_zero_bias_zero(solve):
    # scale=0,bias=0 → 每 pair y*z=0、logσ(0)=-ln2 → loss=(1/B)*B²*ln2 = B*ln2
    sim = torch.randn(3, 3)
    y = solve(sim, 0.0, 0.0)
    assert abs(y.item() - 3 * math.log(2.0)) < 1e-5, y.item()


def check_perfect_pairs_low_loss(solve):
    # 对角极大正、非对角极大负 → y⊙z 全大正 → σ→1 → loss→0
    b = 4
    sim = -torch.ones(b, b) * 30.0
    sim.fill_diagonal_(30.0)
    y = solve(sim, 1.0, 0.0)
    assert y.item() < 1e-3, y.item()


def check_returns_scalar(solve):
    sim = torch.tensor([[1.0, 0.2], [0.3, 0.9]])
    y = solve(sim, 5.0, -1.0)
    assert y.dim() == 0 and y.dtype == torch.float32, (y.dim(), y.dtype)
