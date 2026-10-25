import math

import torch

LR = torch.tensor([0.5, -0.3, 0.2, -0.8])
LABEL = torch.tensor([1.0, 0.0, 1.0, 0.0])
ZREF, BETA, LD, LU = 0.1, 0.1, 1.0, 1.0


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def _oracle(lr, label, zref, beta, ld, lu):
    """独立重算：用 1-σ(x)=σ(-x) 改写、逐元素 Python 循环（结构异于参考解的向量化 where）。"""
    total = 0.0
    for i in range(lr.numel()):
        z = lr[i].item()
        if label[i].item() > 0.5:
            total += ld * _sig(-beta * (z - zref))
        else:
            total += lu * _sig(-beta * (zref - z))
    return total / lr.numel()


def check_example(solve):
    y = solve(LR, LABEL, ZREF, BETA, LD, LU)
    assert abs(y.item() - 0.4887544512748718) < 1e-5, f"got {y.item()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(24)
    lr = torch.randn(12)
    label = (torch.rand(12) > 0.5).float()
    for zref, beta, ld, lu in [(0.0, 0.1, 1.0, 1.0), (0.2, 0.5, 1.0, 0.5), (-0.1, 0.05, 1.3, 1.0)]:
        assert abs(solve(lr, label, zref, beta, ld, lu).item() - _oracle(lr, label, zref, beta, ld, lu)) < 1e-5


def check_at_reference_is_lambda_half(solve):
    # 所有 z == z_ref 时 σ(0)=0.5，损失=λ/2（此处 λ_D=2,λ_U=1，混合标签）
    z = torch.tensor([0.3, 0.3, 0.3])
    lab = torch.tensor([1.0, 0.0, 1.0])
    y = solve(z, lab, 0.3, 0.1, 2.0, 1.0)
    # 期望 (2*0.5 + 1*0.5 + 2*0.5)/3 = (1+0.5+1)/3 = 2.5/3
    assert abs(y.item() - (2.5 / 3)) < 1e-6, f"got {y.item()}"


def check_good_sample_rewarded(solve):
    # 好样本 z 越大（远超 z_ref）损失越小；坏样本 z 越大损失越大（方向相反）
    good_low = solve(torch.tensor([0.0]), torch.tensor([1.0]), 0.1, 1.0, 1.0, 1.0).item()
    good_high = solve(torch.tensor([2.0]), torch.tensor([1.0]), 0.1, 1.0, 1.0, 1.0).item()
    assert good_high < good_low, "好样本 z 更大应损失更小"
    bad_low = solve(torch.tensor([0.0]), torch.tensor([0.0]), 0.1, 1.0, 1.0, 1.0).item()
    bad_high = solve(torch.tensor([2.0]), torch.tensor([0.0]), 0.1, 1.0, 1.0, 1.0).item()
    assert bad_high > bad_low, "坏样本 z 更大应损失更大"


def check_returns_scalar(solve):
    y = solve(LR, LABEL, ZREF, BETA, LD, LU)
    assert y.dim() == 0 and y.dtype == torch.float32, (y.dim(), y.dtype)
