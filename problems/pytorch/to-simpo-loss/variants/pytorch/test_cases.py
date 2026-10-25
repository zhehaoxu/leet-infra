import math

import torch

LPW = torch.tensor([-2.0, -1.5])
LPL = torch.tensor([-3.0, -2.5])
NW = torch.tensor([4.0, 3.0])
NL = torch.tensor([5.0, 4.0])
BETA, GAMMA = 2.0, 0.5


def _oracle(lpw, lpl, nw, nl, beta, gamma):
    """独立重算：logsigmoid(m)=-log1p(exp(-m))，逐元素 Python 稳定式（结构异于参考解的向量化）。"""
    total = 0.0
    for i in range(lpw.numel()):
        r_w = lpw[i].item() / nw[i].item()
        r_l = lpl[i].item() / nl[i].item()
        m = beta * (r_w - r_l) - gamma
        total += math.log1p(math.exp(-m))
    return total / lpw.numel()


def check_example(solve):
    y = solve(LPW, LPL, NW, NL, BETA, GAMMA)
    assert abs(y.item() - 0.8401473164558411) < 1e-5, f"got {y.item()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(31)
    lpw = -torch.rand(10) * 3 - 0.5
    lpl = -torch.rand(10) * 3 - 0.5
    nw = torch.randint(2, 8, (10,)).float()
    nl = torch.randint(2, 8, (10,)).float()
    for beta, gamma in [(2.0, 0.5), (1.0, 0.0), (0.5, 1.25)]:
        assert abs(solve(lpw, lpl, nw, nl, beta, gamma).item() - _oracle(lpw, lpl, nw, nl, beta, gamma)) < 1e-5


def check_gamma_zero_naive(solve):
    # γ=0 时 loss=-mean(logsigmoid(beta*(r_w-r_l)))，用固定小例手算对照
    lpw = torch.tensor([-1.0]); lpl = torch.tensor([-2.0])
    nw = torch.tensor([1.0]); nl = torch.tensor([1.0])
    # r_w=−1, r_l=−2, m=beta*(1)=1.0；loss=log1p(exp(-1))
    y = solve(lpw, lpl, nw, nl, 1.0, 0.0)
    assert abs(y.item() - math.log1p(math.exp(-1.0))) < 1e-6, f"got {y.item()}"


def check_length_normalization_matters(solve):
    # 同 logp、等长时 r_w==r_l、margin=0、loss=log2；长度不同则归一奖励不等、loss 改变（长度归一是 SimPO 灵魂）
    lpw = torch.tensor([-4.0]); lpl = torch.tensor([-4.0])
    y_equal = solve(lpw, lpl, torch.tensor([4.0]), torch.tensor([4.0]), 1.0, 0.0)
    y_diff = solve(lpw, lpl, torch.tensor([2.0]), torch.tensor([4.0]), 1.0, 0.0)
    assert abs(y_equal.item() - math.log(2.0)) < 1e-6, f"got {y_equal.item()}"
    assert y_diff.item() > y_equal.item(), "长度不等应打破 r_w==r_l、抬高损失"


def check_returns_scalar(solve):
    y = solve(LPW, LPL, NW, NL, BETA, GAMMA)
    assert y.dim() == 0 and y.dtype == torch.float32, (y.dim(), y.dtype)
