import math
import torch


def _oracle(new, old, adv, eps):
    """独立纯 python：手算 exp 比值、clip、min、均值取负。"""
    n = len(new)
    tot = 0.0
    for i in range(n):
        ratio = math.exp(new[i] - old[i])
        clipped = min(max(ratio, 1 - eps), 1 + eps)
        tot += min(ratio * adv[i], clipped * adv[i])
    return -tot / n


def check_example(solve):
    out = solve(
        torch.tensor([-1.0, -1.5, -0.7, -2.0]),
        torch.tensor([-1.2, -1.3, -0.9, -1.8]),
        torch.tensor([1.0, -1.0, 0.5, -2.0]),
        0.2,
    )
    assert abs(out.item() - 0.164048) < 1e-5, out.item()


def check_no_clip_when_ratio_in_range(solve):
    # ratio 全在 [0.8,1.2] 且 advantage 正 → min 取未截断（ratio<上界）
    out = solve(torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([2.0]), 0.2)
    # ratio=1, obj=min(2,2)=2, loss=-2
    assert abs(out.item() + 2.0) < 1e-6, out.item()


def check_clip_binds_positive_advantage(solve):
    # ratio=exp(1)=2.718 > 1.2, A>0 → 截断项 1.2*A 生效
    out = solve(torch.tensor([0.0]), torch.tensor([-1.0]), torch.tensor([1.0]), 0.2)
    assert abs(out.item() + 1.2) < 1e-6, out.item()


def check_matches_oracle(solve):
    torch.manual_seed(2)
    for _ in range(50):
        N = int(torch.randint(1, 7, (1,)))
        new = torch.randn(N) - 2
        old = torch.randn(N) - 2
        adv = torch.randn(N) * 3
        eps = 0.1 + float(torch.rand(1)) * 0.3
        got = float(solve(new, old, adv, eps).item())
        exp = _oracle(new.tolist(), old.tolist(), adv.tolist(), eps)
        assert abs(got - exp) < 1e-4, (got, exp)


check_example.example = True
