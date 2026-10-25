import torch

VN = torch.tensor([0.6, 1.2, -0.3, 0.9])
VO = torch.tensor([0.5, 1.0, -0.2, 0.8])
RT = torch.tensor([0.7, 1.5, -0.5, 0.8])
EPS = 0.2


def _oracle(vn, vo, rt, eps):
    """独立重算：逐元素 Python clip 与 max，再平均（结构异于参考解的向量化）。"""
    acc = 0.0
    for i in range(vn.numel()):
        c = min(max(vn[i].item(), vo[i].item() - eps), vo[i].item() + eps)
        acc += max((vn[i].item() - rt[i].item()) ** 2, (c - rt[i].item()) ** 2)
    return acc / vn.numel()


def check_example(solve):
    y = solve(VN, VO, RT, EPS)
    assert abs(y.item() - 0.037499986588954926) < 1e-5, f"got {y.item()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(22)
    vn = torch.randn(10)
    vo = torch.randn(10)
    rt = torch.randn(10)
    for eps in [0.05, 0.2, 0.5]:
        assert abs(solve(vn, vo, rt, eps).item() - _oracle(vn, vo, rt, eps)) < 1e-5, eps


def check_large_eps_is_plain_mse(solve):
    # ε 足够大时 clip 不触发 -> 退化为普通 MSE
    y = solve(VN, VO, RT, 100.0)
    mse = ((VN - RT) ** 2).mean()
    assert torch.allclose(y, mse, atol=1e-6, rtol=1e-6), "大 ε 应等于普通 MSE"


def check_clip_triggers_and_maxpicks_larger(solve):
    # 构造越界：vn 远离 vo，clip 把 vn 拉回 vo 附近；取 max 应选两者中较大的平方误差
    vn = torch.tensor([5.0])
    vo = torch.tensor([0.0])
    rt = torch.tensor([0.0])
    eps = 1.0
    # clip(5, -1, 1)=1 ; unclipped err=25 ; clipped err=1 ; max=25
    y = solve(vn, vo, rt, eps)
    assert abs(y.item() - 25.0) < 1e-4, f"got {y.item()}"


def check_returns_scalar(solve):
    y = solve(VN, VO, RT, EPS)
    assert y.dim() == 0 and y.dtype == torch.float32, (y.dim(), y.dtype)
