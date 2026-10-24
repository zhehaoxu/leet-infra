import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]])
    u = torch.tensor([[0.5, 0.9, 0.1, 0.3]])
    y = solve(logits, u, tau=1.0)
    exp = torch.tensor([[0.2798, 0.6773, 0.0188, 0.0241]])
    assert torch.allclose(y, exp, atol=1e-3), y.tolist()


def check_sum_to_one(solve):
    torch.manual_seed(0)
    logits = torch.randn(5, 7)
    u = torch.rand(5, 7) * 0.98 + 0.01      # 落在 (0,1)
    y = solve(logits, u, tau=0.7)
    assert torch.allclose(y.sum(-1), torch.ones(5), atol=1e-5)


def check_matches_definition(solve):
    torch.manual_seed(1)
    logits = torch.randn(3, 6)
    u = torch.rand(3, 6) * 0.98 + 0.01
    tau = 0.5
    g = -torch.log(-torch.log(u))
    ref = F.softmax((logits + g) / tau, dim=-1)
    assert torch.allclose(solve(logits, u, tau), ref, atol=1e-6)


def check_low_tau_is_hard(solve):
    # tau 极小时输出近似 one-hot：最大值接近 1
    logits = torch.tensor([[3.0, 1, 1, 1]])
    u = torch.tensor([[0.5, 0.2, 0.2, 0.2]])
    y = solve(logits, u, tau=1e-3)
    assert abs(y.max().item() - 1.0) < 1e-3, "小温度应逼近 one-hot"


check_basic.example = True
