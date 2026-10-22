import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])  # (1,2,2,2)
    y_avg = solve(x, "avg")
    y_max = solve(x, "max")
    assert y_avg.shape == (1, 2, 1, 1), f"应 keepdim, got {tuple(y_avg.shape)}"
    assert torch.allclose(y_avg, torch.tensor([[[[2.5]], [[6.5]]]])), f"avg got {y_avg.tolist()}"
    assert torch.allclose(y_max, torch.tensor([[[[4.0]], [[8.0]]]])), f"max got {y_max.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(4, 8, 7, 5)
    assert torch.allclose(solve(x, "avg"), F.adaptive_avg_pool2d(x, 1), atol=1e-6), "GAP 不一致"
    assert torch.allclose(solve(x, "max"), F.adaptive_max_pool2d(x, 1), atol=1e-6), "GMP 不一致"


def check_per_channel_independence(solve):
    # 归约不跨通道：某通道变化不影响其他通道
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 4)
    y1 = solve(x, "avg")
    x2 = x.clone()
    x2[:, 1] = 100.0
    y2 = solve(x2, "avg")
    assert torch.allclose(y1[:, 0], y2[:, 0], atol=1e-7), "通道 0 不应受影响"
    assert torch.allclose(y2[:, 1], torch.full((2, 1, 1), 100.0), atol=1e-5), "常通道均值应为 100"


def check_3d_input(solve):
    # (N, C, L) 序列输入也应工作
    torch.manual_seed(1)
    x = torch.randn(2, 5, 9)
    assert torch.allclose(solve(x, "avg"), x.mean(-1, keepdim=True), atol=1e-6)
    assert torch.allclose(solve(x, "max"), x.amax(-1, keepdim=True), atol=1e-6)
    assert solve(x, "avg").shape == (2, 5, 1)


def check_bad_mode(solve):
    x = torch.randn(1, 2, 3, 3)
    try:
        solve(x, "sum")
    except (ValueError, KeyError):
        return
    raise AssertionError("非法 mode 应抛出 ValueError")


check_basic.example = True
