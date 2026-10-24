import torch
import torch.nn.functional as F

MODE = dict(mode="bilinear", padding_mode="zeros", align_corners=False)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    grid = torch.tensor([[[[0.0, 0.0], [-1.0, -1.0]],
                          [[1.0, 1.0], [0.5, 0.5]]]])
    y = solve(x, grid)
    exp = torch.tensor([[[[4.0, 0.0], [2.0, 7.0]]]], dtype=torch.float32)
    assert torch.allclose(y, exp, atol=1e-4), f"expected {exp.tolist()}, got {y.tolist()}"


def check_matches_F(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 2, 4, 5)
    grid = torch.randn(2, 3, 3, 2).clamp(-1, 1)
    y = solve(x, grid)
    ref = F.grid_sample(x, grid, **MODE)
    assert y.shape == (2, 2, 3, 3)
    assert torch.allclose(y, ref, atol=1e-4), "应与 F.grid_sample 一致"


def check_identity_grid(solve):
    # 恒等网格应近似还原原图（align_corners=False 下栅格对齐到像素中心）
    h = w = 4
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    grid = torch.stack([gx, gy], dim=-1)[None]      # (1,h,w,2)
    torch.manual_seed(1)
    x = torch.randn(1, 3, h, w)
    y = solve(x, grid)
    ref = F.grid_sample(x, grid, **MODE)
    assert torch.allclose(y, ref, atol=1e-4)


def check_out_of_bounds_zero(solve):
    # 完全落在图外的网格 → 全 0（padding_mode='zeros'）
    x = torch.ones(1, 1, 3, 3)
    # 归一化坐标无法 >1，用一个角落使得双线性主要取到补零区
    grid = torch.full((1, 1, 1, 2), -1.0)
    y = solve(x, grid)
    ref = F.grid_sample(x, grid, **MODE)
    assert torch.allclose(y, ref, atol=1e-4)


check_basic.example = True
