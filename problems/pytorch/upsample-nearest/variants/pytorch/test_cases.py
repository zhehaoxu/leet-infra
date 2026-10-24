import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)
    y = solve(x, 2)
    exp = torch.tensor(
        [[[[1, 1, 2, 2],
           [1, 1, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 4, 4]]]], dtype=torch.float32)
    assert torch.equal(y, exp), f"expected {exp.tolist()}, got {y.tolist()}"


def check_matches_F(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5)
    y = solve(x, 3)
    ref = F.interpolate(x, scale_factor=3, mode="nearest")
    assert y.shape == (2, 3, 12, 15)
    assert torch.equal(y, ref), "应与 F.interpolate(mode='nearest') 完全一致"


def check_block_structure(solve):
    # 每个输入像素应被复制成 scale×scale 的方块：值集合不变
    torch.manual_seed(1)
    x = torch.randint(0, 5, (1, 1, 3, 3)).float()
    y = solve(x, 2)
    assert set(y.flatten().tolist()) == set(x.flatten().tolist()), "最近邻不应产生新数值"


def check_anisotropic_scale(solve):
    torch.manual_seed(2)
    x = torch.randn(1, 2, 3, 4)
    y = solve(x, (2, 3))
    ref = F.interpolate(x, scale_factor=(2, 3), mode="nearest")
    assert torch.equal(y, ref), "支持 (sh, sw) 各向异性放大"


check_basic.example = True
