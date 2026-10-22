import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = solve(x, (1, 1, 1, 1))  # 左1右1（最后一维W），上1下1（dim=-2 H）
    expected = [
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
    ]
    assert torch.equal(y, torch.tensor(expected, dtype=torch.float32)), f"got {y.tolist()}"


def check_last_dim_first(solve):
    # padding 顺序：最后两元素补最后一维，之前两元素补倒数第二维
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = solve(x, (1, 0))  # 仅最后一维左侧补 1
    assert torch.equal(y, torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]])), f"顺序错误 got {y.tolist()}"


def check_matches_torch_2d(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    assert torch.equal(solve(x, (1, 2, 0, 1)), F.pad(x, (1, 2, 0, 1)))


def check_value_fill(solve):
    x = torch.ones(2, 2)
    y = solve(x, (1, 1), mode="constant", value=9.0)
    assert torch.equal(y, F.pad(x, (1, 1), mode="constant", value=9.0)), "fill 值应生效"


def check_1d(solve):
    x = torch.tensor([1.0, 2.0, 3.0])
    y = solve(x, (2, 1))
    assert torch.equal(y, F.pad(x, (2, 1))), "一维 pad 应一致"


def check_mode_reject(solve):
    x = torch.ones(2, 2)
    try:
        solve(x, (1, 1), mode="reflect")
        raise AssertionError("应拒绝非 constant 模式")
    except AssertionError:
        raise
    except Exception:
        pass  # 允许以任意错误拒绝


check_basic.example = True
