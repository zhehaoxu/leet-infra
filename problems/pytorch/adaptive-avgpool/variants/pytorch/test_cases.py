import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0]]]])
    y = solve(x, 2)
    expected = torch.tensor([[[[3.5, 5.5], [11.5, 13.5]]]])
    assert torch.allclose(y, expected), f"got {y.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 7, 5)
    for out in (1, 2, 3, 4, (3, 7), (1, 5)):
        y = solve(x, out)
        exp = F.adaptive_avg_pool2d(x, out)
        assert torch.allclose(y, exp, atol=1e-6), f"output_size={out} 与 F.adaptive_avg_pool2d 不一致"


def check_nondivisible_bounds(solve):
    # 5 -> 4：区间长度不均等（[0,1),[1,3),[3,4),[4,5)），朴素等宽切分必错
    x = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    y = solve(x, (4, 4))
    exp = F.adaptive_avg_pool2d(x, (4, 4))
    assert torch.allclose(y, exp, atol=1e-6), "非整除尺寸的边界公式错误"
    # 列区间：[0,2),[1,3),[2,4),[3,5)；行区间 4->4 为单行
    assert torch.allclose(y[0, 0, 1, 1], x[0, 0, 1:2, 1:3].mean(), atol=1e-6), "内部段均值错误"


def check_identity_when_smaller(solve):
    # output >= input 时每段是单元素，等价于恒等
    x = torch.randn(1, 2, 3, 3)
    y = solve(x, 3)
    assert torch.allclose(y, x, atol=1e-6), "output_size == input 应恒等"
    y2 = solve(x, (5, 5))
    assert y2.shape == (1, 2, 5, 5)


def check_global_via_one(solve):
    torch.manual_seed(0)
    x = torch.randn(4, 8, 11, 9)
    assert torch.allclose(solve(x, 1), F.adaptive_avg_pool2d(x, 1), atol=1e-6), "GAP 口径不一致"


check_basic.example = True
