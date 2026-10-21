import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_flip(solve):
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = solve(x, "flip")
    expected = torch.tensor([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])
    assert torch.equal(out, expected), f"flip 结果错误: got {out.tolist()}"
    assert out.is_contiguous(), "flip 输出必须是连续内存（负 stride 视图不合格）"


def check_narrow(solve):
    x = torch.arange(1.0, 13.0).reshape(3, 4)  # 3 行 4 列
    out = solve(x, "narrow")
    assert out.shape == (3, 2), f"应取前一半列: expected (3,2), got {tuple(out.shape)}"
    expected = torch.tensor([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]])
    assert torch.equal(out, expected), f"narrow 值错误: got {out.tolist()}"


def check_concat(solve):
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = solve(x, "concat")
    assert out.shape == (2, 4), f"列数应翻倍: expected (2,4), got {tuple(out.shape)}"
    expected = torch.tensor([[1.0, 2.0, 2.0, 3.0], [3.0, 4.0, 4.0, 5.0]])
    assert torch.equal(out, expected), f"concat 值错误: got {out.tolist()}"


def check_reshape_row_major(solve):
    x = torch.arange(1.0, 7.0).reshape(2, 3)
    out = solve(x, "reshape")
    assert out.shape == (3, 2), f"expected (3,2), got {tuple(out.shape)}"
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert torch.equal(out, expected), f"reshape 应按行优先重排: got {out.tolist()}"


def check_invalid_op(solve):
    try:
        solve(torch.ones(2, 2), "transpose")
    except ValueError:
        return
    raise AssertionError("非法 op 应抛出 ValueError")


def check_input_not_modified(solve):
    x = torch.tensor([[1.0, 2.0, 3.0]])
    _ = solve(x, "flip")
    assert torch.equal(x, torch.tensor([[1.0, 2.0, 3.0]])), "不允许原地修改输入"


check_flip.example = True
