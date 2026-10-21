import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_outer(solve):
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([10.0, 20.0, 30.0])
    out = solve(a, b, "outer")
    expected = torch.tensor([[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]])
    assert out.shape == (2, 3), f"expected (2,3), got {tuple(out.shape)}"
    assert torch.equal(out, expected), f"outer 结果错误: got {out.tolist()}"


def check_row_scale(solve):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b = torch.tensor([10.0, 0.0, -1.0])
    out = solve(a, b, "row_scale")
    expected = torch.tensor([[10.0, 20.0], [0.0, 0.0], [-5.0, -6.0]])
    assert torch.equal(out, expected), f"按行缩放错误: got {out.tolist()}"


def check_col_bias(solve):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([100.0, -1.0])
    out = solve(a, b, "col_bias")
    expected = torch.tensor([[101.0, 1.0], [103.0, 3.0]])
    assert torch.equal(out, expected), f"按列平移错误: got {out.tolist()}"


def check_matches_vectorized_reference(solve):
    torch.manual_seed(42)
    a = torch.randn(50, 20)
    r = torch.randn(50)
    c = torch.randn(20)
    assert torch.allclose(solve(a, r, "row_scale"), a * r[:, None]), "row_scale 与向量化参考不一致"
    assert torch.allclose(solve(a, c, "col_bias"), a + c[None, :]), "col_bias 与向量化参考不一致"
    assert torch.allclose(solve(r, c, "outer"), torch.outer(r, c)), "outer 与 torch.outer 不一致"


def check_no_element_loop_result_shape(solve):
    # 广播失败会直接 raise；这里保证大形状下结果正确且不复制出错
    torch.manual_seed(0)
    a = torch.randn(128, 64)
    out = solve(a, torch.randn(128), "row_scale")
    assert out.shape == (128, 64), f"expected (128,64), got {tuple(out.shape)}"


def check_invalid_op(solve):
    try:
        solve(torch.ones(2, 2), torch.ones(2), "matmul")
    except ValueError:
        return
    raise AssertionError("非法 op 应抛出 ValueError")


check_outer.example = True
