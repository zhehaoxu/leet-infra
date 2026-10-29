import torch

# 示例：MX = [[1,-2,3],[4,0,-1]]，八种 ord 的标量结果（实跑参考答案回填）
EXAMPLE_M = [[1.0, -2.0, 3.0], [4.0, 0.0, -1.0]]
EXAMPLE_VALUES = {
    "fro": 5.5677642822265625,
    "nuc": 7.8606367111206055,
    1: 5.0,
    -1: 2.0,
    "inf": 6.0,
    "-inf": 5.0,
    2: 4.159660339355469,
    -2: 3.7009761333465576,
}


def check_example(solve):
    m = torch.tensor(EXAMPLE_M)
    for ordv, want in EXAMPLE_VALUES.items():
        got = solve(m, ord=ordv)
        assert got.dim() == 0, (ordv, got.shape)
        assert abs(float(got) - want) < 1e-5, (ordv, float(got), want)
    # 1/-1 看列和、inf/-inf 看行和，两个方向的极值不能写反
    assert float(solve(m, ord=1)) == 5.0 and float(solve(m, ord="inf")) == 6.0


check_example.example = True


def check_matches_torch_batched(solve):
    torch.manual_seed(301)
    x = torch.randn(2, 3, 4, 5)
    for ordv in ["fro", "nuc", 1, -1, float("inf"), float("-inf"), 2, -2]:
        for dimv in [(-2, -1), (0, 1), (1, 3), (0, 2)]:
            for keep in [False, True]:
                ref = torch.linalg.matrix_norm(x, ord=ordv, dim=dimv, keepdim=keep)
                got = solve(x, ord=ordv, dim=dimv, keepdim=keep)
                assert got.shape == ref.shape, (ordv, dimv, keep, got.shape, ref.shape)
                assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), (ordv, dimv, keep)
                assert got.dtype == x.dtype


def check_induced_norm_closed_form(solve):
    # 独立重算的闭式：1-范数 = 最大列绝对和，inf-范数 = 最大行绝对和
    torch.manual_seed(302)
    x = torch.randn(4, 6)
    a = x.abs()
    assert abs(float(solve(x, ord=1)) - float(a.sum(dim=0).max())) < 1e-5
    assert abs(float(solve(x, ord=-1)) - float(a.sum(dim=0).min())) < 1e-5
    assert abs(float(solve(x, ord=float("inf"))) - float(a.sum(dim=1).max())) < 1e-5
    assert abs(float(solve(x, ord=float("-inf"))) - float(a.sum(dim=1).min())) < 1e-5
    # frobenius 与全元素 2-范数一致；nuclear >= 谱范数
    assert abs(float(solve(x, ord="fro")) - float(x.norm())) < 1e-5
    assert float(solve(x, ord="nuc")) >= float(solve(x, ord=2)) - 1e-5
    # 正交矩阵的谱范数是 1（用 Hadamard 型构造，避免随机病态）
    q = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / (2.0 ** 0.5)
    assert abs(float(solve(q, ord=2)) - 1.0) < 1e-6
    assert abs(float(solve(q, ord="nuc")) - 2.0) < 1e-6


def check_axis_layout_and_keepdim(solve):
    torch.manual_seed(303)
    x = torch.randn(2, 3, 4, 5)
    # 非相邻两轴 (0, 2)：keepdim 结果必须把长度 1 放回原来的轴位置
    got = solve(x, ord="fro", dim=(0, 2), keepdim=True)
    assert got.shape == (1, 3, 1, 5), got.shape
    assert solve(x, ord="fro", dim=(0, 2), keepdim=False).shape == (3, 5)
    # dim 顺序改变「行/列」的定义，1 与 inf 范数一般不相等
    a = solve(x, ord=1, dim=(1, 3))
    b = solve(x, ord=1, dim=(3, 1))
    assert a.shape == b.shape == (2, 4)
    assert not torch.allclose(a, b, atol=1e-4)
    # 沿 (0,1) 的 fro 与先 reshape 再算一致
    ref = x.double().pow(2).sum(dim=(0, 1)).sqrt().float()
    assert torch.allclose(solve(x, ord="fro", dim=(0, 1)), ref, atol=1e-5, rtol=1e-5)


def check_invalid_args(solve):
    x = torch.randn(3, 4, 5)
    for ordv in [3, "nuclear", 0]:
        try:
            solve(x, ord=ordv)
        except ValueError:
            continue
        raise AssertionError(f"ord={ordv} 应抛 ValueError")
    for dimv in [(1, 1), (0,), [0], "fro", (0, 3)]:
        try:
            solve(x, ord="fro", dim=dimv)
        except ValueError:
            continue
        raise AssertionError(f"dim={dimv} 应抛 ValueError")
    try:
        solve(torch.randn(5), ord="fro")
    except ValueError:
        pass
    else:
        raise AssertionError("一维输入应抛 ValueError")
