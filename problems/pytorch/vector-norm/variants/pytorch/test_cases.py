import torch

# 示例：VX = [[3,-4,0],[0,0,-1]]，各 ord 沿 dim=1 的结果（实跑参考答案回填）
EXAMPLE_X = [[3.0, -4.0, 0.0], [0.0, 0.0, -1.0]]
EXAMPLE_ORD2 = [5.0, 1.0]
EXAMPLE_ORD0 = [2.0, 1.0]
EXAMPLE_ORDINF = [4.0, 1.0]
EXAMPLE_FLAT = 5.099019527435303      # sqrt(26)


def check_example(solve):
    x = torch.tensor(EXAMPLE_X)
    assert torch.allclose(solve(x, ord=2, dim=1), torch.tensor(EXAMPLE_ORD2), atol=1e-6)
    assert torch.allclose(solve(x, ord=None, dim=1), torch.tensor(EXAMPLE_ORD2), atol=1e-6)
    assert torch.allclose(solve(x, ord=0, dim=1), torch.tensor(EXAMPLE_ORD0), atol=1e-6)
    assert torch.allclose(solve(x, ord="inf", dim=1), torch.tensor(EXAMPLE_ORDINF), atol=1e-6)
    assert abs(float(solve(x)) - EXAMPLE_FLAT) < 1e-6


check_example.example = True


def check_matches_torch_ords(solve):
    torch.manual_seed(201)
    x = torch.randn(5, 6, 7)
    for ordv in [None, 1, 2, 3, 4, 0.5, 0, float("inf"), float("-inf"), "inf", "-inf"]:
        # torch.linalg.vector_norm 的 ord 默认值是 2 且只接 Number（不接 None / "inf" 字符串），
        # 本题把 None 定义为 2-范数、并额外接受 "inf"/"-inf" 写法
        if ordv is None:
            ord_ref = 2
        elif isinstance(ordv, str):
            ord_ref = float(ordv)
        else:
            ord_ref = ordv
        for dimv in [None, 0, 1, -1]:
            ref = torch.linalg.vector_norm(x, ord=ord_ref, dim=dimv)
            got = solve(x, ord=ordv, dim=dimv)
            assert got.shape == ref.shape, (ordv, dimv, got.shape, ref.shape)
            assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), (ordv, dimv)
            assert got.dtype == x.dtype


def check_negative_one_is_harmonic(solve):
    # ord=-1 不是绝对值和，而是 (Σ|x|^-1)^-1；含 0 元素时结果为 0
    x = torch.tensor([1.0, 2.0, 3.0])
    ref = torch.linalg.vector_norm(x, ord=-1)
    got = solve(x, ord=-1)
    assert torch.allclose(got, ref, atol=1e-6)
    assert 0.5 < float(got) < 0.55, float(got)
    z = torch.tensor([1.0, 2.0, 0.0])
    assert float(solve(z, ord=-1)) == 0.0
    assert torch.allclose(solve(z, ord=-1), torch.linalg.vector_norm(z, ord=-1))
    # 而 ord=1 才是绝对值和
    assert float(solve(x, ord=1)) == 6.0


def check_multi_axis_and_keepdim(solve):
    torch.manual_seed(202)
    x = torch.randn(3, 4, 5)
    for dimv in [(0, 1), (1, 2), (-2, -1), (0, 2), tuple(range(3))]:
        for keep in [False, True]:
            ref = torch.linalg.vector_norm(x, ord=2, dim=dimv, keepdim=keep)
            got = solve(x, dim=dimv, keepdim=keep)
            assert got.shape == ref.shape, (dimv, keep)
            assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
    # dim=None + keepdim=True 是本题扩展口径：所有轴都保留为长度 1
    assert solve(x, dim=None, keepdim=True).shape == (1, 1, 1)
    assert solve(x, dim=None, keepdim=False).dim() == 0


def check_overflow_free_p_norm(solve):
    # 大数值 + 高次幂：朴素 Σ|x|^p 会 inf，稳定式必须给出有限且正确的值
    x = torch.tensor([[1e3, 2e3, -3e3], [-1e2, 1e2, 0.0]])
    ref = torch.linalg.vector_norm(x, ord=8, dim=1)
    got = solve(x, ord=8, dim=1)
    assert torch.isfinite(got).all(), got.tolist()
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5), (got.tolist(), ref.tolist())
    tiny = torch.tensor([[1e-8, 2e-8, 3e-8]])
    assert torch.allclose(solve(tiny, ord=4, dim=1),
                          torch.linalg.vector_norm(tiny, ord=4, dim=1), rtol=1e-5)
    # 全零向量各 ord 都是 0（ord=0 计非零个数亦为 0）
    z = torch.zeros(2, 5)
    for ordv in [None, 1, 3, 0, "inf"]:
        assert torch.equal(solve(z, ord=ordv, dim=1), torch.zeros(2)), ordv


def check_invalid_args(solve):
    x = torch.randn(3, 4)
    for ordv in ["fro", "nuc"]:
        try:
            solve(x, ord=ordv)
        except ValueError:
            continue
        raise AssertionError(f"ord={ordv} 应抛 ValueError")
    for dimv in [5, -5, (0, 0), ()]:
        try:
            solve(x, dim=dimv)
        except ValueError:
            continue
        raise AssertionError(f"dim={dimv} 应抛 ValueError")
