import torch

# 示例：X(3×2) · Y(2×3)，spec="ij,jk->ik"（实跑参考答案回填）
EXAMPLE_SPEC = "ij,jk->ik"
EXAMPLE_X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
EXAMPLE_Y = [[1.0, 0.0, 2.0], [0.5, 1.0, 0.0]]
EXAMPLE_OUT = [[2.0, 2.0, 2.0], [5.0, 4.0, 6.0], [8.0, 6.0, 10.0]]


def check_example(solve):
    x = torch.tensor(EXAMPLE_X)
    y = torch.tensor(EXAMPLE_Y)
    out = solve(EXAMPLE_SPEC, x, y)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 3)
    assert torch.allclose(out, torch.tensor(EXAMPLE_OUT), atol=1e-6)


check_example.example = True


def check_matches_torch_family(solve):
    torch.manual_seed(101)
    cases = [
        ("ij,jk->ik", [(3, 4), (4, 5)]),
        ("bij,bjk->bik", [(2, 3, 4), (2, 4, 5)]),
        ("bhqd,bhkd->bhqk", [(2, 3, 4, 6), (2, 3, 5, 6)]),
        ("abc,cde->abde", [(2, 3, 4), (4, 5, 6)]),
        ("i,j->ij", [(3,), (4,)]),
        ("i,i->", [(5,), (5,)]),
    ]
    for spec, shapes in cases:
        ops = [torch.randn(*s) for s in shapes]
        out = solve(spec, *ops)
        ref = torch.einsum(spec, *ops)
        assert out.shape == ref.shape, (spec, out.shape, ref.shape)
        assert out.dtype == ref.dtype
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), spec


def check_reduction_and_permutation_only(solve):
    torch.manual_seed(102)
    x = torch.randn(3, 4)
    # 纯置换：输出轴顺序变了，没有归约
    assert torch.allclose(solve("ij->ji", x), x.t(), atol=1e-6)
    # 纯归约：沿 j 求和
    assert torch.allclose(solve("ij->i", x), x.sum(dim=1), atol=1e-6)
    assert torch.allclose(solve("ij->j", x), x.sum(dim=0), atol=1e-6)
    # 全归约到标量：0 维张量
    out = solve("ij->", x)
    assert out.dim() == 0
    assert torch.allclose(out, x.sum(), atol=1e-6)
    # 单轴向量点积
    v = torch.randn(7)
    assert torch.allclose(solve("i,i->", v, v), v @ v, atol=1e-5)


def check_attention_shapes(solve):
    torch.manual_seed(103)
    b, h, q, k, d = 2, 3, 4, 5, 6
    qh = torch.randn(b, h, q, d)
    kh = torch.randn(b, h, k, d)
    vh = torch.randn(b, h, k, d)
    scores = solve("bhqd,bhkd->bhqk", qh, kh)
    assert scores.shape == (b, h, q, k)
    assert torch.allclose(scores, torch.einsum("bhqd,bhkd->bhqk", qh, kh),
                          atol=1e-5, rtol=1e-5)
    probs = torch.softmax(scores, dim=-1)
    ctx = solve("bhqk,bhkd->bhqd", probs, vh)
    assert ctx.shape == (b, h, q, d)
    assert torch.allclose(ctx, torch.einsum("bhqk,bhkd->bhqd", probs, vh),
                          atol=1e-5, rtol=1e-5)


def check_invalid_specs(solve):
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    bad = [
        ("ij,jk", (x, y)),                       # 缺 "->"
        ("...ij,...jk->...k", (x, y)),           # 不支持省略号
        ("ii->i", (torch.randn(3, 3),)),         # 同操作数内标注重复
        ("ij,jk->ix", (x, y)),                   # 输出字母没出现在输入里
        ("ij->ijk", (x,)),                       # 输出字母没出现在输入里
        ("ijk->ij", (x,)),                       # 标注字母数与维数不符
        ("ij", (x,)),                            # 缺 "->"
    ]
    for spec, ops in bad:
        try:
            solve(spec, *ops)
        except ValueError:
            continue
        raise AssertionError(f"非法 spec 应抛 ValueError: {spec}")
