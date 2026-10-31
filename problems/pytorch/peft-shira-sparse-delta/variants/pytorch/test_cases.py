import torch

KEYS = {"delta_weight", "merged_weight", "y"}


def _verify(solve, weight, indices, values, scaling, x):
    w_snap = weight.clone()
    got = solve(weight, indices, values, scaling, x)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    for k, v in got.items():
        assert torch.isfinite(v).all(), f"{k} 含非有限值"
    out_f, in_f = weight.shape
    assert got["delta_weight"].shape == (out_f, in_f)
    assert got["merged_weight"].shape == (out_f, in_f)
    assert got["y"].shape == (*x.shape[:-1], out_f)
    assert torch.equal(weight, w_snap), "不应修改输入 weight"
    # mask 一致性：delta 非零处必然是某个索引（反之未必——值可能恰好为 0）
    return got


def check_example(solve):
    """手算：重复坐标累加、scaling 相乘、加性合并。"""
    W = torch.tensor([[1.0, 2.0, 0.0],
                      [0.0, 0.0, 0.0]])            # out=2, in=3
    indices = [[0, 0], [0, 0], [1, 2]]
    values = [1.0, 2.0, 3.0]
    got = _verify(solve, W, indices, values, 2.0,
                  torch.tensor([[1.0, 1.0, 1.0]]))
    # (0,0) 出现两次：1*2 + 2*2 = 6（累加，不是覆盖成 4）
    # (1,2) 一次：3*2 = 6
    assert torch.equal(got["delta_weight"], torch.tensor([[6.0, 0.0, 0.0],
                                                          [0.0, 0.0, 6.0]]))
    assert torch.equal(got["merged_weight"], torch.tensor([[7.0, 2.0, 0.0],
                                                           [0.0, 0.0, 6.0]]))
    assert torch.equal(got["y"], torch.tensor([[9.0, 6.0]]))


check_example.example = True


def check_matches_oracle(solve):
    """torch.sparse_coo_tensor 对拍 + 随机固定种子。"""
    g = torch.Generator().manual_seed(7)
    out_f, in_f = 5, 7
    K = 20
    W = torch.randn(out_f, in_f, generator=g)
    rows = torch.randint(0, out_f, (K,), generator=g).tolist()
    cols = torch.randint(0, in_f, (K,), generator=g).tolist()
    vals = torch.randn(K, generator=g).tolist()
    x = torch.randn(3, in_f, generator=g)

    got = _verify(solve, W, list(zip(rows, cols)), vals, 1.0, x)

    # 上游同款构造：sparse_coo_tensor 未 coalesce，to_dense 对重复坐标求和
    idx = torch.tensor([rows, cols], dtype=torch.int64)
    ref_delta = torch.sparse_coo_tensor(
        idx, torch.tensor(vals, dtype=torch.float64), (out_f, in_f)
    ).to_dense()
    assert torch.allclose(got["delta_weight"], ref_delta.to(W.dtype), atol=1e-6), \
        "与 torch.sparse_coo_tensor 的累加语义不一致"
    assert torch.allclose(got["merged_weight"], W + ref_delta.to(W.dtype), atol=1e-6)
    assert torch.allclose(got["y"], x @ (W + ref_delta.to(W.dtype)).T, atol=1e-6)

    # scaling 语义：scaling=0.5 等价 values 减半
    got2 = _verify(solve, W, list(zip(rows, cols)), vals, 0.5, x)
    assert torch.allclose(got2["delta_weight"], ref_delta.to(W.dtype) * 0.5, atol=1e-6)

    # values 用一维 tensor 传入
    got3 = _verify(solve, W, list(zip(rows, cols)), torch.tensor(vals), 1.0, x)
    assert torch.allclose(got3["delta_weight"], ref_delta.to(W.dtype), atol=1e-6)

    # 一维 x
    got4 = _verify(solve, W, [[0, 0]], [2.0], 1.0, torch.randn(in_f, generator=g))
    assert got4["y"].shape == (out_f,)


def check_budget_and_additive(solve):
    """参数预算上限 + 与 HiRA 的加性对照（W=0 处 delta 依然生效）。"""
    W = torch.zeros(2, 3)
    got = _verify(solve, W, [[0, 1]], [4.0], 1.0, torch.tensor([[1.0, 1.0, 1.0]]))
    # SHiRA 是加性 delta：W=0 处照样写入（对照 HiRA 的乘性冻结）
    assert torch.equal(got["merged_weight"], torch.tensor([[0.0, 4.0, 0.0],
                                                           [0.0, 0.0, 0.0]]))
    assert torch.equal(got["y"], torch.tensor([[4.0, 0.0]]))

    # 预算：K = in*out 恰好合法（每个元素一个参数）
    indices = [[r, c] for r in range(2) for c in range(3)]
    got = _verify(solve, W, indices, [1.0] * 6, 1.0, torch.zeros(1, 3))
    assert torch.equal(got["delta_weight"], torch.ones(2, 3))


def check_validation(solve):
    """形状、坐标越界、预算、scaling 等非法输入抛 ValueError。"""
    W = torch.randn(2, 3)
    x = torch.randn(1, 3)

    def bad(fn, tag=""):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("应抛 ValueError: " + tag)

    bad(lambda: solve(W, [], [], 1.0, x), "indices 空")
    bad(lambda: solve(W, [[0, 0]], [1.0, 2.0], 1.0, x), "values 数量不匹配")
    bad(lambda: solve(W, [[0]], [1.0], 1.0, x), "坐标非二元组")
    bad(lambda: solve(W, [[0, 3]], [1.0], 1.0, x), "列越界")
    bad(lambda: solve(W, [[2, 0]], [1.0], 1.0, x), "行越界")
    bad(lambda: solve(W, [[0, -1]], [1.0], 1.0, x), "负坐标")
    bad(lambda: solve(W, [[0, 0]], ["a"], 1.0, x), "values 非数")
    bad(lambda: solve(W, [[0, 0]], [float("nan")], 1.0, x), "values NaN")
    bad(lambda: solve(W, [[0, 0]], [1.0], -1.0, x), "scaling 为负")
    bad(lambda: solve(W, [[0, 0]], [1.0], float("inf"), x), "scaling 无穷")
    bad(lambda: solve(W, [[0, 0]], [1.0], True, x), "scaling bool")
    bad(lambda: solve(W, [[0, 0]], [1.0], 1.0, torch.randn(1, 4)), "x 末维不匹配")
    bad(lambda: solve(W, [[0, 0]], [1.0], 1.0, torch.tensor(1.0)), "x 零维")
    # 预算：K > in*out
    big = [[0, 0]] * 7
    bad(lambda: solve(W, big, [1.0] * 7, 1.0, x), "参数量超预算")
