import torch

KEYS = {"merged_weight", "originals"}


def _verify(solve, weight, adapters):
    w_snap = weight.clone()
    got = solve(weight, adapters)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    vocab, dim = weight.shape
    assert got["merged_weight"].shape == (vocab, dim)
    assert torch.isfinite(got["merged_weight"]).all()
    assert set(got["originals"]) == {a["name"] for a in adapters}
    for name, rows in got["originals"].items():
        n = len(next(a for a in adapters if a["name"] == name)["token_indices"])
        assert rows.shape == (n, dim)
    assert torch.equal(weight, w_snap), "不应修改输入 weight"
    return got


def check_example(solve):
    """手算：替换语义、originals 从原始权重抓取、多 adapter 依序应用。"""
    W = torch.tensor([[1.0, 1.0],
                      [2.0, 2.0],
                      [3.0, 3.0],
                      [4.0, 4.0]])                 # vocab=4, dim=2
    got = _verify(solve, W, [
        {"name": "a", "token_indices": [1], "delta": [[9.0, 9.0]]},
        {"name": "b", "token_indices": [3, 0], "delta": [[8.0, 8.0], [7.0, 7.0]]},
    ])
    # 行 1 被 a 替换；行 3、0 被 b 替换；行 2 不动
    assert torch.equal(got["merged_weight"], torch.tensor([[7.0, 7.0],
                                                           [9.0, 9.0],
                                                           [3.0, 3.0],
                                                           [8.0, 8.0]]))
    # originals 是原始行（不是 delta，也不是 merged 中的行）
    assert torch.equal(got["originals"]["a"], torch.tensor([[2.0, 2.0]]))
    assert torch.equal(got["originals"]["b"], torch.tensor([[4.0, 4.0], [1.0, 1.0]]))


check_example.example = True


def check_replacement_not_additive(solve):
    """替换 vs 加法锚点：delta 完全决定新行，与原值无关。"""
    W = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
    got = _verify(solve, W, [{"name": "a", "token_indices": [0], "delta": [[1.0, 2.0]]}])
    # 加法会得 [[101, 202], [300, 400]]；替换直接换成 [1, 2]
    assert torch.equal(got["merged_weight"], torch.tensor([[1.0, 2.0], [300.0, 400.0]]))
    assert torch.equal(got["originals"]["a"], torch.tensor([[100.0, 200.0]]))


def check_matches_oracle(solve):
    """torch.index_copy 对拍 + 随机固定种子 + delta 用 tensor 传入。"""
    g = torch.Generator().manual_seed(11)
    vocab, dim = 8, 5
    W = torch.randn(vocab, dim, generator=g)

    idx_a = [5, 2, 7]
    idx_b = [0, 6]
    delta_a = torch.randn(3, dim, generator=g)
    delta_b = torch.randn(2, dim, generator=g)

    got = _verify(solve, W, [
        {"name": "a", "token_indices": idx_a, "delta": delta_a},
        {"name": "b", "token_indices": idx_b, "delta": delta_b},
    ])
    ref = W.clone().index_copy(0, torch.tensor(idx_a), delta_a) \
            .index_copy(0, torch.tensor(idx_b), delta_b)
    assert torch.allclose(got["merged_weight"], ref, atol=1e-6)
    assert torch.allclose(got["originals"]["a"], W[idx_a], atol=1e-6)
    assert torch.allclose(got["originals"]["b"], W[idx_b], atol=1e-6)

    # delta 用嵌套 list 传入与 tensor 等价
    got2 = _verify(solve, W, [
        {"name": "a", "token_indices": idx_a, "delta": delta_a.tolist()},
    ])
    assert torch.allclose(got2["merged_weight"], W.clone().index_copy(0, torch.tensor(idx_a), delta_a), atol=1e-6)

    # 单 adapter、dim=1 的窄表
    W1 = torch.randn(4, 1, generator=g)
    got3 = _verify(solve, W1, [{"name": "z", "token_indices": [2], "delta": [[0.5]]}])
    assert got3["merged_weight"][2, 0].item() == 0.5
    assert got3["merged_weight"][0, 0].item() == W1[0, 0].item()


def check_overlap_and_validation(solve):
    """重叠 adapter 与各类非法输入抛 ValueError。"""
    W = torch.randn(4, 2)

    def bad(fn, tag=""):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("应抛 ValueError: " + tag)

    ad = lambda name, idx, delta: {"name": name, "token_indices": idx, "delta": delta}

    # 重叠（上游明确不支持，无论 delta 是什么）
    bad(lambda: solve(W, [ad("a", [1], [[0.0, 0.0]]),
                          ad("b", [1], [[0.0, 0.0]])]), "完全重叠")
    bad(lambda: solve(W, [ad("a", [0, 1], [[0.0, 0.0], [0.0, 0.0]]),
                          ad("b", [1, 2], [[0.0, 0.0], [0.0, 0.0]])]), "部分重叠")

    bad(lambda: solve(W, []), "adapters 空")
    bad(lambda: solve(W, [ad("a", [], [[1.0, 1.0]])]), "indices 空")
    bad(lambda: solve(W, [ad("a", [4], [[0.0, 0.0]])]), "index 越界")
    bad(lambda: solve(W, [ad("a", [-1], [[0.0, 0.0]])]), "负 index")
    bad(lambda: solve(W, [ad("a", [1, 1], [[0.0, 0.0], [0.0, 0.0]])]), "adapter 内重复 index")
    bad(lambda: solve(W, [ad("a", [1], [[0.0, 0.0]]),
                          ad("a", [2], [[0.0, 0.0]])]), "adapter 名重复")
    bad(lambda: solve(W, [ad("a", [1], [[0.0]])]), "delta 行宽不符")
    bad(lambda: solve(W, [ad("a", [1, 2], [[0.0, 0.0]])]), "delta 行数不符")
    bad(lambda: solve(W, [ad("a", [1], [[0.0, float("nan")]])]), "delta 含 NaN")
    bad(lambda: solve(W, [ad("a", [True], [[0.0, 0.0]])]), "index 是 bool")
    bad(lambda: solve(W, ["a"]), "adapter 非 dict")
    bad(lambda: solve(W, [{"name": "a", "token_indices": [1]}]), "缺 delta 键")
