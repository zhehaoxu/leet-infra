import torch

KEYS = {"delta_weight", "effective_weight", "hira_result", "merged_weight"}


def _verify(solve, weight, A, B, x, fan_in_fan_out=False):
    w_snap = weight.clone()
    got = solve(weight, A, B, x, fan_in_fan_out=fan_in_fan_out)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    for k, v in got.items():
        assert torch.isfinite(v).all(), f"{k} 含非有限值"
    if fan_in_fan_out:
        out_f, in_f = B.shape[0], A.shape[1]
        assert got["delta_weight"].shape == (in_f, out_f)
        assert got["effective_weight"].shape == (in_f, out_f)
        assert got["merged_weight"].shape == (in_f, out_f)
    else:
        out_f = B.shape[0]
        assert got["delta_weight"].shape == (out_f, A.shape[1])
    assert got["hira_result"].shape == (*x.shape[:-1], B.shape[0])
    assert torch.equal(weight, w_snap), "不应修改输入 weight"
    return got


def check_example(solve):
    """手算 2x3、r=2 的乘性语义（vs LoRA 的加性）。"""
    W = torch.tensor([[1.0, 2.0, 0.5],
                      [0.0, -1.0, 2.0]])
    A = torch.tensor([[1.0, 0.0, -1.0],
                      [0.0, 1.0, 1.0]])          # [r=2, in=3]
    B = torch.tensor([[1.0, 2.0],
                      [0.0, -1.0]])              # [out=2, r=2]
    x = torch.tensor([[1.0, 1.0, 1.0]])          # [1, in=3]
    got = _verify(solve, W, A, B, x)
    # delta = B@A = [[1,2,-? ...]]：BA[0,:] = [1*1+2*0, 1*0+2*1, 1*(-1)+2*1] = [1, 2, 1]
    #                        BA[1,:] = [0*1-1*0, 0*0-1*1, 0*(-1)-1*1] = [0, -1, -1]
    assert torch.equal(got["delta_weight"], torch.tensor([[1.0, 2.0, 1.0],
                                                          [0.0, -1.0, -1.0]]))
    # eff = W ⊙ BA = [[1, 4, 0.5], [0, 1, -2]]
    assert torch.equal(got["effective_weight"], torch.tensor([[1.0, 4.0, 0.5],
                                                              [0.0, 1.0, -2.0]]))
    # hira_result = x @ eff.T = [[1+4+0.5, 0+1-2]] = [[5.5, -1.0]]
    assert torch.equal(got["hira_result"], torch.tensor([[5.5, -1.0]]))
    # merged = W + W⊙BA = [[2, 6, 1.0], [0, 0, 0.0]]
    assert torch.equal(got["merged_weight"], torch.tensor([[2.0, 6.0, 1.0],
                                                           [0.0, 0.0, 0.0]]))


check_example.example = True


def check_matches_oracle(solve):
    """随机固定种子对拍 + oracle 独立重算（fio False/True、批量 x）。"""
    g = torch.Generator().manual_seed(42)

    def oracle(W, A, B, x):
        delta = B @ A
        eff = W * delta
        return x @ eff.T

    for _ in range(4):
        out_f = int(torch.randint(2, 9, (1,), generator=g))
        in_f = int(torch.randint(2, 9, (1,), generator=g))
        r = int(torch.randint(1, 6, (1,), generator=g))
        W = torch.randn(out_f, in_f, generator=g)
        A = torch.randn(r, in_f, generator=g)
        B = torch.randn(out_f, r, generator=g)
        x = torch.randn(3, in_f, generator=g)
        got = _verify(solve, W, A, B, x)
        assert torch.allclose(got["hira_result"], oracle(W, A, B, x), atol=1e-6)
        assert torch.allclose(got["delta_weight"], B @ A, atol=1e-6)
        assert torch.allclose(got["effective_weight"], W * (B @ A), atol=1e-6)
        assert torch.allclose(got["merged_weight"], W + W * (B @ A), atol=1e-6)

        # fio=True：weight 以 [in, out] 存储，数学结果一致、输出布局转置
        got = _verify(solve, W.T.contiguous(), A, B, x, fan_in_fan_out=True)
        assert torch.allclose(got["hira_result"], oracle(W, A, B, x), atol=1e-6)
        assert got["delta_weight"].shape == (in_f, out_f)
        assert torch.allclose(got["delta_weight"].T, B @ A, atol=1e-6)
        assert torch.allclose(got["merged_weight"].T, W + W * (B @ A), atol=1e-6)

    # 一维 x（单个 token）
    got = _verify(solve, torch.randn(4, 5, generator=g), torch.randn(2, 5, generator=g),
                  torch.randn(4, 2, generator=g), torch.randn(5, generator=g))
    assert got["hira_result"].shape == (4,)


def check_multiplicative_vs_additive(solve):
    """区分 ⊙ 与 +：给一个 W+BA 与 W⊙BA 明显不同的用例。"""
    W = torch.zeros(3, 3)
    A = torch.eye(3)
    B = torch.eye(3)
    x = torch.eye(3)
    got = _verify(solve, W, A, B, x)
    # W=0：乘性语义下 eff 与 merged 全零（LoRA 加性下 BA=I 不为零）
    assert torch.equal(got["delta_weight"], torch.eye(3))
    assert torch.equal(got["effective_weight"], torch.zeros(3, 3))
    assert torch.equal(got["hira_result"], torch.zeros(3, 3))
    assert torch.equal(got["merged_weight"], torch.zeros(3, 3))


def check_merge_semantics(solve):
    """merged = W(1+Δ)，与「W + Δ」区分。"""
    W = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    A = torch.tensor([[1.0, 1.0]])
    B = torch.tensor([[1.0], [1.0]])             # Δ = [[1,1],[1,1]]
    x = torch.tensor([[1.0, 0.0]])
    got = _verify(solve, W, A, B, x)
    assert torch.equal(got["merged_weight"], torch.tensor([[4.0, 0.0], [0.0, 4.0]]))
    # 加性会得到 [[3,1],[1,3]]，乘性下非对角恒为 0


def check_fan_in_fan_out(solve):
    """fio=True 的存储布局：Conv1D 风格 [in, out] 入、[in, out] 出。"""
    W_math = torch.tensor([[1.0, 2.0], [3.0, 4.0]])       # [out=2, in=2]
    A = torch.tensor([[1.0, 0.0]])
    B = torch.tensor([[0.0], [1.0]])                       # Δ = [[0,0],[1,0]]
    x = torch.tensor([[1.0, 1.0]])
    got_f = _verify(solve, W_math, A, B, x, fan_in_fan_out=False)
    got_t = _verify(solve, W_math.T.contiguous(), A, B, x, fan_in_fan_out=True)
    assert torch.equal(got_f["hira_result"], got_t["hira_result"])
    assert torch.equal(got_f["delta_weight"], got_t["delta_weight"].T)
    assert torch.equal(got_f["effective_weight"], got_t["effective_weight"].T)
    assert torch.equal(got_f["merged_weight"], got_t["merged_weight"].T)


def check_validation(solve):
    """形状、类型、非有限值等非法输入抛 ValueError。"""
    W = torch.randn(4, 5)
    A = torch.randn(2, 5)
    B = torch.randn(4, 2)
    x = torch.randn(3, 5)

    def bad(fn, tag=""):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("应抛 ValueError: " + tag)

    bad(lambda: solve(torch.randn(4, 6), A, B, x), "weight 形状不匹配")
    bad(lambda: solve(W, torch.randn(2, 4), B, x), "A 形状不匹配")
    bad(lambda: solve(W, A, torch.randn(3, 2), x), "B 形状不匹配")
    bad(lambda: solve(W, A, B, torch.randn(3, 6)), "x 末维不匹配")
    bad(lambda: solve(W, A, B, x, fan_in_fan_out=1), "fio 非 bool")
    bad(lambda: solve(W, A, B, x, fan_in_fan_out=True), "fio=True 但 weight 是 [out,in]")
    bad(lambda: solve(torch.randn(6, 4), A, B, x, fan_in_fan_out=True), "fio=True 形状反例")
    bad(lambda: solve(W, A, B, torch.tensor([float("nan"), 0, 0, 0, 0])), "x 含 NaN")
    bad(lambda: solve(W, A, B, torch.tensor(1.0)), "x 零维")
    bad(lambda: solve(1.0, A, B, x), "weight 非 tensor")
