import numpy as np
import torch

KEYS = {"scores", "valid", "keep_count", "threshold", "threshold_index", "keep_ratio"}
TOL = {torch.float32: 1e-6, torch.float64: 1e-12}


def _oracle(attn, ratio, level):
    """numpy 独立重算：跨 (G, Q) 取 max → flatten → 下标 n 取阈值 / 每头 topk。"""
    a = attn.detach().numpy().astype(np.float64)
    num_layers, num_heads, _, _, num_keys = a.shape
    scores = a.max(axis=(2, 3))
    index, thres = 0, 0.0
    if level == "pair-uniform":
        keep = int(num_keys * ratio)
        valid = np.zeros(scores.shape, dtype=bool)
        for l in range(num_layers):
            for h in range(num_heads):
                row = sorted(((scores[l, h, j], j) for j in range(num_keys)),
                             key=lambda t: (-t[0], t[1]))
                valid[l, h, [j for _, j in row[:keep]]] = True
    elif ratio < 1:
        index = max(int(scores.size * ratio) - 1, 0)
        thres = float(np.sort(scores.reshape(-1))[::-1][index])
        valid = scores > thres
    else:
        valid = np.ones(scores.shape, dtype=bool)
    return scores, valid, thres, index, float(valid.mean())


def _verify(solve, attn, ratio=0.3, level="pair"):
    before = attn.clone()
    got = solve(attn, ratio, level)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    assert torch.equal(attn, before), "输入被就地修改了"
    scores, valid, thres, index, keep_ratio = _oracle(attn, ratio, level)
    want = torch.tensor(valid)
    assert got["scores"].dtype == attn.dtype and got["scores"].shape == scores.shape
    torch.testing.assert_close(got["scores"], torch.tensor(scores, dtype=attn.dtype),
                               rtol=TOL[attn.dtype], atol=TOL[attn.dtype])
    assert got["valid"].dtype == torch.bool, "valid 必须是 bool"
    assert torch.equal(got["valid"], want), "valid 应为 scores > threshold（严格大于）"
    assert got["keep_count"].dtype == torch.int64, "keep_count 必须是 int64"
    assert torch.equal(got["keep_count"], want.sum(-1)), "keep_count 应为每头保留条数"
    assert abs(got["threshold"] - thres) <= TOL[attn.dtype], f"threshold 应是降序第 {index} 个分数"
    assert int(got["threshold_index"]) == index, f"threshold_index 应等于下标 {index}"
    assert abs(got["keep_ratio"] - keep_ratio) <= 1e-9, "keep_ratio 应为 valid 的真实占比"
    return got


def check_example(solve):
    """题面示例：(L,H_kv,G,Q,N)=(1,2,1,2,2)、ratio=0.5 的 pair 与 pair-uniform 两支。"""
    w = torch.tensor([[[[[0.7, 0.3], [0.4, 0.6]]], [[[0.2, 0.8], [0.5, 0.5]]]]])
    pair = _verify(solve, w, 0.5, "pair")
    assert torch.allclose(pair["scores"], torch.tensor([[[0.7, 0.6], [0.5, 0.8]]]), atol=1e-6)
    assert abs(pair["threshold"] - 0.7) < 1e-6 and int(pair["threshold_index"]) == 1
    assert pair["valid"].tolist() == [[[False, False], [False, True]]], "0.7 本身必须被丢掉"
    assert pair["keep_count"].tolist() == [[0, 1]] and pair["keep_ratio"] == 0.25
    uni = _verify(solve, w, 0.5, "pair-uniform")
    assert uni["valid"].tolist() == [[[True, False], [False, True]]]
    assert uni["keep_count"].tolist() == [[1, 1]] and uni["keep_ratio"] == 0.5
    assert uni["threshold"] == 0.0 and int(uni["threshold_index"]) == 0, "uniform 不排序"


def check_max_over_group_and_queries(solve):
    """聚合是跨 (group, query) 的 amax：换成 mean 分数立刻不同。"""
    torch.manual_seed(3)
    w = torch.rand(2, 3, 4, 5, 6, dtype=torch.float64) * 0.1
    w[0, 0, 2, 3, 4] = 0.9                                  # 单点尖峰，必须被 max 抓到
    got = _verify(solve, w, 0.5)
    assert got["scores"][0, 0, 4].item() == 0.9, "尖峰没被 group×query 的 max 抓到"
    assert not torch.allclose(got["scores"], w.mean(dim=(2, 3)), atol=1e-3), "scores 不是 mean"


def check_threshold_is_an_index_not_a_count(solve):
    """ratio=0.3、T=20：n=5 是下标，阈值本身不保留，实际留住 5 个而不是 6 个。"""
    torch.manual_seed(5)
    w = torch.rand(2, 2, 1, 5, 5, dtype=torch.float64)      # T = 2*2*5 = 20
    got = _verify(solve, w, 0.3)
    assert int(got["threshold_index"]) == int(20 * 0.3) - 1 == 5
    assert int(got["valid"].sum()) == 5 and got["keep_ratio"] == 0.25, "应留 5 个而不是 6 个"


def check_degenerate_index_clamps_to_zero(solve):
    """int(T*ratio)-1 < 0 时上游 max(...,0) 夹到 0，阈值=最大值，结果一条都不留。"""
    w = torch.tensor([[[[[0.6, 0.3]], [[0.2, 0.5]]]]], dtype=torch.float64)   # T = 4
    got = _verify(solve, w, 0.01)
    assert int(got["threshold_index"]) == 0, "负下标必须夹到 0"
    assert got["threshold"] == 0.6 and int(got["valid"].sum()) == 0, "阈值=最大值，一条不留"


def check_strict_greater_and_ties(solve):
    """分数全同 → 阈值等于它自己，严格大于导致全丢；同分 topk 按列号稳定。"""
    w = torch.full((1, 1, 2, 3, 4), 0.25, dtype=torch.float64)
    assert int(_verify(solve, w, 0.5)["valid"].sum()) == 0, "全同分时严格大于必须一条不留"
    uni = _verify(solve, w, 0.5, "pair-uniform")             # k = int(4*0.5) = 2
    assert uni["valid"].tolist() == [[[True, True, False, False]]]


def check_ratio_at_least_one_keeps_all(solve):
    """ratio >= 1 走全留分支：threshold 与 threshold_index 都归 0。"""
    torch.manual_seed(7)
    w = torch.rand(1, 2, 2, 2, 3)
    for ratio in (1.0, 1.5):
        got = _verify(solve, w, ratio)
        assert bool(got["valid"].all()) and got["keep_ratio"] == 1.0
        assert got["threshold"] == 0.0 and int(got["threshold_index"]) == 0, "全留分支阈值归 0"


def check_uniform_mode_budgets_each_head(solve):
    """pair-uniform 每头各留 int(N*ratio) 个：弱头不会被强头抢走名额。"""
    w = torch.zeros(1, 2, 1, 1, 8, dtype=torch.float64)
    w[0, 0, 0, 0] = torch.linspace(1.0, 0.1, 8)             # 头 0 分数高
    w[0, 1, 0, 0] = torch.linspace(0.09, 0.02, 8)           # 头 1 分数低但同序
    got = _verify(solve, w, 0.25, "pair-uniform")
    assert got["keep_count"].tolist() == [[2, 2]], got["keep_count"].tolist()
    assert got["valid"][0, 1].tolist() == [True, True] + [False] * 6
    glob = _verify(solve, w, 0.25)                          # 全局阈值：弱头一条不剩
    assert glob["keep_count"].tolist() == [[3, 0]], glob["keep_count"].tolist()


def check_dtype_and_non_contiguous(solve):
    """float32 输入 → float32 输出（对齐 float64 结果）；非连续视图必须与副本一致。"""
    torch.manual_seed(11)
    w = torch.rand(3, 4, 2, 3, 10, dtype=torch.float64)[:, :, :, ::2, ::2]
    a = _verify(solve, w, 0.4)                              # 步长切片，非连续
    assert a["valid"].tolist() == solve(w.contiguous(), 0.4, "pair")["valid"].tolist(), "非连续"
    lo = solve(w.float(), 0.4, "pair")
    assert lo["scores"].dtype == torch.float32, lo["scores"].dtype
    torch.testing.assert_close(lo["scores"], a["scores"].float(), rtol=1e-6, atol=1e-6)


def check_random_against_oracle(solve):
    """随机形状 × 多个 ratio × 两种 level，与 numpy oracle 全键对拍。"""
    for seed, shape in enumerate([(1, 1, 1, 1, 1), (1, 2, 3, 4, 5), (3, 4, 2, 7, 9), (2, 1, 5, 1, 12)]):
        torch.manual_seed(200 + seed)
        w = torch.rand(shape, dtype=torch.float64)
        for ratio, level in ((0.1, "pair"), (0.3, "pair"), (0.75, "pair"),
                             (0.5, "pair-uniform"), (0.99, "pair-uniform")):
            _verify(solve, w, ratio, level)


def check_value_errors(solve):
    w = torch.rand(1, 2, 1, 2, 4)
    bad = [
        (torch.rand(2, 1, 2, 4), 0.3, "pair"),        # 非五维
        (w.to(torch.int32), 0.3, "pair"),             # 非浮点
        (torch.rand(1, 2, 1, 2, 0), 0.3, "pair"),     # 空维
        (w, -0.1, "pair"), (w, "0.3", "pair"),        # ratio 为负 / 非实数
        (w, 0.3, "head"),                             # level 不认识
    ]
    for i, args in enumerate(bad):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 个非法输入应抛 ValueError")


check_example.example = True
