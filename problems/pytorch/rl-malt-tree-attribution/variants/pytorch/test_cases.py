"""MALT -- search-tree credit attribution tests."""

"""
The ledger is the paper formula itself (arXiv:2412.01928, COLM 2025 v3,
sec 4.2 / A.6.1): no official code exists, so the oracle is an
independent NumPy translation of the value-iteration formulas plus
hand-computed anchors (see notes.md). Fixture A is a 3x3x3 tree,
fixture B is a 2x2x2 tree pinning the strict >0.5 threshold.
"""

# ---------- shared fixtures ----------

# j=0 强（7/9 成功）、j=1 弱（2/9）、j=2 全败
TREE_A = [
    [[1, 1, 1], [1, 0, 1], [0, 1, 1]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
]

# n=2: v00=v01=g0 恰 0.5（严格 > 阈值判 ×）、v10=1.0、v11=0.5、g1=0.75
TREE_B = [
    [[1, 0], [1, 0]],
    [[1, 1], [0, 1]],
]


def _close(a, b, tol=1e-12):
    return abs(a - b) <= tol


# ---------- mode="propagate" ----------


def check_propagate(solve):
    verifier_values, generator_values = solve(
        mode="propagate", rewards=TREE_A, n=3)
    # 手工锚点: v00=1.0, v01=v02=2/3, g0=7/9, g1=2/9, g2=0
    assert _close(verifier_values[0][0], 1.0)
    assert _close(verifier_values[0][1], 2 / 3)
    assert _close(verifier_values[0][2], 2 / 3)
    assert _close(generator_values[0], 7 / 9)
    assert _close(generator_values[1], 2 / 9)
    assert generator_values[2] == 0.0

    # n=2 边界: v00=v01=g0 恰 0.5、v10=1.0、v11=0.5、g1=0.75
    verifier_values, generator_values = solve(
        mode="propagate", rewards=TREE_B, n=2)
    assert verifier_values == [[0.5, 0.5], [1.0, 0.5]]
    assert _close(generator_values[0], 0.5)
    assert _close(generator_values[1], 0.75)

    # 树形校验: 不是 n x n x n 抛 ValueError
    try:
        solve(mode="propagate", rewards=[[[1, 1], [1, 1]], [[1]]], n=2)
        assert False, "should raise"
    except ValueError:
        pass
    # 叶层也要校验: branch 数与每 branch len 都 ==n 但某 leaf len!=n 同样抛
    try:
        solve(mode="propagate",
              rewards=[[[1, 1], [1, 1, 1]], [[1, 1], [1, 1]]], n=2)
        assert False, "should raise"
    except ValueError:
        pass


# ---------- mode="binarize" ----------


def check_binarize(solve):
    # 严格 >: 恰等于 theta 判 x
    labels = solve(mode="binarize", values=[0.3, 0.6, 0.5], theta=0.5)
    assert labels == [False, True, False], labels
    # 默认 theta=0.5
    labels = solve(mode="binarize", values=[0.5, 0.51, 0.49])
    assert labels == [False, True, False], labels


# ---------- mode="pairs" ----------


def check_pairs(solve):
    # 同父配对: 父 0 内 1 正 1 负、父 1 内 1 正 1 负；正下标在前、父升序
    pairs = solve(mode="pairs", unit_values=[0.3, 0.6, 0.5, 0.9],
                  parent_ids=[0, 0, 1, 1], theta=0.5)
    # 父 0: 正 [1] 负 [0]；父 1: 正 [3] 负 [2]（0.5 判负）
    assert pairs == [(1, 0), (3, 2)], pairs

    # 同父内正负子集的全组合: 2 正 2 负 -> 4 对
    pairs = solve(mode="pairs", unit_values=[0.9, 0.8, 0.2, 0.1],
                  parent_ids=[7, 7, 7, 7])
    assert pairs == [(0, 2), (0, 3), (1, 2), (1, 3)], pairs

    # 全正 / 全负父 -> 0 对
    assert solve(mode="pairs", unit_values=[0.9, 0.8],
                 parent_ids=[0, 0]) == []
    assert solve(mode="pairs", unit_values=[0.1, 0.2],
                 parent_ids=[0, 0]) == []


# ---------- mode="sft" ----------


def check_sft(solve):
    out = solve(mode="sft", unit_values=[0.9, 0.2, 0.8, 0.5],
                parent_ids=[0, 0, 1, 1], theta=0.5)
    # 正样本保持输入顺序；0.5 判负
    assert out["positives"] == [0, 2], out
    # 逐父 (正数, 负数) 对账，父下标升序
    assert out["counts"] == [(0, 1, 1), (1, 1, 1)], out

    # 全负
    out = solve(mode="sft", unit_values=[0.1], parent_ids=[3])
    assert out == {"positives": [], "counts": [(3, 0, 1)]}, out


# ---------- mode="attribute" ----------


def check_attribute_tree_a(solve):
    attr = solve(mode="attribute", rewards=TREE_A, n=3)
    # G 层标签: g0=7/9 ✓, g1=2/9 ×, g2=0 ×
    assert attr["generator_labels"] == [True, False, False]
    # V 层标签: j0 全 ✓（1.0, 2/3, 2/3）、j1 全 ×、j2 全 ×
    assert attr["verifier_labels"] == [
        True, True, True, False, False, False,
        False, False, False], attr["verifier_labels"]
    # R 层标签与叶奖励一致（叶是 0/1，二值化恒等）
    leaves = [float(x) for branch in TREE_A for sub in branch
              for x in sub]
    assert attr["refiner_values"] == leaves
    assert attr["refiner_labels"] == [x == 1.0 for x in leaves]

    # R 层偏好对: 8 对（(0,1) 2 对、(0,2) 2 对、(1,0) 2 对、(1,2) 2 对）
    assert attr["refiner_pairs"] == [
        (3, 4), (5, 4), (7, 6), (8, 6),
        (9, 10), (9, 11), (17, 15), (17, 16)], attr["refiner_pairs"]
    # V 层偏好对: 三个 generator 父内同号 -> 0 对
    assert attr["verifier_pairs"] == []
    # G 层偏好对: 同题 1 正 2 负 -> 2 对
    assert attr["generator_pairs"] == [(0, 1), (0, 2)]

    # SFT 正样本: R 层 9 个、V 层 3 个（j0）、G 层 1 个
    assert len(attr["sft"]["refiner"]["positives"]) == 9
    assert attr["sft"]["verifier"]["positives"] == [0, 1, 2]
    assert attr["sft"]["generator"]["positives"] == [0]
    # G 层 sft 是单父（同题）: counts 为一条 (0, 1, 2)
    assert attr["sft"]["generator"]["counts"] == [(0, 1, 2)]
    # V 层 sft counts: 逐 generator 父（j0 全正、j1/j2 全负）
    assert attr["sft"]["verifier"]["counts"] == [
        (0, 3, 0), (1, 0, 3), (2, 0, 3)]


def check_attribute_tree_b(solve):
    attr = solve(mode="attribute", rewards=TREE_B, n=2)
    # 0.5 边界: v00=v01=v11 判 x、v10 判 ✓、g0 判 x、g1 判 ✓
    assert attr["verifier_labels"] == [False, False, True, False]
    assert attr["generator_labels"] == [False, True]
    # R 层偏好对: (0,0) 1 对、(0,1) 1 对、(1,0) 0 对、(1,1) 1 对 -> 3 对
    assert attr["refiner_pairs"] == [(0, 1), (2, 3), (7, 6)], \
        attr["refiner_pairs"]
    # V 层偏好对: g0 父全 × -> 0；g1 父 [1.0, 0.5] -> 1 对
    assert attr["verifier_pairs"] == [(2, 3)], attr["verifier_pairs"]
    # G 层: 1 正 1 负 -> 1 对（正下标在前）
    assert attr["generator_pairs"] == [(1, 0)]
    # V 层 SFT: 只有线性 idx 2（v10）
    assert attr["sft"]["verifier"]["positives"] == [2]


TESTS = [
    check_propagate,
    check_binarize,
    check_pairs,
    check_sft,
    check_attribute_tree_a,
    check_attribute_tree_b,
]
