import math

SCORES = [0.10, 0.85, 0.40, 0.85, 0.25, 0.60]
RELS = [0, 3, 2, 1, 2, 3]
K = 4
EXAMPLE_NDCG = 0.8743305942        # 参考解跑出的 nDCG@4（指数增益）
EXAMPLE_DCG = 12.4229594278        # 参考解跑出的 DCG@4
EXAMPLE_IDCG = 14.2085379492       # 参考解跑出的 IDCG@4
EXAMPLE_NDCG_LIN = 0.8872011479    # 同一份数据换线性增益
EXAMPLE_DCG_LIN = 5.9922828697
EXAMPLE_IDCG_LIN = 6.7541423769


def _oracle(scores, rels, k, exp_gain=True):
    """独立重算：手写选择排序（严格大于才交换，天然实现"同分取下标小的"），
    增益用循环倍增而不是 2**rel，折损用 math.log(x, 2)，全程纯 Python 标量累加。"""

    def gain(r):
        g = 1
        for _ in range(int(r)):
            g *= 2
        return (g - 1) if exp_gain else int(r)

    def dcg_of(label_seq):
        total = 0.0
        for i, r in enumerate(label_seq):
            total += gain(r) / math.log(i + 2, 2)   # 名次 i 从 0 起，折底 log2(i+2)
        return total

    n = len(scores)
    cut = min(int(k), n)
    if n == 0 or cut <= 0:
        return 0.0, 0.0, 0.0

    # 选择排序：每轮在未排好的部分里找分数最大的，只有"严格更大"才替换 => 平局留下标小的
    remaining = list(range(n))
    ranked = []
    while len(ranked) < cut and remaining:
        best = remaining[0]
        for j in remaining[1:]:
            if scores[j] > scores[best]:
                best = j
        ranked.append(rels[best])
        remaining.remove(best)

    # 理想排序：把 rels 插入排序成降序（平局无所谓，增益相同）
    sorted_rels = []
    for r in rels:
        pos = len(sorted_rels)
        for idx in range(len(sorted_rels)):
            if sorted_rels[idx] < r:
                pos = idx
                break
        sorted_rels.insert(pos, r)

    dcg = dcg_of(ranked)
    idcg = dcg_of(sorted_rels[:cut])
    ndcg = dcg / idcg if idcg > 0.0 else 0.0
    return ndcg, dcg, idcg


def _close(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-5, rel_tol=1e-5)


def _all_close(got, exp):
    return all(_close(a, b) for a, b in zip(got, exp))


def check_example(solve):
    got = solve(SCORES, RELS, K, True)
    assert _close(got[0], EXAMPLE_NDCG), f"ndcg got {got[0]}"
    assert _close(got[1], EXAMPLE_DCG), f"dcg got {got[1]}"
    assert _close(got[2], EXAMPLE_IDCG), f"idcg got {got[2]}"


check_example.example = True


def check_example_linear_gain(solve):
    got = solve(SCORES, RELS, K, False)
    assert _close(got[0], EXAMPLE_NDCG_LIN), f"ndcg-lin got {got[0]}"
    assert _close(got[1], EXAMPLE_DCG_LIN), f"dcg-lin got {got[1]}"
    assert _close(got[2], EXAMPLE_IDCG_LIN), f"idcg-lin got {got[2]}"


def check_matches_oracle(solve):
    cases = [
        (SCORES, RELS, 4, True),
        (SCORES, RELS, 4, False),
        (SCORES, RELS, 6, True),
        (SCORES, RELS, 1, True),
        ([0.5, 0.5, 0.5], [1, 2, 3], 3, True),
        ([3.0, -1.0, 0.0, 2.0], [0, 0, 0, 5], 3, True),
        ([1.0], [4], 1, False),
        ([0.9, 0.1, 0.5], [0, 0, 0], 2, True),
    ]
    for scores, rels, k, eg in cases:
        got = solve(scores, rels, k, eg)
        exp = _oracle(scores, rels, k, eg)
        assert _all_close(got, exp), (scores, rels, k, eg, got, exp)


def check_random_against_oracle(solve):
    import torch

    g = torch.Generator().manual_seed(299)
    for _ in range(8):
        n = int(torch.randint(1, 12, (1,), generator=g))
        scores = [round(float(x), 2) for x in torch.randn(n, generator=g)]
        rels = [int(x) for x in torch.randint(0, 4, (n,), generator=g)]
        k = int(torch.randint(1, n + 3, (1,), generator=g))
        for eg in (True, False):
            got = solve(scores, rels, k, eg)
            exp = _oracle(scores, rels, k, eg)
            assert _all_close(got, exp), (scores, rels, k, eg, got, exp)


def check_perfect_ranking_is_one(solve):
    # 排序就是理想排序 => nDCG = 1，且 DCG == IDCG
    scores = [0.9, 0.7, 0.5, 0.3, 0.1]
    rels = [4, 3, 2, 1, 0]
    for eg in (True, False):
        ndcg, dcg, idcg = solve(scores, rels, 5, eg)
        assert _close(ndcg, 1.0), (eg, ndcg)
        assert _close(dcg, idcg), (eg, dcg, idcg)
    # 只截前 3 个位置，前 3 个仍是最相关的 3 篇 => 依旧 1
    assert _close(solve(scores, rels, 3, True)[0], 1.0)


def check_k_beyond_n_is_full_list(solve):
    base = solve(SCORES, RELS, 6, True)
    for k in (7, 100):
        got = solve(SCORES, RELS, k, True)
        assert _all_close(got, base), (k, got, base)
    # DCG / IDCG 是逐项累加的非负和 => 随 k 单调不减；nDCG 本身不一定单调
    prev_d = prev_i = -1.0
    for k in range(1, 9):
        _, d, i = solve(SCORES, RELS, k, True)
        assert d >= prev_d - 1e-12 and i >= prev_i - 1e-12, (k, d, i)
        prev_d, prev_i = d, i
    assert _close(solve(SCORES, RELS, 1, True)[0], 1.0)   # top1 恰好是最相关文档之一


def check_all_zero_relevance_sentinel(solve):
    # 全 0 标注 => IDCG = 0 => nDCG 走哨兵 0.0，DCG 也是 0
    got = solve([0.9, 0.1, 0.5], [0, 0, 0], 2, True)
    assert got == (0.0, 0.0, 0.0), got
    got_lin = solve([0.9, 0.1, 0.5], [0, 0, 0], 2, False)
    assert got_lin == (0.0, 0.0, 0.0), got_lin
    assert _all_close(got_lin, _oracle([0.9, 0.1, 0.5], [0, 0, 0], 2, False))


def check_empty_input_sentinel(solve):
    for k in (0, 1, 5):
        assert solve([], [], k, True) == (0.0, 0.0, 0.0)
        assert solve([], [], k, False) == (0.0, 0.0, 0.0)
    # k = 0 但有候选：一个位置都不取
    assert solve(SCORES, RELS, 0, True) == (0.0, 0.0, 0.0)


def check_tie_break_uses_smaller_index(solve):
    # 三个文档打分完全相同 => 名次就是下标 0,1,2
    # rels [1,2,3] 按 0,1,2 顺序进 DCG：1/1 + 3/log2(3) + 7/log2(4)
    got = solve([0.5, 0.5, 0.5], [1, 2, 3], 3, True)
    dcg = 1 / math.log(2, 2) + 3 / math.log(3, 2) + 7 / math.log(4, 2)
    assert _close(got[1], dcg), got
    # 把 rels 换成递减的，同分平局就能拿到 1.0
    ndcg, _, _ = solve([0.5, 0.5, 0.5], [3, 2, 1], 3, True)
    assert _close(ndcg, 1.0), ndcg


def check_single_document(solve):
    ndcg, dcg, idcg = solve([0.42], [3], 1, True)
    assert _close(ndcg, 1.0), ndcg
    assert _close(dcg, 7.0) and _close(idcg, 7.0), (dcg, idcg)
    ndcg0, dcg0, idcg0 = solve([0.42], [0], 1, True)
    assert (ndcg0, dcg0, idcg0) == (0.0, 0.0, 0.0), (ndcg0, dcg0, idcg0)


def check_permutation_invariance_with_distinct_scores(solve):
    # 打分互不相同时，文档重排不改变 nDCG / DCG / IDCG
    scores = [0.10, 0.84, 0.40, 0.86, 0.25, 0.60]
    rels = [0, 3, 2, 1, 2, 3]
    a = solve(scores, rels, 4, True)
    perm = [5, 4, 3, 2, 1, 0]
    b = solve([scores[i] for i in perm], [rels[i] for i in perm], 4, True)
    assert _all_close(a, b), (a, b)


def check_exp_gain_penalises_top_heavily(solve):
    # 多个非零标注时，指数增益把高相关文档"放大"，把它排到后面扣分更狠
    scores = [0.9, 0.8, 0.7, 0.1]
    rels = [1, 0, 0, 3]
    exp_ndcg = solve(scores, rels, 4, True)[0]
    lin_ndcg = solve(scores, rels, 4, False)[0]
    assert exp_ndcg < lin_ndcg, (exp_ndcg, lin_ndcg)
    assert _close(exp_ndcg, 0.5261135977), exp_ndcg
    assert _close(lin_ndcg, 0.6312514507), lin_ndcg
    for eg, val in ((True, exp_ndcg), (False, lin_ndcg)):
        assert _all_close(solve(scores, rels, 4, eg), _oracle(scores, rels, 4, eg))
