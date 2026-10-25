import torch

HYPS = [[1, 2, 4, 3], [1, 2, 5]]
REFS = [[[1, 2, 3, 4]], [[1, 2, 3, 4, 5]]]
EXAMPLE_R1 = 0.875               # 参考解跑出的 ROUGE-1 F1
EXAMPLE_R2 = 0.3333333333        # 参考解跑出的 ROUGE-2 F1
EXAMPLE_RL = 0.75                # 参考解跑出的 ROUGE-L F1


def _oracle(hypotheses, references):
    """独立重算：显式循环数 n-gram（不用 Counter），LCS 用记忆化递归（不用 DP 表）。"""

    def grams(seq, n):
        out = []
        for i in range(len(seq) - n + 1):
            g = tuple()
            for j in range(n):
                g = g + (seq[i + j],)
            out.append(g)
        return out

    def overlap(hyp, ref, n):
        hg = grams(hyp, n)
        rg = grams(ref, n)
        distinct = []
        for g in hg:
            if g not in distinct:
                distinct.append(g)
        hit = 0
        for g in distinct:
            ch = hg.count(g)
            cr = rg.count(g)
            hit += ch if ch < cr else cr
        return hit, len(hg), len(rg)

    def f1(hit, total_h, total_r):
        if total_h == 0 or total_r == 0:
            return 0.0
        recall = hit / total_r
        precision = hit / total_h
        if recall + precision == 0.0:
            return 0.0
        return 2.0 * recall * precision / (recall + precision)

    def lcs(a, b):
        memo = {}

        def go(i, j):
            if i == len(a) or j == len(b):
                return 0
            key = (i, j)
            if key in memo:
                return memo[key]
            if a[i] == b[j]:
                res = 1 + go(i + 1, j + 1)
            else:
                r1 = go(i + 1, j)
                r2 = go(i, j + 1)
                res = r1 if r1 > r2 else r2
            memo[key] = res
            return res

        return go(0, 0)

    r1s, r2s, rls = [], [], []
    for hyp, refs in zip(hypotheses, references):
        best1 = 0.0
        best2 = 0.0
        bestl = 0.0
        for ref in refs:
            hit1, th1, tr1 = overlap(hyp, ref, 1)
            best1 = max(best1, f1(hit1, th1, tr1))
            hit2, th2, tr2 = overlap(hyp, ref, 2)
            best2 = max(best2, f1(hit2, th2, tr2))
            l = lcs(hyp, ref)
            if len(hyp) > 0 and len(ref) > 0:
                bestl = max(bestl, f1(l, len(hyp), len(ref)))
        r1s.append(best1)
        r2s.append(best2)
        rls.append(bestl)
    n = len(hypotheses)
    if n == 0:
        return 0.0, 0.0, 0.0
    return sum(r1s) / n, sum(r2s) / n, sum(rls) / n


def _close(a, b):
    return torch.allclose(
        torch.tensor(float(a), dtype=torch.float64),
        torch.tensor(float(b), dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )


def check_example(solve):
    got = solve(HYPS, REFS)
    assert _close(got[0], EXAMPLE_R1), f"rouge1 got {got[0]}"
    assert _close(got[1], EXAMPLE_R2), f"rouge2 got {got[1]}"
    assert _close(got[2], EXAMPLE_RL), f"rougeL got {got[2]}"


check_example.example = True


def check_matches_oracle(solve):
    corpora = [
        (HYPS, REFS),
        ([[1, 2, 3]], [[[3, 2, 1]]]),
        ([[1, 1, 1]], [[[1, 1]]]),
        ([[5], [], [7, 8]], [[[5, 6]], [[1]], [[7, 8], [8, 7]]]),
        ([[1, 2, 3, 4, 5]], [[[1, 2, 3, 4, 5]], [[1, 3, 5]]]),
    ]
    for hyps, refs in corpora:
        got = solve(hyps, refs)
        exp = _oracle(hyps, refs)
        assert all(_close(a, b) for a, b in zip(got, exp)), (hyps, refs, got, exp)


def check_random_against_oracle(solve):
    g = torch.Generator().manual_seed(29)
    for _ in range(5):
        m = int(torch.randint(1, 4, (1,), generator=g))
        hyps = []
        refs = []
        for _ in range(m):
            lh = int(torch.randint(0, 7, (1,), generator=g))
            hyps.append([int(x) for x in torch.randint(1, 4, (lh,), generator=g)])
            nr = int(torch.randint(1, 3, (1,), generator=g))
            refs.append(
                [
                    [int(x) for x in torch.randint(1, 4, (int(torch.randint(0, 8, (1,), generator=g)),), generator=g)]
                    for _ in range(nr)
                ]
            )
        got = solve(hyps, refs)
        exp = _oracle(hyps, refs)
        assert all(_close(a, b) for a, b in zip(got, exp)), (hyps, refs, got, exp)


def check_identical_is_one(solve):
    # 完美摘要：三项全 1
    hyps = [[1, 2, 3, 4], [9, 8]]
    refs = [[[1, 2, 3, 4]], [[9, 8]]]
    r1, r2, rl = solve(hyps, refs)
    assert _close(r1, 1.0) and _close(r2, 1.0) and _close(rl, 1.0), (r1, r2, rl)


def check_order_sensitive_lcs(solve):
    # 换序：ROUGE-1 仍为 1，ROUGE-L 掉到 0.75（LCS=3），体现 LCS 的顺序敏感性
    r1, r2, rl = solve([[1, 2, 4, 3]], [[[1, 2, 3, 4]]])
    assert _close(r1, 1.0), r1
    assert _close(rl, 0.75), rl
    assert _close(r2, 1.0 / 3.0), r2


def check_empty_hypothesis(solve):
    # 空假设：F1 记 0，但仍占一条假设
    r1, r2, rl = solve([[], [1, 2]], [[[1, 2]], [[1, 2]]])
    assert _close(r1, 0.5) and _close(r2, 0.5) and _close(rl, 0.5), (r1, r2, rl)


def check_disjoint_is_zero(solve):
    # 完全不重合：三项 0
    r1, r2, rl = solve([[1, 2, 3]], [[[7, 8, 9]]])
    assert r1 == 0.0 and r2 == 0.0 and rl == 0.0, (r1, r2, rl)


def check_too_short_for_bigram(solve):
    # k > n：单 token 假设没有任何 bigram -> rouge2 为 0，但 rouge1 / rougeL 正常
    r1, r2, rl = solve([[3]], [[[1, 2, 3]]])
    assert r2 == 0.0, r2
    assert _close(r1, 0.5) and _close(rl, 0.5), (r1, rl)   # R = 1/3, P = 1 -> F = 0.5


def check_best_reference_is_picked(solve):
    # 多参考：取 F1 最大的那条参考，而不是第一条、也不是平均
    hyp = [1, 2, 9]
    ref_a = [9, 9, 9]   # 只有 9 命中：R=P=1/3 -> F1 = 1/3
    ref_b = [1, 2, 9]   # 全命中：F1 = 1
    only_a, _, _ = solve([hyp], [[ref_a]])
    only_b, _, _ = solve([hyp], [[ref_b]])
    both, _, _ = solve([hyp], [[ref_a, ref_b]])
    assert _close(only_a, 1.0 / 3.0), only_a
    assert _close(only_b, 1.0), only_b
    assert _close(both, 1.0), both          # 取最大：若是"参考平均"就会得到 2/3
    rev, _, _ = solve([hyp], [[ref_b, ref_a]])
    assert _close(rev, both), rev          # 与参考顺序无关


def check_empty_candidate_set(solve):
    # 没有任何假设（空候选集）：三项 0
    got = solve([], [])
    assert got == (0.0, 0.0, 0.0), got
    assert _oracle([], []) == (0.0, 0.0, 0.0)


def check_duplicate_reference_invariant(solve):
    base = solve(HYPS, REFS)
    dup = solve(HYPS, [r + [list(r[0])] for r in REFS])
    assert all(_close(a, b) for a, b in zip(base, dup)), (base, dup)
