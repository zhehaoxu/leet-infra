import math

import torch

HYPS = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 8]]
REFS = [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 7, 6]], [[1, 2, 3, 4, 5]]]
EXAMPLE_BLEU = 0.7578848           # 参考解跑出的语料 BLEU
EXAMPLE_PREC = [0.9, 0.875, 0.8333333, 0.75]  # 参考解跑出的 1..4 阶裁剪精度


def _oracle(hypotheses, references, max_n=4, smoothing=False):
    """独立重算：全手动嵌套循环 + 普通 dict，几何平均用连开后四次方而非 exp(sum log)。"""
    counts = [0] * max_n
    matches = [0] * max_n
    c_total = 0
    r_total = 0
    for hyp, refs in zip(hypotheses, references):
        c_total += len(hyp)
        best_diff = None
        best_len = 0
        for ref in refs:
            d = abs(len(ref) - len(hyp))
            if best_diff is None or d < best_diff or (d == best_diff and len(ref) < best_len):
                best_diff = d
                best_len = len(ref)
        r_total += best_len
        for n in range(1, max_n + 1):
            grams = []
            for i in range(len(hyp) - n + 1):
                g = tuple()
                for j in range(n):
                    g = g + (hyp[i + j],)
                grams.append(g)
            counts[n - 1] += len(grams)
            distinct = {}
            for g in grams:
                distinct[g] = distinct.get(g, 0) + 1
            for g, hcnt in distinct.items():
                refmax = 0
                for ref in refs:
                    rcnt = 0
                    for i in range(len(ref) - n + 1):
                        gg = tuple()
                        for j in range(n):
                            gg = gg + (ref[i + j],)
                        if gg == g:
                            rcnt += 1
                    if rcnt > refmax:
                        refmax = rcnt
                matches[n - 1] += hcnt if hcnt < refmax else refmax

    prec = []
    for k in range(max_n):
        if counts[k] == 0:
            prec.append(0.0)
        elif smoothing:
            prec.append((matches[k] + 1.0) / (counts[k] + 1.0))
        else:
            prec.append(matches[k] / counts[k])
    if c_total == 0 or min(prec) <= 0.0:
        bleu = 0.0
    else:
        bp = min(1.0, math.exp(1.0 - r_total / c_total))
        bleu = bp * (math.prod(prec) ** (1.0 / max_n))
    return bleu, prec


def _close(a, b):
    return torch.allclose(
        torch.tensor(float(a), dtype=torch.float64),
        torch.tensor(float(b), dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )


def check_example(solve):
    bleu, prec = solve(HYPS, REFS)
    assert _close(bleu, EXAMPLE_BLEU), f"bleu got {bleu}"
    assert torch.allclose(
        prec, torch.tensor(EXAMPLE_PREC, dtype=torch.float32), atol=1e-5, rtol=1e-5
    ), prec.tolist()


check_example.example = True


def check_matches_oracle(solve):
    corpora = [
        (HYPS, REFS),
        ([[1, 2, 3], [4, 5]], [[[1, 2, 3]], [[4, 5, 6]]]),
        ([[7, 7, 7, 7]], [[[7, 7]]]),
        ([[1, 1, 2], [1, 2, 1]], [[[1, 1, 2], [1, 2, 1]], [[2, 1, 1]]]),
    ]
    for hyps, refs in corpora:
        for mn in (1, 2, 3, 4):
            for sm in (False, True):
                b1, p1 = solve(hyps, refs, mn, sm)
                b2, p2 = _oracle(hyps, refs, mn, sm)
                assert _close(b1, b2), (hyps, refs, mn, sm, b1, b2)
                assert all(_close(x, y) for x, y in zip(p1.tolist(), p2)), (mn, sm, p1.tolist(), p2)


def check_random_against_oracle(solve):
    g = torch.Generator().manual_seed(23)
    for trial in range(4):
        m = int(torch.randint(1, 4, (1,), generator=g))
        hyps = []
        refs = []
        for _ in range(m):
            lh = int(torch.randint(0, 8, (1,), generator=g))
            hyps.append([int(x) for x in torch.randint(1, 5, (lh,), generator=g)])
            nr = int(torch.randint(1, 3, (1,), generator=g))
            refs.append([[int(x) for x in torch.randint(1, 5, (int(torch.randint(0, 9, (1,), generator=g)),), generator=g)] for _ in range(nr)])
        for sm in (False, True):
            b1, p1 = solve(hyps, refs, 3, sm)
            b2, p2 = _oracle(hyps, refs, 3, sm)
            assert _close(b1, b2) and all(_close(x, y) for x, y in zip(p1.tolist(), p2)), (hyps, refs)


def check_identical_hyp_ref_is_one(solve):
    # 完美译文：各阶精度 1、c == r -> BP = 1 -> BLEU = 1
    hyps = [[1, 2, 3, 4, 5], [9, 8, 7]]
    refs = [[[1, 2, 3, 4, 5]], [[9, 8, 7]]]
    bleu, prec = solve(hyps, refs)
    assert _close(bleu, 1.0), bleu
    assert torch.allclose(prec, torch.ones(4), atol=1e-6, rtol=0), prec.tolist()


def check_clipped_precision_limits_repeats(solve):
    # 假设里 4 个 unigram，参考只有 2 个 -> 裁剪命中 2，p1 = 0.5；4-gram 无 -> 未平滑 BLEU = 0
    bleu, prec = solve([[7, 7, 7, 7]], [[[7, 7]]], max_n=1)
    assert _close(bleu, 0.5), (bleu, prec.tolist())
    bleu4, prec4 = solve([[7, 7, 7, 7]], [[[7, 7]]])
    assert prec4[0].item() == 0.5, prec4.tolist()
    assert _close(bleu4, 0.0), bleu4
    sm, smprec = solve([[7, 7, 7, 7]], [[[7, 7]]], smoothing=True)
    assert sm > 0.0 and _close(smprec[0], 3.0 / 5.0), (sm, smprec.tolist())


def check_short_hypothesis_zero_order(solve):
    # 假设长度 < n：该阶没有任何 n-gram 位置 -> 精度记 0（平滑也救不回来）
    bleu, prec = solve([[1, 2]], [[[1, 2, 3, 4, 5]]], max_n=4)
    assert prec[2].item() == 0.0 and prec[3].item() == 0.0, prec.tolist()
    assert _close(bleu, 0.0), bleu
    b2, p2 = solve([[1, 2]], [[[1, 2, 3, 4, 5]]], max_n=2)
    assert _close(p2[0], 1.0) and _close(p2[1], 1.0), p2.tolist()
    assert b2 > 0.0, b2


def check_empty_corpus(solve):
    # 没有任何假设（空候选集）：BLEU 0，精度全 0
    bleu, prec = solve([], [])
    assert _close(bleu, 0.0), bleu
    assert prec.shape == (4,) and prec.sum().item() == 0.0, prec.tolist()
    assert _close(_oracle([], [])[0], bleu)


def check_all_empty_hypotheses(solve):
    # c = 0（每条假设都是空串）：分母为零，BLEU 记 0
    bleu, prec = solve([[], []], [[[1, 2]], [[3, 4]]])
    assert _close(bleu, 0.0) and _close(prec[0], 0.0), (bleu, prec.tolist())


def check_brevity_penalty_matches_formula(solve):
    # 精度为 1 时 BLEU 恰等于 BP = exp(1 - r/c)
    hyps = [[1, 2, 3]]
    refs = [[[1, 2, 3, 4, 5, 6]]]
    bleu, prec = solve(hyps, refs, max_n=1)
    assert torch.allclose(prec, torch.ones(1), atol=1e-6, rtol=0), prec.tolist()
    assert _close(bleu, math.exp(1.0 - 6.0 / 3.0)), bleu
    # c > r 时 BP 饱和为 1，BLEU 就等于几何平均本身（长度红利为零）
    b_over, p_over = solve([[1, 2, 3]], [[[1, 2]]], max_n=1)
    assert _close(p_over[0], 2.0 / 3.0), p_over.tolist()
    assert _close(b_over, 2.0 / 3.0), b_over


def check_duplicate_reference_is_invariant(solve):
    # 把参考复制一份：max 裁剪与最短距离都不变 -> BLEU 不变
    base, bp = solve(HYPS, REFS)
    dup_refs = [r + [list(r[0])] for r in REFS]
    dup, dp = solve(HYPS, dup_refs)
    assert _close(base, dup) and _close(bp[0], dp[0]), (base, dup, bp.tolist(), dp.tolist())
