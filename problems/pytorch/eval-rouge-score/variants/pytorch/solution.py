"""参考答案：ROUGE-1 / ROUGE-2 / ROUGE-L（Lin 2004），逐参考取最大 F1 后对假设求平均。"""

from collections import Counter


def _ngram_f1(hyp, ref, n):
    """单对 (假设, 参考) 的 ROUGE-n F1：裁剪重合 -> 召回 / 精度 -> 调和平均。"""
    if not hyp or not ref:
        return 0.0
    hc = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
    rc = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
    overlap = sum(min(v, rc.get(g, 0)) for g, v in hc.items())
    total_h = sum(hc.values())
    total_r = sum(rc.values())
    if total_h == 0 or total_r == 0:
        return 0.0
    recall = overlap / total_r
    precision = overlap / total_h
    if recall + precision <= 0.0:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


def _lcs_len(a, b):
    """最长公共子序列长度：完整 DP 矩阵。"""
    m, k = len(a), len(b)
    dp = [[0] * (k + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i][j - 1]
    return dp[m][k]


def _lcs_f1(hyp, ref):
    if not hyp or not ref:
        return 0.0
    l = _lcs_len(hyp, ref)
    recall = l / len(ref)
    precision = l / len(hyp)
    if recall + precision <= 0.0:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


def _aggregate(per_hypothesis):
    return sum(per_hypothesis) / len(per_hypothesis) if per_hypothesis else 0.0


def solve(hypotheses, references):
    r1, r2, rl = [], [], []
    for hyp, refs in zip(hypotheses, references):
        r1.append(max((_ngram_f1(hyp, ref, 1) for ref in refs), default=0.0))
        r2.append(max((_ngram_f1(hyp, ref, 2) for ref in refs), default=0.0))
        rl.append(max((_lcs_f1(hyp, ref) for ref in refs), default=0.0))
    return _aggregate(r1), _aggregate(r2), _aggregate(rl)
