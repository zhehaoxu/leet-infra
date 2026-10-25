"""参考答案：语料级 BLEU（Papineni 2002）——n-gram 裁剪精度 + 几何平均 + 简洁惩罚。"""

import math
from collections import Counter

import torch


def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _closest_ref_len(ref_lens, hyp_len):
    """与假设长度最接近的参考长度；并列取更短的那个。"""
    best = None
    for ln in ref_lens:
        key = (abs(ln - hyp_len), ln)
        if best is None or key < best:
            best = key
    return best[1] if best is not None else 0


def solve(hypotheses, references, max_n=4, smoothing=False):
    counts = [0] * max_n
    matches = [0] * max_n
    c_total = 0
    r_total = 0
    for hyp, refs in zip(hypotheses, references):
        c_total += len(hyp)
        r_total += _closest_ref_len([len(x) for x in refs], len(hyp))
        for n in range(1, max_n + 1):
            hc = _ngram_counts(hyp, n)
            counts[n - 1] += sum(hc.values())
            max_ref = {}
            for ref in refs:
                for gram, value in _ngram_counts(ref, n).items():
                    if value > max_ref.get(gram, 0):
                        max_ref[gram] = value
            for gram, value in hc.items():
                matches[n - 1] += min(value, max_ref.get(gram, 0))

    precisions = []
    for k in range(max_n):
        if counts[k] == 0:
            precisions.append(0.0)
        elif smoothing:
            precisions.append((matches[k] + 1.0) / (counts[k] + 1.0))
        else:
            precisions.append(matches[k] / counts[k])

    if c_total == 0 or any(p <= 0.0 for p in precisions):
        bleu = 0.0
    else:
        bp = 1.0 if c_total >= r_total else math.exp(1.0 - r_total / c_total)
        geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
        bleu = bp * geo
    return bleu, torch.tensor(precisions, dtype=torch.float32)
