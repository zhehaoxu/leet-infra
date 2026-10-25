"""参考答案：Unigram 前向 Viterbi DP + 回溯下标。"""

import torch


def solve(text, vocab, log_probs):
    piece2idx = {p: i for i, p in enumerate(vocab)}
    n = len(text)
    NEG = float("-inf")
    best = [NEG] * (n + 1)
    best[0] = 0.0
    back = [None] * (n + 1)
    for j in range(1, n + 1):
        for pi, p in enumerate(vocab):
            l = len(p)
            if l == 0 or j - l < 0:
                continue
            if best[j - l] != NEG and text[j - l:j] == p:
                sc = best[j - l] + log_probs[pi]
                if sc > best[j]:
                    best[j] = sc
                    back[j] = (j - l, pi)
    if best[n] == NEG:
        raise ValueError("no segmentation covers the text")
    seg = []
    j = n
    while j > 0:
        prev, pi = back[j]
        seg.append(pi)
        j = prev
    seg.reverse()
    return torch.tensor(seg, dtype=torch.long)
