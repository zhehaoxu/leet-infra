"""参考答案：CTC 前向算法，log 空间稳定递推。"""

import math

import torch


def _lae(a, b):
    neg = float("-inf")
    if a == neg:
        return b
    if b == neg:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def solve(log_probs, labels):
    T, V = log_probs.shape
    L = labels.shape[0]
    blank = V - 1
    Lp = 2 * L + 1

    def label_at(k):
        return blank if k % 2 == 0 else int(labels[k // 2])

    neg = float("-inf")
    prev = [neg] * Lp
    prev[0] = log_probs[0, blank].item()
    if L >= 1:
        prev[1] = log_probs[0, int(labels[0])].item()

    for tt in range(1, T):
        cur = [neg] * Lp
        for k in range(Lp):
            la = log_probs[tt, label_at(k)].item()
            m = prev[k]
            if k >= 1:
                m = _lae(m, prev[k - 1])
            l_k = label_at(k)
            if l_k != blank and k >= 2 and prev[k - 2] != neg and label_at(k - 2) != l_k:
                m = _lae(m, prev[k - 2])
            cur[k] = m + la if m != neg else neg
        prev = cur

    total = prev[Lp - 1]
    if Lp >= 2 and prev[Lp - 2] != neg:
        total = _lae(total, prev[Lp - 2])
    return torch.tensor(-total, dtype=torch.float32)
