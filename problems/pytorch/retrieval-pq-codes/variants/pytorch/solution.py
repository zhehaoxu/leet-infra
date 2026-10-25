"""参考答案：逐子空间广播算平方距离后 argmin，填入 (n, m)。"""

import torch


def solve(x, codebook):
    m, ksub, d = codebook.shape
    n = x.shape[0]
    codes = torch.zeros(n, m, dtype=torch.long)
    for j in range(m):
        sub = x[:, j * d:(j + 1) * d]                       # (n, d)
        d2 = ((sub[:, None, :] - codebook[j][None, :, :]) ** 2).sum(-1)  # (n, ksub)
        codes[:, j] = d2.argmin(dim=1)
    return codes
