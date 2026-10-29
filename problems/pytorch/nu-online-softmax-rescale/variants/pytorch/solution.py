"""参考答案：段内取 max、段间只做一次 exp(m_s - M) 的重标定，全程不碰未减最大值的 exp。

合并公式里 l_s * exp(m_s - M) 的两种退化情形要分清楚：
- m_s = -inf（整段被掩码掉）而 M 有限：exp(-inf - M) = 0，l_s 也必然为 0，乘积 0 —— 干净；
- m_s = M = -inf（整行都是 -inf）：-inf - -inf = NaN，所以这种行必须提前分支处理，
  不能让 IEEE 把 NaN 泼进整行——这也是「无定义」的正确表达方式。
+inf 行同样单独走一条支路：M = +inf 时所有有限值都会被 exp(finite - inf) = 0 吸收，
剩下的只是「几个 +inf 均分 1」，所以先数 +inf 的个数再赋值，比硬算更稳。

段内最大值用张量版 max() 取（它会传播 NaN），段间归约在 Python 里做，但必须先
把含 NaN 的行分流出去：`max(list)` 遇到 NaN 的结果取决于元素顺序，直接拿它当总
max 就会让判题变成看运气。

还有个小坑：`torch.tensor(一个 python float)` 默认给 float32，如果用它求段间最大值，
1e39 这种「有限但超出 float32 范围」的 max 会被折成 +inf，整行于是被误送进 +inf 支路，
lse 从 1e39 变成 inf。probs 反而看不出问题——分子分母同时带着 exp(m_s - M) 与 exp(x - M)，
M 的误差自己抵消了，所以这个 bug 只在 lse 上暴露，测试专门盯这一点。段间归约请用
Python 的 max，但要先处理 NaN：`max(list)` 遇到 NaN 的结果取决于元素顺序，得先单独分支。
"""

import math

import torch

INF = float('inf')


def solve(x, shard_size):
    if not isinstance(x, torch.Tensor):
        raise ValueError("x 必须是 torch.Tensor")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("x 必须是 float32 或 float64 张量")
    if x.dim() != 2:
        raise ValueError("x 必须是二维张量 (rows, cols)")
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size < 1:
        raise ValueError("shard_size 必须是 >= 1 的整数")
    rows, cols = tuple(x.shape)
    if cols == 0:
        raise ValueError("每行至少要有 1 个元素")

    src = x.detach().to(torch.float64)
    probs = torch.full_like(src, float('nan'))
    lse = torch.full((rows,), float('nan'), dtype=torch.float64)
    starts = list(range(0, cols, shard_size))

    for r in range(rows):
        row = src[r]
        ms = [float(row[s:s + shard_size].max()) for s in starts]
        if any(m != m for m in ms):  # 段内 max 会传播 NaN，先按行分流
            continue
        M = max(ms)
        if M == INF:
            mask = row == INF
            k = int(mask.sum())
            probs[r] = torch.where(mask, torch.full_like(row, 1.0 / k), torch.zeros_like(row))
            lse[r] = INF
            continue
        if M == -INF:
            lse[r] = -INF
            continue
        L = 0.0
        for s, m_s in zip(starts, ms):
            seg = row[s:s + shard_size]
            if m_s == -INF:
                continue  # 整段被掩掉，贡献 0
            l_s = float(torch.exp(seg - m_s).sum())
            L += l_s * math.exp(m_s - M)
        probs[r] = torch.exp(row - M) / L
        lse[r] = M + math.log(L)

    return probs.to(x.dtype), lse.to(x.dtype)
