import torch


def solve(prompt, steps, min_n, max_n, pool_seed, V):
    """Lookahead decoding — Jacobi n-gram candidate pool (arXiv:2402.02057)。"""
    pool = {}
    # 预置：每条 seed 序列的所有长度∈[min_n,max_n] 子串 -> 其后继 token（后者覆盖前者）
    for s in pool_seed:
        for i in range(len(s)):
            for n in range(min_n, max_n + 1):
                if i + n < len(s):
                    pool[tuple(s[i:i + n])] = s[i + n]

    seq = list(prompt)
    emitted = []
    for _ in range(steps):
        drafted = None
        # 最长后缀优先：max_n..min_n
        for n in range(max_n, min_n - 1, -1):
            if len(seq) >= n:
                key = tuple(seq[-n:])
                if key in pool:
                    drafted = pool[key]
                    break
        if drafted is None:
            drafted = (seq[-1] + 1) % V  # 确定性回退
        seq.append(drafted)
        emitted.append(drafted)
        # 自学习：把「紧邻新 token 的 n-gram 前缀 -> 新 token」写回池
        for n in range(min_n, max_n + 1):
            if len(seq) >= n + 1:
                pool[tuple(seq[-(n + 1):-1])] = seq[-1]
    return torch.tensor(emitted, dtype=torch.long)
