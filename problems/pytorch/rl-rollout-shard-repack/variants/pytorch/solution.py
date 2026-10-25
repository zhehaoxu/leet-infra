"""参考答案：rollout 数据的跨 rank 重分片 + padding-free 微批打包 + pad 开销账。"""

import torch


def solve(seq_lens, num_ranks, micro_budget):
    lens = [int(x) for x in seq_lens.reshape(-1).tolist()]
    n = len(lens)
    w = int(num_ranks)
    m = int(micro_budget)

    dest = [0] * n
    micro = [0] * n
    tokens = [0] * w
    saved = [0] * w
    rank_lists = [[] for _ in range(w)]

    if n > 0 and w > 0:
        # 1) 均衡顺序：按长度升序，等长按原下标升序
        order = sorted(range(n), key=lambda i: (lens[i], i))
        # 2) 贪心最短队列优先（同负载取最小 rank）
        for i in order:
            best = 0
            for r in range(1, w):
                if tokens[r] < tokens[best]:
                    best = r
            dest[i] = best
            rank_lists[best].append(i)
            tokens[best] += lens[i]
        # 3) 每 rank 内部做 token 预算下的贪心装箱（单条超预算独占一包）
        mb_id = 0
        for r in range(w):
            if not rank_lists[r]:
                continue
            cur = 0
            for i in rank_lists[r]:
                if cur > 0 and cur + lens[i] > m:
                    mb_id += 1
                    cur = 0
                micro[i] = mb_id
                cur += lens[i]
            mb_id += 1
        # 4) 序列级 padding 的开销差：rank 内全部补齐到最长条
        for r in range(w):
            ids = rank_lists[r]
            if ids:
                longest = max(lens[i] for i in ids)
                saved[r] = longest * len(ids) - tokens[r]

    moved = [0] * n
    for i in range(n):
        init = i * w // n if n > 0 else 0
        moved[i] = 1 if dest[i] != init else 0

    return (torch.tensor(dest, dtype=torch.long),
            torch.tensor(micro, dtype=torch.long),
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(saved, dtype=torch.long),
            torch.tensor(moved, dtype=torch.long))
