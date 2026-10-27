import torch


def solve(prompt, observed, n, capacity):
    """n-gram 池：查表预测 + 未命中即学 + 容量淘汰（计数优先、时间戳老化兜底）。"""
    if prompt.ndim != 1 or observed.ndim != 1:
        raise ValueError("prompt 与 observed 都必须是一维")
    if n < 1:
        raise ValueError("n 必须 >= 1")
    if capacity < 1:
        raise ValueError("capacity 必须 >= 1")
    if int(prompt.numel()) < n + 1:
        raise ValueError("prompt 至少要能切出一个 n-gram（长度 >= n+1）")

    context = [int(x) for x in prompt.tolist()]
    counts = {}                                      # key -> {next_token: 次数}
    stamp = {}                                       # key -> 最近使用时间
    clock = 0

    def put(key, token):
        nonlocal clock
        table = counts.setdefault(key, {})
        table[token] = table.get(token, 0) + 1
        stamp[key] = clock
        clock += 1

    def trim():
        dropped = 0
        while len(counts) > capacity:
            victim = min(counts, key=lambda k: (sum(counts[k].values()), stamp[k], k))
            del counts[victim]
            del stamp[victim]
            dropped += 1
        return dropped

    for i in range(len(context) - n):
        put(tuple(context[i:i + n]), context[i + n])
    trim()

    predicted, hits, sizes, evicted = [], [], [], []
    for step, token in enumerate(int(x) for x in observed.tolist()):
        key = tuple(context[-n:])
        if key in counts:                             # 命中：用计数最多的后继
            table = counts[key]
            guess = min(table, key=lambda t: (-table[t], t))
            predicted.append(guess)
            hits.append(True)
            put(key, token)                           # 用真实 token 强化，而不是强化猜的那个
        else:                                         # 未命中：现场学习这一条 n-gram
            predicted.append(-1)
            hits.append(False)
            put(key, token)
        context.append(token)
        evicted.append(trim())
        sizes.append(len(counts))

    return (torch.tensor(predicted, dtype=torch.int64),
            torch.tensor(hits, dtype=torch.bool),
            torch.tensor(sizes, dtype=torch.int64),
            torch.tensor(evicted, dtype=torch.int64))
