import torch


def _oracle(prompt, steps, min_n, max_n, pool_seed, V):
    """独立纯 Python：与题面一致的 pool 匹配 + 回退 + 更新循环。"""
    pool = {}
    for s in pool_seed:
        for i in range(len(s)):
            for n in range(min_n, max_n + 1):
                if i + n < len(s):
                    pool[tuple(s[i:i + n])] = s[i + n]
    seq = list(prompt)
    emitted = []
    for _ in range(steps):
        drafted = None
        n = max_n
        while n >= min_n:
            if len(seq) >= n:
                key = tuple(seq[len(seq) - n:])
                if key in pool:
                    drafted = pool[key]
                    break
            n -= 1
        if drafted is None:
            drafted = (seq[-1] + 1) % V
        seq.append(drafted)
        emitted.append(drafted)
        for n in range(min_n, max_n + 1):
            if len(seq) >= n + 1:
                pool[tuple(seq[len(seq) - n - 1:len(seq) - 1])] = seq[-1]
    return emitted


def check_example(solve):
    out = solve([0, 1], 4, 2, 2, [[0, 1, 2, 3]], 10)
    assert torch.equal(out, torch.tensor([2, 3, 4, 5])), out


def check_empty_pool_fallback(solve):
    out = solve([3], 3, 2, 2, [], 5)
    assert torch.equal(out, torch.tensor([4, 0, 1])), out


def check_seed_override(solve):
    # 同名 (0,1) 先设 7 再被后一条 seed 覆盖为 9
    out = solve([0, 1], 1, 2, 2, [[0, 1, 7], [0, 1, 9]], 10)
    assert torch.equal(out, torch.tensor([9])), out


def check_matches_oracle(solve):
    torch.manual_seed(21)
    for _ in range(50):
        V = int(torch.randint(2, 9, (1,)))
        plen = int(torch.randint(1, 5, (1,)))
        prompt = [int(x) for x in torch.randint(0, V, (plen,))]
        min_n = int(torch.randint(2, 4, (1,)))
        max_n = min_n + int(torch.randint(0, 2, (1,)))
        steps = int(torch.randint(1, 7, (1,)))
        nseed = int(torch.randint(0, 4, (1,)))
        pool_seed = []
        for _ in range(nseed):
            slen = int(torch.randint(min_n + 1, max_n + 4, (1,)))
            pool_seed.append([int(x) for x in torch.randint(0, V, (slen,))])
        out = solve(prompt, steps, min_n, max_n, pool_seed, V)
        exp = torch.tensor(_oracle(prompt, steps, min_n, max_n, pool_seed, V), dtype=torch.long)
        assert torch.equal(out, exp), (prompt, steps, min_n, max_n, pool_seed, V, out, exp)


check_example.example = True
