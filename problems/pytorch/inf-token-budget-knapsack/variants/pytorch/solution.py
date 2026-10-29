def solve(prompt_lens, values, budget, decode_count):
    remaining = budget - decode_count
    n = len(prompt_lens)

    # dp[c] = (最大权重, 升序下标元组)，只用掉恰好 c 个名额
    dp = [None] * (remaining + 1)
    dp[0] = (0, ())
    for i in range(n):
        cost, val = prompt_lens[i], values[i]
        if cost > remaining:
            continue
        for c in range(remaining, cost - 1, -1):
            prev = dp[c - cost]
            if prev is None:
                continue
            cand = (prev[0] + val, prev[1] + (i,))
            cur = dp[c]
            if cur is None or (-cand[0], cand[1]) < (-cur[0], cur[1]):
                dp[c] = cand

    best_cost = 0
    best = dp[0]
    for c in range(1, remaining + 1):
        cand = dp[c]
        if cand is None:
            continue
        if (-cand[0], c, cand[1]) < (-best[0], best_cost, best[1]):
            best = cand
            best_cost = c

    total_value, indices = best
    served = list(indices)
    chosen = set(served)
    leftover = remaining - best_cost

    chunk_index, chunk_tokens = -1, 0
    if leftover > 0:
        for i in range(n):
            if i not in chosen:
                chunk_index = i
                chunk_tokens = min(leftover, prompt_lens[i])
                break
    return served, total_value, chunk_index, chunk_tokens, leftover - chunk_tokens
