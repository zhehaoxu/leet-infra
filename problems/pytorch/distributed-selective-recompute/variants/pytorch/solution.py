"""参考答案：0/1 背包 DP。dp[j][m] = 恰好保留 j 块、总显存恰为 m 时的最大保留 cost。
逐块倒序更新（每项最多选一次），最后对 j>=min_keep、m<=budget 取最大保留 cost；
不可行（无满足约束状态）抛 ValueError。"""


def solve(mem, cost, budget, min_keep):
    mem = [int(x) for x in mem]
    cost = [int(x) for x in cost]
    budget = int(budget)
    min_keep = int(min_keep)
    N = len(mem)

    NEG = -(10 ** 18)
    # dp[j][m]: 恰好保留 j 块、总显存 = m 时的最大保留 cost
    dp = [[NEG] * (budget + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    for i in range(N):
        mi, ci = mem[i], cost[i]
        # 倒序，保证每块最多被选一次
        for j in range(i, -1, -1):
            row_src = dp[j]
            row_dst = dp[j + 1]
            for m in range(budget - mi, -1, -1):
                if row_src[m] != NEG:
                    v = row_src[m] + ci
                    if v > row_dst[m + mi]:
                        row_dst[m + mi] = v

    best_keep = NEG
    for j in range(min_keep, N + 1):
        row = dp[j]
        for m in range(budget + 1):
            if row[m] > best_keep:
                best_keep = row[m]

    if best_keep == NEG:
        raise ValueError("infeasible: cannot keep min_keep ops within budget")

    return int(sum(cost) - best_keep)
