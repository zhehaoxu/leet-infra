def solve(N, B, P):
    """Sequoia 最优草稿树动态规划（arXiv:2402.12374 Algorithm 1）。

    T[n,b] = 一棵 size=n、根恰有 b 个直接孩子（FirstBranch=b）的树的最大 F(T)。
    转移：T[n,b] = max_{1<=m<=n-1} ( T[n-m, b-1] + P[b] * Tmax[m] )，
    其中把"最后一个（第 b 个）孩子的子树"拆出来 size=m，只有第 b 个孩子被接受（概率
    P[b]）时其子树才计入期望；前 b-1 个孩子及其后代构成 T[n-m, b-1]。
    Tmax[n] = max_{0<=b<=B} T[n,b]。base：T[1,0]=1（只有根），Tmax[1]=1。
    """
    NEG = float("-inf")
    T = [[NEG] * (B + 1) for _ in range(N + 1)]
    Tmax = [NEG] * (N + 1)
    T[1][0] = 1.0
    Tmax[1] = 1.0
    for n in range(2, N + 1):
        for b in range(1, B + 1):
            best = NEG
            for m in range(1, n):
                if T[n - m][b - 1] == NEG or Tmax[m] == NEG:
                    continue
                val = T[n - m][b - 1] + P[b - 1] * Tmax[m]
                if val > best:
                    best = val
            T[n][b] = best
        Tmax[n] = max(T[n][b] for b in range(B + 1))
    return float(Tmax[N])
