"""参考答案：Sequoia 深度受限最优草稿树 DP（arXiv:2402.12374，Algorithm 1 的深度扩展 + §3.3.1 的代价目标）。"""


def solve(A, N, B, D, td, tv):
    """T[n][b][d] = 根在绝对深度 d、恰有 b 个孩子、规模恰为 n、全树深度 <= D 的最大期望 F。
    转移把第 b 个孩子（绝对深度 d+1，接受率 A[d][b-1]）的子树拆出来 size=m，其余 b-1 个孩子
    留在同一深度 d；base T[1][0][d]=1。最后对可行 n 取 max F(n)/(tv+td*n)。"""
    NEG = float("-inf")
    Al = A.tolist()
    T = [[[NEG] * (D + 1) for _ in range(B + 1)] for _ in range(N + 1)]
    Tm = [[NEG] * (D + 1) for _ in range(N + 1)]
    for d in range(D + 1):
        T[1][0][d] = 1.0
        Tm[1][d] = 1.0
    for d in range(D - 1, -1, -1):
        for n in range(2, N + 1):
            for b in range(1, min(B, n - 1) + 1):
                best = NEG
                acc = Al[d][b - 1]
                for m in range(1, n):
                    rest, sub = T[n - m][b - 1][d], Tm[m][d + 1]
                    if rest == NEG or sub == NEG:
                        continue
                    val = rest + acc * sub
                    if val > best:
                        best = val
                T[n][b][d] = best
            Tm[n][d] = max(T[n][bb][d] for bb in range(B + 1))

    best_ratio, best_n = NEG, 1
    for n in range(1, N + 1):
        f = Tm[n][0]
        if f == NEG:
            continue
        ratio = f / (tv + td * n)
        if ratio > best_ratio:
            best_ratio, best_n = ratio, n
    return float(best_ratio), best_n
