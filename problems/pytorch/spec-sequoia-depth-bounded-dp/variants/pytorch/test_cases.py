import random

import torch


def _close(a, b, tol=1e-6):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


def _oracle(A, N, B, D, td, tv):
    """独立重算：递归枚举所有「出度<=B、绝对深度<=D、规模<=N」的有序树（孩子子树的嵌套元组），
    每棵树用显式栈沿根到各节点的路径逐层乘 A[深度-1][序号-1] 得到 F，再逐棵树取
    F/(tv+td*size) 最大。与参考解的 DP 状态转移完全无共享代码路径。"""
    Al = A.tolist()
    memo = {}

    def shapes(d, lim):
        key = (d, lim)
        if key in memo:
            return memo[key]
        out = [((), 1)]
        if d < D and lim >= 2:
            kids = shapes(d + 1, lim - 1)

            def seqs(left, budget):
                if left == 0:
                    yield (), 0
                    return
                for kt, ks in kids:
                    if ks > budget:
                        continue
                    for rest, rs in seqs(left - 1, budget - ks):
                        yield (kt,) + rest, ks + rs

            for b in range(1, B + 1):
                for ch, cs in seqs(b, lim - 1):
                    out.append((ch, 1 + cs))
        memo[key] = out
        return out

    def F_of(tree):
        total = 1.0
        stack = [(ch, 1, Al[0][i]) for i, ch in enumerate(tree)]
        while stack:
            node, depth, prod = stack.pop()
            total += prod
            for i, ch in enumerate(node):
                stack.append((ch, depth + 1, prod * Al[depth][i]))
        return total

    ratios = []
    for tree, sz in shapes(0, N):
        ratios.append((sz, F_of(tree) / (float(tv) + float(td) * sz)))
    best_r = max(r for _, r in ratios)
    opt_sizes = sorted({sz for sz, r in ratios if abs(r - best_r) <= 1e-9 * max(1.0, abs(best_r))})
    return best_r, opt_sizes


def check_example(solve):
    A = torch.tensor([[0.75, 0.5], [0.5, 0.25]])
    got = solve(A, 4, 2, 2, 0.25, 1.0)
    assert _close(got[0], 1.3125) and got[1] == 4, got
    got = solve(A, 4, 2, 2, 0.0, 1.0)
    assert _close(got[0], 2.625) and got[1] == 4, got
    got = solve(torch.zeros(0, 2), 4, 2, 0, 0.25, 1.0)
    assert _close(got[0], 0.8) and got[1] == 1, got


def check_matches_oracle(solve):
    rng = random.Random(20240212)
    for _ in range(120):
        N = rng.randint(1, 7)
        B = rng.randint(1, 3)
        D = rng.randint(0, min(3, max(N - 1, 0)))
        A = torch.tensor([[round(rng.uniform(0.02, 1.0), 3) for _ in range(B)] for _ in range(D)])
        td = round(rng.uniform(0.0, 0.6), 3)
        tv = round(rng.uniform(0.5, 2.0), 3)
        ratio, n = solve(A, N, B, D, td, tv)
        exp_r, opt_sizes = _oracle(A, N, B, D, td, tv)
        assert _close(ratio, exp_r), (N, B, D, A.tolist(), td, tv, ratio, exp_r)
        assert n in opt_sizes, (N, B, D, A.tolist(), td, tv, n, opt_sizes)


def check_depth_cap_kills_bigger_trees(solve):
    # D=1, B=2：最多 1+2=3 个节点，规模 4 不可行；全接受时最优规模被截到 3
    A = torch.ones(1, 2)
    ratio, n = solve(A, 4, 2, 1, 0.25, 1.0)
    assert n == 3, (ratio, n)
    assert _close(ratio, 3.0 / 1.75), (ratio, n)
    # D=2 时同样全接受，规模 6 = 1+2+4 可行且比值最大
    A = torch.ones(2, 2)
    ratio, n = solve(A, 6, 2, 2, 0.25, 1.0)
    assert n == 6, (ratio, n)
    assert _close(ratio, 6.0 / 2.5), (ratio, n)


def check_zero_accept_tie_break_smallest_size(solve):
    # 接受率全 0：任何规模的期望都是 1，比值 1/(tv+td*n) 随 n 递减，最优规模必为 1
    A = torch.zeros(3, 2)
    ratio, n = solve(A, 5, 2, 3, 0.2, 1.3)
    assert n == 1, (ratio, n)
    assert _close(ratio, 1.0 / 1.5), (ratio, n)


def check_chain_matches_geometric_sum(solve):
    # B=1：唯一形状是链，F(n)=sum_{i<=min(n-1,D)} c^i
    for c, D, N, td, tv in [(0.7, 4, 6, 0.1, 1.0), (0.5, 2, 5, 0.0, 1.0), (0.9, 1, 4, 0.4, 2.0)]:
        A = torch.full((D, 1), c)
        F = {}
        for n in range(1, N + 1):
            if n - 1 > D:
                continue
            F[n] = sum(c ** i for i in range(min(n - 1, D) + 1))
        exp = max(f / (tv + td * nn) for nn, f in F.items())
        ratio, nn = solve(A, N, 1, D, td, tv)
        assert _close(ratio, exp), (c, D, N, td, tv, ratio, exp)


def check_reduces_to_177_when_cap_inactive(solve):
    # A 各行相同 + D >= N-1 + td=0, tv=1 时，必须复现 spec-sequoia-optimal-tree(177) 的 Tmax[N]
    for N, B, P, exp in [(3, 1, [0.7], 2.19), (4, 2, [0.9, 0.5], 3.439), (1, 3, [0.9, 0.6, 0.3], 1.0)]:
        D = max(N - 1, 1)
        A = torch.tensor([list(P) for _ in range(D)])
        ratio, n = solve(A, N, B, D, 0.0, 1.0)
        assert _close(ratio, exp, 1e-5), (N, B, P, ratio, exp)
        assert n == N or N == 1, (N, n)


def check_cost_objective_prefers_smaller_tree(solve):
    # 草稿很贵：最优规模应远小于预算 N；把 td 置 0 后最优规模回到 N
    A = torch.tensor([[0.9, 0.2], [0.85, 0.1], [0.8, 0.05]])
    cheap_ratio, cheap_n = solve(A, 6, 2, 3, 0.02, 1.0)
    dear_ratio, dear_n = solve(A, 6, 2, 3, 1.5, 1.0)
    assert dear_n < 6, (dear_ratio, dear_n)
    assert cheap_n >= dear_n, (cheap_n, dear_n)
    exp_r, exp_sizes = _oracle(A, 6, 2, 3, 1.5, 1.0)
    assert _close(dear_ratio, exp_r) and dear_n in exp_sizes, (dear_ratio, dear_n, exp_r, exp_sizes)
    exp_r, exp_sizes = _oracle(A, 6, 2, 3, 0.02, 1.0)
    assert _close(cheap_ratio, exp_r) and cheap_n in exp_sizes, (cheap_ratio, cheap_n, exp_r, exp_sizes)


check_example.example = True
