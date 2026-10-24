def _close(a, b, tol=1e-6):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


def _oracle(N, B, P):
    """独立纯 Python：暴力递归枚举根出度 + 子树规模组合，最大化 F=1+Σ P[i-1]*F(子树)。"""
    def comps(total, parts):
        if parts == 1:
            yield (total,)
            return
        for first in range(1, total - parts + 2):
            for rest in comps(total - first, parts - 1):
                yield (first,) + rest

    memo = {}

    def best(n):
        if n in memo:
            return memo[n]
        if n == 1:
            memo[n] = 1.0
            return 1.0
        r = float("-inf")
        for b in range(1, min(B, n - 1) + 1):
            for comp in comps(n - 1, b):
                val = 1.0 + sum(P[i - 1] * best(comp[i - 1]) for i in range(1, b + 1))
                if val > r:
                    r = val
        memo[n] = r
        return r

    return best(N)


def check_example(solve):
    assert _close(solve(3, 1, [0.7]), 2.19), solve(3, 1, [0.7])
    assert _close(solve(4, 2, [0.9, 0.5]), 3.439), solve(4, 2, [0.9, 0.5])
    assert _close(solve(1, 3, [0.9, 0.6, 0.3]), 1.0), solve(1, 3, [0.9, 0.6, 0.3])


def check_chain_formula(solve):
    # B=1 -> 几何级数 (1-c^N)/(1-c)
    for N, c in [(5, 0.8), (7, 0.5), (4, 0.95)]:
        exp = (1 - c ** N) / (1 - c)
        assert _close(solve(N, 1, [c]), exp), (N, c, solve(N, 1, [c]), exp)


def check_all_accept(solve):
    # 接受率全 1 -> 每棵 size=N 的树 F=N
    for N, B in [(3, 2), (6, 3), (5, 4)]:
        assert _close(solve(N, B, [1.0] * B), float(N)), (N, B, solve(N, B, [1.0] * B))


def _random_cases(solve):
    import random

    rng = random.Random(11)
    for _ in range(200):
        N = rng.randint(1, 8)
        B = max(1, min(N - 1 if N > 1 else 1, rng.randint(1, 4)))
        P = [round(rng.uniform(0.05, 1.0), 4) for _ in range(B)]
        assert _close(solve(N, B, P), _oracle(N, B, P), 1e-9), (N, B, P)


def check_matches_oracle(solve):
    _random_cases(solve)


check_example.example = True
