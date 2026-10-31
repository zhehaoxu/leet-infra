
def _py(correct, tokens, bin_edges):
    n = len(correct)
    k = len(bin_edges) - 1
    bins = []
    assign = []
    for t in tokens:
        j = k - 1
        for b in range(k - 1):
            if bin_edges[b] <= t < bin_edges[b + 1]:
                j = b
                break
        assign.append(j)
    for j in range(k):
        idx = [i for i in range(n) if assign[i] == j]
        cnt = len(idx)
        acc = (sum(1 for i in idx if correct[i]) / cnt) if cnt else None
        mt = (sum(tokens[i] for i in idx) / cnt) if cnt else None
        bins.append({"count": cnt, "accuracy": acc, "mean_tokens": mt})
    nc = sum(1 for c in correct if c)
    lc = [t for c, t in zip(correct, tokens) if c]
    lw = [t for c, t in zip(correct, tokens) if not c]
    len_c = (sum(lc) / len(lc)) if lc else None
    len_w = (sum(lw) / len(lw)) if lw else None
    eff = (len_w / len_c) if (len_c and len_w is not None) else None
    best = None
    best_acc = -1.0
    for j in range(k):
        if bins[j]["count"] > 0 and bins[j]["accuracy"] > best_acc:
            best_acc = bins[j]["accuracy"]
            best = j
    return {
        "n": n,
        "accuracy": nc / n,
        "total_tokens": sum(tokens),
        "bins": bins,
        "len_correct": len_c,
        "len_wrong": len_w,
        "efficiency_ratio": eff,
        "best_bin": best,
    }


def _compare(solve, correct, tokens, bin_edges):
    led = solve(list(correct), list(tokens), list(bin_edges))
    ref = _py(correct, tokens, bin_edges)
    assert led == ref, f"\nled={led}\nref={ref}"
    return led


def check_anchor_basic(solve):
    # 4 条样本、2 箱（edges=[10,30,100] → K=2）：箱0 [10,30)，箱1 [30,∞)
    correct = [True, False, True, True]
    tokens = [5, 20, 30, 100]
    led = _compare(solve, correct, tokens, [10, 30, 100])
    assert led["n"] == 4
    assert abs(led["accuracy"] - 0.75) < 1e-12
    assert led["total_tokens"] == 155
    # t=5 < 10 → 落末箱；t=20 → 箱0；t=30、t=100 → 末箱
    assert led["bins"][0]["count"] == 1
    assert led["bins"][1]["count"] == 3
    assert abs(led["bins"][0]["accuracy"] - 0.0) < 1e-12
    assert abs(led["bins"][1]["accuracy"] - 1.0) < 1e-12
    assert abs(led["len_correct"] - (5 + 30 + 100) / 3) < 1e-12
    assert abs(led["len_wrong"] - 20.0) < 1e-12
    assert abs(led["efficiency_ratio"] - 20.0 / (135 / 3)) < 1e-12
    assert led["best_bin"] == 1


def check_boundary_edges(solve):
    # 边界值归属（edges=[10,30,100] → K=2）：t=10 → 箱0（左闭），t=30 → 末箱
    correct = [True, False]
    tokens = [10, 30]
    led = _compare(solve, correct, tokens, [10, 30, 100])
    assert led["bins"][0]["count"] == 1
    assert led["bins"][1]["count"] == 1


def check_empty_bins(solve):
    # edges=[10,20,40] → K=2：箱0 [10,20) 空，箱1 [20,∞) 有 2 条
    correct = [True, False]
    tokens = [50, 60]
    led = _compare(solve, correct, tokens, [10, 20, 40])
    assert led["bins"][0]["count"] == 0
    assert led["bins"][0]["accuracy"] is None
    assert led["bins"][0]["mean_tokens"] is None
    assert led["bins"][1]["count"] == 2
    assert led["best_bin"] == 1


def check_all_correct(solve):
    led = _compare(solve, [True, True], [10, 30], [0, 20])
    assert led["accuracy"] == 1.0
    assert led["len_wrong"] is None
    assert led["efficiency_ratio"] is None
    # 箱1 并列准确率 1.0：并列取最小箱号（箱0 非空）
    assert led["best_bin"] == 0


def check_all_wrong(solve):
    led = _compare(solve, [False, False, False], [1, 2, 3], [0, 2])
    assert led["accuracy"] == 0.0
    assert led["len_correct"] is None
    assert led["efficiency_ratio"] is None
    assert led["best_bin"] == 0  # 准确率全 0：仍取首个非空箱（best_acc 初值 -1）


def check_tie_smallest_bin(solve):
    # 两箱准确率并列 0.5：取最小箱号
    correct = [True, False, True, False]
    tokens = [1, 2, 10, 20]
    led = _compare(solve, correct, tokens, [0, 5, 100])
    assert led["bins"][0]["accuracy"] == 0.5
    assert led["bins"][1]["accuracy"] == 0.5
    assert led["best_bin"] == 0


def check_single_bin(solve):
    # K=1（edges 两个值 → 1 箱）：全部样本进末箱
    correct = [True, False]
    tokens = [7, 8]
    led = _compare(solve, correct, tokens, [0, 100])
    assert led["bins"][0]["count"] == 2
    assert abs(led["bins"][0]["accuracy"] - 0.5) < 1e-12


def check_oracle_random(solve):
    import random
    rng = random.Random(77)
    for _ in range(150):
        n = rng.randint(1, 30)
        correct = [rng.random() < 0.5 for _ in range(n)]
        tokens = [rng.randint(0, 200) for _ in range(n)]
        e1 = rng.randint(0, 60)
        e2 = rng.randint(e1 + 1, 120)
        e3 = rng.randint(e2 + 1, 200)
        led = solve(list(correct), list(tokens), [e1, e2, e3])
        ref = _py(correct, tokens, [e1, e2, e3])
        assert led == ref


def check_invalid_inputs(solve):
    bad = [
        lambda: solve([True], [1, 2], [0, 10]),
        lambda: solve([1, 0], [1, 2], [0, 10]),
        lambda: solve([True, False], [1], [0, 10]),
        lambda: solve([True], [-1], [0, 10]),
        lambda: solve([True], [True], [0, 10]),
        lambda: solve([True], [1], [0]),
        lambda: solve([True], [1], []),
        lambda: solve([True], [1], [5, 5]),
        lambda: solve([True], [1], [10, 5]),
        lambda: solve([True], [1], [0, 5.5]),
        lambda: solve([], [], [0, 1]),
        lambda: solve([True], [1], [True, 5]),
    ]
    for fn in bad:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
