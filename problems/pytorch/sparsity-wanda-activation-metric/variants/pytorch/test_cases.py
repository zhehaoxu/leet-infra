import math

import torch

KEYS = {"metric", "mask", "pruned_weight", "sparsity"}


def _oracle(w, x, s, prune_n=0, prune_m=0):
    """纯 python 独立重算：逐列 RMS、逐行（或逐块）取最小。"""
    out_features, in_features = len(w), len(w[0])
    col_ms = [sum(row[j] ** 2 for row in x) / len(x) for j in range(in_features)]
    metric = [[abs(w[i][j]) * math.sqrt(col_ms[j]) for j in range(in_features)]
              for i in range(out_features)]
    mask = [[False] * in_features for _ in range(out_features)]
    if prune_n > 0:
        for start in range(0, in_features, prune_m):
            for i in range(out_features):
                order = sorted(range(start, start + prune_m),
                               key=lambda j: (metric[i][j], j))[:prune_n]
                for j in order:
                    mask[i][j] = True
    else:
        count = int(in_features * s)
        for i in range(out_features):
            order = sorted(range(in_features), key=lambda j: (metric[i][j], j))[:count]
            for j in order:
                mask[i][j] = True
    pruned = [[0.0 if mask[i][j] else w[i][j] for j in range(in_features)]
              for i in range(out_features)]
    ratio = sum(sum(row) for row in mask) / (out_features * in_features)
    return metric, mask, pruned, ratio


def _check(solve, dtype, w, x, s, prune_n=0, prune_m=0):
    wt = torch.tensor(w, dtype=dtype)
    xt = torch.tensor(x, dtype=dtype)
    w_snapshot, x_snapshot = wt.clone(), xt.clone()
    got = solve(wt, xt, s, prune_n, prune_m)
    want_metric, want_mask, want_pruned, want_ratio = _oracle(w, x, s, prune_n, prune_m)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    assert torch.equal(wt, w_snapshot) and torch.equal(xt, x_snapshot), "输入被就地修改"
    tol = 1e-5 if dtype == torch.float32 else 1e-12
    assert got["metric"].dtype == dtype and got["metric"].shape == wt.shape
    want_m = torch.tensor(want_metric, dtype=dtype)
    assert torch.allclose(got["metric"], want_m, atol=tol, rtol=tol), (
        f"metric 不对：{got['metric'].tolist()}")
    assert got["mask"].dtype == torch.bool and got["mask"].shape == wt.shape
    assert got["mask"].tolist() == want_mask, f"mask 不对：{got['mask'].tolist()}"
    assert got["pruned_weight"].dtype == dtype
    assert torch.allclose(got["pruned_weight"], torch.tensor(want_pruned, dtype=dtype),
                          atol=tol, rtol=tol), "pruned_weight 不对"
    assert isinstance(got["sparsity"], float), "sparsity 必须是 python float"
    assert abs(got["sparsity"] - want_ratio) < 1e-12
    return got


W = [[1.0, -5.0, 3.0, -2.0], [0.5, 9.0, -2.0, -8.0]]
X = [[2.0, 1.0, 0.0, 1.0], [0.0, 3.0, 0.0, 0.0]]


def check_example(solve):
    got = _check(solve, torch.float64, W, X, 0.5)
    # 列 RMS = [sqrt2, sqrt5, 0, sqrt0.5]；行 0 的 0/3 列并列 1.41421…，小下标 0 先剪
    assert got["mask"].tolist() == [[True, False, True, False],
                                    [True, False, True, False]]
    assert got["pruned_weight"].tolist() == [[0.0, -5.0, 0.0, -2.0],
                                             [0.0, 9.0, 0.0, -8.0]]
    assert abs(got["sparsity"] - 0.5) < 1e-12


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(7)
    for shape in [(3, 8), (4, 7), (2, 10)]:
        w = (torch.randn(*shape, dtype=torch.float64) * 2).tolist()
        x = torch.randn(5, shape[1], dtype=torch.float64).tolist()
        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            _check(solve, torch.float64, w, x, s)


def check_truncation_not_round(solve):
    w = [[float(i + 1) for i in range(7)]]
    x = [[1.0] * 7]
    got = _check(solve, torch.float64, w, x, 0.5)
    assert int(got["mask"].sum()) == 3, "int(7*0.5)=3 是截断，不是 round(3.5)=4"
    assert got["pruned_weight"].tolist() == [[0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0]]


def check_zero_activation_column(solve):
    w = [[100.0, 1.0, 2.0, 3.0]]
    x = [[0.0, 1.0, 1.0, 0.0]]
    got = _check(solve, torch.float64, w, x, 0.5)
    assert got["mask"][0, 0].item(), "激活恒为 0 的列 RMS=0，权重再大也要先剪"
    assert got["mask"][0, 3].item(), "激活为 0 的另一列同样先剪"


def check_tie_smaller_index(solve):
    w = [[1.0] * 6, [2.0] * 6]
    x = [[1.0] * 6]
    got = _check(solve, torch.float64, w, x, 0.5)
    assert got["mask"].tolist() == [[True, True, True, False, False, False]] * 2


def check_nm_2_4(solve):
    torch.manual_seed(11)
    w = torch.randn(3, 8, dtype=torch.float64).tolist()
    x = torch.randn(4, 8, dtype=torch.float64).tolist()
    got = _check(solve, torch.float64, w, x, 0.5, prune_n=2, prune_m=4)
    mask = got["mask"]
    for row in range(3):
        for start in (0, 4):
            assert int(mask[row, start:start + 4].sum()) == 2, "每个 4 列块必须恰好剪 2 个"
    got48 = _check(solve, torch.float64, w, x, 0.5, prune_n=4, prune_m=8)
    assert int(got48["mask"].sum()) == 3 * 4


def check_dtype_and_shapes(solve):
    torch.manual_seed(3)
    w = torch.randn(4, 6, dtype=torch.float32)
    x = torch.randn(9, 6, dtype=torch.float32)
    got = _check(solve, torch.float32, w.tolist(), x.tolist(), 0.5)
    assert got["metric"].dtype == torch.float32
    assert got["mask"].dtype == torch.bool
    assert got["pruned_weight"].dtype == torch.float32
    empty = _check(solve, torch.float64, w.double().tolist(), x.double().tolist(), 0.0)
    assert not bool(empty["mask"].any())


def check_value_errors(solve):
    w = torch.ones(2, 4)
    x = torch.ones(3, 4)
    cases = [
        (torch.ones(4), x, 0.5, 0, 0),                      # weight 非二维
        (w, torch.ones(3, 5), 0.5, 0, 0),                   # 列数不一致
        (w, torch.ones(0, 4), 0.5, 0, 0),                   # 没有 token
        (w, x.double(), 0.5, 0, 0),                         # dtype 不一致
        (w.to(torch.int64), x.to(torch.int64), 0.5, 0, 0),  # 非浮点
        (w, x, -0.1, 0, 0),                                 # sparsity 越界
        (w, x, 1.2, 0, 0),
        (w, x, True, 0, 0),                                 # bool 不是合法标量
        (w, x, 0.5, -1, 4),                                 # prune_n 为负
        (w, x, 0.5, 4, 4),                                  # n >= m
        (w, x, 0.5, 2, 3),                                  # in 不能被 m 整除
        (w, x, 0.6, 2, 4),                                  # N:M 下 sparsity 必须 0.5
    ]
    for index, args in enumerate(cases):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"case {index}: 期望 ValueError")
