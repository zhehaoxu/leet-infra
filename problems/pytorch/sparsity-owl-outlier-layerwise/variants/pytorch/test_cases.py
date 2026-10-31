import math

import torch

KEYS = {"outlier_ratio", "density", "layer_sparsity", "metric", "mask", "pruned_weight"}


def _oracle(weights, act_ms, s, lamda=0.08, hyper_m=3.0):
    """纯 python 独立重算：逐层拼接 metric 求离群比 → min-max 归一 → 逐行取最小。"""
    metrics = []
    for layer_w, layer_a in zip(weights, act_ms):
        layer = []
        for w, a in zip(layer_w, layer_a):
            rms = [math.sqrt(v) for v in a]
            layer.append([[abs(w[i][j]) * rms[j] for j in range(len(w[0]))]
                          for i in range(len(w))])
        metrics.append(layer)

    ratios = []
    for layer in metrics:
        flat = [v for m in layer for row in m for v in row]
        threshold = sum(flat) / len(flat) * hyper_m
        ratios.append(sum(1 for v in flat if v > threshold) / len(flat) * 100.0)

    lo, hi = min(ratios), max(ratios)
    norm = [(r - lo) * (2.0 * lamda / (hi - lo)) for r in ratios]
    center = sum(norm) / len(norm)
    density = [v - center + (1.0 - s) for v in norm]
    sparsity = [0.01 if 1.0 - d <= 0.0 else 1.0 - d for d in density]

    masks, pruned = [], []
    for layer, layer_w, layer_sparsity in zip(metrics, weights, sparsity):
        layer_mask, layer_pruned = [], []
        for m, w in zip(layer, layer_w):
            width = len(m[0])
            count = int(width * layer_sparsity)
            mask = [[False] * width for _ in m]
            for i, row in enumerate(m):
                for j in sorted(range(width), key=lambda j: (row[j], j))[:count]:
                    mask[i][j] = True
            layer_mask.append(mask)
            layer_pruned.append([[0.0 if mask[i][j] else w[i][j] for j in range(width)]
                                 for i in range(len(m))])
        masks.append(layer_mask)
        pruned.append(layer_pruned)
    return metrics, ratios, density, sparsity, masks, pruned


def _compare(solve, dtype, weights, act_ms, s, lamda=0.08, hyper_m=3.0):
    wt = [[torch.tensor(w, dtype=dtype) for w in layer] for layer in weights]
    at = [[torch.tensor(a, dtype=dtype) for a in layer] for layer in act_ms]
    snap_w = [[w.clone() for w in layer] for layer in wt]
    snap_a = [[a.clone() for a in layer] for layer in at]
    got = solve(wt, at, s, lamda, hyper_m)
    want_metric, want_ratio, want_density, want_sparsity, want_mask, want_pruned = \
        _oracle(weights, act_ms, s, lamda, hyper_m)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    for layer_w, layer_a, sw, sa in zip(wt, at, snap_w, snap_a):
        for w, a, x, y in zip(layer_w, layer_a, sw, sa):
            assert torch.equal(w, x) and torch.equal(a, y), "输入被就地修改"
    tol = 1e-5 if dtype == torch.float32 else 1e-9
    assert got["outlier_ratio"].dtype == torch.float64
    assert torch.allclose(got["outlier_ratio"], torch.tensor(want_ratio, dtype=torch.float64),
                          atol=tol, rtol=tol), f"outlier_ratio 不对：{got['outlier_ratio'].tolist()}"
    for key, want in (("density", want_density), ("layer_sparsity", want_sparsity)):
        assert got[key].dtype == torch.float64
        assert torch.allclose(got[key], torch.tensor(want, dtype=torch.float64),
                              atol=max(tol, 1e-9), rtol=tol), f"{key} 不对：{got[key].tolist()}"
    assert len(got["metric"]) == len(weights)
    for index in range(len(weights)):
        assert len(got["metric"][index]) == len(weights[index])
        for k, w in enumerate(wt[index]):
            metric = got["metric"][index][k]
            mask = got["mask"][index][k]
            pruned = got["pruned_weight"][index][k]
            assert metric.dtype == dtype and metric.shape == w.shape
            assert torch.allclose(metric, torch.tensor(want_metric[index][k], dtype=dtype),
                                  atol=tol, rtol=tol), "metric 不对"
            assert mask.dtype == torch.bool and mask.tolist() == want_mask[index][k]
            assert pruned.dtype == dtype
            assert torch.allclose(pruned, torch.tensor(want_pruned[index][k], dtype=dtype),
                                  atol=tol, rtol=tol), "pruned_weight 不对"
    return got


def _ones(*shape):
    return [[1.0] * shape[1] for _ in range(shape[0])]


LAYERS = [
    [[[1.0] * 8]],
    [[[1.0] * 7 + [40.0]]],
    [[[1.0] * 6 + [40.0, 40.0]]],
]
ACTS = [[[1.0] * 8], [[1.0] * 8], [[1.0] * 8]]


def check_example(solve):
    got = _compare(solve, torch.float64, LAYERS, ACTS, 0.5)
    assert got["outlier_ratio"].tolist() == [0.0, 12.5, 25.0]
    assert all(abs(a - b) < 1e-10 for a, b in
               zip(got["density"].tolist(), [0.42, 0.5, 0.58]))
    assert all(abs(a - b) < 1e-10 for a, b in
               zip(got["layer_sparsity"].tolist(), [0.58, 0.5, 0.42]))
    assert got["mask"][0][0].tolist() == [[True, True, True, True, False, False, False, False]]
    assert got["mask"][1][0].tolist() == [[True, True, True, True, False, False, False, False]]
    assert got["mask"][2][0].tolist() == [[True, True, True, False, False, False, False, False]]


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(21)
    # 逐层注入 0/1/2 个大值，保证离群比 [0, 12.5, 25] 有区分度、不触发除零
    for width in [6, 8, 10]:
        weights, acts = [], []
        for count in range(3):
            matrix = (torch.rand(2, width, dtype=torch.float64) * 1.5 + 0.5).tolist()
            for j in range(width - count, width):
                matrix[0][j] = 100.0
            weights.append([matrix])
            acts.append([(torch.rand(width, dtype=torch.float64) + 0.5).tolist()])
        for s in [0.3, 0.5, 0.7]:
            _compare(solve, torch.float64, weights, acts, s)


def check_mean_budget(solve):
    got = _compare(solve, torch.float64, LAYERS, ACTS, 0.5)
    assert abs(float(got["layer_sparsity"].mean()) - 0.5) < 1e-10, \
        "逐层稀疏率的均值必须回到 target"


def check_multi_linear_layer(solve):
    layers = [
        [[[1.0] * 6, [1.0] * 6], [[1.0] * 7 + [50.0]]],
        [[[1.0] * 6, [1.0] * 6], [[1.0] * 6, [1.0] * 6]],
    ]
    acts = [[[1.0] * 6, [1.0] * 8], [[1.0] * 6, [1.0] * 6]]
    got = _compare(solve, torch.float64, layers, acts, 0.5)
    ratio = float(got["layer_sparsity"][0])
    for linear in got["mask"][0]:
        width = linear.shape[1]
        assert int(linear[0].sum()) == int(width * ratio), "同一层的多个 Linear 共用层稀疏率"


def check_strict_threshold(solve):
    # [1,1,1,9] 的均值是 3、阈值 9；元素 9 不满足严格大于，离群比为 0
    layers = [[[[1.0, 1.0, 1.0, 9.0]]], [[[1.0, 1.0, 1.0, 10.0]]]]
    acts = [[[1.0] * 4], [[1.0] * 4]]
    got = _compare(solve, torch.float64, layers, acts, 0.5)
    assert got["outlier_ratio"].tolist() == [0.0, 25.0]


def check_floor_rule(solve):
    got = _compare(solve, torch.float64, LAYERS, ACTS, 0.0)
    assert got["layer_sparsity"].tolist()[2] == 0.01, "density>1 时 layer_sparsity<=0 要落到 0.01"
    assert int(got["mask"][2][0].sum()) == 0, "int(8*0.01)=0，不剪"


def check_dtype_and_errors(solve):
    w32 = [[torch.tensor([[1.0] * 6], dtype=torch.float32)],
           [torch.tensor([[1.0] * 5 + [40.0]], dtype=torch.float32)]]
    a32 = [[torch.ones(6, dtype=torch.float32)],
           [torch.ones(6, dtype=torch.float32)]]
    got = solve(w32, a32, 0.5)
    assert got["metric"][0][0].dtype == torch.float32
    assert got["pruned_weight"][0][0].dtype == torch.float32

    base_w = [[torch.ones(1, 4)], [torch.ones(1, 4)]]
    base_a = [[torch.ones(4)], [torch.ones(4)]]
    cases = [
        (torch.ones(1, 4), base_a, 0.5, 0.08, 3.0),                  # weights 不是 list
        ([], base_a, 0.5, 0.08, 3.0),                                # 层数 < 2
        ([base_w[0]], [base_a[0]], 0.5, 0.08, 3.0),                  # 只有一层
        ([[]], [[]], 0.5, 0.08, 3.0),                                # 空层
        (base_w, [base_a[0]], 0.5, 0.08, 3.0),                       # 激活层数不一致
        ([[torch.ones(2, 4)], [torch.ones(2, 4)]],
         [[torch.ones(3)], [torch.ones(4)]], 0.5, 0.08, 3.0),        # 激活长度不一致
        ([[torch.ones(4)], [torch.ones(4)]], base_a, 0.5, 0.08, 3.0),  # 权重非二维
        ([[torch.ones(1, 4, dtype=torch.float32)], [torch.ones(1, 4)]],
         base_a, 0.5, 0.08, 3.0),                                    # dtype 不一致
        (base_w, base_a, -0.1, 0.08, 3.0),                           # sparsity 越界
        (base_w, base_a, 1.1, 0.08, 3.0),
        (base_w, base_a, True, 0.08, 3.0),                           # bool
        (base_w, base_a, 0.5, 0.0, 3.0),                             # lamda <= 0
        (base_w, base_a, 0.5, 0.08, 0.0),                            # hyper_m <= 0
        (base_w, base_a, 0.5, 0.08, 3.0),                            # 各层离群比相同
    ]
    for index, args in enumerate(cases):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"case {index}: 期望 ValueError")
