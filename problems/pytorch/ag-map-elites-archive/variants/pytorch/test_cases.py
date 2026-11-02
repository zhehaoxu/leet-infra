import math

import torch


def _oracle(desc, scores, bins, ranges):
    dim = len(ranges)
    b = bins if isinstance(bins, list) else [bins] * dim
    total = 1
    for x in b:
        total *= x
    score = [float("-inf")] * total
    index = [-1] * total
    cells = []
    for i, d in enumerate(desc):
        ok = all(ranges[k][0] <= d[k] <= ranges[k][1] for k in range(dim))
        if not ok:
            cells.append(-1)
            continue
        flat = 0
        for k in range(dim):
            slot = int(math.floor((d[k] - ranges[k][0]) / (ranges[k][1] - ranges[k][0]) * b[k]))
            slot = min(slot, b[k] - 1)
            flat = flat * b[k] + slot
        cells.append(flat)
        if scores[i] > score[flat]:
            score[flat] = scores[i]
            index[flat] = i
    return cells, score, index


def check_basic_routing(solve):
    d = torch.tensor([[0.0, 0.0], [0.9, 0.9], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float64)
    s = torch.tensor([1.0, 2.0, 3.0, 0.5], dtype=torch.float64)
    out = solve(d, s, 2, [(0.0, 1.0), (0.0, 1.0)])
    assert out["cells"].tolist() == [0, 3, 3, 3], out["cells"]
    assert out["accepted"].tolist() == [True, True, True, False], out["accepted"]
    assert out["elite_score"].tolist() == [1.0, float("-inf"), float("-inf"), 3.0], out["elite_score"]
    assert out["elite_index"].tolist() == [0, -1, -1, 2], out["elite_index"]
    assert out["occupancy"] == 2
    assert out["best_score"] == 3.0 and out["best_index"] == 2


def check_boundary_bins(solve):
    # lo 落首格、hi 落末格（不丢），恰好落在分隔点上的值进右侧格
    d = torch.tensor([[0.0], [1.0], [0.5], [0.4999999]], dtype=torch.float64)
    s = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    out = solve(d, s, 2, [(0.0, 1.0)])
    assert out["cells"].tolist() == [0, 1, 1, 0], out["cells"]
    assert out["elite_score"].tolist() == [4.0, 3.0], out["elite_score"]


def check_out_of_range_discard(solve):
    d = torch.tensor([[-0.001], [1.001], [float("inf")]], dtype=torch.float64)
    s = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float64)
    out = solve(d, s, 3, [(0.0, 1.0)])
    assert out["cells"].tolist() == [-1, -1, -1], out["cells"]
    assert out["accepted"].tolist() == [False, False, False]
    assert out["occupancy"] == 0 and out["best_index"] == -1


def check_strict_improvement(solve):
    # 同分不替换：先到者留在档案里
    d = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float64)
    s = torch.tensor([2.0, 2.0, 1.9], dtype=torch.float64)
    out = solve(d, s, 1, [(0.0, 1.0)])
    assert out["accepted"].tolist() == [True, False, False], out["accepted"]
    assert out["elite_index"].tolist() == [0], out["elite_index"]
    assert out["best_index"] == 0


def check_multi_dim_flat_order(solve):
    # 3 维各 2/3/2 格：flat = ((i0*3)+i1)*2+i2（行优先）
    d = torch.tensor([[0.9, 0.9, 0.9]], dtype=torch.float64)
    out = solve(d, torch.tensor([1.0], dtype=torch.float64), [2, 3, 2],
                [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    assert out["cells"].tolist() == [11], out["cells"]
    assert out["elite_score"].numel() == 12
    # 中间格
    out2 = solve(torch.tensor([[0.1, 0.4, 0.6]], dtype=torch.float64),
                 torch.tensor([1.0], dtype=torch.float64), [2, 3, 2],
                 [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    assert out2["cells"].tolist() == [3], out2["cells"]


def check_scalar_bins(solve):
    # 标量 bins 表示每维同样多格
    d = torch.tensor([[0.2, 0.7], [0.8, 0.1]], dtype=torch.float64)
    out = solve(d, torch.tensor([1.0, 2.0], dtype=torch.float64), 4,
                [(0.0, 1.0), (0.0, 1.0)])
    assert out["cells"].tolist() == [2, 12], out["cells"]
    assert out["elite_score"].numel() == 16


def check_empty_input(solve):
    out = solve(torch.zeros(0, 2, dtype=torch.float64), torch.zeros(0, dtype=torch.float64),
                2, [(0.0, 1.0), (0.0, 1.0)])
    assert out["cells"].numel() == 0 and out["accepted"].numel() == 0
    assert out["occupancy"] == 0 and out["best_index"] == -1
    assert out["best_score"] == float("-inf")


def check_best_tie_keeps_earliest(solve):
    # 两格并列最高分：best_index 取先到的那个候选下标
    d = torch.tensor([[0.1], [0.9]], dtype=torch.float64)
    s = torch.tensor([2.0, 2.0], dtype=torch.float64)
    out = solve(d, s, 2, [(0.0, 1.0)])
    assert out["best_score"] == 2.0 and out["best_index"] == 0, (out["best_score"], out["best_index"])


def check_all_same_cell(solve):
    d = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float64)
    s = torch.tensor([1.0, 4.0, 2.0, 3.0], dtype=torch.float64)
    out = solve(d, s, 1, [(0.0, 1.0)])
    assert out["accepted"].tolist() == [True, True, False, False], out["accepted"]
    assert out["elite_score"].tolist() == [4.0] and out["elite_index"].tolist() == [1]
    assert out["occupancy"] == 1


def check_validation(solve):
    d = torch.zeros(2, 2, dtype=torch.float64)
    s = torch.zeros(2, dtype=torch.float64)
    bad = [
        dict(bins=0),
        dict(bins=True),
        dict(bins=[2]),
        dict(bins=[2, 2, 2]),
        dict(bins=[2, 0]),
        dict(ranges=[(0.0, 1.0)]),
        dict(ranges=[(1.0, 0.0), (0.0, 1.0)]),
        dict(ranges=[(0.0, 1.0), (0.0, 1.0, 2.0)]),
        dict(ranges=[(0.5, 0.5), (0.0, 1.0)]),
    ]
    for kw in bad:
        args = dict(bins=2, ranges=[(0.0, 1.0), (0.0, 1.0)])
        args.update(kw)
        try:
            solve(d, s, **args)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝 {kw}")
    try:
        solve(torch.zeros(3, 2, dtype=torch.float64), s, 2, [(0.0, 1.0), (0.0, 1.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("scores 长度不符应被拒绝")
    try:
        solve(torch.zeros(2, dtype=torch.float64), s, 2, [(0.0, 1.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("descriptors 必须是 2 维")
    try:
        solve(d, torch.tensor([0.0, float("nan")], dtype=torch.float64), 2,
              [(0.0, 1.0), (0.0, 1.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("NaN 分数应被拒绝")


def check_no_alias(solve):
    d = torch.tensor([[0.5]], dtype=torch.float64)
    s = torch.tensor([1.0], dtype=torch.float64)
    out = solve(d, s, 2, [(0.0, 1.0)])
    assert d.tolist() == [[0.5]] and s.tolist() == [1.0], "入参不应被改"
    out["elite_score"][0] = 99.0
    assert out["best_score"] == 1.0, "best_score 应是标量快照"


def check_against_oracle(solve):
    torch.manual_seed(20261107)
    for trial in range(40):
        dim = int(torch.randint(1, 4, (1,)).item())
        n = int(torch.randint(0, 12, (1,)).item())
        bins = [int(torch.randint(1, 4, (1,)).item()) for _ in range(dim)]
        ranges = []
        for _ in range(dim):
            a = float(torch.randint(-3, 3, (1,)).item())
            ranges.append((a, a + float(torch.randint(1, 4, (1,)).item())))
        desc = []
        for _ in range(n):
            row = []
            for k in range(dim):
                lo, hi = ranges[k]
                span = hi - lo
                row.append(lo + span * float(torch.rand(1).item()) * 1.2 - 0.1 * span)
            desc.append(row)
        scores = [float(torch.randint(0, 4, (1,)).item()) for _ in range(n)]
        dt = torch.tensor(desc, dtype=torch.float64).reshape(n, dim)
        st = torch.tensor(scores, dtype=torch.float64)
        out = solve(dt, st, bins, ranges)
        cells, score, index = _oracle(desc, scores, bins, ranges)
        assert out["cells"].tolist() == cells, (trial, out["cells"].tolist(), cells)
        assert out["elite_index"].tolist() == index, (trial, out["elite_index"].tolist(), index)
        got = out["elite_score"].tolist()
        for a, b in zip(got, score):
            assert (a == b) or (a == a and b == b and abs(a - b) < 1e-12), (a, b)
        assert out["occupancy"] == sum(1 for x in index if x >= 0)


check_basic_routing.example = True
check_strict_improvement.example = True
