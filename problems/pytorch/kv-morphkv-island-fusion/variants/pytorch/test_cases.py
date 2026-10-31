import math

import torch

KEYS = {"keys", "values", "important_idx", "island_idx", "island_counts"}


def _oracle(attn, keys, values, max_capacity, window_size, imp_budget, fuse_temperature, value_norms):
    """纯 Python 重算：softmax/T → 窗口 query 均值 × value 范数 → 重要位 / 岛屿 → 加权融合 → 布局。"""
    attn, keys, values = attn.tolist(), keys.tolist(), values.tolist()
    heads, n_query, n_past = len(attn), len(attn[0]), len(attn[0][0])
    win = window_size
    head_dim = len(keys[0][0])
    n_budget = int(max_capacity * imp_budget)
    n_fused = max_capacity - win - n_budget
    n_important = min(n_budget, n_past)
    n_islands = n_important + 1

    importance, important, unimportant = [], [], []
    for h in range(heads):
        mean = [0.0] * n_past
        for q in range(n_query):
            row = attn[h][q]
            top = max(row)
            ex = [math.exp((x - top) / fuse_temperature) for x in row]
            total = sum(ex)
            for p in range(n_past):
                mean[p] += ex[p] / total / n_query
        if value_norms is None:
            norms = [math.sqrt(sum(c * c for c in values[h][p])) for p in range(n_past)]
        else:
            norms = value_norms.tolist()[h]
        importance.append([mean[p] * norms[p] for p in range(n_past)])
        picked = sorted(sorted(range(n_past), key=lambda p: (-importance[h][p], p))[:n_important])
        important.append(picked)
        rest = [p for p in range(n_past) if p not in picked]
        unimportant.append(rest)

    out_k, out_v, island_ids, counts_out = [], [], [], []
    imp_out = []
    for h in range(heads):
        groups = {}
        for p in unimportant[h]:
            island = sum(1 for q in important[h] if q < p)
            groups.setdefault(island, []).append(p)
        fused_k = [None] * n_islands
        fused_v = [None] * n_islands
        mass = [0.0] * n_islands
        counts = [0] * n_islands
        for island, members in groups.items():
            mass[island] = sum(importance[h][p] for p in members)
            counts[island] = len(members)
            acc_k = [0.0] * head_dim
            acc_v = [0.0] * head_dim
            for p in members:
                w = importance[h][p] / max(mass[island], 1e-8)
                for d in range(head_dim):
                    acc_k[d] += w * keys[h][p][d]
                    acc_v[d] += w * values[h][p][d]
            fused_k[island], fused_v[island] = acc_k, acc_v
        scores = [mass[i] / counts[i] if counts[i] else float("-inf") for i in range(n_islands)]
        ranked = sorted(range(n_islands), key=lambda i: (-scores[i], i))[:min(n_fused, n_islands)]
        ranked.sort()
        pad = n_fused - len(ranked)
        fk = [fused_k[i] if fused_k[i] is not None else [0.0] * head_dim for i in ranked]
        fv = [fused_v[i] if fused_v[i] is not None else [0.0] * head_dim for i in ranked]
        fk += [[0.0] * head_dim] * pad
        fv += [[0.0] * head_dim] * pad
        island_ids.append(ranked + [-1] * pad)
        counts_out.append([counts[i] for i in ranked] + [0] * pad)
        imp_out.append(important[h])
        out_k.append(fk + [keys[h][p] for p in important[h]] + keys[h][n_past:])
        out_v.append(fv + [values[h][p] for p in important[h]] + values[h][n_past:])
    return {
        "keys": out_k,
        "values": out_v,
        "important_idx": imp_out,
        "island_idx": island_ids,
        "island_counts": counts_out,
    }


def _close(got, want, tol, path="out"):
    if isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), f"{path} 形状不符：{got} vs {want}"
        for i, (a, b) in enumerate(zip(got, want)):
            _close(a, b, tol, f"{path}[{i}]")
    else:
        assert abs(got - want) <= tol + tol * abs(want), f"{path} 数值不符：{got} vs {want}"


def _int_eq(got, want, path="out"):
    assert got == want, f"{path} 索引不符：{got} vs {want}"


def _verify(solve, attn, keys, values, max_capacity, window_size, imp_budget=0.5,
            fuse_temperature=1.0, value_norms=None):
    """统一入口：返回值结构 / 输入不被改写 / 与独立 oracle 全量对齐。"""
    snap = [t.clone() for t in (attn, keys, values)] + ([value_norms.clone()] if value_norms is not None else [])
    got = solve(attn, keys, values, max_capacity, window_size, imp_budget, fuse_temperature, value_norms)
    assert isinstance(got, dict) and set(got) == KEYS, f"返回键不对：{sorted(got)}"
    for before, after in zip(snap, [attn, keys, values] + ([value_norms] if value_norms is not None else [])):
        assert torch.equal(before, after), "输入被就地修改了"
    want = _oracle(attn, keys, values, max_capacity, window_size, imp_budget,
                   fuse_temperature, value_norms)
    tol = 3e-5 if keys.dtype == torch.float32 else 1e-9
    for name in ("keys", "values"):
        assert got[name].dtype == keys.dtype, f"{name} dtype {got[name].dtype} != {keys.dtype}"
        _close(got[name].tolist(), want[name], tol, name)
    for name in ("important_idx", "island_idx", "island_counts"):
        assert got[name].dtype == torch.int64, f"{name} 必须是 int64"
        _int_eq(got[name].tolist(), want[name], name)
    return got


def check_example(solve):
    """题面示例：H=1,W=2,P=6,D=2,max_capacity=5,imp_budget=0.5 → 1 个融合槽 + 2 单例 + 窗口。"""
    attn = torch.tensor([[[0.0, 0.7, 2.0, -1.0, 1.0, -2.0],
                          [0.0, 0.7, 2.0, -1.0, 1.0, -2.0]]])
    keys = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0],
                          [8.0, 9.0], [10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]])
    got = _verify(solve, attn, keys, 10 * keys, 5, 2, 0.5, 1.0, torch.ones(1, 6))
    assert got["important_idx"].tolist() == [[2, 4]], got["important_idx"].tolist()
    assert got["island_idx"].tolist() == [[0]] and got["island_counts"].tolist() == [[2]]
    assert torch.allclose(got["keys"][0, 0], torch.tensor([1.336376, 2.336375]), atol=1e-6), got["keys"][0, 0]
    assert torch.allclose(got["values"][0, 0], torch.tensor([13.363755, 23.363756]), atol=1e-5), got["values"][0, 0]
    assert got["keys"][0, 1:].tolist() == [[4.0, 5.0], [8.0, 9.0], [12.0, 13.0], [14.0, 15.0]]


check_example.example = True


def check_single_token_islands(solve):
    """重要位把 past 切碎成单 token 岛：每个融合槽必须原样等于该 token，槽数受预算限制。"""
    torch.manual_seed(11)
    keys = torch.randn(2, 9, 3, dtype=torch.float64)
    # 交替极端 logits → 重要位落在偶数位，奇数位各成一个单例岛
    row = torch.tensor([5.0, -5.0, 4.0, -4.0, 3.0, -3.0, 2.0, -2.0], dtype=torch.float64)
    attn = row.view(1, 1, 8).repeat(2, 1, 1)
    # max_capacity=9, window=1, imp_budget=0.5 → 4 个重要位 + 4 个融合槽
    got = _verify(solve, attn, keys, keys + 1.0, 9, 1, 0.5, 1.0, torch.ones(2, 8, dtype=torch.float64))
    assert got["important_idx"].tolist() == [[0, 2, 4, 6]] * 2, got["important_idx"].tolist()
    assert got["island_counts"].tolist() == [[1, 1, 1, 1]] * 2, got["island_counts"].tolist()
    sel = got["island_idx"][0]
    for slot, island in enumerate(sel.tolist()):
        token = {1: 1, 2: 3, 3: 5, 4: 7}[island]
        assert torch.equal(got["keys"][0, slot], keys[0, token]), (slot, island)


def check_empty_and_padded_islands(solve):
    """num_past 全被单例吃掉：所有融合槽补零、计数 0、岛号 -1；总长仍由预算决定。"""
    keys = torch.randn(1, 4, 2)
    attn = torch.randn(1, 1, 2)
    got = _verify(solve, attn, keys, keys, 6, 2, 0.4, 1.0)  # n_budget=2=num_past, n_fused=2
    assert got["island_idx"].tolist() == [[0, 1]] and got["island_counts"].tolist() == [[0, 0]]
    assert torch.count_nonzero(got["keys"][:, :2]) == 0 and got["keys"].shape == (1, 6, 2)
    assert got["important_idx"].shape == (1, 2), "重要位应全部保留"


def check_zero_fused_budget(solve):
    """预算刚好等于 window + 重要位：n_fused=0，布局退化为 [重要位 | 窗口]。"""
    keys = torch.randn(2, 6, 2, dtype=torch.float64)
    attn = torch.randn(2, 2, 4, dtype=torch.float64)
    got = _verify(solve, attn, keys, keys, 4, 2, 0.5)  # n_budget=2, n_fused=0
    assert got["island_idx"].shape == (2, 0) and got["keys"].shape == (2, 4, 2)
    assert torch.equal(got["keys"][:, 2:], keys[:, 4:]), "窗口必须原样保留"


def check_tie_break(solve):
    """importance 全相等：重要位取更小下标；空岛分数 -inf，选岛同分取更小岛号且升序排列。"""
    keys = torch.arange(1, 13, dtype=torch.float64).view(1, 6, 2)
    attn = torch.zeros(1, 1, 4, dtype=torch.float64)
    got = _verify(solve, attn, keys, keys, 6, 2, 0.4, 1.0,
                  torch.ones(1, 4, dtype=torch.float64))  # n_budget=2, n_fused=2
    assert got["important_idx"].tolist() == [[0, 1]], got["important_idx"].tolist()
    assert got["island_idx"].tolist() == [[0, 2]], "空岛 0 与唯一非空岛 2（两个重要位之后）"
    assert got["island_counts"].tolist() == [[0, 2]]
    assert torch.equal(got["keys"][0, 0], torch.zeros(2, dtype=torch.float64)), "空岛槽位补零"
    assert torch.allclose(got["keys"][0, 1], keys[0, 2:4].mean(0)), "等权平均等价于算术平均"


def check_dtype_and_layout(solve):
    """float64 进 float64 出；窗口块逐位相等；总长 = 融合槽 + 重要位 + 窗口。"""
    torch.manual_seed(21)
    attn = torch.randn(3, 2, 7, dtype=torch.float64)
    keys = torch.randn(3, 10, 4, dtype=torch.float64)
    got = _verify(solve, attn, keys, keys - 1.0, 8, 3, 0.25, 2.0)
    assert got["keys"].dtype == torch.float64 and got["keys"].shape == (3, 8, 4)
    assert torch.equal(got["keys"][:, -3:], keys[:, 7:]) and torch.equal(got["values"][:, -3:], (keys - 1.0)[:, 7:])
    assert got["island_idx"].shape == (3, 3) and got["important_idx"].shape == (3, 2)


def check_inputs_not_modified(solve):
    """调用后 attn/keys/values/value_norms 全部逐位不变（含 float32 与显式范数）。"""
    torch.manual_seed(22)
    attn, keys = torch.randn(2, 2, 5), torch.randn(2, 7, 3)
    vals, norms = torch.randn(2, 7, 3), torch.rand(2, 5) + 0.5
    snap = [t.clone() for t in (attn, keys, vals, norms)]
    solve(attn, keys, vals, 6, 2, 0.5, 1.0, norms)
    for before, after in zip(snap, (attn, keys, vals, norms)):
        assert torch.equal(before, after), "输入被就地修改了"


def check_random_matches_oracle(solve):
    """固定种子随机对拍：多种 H/Q/P/W/D/预算，value_norms 省略与显式两条路径。"""
    torch.manual_seed(2024)
    shapes = [(1, 1, 6, 0, 2), (2, 3, 9, 2, 1), (3, 2, 12, 3, 4), (1, 4, 5, 1, 1)]
    for idx, (heads, n_query, n_past, win, head_dim) in enumerate(shapes):
        for ib in (0.0, 0.25, 0.5, 1.0):
            caps = [mc for mc in range(win + 1, win + n_past + 6)
                    if mc - win - int(mc * ib) >= 0]
            for max_capacity in caps[:2]:
                for explicit in (False, True):
                    attn = torch.randn(heads, n_query, n_past, dtype=torch.float64) * 2
                    keys = torch.randn(heads, n_past + win, head_dim, dtype=torch.float64)
                    vals = torch.randn(heads, n_past + win, head_dim, dtype=torch.float64)
                    norms = (torch.rand(heads, n_past, dtype=torch.float64) + 0.25
                             if explicit else None)
                    _verify(solve, attn, keys, vals, max_capacity, win, ib, 0.7 + 0.1 * idx, norms)
    _verify(solve, torch.randn(2, 2, 6), torch.randn(2, 6, 2), torch.randn(2, 6, 2), 5, 0, 0.5)
    _verify(solve, torch.randn(2, 2, 6), torch.randn(2, 9, 2), torch.randn(2, 9, 2), 6, 3, 0.5)


def check_value_errors(solve):
    """非法预算 / 形状 / dtype / 范数都要抛 ValueError。"""
    attn, keys = torch.randn(2, 2, 4), torch.randn(2, 6, 3)
    vals = torch.randn(2, 6, 3)
    bad = [
        (attn, keys, vals, 2, 2, 0.5, 1.0, None),          # 预算 < window + 重要位
        (attn, keys, vals, 5, 3, 0.5, 1.0, None),          # keys 长度 != P + window
        (attn, keys, vals, 5, 2, 1.5, 1.0, None),          # imp_budget 越界
        (attn, keys, vals, 5, 2, 0.5, 0.0, None),          # 温度非正
        (attn, keys, vals[:, :5], 5, 2, 0.5, 1.0, None),   # value 长度不符
        (attn, keys.double(), vals.double(), 5, 2, 0.5, 1.0, None),  # dtype 混用
        (attn, keys, vals, 5, 2, 0.5, 1.0, torch.zeros(2, 4)),       # 范数为 0
        (attn, keys, vals, 5, 2, 0.5, 1.0, torch.ones(2, 3)),        # 范数形状不符
        (attn[0], keys, vals, 5, 2, 0.5, 1.0, None),       # attn 维度不对
        (attn, keys, vals, 5, 2.0, 0.5, 1.0, None),        # window_size 非 int
    ]
    for idx, args in enumerate(bad):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"第 {idx} 个非法输入应抛 ValueError")
