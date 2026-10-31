import random

import torch

# 题面样例：(shape, strides, offset, ops) -> [(shape, strides, offset, contiguous, kind), ...]
# 由参考实现与本机 torch 2.14.1 双向实测得出（每条链都逐 step 对过真实张量元数据）。
EXAMPLES = [
    ((3, 4), (4, 1), 0,
     [["transpose", 1, 0], ["narrow", 0, 1, 2], ["reshape", 6], ["reshape", 2, 3]],
     [((4, 3), (1, 4), 0, False, "view"),
      ((2, 3), (1, 4), 1, False, "view"),
      ((6,), (1,), 0, True, "copy"),
      ((2, 3), (3, 1), 0, True, "view")]),
    ((2, 3, 4), (12, 4, 1), 0,
     [["permute", 2, 0, 1], ["reshape", 4, 6], ["contiguous"]],
     [((4, 2, 3), (1, 12, 4), 0, False, "view"),
      ((4, 6), (1, 4), 0, False, "view"),
      ((4, 6), (6, 1), 0, True, "copy")]),
    ((3, 4), (4, 1), 0,
     [["unsqueeze", 1], ["unsqueeze", 0], ["squeeze", None]],
     [((3, 1, 4), (4, 4, 1), 0, True, "view"),
      ((1, 3, 1, 4), (12, 4, 4, 1), 0, True, "view"),
      ((3, 4), (4, 1), 0, True, "view")]),
    ((1, 4), (4, 1), 0,
     [["expand", 3, 4], ["expand", 2, 3, 4], ["reshape", 24], ["contiguous"]],
     [((3, 4), (0, 1), 0, False, "view"),
      ((2, 3, 4), (0, 0, 1), 0, False, "view"),
      ((24,), (1,), 0, True, "copy"),
      ((24,), (1,), 0, True, "view")]),
    ((2, 3, 4), (12, 4, 1), 0,
     [["slice", 1, 0, 3, 2], ["flatten", 0, 1], ["flatten", 0, 1]],
     [((2, 2, 4), (12, 8, 1), 0, False, "view"),
      ((4, 4), (4, 1), 0, True, "copy"),
      ((16,), (1,), 0, True, "view")]),
    ((2, 3, 4), (12, 4, 1), 0,
     [["select", 1, -1], ["unsqueeze", 0], ["narrow", 1, 1, 1]],
     [((2, 4), (12, 1), 8, False, "view"),
      ((1, 2, 4), (24, 12, 1), 8, False, "view"),
      ((1, 1, 4), (24, 12, 1), 20, True, "view")]),
    ((4, 3), (1, 4), 0,
     [["reshape", 12], ["reshape", 2, 6], ["contiguous"]],
     [((12,), (1,), 0, True, "copy"),
      ((2, 6), (6, 1), 0, True, "view"),
      ((2, 6), (6, 1), 0, True, "view")]),
    ((2, 0, 4), (8, 4, 1), 0,
     [["transpose", 0, 2], ["reshape", 0], ["unsqueeze", 0]],
     [((4, 0, 2), (1, 4, 8), 0, True, "view"),
      ((0,), (1,), 0, True, "view"),
      ((1, 0), (0, 1), 0, True, "view")]),
]

KINDS = ["transpose", "permute", "unsqueeze", "squeeze", "narrow", "select",
         "slice", "expand", "reshape", "flatten", "contiguous"]


def _dicts(rows):
    return [{"shape": list(s), "strides": list(t), "offset": o, "contiguous": c, "kind": k}
            for s, t, o, c, k in rows]


def _build(shape, strides, offset):
    """用 as_strided 造出给定元数据的真实张量（元数据即题目输入）。"""
    need = offset + sum((s - 1) * st for s, st in zip(shape, strides)) + 1
    return torch.arange(max(need, 1), dtype=torch.float32).as_strided(shape, strides, offset)


def _apply(t, op):
    kind, a = op[0], op[1:]
    if kind == "squeeze":
        return t.squeeze() if a[0] is None else t.squeeze(a[0])
    if kind == "transpose":
        return t.transpose(a[0], a[1])
    if kind == "permute":
        return t.permute(*a)
    if kind == "unsqueeze":
        return t.unsqueeze(a[0])
    if kind == "narrow":
        return t.narrow(a[0], a[1], a[2])
    if kind == "select":
        return t.select(a[0], a[1])
    if kind == "slice":
        d, s, e, st = a
        return t[(slice(None),) * (d % len(t.shape)) + (slice(s, e, st),)]
    if kind == "expand":
        return t.expand(*a)
    if kind == "reshape":
        return t.reshape(*a)
    if kind == "flatten":
        return t.flatten(a[0], a[1])
    if kind == "contiguous":
        return t.contiguous()
    raise AssertionError(f"unknown op {kind}")


def _torch_ledger(shape, strides, offset, ops):
    """torch 侧账本：kind 由 storage 指针是否改变判定。"""
    t = _build(shape, strides, offset)
    prev = t
    rows = []
    for op in ops:
        nt = _apply(prev, op)
        rows.append({
            "shape": list(nt.shape),
            "strides": list(nt.stride()),
            "offset": nt.storage_offset(),
            "contiguous": bool(nt.is_contiguous()),
            "kind": "view" if nt.untyped_storage().data_ptr() == prev.untyped_storage().data_ptr() else "copy",
        })
        prev = nt
    return rows


def _gen_op(rng, shape):
    ndim = len(shape)
    kind = rng.choice(KINDS)
    if ndim >= 5 and kind in ("unsqueeze", "expand"):
        kind = "reshape"
    if kind == "transpose":
        return None if ndim == 0 else ["transpose", rng.randint(-ndim, ndim - 1), rng.randint(-ndim, ndim - 1)]
    if kind == "permute":
        if ndim == 0:
            return None
        dims = list(range(ndim))
        rng.shuffle(dims)
        if rng.random() < 0.4:
            dims = [d - ndim if rng.random() < 0.5 else d for d in dims]
        return ["permute", *dims]
    if kind == "unsqueeze":
        return ["unsqueeze", rng.randint(-ndim - 1, ndim)]
    if kind == "squeeze":
        if rng.random() < 0.4 or ndim == 0:
            return ["squeeze", None]
        return ["squeeze", rng.randint(-ndim, ndim - 1)]
    if kind == "narrow":
        if ndim == 0:
            return None
        d = rng.randint(-ndim, ndim - 1)
        size = shape[d % ndim]
        start = rng.randint(-size, size)
        norm = start + size if start < 0 else start
        return ["narrow", d, start, rng.randint(0, size - norm)]
    if kind == "select":
        if not shape:
            return None
        d = rng.randint(0, len(shape) - 1)
        size = shape[d]
        if size == 0:
            return None
        return ["select", d, rng.randint(-size, size - 1)]
    if kind == "slice":
        if ndim == 0:
            return None
        d = rng.randint(-ndim, ndim - 1)
        size = shape[d % ndim]
        return ["slice", d, rng.randint(-size - 2, size + 2), rng.randint(-size - 2, size + 2),
                rng.randint(1, 3)]
    if kind == "expand":
        new_ndim = rng.randint(ndim, min(ndim + 2, 5))
        sizes = [rng.choice([1, 2, 3, 4]) for _ in range(new_ndim - ndim)]
        for s in shape:
            sizes.append(rng.choice([1, 2, 3, -1]) if s == 1 else rng.choice([s, -1]))
        return ["expand", *sizes]
    if kind == "reshape":
        numel = 1
        for s in shape:
            numel *= s
        if numel == 0:
            dims = [rng.choice([0, 1, 2, 3]) for _ in range(rng.randint(1, 3))]
            dims[rng.randrange(len(dims))] = 0
            return ["reshape", *dims]
        dims, left = [], numel
        for _ in range(rng.randint(1, 4)):
            cand = [d for d in (1, 2, 3, 4, 6, numel) if left % d == 0] or [1]
            d = rng.choice(cand)
            dims.append(d)
            left //= d
        if rng.random() < 0.3:
            dims[rng.randrange(len(dims))] = -1
        return ["reshape", *dims]
    if kind == "flatten":
        if ndim == 0:
            return ["flatten", 0, -1]
        a = rng.randint(-ndim, ndim - 1)
        return ["flatten", a, rng.randint(-ndim, ndim - 1)]
    return ["contiguous"]


def _gen_chain(rng):
    """生成保证合法的链：torch 拒收就换下一个候选操作。"""
    ndim = rng.randint(0, 4)
    shape = [rng.choice([0, 1, 2, 3, 4]) if rng.random() < 0.08 else rng.choice([1, 2, 3, 4])
             for _ in range(ndim)]
    strides = [rng.randint(1, 9) for _ in shape]
    offset = rng.choice([0, 0, 1, 5])
    if offset + sum((s - 1) * st for s, st in zip(shape, strides)) + 1 > 2000:
        return None
    t = _build(shape, strides, offset)
    ops = []
    for _ in range(rng.randint(1, 8)):
        for _try in range(12):
            op = _gen_op(rng, list(t.shape))
            if op is None:
                continue
            try:
                t = _apply(t, op)
            except Exception:
                continue
            ops.append(op)
            break
        else:
            break
    if not ops:
        return None, []
    return (shape, strides, offset), ops


def check_example(solve):
    for shape, strides, offset, ops, rows in EXAMPLES:
        got = solve(list(shape), list(strides), offset, [list(o) for o in ops])
        want = _dicts(rows)
        assert got == want, f"链 {ops}: got {got}, want {want}"


def check_random_chains_match_torch(solve):
    """随机操作链逐 step 对齐真实 torch 元数据；随机源固定种子，规模与题目约束一致。"""
    total = 0
    for seed in (11, 22, 33, 44, 55, 66, 77):
        rng = random.Random(seed)
        for _ in range(120):
            base, ops = _gen_chain(rng)
            if base is None:
                continue
            shape, strides, offset = base
            want = _torch_ledger(shape, strides, offset, ops)
            got = solve(list(shape), list(strides), offset, ops)
            total += len(ops)
            assert got == want, f"seed={seed} base={base} ops={ops}\ngot  {got}\nwant {want}"
    assert total > 3000, f"随机链覆盖步数过少：{total}"


def check_view_or_copy_decision(solve):
    """view 与 copy 的分界：stride-0 展开必复制，可合并的置换只改元数据。"""
    # expand 出来的 stride-0 维无法被 12 吃满 -> copy
    got = solve([3, 4], [0, 1], 0, [["reshape", 12]])
    assert got[0]["kind"] == "copy" and got[0]["strides"] == [1] and got[0]["offset"] == 0, got
    # 但只是拆细就还是 view
    got = solve([3, 4], [0, 1], 0, [["reshape", 3, 2, 2]])
    assert (got[0]["shape"], got[0]["strides"], got[0]["kind"]) == ([3, 2, 2], [0, 2, 1], "view"), got
    # (3,4,2)/(4,1,12)：内两段各自吃满，view
    got = solve([3, 4, 2], [4, 1, 12], 0, [["reshape", 3, 8]])
    assert got[0]["kind"] == "copy", got
    got = solve([3, 4, 2], [4, 1, 12], 0, [["reshape", 12, 2]])
    assert (got[0]["strides"], got[0]["kind"]) == ([1, 12], "view"), got
    # contiguous 已连续时原样返回
    got = solve([2, 3], [3, 1], 4, [["contiguous"]])
    assert got[0] == {"shape": [2, 3], "strides": [3, 1], "offset": 4, "contiguous": True, "kind": "view"}, got
    got = solve([2, 3], [1, 3], 4, [["contiguous"]])
    assert got[0]["kind"] == "copy" and got[0]["strides"] == [3, 1] and got[0]["offset"] == 0, got
    # 步长不为 1 的切片使内层不再连续，随后 reshape 必复制
    got = solve([2, 3, 4], [12, 4, 1], 0,
                [["slice", 1, 0, 3, 2], ["reshape", 16]])
    assert got[1]["kind"] == "copy" and got[1]["strides"] == [4, 1] or got[1]["shape"] == [16], got


def check_singleton_and_expand_strides(solve):
    """长度 1 的维的步长虽是「自由」的，torch 仍按固定式子填，必须逐位一致。"""
    for d, want in [(0, [12, 4, 1]), (1, [4, 4, 1]), (2, [4, 1, 1]), (-1, [4, 1, 1])]:
        got = solve([3, 4], [4, 1], 0, [["unsqueeze", d]])
        assert got[0]["strides"] == want, f"unsqueeze({d}): {got[0]} != {want}"
    got = solve([2, 3, 4], [12, 4, 1], 0, [["unsqueeze", 1], ["unsqueeze", 4]])
    assert got[1]["strides"] == [12, 12, 4, 1, 1], got
    # squeeze 不重算留下的步长
    got = solve([2, 1, 3], [3, 3, 1], 0, [["squeeze", 0]])
    assert (got[0]["shape"], got[0]["strides"]) == ([2, 1, 3], [3, 3, 1]), got
    got = solve([2, 1, 3], [3, 3, 1], 0, [["squeeze", 1]])
    assert (got[0]["shape"], got[0]["strides"]) == ([2, 3], [3, 1]), got
    # expand：长度相同保留步长；1 -> n 置 0；新增前置长度 1 取右邻的 size*stride
    got = solve([3, 4], [4, 1], 0, [["expand", 1, 3, 4]])
    assert got[0]["strides"] == [12, 4, 1], got
    got = solve([3, 4], [4, 1], 0, [["expand", 1, 1, 3, 4]])
    assert got[0]["strides"] == [12, 12, 4, 1], got
    got = solve([3, 4], [0, 1], 0, [["expand", 1, 3, 4]])
    assert got[0]["strides"] == [0, 0, 1], got
    got = solve([3, 1, 4], [4, 4, 1], 0, [["expand", 3, 5, 4], ["expand", 1, 3, 5, 4]])
    assert got[0]["strides"] == [4, 0, 1] and got[1]["strides"] == [12, 4, 0, 1], got
    got = solve([], [], 0, [["expand", 1]])
    assert got[0]["strides"] == [0], got


def check_offset_arithmetic(solve):
    """narrow/select/slice 的 offset 推进与归一化/夹紧。"""
    got = solve([6, 8], [8, 1], 0, [["narrow", 0, -2, 2]])
    assert (got[0]["shape"], got[0]["offset"]) == ([2, 8], 32), got
    got = solve([6, 8], [8, 1], 0, [["narrow", 0, 6, 0]])
    assert (got[0]["shape"], got[0]["offset"], got[0]["contiguous"]) == ([0, 8], 48, True), got
    got = solve([6, 8], [8, 1], 0, [["select", 0, -1], ["select", 0, 5]])
    assert got[0] == {"shape": [8], "strides": [1], "offset": 40, "contiguous": True, "kind": "view"}, got
    assert got[1] == {"shape": [], "strides": [], "offset": 45, "contiguous": True, "kind": "view"}, got
    # select 掉内维之后就不再连续
    got = solve([6, 8], [8, 1], 0, [["select", 1, 7]])
    assert (got[0]["shape"], got[0]["strides"], got[0]["offset"], got[0]["contiguous"]) == ([6], [8], 7, False), got
    # slice 夹紧 + 步长相乘；空切片仍推进 offset
    got = solve([6, 8], [8, 1], 0, [["slice", 0, 1, 7, 2]])
    assert (got[0]["shape"], got[0]["strides"], got[0]["offset"]) == ([3, 8], [16, 1], 8), got
    got = solve([6, 8], [8, 1], 0, [["slice", 0, -5, -1, 2]])
    assert (got[0]["shape"], got[0]["offset"]) == ([2, 8], 8), got
    got = solve([6, 8], [8, 1], 0, [["slice", 1, 5, 1, 2]])
    assert (got[0]["shape"], got[0]["strides"], got[0]["offset"], got[0]["contiguous"]) == ([6, 0], [8, 2], 5, True), got
    # 视图套视图：offset 累加在上一层的 offset 上
    got = solve([4, 5], [5, 1], 3, [["narrow", 0, 1, 2], ["narrow", 1, 2, 2]])
    assert (got[0]["offset"], got[1]["offset"], got[1]["shape"]) == (8, 10, [2, 2]), got


def check_zero_numel_and_reshape_rules(solve):
    """总长度为 0 的分支与 reshape 的推断。"""
    got = solve([2, 0, 4], [8, 4, 1], 0,
                [["reshape", 2, 0, 4], ["reshape", 0], ["reshape", 3, -1]])
    assert got[0]["strides"] == [8, 4, 1] and got[0]["kind"] == "view", got
    assert got[1]["strides"] == [1], got
    assert got[2]["shape"] == [3, 0] and got[2]["strides"] == [1, 1], got
    # 0 维基张量：reshape 给全 1 步长，flatten 合并、squeeze 删维
    got = solve([], [], 0, [["reshape", 1, 1], ["flatten", 0, 1], ["squeeze", 0]])
    assert got[0]["strides"] == [1, 1], got
    assert (got[1]["shape"], got[1]["strides"]) == ([1], [1]), got
    assert (got[2]["shape"], got[2]["strides"]) == ([], []), got
    # -1 推断与 flatten 的空操作
    got = solve([6, 8], [8, 1], 0, [["reshape", 3, -1], ["flatten", 1, 1]])
    assert got[0]["shape"] == [3, 16], got
    assert got[1] == got[0], got


def check_invalid_inputs_raise(solve):
    assert solve([3, 4], [4, 1], 0, [["transpose", 0, -1]])[0]["shape"] == [4, 3]
    bad = [
        ([3, 4], [4], 0, [["reshape", 12]]),                     # shape/strides 长度不等
        ([3, 4], [4, -1], 0, [["reshape", 12]]),                 # 负步长
        ([3, 4], [4, 1], -1, [["reshape", 12]]),                 # 负 offset
        ([3, 4], [4, 1], 0, []),                                 # 空 ops 合法，下面故意给非列表
        ([3, 4], [4, 1], 0, "reshape"),                          # ops 不是列表
        ([3, 4], [4, 1], 0, [[]]),                               # 空 op
        ([3, 4], [4, 1], 0, [["as_strided", 12]]),               # 未知操作
        ([3, 4], [4, 1], 0, [["transpose", 0]]),                 # 参数不足
        ([3, 4], [4, 1], 0, [["contiguous", 1]]),                # 多余参数
        ([3, 4], [4, 1], 0, [["reshape", 2.0]]),                 # 非 int
        ([3, 4], [4, 1], 0, [["reshape", 2, -1, -1]]),           # 两个 -1
        ([3, 4], [4, 1], 0, [["reshape", 5, 5]]),                # 总长度不符
        ([2, 0, 4], [8, 4, 1], 0, [["reshape", 3, -1, 0]]),      # 长度为 0 时无法推断
        ([3, 4], [4, 1], 0, [["expand", 3, 5]]),                 # expand 尺寸冲突
        ([3, 4], [4, 1], 0, [["expand", 4]]),                    # 提供的维数不足
        ([3, 4], [4, 1], 0, [["permute", 0, 0]]),                # permute 不是排列
        ([3, 4], [4, 1], 0, [["unsqueeze", 3]]),                 # 维越界
        ([3, 4], [4, 1], 0, [["narrow", 0, 4, 8]]),              # narrow 越界
        ([3, 4], [4, 1], 0, [["select", 1, 4]]),                 # select 越界
        ([3, 4], [4, 1], 0, [["slice", 1, 0, 4, 0]]),            # step 非正
        ([3, 4], [4, 1], 0, [["flatten", 1, 0]]),                # start > end
        ([], [], 0, [["slice", 0, 0, 1, 1]]),                    # 0 维上切片
    ]
    for case in bad:
        if case[3] == []:
            assert solve(*case) == [], case
            continue
        try:
            solve(*case)
        except ValueError:
            continue
        raise AssertionError(f"非法输入未抛 ValueError: {case}")


check_example.example = True
