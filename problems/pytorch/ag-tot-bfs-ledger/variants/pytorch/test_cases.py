import math
import random

# ============ 独立 oracle（与 solution 不同构的实现） ============

_W = {"sure": 20.0, "likely": 1.0, "impossible": 0.001}


def _oracle_value(y, evals):
    lines = y.strip().split("\n")
    low = y.lower()
    if len(lines) == 4 and "answer" not in low:
        return 0
    counts = {}
    for text in evals:
        tail = text.split("\n")[-1]
        counts[tail] = counts.get(tail, 0) + 1
    # 与官方 sum() 同序折叠：0.001*imp + 1*likely + 20*sure（计数法非逐条累加）
    return (0.001 * counts.get("impossible", 0)
            + 1 * counts.get("likely", 0)
            + 20 * counts.get("sure", 0))


def _oracle(x, plan, n_select_sample=5):
    if not isinstance(x, str):
        raise ValueError("x must be a str")
    if isinstance(n_select_sample, bool) or not isinstance(n_select_sample, int) \
            or n_select_sample < 1:
        raise ValueError("n_select_sample must be a positive int (b in the paper)")
    if not isinstance(plan, (list, tuple)) or len(plan) == 0:
        raise ValueError("plan must be a non-empty list of step dicts")

    ys = [""]
    ledger = []
    for idx, rnd in enumerate(plan):
        method, gen, evals = rnd["method"], rnd["gen"], rnd["evals"]
        if method not in ("propose", "sample"):
            raise ValueError("method must be 'propose' or 'sample'")
        kids, kid_evals = [], []
        for slot, parent in enumerate(ys):
            if slot >= len(gen) or slot >= len(evals):
                raise ValueError(f"step {idx} needs gen/evals for {len(ys)} parents")
            if method == "propose":
                if not isinstance(gen[slot], str):
                    raise ValueError("propose gen[i] must be a single output str")
                props = gen[slot].split("\n")
                if len(evals[slot]) != len(props):
                    raise ValueError("propose evals[i] must match proposal count")
                for pj, piece in enumerate(props):
                    kids.append(parent + piece + "\n")
                    kid_evals.append(evals[slot][pj])
            else:
                if not isinstance(gen[slot], (list, tuple)):
                    raise ValueError("sample gen[i] must be a list of sample strs")
                if len(evals[slot]) != len(gen[slot]):
                    raise ValueError("sample evals[i] must match sample count")
                for sj, piece in enumerate(gen[slot]):
                    kids.append(parent + piece)
                    kid_evals.append(evals[slot][sj])

        scored, memo = [], {}
        for cand, ev in zip(kids, kid_evals):
            if cand in memo:
                scored.append(0)
            else:
                v = _oracle_value(cand, ev)
                memo[cand] = v
                scored.append(v)

        ranked = sorted(range(len(kids)),
                        key=lambda k: scored[k], reverse=True)
        kept = [kids[k] for k in ranked[:n_select_sample]]
        ledger.append({"step": idx, "x": x, "ys": list(ys),
                       "new_ys": list(kids), "values": scored,
                       "select_new_ys": list(kept)})
        ys = kept
    return {"final": ys, "steps": ledger}


# ============ 比较工具 ============

def _eq(a, b, path="root"):
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            raise AssertionError(f"{path}: {a!r} vs {b!r}")
        assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9), \
            f"{path}: {a!r} vs {b!r}"
        return
    if type(a) is not type(b) and not (isinstance(a, (int, float))
                                      and isinstance(b, (int, float))):
        raise AssertionError(f"{path}: type {type(a)} vs {type(b)}")
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {set(a)} vs {set(b)}"
        for k in a:
            _eq(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: len {len(a)} vs {len(b)}"
        for i, (va, vb) in enumerate(zip(a, b)):
            _eq(va, vb, f"{path}[{i}]")
        return
    assert a == b, f"{path}: {a!r} vs {b!r}"


def _compare(solve, x, plan, **kw):
    got = solve(x, _deep_copy(plan), **dict(kw))
    exp = _oracle(x, _deep_copy(plan), **dict(kw))
    _eq(got, exp)
    return got


def _deep_copy(v):
    if isinstance(v, dict):
        return {k: _deep_copy(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [ _deep_copy(x) for x in v ]
    return v


def _rnd(method, gen, evals):
    return {"method": method, "gen": gen, "evals": evals}


# ============ 手算锚点 ============

def check_basic_propose(solve):
    """单父节点 propose：y=''+p+'\\n'、值=名称计数加权和、不除以样本数。"""
    plan = [_rnd("propose", ["a\nb"],
                 [[["sure", "sure"], ["likely", "impossible"]]])]
    led = _compare(solve, "1 2 3 4", plan, n_select_sample=5)
    st = led["steps"][0]
    assert st["new_ys"] == ["a\n", "b\n"]
    assert st["values"][0] == 40.0            # 2×sure，官方不归一化
    assert abs(st["values"][1] - 1.001) < 1e-9
    assert st["select_new_ys"] == ["a\n", "b\n"]  # b=5 全保留
    assert st["ys"] == [""]
    assert st["x"] == "1 2 3 4"
    assert led["final"] == ["a\n", "b\n"]


def check_duplicate_zero(solve):
    """同父节点提出重复候选：第二次直接 0（其评估文本即使全 sure 也被无视）。"""
    plan = [_rnd("propose", ["x\nx"], [[["sure"], ["sure", "sure", "sure"]]])]
    led = _compare(solve, "p", plan)
    st = led["steps"][0]
    assert st["new_ys"] == ["x\n", "x\n"]
    assert st["values"] == [20.0, 0]          # 重复候选强制 0
    assert st["select_new_ys"] == ["x\n", "x\n"]  # 平分仍按原序双选
    # 跨步重复不共享缓存（每步独立重评）：同串在同步内仍只算首发
    plan2 = [
        _rnd("propose", ["a"], [[["sure"]]]),
        _rnd("propose", ["a\na"], [[["likely"], ["sure"]]]),
    ]
    led2 = _compare(solve, "p", plan2, n_select_sample=2)
    assert led2["steps"][1]["new_ys"] == ["a\na\n", "a\na\n"]
    assert led2["steps"][1]["values"] == [1.0, 0]


def check_value_map_lastline(solve):
    """权重表按评估文本最后一行精确匹配；未知行/空行计 0；多行文本只看末行。"""
    plan = [_rnd("propose", ["a"],
                 [[["sure\nbecause of x", "nope", "likely\ny", ""]]])]
    led = _compare(solve, "p", plan)
    st = led["steps"][0]
    # 末行 "because of x"/"nope"/"y"/"" 均不匹配 → 0.0（最后一行才是名称）
    assert st["values"][0] == 0.0
    plan2 = [_rnd("propose", ["a"],
                  [[["sure", "SURE", "likely", "impossible"]]])]
    led2 = _compare(solve, "p", plan2)
    assert led2["steps"][0]["values"][0] == 21.001  # 大小写敏感


def check_four_line_rule(solve):
    """4 行且无 answer → 0；'Answer' 大小写不敏感豁免；尾部换行 strip 后行数变化。
    （sample 样本可含换行一次成 4 行；propose 每步只加一行需多步累积。）"""
    # 4 行无 answer：即便全 sure 也是 0
    plan = [_rnd("sample", [["a\nb\nc\nd"]], [[["sure", "sure", "sure", "sure"]]])]
    led = _compare(solve, "4 6 2", plan)
    assert led["steps"][0]["values"][0] == 0
    # 第 4 行含 Answer（大小写不敏感）→ 正常计权
    plan2 = [_rnd("sample", [["a\nb\nc\nAnswer: x"]], [[["sure"]]])]
    led2 = _compare(solve, "p", plan2)
    assert led2["steps"][0]["values"][0] == 20.0
    # propose 每步只加一行：3 步累积到 3 行 + 尾换行 strip 后 3 行，不触发惩罚
    plan3 = [_rnd("propose", ["a"], [[["sure"]]]),
             _rnd("propose", ["b"], [[["sure"]]]),
             _rnd("propose", ["c"], [[["sure"]]])]
    led3 = _compare(solve, "p", plan3)
    assert led3["steps"][2]["new_ys"][0] == "a\nb\nc\n"
    assert led3["steps"][2]["values"][0] == 20.0
    # 4 行内容 + 尾换行：strip 后恰 4 行 → 惩罚生效
    plan4 = [_rnd("sample", [["a\nb\nc\nd\n"]], [[["sure"]]])]
    led4 = _compare(solve, "p", plan4)
    assert led4["steps"][0]["values"][0] == 0
    # 5 行：恰好 4 行的等值变异（>=）不得误伤
    plan5 = [_rnd("sample", [["a\nb\nc\nd\ne"]], [[["sure"]]])]
    led5 = _compare(solve, "p", plan5)
    assert led5["steps"][0]["values"][0] == 20.0


def check_greedy_tie_and_cap(solve):
    """greedy 稳定排序：值平分保持原下标序；b 截断取前 b。"""
    # 三个候选同值 1.0（各一条 likely）：全保留且顺序不变
    plan = [_rnd("propose", ["a\nb\nc"],
                 [[["likely"], ["likely"], ["likely"]]])]
    led = _compare(solve, "p", plan, n_select_sample=5)
    assert led["steps"][0]["values"] == [1.0, 1.0, 1.0]
    assert led["steps"][0]["select_new_ys"] == ["a\n", "b\n", "c\n"]
    # b=2 截断：平分时保留下标 0、1
    led2 = _compare(solve, "p", plan, n_select_sample=2)
    assert led2["steps"][0]["select_new_ys"] == ["a\n", "b\n"]
    # 高低分交错：按值降序，平分段内小 id 优先
    plan3 = [_rnd("propose", ["a\nb\nc\nd"],
                  [[["likely"], ["sure"], ["likely"], ["sure"]]])]
    led3 = _compare(solve, "p", plan3, n_select_sample=3)
    assert led3["steps"][0]["select_new_ys"] == ["b\n", "d\n", "a\n"]


def check_sample_child_format(solve):
    """sample 拼接无分隔符（y+s 直接相连）；逐父槽位展开、父优先序。"""
    plan = [
        _rnd("sample", [["a", "b"]], [[["sure"], ["sure"]]]),
        # 第二轮父节点 ['a','b']：槽位逐父展开
        _rnd("sample", [["S1", "S2"], ["T1"]],
             [[["sure"], ["likely"]], [["impossible"]]]),
    ]
    led = _compare(solve, "X", plan, n_select_sample=5)
    st = led["steps"][1]
    assert st["ys"] == ["a", "b"]
    assert st["new_ys"] == ["aS1", "aS2", "bT1"]  # 父优先序 + 无分隔拼接
    assert st["values"] == [20.0, 1.0, 0.001]
    assert st["select_new_ys"] == ["aS1", "aS2", "bT1"]  # b=5 全保留


def check_carry_and_extras(solve):
    """本轮 select_new_ys 成为下轮父节点；多余槽位被无视；账本逐步完整。"""
    plan = [
        _rnd("propose", ["a\nb"], [[["sure"], ["likely"]]]),
        # 下轮父节点是 [a\n]（1 个）；提供 3 个槽位，后 2 个必须被无视
        _rnd("propose", ["p\nq", "zz", "yy"],
             [[["sure", "impossible"], ["sure"]], [["sure"]], [["sure"]]]),
    ]
    led = _compare(solve, "X", plan, n_select_sample=1)
    s0, s1 = led["steps"]
    assert s0["select_new_ys"] == ["a\n"]
    assert s1["ys"] == ["a\n"]
    assert s1["new_ys"] == ["a\np\n", "a\nq\n"]  # 槽位 1/2 的 zz/yy 未参与
    assert s1["values"][0] == 20.001
    assert s1["values"][1] == 20.0
    assert s1["select_new_ys"] == ["a\np\n"]
    assert led["final"] == ["a\np\n"]
    assert [s["step"] for s in led["steps"]] == [0, 1]


def check_infos_ledger(solve):
    """每步账字段齐全：step/x/ys/new_ys/values/select_new_ys。"""
    plan = [_rnd("propose", ["a"], [[["sure"]]]),
            _rnd("sample", [["s"]], [[["likely"]]])]
    led = _compare(solve, "IN", plan, n_select_sample=3)
    assert set(led.keys()) == {"final", "steps"}
    for st in led["steps"]:
        assert set(st.keys()) == {"step", "x", "ys", "new_ys", "values",
                                  "select_new_ys"}
        assert st["x"] == "IN"
    assert led["steps"][1]["ys"] == ["a\n"]
    assert led["steps"][1]["new_ys"] == ["a\ns"]  # sample 拼接无 \n
    assert led["final"] == ["a\ns"]


def check_degenerate_paths(solve):
    """sample 空样本列表 → 该父节点零候选；全零候选父节点只让其余槽位继续走。"""
    plan = [
        _rnd("sample", [["s1", "s2"]], [[["sure"], ["likely"]]]),
        # 父节点 ['s1','s2']：s1 零样本，s2 一条
        _rnd("sample", [[], ["t"]], [[], [["sure"]]]),
        _rnd("propose", ["k"], [[["likely"]]]),
    ]
    led = _compare(solve, "X", plan, n_select_sample=2)
    s0, s1, s2 = led["steps"]
    assert s0["select_new_ys"] == ["s1", "s2"]
    assert s1["ys"] == ["s1", "s2"]
    assert s1["new_ys"] == ["s2t"]
    assert s1["select_new_ys"] == ["s2t"]
    assert s2["ys"] == ["s2t"]
    assert s2["new_ys"] == ["s2tk\n"]
    assert led["final"] == ["s2tk\n"]
    # 空评估文本列表：官方 sum 空 → 0（不报错）
    plan2 = [_rnd("sample", [["s"]], [[[]]])]
    led2 = _compare(solve, "X", plan2)
    assert led2["steps"][0]["values"] == [0]


def check_invalid_inputs(solve):
    pred = _rnd("propose", ["a"], [[["sure"]]])
    for bad in [
        lambda: solve(1, [pred]),                              # x 非字符串
        lambda: solve("x", [pred], n_select_sample=0),         # b=0
        lambda: solve("x", [pred], n_select_sample=True),      # bool 混入
        lambda: solve("x", [pred], n_select_sample=2.0),       # 非整数 b
        lambda: solve("x", []),                                # 空计划
        lambda: solve("x", "nope"),                            # plan 非列表
        lambda: solve("x", [{"method": "dfs", "gen": ["a"], "evals": [["sure"]]}]),
        lambda: solve("x", [{"method": "propose", "gen": ["a"]}]),  # 缺 evals
        lambda: solve("x", [_rnd("propose", [1], [[["sure"]]])]),   # propose 非字符串
        lambda: solve("x", [_rnd("sample", ["a"], [[["sure"]]])]),  # sample 非列表
        lambda: solve("x", [_rnd("propose", ["a\nb"], [[["sure"]]])]),  # evals 缺项
        lambda: solve("x", [_rnd("sample", [["a", "b"]], [[["sure"]]])]),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


# ============ 随机对拍 ============

_EVAL_VOCAB = ["sure", "likely", "impossible", "sure\nextra", "unknown",
               "", "likely\nsure"]


def _gen_plan(rng):
    steps = rng.randint(1, 3)
    plan = []
    for _ in range(steps):
        method = rng.choice(["propose", "sample"])
        gen, evals = [], []
        for _slot in range(5):  # 槽位数≥实际父节点数（b≤5），多余被无视
            if method == "propose":
                n_prop = rng.randint(1, 3)
                gen.append("\n".join(
                    rng.choice(["w", "x", "y", "z", ""]) for _ in range(n_prop)))
                evals.append([_gen_evals(rng) for _ in range(n_prop)])
            else:
                n_sam = rng.randint(0, 3)
                gen.append([rng.choice(["s", "t", "u", ""])
                            for _ in range(n_sam)])
                evals.append([_gen_evals(rng) for _ in range(n_sam)])
        plan.append(_rnd(method, gen, evals))
    return plan


def _gen_evals(rng):
    k = rng.randint(0, 3)
    return [rng.choice(_EVAL_VOCAB) for _ in range(k)]


def check_random(solve):
    rng = random.Random(20260220)
    for _ in range(200):
        plan = _gen_plan(rng)
        b = rng.choice([2, 3, 5])
        got = solve("RND", _deep_copy(plan), n_select_sample=b)
        exp = _oracle("RND", _deep_copy(plan), n_select_sample=b)
        _eq(got, exp)


TEST_CASES = [
    ("basic_propose", check_basic_propose),
    ("duplicate_zero", check_duplicate_zero),
    ("value_map_lastline", check_value_map_lastline),
    ("four_line_rule", check_four_line_rule),
    ("greedy_tie_and_cap", check_greedy_tie_and_cap),
    ("sample_child_format", check_sample_child_format),
    ("carry_and_extras", check_carry_and_extras),
    ("infos_ledger", check_infos_ledger),
    ("degenerate_paths", check_degenerate_paths),
    ("invalid_inputs", check_invalid_inputs),
    ("random", check_random),
]
