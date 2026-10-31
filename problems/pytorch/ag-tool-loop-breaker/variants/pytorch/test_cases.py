import hashlib
import json
from collections import Counter


def _py(calls, threshold):
    """oracle：独立实现（Counter 计数、倒序求首个熔断、规范化走另一条递归路径）。"""
    def norm(v):
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return round(v, 6)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            out = {}
            for key in sorted(v.keys()):
                out[key] = norm(v[key])
            return out
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        raise ValueError("args must be JSON-able")

    allowed = []
    fingerprints = []
    counts_at_call = []
    counter = Counter()
    for c in calls:
        payload = json.dumps(norm(c["args"]), sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False)
        fp = c["tool"] + "|" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        seen = counter[fp]
        counts_at_call.append(seen)
        fingerprints.append(fp)
        if seen < threshold:
            counter[fp] += 1
            allowed.append(True)
        else:
            allowed.append(False)
    blocked = [i for i, a in enumerate(allowed) if not a]
    return {
        "allowed": allowed,
        "fingerprints": fingerprints,
        "counts_at_call": counts_at_call,
        "first_block_index": blocked[0] if blocked else None,
    }


def _compare(solve, calls, threshold):
    led = solve(calls, threshold)
    ref = _py(calls, threshold)
    assert led == ref, f"solve vs oracle mismatch:\n{led}\nvs\n{ref}"
    return led


def check_anchor_basic(solve):
    # 同 tool 同 args 三步，threshold=2：放 2 拒 1，first_block=2
    calls = [
        {"tool": "search", "args": {"q": "weather"}},
        {"tool": "search", "args": {"q": "weather"}},
        {"tool": "search", "args": {"q": "weather"}},
    ]
    led = _compare(solve, calls, 2)
    assert led["allowed"] == [True, True, False]
    assert led["counts_at_call"] == [0, 1, 2]
    assert led["first_block_index"] == 2
    assert len(set(led["fingerprints"])) == 1
    # threshold=3：三步全放
    led3 = _compare(solve, calls, 3)
    assert led3["allowed"] == [True, True, True]
    assert led3["first_block_index"] is None
    # threshold=0：第一步即熔断
    led0 = _compare(solve, calls, 0)
    assert led0["allowed"] == [False, False, False]
    assert led0["counts_at_call"] == [0, 0, 0]
    assert led0["first_block_index"] == 0


def check_distinct_tools(solve):
    # tool 不同则指纹不同：各自独立计数
    calls = [
        {"tool": "read", "args": {"path": "a.txt"}},
        {"tool": "write", "args": {"path": "a.txt"}},
        {"tool": "read", "args": {"path": "a.txt"}},
    ]
    led = _compare(solve, calls, 1)
    assert led["allowed"] == [True, True, False]
    assert led["counts_at_call"] == [0, 0, 1]
    assert led["fingerprints"][0] != led["fingerprints"][1]
    assert led["fingerprints"][0] == led["fingerprints"][2]
    # args 不同亦然
    calls2 = [
        {"tool": "t", "args": {"x": 1}},
        {"tool": "t", "args": {"x": 2}},
        {"tool": "t", "args": {"x": 1}},
    ]
    led2 = _compare(solve, calls2, 1)
    assert led2["allowed"] == [True, True, False]
    assert led2["fingerprints"][0] != led2["fingerprints"][1]


def check_key_order_canonical(solve):
    # 键序不同视为同一指纹（sort_keys）
    calls = [
        {"tool": "api", "args": {"a": 1, "b": 2}},
        {"tool": "api", "args": {"b": 2, "a": 1}},
        {"tool": "api", "args": {"a": 1, "b": 2}},
    ]
    led = _compare(solve, calls, 2)
    assert led["allowed"] == [True, True, False]
    assert len(set(led["fingerprints"])) == 1
    # 嵌套结构同样键序无关
    calls2 = [
        {"tool": "api", "args": {"outer": {"u": 1, "v": 2}, "rest": [3, {"w": 4}]}},
        {"tool": "api", "args": {"rest": [3, {"w": 4}], "outer": {"v": 2, "u": 1}}},
    ]
    led2 = _compare(solve, calls2, 1)
    assert led2["allowed"] == [True, False]
    assert len(set(led2["fingerprints"])) == 1


def check_float_rounding(solve):
    # float round 6 位：1e-7 级抖动同指纹；1e-5 级差异不同指纹
    calls = [
        {"tool": "t", "args": {"x": 1.0000001}},
        {"tool": "t", "args": {"x": 1.0000002}},
    ]
    led = _compare(solve, calls, 1)
    assert led["allowed"] == [True, False]
    assert led["fingerprints"][0] == led["fingerprints"][1]
    calls2 = [
        {"tool": "t", "args": {"x": 1.00001}},
        {"tool": "t", "args": {"x": 1.00002}},
    ]
    led2 = _compare(solve, calls2, 2)
    assert led2["allowed"] == [True, True]
    assert led2["fingerprints"][0] != led2["fingerprints"][1]


def check_first_block_scenarios(solve):
    # 多处熔断取首个；穿插不同指纹不干扰
    calls = [
        {"tool": "a", "args": {"k": 1}},
        {"tool": "b", "args": {"k": 1}},   # 新指纹
        {"tool": "a", "args": {"k": 1}},
        {"tool": "a", "args": {"k": 1}},   # a 第 3 次 → 熔断（idx 3）
        {"tool": "b", "args": {"k": 1}},
        {"tool": "b", "args": {"k": 1}},   # b 第 3 次 → 熔断（idx 5）
    ]
    led = _compare(solve, calls, 2)
    assert led["allowed"] == [True, True, True, False, True, False]
    assert led["counts_at_call"] == [0, 0, 1, 2, 1, 2]
    assert led["first_block_index"] == 3


def check_fingerprint_format(solve):
    calls = [{"tool": "grep", "args": {"pat": "x"}}]
    led = _compare(solve, calls, 1)
    fp = led["fingerprints"][0]
    digest = fp.split("|", 1)[1]
    assert fp.startswith("grep|")
    assert len(digest) == 64
    int(digest, 16)  # 64 位十六进制
    expect = "grep|" + hashlib.sha256(
        json.dumps({"pat": "x"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert fp == expect


def check_invalid_inputs(solve):
    def ve(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")
    ve([], 1)
    ve([{"tool": "t", "args": {}}], True)          # threshold bool
    ve([{"tool": "t", "args": {}}], -1)
    ve([{"tool": "t", "args": {}}], 1.5)
    ve([{"tool": "t", "args": {}}], "2")
    ve([{"tool": "t"}], 1)                          # 缺 args
    ve([{"tool": "t", "args": {}, "extra": 1}], 1)  # 多键
    ve([{"tool": "", "args": {}}], 1)               # 空 tool
    ve([{"tool": 3, "args": {}}], 1)                # tool 非 str
    ve([{"tool": "t", "args": [1, 2]}], 1)          # args 非 dict
    ve("nope", 1)                                   # calls 非 list
    ve(None, 1)


def check_oracle_random(solve):
    import random
    rng = random.Random(20260667)

    def rand_args(depth):
        if depth == 0:
            kind = rng.randint(0, 4)
            if kind == 0:
                return rng.uniform(-5, 5)
            if kind == 1:
                return rng.randint(-10, 10)
            if kind == 2:
                return rng.choice(["alpha", "beta", "gamma"])
            if kind == 3:
                return rng.choice([True, False, None])
            return rng.choice([1.0, 2.5, 0.1])  # 易撞车的 float
        if rng.randint(0, 1) == 0:
            return {f"k{rng.randint(0, 3)}": rand_args(depth - 1)
                    for _ in range(rng.randint(1, 3))}
        return [rand_args(depth - 1) for _ in range(rng.randint(1, 3))]

    for _ in range(150):
        n = rng.randint(1, 10)
        tools = ["Read", "read", "Write", "Grep"]
        calls = []
        for _ in range(n):
            args = rand_args(2)
            if not isinstance(args, dict):
                args = {"v": args}
            calls.append({"tool": rng.choice(tools), "args": args})
        threshold = rng.randint(0, 3)
        led = solve(calls, threshold)
        ref = _py(calls, threshold)
        assert led == ref, f"random mismatch at n={n}, threshold={threshold}"
