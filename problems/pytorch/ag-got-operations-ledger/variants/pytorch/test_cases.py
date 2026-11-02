import math
import random

from copy import deepcopy


# ---------------------------------------------------------------- helpers

def _deep_copy(x):
    return deepcopy(x)


def _t(state, score=0.0, scored=False, valid=False, validated=False,
       solved=False, compared=False):
    return {"state": state, "score": score, "scored": scored,
            "valid": valid, "validated": validated,
            "solved": solved, "compared_to_ground_truth": compared}


# ---------------------------------------------------------------- oracle
# 独立实现（不 import solution），同官方 spcl/graph-of-thoughts 语义

def _oracle(ops, base_state=None):
    prev = []
    steps = []
    first = True
    for spec in ops:
        kind = spec["op"]
        rec = {"op": kind}
        if kind == "score":
            if first:
                raise ValueError("no pred")
            scores = list(spec["scores"])
            new = []
            assigned = []
            if spec.get("combined"):
                for t, s in zip(prev, scores):
                    nt = dict(t)
                    nt["state"] = deepcopy(t["state"])
                    nt["score"] = s
                    nt["scored"] = True
                    assigned.append(s)
                    new.append(nt)
            else:
                for i, t in enumerate(prev):
                    nt = dict(t)
                    nt["state"] = deepcopy(t["state"])
                    nt["score"] = scores[i]
                    nt["scored"] = True
                    assigned.append(scores[i])
                    new.append(nt)
            prev = new
            rec["scores"] = assigned
        elif kind == "validate_and_improve":
            if first:
                raise ValueError("no pred")
            valids = list(spec["valids"])
            updates = list(spec.get("updates", []))
            improve = spec.get("improve", True)
            num_tries = spec.get("num_tries", 3)
            attempts = []
            out = []
            vi = ui = 0
            for t in prev:
                chain = []
                cur = dict(t)
                cur["state"] = deepcopy(t["state"])
                cur_try = 0
                while True:
                    cur["valid"] = valids[vi]
                    vi += 1
                    cur["validated"] = True
                    chain.append(cur)
                    if not improve or cur["valid"] or cur_try >= num_tries:
                        break
                    cur = _t({**cur["state"], **updates[ui]})
                    ui += 1
                    cur_try += 1
                attempts.append(len(chain))
                out.append(chain[-1])
            prev = out
            rec["attempts"] = attempts
        elif kind == "generate":
            if not prev and not first:
                rec["skipped"] = True
                rec["count"] = 0
                rec["thoughts"] = []
                steps.append(rec)
                first = False
                continue
            parents = prev if prev else [_t(deepcopy(base_state)
                                            if base_state else {})]
            parsed = spec["parsed"]
            out = []
            for t, nss in zip(parents, parsed):
                for ns in nss:
                    out.append(_t({**t["state"], **ns}))
            prev = out
            rec["skipped"] = False
        elif kind == "improve":
            if first:
                raise ValueError("no pred")
            out = []
            for t, upd in zip(prev, spec["updates"]):
                out.append(_t({**t["state"], **upd}))
            prev = out
        elif kind == "aggregate":
            if first:
                raise ValueError("no pred")
            if not prev:
                rec["count"] = 0
                rec["thoughts"] = []
                steps.append(rec)
                first = False
                continue
            base = {}
            for t in sorted(prev, key=lambda x: x["score"]):
                base = {**base, **t["state"]}
            parsed = spec["parsed"]
            if isinstance(parsed, dict):
                parsed = [parsed]
            out = [_t({**base, **ns}) for ns in parsed]
            prev = out
            rec["base_state"] = deepcopy(base)
        elif kind == "keep_best_n":
            if first:
                raise ValueError("no pred")
            n = spec["n"]
            higher = spec.get("higher_is_better", True)
            if not all(t["scored"] for t in prev):
                raise ValueError("unscored")
            fallback = False
            try:
                ordered = sorted(prev, key=lambda t: t["score"],
                                 reverse=higher)
            except TypeError:
                fallback = True
                ordered = sorted(
                    [t for t in prev if isinstance(t["score"], float)],
                    key=lambda t: t["score"], reverse=higher)
            prev = [dict(t) for t in ordered[:n]]
            rec["fallback"] = fallback
        elif kind == "keep_valid":
            if first:
                raise ValueError("no pred")
            prev = [dict(t) for t in prev
                    if not t["validated"] or t["valid"]]
        elif kind == "ground_truth":
            if first:
                raise ValueError("no pred")
            results = list(spec["results"])
            raises = list(spec.get("raises", [False] * len(results)))
            out = []
            sc = 0
            for i, t in enumerate(prev):
                nt = dict(t)
                nt["solved"] = False if raises[i] else results[i]
                nt["compared_to_ground_truth"] = True
                sc += 1 if nt["solved"] else 0
                out.append(nt)
            prev = out
            rec["solved_count"] = sc
        elif kind == "selector":
            if not prev and first:
                prev = [_t(deepcopy(base_state) if base_state else {})]
            prev = [dict(prev[i]) for i in spec["indices"]]
        else:
            raise ValueError("unknown op")
        if kind != "generate" or not rec.get("skipped"):
            if "count" not in rec:
                rec["count"] = len(prev)
                rec["thoughts"] = [dict(t) for t in prev]
        steps.append(rec)
        first = False
    return {"steps": steps, "final": [dict(t) for t in prev]}


# ---------------------------------------------------------------- compare

_FLOAT_KEYS = ("score",)


def _norm_thought(t):
    d = dict(t)
    for k in _FLOAT_KEYS:
        if isinstance(d.get(k), float) and isinstance(d.get(k), float):
            pass
    return d


def _eq(a, b):
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), (len(a), len(b))
        for x, y in zip(a, b):
            if isinstance(x, (list, dict)):
                _eq(x, y)
            elif isinstance(x, float) and isinstance(y, float):
                assert math.isclose(x, y, rel_tol=0, abs_tol=1e-12), (x, y)
            else:
                assert x == y, (x, y)
        return
    assert isinstance(a, dict) and isinstance(b, dict), (a, b)
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    for k in a:
        if isinstance(a[k], float) and isinstance(b[k], float):
            assert math.isclose(a[k], b[k], rel_tol=0, abs_tol=1e-12), \
                (k, a[k], b[k])
        elif isinstance(a[k], (list, dict)):
            _eq(a[k], b[k])
        else:
            assert a[k] == b[k], (k, a[k], b[k])


def _compare(solve, ops, base_state=None):
    got = solve(_deep_copy(ops), _deep_copy(base_state))
    exp = _oracle(_deep_copy(ops), _deep_copy(base_state))
    _eq(got, exp)
    return got


# ---------------------------------------------------------------- checks

def check_score_combined_zip_truncation(solve):
    """combined 模式 zip 截断：scores 更短时多出的 thought 静默丢弃。"""
    ops = [{"op": "score", "combined": True, "scores": [0.5]}]
    # 无前驱 → ValueError
    try:
        solve(ops)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    gen = {"op": "generate",
           "parsed": [[{"x": "a"}, {"x": "b"}, {"x": "c"}]]}
    led = _compare(solve, [gen, {"op": "score", "combined": True,
                                 "scores": [1.0, 2.0]}])
    st = led["steps"][1]
    assert st["count"] == 2                       # c 被 zip 截掉
    assert [t["state"]["x"] for t in st["thoughts"]] == ["a", "b"]
    assert st["scores"] == [1.0, 2.0]
    assert all(t["scored"] for t in st["thoughts"])
    # combined 单条 score：三 thought 只剩第一条
    led2 = _compare(solve, [gen, {"op": "score", "combined": True,
                                  "scores": [7.0]}])
    assert led2["steps"][1]["count"] == 1


def check_score_individual_and_flags(solve):
    """individual 模式逐条赋分；score setter 置 scored=True、缺省 0.0。"""
    gen = {"op": "generate", "parsed": [[{"k": 1}, {"k": 2}]]}
    led = _compare(solve, [gen, {"op": "score", "combined": False,
                                 "scores": [0.25, -1.5]}])
    st = led["steps"][1]
    assert st["scores"] == [0.25, -1.5]
    assert [t["score"] for t in st["thoughts"]] == [0.25, -1.5]
    assert all(t["scored"] for t in st["thoughts"])
    assert led["steps"][0]["thoughts"][1]["score"] == 0.0   # 缺省
    assert led["steps"][0]["thoughts"][0]["scored"] is False


def check_validate_improve_tries(solve):
    """恒定无效：恰 num_tries+1 次验证、num_tries 次改进；
    改进产物是全新缺省 thought（score 复位 0.0、标志全复位）。"""
    ops = [{"op": "generate", "parsed": [[{"v": 1}]]},
           {"op": "score", "combined": False, "scores": [3.0]},
           {"op": "validate_and_improve", "improve": True, "num_tries": 2,
            "valids": [False, False, False],
            "updates": [{"v": 2}, {"v": 3}]}]
    led = _compare(solve, ops)
    va = led["steps"][2]
    assert va["attempts"] == [3]                  # num_tries+1 次验证
    fin = va["thoughts"][0]
    assert fin["state"] == {"v": 3}
    assert fin["valid"] is False and fin["validated"] is True
    assert fin["score"] == 0.0 and fin["scored"] is False   # 全新 thought
    assert fin["solved"] is False
    # 先无效后有效：恰 2 次验证 1 次改进
    ops2 = [{"op": "generate", "parsed": [[{"v": 1}]]},
            {"op": "validate_and_improve", "improve": True, "num_tries": 3,
             "valids": [False, True], "updates": [{"v": 9}]}]
    led2 = _compare(solve, ops2)
    assert led2["steps"][1]["attempts"] == [2]
    assert led2["final"][0]["state"] == {"v": 9}
    assert led2["final"][0]["valid"] is True
    # improve=False：单次验证、无改进、状态不变
    ops3 = [{"op": "generate", "parsed": [[{"v": 1}]]},
            {"op": "validate_and_improve", "improve": False,
             "valids": [False], "updates": []}]
    led3 = _compare(solve, ops3)
    assert led3["steps"][1]["attempts"] == [1]
    assert led3["final"][0]["state"] == {"v": 1}
    assert led3["final"][0]["validated"] is True
    assert led3["final"][0]["valid"] is False
    # 恰在最后一次尝试有效：num_tries 次改进后验证通过
    ops4 = [{"op": "generate", "parsed": [[{"v": 1}]]},
            {"op": "validate_and_improve", "improve": True, "num_tries": 2,
             "valids": [False, False, True],
             "updates": [{"v": 2}, {"v": 3}]}]
    led4 = _compare(solve, ops4)
    assert led4["steps"][1]["attempts"] == [3]
    assert led4["final"][0]["valid"] is True


def check_generate_merge_and_skip(solve):
    """generate：父态为底、新 state 覆盖合并；前驱为空且有前驱 → 跳过。"""
    ops = [{"op": "generate", "parsed": [[{"a": 1}]]},
           {"op": "generate", "parsed": [[{"a": 2, "b": 3},
                                          {"b": 4}]]}]
    led = _compare(solve, ops)
    g1, g2 = led["steps"]
    assert g1["count"] == 1 and g1["skipped"] is False
    assert g2["count"] == 2                       # 单父 → 两条
    assert g2["thoughts"][0]["state"] == {"a": 2, "b": 3}
    assert g2["thoughts"][1]["state"] == {"a": 1, "b": 4}  # 父态保留
    # selector 选出空集 → 下一个 generate 跳过
    ops2 = [{"op": "generate", "parsed": [[{"x": 1}]]},
            {"op": "selector", "indices": []},
            {"op": "generate", "parsed": [[{"y": 1}]]}]
    led2 = _compare(solve, ops2)
    assert led2["steps"][2]["skipped"] is True    # 级联跳过
    assert led2["steps"][1]["count"] == 0         # selector 清空
    assert led2["final"] == []


def check_aggregate_score_order(solve):
    """aggregate：score 升序 stable 合并，高 score 覆盖同名键；
    未打分缺省 0.0。"""
    ops = [{"op": "generate",
            "parsed": [[{"v": "low"}, {"v": "hi"}, {"w": 1}]]},
           {"op": "score", "combined": False, "scores": [1.0, 9.0, 5.0]},
           {"op": "aggregate", "parsed": {"z": 0}}]
    led = _compare(solve, ops)
    ag = led["steps"][2]
    assert ag["base_state"] == {"v": "hi", "w": 1}   # 9.0 的 v 覆盖 1.0 的
    assert ag["count"] == 1
    assert ag["thoughts"][0]["state"] == {"v": "hi", "w": 1, "z": 0}
    assert ag["thoughts"][0]["score"] == 0.0          # 全新缺省 thought
    # 平分（全 0.0 未打分）保持原序：后 thought 覆盖前
    ops2 = [{"op": "generate", "parsed": [[{"v": 1}, {"v": 2}]]},
            {"op": "aggregate", "parsed": []}]
    led2 = _compare(solve, ops2)
    assert led2["steps"][1]["base_state"] == {"v": 2}
    assert led2["steps"][1]["count"] == 0             # parsed 空列表


def check_keep_best_n(solve):
    """keep_best_n：stable 排序平分保原序；未打分 ValueError；n<=0。"""
    gen = {"op": "generate",
           "parsed": [[{"x": "a"}, {"x": "b"}, {"x": "c"}]]}
    ops = [gen, {"op": "score", "combined": False,
                 "scores": [2.0, 3.0, 2.0]},
           {"op": "keep_best_n", "n": 2, "higher_is_better": True}]
    led = _compare(solve, ops)
    kb = led["steps"][2]
    assert [t["state"]["x"] for t in kb["thoughts"]] == ["b", "a"]  # a 先于 c
    assert kb["fallback"] is False
    # higher_is_better=False：升序
    ops2 = [gen, {"op": "score", "combined": False,
                  "scores": [2.0, 3.0, 1.0]},
            {"op": "keep_best_n", "n": 2, "higher_is_better": False}]
    led2 = _compare(solve, ops2, {"p": 1})
    assert [t["state"]["x"] for t in led2["steps"][2]["thoughts"]] == \
        ["c", "a"]
    # n 大于总数：全保留
    led3 = _compare(solve, [gen, {"op": "score", "combined": False,
                                  "scores": [1.0, 2.0, 3.0]},
                            {"op": "keep_best_n", "n": 10}])
    assert led3["steps"][2]["count"] == 3
    # 未打分 → ValueError
    try:
        solve([gen, {"op": "keep_best_n", "n": 1}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    # n<=0 → ValueError
    try:
        solve([gen, {"op": "score", "combined": False, "scores": [1, 2, 3]},
               {"op": "keep_best_n", "n": 0}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def check_keep_best_n_fallback(solve):
    """score=None（已打分）触发 TypeError → float-only fallback：
    None 与 int score 的 thought 全部丢弃。"""
    gen = {"op": "generate",
           "parsed": [[{"x": "a"}, {"x": "b"}, {"x": "c"}, {"x": "d"}]]}
    ops = [gen,
           {"op": "score", "combined": False,
            "scores": [None, 2.5, 1, 3.5]},
           {"op": "keep_best_n", "n": 2}]
    led = _compare(solve, ops)
    kb = led["steps"][2]
    assert kb["fallback"] is True
    # isinstance(1, float) is False → int score 的 c 也被丢弃
    assert [t["state"]["x"] for t in kb["thoughts"]] == ["d", "b"]
    assert kb["count"] == 2


def check_keep_valid(solve):
    """keep_valid：`not validated or valid`——未验证直接放行。"""
    ops = [{"op": "generate",
            "parsed": [[{"x": 1}, {"x": 2}, {"x": 3}]]},
           {"op": "validate_and_improve", "improve": False,
            "valids": [True, False, False]},
           {"op": "keep_valid"}]
    led = _compare(solve, ops)
    kv = led["steps"][2]
    # 三个都 validated：只留 valid 的第一个
    assert [t["state"]["x"] for t in kv["thoughts"]] == [1]
    # 未验证的 thought 放行：selector 跳过验证后 keep_valid 全留
    ops2 = [{"op": "generate", "parsed": [[{"x": 1}, {"x": 2}]]},
            {"op": "selector", "indices": [0]},
            {"op": "validate_and_improve", "improve": False,
             "valids": [False]},
            {"op": "keep_valid"}]
    led2 = _compare(solve, ops2)
    # selector 后只剩 x=1（invalid 被剔）；无未验证 thought 可试
    # 构造：直接 generate 后不验证 → keep_valid 全留
    ops3 = [{"op": "generate", "parsed": [[{"x": 1}, {"x": 2}]]},
            {"op": "keep_valid"}]
    led3 = _compare(solve, ops3)
    assert led3["steps"][1]["count"] == 2         # 未验证放行


def check_ground_truth(solve):
    """ground_truth：异常 → solved=False；False 也置
    compared_to_ground_truth=True。"""
    ops = [{"op": "generate",
            "parsed": [[{"x": 1}, {"x": 2}, {"x": 3}]]},
           {"op": "ground_truth", "results": [True, False, True],
            "raises": [False, False, True]}]
    led = _compare(solve, ops)
    gt = led["steps"][1]
    assert [t["solved"] for t in gt["thoughts"]] == [True, False, False]
    assert gt["solved_count"] == 1
    assert all(t["compared_to_ground_truth"] for t in gt["thoughts"])
    # raises 位不消费 results 之外的位置——results 仍按 thought 序
    ops2 = [{"op": "generate", "parsed": [[{"x": 1}, {"x": 2}]]},
            {"op": "ground_truth", "results": [True, True],
             "raises": [True, False]}]
    led2 = _compare(solve, ops2)
    assert [t["solved"] for t in led2["steps"][1]["thoughts"]] == \
        [False, True]


def check_selector_and_flags_copy(solve):
    """selector 按索引选取；from_thought 复制全部标志。"""
    ops = [{"op": "generate",
            "parsed": [[{"x": 1}, {"x": 2}, {"x": 3}]]},
           {"op": "score", "combined": False, "scores": [1.0, 2.0, 3.0]},
           {"op": "selector", "indices": [2, 0]}]
    led = _compare(solve, ops)
    sel = led["steps"][2]
    assert [t["state"]["x"] for t in sel["thoughts"]] == [3, 1]
    assert [t["score"] for t in sel["thoughts"]] == [3.0, 1.0]
    assert all(t["scored"] for t in sel["thoughts"])   # 标志被复制
    # 首个操作为 selector + base_state
    led2 = _compare(solve, [{"op": "selector", "indices": [0]}],
                    {"seed": "s"})
    assert led2["final"][0]["state"] == {"seed": "s"}


def check_improve_resets(solve):
    """improve 产物是全新缺省 thought（score 复位、标志复位）。"""
    ops = [{"op": "generate", "parsed": [[{"v": 1}]]},
           {"op": "score", "combined": False, "scores": [5.0]},
           {"op": "improve", "updates": [{"v": 2, "w": 3}]}]
    led = _compare(solve, ops)
    im = led["steps"][2]
    assert im["thoughts"][0]["state"] == {"v": 2, "w": 3}
    assert im["thoughts"][0]["score"] == 0.0
    assert im["thoughts"][0]["scored"] is False
    assert im["thoughts"][0]["validated"] is False


def check_domain_errors(solve):
    """无前驱时各操作 ValueError（官方 assert 转 ValueError）。"""
    for op in [{"op": "improve", "updates": []},
               {"op": "aggregate", "parsed": []},
               {"op": "keep_valid"},
               {"op": "ground_truth", "results": []},
               {"op": "validate_and_improve", "valids": []},
               {"op": "keep_best_n", "n": 1}]:
        try:
            solve([op])
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {op['op']}")
    # 未知操作
    try:
        solve([{"op": "nope"}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown op")


def check_random(solve):
    """100 组随机操作链对拍 oracle。"""
    rng = random.Random(20261102)

    def rc():
        return rng.choice([0.0, 0.5, 1.0, 2.0, -1.0, 3.0])

    for _ in range(100):
        ops = []
        if rng.random() < 0.5:
            n0 = rng.randint(1, 3)
            ops.append({"op": "generate",
                        "parsed": [[{"v": i, "tag": "g0"}
                                    for i in range(n0)]]})
        else:
            ops.append({"op": "selector", "indices": [0]})
            ops.append({"op": "generate",
                        "parsed": [[{"v": i}
                                    for i in range(rng.randint(1, 3))]]})
        for step in range(rng.randint(1, 3)):
            kind = rng.choice(["score", "validate_and_improve", "generate",
                               "improve", "aggregate", "keep_best_n",
                               "keep_valid", "ground_truth", "selector"])
            if kind == "score":
                combined = rng.random() < 0.5
                ns = rng.randint(1, 3)
                k = rng.randint(1, ns) if combined else ns
                ops.append({"op": "score", "combined": combined,
                            "scores": [rc() if rng.random() < 0.85 else None
                                       for _ in range(k)]})
            elif kind == "validate_and_improve":
                ops.append({"op": "validate_and_improve",
                            "improve": rng.random() < 0.7,
                            "num_tries": rng.randint(0, 2),
                            "valids": [rng.random() < 0.4
                                       for _ in range(6)],
                            "updates": [{"u": i} for i in range(4)]})
            elif kind == "generate":
                ops.append({"op": "generate",
                            "parsed": [[{"g": step, "i": i}
                                        for i in range(rng.randint(0, 2))]
                                       for _ in range(3)]})
            elif kind == "improve":
                ops.append({"op": "improve",
                            "updates": [{"m": step} for _ in range(4)]})
            elif kind == "aggregate":
                p = rng.random() < 0.5
                ops.append({"op": "aggregate",
                            "parsed": {"a": step} if p
                            else [{"a": step}, {"b": step}]})
            elif kind == "keep_best_n":
                ops.append({"op": "keep_best_n",
                            "n": rng.randint(1, 3),
                            "higher_is_better": rng.random() < 0.5})
            elif kind == "ground_truth":
                ops.append({"op": "ground_truth",
                            "results": [rng.random() < 0.5
                                        for _ in range(5)],
                            "raises": [rng.random() < 0.2
                                       for _ in range(5)]})
            else:
                ops.append({"op": "selector",
                            "indices": rng.sample(range(3),
                                                  rng.randint(0, 3))})
        # 逐操作尝试；随机链可能触发域/索引/类型错误，两侧实现
        # 行为一致 → 只要求对称（同抛或同过）
        try:
            got = solve(_deep_copy(ops), {"p": 1})
            err = None
        except Exception as e:
            got, err = None, type(e).__name__
        try:
            exp = _oracle(_deep_copy(ops), {"p": 1})
            exp_err = None
        except Exception as e:
            exp, exp_err = None, type(e).__name__
        if err is not None or exp_err is not None:
            assert err == exp_err, (err, exp_err, ops)
            continue
        _eq(got, exp)
