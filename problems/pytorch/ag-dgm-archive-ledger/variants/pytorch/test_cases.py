import math
import torch

TOL = 1e-12


def _close(a, b):
    if a == b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(a)), abs(float(b)))
        except (TypeError, ValueError):
            return False
    return a == b


def _eq(got, ref, path=""):
    if isinstance(ref, dict):
        assert isinstance(got, dict), path
        assert set(got) == set(ref), (path, sorted(got), sorted(ref))
        for k in ref:
            _eq(got[k], ref[k], path + "/" + str(k))
        return
    if isinstance(ref, (list, tuple)):
        assert isinstance(got, (list, tuple)), (path, type(got))
        assert len(got) == len(ref), (path, len(got), len(ref))
        for i, (g, r) in enumerate(zip(got, ref)):
            _eq(g, r, path + "[%d]" % i)
        return
    assert _close(got, ref), (path, got, ref)


def _sig(score):
    t = torch.tensor(10.0 * (score - 0.5), dtype=torch.float64)
    return float(torch.sigmoid(t))


def _py(commits, children, method="score_child_prop", size=2,
        update="keep_all", noise_leeway=0.1, num_issues=60,
        num_full_eval=500, parent_picks=None, pad_picks=None, draws=None,
        baseline=None):
    arch = [dict(c) for c in commits]
    kids = [dict(c) for c in children]
    ids = [c["id"] for c in arch]
    score = {c["id"]: float(c["score"]) for c in arch}
    ctx = {c["id"]: bool(c.get("context_exceeded", False)) for c in arch}
    cand, seen = {}, []
    for c in arch:
        if c.get("metadata_ok", True) is False:
            continue
        cand[c["id"]] = {"id": c["id"], "score": float(c["score"]),
                         "resolved": c["resolved"], "unresolved": c["unresolved"],
                         "empty": c["empty"], "submitted": c["submitted"],
                         "context_exceeded": bool(c.get("context_exceeded", False)),
                         "children_count": 0}
        seen.append(c["id"])
    for c in arch:
        p = c.get("parent")
        if p is not None and c["id"] in cand and p in cand:
            cand[p]["children_count"] += 1

    probs = None
    if baseline == "no_darwin":
        parents = [seen[-1]]
    elif method in ("score_prop", "score_child_prop"):
        w = [_sig(cand[i]["score"]) for i in seen]
        if method == "score_child_prop":
            w = [x / (1 + cand[i]["children_count"]) for x, i in zip(w, seen)]
        s = sum(w)
        probs = [x / s for x in w]
        parents = [seen[k] for k in parent_picks]
    elif method == "best":
        ranked = sorted(seen, key=lambda i: cand[i]["score"])
        parents = ranked[:min(size, len(ranked))]
        if len(parents) < size:
            for k in pad_picks:
                parents.append(parents[k])
    else:
        parents = [seen[k] for k in parent_picks]

    used = 0
    entries = []
    for pid in parents:
        c = cand[pid]
        total = c["empty"] + c["resolved"] + c["unresolved"]
        done = False
        if c["empty"] >= 0.1 * total:
            u = draws[used]
            used += 1
            if u < 0.25:
                entries.append({"parent": pid, "entry": "solve_empty_patches", "index": None})
                done = True
        if not done:
            u = draws[used]
            used += 1
            if u < 0.25:
                entries.append({"parent": pid, "entry": "solve_stochasticity", "index": None})
                done = True
        if not done and ctx[pid]:
            u = draws[used]
            used += 1
            if u < 0.25:
                entries.append({"parent": pid, "entry": "solve_contextlength", "index": None})
                done = True
        if not done:
            if c["unresolved"] == 0:
                entries.append({"parent": pid, "entry": "skipped", "index": None})
            else:
                u = draws[used]
                used += 1
                entries.append({"parent": pid, "entry": "unresolved",
                                "index": min(int(u * c["unresolved"]), c["unresolved"] - 1)})

    ok, bad = [], []
    for k in kids:
        if k.get("metadata_ok", True) is False:
            bad.append({"id": k["id"], "reason": "no_metadata"})
        elif k["resolved"] + k["unresolved"] == 0:
            bad.append({"id": k["id"], "reason": "no_patch"})
        elif k["submitted"] < num_issues:
            bad.append({"id": k["id"], "reason": "not_enough_tests"})
        else:
            ok.append(k["id"])

    pool = [score["initial"]]
    for c in arch:
        if c["submitted"] >= num_full_eval * 0.9:
            pool.append(float(c["score"]))
    pool.sort(reverse=True)
    thr = pool[1] if len(pool) > 1 else pool[0]
    thr = max(thr, 0.4)

    final = list(ids)
    if update == "keep_better":
        gate = score["initial"] - noise_leeway
        for k in kids:
            if k["id"] in ok and float(k["score"]) >= gate:
                final.append(k["id"])
    else:
        final.extend(ok)

    return {"candidates": [cand[i] for i in seen], "probabilities": probs,
            "parents": parents, "entries": entries, "draws_used": used,
            "compiled": ok, "rejected": bad, "archive": final, "threshold": thr}


def _compare(solve, commits, children, **kw):
    ref = _py(commits, children, **kw)
    got = solve([dict(c) for c in commits], [dict(c) for c in children], **kw)
    _eq(got, ref)
    return got


def _c(cid, score, resolved=5, unresolved=5, empty=0, submitted=100,
       parent=None, meta=True, ctx=False):
    return {"id": cid, "score": score, "resolved": resolved,
            "unresolved": unresolved, "empty": empty, "submitted": submitted,
            "parent": parent, "metadata_ok": meta, "context_exceeded": ctx}


BASE = [_c("initial", 0.30, submitted=100)]


def check_candidates_children_count(solve):
    """children_count 数的是**存档内**父代链接；父代不在候选表里就不计数。"""
    rows = [_c("initial", 0.3), _c("a", 0.4, parent="initial"),
            _c("b", 0.5, parent="initial"), _c("c", 0.6, parent="a"),
            _c("ghost", 0.9, meta=False), _c("d", 0.7, parent="ghost")]
    led = _compare(solve, rows, [], method="random", size=1, parent_picks=[0],
                   draws=[0.9] * 6)
    counts = {c["id"]: c["children_count"] for c in led["candidates"]}
    assert counts == {"initial": 2, "a": 1, "b": 0, "c": 0, "d": 0}
    # meta 读不出来的不算候选，但它仍是 d 的父代 -> d 留着、计数不动
    assert [c["id"] for c in led["candidates"]] == ["initial", "a", "b", "c", "d"]


def check_score_prop_sigmoid(solve):
    """父代概率 = sigmoid(10(s-0.5)) 归一化（不是分数本身归一化）。"""
    rows = [_c("initial", 0.3), _c("a", 0.7)]
    led = _compare(solve, rows, [], method="score_prop", size=1, parent_picks=[0],
                   draws=[0.9] * 6)
    probs = led["probabilities"]
    assert abs(sum(probs) - 1.0) < 1e-12
    ratio = _sig(0.7) / _sig(0.3)
    assert abs(probs[1] / probs[0] - ratio) < 1e-9
    # 顺序与存档顺序一致
    assert [c["id"] for c in led["candidates"]] == ["initial", "a"]


def check_score_child_prop(solve):
    """score_child_prop 再把权重除以 (1 + children_count)——子代多的父代被压。"""
    rows = [_c("initial", 0.5), _c("a", 0.5, parent="initial"),
            _c("b", 0.5, parent="initial")]
    led = _compare(solve, rows, [], method="score_child_prop", size=1,
                   parent_picks=[0], draws=[0.9] * 6)
    probs = led["probabilities"]
    w = [1.0 / (1 + 2), 1.0 / (1 + 0), 1.0 / (1 + 0)]
    assert abs(probs[0] - w[0] / sum(w)) < 1e-12
    assert abs(probs[1] - w[1] / sum(w)) < 1e-12


def check_best_takes_worst(solve):
    """`best` 按分数**升序**取前 size 个：官方实现取到的是最差的那几个。"""
    rows = [_c("initial", 0.2), _c("a", 0.9), _c("b", 0.5), _c("c", 0.7)]
    led = _compare(solve, rows, [], method="best", size=2, draws=[0.9] * 6)
    assert led["parents"] == ["initial", "b"]
    assert led["probabilities"] is None

    # 名额不足时用 pad_picks 从已选中的里面补
    led = _compare(solve, BASE, [], method="best", size=3, pad_picks=[0, 0],
                   draws=[0.9] * 8)
    assert led["parents"] == ["initial", "initial", "initial"]


def check_no_darwin_last(solve):
    """no_darwin 基线下永远取候选表最后一条（不做任何抽样）。"""
    rows = [_c("initial", 0.2), _c("a", 0.1), _c("b", 0.9)]
    led = _compare(solve, rows, [], method="score_prop", size=2,
                   baseline="no_darwin", draws=[0.9] * 6)
    assert led["parents"] == ["b"]
    assert led["draws_used"] == 2


def check_gate_empty_short_circuit(solve):
    """空补丁占比不够时第一道闸门**不抛硬币**，draws 少消耗一个。"""
    # empty=10 / total=20 -> 10 >= 2 成立，第一道先抛
    rich = [_c("initial", 0.3, resolved=5, unresolved=5, empty=10)]
    led = _compare(solve, rich, [], method="random", size=1, parent_picks=[0],
                   draws=[0.5, 0.1])
    assert led["entries"][0]["entry"] == "solve_stochasticity"
    assert led["draws_used"] == 2

    # empty=1 / total=19 -> 1 >= 1.9 不成立，第一道短路，0.1 落在第二道上
    thin = [_c("initial", 0.3, resolved=9, unresolved=9, empty=1)]
    led = _compare(solve, thin, [], method="random", size=1, parent_picks=[0],
                   draws=[0.1, 0.9])
    assert led["entries"][0]["entry"] == "solve_stochasticity"
    assert led["draws_used"] == 1

    # 同一条 thin，把 0.1 放在第二位就没有闸门命中，落到未解决项抽取
    led = _compare(solve, thin, [], method="random", size=1, parent_picks=[0],
                   draws=[0.9, 0.1])
    assert led["entries"][0]["entry"] == "unresolved"
    assert led["entries"][0]["index"] == 0
    assert led["draws_used"] == 2


def check_gate_empty_patches(solve):
    """空补丁闸门命中：`empty >= 0.1 * total` 且 coin < 0.25，直接用掉第一枚硬币。"""
    rows = [_c("initial", 0.3, resolved=0, unresolved=0, empty=10)]
    led = _compare(solve, rows, [], method="random", size=1, parent_picks=[0],
                   draws=[0.24, 0.9])
    assert led["entries"] == [{"parent": "initial",
                               "entry": "solve_empty_patches", "index": None}]
    assert led["draws_used"] == 1

    # 硬币恰好 0.25 不中第一道，第二道同样 0.25 不中，且没有未解决项 -> 跳过
    led = _compare(solve, rows, [], method="random", size=1, parent_picks=[0],
                   draws=[0.25, 0.25])
    assert led["entries"][0]["entry"] == "skipped"
    assert led["draws_used"] == 2


def check_gate_contextlength(solve):
    """第三道闸门只在 `context_exceeded` 为真时抛硬币；假则不消耗。"""
    plain = [_c("initial", 0.3, unresolved=4)]
    led = _compare(solve, plain, [], method="random", size=1, parent_picks=[0],
                   draws=[0.9, 0.5])
    assert led["entries"][0] == {"parent": "initial", "entry": "unresolved", "index": 2}
    assert led["draws_used"] == 2       # 第三道没抛

    ctx = [_c("initial", 0.3, unresolved=4, ctx=True)]
    led = _compare(solve, ctx, [], method="random", size=1, parent_picks=[0],
                   draws=[0.9, 0.1, 0.5])
    assert led["entries"][0]["entry"] == "solve_contextlength"
    assert led["draws_used"] == 2       # 第二道 + 第三道


def check_entry_skipped_no_unresolved(solve):
    """没未解决项时直接跳过，不再抽第四枚硬币。"""
    rows = [_c("initial", 0.3, resolved=5, unresolved=0)]
    led = _compare(solve, rows, [], method="random", size=1, parent_picks=[0],
                   draws=[0.9] * 6)
    assert led["entries"] == [{"parent": "initial", "entry": "skipped", "index": None}]
    assert led["draws_used"] == 1        # 只有第二道闸门抛了硬币


def check_entry_index_clamped(solve):
    """未解决项下标 `min(int(coin * n), n - 1)`，coin → 1 也不会越界。"""
    rows = [_c("initial", 0.3, unresolved=3)]
    for coin, want in ((0.0, 0), (0.34, 1), (0.99, 2)):
        led = _compare(solve, rows, [], method="random", size=1, parent_picks=[0],
                       draws=[0.9, coin])
        assert led["entries"][0]["index"] == want, (coin, led["entries"][0])


def check_compile_filter(solve):
    """子代要过三关：有 metadata、补丁非全空、题数够；否则给三种不同的理由。"""
    kids = [_c("k1", 0.6, resolved=0, unresolved=0),
            _c("k2", 0.6, submitted=59),
            _c("k3", 0.6, meta=False),
            _c("k4", 0.6, submitted=60, resolved=0, unresolved=3)]
    led = _compare(solve, BASE, kids, method="random", size=1, parent_picks=[0],
                   draws=[0.9] * 6)
    assert led["compiled"] == ["k4"]
    assert led["rejected"] == [{"id": "k1", "reason": "no_patch"},
                               {"id": "k2", "reason": "not_enough_tests"},
                               {"id": "k3", "reason": "no_metadata"}]
    assert led["archive"] == ["initial", "k4"]


def check_keep_better_leeway(solve):
    """keep_better 的门槛是「初始分 - 噪声余量」，且只对通过编译的子代生效。"""
    kids = [_c("lo", 0.19), _c("edge", 0.20), _c("hi", 0.9)]
    led = _compare(solve, BASE, kids, method="random", size=1, parent_picks=[0],
                   update="keep_better", noise_leeway=0.1, draws=[0.9] * 6)
    # 门槛 = 0.30 - 0.1 = 0.20，lo 被拒、edge 恰好等于门槛保留
    assert led["archive"] == ["initial", "edge", "hi"]

    led = _compare(solve, BASE, kids, method="random", size=1, parent_picks=[0],
                   update="keep_better", noise_leeway=0.0, draws=[0.9] * 6)
    assert led["archive"] == ["initial", "hi"]  # 门槛 0.30，只有 0.9 那条够


def check_threshold_double_counts_initial(solve):
    """满评门槛的候选池里 initial 会被数两次（无条件一次 + 过筛一次）。"""
    rows = [_c("initial", 0.60, submitted=500), _c("a", 0.50, submitted=500)]
    led = _compare(solve, rows, [], method="random", size=1, parent_picks=[0],
                   num_full_eval=500, draws=[0.9] * 6)
    assert abs(led["threshold"] - 0.60) < 1e-12   # 池 = [0.60, 0.60, 0.50]

    # initial 不满评 -> 池 = [0.60, 0.50]，降序第二是 0.50
    rows = [_c("initial", 0.60, submitted=10), _c("a", 0.50, submitted=500)]
    led = _compare(solve, rows, [], method="random", size=1, parent_picks=[0],
                   num_full_eval=500, draws=[0.9] * 6)
    assert abs(led["threshold"] - 0.50) < 1e-12

    # 池里只剩一个元素 -> 用它自己，再与 0.4 取大
    rows = [_c("initial", 0.30, submitted=10)]
    led = _compare(solve, rows, [], method="random", size=1, parent_picks=[0],
                   num_full_eval=500, draws=[0.9] * 6)
    assert abs(led["threshold"] - 0.4) < 1e-12


def check_threshold_floor(solve):
    """门槛至少 0.4；满评判据是 `submitted >= 0.9 * num_full_eval`（450 过、449 不过）。"""
    rows = [_c("initial", 0.10), _c("a", 0.90, submitted=450),
            _c("b", 0.80, submitted=450)]
    led = _compare(solve, rows, [], method="random", size=1, parent_picks=[0],
                   num_full_eval=500, draws=[0.9] * 6)
    assert abs(led["threshold"] - 0.80) < 1e-12   # 池 = [0.10, 0.90, 0.80]

    # a 差一题就掉出池子，池里只剩 0.10 与 0.80 -> 第二是 0.10 -> 夹到 0.4
    rows = [_c("initial", 0.10), _c("a", 0.90, submitted=449),
            _c("b", 0.80, submitted=450)]
    led = _compare(solve, rows, [], method="random", size=1, parent_picks=[0],
                   num_full_eval=500, draws=[0.9] * 6)
    assert abs(led["threshold"] - 0.4) < 1e-12


def check_boundary_equalities(solve):
    """两处 `>=` 的等号都算数：空补丁占比恰好 0.1、入档分恰好等于门槛。"""
    # empty=2, total=20 -> 2 >= 2.0 命中第一道
    rows = [_c("initial", 0.3, resolved=9, unresolved=9, empty=2)]
    led = _compare(solve, rows, [], method="random", size=1, parent_picks=[0],
                   draws=[0.1, 0.9])
    assert led["entries"][0]["entry"] == "solve_empty_patches"
    assert led["draws_used"] == 1

    # 初始 0.5、余量 0.25 -> 门槛恰为 0.25，子代 0.25 要留下
    rows = [_c("initial", 0.50)]
    kids = [_c("edge", 0.25), _c("low", 0.24)]
    led = _compare(solve, rows, kids, method="random", size=1, parent_picks=[0],
                   update="keep_better", noise_leeway=0.25, draws=[0.9] * 6)
    assert led["archive"] == ["initial", "edge"]


def check_scalar_boundaries(solve):
    """size=0、硬币取到 1.0、bool 当实数：三条都要拦。"""
    rows = [_c("initial", 0.3)]
    try:
        solve(rows, [], method="random", size=0, parent_picks=[], draws=[])
    except ValueError:
        pass
    else:
        raise AssertionError("size=0 应当报错")
    try:
        solve(rows, [], method="random", size=1, parent_picks=[0],
              draws=[1.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("coins 取 1.0 应当报错")
    try:
        solve(rows, [], method="random", size=1, parent_picks=[0],
              draws=[0.9, 0.9], noise_leeway=True)
    except ValueError:
        pass
    else:
        raise AssertionError("noise_leeway=True 应当报错")


def check_draws_exhausted(solve):
    """硬币用光必须报错，不能默默用最后一个。"""
    rows = [_c("initial", 0.3, unresolved=3)]
    try:
        solve(rows, [], method="random", size=1, parent_picks=[0], draws=[0.9])
    except ValueError:
        pass
    else:
        raise AssertionError("draws 不够时应当报错")


def check_validation(solve):
    """标量与结构校验。"""
    rows = [_c("initial", 0.3)]
    for kw in ({"size": 0}, {"size": True}, {"method": "nope"}, {"update": "nope"},
               {"num_issues": -1}, {"num_full_eval": 0},
               {"noise_leeway": float("nan")}, {"baseline": "nope"},
               {"draws": [1.0]}, {"draws": [-0.1]}):
        base = {"method": "random", "size": 1, "parent_picks": [0], "draws": [0.9, 0.9]}
        base.update(kw)
        try:
            solve(rows, [], **base)
        except ValueError:
            pass
        else:
            raise AssertionError("%r 应当报错" % (kw,))
    # 存档缺 initial / id 重复 / children id 重复 / 欠 parent_picks
    for bad in ([_c("a", 0.3)], [_c("initial", 0.3), _c("initial", 0.4)]):
        try:
            solve(bad, [], method="random", size=1, parent_picks=[0],
                  draws=[0.9] * 6)
        except ValueError:
            pass
        else:
            raise AssertionError("%r 应当报错" % (bad,))
    try:
        solve(rows, [_c("k", 0.5), _c("k", 0.6)], method="random", size=1,
              parent_picks=[0], draws=[0.9] * 6)
    except ValueError:
        pass
    else:
        raise AssertionError("children 重复 id 应当报错")
    try:
        solve(rows, [], method="score_prop", size=2, parent_picks=[0],
              draws=[0.9] * 8)
    except ValueError:
        pass
    else:
        raise AssertionError("parent_picks 数量不符应当报错")


def check_against_oracle(solve):
    """两代混合场景整体对拍。"""
    rows = [_c("initial", 0.30, resolved=40, unresolved=20, empty=0, submitted=60),
            _c("a", 0.42, resolved=45, unresolved=12, empty=3, submitted=60,
               parent="initial"),
            _c("b", 0.51, resolved=48, unresolved=10, empty=2, submitted=60,
               parent="initial"),
            _c("c", 0.61, resolved=50, unresolved=8, empty=2, submitted=60,
               parent="a"),
            _c("d", 0.20, resolved=10, unresolved=40, empty=10, submitted=60,
               parent="b", ctx=True)]
    kids = [_c("e", 0.65, resolved=52, unresolved=6, empty=2, submitted=60,
               parent="c"),
            _c("f", 0.10, resolved=5, unresolved=50, empty=5, submitted=60,
               parent="a"),
            _c("g", 0.70, resolved=0, unresolved=0, empty=60, submitted=60,
               parent="b")]
    led = _compare(solve, rows, kids, method="score_child_prop", size=3,
                   parent_picks=[4, 2, 0], update="keep_better",
                   noise_leeway=0.1, num_issues=60, num_full_eval=60,
                   draws=[0.9, 0.9, 0.9, 0.9, 0.2, 0.9, 0.5, 0.9, 0.9, 0.5])
    assert led["parents"] == ["d", "b", "initial"]
    # d：empty=10/60 过第一道占比（6）但硬币 0.9 不中，第二、三道也不中 -> 抽未解决项
    assert led["entries"][0] == {"parent": "d", "entry": "unresolved", "index": 36}
    # b：empty=2 < 6 短路，第二枚硬币 0.2 直接命中
    assert led["entries"][1] == {"parent": "b", "entry": "solve_stochasticity",
                                 "index": None}
    # initial：三枚硬币都不中，抽未解决项
    assert led["entries"][2] == {"parent": "initial", "entry": "unresolved",
                                 "index": 10}
    assert led["draws_used"] == 7
    assert led["compiled"] == ["e", "f"]
    assert led["rejected"] == [{"id": "g", "reason": "no_patch"}]
    # keep_better 门槛 0.30-0.1=0.20，e(0.65) 与 f(0.10) 只有 e 够
    assert led["archive"] == ["initial", "a", "b", "c", "d", "e"]


def run_examples(solve):
    print(_compare(solve, [_c("initial", 0.2), _c("a", 0.9), _c("b", 0.5)],
                   [], method="best", size=2, draws=[0.9] * 6)["parents"])
    print(_compare(solve, BASE, [_c("k", 0.6)], method="random", size=1,
                   parent_picks=[0], draws=[0.9] * 6)["archive"])


check_best_takes_worst.example = True
check_compile_filter.example = True
