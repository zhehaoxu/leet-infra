import torch

TOL = 1e-12


def _close(a, b):
    if a == b:          # 含 -inf == -inf
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


def _sm(logits, temperature):
    t = torch.tensor(logits, dtype=torch.float64) / temperature
    return [float(x) for x in torch.softmax(t, dim=-1)]


def _prog_probs(programs):
    lengths = [p["size"] for p in programs]
    lo, hi = min(lengths), max(lengths)
    norm = [(x - lo) / (hi + 1e-6) for x in lengths]
    return _sm([-x for x in norm], 1.0)


def _py(events, num_islands=10, functions_per_prompt=2,
        cluster_sampling_temperature_init=0.1,
        cluster_sampling_temperature_period=30000, function_name="f"):
    """独立参考实现：簇表用 (signature -> [score, programs]) 的普通 dict。"""
    pools = [dict() for _ in range(num_islands)]
    order = [[None] for _ in range(num_islands)]  # order[i][0] 是 signature 列表
    for i in range(num_islands):
        order[i] = []
    counts = [0] * num_islands
    best = [-float("inf")] * num_islands
    best_p = [None] * num_islands
    best_t = [None] * num_islands
    regs, prompts, resets = [], [], []

    def put(program, island_id, scores, via):
        sig = tuple(scores[k] for k in sorted(scores))
        score = scores[list(scores)[-1]]
        if sig in pools[island_id]:
            pools[island_id][sig][1].append(program)
            new = False
        else:
            pools[island_id][sig] = [score, [program]]
            order[island_id].append(sig)
            new = True
        counts[island_id] += 1
        imp = score > best[island_id]
        if imp:
            best[island_id] = score
            best_p[island_id] = program
            best_t[island_id] = scores
        regs.append({"program": program["id"], "island": island_id,
                     "cluster": order[island_id].index(sig), "new_cluster": new,
                     "score": score, "improved": imp, "via": via})

    for ev in events:
        if ev["kind"] == "register":
            scores = {k: float(v) for k, v in ev["scores"].items()}
            prog = {"id": ev["program"]["id"], "doc": ev["program"]["doc"],
                    "size": ev["program"]["size"]}
            if ev.get("island") is None:
                for i in range(num_islands):
                    put(prog, i, scores, "broadcast")
            else:
                put(prog, ev["island"], scores, "event")
        elif ev["kind"] == "prompt":
            iid = ev["island"]
            sigs = order[iid]
            total = len(sigs)
            take = min(total, functions_per_prompt)
            temp = cluster_sampling_temperature_init * (
                1 - (counts[iid] % cluster_sampling_temperature_period)
                / cluster_sampling_temperature_period)
            scores = [pools[iid][s][0] for s in sigs]
            probs = _sm(scores, temp)
            chosen = list(zip(ev["clusters"], ev["programs"]))
            seq = sorted(range(take), key=lambda i: (scores[chosen[i][0]], i))
            versions = []
            for slot, idx in enumerate(seq):
                c, p = chosen[idx]
                src = pools[iid][sigs[c]][1][p]
                doc = src["doc"] if slot == 0 else (
                    "Improved version of `%s_v%d`." % (function_name, slot - 1))
                versions.append({"name": "%s_v%d" % (function_name, slot),
                                 "docstring": doc, "source": src["id"]})
            prompts.append({
                "island": iid, "temperature": temp, "num_programs": counts[iid],
                "cluster_scores": scores, "probabilities": probs,
                "chosen_clusters": [c for c, _ in chosen],
                "chosen_programs": [p for _, p in chosen],
                "program_probs": [_prog_probs(pools[iid][sigs[c]][1])
                                  for c, _ in chosen],
                "sorted_clusters": [chosen[i][0] for i in seq],
                "versions": versions, "version_generated": take + 1,
                "header": {"name": "%s_v%d" % (function_name, take),
                           "docstring": "Improved version of `%s_v%d`."
                                        % (function_name, take - 1)}})
        else:
            need = num_islands // 2
            ranked = sorted(range(num_islands), key=lambda i: (best[i], i))
            reset_ids, keep_ids = ranked[:need], ranked[need:]
            for iid, founder in zip(reset_ids, ev["founders"]):
                pools[iid] = dict()
                order[iid] = []
                counts[iid] = 0
                best[iid] = -float("inf")
                best_p[iid] = None
                best_t[iid] = None
                put(dict(best_p[founder]), iid, dict(best_t[founder]), "reset")
            resets.append({"reset": reset_ids, "keep": keep_ids,
                           "founders": {i: f for i, f in zip(reset_ids, ev["founders"])}})

    clusters = []
    for iid in range(num_islands):
        row = []
        for sig in order[iid]:
            score, progs = pools[iid][sig]
            row.append({"signature": list(sig), "score": score,
                        "programs": [p["id"] for p in progs]})
        clusters.append(row)
    return {"clusters": clusters,
            "cluster_count": [len(order[i]) for i in range(num_islands)],
            "num_programs": list(counts), "best_score": list(best),
            "best_program": list(best_p), "registrations": regs,
            "prompts": prompts, "resets": resets}


def _compare(solve, events, **kw):
    got = solve([dict(e) for e in events], **kw)
    ref = _py(events, **kw)
    _eq(got, ref)
    return got


def _prog(pid, size=10, doc=None):
    return {"id": pid, "doc": doc if doc is not None else "does %s" % pid,
            "size": size}


def _reg(pid, scores, island=0, size=10):
    return {"kind": "register", "program": _prog(pid, size), "scores": scores,
            "island": island}


def check_signature_sorted_reduce_last(solve):
    """簇键按测试名升序、簇分取**最后插入**的测试（不是最大、不是字典序末位）。"""
    # 插入序 t2, t1 -> 簇分取 t1=2.0；键序 (t1, t2) -> signature [2.0, 1.0]
    led = _compare(solve, [_reg("p", {"t2": 1.0, "t1": 2.0})])
    assert led["clusters"][0][0]["signature"] == [2.0, 1.0]
    assert led["clusters"][0][0]["score"] == 2.0
    assert led["best_score"][0] == 2.0

    # 反过来插入 -> 同一 signature，但簇分变成 1.0
    led = _compare(solve, [_reg("p", {"t1": 2.0, "t2": 1.0})])
    assert led["clusters"][0][0]["signature"] == [2.0, 1.0]
    assert led["clusters"][0][0]["score"] == 1.0
    assert led["best_score"][0] == 1.0


def check_cluster_score_frozen(solve):
    """同 signature 的后来者只进程序表，不改簇分（哪怕它的 reduce 分更高）。"""
    rows = [_reg("a", {"t2": 9.0, "t1": 1.0}), _reg("b", {"t1": 1.0, "t2": 9.0})]
    led = _compare(solve, rows)
    assert led["cluster_count"][0] == 1
    cell = led["clusters"][0][0]
    assert cell["score"] == 1.0          # 首个成员的 reduce
    assert cell["programs"] == ["a", "b"]
    # 但 best 用「本次注册的 reduce 分」比较：b 的 reduce=9.0 反而成了岛内最优
    assert led["best_score"][0] == 9.0
    assert led["best_program"][0]["id"] == "b"
    assert [r["improved"] for r in led["registrations"]] == [True, True]


def check_broadcast_register(solve):
    """island=None 的种子程序广播到所有岛，每岛都建簇、都记一次 num_programs。"""
    led = _compare(solve, [_reg("seed", {"t": 1.0}, island=None)], num_islands=3)
    assert led["cluster_count"] == [1, 1, 1]
    assert led["num_programs"] == [1, 1, 1]
    assert [r["via"] for r in led["registrations"]] == ["broadcast"] * 3
    assert led["best_score"] == [1.0, 1.0, 1.0]


def check_best_strictly_greater(solve):
    """岛内最优只在**严格**更高时替换，同分保留先到者。"""
    rows = [_reg("a", {"t": 5.0}), _reg("b", {"t": 5.0}), _reg("c", {"t": 4.0})]
    led = _compare(solve, rows)
    assert [r["improved"] for r in led["registrations"]] == [True, False, False]
    assert led["best_program"][0]["id"] == "a"


def check_temperature_schedule(solve):
    """温度随该岛累计程序数线性衰减，`n % period` 归零时回到初值。"""
    rows = [_reg("p%d" % i, {"t": float(i)}) for i in range(4)]
    probe = {"kind": "prompt", "island": 0, "clusters": [0], "programs": [0]}
    led = _compare(solve, rows + [probe], functions_per_prompt=1,
                   cluster_sampling_temperature_init=0.3,
                   cluster_sampling_temperature_period=3)
    rec = led["prompts"][0]
    assert rec["num_programs"] == 4
    # 4 % 3 == 1 -> 0.3 * (1 - 1/3)
    assert abs(rec["temperature"] - 0.2) < 1e-12

    rows = [_reg("p%d" % i, {"t": float(i)}) for i in range(3)]
    led = _compare(solve, rows + [probe], functions_per_prompt=1,
                   cluster_sampling_temperature_init=0.3,
                   cluster_sampling_temperature_period=3)
    assert led["prompts"][0]["num_programs"] == 3
    assert abs(led["prompts"][0]["temperature"] - 0.3) < 1e-12  # 3 % 3 == 0


def check_softmax_over_clusters(solve):
    """簇概率是分数 / 温度的 softmax；低温时明显尖锐。"""
    rows = [_reg("lo", {"t": 0.0}), _reg("hi", {"t": 2.0})]
    probe = {"kind": "prompt", "island": 0, "clusters": [0], "programs": [0]}
    led = _compare(solve, rows + [probe], functions_per_prompt=1,
                   cluster_sampling_temperature_init=0.5)
    rec = led["prompts"][0]
    probs = rec["probabilities"]
    assert abs(sum(probs) - 1.0) < 1e-12
    assert probs[1] > probs[0]
    # 注意温度已被 (num_programs % period) 削过一点点，比值要按实际温度算
    gap = 2.0 / rec["temperature"]
    assert abs(probs[1] / probs[0] - torch.exp(torch.tensor(gap, dtype=torch.float64)).item()) < 1e-9
    assert rec["temperature"] < 0.5


def check_take_min_clusters(solve):
    """prompt 里的函数条数取 min(簇数, functions_per_prompt)，版本号随之为 n+1。"""
    rows = [_reg("only", {"t": 1.0}, size=7)]
    probe = {"kind": "prompt", "island": 0, "clusters": [0], "programs": [0]}
    led = _compare(solve, rows + [probe], functions_per_prompt=4)
    rec = led["prompts"][0]
    assert len(rec["versions"]) == 1
    assert rec["version_generated"] == 2
    assert rec["header"] == {"name": "f_v1",
                            "docstring": "Improved version of `f_v0`."}
    assert rec["versions"][0]["name"] == "f_v0"
    assert rec["versions"][0]["docstring"] == "does only"   # v0 保留原 docstring

    rows = [_reg("a", {"t": 1.0}), _reg("b", {"t": 3.0})]
    led = _compare(solve, rows + [{"kind": "prompt", "island": 0,
                                   "clusters": [0, 1], "programs": [0, 0]}],
                   functions_per_prompt=4)
    rec = led["prompts"][0]
    assert rec["version_generated"] == 3
    assert [v["name"] for v in rec["versions"]] == ["f_v0", "f_v1"]
    assert rec["header"]["name"] == "f_v2"


def check_versions_sorted_by_score(solve):
    """版本号按**簇分升序**重排（同分按取样位置），取样序与版本序可以不同。"""
    rows = [_reg("c0", {"t": 5.0}), _reg("c1", {"t": 1.0}), _reg("c2", {"t": 3.0})]
    probe = {"kind": "prompt", "island": 0, "clusters": [0, 2], "programs": [0, 0]}
    led = _compare(solve, rows + [probe], functions_per_prompt=2)
    rec = led["prompts"][0]
    assert rec["chosen_clusters"] == [0, 2]
    assert rec["sorted_clusters"] == [2, 0]          # 3.0 < 5.0
    assert [v["source"] for v in rec["versions"]] == ["c2", "c0"]
    assert rec["versions"][1]["docstring"] == "Improved version of `f_v0`."

    # 同分时按取样位置升序
    # 两个不同签名但簇分同为 2.0：y 的插入序是 t2, t1，reduce 取 t1
    rows = [_reg("x", {"t": 2.0}), _reg("y", {"t2": 0.0, "t1": 2.0})]
    led = _compare(solve, rows + [{"kind": "prompt", "island": 0,
                                   "clusters": [1, 0], "programs": [0, 0]}],
                   functions_per_prompt=2)
    assert led["prompts"][0]["sorted_clusters"] == [1, 0]


def check_program_probs_denominator(solve):
    """簇内取样概率的归一化分母是 `max + 1e-6`（不是 `max - min`）。"""
    rows = [_reg("short", {"t": 1.0}, size=10), _reg("long", {"t": 1.0}, size=20)]
    probe = {"kind": "prompt", "island": 0, "clusters": [0], "programs": [0]}
    led = _compare(solve, rows + [probe], functions_per_prompt=1)
    probs = led["prompts"][0]["program_probs"][0]
    assert len(probs) == 2
    assert probs[0] > probs[1]
    # 归一化后 -x = [0, -10/20.000001]；softmax 后比值为 exp(10/20.000001)
    ratio = torch.exp(torch.tensor(-10.0 / 20.000001, dtype=torch.float64)).item()
    assert abs(probs[1] / probs[0] - ratio) < 1e-9

    # 长度全相同 -> 归一化全 0 -> 均匀
    rows = [_reg("a", {"t": 1.0}, size=4), _reg("b", {"t": 1.0}, size=4)]
    led = _compare(solve, rows + [probe], functions_per_prompt=1)
    assert abs(led["prompts"][0]["program_probs"][0][0] - 0.5) < 1e-12


def check_reset_weaker_half(solve):
    """重置按 (岛内最优分, 岛号) 升序取弱的一半；保留岛的相对顺序即排名。"""
    rows = [_reg("seed", {"t": 1.0}, island=None)]
    rows += [_reg("p%d" % i, {"t": float(i)}, island=i) for i in range(1, 4)]
    rows.append({"kind": "reset", "founders": [3, 3]})
    led = _compare(solve, rows, num_islands=4)
    rec = led["resets"][0]
    # 分数：0 岛 1.0、1 岛 1.0（种子）、2 岛 2.0、3 岛 3.0
    assert rec["reset"] == [0, 1]
    assert rec["keep"] == [2, 3]
    assert rec["founders"] == {0: 3, 1: 3}


def check_reset_founder_clone(solve):
    """重置后的岛 = 空岛 + 创始人最优程序克隆，最优分随克隆继承。"""
    rows = [_reg("seed", {"t": 1.0}, island=None)]
    rows.append(_reg("win", {"t": 7.0}, island=3))
    rows.append({"kind": "reset", "founders": [3, 3]})
    led = _compare(solve, rows, num_islands=4)
    for iid in (0, 1):
        assert led["cluster_count"][iid] == 1
        assert led["num_programs"][iid] == 1
        assert led["best_score"][iid] == 7.0
        assert led["best_program"][iid]["id"] == "win"
    # 保留岛不受影响
    assert led["num_programs"][3] == 2
    assert led["best_score"][3] == 7.0
    assert led["registrations"][-2]["via"] == "reset"


def check_reset_never_zero_islands(solve):
    """num_islands=1 时 `1 // 2 == 0`，重置不删任何岛（founders 必须为空）。"""
    led = _compare(solve, [_reg("a", {"t": 1.0}), {"kind": "reset", "founders": []}],
                   num_islands=1)
    assert led["resets"][0] == {"reset": [], "keep": [0], "founders": {}}
    assert led["cluster_count"] == [1]


def check_reset_rejects_bad_founder(solve):
    """创始人必须在保留岛里、且手上真有程序；数量必须等于 num_islands // 2。"""
    # 岛 0 有 a（1.0）、岛 3 有 win（9.0），岛 1/2 空 -> 重置 [1, 2]、保留 [0, 3]
    base = [_reg("a", {"t": 1.0}, island=0), _reg("win", {"t": 9.0}, island=3)]
    for founders in ([0], [1, 1], [2, 2], [0, 1]):
        try:
            solve([dict(e) for e in base] + [{"kind": "reset", "founders": list(founders)}],
                  num_islands=4)  # noqa: PT012
        except ValueError:
            pass
        else:
            raise AssertionError("应当拒绝 founders=%r" % (founders,))
    led = solve([dict(e) for e in base] + [{"kind": "reset", "founders": [0, 3]}],
                num_islands=4)
    assert led["resets"][0]["reset"] == [1, 2]
    assert led["best_program"][1]["id"] == "a"
    assert led["best_program"][2]["id"] == "win"

    # 创始人指向**被重置**的岛也要拒（那些岛手上有程序，单靠「有没有程序」拦不住）
    deep = [_reg("a", {"t": 0.0}, island=0), _reg("b", {"t": 0.5}, island=1),
            _reg("w", {"t": 8.0}, island=2), _reg("v", {"t": 9.0}, island=3)]
    try:
        solve([dict(e) for e in deep] + [{"kind": "reset", "founders": [0, 3]}],
              num_islands=4)
    except ValueError:
        pass
    else:
        raise AssertionError("应当拒绝指向被重置岛的创始人")

    # 保留岛手上没有程序也要拒
    lone = [_reg("a", {"t": 1.0}, island=0)]
    try:
        solve([dict(e) for e in lone] + [{"kind": "reset", "founders": [3, 0]}],
              num_islands=4)
    except ValueError:
        pass
    else:
        raise AssertionError("应当拒绝空手创始人")


def check_signed_zero_signature(solve):
    """`-0.0` 与 `0.0` 在 dict 里是同一个键，故落进同一簇（Python 语义，非 bug）。"""
    rows = [_reg("a", {"t": -0.0}), _reg("b", {"t": 0.0})]
    led = _compare(solve, rows)
    assert led["cluster_count"][0] == 1
    assert led["clusters"][0][0]["programs"] == ["a", "b"]


def check_prompt_rejects_empty_island(solve):
    """空岛没有簇，取提示词必须报错；簇 / 程序下标越界同样报错。"""
    rows = [_reg("a", {"t": 1.0}), _reg("b", {"t": 2.0})]
    for probe in ({"kind": "prompt", "island": 1, "clusters": [], "programs": []},
                  {"kind": "prompt", "island": 0, "clusters": [2], "programs": [0]},
                  {"kind": "prompt", "island": 0, "clusters": [0], "programs": [1]},
                  {"kind": "prompt", "island": 0, "clusters": [0, 0], "programs": [0, 0]}):
        try:
            solve([dict(e) for e in rows] + [dict(probe)], num_islands=2,
                  functions_per_prompt=1)
        except ValueError:
            pass
        else:
            raise AssertionError("应当拒绝 %r" % (probe,))


def check_validation(solve):
    """标量校验：bool 不算整数、非有限分数、未知 kind、空岛广播都要报错。"""
    for bad in (True, 0, -3, 2.0):
        try:
            solve([], num_islands=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("num_islands=%r 应当报错" % (bad,))
    for kw in ({"functions_per_prompt": 0}, {"functions_per_prompt": True},
               {"cluster_sampling_temperature_init": 0.0},
               {"cluster_sampling_temperature_init": float("nan")},
               {"cluster_sampling_temperature_period": 0},
               {"cluster_sampling_temperature_period": True}):
        try:
            solve([], **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("%r 应当报错" % (kw,))
    # 未知 kind 不能顺着 reset 分支溜过去
    try:
        solve([_reg("a", {"t": 5.0}, island=1),
               {"kind": "reset_typo", "founders": [1]}], num_islands=2)
    except ValueError:
        pass
    else:
        raise AssertionError("未知 kind 应当报错")
    for rows in ([{"kind": "nope"}],
                 [_reg("a", {})],
                 [_reg("a", {"t": float("inf")})],
                 [_reg("a", {"t": True})],
                 [{"kind": "register", "program": {"id": "", "doc": "x", "size": 1},
                   "scores": {"t": 1.0}, "island": None}]):
        try:
            solve([dict(r) for r in rows])
        except ValueError:
            pass
        else:
            raise AssertionError("%r 应当报错" % (rows,))


def check_against_oracle(solve):
    """一串混合事件（广播种子 + 三岛注册 + 两次取样 + 重置）整体对拍。"""
    rows = [_reg("seed", {"t1": 1.0, "t2": 0.0}, island=None, size=30)]
    rows += [
        _reg("a", {"t1": 2.0, "t2": 5.0}, island=0, size=12),
        _reg("b", {"t2": 3.0, "t1": 2.0}, island=0, size=40),
        _reg("c", {"t1": 2.0, "t2": 5.0}, island=0, size=8),
        _reg("d", {"t": 4.0}, island=1, size=20),
        _reg("e", {"t": 4.0}, island=1, size=9),
        {"kind": "prompt", "island": 0, "clusters": [1, 2], "programs": [1, 0]},
        {"kind": "prompt", "island": 1, "clusters": [1, 1], "programs": [0, 1]},
        _reg("f", {"t": -1.0}, island=2, size=5),
    ]
    led = _compare(solve, rows, num_islands=3, functions_per_prompt=2,
                   cluster_sampling_temperature_init=0.2,
                   cluster_sampling_temperature_period=2, function_name="solve_it")
    assert led["cluster_count"] == [3, 2, 2]
    # 岛 0：簇 0 是广播种子，簇 1 是签名 (2,5)（a 先进、c 后进），簇 2 是 (2,3)
    assert led["clusters"][0][0]["programs"] == ["seed"]
    assert led["clusters"][0][1]["programs"] == ["a", "c"]
    assert led["clusters"][0][1]["score"] == 5.0
    assert led["clusters"][0][2]["score"] == 2.0   # b 的插入序是 t2,t1 -> 取 t1
    assert [v["source"] for v in led["prompts"][0]["versions"]] == ["b", "c"]
    assert led["prompts"][0]["versions"][1]["name"] == "solve_it_v1"
    # 岛 1 累计 3 个程序，period=2 -> 3 % 2 == 1 -> 0.2 * 0.5
    assert abs(led["prompts"][1]["temperature"] - 0.1) < 1e-12


def run_examples(solve):
    print(_compare(solve, [_reg("p", {"t2": 1.0, "t1": 2.0})])["clusters"][0])
    print(_compare(solve, [_reg("a", {"t": 1.0}), _reg("b", {"t": 3.0}),
                           {"kind": "prompt", "island": 0,
                            "clusters": [1, 0], "programs": [0, 0]}],
                   functions_per_prompt=2)["prompts"][0]["header"])


check_signature_sorted_reduce_last.example = True
check_softmax_over_clusters.example = True
