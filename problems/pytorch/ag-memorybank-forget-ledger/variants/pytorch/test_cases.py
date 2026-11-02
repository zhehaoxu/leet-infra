import copy
import math
import random

# ============ 独立 oracle（与 solution 不同构的实现） ============


def _ocurve(t, s):
    # 官方优先级：exp((-t / 5) * S)
    return math.exp((-t / 5) * s)


def _odate(s):
    import datetime
    return datetime.datetime.strptime(s, "%Y-%m-%d")


_KW = {"cn": {"mem": "时间{d}的对话内容：", "u": "[|用户|]：", "a": "[|AI伴侣|]：",
              "s": "时间{d}的对话总结为：{c}"},
       "en": {"mem": "Conversation content on {d}:", "u": "[|User|]:",
              "a": "[|AI|]:",
              "s": "The summary of the conversation on {d} is: {c}"}}


def _oracle(bank, now_date, draws, recalled_ids=(), user=None,
            cur_recall_date=None, language="cn"):
    import datetime
    if not isinstance(bank, dict):
        raise ValueError("bank must be a dict")
    _odate(now_date)
    if not isinstance(draws, (list, tuple)):
        raise ValueError("draws must be a list")
    if language not in _KW:
        raise ValueError("language must be 'cn' or 'en'")
    kw = _KW[language]
    pos = [0]

    def take():
        if pos[0] >= len(draws):
            raise ValueError("not enough draws")
        v = draws[pos[0]]
        pos[0] += 1
        return v

    out, dlg_acc, doc_acc = {}, [], []
    for name, mem in bank.items():
        if not isinstance(mem, dict) or "history" not in mem:
            continue
        import copy as _c
        um = _c.deepcopy(mem)
        out[name] = um
        for day in list(um["history"].keys()):
            _odate(day)
            rows = um["history"][day]
            killed = []
            for i, old in enumerate(rows):
                dlg = old if isinstance(old, dict) \
                    else {"query": old[0], "response": old[1]}
                if not isinstance(old, dict):
                    rows[i] = dlg
                s = dlg.get("memory_strength", 1)
                lr = dlg.get("last_recall_date", day)
                mid = dlg.get("memory_id", f"{name}_{day}_{i}")
                meta = {"memory_strength": s, "memory_id": mid,
                        "last_recall_date": lr, "source": mid}
                rows[i].update(meta)
                dd = (_odate(now_date) - _odate(lr)).days
                ret = _ocurve(dd, s)
                d = take()
                gone = d > ret
                dlg_acc.append({"user": name, "date": day, "index": i,
                                "memory_id": mid, "strength": s,
                                "days_diff": dd, "retention": ret,
                                "draw": d, "forgotten": gone})
                if gone:
                    killed.append(i)
                else:
                    doc_acc.append(kw["mem"].format(d=day)
                                   + f'{kw["u"]} {dlg["query"].strip()}; '
                                   + f'{kw["a"]} {dlg["response"].strip()}')
            for i in sorted(killed, reverse=True):
                rows.pop(i)
            if len(rows) == 0:
                del um["history"][day]
                if "summary" not in um or day not in um["summary"]:
                    raise ValueError(f"summary missing for emptied date {day}")
                del um["summary"][day]
        if "summary" in um:
            for day in list(um["summary"].keys()):
                cell = um["summary"][day]
                if not isinstance(cell, dict):
                    cell = {"content": cell}
                    um["summary"][day] = cell
                meta = {"memory_strength": cell.get("memory_strength", 1),
                        "memory_id": f"{name}_{day}_summary",
                        "last_recall_date": cell.get("last_recall_date", day),
                        "source": f"{name}_{day}_summary"}
                cell.update(meta)
                doc_acc.append(kw["s"].format(d=day, c=cell["content"]))

    ups = []
    if recalled_ids:
        if user is None or cur_recall_date is None:
            raise ValueError("user and cur_recall_date required")
        if user not in out:
            raise ValueError("unknown user for recall")
        for rid in recalled_ids:
            if not isinstance(rid, str) or len(rid.split("_")) < 2:
                raise ValueError("malformed memory id")
            day = rid.split("_")[1]
            hit = {"memory_id": rid, "matched": False}
            if day not in out[user]["history"]:
                raise ValueError(f"recalled date not in history: {day!r}")
            for m in out[user]["history"][day]:
                if m["memory_id"] == rid:
                    hit["matched"] = True
                    hit["strength_after"] = m["memory_strength"] + 1
                    m["memory_strength"] += 1
                    m["last_recall_date"] = cur_recall_date
                    break
            ups.append(hit)
    return {"bank": out, "dialogs": dlg_acc, "docs": doc_acc, "updates": ups}


# ============ 比较工具 ============

def _eq(a, b, path="root"):
    if isinstance(a, float) or isinstance(b, float):
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)), \
            f"{path}: {a!r} vs {b!r}"
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


def _dc(v):
    if isinstance(v, dict):
        return {k: _dc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_dc(x) for x in v]
    return v


# oracle 调用统一封装（签名一致）
def _both(solve, bank, now_date, draws, **kw):
    got = solve(_dc(bank), now_date, _dc(draws), **kw)
    exp = _oracle(_dc(bank), now_date, _dc(draws), **kw)
    _eq(got, exp)
    return got


# ============ 手算锚点 ============

def check_curve_precedence(solve):
    """官方 exp(-t/5*S) = exp(-t*S/5)：S 越大衰减越快（与论文意图相反）。"""
    bank = {"u": {"history": {"2026-01-01": [
        {"query": "q", "response": "r", "memory_strength": 7,
         "last_recall_date": "2025-12-31"}]},
        "summary": {"2026-01-01": "s"}}}
    led = _both(solve, bank, "2026-01-01", [0.5])
    d = led["dialogs"][0]
    assert d["days_diff"] == 1
    assert math.isclose(d["retention"], math.exp(-1.4), rel_tol=0,
                        abs_tol=1e-12), d["retention"]   # exp(-t*S/5)
    assert not math.isclose(d["retention"], math.exp(-1 / 35), abs_tol=1e-3)
    assert d["forgotten"] is True                   # 0.5 > 0.2466 → 忘
    assert d["memory_id"] == "u_2026-01-01_0"
    # 对话被遗忘 → 日期清空 → summary 同步 pop，摘要文档也不产
    assert led["docs"] == []


def check_basic_keep_and_docs(solve):
    """保留路径：docs 精确串（cn 双语前缀 + strip）、旧格式迁移写回。"""
    bank = {"u": {"history": {"2026-01-05": [
        ["  hi  ", " hello "],
        {"query": "a", "response": "b", "memory_strength": 2,
         "last_recall_date": "2026-01-03", "memory_id": "u_x_9"}]}}}
    led = _both(solve, bank, "2026-01-05", [1.0, 0.0])
    d0, d1 = led["dialogs"]
    assert d0["strength"] == 1 and d0["days_diff"] == 0
    assert d0["retention"] == 1.0
    assert d0["forgotten"] is False             # draw==1.0 恰等不遗忘（严格 >）
    assert d1["days_diff"] == 2
    assert math.isclose(d1["retention"], math.exp(-2 * 2 / 5), abs_tol=1e-12)
    assert d1["forgotten"] is False
    assert led["docs"] == ["时间2026-01-05的对话内容：[|用户|]： hi; "
                           "[|AI伴侣|]： hello",
                           "时间2026-01-05的对话内容：[|用户|]： a; "
                           "[|AI伴侣|]： b"]
    # 旧格式迁移：bank 中该条变 dict 且补齐元数据
    dlg = led["bank"]["u"]["history"]["2026-01-05"][0]
    assert dlg["query"] == "  hi  " and dlg["response"] == " hello "
    assert dlg["memory_strength"] == 1
    assert dlg["memory_id"] == "u_2026-01-05_0"
    assert dlg["last_recall_date"] == "2026-01-05"


def check_multi_forget_pop_order(solve):
    """同日多条遗忘：官方 forget_ids 逆序 pop（升序 pop 会串位）。"""
    bank = {"u": {"history": {"2026-01-01": [
        {"query": "a", "response": "r", "memory_strength": 100},
        {"query": "b", "response": "r", "memory_strength": 1},
        {"query": "c", "response": "r", "memory_strength": 100}]}}}
    # 2026-01-06: days=5 → S=1 保留 exp(-1)≈0.368, S=100 → exp(-100)≈0
    led = _both(solve, bank, "2026-01-06", [0.0, 0.5, 0.0])
    assert [d["forgotten"] for d in led["dialogs"]] == [False, True, False]
    hist = led["bank"]["u"]["history"]["2026-01-01"]
    assert [m["query"] for m in hist] == ["a", "c"]
    # 全部遗忘 → 日期从 history 与 summary 同步删除
    bank2 = {"u": {"history": {"2026-01-01": [
        {"query": "a", "response": "r", "memory_strength": 100}]},
        "summary": {"2026-01-01": "s1", "2026-01-02": "s2"}}}
    led2 = _both(solve, bank2, "2026-01-06", [0.9])
    assert led2["bank"]["u"]["history"] == {}
    assert led2["bank"]["u"]["summary"] == {
        "2026-01-02": {"content": "s2",
                       "memory_id": "u_2026-01-02_summary",
                       "memory_strength": 1,
                       "last_recall_date": "2026-01-02",
                       "source": "u_2026-01-02_summary"}}
    # summary 缺失被清日期 → ValueError（官方 KeyError）
    bank3 = {"u": {"history": {"2026-01-01": [
        {"query": "a", "response": "r", "memory_strength": 100}]}}}
    try:
        solve(bank3, "2026-01-06", [0.9])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing summary")


def check_summary_accounts(solve):
    """摘要不参与遗忘、非 dict 包 content、metadata 写回、docs 追加。"""
    bank = {"u": {"history": {"2026-01-01": [
        {"query": "q", "response": "r", "memory_strength": 100}]},
        "summary": {"2026-01-01": {"content": "sum", "memory_strength": 3,
                                   "last_recall_date": "2026-01-02"}}}}
    led = _both(solve, bank, "2026-01-06", [0.0])
    assert len(led["dialogs"]) == 1                    # 摘要不消耗 draw
    assert led["docs"][-1] == "时间2026-01-01的对话总结为：sum"
    cell = led["bank"]["u"]["summary"]["2026-01-01"]
    assert cell["memory_id"] == "u_2026-01-01_summary"
    assert cell["last_recall_date"] == "2026-01-02"
    # 非 dict 摘要被包成 {'content': ...}
    bank2 = {"u": {"history": {"2026-01-01": [
        {"query": "q", "response": "r", "memory_strength": 100}]},
        "summary": {"2026-01-01": "raw"}}}
    led2 = _both(solve, bank2, "2026-01-06", [0.0])
    assert led2["bank"]["u"]["summary"]["2026-01-01"]["content"] == "raw"


def check_no_history_user_dropped(solve):
    """无 history 键的用户整体不进结果库（官方 continue 位置 quirk）。"""
    bank = {"a": {"summary": {"2026-01-01": "x"}},
            "b": {"history": {"2026-01-01": [
                {"query": "q", "response": "r", "memory_strength": 1}]}}}
    led = _both(solve, bank, "2026-01-06", [0.0])
    assert set(led["bank"].keys()) == {"b"}   # a 被整体丢弃
    # en 语言前缀
    bank2 = {"b": {"history": {"2026-01-01": [
        {"query": "q", "response": "r", "memory_strength": 100}]}}}
    led2 = _both(solve, bank2, "2026-01-06", [0.0], language="en")
    assert led2["docs"] == [
        "Conversation content on 2026-01-01:[|User|]: q; [|AI|]: r"]


def check_update_accounts(solve):
    """检索强化：命中 +1 且刷新 last_recall_date、只改第一个匹配、
    已被遗忘的记忆无法命中（按 id 内日期定位）。"""
    bank = {"alice": {"history": {"2026-01-01": [
        {"query": "q1", "response": "r", "memory_strength": 2,
         "memory_id": "alice_2026-01-01_0"},
        {"query": "q2", "response": "r", "memory_strength": 2,
         "memory_id": "alice_2026-01-01_1"}]}}}
    led = _both(solve, bank, "2026-01-06", [0.0, 0.0],
                recalled_ids=["alice_2026-01-01_1"],
                user="alice", cur_recall_date="2026-01-07")
    hist = led["bank"]["alice"]["history"]["2026-01-01"]
    assert hist[0]["memory_strength"] == 2
    assert hist[1]["memory_strength"] == 3
    assert hist[1]["last_recall_date"] == "2026-01-07"
    assert led["updates"] == [{"memory_id": "alice_2026-01-01_1",
                               "matched": True, "strength_after": 3}]
    # 遗忘后的记忆：整日被清空 → id 定位日期不存在 → ValueError（官方 KeyError）
    bank2 = {"alice": {"history": {"2026-01-01": [
        {"query": "q", "response": "r", "memory_strength": 100,
         "memory_id": "alice_2026-01-01_0"}]}}}
    try:
        solve(_dc(bank2), "2026-01-06", [0.9],
              recalled_ids=["alice_2026-01-01_0"],
              user="alice", cur_recall_date="2026-01-07")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for emptied date id")
    # 同日期存在但 id 不匹配 → matched False（官方循环不命中即静默跳过）
    bank4 = {"alice": {"history": {"2026-01-01": [
        {"query": "q", "response": "r", "memory_strength": 1}]}},
        "2026-01-06": None}
    del bank4["2026-01-06"]
    led4 = _both(solve, bank4, "2026-01-06", [0.0],
                 recalled_ids=["alice_2026-01-01_9"],
                 user="alice", cur_recall_date="2026-01-07")
    assert led4["updates"][0]["matched"] is False
    # 重复 id：只改第一个
    bank3 = {"alice": {"history": {"2026-01-01": [
        {"query": "a", "response": "r", "memory_strength": 1,
         "memory_id": "alice_2026-01-01_0"},
        {"query": "b", "response": "r", "memory_strength": 1,
         "memory_id": "alice_2026-01-01_0"}]}}}
    led3 = _both(solve, bank3, "2026-01-01", [0.0, 0.0],
                 recalled_ids=["alice_2026-01-01_0"], user="alice",
                 cur_recall_date="2026-01-02")
    hist3 = led3["bank"]["alice"]["history"]["2026-01-01"]
    assert [m["memory_strength"] for m in hist3] == [2, 1]


def check_invalid_inputs(solve):
    good = {"u": {"history": {"2026-01-01": [
        {"query": "q", "response": "r", "memory_strength": 100}]}}}
    for bad in [
        lambda: solve([], "2026-01-06", [0.5]),                 # bank 非字典
        lambda: solve(good, "20260106", [0.5]),                 # 日期格式
        lambda: solve(good, "2026-01-06", [0.5], language="jp"),
        lambda: solve(good, "2026-01-06", ["x"]),               # draw 非数值
        lambda: solve(good, "2026-01-06", [True]),              # bool draw
        lambda: solve(good, "2026-01-06", [1.5]),               # 越界 draw
        lambda: solve(good, "2026-01-06", []),                  # draw 不足
        lambda: solve({"u": {"history": "x"}}, "2026-01-06", [0.5]),
        lambda: solve(good, "2026-01-06", [0.0],
                      recalled_ids=["alice_2026-01-01_0"]),      # 缺 user
        lambda: solve(good, "2026-01-06", [0.0], recalled_ids=["bad"],
                      user="u", cur_recall_date="2026-01-07"),   # id 无 '_'
        lambda: solve(good, "2026-01-06", [0.0],
                      recalled_ids=["u_2026-03-01_0"], user="u",
                      cur_recall_date="2026-01-07"),             # 未知日期 id
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


# ============ 随机对拍 ============

def _gen_case(rng):
    users = {}
    names = rng.sample(["al", "bo_1", "cy"], rng.randint(1, 3))
    for name in names:
        if rng.random() < 0.2:
            users[name] = {"summary": {"2026-01-0%d" % rng.randint(1, 5): "s"}}
            continue
        hist = {}
        for _ in range(rng.randint(1, 3)):
            day = "2026-01-%02d" % rng.randint(1, 9)
            rows = []
            for i in range(rng.randint(1, 3)):
                dlg = {"query": rng.choice([" q1", "q2", "  q3"]),
                       "response": rng.choice(["r1", " r2"]),
                       "memory_strength": rng.choice([1, 2, 7, 100]),
                       "last_recall_date":
                           "2025-12-%02d" % rng.randint(25, 31)}
                if rng.random() < 0.3:
                    dlg["memory_id"] = f"{name}_{day}_{i}"
                rows.append(dlg)
            hist[day] = rows
        um = {"history": hist}
        if rng.random() < 0.6:
            um["summary"] = {
                "2026-01-%02d" % rng.randint(1, 9):
                rng.choice(["s", {"content": "s2",
                                  "memory_strength": 2}])}
        users[name] = um
    n_dialogs = sum(len(rows) for u in users.values()
                    if isinstance(u, dict) and "history" in u
                    for rows in u["history"].values())
    draws = [round(rng.random(), 3) for _ in range(n_dialogs)]
    ids = []
    for name, u in users.items():
        if "history" in u:
            for day, rows in u["history"].items():
                for i in range(len(rows)):
                    if rng.random() < 0.4:
                        ids.append(f"{name}_{day}_{i}")
    kw = dict(language=rng.choice(["cn", "en"]))
    if ids and rng.random() < 0.8:
        kw.update(recalled_ids=ids, user=rng.choice(names),
                  cur_recall_date="2026-01-10")
    return users, draws, kw


def check_random(solve):
    rng = random.Random(7220722)
    for _ in range(200):
        bank, draws, kw = _gen_case(rng)
        try:
            got = solve(_dc(bank), "2026-01-06", _dc(draws), **kw)
        except ValueError:
            try:
                _oracle(_dc(bank), "2026-01-06", _dc(draws), **kw)
                raise AssertionError("solve raised but oracle did not")
            except ValueError:
                continue
        exp = _oracle(_dc(bank), "2026-01-06", _dc(draws), **kw)
        _eq(got, exp)


TEST_CASES = [
    ("curve_precedence", check_curve_precedence),
    ("basic_keep_and_docs", check_basic_keep_and_docs),
    ("multi_forget_pop_order", check_multi_forget_pop_order),
    ("summary_accounts", check_summary_accounts),
    ("no_history_user_dropped", check_no_history_user_dropped),
    ("update_accounts", check_update_accounts),
    ("invalid_inputs", check_invalid_inputs),
    ("random", check_random),
]
