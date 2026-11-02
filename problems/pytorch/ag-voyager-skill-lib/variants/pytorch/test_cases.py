GUARD = "Deposit useless items into the chest at"


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
    assert got == ref, (path, got, ref)


def _desc(name, text):
    return "async function %s(bot) {\n    // %s\n}" % (name, text)


def _py(skills, events, retrieval_top_k=5, dump_files=None, primitives=None):
    book, order = {}, []
    for raw in skills:
        book[raw["name"]] = {"code": raw["code"],
                             "description": raw["description"]}
        order.append(raw["name"])
    files = set(dump_files or ())
    dumps, retrieved, skipped = [], [], []
    for i, ev in enumerate(events):
        if ev["kind"] == "add":
            if ev["task"].startswith(GUARD):
                skipped.append({"event": i, "program_name": ev["program_name"]})
                continue
            name = ev["program_name"]
            desc = _desc(name, ev["llm_description"])
            rewritten = name in book
            if rewritten:
                n = 2
                while "%sV%d.js" % (name, n) in files:
                    n += 1
                dumped = "%sV%d" % (name, n)
            else:
                dumped = name
                order.append(name)
            book[name] = {"code": ev["program_code"], "description": desc}
            files.add(dumped + ".js")
            dumps.append({"event": i, "program_name": name, "file": dumped,
                          "rewritten": rewritten})
        else:
            dist = list(ev["distances"])
            k = min(len(order), retrieval_top_k)
            if k == 0:
                retrieved.append({"event": i, "query": ev["query"], "k": 0,
                                  "names": [], "codes": []})
                continue
            pairs = sorted(((dist[j], j) for j in range(len(order))))[:k]
            names = [order[j] for _, j in pairs]
            retrieved.append({"event": i, "query": ev["query"], "k": k,
                              "names": names,
                              "codes": [book[x]["code"] for x in names]})
    out = ""
    for name in order:
        out += book[name]["code"] + "\n\n"
    for code in (primitives or ()):
        out += code + "\n\n"
    return {"order": order, "skills": book, "dumps": dumps,
            "retrieved": retrieved, "skipped": skipped, "programs": out,
            "count": len(book), "synced": len(book) == len(order)}


def _compare(solve, skills, events, **kw):
    ref = _py([dict(s) for s in skills], [dict(e) for e in events], **kw)
    got = solve([dict(s) for s in skills], [dict(e) for e in events], **kw)
    _eq(got, ref)
    return got


def _add(task, name, code=None, text="does it"):
    return {"kind": "add", "task": task, "program_name": name,
            "program_code": code or ("async function %s(bot) {}" % name),
            "llm_description": text}


def _sk(name, code=None):
    return {"name": name, "code": code or ("async function %s(bot) {}" % name),
            "description": _desc(name, "existing")}


def check_guard_prefix(solve):
    """只有以 `...into the chest at` 开头的任务被跳过；前缀差一个词就不算。"""
    events = [
        _add("Deposit useless items into the chest at 10, 64, 10", "deposit"),
        _add("Deposit useless items into the chest", "deposit_near"),
        _add("deposit useless items into the chest at 1, 2, 3", "deposit_lower"),
        _add("Mine 3 wood logs", "mine"),
    ]
    led = _compare(solve, [], events)
    assert [s["program_name"] for s in led["skipped"]] == ["deposit"]
    assert led["order"] == ["deposit_near", "deposit_lower", "mine"]


def check_same_name_overwrites(solve):
    """同名技能是**覆盖**不是新增：条目数不变、字典位置不变、落盘名带 V2。"""
    led = _compare(solve, [_sk("mine"), _sk("craft")], [
        _add("Mine wood", "mine", code="async function mine(bot) { /* v2 */ }",
             text="mine wood"),
    ])
    assert led["order"] == ["mine", "craft"]      # 位置保持原样
    assert led["count"] == 2 and led["synced"] is True
    assert led["skills"]["mine"]["code"] == "async function mine(bot) { /* v2 */ }"
    assert led["dumps"] == [{"event": 0, "program_name": "mine",
                             "file": "mineV2", "rewritten": True}]


def check_version_suffix_scan(solve):
    """落盘名从 V2 起找空位：V2 被占就 V3，且新建的技能从已有文件里挑空位。"""
    events = [_add("Mine wood", "mine"),
              _add("Mine wood again", "mine"),
              _add("Mine wood third", "mine")]
    led = _compare(solve, [_sk("mine")], events,
                   dump_files=["mine.js", "mineV2.js"])
    assert [d["file"] for d in led["dumps"]] == ["mineV3", "mineV4", "mineV5"]
    # 新技能与 V 后缀无关
    led = _compare(solve, [], [_add("Craft planks", "craft")],
                   dump_files=["craft.js", "craftV2.js"])
    assert led["dumps"][0]["file"] == "craft"


def check_description_template(solve):
    """描述模板：`async function <name>(bot) {` + 四空格注释 + 收尾花括号。"""
    led = _compare(solve, [], [_add("Mine wood", "mine", text="mine the nearest log")])
    assert led["skills"]["mine"]["description"] == (
        "async function mine(bot) {\n    // mine the nearest log\n}")
    assert led["skills"]["mine"]["description"].count("\n") == 2


def check_retrieve_top_k(solve):
    """检索条数是 `min(库大小, retrieval_top_k)`，空库返回 [] 且 k=0。"""
    base = [_sk("a"), _sk("b"), _sk("c")]
    led = _compare(solve, base, [{"kind": "retrieve", "query": "q",
                                  "distances": [0.5, 0.1, 0.3]}],
                   retrieval_top_k=2)
    rec = led["retrieved"][0]
    assert rec["k"] == 2
    assert rec["names"] == ["b", "c"]      # 距离升序
    assert rec["codes"] == [base[1]["code"], base[2]["code"]]

    led = _compare(solve, base, [{"kind": "retrieve", "query": "q",
                                  "distances": [0.5, 0.1, 0.3]}],
                   retrieval_top_k=10)
    assert led["retrieved"][0]["k"] == 3

    led = _compare(solve, [], [{"kind": "retrieve", "query": "q", "distances": []}])
    assert led["retrieved"][0] == {"event": 0, "query": "q", "k": 0,
                                   "names": [], "codes": []}


def check_retrieve_tie_by_insertion(solve):
    """距离并列时按插入顺序（不是字典序、不是后到先得）。"""
    base = [_sk("zeta"), _sk("alpha"), _sk("mid")]
    led = _compare(solve, base, [{"kind": "retrieve", "query": "q",
                                  "distances": [0.2, 0.2, 0.2]}],
                   retrieval_top_k=2)
    assert led["retrieved"][0]["names"] == ["zeta", "alpha"]


def check_retrieve_after_rewrite(solve):
    """覆盖后检索到的是新代码，且位置仍在原处。"""
    base = [_sk("a", code="A1"), _sk("b", code="B1")]
    led = _compare(solve, base, [
        _add("task", "a", code="A2"),
        {"kind": "retrieve", "query": "q", "distances": [0.0, 9.0]},
    ], retrieval_top_k=2)
    assert led["retrieved"][0]["names"] == ["a", "b"]
    assert led["retrieved"][0]["codes"] == ["A2", "B1"]


def check_programs_concatenation(solve):
    """programs：先所有技能（插入序）、再控制原语，每段后跟一个空行。"""
    led = _compare(solve, [_sk("a", code="A"), _sk("b", code="B")], [],
                   primitives=["P1", "P2"])
    assert led["programs"] == "A\n\nB\n\nP1\n\nP2\n\n"
    assert _compare(solve, [], [])["programs"] == ""
    led = _compare(solve, [], [], primitives=["P"])
    assert led["programs"] == "P\n\n"


def check_validation(solve):
    """结构校验：重名、top_k 非正、距离条数不符、未知事件、bool 距离。"""
    for args in (([_sk("a"), _sk("a")], []),
                 ([], [{"kind": "nope"}]),
                 ([_sk("a")], [{"kind": "retrieve", "query": "q",
                                "distances": []}]),
                 ([_sk("a")], [{"kind": "retrieve", "query": "q",
                                "distances": [True]}])):
        try:
            solve(list(args[0]), list(args[1]))
        except ValueError:
            pass
        else:
            raise AssertionError("%r 应当报错" % (args,))
    for kw in ({"retrieval_top_k": 0}, {"retrieval_top_k": True}):
        try:
            solve([], [], **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("%r 应当报错" % (kw,))
    try:
        solve([], [_add("task", "x", text=None)])
    except ValueError:
        pass
    else:
        raise AssertionError("llm_description=None 应当报错")


def check_against_oracle(solve):
    """混合场景：跳过 + 新增 + 覆盖 + 两次检索。"""
    base = [_sk("mine", code="M1")]
    events = [
        _add("Deposit useless items into the chest at 0, 64, 0", "deposit"),
        _add("Craft a wooden pickaxe", "craft", text="craft pickaxe"),
        _add("Mine iron", "mine", code="M2", text="mine iron"),
        {"kind": "retrieve", "query": "dig", "distances": [0.9, 0.1]},
        _add("Craft a stone sword", "sword", code="S1"),
        {"kind": "retrieve", "query": "fight", "distances": [0.7, 0.2, 0.2]},
    ]
    led = _compare(solve, base, events, retrieval_top_k=2,
                   dump_files=["mine.js"], primitives=["P"])
    assert led["order"] == ["mine", "craft", "sword"]
    assert [d["file"] for d in led["dumps"]] == ["craft", "mineV2", "sword"]
    assert led["retrieved"][0]["names"] == ["craft", "mine"]
    assert led["retrieved"][1]["names"] == ["craft", "sword"]
    assert led["programs"].endswith("P\n\n")


def run_examples(solve):
    print(_compare(solve, [_sk("mine")], [_add("Mine wood", "mine")])["dumps"])
    print(_compare(solve, [_sk("a")], [{"kind": "retrieve", "query": "q",
                                        "distances": [0.4]}])["retrieved"][0])


check_same_name_overwrites.example = True
check_retrieve_top_k.example = True
