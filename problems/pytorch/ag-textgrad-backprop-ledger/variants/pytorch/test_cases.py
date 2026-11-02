"""TextGrad 文本反传图账的检查点。

每个检查点先拿一份**另一套写法**的 oracle（迭代式后序遍历、扁平字典记账）
与 `solve` 的全量账本比对，再补硬断言钉住容易写错的口径。
"""


# ---------------------------------------------------------------- oracle --
def _text(grads):
    return "\n".join(g["v"] for g in grads)


def _oracle(nodes, grads, reduces=None, root=None, rounds=1):
    table = dict(reduces or {})
    spec_of, order_ids = {}, []
    for spec in nodes:
        spec_of[spec["id"]] = spec
        order_ids.append(spec["id"])

    preds, value, role, req, fn, meta = {}, {}, {}, {}, {}, {}
    for nid in order_ids:
        spec = spec_of[nid]
        kind = spec.get("kind", "input")
        if kind == "input":
            preds[nid] = []
        elif kind == "llm":
            preds[nid] = ([spec["system"]] if spec.get("system") else []) \
                + list(spec.get("inputs", []))
        else:
            preds[nid] = list(spec.get("inputs", []))

        if kind in ("input", "llm"):
            value[nid] = spec.get("value", "")
            role[nid] = spec.get("role", "")
            req[nid] = bool(spec.get("requires_grad", True))
            fn[nid] = None if kind == "input" else "llm"
            meta[nid] = []
            continue

        value[nid] = "\n".join(value[p] for p in preds[nid])
        distinct = []
        for p in preds[nid]:
            if role[p] not in distinct:
                distinct.append(role[p])
        joined = ", ".join(distinct)
        if kind == "add":
            role[nid] = role[preds[nid][0]] + " and " + role[preds[nid][1]]
            req[nid] = req[preds[nid][0]] or req[preds[nid][1]]
        elif kind == "sum":
            role[nid] = "a combination of the following: " + joined
            req[nid] = any(req[p] for p in preds[nid])
        else:
            role[nid] = ("a combination of the following variables: "
                         + joined + ".")
            req[nid] = any(req[p] for p in preds[nid])
        fn[nid] = kind
        meta[nid] = [{"id": spec.get("group", nid)}] if kind == "aggregate" else []

    if root is None:
        root = order_ids[-1]
    grad = {nid: [] for nid in order_ids}
    prompts, steps, walk = [], [], []

    for rnd in range(1, rounds + 1):
        post, seen = [], set()
        stack = [(root, False)]
        while stack:
            nid, done = stack.pop()
            if done:
                post.append(nid)
                continue
            if nid in seen:
                continue
            seen.add(nid)
            stack.append((nid, True))
            for p in reversed(preds[nid]):
                if p not in seen:
                    stack.append((p, False))
        proc = post[::-1]
        walk.extend(proc)
        grad[root] = []

        for nid in proc:
            if not req[nid]:
                steps.append({"round": rnd, "id": nid, "skipped": True})
                continue
            cur = grad[nid]
            before = [g["v"] for g in cur]
            reduced = False
            if meta[nid] and _text(cur) != "" and len(cur) > 1:
                buckets, index = [], {}
                for g in cur:
                    for item in g["m"]:
                        if item["id"] not in index:
                            index[item["id"]] = []
                            buckets.append(item["id"])
                        index[item["id"]].append(g)
                fresh = []
                for gid in buckets:
                    group = index[gid]
                    prompts.append("\n".join("<FEEDBACK>%s</FEEDBACK>" % g["v"]
                                             for g in group))
                    roles = []
                    for g in group:
                        if g["r"] not in roles:
                            roles.append(g["r"])
                    fresh.append({"v": table[gid], "r": ", ".join(roles), "m": []})
                cur = fresh
                grad[nid] = cur
                reduced = True

            pushed, path = [], None
            if fn[nid] == "llm":
                path = "base" if _text(cur) == "" else "chain"
                for p in preds[nid]:
                    if not req[p]:
                        continue
                    grad[p].append({"v": grads["%s:%s:%s" % (nid, path, p)],
                                    "r": "feedback to " + role[p], "m": []})
                    pushed.append({"to": p, "count": 1})
            elif fn[nid] in ("sum", "add"):
                body = _text(cur)
                for p in preds[nid]:
                    val = "" if body == "" else (
                        "Here is the combined feedback we got for this "
                        "specific %s and other variables: %s." % (role[p], body))
                    first = {"v": val, "r": "feedback to " + role[p],
                             "m": [dict(m) for m in meta[nid]]}
                    grad[p].append(first)
                    if meta[nid]:
                        meta[p].extend(dict(m) for m in meta[nid])
                    grad[p].append({"v": val, "r": "feedback to " + role[p],
                                    "m": []})
                    pushed.append({"to": p, "count": 2})
            elif fn[nid] == "aggregate" and preds[nid]:
                last = preds[nid][-1]
                body = _text(cur)
                val = "" if body == "" else (
                    "Here is the combined feedback we got for this "
                    "specific %s and other variables: %s." % (role[last], body))
                grad[last].append({"v": val, "r": "feedback to " + role[last],
                                   "m": [dict(m) for m in meta[nid]]})
                if meta[nid]:
                    meta[last].extend(dict(m) for m in meta[nid])
                pushed.append({"to": last, "count": 1})

            step = {"round": rnd, "id": nid, "skipped": False,
                    "in": before, "out": [g["v"] for g in cur],
                    "reduced": reduced, "called": fn[nid] is not None,
                    "pushed": pushed}
            if path is not None:
                step["path"] = path
            steps.append(step)

    return {
        "order": walk,
        "skipped": [s["id"] for s in steps if s["skipped"]],
        "steps": steps,
        "grad_text": {nid: _text(grad[nid]) for nid in order_ids},
        "grad_count": {nid: len(grad[nid]) for nid in order_ids},
        "values": value,
        "roles": role,
        "reduce_prompts": prompts,
        "rounds": rounds,
    }


def _view(ledger):
    return {
        "order": ledger["order"],
        "skipped": ledger["skipped"],
        "steps": [(s["round"], s["id"], s["skipped"], s.get("in"),
                   s.get("out"), s.get("reduced"), s.get("called"),
                   s.get("path"),
                   [(p["to"], p["count"]) for p in s.get("pushed", [])])
                  for s in ledger["steps"]],
        "grad_text": ledger["grad_text"],
        "grad_count": ledger["grad_count"],
        "values": ledger["values"],
        "roles": ledger["roles"],
        "prompts": ledger["reduce_prompts"],
    }


def _same(solve, nodes, grads, reduces=None, root=None, rounds=1):
    ledger = solve(nodes, grads, reduces=reduces, root=root, rounds=rounds)
    got = _view(ledger)
    want = _view(_oracle(nodes, grads, reduces=reduces, root=root, rounds=rounds))
    if got != want:
        for key in want:
            if got[key] != want[key]:
                raise AssertionError("%s mismatch\n got=%r\nwant=%r"
                                     % (key, got[key], want[key]))
    return ledger


def _feedback(role, text):
    return ("Here is the combined feedback we got for this specific "
            "%s and other variables: %s." % (role, text))


# ---------------------------------------------------------------- graph ----
def _chain():
    """question -> answer（单条 LLM 边）。"""
    return [
        {"id": "q", "kind": "input", "value": "3 * 7", "role": "question"},
        {"id": "a", "kind": "llm", "inputs": ["q"], "value": "20",
         "role": "answer"},
    ]


CHAIN_GRADS = {"a:base:q": "the question asks for 21"}


# --------------------------------------------------------------- checks --
def check_example(solve):
    """题面示例：单条 LLM 边。"""
    r = _same(solve, _chain(), CHAIN_GRADS)
    assert r["order"] == ["a", "q"]
    assert r["grad_text"]["q"] == "the question asks for 21"
    assert r["grad_text"]["a"] == ""
    assert r["steps"][0]["path"] == "base"
    assert r["steps"][0]["called"] is True
    assert r["steps"][1]["called"] is False
    assert r["grad_count"] == {"q": 1, "a": 0}


def check_base_vs_chain(solve):
    """响应自己有梯度就换 chain 路，两条路的梯度都只推一条。"""
    nodes = [
        {"id": "q", "kind": "input", "value": "2 + 2", "role": "question"},
        {"id": "a", "kind": "llm", "inputs": ["q"], "value": "4",
         "role": "answer"},
        {"id": "r", "kind": "input", "value": "4", "role": "reference"},
        {"id": "s", "kind": "sum", "inputs": ["a", "r"]},
    ]
    grads = {"a:chain:q": "off by a bit"}
    r = _same(solve, nodes, grads, root="s")
    # 求和节点自己没有梯度，两条空串拼起来却是 "\n"（非空）
    assert r["grad_text"]["s"] == ""
    assert r["steps"][0]["pushed"] == [{"to": "a", "count": 2},
                                       {"to": "r", "count": 2}]
    assert r["steps"][2]["id"] == "a"
    assert r["steps"][2]["path"] == "chain"
    assert r["grad_text"]["q"] == "off by a bit"
    assert r["grad_count"]["a"] == 2
    assert r["grad_count"]["q"] == 1


def check_empty_gradient_still_chain(solve):
    """求和推来的两条空串 -> 文本是 "\\n" -> 下游仍判定为「有梯度」。"""
    nodes = [
        {"id": "q", "kind": "input", "value": "hello", "role": "question"},
        {"id": "a", "kind": "llm", "inputs": ["q"], "value": "hi",
         "role": "answer"},
        {"id": "s", "kind": "sum", "inputs": ["a"]},
    ]
    r = _same(solve, nodes, {"a:chain:q": "too terse"}, root="s")
    assert r["steps"][0]["id"] == "s"
    assert r["steps"][0]["pushed"] == [{"to": "a", "count": 2}]
    assert r["steps"][1]["in"] == ["", ""]
    assert r["steps"][1]["path"] == "chain"
    # 反馈模板用的是「一条空串」而不是「两条空串」
    assert r["grad_text"]["q"] == "too terse"

    # 对照：根自己就是 llm -> 没有梯度 -> base
    r = _same(solve, nodes[:2], CHAIN_GRADS, root="a")
    assert r["steps"][0]["path"] == "base"


def check_sum_broadcast_text(solve):
    """求和节点有梯度时，每个前驱各拿两条完全相同的反馈。"""
    nodes = [
        {"id": "q1", "kind": "input", "value": "A", "role": "first half"},
        {"id": "q2", "kind": "input", "value": "B", "role": "second half"},
        {"id": "s", "kind": "sum", "inputs": ["q1", "q2"]},
        {"id": "u", "kind": "llm", "inputs": ["s"], "value": "AB",
         "role": "merged"},
    ]
    r = _same(solve, nodes, {"u:base:s": "merge is wrong"})
    assert r["steps"][1]["id"] == "s"
    assert r["steps"][1]["in"] == ["merge is wrong"]
    assert r["steps"][1]["pushed"] == [{"to": "q1", "count": 2},
                                       {"to": "q2", "count": 2}]
    body = _feedback("first half", "merge is wrong")
    assert r["grad_text"]["q1"] == body + "\n" + body
    assert r["grad_count"]["q1"] == 2
    body2 = _feedback("second half", "merge is wrong")
    assert r["grad_text"]["q2"] == body2 + "\n" + body2
    # 角色文本按声明序去重
    assert r["roles"]["s"] == "a combination of the following: first half, second half"


def check_add_role(solve):
    """__add__ 的角色是 "A and B"，且同样双推。"""
    nodes = [
        {"id": "x", "kind": "input", "value": "1", "role": "lhs"},
        {"id": "y", "kind": "input", "value": "2", "role": "rhs"},
        {"id": "t", "kind": "add", "inputs": ["x", "y"]},
        {"id": "u", "kind": "llm", "inputs": ["t"], "value": "3",
         "role": "result"},
    ]
    r = _same(solve, nodes, {"u:base:t": "sum is off"})
    assert r["roles"]["t"] == "lhs and rhs"
    assert r["values"]["t"] == "1\n2"
    assert r["steps"][1]["pushed"] == [{"to": "x", "count": 2},
                                       {"to": "y", "count": 2}]
    body = _feedback("rhs", "sum is off")
    assert r["grad_text"]["y"] == body + "\n" + body


def check_aggregate_last_only(solve):
    """aggregate 只给声明序最后一个前驱推一条梯度。"""
    nodes = [
        {"id": "x", "kind": "input", "value": "alpha", "role": "left"},
        {"id": "y", "kind": "input", "value": "beta", "role": "right"},
        {"id": "g", "kind": "aggregate", "inputs": ["x", "y"]},
        {"id": "u", "kind": "llm", "inputs": ["g"], "value": "alpha beta",
         "role": "merged"},
    ]
    r = _same(solve, nodes, {"u:base:g": "order flipped"})
    assert r["roles"]["g"] == ("a combination of the following variables: "
                               "left, right.")
    assert r["steps"][1]["pushed"] == [{"to": "y", "count": 1}]
    assert r["grad_text"]["y"] == _feedback("right", "order flipped")
    assert r["grad_text"]["x"] == ""
    assert r["grad_count"] == {"x": 0, "y": 1, "g": 1, "u": 0}


def check_llm_skip_nograd(solve):
    """LLM 后向跳过 requires_grad=False 的前驱（系统提示词）。"""
    nodes = [
        {"id": "sys", "kind": "input", "value": "be terse", "role": "system",
         "requires_grad": False},
        {"id": "q", "kind": "input", "value": "why", "role": "question"},
        {"id": "a", "kind": "llm", "inputs": ["q"], "system": "sys",
         "value": "because", "role": "answer"},
    ]
    r = _same(solve, nodes, {"a:base:q": "too vague"})
    assert r["order"] == ["a", "q", "sys"]
    assert r["steps"][0]["pushed"] == [{"to": "q", "count": 1}]
    assert r["grad_text"]["sys"] == ""
    assert r["skipped"] == ["sys"]


def check_topo_reverse_order(solve):
    """逆后序：同一个输入被两条支路喂时，后处理的那条先写进梯度。"""
    nodes = [
        {"id": "q", "kind": "input", "value": "1/0", "role": "question"},
        {"id": "a", "kind": "llm", "inputs": ["q"], "value": "0",
         "role": "answer"},
        {"id": "b", "kind": "llm", "inputs": ["q"], "value": "inf",
         "role": "guess"},
        {"id": "s", "kind": "sum", "inputs": ["a", "b"]},
    ]
    grads = {"a:chain:q": "from answer", "b:chain:q": "from guess"}
    r = _same(solve, nodes, grads, root="s")
    assert r["order"] == ["s", "b", "a", "q"]
    assert r["grad_text"]["q"] == "from guess\nfrom answer"


def check_root_only_reset(solve):
    """只清根节点梯度：中间节点第二轮继续累积。"""
    nodes = [
        {"id": "q", "kind": "input", "value": "x", "role": "question"},
        {"id": "a", "kind": "llm", "inputs": ["q"], "value": "y",
         "role": "answer"},
        {"id": "s", "kind": "sum", "inputs": ["a"]},
    ]
    r = _same(solve, nodes, {"a:chain:q": "sharpen it"}, root="s", rounds=2)
    assert r["order"] == ["s", "a", "q"] * 2
    assert r["grad_count"]["a"] == 4
    assert r["grad_count"]["q"] == 2
    assert r["skipped"] == []


def check_reduce_group(solve):
    """同 group 的两条梯度并成一次归约；提示词按梯度到达顺序逐条包 <FEEDBACK>。"""
    nodes = [
        {"id": "x", "kind": "input", "value": "alpha", "role": "left"},
        {"id": "y", "kind": "input", "value": "beta", "role": "right"},
        {"id": "g1", "kind": "aggregate", "inputs": ["x", "y"], "group": "G"},
        {"id": "g2", "kind": "aggregate", "inputs": ["x", "y"], "group": "G"},
        {"id": "u1", "kind": "llm", "inputs": ["g1"], "value": "one",
         "role": "pass one"},
        {"id": "u2", "kind": "llm", "inputs": ["g2"], "value": "two",
         "role": "pass two"},
        {"id": "t", "kind": "add", "inputs": ["u1", "u2"]},
    ]
    grads = {"u1:chain:g1": "first pass wrong", "u2:chain:g2": "second pass wrong"}
    r = _same(solve, nodes, grads, reduces={"G": "two in one"}, root="t")
    assert r["order"] == ["t", "u2", "g2", "u1", "g1", "y", "x"]
    # 两个 aggregate 各自只收到一条梯度 -> 不归约，原样往下推
    assert r["grad_text"]["g1"] == "first pass wrong"
    assert r["grad_text"]["g2"] == "second pass wrong"
    assert r["grad_count"]["y"] == 1
    assert r["grad_text"]["y"] == "two in one"
    assert len(r["reduce_prompts"]) == 1
    second = _feedback("right", "second pass wrong")
    first = _feedback("right", "first pass wrong")
    assert r["reduce_prompts"][0] == ("<FEEDBACK>" + second + "</FEEDBACK>\n"
                                      "<FEEDBACK>" + first + "</FEEDBACK>")


def check_aggregate_self_reduce(solve):
    """aggregate 节点自带归约标记：收到两条无标记梯度时当场被抹平。"""
    nodes = [
        {"id": "x", "kind": "input", "value": "alpha", "role": "left"},
        {"id": "y", "kind": "input", "value": "beta", "role": "right"},
        {"id": "g1", "kind": "aggregate", "inputs": ["x", "y"], "group": "G"},
        {"id": "g2", "kind": "aggregate", "inputs": ["x", "y"], "group": "G"},
        {"id": "t", "kind": "add", "inputs": ["g1", "g2"]},
    ]
    r = _same(solve, nodes, {}, reduces={"G": "two in one"}, root="t")
    # t 自己没有梯度 -> 推给 g1/g2 的都是空串；两条空串拼成 "\n" 非空 -> 触发归约
    # 但这两条梯度没带标记，一个分组都没有：不调总结、梯度直接清空
    assert r["steps"][1]["id"] == "g2"
    assert r["steps"][1]["reduced"] is True
    assert r["steps"][1]["out"] == []
    assert r["steps"][2]["id"] == "g1"
    assert r["steps"][2]["out"] == []
    assert r["grad_text"]["g1"] == ""
    assert r["grad_count"]["g1"] == 0
    # 唯一一次真正的总结发生在 y：两条空梯度同组
    assert r["reduce_prompts"] == ["<FEEDBACK></FEEDBACK>\n<FEEDBACK></FEEDBACK>"]


def check_reduce_needs_two(solve):
    """只有一条梯度就不归约（哪怕带标记）。"""
    nodes = [
        {"id": "x", "kind": "input", "value": "alpha", "role": "left"},
        {"id": "y", "kind": "input", "value": "beta", "role": "right"},
        {"id": "g", "kind": "aggregate", "inputs": ["x", "y"]},
        {"id": "u", "kind": "llm", "inputs": ["g"], "value": "alpha beta",
         "role": "merged"},
    ]
    r = _same(solve, nodes, {"u:base:g": "order flipped"}, root="u")
    assert r["reduce_prompts"] == []
    assert r["grad_count"]["y"] == 1
    assert r["grad_text"]["y"] == _feedback("right", "order flipped")


def check_sum_carries_reduce_meta(solve):
    """aggregate -> sum：归约标记沿幂等广播下发，双推里只有一条带标记。"""
    nodes = [
        {"id": "x", "kind": "input", "value": "1", "role": "left"},
        {"id": "z", "kind": "input", "value": "2", "role": "right"},
        {"id": "s", "kind": "sum", "inputs": ["x", "z"]},
        {"id": "g", "kind": "aggregate", "inputs": ["s"], "group": "G"},
        {"id": "u", "kind": "llm", "inputs": ["g"], "value": "12",
         "role": "merged"},
    ]
    r = _same(solve, nodes, {"u:base:g": "digits swapped"}, root="u",
              reduces={"G": "one summary"})
    assert r["grad_text"]["x"] == "one summary"
    assert r["grad_text"]["z"] == "one summary"
    assert r["grad_count"]["x"] == 1
    assert r["grad_count"]["z"] == 1
    # 每个前驱各归约一次，各丢掉那条没带标记的重复梯度
    assert len(r["reduce_prompts"]) == 2
    inner = _feedback(r["roles"]["s"], "digits swapped")
    assert r["reduce_prompts"][0] == ("<FEEDBACK>" + _feedback("right", inner)
                                      + "</FEEDBACK>")
    assert r["reduce_prompts"][1] == ("<FEEDBACK>" + _feedback("left", inner)
                                      + "</FEEDBACK>")


def check_no_grad_construction(solve):
    """requires_grad=False 的变量不允许有需要梯度的前驱。"""
    nodes = [
        {"id": "q", "kind": "input", "value": "x", "role": "question"},
        {"id": "a", "kind": "llm", "inputs": ["q"], "value": "y",
         "role": "answer", "requires_grad": False},
    ]
    try:
        solve(nodes, {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def check_empty_sum(solve):
    """空求和：值为空串、不需要梯度、轮到它时整段跳过。"""
    nodes = [
        {"id": "s", "kind": "sum", "inputs": []},
        {"id": "q", "kind": "input", "value": "x", "role": "question"},
    ]
    r = _same(solve, nodes, {}, root="s")
    assert r["values"]["s"] == ""
    assert r["roles"]["s"] == "a combination of the following: "
    assert r["skipped"] == ["s"]
    assert r["grad_count"] == {"s": 0, "q": 0}

def check_duplicate_roles(solve):
    """角色描述去重：两个同角色的前驱在组合角色里只出现一次。"""
    nodes = [
        {"id": "c1", "kind": "input", "value": "one", "role": "chunk"},
        {"id": "c2", "kind": "input", "value": "two", "role": "chunk"},
        {"id": "s", "kind": "sum", "inputs": ["c1", "c2"]},
        {"id": "u", "kind": "llm", "inputs": ["s"], "value": "one two",
         "role": "answer"},
    ]
    r = _same(solve, nodes, {"u:base:s": "missing a chunk"})
    assert r["roles"]["s"] == "a combination of the following: chunk"
    # 两个前驱各收两条同样的反馈，文本里角色只出现一次
    assert r["grad_text"]["c1"].count("for this specific chunk and other") == 2
    assert r["grad_text"]["c1"] == r["grad_text"]["c2"]


def check_bad_specs(solve):
    """建图期的硬约束一律报 ValueError。"""
    bad = [
        ([{"id": "a", "kind": "input", "value": "x", "role": "r"}], 0),
        ([{"id": "a", "kind": "input", "value": "x", "role": "r"}], -1),
        ([{"id": "a", "kind": "input", "value": "x", "role": "r"},
          {"id": "a", "kind": "input", "value": "y", "role": "r"}], 1),
        ([{"id": "a", "kind": "input", "value": "x", "role": "r"},
          {"id": "b", "kind": "llm", "inputs": ["zz"], "value": "y",
           "role": "r"}], 1),
    ]
    for nodes, rounds in bad:
        try:
            solve(nodes, {}, rounds=rounds)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % (nodes,))

    try:
        solve([{"id": "a", "kind": "input", "value": "x", "role": "r"}], {},
              root="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown root")

    try:
        solve([], {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty graph")

    try:
        solve([{"id": "a", "kind": "input", "value": "x", "role": "r"},
               {"id": "b", "kind": "input", "value": "y", "role": "r"},
               {"id": "c", "kind": "input", "value": "z", "role": "r"},
               {"id": "t", "kind": "add", "inputs": ["a", "b", "c"]}], {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for three-operand add")

def check_bare_aggregate_root(solve):
    """aggregate 当根、自己又没梯度：不进归约分支，只把空反馈推给最后一个前驱。"""
    nodes = [
        {"id": "x", "kind": "input", "value": "alpha", "role": "left"},
        {"id": "y", "kind": "input", "value": "beta", "role": "right"},
        {"id": "g", "kind": "aggregate", "inputs": ["x", "y"], "group": "G"},
    ]
    r = _same(solve, nodes, {}, reduces={"G": "never used"}, root="g")
    assert r["steps"][0]["id"] == "g"
    assert r["steps"][0]["in"] == []
    assert r["steps"][0]["reduced"] is False
    assert r["steps"][0]["pushed"] == [{"to": "y", "count": 1}]
    assert r["reduce_prompts"] == []
    assert r["grad_text"]["y"] == ""
    assert r["grad_count"]["y"] == 1


check_example.example = True
