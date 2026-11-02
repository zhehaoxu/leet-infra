import random

from copy import deepcopy


def _dc(x):
    return deepcopy(x)


# ---------------------------------------------------------------- oracle
# 独立实现（不 import solution）：去重直接用 `x not in list`（dict 等值），
# 地址判定用集合交集，广播/转交按分支展开。

def _oracle(roles, messages, tl_name="Mike", is_public_chat=True, rounds=1,
            all_tag="<all>", no_one_tag="no one"):
    if rounds < 0:
        raise ValueError("negative rounds")
    names = [r["name"] for r in roles]
    if len(set(names)) != len(names):
        raise ValueError("duplicate")
    by = {r["name"]: r for r in roles}
    if tl_name not in by:
        raise ValueError("no leader")
    tl = by[tl_name]
    addrs = {}
    for r in roles:
        a = r.get("addresses")
        addrs[r["name"]] = set(a) if a is not None else {r["cls"], r["name"]}

    def norm(raw):
        return {"id": raw["id"], "content": raw["content"],
                "role": raw.get("role", "user"),
                "cause_by": raw.get("cause_by", ""),
                "sent_from": raw.get("sent_from", ""),
                "send_to": set(raw.get("send_to", [all_tag])),
                "metadata": dict(raw.get("metadata", {})),
                "publicer": raw.get("publicer", ""),
                "user_defined": raw.get("user_defined_recipient", "")}

    buf = {n: [] for n in names}
    memo = {n: [] for n in names}
    news = {n: [] for n in names}
    anch = {}
    for n in names:
        raw = by[n].get("latest_observed")
        anch[n] = None if raw is None else norm(raw)
    hist = []
    direct = set()

    def idle(n):
        return len(news[n]) == 0 and len(buf[n]) == 0

    def conv(m):
        c = dict(m)
        c["send_to"] = set(m["send_to"])
        c["metadata"] = dict(m["metadata"])
        if c["role"] not in ("system", "user", "assistant"):
            c["role"] = "assistant"
        who = c["metadata"].get("agent", c["sent_from"])
        if c["send_to"] == {all_tag}:
            to = tl_name
        else:
            to = ", ".join(sorted(c["send_to"] - {all_tag}))
        c["content"] = "[Message] from %s to %s: %s" % (
            who or "User", to, c["content"])
        return c

    def fanout(m):
        if is_public_chat:
            m["send_to"].add(all_tag)
        c = conv(m)
        got = []
        for n in names:
            if all_tag in c["send_to"] or len(addrs[n] & c["send_to"]) > 0:
                buf[n].append(c)
                got.append(n)
        return got, c

    pubs = []
    for raw in messages:
        m = norm(raw)
        added, published, c = [], False, None
        if m["user_defined"]:
            branch = "direct_in"
            for n in m["send_to"]:
                if n not in by:
                    raise ValueError("bad recipient")
                if idle(n):
                    direct.add(n)
                    added.append(n)
            got, c = fanout(m)
            published = True
        elif m["sent_from"] in direct:
            branch = "direct_out"
            direct.remove(m["sent_from"])
            if is_public_chat:
                got, c = fanout(m)
                published = True
            else:
                got = []
        elif m["publicer"] == tl["profile"]:
            if m["send_to"] == {no_one_tag}:
                branch = "skipped"
                got = []
            else:
                branch = "tl"
                got, c = fanout(m)
                published = True
        else:
            branch = "regular"
            m["send_to"].add(tl_name)
            got, c = fanout(m)
            published = True
        rec = 0
        if branch != "skipped":
            if published and c not in hist:
                hist.append(c)
                rec += 1
            if m not in hist:
                hist.append(m)
                rec += 1
        pubs.append({"branch": branch,
                     "send_to": sorted(m["send_to"]),
                     "delivered": sorted(got),
                     "direct_chat_added": sorted(added),
                     "history_added": rec,
                     "content_out": None if c is None else c["content"],
                     "role_out": None if c is None else c["role"]})

    seen = {n: [] for n in names}
    for _ in range(rounds):
        for n in names:
            if idle(n):
                seen[n].append(None)
                continue
            r = by[n]
            batch = []
            if r.get("recovered") and anch[n] is not None:
                if anch[n] not in memo[n][-10:]:
                    batch = [anch[n]]
            if not batch:
                batch = buf[n]
                buf[n] = []
            known = [] if not r.get("enable_memory", True) else list(memo[n])
            watch = set(r.get("watch") or ())
            pick = [x for x in batch
                    if (x["cause_by"] in watch or n in x["send_to"])
                    and x not in known]
            news[n] = pick
            for x in (batch if r.get("observe_all") else pick):
                if x not in memo[n]:
                    memo[n].append(x)
            anch[n] = pick[-1] if pick else None
            seen[n].append([x["id"] for x in pick])

    out = {n: {"news": seen[n], "memory": [x["id"] for x in memo[n]],
               "buffer": [x["id"] for x in buf[n]], "idle": idle(n)}
           for n in names}
    return {"publishes": pubs, "roles": out,
            "history_ids": [x["id"] for x in hist],
            "direct_chat_roles": sorted(direct)}


def _eq(a, b):
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), (a, b)
        for x, y in zip(a, b):
            _eq(x, y)
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), (a.keys() ^ b.keys())
        for k in a:
            _eq(a[k], b[k])
        return
    assert a == b, (a, b)


def _compare(solve, *args, **kw):
    got = solve(*[_dc(x) for x in args], **deepcopy(kw))
    exp = _oracle(*[_dc(x) for x in args], **deepcopy(kw))
    _eq(got, exp)
    return got


# ---------------------------------------------------------------- fixtures

def _role(name, watch=("WriteCode",), **kw):
    d = {"name": name, "profile": name, "cls": "metagpt.roles.role.Role",
         "watch": list(watch)}
    d.update(kw)
    return d


def _msg(mid, content, sent_from="User", send_to=("Alice",), cause_by="WriteCode",
         **kw):
    d = {"id": mid, "content": content, "sent_from": sent_from,
         "send_to": list(send_to), "cause_by": cause_by}
    d.update(kw)
    return d


_TL = _role("Mike", watch=("UserRequirement",))
_ALICE = _role("Alice")
_BOB = _role("Bob")


def _led(solve, roles, messages, **kw):
    return _compare(solve, roles, messages, **kw)


def check_addresses_fanout(solve):
    """投递只看 `addresses`：默认 `{cls, name}`；`<all>` 广播给所有人；
    自定义 addresses 之后连团队负责人的名字也收不到（私聊时可见）。"""
    led = _led(solve, [_TL, _ALICE], [_msg("m1", "hi")], rounds=1)
    p = led["publishes"][0]
    assert p["send_to"] == ["<all>", "Alice", "Mike"]
    assert p["delivered"] == ["Alice", "Mike"]      # Mike 因转交收件
    # 私聊 + 自定义地址：转交的 tl.name 不是地址 → TL 收不到
    tl2 = _role("Mike", watch=("UserRequirement",),
                cls="x.Role", addresses=["mike-bot"])
    led = _led(solve, [tl2, _ALICE], [_msg("m1", "hi")],
               is_public_chat=False, rounds=1)
    p = led["publishes"][0]
    assert p["send_to"] == ["Alice", "Mike"]
    assert p["delivered"] == ["Alice"]
    # 收件人换成 TL 的类名：普通分支会顺手加 tl.name，看不出 cls 的作用，
    # 所以走**转交分支**（不加 tl.name），只能靠 addresses 里的 cls 命中
    tl3 = _role("Mike", watch=("UserRequirement",), cls="x.Role")
    led = _led(solve, [tl3, _ALICE],
               [_msg("m1", "hi", send_to=("x.Role",), publicer="Mike",
                     cause_by="")], is_public_chat=False, rounds=1)
    assert led["publishes"][0]["branch"] == "tl"
    assert led["publishes"][0]["send_to"] == ["x.Role"]
    assert led["publishes"][0]["delivered"] == ["Mike"]
    # 广播无视 addresses
    led = _led(solve, [tl2, _ALICE], [_msg("m1", "hi", send_to=["<all>"])],
               rounds=1)
    assert led["publishes"][0]["delivered"] == ["Alice", "Mike"]


def check_mgx_branches(solve):
    """四条分支：普通消息加上 tl.name、`publicer == tl.profile` 才转交、
    只有 `send_to == {"no one"}` 才丢（多带一个收件人就不丢）。"""
    led = _led(solve, [_TL, _ALICE], [
        _msg("m1", "plain"),
        _msg("m2", "dummy", send_to=("no one",), publicer="Mike", cause_by=""),
        _msg("m3", "done", sent_from="Mike", publicer="Mike", cause_by=""),
        _msg("m4", "ok", send_to=("no one", "Alice"), publicer="Mike",
             cause_by=""),
    ], rounds=1)
    got = [p["branch"] for p in led["publishes"]]
    assert got == ["regular", "skipped", "tl", "tl"]
    assert led["publishes"][0]["send_to"] == ["<all>", "Alice", "Mike"]
    assert led["publishes"][1]["delivered"] == []
    assert led["publishes"][1]["history_added"] == 0
    assert led["publishes"][1]["content_out"] is None
    assert led["publishes"][2]["send_to"] == ["<all>", "Alice"]   # 不加 tl
    assert led["publishes"][3]["delivered"] == ["Alice", "Mike"]
    # profile ≠ name：转交加的是 **name**，判定用的是 **profile**
    tl_p = {"name": "Mike", "profile": "Mike-bot",
            "cls": "metagpt.roles.role.Role", "watch": ["UserRequirement"]}
    led = _led(solve, [tl_p, _ALICE],
               [_msg("m1", "hi"),
                _msg("m2", "x", publicer="Mike-bot", cause_by="")], rounds=1)
    ps = led["publishes"]
    assert ps[0]["send_to"] == ["<all>", "Alice", "Mike"]
    assert ps[0]["content_out"] == "[Message] from User to Alice, Mike: hi"
    assert ps[1]["branch"] == "tl"
    # 名字等于 profile 才转交（publicer 用名字也不行）
    led = _led(solve, [_TL, _ALICE],
               [_msg("m5", "x", publicer="Mike-name", cause_by="")], rounds=1)
    assert led["publishes"][0]["branch"] == "regular"


def check_display_prefix(solve):
    """转成文本时：`send_to == {<all>}` 显示团队负责人名字，否则列收件人
    （去掉 `<all>`）；role 只留 system/user/assistant；metadata 的 agent 优先。"""
    led = _led(solve, [_TL, _ALICE, _BOB], [
        _msg("m1", "a", send_to=["<all>"], publicer="Mike", cause_by=""),
        _msg("m2", "b", send_to=("Alice", "Bob"), publicer="Mike",
             cause_by="", role="system"),
        _msg("m3", "c", sent_from="", metadata={"agent": "Coder"}, role="coder"),
        _msg("m4", "d", sent_from="", role="tool"),
    ], rounds=1)
    ps = led["publishes"]
    assert ps[0]["content_out"] == "[Message] from User to Mike: a"
    assert ps[1]["role_out"] == "system"
    assert ps[1]["content_out"] == "[Message] from User to Alice, Bob: b"
    assert ps[2]["content_out"] == "[Message] from Coder to Alice, Mike: c"
    assert ps[2]["role_out"] == "assistant"          # coder → assistant
    assert ps[3]["content_out"] == "[Message] from User to Alice, Mike: d"
    assert ps[3]["role_out"] == "assistant"          # tool → assistant


def check_direct_chat(solve):
    """`user_defined_recipient`：只有**空转**的收件人进直聊簿；回包由
    `sent_from` 命中簿子消费；私聊时回包被丢但仍记 history。"""
    led = _led(solve, [_TL, _ALICE], [
        _msg("d1", "ask", user_defined_recipient="u"),
        _msg("d2", "ans", sent_from="Alice", send_to=("User",), cause_by=""),
    ], rounds=1)
    ps = led["publishes"]
    assert ps[0]["branch"] == "direct_in"
    assert ps[0]["direct_chat_added"] == ["Alice"]
    assert ps[1]["branch"] == "direct_out"
    assert led["direct_chat_roles"] == []            # 回包把簿子消费掉
    # 收件人已经忙（buffer 非空）→ 不进簿子
    led = _led(solve, [_TL, _ALICE], [
        _msg("pre", "busy"),
        _msg("d1", "ask", user_defined_recipient="u"),
        _msg("d2", "ans", sent_from="Alice", send_to=("User",), cause_by=""),
    ], rounds=1)
    ps = led["publishes"]
    assert ps[1]["direct_chat_added"] == []
    assert ps[2]["branch"] == "regular"              # 不在簿子 → 走普通分支
    assert ps[2]["send_to"] == ["<all>", "Mike", "User"]
    # 私聊：回包不发布，但 history 仍加一条原文
    led = _led(solve, [_TL, _ALICE], [
        _msg("d1", "ask", user_defined_recipient="u"),
        _msg("d2", "ans", sent_from="Alice", send_to=("User",), cause_by=""),
    ], is_public_chat=False, rounds=1)
    ps = led["publishes"]
    assert ps[1]["branch"] == "direct_out"
    assert ps[1]["delivered"] == []
    assert ps[1]["history_added"] == 1
    assert ps[1]["content_out"] is None
    assert led["history_ids"] == ["d1", "d1", "d2"]


def check_history_dedup(solve):
    """`Memory.add` 按整条消息等值去重：同样的消息再发一次 history 不再涨，
    但**投递照旧**（buffer 里会有多份）。"""
    led = _led(solve, [_TL, _ALICE],
               [_msg("s1", "x"), _msg("s1", "x"), _msg("s1", "x")], rounds=1)
    assert [p["history_added"] for p in led["publishes"]] == [2, 0, 0]
    assert led["history_ids"] == ["s1", "s1"]        # conv + 原文各一条
    r = led["roles"]["Alice"]
    assert r["news"] == [["s1", "s1", "s1"]]         # 批内重复全部算 news
    assert r["memory"] == ["s1"]                     # 但记忆只留一份
    # content 变了就不是同一条
    led = _led(solve, [_TL, _ALICE],
               [_msg("s2", "x"), _msg("s2", "x!"), _msg("s2", "x")], rounds=1)
    assert [p["history_added"] for p in led["publishes"]] == [2, 2, 0]


def check_watch_filter(solve):
    """`_observe` 的两道门：`cause_by in watch` **或** 名字在收件人里；
    再过一道「不在已有记忆里」。"""
    led = _led(solve, [_TL, _ALICE, _role("Bob", watch=())], [
        _msg("w1", "a", send_to=("Alice",), cause_by="WriteCode"),
        _msg("w2", "b", send_to=("Alice",), cause_by="Other"),
        _msg("w3", "c", send_to=("Bob",), cause_by="Other"),
    ], rounds=1)
    assert led["roles"]["Alice"]["news"] == [["w1", "w2"]]
    assert led["roles"]["Bob"]["news"] == [["w3"]]   # 只靠收件人门
    assert led["roles"]["Mike"]["news"] == [["w1", "w2", "w3"]]  # 转交点名
    # 已观测过的不再算 news（第二轮为空）
    led = _led(solve, [_TL, _ALICE], [_msg("w1", "a")], rounds=2)
    assert led["roles"]["Alice"]["news"] == [["w1"], []]
    assert led["roles"]["Alice"]["memory"] == ["w1"]


def check_observe_all(solve):
    """`observe_all_msg_from_buffer`：buffer 里**全部**消息进记忆，
    但只有被 watch 到的进 news。"""
    led = _led(solve, [_TL, _ALICE],
               [_msg("o1", "x", send_to=("Mike",), cause_by="Other")], rounds=2)
    r = led["roles"]["Alice"]
    assert r["news"] == [[], None]                   # 第一轮空手，第二轮已空转
    assert r["memory"] == []                         # 没开开关 → 不入记忆
    led = _led(solve, [_TL, _role("Alice", observe_all=True)],
               [_msg("o1", "x", send_to=("Mike",), cause_by="Other")], rounds=2)
    assert led["roles"]["Alice"]["news"] == [[], None]
    assert led["roles"]["Alice"]["memory"] == ["o1"]  # 存了但没反应
    assert led["roles"]["Alice"]["idle"] is True


def _conv_anchor(mid, content="hi"):
    """构造一条**与投递件逐字段相同**的锚点：MGX 会把 content 改写成
    `[Message] from X to Y: ...`，只有这种形态才会与 buffer 里的副本等值。"""
    return {"id": mid, "content": "[Message] from User to Alice, Mike: " + content,
            "sent_from": "User", "send_to": ["<all>", "Alice", "Mike"],
            "cause_by": "WriteCode", "role": "user", "metadata": {}}


def check_memory_gate(solve):
    """`_observe` 的第三道门是「不在已有记忆里」；只有让锚点与投递件等值
    才能构造出「记忆里有、buffer 里也有」的可达状态。"""
    anchor = _conv_anchor("a0")
    led = _led(solve, [_TL, _role("Alice", recovered=True,
                                  latest_observed=anchor)],
               [_msg("a0", "hi")], rounds=2)
    assert led["publishes"][0]["content_out"] == anchor["content"]
    r = led["roles"]["Alice"]
    assert r["news"] == [["a0"], []]      # 第二轮：投递件已在记忆里 → 挡下
    assert r["memory"] == ["a0"]
    assert r["idle"] is True


def check_enable_memory_off(solve):
    """`enable_memory=False` 时去重门失效：同一条消息重复投递就被重复观测。"""
    led = _led(solve, [_TL, _role("Alice", enable_memory=False)],
               [_msg("r1", "x"), _msg("r1", "x")], rounds=2)
    r = led["roles"]["Alice"]
    assert r["news"] == [["r1", "r1"], []]
    assert r["memory"] == ["r1"]
    assert led["history_ids"] == ["r1", "r1"]
    # 关掉记忆 → 即使是已观测过的投递件也会再进 news
    anchor = _conv_anchor("a0")
    led = _led(solve, [_TL, _role("Alice", enable_memory=False, recovered=True,
                                  latest_observed=anchor)],
               [_msg("a0", "hi")], rounds=2)
    r = led["roles"]["Alice"]
    assert r["news"] == [["a0"], ["a0"]]
    assert r["memory"] == ["a0"]


def check_recover(solve):
    """恢复口径：锚点在记忆里（最后 10 条内）→ 正常读 buffer；锚点不在
    → 只捞锚点、**buffer 这一轮不动**；空转时根本不会走恢复。"""
    anchor = _msg("a0", "old")
    # 锚点不在记忆 → 注入锚点，buffer 留到下一轮
    led = _led(solve, [_TL, _role("Alice", recovered=True,
                                  latest_observed=anchor)],
               [_msg("n1", "new")], rounds=3)
    r = led["roles"]["Alice"]
    assert r["news"] == [["a0"], ["n1"], []]
    assert r["memory"] == ["a0", "n1"]
    assert r["idle"] is True
    # 空转 + 锚点 → 直接跳过（恢复不触发）
    led = _led(solve, [_TL, _role("Alice", recovered=True,
                                  latest_observed=anchor)], [], rounds=2)
    r = led["roles"]["Alice"]
    assert r["news"] == [None, None]
    assert r["memory"] == []
    # 锚点已在记忆最后 10 条内（先发一条同样的消息）→ 直接 drain buffer
    led = _led(solve, [_TL, _role("Alice", recovered=True,
                                  latest_observed=anchor)],
               [_msg("a0", "old")], rounds=2)
    r = led["roles"]["Alice"]
    assert r["news"] == [["a0"], ["a0"]]             # 锚点与投递件是两条
    assert r["memory"] == ["a0", "a0"]
    assert r["buffer"] == []


def check_rounds_and_idle(solve):
    """`is_idle` = 没 news 且 buffer 空；`env.run` 跳过空转角色（账上记
    `None`），所以至少要两轮才全部空转。"""
    led = _led(solve, [_TL, _ALICE], [_msg("m1", "hi")], rounds=1)
    assert led["roles"]["Alice"]["news"] == [["m1"]]      # 观测完仍非空转
    assert led["roles"]["Alice"]["idle"] is False
    led = _led(solve, [_TL, _ALICE], [_msg("m1", "hi")], rounds=2)
    assert led["roles"]["Alice"]["news"] == [["m1"], []]
    assert led["roles"]["Alice"]["idle"] is True
    led = _led(solve, [_TL, _ALICE], [_msg("m1", "hi")], rounds=3)
    assert led["roles"]["Alice"]["news"] == [["m1"], [], None]
    assert led["roles"]["Alice"]["idle"] is True


def check_errors(solve):
    """域约束：缺团队负责人、重名、负轮数、直聊收件人不存在。"""
    def _bad(fn):
        try:
            fn()
        except ValueError:
            return True
        except Exception as exc:                       # noqa: BLE001
            raise AssertionError("wrong error: %r" % (exc,))
        raise AssertionError("expected ValueError")

    assert _bad(lambda: solve(_dc([_ALICE]), _dc([_msg("m1", "x")])))
    assert _bad(lambda: solve(_dc([_TL, _ALICE, _ALICE]),
                              _dc([_msg("m1", "x")])))
    assert _bad(lambda: solve(_dc([_TL]), [], rounds=-1))
    assert _bad(lambda: solve(_dc([_TL, _ALICE]),
                              _dc([_msg("m1", "x", user_defined_recipient="u",
                                        send_to=["Ghost"])])))
    # 默认 `send_to` 含 `<all>`：直聊场景必须显式给收件人，否则算错
    assert _bad(lambda: solve(_dc([_TL, _ALICE]),
                              _dc([{"id": "m2", "content": "x",
                                    "user_defined_recipient": "u"}])))


def check_default_args(solve):
    """默认值：tl_name="Mike"、is_public_chat=True、rounds=1、
    all_tag="<all>"、no_one_tag="no one"。"""
    led = _compare(solve, [_TL, _ALICE], [_msg("m1", "hi")])
    p = led["publishes"][0]
    assert p["branch"] == "regular"
    assert "Mike" in p["send_to"] and "<all>" in p["send_to"]
    assert led["roles"]["Alice"]["news"] == [["m1"]]
    assert len(led["roles"]["Alice"]["news"]) == 1      # rounds 默认 1
    led = _compare(solve, [_TL, _ALICE],
                   [_msg("m2", "x", send_to=("no one",), publicer="Mike",
                         cause_by="")])
    assert led["publishes"][0]["branch"] == "skipped"


def check_random_cross(solve):
    """随机交叉：随机角色表（含自定义地址/开关/恢复锚点）与随机消息流。"""
    pools = ["Alice", "Bob", "Mike", "Cara"]
    causes = ["WriteCode", "Other", "UserRequirement", ""]
    for seed in range(50):
        rng = random.Random(seed)
        roles = []
        for name in ["Mike", "Alice", "Bob", "Cara"]:
            r = _role(name, watch=rng.sample(causes, rng.randint(0, 2)))
            if rng.random() < 0.3:
                r["addresses"] = rng.sample(
                    ["Mike", "Alice", "Bob", "x.Role", "bot"], rng.randint(1, 2))
            if rng.random() < 0.2:
                r["enable_memory"] = False
            if rng.random() < 0.2:
                r["observe_all"] = True
            if rng.random() < 0.15:
                r["recovered"] = True
                r["latest_observed"] = _msg(
                    "a%d" % rng.randint(0, 2), "old",
                    send_to=(name,), cause_by=rng.choice(causes))
            roles.append(r)
        n_msg = rng.randint(0, 5)
        msgs = []
        for i in range(n_msg):
            to = rng.sample(pools, rng.randint(1, 2))
            d = _msg("q%d" % i, "c%d" % rng.randint(0, 1),
                     sent_from=rng.choice(["User", "Mike", "Alice"]),
                     send_to=tuple(to), cause_by=rng.choice(causes),
                     role=rng.choice(["user", "assistant", "coder"]))
            if rng.random() < 0.25:
                d["publicer"] = "Mike"
            if rng.random() < 0.2:
                d["user_defined_recipient"] = "u"
                d["send_to"] = [rng.choice(pools)]        # 直聊必须点名
            if rng.random() < 0.2:
                d["metadata"] = {"agent": rng.choice(["Coder", ""])}
            msgs.append(d)
        _compare(solve, roles, msgs,
                 is_public_chat=rng.random() < 0.7,
                 rounds=rng.randint(0, 3))


def check_example(solve):
    """run 模式样例：一条普通消息 → 转交 TL、广播 → 两轮观测后空转。"""
    led = _led(solve, [_TL, _ALICE], [_msg("m1", "hi")], rounds=2)
    p = led["publishes"][0]
    assert p["branch"] == "regular"
    assert p["delivered"] == ["Alice", "Mike"]
    assert p["history_added"] == 2
    assert p["content_out"] == "[Message] from User to Alice, Mike: hi"
    assert led["roles"]["Alice"]["news"] == [["m1"], []]
    assert led["roles"]["Alice"]["idle"] is True


check_example.example = True
