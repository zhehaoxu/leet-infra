"""MetaGPT 消息池路由账。

MetaGPT 的 Environment 是 publish-subscribe 消息池：角色发布 `Message`，
环境按角色的 **addresses** 投递到其私有 buffer，角色再在自己的 `_observe`
里按 **watch/cause_by** 与 **收件人** 过滤出一批 news。MGX 版本又在环境层
叠了团队负责人（TeamLeader）转交、公开/私聊、直聊簿记三条规则。

回源 FoundationAgents/MetaGPT main：
`metagpt/environment/base_env.py`（`Environment.publish_message` /
`add_roles` / `set_addresses` / `is_idle`）、
`metagpt/environment/mgx/mgx_env.py`（`publish_message` /
`_publish_message` / `move_message_info_to_content`）、
`metagpt/roles/role.py`（`_watch` / `_observe` / `put_message` / `is_idle`）、
`metagpt/memory/memory.py`（`add` / `get` / `find_news`）、
`metagpt/utils/common.py::is_send_to`、`metagpt/schema.py`（`Message` /
`MessageQueue.pop_all`）。

纯函数口径：LLM 全作显式入参（这里不需要），投递与观测都是确定性的
集合/队列运算。
"""

_ROUTE_ALL = "<all>"


def _norm(raw, all_tag):
    """把一条消息入参规范成内部 dict（`send_to` 变可改集合）。"""
    return {
        "id": raw["id"],
        "content": raw["content"],
        "role": raw.get("role", "user"),
        "cause_by": raw.get("cause_by", ""),
        "sent_from": raw.get("sent_from", ""),
        "send_to": set(raw.get("send_to", [all_tag])),
        "metadata": dict(raw.get("metadata", {})),
        "publicer": raw.get("publicer", ""),
        "user_defined": raw.get("user_defined_recipient", ""),
    }


def _key(m):
    """`Message` 的等值口径（pydantic 比全部字段）；只取本题建模的字段。"""
    return (m["id"], m["content"], m["role"], m["cause_by"], m["sent_from"],
            tuple(sorted(m["send_to"])), tuple(sorted(m["metadata"].items())))


def _add(store, keys, m):
    """`Memory.add`：等值去重后追加，返回是否真的加了。"""
    k = _key(m)
    if k in keys:
        return False
    keys.add(k)
    store.append(m)
    return True


def solve(roles, messages, tl_name="Mike", is_public_chat=True, rounds=1,
          all_tag=_ROUTE_ALL, no_one_tag="no one"):
    """返回 {"publishes", "roles", "history_ids", "direct_chat_roles"}。"""
    if rounds < 0:
        raise ValueError("negative rounds")
    order = [r["name"] for r in roles]
    by_name = {r["name"]: r for r in roles}
    if len(by_name) != len(order):
        raise ValueError("duplicate role name")
    addr = {}
    for r in roles:
        a = r.get("addresses")
        addr[r["name"]] = set(a) if a is not None else {r["cls"], r["name"]}
    if tl_name not in by_name:
        raise ValueError("missing team leader: %r" % (tl_name,))
    tl = by_name[tl_name]

    buffer = {n: [] for n in order}
    mem = {n: [] for n in order}
    mem_keys = {n: set() for n in order}
    news = {n: [] for n in order}
    latest = {}
    for n, r in by_name.items():
        raw = r.get("latest_observed")
        latest[n] = None if raw is None else _norm(raw, all_tag)
    history, history_keys = [], set()
    direct = set()

    def _idle(name):
        return not news[name] and not buffer[name]

    def _convert(m):
        conv = {}
        for k, v in m.items():
            conv[k] = set(v) if k == "send_to" else dict(v) if k == "metadata" \
                else v
        if conv["role"] not in ("system", "user", "assistant"):
            conv["role"] = "assistant"
        who = conv["metadata"]["agent"] if "agent" in conv["metadata"] \
            else conv["sent_from"]
        if conv["send_to"] == {all_tag}:
            to = tl_name
        else:
            to = ", ".join(sorted(x for x in conv["send_to"] if x != all_tag))
        conv["content"] = "[Message] from %s to %s: %s" % (
            who or "User", to, conv["content"])
        return conv

    def _fanout(m):
        """`_publish_message` + `Environment.publish_message`：广播 + 投递。"""
        if is_public_chat:
            m["send_to"].add(all_tag)
        conv = _convert(m)
        got = []
        for name in order:
            if all_tag in conv["send_to"] or (addr[name] & conv["send_to"]):
                buffer[name].append(conv)
                got.append(name)
        return got, conv

    publishes = []
    for raw in messages:
        m = _norm(raw, all_tag)
        added, published, conv = [], False, None
        branch = None
        if m["user_defined"]:
            branch = "direct_in"
            for name in sorted(m["send_to"]):
                if name not in by_name:
                    raise ValueError("unknown recipient: %r" % (name,))
                if _idle(name):
                    direct.add(name)
                    added.append(name)
            got, conv = _fanout(m)
            published = True
        elif m["sent_from"] in direct:
            branch = "direct_out"
            direct.discard(m["sent_from"])
            if is_public_chat:
                got, conv = _fanout(m)
                published = True
            else:
                got = []
        elif m["publicer"] == tl["profile"]:
            if m["send_to"] == {no_one_tag}:
                branch = "skipped"
                got = []
            else:
                branch = "tl"
                got, conv = _fanout(m)
                published = True
        else:
            branch = "regular"
            m["send_to"].add(tl_name)
            got, conv = _fanout(m)
            published = True
        recorded = 0
        if branch != "skipped":
            if published:
                recorded += 1 if _add(history, history_keys, conv) else 0
            recorded += 1 if _add(history, history_keys, m) else 0
        publishes.append({
            "branch": branch,
            "send_to": sorted(m["send_to"]),
            "delivered": sorted(got),
            "direct_chat_added": sorted(added),
            "history_added": recorded,
            "content_out": None if conv is None else conv["content"],
            "role_out": None if conv is None else conv["role"],
        })

    rounds_out = {n: [] for n in order}
    for _ in range(rounds):
        for name in order:
            if _idle(name):
                rounds_out[name].append(None)
                continue
            r = by_name[name]
            batch = []
            if r.get("recovered") and latest[name] is not None:
                anchor = latest[name]
                recent = [_key(x) for x in mem[name][-10:]]
                if _key(anchor) not in recent:
                    batch = [anchor]        # 从 checkpoint 捞回、buffer 不动
            if not batch:
                batch = buffer[name]
                buffer[name] = []
            if r.get("enable_memory", True):
                old = {_key(x) for x in mem[name]}
            else:
                old = set()
            watch = set(r.get("watch") or ())
            picked = [n for n in batch
                      if (n["cause_by"] in watch or name in n["send_to"])
                      and _key(n) not in old]
            news[name] = picked
            for n in (batch if r.get("observe_all") else picked):
                _add(mem[name], mem_keys[name], n)
            latest[name] = picked[-1] if picked else None
            rounds_out[name].append([n["id"] for n in picked])

    out_roles = {}
    for name in order:
        out_roles[name] = {
            "news": rounds_out[name],
            "memory": [x["id"] for x in mem[name]],
            "buffer": [x["id"] for x in buffer[name]],
            "idle": _idle(name),
        }
    return {"publishes": publishes, "roles": out_roles,
            "history_ids": [x["id"] for x in history],
            "direct_chat_roles": sorted(direct)}
