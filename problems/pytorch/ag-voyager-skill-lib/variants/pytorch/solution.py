"""Voyager 技能库：同名覆盖、版本化落盘与 top-k 检索账。

对照 MineDojo/Voyager 的 `voyager/agents/skill.py`（127 行）：
- `add_new_skill` 对以 `Deposit useless items into the chest at` 开头的任务直接跳过；
- 技能以 `program_name` 为**字典键**，同名即覆盖（不新增条目），但落盘文件名要从 `V2` 起
  找一个空位；
- 描述是 `async function {name}(bot) {\\n    // {llm 内容}\\n}`；
- `retrieve_skills` 的 k 是 `min(库大小, retrieval_top_k)`，空库直接返回 `[]`；
- `programs` 先拼所有技能代码、再拼控制原语，各自后跟一个空行。
向量库 / 落盘 IO / LLM 全部替换成显式入参。
"""
_GUARD = "Deposit useless items into the chest at"


def _as_text(value, name, allow_empty=False):
    if not isinstance(value, str):
        raise ValueError("%s 必须是字符串" % name)
    if not allow_empty and not value.strip():
        raise ValueError("%s 不能为空" % name)
    return value


def _as_int(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s 必须是整数" % name)
    if minimum is not None and value < minimum:
        raise ValueError("%s 必须 >= %d" % (name, minimum))
    return value


def _wrap_description(name, llm_text):
    return "async function %s(bot) {\n    // %s\n}" % (name, llm_text)


def solve(skills, events, retrieval_top_k=5, dump_files=None, primitives=None):
    if not isinstance(skills, (list, tuple)):
        raise ValueError("skills 必须是序列")
    if not isinstance(events, (list, tuple)):
        raise ValueError("events 必须是序列")
    top_k = _as_int(retrieval_top_k, "retrieval_top_k", 1)
    files = set()
    for i, name in enumerate(dump_files or ()):
        files.add(_as_text(name, "dump_files[%d]" % i, allow_empty=True))
    prims = []
    for i, code in enumerate(primitives or ()):
        prims.append(_as_text(code, "primitives[%d]" % i))

    book, order = {}, []
    for i, raw in enumerate(skills):
        if not isinstance(raw, dict):
            raise ValueError("skills[%d] 必须是字典" % i)
        for key in ("name", "code", "description"):
            if key not in raw:
                raise ValueError("skills[%d] 缺少字段 %r" % (i, key))
        name = _as_text(raw["name"], "skills[%d].name" % i)
        if name in book:
            raise ValueError("skills 内 name 重复")
        book[name] = {"code": _as_text(raw["code"], "skills[%d].code" % i),
                      "description": _as_text(raw["description"],
                                              "skills[%d].description" % i)}
        order.append(name)

    dumps, retrieved, skipped = [], [], []
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("events[%d] 必须是字典" % i)
        kind = event.get("kind")
        if kind == "add":
            task = _as_text(event.get("task"), "add.task")
            name = _as_text(event.get("program_name"), "add.program_name")
            code = _as_text(event.get("program_code"), "add.program_code")
            llm_text = _as_text(event.get("llm_description"),
                                "add.llm_description", allow_empty=True)
            if task.startswith(_GUARD):
                skipped.append({"event": i, "program_name": name})
                continue
            description = _wrap_description(name, llm_text)
            rewritten = name in book
            if rewritten:
                # 同名覆盖：字典键不变（位置也不变），只是内容换新
                index = 2
                while "%sV%d.js" % (name, index) in files:
                    index += 1
                dumped = "%sV%d" % (name, index)
            else:
                dumped = name
                order.append(name)
            book[name] = {"code": code, "description": description}
            files.add(dumped + ".js")
            dumps.append({"event": i, "program_name": name, "file": dumped,
                          "rewritten": rewritten})
        elif kind == "retrieve":
            query = event.get("query")
            if not isinstance(query, str):
                raise ValueError("retrieve.query 必须是字符串")
            distances = event.get("distances")
            if not isinstance(distances, (list, tuple)):
                raise ValueError("retrieve.distances 必须是序列")
            if len(distances) != len(order):
                raise ValueError("retrieve.distances 条数必须等于当前技能数")
            clean = []
            for j, value in enumerate(distances):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("retrieve.distances[%d] 必须是实数" % j)
                clean.append(float(value))
            k = min(len(order), top_k)
            if k == 0:
                retrieved.append({"event": i, "query": query, "k": 0,
                                  "names": [], "codes": []})
                continue
            ranked = sorted(range(len(order)), key=lambda j: (clean[j], j))[:k]
            names = [order[j] for j in ranked]
            retrieved.append({"event": i, "query": query, "k": k, "names": names,
                              "codes": [book[n]["code"] for n in names]})
        else:
            raise ValueError("未知的事件类型: %r" % (kind,))

    parts = []
    for name in order:
        parts.append(book[name]["code"] + "\n\n")
    for code in prims:
        parts.append(code + "\n\n")
    return {"order": list(order), "skills": book, "dumps": dumps,
            "retrieved": retrieved, "skipped": skipped,
            "programs": "".join(parts),
            "count": len(book), "synced": len(book) == len(order)}
