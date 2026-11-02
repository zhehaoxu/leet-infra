"""TextGrad 文本反传图账（arXiv 2406.07496 §2 + zou-group/textgrad main）。

TextGrad 把「用自然语言反馈做梯度」摊成一张计算图：`Variable` 是节点，
每次算子调用建一个新节点并挂一个 `grad_fn`；`root.backward()` 先做一遍
DFS 后序拓扑排序，再**逆序**逐节点「先归约自己的梯度、再把梯度推给前驱」。

official 里所有容器都是 `set`（`predecessors`、`gradients`、角色描述集合），
迭代序不保证，因此本题把三处集合统一规范成「按加入顺序去重」，其余规则
逐条对齐官方实现：

* `Variable.__init__`：`requires_grad=False` 但存在 `requires_grad=True`
  的前驱 → 直接抛异常（不等反传）。
* `backward()`：只把**根节点**的 `gradients` 清空，中间节点留着上一轮的
  梯度继续累积；`requires_grad=False` 的节点整段跳过（不清梯度、不调
  `grad_fn`）；处理顺序是 DFS 后序的**逆序**。
* `sum` / `add`（`Variable.__add__`）的后向是「幂等广播」：把求和节点的
  梯度文本按模板塞给每个前驱，并且**每上一次加两个完全相同的梯度条目**
  （官方连着 `add` 两次，第二条不带归约标记）。
* `aggregate` 的后向有个循环作用域 bug：`for` 里只赋值不推送，循环结束后
  只给**声明序最后一个**前驱推一条梯度。
* `LLMCall` 的后向：响应自己没梯度走 `base`（直接评估变量），有梯度走
  `chain`（把响应梯度当链式反馈带上），两条路都只推**一条**梯度，且跳过
  `requires_grad=False` 的前驱。
* `_check_and_reduce_gradients`：没有归约标记 / 梯度文本为空 / 只有一条
  梯度 → 原样返回；否则按标记里的分组 id 分组，每组调一次「总结」并把
  组内 `<FEEDBACK>` 拼成提示词；**没带标记的梯度在归约时被丢掉**。
"""


def _uniq(seq):
    out = []
    for item in seq:
        if item not in out:
            out.append(item)
    return out


def _grad_text(gradients):
    return "\n".join(g["value"] for g in gradients)


def _feedback(role, text):
    return ("Here is the combined feedback we got for this specific "
            "%s and other variables: %s." % (role, text))


def _lookup(table, node_id):
    if node_id not in table:
        raise ValueError("unknown predecessor: %s" % (node_id,))
    return table[node_id]


def solve(nodes, grads, reduces=None, root=None, rounds=1):
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    reduces = dict(reduces or {})
    table, declared = {}, []

    for spec in nodes:
        if not isinstance(spec, dict) or "id" not in spec:
            raise ValueError("bad node spec")
        nid = spec["id"]
        if nid in table:
            raise ValueError("duplicate node id: %s" % (nid,))
        kind = spec.get("kind", "input")

        if kind == "input":
            node = {
                "id": nid, "kind": kind,
                "value": spec.get("value", ""),
                "role": spec.get("role", ""),
                "predecessors": [],
                "requires_grad": bool(spec.get("requires_grad", True)),
                "gradients": [], "grad_fn": None, "reduce_meta": [],
            }
        elif kind == "llm":
            predecessors = []
            system = spec.get("system")
            if system is not None:
                predecessors.append(_lookup(table, system))
            for pid in spec.get("inputs", []):
                predecessors.append(_lookup(table, pid))
            node = {
                "id": nid, "kind": kind,
                "value": spec.get("value", ""),
                "role": spec.get("role", ""),
                "predecessors": predecessors,
                "requires_grad": bool(spec.get("requires_grad", True)),
                "gradients": [], "grad_fn": "llm", "reduce_meta": [],
            }
        elif kind in ("sum", "add", "aggregate"):
            predecessors = [_lookup(table, pid) for pid in spec.get("inputs", [])]
            if kind == "add" and len(predecessors) != 2:
                raise ValueError("add takes exactly two operands")
            roles = ", ".join(_uniq([p["role"] for p in predecessors]))
            if kind == "add":
                role = "%s and %s" % (predecessors[0]["role"],
                                      predecessors[1]["role"])
                requires_grad = (predecessors[0]["requires_grad"]
                                 | predecessors[1]["requires_grad"])
            elif kind == "sum":
                role = "a combination of the following: %s" % (roles,)
                requires_grad = any(p["requires_grad"] for p in predecessors)
            else:
                role = "a combination of the following variables: %s." % (roles,)
                requires_grad = any(p["requires_grad"] for p in predecessors)
            node = {
                "id": nid, "kind": kind,
                "value": "\n".join(p["value"] for p in predecessors),
                "role": role,
                "predecessors": predecessors,
                "requires_grad": bool(requires_grad),
                "gradients": [], "grad_fn": kind, "reduce_meta": [],
            }
            if kind == "aggregate":
                node["reduce_meta"].append({"op": "mean",
                                            "id": spec.get("group", nid)})
        else:
            raise ValueError("unknown kind: %s" % (kind,))

        if (not node["requires_grad"]) and any(p["requires_grad"]
                                               for p in node["predecessors"]):
            raise ValueError(
                "If the variable does not require grad, none of its "
                "predecessors should require grad.")
        table[nid] = node
        declared.append(nid)

    if root is None:
        if not declared:
            raise ValueError("empty graph")
        root = declared[-1]
    if root not in table:
        raise ValueError("unknown root: %s" % (root,))

    prompts = []

    def check_and_reduce(node):
        if not node["reduce_meta"]:
            return list(node["gradients"]), False
        if _grad_text(node["gradients"]) == "":
            return list(node["gradients"]), False
        if len(node["gradients"]) == 1:
            return list(node["gradients"]), False

        buckets, index = [], {}
        for gradient in node["gradients"]:
            for item in gradient["reduce_meta"]:
                gid = item["id"]
                if gid not in index:
                    index[gid] = []
                    buckets.append(gid)
                index[gid].append(gradient)
        fresh = []
        for gid in buckets:
            group = index[gid]
            prompts.append("\n".join("<FEEDBACK>%s</FEEDBACK>" % (g["value"],)
                                     for g in group))
            if gid not in reduces:
                raise KeyError("no reduce output for group %r" % (gid,))
            fresh.append({
                "value": reduces[gid],
                "role": ", ".join(_uniq([g["role"] for g in group])),
                "reduce_meta": [],
            })
        return fresh, True

    def backward_idempotent(node):
        text = _grad_text(node["gradients"])
        pushed = []
        for pred in node["predecessors"]:
            value = "" if text == "" else _feedback(pred["role"], text)
            gradient = {"value": value,
                        "role": "feedback to %s" % (pred["role"],),
                        "reduce_meta": []}
            pred["gradients"].append(gradient)
            if node["reduce_meta"]:
                gradient["reduce_meta"].extend(dict(m)
                                               for m in node["reduce_meta"])
                pred["reduce_meta"].extend(dict(m) for m in node["reduce_meta"])
            pred["gradients"].append({
                "value": value,
                "role": "feedback to %s" % (pred["role"],),
                "reduce_meta": [],
            })
            pushed.append({"to": pred["id"], "count": 2})
        return pushed

    def backward_aggregate(node):
        last = node["predecessors"][-1]
        text = _grad_text(node["gradients"])
        value = "" if text == "" else _feedback(last["role"], text)
        gradient = {"value": value,
                    "role": "feedback to %s" % (last["role"],),
                    "reduce_meta": []}
        last["gradients"].append(gradient)
        if node["reduce_meta"]:
            gradient["reduce_meta"].extend(dict(m) for m in node["reduce_meta"])
            last["reduce_meta"].extend(dict(m) for m in node["reduce_meta"])
        return [{"to": last["id"], "count": 1}]

    def backward_llm(node):
        path = "base" if _grad_text(node["gradients"]) == "" else "chain"
        pushed = []
        for pred in node["predecessors"]:
            if not pred["requires_grad"]:
                continue
            key = "%s:%s:%s" % (node["id"], path, pred["id"])
            if key not in grads:
                raise KeyError("missing gradient for %r" % (key,))
            pred["gradients"].append({
                "value": grads[key],
                "role": "feedback to %s" % (pred["role"],),
                "reduce_meta": [],
            })
            pushed.append({"to": pred["id"], "count": 1})
        return path, pushed

    order, steps = [], []

    for rnd in range(1, rounds + 1):
        topo, seen = [], set()

        def visit(node):
            if node["id"] in seen:
                return
            seen.add(node["id"])
            for pred in node["predecessors"]:
                visit(pred)
            topo.append(node)

        visit(table[root])
        table[root]["gradients"] = []
        order.extend(node["id"] for node in reversed(topo))

        for node in reversed(topo):
            if not node["requires_grad"]:
                steps.append({"round": rnd, "id": node["id"], "skipped": True})
                continue
            before = [g["value"] for g in node["gradients"]]
            fresh, did_reduce = check_and_reduce(node)
            node["gradients"] = fresh
            path, pushed = None, []
            if node["grad_fn"] == "llm":
                path, pushed = backward_llm(node)
            elif node["grad_fn"] in ("sum", "add"):
                pushed = backward_idempotent(node)
            elif node["grad_fn"] == "aggregate":
                pushed = backward_aggregate(node)
            step = {
                "round": rnd, "id": node["id"], "skipped": False,
                "in": before, "reduced": did_reduce,
                "out": [g["value"] for g in fresh],
                "called": node["grad_fn"] is not None,
                "pushed": pushed,
            }
            if path is not None:
                step["path"] = path
            steps.append(step)

    return {
        "order": order,
        "skipped": [s["id"] for s in steps if s["skipped"]],
        "steps": steps,
        "grad_text": {nid: _grad_text(table[nid]["gradients"]) for nid in declared},
        "grad_count": {nid: len(table[nid]["gradients"]) for nid in declared},
        "values": {nid: table[nid]["value"] for nid in declared},
        "roles": {nid: table[nid]["role"] for nid in declared},
        "reduce_prompts": prompts,
        "rounds": rounds,
    }
