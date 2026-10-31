"""autograd 血缘账的纯 Python 推演。

口径来源（torch 2.14.1 实测，实现即规范）：
- ``TensorImpl::is_leaf`` 定义为「没有 grad_fn」，于是本题里 ``leaf == (node is None)``。
- ``aten/src/ATen/core/Variable.cpp``（``set_history``）：只有求导态且操作数里存在
  「非 inference 的 requires_grad 张量」时结果才挂 backward 节点；inference 张量自己不驱动建图。
- ``aten/src/ATen/native/TensorViews.cpp``：视图沿用基张量的 ``_base`` 与 requires_grad
  （不受 grad mode 影响），``detach`` 切断 ``_base`` 并关掉 requires_grad。
- ``torch/csrc/autograd/engine.cpp``（``run_backward``）：隐式梯度只对标量输出合法，
  反向经过的节点在 ``retain_graph=False`` 时被消费，叶子输入拿到/累加 ``.grad``。
- 就地写会替换主体的 grad_fn，且「requires_grad 的叶、叶的视图」不允许就地写
  （``VariableTypeManifold`` / ``ATenNoVariableHooks`` 的两条报错）。
"""

KEYS = ("ok", "rg", "leaf", "infer", "rank", "node", "view_of", "grad", "grad_id")

CTX = ("grad", "no_grad", "inference_mode")

# 数值/复制类语句：节点名 + 参数个数（1 元或 2 元）
VALUE = {
    "add": ("AddBackward0", 2),
    "mul": ("MulBackward0", 2),
    "neg": ("NegBackward0", 1),
    "sigmoid": ("SigmoidBackward0", 1),
    "relu": ("ReluBackward0", 1),
    "sum": ("SumBackward0", 1),
    "mean": ("MeanBackward0", 1),
    "clone": ("CloneBackward0", 1),
}
VIEW = {"t": "TBackward0", "index": "SelectBackward0", "unsqueeze": "UnsqueezeBackward0"}
# 带 ctx 的创建类语句
CREATING = set(VALUE) | set(VIEW) | {"detach", "leaf"}
MUTATING = {"detach_", "add_", "requires_grad_", "retain_grad", "backward",
            "grad", "grad_none", "grad_zero_", "probe"}
ALL_KINDS = CREATING | MUTATING | {"grad"}

FREE, CONSUMED = 0, 1          # 节点是否已被一次不带 retain_graph 的反向消费
# 反向时真的要回读保存张量的节点：只有它们被释放后再经过才会撞上
# 「Trying to backward through the graph a second time」。add/neg/sum/mean/clone
# 与三种视图的 backward 不读回保存的数据，释放动作对它们没有影响（torch 2.14.1 实测）。
SAVING = frozenset(("MulBackward0", "SigmoidBackward0", "ReluBackward0"))


def _fail():
    return {"ok": False, "rg": None, "leaf": None, "infer": None, "rank": None,
            "node": None, "view_of": None, "grad": None, "grad_id": None}


def _row(t, rg, infer):
    return {"ok": True, "rg": rg, "leaf": t["node"] is None, "infer": infer,
            "rank": t["rank"], "node": t["node"], "view_of": t["vo"],
            "grad": t["grad"], "grad_id": t["gid"]}


def _tensor(idx, rg, infer, rank, node=None, vo=None):
    return {"idx": idx, "rg": rg, "infer": infer, "rank": rank, "node": node, "vo": vo,
            "grad": False, "gid": None, "retain": False,
            "node_id": idx if node is not None else None}


def _int(value, limit, name):
    if type(value) is not int or not 0 <= value < limit:
        raise ValueError(f"非法下标 {name}={value!r}")


def solve(stmts):  # noqa: C901 - 语句表本身就是一张分发表，拆成小函数反而更难读
    if type(stmts) not in (list, tuple):
        raise ValueError("stmts 必须是列表")

    ts = {}        # 创建类语句下标 -> 张量状态
    deps = {}      # 节点 id -> 入边 (("node", nid) / ("leaf", idx))
    host = {}      # 节点 id -> 宿主张量下标
    nodename = {}  # 节点 id -> 节点类名（就地改写会把宿主的 node 换掉，节点自身名字不变）
    consumed = {}  # 节点 id -> 是否已被消费

    def root_of(t):
        """视图血缘的根：vo 记录的就是根下标，根本人 vo 为 None。"""
        return ts[t["vo"]] if t["vo"] is not None else t

    def own_meta(t):
        """自己持有 autograd meta（标志与 .grad 记在自己身上）而非从基张量借。"""
        return t["vo"] is None or t["node"] is not None

    def live_rg(t):
        return t["rg"] if own_meta(t) else root_of(t)["rg"]

    def live_infer(t):
        return t["infer"] if own_meta(t) else root_of(t)["infer"]

    def drives(t):
        """操作数是否让求导结果建节点：借标志的视图也算，inference 张量不算。"""
        return live_rg(t) and not live_infer(t)

    def contribution(t):
        """操作数对新节点的入边：借标志的视图没有 AccumulateGrad 宿主，不留边。"""
        if not drives(t):
            return None
        if t["node_id"] is not None:
            return ("node", t["node_id"])
        if own_meta(t):
            return ("leaf", t["idx"])
        return None

    def add_node(nid, host_idx, refs, name):
        deps[nid] = [r for r in refs if r is not None]
        host[nid] = host_idx
        nodename[nid] = name
        consumed[nid] = FREE

    def walk(start):
        """从 start 节点出发遍历整张反向图。"""
        stack, seen, nodes, targets = [start], set(), [], set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            if consumed[nid]:
                return None
            nodes.append(nid)
            if ts[host[nid]]["retain"]:
                targets.add(host[nid])
            for kind, payload in deps[nid]:
                if kind == "node":
                    stack.append(payload)
                elif ts[payload]["rg"]:
                    targets.add(payload)
        return nodes, targets

    def subject(stmt, i, count):
        _int(stmt[1], i, "a")
        if stmt[1] not in ts:
            raise ValueError(f"主体不是成功创建的张量: {stmt!r}")
        if len(stmt) != count:
            raise ValueError(f"参数个数不对: {stmt!r}")
        return ts[stmt[1]]

    def operand(stmt, i, pos):
        _int(stmt[pos], i, "b")
        if stmt[pos] not in ts:
            raise ValueError(f"引用的语句没有创建张量: {stmt!r}")
        return ts[stmt[pos]]

    rows = []
    for i, stmt in enumerate(stmts):
        if type(stmt) not in (list, tuple) or not stmt:
            raise ValueError(f"语句必须是非空列表: {stmt!r}")
        kind = stmt[0]
        if kind not in ALL_KINDS:
            raise ValueError(f"未知语句: {kind!r}")
        if kind in CREATING:
            if len(stmt) < 2 or stmt[-1] not in CTX:
                raise ValueError(f"缺少或非法 ctx: {stmt!r}")
            ctx = stmt[-1]
            body = stmt[1:-1]
        else:
            if len(stmt) and stmt[-1] in CTX:
                raise ValueError(f"该语句不接受 ctx: {stmt!r}")
            ctx = "grad"
            body = stmt[1:]

        if kind == "leaf":
            if len(body) != 1 or type(body[0]) is not bool:
                raise ValueError(f"leaf 参数非法: {stmt!r}")
            ts[i] = _tensor(i, body[0], ctx == "inference_mode", 2)
            rows.append(_row(ts[i], body[0], ctx == "inference_mode"))
            continue

        if kind in VALUE:
            name, arity = VALUE[kind]
            if len(body) != arity:
                raise ValueError(f"参数个数不对: {stmt!r}")
            srcs = [operand(stmt, i, p + 1) for p in range(arity)]
            refs = [contribution(s) for s in srcs]
            rg = ctx == "grad" and any(drives(s) for s in srcs)
            if kind in ("sum", "mean"):
                rank = 0
            elif arity == 1:
                rank = srcs[0]["rank"]
            else:
                rank = max(s["rank"] for s in srcs)
            infer = ctx == "inference_mode"
            ts[i] = _tensor(i, rg, infer, rank, name if rg else None)
            if rg:
                add_node(i, i, refs, name)
            rows.append(_row(ts[i], rg, infer))
            continue

        if kind == "detach":
            t = operand(stmt, i, 1)
            ts[i] = _tensor(i, False, live_infer(t), t["rank"])
            rows.append(_row(ts[i], False, live_infer(t)))
            continue

        if kind in VIEW:
            if len(body) != 1:
                raise ValueError(f"参数个数不对: {stmt!r}")
            t = operand(stmt, i, 1)
            if kind == "t":
                rank, bad = t["rank"], t["rank"] > 2
            elif kind == "index":
                rank, bad = t["rank"] - 1, t["rank"] == 0
            else:
                rank, bad = t["rank"] + 1, False
            if bad:
                rows.append(_fail())
                continue
            node = VIEW[kind] if (ctx == "grad" and live_rg(t)) else None
            infer = live_infer(t)
            vo = None if infer else (t["vo"] if t["vo"] is not None else t["idx"])
            ts[i] = _tensor(i, live_rg(t), infer, rank, node, vo)
            if node is not None:
                add_node(i, i, [contribution(t)], node)
            rows.append(_row(ts[i], live_rg(ts[i]), infer))
            continue

        if kind == "grad":
            t = subject(stmt, i, 2)
            if not t["grad"]:
                rows.append(_fail())
                continue
            ts[i] = _tensor(i, False, False, t["rank"])
            rows.append(_row(ts[i], False, False))
            continue

        if kind == "add_":
            s = subject(stmt, i, 3)
            o = operand(stmt, i, 2)
            root = root_of(s)
            if live_rg(s) and (s["node"] is None or
                               (s["vo"] is not None and root["node"] is None and root["rg"])):
                rows.append(_fail())
                continue
            prev, other = contribution(s), contribution(o)
            if live_rg(s) or other is not None:
                s["rg"], s["node"] = True, "AddBackward0"
                s["node_id"] = i
                add_node(i, s["idx"], [prev, other], "AddBackward0")
            else:
                s["rg"], s["node"], s["node_id"] = False, None, None
            rows.append(_row(s, live_rg(s), live_infer(s)))
            continue

        if kind == "detach_":
            s = subject(stmt, i, 2)
            if s["vo"] is not None:
                rows.append(_fail())
                continue
            s["node"], s["node_id"] = None, None
            if not s["infer"]:
                # inference 张量没有可复位的 autograd meta，就地 detach 关不掉它的标志
                s["rg"] = False
            rows.append(_row(s, s["rg"], s["infer"]))
            continue

        if kind == "requires_grad_":
            s = subject(stmt, i, 3)
            if type(stmt[2]) is not bool:
                raise ValueError(f"requires_grad_ 的标志必须是 bool: {stmt!r}")
            flag = stmt[2]
            cur = live_rg(s)
            if flag and live_infer(s):
                rows.append(_fail())
                continue
            if flag == cur:
                rows.append(_row(s, cur, live_infer(s)))
                continue
            if not flag and s["node"] is not None:
                rows.append(_fail())
                continue
            s["rg"] = flag
            rows.append(_row(s, live_rg(s), live_infer(s)))
            continue

        if kind == "retain_grad":
            s = subject(stmt, i, 2)
            if not live_rg(s):
                rows.append(_fail())
                continue
            s["retain"] = True
            rows.append(_row(s, live_rg(s), live_infer(s)))
            continue

        if kind == "grad_none":
            s = subject(stmt, i, 2)
            s["grad"], s["gid"] = False, None
            rows.append(_row(s, live_rg(s), live_infer(s)))
            continue

        if kind == "grad_zero_":
            s = subject(stmt, i, 2)
            if not s["grad"]:
                rows.append(_fail())
                continue
            rows.append(_row(s, live_rg(s), live_infer(s)))
            continue

        if kind == "backward":
            s = subject(stmt, i, 3)
            if type(stmt[2]) is not bool:
                raise ValueError(f"backward 的 retain_graph 必须是 bool: {stmt!r}")
            if s["rank"] > 0 or not live_rg(s):
                rows.append(_fail())
                continue
            if s["node_id"] is None:
                # 借标志的视图自己没有 autograd meta，引擎既找不到节点也无处累加
                if not own_meta(s):
                    rows.append(_fail())
                    continue
                _give_grad(s, i)
                rows.append(_row(s, live_rg(s), live_infer(s)))
                continue
            reached = walk(s["node_id"])
            if reached is None:
                rows.append(_fail())
                continue
            nodes, targets = reached
            for idx in targets:
                _give_grad(ts[idx], i)
            if not stmt[2]:
                for nid in nodes:
                    if nodename[nid] in SAVING:
                        consumed[nid] = CONSUMED
            rows.append(_row(s, live_rg(s), live_infer(s)))
            continue

        if kind == "probe":
            s = subject(stmt, i, 2)
            rows.append(_row(s, live_rg(s), live_infer(s)))
            continue

        raise ValueError(f"未知语句: {kind!r}")

    return rows


def _give_grad(t, i):
    if not t["grad"]:
        t["grad"], t["gid"] = True, i
