import contextlib
import random

import torch

# ---------------------------------------------------------------------------
# 题面样例：语句列表 -> 观测行（九列）。
# 每条链都由参考实现与真实 torch 2.14.1 双向实测得到（_torch_program 同一份期望值），
# 行由 _row 构造：缺省的 view_of / grad / grad_id 都是 None。
# ---------------------------------------------------------------------------

R = ("ok", "rg", "leaf", "infer", "rank", "node", "view_of", "grad", "grad_id")


def _row(ok, rg=None, leaf=None, infer=None, rank=None, node=None, vo=None, grad=False, gid=None):
    if not ok:
        return dict(zip(R, (False, None, None, None, None, None, None, None, None)))
    return dict(zip(R, (True, rg, leaf, infer, rank, node, vo, grad, gid)))


EXAMPLES = [
    # 1 建图与消费
    ([["leaf", True, "grad"],
      ["mul", 0, 0, "grad"],
      ["sum", 1, "grad"],
      ["backward", 2, False],
      ["backward", 2, False],
      ["backward", 2, True],
      ["probe", 0]],
     [
      _row(1, True, True, False, 2),
      _row(1, True, False, False, 2, "MulBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(0),
      _row(0),
      _row(1, True, True, False, 2, grad=True, gid=3)]),

    # 2 梯度缓冲的身份
    ([["leaf", True, "grad"],
      ["mul", 0, 0, "grad"],
      ["sum", 1, "grad"],
      ["backward", 2, True],
      ["probe", 0],
      ["backward", 2, True],
      ["probe", 0],
      ["grad_none", 0],
      ["probe", 0],
      ["backward", 2, True],
      ["probe", 0],
      ["grad_zero_", 0],
      ["probe", 0]],
     [
      _row(1, True, True, False, 2),
      _row(1, True, False, False, 2, "MulBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, True, False, 2, grad=True, gid=3),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, True, False, 2, grad=True, gid=3),
      _row(1, True, True, False, 2),
      _row(1, True, True, False, 2),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, True, False, 2, grad=True, gid=9),
      _row(1, True, True, False, 2, grad=True, gid=9),
      _row(1, True, True, False, 2, grad=True, gid=9)]),

    # 3 三种 ctx
    ([["leaf", True, "grad"],
      ["mul", 0, 0, "no_grad"],
      ["mul", 0, 0, "inference_mode"],
      ["leaf", True, "inference_mode"],
      ["detach", 3, "grad"],
      ["backward", 3, False],
      ["neg", 3, "inference_mode"],
      ["probe", 4]],
     [
      _row(1, True, True, False, 2),
      _row(1, False, True, False, 2),
      _row(1, False, True, True, 2),
      _row(1, True, True, True, 2),
      _row(1, False, True, True, 2),
      _row(0),
      _row(1, False, True, True, 2),
      _row(1, False, True, True, 2)]),

    # 4 视图血缘
    ([["leaf", True, "grad"],
      ["t", 0, "grad"],
      ["index", 1, "grad"],
      ["sum", 2, "grad"],
      ["backward", 3, False],
      ["t", 0, "no_grad"],
      ["index", 5, "no_grad"],
      ["add_", 5, 5],
      ["add_", 2, 2],
      ["detach_", 1],
      ["detach_", 0]],
     [
      _row(1, True, True, False, 2),
      _row(1, True, False, False, 2, "TBackward0", vo=0),
      _row(1, True, False, False, 1, "SelectBackward0", vo=0),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, True, False, 2, vo=0),
      _row(1, True, True, False, 1, vo=0),
      _row(0),
      _row(0),
      _row(0),
      _row(1, False, True, False, 2, grad=True, gid=4)]),

    # 5 detach/clone/就地拉叶进图
    ([["leaf", True, "grad"],
      ["detach", 0, "grad"],
      ["clone", 0, "grad"],
      ["backward", 1, False],
      ["backward", 2, False],
      ["leaf", False, "grad"],
      ["add_", 5, 0],
      ["probe", 5],
      ["grad", 0],
      ["grad", 5]],
     [
      _row(1, True, True, False, 2),
      _row(1, False, True, False, 2),
      _row(1, True, False, False, 2, "CloneBackward0"),
      _row(0),
      _row(0),
      _row(1, False, True, False, 2),
      _row(1, True, False, False, 2, "AddBackward0"),
      _row(1, True, False, False, 2, "AddBackward0"),
      _row(0),
      _row(0)]),

    # 6 requires_grad_ 与 retain_grad
    ([["leaf", False, "grad"],
      ["requires_grad_", 0, True],
      ["mul", 0, 0, "grad"],
      ["retain_grad", 2],
      ["sum", 2, "grad"],
      ["backward", 4, False],
      ["probe", 2],
      ["grad", 2],
      ["requires_grad_", 2, True],
      ["probe", 0],
      ["requires_grad_", 0, False],
      ["probe", 0]],
     [
      _row(1, False, True, False, 2),
      _row(1, True, True, False, 2),
      _row(1, True, False, False, 2, "MulBackward0"),
      _row(1, True, False, False, 2, "MulBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, False, False, 2, "MulBackward0", grad=True, gid=5),
      _row(1, False, True, False, 2),
      _row(1, True, False, False, 2, "MulBackward0", grad=True, gid=5),
      _row(1, True, True, False, 2, grad=True, gid=5),
      _row(1, False, True, False, 2, grad=True, gid=5),
      _row(1, False, True, False, 2, grad=True, gid=5)]),

    # 7 rank 边界与标量叶反向
    ([["leaf", True, "grad"],
      ["index", 0, "grad"],
      ["index", 1, "grad"],
      ["index", 2, "grad"],
      ["unsqueeze", 2, "grad"],
      ["t", 4, "grad"],
      ["t", 0, "no_grad"],
      ["index", 6, "no_grad"],
      ["index", 7, "no_grad"],
      ["backward", 8, False],
      ["probe", 8],
      ["probe", 0]],
     [
      _row(1, True, True, False, 2),
      _row(1, True, False, False, 1, "SelectBackward0", vo=0),
      _row(1, True, False, False, 0, "SelectBackward0", vo=0),
      _row(0),
      _row(1, True, False, False, 1, "UnsqueezeBackward0", vo=0),
      _row(1, True, False, False, 1, "TBackward0", vo=0),
      _row(1, True, True, False, 2, vo=0),
      _row(1, True, True, False, 1, vo=0),
      _row(1, True, True, False, 0, vo=0),
      _row(0),
      _row(1, True, True, False, 0, vo=0),
      _row(1, True, True, False, 2)]),

    # 8 inference 张量当操作数
    ([["leaf", True, "grad"],
      ["leaf", False, "inference_mode"],
      ["add", 0, 1, "grad"],
      ["sum", 2, "grad"],
      ["backward", 3, False],
      ["probe", 0],
      ["probe", 1],
      ["grad", 1],
      ["mul", 1, 1, "no_grad"]],
     [
      _row(1, True, True, False, 2),
      _row(1, False, True, True, 2),
      _row(1, True, False, False, 2, "AddBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, True, False, 2, grad=True, gid=4),
      _row(1, False, True, True, 2),
      _row(0),
      _row(1, False, True, False, 2)]),

    # 9 共用中间节点的两条路径
    ([["leaf", True, "grad"],
      ["leaf", True, "grad"],
      ["mul", 0, 1, "grad"],
      ["sum", 2, "grad"],
      ["backward", 3, False],
      ["probe", 0],
      ["probe", 1],
      ["mul", 2, 0, "grad"],
      ["sum", 7, "grad"],
      ["backward", 8, True],
      ["probe", 7],
      ["probe", 0]],
     [
      _row(1, True, True, False, 2),
      _row(1, True, True, False, 2),
      _row(1, True, False, False, 2, "MulBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, True, False, 2, grad=True, gid=4),
      _row(1, True, True, False, 2, grad=True, gid=4),
      _row(1, True, False, False, 2, "MulBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(0),
      _row(1, True, False, False, 2, "MulBackward0"),
      _row(1, True, True, False, 2, grad=True, gid=4)]),

    # 10 就地写改写图
    ([["leaf", True, "grad"],
      ["leaf", True, "grad"],
      ["mul", 0, 0, "grad"],
      ["add_", 2, 1],
      ["sum", 2, "grad"],
      ["backward", 4, True],
      ["probe", 1],
      ["add_", 2, 0],
      ["sum", 2, "grad"],
      ["backward", 8, False],
      ["probe", 2]],
     [
      _row(1, True, True, False, 2),
      _row(1, True, True, False, 2),
      _row(1, True, False, False, 2, "MulBackward0"),
      _row(1, True, False, False, 2, "AddBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, True, False, 2, grad=True, gid=5),
      _row(1, True, False, False, 2, "AddBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, False, False, 0, "SumBackward0"),
      _row(1, True, False, False, 2, "AddBackward0")]),
]

# ---------------------------------------------------------------------------
# torch 侧 oracle
# ---------------------------------------------------------------------------

CREATING = {"leaf", "add", "mul", "neg", "sigmoid", "relu", "sum", "mean",
            "t", "index", "unsqueeze", "detach", "clone", "grad"}

KEYS_ORDER = ("ok", "rg", "leaf", "infer", "rank", "node", "view_of", "grad", "grad_id")

EXPECTED_ERRORS = (
    "grad can be implicitly created only for scalar outputs",
    "does not require grad and does not have a grad_fn",
    "Trying to backward through the graph a second time",
    "a leaf Variable that requires grad is being used in an in-place operation",
    "a view of a leaf Variable that requires grad is being used in an in-place operation",
    "Can't detach views in-place",
    "you can only change requires_grad flags of leaf variables",
    "can't retain_grad on Tensor that has requires_grad=False",
    "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed",
    "invalid index of a 0-dim tensor",
    "t() expects a tensor with <= 2",
    "NoneType",
    "A view was created in no_grad mode",
    "Inplace with an inference",
)


class OutOfDomain(Exception):
    """torch 报的是本题约定「程序保证不会出现」的错，语句作废、换一条。"""


def _ctx(kind):
    if kind == "no_grad":
        return torch.no_grad()
    if kind == "inference_mode":
        return torch.inference_mode()
    return contextlib.nullcontext()


class Runner:
    """真实执行语句并观测 torch 的元数据。"""

    def __init__(self):
        self.ts = []        # 语句下标 -> 张量（只有创建类语句占位）
        self.idx_of = {}    # id(张量) -> 语句下标
        self.alloc_of = {}  # id(.grad 缓冲) -> 分配它的 backward 语句下标
        self.keep = []      # 兜住引用，避免 id 复用把身份判据带歪

    def submit(self, stmt):
        i = len(self.ts)
        self.ts.append(None)
        before = {id(t.grad) for t in self.ts if t is not None and t.grad is not None}
        try:
            row = self._step(i, stmt)
        except Exception as exc:
            msg = str(exc)
            if not any(e in msg for e in EXPECTED_ERRORS):
                self.ts.pop()
                raise OutOfDomain(msg) from exc
            after = {id(t.grad) for t in self.ts if t is not None and t.grad is not None}
            if after != before:
                # 反向跑到一半已经给别的张量分配了梯度，本题保证不出现这种语句
                self.ts.pop()
                raise OutOfDomain("partial backward")
            return {k: None if k != "ok" else False for k in KEYS_ORDER}
        return row

    def _snapshot(self, t):
        base = t._base
        grad = t.grad
        if grad is not None:
            self.keep.append(grad)
        return {
            "ok": True,
            "rg": bool(t.requires_grad),
            "leaf": bool(t.is_leaf),
            "infer": bool(torch.is_inference(t)),
            "rank": int(t.dim()),
            "node": None if t.grad_fn is None else type(t.grad_fn).__name__,
            "view_of": None if base is None else self.idx_of[id(base)],
            "grad": grad is not None,
            "grad_id": None if grad is None else self.alloc_of.get(id(grad)),
        }

    def _register(self, i, t):
        self.idx_of[id(t)] = i
        self.keep.append(t)
        self.ts[i] = t

    def _step(self, i, st):  # noqa: C901 - 语句分发表，与参考实现一一对应
        kind, args = st[0], list(st[1:])
        if kind in CREATING and kind != "grad":
            ctx = args.pop()
            with _ctx(ctx):
                t = self._create(kind, args)
            self._register(i, t)
            return self._snapshot(t)

        if kind == "grad":
            g = self.ts[args[0]].grad
            if g is None:
                raise TypeError("NoneType 没有 grad 属性")
            self._register(i, g)
            return self._snapshot(g)
        if kind == "add_":
            self.ts[args[0]].add_(self.ts[args[1]])
            return self._snapshot(self.ts[args[0]])
        if kind == "detach_":
            self.ts[args[0]].detach_()
            return self._snapshot(self.ts[args[0]])
        if kind == "requires_grad_":
            self.ts[args[0]].requires_grad_(args[1])
            return self._snapshot(self.ts[args[0]])
        if kind == "retain_grad":
            self.ts[args[0]].retain_grad()
            return self._snapshot(self.ts[args[0]])
        if kind == "backward":
            self.ts[args[0]].backward(retain_graph=args[1])
            for j, t in enumerate(self.ts):
                if t is None or t.grad is None:
                    continue
                if id(t.grad) not in self.alloc_of:
                    self.keep.append(t.grad)
                    self.alloc_of[id(t.grad)] = i
            return self._snapshot(self.ts[args[0]])
        if kind == "grad_none":
            self.ts[args[0]].grad = None
            return self._snapshot(self.ts[args[0]])
        if kind == "grad_zero_":
            self.ts[args[0]].grad.zero_()
            return self._snapshot(self.ts[args[0]])
        if kind == "probe":
            return self._snapshot(self.ts[args[0]])
        raise AssertionError(kind)

    def _create(self, kind, args):
        if kind == "leaf":
            return torch.ones(2, 3, requires_grad=args[0])
        if kind == "grad":
            raise AssertionError("grad 语句不走 _create")
        t = self.ts[args[0]]
        if kind == "add":
            return t + self.ts[args[1]]
        if kind == "mul":
            return t * self.ts[args[1]]
        if kind == "index":
            return t[0]
        if kind == "unsqueeze":
            return t.unsqueeze(0)
        return getattr(t, kind)()


def _torch_program(stmts):
    r = Runner()
    return [r.submit(st) for st in stmts]


# ---------------------------------------------------------------------------
# 随机程序生成：候选语句在真实 torch 上试跑，越界（OutOfDomain）就换一条
# ---------------------------------------------------------------------------

UNARY_VALUE = ["neg", "sigmoid", "relu", "sum", "mean", "clone"]
VIEWS = ["t", "index", "unsqueeze"]
MUTATES = ["detach_", "retain_grad", "backward", "grad_none", "grad_zero_", "probe", "grad"]


def _created(stmts, rows):
    """成功创建了张量的语句下标：只有这些下标能被后续语句引用。"""
    return [i for i, (st, r) in enumerate(zip(stmts, rows)) if r["ok"] and st[0] in CREATING]


def _usable(created, rows):
    """可作为操作数的张量：排除「inference 且 requires_grad」这种本题不出现的组合。"""
    return [i for i in created if not (rows[i]["infer"] and rows[i]["rg"])]


def _gen_stmt(rng, stmts, rows):
    created = _created(stmts, rows)
    pool = _usable(created, rows)
    ctx = rng.choice(["grad", "grad", "grad", "no_grad", "inference_mode"])
    pick = rng.random()
    if not pool or pick < 0.12:
        return ["leaf", rng.random() < 0.6, ctx]
    if pick < 0.34:
        a = rng.choice(pool)
        cand = [b for b in pool
                if rows[a]["rank"] == rows[b]["rank"] or min(rows[a]["rank"], rows[b]["rank"]) == 0]
        if not cand:
            return ["neg", a, ctx]
        return [rng.choice(["add", "mul"]), a, rng.choice(cand), ctx]
    if pick < 0.62:
        return [rng.choice(UNARY_VALUE + VIEWS), rng.choice(pool), ctx]
    if created and rng.random() < 0.25:
        # requires_grad_ 只改「自己持有 requires_grad」的张量：视图的标志是从基张量借的，
        # 改了也不会变（torch 2.14.1 实测），本题约定不出现这种语句。
        own = [i for i in created if rows[i]["view_of"] is None]
        return ["requires_grad_", rng.choice(own), rng.random() < 0.5]
    a = rng.choice(created) if rng.random() < 0.5 else rng.choice(pool)
    kind = rng.choice([k for k in MUTATES if k != "requires_grad_"])
    if kind == "backward":
        return ["backward", a, rng.random() < 0.5]
    return [kind, a]


def _gen_program(rng, n):
    r = Runner()
    stmts, rows = [], []
    attempts = 0
    while len(stmts) < n and attempts < n * 30:
        attempts += 1
        stmt = _gen_stmt(rng, stmts, rows)
        try:
            row = r.submit(stmt)
        except OutOfDomain:
            continue
        stmts.append(stmt)
        rows.append(row)
    return stmts, rows


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------

def check_example(solve):
    for stmts, expect in EXAMPLES:
        got = solve(stmts)
        assert got == expect, (stmts, got, expect)
        assert _torch_program([s[:] for s in stmts]) == expect, stmts


def check_random_programs_match_torch(solve):
    total = 0
    fails = 0
    for seed in (7, 17, 27, 37, 47, 57, 67, 77):
        rng = random.Random(seed)
        for _ in range(40):
            stmts, expect = _gen_program(rng, rng.randint(6, 26))
            got = solve(stmts)
            assert got == expect, (seed, stmts, got, expect)
            total += len(stmts)
            fails += sum(1 for row in expect if not row["ok"])
    assert total > 4500, total
    assert fails > 500, fails


def check_backward_graph_consumption(solve):
    """节点消费沿图传播：两条共用同一中间节点的路径，第二条反向必须失败。"""
    prog = [["leaf", True, "grad"], ["leaf", True, "grad"], ["mul", 0, 1, "grad"],
            ["sum", 2, "grad"], ["backward", 3, False], ["probe", 0], ["probe", 1],
            ["mul", 2, 0, "grad"], ["sum", 7, "grad"], ["backward", 8, True],
            ["probe", 7], ["probe", 0]]
    expect = _torch_program([s[:] for s in prog])
    assert solve(prog) == expect
    assert expect[9]["ok"] is False
    assert expect[10] == _row(1, True, False, False, 2, "MulBackward0")
    assert expect[11]["grad_id"] == 4


def check_non_saving_nodes_replay(solve):
    """反向跑过后图就「释放」了，但只有反向要回读保存张量的节点（mul/sigmoid/relu）
    真的拦路；sum/add/neg/clone/视图这些不读回，同一条图可以再反好几次。"""
    prog = [["leaf", True, "grad"], ["sum", 0, "grad"], ["backward", 1, False], ["probe", 0],
            ["grad", 0], ["grad_none", 0], ["backward", 1, True], ["probe", 0],
            ["backward", 1, False], ["probe", 0], ["grad", 0], ["probe", 1]]
    expect = _torch_program([s[:] for s in prog])
    assert solve(prog) == expect
    assert expect[3] == _row(1, True, True, False, 2, grad=True, gid=2)
    # 缓冲被清空后再反一次：新一次 backward 重新分配，此后连着的反向仍累加同一块
    assert expect[5] == _row(1, True, True, False, 2)
    assert expect[7] == _row(1, True, True, False, 2, grad=True, gid=6)
    assert expect[9] == _row(1, True, True, False, 2, grad=True, gid=6)

    mul = [["leaf", True, "grad"], ["mul", 0, 0, "grad"], ["sum", 1, "grad"],
           ["backward", 2, False], ["backward", 2, True], ["backward", 1, False],
           ["probe", 0]]
    expect = _torch_program([s[:] for s in mul])
    assert solve(mul) == expect
    # mul 的节点回读保存的输入，第二次反向无论带不带 retain 都不行
    assert expect[4] == _row(0) and expect[5] == _row(0)
    assert expect[6] == _row(1, True, True, False, 2, grad=True, gid=3)


def check_inference_flag_and_detach(solve):
    """inference 张量：detach_ 关不掉它的标志、retain_grad 跟着实时标志走、
    detach 出来的新张量仍然带着 inference 身份。"""
    prog = [["leaf", True, "inference_mode"], ["detach_", 0], ["probe", 0],
            ["requires_grad_", 0, False], ["probe", 0], ["retain_grad", 0],
            ["sum", 0, "grad"], ["detach", 0, "grad"], ["probe", 7]]
    expect = _torch_program([s[:] for s in prog])
    assert solve(prog) == expect
    # 就地 detach 只能复位「自己的 autograd meta」，inference 张量没有可复位的 meta
    assert expect[1] == _row(1, True, True, True, 2)
    assert expect[3] == _row(1, False, True, True, 2)
    assert expect[5] == _row(0)
    # 关掉标志后 sum 不进图；detach 出来的张量不再要梯度、但仍是 inference 张量
    assert expect[6] == _row(1, False, True, False, 0)
    assert expect[7] == _row(1, False, True, True, 2)


def check_diamond_allocates_one_buffer(solve):
    """菱形的两条边指向同一个中间节点：叶只开一个梯度缓冲；该节点被消费后，
    第二次反向整条失败，被 retain_grad 留住的中间张量也就永远拿不到缓冲。"""
    prog = [["leaf", True, "grad"], ["mul", 0, 0, "grad"], ["add", 1, 1, "grad"],
            ["sum", 2, "grad"], ["backward", 3, False], ["probe", 0], ["probe", 1],
            ["retain_grad", 1], ["mul", 0, 1, "grad"], ["backward", 8, True],
            ["probe", 0], ["probe", 1], ["probe", 8]]
    expect = _torch_program([s[:] for s in prog])
    assert solve(prog) == expect
    assert expect[5] == _row(1, True, True, False, 2, grad=True, gid=4)
    assert expect[6] == _row(1, True, False, False, 2, "MulBackward0")
    assert expect[9] == _row(0)
    assert expect[10] == _row(1, True, True, False, 2, grad=True, gid=4)
    assert expect[12] == _row(1, True, False, False, 2, "MulBackward0")


def check_inplace_replaces_grad_fn(solve):
    prog = [["leaf", True, "grad"], ["mul", 0, 0, "grad"], ["leaf", True, "grad"],
            ["add_", 1, 2], ["sum", 1, "grad"], ["backward", 4, True],
            ["probe", 0], ["probe", 2], ["add_", 1, 0], ["probe", 1]]
    expect = _torch_program([s[:] for s in prog])
    assert solve(prog) == expect
    assert expect[3] == _row(1, True, False, False, 2, "AddBackward0")
    assert expect[9] == _row(1, True, False, False, 2, "AddBackward0")


def check_region_and_lineage(solve):
    prog = [["leaf", True, "grad"], ["t", 0, "grad"], ["t", 1, "grad"],
            ["index", 2, "no_grad"], ["detach", 3, "inference_mode"],
            ["leaf", False, "inference_mode"], ["t", 5, "grad"], ["clone", 5, "grad"],
            ["add", 0, 5, "grad"], ["sum", 8, "grad"], ["backward", 9, False],
            ["probe", 0], ["unsqueeze", 1, "grad"]]
    expect = _torch_program([s[:] for s in prog])
    assert solve(prog) == expect
    assert expect[2]["view_of"] == 0 and expect[3]["view_of"] == 0
    assert expect[3]["rg"] is True and expect[3]["node"] is None and expect[3]["leaf"] is True
    assert expect[6]["view_of"] is None and expect[6]["infer"] is True
    assert expect[8]["node"] == "AddBackward0" and expect[11]["grad_id"] == 10


def check_leaf_flag_and_grad_buffer(solve):
    """叶的开关、梯度缓冲的身份，以及 inference 张量沾不得的两条边界。"""
    prog = [["leaf", False, "grad"], ["requires_grad_", 0, True], ["leaf", True, "no_grad"],
            ["leaf", True, "inference_mode"], ["detach", 3, "grad"], ["mul", 0, 2, "grad"],
            ["sum", 5, "grad"], ["backward", 6, False], ["grad", 0], ["grad", 2],
            ["grad", 4], ["grad_none", 0], ["grad", 0], ["grad_zero_", 2], ["probe", 2],
            ["backward", 6, True], ["requires_grad_", 4, True], ["requires_grad_", 2, False],
            ["probe", 2], ["grad", 2]]
    expect = _torch_program([s[:] for s in prog])
    assert solve(prog) == expect
    # no_grad 里造出来的叶照样带着 requires_grad；inference 叶被 detach 出来仍是 inference 张量
    assert expect[2] == _row(1, True, True, False, 2)
    assert expect[4] == _row(1, False, True, True, 2)
    assert expect[8] == _row(1, False, True, False, 2)
    assert expect[9] == _row(1, False, True, False, 2)
    # 没有缓冲就取不了梯度快照；grad_none 之后快照失败，grad_zero_ 不换缓冲身份
    assert expect[10] == _row(0)
    assert expect[12] == _row(0)
    assert expect[13] == _row(1, True, True, False, 2, grad=True, gid=7)
    # 图已被第一次反向消费掉，第二次反向整条失败
    assert expect[15] == _row(0)
    # inference 张量一律不许开标志；关标志只动元数据，不动已有的梯度缓冲
    assert expect[16] == _row(0)
    assert expect[17] == _row(1, False, True, False, 2, grad=True, gid=7)
    assert expect[19] == _row(1, False, True, False, 2)


def check_borrowed_view_flag(solve):
    """没带 autograd meta 的视图：标志实时跟着根走，能驱动建图却进不了累加名单。"""
    prog = [["leaf", False, "grad"], ["index", 0, "grad"], ["requires_grad_", 0, True],
            ["probe", 1], ["mul", 1, 1, "grad"], ["sum", 4, "grad"], ["backward", 5, False],
            ["requires_grad_", 0, False], ["probe", 1], ["probe", 0], ["leaf", True, "grad"],
            ["index", 10, "grad"], ["add", 11, 11, "grad"], ["sum", 12, "grad"],
            ["backward", 13, False], ["probe", 11], ["grad", 11], ["grad", 10]]
    expect = _torch_program([s[:] for s in prog])
    assert solve(prog) == expect
    # 视图自己没有元数据：根点亮则它看着像要梯度，根关掉它也跟着关
    assert expect[3] == _row(1, True, True, False, 1, vo=0)
    assert expect[8] == _row(1, False, True, False, 1, vo=0)
    # 反向照建图、照跑通，但借标志的视图不给任何张量开缓冲
    assert expect[6] == _row(1, True, False, False, 0, "SumBackward0")
    assert expect[9]["grad"] is False
    # 带了自己节点的视图才有血缘（vo 指根），取快照也只认自己的缓冲
    assert expect[11] == _row(1, True, False, False, 1, "SelectBackward0", vo=10)
    assert expect[15] == _row(1, True, False, False, 1, "SelectBackward0", vo=10)
    assert expect[16] == _row(0)
    assert expect[17] == _row(1, False, True, False, 2)


def check_view_flag_and_detach(solve):
    """就地 detach 关得掉自己的标志、关不掉 inference 的，也碰不得视图。"""
    prog = [["leaf", True, "grad"], ["t", 0, "grad"], ["detach_", 0], ["probe", 1],
            ["leaf", False, "grad"], ["t", 4, "grad"], ["requires_grad_", 4, True],
            ["probe", 5], ["sum", 5, "grad"], ["backward", 8, False], ["probe", 4],
            ["grad", 4], ["requires_grad_", 1, True], ["detach_", 1], ["grad_none", 4],
            ["requires_grad_", 4, False], ["probe", 5]]
    expect = _torch_program([s[:] for s in prog])
    assert solve(prog) == expect
    # grad 区里从带标志的根切出来的视图自带元数据：根就地 detach 后它的快照仍是 True
    assert expect[1] == _row(1, True, False, False, 2, "TBackward0", vo=0)
    assert expect[2] == _row(1, False, True, False, 2)
    assert expect[3] == _row(1, True, False, False, 2, "TBackward0", vo=0)
    # 借标志的视图反向跑通却不给根开缓冲，非叶的快照也永远取不到
    assert expect[9] == _row(1, True, False, False, 0, "SumBackward0")
    assert expect[10] == _row(1, True, True, False, 2)
    assert expect[11] == _row(0)
    # 同值改写是空操作；视图不许就地 detach
    assert expect[12] == _row(1, True, False, False, 2, "TBackward0", vo=0)
    assert expect[13] == _row(0)
    # 根的标志关掉后，借标志的视图实时跟着关
    assert expect[16] == _row(1, False, True, False, 2, vo=4)


def check_flag_retain_and_scalar_leaf_backward(solve):
    """三条边界：同值改写是空操作、非叶关不掉标志、视图借来的标志反不出梯度。"""
    prog = [["leaf", False, "grad"], ["requires_grad_", 0, False], ["requires_grad_", 0, True],
            ["sum", 0, "grad"], ["retain_grad", 3], ["backward", 3, False], ["grad", 3],
            ["requires_grad_", 6, True], ["requires_grad_", 3, False], ["backward", 6, False],
            ["probe", 6], ["index", 0, "grad"], ["requires_grad_", 11, False],
            ["leaf", True, "grad"], ["index", 13, "no_grad"], ["backward", 14, False],
            ["retain_grad", 14], ["backward", 0, False], ["probe", 14],
            ["leaf", False, "no_grad"], ["requires_grad_", 19, True], ["retain_grad", 19],
            ["probe", 19]]
    expect = _torch_program([s[:] for s in prog])
    assert solve(prog) == expect
    assert expect[1] == _row(1, False, True, False, 2)
    assert expect[8] == _row(0) and expect[12] == _row(0)
    assert expect[9] == _row(1, True, True, False, 0, grad=True, gid=9)
    assert expect[15] == _row(0) and expect[18] == _row(1, True, True, False, 1, vo=13)


def check_invalid_inputs_raise(solve):
    ok_cases = [
        [],
        [["leaf", False, "grad"]],
        [["leaf", True, "grad"], ["backward", 0, True]],
    ]
    for case in ok_cases:
        assert isinstance(solve(case), list), case

    bad = [
        "not-a-list",
        [[]],
        [["nope", 0, "grad"]],
        [["leaf", 1, "grad"]],                        # flag 不是 bool
        [["leaf", True]],                             # 缺 ctx
        [["leaf", True, "enable_grad"]],              # ctx 取值非法
        [["leaf", True, "grad"], ["probe", 0, "grad"]],   # 读取类语句不接受 ctx
        [["leaf", True, "grad"], ["add", 0, "grad"]],     # 参数个数不对
        [["leaf", True, "grad"], ["mul", 0, 1, "grad"]],  # 下标越界
        [["leaf", True, "grad"], ["sum", 1, "grad"]],     # 下标越界
        [["leaf", True, "grad"], ["neg", -1, "grad"]],    # 负下标
        [["leaf", True, "grad"], ["backward", 0, 1]],     # retain_graph 不是 bool
        [["leaf", True, "grad"], ["add_", 0, 1]],         # 操作数引用了尚未创建的语句
        [["leaf", True, "grad"], ["detach_", 99]],
        [["leaf", True, "grad"], ["requires_grad_", 0, "True"]],
        [["leaf", True, "grad"], ["t", True, "grad"]],    # 下标是 bool 而非 int
    ]
    for case in bad:
        try:
            solve(case)
        except ValueError:
            continue
        raise AssertionError(f"非法输入未抛 ValueError: {case}")
