"""Stream of Search 轨迹序列化账的检查点。

每个检查点先拿一份**另一套写法**的 oracle 与 `solve` 的全量账本比对：oracle
先把一层子节点整体物化成列表（用切片取剩余数字），再用"把操作序列重放一遍、
可用数字池 = 初始数字 + 已产出的结果"这种等价表述复核轨迹，而不是照抄一遍
`available` 列表的增删。硬断言再钉住官方那些容易写错的口径。
"""

import re

_MAX = {2: 4, 3: 48, 4: 1152, 5: 46080}
_GOAL = re.compile(r"\d+,\d+ equal: Goal Reached")
_OUTCOME = re.compile(r"^(\d+),(\d+) (equal: Goal Reached|unequal: No Solution)$")
_ARITH = {"+": lambda a, b: a + b,
          "-": lambda a, b: a - b,
          "*": lambda a, b: a * b,
          "/": lambda a, b: a / b}
_BAD_HEAD = "Invalid input: Cannot find the initial state in the first line."


# ---------------------------------------------------------------- oracle --
def _o_ops(a, b):
    """固定四序：加、乘、大减小、大除小（只在整除时出现）。"""
    a, b = int(a), int(b)
    seq = [(a + b, "%d+%d=%d" % (a, b, a + b)),
           (a * b, "%d*%d=%d" % (a, b, a * b))]
    big, small = (b, a) if a <= b else (a, b)
    seq.append((big - small, "%d-%d=%d" % (big, small, big - small)))
    if small and big % small == 0:
        seq.append((big // small, "%d/%d=%d" % (big, small, big // small)))
    return seq


def _o_sum_h(nums, target):
    if len(nums) == 1:
        return abs(nums[0] - target)
    return sum(abs(num - target) for num in nums) / len(nums)


def _o_mult_h(nums, target):
    factors = [factor for factor in range(2, target + 1) if target % factor == 0]
    return sum(min(abs(num - factor) for factor in factors) for num in nums)


_O_H = {"sum": _o_sum_h, "mult": _o_mult_h}


def _o_kids(nums, target, heuristic, threshold):
    """把一层的全部子节点物化出来，再过滤、再按启发式稳定排序。"""
    kids = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            rest = nums[:i] + nums[i + 1:j] + nums[j + 1:]
            for value, operation in _o_ops(nums[i], nums[j]):
                child = rest + [value]
                kids.append({"nums": child, "op": operation,
                             "h": heuristic(child, target)})
    kids = [kid for kid in kids if threshold is None or kid["h"] <= threshold]
    kids.sort(key=lambda kid: kid["h"])
    return kids


def _o_emit(target, nums, ops, prefix, lines, heuristic, threshold):
    lines.append("Current State: %d:%s, Operations: %s" % (target, nums, ops))
    kids = _o_kids(nums, target, heuristic, threshold)
    for pos, kid in enumerate(kids):
        node_id = "%s,%d" % (prefix, pos)
        lines.append("Exploring Operation: %s, Resulting Numbers: %s"
                     % (kid["op"], kid["nums"]))
        if len(kid["nums"]) == 1:
            if kid["nums"][0] == target:
                lines.append("%d,%d equal: Goal Reached" % (kid["nums"][0], target))
                return True
            lines.append("%d,%d unequal: No Solution" % (kid["nums"][0], target))
        else:
            lines.append("Generated Node #%s: %d:%s Operation: %s"
                         % (node_id, target, kid["nums"], kid["op"]))
            lines.append("Moving to Node #%s" % (node_id,))
            if _o_emit(target, kid["nums"], ops + [kid["op"]], node_id,
                       lines, heuristic, threshold):
                return True
        if pos + 1 < len(kids):
            lines.append("Moving to Node #%s" % (prefix,))
            lines.append("Current State: %d:%s, Operations: %s"
                         % (target, nums, ops))
    return False


def _o_head(head):
    prefix = "Current State: "
    if not head.startswith(prefix):
        return None
    rest = head[len(prefix):]
    if ":" not in rest:
        return None
    target, tail = rest.split(":", 1)
    if not target.isdigit() or not tail.startswith("["):
        return None
    close = tail.find("]")
    if close < 0:
        return None
    try:
        nums = [int(piece) for piece in tail[1:close].split(", ")]
    except ValueError:
        return None
    return int(target), nums


def _o_eval(expression):
    match = re.fullmatch(r"(\d+)([-+*/])(\d+)", expression)
    if not match:
        raise ValueError("malformed expression %r" % (expression,))
    return _ARITH[match.group(2)](int(match.group(1)), int(match.group(3)))


def _o_verdict(text):
    head = text.strip().split("\n")[0]
    parsed = _o_head(head)
    if parsed is None:
        return _BAD_HEAD
    target, nums = parsed
    lines = text.replace("<|endoftext|>", "").split("\n")

    stop = None
    for pos, line in enumerate(lines):
        if _GOAL.search(line):
            stop = pos
            break
    if stop is None:
        return "No goal reached statement found."

    before = lines[:stop]
    found = []
    for line in before:
        if not line.startswith("Exploring Operation: "):
            continue
        body = line[len("Exploring Operation: "):]
        marker = body.find(", Resulting Numbers: [")
        if marker < 0 or not body.endswith("]"):
            continue
        if not re.search(r"=\d+$", body[:marker]):
            continue
        found.append(body[:marker])
    if not found:
        return "No operations found leading to the goal."

    final_operation = found[-1]
    predicted = int(final_operation.split("=")[1])
    if predicted != target:
        return "Invalid path: Final operation does not result in target."

    carried = None
    for line in before:
        if line.startswith("Current State: ") and ", Operations: [" in line:
            carried = line.split(", Operations: [", 1)[1][:-1]
    pieces = carried.split(", ") if carried is not None else []
    pieces = [piece.replace("'", "") for piece in pieces] + [final_operation]

    produced = []
    for operation in pieces:
        parts = operation.split("=")
        if len(parts) != 2:
            return "Could not split operation into lhs, rhs"
        left, right = parts
        try:
            value = _o_eval(left)
            if value != int(right):
                return "Invalid operation: %s" % (operation,)
        except Exception as exc:
            return "Error in evaluating operation %s: %s" % (operation, exc)
        pool = nums + produced
        for number in re.findall(r"\d+", left):
            if int(number) not in pool:
                return ("Invalid operation: %s, number %s not available in %s"
                        % (operation, number, pool))
        produced.append(int(right))
    return "Valid path."


def _o_solve(target, nums, heuristic="sum", threshold=None, trace=None):
    if trace is None:
        lines = []
        _o_emit(int(target), [int(num) for num in nums], [], "0", lines,
                _O_H[heuristic], threshold)
        text = "".join(line + "\n" for line in lines)
    else:
        text = trace

    verdict = _o_verdict(text)
    lines = text.split("\n")
    explored = text.count("Exploring Operation")
    parsed = _o_head(text.strip().split("\n")[0])
    size = len(parsed[1]) if parsed else None
    max_nodes = _MAX.get(size)

    outcomes = [match.groups() for match in
                (_OUTCOME.match(line) for line in lines) if match]
    if outcomes:
        value, goal, kind = outcomes[-1]
        final = {"value": int(value), "target": int(goal),
                 "equal": kind == "equal: Goal Reached"}
    else:
        final = None

    rating = 0.0
    if verdict == "Valid path." and max_nodes is not None:
        rating = max(1.0 - (explored + 1) / max_nodes, 0.0)

    return {
        "trace": text,
        "encoded": trace is None,
        "found": any(_GOAL.search(line) for line in lines),
        "states": text.count("Current State"),
        "moves": text.count("Moving to Node"),
        "explored": explored,
        "node_ids": [line[len("Generated Node #"):].split(":", 1)[0]
                     for line in lines if line.startswith("Generated Node #")],
        "ops": [line[len("Exploring Operation: "):].split(", Resulting Numbers: [")[0]
                for line in lines if line.startswith("Exploring Operation: ")],
        "final": final,
        "verdict": verdict,
        "score": explored + 1,
        "max_nodes": max_nodes,
        "rating": rating,
    }


# ------------------------------------------------------------- assertions --
def _cmp(got, want, where="root"):
    if isinstance(want, float) or isinstance(got, float):
        assert isinstance(got, (int, float)) and isinstance(want, (int, float)), \
            "%s: %r vs %r" % (where, got, want)
        assert abs(got - want) <= 1e-12, "%s: %r vs %r" % (where, got, want)
        return
    if isinstance(want, dict):
        assert isinstance(got, dict), "%s: not a dict (%r)" % (where, got)
        assert sorted(got) == sorted(want), \
            "%s: keys %r vs %r" % (where, sorted(got), sorted(want))
        for key in want:
            _cmp(got[key], want[key], where + "/" + str(key))
        return
    if isinstance(want, (list, tuple)):
        assert isinstance(got, (list, tuple)), "%s: not a list (%r)" % (where, got)
        assert len(got) == len(want), \
            "%s: len %d vs %d" % (where, len(got), len(want))
        for pos, (a, b) in enumerate(zip(got, want)):
            _cmp(a, b, where + "[%d]" % (pos,))
        return
    assert got == want, "%s: %r vs %r" % (where, got, want)


def _same(solve, target, nums, heuristic="sum", threshold=None):
    got = solve(target, nums, heuristic=heuristic, threshold=threshold)
    _cmp(got, _o_solve(target, nums, heuristic=heuristic, threshold=threshold))
    return got


def _decoded(solve, trace):
    got = solve(0, [1, 1], trace=trace)
    _cmp(got, _o_solve(0, [1, 1], trace=trace))
    return got


def _line(*parts):
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------- checks --
def check_example(solve):
    """题面示例：3 个数字、先绕一步再命中，评级按 3/48 计。"""
    r = _same(solve, 6, [1, 2, 3], "sum", 6)
    assert r["trace"] == (
        "Current State: 6:[1, 2, 3], Operations: []\n"
        "Exploring Operation: 2*3=6, Resulting Numbers: [1, 6]\n"
        "Generated Node #0,0: 6:[1, 6] Operation: 2*3=6\n"
        "Moving to Node #0,0\n"
        "Current State: 6:[1, 6], Operations: ['2*3=6']\n"
        "Exploring Operation: 1*6=6, Resulting Numbers: [6]\n"
        "6,6 equal: Goal Reached\n"
    )
    assert r["encoded"] is True
    assert r["found"] is True
    assert r["states"] == 2
    assert r["moves"] == 1
    assert r["explored"] == 2
    assert r["node_ids"] == ["0,0"]
    assert r["ops"] == ["2*3=6", "1*6=6"]
    assert r["final"] == {"value": 6, "target": 6, "equal": True}
    assert r["verdict"] == "Valid path."
    assert r["score"] == 3
    assert r["max_nodes"] == 48
    assert r["rating"] == 1.0 - 3.0 / 48


def check_op_order(solve):
    """一次合并的固定四序：加、乘、大减小、大除小（只在整除时）。

    排序会把不同启发式的操作打散，所以这里挑**同分**的实例：同分保持生成序，
    加在乘前、减在除前，减法是「大减小」、除法是「大除小」。
    """
    # 7:[2,4]：2+4 与 2*4 都离 7 一步，4-2 与 4/2 也同分。
    r = _same(solve, 7, [2, 4])
    assert r["ops"] == ["2+4=6", "2*4=8", "4-2=2", "4/2=2"]
    # 5:[3,4]：不整除，没有除法；减法只能是 4-3。
    r = _same(solve, 5, [3, 4])
    assert sorted(r["ops"]) == sorted(["3+4=7", "3*4=12", "4-3=1"])
    assert "3-4=-1" not in r["trace"]
    assert "3/4" not in r["trace"]
    # 9:[0,2]：a 为 0 时不算除法；加与减同值同分。
    r = _same(solve, 9, [0, 2])
    assert r["ops"] == ["0+2=2", "2-0=2", "0*2=0"]
    # 5:[2,2]：四个运算全齐，且同分的加在乘之前。
    r = _same(solve, 5, [2, 2])
    assert r["ops"] == ["2+2=4", "2*2=4", "2/2=1", "2-2=0"]


def check_heuristic_order(solve):
    """子节点按启发式升序：第一个被探索的就是最"接近"目标的那个。"""
    # sum 启发式下 [1,2,3] 对 6：2*3=6 得 [1,6]（h=2.5）最优，先探索。
    r = _same(solve, 6, [1, 2, 3], "sum", 6)
    assert r["ops"][0] == "2*3=6"
    # mult 启发式把结果往 target 的因子上拉：[4,9] 对 36，4*9=36 直接命中。
    r = _same(solve, 36, [4, 9], "mult")
    assert r["ops"] == ["4*9=36"]
    assert r["score"] == 2
    # 同一组数字换启发式，探索次序就换。
    a = _same(solve, 10, [1, 2, 3, 4], "sum", 6)
    b = _same(solve, 10, [1, 2, 3, 4], "mult", 6)
    assert a["ops"][:3] != b["ops"][:3]


def check_tie_keeps_generation_order(solve):
    """同分不打乱：2+2=4 与 2*2=4 都得到 [5, 4]，编号连着且按生成序。"""
    r = _same(solve, 14, [2, 2, 5])
    trace = r["trace"]
    assert "Generated Node #0,2: 14:[5, 4] Operation: 2+2=4" in trace
    assert "Generated Node #0,3: 14:[5, 4] Operation: 2*2=4" in trace
    assert trace.index("#0,2") < trace.index("#0,3")


def check_threshold_before_numbering(solve):
    """threshold 过滤发生在排序与编号之前：被剪掉的分支不占编号。"""
    full = _same(solve, 24, [4, 9, 3], "mult")
    cut = _same(solve, 24, [4, 9, 3], "mult", 10)
    assert full["found"] is True
    # 剪枝后编号从 0 连续重排，不会留空洞。
    ids = cut["node_ids"]
    assert ids, "expected at least one surviving branch"
    for node_id in ids:
        depth, pos = node_id.split(",")
        assert depth == "0"
        assert int(pos) < 12
    assert cut["explored"] <= full["explored"]
    # threshold 卡在「sum/len」与「sum/(len+1)」之间：分母写错就会多留分支。
    tight = _same(solve, 6, [1, 2, 3], "sum", 2)
    assert tight["found"] is False
    assert tight["explored"] == 0
    assert tight["trace"] == "Current State: 6:[1, 2, 3], Operations: []\n"


def check_goal_stops_everything(solve):
    """命中即整棵停下：轨迹最后一行就是 equal，后面不再有任何标记。"""
    r = _same(solve, 6, [1, 2, 3], "sum", 6)
    assert r["trace"].endswith("equal: Goal Reached\n")
    assert r["trace"].count("Goal Reached") == 1
    # 命中后兄弟分支一个字都不写。
    assert "unequal" not in r["trace"]


def check_no_solution_tail(solve):
    """整棵搜完没命中：结尾停在某个 unequal 行，没有收尾回溯标记。"""
    r = _same(solve, 10, [1, 2, 3], "sum")
    assert r["found"] is False
    assert r["trace"].endswith("unequal: No Solution\n")
    assert r["final"] == {"value": r["final"]["value"], "target": 10,
                          "equal": False}
    assert r["verdict"] == "No goal reached statement found."
    assert r["rating"] == 0.0
    # 官方 dfs 末尾那段回溯提示是死代码：搜完不会补一行 Moving / Current State。
    assert r["moves"] > 0
    assert not r["trace"].rstrip().split("\n")[-1].startswith("Moving to Node")


def check_leaf_not_descended(solve):
    """只剩一个数但不等于 target：写 unequal 后不下探，也不再写 Generated。"""
    r = _same(solve, 5, [1, 2, 3], "sum")
    lines = r["trace"].split("\n")
    for pos, line in enumerate(lines):
        if line.endswith("unequal: No Solution"):
            nxt = lines[pos + 1] if pos + 1 < len(lines) else ""
            assert not nxt.startswith("Generated Node"), \
                "a one-number leaf must not be descended into"
    # 反过来说，每个 Generated 的状态里都还剩至少两个数字。
    for line in lines:
        if not line.startswith("Generated Node #"):
            continue
        state = line.split(" Operation: ")[0].split(":", 1)[1]
        assert ", " in state


def check_sibling_restores_parent(solve):
    """兄弟分支之间的衔接写的是**父节点**的状态，不是兄弟自己的。"""
    r = _same(solve, 5, [1, 2, 3], "sum")
    lines = r["trace"].split("\n")
    picks = [pos for pos, line in enumerate(lines)
             if line.startswith("Moving to Node #0\n") or line == "Moving to Node #0"]
    assert picks, "expected a sibling hand-off"
    for pos in picks:
        assert lines[pos + 1] == "Current State: 5:[1, 2, 3], Operations: []"


def check_two_number_parse_bug(solve):
    """2 个数字一步命中：轨迹完全合法，但官方解析器读父状态的操作列表为空。"""
    r = _same(solve, 24, [3, 8], "sum")
    assert r["found"] is True
    assert r["trace"].count("Goal Reached") == 1
    assert r["verdict"] == "Could not split operation into lhs, rhs"
    assert r["score"] == 2
    assert r["max_nodes"] == 4
    # 明明命中却拿 0 分，这是官方口径。
    assert r["rating"] == 0.0
    r = _same(solve, 9, [3, 3], "sum")
    assert r["verdict"] == "Could not split operation into lhs, rhs"


def check_pool_never_shrinks(solve):
    """可用数字池只增不减：同一个数字可以被反复用而不报错。"""
    r = _same(solve, 14, [2, 2, 5], "sum")
    assert r["ops"][-2:] == ["2+5=7", "2*7=14"]
    assert r["verdict"] == "Valid path."
    assert r["rating"] == 1.0 - 21.0 / 48
    # 换成"用完就删一个"的实现会判这条轨迹不合法。
    assert "number 2 not available" not in r["verdict"]


def check_rating_table_and_clamp(solve):
    """满树节点数只按初始数字个数查表，算出来为负一律夹到 0。"""
    for nums, expect in (([3, 8], 4), ([1, 2, 3], 48), ([1, 2, 3, 4], 1152),
                         ([1, 2, 3, 4, 5], 46080)):
        r = _same(solve, 7, nums, "sum", 7)
        assert r["max_nodes"] == expect
    # 搜完整棵 3 数树：分数 52 > 48，评级被夹到 0。
    r = _same(solve, 10, [1, 2, 3], "sum")
    assert r["score"] == 52
    assert r["rating"] == 0.0
    # 命中且合法时按 1 - 分数/满树 计。
    r = _same(solve, 47, [42, 61, 66], "sum")
    assert r["score"] == 7
    assert r["rating"] == 1.0 - 7.0 / 48
    # 轨迹合法、但分数已经超过满树节点数：1 - 51/48 是负数，要夹到 0。
    r = _same(solve, 18, [2, 4, 4], "mult")
    assert r["verdict"] == "Valid path."
    assert r["score"] == 51
    assert 1.0 - r["score"] / r["max_nodes"] < 0
    assert r["rating"] == 0.0


def check_bad_arguments(solve):
    """建图期的硬约束一律 ValueError；给了 trace 就不看其余入参。"""
    bad = [dict(nums=[1]), dict(nums=[1, 2, 3, 4, 5, 6]),
           dict(nums=[1, -2]), dict(nums=[1, 2], heuristic="max"),
           dict(nums=[1, 2], heuristic="mult", target=1)]
    for case in bad:
        target = case.pop("target", 4)
        try:
            solve(target, **case)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % (case,))
    try:
        solve(-3, [1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative target")
    # 只解码时 nums/target 完全不参与。
    r = solve(-1, [], trace=_line("Current State: 6:[1, 2, 3], Operations: []"))
    assert r["encoded"] is False
    assert r["max_nodes"] == 48


def check_decode_bad_head(solve):
    """首行不合规：直接判无效，且不评级。"""
    bad = [
        "",
        "no state here\n",
        "Current State: 6:1, 2, 3], Operations: []\n",
        "Current state: 6:[1, 2, 3], Operations: []\n",
        "Current State: -6:[1, 2, 3], Operations: []\n",
        "Current State: 6:[a, b], Operations: []\n",
    ]
    for trace in bad:
        r = _decoded(solve, trace)
        assert r["verdict"] == _BAD_HEAD, repr(trace)
        assert r["rating"] == 0.0
        assert r["max_nodes"] is None
    # 首行前面多出别的字符：`re.match` 锚在开头，不能从中间捞。
    r = _decoded(solve, "log: Current State: 6:[1, 2, 3], Operations: []\n")
    assert r["verdict"] == _BAD_HEAD
    # 首行前后空白会被 strip 掉，所以缩进不影响解析。
    r = _decoded(solve, "  Current State: 6:[1, 2, 3], Operations: []\n")
    assert r["verdict"] == "No goal reached statement found."
    assert r["max_nodes"] == 48


def check_decode_no_goal(solve):
    """有状态、有操作，但没有 equal 行。"""
    trace = _line("Current State: 6:[1, 2, 3], Operations: []",
                  "Exploring Operation: 1+2=3, Resulting Numbers: [3, 3]")
    r = _decoded(solve, trace)
    assert r["verdict"] == "No goal reached statement found."
    assert r["found"] is False
    assert r["score"] == 2


def check_decode_no_operations(solve):
    """有 equal 行但前面一个操作都没有。"""
    trace = _line("Current State: 6:[6], Operations: []",
                  "6,6 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == "No operations found leading to the goal."


def check_decode_final_mismatch(solve):
    """最后一步的结果不等于首行 target。"""
    trace = _line("Current State: 6:[1, 2, 3], Operations: []",
                  "Exploring Operation: 1*2=2, Resulting Numbers: [2, 3]",
                  "Generated Node #0,0: 6:[2, 3] Operation: 1*2=2",
                  "Moving to Node #0,0",
                  "Current State: 6:[2, 3], Operations: ['1*2=2']",
                  "Exploring Operation: 2*3=9, Resulting Numbers: [9]",
                  "6,6 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == "Invalid path: Final operation does not result in target."
    assert r["rating"] == 0.0


def check_decode_arithmetic(solve):
    """等号左边算不对就判非法，判词带原操作。"""
    trace = _line("Current State: 8:[2, 5, 7], Operations: []",
                  "Exploring Operation: 2+5=7, Resulting Numbers: [7, 7]",
                  "Generated Node #0,0: 8:[7, 7] Operation: 2+5=7",
                  "Moving to Node #0,0",
                  "Current State: 8:[7, 7], Operations: ['2+5=7']",
                  "Exploring Operation: 7+7=8, Resulting Numbers: [8]",
                  "8,8 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == "Invalid operation: 7+7=8"
    # 达成步算术没问题、但前面某一步是错的：同样判非法。
    trace = _line("Current State: 8:[2, 5, 7], Operations: []",
                  "Exploring Operation: 2+5=9, Resulting Numbers: [7, 9]",
                  "Generated Node #0,0: 8:[7, 9] Operation: 2+5=9",
                  "Moving to Node #0,0",
                  "Current State: 8:[7, 9], Operations: ['2+5=9']",
                  "Exploring Operation: 9-7=8, Resulting Numbers: [2]",
                  "8,8 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == "Invalid operation: 2+5=9"


def check_decode_number_not_available(solve):
    """用了初始状态里没有、也没产出过的数字。"""
    trace = _line("Current State: 10:[1, 2, 8], Operations: []",
                  "Exploring Operation: 1+2=3, Resulting Numbers: [8, 3]",
                  "Generated Node #0,0: 10:[8, 3] Operation: 1+2=3",
                  "Moving to Node #0,0",
                  "Current State: 10:[8, 3], Operations: ['1+2=3']",
                  "Exploring Operation: 5+5=10, Resulting Numbers: [10]",
                  "10,10 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == (
        "Invalid operation: 5+5=10, number 5 not available in [1, 2, 8, 3]")
    # 池子里只多不少：同样的 5 先被用过一次，第二次照样能过。
    trace = _line("Current State: 10:[5, 1, 2], Operations: []",
                  "Exploring Operation: 5+1=6, Resulting Numbers: [2, 6]",
                  "Generated Node #0,0: 10:[2, 6] Operation: 5+1=6",
                  "Moving to Node #0,0",
                  "Current State: 10:[2, 6], Operations: ['5+1=6']",
                  "Exploring Operation: 5+5=10, Resulting Numbers: [10]",
                  "10,10 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == "Valid path."


def check_decode_split_failure(solve):
    """操作序列里混进空串（空 Operations 列表）就拆不开等号。"""
    trace = _line("Current State: 24:[3, 8], Operations: []",
                  "Exploring Operation: 3*8=24, Resulting Numbers: [24]",
                  "24,24 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == "Could not split operation into lhs, rhs"

    trace = _line("Current State: 24:[3, 8], Operations: []",
                  "Exploring Operation: 3*8=24, Resulting Numbers: [24]",
                  "Exploding Operation: 3*8=24, Resulting Numbers: [24]",
                  "24,24 equal: Goal Reached")
    r = _decoded(solve, trace)
    # 只认 Exploring Operation，别的行当不存在。
    assert r["verdict"] == "Could not split operation into lhs, rhs"


def check_decode_eval_error(solve):
    """左边算不出来（除零）落到 evaluating 分支。"""
    trace = _line("Current State: 3:[3, 0, 6], Operations: []",
                  "Exploring Operation: 3+0=3, Resulting Numbers: [6, 3]",
                  "Generated Node #0,0: 3:[6, 3] Operation: 3+0=3",
                  "Moving to Node #0,0",
                  "Current State: 3:[6, 3], Operations: ['3+0=3']",
                  "Exploring Operation: 3/0=3, Resulting Numbers: [3]",
                  "3,3 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"].startswith("Error in evaluating operation 3/0=3: ")
    assert "division by zero" in r["verdict"]


def check_decode_valid_replay(solve):
    """手工轨迹：父状态的操作列表 + 达成步拼起来逐步重放。"""
    trace = _line("Current State: 47:[42, 61, 66], Operations: []",
                  "Exploring Operation: 61-42=19, Resulting Numbers: [66, 19]",
                  "Generated Node #0,0: 47:[66, 19] Operation: 61-42=19",
                  "Moving to Node #0,0",
                  "Current State: 47:[66, 19], Operations: ['61-42=19']",
                  "Exploring Operation: 66-19=47, Resulting Numbers: [47]",
                  "47,47 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == "Valid path."
    assert r["encoded"] is False
    assert r["found"] is True
    assert r["states"] == 2
    assert r["moves"] == 1
    assert r["explored"] == 2
    assert r["node_ids"] == ["0,0"]
    assert r["ops"] == ["61-42=19", "66-19=47"]
    assert r["final"] == {"value": 47, "target": 47, "equal": True}
    assert r["score"] == 3
    assert r["max_nodes"] == 48
    assert r["rating"] == 1.0 - 3.0 / 48

    # 达成步之前还有更多状态时，取的是**最后一条**（即达成步的父节点）。
    trace = _line("Current State: 47:[42, 61, 66], Operations: []",
                  "Exploring Operation: 66-42=24, Resulting Numbers: [61, 24]",
                  "Generated Node #0,0: 47:[61, 24] Operation: 66-42=24",
                  "Moving to Node #0,0",
                  "Current State: 47:[61, 24], Operations: ['66-42=24']",
                  "Exploring Operation: 61-24=37, Resulting Numbers: [37]",
                  "37,47 unequal: No Solution",
                  "Moving to Node #0",
                  "Current State: 47:[42, 61, 66], Operations: []",
                  "Exploring Operation: 61-42=19, Resulting Numbers: [66, 19]",
                  "Generated Node #0,1: 47:[66, 19] Operation: 61-42=19",
                  "Moving to Node #0,1",
                  "Current State: 47:[66, 19], Operations: ['61-42=19']",
                  "Exploring Operation: 66-19=47, Resulting Numbers: [47]",
                  "47,47 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == "Valid path."
    assert r["node_ids"] == ["0,0", "0,1"]
    assert r["final"] == {"value": 47, "target": 47, "equal": True}


def check_decode_two_goals(solve):
    """轨迹里出现两条 equal 行：只认**第一条**，后面的当没看见。"""
    trace = _line("Current State: 6:[1, 2, 3], Operations: []",
                  "Exploring Operation: 2*3=6, Resulting Numbers: [1, 6]",
                  "Generated Node #0,0: 6:[1, 6] Operation: 2*3=6",
                  "Moving to Node #0,0",
                  "Current State: 6:[1, 6], Operations: ['2*3=6']",
                  "Exploring Operation: 1*6=6, Resulting Numbers: [6]",
                  "6,6 equal: Goal Reached",
                  "Moving to Node #0",
                  "Current State: 6:[1, 2, 3], Operations: []",
                  "Exploring Operation: 1+2=3, Resulting Numbers: [3, 3]",
                  "3,3 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == "Valid path."
    assert r["ops"] == ["2*3=6", "1*6=6", "1+2=3"]
    # 找的是第一条：取它前面的最后一个操作当达成步，而不是最后那条。
    assert r["final"] == {"value": 3, "target": 3, "equal": True}


def check_decode_inexact_division(solve):
    """除法按真除算：3/2=1 这种取整写法的轨迹判非法。"""
    trace = _line("Current State: 1:[3, 2, 9], Operations: []",
                  "Exploring Operation: 9-3=6, Resulting Numbers: [2, 6]",
                  "Generated Node #0,0: 1:[2, 6] Operation: 9-3=6",
                  "Moving to Node #0,0",
                  "Current State: 1:[2, 6], Operations: ['9-3=6']",
                  "Exploring Operation: 3/2=1, Resulting Numbers: [1]",
                  "1,1 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == "Invalid operation: 3/2=1"
    # 等号左边不是 a<op>b 两数形态（这里带空格）：一律落到求值兜底分支。
    trace = _line("Current State: 7:[3, 4, 9], Operations: []",
                  "Exploring Operation: 9-4=5, Resulting Numbers: [3, 5]",
                  "Generated Node #0,0: 7:[3, 5] Operation: 9-4=5",
                  "Moving to Node #0,0",
                  "Current State: 7:[3, 5], Operations: ['9-4=5']",
                  "Exploring Operation: 3 + 4=7, Resulting Numbers: [7]",
                  "7,7 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"].startswith("Error in evaluating operation 3 + 4=7: ")
    # 整除时才两边一致。
    trace = _line("Current State: 3:[6, 2, 9], Operations: []",
                  "Exploring Operation: 9-6=3, Resulting Numbers: [2, 3]",
                  "Generated Node #0,0: 3:[2, 3] Operation: 9-6=3",
                  "Moving to Node #0,0",
                  "Current State: 3:[2, 3], Operations: ['9-6=3']",
                  "Exploring Operation: 6/2=3, Resulting Numbers: [3]",
                  "3,3 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["verdict"] == "Valid path."


def check_decode_unknown_size(solve):
    """首行数字个数不在查表范围：判词照算，但不给评级。"""
    trace = _line("Current State: 6:[6], Operations: []",
                  "Exploring Operation: 3+3=6, Resulting Numbers: [6]",
                  "6,6 equal: Goal Reached")
    r = _decoded(solve, trace)
    assert r["max_nodes"] is None
    assert r["verdict"] == "Could not split operation into lhs, rhs"
    assert r["rating"] == 0.0


def check_encoded_flag(solve):
    """两次调用互不影响：同一段文本，编码来的和喂进来的账本一致。"""
    text = _same(solve, 6, [1, 2, 3], "sum", 6)["trace"]
    a = solve(6, [1, 2, 3], heuristic="sum", threshold=6)
    b = solve(6, [1, 2, 3], heuristic="sum", threshold=6, trace=text)
    assert a["encoded"] is True
    assert b["encoded"] is False
    for key in ("trace", "found", "states", "moves", "explored",
                "node_ids", "ops", "final", "verdict", "score",
                "max_nodes", "rating"):
        assert a[key] == b[key], key


check_example.example = True
