"""Stream of Search 轨迹序列化账（arXiv 2404.03683 + kanishkg/stream-of-search main）。

Stream of Search 把「搜索过程本身」当成语言来学：在 Countdown 游戏上跑一个
符号搜索器，把每一步状态、操作、前进/回溯都**序列化成一段文本轨迹**，再用
这段轨迹训练语言模型。官方实现里，编码器是 `src/countdown_dfs.py::dfs`（配
`src/countdown_utils.py::combine_nums` 与两个启发式），解码与打分器是
`src/countdown_utils.py::parse_trajectory` / `metric_fn`（判轨迹合不合法、
用了多少节点、给出 0~1 的评级）。

本题把这条流水线的**确定性的那一半**做成一本账：给定 `(target, nums)` 与一个
纯启发式就逐行编码出轨迹（含 `Moving to Node #` 前进/回溯标记），也可以直接
把一段轨迹喂进来；两种情况下后面的解码与评级走同一条路，账本上的每个统计
字段都**从轨迹文本上数出来**。

编码侧口径逐条对齐官方：

* 一次合并 `combine_nums(a, b)` 最多产 4 个结果，**次序固定**：先 `a+b`、
  再 `a*b`、然后减法、最后除法；减法方向恒为「大减小」（结果非负），除法
  方向恒为「大除小」且**只在整除时**才出现，因此不会产生小数或负数。
* 合并后剩余数字**保持原顺序**，新结果**追加在末尾**。
* 子节点按 `(启发式, 生成次序)` 稳定升序（同分保持「先 (i,j) 后运算符」的
  生成序）；`threshold` 过滤用 `<=`，过滤发生在**排序与编号之前**，剪掉的
  分支不占编号。
* 编号是 `父编号,本层序号`，根记 `0`；序号是**过滤排序后**的位置。
* 目标检查发生在**进入子节点之前**：子节点只剩一个数且等于 `target` 就立刻
  写 `equal: Goal Reached` 并整棵树停下；只剩一个数但不等于 `target` 就写
  `unequal: No Solution`，**不再下探**（也不再多写一行 `Moving`）。
* 一个子分支失败返回后要接着试下一个兄弟，此时写两行：`Moving to Node #父`
  与父节点自己的 `Current State`（不是兄弟的状态）。
* 官方 `dfs` 末尾那段 `if open_set:` 的回溯提示在递归实现下 `open_set` 恒为
  空，是**死代码**，永远不会写出——所以轨迹可以戛然而止于某个 `unequal`
  行，没有收尾的回溯标记。

解码侧（官方 `parse_trajectory`）：

* 首行必须是 `Current State: <target>:[a, b, ...]`，`re.match` 锚在开头，
  `target` 只认**无符号数字**，数字列表按 `", "` 切开。
* 找**第一个** `X,Y equal: Goal Reached`；没有就直接判
  `No goal reached statement found.`。
* 取该行**之前**最后一个 `Exploring Operation: <op>, Resulting Numbers: [...]`
  作为「达成目标的那一步」，它的等号右边必须等于首行 `target`。
* 再从该行之前**最后一条** `Current State` 里取出 `Operations: [...]` 列表
  （这条状态是达成步的父节点），拼上达成步本身得到整条操作序列，然后逐步
  重放：等号左边要算得对、用到的每个数字必须「当下可用」。
* 重放的可用数字池只增不减：官方那句 `[n for n in available if n not in
  used_numbers]` 拿 `int` 去比 `str` 列表恒为真，**一个都不会被移除**，所以
  同一个数字可以被反复使用而检查不出来。

评级（官方 `metric_fn`）：

* 合法轨迹的分数 = 轨迹里 `Exploring Operation` 的行数 `+ 1`（那个 `+1` 是
  根），评级 = `max(1 - 分数 / 满树节点数, 0.0)`；满树节点数只按初始数字个数
  查表（2→4、3→48、4→1152、5→46080），所以 2 个数字的实例里「一步就中」
  也可能被夹到 0。
* 不合法轨迹不评级，直接 `(0.0, 判词)`。

两处显式化（note 里明列）：官方 `eval(lhs)` 换成只认 `a<op>b` 两数形态的
求值器（生成的轨迹只可能是这种形态）；官方「抽不出操作列表时 `[-1]` 抛
`IndexError`」与「`Couldnt parse last op` 分支」都够不到，前者规范成空列表、
后者删掉。
"""

import re

_MAX_NODES = {2: 4, 3: 48, 4: 1152, 5: 46080}
_OPERATORS = "+-*/"

_FIRST_LINE = re.compile(r"Current State: (\d+):\[(.*?)\]")
_GOAL_LINE = re.compile(r"\d+,\d+ equal: Goal Reached")
_OUTCOME = re.compile(r"(\d+),(\d+) (equal: Goal Reached|unequal: No Solution)")
_EXPLORE_LINE = re.compile(
    r"Exploring Operation: (.*?=\d+), Resulting Numbers: \[(.*?)\]")
_STATE_OPS = re.compile(
    r"Current State: \d+:\[.*?\], Operations: \[(.*?)\]")
_GENERATED = re.compile(r"Generated Node #([\d,]+):")
_DIGITS = re.compile(r"\d+")

_BAD_HEAD = "Invalid input: Cannot find the initial state in the first line."


def _combine(a, b):
    """官方 combine_nums：固定四序，减法大减小、除法只在整除时出现。"""
    a = int(a)
    b = int(b)
    pairs = [[a + b, "%d+%d=%d" % (a, b, a + b)],
             [a * b, "%d*%d=%d" % (a, b, a * b)]]
    if a <= b:
        pairs.append([b - a, "%d-%d=%d" % (b, a, b - a)])
        if a != 0 and b % a == 0:
            pairs.append([b // a, "%d/%d=%d" % (b, a, b // a)])
    else:
        pairs.append([a - b, "%d-%d=%d" % (a, b, a - b)])
        if b != 0 and a % b == 0:
            pairs.append([a // b, "%d/%d=%d" % (a, b, a // b)])
    return pairs


def _sum_heuristic(nums, target):
    if len(nums) == 1:
        return abs(nums[0] - target)
    return sum(abs(num - target) for num in nums) / len(nums)


def _mult_heuristic(nums, target):
    factors = [i for i in range(2, target + 1) if target % i == 0]
    return sum(min(abs(num - factor) for factor in factors) for num in nums)


_HEURISTICS = {"sum": _sum_heuristic, "mult": _mult_heuristic}


def _expand(nums, ops, target, heuristic):
    """把所有后继按「(i,j) 字典序 × 运算固定序」摊平成一串。"""
    children = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            for result, operation in _combine(nums[i], nums[j]):
                rest = [nums[k] for k in range(len(nums)) if k != i and k != j]
                new_nums = rest + [result]
                children.append((heuristic(new_nums, target), new_nums,
                                 ops + [operation], operation))
    return children


def _walk(target, node, threshold, heuristic, lines):
    """官方 dfs 的递归骨架：进入即写状态，按序试子节点，命中即整棵停下。"""
    lines.append("Current State: %d:%s, Operations: %s"
                 % (target, node["nums"], node["ops"]))
    children = _expand(node["nums"], node["ops"], target, heuristic)
    kept = [child for child in children
            if threshold is None or child[0] <= threshold]
    kept.sort(key=lambda child: child[0])

    for pos, (_, nums, ops, operation) in enumerate(kept):
        node_id = "%s,%d" % (node["id"], pos)
        lines.append("Exploring Operation: %s, Resulting Numbers: %s"
                     % (operation, nums))
        if len(nums) == 1 and nums[0] == target:
            lines.append("%d,%d equal: Goal Reached" % (nums[0], target))
            return True
        if len(nums) == 1:
            lines.append("%d,%d unequal: No Solution" % (nums[0], target))
        else:
            lines.append("Generated Node #%s: %d:%s Operation: %s"
                         % (node_id, target, nums, operation))
            lines.append("Moving to Node #%s" % (node_id,))
            if _walk(target, {"id": node_id, "nums": nums, "ops": ops},
                     threshold, heuristic, lines):
                return True
        if pos < len(kept) - 1:
            lines.append("Moving to Node #%s" % (node["id"],))
            lines.append("Current State: %d:%s, Operations: %s"
                         % (target, node["nums"], node["ops"]))
    return False


def _evaluate(expression):
    """把 `a<op>b` 算出来；不是两数形态返回 None。"""
    for pos in range(1, len(expression) - 1):
        if expression[pos] not in _OPERATORS:
            continue
        left, right = expression[:pos], expression[pos + 1:]
        if not (left.isdigit() and right.isdigit()):
            return None
        a, b = int(left), int(right)
        if expression[pos] == "+":
            return a + b
        if expression[pos] == "-":
            return a - b
        if expression[pos] == "*":
            return a * b
        if b == 0:
            raise ZeroDivisionError("division by zero")
        return a / b
    return None


def _decode(text):
    """官方 parse_trajectory：把轨迹解回一条操作序列并逐步重放。"""
    info = {"verdict": "", "target": None, "nums": None}
    first_line = text.strip().split("\n")[0]
    match = _FIRST_LINE.match(first_line)
    if not match:
        info["verdict"] = _BAD_HEAD
        return info
    try:
        nums = [int(num) for num in match.group(2).split(", ")]
    except ValueError:
        info["verdict"] = _BAD_HEAD
        return info
    target = int(match.group(1))
    info["target"] = target
    info["nums"] = nums

    cleaned = text.replace("<|endoftext|>", "")
    goals = list(_GOAL_LINE.finditer(cleaned))
    if not goals:
        info["verdict"] = "No goal reached statement found."
        return info
    goal = goals[0]

    operations = _EXPLORE_LINE.findall(cleaned[:goal.start()])
    if not operations:
        info["verdict"] = "No operations found leading to the goal."
        return info

    final_operation = operations[-1][0]
    predicted = int(final_operation.split("=")[1])
    if predicted != target:
        info["verdict"] = "Invalid path: Final operation does not result in target."
        return info

    states = _STATE_OPS.findall(cleaned[:goal.start()])
    operation_list = states[-1].split(", ") if states else []
    operation_list = [operation.replace("'", "") for operation in operation_list]
    operation_list.append(final_operation)

    available = nums
    for operation in operation_list:
        try:
            left, right = operation.split("=")
        except Exception:
            info["verdict"] = "Could not split operation into lhs, rhs"
            return info
        try:
            value = _evaluate(left)
            if value is None:
                raise ValueError("malformed expression %r" % (left,))
            if value != int(right):
                info["verdict"] = "Invalid operation: %s" % (operation,)
                return info
        except Exception as exc:
            info["verdict"] = ("Error in evaluating operation %s: %s"
                               % (operation, exc))
            return info
        used_numbers = _DIGITS.findall(left)
        for number in used_numbers:
            if int(number) not in available:
                info["verdict"] = (
                    "Invalid operation: %s, number %s not available in %s"
                    % (operation, number, available))
                return info
        # 官方这里拿 int 去比 str 列表，等于什么都不删——照抄。
        available = [item for item in available if item not in used_numbers]
        available.append(int(right))

    info["verdict"] = "Valid path."
    return info


def solve(target, nums, heuristic="sum", threshold=None, trace=None):
    """编码（或直接吃）一段 Countdown 搜索轨迹，再解码评级。"""
    if trace is None:
        if heuristic not in _HEURISTICS:
            raise ValueError("unknown heuristic: %r" % (heuristic,))
        values = [int(num) for num in nums]
        if not 2 <= len(values) <= 5:
            raise ValueError("nums must hold between 2 and 5 numbers")
        if any(num < 0 for num in values) or target < 0:
            raise ValueError("countdown numbers and target must be non-negative")
        if heuristic == "mult" and target < 2:
            raise ValueError("mult heuristic needs target >= 2")
        lines = []
        _walk(target, {"id": "0", "nums": values, "ops": []},
              threshold, _HEURISTICS[heuristic], lines)
        text = "".join(line + "\n" for line in lines)
    else:
        text = trace

    info = _decode(text)
    outcomes = _OUTCOME.findall(text)
    if outcomes:
        value, goal, kind = outcomes[-1]
        final = {"value": int(value), "target": int(goal),
                 "equal": kind == "equal: Goal Reached"}
    else:
        final = None

    explored = text.count("Exploring Operation")
    score = explored + 1
    size = len(info["nums"]) if info["nums"] is not None else None
    max_nodes = _MAX_NODES.get(size)
    valid = info["verdict"] == "Valid path."
    rating = 0.0
    if valid and max_nodes is not None:
        rating = max(1.0 - score / max_nodes, 0.0)

    return {
        "trace": text,
        "encoded": trace is None,
        "found": "Goal Reached" in text,
        "states": text.count("Current State"),
        "moves": text.count("Moving to Node"),
        "explored": explored,
        "node_ids": _GENERATED.findall(text),
        "ops": [op for op, _ in _EXPLORE_LINE.findall(text)],
        "final": final,
        "verdict": info["verdict"],
        "score": score,
        "max_nodes": max_nodes,
        "rating": rating,
    }
