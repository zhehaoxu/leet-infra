# Stream of Search 轨迹序列化账

Stream of Search（arXiv 2404.03683）把「搜索过程本身」当成语言：在 Countdown
游戏上跑一个符号搜索器，把每一步状态、操作、前进/回溯**序列化成一段文本轨迹**，
再拿这段轨迹去训练语言模型。

本题把这条流水线的确定性那一半做成一本账：一个纯启发式驱动的深度优先搜索把
`(target, nums)` 编码成轨迹，官方的解析器再把轨迹解码回一条操作序列逐步重放，
最后按「用掉多少节点 / 满树有多少节点」折成 0~1 的评级。对应官方
`kanishkg/stream-of-search` 的 `src/countdown_dfs.py::dfs` 与
`src/countdown_utils.py`（`combine_nums` / 两个启发式 / `parse_trajectory` /
`metric_fn`）。

实现 `solve(target, nums, heuristic="sum", threshold=None, trace=None)`。

## 输入

- `target`：目标数，非负整数。
- `nums`：初始数字，2~5 个、非负整数。
- `heuristic`：`"sum"` 或 `"mult"`，见下。
- `threshold`：给了就只保留启发式 `<= threshold` 的子节点。
- `trace`：**给了就直接解码这段轨迹**，`target` / `nums` / `heuristic` /
  `threshold` 都不参与；此时 `encoded` 为 `False`。

两个启发式（都只作用于「这一步之后的数字列表」）：

- `sum`：只有一个数时是 `|n - target|`；否则是各数到 `target` 距离的**均值**
  （除以数字个数）。
- `mult`：先取 `target` 在 `[2, target]` 里的全部因子，再把每个数到**最近因子**
  的距离加起来。`target < 2` 时无因子可用 → `ValueError`。

## 编码：一次合并

两个数 `a`、`b` 的合并结果最多 4 个，**次序固定**：

1. `a+b`
2. `a*b`
3. 大减小：`a<=b` 时是 `b-a`，否则 `a-b`（结果恒非负）
4. 大除小：只在**整除**时出现——`a<=b` 且 `a != 0` 且 `b % a == 0` 时给
   `b/a`，`a>b` 且 `b != 0` 且 `a % b == 0` 时给 `a/b`

每个结果的文本是 `<左><op><右>=<结果>`，全是十进制整数。合并后的数字列表是
「**按原顺序**保留没用到的数」+「新结果**追加在末尾**」。

## 编码：子节点顺序与编号

- 一层子节点按 `(i, j)` 字典序遍历所有数对（`i < j`），对内按上面四序展开。
- 先按 `threshold` 过滤（`<=` 保留），**再**排序、**再**编号——被剪掉的分支
  不占编号。
- 排序键是子节点的启发式值，**稳定**升序：同分保持生成次序。
- 编号形如 `<父编号>,<本层序号>`，根编号是 `0`，本层序号是**过滤排序后**的
  0 起位置。

## 编码：每行写什么

进入一个节点先写它自己：

```
Current State: <target>:[<数字...>], Operations: [<操作...>]
```

数字与操作都是 Python 字面量形式（`[1, 2, 3]`、`['2*3=6']`，逗号后带空格，
操作串带单引号），根节点的操作列表是 `[]`。

然后按排序后的顺序逐个试子节点，每个子节点：

- 先写 `Exploring Operation: <op>, Resulting Numbers: [<合并后的数字...>]`。
- 子节点只剩一个数：等于 `target` 就写 `<数>,<target> equal: Goal Reached` 并
  **整棵搜索立刻停下**；不等于就写 `<数>,<target> unequal: No Solution`，
  **不再下探**。
- 否则写 `Generated Node #<编号>: <target>:[<数字...>] Operation: <op>`，接着写
  `Moving to Node #<编号>`，然后递归进去；递归报告命中就整体返回。
- 不是本层最后一个子节点时，往下试之前还要写两行：`Moving to Node #<父编号>`
  与**父节点自己**的 `Current State`（不是兄弟节点的状态）。

每一行都以换行结尾（包括最后一行）。整棵搜完仍没命中就到此为止——官方 `dfs`
末尾那段「`if open_set:` 回溯提示」在递归实现下恒为空，是**死代码**，所以轨迹
可以停在一个 `unequal` 行上，没有收尾标记。

## 解码：判词

把轨迹解回一条操作序列并逐步重放：

1. 取 `trace.strip()` 的**第一行**，必须从 `Current State: <数字>:[<数字列表>]`
   开头（锚在行首）；`target` 只认无符号数字，数字列表按 `", "` 切开。
2. 在全文里找**第一个** `<数>,<数> equal: Goal Reached`。
3. 取它**之前**最后一个 `Exploring Operation: <op>, Resulting Numbers: [...]`
   当作「达成目标的那一步」，它的等号右边必须等于首行的 `target`。
4. 取它之前**最后一条** `Current State` 的 `Operations: [...]`（那是达成步的
   父节点）拆成操作列表，拼上达成步本身，然后按顺序逐步重放：等号左边要算得
   对、用到的每个数字必须「当下可用」。
5. 可用数字池**只增不减**：一开始是首行的数字，每步把结果追加进去，但**从不
   移除**已用掉的数字——所以同一个数字可以被反复使用而查不出来。

判词就是下面这些字符串之一：

```
Invalid input: Cannot find the initial state in the first line.
No goal reached statement found.
No operations found leading to the goal.
Invalid path: Final operation does not result in target.
Could not split operation into lhs, rhs
Invalid operation: <op>
Invalid operation: <op>, number <n> not available in [<池子...>]
Error in evaluating operation <op>: <原因>
Valid path.
```

- 只有 `Valid path.` 是合法。
- 等号左边只接受 `a<op>b` 两数形态（`+ - * /`，两边都是十进制非负整数）；
  其余形态一律落到 `Error in evaluating operation ...`。
- 除法按**真除**算，所以 `3/2=1` 这种取整写法判 `Invalid operation`。
- 池子打印的就是「首行数字 + 此前各步结果」，按顺序、保留重复。

## 评级

- `explored` = 全文里 `Exploring Operation` 出现次数；
  `score` = `explored + 1`（那个 `+1` 是根）。
- `max_nodes` 只按**首行解析出的数字个数**查表：2→4、3→48、4→1152、5→46080；
  其它个数是 `None`。
- 合法且有表项：`rating = max(1 - score / max_nodes, 0.0)`；否则 `rating = 0.0`。
  所以搜得比满树还多时会被夹到 0，2 个数字的实例「一步就中」也可能被夹到 0。

## 输出

```
{
  "trace": "<轨迹全文>",
  "encoded": true,
  "found": true,
  "states": 2,
  "moves": 1,
  "explored": 2,
  "node_ids": ["0,0"],
  "ops": ["2*3=6", "1*6=6"],
  "final": {"value": 6, "target": 6, "equal": true},
  "verdict": "Valid path.",
  "score": 3,
  "max_nodes": 48,
  "rating": 0.9375
}
```

- `states` / `moves` / `explored` 分别是 `Current State` / `Moving to Node` /
  `Exploring Operation` 的出现次数，全部从轨迹文本上数。
- `node_ids` 是每个 `Generated Node #<编号>` 的编号，按写出序。
- `ops` 是每条 `Exploring Operation` 的操作，按写出序。
- `final` 是**最后一条** `equal: Goal Reached` / `unequal: No Solution` 行解析
  出来的 `{值, 目标, 是否相等}`；一条都没有就是 `None`。
- `found` 只看文本里有没有 `Goal Reached`。

## 样例

```
led = solve(6, [1, 2, 3], heuristic="sum", threshold=6)
led["trace"]
# Current State: 6:[1, 2, 3], Operations: []
# Exploring Operation: 2*3=6, Resulting Numbers: [1, 6]
# Generated Node #0,0: 6:[1, 6] Operation: 2*3=6
# Moving to Node #0,0
# Current State: 6:[1, 6], Operations: ['2*3=6']
# Exploring Operation: 1*6=6, Resulting Numbers: [6]
# 6,6 equal: Goal Reached
led["ops"]        # ["2*3=6", "1*6=6"]
led["node_ids"]   # ["0,0"]
led["score"]      # 3
led["rating"]     # 1 - 3/48
```

反例：2 个数字一步命中时轨迹完全合法，但达成步的父节点（根）操作列表是空的，
`"".split(", ")` 会切出一个空串，于是判词是
`Could not split operation into lhs, rhs`、评级 0——这是官方口径，不是笔误。

## 约束

- 纯函数：不联网、不读写文件、不修改入参；同样的输入必须给同样的账本。
- `rating` 按普通除法算，比较时按 `1e-12` 容差。
- `heuristic` 不是 `"sum"` / `"mult"`、`nums` 个数不在 2~5、`nums` 或 `target`
  有负数、`mult` 且 `target < 2` → `ValueError`。这些检查只在**编码**时做。
- 给了 `trace` 时不做上面任何入参校验，也不会去改它。
