# notes（rs-sos-trace-ledger）

## 回源

- 论文：Stream of Search（arXiv 2404.03683）。主张是「把搜索过程当语言」：在
  Countdown 上跑符号搜索器，把状态/操作/前进回溯序列化成文本轨迹，再拿轨迹训练
  LM。论文给的是这套思路与曲线，**轨迹的逐行格式、编号规则、解析口径**都没写。
- 代码：`kanishkg/stream-of-search` main（论文官方实现）。
  - `src/countdown_dfs.py::dfs`（6-65）：19-26 枚举后继、28-34 `threshold` 过滤、
    35 排序、39 写编号、41-45 目标判定、50-52 递归、54-57 兄弟衔接、60-63 回溯提示。
  - `src/countdown_utils.py::combine_nums`（7-20）、`CountdownNode`（22-31）、
    `sum_heuristic`（35-38）、`mult_heuristic`（40-45）、`simple_rating`（57-60）、
    `parse_trajectory`（77-140）、`metric_fn`（143-165）。
  - `src/countdown_bfs.py` 是另一条（beam + `random.shuffle`）路线，不可判，本题不用。
- 编码侧本题逐字复刻 `dfs` 的输出：拿官方模块直接跑 400 组随机实例，轨迹**逐字节
  相同**（0 差异）；判词与评级也逐条对齐 `parse_trajectory` / `metric_fn`。

## 官方实现里的坑

1. **`dfs` 的 `while open_set` 每层只转一圈**：根调用靠空列表补一个初始节点，递归
   调用传的是单元素 `new_set`，pop 完就空。所以它实质是「进入即展开」的递归 DFS，
   `while` 只是壳。
2. **`open_set.sort(key=lambda x: -x[0])` + `pop()`** = 取启发式**最小**的那个；
   同分时 `CountdownNode.__lt__` 只比 `heuristic`，`(h, node)` 元组的第二元素比较恒
   `False`，于是被当作相等、`sort` 稳定 → **同分保持生成序**。
3. **`node_index` 与 `enumerate` 的 `g` 恒等**：它只在循环体末尾 `+1`，而命中时
   直接 `return` 不加。所以编号里的"本层序号"就是排序后位置。
4. **`threshold` 过滤在排序与编号之前**：被剪掉的分支不占编号，剪枝后编号会重排。
5. **目标判定在进入子节点之前**：只剩一个数且等于 `target` 就写 `equal` 并整棵停下；
   只剩一个数但不等于就写 `unequal` 且**不下探**（也就不写 `Generated`）。
6. **兄弟衔接写的是父节点状态**：失败返回后写 `Moving to Node #父` 与
   `Current State: <父的数字>`（代码里是 `new_node.parent.*`），不是兄弟的状态。
7. **`if open_set:` 的回溯块是死代码**：本地 `open_set` 在 pop 后恒空，那段
   `Moving to Node #next` 永远不会写。所以轨迹可以戛然停在一个 `unequal` 行上。
8. **`combine_nums` 的 `round(b // a, 0)` 是多余的**：`//` 已是 int，`round` 不改变
   任何东西；本题直接写 `b // a`（等价变异可证）。
9. **减法/除法方向恒「大减小 / 大除小」**：结果不可能为负、也不可能出现小数，
   所以轨迹里永远看不到 `3-5=-2` 或 `3/2=1.5`。
10. **`a+b` 与 `a*b` 无条件出现**，即使结果与别的运算重复（`2+2=4` 与 `2*2=4` 都留），
    除法只在整除时补第 4 项。
11. **`parse_trajectory` 三处「取哪个」**：goal 取**第一个**匹配、达成步取它之前
    **最后一个** `Exploring Operation`、父状态取它之前**最后一条** `Current State`。
12. **空 `Operations` 列表的坑**：`Operations: []` 被 `\[(.*?)\]` 抓成空串，
    `"".split(", ")` 是 `[""]` 而不是 `[]`，于是重放第一步就拆不开等号 →
    `Could not split operation into lhs, rhs`。**2 个数字的实例一步命中反而拿 0 分**，
    这是本题最反直觉的一条。
13. **可用数字池只增不减**：`[n for n in available if n not in used_numbers]` 拿 `int`
    去比 `str` 列表，恒为真，一个都不删；池子等于「首行数字 + 此前各步结果」。
    所以同一个数字可以被反复使用而查不出来。
14. **`metric_fn` 的 `max_nodes` 只认 2~5 个数**：`len(nums)` 是 1 或 ≥6 时
    `max_nodes` 未赋值，直接 `UnboundLocalError`。
15. **`simple_rating` 是「行数 + 1」**：那个 `+1` 是根节点（根只写 `Current State`，
    不计 `Exploring Operation`）。
16. **满树节点数公式偏小**：`3c2×4 × 2c2×4 = 48` 假设每对恰好 4 个运算，实际因
    「整除才给除法」「加法与乘法结果可能重复」而偏多，所以**合法轨迹也可能算出
    负评级**，靠 `max(..., 0.0)` 夹住。
17. **`get_target_nums`（63-74）是没人调用的重复实现**（与 `parse_trajectory`
    开头几行一样），本题不复刻。

## 本题的规范化

- **`eval(lhs)` → 显式两数求值器**：官方直接 `eval`，本题只接受 `a<op>b` 两数形态
  （`+ - * /`，两侧十进制非负整数），其余落到 `Error in evaluating operation ...`。
  轨迹编码只会产出这种形态，所以这是**接口收窄**而不是语义变更；`eval` 的异常文案
  还依赖运行环境，收窄后兜底文案才可判。除法按真除算（`3/2=1` 判非法）。
- **删掉 `Couldnt parse last op` 分支**：上游正则 `(.*?=\d+)` 已保证等号右边是数字，
  `int(...)` 不可能失败，该分支不可达。
- **空 `Current State` 列表的 `[-1]` 规范成空列表**：官方在「轨迹里有 goal 但一条
  `Current State` 都没有」时 `[-1]` 抛 `IndexError`；本题退化成空操作列表（这种输入
  编码侧产不出来，只影响手喂轨迹）。
- **数字列表解析失败归到 `Invalid input: ...`**：官方 `[int(n) for n in ...]` 会抛
  `ValueError`，本题把它与「首行不合规」合并成同一个判词。
- **首行数字个数不在 2~5 时 `max_nodes = None`、评级 0**：官方此处
  `UnboundLocalError`，本题给可判的 `None`。
- **编码侧入参校验显式化**：`heuristic` 名非法、`nums` 不在 2~5 个、出现负数、
  `mult` 且 `target < 2` → `ValueError`。给了 `trace` 时一律不校验。

## 变异

59 条变异：**58 条实质变异全部 CAUGHT**，1 条判为等价。等价的那条有实证：

- `m-mult-guard-one`（`target < 2` → `target < 1`）：差别只在 `target == 1`——守卫
  放过它之后，`_mult_heuristic` 里 `factors` 为空、`min()` 抛
  `ValueError: min() arg is an empty sequence`；两条路对外都是 `ValueError`，
  契约只规定「`mult` 且 `target < 2` → `ValueError`」。42 组
  （`target ∈ {0,1}` × `nums` 组合）逐一比对异常类型，0 差异。

首轮 SURVIVED、补用例后才抓住的（都被"另一个兜底"挡住了）：

- `m-sum-div-plus1`（`/ len(nums)` → `/ (len+1)`）：所有距离被同一个因子缩放，
  **排序结果不变**，只有 `threshold` 的绝对量级能看出来。补了「`threshold` 卡在
  `sum/len` 与 `sum/(len+1)` 之间」的实例（`solve(6, [1,2,3], "sum", 2)`，正确实现
  一个子节点都不留）后才抓住。
- `m-goal-last`（`goals[0]` → `goals[-1]`）：编码侧命中就停，永远只有一条 `equal`
  行。补了手喂的「两条 `equal` 行」轨迹（第二条在另一个兄弟分支下）才抓住。
- `m-eval-floordiv`（`a / b` → `a // b`）：编码侧只在整除时写除法，两种写法同值。
  补了 `3/2=1` 这种取整写法的轨迹才抓住。
- `m-eval-accept-any`（去掉 `isdigit` 双检）：只影响非两数形态的左式。补了
  `3 + 4=7`（带空格）的轨迹——`int("3 ")` 能过，于是会误判 `Valid path.`。
- `m-head-slack`（`re.match` → `re.search`）：补了「首行前面多出 `log: `」的轨迹。
- `m-no-clamp`（去掉 `max(..., 0.0)`）：需要一条**合法且分数超过满树节点数**的
  轨迹。暴力扫 3 数实例找到 `solve(18, [2, 4, 4], "mult")`：合法、`score` 51 > 48。

写用例时踩到的坑（都已修）：

- 「运算次序」不能拿普通实例断言——排序会按启发式重排，看起来像"乘在加前面"。
  必须挑**同分**实例（`7:[2,4]` 里 `2+4=6` 与 `2*4=8` 同分）才能看见生成序。
- 手喂轨迹时，第一步之前只有根状态、其 `Operations` 是 `[]`，会先撞上第 12 条那个
  空串坑；想测后面的分支必须让达成步的父状态**带一条操作**。
- `parse_trajectory` 先查「达成步右端 == target」，再重放；想让「算术错」的分支
  生效，达成步的右端必须**等于** `target`，错误要留在前一步。

## 查重

- 与 `ag-lats-uct-ledger`（order 724）划界：那题是 UCT 的**选择/回传**数值账
  （Q/N/c 更新、PUCT 打分），本题是**轨迹文本的编解码**与评级，没有选择公式。
- 与 `ag-toollm-dfs-ledger`（order 728）划界：那题是 DFS 调度工具调用（栈式
  backtrack、去重剪枝），不产生文本轨迹，也不解析。
- 与 `rs-ttrl-pseudo-reward-ledger` / `rs-self-consistency-vote` 划界：那两题是采样
  聚合与投票，本题是单个确定性搜索的序列化。
- 全库无「轨迹序列化 / 解析回放 / 节点评级」轴的题目。
