# ADaPT 按需递归分解账

ADaPT（As-Needed Decomposition and Planning，arXiv 2311.05772）的核心主张是「**先直接试，
试不动才分解**」：执行器先自己对整条任务跑一轮，只有在失败、且还没到深度上限时，才让规划器
把它拆成带 `AND` / `OR` 的子任务，再逐个递归下去。

本题目把官方 `archiki/ADaPT` 的 `run_alfworld.py` 剥成纯函数：LLM 调用全部变成查表，
环境步进变成脚本，只保留**控制流与账本**——谁在什么深度先试了一次、失败了怎么拆、`AND`/`OR`
怎么短路、动作与信息如何在节点之间传递。

实现 `solve(task, exec_table, plan_table, max_depth=3, max_runs=20, max_patience=5, info_prop_mode="last-step-last-act")`。

## 输入

- `task`：根任务文本（非空字符串）。
- `exec_table`：执行器脚本，`{任务文本: {"steps": [{"act", "obs", "reward", "done"}, ...]}}`。
  `obs` 缺省为 `"Nothing happens."`，`reward` 缺省 0，`done` 缺省 `False`。
- `plan_table`：规划器输出，`{任务文本: 规划文本}`。
- `max_depth`：深度上限，根节点深度为 1。
- `max_runs`：单次执行的动作上限。
- `max_patience`：连续「没进展」观测的中断阈值。
- `info_prop_mode`：`"last-step-last-act"` 或 `"all-step-last-act"`。

## 输出

```
{
  "nodes": [ 每个被访问节点一条，先序编号 ],
  "runs": 动作步数 + 规划调用次数,
  "checkpoint": 成功的真实动作序列,
  "completion": 根节点累计的执行文本,
  "info_prop": 根节点算出的信息串,
  "success": bool, "reward": int, "terminate": bool,
  "depth": 根节点返回时的深度,
}
```

节点字段：`id`（先序编号）、`task`（字符串任务，或括号分组的**已解析子计划**字典）、
`depth`、`replayed`（进入时复放的 checkpoint 长度）、`direct`（直接执行结果：
`success` / `reward` / `terminate` / `turns` / `history`；分组节点为 `None`）、
`plan`（解析后的 `{"steps": [...], "logic": ...}`，叶子为 `None`）、`children`（子节点 id）、
`stop`（`"success"` / `"depth-cap"` / `"and-fail"` / `"or-success"` / `"children-done"`）、
`success`、`reward`。

## 执行器语义

单个任务的一次尝试（`direct`）：

- 从脚本第 1 步开始逐条执行，**最多 `max_runs - 1` 步**（官方写的是 `range(1, max_runs)`）。
- 动作以 `think:` 开头时，观测强制记为 `"OK."`；正文小写包含 `task completed!` 记成功、
  包含 `task failed!` 记失败；think 动作**不进入 `history`**，但照常计入 `turns` 与
  `completion`。
- 非 think 动作进入 `history`。
- 观测属于 `{"Nothing happens.", "OK."}` 时耐心计数 +1，否则清零；计数等于 `max_patience`
  时立刻中断（`terminate` 置 `True`），且**触发中断的这一条不写进 `completion`**。
- 该步 `reward` 非 0 时把成功置 `True`、把 `terminate` 清回 `False`。
- 该步 `done` 为真时立即结束。
- 循环自然结束（没走到 `done`）时：成功压回 `False`、`terminate` 置 `True`、**`reward` 归 0**。
- `completion` 是每步 `" {act}\n{obs}\n>"` 的拼接。

## 递归语义

对字符串任务：

1. 先跑一次执行器（计入 `turns`）。
2. 若成功、或 `depth >= max_depth`，直接返回。返回时把自己的 `completion + "\n"` 拼进
   返回值——注意**失败也拼**（到上限而失败时，那次尝试的文本仍会带回去）。
3. 否则查 `plan_table`（规划调用使 `runs` +1），解析计划；若顶层只有 1 个子任务则解包
   （子任务是字符串时补成 `{"steps": [x], "logic": "AND"}`），然后 `depth` 加 1。

对字典任务（括号分组）：不跑执行器、不规划、**不加深度**，直接按它的 `steps` / `logic`
展开。

子任务循环：

- `logic` 为 `or` 且子任务成功 → 立即返回（`or-success`）。
- 任何子任务成功 → 把它的 `completion + "\n"` 拼进当前累计文本。
- `logic` 为 `and` 且子任务失败 → 立即返回（`and-fail`）。
- 全部跑完 → 返回最后一个子任务的结果（`children-done`）。

只有成功节点的真实动作会追加进 `checkpoint`（根到叶的先序顺序）。

## 计划解析

规划文本按行扫描：

- 以 `Step` 开头的行是子任务，剥掉行内所有 `Step <数字>: ` 前缀后作为子任务文本；
- 含 `execution order`（大小写不敏感）的行给出执行顺序，取 `": "` 切分后的**最后一段**；
- 表达式词法只认 `Step <数字>` / `AND` / `OR` / `(` / `)`，其余字符忽略；同一层出现多个
  逻辑词时**后写的覆盖先写的**；括号内形成嵌套子计划；
- 解析过程任何异常（没有执行顺序行、括号不配对、`Step` 编号越界）都整体兜底为
  `{"steps": [子任务文本...], "logic": "AND"}`——嵌套结构就此丢失。

## 信息传播

`info_prop` 从 `completion` 里抠：先 `rstrip("\n>")`，要求剩下的文本**字面**含
`Task completed!`（大小写敏感，与执行器的判定口径不同），按行切开后丢掉最后两行，再跳过
尾部的 think 对，取剩下最后两行（`act` 行会剥掉其中出现的 `Task completed!`）。

`"last-step-last-act"` 模式在每处成功（叶子、子节点、`OR` 命中）都刷新；
`"all-step-last-act"` 模式只在叶子成功时把 `past + "\n" + 提取值` 去掉前导换行后写回。

## 样例

```
task   = "t"
exec   = {"t": {"steps": [{"act": "look", "obs": "Nothing happens."}]}}
plan   = {"t": "Step 1: wash the mug\nStep 2: put the mug\n"
               "Execution order: (Step 1 AND Step 2)"}
返回   = success=True, runs=6, checkpoint=["go to sink 1", "go to cabinet 1",
                                          "put mug 1 in/on cabinet 1"]
```

## 约束

- 纯函数：不联网、不读写文件、不修改入参（内部需自行深拷贝）。
- 缺执行脚本、需要分解却缺计划、计划解析出 0 个子任务、`max_depth < 1`、
  `max_patience < 1`、`task` 为空、`info_prop_mode` 非法，一律抛 `ValueError`。
