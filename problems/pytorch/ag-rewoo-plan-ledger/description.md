# ReWOO 占位符计划账

## 题意

ReWOO（Xu et al., 2305.18323）指出 ReAct 式交错推理的冗余：Planner 一次性产出完整计划，Worker 按计划逐条执行工具，Solver 只在最后看到全部证据——观察结果不再反反复复灌回 LLM。官方实现（billxbf/ReWOO）的 `algos/PWS.py` 是这套 Plan-Worker-Solve 的主类。

Planner 的输出文本、每个工具的真实返回、Solver 的最终答案都是不可判定的 LLM/工具调用，本题把它们**全部作显式入参**，只考三段账：

- **Plan 解析**：以 `Plan:` 开头的行进 plans；`#E<d> = ...` 行按第一个 `=` 切分（两侧 strip），证据号 strip 后**恰为 `#Ed`（单数字下标，len==3）才有效**——`#E12 = Google[q]` 整行被记成 `No evidence found`，工具调用被丢弃。
- **Worker 替换**（按证据插入序逐条）：无 `[` 的声明行原文即证据（pass-through）；有 `[` 的行 `split("[",1)` 取工具名，其余部分**盲删最后一个字符**（约定为 `]`），再按原文出现序对 `#E\d+` 占位符 replace 全量替换为 `[证据]`——**插入文本中新出现的占位符也会被后续替换命中**（官方 replace-all quirk）；尚未算出的前向引用原样保留。工具在 tools 中才执行，否则证据 = `No evidence found`；Google 计调用次数，LLM 计 `len(调用串+证据)//4` 个 token。
- **装配与成本**：第 i 个 Plan 行与证据 `#E{i+1}` 按**下标**配对拼 `worker_log`（多出的证据不进 log）；`token_cost = planner价×planner_tokens + solver价×solver_tokens + 固定 davinci-003 价×LLM token`；`tool_cost = Google次数×0.01`；`steps = len(plans)+1`。

价格表（官方 `get_token_unit_price`）：text-davinci-003=0.00002、gpt-3.5-turbo=0.000002、gpt-4=0.00003、两个 alpaca-lora 权重名=0.0，未知名报 `ValueError`。

## 求解

实现 `solve(planner_response, worker_outputs, solver_output, planner_tokens=0, solver_tokens=0, tools=("Google","LLM"), planner_model=..., solver_model=...)`，返回账本 dict（plans/planner_evidences/worker_calls/worker_evidences/worker_log/output/tool_usage/steps/total_tokens/token_cost/tool_cost/total_cost）。域约束：证据行缺 `=`、有效工具调用缺 worker 输出、Plan 行缺配对证据 → `ValueError`。

## 样例

```python
resp = "Plan: a\n#E1 = Quote #E2 line\n#E2 = Google[q]\n#E3 = LLM[#E1 and #E2]\n"
solve(resp, {"#E2": "cats", "#E3": "ans"}, "ans")["worker_calls"]["#E3"]
# == "[Quote [cats] line] and [cats]"
# #E1 先替换成 "[Quote #E2 line]"，随后 #E2 的全量替换把插入文本里的 #E2 也换掉
```

## 数据范围

- 步数 ≤ 6、证据号 1–12（含多位下标 quirk）、文本为短字符串；200 组随机对拍 + 全锚点精确断言。
