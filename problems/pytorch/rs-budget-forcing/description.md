# Budget Forcing 推理预算账：截断 / Wait 续跑 / 回答截断

## 题意

s1（Skip Thinking）的 budget forcing 是一种推理时控制：给思维链一个 token 预算，超预算就把模型硬切到最终回答（`</think>` + 强制 final）。本题转写其核心状态机账本：给定思考步序列 `think_steps`（每步 token 数）、预算 `budget`、wait 步序列 `wait_steps`（续跑提示的 token 数）与 `wait_cost`、回答 token 序列 `final_tokens`，逐账输出三段状态机的发射记录。

## 求解

三段状态机口径题面钉死：

1. **思考段**：按序发射 `think_steps`。若 `emitted + c <= budget` 整步发射；否则发射 `budget - emitted` 个、`forced_final=True`、思考终止（length forcing：被切的那步只发前缀，记录进 `step_emitted`）。
2. **Wait 段**（仅在未被截断时）：按序尝试 `wait_steps`。每步先要求 `remaining >= wait_cost`（付得起续跑提示），付则扣 `wait_cost` 再按规则 1 发射该 Wait 步（可能再次截断，截断后停止后续 Wait）；付不起则停止尝试（该步 `wait_log=False`、发射 0）。已被截断则跳过全部 Wait。
3. **回答段**：`final_tokens` 总量按 `remaining` 截断发射，被切则 `answer_truncated=True`。

返回 `{"step_emitted", "think_emitted", "forced_final", "waits_used", "wait_log", "wait_emitted", "final_emitted", "answer_truncated", "total"}`；`think_emitted` 只累计思考段（Wait 步的 token 计入 `wait_emitted` 与总量，不计入 `think_emitted`）。

## 数据规模

- 1 ≤ budget ≤ 10^6，序列长度 ≤ 10^4，各 token 数为非负 int。
- think_steps/final_tokens/wait_steps 非列表、条目为 bool/负数/非 int、budget 非正 int、wait_cost 为 bool/负数 → `ValueError`。
