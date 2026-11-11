# L4 设计复盘：为什么不能有状态判题

## 背景

最后一步不写代码逻辑，而是考整体认知：六个阶段的顺序能否说清，以及哪些决策在项目里被明令禁止。答案完全确定，仍可自动判。

## 任务

用冻结的 `scaffold.dialog_runner` 阶段词表与审计函数，判定一组架构决策与一条数据流 trace 是否合规。

实现 `solve(decisions, trace)`。

## 输入

见下方示例与规则；所有「模型输出」「工具返回」都是显式入参（fixture），判题环境里没有模型、网络、随机源与时钟。

## 输出

一个可直接逐字段比对的字典；字段名与取值见示例。

## 规则

- 官方阶段顺序为 `validate → loop → ledger → retry → dialog → review`。\n- `decisions` 含禁止项（`replay_user_code`、`judge_convergence`、`call_llm`）时 `reason="forbidden"`；顺序不符 `reason="order"`；trace 与官方阶段不一致 `reason="trace"`；全部合规 `reason="ok"`。\n- 判定优先级：禁止项 > 顺序 > trace。\n- 返回 `{"accepted", "reason", "stages", "trace_len"}`；`stages` 恒为官方六阶段，`trace_len = len(trace)`。

## 示例

```python
stages = ["validate", "loop", "ledger", "retry", "dialog", "review"]
solve(stages, stages)
# {"accepted": True, "reason": "ok",
#  "stages": [...], "trace_len": 6}
solve(stages + ["replay_user_code"], stages)
# {"accepted": False, "reason": "forbidden", ...}

```

## 约束

- 这是 L4 设计复盘，不是选择题 UI：仍按普通题形态用 `solve` 返回字典判定。\n- 不得改写官方阶段词表；重复调用结果必须一致。
