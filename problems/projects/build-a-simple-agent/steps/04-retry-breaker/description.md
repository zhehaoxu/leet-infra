# 重试退避与熔断预算

## 背景

工具会失败。若把失败交给真实时间或随机抖动，判题就不再确定。退避必须是**算出来的数字计划**：所有结果由 fixture 给定，没有 sleep、没有时钟。

## 任务

复用冻结的循环拿到本轮有效工具序列，再按 `policy` 把 `outcomes` 编成重试/退避/熔断计划。

实现 `solve(script, tools, outcomes, policy)`。

## 输入

见下方示例与规则；所有「模型输出」「工具返回」都是显式入参（fixture），判题环境里没有模型、网络、随机源与时钟。

## 输出

一个可直接逐字段比对的字典；字段名与取值见示例。

## 规则

- 逐条读 `outcomes`：`"ok"` 立即成功；其余视为失败并把退避 `base_delay_ms * factor**len(delays)` 追加进计划。\n- 累计退避超过 `budget_ms` 时停在 `budget`；尝试数用尽停在 `attempts`；fixture 先耗尽停在 `outcomes`。\n- 最近 `window` 次内连续失败数达到 `breaker_threshold` 时打开熔断，停在 `breaker`；熔断优先于退避。\n- 返回 `{"actions", "terminated_by", "retry"}`，`retry` 含 `attempts`、`delays`、`total_delay_ms`、`breaker_open`、`succeeded`、`stopped_by`。

## 示例

```python
policy = {"max_steps": 6, "max_attempts": 4, "base_delay_ms": 10,
          "factor": 2, "budget_ms": 100,
          "breaker_threshold": 2, "window": 3}
solve(script, tools, ["error", "error", "ok"], policy)
# {"actions": ["search"], "terminated_by": "final",
#  "retry": {"attempts": ["error", "error"], "delays": [10],
#            "total_delay_ms": 10, "breaker_open": True,
#            "succeeded": False, "stopped_by": "breaker"}}

```

## 约束

- `actions` 只含轨迹中 `valid=True` 的工具名，顺序不变；熔断/预算只影响计划，不修改轨迹。\n- 不得调用 `time`、`random`、`sleep`；重复调用结果必须一致。
