# 记忆账本与确定性压缩

## 背景

循环轨迹会越攒越长。把轨迹写成账本，并在阈值处做**确定性**压缩，长对话才有界、可回放、可复现。压缩文本由条目本身拼出，不调用模型。

## 任务

循环用冻结的 `scaffold.react_loop`；**账本的 append / compact / total_chars 是本步缺失的那块，由你在本文件里实现**。最后返回终止信息、账本、压缩次数与总字符数。

实现 `solve(script, tools, policy)`。

## 输入

见下方示例与规则；所有「模型输出」「工具返回」都是显式入参（fixture），判题环境里没有模型、网络、随机源与时钟。

## 输出

一个可直接逐字段比对的字典；字段名与取值见示例。

## 规则

- 每轨迹项生成 `{"turn": step, "role": type, "text": ...}`：`action` 用 `action:<tool>`，`final` 用 `final:<answer>`，其余用 `noop:`。\n- 每次 append 后调用 `compact(ledger, keep_last, max_chars)`：条目数超过 `keep_last` 时，把最旧的若干条折成一条 `role="summary"`，文本为 `[N]<拼接>` 且截断到 `max_chars*4`。\n- 返回 `{"terminated_by", "final", "ledger", "compactions", "ledger_chars"}`；`ledger_chars` 为最终账本各 `text` 长度之和。\n- `policy` 提供 `max_steps`、`keep_last`、`max_chars`。

## 示例

```python
policy = {"max_steps": 6, "keep_last": 2, "max_chars": 24}
solve(script, tools, policy)
# {"terminated_by": "final", "final": "answer-1",
#  "ledger": [{"turn": 1, "role": "summary", "text": "[1]action:search"}, ...],
#  "compactions": 1, "ledger_chars": 43}

```

## 约束

- 本步只注入前序官方模块（`tool_schema`、`react_loop`）；账本三个函数由你自己实现，不要维护全局账本，重复调用结果必须一致。\n- 本步的参考实现会被冻结成 `scaffold/memory_ledger.py`，供第 05 步使用；`keep_last=0` 时压缩全部条目，`max_chars` 为正整数。
