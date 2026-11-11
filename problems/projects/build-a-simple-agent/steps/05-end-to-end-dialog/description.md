# 多轮脚本化对话终局验收

## 背景

前四块各自可判，但 Agent 的价值在整条数据流。终局把多轮脚本化对话跑一遍，断言每轮工具调用序列、失败断点、最终答案与账本规模——仍然全部是显式入参。

## 任务

用四个冻结模块拼装 `run_dialog`：每轮跑循环、统计有效/无效调用、算重试断点、写账本并压缩，最后汇总多轮报告。

实现 `solve(turns, tools, policy)`。

## 输入

见下方示例与规则；所有「模型输出」「工具返回」都是显式入参（fixture），判题环境里没有模型、网络、随机源与时钟。

## 输出

一个可直接逐字段比对的字典；字段名与取值见示例。

## 规则

- 每个 `turn` 含 `user`、`script`（fixture 回复）、`outcomes`（fixture 工具结果）。\n- 每轮报告含 `turn`、`user`、`tool_calls`（仅有效调用）、`invalid_calls`、`breakpoint`、`breaker_open`、`final`。\n- 账本每轮追加 `"<逗号连接的工具名>|<final 或空>"`，按 `keep_last` / `max_chars` 压缩；`policy.retry` 传给第 04 步的计划器。\n- 返回 `{"turns", "final_answer", "total_tool_calls", "compactions", "ledger_chars"}`；`final_answer` 取最后一轮的 `final`。

## 示例

```python
turns = [{"user": "find a",
          "script": [{"thought": "x",
                      "action": {"tool": "search", "args": {"query": "a"}}},
                     {"thought": "y", "final": "A"}],
          "outcomes": ["ok"]}]
solve(turns, tools, policy)
# {"turns": [{"turn": 1, "user": "find a", "tool_calls": ["search"],
#             "invalid_calls": 0, "breakpoint": "success",
#             "breaker_open": False, "final": "A"}],
#  "final_answer": "A", "total_tool_calls": 1,
#  "compactions": 0, "ledger_chars": 8}

```

## 约束

- 本步通过后解锁项目总览页；进度单独记在 `data/project_progress.json`。\n- 没有训练、没有收敛判定；不得调用模型、网络或时钟，重复调用结果必须一致。
