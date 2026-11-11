# ReAct 循环状态机

## 背景

模型回复不可复现，但「给定一串回复，循环该怎么走」完全可判。把模型输出外置成 `script` 入参，循环本身就是纯状态机。

## 任务

消费至多 `max_steps` 条 fixture 回复，产出逐轮轨迹、终止原因与最终答案；action 用冻结的 step-01 校验器判断是否可执行。

实现 `solve(script, tools, max_steps)`。

## 输入

见下方示例与规则；所有「模型输出」「工具返回」都是显式入参（fixture），判题环境里没有模型、网络、随机源与时钟。

## 输出

一个可直接逐字段比对的字典；字段名与取值见示例。

## 规则

- `script` 每项含 `thought`；带 `action` 的是工具调用，带 `final` 的是终止回复（即使 `final` 为空字符串也算终止）。\n- 轨迹项 `type` 为 `action` / `noop` / `final`；`action` 项额外含 `tool`、`args`、`valid`。\n- `valid = 工具名在 tools 中 且 validate(spec, args)["ok"]`；无效 action 只记录，不执行。\n- `terminated_by` 为 `final`（遇到终止回复）/ `budget`（用满额度且脚本仍有条目）/ `script_end`（脚本先耗尽）；`steps_used = len(trace)`。

## 示例

```python
tools = {"search": {"name": "search", "strict": True,
                    "params": {"query": {"type": "str", "required": True}}}}
script = [{"thought": "need info",
           "action": {"tool": "search", "args": {"query": "a"}}},
          {"thought": "done", "final": "answer-1"}]
solve(script, tools, 6)
# {"trace": [...], "steps_used": 2,
#  "terminated_by": "final", "final": "answer-1"}

```

## 约束

- 官方模块以只读方式注入沙箱：`from scaffold.tool_schema import validate`；不得 import 自己的前序代码。\n- 不得修改 `script` / `tools`；重复调用同一入参必须返回完全一致的结果。
