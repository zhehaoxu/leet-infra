# 工具 schema 校验器

## 背景

Agent 的第一步不是「让模型说话」，而是把一次工具调用是否合法变成可断言的纯函数。模型输出是 fixture，校验器只吃 `spec` 与 `args`。

## 任务

按 `spec` 校验 `args`，补齐默认值，报告三类问题，并给出规范化的调用字典。

实现 `solve(spec, args)`。

## 输入

见下方示例与规则；所有「模型输出」「工具返回」都是显式入参（fixture），判题环境里没有模型、网络、随机源与时钟。

## 输出

一个可直接逐字段比对的字典；字段名与取值见示例。

## 规则

- 返回 `{"ok", "tool", "missing", "unexpected", "type_errors", "normalized"}`；三个问题列表按字典序排序。\n- 缺必填进 `missing`；`strict=True` 时多余键进 `unexpected`；类型不符的键进 `type_errors` 且不进入 `normalized`。\n- 有 `default` 的声明参数在缺省时补进 `normalized`；`type` 支持 `any/str/int/float/number/bool/list/dict`，`bool` 不算 `int`。\n- `ok = name 非空 且 三个问题列表都为空`；`normalized` 只含已声明且类型正确的键。

## 示例

```python
spec = {"name": "search", "strict": True,
        "params": {"query": {"type": "str", "required": True},
                   "top_k": {"type": "int", "required": False, "default": 3}}}
solve(spec, {"query": "hello"})
# {"ok": True, "tool": "search", "missing": [], "unexpected": [],
#  "type_errors": [], "normalized": {"query": "hello", "top_k": 3}}

```

## 约束

- 纯函数：不得修改入参、不得读全局状态、不得抛异常；`args` 为空时按 `{}` 处理。\n- 判题环境只有 Python 标准库 + torch/numpy；本步不需要张量。
