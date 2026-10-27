# Agent 基建：工具调用参数校验与归一

## 问题

function calling 时代，模型的「动作」不再是字符串而是结构化参数：`{"city": "Beijing", "unit": "c"}`。这些 JSON 是模型**生成**的，一定会有偏差——漏必填字段、把整数写成字符串、把枚举写成近义词、多塞一个没声明的键。运行时如果直接把字典喂给工具，异常会以 `TypeError` 甚至静默错误的形式在下游炸开。

所以每个 agent 框架都有一层 schema 校验（OpenAI 的 `strict: true`、LangChain 的 `Pydantic` 包装、MCP 的 `inputSchema`）。本题把这层写成纯函数：按 schema 校验、把能兜的类型兜回来（`3.0 → 3`）、补默认值、把错误按稳定顺序报出来。

## 任务

实现 `solve(schema, arguments)` → `(normalized, errors)`：

- **必填**：`required` 中的字段缺失报 `$.<name>: missing required property`；若该字段自带 `default`，按默认值补齐、不算缺失。
- **类型**：`string` / `boolean` 严格；`integer` 接受 `int` 与「值为整数的 `float`」并归一成 `int`，拒绝 `bool`；`number` 接受 `int` / `float` 并归一成 `float`，拒绝 `bool`；`array` 走 `items` 递归（路径 `$.tags[i]`）；`object` 走嵌套 `properties` / `required` 递归（路径 `$.meta.days`）。
- **枚举**：`enum` 按「类型 + 值」双重要求匹配，不命中报 `$.<name>: value <repr> is not one of <list>`。
- **默认值**：缺失且非必填的字段若有 `default` 则填入。
- **多余字段**：`additionalProperties` 为 `False` 时报 `$.<name>: unexpected property`，否则原样保留。
- 校验失败的字段不写入 `normalized`；错误顺序为「必填缺失 → properties 插入顺序 → 多余字段」。

## 示例

```
schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer", "default": 1},
        "unit": {"type": "string", "enum": ["c", "f"]},
    },
    "required": ["city", "unit"],
}
arguments = {"city": "Beijing", "unit": "c", "days": 3.0}

solve(schema, arguments) -> ({"city": "Beijing", "days": 3, "unit": "c"}, [])

solve(schema, {"city": 42, "unit": "k"}) -> ({"days": 1}, [
    "$.city: expected string, got int",
    "$.unit: value 'k' is not one of ['c', 'f']",
])
```

## 约束

- **`bool` 是 `int` 的子类**：`isinstance(True, int)` 为真，所以 `integer` 必须先显式排除 `bool`，否则 `True` 会被当成 1 送进工具。
- **枚举也要看类型**：`0 == 0.0 == False` 在 Python 里都成立，只比 `==` 会让 `True` 命中 `enum=[0, 1]`；必须同时比 `type`。
- **模型最常见的漂移**：把 `3` 写成 `3.0`，把 `true` 写成 `"true"`。前者可兜（归一成 `int`），后者不可兜（报错让模型重试），这条线要划清楚。
- **错误路径必须可定位**：`$` 表示根、`.` 表示字段、`[i]` 表示数组下标，嵌套时逐层拼接。
- **非法输入抛 `ValueError`**：`schema` / `arguments` 不是 `dict`，`properties` 不是 `dict`，`required` 不是 `list`。
