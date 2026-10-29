# 分词：ChatML 模板与角色字节边界

## 任务

实现 `solve`：把消息渲染成本题指定的固定 ChatML 字符串，并标注每个 **UTF-8 字节** 的正文角色和 assistant 监督掩码。输入参数为 `messages`、`add_generation_prompt`（默认 `False`）、`continue_final_message`（默认 `False`）。只需 Python 标准库。

**这是字节标注练习，不是模型分词。** `input_bytes` 不是模型 token ID；其长度不是模型 token 数。本题不使用 BPE、Jinja 解释器或任何模型默认模板，不声称该固定格式是 Qwen 的完整默认模板。

## 输入与错误

- `messages` 是长度 0..20 的列表。每项必须是恰含 `role`、`content` 两个键的 `dict`。
- `role` 只能是 `"system"`、`"user"`、`"assistant"`。消息记录结构或角色不合法时，抛出 `ValueError`。
- `content` 保证是长度 0..1000 的 Unicode 标量字符串：可含非 ASCII 字符、组合字符、换行、空白、空字符串和字面控制标记样式，但不含代理码点。长度按 Python 字符串码点数计。
- 两个开关均为 `bool`。同时为 `True` 时抛出 `ValueError`。
- `continue_final_message=True` 且 `messages` 为空时抛出 `ValueError`。
- 除明确要求报错的情形外，其余输入保证合法。不得修改列表或消息字典。

## 精确渲染规则

每条消息通常严格按下式拼接；代码字符串中的 `\n` 表示一个换行字符：

```text
'<|im_start|>' + role + '\n' + content + '<|im_end|>\n'
```

1. 按输入顺序连接消息，不添加任何隐式 system 消息。
2. 续写开关开启时，**仅最后一条消息**省略 `'<|im_end|>\n'`；与其角色无关。之前的消息仍闭合，最后正文的所有尾部空白仍保留。
3. 生成提示开关开启时，先闭合全部原消息，再追加 `'<|im_start|>assistant\n'`。空消息列表也能添加生成提示。
4. 不转义、不执行、不 trim、不规范化正文。正文里的 `'<|im_end|>'`、`'{{ expression }}'` 等始终是普通正文，不能据此重新解析结构。

## 返回值

返回一个 `dict`，**恰含**以下五个键：

| 键 | 类型与含义 |
| --- | --- |
| `text` | 完整渲染字符串 |
| `input_bytes` | Python `int` 列表，严格等于 `list(text.encode('utf-8'))` |
| `role_ids` | 与字节数相同的 Python `int` 列表；正文 system=0、user=1、assistant=2，其余为 -1 |
| `assistant_mask` | 与字节数相同的 Python `bool` 列表，仅 assistant 正文字节为 `True` |
| `content_spans` | 每条原消息对应一个 `(start, end)` 元组的列表；采用 UTF-8 字节半开区间 |

所有头部、语法、结束标记和新增生成头均为角色 -1、掩码 `False`。结束标记不参与监督。空正文的 `start == end`；新增生成头不增加 span。多字节字符的每个字节都继承同一正文角色。

## 示例 1：多字节正文

输入：

```python
messages = [{"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "好"}]
add_generation_prompt = False
continue_final_message = False
```

输出如下。为方便阅读，列表使用 Python 的重复与连接表达式表示，求值后就是完整返回值：

```python
{
    "text": "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n好<|im_end|>\n",
    "input_bytes": [60, 124, 105, 109, 95, 115, 116, 97, 114, 116, 124, 62,
                    117, 115, 101, 114, 10, 72, 105, 60, 124, 105, 109, 95,
                    101, 110, 100, 124, 62, 10, 60, 124, 105, 109, 95, 115,
                    116, 97, 114, 116, 124, 62, 97, 115, 115, 105, 115, 116,
                    97, 110, 116, 10, 229, 165, 189, 60, 124, 105, 109, 95,
                    101, 110, 100, 124, 62, 10],
    "role_ids": [-1] * 17 + [1] * 2 + [-1] * 33 + [2] * 3 + [-1] * 11,
    "assistant_mask": [False] * 52 + [True] * 3 + [False] * 11,
    "content_spans": [(17, 19), (52, 55)]
}
```

共 66 个 UTF-8 字节；`好` 占 3 字节，因此恰有 3 个 `True`。

## 示例 2：空会话的生成提示

输入 `messages=[]`、`add_generation_prompt=True`、`continue_final_message=False`，输出：

```python
{
    "text": "<|im_start|>assistant\n",
    "input_bytes": [60, 124, 105, 109, 95, 115, 116, 97, 114, 116, 124, 62,
                    97, 115, 115, 105, 115, 116, 97, 110, 116, 10],
    "role_ids": [-1] * 22,
    "assistant_mask": [False] * 22,
    "content_spans": []
}
```

若两个开关均关闭，空会话返回空字符串及四个空列表。

## 示例 3：续写保留尾空白

输入 `messages=[{"role": "assistant", "content": "答 \n"}]`、`add_generation_prompt=False`、`continue_final_message=True`，输出：

```python
{
    "text": "<|im_start|>assistant\n答 \n",
    "input_bytes": [60, 124, 105, 109, 95, 115, 116, 97, 114, 116, 124, 62,
                    97, 115, 115, 105, 115, 116, 97, 110, 116, 10,
                    231, 173, 148, 32, 10],
    "role_ids": [-1] * 22 + [2] * 5,
    "assistant_mask": [False] * 22 + [True] * 5,
    "content_spans": [(22, 27)]
}
```

末尾的空格与换行都属于 assistant 正文，而非自动添加的语法。
