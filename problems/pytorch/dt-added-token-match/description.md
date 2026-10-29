# Added Token 隔离与最长匹配

## 任务

实现 `solve`，从 ASCII 原文中隔离已注册的 added token，并保留所有普通原文及其偏移。本题只实现 identity normalization、无空白吸收、同一匹配阶段的受限模型，不做 BPE。

## 输入

- `text`：长度 0 到 300 的 ASCII 字符串。
- `added_tokens`：长度 0 到 30 的注册列表，每项字典恰含：
  - `content`：非空、无空白 ASCII 字符串，长度不超过 30。
  - `single_word`：布尔值，是否在选出原始候选后检查左右单词边界。
  - `special`：布尔值，是否为可跳过的特殊候选。
- `split_special_tokens`：布尔值，默认 `False`。
- 若 `content` 为空或重复，必须抛出 `ValueError`，包括原文为空、重复项将被 special 过滤的情况。其他输入保证合法。
- 不得修改输入列表或其中的字典。

## 精确匹配规则

1. **先过滤 special**：若 `split_special_tokens=True`，先从候选集合移除全部 `special=True` 的 pattern；其他 pattern 仍可匹配，注册下标不变。
2. **再生成非重叠原始候选**：从左到右，选起点最早的出现；同一起点选最长的 content。选中后从该区间末尾继续。不是从全文选择最长，也不是按注册先后选短串。
3. **最后检查边界**：若选中的候选 `single_word=True`，其左右紧邻字符若存在，均不能是 ASCII 字母 `A-Z/a-z`、数字 `0-9` 或 `_`。只检查邻居，不要求 content 本身以单词字符开头或结尾。
4. 边界失败的长候选归入普通原文，但它已经消耗了整个原始候选区间。**不能回退短 pattern，也不能重新扫描其内部**，包括内部开始、延伸到候选之外的匹配。
5. 匹配大小写敏感。special token 不自带额外优先权。所有 pattern 同一阶段，没有数字 priority；不执行规范化、BPE 或 `lstrip` / `rstrip` 空白吸收。

本题“先移除 special”的规则与 tokenizers 0.23.3 原生实现的“生成候选后跳过 special”不同，以本题契约为准；详见 notes。

## 输出

返回 `(piece, start, end, added_id)` 元组列表：

- `piece` 必须是原文切片，`start`、`end` 是 Python 字符半开偏移。
- 成功的 added token 单独输出，`added_id` 为**原注册列表下标**。
- 所有连续未匹配原文（包括边界失败候选）合并为一个普通片段，ID 为 `-1`。
- 相邻成功匹配即使 ID 相同，也不能合并。不得输出空片段。
- 输出连续覆盖原文；空原文返回 `[]`，但仍需先验证注册表。

## 示例 1：左最早，而非全局最长

```python
text = 'abcde'
added_tokens = [
    {'content': 'ab', 'single_word': False, 'special': False},
    {'content': 'bcde', 'single_word': False, 'special': False},
]
split_special_tokens = False
输出 = [('ab', 0, 2, 0), ('cde', 2, 5, -1)]
```

`ab` 起点更早，因此阻止后面的重叠 `bcde`。

## 示例 2：边界失败不能回退

```python
text = 'zabc!'
added_tokens = [
    {'content': 'ab', 'single_word': False, 'special': False},
    {'content': 'abc', 'single_word': True, 'special': False},
    {'content': 'b', 'single_word': False, 'special': False},
]
split_special_tokens = False
输出 = [('zabc!', 0, 5, -1)]
```

先选中 `abc`，再因左邻 `z` 拒绝。不能回退到 `ab` 或重新扫描内部的 `b`。

## 示例 3：预先移除 special 改变候选

```python
text = 'abc!'
added_tokens = [
    {'content': 'ab', 'single_word': False, 'special': False},
    {'content': 'abc', 'single_word': True, 'special': True},
]
split_special_tokens = True
输出 = [('ab', 0, 2, 0), ('c!', 2, 4, -1)]
```

若使用同一输入但将开关设为 `False`，结果为：

```python
[('abc', 0, 3, 1), ('!', 3, 4, -1)]
```

## 实现要求

仅使用 Python 标准库。推荐用 trie 在各扫描位置选择最长原始候选，再单独检查边界。无需实现完整 AddedVocabulary。
