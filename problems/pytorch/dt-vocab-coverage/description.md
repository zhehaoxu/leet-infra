# 词表覆盖率与 OOV 统计

## 题目描述

给定已分词的文档列表 `documents`、词表 `vocab` 和忽略词列表 `ignored_tokens`，统计保留语料中词表内词与词表外词（OOV）的出现次数、文档频次，以及 token 覆盖率和词种覆盖率。

先移除与忽略词**精确相等**的 token 出现，再计算全部统计。不做小写转换、Unicode 归一化或子串匹配。OOV 必须保留独立词形，不能合并到一个 unknown ID。

## 输入与约束

- `documents` 为 `list[list[str]]`，包含 0..30 篇文档，每篇含 0..100 个 token。
- 每个 token 是长度为 1..30 的非空 Unicode 字符串，长度按 Python code point 计算。
- `vocab` 为含 0..100 个唯一 token 的列表。**有重复项时抛出 `ValueError`**，即使该词被忽略或语料为空；其余输入保证合法。
- `ignored_tokens` 是 token 列表，允许重复；重复忽略词不额外影响结果。
- 保留文档顺序，空文档和过滤后为空的文档都不能丢弃。
- 不修改任何输入，包括文档的内层列表。仅需 Python 标准库，不需要张量。

## 返回结果

返回恰好包含以下 8 个键的 Python `dict`，字典键的插入顺序不限：

| 键 | Python 类型 | 含义 |
| --- | --- | --- |
| `vocab_counts` | `list[int]` | 每个词表词在保留语料中的总出现次数，与 `vocab` 原顺序对齐 |
| `vocab_doc_freq` | `list[int]` | 包含该词表词的保留文档数，每篇文档对同一词最多贡献 1，与 `vocab` 对齐 |
| `oov_tokens` | `list[str]` | 保留语料中的唯一 OOV token，按下述排序规则排列 |
| `oov_counts` | `list[int]` | 与 `oov_tokens` 对齐的语料总出现次数 |
| `oov_doc_freq` | `list[int]` | 与 `oov_tokens` 对齐的文档频次，每篇对同一词最多贡献 1 |
| `document_stats` | `list[tuple[int, int]]` | 每篇文档的 `(保留出现次数, OOV 出现次数)`，按输入文档顺序排列 |
| `token_coverage` | `float` | 保留的词表内 token 出现次数 / 全部保留 token 出现次数 |
| `type_coverage` | `float` | 被观察到的保留词表内词种数 / 全部被观察到的保留词种数 |

额外规则：

- OOV 按 `(-corpus_count, token)` 排序：先总次数降序，再按 Python 字符串字典序升序，不采用语言区域排序或数字排序。
- 被忽略或从未出现的词表词在 `vocab_counts`、`vocab_doc_freq` 中均为 0，不从词表对齐结果中删除。
- `document_stats` 统计的是**出现次数**而不是唯一词种数；空文档对应 `(0, 0)`。
- 两种覆盖率分别使用各自的分母。分母为 0 时返回 `0.0`；未观察到的词表词不计入词种分母。不对覆盖率做额外舍入。
- 所有计数均为 Python `int`，容器和浮点数使用表中的 Python 内置类型。

## 示例 1：忽略词与空文档

输入：

```python
documents = [['a', 'x', 'a'], [], ['x', 'b']]
vocab = ['b', 'a', 'z']
ignored_tokens = ['b']
```

输出：

```python
{
    'vocab_counts': [0, 2, 0],
    'vocab_doc_freq': [0, 1, 0],
    'oov_tokens': ['x'],
    'oov_counts': [2],
    'oov_doc_freq': [2],
    'document_stats': [(3, 1), (0, 0), (1, 1)],
    'token_coverage': 0.5,
    'type_coverage': 0.5
}
```

忽略 `b` 后共保留 4 次出现，其中已知词 `a` 出现 2 次；观察到的两个词种是 `a` 与 `x`，其中一个属于词表。`a` 虽出现两次，但只出现在一篇文档里。

## 示例 2：精确 Unicode 匹配与不同分母

转义按 Python 字符串字面量解释：`\xe9` 是 `é`，`\xc9` 是 `É`，`\u732b` 是 `猫`。`e\u0301` 不会被归一化为 `é`。

输入：

```python
documents = [
    ['\xe9', 'e\u0301', '\xc9', '\u732b', '\U0001f642', '\U0001f642'],
    ['\xe9', '\xc9', '\U0001f642']
]
vocab = ['\xe9', 'unused']
ignored_tokens = ['\xc9']
```

输出：

```python
{
    'vocab_counts': [2, 0],
    'vocab_doc_freq': [2, 0],
    'oov_tokens': ['\U0001f642', 'e\u0301', '\u732b'],
    'oov_counts': [3, 1, 1],
    'oov_doc_freq': [2, 1, 1],
    'document_stats': [(5, 4), (2, 1)],
    'token_coverage': 0.2857142857142857,
    'type_coverage': 0.25
}
```

保留 7 次出现，其中 2 次已知，故 token 覆盖率为 $2/7$；观察到 4 个词种，其中 1 个已知，词种覆盖率为 $1/4$。两个出现一次的 OOV 按 Python 字典序排列。
