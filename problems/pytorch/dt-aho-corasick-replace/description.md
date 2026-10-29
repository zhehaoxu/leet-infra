# Aho–Corasick 多模式替换

## 题目描述

给定原始字符串 `text`、模式列表 `patterns` 和一一对应的替换字符串列表 `replacements`，先找出原文中的所有模式匹配，再按“最左优先、同起点最长优先”的规则选择不重叠片段，执行一次性替换。

请使用 **trie、BFS 构建的 failure links（失败链接）及继承输出的 Aho–Corasick 自动机**完成多模式检测，不要在原文每个位置逐一尝试所有模式。

## 输入与约束

- `text`：由 Unicode 标量字符组成的 Python `str`，长度为 0..500。
- `patterns`：含 0..30 项的字符串列表，每项长度为 1..30，允许重复。
- `replacements`：与 `patterns` 等长的字符串列表，单个替换字符串可以为空。
- **只要任意模式为空字符串，就抛出 `ValueError`**，即使原文为空；其余输入保证合法。
- 长度和偏移按 Python Unicode code point 计算，不按 UTF-8 字节或字素簇计算；不做大小写转换或 Unicode 归一化。
- 不修改任何输入。仅需 Python 标准库，不需要张量。

## 输出与规则

返回 Python 元组 `(rewritten, all_matches, selected)`：

1. `rewritten` 是替换后的 Python 字符串。
2. `all_matches` 是由 Python 三元组 `(start, end, pattern_index)` 组成的列表。下标均从 0 开始，`[start, end)` 是**原始** `text` 的半开区间。列出所有匹配，包括重叠匹配、后缀匹配，以及重复模式在各自下标下的独立记录；按 `(start, end, pattern_index)` 升序排列。
3. `selected` 是同样格式的三元组列表，按以下贪心选择顺序排列：
   - 初始已消费位置为 0，只考虑 `start` 不小于已消费位置的匹配。
   - 优先选择最小 `start`；同起点选择最大 `end - start`；仍相同则选择最小 `pattern_index`。
   - 选中后将已消费位置更新为该匹配的 `end`，跳过所有 `start < end` 的匹配；恰好从 `end` 开始的匹配仍可选。
   - 重复直到没有可选匹配。较晚开始的更长匹配不能抢占较早开始的匹配。

使用每个选中匹配对应的 `replacements[pattern_index]` 替换原文片段。未覆盖原文原样保留，替换文本和拼接后产生的新匹配都**不再扫描**。所有区间始终相对于原文。列表内每个下标必须为 Python `int`，不能用张量或其他容器替代规定类型。

没有模式或没有匹配时，返回原文及两个空列表。

## 示例 1：失败链接、重复模式和最左优先

输入：

```python
text = 'ushers'
patterns = ['he', 'she', 'hers', 'he']
replacements = ['H', 'S', 'R', 'D']
```

输出：

```python
('uSrs', [(1, 4, 1), (2, 4, 0), (2, 4, 3), (2, 6, 2)], [(1, 4, 1)])
```

`she` 的结束位置也产生两条 `he` 的后缀匹配。虽然 `hers` 更长，但它开始得更晚，所以选中的是 `[1, 4)` 的 `she`。原文前缀 `u` 和后缀 `rs` 保留。

## 示例 2：组合字符、Unicode 偏移和删除

字符串中的转义按 Python 字符串字面量解释；`\xe9` 是 `é`，`\u0301` 是组合重音，`\U0001f642` 是一个补充平面字符。

输入：

```python
text = 'e\u0301\xe9\U0001f642\xe9'
patterns = ['\xe9', 'e\u0301', '\U0001f642\xe9']
replacements = ['E', 'C', '']
```

输出：

```python
('CE', [(0, 2, 1), (2, 3, 0), (3, 5, 2), (4, 5, 0)], [(0, 2, 1), (2, 3, 0), (3, 5, 2)])
```

`e` 加组合重音占两个 code point，预组合的 `é` 占一个；二者不会被归一化为同一模式。最后选中的 `[3, 5)` 被空字符串删除。
