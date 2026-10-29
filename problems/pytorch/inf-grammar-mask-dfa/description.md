# 受限解码：正则文法约束下的候选 token 掩码

## 题目描述

要让模型只输出合法结构（JSON、SQL、函数调用参数），靠 prompt 求它"听话"并不可靠，工程上的做法是**约束解码**：每一步在采样之前，把词表里会导致文法走不下去的 token 直接屏蔽掉。推理引擎由此算出张布尔掩码，采样器只在掩码为真的位置里挑。

难点不在采样，而在**状态**。token 不是单个字符：一个 token 可能把自动机推过好几个状态，也可能第一个字符就把状态走死。而"这个 token 现在能不能用"问的是"加上它之后还能不能补完"，不是"加上它之后是不是已经完整"。这两件事混在一起，是这类实现最常见的 bug 来源。

本题要求实现一个小型正则文法的自动机，并对给定词表同时算出三件事：哪些 token 还能选、当前 prefix 是否已经可以结束、哪些 token 拼上之后刚好完整。

## 规则

**文法**（pattern 的合法形式，优先级从低到高）：

- `A|B`：选择，取 A 或 B；
- `AB`：连接，相邻单元直接拼接；
- `X*` / `X+` / `X?`：重复 0+ 次 / 1+ 次 / 0–1 次；
- `(X)`：分组；
- `[abc]`：字符类，类内每个字符都是字面字符（不支持 `^` 与 `-`）；
- 其余字符：字面字符。

不会出现空分支（如 `a|`）、空组 `()`、转义字符、`.`、`^`、`$`、`{m,n}`。

记 `L(pattern)` 为被 `pattern` 完整匹配的全部字符串。给定 `pattern`、候选词表 `vocab` 与已生成文本 `prefix`，按顺序返回三元组：

- `mask`：长度 `len(vocab)` 的 `bool` 列表。`mask[i]` 为真当且仅当**存在**（可为空的）字符串 `w`，使 `prefix + vocab[i] + w` 属于 `L(pattern)`；
- `accept`：单个 `bool`，`prefix` 本身是否属于 `L(pattern)`，即"现在就可以停下来"；
- `complete`：长度 `len(vocab)` 的 `bool` 列表。`complete[i]` 为真当且仅当 `prefix + vocab[i]` 属于 `L(pattern)`。恒有 `complete[i]` 蕴含 `mask[i]`。

注意可空模式：`pattern = "a*"` 时空串也是完整匹配，所以空前缀下 `accept` 为真；但这不影响 `mask` —— `mask` 只回答"还能不能继续"。

## 示例

**示例 1**：`pattern = "a(b|c)*d"`、`prefix = "a"`、`vocab = ["b", "c", "d", "e", "bb"]`

```
"a" + "b"  = "ab"   后面还能补 "d"          -> mask 真，complete 假
"a" + "c"  = "ac"   后面还能补 "d"          -> mask 真，complete 假
"a" + "d"  = "ad"   已经是完整匹配          -> mask 真，complete 真
"a" + "e"  = "ae"   'e' 不在文法字母表       -> mask 假，complete 假
"a" + "bb" = "abb"  后面还能补 "d"          -> mask 真，complete 假

mask     = [True, True, True, False, True]
accept   = False
complete = [False, False, True, False, False]
```

**示例 2**：`pattern = "ab[cd]?e"`、`prefix = "ab"`、`vocab = ["c", "d", "e", "cde", "ce", "b"]`

```
"abc"   可选位被 'c' 占掉，还差 'e'      -> 真 / 假
"abd"   同上                             -> 真 / 假
"abe"   完整                             -> 真 / 真
"abcde" 可选位只有一位，'d' 多余          -> 假 / 假
"abce"  完整                             -> 真 / 真
"abb"   'b' 不在 [cd]?e 里                -> 假 / 假

mask     = [True, True, True, False, True, False]
accept   = False
complete = [False, False, True, False, True, False]
```

**示例 3**（可空模式）：`pattern = "a*"`、`prefix = "aa"`、`vocab = ["a", "b", "aa"]`

```
mask     = [True, False, True]
accept   = True
complete = [True, False, True]
```

`"b"` 不可能出现在 `a*` 的任何匹配里，所以两个列表的第二项都是假；而 `prefix` 本身已经是完整匹配，故 `accept` 为真。

## 约束

- `pattern` 长度不超过 32；`vocab` 长度 1–64，每个 token 非空、长度不超过 8；`prefix` 长度不超过 32。
- `vocab` 与 `prefix` 中允许出现文法字母表以外的字符，这类 token 的 `mask` / `complete` 必然全假。
- 判题只比较三个输出的语义，不限定自动机实现（位置自动机、NFA 子集构造、带记忆的导数都可以）。
- `mask` / `complete` 必须是 `bool` 列表，`accept` 必须是 `bool`。
