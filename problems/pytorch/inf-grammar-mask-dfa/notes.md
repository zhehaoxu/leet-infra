# 解题说明 · 正则约束下的 token 掩码

## 出处

- Outlines: *Efficient Guided Generation for Large Language Models*（Willard & Louf，arXiv:2307.09702）——把正则/JSON-Schema 编译成有限自动机（索引自动机），再按自动机状态给词表打掩码，是"FSA 约束解码"最常被引用的实现。
- xgrammar（MLC）与 llama.cpp 的 GBNF 栈：同一思想在 JSON-Schema / 上下文无关文法上的工程实现，把"当前状态"当作缓存键，预计算每个状态下的 token 掩码。

本题把正则子集与词表掩码拆出来单做，不涉及 JSON-Schema 归一化、tokenizer 对齐与文法上下文栈，只考自动机状态本身。

## 核心与易错点

**状态是"刚读完的字符落在哪些位置"。** Glushkov（位置）自动机给 `pattern` 的每个字面位置编号，构造出 `first` / `last` / `follow`：读第一个字符时在 `first` 里挑符号匹配的位置，之后每一步在 `follow` 的后继里挑。初态不是普通状态——若把 `first` 当成"已读位置"再去走 `follow`，第一个字符永远匹配不上（`a*` 会被整体判成非法）。

**合法 ≠ 完整 ≠ 可停。** `mask` 问"还能不能补完"，`complete` 问"拼上就完整了"，`accept` 问"现在的 prefix 已经完整了吗"。三者在 `a(b|c)*d` 上就分得开：`"a"` 后接 `"b"`，`mask` 真而 `complete` 假；接 `"d"` 两者都真；`prefix="abc"` 在 `abc` 下 `accept` 真，但任何 token 的 `mask` 都假（终态之后再无出边）。

**可空模式。** `a*`、`x?` 含 ε 路径。空前缀下 `accept` 必须为真；若允许空 token，它合法当且仅当 prefix 本身已可接受（本题约定 token 非空）。

**多字符 token 要逐字符推进。** token 是一串字符而不是字母表里的一个符号：中途任一字符走空，整个 token 非法；走完还要看落点决定 `complete`。`[ab]+c` 下的 `"cc"` 第一个字符就死，`"abc"` 走完正好完整。

**文法外的字符。** 词表里常有 pattern 从未出现的字符（数字、标点、特殊 token）。它们不可能被消费，第一步即死——不要把它们当成通配。

**为什么这里"合法"等价于"状态非空"。** 本题文法不含空分支，对语法树归纳可证每个位置都能走到某个 `last` 位置，因此任何非空状态都存在补全后缀。若文法允许空分支或不可达分支，就必须额外做一次反向可达性（co-accessible）判定，否则会把死状态误判成合法。

## 对拍口径

- 参考实现：递归下降解析 → Glushkov 位置自动机（`nullable` / `first` / `follow` / `last`）。
- 独立 oracle：Thompson NFA（显式 ε 转移）+ 子集构造 + ε-闭包，两套结构不共享代码。
- 第三方见证：用 Python `re.fullmatch` 暴力枚举后缀（长度 ≤ 4），确认每个 `mask` 为真的 token 真的能补完、每个 `complete` 为真的 token 直接 `fullmatch`；`accept` 与 `re.fullmatch(pattern, prefix)` 逐条对照。
- 400 组随机 pattern / 词表 / prefix 上参考实现与 oracle 三元组完全一致，且全部 400 个模式里"所有位置可达终态"成立。

## 复杂度

- 解析与自动机构造 `O(|pattern|)`；每个 token 走 `O(|token|)` 次转移；每给一个 prefix 只需重放 `O(|prefix|)`，整体 `O(|pattern| + |prefix| + Σ|token|)`。

## 相关题目

- `inf-spec-decoding-rejection`：同样在候选 token 层面做取舍，但判据是接受率而不是文法。
- `inf-cache-aware-routing`：把"状态 → 预计算表"当缓存复用，也是同一类工程账。
