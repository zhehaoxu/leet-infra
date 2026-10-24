# Suffix / REST：最长后缀检索草稿

## 题目描述

不用训练 draft 模型也能投机：**检索式草稿**直接从历史生成/语料里"抄"续写。**REST**（Zhao et al., 2023, [arXiv:2311.08252](https://arxiv.org/abs/2311.08252)）在离线构建的**数据存储 (datastore)** 里，用当前上下文去检索匹配的后缀并取其后继作草稿；**Suffix Decoding**（Arora et al., 2024, [arXiv:2411.04975](https://arxiv.org/abs/2411.04975)）用后缀树把这一"最长后缀匹配"做到在线、动态深度。二者共用一个核心操作：**找 context 的最长后缀，使其作为 datastore 某条记录的连续子串出现，草稿就是匹配点后面紧跟的若干 token**。本题实现这个检索操作。

## 规则

入参：
- `datastore`：token 序列列表 `list[list[int]]`，每条是一个 entry。
- `context`：当前上下文 `list[int]`（含其最后若干 token）。
- `gamma`：最多草稿的 token 数。
- `min_match`：后缀匹配的最小长度门槛；短于此不算匹配。

在 datastore 中寻找 context 的一个**后缀** $s = (\text{context}_{-n},\dots,\text{context}_{-1})$（$n\ge \text{min\_match}$），使得 $s$ 作为某条 entry 的**连续子串**出现在位置 $i$（即 $\text{entry}[i:i+n]=s$）。草稿取 $[\text{entry}_{i+n},\dots,\text{entry}_{i+n+\gamma-1}]$（越界则截断）。

**选择与 tie-break**（保证唯一）：
1. 优先**最长**的后缀匹配（$n$ 最大）；
2. $n$ 相同时取 **entry 序号最小**（更早的 entry）；
3. entry、$n$ 都相同时取**匹配位置 $i$ 最小**（该 entry 中最早出现处）。

没有任何满足 `min_match` 的后缀匹配时，返回**空**草稿。返回 `torch.long` 一维张量。

- 与 Lookahead（题 179）用定长窗、逐步回退不同，这里是**变长最长后缀 + 一次性取 ≤gamma 个草稿**，不做逐 token 回退。
- "最长后缀优先"对应直觉：匹配得越长，抄来的续写越可信。

## 示例

```
datastore = [[1,2,3,4,5],
             [9,2,3,7,8],
             [2,3,6]]
context=[0,1,2,3], gamma=2, min_match=2

后缀由长到短：
 n=4 [0,1,2,3] -> 无
 n=3 [1,2,3]   -> entry0 在 i=0 命中，草稿 entry0[3:5] = [4,5]   （最长，锁定）
 n=2 [2,3]     -> 已锁更长，忽略
输出 = [4, 5]
```

## 约束

- CPU；返回 `torch.long` 一维（可为空），用 `torch.equal` 比较。
- 草稿长度 $\le \gamma$，且不超过匹配点在 entry 内的剩余长度。
- tie-break 三优先级严格按题面：更长 → 更早 entry → 更早位置。
