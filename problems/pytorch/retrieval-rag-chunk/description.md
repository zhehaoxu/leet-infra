# 检索：RAG 重叠滑窗分块

## 题目描述

RAG（检索增强生成）在建库前要把长文档切成可嵌入的片段（chunk）。切太长稀释语义、切太短丢失上下文，于是主流做法是**带重叠的滑窗分块**（LangChain `CharacterTextSplitter`/`RecursiveCharacterTextSplitter` 的 `chunk_overlap`、LlamaIndex 的窗口分块）：窗口大小 `chunk_size`、相邻块之间共享 `overlap` 个 token，步长 `step = chunk_size - overlap`。重叠保证跨块边界的语义不被子序列切断。本题把一篇「已经是 token 序列」的文档做重叠切块，返回每块覆盖的 **token 下标矩阵**（末尾不足处用 `-1` 填充）。

## 规则

实现 `solve(num_tokens, chunk_size, overlap)`。整数入参：文档共 `num_tokens` 个 token（下标 \(0\dots \text{num\_tokens}-1\)），窗口 `chunk_size`，重叠 `overlap`（\(0\le\) `overlap` \(<\) `chunk_size`）。步长 \(\text{step}=\text{chunk\_size}-\text{overlap}\)。块起点从 0 开始、每隔 step 一个，只要起点 \(<\) `num_tokens` 就成块。第 \(b\) 块第 \(j\) 列的 token 下标为 \(\text{start}_b + j\)；若 \(\ge\) `num_tokens` 则该位置填 `-1`。返回 int64 张量，形状 \((\text{块数},\ \text{chunk\_size})\)。

## 示例

\(num\_tokens=7, chunk\_size=3, overlap=1\)（step=2，起点 0,2,4,6）：

```
块0: 0 1 2
块2: 2 3 4
块4: 4 5 6
块6: 6 -1 -1     (起点6<7 成块，但 7、8 越界 -> -1)
out = [[0,1,2],[2,3,4],[4,5,6],[6,-1,-1]]   shape = (4, 3)
```

## 约束

- \(0\le\) `overlap` \(<\) `chunk_size\)，故 `step ≥ 1`（避免不退进度）。
- 起点 \(<\) `num_tokens` 即成一块——哪怕该块大部分越界（如上例起点 6）。
- 越界位置填 `-1`（哨兵），不是回绕、不是截断到不满 `chunk_size` 宽。
- 返回 int64 下标矩阵；纯切块，不涉及任何文本内容或嵌入。
