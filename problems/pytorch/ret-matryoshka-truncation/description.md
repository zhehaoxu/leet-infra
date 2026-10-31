# Matryoshka 嵌入截断：低维排名保持账

## 题目描述

Matryoshka Representation Learning（MRL, arXiv:2205.13147）训练出的嵌入有个套娃性质：
**前 d 维本身就是可用的低维嵌入**——检索时可以把 1024 维截到 64 维先做粗排，再回全维精排，
存储与计算都省一个量级。这件事成立的前提是「截断 + **重新归一化** + 余弦相似度」后，
低维排名要尽量贴近全维排名。本题把这个检索账钉死成可判题的确定性形式：给定 query 与
文档库嵌入，对一串截断维度逐档算相似度、排名与全维 top-k 的重叠账。

> 为保证判题**确定性**：嵌入全部显式入参（不加载任何模型）；归一化前先转 float64；
> 零向量归一化为全 0（模长下限 1e-12）；并列按 index 升序。

## 规则

- 实现 `solve(query_emb, doc_embs, dims, top_k)`：
  - `query_emb: torch.Tensor [D]`，`doc_embs: torch.Tensor [N, D]`；
  - `dims: list[int]`，截断维度档位（每档 `1 ≤ d ≤ D`，可重复、顺序即返回顺序）；
  - `top_k: int ≥ 1`（`> N` 报 `ValueError`）。
- 对每个维度 `d`（含全维 `D` 的 `full` 档）：
  1. `q_d = normalize(query_emb[:d])`，`docs_d = normalize(doc_embs[:, :d])`——**截前 d 维
     后重新 L2 归一化**（这是 MRL 的核心约定，不归一化就退化成点积）；
  2. `sims = docs_d @ q_d`（余弦相似度）；
  3. `top =` 按 `(-sim, index)` 升序取前 `top_k` 个 index。
- `full` 档用 `d = D` 同法计算；每个截断档记
  `overlap_full = |top_d ∩ top_full|`（**集合**交集，与顺序无关）。
- 返回 dict：`full`（`sims` 列表 + `top` 列表）、`truncated`（`d` → 该档账本 dict，
  含 `sims` / `top` / `overlap_full`）、`n_docs`。
- 校验（`ValueError`）：`top_k` 非正整数（含 bool）或 `> N`；`d` 非整数（含 bool）、
  `< 1` 或 `> D`；`query_emb` 与 `doc_embs` 维度不匹配。

## 示例

```
输入：query_emb = [1, 1]（45°）
      doc_embs = [[2, 0.5], [1, 2], [-1, 1], [0.5, 0.1]]，dims = [1, 2]，top_k = 3
  full（d=2）：sims ≈ [0.857, 0.949, 0.0, 0.829] → top = [1, 0, 3]
  d=1：只留第一维，幅值被归一化抹掉，sims = [1, 1, -1, 1] → top = [0, 1, 3]
  d=1 的 overlap_full = 3（top 集合恰好相同）
输出：full.top = [1, 0, 3]，truncated[1].top = [0, 1, 3]，truncated[1].overlap_full = 3
```

## 约束

- float32 入参，内部转 float64 计算（比较容差 1e-6）
- `dims` 可以包含 `D` 本身（该档账本与 `full` 完全一致）；可以乱序、可重复
- 零向量（截断后该档全 0 的行）相似度记 0，参与并列排序
- 返回的 `sims` 用 Python float 列表（`item()` 后），顺序与 `doc_embs` 行序一致
