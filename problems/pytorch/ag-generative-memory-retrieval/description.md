# 生成式智能体记忆检索：三因子加权排序账

## 题意

Generative Agents（斯坦福小镇）的记忆检索用三因子加权排序：**recency**（时间衰减）、**relevance**（与当前焦点的余弦相似度）、**importance**（poignancy 记忆强度）。本题转写 `new_retrieve` 的打分账：给定查询向量 `query_emb` 与记忆列表 `memories`（每条含 `emb` 与 `importance`），对三分量各自做 min-max 归一化到 [0,1]，再按权重 `gw` 加权求和，返回每条记忆的分数、三分量明细与按分数稳定降序取前 `k` 的下标。

## 求解

口径题面钉死（照官方 `retrieve.py`，含其 quirk）：

- `recency[i] = decay ** (i + 1)`，时间线 0 起：**最旧**的记忆指数最小、值最大（官方 `extract_recency` 原样行为，题面不修正）；
- `relevance[i] = cos(emb_i, query_emb)`：点积除以两个 L2 范数之积；
- 三分量各自 min-max 归一化 `(v - lo) / (hi - lo)`；**range == 0（含单节点）时全部置 0.5**（官方 `normalize_dict_floats` 取目标区间中点）；
- `score[i] = gw[0]*recency_n[i] + gw[1]*relevance_n[i] + gw[2]*importance_n[i]`，默认 `gw = (0.5, 3, 2)`；
- `top` 为按 score **稳定降序**（Python `sorted` 稳定性：总分并列时保持时间线顺序）取前 `k` 个下标。

返回 `{"scores", "top", "components": {"recency", "relevance", "importance"}}`。

## 数据规模

- 1 ≤ len(memories) ≤ 10^4，1 ≤ D = len(query_emb) ≤ 64，1 ≤ k ≤ 10^4。
- query_emb/memories 为空、维度不齐、emb 含非数、importance 非 数或 NaN/Inf、k 非正整数或 bool、decay 不在 (0,1]、gw 长度非 3 → `ValueError`。
