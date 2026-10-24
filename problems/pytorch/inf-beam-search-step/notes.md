# 解题说明 · Beam Search 单步

## 展平 + topk 的技巧

把 $(B,V)$ 的候选分按行优先展平成 $B\cdot V$，一次 `topk` 就能同时完成"跨 beam 比较"和"选 token"。还原时用整除/取模：`parent = f // V` 找回父 beam，`token = f % V` 找回新 token。这正是 HF `BeamSearchScorer` 里 `_beam_search` 的核心一行。

## 为什么需要回退

如果每条 beam 独立取自己最好的下一 token（贪心 per-beam），就会漏掉"A beam 的第二优其实全局更优"的情况。全局 top-k 允许某条父 beam 贡献 0 个、另一条贡献 2 个后继，从而动态把算力倾斜到更有希望的路径——这就是"回退/重排"的意义。

## 与 finished beam 的关系

真实系统里，命中 EOS 的序列会被挪到 finished 池并做**长度惩罚** $\frac{\text{score}}{((L+5)^\alpha / 5^\alpha)}$ 后参选。本题只考察"活 beam 的一步扩展与重排"这一最易错的骨架，把 finished/惩罚留作概念延伸，避免判题分支过多。

## 并列处理

`torch.topk` 对完全相等的分数不保证索引顺序；本题 fuzz 用连续 log 概率，几乎不会出现 float32 精确相等。oracle 里显式用 `(-score, flat_index)` 排序定义"并列取更小下标"，与题面约束一致。
