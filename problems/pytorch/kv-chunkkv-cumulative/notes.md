# 解题说明 · ChunkKV

## 语义块：为什么整块保留更好

逐 token 打分（Knorm/SnapKV/TOVA 等）会挑出散落在上下文里的"高分点"，但语言里**一段话的语义往往跨越多个连续 token**：只保留句子里的三五个 token，反而把上下文切碎、破坏连贯性。ChunkKV 的观察是——把连续 $c$ 个 token 视作一个语义块，按块的重要性**整块取舍**，保留下来的 KV 更像"原文的缩略本"。

## 块级分数的两步聚合

$$cs_i = \frac{1}{|\text{seg}_i|}\sum_{t\in \text{seg}_i}\ \sum_{h} \text{scores}_{h,t}$$

先**跨 head 求和**（把所有 head 的意见合起来），再**块内平均**（除以块长，让短尾块和满块可比）。这与 ChunkPress 的"块内各自 top-k"是两条完全不同的路：ChunkPress 保 token、ChunkKV 保整块。

## 输出是升序、且跨 head 共享

因为块选择已经把 head 维聚合掉了，所有 head 用的是同一批块，结果天然是一维的位置列表。最后 `.sort()` 成升序，保证 gather 时按原文顺序读取（KV 的相对顺序不能乱）。

## 保底与边界

- `max(1, int(...))` 保证至少留一个块，`compression_ratio→1` 也不会清空。
- $L\le c$ 时只有一个块 → 全保留。本题把"完整块 + 尾块"两种情况统一成 `ceil(L/c)` 段、每段按实际长度平均，与官方分支写法等价但更简洁。

## 与 PyramidKV / AdaKV 的正交性

PyramidKV/AdaKV 管的是"每层/每头分多少预算"，ChunkKV 管的是"在同一预算下保留的最小单位是块还是 token"。可以叠加：先用某 scorer 出分，再用 ChunkKV 聚合到块级选择。
