# KV Cache：KVzip 跨查询聚合的全局阈值裁剪

## 题目描述

KVzip（arXiv 2505.23416）在 prefill 之后只回答一个问题：整条上下文里**哪些 KV pair 可以丢**。它不看单条 query，而是把 prefill 期间**所有 query 位置**的注意力摊开，给每个 KV pair 取一个"历史最大"分数，再跨层跨头算出一个**全局阈值**，低于阈值的整对驱逐。

分数是这样来的（`attention/score.py::KVScore._get_score`）：

- GQA 下每个 KV 头对应 \(G\) 个 query 头，注意力权重在 **group 轴与 query 轴上一并取 max**——上游就一句 `attn_weights.amax(dim=(-3, -2))`，注释写着 `max over group, q`；
- 于是每层每个 KV 头得到一条长度 \(N\) 的分数向量，把 \(L\) 层全部 flatten 成 \(T=L\cdot H_{\text{kv}}\cdot N\) 个分数；
- 降序排序，取**下标** \(n=\max(\lfloor T\cdot\text{ratio}\rfloor-1,\ 0)\) 处的值为阈值，**严格大于**阈值的留（`attention/score.py::KVScore._threshold`）。

\(n\) 是**下标**不是个数：阈值自己一定被丢掉，所以 ratio 只会"少留"。`ratio >= 1` 时上游直接全留、阈值归 0。`--level pair-uniform` 换成每层每头各自 topk（`_threshold_uniform`，\(k=\lfloor N\cdot\text{ratio}\rfloor\)）。

## 任务

实现 `solve(attn_weights, ratio=0.3, level="pair")`：

- `attn_weights`：`(L, H_kv, G, Q, N)` 的 float32/float64，**已经 softmax 过、且已经切掉 sink 前缀**的权重（sink 保留与因果 mask 不在此题范围内）；
- `ratio`：实数且 `>= 0`，默认 0.3；
- `level`：`"pair"`（全局排序取阈值）或 `"pair-uniform"`（每头各自 topk）。

返回 6 键字典（张量按输入 dtype，整数按 int64）：

- `scores`：`(L, H_kv, N)`，group 轴与 query 轴上的 amax；
- `valid`：`(L, H_kv, N)` bool，保留掩码；
- `keep_count`：`(L, H_kv)` int64，每头保留条数；
- `threshold`：Python float 全局阈值，`pair-uniform` 与 `ratio >= 1` 两支为 0.0；
- `threshold_index`：0 维 int64，即 \(n\)，同两支为 0；
- `keep_ratio`：Python float，真实保留比例 `valid.float().mean()`。

同一 (layer, head) 内并列时取**更小的 key 下标**（上游 `torch.topk` 不定义同分顺序，本题钉死）。非法输入抛 `ValueError`：非五维、dtype 不是 float32/float64、任一维为 0、`ratio` 不是实数或为负、`level` 不认识。

## 示例

```
attn_weights: L=1, H_kv=2, G=1, Q=2, N=2   -> T = 1*2*2 = 4
head 0: query 0 [0.7, 0.3], query 1 [0.4, 0.6]   -> scores[0] = [0.7, 0.6]
head 1: query 0 [0.2, 0.8], query 1 [0.5, 0.5]   -> scores[1] = [0.5, 0.8]

ratio = 0.5, level = "pair"
flat       = [0.7, 0.6, 0.5, 0.8]
descending = [0.8, 0.7, 0.6, 0.5]
n          = max(int(4*0.5) - 1, 0) = 1        # index, not count
threshold  = descending[1] = 0.7
valid      = scores > 0.7 = [[F, F], [F, T]]   # 0.7 itself is dropped
keep_count = [[0, 1]]        keep_ratio = 0.25 # not 0.5

ratio = 0.5, level = "pair-uniform"            # k = int(N*ratio) = 1 per head
valid      = [[T, F], [F, T]]
keep_count = [[1, 1]]        keep_ratio = 0.5
threshold  = 0.0             threshold_index = 0
```

同一个 ratio，`pair` 只留 1 条（25%），`pair-uniform` 留 2 条（50%）：两种口径的真实保留率并不相同。

## 约束

- **聚合是 max**：`amax` 同时跨 group 与 query，不是 mean 也不是 sum。mean 会把"某条 query 特别关注它"的证据抹平，而长上下文里正是这种尖峰决定去留。
- **阈值是下标**：`n = max(int(T*ratio) - 1, 0)`。`-1` 让它比"个数"少一个；`max(...,0)` 是上游对退化情形的兜底——`T*ratio < 1` 时阈值等于最大值，**一条都不留**。
- **严格大于**：`valid = scores > threshold`。全部同分时阈值就是该分数本身，结果为空集。
- **全局 vs 每头**：`pair` 把 \(L\cdot H_{\text{kv}}\cdot N\) 个分数一起排序，弱头可能一条不剩；`pair-uniform` 每头各留 \(\lfloor N\cdot\text{ratio}\rfloor\) 个，头之间不抢预算。
- **不要加 value 范数**：上游打分路径里没有任何 `linalg.norm`，分数只由注意力权重聚合而来。
- **内部 float64**：上游 softmax 的注释写着 `not fp32`、分数沿用该 dtype；本题统一在 float64 里比较，避免 float32 下阈值附近的并列抖动。
