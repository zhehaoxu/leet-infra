# 解题说明 · ZSMerge 零丢弃加权合并

## 一句话

缓存满了不驱逐：在非 dense 区挤掉分数最低的槽位 \(i\)，按 `counts` 加权并进 dense 区的最近邻 \(j\)，再让新 token 占住 \(i\)。

## 三个必须写对的点

**1. K 和 V 用同一套权重一起并。** 上游 `cache_merge` 里

```
w_i = pos_weight[i]         # = counts[i]
w_j = pos_weight[j]         # = counts[j]
k[j] = (w_i * k[i] + w_j * k[j]) / (w_i + w_j)
v[j] = (w_i * v[i] + w_j * v[j]) / (w_i + w_j)
```

`w_i / w_j` 只算一次，K 与 V 共用。如果给 V 另算一套权重（例如按值范数加权），或者只并 K 不并 V，注意力分布和值的内容就会互相错位——这是一道题里最容易「看起来对」的错。

**2. 计数是继承的，不是加一。** 合并后 `counts[j] = w_i + w_j`，`counts[i] = 1`（新 token 只代表它自己）。若写成 `counts[j] += 1`，一个已经并过 100 次的槽位与只并过 1 次的槽位权重相同，后续合并里新 token 会被反复放大，历史 token 被稀释到消失。

**3. dense 区只约束 \(j\)，不约束 \(i\)。** \(i\) 是**非 dense 区**里 `scores` 最小的槽位，\(j\) 是 **dense 区**里离 \(k_i\) 最近的槽位。两者互斥：dense 槽位分数再低也不能被牺牲，非 dense 槽位再近也不能当锚点。测试里专门放了「dense 区里某个 key 离 \(k_i\) 更近」的局面来验证这条。

## 判分点

- `merged_slots`：每 head 独立的 `[i, j]`，`i` 在非 dense、`j` 在 dense；无合并时 \(j=-1\)。
- `counts`：\(j\) 处相加、\(i\) 处为 1，其余不变。
- `scores`：\(j\) 处**相加**（上游 `score_update="sum"`），不是取 max，也不是取平均。
- `metric`：`l2` 用平方距离、`dot_product` 用负点积，两者在示例数据上选出的 \(j\) 不同。
- 同分：\(i\) 与 \(j\) 都取更小的槽位号（上游 `argmin` 不定义同分顺序）。
- 不改输入、内部 float64、输出回输入 dtype。

## 边界

- **dense 全 False**：上游走 `cache_dense == 0` 的独立分支——不找 \(j\)、不合并，直接把全局最低分槽位交给新 token，题面用 \(j=-1\) 标记；**dense 全 True** 时没有槽位可挤，抛 `ValueError`。
- **\(i\neq j\)**：靠 dense / 非 dense 的分区天然成立，所以「先把 \(j\) 写回、再写 \(i\)」不会互相覆盖；反过来先写 \(i\) 再读 \(k_i\) 就会读到新 token 的 key。
- **上游的分数衰减**：`scores` 每步先乘 `shrink_factor`（0.98）再加新分数，因而分数是带遗忘的累计。本题把 `scores` 当显式输入，衰减发生在调用方，`solve` 只负责相加——测试因此能对分数做精确断言。

## 复杂度

每步 \(O(THD)\) 的归约、\(O(|D_{\text{dense}}| H D)\) 的最近邻比较和 \(O(THD)\) 的写入量；相对全量 attention 的 \(O(L^2 D)\) 只是常数级维护成本。

## 出处

`SusCom-Lab/ZSMerge`（arXiv 2503.10714，编号取自论文页——仓库里没有 README 可对照）：

- `mergekv/mergekv.py` L354-428 `cache_merge`：整体路径；
- L362-374 `cache_dense == 0` 分支：不合并、直接顶替；
- L377-378 `mrg_i = argmin(pos_score[..., cache_dense:]) + cache_dense`；
- L384-391：`dot_product` 取负 `einsum`、`l2` 取平方距离，`j` 在 dense 前缀内取 `argmin`；
- L395-404：`w_ij = w_i + w_j`，K 与 V 共用这套权重；
- L411-412：`s_ij = s_i + s_j`；L418-426：先 scatter 合并结果到 \(j\)，再把新 token 写到 \(i\)；
- L27-41 与 L48-80：`cache_budget / cache_tail / cache_dense` 的 `int()` 逐级截断换算出 dense 前缀长度；
- 同文件 `topk_split_new` 是配套的槽位切分入口，本题把切分结果抽象成 `dense` 掩码入参（不单独判）。

**本题自己的补充**：同分取更小槽位号；dense 掩码与 `scores` / `counts` 走显式入参而不是内部状态；`cache_dense == 0` 用 \(j=-1\) 显式标记。

## 相关题目

- `kv-h2o-heavy-hitters`：同样按分数挑受害者，但直接丢弃、不合并，也不继承计数。
- `kv-merging-cos-nearest`：按 key 的余弦相似度把被驱逐 token 的 value 折进最近邻，槽位数量减少；本题按平方 L2 / 负点积在 **dense 区**找锚点，K 与 V 用同一套计数权重一起并，槽位总数不变。
- `kv-wkvd-least-squares-merge`：用最小二乘把被驱逐的 value 投影到存活 value 张成的子空间；本题是一对一的计数加权平均，不做回归求解。
- `kv-morphkv-island-fusion`：把连续的一段位置整体融成一个槽位，是「多对一」的加权平均；本题是「一对一」。
- `kv-snapkv-pooling`：在观测窗内做池化压缩，颗粒是窗口而非单个槽位。
