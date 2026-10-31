# KV Cache：ZSMerge 零丢弃加权合并

## 题目描述

KV cache 写满以后，主流做法是把一部分 token **驱逐**掉（H2O、SnapKV 都是如此）。ZSMerge（arXiv 2503.10714）反过来：一个槽位都不丢。每来一个新 token，先在**非 dense 区**里挑出分数最低的槽位 \(i\)，把它合并进 **dense 区**里离它最近的槽位 \(j\)，再让新 token 直接占用空出来的 \(i\)。

合并是**计数加权**的。每个槽位带一个 `counts`，记录它由多少个原始 token 平均而成。把 \(i\) 并进 \(j\) 时

\[
w_i=\text{counts}_i,\quad w_j=\text{counts}_j,\qquad
k_j\leftarrow\frac{w_i k_i+w_j k_j}{w_i+w_j},\quad
v_j\leftarrow\frac{w_i v_i+w_j v_j}{w_i+w_j}
\]

权重**继承**：\(\text{counts}_j\leftarrow w_i+w_j\)，\(\text{counts}_i\leftarrow 1\)。于是连续多次合并等价于对原始 token 做一次加权平均，后加入的 token 不会因为「只并过一次」而被重复计权。

分数同样累加：\(\text{scores}_j\leftarrow\text{scores}_i+\text{scores}_j\)，\(\text{scores}_i\leftarrow\) 新 token 的分数。

**dense 区**是缓存前沿的一段位置前缀（上游按 `cache_dense` 比例换算），\(j\) 只能在这里面选；非 dense 区才是可牺牲的槽位。上游用平方欧氏距离（`metric="l2"`）或负点积（`metric="dot_product"`）衡量「最近」。

## 任务

输入是一份已经带 `counts` / `scores` 的 KV cache 快照、一个 dense 掩码，以及一个待写入的新 token：

- `keys`、`values`：`(T, H, D)`，同一 dtype（float32 或 float64）；
- `scores`、`counts`：`(T, H)`，与上面同 dtype，`counts` 全为正；
- `dense`：`(T,)` 的 bool 掩码；上游恒为位置前缀，本题不校验前缀性；
- `new_key`、`new_value`：`(H, D)`；`new_score`：`(H,)`；
- `metric`：`"l2"`（默认）或 `"dot_product"`，只影响 \(j\) 的选取。

返回 5 键字典：

- `keys`、`values`：`(T, H, D)`，合并写回后的整份缓存；
- `counts`、`scores`：`(T, H)`，同上；
- `merged_slots`：`(H, 2)` int64，第 \(h\) 行是 `[i, j]`；没有合并时 \(j=-1\)。

四个张量输出的形状与 dtype 都跟输入一致；`merged_slots` 恒为 int64。内部按 float64 计算，输入不得就地修改。

## 示例

```
# T=5 slots, H=1 head, D=2
keys   = [[0, 0], [4, 4], [1, 0], [2, 2], [0, 1]]
values = 10 * keys
scores = [1, 2, -2, 5, 3]
counts = [1, 1, 3, 2, 1]
dense  = [1, 1, 0, 0, 0]
new_key, new_value, new_score = [3, 3], [30, 30], 1.5

i = 2                  # lowest score outside dense: min(-2, 5, 3)
j = 0                  # nearest dense key to k_i=[1, 0]: 1 vs 25 (squared L2)
w_i, w_j = 3, 1
keys[0]   = (3*[1, 0] + 1*[0, 0]) / 4   = [0.75, 0]
values[0] = (3*[10, 0] + 1*[0, 0]) / 4  = [7.5, 0]
counts[0] = 3 + 1 = 4
scores[0] = -2 + 1 = -1
keys[2], values[2], counts[2], scores[2] = [3, 3], [30, 30], 1, 1.5
merged_slots = [[2, 0]]
```

`l2` 与 `dot_product` 在这里正好会挑出不同的 \(j\)：\(k_i\) 到 `[4, 4]` 的负点积是 \(-4\)，比到 `[0, 0]` 的 \(0\) 更小，可惜 dense 区里 `[0, 0]` 与 `[4, 4]` 的负点积分别是 \(0\) 和 \(-16\)，所以 `dot_product` 下 \(j=1\)。见测试。

## 约束

- **\(i\) 只从非 dense 区选**：按 `scores` 取最小，同分取更小的槽位号。dense 区里分数再低也不能被牺牲。
- **\(j\) 只从 dense 区选**：`l2` 是**平方**欧氏距离（不开根，单调性一样），`dot_product` 是**负**点积；同分取更小的槽位号。
- **先写 \(j\) 再写 \(i\)**：\(i\) 在非 dense 区、\(j\) 在 dense 区，恒有 \(i\neq j\)，两次写入不冲突。
- **dense 全 False**：退化成「不合并」，\(j=-1\)，全局最低分槽位直接让给新 token（上游 `cache_dense == 0` 分支）。
- **dense 全 True**：没有可牺牲的槽位，抛 `ValueError`。
- **确定性**：上游用 `torch.argmin` / `torch.topk`，同分顺序不作保证；本题把同分一律钉成「取更小槽位号」。
