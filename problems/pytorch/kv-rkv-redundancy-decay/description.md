# KV Cache：R-KV 冗余感知打分与预算驱逐

## 题目描述

长上下文 decode 时 KV cache 只会越写越长，H2O、SnapKV 这类方法按注意力分数驱逐 token。麻烦在于注意力分数高的 token 常常成簇重复：一段复读的文本里每条都被 attend，却几乎不提供新信息。R-KV（arXiv 2505.24133）因此把驱逐分数拆成**注意力项减冗余项**，让「说得响但说的是废话」的 token 先走。

上游 `rkv/compression/r1_kv.py::_compute_scores` 与 `rkv/utils.py::cal_similarity` 的四步口径：

1. 只取最后 `window_size=8` 条 query，对前 `L-window_size` 条 key 做 softmax 后按 query 取均值，再沿 token 轴 `max_pool1d(kernel=7, padding=3, stride=1)`；
2. key 按行归一化后两两余弦、对角线清 0，`> threshold`（默认 0.5）记为冗余；每行命中的最小下标集合里取 `retain_direction`（默认 `"last"`）那一端当代表位，把该位相似度清 0，再对列取均值、`softmax`；
3. `final = mix_lambda * attn - (1 - mix_lambda) * redundancy`，`mix_lambda` 默认 **0.07**；
4. 每 head 取 `budget - window_size` 条最高分，最后 `window_size` 条恒留。

第 3 步的 0.07 是本题的重点：注意力项只是微调，**决定去留的主要是冗余项**——同一份预算下，重复的另一半会被换成一个独特但注意力稍低的 token。

## 任务

实现 `solve(k, q, v, budget=128, window_size=8, kernel_size=7, mix_lambda=0.07, threshold=0.5, retain_direction="last")`：

- `k`、`v`：`(L, H, D)`，同 dtype（float32/float64）；
- `q`：`(window_size, H, D)`，恰好是最后 `window_size` 个 query，**不打 causal mask**（每条 query 都能看全部 `N = L - window_size` 个 key）；
- `L >= budget > window_size >= 1`，`kernel_size` 为正奇数。

返回 6 键字典，`N = L - window_size`：

- `attn`：`(H, N)`，第 1 步的池化注意力分；
- `redundancy`：`(H, N)`，第 2 步的冗余分；
- `final`：`(H, N)`，`mix_lambda*attn - (1-mix_lambda)*redundancy`；
- `kept`：`(H, budget)` int64；前 `budget-window_size` 条是按 `final` 降序选出的下标（同分取小下标），后 `window_size` 条是 `N..L-1` 原序；
- `kept_keys`、`kept_values`：`(H, budget, D)`，按 `kept` 顺序 gather 出来的压缩缓存。

非法输入抛 `ValueError`：非三维、dtype 不一致或非浮点、head 数/head_dim 不一致、`L < budget`（上游此时直接原样返回、不打分）、`budget <= window_size`、`kernel_size` 非正奇数、`q` 行数不等于 `window_size`、`mix_lambda` 不在 `[0,1]`、`threshold` 不在 `[-1,1]`、`retain_direction` 不是 `last`/`first`。

## 示例

```
k = [[1,0], [1,0], [0,1], [1,1]]          # L=4 keys
q = [[1,0]]                               # window_size=1: only the last query
v = [[0,1], [2,3], [4,5], [6,7]]
budget=3, window_size=1, kernel_size=1, threshold=0.5, mix_lambda=0.07   # N=3, pick 2

cosine, diagonal cleared                    rep = max hit index in the row
        k0      k1      k2      k3
k0  [ 0.000   1.000   0.000   0.707 ]  -> 3     k0 and k1 are duplicates
k1  [ 1.000   0.000   0.000   0.707 ]  -> 3
k2  [ 0.000   0.000   0.000   0.707 ]  -> 3     k2 only has the window key k3
k3  [ 0.707   0.707   0.707   0.000 ]  -> 2     k3's rep is k2, not k3 itself

column mean after zeroing reps = [0.4268, 0.4268, 0.0000, 0.0000]
redundancy = softmax([0.4268, 0.4268, 0, 0])[:3] = [0.3026, 0.3026, 0.1975]
attn       = softmax([1, 1, 0] / sqrt(2))        = [0.4011, 0.4011, 0.1978]
final      = 0.07 * attn - 0.93 * redundancy     = [-0.2533, -0.2533, -0.1698]
kept       = [2, 0, 3]     # k2 is unique: it wins although k0/k1 have more attention
```

k0 与 k1 的注意力并列最高，却双双被冗余项扣到 -0.2533，预算里只剩 k2 和一个重复副本。注意 `redundancy` 三项之和只有 0.8025：softmax 是在**全长 4 条**上做的，窗口里那条 k3 也占分母。

## 约束

- **λ 只有 0.07 不是笔误**：`final = 0.07*attn - 0.93*redundancy`。把两项等权或只留注意力，选出来的集合就完全变了。
- **代表位取的是「命中下标行」的 max**：该行非命中位补 0，所以 `retain_direction="last"` 在没有命中时返回 0，`"first"` 的 min 恒为 0（等于每次都清 0 号列）——这是上游代码的真实行为，照抄。
- **softmax 在切片之前**：`redundancy` 是 `softmax(dim=-1)` 之后才切掉最后 `window_size` 列，行和小于 1，不是窗口上的条件分布。
- **threshold 是严格大于**：余弦恰好等于 0.5 不算冗余；对角线先清 0 再比较。
- **窗口硬保留**：`kept` 末尾恒为 `N..L-1`，前面按分数降序（同分取小下标，钉死 `torch.topk` 未定义的顺序）；`cache_len == budget` 时一条都不丢。
