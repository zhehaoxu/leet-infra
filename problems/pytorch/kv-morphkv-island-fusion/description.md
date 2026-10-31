# KV Cache：MorphKV 常数缓存的岛屿融合

## 题目描述

**MorphKV**（[arXiv:2503.00979](https://arxiv.org/abs/2503.00979)，仓库 [ghadiaravi13/MorphKV](https://github.com/ghadiaravi13/MorphKV)）：把 KV cache 压到**常数大小**。与「逐 token 打分、保留 top-K」不同，MorphKV 的层级（hierarchical）路径先把最重要的 token 留成**单例**，剩下的 token 按「重要位」切分成若干段**连续岛屿（island）**，每个岛用 importance 加权平均**融成 1 个槽位**，再按岛的平均重要性取 top-K 个岛。

最终布局固定为 `[融合岛 | 重要单例 | 最近窗口]`——融合岛提供「远距离摘要」，重要单例保留精确的尖峰 token，窗口保生成不失忆。

## 规则

实现 `solve(attn, keys, values, max_capacity, window_size, imp_budget=0.5, fuse_temperature=1.0, value_norms=None)`。

**输入**

- `attn`：$(H, Q, P)$，窗口内 `Q` 个 query 对前 `P` 个 key 的**未 softmax logits**（每个 KV head 一份）。
- `keys` / `values`：$(H, P + W, D)$，**post-RoPE** 的 KV cache；最后 `W = window_size` 个是窗口，不参与压缩。
- `max_capacity`：压缩后的总槽位预算；`window_size`：`W`；`imp_budget`：重要单例占比。
- `fuse_temperature`：重要性 softmax 的温度；`value_norms`：$(H, P)$ 的 value 范数，省略时取 `values[:, :P, :].norm(dim=-1)`。

**步骤**

1. 重要性：`importance = softmax(attn / fuse_temperature, dim=-1).mean(dim=-2) * value_norms`，形状 $(H, P)$。
2. 预算：`n_budget = int(max_capacity * imp_budget)`，`n_fused = max_capacity - window_size - n_budget`（必须 $\ge 0$），`n_important = min(n_budget, P)`。
3. 单例：按重要性**降序**取前 `n_important` 个位置（同分取更小下标），返回时改为**位置升序**。
4. 岛屿：past 位置 $p$ 的岛号 = $p$ 左侧重要 token 的个数（不含 $p$；范围 $[0, n_{important}]$）。连续的非重要 token 共享岛号；相邻重要 token 会让某些岛号**空洞**。
5. 融合：岛内权重 $w_i = \mathrm{imp}_i / \sum_{j \in \text{island}} \mathrm{imp}_j$，`fused_key = Σ w_i · k_i`，`fused_value = Σ w_i · v_i`（逐 head、逐维度）。岛分数 = 岛内**平均**重要性；空岛分数记 $-\infty$。
6. 选岛：`K = min(n_fused, n_important + 1)`，按岛分数降序取 `K` 个（同分取更小岛号），再按**岛号升序**排列；不足 `n_fused` 的槽位补零。
7. 布局：`[融合岛 (n_fused) | 重要单例 (n_important) | 窗口 (W)]`，窗口原样搬运。

**返回** `dict`：

| 键 | 形状 / dtype | 含义 |
| --- | --- | --- |
| `keys` | $(H, n_{fused}+n_{important}+W, D)$，同输入 dtype | 压缩后 K |
| `values` | 同上 | 压缩后 V |
| `important_idx` | $(H, n_{important})$，int64 | 单例的原始位置，升序 |
| `island_idx` | $(H, n_{fused})$，int64 | 每槽的岛号，补位槽为 `-1` |
| `island_counts` | $(H, n_{fused})$，int64 | 每槽吃进的 token 数，补位槽为 `0` |

## 示例

```
H=1, Q=2, P=6, W=2, D=2, max_capacity=5, imp_budget=0.5, fuse_temperature=1.0
attn[0] = [[0.0, 0.7, 2.0, -1.0, 1.0, -2.0],
           [0.0, 0.7, 2.0, -1.0, 1.0, -2.0]]
keys[0] = past [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11]]
        + window [[12,13], [14,15]]
values = 10 * keys，value_norms = [1, 1, 1, 1, 1, 1]

n_budget = int(5*0.5) = 2，n_fused = 5-2-2 = 1，n_important = 2
importance = [0.073398, 0.147806, 0.542344, 0.027002, 0.199517, 0.009933]
important_idx = [2, 4]
islands: id0 = {0,1}（均分 0.110602）> id1 = {3}（0.027002）> id2 = {5}（0.009933）
→ 选 id0，权重 [0.331812, 0.668188]
fused_key   = 0.331812*[0,1] + 0.668188*[2,3] = [1.336376, 2.336375]
fused_value = 10 * fused_key            = [13.363755, 23.363756]

keys    = [[1.336376, 2.336375], [4,5], [8,9], [12,13], [14,15]]
values  = [[13.363755, 23.363756], [40,50], [80,90], [120,130], [140,150]]
island_idx = [[0]]，island_counts = [[2]]，重要单例占第 1、2 槽，窗口占最后 2 槽
```

## 约束

- CPU float32 / float64，所有浮点输入（含 `value_norms`）必须同 dtype；输出沿用该 dtype，内部用 float64 累加。
- $P \ge 1$、$Q \ge 1$、$D \ge 1$、$W \ge 0$，且 `keys.shape[1] == P + W`。
- `max_capacity ≥ window_size + int(max_capacity * imp_budget)`，否则抛 `ValueError`；`0 ≤ imp_budget ≤ 1`，`fuse_temperature > 0`，`value_norms` 全为正。
- 不得就地修改任何输入。
