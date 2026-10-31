# KV Cache：FastKV 跨层 token 选择性传播

## 题目描述

**FastKV**（Jo et al., 2025, [arXiv:2502.01068](https://arxiv.org/abs/2502.01068)）观察到：KV cache 的压缩不必在每一层独立做。它在某一层（TSP 层，Token-Selective Propagation）先用跨 KV head 汇总的分数选出一个 token 子集，**后续层只携带这个子集**；由于后续层的序列变短，保留率可以从 `retain_rate` 放宽到 `retain_rate / tsp_rate`，使绝对预算大致不变。每个 head 内部仍然是 SnapKV 式的「保留分数最高的若干历史 token + 最近 `window_size` 个 token」。

本题只考这条**跨层选择与传播**链路：`attn_caches` 已是逐层、逐 token 的分数，`solve` 负责算预算、逐 head 选 token、在 TSP 层选子集并切出对应的 `hidden_states` / `position_ids`。

## 规则

实现 `solve(attn_caches, q_len, tsp_idx, hidden_states, position_ids, window_size=8, retain_rate=0.1, tsp_rate=0.2)`：

### 入参

- `attn_caches`：长度 \(L\) 的非空 `list` / `tuple`，第 \(l\) 项是形状 \((T_l, H_{\text{kv}})\) 的 float32 / float64 张量，表示该层每个 KV head 对每个 token 的重要性分数。所有层同 dtype、同 \(H_{\text{kv}}\)；第 0 层长度必须等于 `q_len`。
- `q_len`：`int >= 1`，当前段的 token 数。
- `tsp_idx`：`int`，TSP 层下标，\(0 \le \text{tsp\_idx} < L\)。
- `hidden_states`：\((q_len, D)\)，与 `attn_caches` 同 dtype；TSP 选中哪些 token 就切哪些行。
- `position_ids`：\((q_len,)\) int64，原始绝对位置。
- `window_size`：`int >= 1`，每层强制保留的最近 token 数，默认 8。
- `retain_rate`、`tsp_rate`：`int` / `float`，落在 \((0,1]\)，默认 0.1 / 0.2。

先算
\[
\text{tsp\_length}=\lfloor q\_len \times \text{tsp\_rate}\rfloor,\qquad
\text{gating}=[q\_len > \text{tsp\_length}].
\]
仅当 `gating` 为真时才做跨层传播。若 `gating` 且 \(\text{tsp\_length}\le \text{window\_size}\)，抛 `ValueError`。

### 计算

1. **各层长度**：
   - \(l \le \text{tsp\_idx}\) 或 `gating` 为假：\(T_l = q\_len\)；
   - \(l > \text{tsp\_idx}\) 且 `gating` 为真：\(T_l = \text{tsp\_length}\)。
   输入的第 \(l\) 层 cache 必须正好是这个长度，否则抛 `ValueError`。
2. **各层预算**：
   \[
   \text{rate}_l=
   \begin{cases}
   \text{retain\_rate}, & l \le \text{tsp\_idx},\\
   \text{retain\_rate}/\text{tsp\_rate}, & l > \text{tsp\_idx},
   \end{cases}
   \qquad
   \text{capacity}_l=\lfloor T_l \times \text{rate}_l\rfloor .
   \]
   若 \(\text{capacity}_l \le \text{window\_size}\)，抛 `ValueError`。
3. **逐层、逐 head 保留**：
   - 若 \(T_l < \text{capacity}_l\)：容量比 cache 还大，**原样保留**全部下标 `0..T_l-1`。
   - 否则把前 \(T_l-\text{window\_size}\) 个 token 按分数**降序**排序（stable，同分取更小下标），取前 \(\text{capacity}_l-\text{window\_size}\) 个，再与尾部窗口 `T_l-window_size .. T_l-1` 合并，最后升序排列。
4. **TSP 选择**（仅 `l == tsp_idx` 且 `gating` 为真）：
   - 把该层前 \(T_l-\text{window\_size}\) 个 token 的分数**跨 KV head 求和**；
   - 取最高的 \(\text{tsp\_length}-\text{window\_size}\) 个（stable，同分取更小下标）；
   - 与尾部窗口 `q_len-window_size .. q_len-1` 合并并升序排列，得到 `tsp_indices`。
5. `hidden_out = hidden_states[tsp_indices]`，`position_out = position_ids[tsp_indices]`；位置**保持原始绝对下标**，不重新编号。

### 返回值

返回 5 键字典：

- `tsp_indices`：`None`（不传播）或 \(( \text{tsp\_length}, )\) int64，升序；
- `hidden_out`：`None` 或 \(( \text{tsp\_length}, D)\)，dtype 与 `hidden_states` 相同；
- `position_out`：`None` 或 \(( \text{tsp\_length}, )\) int64，原始绝对位置；
- `capacities`：\((L,)\) int64，逐层保留预算；
- `kept`：长度 \(L\) 的 list，第 \(l\) 项是 \((H_{\text{kv}}, n_l)\) int64，每行升序、无重复；\(n_l=\text{capacity}_l\)，但当 \(T_l<\text{capacity}_l\) 时为 \(T_l\)。

不得就地修改任何输入。

## 示例

```
q_len=12, L=3, tsp_idx=1, H_kv=1, window_size=2,
retain_rate=0.5, tsp_rate=0.25

attn_caches[0] = [[0.1], [0.2], [0.9], [0.3], [0.4], [0.8],
                  [0.5], [0.7], [0.6], [0.85], [0.0], [0.0]]     # (12,1)
attn_caches[1] = [[0.1], [0.5], [0.2], [0.3], [0.95], [0.4],
                  [0.6], [0.2], [0.7], [0.1], [0.0], [0.0]]      # (12,1)
attn_caches[2] = [[0.3], [0.2], [0.1]]                            # (3,1)
hidden_states  = [[0,100], [1,101], ..., [11,111]]                # (12,2)
position_ids   = [100, 101, ..., 111]                             # (12,)

tsp_indices  = [4, 10, 11]
hidden_out   = [[4.0, 104.0], [10.0, 110.0], [11.0, 111.0]]
position_out = [104, 110, 111]
capacities   = [6, 6, 6]

kept[0] = [[2, 5, 7, 9, 10, 11]]
kept[1] = [[1, 4, 6, 8, 10, 11]]
kept[2] = [[0, 1, 2]]
```

`tsp_length = floor(12 × 0.25) = 3`，`gating = 12 > 3`，所以第 2 层只有 3 个 token。第 0、1 层长度 12、保留率 0.5 → 容量 6；第 2 层长度 3、保留率 \(0.5/0.25=2\) → 容量 6，容量大于 cache，原样保留。第 0 层从历史 10 个 token 里取分数最高的 4 个（下标 2、5、7、9），再加窗口 `[10,11]`；第 1 层同理得到 `[1,4,6,8]`。TSP 在第 1 层，跨 KV head 求和后取 1 个历史 token（下标 4，分数 0.95），再加窗口 `[10,11]`，得到 `[4,10,11]`。

## 约束

- CPU float32 / float64；`attn_caches` 每项必须是同 dtype 的二维张量，且 KV head 数一致。
- `q_len >= 1`；`0 <= tsp_idx < len(attn_caches)`；`window_size >= 1`；`retain_rate, tsp_rate ∈ (0,1]`。
- `hidden_states` 必须是 \((q_len, D)\) 且与 cache 同 dtype；`position_ids` 必须是长度 `q_len` 的一维 int64 张量；第 0 层 cache 长度必须等于 `q_len`。
- `tsp_length = int(q_len * tsp_rate)` 向下取整；`gating = q_len > tsp_length`。`gating` 为真时要求 `tsp_length > window_size`。
- 每层 `capacity = int(length * rate)` 向下取整；`capacity` 必须严格大于 `window_size`；若 `length < capacity`，该层保留全部 token。
- 排序 stable，同分取更小下标；`kept` 每行升序、无重复；尾部 `window_size` 个 token 恒被保留。
- 不传播时 `tsp_indices` / `hidden_out` / `position_out` 都是 `None`；`capacities` 与 `kept` 始终返回。
- 非法形状、dtype、比例、预算（如 `tsp_rate=1.0` 导致不传播、`capacity <= window_size`、cache 长度与 gating 规则不符）抛 `ValueError`。
