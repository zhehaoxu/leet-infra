# KV Cache：Page-EntroKV 的分组 Rényi 熵与页级保留

## 题目描述

Page-EntroKV（arXiv 2610.03135）针对 GQA 的一个麻烦：多个 query head 共享同一块物理 KV 缓冲，如果每个 query head 各自 top-k，引擎要保留它们选择的并集，组内越分歧缓存越肥；而朴素均值池化又会把专家 head 的偏好稀释到 `1/r`。

它的解法是**先在物理组内把 head 池化掉，再在页的粒度上做 top-k**：每个 query head 的 sink 隔离 Rényi 熵当作「这个 head 有多集中」的代理，组内 `softmax(-熵 / tau_g)` 加权，把 `r` 个 head 的注意力合成一条曲线，按 `block_size` 分页取 `max`，最后每页打分取 top-k。

上游唯一可运行的实现是 `page_entrokv/core.py::PageEntroKVGroupPooler`（`forward` 只有 19 行）；仓库其余部分（`kernels/`、`baselines/`、`evaluation/run_guardrails.py`）都自标 `PENDING`。本题只考这个 pooler 的完整账。

对每个 batch `b` 和每个物理 KV head `g`（共 `H_KV` 个），组内 `r = H_Q // H_KV` 个 query head 按连续顺序分组：`group(g) = [g*r, g*r + r)`。三个中间量：

- **熵**：先把该行**前 `min(sink_size, T)` 个位置置 0**（sink 隔离），再对**剩下的位置**重新归一化（分母下限 `1e-12`），然后算 Rényi 熵；`alpha == 1.0` 退化为 Shannon 熵，否则 `log(sum p^alpha + 1e-12) / (1 - alpha)`。
- **组权重**：`softmax(-ent / tau_g)`，只在这组的 `r` 个 head 上做。
- **页分数**：池化曲线 `A_pooled[g][t] = sum_i w_i * attn[b][g*r + i][t]`，**用的是原始注意力**（sink 位置不置零、不重归一化）；零填充到 `block_size` 的整数倍后每页取 `max`。

最后每个 `(b, g)` 在页分数上取 top-`num_retained_blocks` 页，输出页掩码；token 掩码按 `t // block_size` 展开。

## 任务

实现 `solve(attn, num_retained_blocks, num_query_heads, num_kv_heads, block_size=16, tau_g=0.5, alpha=2.0, sink_size=4)`：

- `attn`：float 三维张量 `[B, H_Q, T]`，三轴都 `>= 1`，**必须非负**（「零填充不改变页 max」依赖它）。数值本身参与加权池化和页内取 `max`，不必是概率、不必归一化。
- `num_query_heads` 必须等于 `H_Q`；`num_kv_heads >= 1` 且整除 `H_Q`。
- `block_size` 是 `>= 1` 的整数，`sink_size` 是 `>= 0` 的整数；`sink_size > T` 时按 `T` 截断（整行都是 sink）。
- `tau_g > 0`、`alpha > 0`；`num_retained_blocks` 是整数且 `0 <= num_retained_blocks <= nb`，其中 `nb = ceil(T / block_size)`。

按上面的定义对每个 `(b, g)` 独立算，返回 5 键字典：

- `entropies`：float `[B, H_Q]`
- `group_weights`：float `[B, H_KV, r]`
- `page_scores`：float `[B, H_KV, nb]`
- `retained_pages`：int64 `[B, H_KV, nb]`，取值 0/1
- `retained_tokens`：int64 `[B, H_KV, T]`，取值 0/1

`attn` 不是浮点张量 / 维度不是 3 / 含负值、`num_retained_blocks` / `num_query_heads` / `num_kv_heads` / `block_size` / `sink_size` 任一不是 `int`（`bool` 也算不合法）或 `num_retained_blocks < 0` / `sink_size < 0` / 其余三个 `< 1`、`num_query_heads != H_Q`、`H_Q` 不能被 `num_kv_heads` 整除、`num_retained_blocks > nb`、`tau_g` / `alpha` 不是实数或 `<= 0` 时抛 `ValueError`。

## 示例

```
B = 1, H_Q = 4, H_KV = 2, T = 5, block_size = 2, tau_g = 0.5, alpha = 2.0, sink_size = 1
num_retained_blocks = 1
attn[0], rows = query heads 0..3, columns = positions 0..4
  h=0   0.10 0.20 0.05 0.40 0.30
  h=1   0.30 0.10 0.20 0.05 0.15
  h=2   0.05 0.45 0.10 0.20 0.25
  h=3   0.20 0.05 0.35 0.10 0.30

nb = ceil(5 / 2) = 3; pages cover positions {0,1} {2,3} {4, pad}
entropies       = [1.1267, 1.2040, 1.1552, 1.0454]      # over positions 1..4 only
group_weights   = [[0.5386, 0.4614], [0.4453, 0.5547]]  # kv 0 = heads 0,1; kv 1 = heads 2,3
page_scores     = [[0.1923, 0.2385, 0.2308], [0.2281, 0.2387, 0.2777]]
retained_pages  = [[0, 1, 0], [0, 0, 1]]                # kv 0 keeps page 1, kv 1 keeps page 2
retained_tokens = [[0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]
```

```
tie break: B = 1, H_Q = 1, H_KV = 1, T = 4, block_size = 2, tau_g = 0.5, alpha = 2.0
num_retained_blocks = 1, sink_size = 0, attn[0][0] = [0.5, 0.5, 0.5, 0.5]

entropies       = [1.3863]        # log 4
group_weights   = [[1.0]]
page_scores     = [[0.5, 0.5]]    # tie -> lower page index wins
retained_pages  = [[1, 0]]
retained_tokens = [[1, 1, 0, 0]]
```

## 约束

- **熵和池化用的不是同一行数据**：sink 位置只在算熵时置零并重新归一化，池化回的是原始 `attn`。把 sink 也池化掉是最常见的错。
- **`alpha == 1.0` 是单独分支**：走 Shannon 熵；`alpha` 传 `1`（整数）也走这里。
- **`clamp(min=1e-12)` 让全 sink 行也有有限熵**：整行都是 sink 时归一化后全是 0，`0**alpha = 0`、`log(0 + 1e-12)` 仍是有限值，不出 NaN。
- **零填充要在页 max 之前、`nb` 用 `ceil`**：`attn >= 0` 时池化值非负，补的 0 不会顶替页内最大值；但页分数仍按「含填充的整页」取 max。
- **top-k 平手按页号小的在前**：上游用 `torch.topk`，并列顺序不属于公开契约，这条是本题自行定死的口径。
- **`num_retained_blocks` 可以取 0 或 `nb`**：0 时页掩码全 0，`nb` 时全 1。
- 判分逐元素比：3 个 float 键容差 `1e-5`；2 个掩码键是 int64 的 0/1（不是 bool）。`B <= 4`、`H_Q <= 16`、`H_KV <= 8`、`T <= 512`。
