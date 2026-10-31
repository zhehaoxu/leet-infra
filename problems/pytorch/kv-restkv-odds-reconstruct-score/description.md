# KV Cache：ReST-KV 的赔率分与输出重建乘子

## 题目描述

ReST-KV（*Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing*，arXiv 2605.08840，ICLR 2026）把 SnapKV 的「末窗注意力按列求和」换成两项相乘的重要性分：

- **赔率项** `p / (1 − p)`：某个历史 key 被末窗 query 关注得越接近 1，赔率涨得越猛；
- **重建误差项** `‖W_o·v̄ − W_o·v_l‖₂`：这个 key 的 value 离「整段历史压缩出来的代表 value」越远越独特，越该留。

分数再做一次沿 token 轴的一维池化（空间平滑），然后 top-k 选历史段、末窗 `W` 个恒留。

上游把算分顺序做成两个可切换分支：`metric_mode="after"` 先把 `W` 行注意力聚合成 `p̄` 再算赔率与乘子，`"before"` 先逐 query 算完整分数再沿 query 轴聚合。两者**不等价**——`before` 的赔率是 `A(q,l)/(1−A(q,l))` 的平均，`after` 是平均值的赔率，且代表 value 一个是 `p̄` 加权、一个是逐 query 的 `A(q,·) V`。同分不同排序就是本题的主要判分信号。

## 任务

实现 `solve(query, key, value, wo, window_size=32, max_capacity_prompt=4096, kernel_size=5, pooling="avgpool", metric_mode="after", use_wo=True, use_norm=False, use_ema=True, alpha=0.3, tau=1.0)`：

- `query` / `key` / `value`：同形状同 dtype 的 float32 或 float64 `(B, H, S, D)`；上游调用点传进来的 K/V 已按 query head 展开，本题不再做 GQA 复制，`H` 恒为 query head 数。
- `wo`：与三者同 dtype 的二维 `(C_out, H * D)` 输出投影权重；`use_wo=False` 时不参与计算但仍校验。
- `window_size=W`、`max_capacity_prompt`、`kernel_size` 为正整数；`pooling ∈ {"avgpool","maxpool","none"}`、`metric_mode ∈ {"before","after"}`；`use_wo` / `use_norm` / `use_ema` 为布尔；`alpha ∈ [0,1]`、`tau ≥ 0`。记 `L = S − W`、`k = max_capacity_prompt − W`。

规则（`S == 7, W == 2, L == 5, k == 3` 时下列下标都按 `L` 段解释）：

1. `A = (query[..., -W:, :] @ key.transpose(2,3)) / sqrt(D)`，形状 `(B,H,W,S)`；给右下角 `W×W` 块**加**上严格上三角的 `finfo(A.dtype).min`（窗内第 `i` 行只能看前 `i+1` 个 key），再 `softmax(dim=-1, dtype=float32)` 后转回原 dtype——被掩位置的概率恒为 0。
2. `A_past = A[:, :, -W:, :-W]`，形状 `(B,H,W,L)`。
3. `metric_mode == "after"`：`p̄` 取 `A_past` 沿 query 轴的均值，或 `use_ema` 时的 EMA（`e_0 = A_past[0]`、`e_i = alpha·A_past[i] + (1−alpha)·e_{i−1}`，**只回吐最后一个状态**）；代表 value `v̄ = einsum("bhl,bhld->bhd", p̄, value[:, :, :-W])`（权重和 `< 1`，不是凸组合）；`score_l = p̄_l/(1−p̄_l)` 再乘 `‖W_o v̄ − W_o V_l‖₂ ** tau`（`use_wo`）或 `‖v̄ − V_l‖₂`（不开 `use_wo`，**这一支没有 tau**）。`W_o` 先 reshape 成 `(C_out,H,D)`，逐 head 在 `(h,d)` 两轴收缩到输出通道再取 L2。
4. `metric_mode == "before"`：`O = A @ value`（整段 softmax 乘**全量** value，形状 `(B,H,W,D)`）；`u = A/(1−A)` 形状 `(B,H,W,S)`；乘子逐 query 逐位置：`use_wo` 时 `‖W_o O_q − W_o V_l‖₂`、否则 `‖O_q − V_l‖₂`；`use_norm` 时再除以 `‖W_o O_q‖₂`（不开 `use_wo` 时除以 `‖O_q‖₂`）；**然后**才切 `[-W:, :-W]` 成 `(B,H,W,L)` 并沿 query 轴取均值或 EMA 得到 `score`。`use_norm` 与 `tau` 只在 `before` 支有意义（`after` 支根本不读 `use_norm`）。
5. 平滑：`avgpool` / `maxpool` 用 `kernel_size`（奇数）、`padding=kernel_size//2`、`stride=1` 的一维池化，`(B,H,L)` 按 `(batch, channel, length)` 理解即逐 head 独立；`none` 原样。
6. 选择：仅当 `S >= max_capacity_prompt` 时取 `pooled` 的 top-`k`（分数降序，**同分取下标小的**）；`kept_positions` 是这批位置后接末窗 `L..S-1` 升序。`S < max_capacity_prompt` 时不选择（`selected_indices` 长度 0、`kept_positions = range(S)`），但第 1–5 步的分数**照常算并回吐**。

返回 5 键：`attn_weights (B,H,W,S)`、`important_score (B,H,L)`、`pooled_score (B,H,L)`、`selected_indices` int64 `(B,H,k)`、`kept_positions` int64 `(B,H,k+W)`；三个张量键与输入同 dtype，历史段下标从 0 起即是绝对下标。

## 示例

```
setup: B=H=1, S=7, D=2, W=2, L=5, k=3
  query[0,0] = [[0.6,0.2], [0.1,0.9], [-0.4,0.3], [0.8,-0.1], [0.0,0.5], [-0.7,0.4], [0.2,0.2]]
  key[0,0]   = [[0.3,-0.2], [0.7,0.1], [-0.5,0.6], [0.2,0.2], [0.9,-0.3], [-0.1,0.8], [0.4,0.0]]
  value[0,0] = [[0.5,0.5], [-0.2,0.7], [0.9,0.1], [0.0,-0.6], [0.3,0.3], [-0.8,0.2], [0.6,-0.4]]
  wo         = [[0.4,-0.3], [0.2,0.7]]                # (C_out=2, H*D=2)

A: solve(query, key, value, wo, window_size=2, max_capacity_prompt=5, kernel_size=3,
         pooling="avgpool", metric_mode="after", use_wo=False, use_ema=False)
  attn_weights row0 = [0.1375, 0.1228, 0.2562, 0.1618, 0.0993, 0.2224, 0.0]     # q = 5
  attn_weights row1 = [0.136, 0.1502, 0.136, 0.1419, 0.146, 0.1481, 0.1419]     # q = 6
  important_score   = [0.0705, 0.1153, 0.1577, 0.1383, 0.0247]
  pooled_score      = [0.0619, 0.1145, 0.1371, 0.1069, 0.0543]                  # kernel 3, pad 1
  selected_indices  = [2, 1, 3]        kept_positions = [2, 1, 3, 5, 6]

B: same inputs, metric_mode="before"           # per-query odds x diff, then mean
  important_score = [0.0801, 0.1053, 0.191, 0.1352, 0.0314]      # differs from A
  pooled_score    = [0.0618, 0.1254, 0.1438, 0.1192, 0.0555]

C: pooling="none", use_ema=True, alpha=0.5    # alpha=0.5 makes EMA equal the mean
  pooled_score    = [0.0705, 0.1153, 0.1577, 0.1383, 0.0247]     # == A's important_score
  selected_indices = [2, 3, 1]               kept_positions = [2, 3, 1, 5, 6]

D: solve(..., max_capacity_prompt=9)          # S = 7 < 9, nothing is selected
  selected_indices = []                      kept_positions = [0,1,2,3,4,5,6]
  important_score / pooled_score unchanged from A
```

## 约束

- **两条支路不等价**：`after` 是「先聚合再取赔率」，`before` 是「先取赔率再聚合」，`p/(1−p)` 是凸的，两者排序可以不同；`before` 的代表 value 还带着末窗 key 的贡献（`A @ value` 用的是**全量** value），`after` 的 `v̄` 只由历史段加权。
- **`use_norm` 只在 `before` 生效**：`after` 支里这个开关根本不读，传 `True` 也不改变任何数值。
- **`tau` 只在 `after` 且 `use_wo=True` 时生效**：`before` 支与 `use_wo=False` 的 `after` 支都不做幂。
- **EMA 只回吐最后一个状态**：`alpha` 是**最新一项**的权重，`alpha=1` 等于「只取窗内最后一个 query」，`alpha=0` 等于「只取第一个」。
- **池化不改长度**：`kernel_size` 必须为奇数且 `padding = kernel_size//2`、`stride=1`，输出仍是 `(B,H,L)`；`L=1` 时池化结果就是它自己（两侧都是补边）。
- **掩码是加法不是置零**：加完 `finfo.min` 后在 float32 softmax 下这些位置恰好得 0，行和仍为 1；窗内最后一个 query 能看到的 key 数逐行递增。
- **不就地改输入**；`selected_indices` 是 top-k 的**选择序**（分数降序），不是升序；`kept_positions` 前半段跟着选择序、后半段升序。
- **非法输入抛 ValueError**：`query/key/value` 不是四维、形状或 dtype 不一致、不是 float32/float64；`wo` 不是二维、dtype 不一致或 `shape[1] != H*D`；`S <= W`、`W < 1`、`max_capacity_prompt <= W`（上游的 `assert`）、`kernel_size` 为偶数或小于 1；`pooling` / `metric_mode` 取未知字符串；三个开关不是布尔；`alpha` 不在 `[0,1]`、`tau < 0`；整数参数不是 `int`（`bool` 也算不合法）。
