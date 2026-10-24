# MoE 并行：All-to-All 分派与合并

## 题目描述

专家并行（Expert Parallelism）下，$E$ 个专家分布在 $P = E/\text{EPD}$ 张卡上（每卡 $\text{EPD}$ 个专家、连续编号）。token 序列**沿 $T$ 维平均切分**到 $P$ 张卡；每张卡先做本地 top-k 路由，然后通过一次 all-to-all 把每个 (token, slot) 送到「目标专家所在的卡」上；本地卡执行完再 all-to-all 送回原卡、按 gate 权重加权合并。

## 规则

约定 expert 输出为「输入 + 专家 id」：$\text{expert}_e(x) = x + e$（每张卡上的 $e$ 都相同）。

实现 `solve(hidden_states, topk_indices, topk_weights, num_experts, experts_per_device)`：
- `hidden_states` 形状 $(T, H)$ float32
- `topk_indices` 形状 $(T, k)$ long；`topk_weights` 形状 $(T, k)$ float32（**每行和为 1**，已经 renorm）
- `num_experts = E`；`experts_per_device = EPD`；`P = E // EPD`
- 每张卡的 token 数 `tokens_per_device = (T + P - 1) // P`；token $t$ 起始在 `src_device = t // tokens_per_device`
- 专家 $e$ 在 `dst_device = e // EPD`

返回 `(send_matrix, combined)`：
- `send_matrix` 形状 $(P, P)$ long：`send_matrix[i, j]` = 从卡 $i$ 送往卡 $j$ 的 **(token, slot) 对**数量（已按 top-k 展平，即每 token 贡献 $k$ 次）。
- `combined` 形状 $(T, H)$ float：$\text{combined}[t] = \sum_j \text{topk\_weights}[t, j] \cdot (\text{hidden}[t] + \text{topk\_indices}[t, j])$

## 示例

```
T=8, H=2, E=4, EPD=2 → P=2, tokens_per_device=4；k=2
hidden = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
topk_indices = [[0,1],[1,2],[2,3],[3,0],[0,2],[1,3],[2,0],[3,1]]
topk_weights = [[.6,.4],[.5,.5],[.7,.3],[.4,.6],
                [.5,.5],[.3,.7],[.55,.45],[.65,.35]]

send_matrix = [[4,4],[4,4]]
combined = [[1.4, 2.4], [4.5, 5.5], [7.3, 8.3], [8.2, 9.2],
            [10.0, 11.0], [13.4, 14.4], [14.1, 15.1], [17.3, 18.3]]
```

（`combined[t] = hidden[t] + Σ_j w[t,j] * idx[t,j]`，例如 t=0：`Σ w*idx = 0.6*0 + 0.4*1 = 0.4` → `1+0.4=1.4`, `2+0.4=2.4`。）

## 约束

- CPU float32；`send_matrix.dtype == torch.long`。
