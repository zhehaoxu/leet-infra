# 分布式梯度裁剪（全局范数）

## 题目描述

梯度裁剪（clip_grad_norm）把整体梯度范数限制到阈值内以防梯度爆炸。在分布式下参数**分片**在各 rank，
必须用**全局范数**统一裁剪——**不能各 rank 独立裁自己的**，否则缩放系数不一致，等价于改了更新方向。
本题就是那个「最容易写错」的点。

## 规则

- 实现 `solve(shards, max_norm, norm_type=2.0)`：
  - `shards`：长度 $W$ 的列表，`shards[r]` 是 rank `r` 上的一维梯度分片（各 rank **长度可不同**）；
  - 把所有分片视作同一个大向量，计算**全局 $p$-范数**：
    $$ \|\mathbf{g}\|_p = \left(\sum_{r}\sum_{i} |g_{r,i}|^{\,p}\right)^{1/p} $$
  - 缩放系数取 $\text{scale} = \min\!\left(1.0,\ \dfrac{\text{max\_norm}}{\|\mathbf{g}\|_p}\right)$
    （范数为 $0$ 或已 $\le$ `max_norm` 时 scale=1，不裁剪）；
  - **每个分片都乘同一个 scale**：$g_{r,i} \leftarrow g_{r,i}\cdot \text{scale}$。
- 返回 `(clipped, total_norm)`：`clipped` 是与输入等长的 `list[torch.Tensor]`，`total_norm` 是全局范数（float）。
- 不得修改输入张量。

## 示例

```
输入：shards = [ [3.0, 4.0], [0.0] ], max_norm=2.5, norm_type=2.0
  total_norm = sqrt(3² + 4² + 0²) = 5.0
  scale = min(1, 2.5/5.0) = 0.5
输出：( [ [1.5, 2.0], [0.0] ], 5.0 )
```

> 注意：若错误地各 rank 独立裁自己的，rank0 局部范数 5 会被裁、rank1 局部范数 0 不裁——与全局结果不同。

## 约束

- `1 ≤ W ≤ 64`；分片为一维、float、长度可不等；`max_norm ≥ 0`；`norm_type ≥ 1`（本题取有限 $p$，不含 $\infty$）
- 返回 `(list[torch.Tensor], float)`；`total_norm` 与裁剪后张量用相对容差 `1e-5` 判定
