# SHiRA 稀疏增量：COO 散点重建与重复索引累加

## 题目描述

LoRA 把可训练参数组织成两个稠密低秩矩阵，SHiRA（Sparse High-Rank Adaptation，huggingface/peft `tuners/shira`）则干脆放弃结构：直接在权重矩阵里挑 $K$ 个**离散坐标**，只训练这 $K$ 个标量。它保证参数预算与 LoRA 相同——$K = r \cdot (\text{in} + \text{out})$，且不超过 $r \cdot (\text{in} + \text{out}) \le \text{in} \cdot \text{out}$（不然「稀疏」比稠密基座还大）。增量矩阵按 COO 散点重建后**加**在基权重上（与 HiRA 的乘性相反）：

$$W' = W + \Delta, \quad \Delta_{ij} = \sum_{k:\ (r_k,c_k)=(i,j)} s \cdot v_k$$

公式里的求和号是本题的题眼：上游用 `torch.sparse_coo_tensor(indices, values, shape)` 构造增量再 `to_dense()`，而 COO 张量对**重复坐标的语义是累加**——同一个位置出现两次的值会相加，不是后者覆盖。给 mask 找坐标的路径恰好可能产生重复（上游 `torch.where(mask == 1.0)`），这一点没写对，整份 Δ 就错了。本题给定坐标对、值与缩放系数，复现这份稀疏增量、加性合并与前向。

## 任务

实现 `solve(weight, indices, values, scaling, x)`：

- `weight`：`[out, in]` 的二维浮点 tensor；
- `indices`：非空 `list`，每项是 `[row, col]` 的二元组（`list` 或 `tuple`），整数坐标且 `0 <= row < out`、`0 <= col < in`；允许重复坐标；
- `values`：长度与 `indices` 一致的数值 `list` 或一维 tensor；
- `scaling`：非负有限实数（训练期为 1.0，推理期可任意缩放），乘在每个 value 上；
- `x`：至少一维的 tensor，末维等于 `in`。

返回 `dict`，键为 `delta_weight`（稀疏重建的 `[out, in]`）、`merged_weight`（`weight + delta`）、`y`（`x @ merged.T`，形状 `[..., out]`）。

## 精确规则

1. **参数预算**：坐标数 $K$ 不得超过 `in * out`（上游 `update_layer` 的显式检查：稀疏参数多于基权重元素数直接拒绝）。
2. **散点重建**：`delta` 初始化为零矩阵，对每个坐标 $(r_k, c_k)$ 做 `delta[r_k, c_k] += scaling * v_k`——**重复坐标累加**。
3. **加性合并**：`merged = weight + delta`（对比 HiRA：`W⊙(1+Δ)`；SHiRA 在 W 为 0 的位置照样写入）。
4. **前向**：`y = x @ merged.T`（上游 forward 把 delta 加进基权重后整体走 `F.linear`，即用合并后的权重）。

## 示例

```text
W = [[1, 2, 0], [0, 0, 0]]（out=2, in=3）
indices = [[0,0], [0,0], [1,2]]，values = [1, 2, 3]，scaling = 2
x = [[1, 1, 1]]

(0,0) 出现两次：1*2 + 2*2 = 6（累加；若覆盖则错成 4）
(1,2) 一次：3*2 = 6
delta = [[6, 0, 0], [0, 0, 6]]
merged = [[7, 2, 0], [0, 0, 6]]
y = x @ merged.T = [[9, 6]]
```

语义锚点：`W` 全零、坐标 `[[0,1]]`、值 4 时 `merged = [[0,4,0],[0,0,0]]`、`y = [[4, 0]]`——加性语义下基座为零的元素照样能被增量抬起（乘性的 HiRA 会永远冻结它们）。

## 约束

- 不修改输入；非法输入（weight 非二维、indices 空/坐标非二元组/坐标越界/负坐标、values 数量不匹配或非数或非有限、scaling 为负/无穷/bool、x 末维不匹配或零维或非有限、K 超预算）抛 `ValueError`。
