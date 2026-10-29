# Aux-free 分组路由与偏置更新

## 任务

实现 `solve`：用 sigmoid 亲和度加一个**只参与选择、不参与加权**的逐专家偏置，先按组选择，再在选中组内做 top-k，最后按原始 sigmoid 分数归一化。可选地按本 batch 的专家负载更新偏置。

这是 DeepSeek 风格 auxiliary-loss-free 负载均衡的受限数值模型。题目显式规定分组、并列顺序和偏置更新规则；它不是任何训练框架的逐步复刻，也不执行训练或梯度更新。

## 输入

- `x`：CPU float64 `[T,D]`。
- `gate_w`：CPU float64 `[E,D]`；路由器 logits 为 `x @ gate_w.T`。
- `bias`：CPU float64 `[E]`，逐专家选择偏置。
- `group_count`：Python `int`，必须整除 E。
- `top_groups`：Python `int`，`1 <= top_groups <= group_count`。
- `k`：Python `int`，`1 <= k <= top_groups * group_size`。
- `gamma`：Python 浮点数，`>= 0`。
- `update_bias`：Python `bool`，默认 `False`。
- 专家按编号连续划分为长度相等的组，组内专家数 `group_size = E / group_count`，必须至少为 2。
- `T,D,E=1..8`；输入值合法，允许非连续张量，不修改输入。

结构非法、`k` 越界或 `gamma < 0` 时抛出 `ValueError`。

## 精确规则

sigmoid 分数、选择分数：

$$
s_i=\sigma(xW_i^{\mathsf T}),\qquad c_i=s_i+b_i.
$$

组 $g$ 的得分是组内**最大的两个** $c_i$ 之和；组内专家数至少为 2，因此不会重复计数。按组得分降序选择 `top_groups` 个组，完全相同时组号较小者优先。

只在选中组内按 $c_i$ 降序取 k 个专家，同分时专家编号较小者优先。**偏置不影响加权**：

$$
a_j=\frac{s_{i_j}}{\sum_{\ell=1}^{k}s_{i_\ell}}.
$$

返回的 `weights` 是 `[T,k]` 的归一化权重，不是 `[T,E]` 稀疏矩阵。

负载为整个 batch 中每个专家被选中的次数。令 $\bar L=Tk/E$，更新只在 `update_bias=True` 时发生：

$$
b_i'=\begin{cases}
b_i+\gamma, & L_i<\bar L,\\
b_i-\gamma, & L_i>\bar L,\\
b_i, & L_i=\bar L.
\end{cases}
$$

过载专家降低偏置，欠载专家提高偏置。`update_bias=False` 时必须返回与输入 `bias` 数值相同、但不共享存储的新张量。

## 返回

返回七项 `tuple`：

```text
(weights, indices, scores, group_scores, selected_groups, load, new_bias)
```

| 项 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `weights` | float64 `[T,k]` | 原始 sigmoid 分数的组内归一化 |
| `indices` | int64 `[T,k]` | 选中专家，按选择分数降序 |
| `scores` | float64 `[T,E]` | 原始 sigmoid 分数 |
| `group_scores` | float64 `[T,G]` | 每组 top-2 选择分数之和 |
| `selected_groups` | int64 `[T,top_groups]` | 被选中的组，按组得分降序 |
| `load` | int64 `[E]` | 整个 batch 的专家被选次数 |
| `new_bias` | float64 `[E]` | 更新后或原样的偏置 |

浮点结果按 `rtol=2e-11, atol=2e-11` 比较。

## 示例

```python
x = [[1.0]]
gate_w = [[1.0], [0.9], [-0.5], [-0.6]]
bias = [0.0, 0.0, 0.0, 0.0]
group_count = 2
top_groups = 1
k = 2
gamma = 0.1
update_bias = True
```

组 0（专家 0、1）的 top-2 和为 `sigmoid(1)+sigmoid(0.9)≈1.4420`，组 1 为 `≈0.7319`，因此选中组 0。输出按张量内容表示：

```python
weights = [[0.5070, 0.4930]]
indices = [[0, 1]]
scores = [[0.7311, 0.7109, 0.3775, 0.3543]]
group_scores = [[1.4420, 0.7319]]
selected_groups = [[0]]
load = [1, 1, 0, 0]
new_bias = [-0.1, -0.1, 0.1, 0.1]
```

把专家 2 的偏置改为 `2.0` 后，组 1 的 top-2 和反超并只选专家 2、3；但权重仍由原始 sigmoid 分数归一化，偏置不出现在权重中。
