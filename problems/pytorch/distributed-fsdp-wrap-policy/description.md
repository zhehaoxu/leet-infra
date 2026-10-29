# 分布式：FSDP wrap policy 与块级显存-通信账

## 题目描述

为一个简化的分片训练流程规划参数块，并计算临时完整参数峰值和集合通信调用次数。本题采用**玩具相邻层装箱策略**：按层顺序累加正的参数量，只要下一层加入后不超过 `wrap_size` 就放入当前块，否则开启新块。层不可拆分，单层不能超过上限，恰好达到上限允许放入。

这里的 `wrap_size` 是块大小的**最大值**，不是 PyTorch 递归、最小阈值式 `size_based_auto_wrap_policy`；本题不复现其模块树遍历规则。

## 计算规则

- 前向按块顺序执行：参数 all-gather → 计算 → 参数 reshard。
- 反向按相反顺序执行：参数 all-gather → 计算 → 参数 reshard，并对梯度执行 reduce-scatter。
- 每块共 **3 次集合通信调用**：前向参数 all-gather、反向参数 all-gather、梯度 reduce-scatter。reshard 在本模型中仅释放完整参数，不额外计通信。这不是网络轮次，也不是按 rank 或硬件传输步骤累加的次数。
- `prefetch_distance` **包含当前块**：每次计算前，保留当前块和沿当前遍历方向后续至多 `prefetch_distance - 1` 个连续块的完整参数。前向和反向使用相同窗口，且不假设前反向重叠。
- 峰值仅计**临时完整参数的数量**，不是字节数或设备总显存；不含常驻参数分片、梯度、激活。

设块大小依次为 \(b_0,\ldots,b_{n-1}\)，\(d=\text{prefetch_distance}\)，则

\[
\text{peak}=\max_{0\le i<n}\sum_{j=i}^{\min(i+d,n)-1}b_j,
\qquad \text{collective calls}=3n.
\]

反向采用逆序的连续窗口，峰值与前向相同，因此不能把两者相加。空层列表的块数、峰值和调用次数均为零。完整输入、输出类型与异常约定见 starter。

## 示例

```text
param_sizes=[30, 40, 50, 20], wrap_size=80, prefetch_distance=2
block_sizes=[70, 70]
peak_params=140
返回 (2, [70, 70], 140, 6)
```

第一块放入 30 和 40 后，无法再容纳 50；第二块为 50 和 20。两个块同时驻留时峰值为 140，共调用集合通信 6 次。

```text
param_sizes=[90, 20, 90], wrap_size=100, prefetch_distance=2
返回 (3, [90, 20, 90], 110, 9)
```

窗口必须连续，两个 90 不会同时出现在长度为 2 的窗口内，故峰值为 110 而非 180。距离为 1 时只保留当前块；距离超过块数时最多保留全部块。
