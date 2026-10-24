# ZeRO 通信量：DDP vs Stage 1/2 vs Stage 3

## 题目描述

`training-state-memory` 算的是 ZeRO 各阶段**省下多少显存**；本题算它的代价——**每 GPU 每次迭代要
通信多少字节**。显存分区得越彻底，需要额外搬运的参数就越多。这是一道纯计算题（自定简化公式，不跑框架）。

## 模型与记号

$W$ 个 rank 成环，All-Reduce / Reduce-Scatter / All-Gather 这类环上原语，**每 GPU 的通信量**都带同一个
环系数：
$$ c = \frac{W-1}{W} $$
（把大小为 $S$ 的数据做一次 All-Reduce，每 GPU 收发合计 $2cS$；只做 Reduce-Scatter 或只做 All-Gather
则是 $cS$）。令参数量 $N$、每次迭代要同步的**梯度字节** $G = N \cdot \texttt{grad\_bytes}$、
**参数字节** $P = N \cdot \texttt{param\_bytes}$。

各方案每 GPU 的通信量（简化教学模型，取整数倍）：

- **DDP**：对梯度做一次 All-Reduce $= 2cG$。参数、优化器状态都在本地，无额外搬运。
- **ZeRO Stage 1 / 2**：梯度用 Reduce-Scatter（$cG$，各 rank 只归约自己负责的分片）$+$ 把更新后的
  fp16 参数 All-Gather 回来（$cG$）$= 2cG$。**与 DDP 相同**——省的是显存不是通信。
- **ZeRO Stage 3**：在上面 $2cG$ 之外，参数也被分区，前向、反向各需临时 All-Gather 一份参数
  （各 $cP$）$= 2cG + 2cP$。**通信更多**，用带宽换显存。

## 规则

实现 `solve(n_params, world, grad_bytes=2, param_bytes=2)`，返回三元组
`(ddp, zero_1_2, zero_3)`——三个方案各自的**每 GPU 每迭代通信字节**（float）。
$W=1$ 时 $c=0$，三者都应为 `0.0`（单卡无通信）。

## 示例

```
N=1e9, W=8, grad_bytes=2, param_bytes=2
G = P = 2e9,  c = 7/8 = 0.875
ddp     = 2·0.875·2e9 = 3.5e9
zero_1_2= 2·0.875·2e9 = 3.5e9
zero_3  = 3.5e9 + 2·0.875·2e9 = 7.0e9
→ (3.5e9, 3.5e9, 7.0e9)
```

## 约束

- `1 ≤ world ≤ 4096`；`n_params ≥ 1`；`grad_bytes`、`param_bytes` 通常 2（fp16/bf16）或 4（fp32）
- 返回 `(float, float, float)`，相对误差 $10^{-6}$ 内判等
- 只按上面定义的简化模型计算，不涉及真实带宽/重叠
