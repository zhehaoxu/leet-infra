# 张量并行的每层通信量（Megatron TP）

## 题目描述

数据并行每步同步的是**梯度**（大小 $\propto$ 参数量），张量并行（Tensor Parallel, TP）每层同步的却是
**激活**（大小 $\propto$ batch×seq×hidden）——它把单层的大矩阵按列/行切到多卡，前向/反向都要用 all-reduce
把分片结果合起来。这就是「TP 只在机内 NVLink 域做、且 $W$ 一小就饱和」的原因。本题算 TP 每层的通信字节
（纯计算，自定简化模型）。

## 模型

Megatron 风格：一个 transformer 层的 MLP 与注意力各用「列并行→行并行」组合，导致
- **前向**：注意力 1 次 all-reduce + MLP 1 次 all-reduce $=2$ 次；
- **反向**：对应再来 $2$ 次。

合计**每层 4 次 all-reduce**，每次的消息是一张**激活**：
$$ \text{act\_bytes} = \texttt{batch}\times \texttt{seq}\times \texttt{hidden}\times \texttt{bytes\_per\_elem} $$

$W$ 卡成环，一次 all-reduce 每 GPU 收发 $2c\cdot \text{act\_bytes}$（环系数 $c=\frac{W-1}{W}$，与
`parallel-allreduce`、`distributed-zero-communication` 同一系数）。于是每层每 GPU 通信量：
$$ \text{per\_layer} = 4\times\big(2c\cdot \text{act\_bytes}\big) = 8\,\frac{W-1}{W}\cdot \text{act\_bytes} $$
$$ \text{total} = \texttt{n\_layers}\times \text{per\_layer} $$

## 规则

实现 `solve(batch, seq, hidden, world, n_layers=1, bytes_per_elem=2)`，返回三元组
`(act_bytes, per_layer_comm, total_comm)`（float 字节数）。$W=1$ 时 $c=0$，后两项为 `0.0`（单卡无 TP 通信）。

## 示例

```
batch=8, seq=2048, hidden=4096, world=8, n_layers=32, bytes_per_elem=2
act_bytes    = 8·2048·4096·2      = 1.34217728e8
c            = 7/8 = 0.875
per_layer    = 8·0.875·1.342e8    = 9.39524096e8
total        = 32·9.395e8         = 3.0064771072e10
→ (1.34217728e8, 9.39524096e8, 3.0064771072e10)
```

## 约束

- `batch, seq, hidden ≥ 1`；`world ≥ 1`；`n_layers ≥ 0`；`bytes_per_elem` 常用 2（fp16/bf16）或 4
- 返回 `(float, float, float)`，相对误差 $10^{-6}$ 判等
- 固定「每层 4 次 all-reduce」的 Megatron 简化：不建模 Sequence Parallel 带来的 all-gather/reduce-scatter
  替换，也不考虑通信与计算重叠
