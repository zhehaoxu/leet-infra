# ZeRO-Offload：显存与 PCIe 带宽的取舍

## 题目描述

`training-state-memory` 算 ZeRO 分区省多少显存、`distributed-zero-communication` 算它多花多少**跨卡
带宽**。ZeRO-Offload 走另一条路：把优化器状态（和梯度）**卸载到 CPU 内存**，用 **PCIe 带宽换 GPU 显存**——
单卡也能训大模型。本题算清楚这笔交换（纯计算，自定简化模型）。

## 模型

设参数量 $N$。每个参数训练态字节数（与 `training-state-memory` 对齐）：

| 分量 | 每参数字节 | 总字节 |
|---|---|---|
| fp16 参数 | `param_bytes`（默认 2） | $2N$ |
| fp16 梯度 | `grad_bytes`（默认 2） | $2N$ |
| fp32 优化器状态（master+momentum+variance） | `opt_bytes`（默认 12） | $12N$ |

ZeRO-Offload 的放置策略：**只有 fp16 参数常驻 GPU**；梯度算出来立即送 CPU、优化器状态全在 CPU。于是：

- **GPU 显存**（训练态部分）$= \texttt{param\_bytes}\cdot N$
- **CPU 内存**（卸载部分）$= (\texttt{grad\_bytes}+\texttt{opt\_bytes})\cdot N$
- **每次迭代 PCIe 传输量**：梯度 GPU→CPU（$\texttt{grad\_bytes}\cdot N$）$+$ 更新后 fp16 参数 CPU→GPU
  （$\texttt{param\_bytes}\cdot N$）$= (\texttt{grad\_bytes}+\texttt{param\_bytes})\cdot N$

PCIe 传输耗时 $t_{\text{pcie}} = \dfrac{\text{pcie\_bytes}}{\texttt{pcie\_bandwidth}}$。若它**大于** GPU
每迭代计算时间 `compute_time`，PCIe 就成了瓶颈（计算无法掩盖传输）：$\text{bottleneck} = (t_{\text{pcie}} > \texttt{compute\_time})$。

## 规则

实现 `solve(n_params, pcie_bandwidth, compute_time, param_bytes=2, grad_bytes=2, opt_bytes=12)`，返回
`(gpu_mem, cpu_mem, pcie_bytes, is_bottleneck)`：前三项 float 字节数，末项 bool。

## 示例

```
N=1e9, pcie_bandwidth=3.2e10 (32 GB/s), compute_time=0.1s, 默认字节
gpu_mem   = 2·1e9  = 2.0e9
cpu_mem   = 14·1e9 = 14.0e9
pcie_bytes= 4·1e9  = 4.0e9
t_pcie    = 4.0e9 / 3.2e10 = 0.125s  >  0.1  → is_bottleneck = True
→ (2.0e9, 14.0e9, 4.0e9, True)
```

## 约束

- `n_params ≥ 1`；`pcie_bandwidth > 0`；`compute_time ≥ 0`
- 返回 `(float, float, float, bool)`；字节用相对误差 $10^{-6}$ 判等
- 忽略激活显存、CPU↔GPU 双向并发、通信/计算重叠优化——纯按上面定义算
- `is_bottleneck` 用**严格大于**（$t_{\text{pcie}} == \texttt{compute\_time}$ 记 `False`）
