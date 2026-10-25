# Checkpoint I/O 带宽账与异步持久化

## 题目描述

训练随时会挂，**检查点（checkpoint）** 存得越勤，崩了以后重跑的代价越小；但每次存盘都要把整份训练态
从 HBM 搬到 host 内存再落到并行文件系统/NVMe，**慢存储会把训练一步一停地拖死**。**异步保存**
（Megatron 的 `async_save`）把「拷进 pinned staging」和「落盘」拆开：只有前者阻塞训练，落盘与后续
step 重叠，用 B 个 staging 缓冲位换落后空间。存多勤还有一个经典的折中公式（Young-Daly / CheckFreq）：
周期 $\approx\sqrt{2CF}$，$C$ 是每次检查点**阻塞**的时长、$F$ 是平均故障间隔——异步把 $C$ 从「拷贝+落盘」
压到「只剩拷贝」，于是周期可以更短。

`training-state-memory` 给出要搬多少字节、`distributed-zero-offload` 算的是训练期逐 step 的 HBM↔DRAM
预取（读侧）、`distributed-sublinear-checkpoint` 算的是激活重算（算侧）；本题只管**存侧的 I/O 账**。

## 规则

记 $N=$ `n_params`，训练态字节沿用 `training-state-memory` 的混合精度 + Adam 口径
（fp16 参数 2 + fp16 梯度 2 + fp32 master 4 + $m$ 4 + $v$ 4）：

$$ \text{state\_bytes}=N\cdot\texttt{state\_bytes\_per\_param}\quad(\texttt{state\_bytes\_per\_param}=16\text{ 默认}) $$

两级带宽各除一次（毫秒口径）：

$$ \text{copy\_ms}=\frac{1000\cdot\text{state\_bytes}}{\texttt{device\_to\_host\_bw}},\qquad
   \text{write\_ms}=\frac{1000\cdot\text{state\_bytes}}{\texttt{storage\_bw}},\qquad
   \text{period\_ms}=\texttt{checkpoint\_interval}\cdot\texttt{step\_ms} $$

- **阻塞时长**：`async_save=False` 时 $\text{stall}=\text{copy}+\text{write}$；
  `async_save=True` 时落盘与训练重叠，只有 pinned 拷贝阻塞：$\text{stall}=\text{copy}$。
- **检查点开销占比**：$\text{overhead}=\text{stall}/\text{period}$（可 $>1$，表示存得比跑得快）。
- **写盘压不住判据**：槽位从拷贝起一直被占用到落盘结束，B 个 staging 位允许一次检查点最多落后 B 个周期：
  $$ \text{writer\_limited}=\big(\text{copy}+\text{write}>\texttt{staging\_buffers}\cdot\text{period}\big) $$
  （严格大于；持平不算。**同步/异步都用这一判据**，同步时它只是参考指标。）
- **最优周期**（Young-Daly / CheckFreq）：以 step 为单位的阻塞成本 $C=\text{stall}/\texttt{step\_ms}$，
  $F=$ `mtbf_steps`：
  $$ \text{optimal\_interval\_steps}=\max\!\Big(1,\ \Big\lfloor\sqrt{2CF}\ \Big\rfloor\Big) $$

## 示例

```
输入：n_params=1e9, step_ms=100, checkpoint_interval=200, device_to_host_bw=64e9, storage_bw=10e9,
      async_save=True, staging_buffers=2, mtbf_steps=1e6, state_bytes_per_param=16
  state_bytes = 1e9 * 16 = 16,000,000,000（约 14.9 GiB）
  copy_ms = 1000*16e9/64e9 = 250.0   write_ms = 1000*16e9/10e9 = 1600.0   period_ms = 200*100 = 20000.0
  stall = 250.0（异步只付拷贝）  overhead = 250/20000 = 0.0125（1.25%）
  writer_limited: 250+1600=1850 > 2*20000=40000? 否
  optimal = floor(sqrt(2*(250/100)*1e6)) = floor(sqrt(5e6)) = 2236
输出：(16000000000, 250.0, 1600.0, 250.0, 20000.0, 0.0125, False, 2236)

输入：同上但 async_save=False
  stall = 250 + 1600 = 1850.0   overhead = 1850/20000 = 0.0925（9.25%）
  optimal = floor(sqrt(2*18.5*1e6)) = floor(sqrt(3.7e7)) = 6082
输出：(16000000000, 250.0, 1600.0, 1850.0, 20000.0, 0.0925, False, 6082)
```

## 约束

- `n_params` $\ge 1$；`step_ms` $>0$；`checkpoint_interval` $\ge 1$；两级带宽 $>0$；
  `staging_buffers` $\ge 1$；`mtbf_steps` $>0$；`state_bytes_per_param` $\ge 1$（默认 16）
- 返回八元组 `(state_bytes, copy_ms, write_ms, stall_ms, period_ms, overhead_fraction,
  writer_limited, optimal_interval_steps)`：第 1、8 项是 int，第 7 项是 bool，
  第 2-6 项是 float（`atol=1e-5, rtol=1e-5`）
- 单机单存储链路口径：只算「拷贝阻塞 + 落盘重叠 + 槽位落后 B 个周期」这一简化模型，
  不模拟多 rank 并发写同一文件系统的带宽争抢
