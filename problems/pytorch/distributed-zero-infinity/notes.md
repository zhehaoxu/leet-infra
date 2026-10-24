# ZeRO-Infinity 三层卸载账 详解

## 从 Offload 到 Infinity 的两处升级

1. **多一层存储介质**：ZeRO-Offload 是 HBM↔DRAM 两层；Infinity 再加 **NVMe SSD**，形成 HBM（最快最小）→ DRAM → NVMe（最慢最大「无限」）三层。优化器状态这种「体积大、访问不频繁」的数据最适合沉到 SSD，让显存只放当前真正需要的分片。
2. **卸载数据也沿 DP 分片**：这是 Infinity 相对 Offload 的关键区别。ZeRO-Offload 里**每个数据并行 rank 都把完整的优化器状态卸到自己的 CPU**（卸载量与 $d$ 无关）；Infinity 把卸载数据**切片分散到各 rank**，每 rank 只存 $1/d$。所以本题 $P,G,O$ 全部除以 $d$——包括卸到 DRAM/SSD 的那部分。

## 三层账本怎么摆

模型刻意简化到「一步预取整份被卸载分片」：

- HBM：梯度分片 $G$（每步即时用，不卸）+ 未卸载的参数/优化器分片。
- 被卸载分片 $= \mathbb{1}[\text{offload\_param}]P + \mathbb{1}[\text{offload\_opt}]O$，整体进 target 层（dram 或 ssd），另一层 0。
- `step_transfer_bytes` = 被卸载分片：每步从低层搬回 HBM 的量（衡量对 PCIe/NVMe 带宽的压力）。

## 边界与直觉

- 两个开关全 False → 退化成 **ZeRO-3 全驻 HBM**：`hbm_bytes=(P+G+O)=16N/d`、搬运 0。用来对照「不卸载要多大显存」。
- 只卸优化器（最常见的 Infinity 配置）→ HBM 省掉最大的 $O=12N/d$，$P,G$ 仍常驻。
- target 选 dram 还是 ssd **只改占用落在哪层**，不改 HBM 与搬运总量——真实系统里差异在带宽/延迟（NVMe 更慢、需 `afr` 预取），本题不计。

## 实现坑

- 别忘了**全部除以 $d$**（含卸载部分），这是 Infinity≠Offload 的考点；写成整份会显著偏大。
- $P,G,O$ 用 float 除法，避免整型截断；返回统一 `float`。
- dram/ssd 互斥：target 之外的层必须是 0，不是「也放一份」。

## 关联

- 对照：`distributed-zero-offload`（两层、卸载不随 $d$ 分片）、`distributed-zero-communication`（分区通信量）、`training-state-memory`（每参数 16 字节基线）。
- 判题：纯闭式 + `test_cases` 独立重算核对，相对误差 $10^{-6}$。
