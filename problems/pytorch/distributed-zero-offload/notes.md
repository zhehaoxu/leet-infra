# 解题思路

## 面试官意图

ZeRO 家族有三条省钱路线：**分区**（跨卡省显存、多花跨卡带宽）、**Offload**（把优化器状态丢 CPU，用 PCIe
换显存、单卡可训大模型）、**Infinity**（再叠 NVMe）。能说清 Offload 到底拿什么换什么、瓶颈会在哪，是区分
「调过 DeepSpeed 配置」和「懂显存/带宽 tradeoff」的分水岭。

## 这笔交换怎么算

默认每参数 16 bytes（fp16 参数 2 + fp16 梯度 2 + fp32 优化器 12）。Offload 后：
- GPU 只留 **fp16 参数** = $2N$（对比全放 GPU 的 $16N$，显存降到 $\frac18$）。
- CPU 扛 **梯度 + 优化器状态** = $14N$。
- 每迭代 PCIe 搬运 = 梯度下去 $2N$ + 更新后参数上来 $2N$ = $4N$。

于是 $t_{\text{pcie}}=\frac{4N}{\text{BW}}$。$N=10^9$、BW$=32$GB/s 时 $t=0.125$s——这正是小模型
（本身计算快）用 Offload 反而变慢的原因：$t_{\text{pcie}}$ 盖过计算，PCIe 成瓶颈。模型越大、每迭代计算越久，
越容易把 $0.125$s 藏起来，Offload 才划算。

## 判定口径

`is_bottleneck = t_pcie > compute_time`，**严格大于**相等记 False——别写成 `>=`，`check_bottleneck_strict_gt`
专门卡这个。带宽/参数量/字节宽度任一变化都直接体现在四个返回值上，测试用逐项累加的独立参考对拍。

## 与两兄弟题的三角关系

- `training-state-memory`：全放 GPU / ZeRO 分区后的**显存**（$16N$、$16N/W$）。
- `distributed-zero-communication`：ZeRO 分区多花的**跨卡带宽** $2cG(+2cP)$。
- `distributed-zero-offload`（本题）：Offload 把「跨卡」换成「卡↔CPU 的 PCIe」，省的是**单卡显存**。
三者合起来才是 ZeRO 全家桶的显存-带宽完整账本。

## 真实工程补充

实际 DeepSpeed 会把「下一层前向」和「上一层的优化器更新 + PCIe 回传」重叠，故真实 $t_{\text{pcie}}$ 常小于
纯带宽估计；本题忽略重叠是刻意的简化，判题只认题面定义。
