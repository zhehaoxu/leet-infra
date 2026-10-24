# Ring vs Recursive-Doubling 延迟账 详解

## α-β 模型是什么

一次集合通信的耗时拆成两部分：**启动开销** $\alpha$（每发起一次 send/recv 的固定延迟：软件栈、握手、kernel 启动）与**传输开销** $\beta\cdot\text{bytes}$（受带宽限制）。总时间 $=\text{步数}\cdot\alpha + \text{总字节}\cdot\beta$。不同算法只是「步数」与「每步字节数」的取舍不同。

## 两种拓扑的取舍

- **Ring**：数据切成 $W$ 块绕环走。做 reduce-scatter + all-gather 两趟 → $2(W-1)$ 步，每步只传 $\frac{S}{W}$。步数多，但**每步数据小、链路全用满**，带宽项 $\frac{2(W-1)}{W}S\beta\to 2S\beta$（与 $W$ 几乎无关）。适合**大消息**。
- **Recursive-Doubling**：第 $i$ 步和距离 $2^i$ 的伙伴交换，$\\log_2 W$ 步就完成。步数少（延迟项 $\\log_2 W\cdot\\alpha$ 碾压 ring），但每步搬运整份数据。**要求 $W$ 是 2 的幂**。适合**小消息 / 高延迟网络**（尤其单机 NVLink 多卡、$W$ 大时 ring 的 $2(W-1)$ 步启动很贵）。

> 注：本题用「rd 每步搬 $S$ 字节」的简化教学模型钉死公式；真实 RD 的带宽项更精细，但**取舍方向**（延迟 vs 带宽）与结论一致。判题只认题面两条闭式。

## 交叉点直觉

令两者相等可解出临界 $S^*$：$W$ 越大、$\alpha$ 越大，越偏向 rd；$S$（消息）越大越偏向 ring。这正是 NCCL 在 `all_reduce` 时按消息大小和拓扑自动切换 ring/tree(≈rd) 的依据。

## 实现坑

- **tie 规则要钉死**：$T_{ring}=T_{rd}$ 时判 `"ring"`（用 `<=`）。否则浮点相等边界会歧义。
- `speedup = max/min ≥ 1`，恒非负；别写成快/慢导致 <1。
- $\log_2 W$ 用 `math.log2`（2 的幂下精确），避免 `log(x)/log(2)` 的末位差影响 tie 判定。

## 关联

- 上游：`distributed-all-reduce` / `distributed-reduce`（这里分析的是它们的**耗时**）。
- 同类账本：`distributed-pipeline-bubble`（气泡率）、`distributed-zero-offload`（显存-带宽）。
- 判题：纯闭式代数 + `test_cases` 独立重算（展开式 vs 因式两条路径）核对。
