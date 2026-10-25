# 详解：step 时间分解、Amdahl 与取数掩盖

## 出处

Amdahl 定律（G. Amdahl, 1967）：整机加速比受「未被加速部分」的占比限制，
$\text{speedup}=\big[(1-f)+f/k\big]^{-1}$，上限 $1/(1-f)$。训练侧的相位表来自 PyTorch Profiler 的
串行时间线口径（dataloader / fwd / bwd / optimizer）。取数模型对应 `torch.utils.data.DataLoader` 的
`num_workers`（并行生产者）与 `prefetch_factor`（每 worker 预取批数）：`num_workers=0` 就是同步取数，
`prefetch_factor>=1` 才存在预取缓冲。本题用「供给间隔 vs 计算时长」的稳态摊销把它变成一个判据。

## 核心与易错点

1. **Amdahl 只压缩被点名的相位**：$\text{new\_step}=(\text{total}-p)+p/k$。写成 $\text{total}/k$ 是最常见错误
   （那等于整个 step 都变快）。
2. **`ceiling` 是极限不是实测**：$\text{ceiling}=\text{total}/(\text{total}-p)$ 对应 $k\to\infty$，即「把这段彻底
   消掉」。占比 5% 的相位哪怕优化到无穷快也只带来 5.3% 收益——这正是本题第一个示例的结论。
3. **`k=1` 必须是恒等**：$\text{amdahl}=1$、$\text{new\_step}=\text{total}$；且恒有
   $\text{ceiling}\ge\text{amdahl}$（$k\ge1$ 时）。
4. **`compute` 不含取数**：它是 `phase_ms[1]+[2]+[3]`，不是 `total`。把 `total` 当计算时长会让掩盖判据永远成立。
5. **`prefetch` 决定有没有重叠**：`prefetch=0` 是同步取数，每步都要全额付 $\text{supply}$；
   `prefetch>=1` 才只需补 $\max(0,\text{supply}-\text{compute})$。注意预取深度本身**不改变稳态供给率**——
   供给率只由 `num_workers` 决定，加 worker 才是真解药，这与 `dataset-dataloader` 的口径一致。
6. **掩盖判据含等号**：$\text{supply}=\text{compute}$ 算掩盖成功（`data_hidden=True`）。
7. **两套视角别混用**：`fracs` 用 profiler 实测的 `phase_ms[0]`（真实发生的等待），
   `effective_step_ms` 用 worker 模型（假设把取数挪到异步后还剩多少）。所以 `phase_ms[0]` 不出现在 `compute` 里。
8. **稳态摊销**：`effective` 不含首个 batch 的暖机代价，用例里用 4000 步流水模拟验证了这个口径。

## 对拍口径

参考解走闭合式 $\text{total}-p+p/k$、`compute = p[1]+p[2]+p[3]`；判题 `_oracle` 用另一条路：显式造出
「把第 `target_phase` 项除以 $k$ 之后的相位表」再整体求和得到 `new_step`，`ceiling` 用「其余项逐个累加」的分母，
`compute` 改走 `total - phase_ms[0]`。额外断言：$k=1$ 恒等、$k=10^9$ 逼近 `ceiling`、其余相位全为 0 时
`ceiling=inf` 且 $\text{amdahl}=k$、`prefetch=0` 无重叠、加 `num_workers` 时 `effective` 单调不增、
`fracs` 之和为 1，以及一个 $S=4000$ 步的取数-计算流水暴力模拟与闭合式对齐。

## 复杂度

参考解 $O(1)$（4 个相位与一次常数次除法）；判题的流水模拟断言 $O(S)$，$S=4000$，总耗时毫秒级。
