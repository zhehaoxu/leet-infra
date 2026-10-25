# LR 调度器在训练循环中的位置

## 题目描述

`optimizer.step()` 和 `scheduler.step()` 谁先谁后，是一个 `last_epoch` 计数器的问题——它决定调度表第一格 lr 是否被"跳过"，进而改变真实训练轨迹；`ReduceLROnPlateau` 更特殊，它由指标序列驱动。本题用同一份 loss 跑三种挂法，把逐 step 生效 lr、权重终值与调度器内部状态三本账对齐。

## 规则

实现 `solve()`，按 starter docstring 跑三个变体并返回账目 dict：

1. **变体 A**（推荐次序）：每步先记录当前 `param_groups[0]["lr"]`，再 `zero_grad → backward → opt.step() → sch.step()`。
2. **变体 B**（scheduler 抢跑）：`sch.step()` 放在记录 lr 之前，官方 UserWarning 应被捕获（`warned_skipping_first`）。
3. **变体 C**：`ReduceLROnPlateau(mode='min', factor=0.5, patience=1, min_lr=0.3, threshold=1e-4)`，指标序列 \([1.0,1.1,\dots,1.6]\)，每次 `schC.step(m)` 后记录 lr。

`StepLR(step_size=1, gamma=0.5)`；\(loss=w^2\)，\(w_0=1\)，SGD(lr=0.1)。生效 lr 序列可直接手算：A 第 \(i\) 步 \(lr=0.1\cdot0.5^i\)；B 整体前移一格 \(lr=0.1\cdot0.5^{i+1}\)；C 由 patience/min_lr 状态机决定。

## 示例

```
输入：无
输出：
{
 "lrs_opt_first": [0.1, 0.05, 0.025, 0.0125],
 "lrs_sched_first": [0.05, 0.025, 0.0125, 0.00625],
 "w_opt_first": 0.6669000387191772, "w_sched_first": 0.8232046365737915,
 "last_epoch_A": 4, "last_epoch_B": 4, "warned_skipping_first": true,
 "plateau_lrs": [1.0, 1.0, 0.5, 0.5, 0.3, 0.3, 0.3],
 "plateau_best": 1.0, "plateau_num_bad": 0, "plateau_last_epoch": 7
}
```

推演 C：指标单调变差 → `num_bad` 每步 +1，攒过 `patience=1` 的那一步降 lr（第 3、5 个指标处）；第 2 次降档想从 \(0.5\) 到 \(0.25\) 时触底 `min_lr=0.3`，第 3 次（\(0.15\)）依旧被压在 \(0.3\)。

## 约束

- lr/指标账逐项精确一致（\(0.1\cdot0.5^i\) 在 IEEE 下与框架内部计算逐位相同）；权重终值 `atol=1e-5, rtol=1e-5`。
- 两个 StepLR 变体的 `last_epoch` 相同但 lr 账不同——不许拿 last_epoch 当生效 lr 序列用。
