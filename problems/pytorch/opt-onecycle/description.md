# 调度器：OneCycle 单周期学习率与动量（OneCycleLR）

## 题目描述

单调下降的调度（余弦、线性）有一个共同前提：lr 从最大值开始一路降。Smith & Topin 的 **1cycle 策略**（arXiv:1708.07120，"Super-Convergence"）反过来：从一个小 lr **先升**到峰值 `max_lr`，**再降**到一个远低于起点的最小值，全程只走一个峰；同时动量沿**反对称**方向走——lr 升时动量降、lr 降时动量升。下限由 `final_div_factor` 单独控制，可以远低于 base_lr，这正是"超大学习率快速收敛"的配方。

本题要求你**继承 `torch.optim.lr_scheduler.LRScheduler`** 实现这个调度器，并且**同时写回学习率和动量**两项。评测按「构造完成即第 0 步、每调用一次 `step()` 前进到下一步」的约定，把 `param_groups[0]` 里的 `lr` 与 `momentum` 序列读出，与**原生 `torch.optim.lr_scheduler.OneCycleLR`**（逐项 `10^{-12}`）以及独立重算对拍。

## 规则

- 工厂函数 `solve(params, max_lr=0.1, total_steps=10, pct_start=0.3, div_factor=25.0, final_div_factor=1e4, anneal_strategy="cos", cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, three_phase=False)`：内部用 `torch.optim.SGD(params, lr=max_lr/div_factor, momentum=max_momentum)` 建优化器，返回其上的 `LRScheduler` 子类实例；`step()` 只改超参，不动参数值。
- 记 `initial = max_lr / div_factor`、`floor = initial / final_div_factor`。两阶段（`three_phase=False`）的段上界为
  \[ \text{end}_1=\text{pct\_start}\cdot\text{total\_steps}-1,\qquad \text{end}_2=\text{total\_steps}-1 \]
  段内 lr 依次走 \( \text{initial}\to\text{max\_lr}\to\text{floor} \)，momentum 依次走 \( \text{max\_momentum}\to\text{base\_momentum}\to\text{max\_momentum} \)。
  三阶段（`three_phase=True`）多一段：\(\text{initial}\to\text{max\_lr}\to\text{initial}\to\text{floor}\)，其中第三段上界 \(\text{end}_3=\text{total\_steps}-1\)、第二段上界 \(2\,\text{pct\_start}\cdot\text{total\_steps}-2\)，momentum 走 \(\text{max}\to\text{base}\to\text{max}\to\text{max}\)。
- 落在第 \(i\) 段时（\(t\le \text{end}_i\)，首段的左端点是 \(0.0\)）：
  \[ \text{pct}=\frac{t-\text{end}_{i-1}}{\text{end}_i-\text{end}_{i-1}}\in[0,1],\qquad
  \text{anneal}(a,b,\text{pct})=\begin{cases}a+(b-a)\dfrac{1-\cos(\pi\,\text{pct})}{2}, & \texttt{cos}\\[4pt] a+(b-a)\,\text{pct}, & \texttt{linear}\end{cases} \]
  lr 用 \(\text{anneal}(a_{lr},b_{lr},\text{pct})\)、momentum 用**同一个 pct** 的 \(\text{anneal}(a_m,b_m,\text{pct})\)；`cycle_momentum=False` 时 momentum 一律不动。
- 段边界是**浮点**（不要取整），\(t\) 超出 `total_steps-1` 后 pct 夹紧为 1（保持终点值）。
- **禁止**调用 `OneCycleLR` / `SequentialLR` / `LambdaLR` 等现成调度器。

## 示例

`max_lr=1.0`、`total_steps=10`、`pct_start=0.3`、`div_factor=10.0`、`final_div_factor=100.0`、`anneal_strategy="cos"`、`cycle_momentum=True`、`base_momentum=0.8`、`max_momentum=0.95`，读出 \((lr,\ momentum)\) 于 \(t=0..9\)：

```
t=0 (0.1,      0.95 )   t=5 (0.611649, 0.858311)
t=1 (0.55,     0.875)   t=6 (0.389351, 0.891689)
t=2 (1.0,      0.8  )   t=7 (0.189067, 0.921762)
t=3 (0.950534, 0.807427) t=8 (0.050466, 0.942573)
t=4 (0.811933, 0.828238) t=9 (0.001,    0.95 )
```

第 2 步同时是 lr 峰（1.0）与动量谷（0.8）；终点 lr \(=0.001\) 远低于起点 \(0.1\)。

## 约束

- `pct_start ∈ (0,1)`、`div_factor > 1`、`final_div_factor ≥ 1`、`base_momentum ≤ max_momentum`，`total_steps ≥ 2`
- 须与原生 `OneCycleLR` 逐步完全一致（lr 与 momentum 都是，对拍口径见 notes.md）
- 全程 float，无张量状态
