# 详解：RAdam 整流自适应矩估计

## 出处
RAdam 论文：*On the Convergence of Adam and Beyond*，arXiv:1908.03213（OpenReview）。算法骨架对齐 kozistr/pytorch_optimizer 的 `RAdam`（`degenerated_to_sgd=False, adam_debias=False, foreach=False` 的 per-param 路径，`weight_decouple=True`）。

## 一句话机制
Adam 自适应学习率的方差会随步数衰减，早期方差估计不可信 → 训练不稳。RAdam 推导出二阶矩估计的方差闭式，用稳定系数 \(r_t\) 缩放自适应步长：方差大（早期）时把自适应项「降温」，方差可忽略（\(n_t\) 够大）时退回标准 Adam。

## 关键顺序与陷阱（易错点）
1. **内建 warmup 的极端形式**：当 \(n_t < \text{n\_sma\_threshold}\) 且 `degenerated_to_sgd=False` 时，库把 \(r_t=-1\)，于是 `step_size<0`，整个更新块被 `if step_size>0 or n_sma>=threshold` 拦下——**这一步连权重衰减、动量累积都不做**（`m,v` 仍为零）。因此 \(\beta_2=0.999\) 时前 4 步完全 no-op，第 5 步起才真正更新。若你以为每步都在更新，结果会错。
2. \(n_{\max}=\frac{2}{1-\beta_2}-1\) 与 \(\beta_2^t\) 是纯标量；\(n_t\) 也是标量（非张量）。
3. \(r_t\) 表达式里四项相乘顺序照抄库：\(\sqrt{(1-\beta_2^t)\cdot\frac{n_t-4}{n_{\max}-4}\cdot\frac{n_t-2}{n_t}\cdot\frac{n_{\max}}{n_{\max}-2}}\)。
4. 分母 \(\sqrt{v}+\epsilon\) 用的是**未偏差修正**的 \(v\)；对 \(m\) 的修正折进 `step_size = lr·r_t/(1−β1^t)`（即 `apply_adam_debias=False` 除以 `bias_correction1`）。

## 与 Adam 的关系
达到阈值后 \(r_t\to 1\)、\(n_t\to n_{\max}\)，更新退化为 \(p \mathrel{-}= \frac{lr}{1-\beta_1^t}\cdot\frac{m}{\sqrt v+\epsilon}\)——正是不带分母偏差修正的 Adam 形式（与本项目 opt-adamw 的口径不同，注意区分）。

## 对拍口径
外部参照用独立向量化重算（显式算标量 \(n_t, r_t\)，与 Optimizer 不同的实现路径）。全程 float32，per-param 与向量化逐元素一致（示例值实测 maxdiff 0，随机多步 20 步 ×50 seed 最大偏差 ≈1.2e-7）。

## 复杂度
每参数每步 \(O(\text{numel})\) + 若干标量运算，额外两份状态 \((m, v)\)。
