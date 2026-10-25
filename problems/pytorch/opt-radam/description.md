# 优化器：RAdam 整流自适应矩估计（Rectified Adam）

## 题目描述

Adam 在训练初期的方差估计噪声很大，导致自适应学习率剧烈抖动，常常需要一个手工的 warmup 来稳定。RAdam（arXiv:1908.03213）把 warmup **内生化**：它显式地估计二阶矩的方差，用一个随步数变化的**稳定系数 \(r_t\)** 去「整流」学习率——早期方差不可靠时 \(r_t\) 逼近甚至关闭自适应项，等简单滑动平均长度 \(n_t\) 足够长后再切换回标准 Adam。

本题要求你**继承 `torch.optim.Optimizer`** 实现 RAdam（`degenerated_to_sgd=False` 变体），通过 `step()` 就地更新参数；评测喂固定梯度序列、比对最终参数与独立参照是否逐元素一致。

## 规则

- 实现工厂函数 `solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, n_sma_threshold=5)`，返回一个 `torch.optim.Optimizer` 子类实例；`params` 是待优化张量列表，`step()` 须**就地更新**这些张量本身。
- 每个参数 \(p\)、梯度 \(g\)，状态 \(m, v\)（初始零），第 \(t\) 步（\(t\) 从 1 计，**每个参数各自计数、仅在该参数有梯度时自增**）：
  1. 一阶动量：\(m \leftarrow \beta_1 m + (1-\beta_1)\, g\)
  2. 二阶矩：\(v \leftarrow \beta_2 v + (1-\beta_2)\, g^2\)
  3. 有效平均长度：令 \(n_{\max}=\dfrac{2}{1-\beta_2}-1\)，\(\beta_2^t=\beta_2^{\,t}\)，\(n_t = n_{\max}-\dfrac{2\,t\,\beta_2^{\,t}}{1-\beta_2^{\,t}}\)
  4. **整流分支**（当 \(n_t \ge \texttt{n\_sma\_threshold}\) 时）：稳定系数
     \[ r_t=\sqrt{\frac{(1-\beta_2^{\,t})(n_t-4)}{n_{\max}-4}\cdot\frac{n_t-2}{n_t}\cdot\frac{n_{\max}}{n_{\max}-2}} \]
     步长 \(\text{step}=\dfrac{lr\cdot r_t}{1-\beta_1^{\,t}}\)，更新 \(p \leftarrow p - \text{step}\cdot \dfrac{m}{\sqrt{v}+\epsilon}\)。
  5. **未达阈值**：\(r_t=-1\) 使步长为负，此时**整步跳过**——不动 \(p\)、不施加权重衰减（这是 RAdam 内建 warmup 的直接体现）。
- 权重衰减采用解耦式（仅当 `weight_decay`≠0，**且只在整流分支真正生效的那一步**施加）：\(p \leftarrow p\,(1 - lr\cdot\text{wd})\)。全程 float32。`grad is None` 的参数整体跳过。**禁止**调用现成优化器。

## 示例

初始化单个参数 \(p=[1.0,\,-2.0,\,0.5]\)，`lr=0.1`、`betas=(0.9, 0.999)`、`eps=1e-8`、`weight_decay=0.01`、`n_sma_threshold=5`，连续喂入 **8 步**梯度（前 4 步因未达阈值而几乎不动）：

```
g₁=[0.10,-0.20,0.30]   g₂=[-0.40,0.05,0.20]  g₃=[0.15,-0.10,-0.25] g₄=[0.05,0.30,-0.10]
g₅=[-0.20,-0.10,0.40]  g₆=[0.30,0.10,-0.20]  g₇=[-0.10,0.25,0.05]  g₈=[0.20,-0.15,-0.30]
八步后 p ≈ [0.992954, -1.998415, 0.503093]
```

## 约束

- 输入张量 dtype 为 float32；输出就地更新、dtype 不变
- 支持任意形状参数；每个参数各维护一份 \((m, v)\) 与独立步数计数
- 未达阈值的步必须完全 no-op（连权重衰减都不做）
- 与独立参照逐步完全一致（口径见 notes.md）
