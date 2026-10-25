# 优化器：GaLore 低秩投影梯度 AdamW

## 题目描述

AdamW 的一阶/二阶矩状态和被测张量一样大，是显存大户。GaLore（Zhao et al., arXiv:2403.03507）的思路是：梯度矩阵其实长期近似低秩，于是先把梯度投影到一个**低秩子空间**里，让矩估计与自适应更新都在这个小空间里做，最后再把更新映回原空间——状态显存从 \(O(d)\) 降到 \(O(r)\)（\(r\ll d\)）。原论文每隔 `update_proj_gap` 步用（随机化）SVD 重算投影基；本题为了在 CPU 上可判、消除一切随机性，把投影基 \(P\) 作为**显式输入**逐参数给定，只考察"投影—小空间 Adam—映回"这条主干，外加**基重算步数账本**（纯整数日程）。

本题要求你**继承 `torch.optim.Optimizer`** 实现该优化器，`step()` 就地更新参数；评测构造小参数（≤8×8）与固定梯度序列反复 `step()`，比对参数轨迹与独立重算，并核对每个投影参数的重算账本。

## 规则

- 工厂函数 `solve(params, proj, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, update_proj_gap=200)`，返回 `torch.optim.Optimizer` 子类实例：
  - `params`：待优化张量列表（float32，1D 或 2D）；
  - `proj`：与 `params` 同长的列表，每项为 `None`（该参数不投影，走原空间 AdamW）或形状 `[p.numel(), r]`（\(r\ge 0\)）的 float32 二维张量 \(P\)。约定 \(P\) 的列已正交归一，但实现**不得依赖、不得重新正交化、不得做 SVD**；
  - `update_proj_gap`：≥1 的整数，只影响账本，**不影响数值轨迹**。
- 对每个在第 \(t\) 步（\(t\) 从 1 计、逐参数独立计数）有梯度的参数 \(p\)、梯度 \(g\)，按序执行：
  1. **状态首见初始化**：\(m=0\)、\(v=0\)，形状为投影空间维数 \(r\)（不投影参数为 \(p.numel()\)，即原空间）。
  2. **账本**（仅投影参数）：若 \((t-1)\bmod \text{update\_proj\_gap}=0\)（首步 \(t=1\) 建基，此后每隔 gap 步一次），把 \(t\) 追加到 `state[p]["recompute_steps"]`（升序、无重复）。同时在优化器实例上提供纯方法 `recompute_at(t)` 返回该布尔日程。
  3. **解耦权重衰减**（仅当 `weight_decay`≠0，先于矩估计，作用在原空间）：\(p\leftarrow p\,(1-lr\cdot\text{wd})\)。
  4. **投影**（row-major 展平）：\(\tilde g = P^\top g \in \mathbb{R}^{r}\)。
  5. **投影空间矩估计**：\(m \leftarrow \beta_1 m + (1-\beta_1)\tilde g\)；\(v \leftarrow \beta_2 v + (1-\beta_2)\,\tilde g\odot\tilde g\)。
  6. **偏差修正与方向**（都在 \(\mathbb{R}^r\) 里）：\(bc_1 = 1-\beta_1^{\,t}\)，\(bc_2 = 1-\beta_2^{\,t}\)，\(\hat u = \dfrac{m/bc_1}{\sqrt{v}/\sqrt{bc_2} + \epsilon}\)。
  7. **映回并更新**（就地）：\(\Delta = \operatorname{reshape}(P\hat u,\ \text{shape}(p))\)；\(p \leftarrow p - lr\cdot\Delta\)。
- \(r=0\) 时 \(\Delta\equiv 0\)（只剩权重衰减）；`grad is None` 的参数整步跳过（连权重衰减与账本都不施加）。全程 float32；禁止调用现成优化器。

## 示例

单参数 \(p=\begin{bmatrix}1.0 & -2.0\\ 0.5 & 0.0\end{bmatrix}\)，展平后 \(d=4\)，取 \(2\times2\) 的列正交基 \(P=\tfrac{1}{\sqrt2}\begin{bmatrix}1&0\\ 0&1\\ 1&0\\ 0&-1\end{bmatrix}\)，`lr=0.1`、`betas=(0.9, 0.999)`、`eps=1e-8`、`weight_decay=0.01`、`update_proj_gap=2`，喂入三步梯度：

```
g₁ = [[ 0.10, -0.20], [ 0.30,  0.10]]
g₂ = [[-0.40,  0.05], [ 0.20, -0.10]]
g₃ = [[ 0.15, -0.10], [-0.25,  0.30]]
三步后 p ≈ [[0.903279, -1.860648], [0.404777, -0.133358]]
基重算账本 recompute_steps = [1, 3]   # (t−1) mod 2 == 0：首步建基，t=3 再起新周期
```

## 约束

- 参数 ≤8×8、\(0\le r\le p.numel()\)、全程 float32（CPU 判题）
- 矩与偏差修正在投影空间维护，更新映回原空间——两者维数不同，勿混用全空间矩
- 参数轨迹与独立重算逐元素一致（`atol=1e-5, rtol=1e-5`）；账本为整数序列，须逐步精确相等；`P=I` 时结果须与原生 AdamW 一致
