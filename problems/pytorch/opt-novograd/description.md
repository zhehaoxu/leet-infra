# 优化器：NovoGrad 逐层梯度归一（Layer-wise Gradient Normalization）

## 题目描述

Adam 用逐元素的二阶矩 \(v\) 做分母，显存开销与参数同量级，且对每个坐标单独缩放。Ginsburg 等人（arXiv:1905.11286）提出 **NovoGrad**：把二阶矩**降维成“整层梯度平方和”这一个标量**，用它对整层梯度做归一化，再叠加普通动量，并对权重衰减同样做“先归一后解耦”的处理。它显存只需一份动量加每层一个标量，在大 batch、大模型上尤其实用，是 NVIDIA 训练 ResNet 时的默认优化器之一。

本题要求你**继承 PyTorch 的 `torch.optim.Optimizer` 基类**、实现一个 NovoGrad，通过 `step()` 逐参数更新。评测会构造若干待优化张量交给你的优化器，反复喂入固定梯度序列并调用 `step()`，比对最终参数是否与参考实现逐元素一致。

## 规则

- 实现工厂函数 `solve(params, lr=1e-3, betas=(0.95, 0.98), eps=1e-8, weight_decay=0.0, grad_averaging=False)`，返回一个 `torch.optim.Optimizer` 子类的实例；`params` 是待优化张量列表。实例的 `step()` 必须**就地更新** `params` 里的张量本身。
- 记第 \(t\) 步（\(t\) 从 1 计）、某参数 \(p\) 的梯度为 \(g\)、该参数**整层梯度平方和**为 \(\lVert g\rVert^2=\sum_i g_i^2\)（一个标量）。每个参数维护标量 \(s\)（层二阶矩 EMA）与同形 \(m\)（动量）：
  1. 层二阶矩：首步 \(s \leftarrow \lVert g\rVert^2\)；其后 \(s \leftarrow \beta_2 s + (1-\beta_2)\lVert g\rVert^2\)
  2. 归一化梯度：\(\hat g = \dfrac{g}{\sqrt{s}+\epsilon}\)
  3. 解耦权重衰减（仅 `weight_decay`≠0）：\(p \leftarrow p\,(1-lr\cdot\text{wd})\)
  4. 动量：首步 \(m \leftarrow \hat g\)；其后（若 `grad_averaging` 则先 \(\hat g\leftarrow(1-\beta_1)\hat g\)）\(m \leftarrow \beta_1 m + \hat g\)
  5. 参数更新：\(p \leftarrow p - lr\cdot m\)
- **注意 NovoGrad 没有偏差修正**：动量与分母都不除以 \(1-\beta^t\)；步长就是 `lr`，\(\hat g\) 前面的系数是 \(1\)（除非 `grad_averaging=True` 才乘 \((1-\beta_1)\)）。
- `grad is None` 的参数必须原样跳过（连权重衰减也不施加）。全程 float32。禁止调用 `torch.optim` 现成优化器。

## 示例

单个参数 \(p=[1.0,\,-2.0,\,0.5]\)，`lr=0.01`、`betas=(0.95, 0.98)`、`eps=1e-8`、`weight_decay=0.01`、`grad_averaging=False`，喂入三步梯度：

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
三步后 p ≈ [1.008826, -1.984079, 0.473279]
```

## 约束

- 输入 dtype 为 float32；就地更新、dtype 不变
- 支持 1D 与多维参数混合（多维参数的 \(s\) 是**对整层所有元素求和**得到的标量）
- 须与参考实现逐步完全一致（对拍口径见 notes.md）
