# 优化器：Cautious 符号一致性掩码（Cautious Optimizer）

## 题目描述

动量优化器的更新方向 \(u_t\) 来自历史梯度的滑动平均，某些坐标上 \(u_t\) 可能与**当前**梯度 \(g_t\) 符号相反——即动量正把参数往“反着最新梯度”的方向推。Liang 等人（arXiv:2411.16085）发现：**只需一行代码**——把更新方向里与当前梯度符号不一致的坐标**清零**、再按存活比例重新缩放——就能显著提升 Adam/AdamW 等的收敛。这就是 **Cautious Optimizer**（如 C-AdamW）。

本题要求你**继承 `torch.optim.Optimizer`** 实现一个 Cautious AdamW：先算 AdamW 的更新方向，再套上谨慎掩码。`step()` 逐参数就地更新，评测喂入固定梯度序列比对最终参数。

## 规则

- 工厂函数 `solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)`，返回 `torch.optim.Optimizer` 子类实例，`step()` 就地更新 `params`。
- 第 \(t\) 步（从 1 计）、梯度 \(g\)，先按 AdamW 求**更新方向** \(u\)（尚未乘 \(lr\)）：
  1. 解耦权重衰减（wd≠0，先做）：\(p \leftarrow p(1-lr\cdot\text{wd})\)
  2. \(m \leftarrow \beta_1 m+(1-\beta_1)g\)；\(v \leftarrow \beta_2 v+(1-\beta_2)g^2\)
  3. \(bc_1=1-\beta_1^{\,t},\ bc_2=1-\beta_2^{\,t}\)，方向 \(u=\dfrac{m/bc_1}{\sqrt{v}/\sqrt{bc_2}+\epsilon}\)
  4. **谨慎掩码**：\(M=\mathbb{1}[\,u\odot g>0\,]\)（逐元素指示 \(u\) 与 \(g\) 同号），\(M \leftarrow M\cdot\dfrac{N}{\sum M+1}\)（\(N\) 为该参数元素个数）
  5. 缩放并更新：\(u \leftarrow u\odot M\)；\(p \leftarrow p - lr\cdot u\)
- \(M\) 的重缩放分母是 \(\sum M+1\)（防零除），务必照此实现。`grad is None` 参数整体跳过。全程 float32，禁止调用现成优化器。

## 示例

单个参数 \(p=[1.0,\,-2.0,\,0.5]\)，`lr=0.1`、`betas=(0.9,0.999)`、`eps=1e-8`、`weight_decay=0.01`，喂入三步梯度：

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
三步后 p ≈ [0.978047, -1.828335, 0.326713]
```

## 约束

- dtype float32、就地更新；支持 1D 与多维混合
- 须与参考实现逐步一致（对拍口径见 notes.md）
