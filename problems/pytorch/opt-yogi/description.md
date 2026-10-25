# 优化器：Yogi 符号控制的二阶矩（Sign-controlled Second Moment）

## 题目描述

Adam 的二阶矩 \(v\) 用 \(g^2\) 的指数平均，遇到稀疏/突发梯度时 \(v\) 会瞬间跳高、学习率被压死。Zaheer 等人（arXiv:1807.01874）提出 **Yogi**：把 \(v\) 的更新从“加性平均”改成**受符号控制的加法**——只在 \(v\) 与 \(g^2\) 的差异方向上、以 \((1-\beta_2)\) 的步长做**增减**，使 \(v\) 的变化更平缓、对梯度噪声更鲁棒。它是推荐系统、NLP 里常用的 Adam 变体。

本题要求你**继承 `torch.optim.Optimizer`** 实现 Yogi，通过 `step()` 逐参数就地更新。评测喂入固定梯度序列并比对最终参数。

## 规则

- 工厂函数 `solve(params, lr=1e-2, betas=(0.9, 0.999), initial_accumulator=1e-6, eps=1e-3, weight_decay=0.0)`，返回 `torch.optim.Optimizer` 子类实例，`step()` 就地更新 `params`。
- 每个参数初始化 \(m=v=\texttt{initial\_accumulator}\)（**非零**，逐元素填充常数）。第 \(t\) 步（从 1 计）、梯度 \(g\)：
  1. 解耦权重衰减（wd≠0，先做）：\(p \leftarrow p(1-lr\cdot\text{wd})\)
  2. 一阶矩：\(m \leftarrow \beta_1 m + (1-\beta_1) g\)
  3. 二阶矩（Yogi 核心）：\(v \leftarrow v - (1-\beta_2)\cdot \operatorname{sign}(v - g^2)\cdot g^2\)
  4. 偏差修正：\(bc_1=1-\beta_1^{t},\ bc_2=1-\beta_2^{t}\)，步长 \(\text{step}=lr/bc_1\)
  5. 更新：\(p \leftarrow p - \text{step}\cdot\dfrac{m}{\sqrt{v}/\sqrt{bc_2}+\epsilon}\)
- 分母算子顺序须为 \(\sqrt{v}/\sqrt{bc_2}+\epsilon\)（把 \(bc_2\) 的开方放进分母、eps 加在最后）。`grad is None` 参数整体跳过。全程 float32，禁止调用现成优化器。

## 示例

单个参数 \(p=[1.0,\,-2.0,\,0.5]\)，`lr=1e-2`、`betas=(0.9,0.999)`、`initial_accumulator=1e-6`、`eps=1e-3`、`weight_decay=0.01`，喂入五步梯度：

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
g₄ = [ 0.05,  0.30, -0.10]
g₅ = [-0.20, -0.10,  0.40]
五步后 p ≈ [1.001245, -1.980141, 0.472945]
```

## 约束

- dtype float32、就地更新；支持 1D 与多维混合
- 须与参考实现逐步一致（对拍口径见 notes.md）
