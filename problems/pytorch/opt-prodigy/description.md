# 优化器：Prodigy 自适应学习率（D-Adaptation）

## 题目描述

调学习率（\(lr\)）是大模型训练里最费时的一环。Defazio & Mishchenko（arXiv:2301.07733）的 **D-Adaptation** 思路是：**在线估计“到最优解的距离 \(D\)”**，用它自动决定有效步长，几乎无需手动设 \(lr\)。**Prodigy** 是该思路对 Adam 的实用化实现——它维护一个随步数增长的估计量 \(d\)，把 \(d\) 乘进等效学习率，从而“边训边学 \(lr\)”。

本题要求你**继承 `torch.optim.Optimizer`** 实现 Prodigy，`step()` 逐参数就地更新。评测喂入固定梯度序列并比对最终参数。这是合集里最复杂的一题：存在**全局标量估计 \(d\)** 与**两趟遍历**（先统计、后更新）。

## 规则

- 工厂函数 `solve(params, lr=1.0, betas=(0.9, 0.999), d0=1e-6, d_coef=1.0, growth_rate=float('inf'), weight_decay=0.0, eps=1e-8)`，返回 `torch.optim.Optimizer` 子类实例，`step()` 就地更新 `params`。
- 记 \(\beta_3=\sqrt{\beta_2}\)。组级状态：\(d\leftarrow d_0\)、\(d_{\max}\leftarrow d_0\)、\(\mathit{num}\leftarrow 0\)（标量，跨步保留）；每个参数状态：\(s=0\)、\(p_0=p\) 初值快照、\(m=0\)、\(v=0\)。每一步：
  1. 等效步长 \(d_{lr}=d\cdot lr\)（本题不启用偏差修正）。
  2. \(\mathit{num}\leftarrow \beta_3\cdot \mathit{num}\)（**先衰减**）；另置本步累加分母 \(\mathit{den}=0\)。
  3. **第一趟**（对每个有梯度的参数 \(p\)，梯度 \(g\)）：
     - \(\mathit{num}\ \mathrel{+}= \dfrac{d}{d_0}\,d_{lr}\,\langle g,\ p_0-p\rangle\)
     - \(m \leftarrow \beta_1 m + d(1-\beta_1) g\)
     - \(v \leftarrow \beta_2 v + d^2(1-\beta_2) g^2\)
     - \(s \leftarrow \beta_3 s + \dfrac{d}{d_0}\,d_{lr}\, g\)
     - \(\mathit{den}\ \mathrel{+}= \sum_i |s_i|\)
  4. 若 \(\mathit{den}=0\) 则本步不更新参数、直接结束。否则 \(\hat d = d_{coef}\cdot \mathit{num}/\mathit{den}\)（取标量），并：
     - 若 \(d==d_0\)：\(d\leftarrow\max(d,\hat d)\)
     - \(d_{\max}\leftarrow\max(d_{\max},\hat d)\)；\(d\leftarrow\min(d_{\max},\ d\cdot \texttt{growth\_rate})\)
  5. **第二趟**（对每个有梯度的参数）：解耦权重衰减（wd≠0）\(p\leftarrow p(1-d_{lr}\cdot\text{wd})\)；分母 \(\text{denom}=\sqrt{v}+d\cdot\epsilon\)；\(p\leftarrow p-d_{lr}\cdot\dfrac{m}{\text{denom}}\)。
- 注意：**第二趟用的是步首的旧 \(d_{lr}\)**，而分母里的 \(d\cdot\epsilon\) 用的是**刚更新后的新 \(d\)**。`grad is None` 参数两趟都跳过。全程 float32，禁止调用现成优化器。

## 示例

单个参数 \(p=[1.0,\,-2.0,\,0.5]\)，`lr=1.0`、`betas=(0.9,0.999)`、`d0=1e-2`、`d_coef=1.0`、`growth_rate=1.1`、`weight_decay=0.0`、`eps=1e-8`，喂入 30 步梯度（`torch.manual_seed(0)` 下 \(\texttt{randn}\cdot0.5\)）。因 \(d\) 逐步抬升、\(d_{lr}\) 随之增大，参数在早期几乎不动、后期才明显下降：

```
30 步后 p ≈ [0.852818, -2.713569, 0.255777]
```

## 约束

- dtype float32、就地更新；支持 1D 与多维混合（内积 \(\langle g,p_0-p\rangle\) 展平后点乘）
- 须与参考实现逐步一致（对拍口径见 notes.md）
