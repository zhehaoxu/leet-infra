# 优化器：AdamP 尺度不变方向投影（Cross-Parameter Wedge）

## 题目描述

权重衰减在归一化层（BN/LayerNorm）等**尺度不变**的方向上是多余的：沿参数自身方向放大/缩小 \(w\) 并不改变函数，因此那一方向上的衰减既无正则作用、又与自适应更新相互干扰，易造成训练后期「过拟合式震荡」。AdamP（arXiv:2009.00329，naver）在 Adam 更新向量里**剔除与参数方向平行的（径向）分量**，只保留真正改变方向的切向分量，并对尺度不变参数施以更小的权重衰减（乘以 `wd_ratio`）。

本题要求你**继承 `torch.optim.Optimizer`** 实现 AdamP，通过 `step()` 就地更新参数；评测喂固定梯度序列、比对最终参数与独立参照是否逐元素一致。

## 规则

- 实现工厂函数 `solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, delta=0.1, wd_ratio=0.1)`，返回一个 `torch.optim.Optimizer` 子类实例；`params` 是待优化张量列表，`step()` 须**就地更新**这些张量本身。固定 `nesterov=False`、`cautious=False`、`weight_decouple=True`。
- 每个参数 \(p\)、梯度 \(g\)，状态 \(m, v\)（初始零），第 \(t\) 步（该参数有梯度才自增）：
  1. 一阶/二阶矩：\(m \leftarrow \beta_1 m + (1-\beta_1)g\)；\(v \leftarrow \beta_2 v + (1-\beta_2)g^2\)
  2. 令 \(\sqrt{\hat\beta_2^{(t)}}=\sqrt{1-\beta_2^{\,t}}\)，\(\text{step}=\dfrac{lr}{1-\beta_1^{\,t}}\)，更新向量
     \[ \text{perturb} = \frac{\sqrt{\hat\beta_2^{(t)}}}{\sqrt{v}+\epsilon}\odot m \]
  3. **投影（仅当 `p.dim() > 1`）**：设 \(r=1.0\)（衰减比例），依次按「逐通道 view」和「整体 layer view」判定，若命中则剔除径向分量并置 \(r=\texttt{wd\_ratio}\)，随即结束（详见下）。
  4. **解耦权重衰减**（仅当 `weight_decay`≠0）：\(p \leftarrow p\,(1 - lr\cdot\text{wd}\cdot r)\)
  5. 更新：\(p \leftarrow p - \text{step}\cdot \text{perturb}\)
- **投影判据与剔除**：把张量按 \(v(\cdot)\) 展平成二维视图（channel view：\((C, \prod_{i>0}d_i)\)；layer view：\((1, \text{numel})\)）。计算 \(\big|\cos(g, p)\big|\)（按视图逐行，分母下限 \(\epsilon\)）。若
  \[ \max|\cos| < \frac{\texttt{delta}}{\sqrt{\text{视图列数}}} \]
  则认为该组是尺度不变的：令 \(\hat n = p / (\|p\|_{\text{row}} + \epsilon)\)（逐行归一），径向分量 \(\text{radial}=(\hat n\odot \text{perturb})\) 按行求和，
  \[ \text{perturb} \leftarrow \text{perturb} - \hat n \cdot \text{radial}, \quad r \leftarrow \texttt{wd\_ratio} \]
  一旦某个视图命中就 `break`，不再检查另一个视图。若两个视图都不命中，\(r\) 保持 1.0、perturb 不变。
- 全程 float32。`grad is None` 的参数整体跳过。**禁止**调用现成优化器（可自行使用 `torch.nn.functional.cosine_similarity` 或手写余弦）。

## 示例

初始化单个 2×3 参数 \(p=\begin{bmatrix}0.5&-0.2&0.1\\0.3&0.4&-0.1\end{bmatrix}\)，`lr=0.1`、`betas=(0.9, 0.999)`、`eps=1e-8`、`weight_decay=0.01`、`delta=0.1`、`wd_ratio=0.1`，连续喂入五步梯度（每步为同形状 \(2\times3\)）：

```
g₁=[[ 0.20,-0.10, 0.05],[ 0.10, 0.30,-0.20]]  g₂=[[-0.15, 0.20, 0.10],[ 0.05,-0.10, 0.30]]
g₃=[[ 0.10, 0.05,-0.20],[-0.30, 0.15, 0.10]]  g₄=[[ 0.25,-0.05, 0.10],[ 0.10,-0.20,-0.15]]
g₅=[[-0.10, 0.30,-0.05],[ 0.20, 0.10, 0.20]]
五步后 p ≈ [[ 0.268631, -0.252074, -0.082198],
            [ 0.129582,  0.171029, -0.093757]]
```

## 约束

- 输入张量 dtype 为 float32；输出就地更新、dtype 不变
- 一维及零维参数**不做投影**（\(r\) 恒为 1.0）；二维及以上走投影判据
- 与独立参照逐步完全一致（口径见 notes.md）
