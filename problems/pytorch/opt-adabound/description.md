# 优化器：AdaBound 动态学习率夹逼（Adaptive Bounds → SGD）

## 题目描述

自适应优化器（Adam 等）收敛快，但在很多任务上泛化不如 SGD。AdaBound（arXiv:1902.11010，微软）给出一个「两全」思路：给 Adam 的**逐坐标有效学习率**套上一对随步数收紧的上下界 \([\alpha_t,\beta_t]\)，让它们**从两端收敛到同一个常数 \( \text{final\_lr} \)**——早期界很宽，行为近似 Adam；后期界收拢到一点，行为退化为带固定学习率的 SGD，从而兼得快速早期收敛与好的最终泛化。

本题要求你**继承 `torch.optim.Optimizer`** 实现 AdaBound，通过 `step()` 就地更新参数；评测喂固定梯度序列、比对最终参数与独立参照是否逐元素一致。

## 规则

- 实现工厂函数 `solve(params, lr=1e-3, final_lr=0.1, betas=(0.9, 0.999), gamma=1e-3, eps=1e-8, weight_decay=0.0)`，返回一个 `torch.optim.Optimizer` 子类实例；`params` 是待优化张量列表，`step()` 须**就地更新**这些张量本身。`base_lr` 取构造时各组的初始 `lr`。
- 每个参数 \(p\)、梯度 \(g\)，状态 \(m, v\)（初始零），第 \(t\) 步（该参数有梯度才自增）：
  1. 偏差修正因子：\(\hat\beta_1^{(t)}=1-\beta_1^{\,t}\)，\(\sqrt{\hat\beta_2^{(t)}}=\sqrt{1-\beta_2^{\,t}}\)
  2. 动态上下界（对**有效学习率倒数**夹逼）：\( \text{flr}=\text{final\_lr}\cdot lr / \text{base\_lr}\)，\(\ \text{lower}=\text{flr}\big(1-\tfrac{1}{\gamma t+1}\big),\ \text{upper}=\text{flr}\big(1+\tfrac{1}{\gamma t}\big)\)
  3. 步长标量：\(\text{step}=\dfrac{lr\cdot\sqrt{\hat\beta_2^{(t)}}}{\hat\beta_1^{(t)}}\)
  4. **解耦权重衰减**（仅当 `weight_decay`≠0，先于方向）：\(p \leftarrow p\,(1 - lr\cdot\text{wd})\)
  5. 一阶/二阶矩：\(m \leftarrow \beta_1 m + (1-\beta_1)g\)；\(v \leftarrow \beta_2 v + (1-\beta_2)g^2\)
  6. 分母：\(\text{denom}=\sqrt{v+10^{-15}}+\epsilon\)
  7. 逐坐标夹逼后更新：\(\text{update}=\operatorname{clamp}\!\big(\text{denom}/\text{step},\ \text{lower},\ \text{upper}\big)\odot m\)，\(p \leftarrow p - \text{update}\)
- 说明：库对 \(v\) 先加常数 \(10^{-15}\) 再开方（数值稳定），最终分母再加 \(\epsilon\)。全程 float32。`grad is None` 的参数整体跳过。**禁止**调用现成优化器。

## 示例

初始化单个参数 \(p=[1.0,\,-2.0,\,0.5]\)，`lr=0.1`、`final_lr=0.1`、`betas=(0.9, 0.999)`、`gamma=0.01`、`eps=1e-8`、`weight_decay=0.01`，连续喂入五步梯度：

```
g₁=[ 0.10,-0.20, 0.30]  g₂=[-0.40, 0.05, 0.20]  g₃=[ 0.15,-0.10,-0.25]
g₄=[ 0.05, 0.30,-0.10]  g₅=[-0.20,-0.10, 0.40]
五步后 p ≈ [1.048022, -1.981064, 0.400048]
```

## 约束

- 输入张量 dtype 为 float32；输出就地更新、dtype 不变
- 支持任意形状参数；每个参数各维护一份 \((m, v)\)
- 上下界对**每个坐标独立**夹逼 \(\text{denom}/\text{step}\)（而非对最终步长）
- 与独立参照逐步完全一致（口径见 notes.md）
