# 详解：Lion 符号动量更新

## 出处
Lion 论文：*Lion: Symbolic Discovery of Optimization Algorithms*，arXiv:2302.06675（Google, 2023）。算法骨架对齐 kozistr/pytorch_optimizer 的 `Lion`（eager per-param 路径，`weight_decouple=True`）。

## 一句话机制
更新方向 = \(\operatorname{sign}(\beta_1 m + (1-\beta_1) g)\)，即"旧动量与当前梯值的插值取符号"；随后动量以另一个衰减率 \(\beta_2\) 继续 EMA。因为取 sign，每个坐标这一步严格移动 \(\pm lr\)（方向由符号决定），这与 Adam 用 \(\hat m/(\sqrt{\hat v}+\epsilon)\) 做逐坐标缩放形成鲜明对比。

## 两个 beta 的分工（易错点）
- \(\beta_1\)（默认 0.9）只用于**合成这一步的方向** \(u\)，不写回状态；
- \(\beta_2\)（默认 0.99）才是真正更新动量缓冲 \(m\) 的衰减率。
- 关键顺序：先用**旧的** \(m\) 算 \(u=\operatorname{sign}(\beta_1 m + (1-\beta_1)g)\)，再执行 \(m \leftarrow \beta_2 m + (1-\beta_2)g\)。若先把 \(m\) 更新了再算方向，结果就错了。参考实现里对应 `update = m.lerp(g, 1-beta1).sign_()` 之后再 `m.lerp_(g, 1-beta2)`。

`lerp(a, b, w) = a + (b-a)w = (1-w)a + wb`，故 `m.lerp(g, 1-beta1) = beta1*m + (1-beta1)*g`。

## 解耦权重衰减
采用与 AdamW 同款：`p ← p·(1 − lr·wd)`，且在方向计算之前施加。本题不含 `fixed_decay`（即衰减按 lr 缩放），对齐论文默认。

## 对拍口径
外部参照用等价的向量化重算（每步显式做 sign 与两次 EMA）。`sign(0)=0`，故某坐标插值恰为 0 时不动。`grad is None` 参数跳过（连 wd 也不施加）。全程 float32，两步法与向量化重算逐元素一致（示例值实测 maxdiff 0）。

## 复杂度
每参数每步 \(O(\text{numel})\)，额外状态仅一份 \(m\)，显存约 Adam 的一半。
