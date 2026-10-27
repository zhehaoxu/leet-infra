# PEFT：LoKr Kronecker 积权重合并

## 题目描述

**LoKr**（Kronecker Product adaptation，Yeh et al., arXiv:2108.07383；PEFT 库 `tuners/lokr`）用 **Kronecker 积**构造权重增量——比 LoRA 的加法旁路更激进地压缩参数：两个小因子 \(W_1\in\mathbb R^{a\times b}\)、\(W_2\in\mathbb R^{c\times d}\)，撑起一个 \((a\,c)\times(b\,d)\) 的大增量：

\[\Delta W = s\,\bigl(W_1\otimes W_2\bigr),\qquad s=\frac{\alpha}{r}\]

Kronecker 积的**块布局**必须记牢（`torch.kron` 与数学定义一致）：

\[(W_1\otimes W_2)\,\bigl[\,i\cdot c+u,\; j\cdot d+v\,\bigr] \;=\; W_1[i,j]\cdot W_2[u,v]\]

即 \(W_1\) 的每个元素放大成一块 \(W_2\)：\(W_1\) 是"外层结构"，\(W_2\) 是"内层纹理"。PEFT 的 `make_kron(w1, w2, scale)` 一行就是 `torch.kron(w1, w2) * scale`，合并即 \(W' = W_0 + \Delta W\)。

LoKr 的参数量是 \(a\,b+c\,d\)，而 \(\Delta W\) 尺寸 \(a\,c\times b\,d\)——比如 \(W_1,W_2\) 都是 \(16\times16\) 时，仅用 512 个参数就表达 \(256\times256\) 的更新（65 536 个元素），这是它比 LoHa/LoRA 更省的来源。

## 任务

给定冻结基座 \(W_0\in\mathbb R^{(a\,c)\times(b\,d)}\)、Kronecker 因子 \(W_1\in\mathbb R^{a\times b}\)、\(W_2\in\mathbb R^{c\times d}\)、缩放参数 \(\alpha\) 和秩 \(r\)，返回**合并后权重**

\[W' = W_0 + s\,\bigl(W_1\otimes W_2\bigr),\qquad s=\alpha/r\]

**约定**：不使用任何现成 PEFT/LyCORIS 库；可用 `torch.kron`，但判题 oracle 会用显式四重循环逐位置重建，结果按 float32 容差 \(10^{-5}\) 判定。

## 提示

- 先判形状：`W0.shape == (a*c, b*d)`，其中 `(a,b)=W1.shape`、`(c,d)=W2.shape`。
- 注意块布局是「\(W_1\) 外层、\(W_2\) 内层」——行索引先按 \(W_1\) 的行分块、再按 \(W_2\) 的行排布，列同理。
- LyCORIS 风格初始化把 \(W_2\) 置零：起点 \(W'=W_0\)。
