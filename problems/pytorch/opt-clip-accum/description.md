# 优化器：梯度累积 + 全局范数裁剪 + 动量 SGD

## 题目描述

显存不够时，我们用**梯度累积**（gradient accumulation）：连续跑若干个「微批次」（micro-batch）把梯度累加起来，再当成一个大批次更新一次。大模型训练还几乎必开**梯度范数裁剪**（gradient clipping）：当所有参数拼起来的全局梯度范数超过阈值时，整体等比缩小，防止梯度爆炸把训练带偏。二者再叠上经典的**动量 SGD**，就是这套题要实现的组合机制。

本题要求你**继承 `torch.optim.Optimizer`**，把「累积 → 求均值 → 全局范数裁剪 → 动量更新」封装进一个 `step()`。评测逐微批次喂梯度、比对最终参数。

## 规则

- 工厂函数 `solve(params, lr, momentum=0.0, accum_steps=1, max_norm=1.0)`，返回 `torch.optim.Optimizer` 子类实例，`step()` 就地更新 `params`。
- 每个参数维护两份状态：累积器 \(a=0\)、动量缓冲 \(b=0\)；调度器级维护微步计数 \(m\)（初始 \(0\)）。
- 每次 `step()`（对应一个微批次）：
  1. 对每个**有梯度**的参数 \(p\)（梯度 \(g\)）：\(a \leftarrow a + g\)。
  2. \(m \leftarrow m+1\)。若 \(m <\) `accum_steps`，**本次不更新参数**，直接返回（继续累积）。
  3. 否则（凑满一个大批次）：
     - 有效梯度（均值）\(\bar g_p = a_p / \texttt{accum\_steps}\)。
     - **全局范数**（跨所有参与参数）：\(\|\bar g\|=\sqrt{\sum_p \|\bar g_p\|_2^2}\)。
     - 裁剪系数 \(c=\min\!\left(\dfrac{\texttt{max\_norm}}{\|\bar g\|+10^{-6}},\ 1\right)\)（与 `torch.nn.utils.clip_grad_norm_` 同口径：只在超阈时缩小，否则不动）。
     - 裁剪后梯度 \(\hat g_p = \bar g_p \cdot c\)。
     - 动量：\(b_p \leftarrow \text{momentum}\cdot b_p + \hat g_p\)；更新 \(p \leftarrow p - lr\cdot b_p\)。
     - 复位：把每个参与参数的 \(a_p\leftarrow 0\)；\(m\leftarrow 0\)。
- `grad is None` 的参数：既不累积也不参与全局范数、也不更新。全程 float32，禁止调用现成优化器或 `clip_grad_norm_`。

## 示例

两个参数 \(p_1\in\mathbb R^{3\times2},\,p_2\in\mathbb R^{4}\)（`torch.manual_seed(7)` 下 `randn` 初始化），`lr=0.1`、`momentum=0.9`、`accum_steps=4`、`max_norm=1.0`。喂入 12 个微批次梯度（每步对每个参数 `randn_like * 2.0`），即 3 次有效更新后：

```
p1 ≈ [[-0.211246,  0.827588],
      [ 0.988395, -0.952307],
      [ 1.620651, -0.932862]]
p2 ≈ [-0.524546, 1.210246, 0.033766, -1.750957]
```

（梯度被 \(2.0\) 放大、全局范数远超 1.0，故每次更新都被裁剪到范数≈1，再累积动量。）

## 约束

- 累积、均值、裁剪、动量的顺序**必须**如上；先求均值再算全局范数、再裁剪。
- 须与参考实现逐参数一致（对拍口径见 notes.md）。
- 支持 1D 与多维混合参数；`accum_steps=1` 时退化为「单步即裁剪+动量更新」。
