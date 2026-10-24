# autograd 求导入门 详解

## 自动微分是什么

PyTorch 对每个参与运算且 `requires_grad=True` 的张量**动态建一张计算图**，记录前向的算子链；`.backward()` 沿图反传，把 $\partial(\text{标量})/\partial(\text{叶子})$ 累加进 `.grad`。无需手写导数公式。

## 本题三段式

1. **造叶子**：`leaf = x.detach().requires_grad_(True)`。
2. **前向**：按 `expr` 组合出 $y$（quadratic $3x^2+2x$、cubic $x^3$、exp $e^x$）。
3. **反传**：`y.sum().backward()`，读 `leaf.grad`。

## 两个必懂口径

- **为什么 `.detach()`**：入参 `x` 可能本身带 `grad_fn` 或已被别的图引用。`detach()` 切断与旧图的联系，`requires_grad_(True)` 把它变成**新叶子**——既满足「对 x 求导」，又**不污染调用方传入的张量**（不写它的 `.grad`、不改它的 `requires_grad`）。题面明令不得污染 `x`，直接 `x.requires_grad_(True)` 会改到外来张量。
- **为什么 `.sum()`**：`.backward()` 默认对**标量**求梯度。$y$ 与 $x$ 同形时，`y.sum().backward()` 得到的 `leaf.grad[i] = ∂y_i/∂x_i`（各分量对各自输入求导），正是题面要的逐元素导数。不 sum 直接 `y.backward()` 会因非标量而要求传 `gradient=`。

## 验证对拍

解析导数：$\frac{d}{x}(3x^2+2x)=6x+2$；$(x^3)'=3x^2$；$(e^x)'=e^x$。autograd 结果应与之一致。

## 坑

- `backward()` 只跑一次：同一张图二次反传需 `retain_graph=True`。
- `.grad` 是**累加**的，反复 backward 不清零会叠加（训练循环里 `zero_grad` 的由来）。
- 想让某段不被求导用 `torch.no_grad()` / `detach()`。

## 关联

optimizer-training-loop 是「构图→backward→step」的最小闭环；lstm-cell / attention 等题的前向都可被 autograd 自动求梯度。
