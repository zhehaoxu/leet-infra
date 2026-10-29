# 详解：LayerNorm 反向的两条耦合路径

## 出处

LayerNorm 反向是「写一个 fused LN kernel」这类岗位的标准笔试题（fused backward 在 Triton/CUDA 面试里出现频率极高），也是推理框架做 kernel 融合时最容易在 `dgamma` 归约顺序上翻车的一处。面试官真正想听的是：为什么 dX 有两项减法，以及这两项各来自前向的哪一步。

## 用链式法则把两项减法定位到来源

对某一行，\(y_i=\gamma_i x̂_i+\beta_i\)，\(x̂_i=(x_i-\mu)/\sigma\)。设 \(g_i=dY_i\gamma_i\)，则

$$\frac{\partial L}{\partial x_j}=\sum_i g_i\frac{\partial x̂_i}{\partial x_j},\qquad
\frac{\partial x̂_i}{\partial x_j}=\frac{1}{\sigma}\Big(\delta_{ij}-\frac1H-\frac{(x_i-\mu)(x_j-\mu)}{H\sigma^2}\Big)$$

第一项 \(\delta_{ij}\) 给出 \(g/\sigma\)；第二项来自 \(\partial\mu/\partial x_j=1/H\)，给出 \(-\text{mean}(g)/\sigma\)；第三项来自 \(\partial\sigma/\partial x_j=(x_j-\mu)/(H\sigma)\)，给出 \(-\hat x_j\text{mean}(g\circ\hat x)/\sigma\)。**三项都保留、后两项带负号**，这就是标准写法；把第三项写成 \(\hat x_j\text{mean}(g\circ\hat x)/\sigma\)（漏负号）是最常见的错法，它在前向输出对称时才恰好掩盖。

## 核心与易错点

- **有偏方差**：`unbiased=False`，分母是 \(H\) 而不是 \(H-1\)；抄 BatchNorm 的估计量会整体缩放。
- **dGamma/dBeta 沿前导轴折叠**：`(B,T,H)` 要 sum 到 `(H,)`，用 `dim=tuple(range(x.dim()-1))` 而不是硬编码 `dim=0`。
- **一维输入**：`x` 为 `(H,)` 时折叠轴为空集，必须返回 `(H,)` 而非标量。
- **beta 的梯度就是折叠后的 dY**，与 `x`、`gamma` 都无关——测试里用「改 beta 结果不变」来钉死这一点。

## 与邻近题的区别

`ag-rmsnorm-backward` 少了减均值，因此 dX 只剩一项减法；`ag-batchnorm-backward` 沿 batch 轴而非特征轴归约，并且 `dx` 还要考虑 batch 统计量本身依赖 `x`（三重耦合）。`ag-weightnorm-backward` 则是重参数化，与归一化轴无关。

## 对拍口径与复杂度

oracle 用 `F.layer_norm(x, (H,), gamma, beta, eps)` 加 `torch.autograd.grad` 取三个梯度，float64 atol=1e-9。示例刻意取 `eps=0.0` 且 \(x\) 为 \(\pm1\)，使 \(\mu=0,\sigma=1\)，可完全手算。复杂度 \(O(BTH)\) 时间，两次沿末轴归约 + 一次沿前导轴折叠。
