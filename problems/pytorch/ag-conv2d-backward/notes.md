# 详解：卷积反向的两条对偶路径

## 出处

`conv2d` 反向是 CNN 时代框架岗的必考题，今天仍然是「手写算子反向」的标准样本，因为它把三个容易混淆的概念压在两行公式里：相关 vs 卷积、padding 的梯度边界、以及 \(dw\) 的 batch/空间折叠。视觉岗还会追问 im2col 与 Winograd，本题只考察最基础也最容易写错的那部分。

## dx 为什么是「翻转核的相关」

前向是相关：\(y_u=\sum_k w_k x_{u+k}\)（省略通道与二维细节）。反向 \(dx_j=\sum_u dY_u\,[\text{指标满足 } j=u+k]\,w_k=\sum_k dY_{j-k}w_k\)，这正是「把核翻转后再做一次相关」的定义。工程上有两种等价写法：`conv_transpose1d/2d`（把 `weight.flip(-1,-2)` 当作核），或者对 \(dY\) 做零填充后用翻转核做普通卷积，填充量是 \(k_h-p-1\)。**忘记翻转**时，只有对称核（例如示例里全 1 的 \(2\times2\)）看不出问题——这正是测试要额外放一组非对称核随机用例的原因。

## dw 的求和范围

\(dw[co,ci,k,l]=\sum_{b,u,v}dY[b,co,u,v]\,x[b,ci,us+k-p,vs+l-p]\)，指标出界的 \((u,v)\) 直接跳过。这一条与 \(dx\) 的「丢弃越界贡献」是同一件事的两面：padding 区在 \(dw\) 里表现为「少几项」，在 \(dx\) 里表现为「那几行的梯度不进输入」。折叠轴必须包含 batch，与 `ag-matmul-backward` 里 dW 的教训一样。

## 核心与易错点

- **`stride>1` 的补零**：\(dx\) 等价于把 \(dY\) 按 `stride` 插空后再卷积，漏插空会让空间尺寸整体缩水。
- **`padding>0` 时 `dx` 的形状**：仍是 `x` 的形状，不是「去 padding 后」的形状。
- **多通道**：\(dx\) 要在 `Cout` 上求和、\(dw\) 要在 `B/H/W` 上求和，`Cin` 与 `Cout` 各自保留，四个轴混一个就全错。
- **数值**：示例用全 1 核对 \(1..9\) 求和，所有值都是整数，判题可 `atol=1e-9` 逐位比。

## 与邻近题的区别

`ag-pooling-backward` 没有可学参数、只需把梯度路由回窗口；`ag-matmul-backward` 的折叠只有一层空间轴。本题的独特点是「同一份 \(dY\) 要分别做翻转相关与不翻转相关」。

## 对拍口径与复杂度

oracle 用 `F.conv2d(x, weight, stride=..., padding=...)` 加 `torch.autograd.grad((y*dY).sum(), [x, weight])`，float64 atol=1e-9。复杂度 \(O\!\cdot\!(B\,C_{in}C_{out}HWH^2)\)，\(dx\) 与 \(dw\) 同阶。
