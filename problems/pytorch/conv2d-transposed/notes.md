# 转置卷积详解

## 它到底是什么

"反卷积"是误导性叫法——它不是卷积的逆运算，而是**卷积关于输入的梯度**。
设普通卷积 $y = W x$（线性算子），则转置卷积算的就是 $x' = W^\top y$：把上游每个元素
沿核**散射累加**回更大的网格。这也是 PyTorch 反向传播里对 `padding='valid'` 卷积求 `grad_input`
所用的运算。

## 尺寸公式

$$ o = (i - 1)\cdot \text{stride} - 2\cdot \text{padding} + k \;(+ \text{output\_padding}) $$

`output_padding`（本题不涉及）用于消除 stride>1 时同一输入尺寸对应多个 $o$ 的歧义。
`padding` 在转置里表示「先当普通卷积裁掉多少」，所以它**减小**输出，与普通卷积方向一致但意义相反。

## 实现要点（散射视角）

对每个输入位置 `(iy,ix)`，其贡献散布到输出的
$(iy\cdot\text{stride}-padding+ky,\; ix\cdot\text{stride}-padding+kx)$，
**越界丢弃、重叠累加**。与普通卷积的「输出中心采集输入窗口」正好反过来，容易写错方向。

- weight 约定 $(C_{in}, C_{out}, k_h, k_w)$，与 conv2d 的 $(C_{out}, C_{in},...)$ 相反。
- `stride` 时输出出现棋盘状空洞（checkerboard artifacts），因为相邻输入的贡献不连续重叠——
  这也是实践中常用「最近邻上采样 + 普通卷积」替代转置卷积的原因。

## 应用

语义分割解码器（FCN/UNet 上采样头）、VAE/GAN 生成器、超分。UNet 里也常用
`ConvTranspose2d(stride=2)` 做 2× 放大；对棋盘敏感的场景改用 `upsample+conv` 或转置卷积取 `k=stride` 的整除配置。
