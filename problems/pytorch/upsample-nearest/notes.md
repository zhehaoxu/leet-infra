# 最近邻上采样详解

## 索引映射：为什么是 `i // scale`

放大 `scale` 倍时，输出坐标 `i` 要回指到输入坐标。PyTorch 的 `nearest`（以及底层
`downsample_nearest`）用的是**比例映射后向下取整**：

$$ \text{src}(i) = \Big\lfloor i \cdot \frac{H_{in}}{H_{out}} \Big\rfloor = \Big\lfloor \frac{i}{\text{scale}} \Big\rfloor $$

整数倍放大时 $H_{out}=\text{scale}\cdot H_{in}$，于是就是简单的 `i // scale`——每段连续的
`scale` 个输出落到同一个输入像素，形成 `scale×scale` 的复制方块。

## 与双线性/插值的取舍

- **不产生新数值**：输出值集合恒等于输入，适合标签图 / 语义分割 mask / 离散 token 网格，
  插值会把类别索引平均成无意义的中间值。
- **块状锯齿**：因为没有平滑，放大倍率大时能看到明显方块；需要平滑就改用
  `bilinear`/`bicubic`（见「双线性插值上采样」与「双线性网格采样」两题）。

## 注意 align_corners 只影响线性类插值

`nearest` 与 `align_corners` 无关；只有 bilinear/bicubic 才有角点对齐的歧义。

## 高效写法

无需三重循环，用 repeat_interleave 或直接索引即可：
`x.repeat_interleave(sh, dim=2).repeat_interleave(sw, dim=3)`，
或展开成 `torch.gather`。教学实现里保留显式循环只为突出「复制」这一本质。
