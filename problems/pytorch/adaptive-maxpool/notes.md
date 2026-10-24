# 自适应最大池化详解

## 自适应的本质：由输出反推窗口

固定 `MaxPool2d(kernel, stride)` 的输出尺寸是 $\lfloor (H-k)/s\rfloor+1$，输入一变就跟着变。
`AdaptiveMaxPool2d(out)` 把关系反过来：给定输出格数 `out`，用
$$ \text{start}(i)=\lfloor i\cdot H_{in}/H_{out}\rfloor,\quad \text{end}(i)=\lceil (i+1)\cdot H_{in}/H_{out}\rceil $$
为每个输出位切出一段。当 $H_{in}$ 不能被 $H_{out}$ 整除时，各段长度会相差 1（ceil/floor
自然吸收余数），这保证所有像素被尽量均匀覆盖、且**每个输出段非空**。

## 与 adaptive-avgpool 的唯一区别

边界公式完全相同，只是聚合从 `mean` 换成 `max`。因此：

- `avg` 平滑、保留整体能量水平；`max` 保留**最强响应**，对显著特征更敏感。
- 二者组合（concat）是一种经典的注意力/描述子增强（如 max+avg 池化拼接送分类头）。

## max 池化对负数仍成立

`max` 取的是段内**代数值最大**者，不是绝对值最大：一段全为负时取「最不负」的那个
（见 `check_max_not_mean`）。想按幅值选需用 `abs` 预处理，那是另一个算子。

## 退化情形

- `output_size=1` → 逐通道全局最大值，等价 global max pooling（很多 backbone 末端的
  `AdaptiveMaxPool2d(1)` 就是它）。
- 段可能重叠：当 $H_{out}>H_{in}$（放大）时 floor/ceil 会给出长度 1 甚至重复覆盖的段，
  PyTorch 也允许这种「上采样式」的自适应池化。
