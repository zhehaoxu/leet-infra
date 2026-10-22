# AdaptiveAvgPool2d 详解

## 定义

不给你 kernel/stride，而是给**目标输出尺寸** $H_{out}\times W_{out}$，算子自动决定每个输出位覆盖输入的哪一段。经典用途：把任意分辨率特征图压成固定尺寸（如分类前 `adaptive(1,1)` = GAP，或 Inception 的多尺度池化）。

## PyTorch 的边界公式（本关灵魂，别用整除切段）

第 $i$ 个输出位覆盖输入的半开区间 $[s_i, e_i)$：

$$ s_i=\Big\lfloor\frac{i\cdot H_{in}}{H_{out}}\Big\rfloor,\qquad e_i=\Big\lceil\frac{(i+1)H_{in}}{H_{out}}\Big\rceil $$

- 起点 **floor**、终点 **ceil**，因此各段**长度可能不等**（有的段 1 个像素、有的 2 个），段之间可能重叠也可能留缝。输出取该段内均值。
- 常见错误：把输入等长切成 $H_{out}$ 段（`H_in//H_out`）——当 $H_{in}$ 不被 $H_{out}$ 整除时与 torch 不一致。

## 尺寸关系

- $H_{in}\ge H_{out}$ 时是降采样；$H_{in}<H_{out}$ 时会**复制/放大**（某些段长度可能为 1 甚至出现相邻同段），也允许。
- `output_size` 可为 int（方形）或 `(oh,ow)`。

## 实现路线

按上面 `_bounds` 预计算每段 (start,end)，双层循环切 `x[:,:,hs:he,ws:we].mean((2,3))` 填入。对拍 `F.adaptive_avg_pool2d`。

## 全局池化关系

`adaptive_avg_pool2d(x, 1)` 与 `global average pooling` 等价——都是对整幅求均值，见「全局池化」关。
