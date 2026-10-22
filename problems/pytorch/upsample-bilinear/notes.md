# 双线性插值上/下采样详解

## 定义

把 $(N,C,H,W)$ 重采样到目标 size，每个新像素取其源坐标周围 **4 个邻居**的加权平均：

$$ \text{out}(y,x)=(1-w_y)(1-w_x)v_{00}+(1-w_y)w_x v_{01}+w_y(1-w_x)v_{10}+w_y w_x v_{11} $$

权重由源坐标的小数部分决定；先在行方向线性、再在列方向线性（或反之，等价）。

## 核心坑：`align_corners`（坐标映射两套公式）

输出第 $i$ 个像素映射回输入的源坐标，取决于这个开关：

- **`align_corners=True`**：让**角对角**对齐，scale $=\dfrac{H_{in}-1}{H_{out}-1}$，$\text{src}=i\cdot\text{scale}$。两端点精确落在输入角像素上。
- **`align_corners=False`（torch 默认）**：按像素**中心**对齐，scale $=\dfrac{H_{in}}{H_{out}}$，$\text{src}=(i+0.5)\cdot\text{scale}-0.5$。保持「像素尺寸」一致，但角不齐。

两者数值全不同；判题必须对齐同一口径。忘了 `-0.5` 或用错 scale 是最常见 bug。

## 边界处理

- 源坐标 `clamp` 到 $[0,H_{in}-1]$；$y_1=(y_0+1).\text{clamp}(\max=H_{in}-1)$ —— 边缘行/列重复，避免越界。
- size 可为 int（方形）或 `(oh,ow)`。

## 与其它插值

nearest（取最近）、bicubic（16 邻域）是同类重采样，只换核。反卷积（`ConvTranspose`）、`F.interpolate` 都覆盖这些模式；本关手写双线性最能理解「坐标映射 + 权重」的本质。
