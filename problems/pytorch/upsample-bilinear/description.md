# 双线性插值上采样

## 题目描述

Bilinear 采样把源图每个通道看作网格上的双线性函数，在目标像素坐标处求值。核心是**坐标映射**，PyTorch 两种口径：

- `align_corners=False`（默认）：把像素当作面积元，目标像素中心映射回源坐标
  $$ x_{\text{src}} = (x_{\text{dst}} + 0.5)\cdot\frac{H_{\text{src}}}{H_{\text{dst}}} - 0.5 $$
- `align_corners=True`：四个角严格对齐
  $$ x_{\text{src}} = x_{\text{dst}} \cdot \frac{H_{\text{src}}-1}{H_{\text{dst}}-1} $$

拿到源坐标 $(y, x)$ 后，取周围四个整数格点按面积权重混合：

$$
v = (1-w_y)(1-w_x)v_{00} + (1-w_y)w_x v_{01} + w_y(1-w_x)v_{10} + w_y w_x v_{11}
$$

## 规则

- 实现 `solve(x, size, align_corners=False)`，x 形状 `(N, C, H, W)`，`size = (oh, ow)`，支持上采样与下采样。
- 源坐标先 `clamp` 到 $[0, H-1]$（映射可能为负或超界），再 floor 取左下角、小数部分作权重；上层格点索引同样要 clamp。
- 禁止调用 `F.interpolate` / `F.grid_sample`；用按行/列索引四邻域 + 权重广播即可全向量化。
- **陷阱**：两种 `align_corners` 在同一输入上除角点外几乎处处不同；用错映射公式时下采样结果尤其明显。

## 示例

```
输入：x = [[[1,2],[3,4]]], size=(4,4), align_corners=False
输出：[[1.00, 1.25, 1.75, 2.00],
      [1.50, 1.75, 2.25, 2.50],
      [2.50, 2.75, 3.25, 3.50],
      [3.00, 3.25, 3.75, 4.00]]
```

## 约束

- float32/float64；$1 \le$ 目标尺寸，任意放大/缩小倍数
- 与 `torch.nn.functional.interpolate(x, size=..., mode="bilinear", align_corners=...)` 一致
