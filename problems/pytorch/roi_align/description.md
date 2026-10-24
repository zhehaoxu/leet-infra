# RoI 对齐（RoI Align）

## 题目描述

RoI Align（Mask R-CNN 提出）修正了 RoI Pooling 的**量化误差**：不做任何取整，ROI 与 bin
边界都保留浮点；对每个输出 bin 在其特征坐标范围内布一个 $s \times s$ 的**规则采样网格**，
每个采样点用**双线性插值**从特征图取值，最后对网格内所有采样点取**平均**。这样输出与 ROI
亚像素对齐，小目标/掩码预测的精度显著优于 RoI Pooling。

## 规则

- 实现 `solve(features, rois, pooled_size, spatial_scale, sampling_ratio, aligned)`。
  - `features` 形状 `(N, C, H, W)`；`rois` 形状 `(K, 5)`，每行 `[batch_index, x1, y1, x2, y2]`（原图坐标 float）。
  - `pooled_size` $P$（输出 $P\times P$）；`spatial_scale` $s$ 为原图→特征图缩放；`sampling_ratio` $r\ge1$ 为每 bin 每轴的采样点数；`aligned` 为布尔。
- 对每个 ROI：
  1. 映射（**不量化**）：$x_1 = X_1 s,\ y_1 = Y_1 s,\ x_2 = X_2 s,\ y_2 = Y_2 s$；若 `aligned` 为真，四项各再减 $0.5$。
  2. bin 尺寸：$\text{bin}_w=(x_2-x_1)/P,\ \text{bin}_h=(y_2-y_1)/P$。
  3. 第 $(i,j)$ 个 bin 的浮点范围：$[\,x_1+j\,\text{bin}_w,\ x_1+(j+1)\text{bin}_w)$（$y$ 同理）。
  4. 在 bin 内布 $r\times r$ 网格，第 $(a,b)$ 个采样点特征坐标：
     $$ f_x = x_{start} + (b+0.5)\frac{\text{bin}_w}{r},\qquad f_y = y_{start} + (a+0.5)\frac{\text{bin}_h}{r} $$
  5. 每个采样点对每通道做**双线性插值**：取 $\lfloor f_x\rfloor,\lfloor f_y\rfloor$ 及其 +1 的四邻域加权，
     **落在特征图外的格点值记 0**（不做 clamp-to-edge）。bin 输出 = $r^2$ 个采样点的**平均**。
- 输出形状 `(K, C, P, P)`（建议以 float64 累加以贴近参考精度）。

> 本环境无 torchvision，正确性以**独立 numpy 参考实现**为准（判题用另一套 numpy 实现核对）。
> 请严格按上述「不量化 / aligned 偏移 / 网格采样 / 越界记 0 / 平均」规则实现。

## 示例

```
输入：features = arange(16).reshape(1,1,4,4)      # 值 0..15
     rois = [[0, 0, 0, 3, 3]], pooled_size=2, spatial_scale=1.0
     sampling_ratio=2, aligned=False
   ROI 特征范围 [0,3]×[0,3]，bin_w=bin_h=1.5；每 bin 布 2×2 网格、双线性平均
输出（float64）：[[[[3.75, 5.25],
                   [9.75, 11.25]]]]
```

## 约束

- float32 输入；`pooled_size >= 1`，`sampling_ratio >= 1`，`spatial_scale > 0`
- 采样点越界时其四邻域中越界的格点贡献 0（等价双线性在 pad-zero 图上求值）
- 不使用 `torchvision.ops.roi_align` / `F.grid_sample`；需手写网格采样 + 双线性插值
