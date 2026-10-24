# RoI Align 详解

## 它改进了 RoI Pooling 的什么

RoI Pooling 有两处不可逆的取整：ROI 坐标 `floor` 到整数格点、bin 边界整数除法。对小 ROI，
这几像素的位移让采到的特征与框真实位置错位，且 max 会放大这种噪声。RoI Align 的核心是
**全程不量化**：

1. ROI 与 bin 边界都是浮点；
2. 每个 bin 不取「窗口 max」，而是布 $r\times r$ 网格，对每个采样点做**双线性插值**再**平均**。

采样点落在四个整数格点之间，用距离加权 $\Rightarrow$ 亚像素级、可微、对位置扰动平滑。

## aligned 的含义

`aligned=True` 对应 `align_corners=True` 的坐标约定：ROI 映射后四项各减 $0.5$，让特征格点中心
与像素中心对齐；`aligned=False` 则不减。两种约定在工程里都出现过，本题把 `aligned` 作为显式参数，
判题按同一规则实现，避免歧义。

## 越界记 0

本实现约定采样点四邻域中**越出特征图的格点值记 0**（等价于把特征图用 0 填充后再双线性求值）。
这与「先 clamp 到边缘再插值」不同，是对 torchvision RoIAlign 越界行为的一种简化约定；因为环境无
torchvision，正确性以题面这条约定 + 独立 numpy 实现为准。

## 实现要点

- 累加用 **float64**：插值/平均涉及很多小数，float32 累积误差可能超出 `atol`。
- 网格采样中心：`f = start + (idx + 0.5) * bin / r`，注意 `+0.5` 是取每个小格中心。
- 双线性：`y0=floor(fy), x0=floor(fx)`，四邻域按 `(1-ly)(1-lx)` 等权重组合，越界项返回 0。

## 易错点

- 把 `spatial_scale` 忘了乘，或对 `sampling_ratio` 用了 `ceil(bin)` 的自适应分支（本题要求固定 `r`）。
- 平均时除以 `r*r`（不是采样到的有效点数）。
- `aligned` 分支只对 ROI 端点做 `-0.5`，别在采样点上重复减。

## 应用

Mask R-CNN 用 RoI Align 生成与框精确对齐的 mask 特征，是实例分割的关键；检测/分割/姿态等
需要像素级定位的任务几乎都从 RoI Pooling 迁到了 RoI Align。多任务时不同 head 可各用不同 `P`。
