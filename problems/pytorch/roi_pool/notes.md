# RoI Pooling 详解

## 为什么需要它

两阶段检测器（Faster R-CNN）先由 RPN  propose 出一堆**任意尺寸**的候选框，但后面的全连接头
要求**固定长度**的特征向量。RoI Pooling 负责把每个 ROI 在特征图上「拍」成统一的
$P\times P$，从而接上共享的分类/回归头。

## 量化是它的原罪

RoI Pooling 的两处取整引入了**位置量化误差**：

1. ROI 坐标乘 `spatial_scale` 后 `floor` 成整数格点；
2. 每个 bin 边界再用**整数除法**离散化。

对大 ROI 这点误差不明显，但小目标上，几个像素的偏移就会让特征与框错位，损害定位精度，
而且 max 池对这种错位很敏感。此外反向时梯度只能路由到取到 max 的那个格点，量化使
forward/backward 的坐标对齐更脆弱。

## 与 RoI Align 的对比

| | RoI Pooling | RoI Align |
|---|---|---|
| 坐标量化 | floor/整除取整 | 保留浮点 |
| bin 聚合 | 整数窗口取 max | 规则采样点 + 双线性插值取**平均** |
| 小目标 | 误差明显 | 亚像素对齐 |

RoI Align（见本题合集另一题）正是为消除量化误差而生：不取整、对每个 bin 用 `sampling_ratio`
个采样点做双线性插值再平均。

## 易错点

- **bin 边界用地板除**：`ylo = fy1 + (fy2-fy1)*i // P`，与参考 numpy 完全一致；若用浮点插值就变成
  Align 的味道、对不上。
- **空 bin → 0**：当 ROI 太小、多个 bin 落到同一格或越界被裁空时，输出 0 而非报错。
- **裁剪到边界**：`max(0,·)`、`min(H/W,·)`，防止越界索引。
- **batch_index** 是 ROI 第一列，用于在 `features` 的 N 维上选图。

## 历史

SPPnet 首先用空间金字塔池化连接任意尺寸输入与固定 FC；Fast RCNN 简化为单尺度 RoI Pooling；
Faster RCNN 沿用并接入 RPN；Mask RCNN 指出量化误差问题，提出 RoI Align。本环境无 torchvision，
故以纯 torch 手写、并用独立 numpy 实现交叉验证正确性。
