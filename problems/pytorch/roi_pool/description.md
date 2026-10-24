# RoI 池化（RoI Pooling）

## 题目描述

RoI Pooling 是 Faster R-CNN 里的关键算子：把候选框（Region of Interest，ROI）在**原图**
坐标系下的矩形，按 `spatial_scale` 映射到特征图，再把这个（量化到整数格点的）矩形**均匀切成
`pooled_size × pooled_size` 个 bin，每个 bin 取窗口内最大值**，从而把任意大小的 ROI 归一成
固定尺寸的 feature。它的「量化取整」会带来位置误差，正是后续 RoI Align 要消除的。

## 规则

- 实现 `solve(features, rois, pooled_size, spatial_scale)`。
  - `features` 形状 `(N, C, H, W)`。
  - `rois` 形状 `(K, 5)`，每行 `[batch_index, x1, y1, x2, y2]`，坐标在**原图**尺度（float）。
  - `pooled_size` 为整数 $P$（输出 $P\times P$，方阵）；`spatial_scale` 为原图→特征图的缩放系数。
- 对每个 ROI：
  1. 映射并量化到特征图：$f_{x1}=\lfloor x_1 s\rfloor,\ f_{y1}=\lfloor y_1 s\rfloor,\ f_{x2}=\lfloor x_2 s\rfloor,\ f_{y2}=\lfloor y_2 s\rfloor$。
  2. 把整数矩形 $[f_{x1},f_{x2}]\times[f_{y1},f_{y2}]$ 用**整数除法**切成 $P\times P$ 个 bin，
     第 $(i,j)$ 个 bin 的行范围 $[\,f_{y1}+(f_{y2}-f_{y1})\lfloor i/P\rfloor,\ \dots)$，即
     $$ y_{lo}=f_{y1}+(f_{y2}-f_{y1})\cdot i\ /\ P\ (\text{整除}),\quad y_{hi}=f_{y1}+(f_{y2}-f_{y1})\cdot(i+1)\ /\ P $$
     $x$ 同理。
  3. bin 若为空（$y_{hi}\le y_{lo}$ 或 $x_{hi}\le x_{lo}$）则输出 0；否则把范围**裁剪**到
     $[0,H]\times[0,W]$，对每个通道取窗口内 **max**。
- 输出形状 `(K, C, P, P)`。

> 本环境无 torchvision，正确性以**独立 numpy 参考实现**为准（判题用另一套 numpy 实现核对）。
> 请严格按上述量化 / 整除 / 裁剪 / max 规则实现。

## 示例

```
输入：features = arange(16).reshape(1,1,4,4)
       [[ 0, 1, 2, 3]
        [ 4, 5, 6, 7]
        [ 8, 9,10,11]
        [12,13,14,15]]
     rois = [[0, 0, 0, 3, 3]],  pooled_size=2, spatial_scale=1.0
   fx1=0,fy1=0,fx2=3,fy2=3；整除切 2×2：
     bin(0,0)=rows[0,1)cols[0,1)={0}          max=0
     bin(0,1)=rows[0,1)cols[1,3)={1,2}        max=2
     bin(1,0)=rows[1,3)cols[0,1)={4,8}        max=8
     bin(1,1)=rows[1,3)cols[1,3)={5,6,9,10}   max=10
输出：[[[[0, 2], [8, 10]]]]   形状 (1,1,2,2)
```

## 约束

- float32 输入；`batch_index` 为整数索引（取 `int`）
- `pooled_size >= 1`；`spatial_scale > 0`；bin 边界用**整数地板除**
- 越界 ROI 范围裁剪到特征图边界；空 bin 输出 0
- 不使用 `torchvision.ops.roi_pool`；需手写量化分 bin + 窗口 max
