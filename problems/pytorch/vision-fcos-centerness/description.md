# 视觉检测：FCOS center-ness 与多尺度层分配

## 题目描述

FCOS（Tian et al., *FCOS: Fully Convolutional One-Stage Object Detection*, arXiv:1904.01355）去掉了 anchor，直接在特征图的每个位置上做逐点检测：位置 \((x,y)\) 到 GT 四条边的距离 \((l,t,r,b)\) 就是回归目标。它带来两个必须自己设计的问题，也是本题的全部内容：

- **多尺度层分配**：同一张图在多个 stride 的层上都有位置，哪些位置负责哪个尺寸的物体？FCOS 的答案是「回归范围」（regress range）——每层只负责 \( \max(l,t,r,b) \) 落在某个区间内的目标，超出区间的点既不回传回归损失也不当负样本，而是**忽略**。
- **center-ness**：逐点分类会产出一大批「框内但离中心很远」的低质量预测，于是 FCOS 给每个正样本位置附加一个软标签 \( \sqrt{\frac{\min(l,r)}{\max(l,r)}\cdot\frac{\min(t,b)}{\max(t,b)}} \)，训练时乘到分类 loss 上，推理时再乘到置信度上，把边角位置的响应压下去。

此外还要实现 center sampling：只有落在 GT 中心区域（半径 = `center_radius` × 该层 stride 的正方形）内的点才算正样本，框内其余点记忽略。多个 GT 都能接收某个点时，按 FCOS 原文取**面积最小**的那个——这样小目标不会被大目标吞掉。

全 CPU、确定性、不调用随机数。

## 规则

完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。

- 采样点以**绝对像素坐标**给出，并逐点带上所属 `levels`；每层的 stride 与尺度区间分别由 `strides`、`regress_ranges` 提供，长度都等于层数。层下标越界、长度不一致、stride 非正、区间不满足 `lo < hi`、`center_radius` 为负都抛 `ValueError`。
- 对点 \(p=(x,y)\) 与 GT \(g=(x_1,y_1,x_2,y_2)\) 定义四个距离（**可以为负**，表示点在框外）：
  \[
  l=x-x_1,\quad t=y-y_1,\quad r=x_2-x,\quad b=y_2-y .
  \]
- 三个判定都是逐 GT 的：
  - `inside(g)`：\(l,t,r,b\ge 0\)，即点在框内（**含边界**）；
  - `in_center(g)`：\( |x-\frac{x_1+x_2}{2}|\le \rho \) 且 \( |y-\frac{y_1+y_2}{2}|\le \rho \)，其中 \( \rho=\text{center\_radius}\times\text{该点所在层的 stride} \)；
  - `in_range(g)`：\( \mathrm{lo}\le \max(l,t,r,b)<\mathrm{hi} \)，区间取该点所在层的 `regress_ranges`，**左闭右开**；这里的 \(\max\) 是**该 GT 自己的**四个距离。
  - 候选 GT = 三者同时成立。
- 分配：有候选时取**面积最小**者（面积 \( =\max(x_2-x_1,0)\cdot\max(y_2-y_1,0) \)，并列取 GT 下标最小者），标签为该 GT 下标；无候选但点落在至少一个 GT 框内 → 标签 -2（忽略）；否则标签 -1（负）。\(G=0\) 时全部 -1。
- 正样本的 center-ness 用**被分配 GT** 的四个距离算：
  \[
  c=\sqrt{\frac{\min(l,r)}{\max(l,r)}\cdot\frac{\min(t,b)}{\max(t,b)}},
  \]
  任一 \(\max\) 为 0 时该因子取 0（例如点正好落在框边上、或 GT 退化成一个点）。非正样本的 center-ness 恒为 0。
- 正样本的 `reg_targets` 是 \( [l,t,r,b] \)（被分配 GT 的距离），非正样本全 0；dtype 跟随 `gt_boxes`（float32/float64），`labels` 恒为 int64。

## 示例

单层 stride 8、尺度区间 \([0,64)\)、`center_radius=1.5`（半径 \(1.5\times8=12\)），GT 为 \( [2,2,22,22] \)（中心 \((12,12)\)），5 个采样点：

```
点 (4,4):   l,t,r,b = 2, 2, 18,18   max=18 ∈[0,64)  在中心区  -> 正样本  c = 2/18   ≈ 0.111111
点 (12,4):  l,t,r,b = 10,2, 10,18   max=18 ∈[0,64)  在中心区  -> 正样本  c = √(1·2/18) ≈ 0.333333
点 (4,12):  l,t,r,b = 2,10, 18,10   max=18 ∈[0,64)  在中心区  -> 正样本  c = √(2/18·1) ≈ 0.333333
点 (12,12): l,t,r,b = 10,10,10,10   max=10 ∈[0,64)  在中心区  -> 正样本  c = 1.0
点 (24,12): x=24 > 22，不在框内                        -> 负样本 -1
```

```
labels     = [0, 0, 0, 0, -1]
centerness = [0.1111111111, 0.3333333333, 0.3333333333, 1.0, 0.0]
reg_targets = [[2,2,18,18], [10,2,10,18], [2,10,18,10], [10,10,10,10], [0,0,0,0]]
```

把尺度区间改成 \([0,16)\)，同一组点里前两个的 \(\max=18\) 越界：它们仍在框内，于是从正样本降级为**忽略**而不是负样本：

```
labels = [-2, -2, -2, 0, -1]
```

注意点 \((24,12)\) 只差 2 像素出框，却已经是负样本——`inside` 的边界判定就发生在 \(x=22\) 这一列上。

## 约束

- 返回 `(labels, centerness, reg_targets)`：`labels` 为 \((P,)\) int64，取值 \( \in\{-1,-2\}\cup[0,G) \)；`centerness` 为 \((P,)\)，`reg_targets` 为 \((P,4)\)，两者 dtype 与 `gt_boxes` 一致。
- 判题对 `labels` 用 `torch.equal`，对 `centerness`、`reg_targets` 用绝对容差 \(10^{-6}\)（float32）/ \(10^{-12}\)（float64）。
- 边界口径固定为：`inside` 含等号、`in_center` 含等号、`in_range` 左闭右开；退化 GT（零宽或零高，面积为 0）参与匹配且因面积最小而优先被选中，center-ness 按 0 处理。
- CPU、确定性、不调用随机数；空 GT 与空点集都必须返回正确形状的空/退化结果，而不是抛错。
