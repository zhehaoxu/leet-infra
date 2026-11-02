# notes（vision-rfcn-psroi-pool）

## 回源

- 论文：R-FCN（arXiv 1605.06409）§2.3。位置敏感得分图把 `k×k` 个空间网格与 `k²` 个通道
  一一对应，每个 bin 只对 RoI 内对应的子区域做平均池化，最后 `k²` 个 bin 等权投票得到类别
  得分；回归分支同理（`4k²` 通道）。
- **实现源码没能钉死**：官方仓 `daijifeng001/R-FCN` 把 Caffe 放在 `external/caffe` 子模块
  （`.gitmodules` 指向 `daijifeng001/caffe-rfcn` 的 `faster-R-CNN` 分支），该分支的 tree 取回
  是空的，`psroi_pooling_layer.cu` 拿不到。故本题**不声称复刻某个 kernel 的浮点 round 口径**，
  而是把 bin 划分显式规范化成整数整除，并在题面写死。
- 这也意味着本题与 Caffe 原实现可能有 ±1 像素的 bin 边界差异：那是官方口径不确定的部分，
  不是本题的考点。

## 本题的规范化

- 坐标必须是整数特征坐标（浮点会 `ValueError`），`spatial_scale` 折算交给调用方；这样
  bin 边界、`bounds`、`bin_sizes` 全部可用整数复算。
- bin 划分：`[y1 + i*h//k, y1 + (i+1)*h//k)`，行优先绑定通道 `i*k + j`。
- 空 bin：`h < k` 时靠前的行 bin 退化为空区间（例如 1×1 的 RoI 在 k=2 下只有最后一个 bin
  有像素），空 bin 值记 `0.0` 且仍进 `k²` 分母——这是「等权投票」的直接后果，也是本题最
  容易被写成「按面积加权」的地方。
- 零面积 RoI（`x2 <= x1` 或 `y2 <= y1`）与完全出界的 RoI 全部 bin 为空，输出 0。

## 变异

15 条全部 CAUGHT：

```
m-bin-start / m-bin-end      区间起点/终点公式改坏     → bin_sizes + 数值抓
m-channel-col-major          通道按列优先绑定          → 通道常数图抓
m-empty-ones                 空 bin 填 1 而非 0        → 空 bin 用例抓
m-bin-sum                    池化用 sum                → 均值用例抓
m-pooled-sum                 投票用 sum                → pooled 断言抓
m-pooled-size-weight         投票按面积加权            → 1 像素 vs 3 像素用例抓
m-no-clip-x1 / m-clip-x2-1   不夹取 / 上界少 1         → 出界 RoI 用例抓
m-neg-width                  宽度不 clamp 到 0         → 反向坐标用例抓
m-roi-int-check              放过非整数坐标            → validation 抓
m-kk-check / m-k-bool        通道数校验与 bool 拒收    → validation 抓（bool 单独用例）
m-size-count                 像素数用加代替乘          → bin_sizes 抓
m-bounds-order               bounds 存成转置           → 非对称 RoI 用例抓
```

## 查重

- `vision-yolov3-anchor-decode`(702)、`vision-yolov5-build-targets`(707)、
  `vision-yolov8-tal-assign`(708)：正样本分配与框编码，不涉及 RoI 内分桶池化。
- `vision-kl-var-voting`(705)：多框融合权重，不做位置敏感通道。
- `vision-spp-pyramid-pool`(703)：多尺度金字塔池化，bin 由 `ceil(H/k)` 决定且无通道分组；
  本题的通道-位置绑定与等权投票是独立机制。
