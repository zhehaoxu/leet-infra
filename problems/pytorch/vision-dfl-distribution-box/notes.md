# notes（vision-dfl-distribution-box）

## 回源

- 损失：`open-mmlab/mmdetection` main `mmdet/models/losses/gfocal_loss.py::distribution_focal_loss`
  （144-165 行）。原文实现：
  `dis_left = label.long()`、`dis_right = dis_left + 1`、
  `weight_left = dis_right.float() - label`、`weight_right = label - dis_left.float()`、
  `loss = CE(pred, dis_left) * weight_left + CE(pred, dis_right) * weight_right`。
- 解码：GFL head 的 `integral`——`softmax(x.reshape(-1, reg_max+1))` 再与
  `arange(reg_max+1)` 做线性组合；`distance2bbox` 里再乘 stride。本题把 `stride` 作显式入参。
- 论文：arXiv 2006.04388（Generalized Focal Loss）§3.2。

## 官方实现里的坑

1. **两条 CE，不是一条**：`distribution_focal_loss` 对 `left` 与 `right` 各算一次
   `cross_entropy` 再按 `wl/wr` 加权；把目标当软标签做一次 CE 会得到不同的梯度与数值。
2. **权重恒和为 1**：`wl = right - label`、`wr = label - left`，且 `right = left + 1`，
   所以 `wl + wr = 1`；写反（`wl = label - left`）在整数目标上恰好也等于 0/1，只有分数
   目标才能测出来。
3. **`label.long()` 是向零截断**：正目标等价 floor，但不要写成 `round`——`3.7` 应落到
   bin 3 而不是 4。
4. **`right` 会越界**：`label = reg_max` 时 `right = reg_max + 1` 超出 bin 范围，官方实现
   直接崩；本题把它规范化为 `ValueError` 并要求 `target ∈ [0, reg_max-1]`。
5. **期望的下标从 0 起**：`arange(reg_max+1)` 从 0 开始，所以均匀分布下期望是 `reg_max/2`
   而不是 `(reg_max+1)/2`。
6. **归一化分母**：`@weighted_loss` 的 `reduction='mean'` 是除以元素总数（`N × 4`），
   不是除以正样本数。

## 本题的规范化

- 边数 `S` 不写死为 4（检测里是 4），方便用单边用例把权重、下标、CE 拆开验。
- 账本同时给出 `left`/`right`/`w_left`/`w_right`，使「两条 CE」与「权重方向」都可观测，
  避免只看标量 loss 时把两种写法混淆。
- 解码与损失解耦：不给 `targets` 就只返回 `probs`/`dist`/`bin_index`，且不得出现 loss 键。
- 概率用 `softmax`、损失用 `log_softmax`+gather（与 `F.cross_entropy` 同口径）。
- 校验门：`reg_max` 正整数（`bool` 拒收，单独用例防被维度检查掩盖）、最后一维必须
  `reg_max+1`、`stride` 有限正数、`targets`/`weights` 形状一致。

## 变异

14 条全部 CAUGHT：

```
m-bin-offset                 下标从 1 起                  → 均匀分布期望用例抓
m-dist-no-stride             解码不乘 stride              → stride 用例抓
m-wl-swap                    左右权重写反                 → 分数目标用例抓
m-single-ce / m-ce-no-weight 只留一条 CE / 权重不乘        → 混合用例抓
m-right-eq-left              right 等于 left              → 下标断言抓
m-target-gate                放过越界目标                 → 越界用例抓
m-mean-denom                 分母用 N                     → 归一化用例抓
m-no-weights                 忽略 weights                 → 权重用例抓
m-softmax-dim                softmax 走错维               → decode 抓
m-left-round                 截断写成四舍五入             → 3.7 类用例抓
m-regmax-bool / m-logit-dim  两条校验门改坏               → validation（bool 单独用例）
m-stride-finite              漏掉有限性检查               → stride=inf 用例抓
```

## 查重

- `vision-yolov3-anchor-decode`(702) / `vision-yolov5-build-targets`(707)：框编码与目标分配，
  不做分布化回归。
- `vision-yolox-simota-assign`(706)：正样本分配代价，不涉及 DFL。
- 本题只考「分布解码期望 + 左右 bin 的双 CE」，与上述均不重叠。
