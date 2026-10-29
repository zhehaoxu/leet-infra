# 视觉检测：多尺度 anchor 生成与 IoU 匹配

## 题目描述

单阶段检测器（SSD, arXiv:1512.02325；RetinaNet, arXiv:1708.02002）在特征图的每个格子上铺一组预设框（anchor），再用「anchor 与 GT 的 IoU」决定每个 anchor 是正样本、负样本还是忽略。这一步叫 anchor matching，是检测 loss 的前置：匹配错了，分类头再强也学不出框。

本题要求把这一步完整实现出来，两件事都要自己做：

- **anchor 生成**：给定多层特征图尺寸、步长、尺度与宽高比，按固定顺序铺出全部 anchor，并把越界坐标裁剪回图像内；
- **IoU 阈值匹配**：逐 anchor 找 IoU 最大的 GT，按正/负/忽略三档打标签，再补一条 SSD 式的**保底规则**——每个 GT 至少要有一个正样本，即使所有 anchor 的 IoU 都低于正阈值。

难点集中在「顺序」和「退化」上：anchor 的枚举顺序（层 → 行 → 列 → 尺度 → 宽高比）决定下标，下标决定保底规则挑中谁；坐标先成框再裁剪，裁剪会让框退化但仍要参与 IoU；无 GT 时仍要返回完整的 anchor 列表，不能提前返回空。全 CPU、确定性、不调用随机数。

## 规则

完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。

- **anchor 生成顺序**固定为：层（按输入顺序）→ 行 \(y\)（从 0 起）→ 列 \(x\)（从 0 起）→ 尺度 \(s\)（按输入顺序）→ 宽高比 \(r\)（按输入顺序）。第 \(l\) 层的格子 \((x,y)\) 的中心是 \( \left((x+0.5)\cdot\mathrm{stride}_l,\ (y+0.5)\cdot\mathrm{stride}_l\right) \)，与该层特征图尺寸无关（只看步长）。
- 每个 \((s,r)\) 组合给出宽 \( s\sqrt{r} \)、高 \( s/\sqrt{r} \)，即以格心为中心、不随层缩放；坐标先算出完整框，再把四个坐标分别裁剪到 \( [0, W_{\text{img}}] \) 与 \( [0, H_{\text{img}}] \) 之间。裁剪后宽或高可能为 0，仍然保留该 anchor（不丢弃、不去重）。
- anchor 总数 \( A=\sum_l H_l W_l \cdot |\mathrm{scales}|\cdot|\mathrm{ratios}| \)，顺序如上；返回的 `anchors` 是 \( (A,4) \)，dtype 跟随 `gt_boxes`（float32 或 float64）。
- **IoU 口径**：交集面积按裁剪后的坐标算，\( \mathrm{IoU}=\frac{\text{inter}}{S_a+S_g-\text{inter}} \)；两个框面积都为 0 时记 0（不产生 0/0）。
- **匹配**（对每个 anchor）：
  1. 在所有 GT 中取 IoU 最大者，得到 `best_gt` 与 `best_iou`；IoU 完全并列时取 GT 下标最小者。
  2. `best_iou >= pos_thr` 记正样本，标签为匹配到的 GT 下标；`best_iou < neg_thr` 记负样本，标签 -1；其余（\( \mathrm{neg\_thr}\le \mathrm{best\_iou}<\mathrm{pos\_thr} \)）记忽略样本，标签 -2。
  3. **保底规则在上一步之后执行**：对每个 GT \(g\)，在「`best_gt == g`」的 anchor 中取 `best_iou` 最大者（并列取 anchor 下标最小者）强制改成正样本、标签 \(g\)，即使它的 IoU 低于 `pos_thr`。该 GT 没有任何 anchor 指向它时跳过。
- `gt_boxes` 为空（0 行）时：anchors 仍按上述规则完整生成，全部标签为 -1，计数的正/忽略档为 0。
- 返回 `counts` 是长度 3 的 int64，依次为 `[正样本数, 负样本数, 忽略样本数]`，三者之和等于 \(A\)。
- 非法输入一律抛 `ValueError`：`gt_boxes` 不是 float32/float64 张量、形状不是 \( (G,4) \)、层数不匹配或为空、特征图尺寸或 stride 非正、`scales`/`ratios` 为空或含非正值、`image_size` 不是两个正整数、阈值不满足 \( 0\le \mathrm{neg\_thr}\le \mathrm{pos\_thr} \)。

## 示例

单个特征图 \(2\times2\)、步长 8、尺度 16、宽高比 1.0，图像 \(32\times32\)；GT 只有一个框 \( [4,4,20,20] \)：

```
anchor 0: 格(0,0) 中心(4,4)   框(-4,-4,12,12) 裁剪后 (0,0,12,12)  IoU =  64/336 ≈ 0.190476
anchor 1: 格(1,0) 中心(12,4)  框( 4,-4,20,12) 裁剪后 (4,0,20,12)  IoU = 128/320 =  0.4
anchor 2: 格(0,1) 中心(4,12)  框(-4, 4,12,20) 裁剪后 (0,4,12,20)  IoU = 128/320 =  0.4
anchor 3: 格(1,1) 中心(12,12) 框( 4, 4,20,20) 裁剪后 (4,4,20,20)  IoU = 256/256 =  1.0
```

anchor 1、2 的 IoU **恰好等于** `neg_thr=0.4`：落在忽略带的判定是 \( \mathrm{neg\_thr}\le\mathrm{best\_iou}<\mathrm{pos\_thr} \)，等号归忽略，所以它们记 -2 而不是 -1（float32 下这两个值算出 0.40000000596，仍在忽略带内）。anchor 0 的 0.190476 低于 0.4 记 -1，anchor 3 的 1.0 记 0：

```
anchors = [[0,0,12,12], [4,0,20,12], [0,4,12,20], [4,4,20,20]]
labels  = [-1, -2, -2, 0]
counts  = [1, 1, 2]
```

保底规则在「没有任何 anchor 达到 `pos_thr`」时体现。把 GT 换成中心处的极小框 `[(15.9,15.9,16.1,16.1)]`，同一套 anchor 的 IoU 依次为 0、0、0、0.00015625（前三个框与 GT 无交集）：全部低于 0.5 也全部低于 0.4，若只看阈值全是负样本；保底规则把 IoU 最大的 anchor 3 强制置为 0：

```
labels = [-1, -1, -1, 0]
counts = [1, 3, 0]
```

## 约束

- 返回 `(anchors, labels, counts)`：`anchors` 为 \( (A,4) \) 且 dtype 与 `gt_boxes` 一致；`labels` 为 \( (A,) \) int64；`counts` 为 \( (3,) \) int64。
- 判题对 `anchors` 用绝对容差 \(10^{-5}\)（float32）/ \(10^{-12}\)（float64），对 `labels`、`counts` 用 `torch.equal`。
- anchor 的顺序、中心公式、裁剪边界、IoU 的零面积约定、保底规则的执行时机与并列打破方式都是判题的一部分，不能换一种等价但顺序不同的枚举。
- CPU、确定性、不调用随机数；无 GT 与空 `image_size` 等退化分支必须按上述约定返回，而不是抛错。
