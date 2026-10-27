# Soft-NMS：置信度连续衰减的软非极大值抑制

## 题目描述

硬 NMS 在 IoU 超阈值时**直接删框**，人群/遮挡场景里第二个真实目标往往被吃掉。Soft-NMS（Bodla et al., ICCV 2017, arXiv:1704.04503）改为只**压低置信度**：每轮仍取当前最高分框入列，但不删除与它重叠的框，而是按 IoU 给它们乘一个衰减因子；分数跌出 `score_thr` 才真正淘汰。本题实现两种衰减核——`"linear"`（IoU 严格大于阈值才乘 \(1-\text{IoU}\)）和 `"gaussian"`（无条件乘 \(e^{-\text{IoU}^2/\sigma}\)），并返回保留序的**原始下标**与**入列那一刻的分数**。CPU、全程确定性、无随机数调用。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。boxes 是 (N,4) 的 \([x_1,y_1,x_2,y_2]\) 合法框（不校验），scores 是 (N,)；N=0 允许。
- **每轮四步**（顺序不可换）：
  1. 剩余框里取当前分数最高者，**并列取原始下标最小者**，加入保留序，并记录它**被选中那一刻**（已经历此前各轮衰减）的分数；
  2. 剩余每个框按它与刚选中框的 IoU 衰减分数；
  3. 删除衰减后分数 `< score_thr` 的剩余框（**严格小于才删，等于阈值保留**）；
  4. 剩余为空则停。
- **两种衰减核**（method 只接受这两个取值，其余抛 ValueError）：
  \[\text{"linear"}:\ \text{IoU}>\text{iou\_thr}\ \text{时}\ s\leftarrow s\,(1-\text{IoU})\ \text{，否则不变；}\qquad
    \text{"gaussian"}:\ s\leftarrow s\,\exp\!\big(-\text{IoU}^2/\sigma\big)\ \text{（无条件，完全不看 iou\_thr）.}\]
  `sigma` 只在 gaussian 下使用且必须 > 0（否则 ValueError）；`method="linear"` 时 sigma 取值再离谱也必须被**完全忽略**（不得校验）。`iou_thr` 只在 linear 下使用。
- **IoU 口径与 `iou-compute` 一致**：交集宽高 `clamp_min(0)`，inter = iw·ih，union = 面积和减交集，union 为 0 时 IoU 记 0。
- 衰减是**逐轮累乘**的：一个框可以被多个已选中框依次压低（\(s\) 乘上每一轮的因子），所以它的入列分数一般低于原始分数。
- 返回 2 元 tuple `(keep_indices, keep_scores)`：前者 long 型 (M,) 的原始下标、按选中顺序排列，后者 float (M,) 的入列分数。N=0 返回两个 (0,) 空张量；N≥1 时首轮最高分框必被保留（M≥1），其余框可被 `score_thr` 全部剔光。

## 示例

boxes \(=[[0,0,10,10],[1,1,11,11],[50,50,60,60]]\)，scores \(=[0.9,0.8,0.7]\)，全默认（linear、iou_thr=0.3、score_thr=0.001）。

- 第 1 轮：选中 0（0.9）。框 1 与它的交集是 \(9\times9=81\)、union \(=100+100-81=119\) → IoU \(=81/119=0.680672>0.3\) → \(s_1=0.8\times(1-0.680672)=0.2554622\)；框 2 不相交 → 0.7 不变。
- 第 2 轮：剩余 \(\{0.2554622,\,0.7\}\)，最高是**框 2**（0.7）→ 入列，衰减对框 1 无作用（IoU=0）。
- 第 3 轮：框 1 以 **0.2554622** 入列（这是它被压低后的分数，不是原始的 0.8）。

返回（实测回填的 EXAMPLE_*）：

```
keep_indices = [0, 2, 1]
keep_scores  = [0.8999999762, 0.6999999881, 0.2554621696]
```

对照硬 NMS：IoU=0.68 > 0.3 会把框 1 直接删掉，输出只有 `[0, 2]`。

## 约束

- method ∈ {`"linear"`, `"gaussian"`}，其余 ValueError；gaussian 且 sigma ≤ 0 ValueError；linear 时 sigma 被忽略且不得报错；N=0 与 N=1 都是合法输入
- 并列最高分取**原始下标最小**者；`score_thr` 判定是"衰减后严格小于才删"，等于阈值必须保留
- linear 的阈值比较是**严格大于**：IoU 恰等于 iou_thr 时不衰减
- gaussian 下 IoU=0 的衰减因子是 \(e^0=1\)，不相交框的分数必须逐位不变
- 下标张量 dtype 为 long、分数张量 dtype 同输入 float；索引序列用 `torch.equal`/tolist 精确比对，分数用 \(10^{-5}\)（示例 \(10^{-6}\)）容差对拍
