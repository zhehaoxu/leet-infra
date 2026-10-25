# 采样：Dripping 高概率抑制

## 题目描述

min-p 把"低于峰值某个相对比例"的**尾部**砍掉；它的**头尾对偶**是反过来砍**头部**：概率太高到"没有信息量"的 token（分布里扎堆的那个尖峰）被滴滤掉，把概率质量还给尾巴，迫使采样走出尖峰、探索低概率区域。Dripping（滴滤）采样按整行统计量定一条**相对阈值**，把高于它的头部整批砍掉：

\[\text{mode="mean"}:\ \text{thr}=\alpha\cdot\bar p,\qquad
\text{mode="std"}:\ \text{thr}=\bar p+\alpha\cdot\sigma,\qquad
\text{cut}=\{i: p_i>\text{thr}\}\]

其中 \(\bar p\)、\(\sigma\) 是 softmax 概率的均值与**总体**标准差（只在"有效项" \(p_i>10^{-35}\) 上统计，下溢项不污染统计量）。触发条件是"**确实砍掉了至少一个 token** 且 **存活数 \(\ge\)`min_keep`**"——任一不满足就整行不砍（宁可不砍也不能砍空）。存活集合重新归一化得 `filtered`，用显式均匀数 `u` 在原顺序 CDF 上逆变换采样。CPU float32、全程无随机性。

## 规则

- 工厂函数 `solve(logits, mode, drip_amount, min_keep, u)`；完整口径只写在 starter 的注释里，那是唯一判题标准。
- 流程：
  1. `probs = softmax(logits)`；
  2. 统计量只在**有效项**（`probs > 1e-35`）上计算：\(n\)=有效项数、\(\bar p=\sum p_i/n\)、\(\sigma=\sqrt{\sum p_i^2/n-\bar p^2}\)（**总体**标准差，分母是 \(n\)，不是 \(n-1\)）；
  3. 阈值按 `mode` 二选一：`"mean"` → \(\text{thr}=\alpha\bar p\)；`"std"` → \(\text{thr}=\bar p+\alpha\sigma\)；其他 `mode` 抛异常；
  4. **砍头部**：`cut = probs > thr`（**等于阈值不砍**）——softmax 单调，被砍的恰是概率最高的那一批；
  5. **兜底**：`cut` 为空（没有任何 token 被砍）**或**存活数 \(<\)`min_keep` → 整行不砍，`filtered` 等于原始 softmax、`cut_count=0`；
  6. 存活集合重新归一化得 `filtered`（原顺序，被砍位置严格为 0）；
  7. `chosen`：`u` 在 `filtered` 原顺序 CDF 上逆变换（前缀和首次 \(\ge u\) 的下标，兜底取最后一个下标）。
- 返回 `(filtered, chosen, cut_count)`：`cut_count` 为 int64 张量，形状与 logits 批维一致；**未触发行记 0**（不是"想砍但被兜底拦下"的数目）。
- 全程 float32、CPU、确定性；不许调用现成采样器。

## 示例

`logits=[2.0, 1.5, 1.0, 0.5, 0.0, -0.5]`，`mode="std"`、`drip_amount=1.0`、`min_keep=1`、`u=0.55`。

softmax 得 \(p=[0.414,\,0.251,\,0.152,\,0.092,\,0.056,\,0.034]\)；全体有效，\(\bar p=1/6=0.1667\)，\(\sigma=\sqrt{0.2704/6-0.1667^2}=0.1315\)。`std` 模式阈值 \(=0.1667+0.1315=0.2982\) → 只有 token 0（0.414）高于阈值，砍 1 个、存活 5 ≥ 1，触发。归一化后 `filtered=[0, 0.4287, 0.2600, 0.1577, 0.0956, 0.0580]`；CDF 在 token 2 处首次 \(\ge0.55\) → `chosen=2`、`cut_count=1`。

同一行换 `mode="mean"`、其余不变：阈值 \(=1.0\cdot0.1667=0.1667\) → token 0 与 token 1（0.251）都被砍，`cut_count=2`。

```
std  模式:  filtered=[0.0, 0.4287, 0.2600, 0.1577, 0.0956, 0.0580]  chosen=2  cut_count=1
mean 模式:  cut_count=2
```

## 约束

- `mode` 只接受 `"mean"`/`"std"`；`drip_amount\ge0`、`min_keep\ge1`；`logits` 可带批维（统计量与阈值逐行独立）
- `filtered` 每行和为 1，被砍位置严格为 0；`chosen` 必落在存活集合内；浮点比对 `atol=1e-5, rtol=1e-5`，`chosen`/`cut_count` 用 `torch.equal`
- 四个判别口径评测直接断言：**完全平坦**的 logits（所有 logit 相等）→ 没有任何项"高于均值" → 两种模式都不触发，`filtered` 等于原始 softmax；`drip_amount` 越大阈值越高、砍得越**少**（单调）；存活数不足 `min_keep` 时整行不砍且 `cut_count=0`；与 min-p 相反，被清零的是**头部**、尾部质量保留
