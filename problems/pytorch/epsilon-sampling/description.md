# 采样：Epsilon Sampling 相对阈值截断

## 题目描述

top-p 按**累计质量**截尾、top-k 按**名次**截尾，都有一个共同盲区：它们对"该留多少"的判断与分布的**绝对概率**无关——一条病态平滑的分布（长尾摊得很开）在 top-p 眼里照样头头有理。epsilon 截断（Hewitt et al., *Truncation Sampling as Language Model Desmoothing*, arXiv:2210.15191）直接盯概率本身：**低于阈值的尾部整体剔除**。本题采用**相对阈值**口径

\[\text{thr}=\varepsilon\cdot p_{\max},\qquad \text{keep}=\{i: p_i\ge\text{thr}\}\]

（阈值锚定在行内最大概率上，跨分布可比；**等于阈值保留**，因此最高概率 token 永远存活、`filtered` 分母恒正）。截断后有两种质量处置：`renorm=True` 传统做法——保留集合重新归一化；`renorm=False`（去平滑论文口径）——保留项**原样不动**，被剔除的质量"窟窿"全部补给该行**原始概率最大**的 token（并列取下标最小者），其余保留项的概率不变。实现它：随机性以显式均匀数 `u` 注入，在 `filtered` 原顺序 CDF 上逆变换采样。CPU float32、全程无随机性。

## 规则

- 工厂函数 `solve(logits, eps, renorm, u)`；完整口径只写在 starter 的注释里，那是唯一判题标准。
- 流程：
  1. `probs = softmax(logits)`；
  2. 逐行阈值 \(\text{thr}=\varepsilon\cdot p_{\max}\)（`p_max` 是该行最大概率）；**剔除 \(p_i<\text{thr}\)** 的候选（判据用 \(\ge\)，取等保留）；
  3. 两种 renorm 口径：
     - `renorm=True`：保留集合重新归一化 `filtered = probs·keep / sum(kept)`；
     - `renorm=False`：保留集合**不**缩放；窟窿 \(=1-\sum_{\text{kept}}p_i\) 逐元素 `scatter_add` 给该行 `argmax`（原始概率最大的 token，并列取最小下标），其余保留项保持原始概率；
  4. `chosen`：`u` 在 `filtered` **原顺序** CDF 上逆变换（前缀和首次 \(\ge u\) 的下标，兜底取最后一个下标）。
- 返回 `(filtered, chosen)`：`filtered` 与 `probs` 同形状、和为 1、被剔位置严格为 0。
- 全程 float32、CPU、确定性；不许调用现成采样器。

## 示例

`logits=[2.0, 1.0, 0.5, -1.0, -2.0]`，`eps=0.15`，`u=0.95`。

softmax 得 \(p=[0.6027,\,0.2217,\,0.1345,\,0.0300,\,0.0110]\)；阈值 \(=0.15\times0.6027=0.0904\) → 保留前 3 个，剔除后两个（合计 0.0411）。

- `renorm=False`：保留三项**原样**，0.0411 全补给 token 0：`filtered=[0.6438, 0.2217, 0.1345, 0.0, 0.0]`。
- `renorm=True`：三项除以 0.9589：`filtered=[0.6285, 0.2312, 0.1402, 0.0, 0.0]`。

两种口径下 CDF 首次 \(\ge0.95\) 都在 token 2 → `chosen=2`。

```
renorm=False:  [0.6438, 0.2217, 0.1345, 0.0, 0.0]   chosen=2
renorm=True :  [0.6285, 0.2312, 0.1402, 0.0, 0.0]   chosen=2
```

## 约束

- \(0\le\varepsilon\le1\)；`logits` 可带批维（阈值、剔除、归一化逐行独立）；`u` 形状与批维匹配
- `filtered` 每行和为 1、被剔位置严格为 0、`chosen` 必落在保留集合内；浮点 `atol=rtol=1e-5`，`chosen` 用 `torch.equal`
- 四个判别口径评测直接断言：\(\varepsilon=0\) → 阈值 0 全保留，两种口径都等于原始 softmax（去平滑口径下窟窿为 0，"补峰"是无害空操作）；\(\varepsilon=1\) → 只有 argmax 达标，两种口径都是 one-hot、`chosen` 恰为 `argmax`；`renorm=False` 时非 argmax 的保留项概率与原始 softmax **逐位相同**（`atol=1e-7`）——质量只流向 argmax；高阈值（\(\varepsilon=0.9\)）下保留集合收缩、两种口径都趋于 one-hot
