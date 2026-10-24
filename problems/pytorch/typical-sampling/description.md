# Typical / 局部典型采样（ε-sampling）

## 题目描述

top-k / top-p 都默认「概率高的就更该留」。**局部典型采样**（Locally Typical Sampling，
Tavakoli et al. 2020，llama.cpp 里叫 `typical`）反其道而行：它保留**信息量典型**的 token——
即自信息（surprisal）$-\ln p$ 接近整个分布熵 $H=\sum_i -p_i\ln p_i$ 的那些。直觉是「符合预期」的词
才该被采样；既剔除令人意外的长尾（surprisal 远大于 $H$），也剔除**过于自信**的 argmax
（surprisal 远小于 $H$）——后者是它区别于所有截断法的关键怪癖。

> 为保证判题**确定性**，均匀随机数 `u` 作为显式输入传入，采样用逆变换法实现。

## 规则

- 实现 `solve(logits, tau, u)`：`logits` 形状 `(..., V)`，`tau ∈ (0,1)` 是累积质量阈值，
  `u` 形状 `(...,)` 或标量、取值 $[0,1)$。
- 步骤：① `probs = softmax(logits)`；② 熵 $H=\sum_i -p_i\ln p_i$；③ 每个 token 的典型偏差
  $d_i = |(-\ln p_i) - H|$；④ 按 $d_i$ **升序**（并列按原索引升序）排列 token，沿此顺序累加概率，
  保留到累积和**首次 > tau** 的那个 token 为止（至少留 1 个）；⑤ 保留集合在原顺序上重新归一化
  得 `filtered`；⑥ 用 `u` 在 `filtered`（原顺序）CDF 上逆变换采样，返回选中 token 的原始索引。
- 返回 `(filtered, chosen)`。

## 示例

```
输入：logits = [3, 2.95, -5, -5, -5, -5, -5, -5, -5, -5]，tau = 0.4，u = 0.3
  probs ≈ [0.511793, 0.486833, 0.000172 ×8]
  熵 H = 0.705164
  surprisal: token0 = 0.669834 → d0 = |0.669834-0.705164| = 0.035330
             token1 = 0.719834 → d1 = |0.719834-0.705164| = 0.014670
             长尾 surprisal ≈ 8.67 → d ≈ 7.96
  按 d 升序：token1(0.0147) < token0(0.0353) < 长尾
  累积：先取 token1，cum = 0.486833 > tau=0.4 → 停，只保留 {token1}
  ⚠️ 概率最高的 token0(argmax) 被丢弃——典型采样的标志行为
  归一化 → filtered = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]，chosen = 1
输出：filtered = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], chosen = 1
```

## 约束

- float32 输入；`tau ∈ (0,1)`；`u ∈ [0,1)`
- `filtered` 每行和为 1，被丢弃位置严格为 0；`chosen` 必落在保留集合内
- 支持批量 `(..., V)`；对数用 $\ln$，$p=0$ 以下限 `1e-30` 兜底；并列按原索引先后稳定处理
- 与 top-k/top-p/min-p 不同：本方法**可能丢弃 argmax**
