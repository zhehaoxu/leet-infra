# 解题说明 · Typical / 局部典型采样

## 信息论视角

一个 token 的**自信息**（surprisal）是 $-\ln p$：越不可能出现，被选中时"信息量/惊讶度"越大。
分布的**熵** $H=\mathbb{E}_{p}[-\ln p]=\sum_i -p_i\ln p_i$ 是「平均惊讶度」。信息论的
典型集（typical set）思想：真正携带信息的 token，其 surprisal 应当**接近平均值 $H$**。

- surprisal $\gg H$：极不可能（长尾噪声）→ 丢。
- surprisal $\ll H$：几乎是必然（模型已经决定了）→ 留着也没"信息"，丢了反而让输出更生动 → 丢。

保留 $|-\ln p_i - H|$ 最小的一批，就是对"既不出格、也不无聊"的候选做采样。

## 截断规则（对齐 llama.cpp `typical`）

1. softmax 得 `probs`。
2. `H = -(probs * log(probs)).sum()`。
3. `d_i = abs(-log(probs_i) - H)`。
4. 按 `d` 升序排 token（并列按原索引升序，`argsort(..., stable=True)`）。
5. 沿此顺序累加概率，保留到累积和**首次 > tau** 的 token（含该 token）；一个都没跨过则全留。
6. 保留集合映射回原顺序、重新归一化 → `filtered`。
7. 逆变换采样（对 `filtered` 原顺序 CDF 取首个 $\ge u$）→ `chosen`。

## 为什么实现要注意

- **argmax 可能被丢弃**：当存在若干中等概率 token 把 $H$ 抬到明显高于 argmax 的 surprisal 时，
  argmax 的偏差反而比中概率 token 大，于是不在最早累积的"典型"集合里（见题面示例）。
- `p=0` 的 token：`log(0) = -inf` → surprisal 无穷大 → 偏差最大 → 排最后 → 被截断，
  归一化时置 0 无影响；代码用 `clamp_min(1e-30)` 避免出现 `inf/nan`。
- 归一化除数用 `filtered.sum(-1, keepdim=True)`；因至少保留 1 个 token，分母恒 > 0。

## 参数

`tau` 对应 llama.cpp 的 `typ_p`（默认接近 1.0 表示基本不截断）；越小截断越狠。
