# 解题说明 · Top-k 采样

## 核心思想

softmax 得到的完整词表分布里，绝大多数 token 概率极低（长尾）。若直接从完整分布采样，
长尾累积质量会让模型偶尔蹦出无意义的词。Top-k 用**硬截断**：只留最可能的 `k` 个 token，
丢弃其余，再对这 `k` 个归一化后采样。

## 为什么按 logit 选 = 按概率选

softmax 是单调函数：$p_i > p_j \iff \text{logit}_i > \text{logit}_j$。所以「logit 最大的 k 个」
与「概率最大的 k 个」是同一集合。取第 k 大的 logit 作阈值 `thr`，`logits >= thr` 即保留掩码。
（实现上二者在无并列时完全等价；并列时本实现用阈值法会保留全部并列项，测试数据用随机 float32
基本无并列。）

## 逆变换采样（inverse-CDF）

要把「连续均匀随机数 u」映射到「按 filtered 分布抽一个离散 token」：算 `filtered` 的累积和 CDF，
取首个 $\text{CDF}[i] \ge u$ 的位置 i。因为 CDF 从 0 单调爬到 1，落在 $[\text{CDF}[i-1],\text{CDF}[i])$
的 u 都会选到 i，概率恰为 $\text{filtered}[i]$——正是我们想要的分布。把 u 作显式入参，判题就能
逐位复现，避免 true RNG 造成的 flaky。

## 与 top-p / min-p 的对比

| 方法 | 截断依据 | 与置信度关系 |
|------|----------|--------------|
| top-k | 固定个数 k | 无关（永远留 k 个） |
| top-p | 累积质量达 p 的最小前缀 | 相关（尖锐时核小） |
| min-p | 相对最高概率 p·p_max 的地板 | 相关 |

top-k 简单但「一刀切」：模型很确信时仍强留 k 个噪声候选，很困惑时又可能砍掉本应保留的词。
这也是社区视其为 legacy、推荐 top-p/min-p 组合的原因（llama.cpp 里 top-k 默认 40）。
