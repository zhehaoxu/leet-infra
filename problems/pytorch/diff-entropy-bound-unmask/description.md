# 熵界解掩码：按温度退火的低熵优先提交

## 题目描述

离散扩散（尤其是 uniform-corruption 那一类，没有 MASK token、被破坏的位置直接换成均匀随机 token）一次性提交多少个位置，直接决定采样步数与质量。提交太少浪费步数，提交太多则被同时固定的 token 之间强相关，整块解码会退化成「各位置独立取 argmax」的贪心。

熵界解掩码（Entropy-Bounded Unmasking）的思路是：每步先在退火温度下算出每个位置的分布熵，按熵**从低到高**排序，只接收那些「联合熵不超过阈值」的前缀。被接收的位置保留本步采到的候选 token，其余位置换成均匀随机 token 重新加噪，下一轮再试。

本题只做**单步**：给定温度参数、当前步号、模型 logits，以及调用方已经采好的候选 token 与随机 token，输出本步的 token 块、接收掩码与每个位置的熵。

## 规则

记步数为 \(N\)、当前步号为 \(s\)（\(0\le s<N\)），熵界为 \(B\)。

1. **温度退火**（第一步最热、越往后越尖锐）：
   \[
   \tau=t_{\min}+(t_{\max}-t_{\min})\cdot\frac{N-s}{N}
   \]
   注意末步 \(s=N-1\) 的温度是 \(t_{\min}+(t_{\max}-t_{\min})/N\)，只是逼近 \(t_{\min}\)，并不等于它。
2. **缩放 logits 并算熵**：\(z=\text{logits}/\tau\)，逐位置
   \[
   p=\mathrm{softmax}(z),\qquad H=-\sum_v p_v\log p_v
   \]
   熵与候选 token 的采样来自**同一个**缩放后分布。
3. **按熵升序排序**（并列时下标小的在前），对排序后的序列求前缀和 \(\text{cum}\)。
4. **接收判据是「不含自己」的前缀和**：
   \[
   \text{accepted}_k \iff \big(\text{cum}_k-H_k\big)\le B
   \]
   即位置 \(k\) 被接收，当且仅当所有**比它熵更低**的位置的熵之和不超过 \(B\)。因此被接收的位置一定是最低熵的一段前缀，且 \(B=0\) 时只有唯一最低熵的位置被接收。
5. **散回原位置**后合成：`prev_sample = where(accepted, sampled_tokens, random_tokens)`。

返回 `(prev_sample, accepted, token_entropy)`：分别为 `int64`、`bool`、`float32`，形状都是 `(batch, block_length)`；`token_entropy` 是温度缩放后的熵。

## 示例

`logits` 形状 `(1, 3, 3)`：

```
位置 0: [2, 0, -1]      位置 1: [0, 0, 0]      位置 2: [1, 1, 0]
sampled_tokens = [[1, 1, 0]]      random_tokens = [[2, 2, 1]]
timestep = 0, num_inference_steps = 4, t_max = 0.8, t_min = 0.4
```

温度为 \(0.4+0.4\times 1=0.8\)，三个位置的熵是：

```
entropy = [0.365770, 1.098612, 0.983653]
```

- \(B=1.0\)：排序后为「位置 0（0.3658）、位置 2（0.9837）、位置 1（1.0986）」。位置 0 的前缀和为 0、位置 2 的前缀和为 0.3658，都不超过 1.0；位置 1 的前缀和为 1.3494，超界。于是 `accepted = [True, False, True]`，`prev_sample = [1, 2, 0]`（位置 1 用随机 token 2）。
- \(B=0.2\)（默认值 0.1 结果相同）：只有位置 0 被接收，`prev_sample = [1, 2, 1]`。
- \(B=10\)：全部接收，`prev_sample = [1, 1, 0]`。

## 约束

- `logits` 必须是三维浮点张量 `(batch, block_length, vocab_size)`，三个维度都 \(\ge 1\)，不含 nan / inf；
- `sampled_tokens` 与 `random_tokens` 必须是 `int32` / `int64` 张量、形状恰为 `(batch, block_length)`、取值落在 \([0,\text{vocab\_size})\)；
- `timestep` 是 Python 整数（布尔值不算），取值 \([0, N)\)；`num_inference_steps` 是正整数；
- `entropy_bound` 必须是非负有限实数；`t_max`、`t_min` 必须是正的有限实数（温度为 0 会除零）；
- 违反以上任一条抛 `ValueError`；
- 性质：`accepted` 恒为按熵升序的一段前缀；熵界增大时接收集合单调扩张（子集关系）；`B=0` 只接收熵最小的位置；同分布的位置熵相等，并列时下标小的优先；`prev_sample` 在被接收处等于 `sampled_tokens`、其余处等于 `random_tokens`。
